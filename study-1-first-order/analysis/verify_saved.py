"""Read-only verification of the frozen study and its saved outputs (make verify).

1. Hashes: the 182 frozen sources and inputs (study/freeze.json), the 3,228 original evidence files
   (provenance/original-records.json) and the 33 files of the earlier study's freeze still match.
2. Saved responses: every record of the 50 models (study/results/runs/**) matches the checksum and simulation
   signature in its metadata, and its DSI, denominators, peaks, mean voltages and integrals are recomputed from the
   stored traces with the frozen check (study/code/verify_revision.py, check_record). Nothing is written; the fresh
   CPU replays are recorded in study/results/verification/ and are not repeated here.
3. Frozen analysis: study/code/analyse_revision.py (sources legacy, development, confirmation and all) and
   study/code/supplement.py run unchanged in a temporary mirror of the project that links to the saved runs, and
   every output is compared with the saved file in study/results/analysis/ (JSON files without their creation
   times). The mirror lives under $TMPDIR and is removed afterwards.
4. Relocations: every file moved or copied on 9 October (provenance/cleanup-2026-10-09.json) is at its destination
   with its recorded size and SHA-256.

Options: --skip-records and --skip-analysis leave out parts 2 and 3 (about two minutes each).
"""
from __future__ import annotations

import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "study/code"
sys.path.insert(0, str(CODE))
failures = []


def fail(message):
    failures.append(message)
    print("FAIL:", message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hashes():
    from revision_common import freeze_check, original_freeze_check
    try:
        freeze_check()
        frozen = len(json.loads((ROOT / "study/freeze.json").read_text())["files"])
        stage1 = original_freeze_check()["files"]
    except RuntimeError as error:
        fail(str(error))
        return
    original = json.loads((ROOT / "provenance/original-records.json").read_text())["files"]
    changed = [rel for rel, digest in original.items() if not (ROOT / rel).is_file() or sha256(ROOT / rel) != digest]
    if changed:
        fail(f"original evidence changed: {changed[:10]}")
        return
    print(f"ok  hashes: {frozen} frozen files, {len(original)} original evidence files, {stage1} files of the "
          f"earlier study's freeze", flush=True)


def check_records():
    from revision_common import CONFIG, MODELS, RESULTS, stimulus
    from run_experiments import simulation_signature
    from verify_revision import check_record
    signature = simulation_signature()
    maximum, files = {}, 0
    started = time.time()
    for model in MODELS:
        for key in CONFIG["common_stimuli"]:
            dest = RESULTS / "runs" / model / key
            if json.loads((dest / "complete.json").read_text())["simulation_signature"] != signature:
                fail(f"panel signature differs: {model} {key}")
                continue
            strengths = CONFIG["main_strengths"] + (CONFIG["development_extra_strengths"]
                                                    if model in CONFIG["development_models"] else [])
            paths = [dest / "baseline.npz", dest / "gradient.npz"]
            paths += [dest / f"s{round(100 * s):03d}-cell{i:02d}.npz" for s in strengths for i in range(65)]
            if model in CONFIG["development_models"] and key == "dev_fast":
                paths.append(dest / "s100-ct1_joint.npz")
            stim = stimulus(key)
            for path in paths:
                try:
                    check_record(path, stim, signature, maximum)
                except RuntimeError as error:
                    fail(str(error))
                files += 1
    print(f"ok  {files} saved response records match their checksums and recomputed metrics "
          f"(largest DSI difference {maximum.get('dsi', 0):.1e}; {time.time() - started:.0f} s)", flush=True)


def comparable(path: Path):
    if path.suffix == ".json":
        value = json.loads(path.read_text())

        def strip(item):
            if isinstance(item, dict):
                return {k: strip(v) for k, v in item.items() if k != "created"}
            if isinstance(item, list):
                return [strip(v) for v in item]
            return item
        return strip(value)
    return None


def check_analysis():
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="ai06-analysis-mirror-") as tmp:
        mirror = Path(tmp)
        (mirror / "study/results/analysis").mkdir(parents=True)
        shutil.copytree(CODE, mirror / "study/code", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "study/config.json", mirror / "study/config.json")
        os.symlink(ROOT / "study/results/runs", mirror / "study/results/runs")
        for folder in ("01-Papers", "02-Code", "03-Data", "04-Results"):
            os.symlink(ROOT / folder, mirror / folder)
        changed_code = [p.name for p in (mirror / "study/code").glob("*.py") if not filecmp.cmp(p, CODE / p.name, shallow=False)]
        if changed_code:
            fail(f"mirror code differs: {changed_code}")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        for args in (["analyse_revision.py", "--source", source] for source in ("legacy", "development", "confirmation", "all")):
            result = subprocess.run([sys.executable, "-B", str(mirror / "study/code" / args[0]), *args[1:]], cwd=mirror,
                                    env=env, capture_output=True, text=True)
            if result.returncode:
                fail(f"frozen analysis failed in the mirror ({' '.join(args)}): {result.stderr[-500:]}")
                return
        result = subprocess.run([sys.executable, "-B", str(mirror / "study/code/supplement.py")], cwd=mirror, env=env,
                                capture_output=True, text=True)
        if result.returncode:
            fail(f"frozen supplement failed in the mirror: {result.stderr[-500:]}")
            return
        saved_root, new_root = ROOT / "study/results/analysis", mirror / "study/results/analysis"
        saved = sorted(p.relative_to(saved_root) for p in saved_root.rglob("*") if p.is_file())
        new = sorted(p.relative_to(new_root) for p in new_root.rglob("*") if p.is_file())
        if saved != new:
            fail(f"the mirror wrote a different set of files: {sorted(set(map(str, saved)) ^ set(map(str, new)))[:10]}")
        differing = []
        for rel in saved:
            a, b = saved_root / rel, new_root / rel
            if not b.exists():
                continue
            same = comparable(a) == comparable(b) if a.suffix == ".json" else filecmp.cmp(a, b, shallow=False)
            if not same:
                differing.append(str(rel))
        if differing:
            fail(f"recomputed analysis differs from the saved outputs: {differing}")
        else:
            print(f"ok  the frozen analysis, rerun in a temporary mirror, reproduces all {len(saved)} saved analysis "
                  f"files ({time.time() - started:.0f} s)", flush=True)


def check_relocations():
    log = json.loads((ROOT / "provenance/cleanup-2026-10-09.json").read_text())
    bad = []
    for entry in log["entries"]:
        dest = ROOT / entry["destination"]
        if not dest.is_file() or dest.stat().st_size != entry["bytes"] or sha256(dest) != entry["sha256"]:
            bad.append(entry["destination"])
    if bad:
        fail(f"relocated files changed or missing: {bad[:10]}")
    else:
        print(f"ok  all {len(log['entries'])} files moved or copied on 9 October are at their destinations unchanged",
              flush=True)


if __name__ == "__main__":
    check_hashes()
    if "--skip-records" not in sys.argv:
        check_records()
    if "--skip-analysis" not in sys.argv:
        check_analysis()
    check_relocations()
    if failures:
        sys.exit(f"{len(failures)} verification check(s) failed")
    print("all verification checks passed")
