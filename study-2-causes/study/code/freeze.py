"""Record the predictions (step 2) and freeze the study (step 4) of PROTOCOL.md Section 10. Neither overwrites.

  python -B study/code/freeze.py predictions   writes study/PREDICTIONS-RECORD.json (hash and full text)
  python -B study/code/freeze.py addendum      writes study/PREDICTIONS-ADDENDUM-RECORD.json (Amendment 1, Section 13.2)
  python -B study/code/freeze.py freeze        writes study/freeze.json (hashes of every file the study depends on)
  python -B study/code/freeze.py freeze2       writes study/freeze-2.json after Amendment 2 (PROTOCOL.md Section 14)
"""
from __future__ import annotations

import json
import sys

from ai09_common import AI06, CONFIRMATION, DEVELOPMENT, RESULTS, ROOT, STUDY, save_json, sha256, utc_now

AI06_INPUTS = ["02-Code/stage1/core.py", "02-Code/stage1/common.py", "study/code/revision_common.py",
               "study/code/analyse_revision.py", "study/freeze.json", "03-Data/stage1/model-selection.json",
               "03-Data/stage1/stimulus-manifest.json", "03-Data/stage1/network-structure.json"]


def record_predictions() -> None:
    path = STUDY / "PREDICTIONS-RECORD.json"
    if path.exists():
        raise SystemExit(f"{path} exists; predictions are recorded once")
    if (RESULTS / "runs").exists():
        raise SystemExit("AI-09 runs exist already; predictions must be recorded before any run")
    text = (STUDY / "PREDICTIONS.md").read_text()
    save_json(path, {"written_at_utc": utc_now(), "predictions_sha256": sha256(STUDY / "PREDICTIONS.md"),
                     "protocol_sha256": sha256(STUDY / "PROTOCOL.md"), "predictions_text": text,
                     "note": "Recorded before any AI-09 outcome was computed on any model."})


def record_addendum() -> None:
    path = STUDY / "PREDICTIONS-ADDENDUM-RECORD.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the addendum is recorded once")
    present = [m for m in CONFIRMATION if (RESULTS / "runs" / m).exists()]
    if present or (STUDY / "freeze.json").exists():
        raise SystemExit("The addendum must be recorded before the freeze and before any confirmation run")
    save_json(path, {"written_at_utc": utc_now(), "addendum_sha256": sha256(STUDY / "PREDICTIONS-ADDENDUM.md"),
                     "protocol_sha256": sha256(STUDY / "PROTOCOL.md"),
                     "development_summary_sha256": sha256(RESULTS / "analysis/development/summary.json"),
                     "addendum_text": (STUDY / "PREDICTIONS-ADDENDUM.md").read_text(),
                     "note": "Development-informed; recorded before any AI-09 quantity was computed on a confirmation "
                             "model and before the freeze."})


def freeze() -> None:
    path = STUDY / "freeze.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the study is frozen once")
    if not (STUDY / "PREDICTIONS-RECORD.json").exists() or not (STUDY / "PREDICTIONS-ADDENDUM-RECORD.json").exists():
        raise SystemExit("Record the predictions and the addendum first")
    present = [m for m in CONFIRMATION if (RESULTS / "runs" / m).exists()]
    if present:
        raise SystemExit(f"Confirmation outputs exist before the freeze: {present}")
    files = ["study/PROTOCOL.md", "study/PROTOCOL-approved-2026-10-16.md", "study/PREDICTIONS.md",
             "study/PREDICTIONS-RECORD.json", "study/PREDICTIONS-ADDENDUM.md", "study/PREDICTIONS-ADDENDUM-RECORD.json",
             "study/config.json", "study/DEVELOPMENT-NOTES.md", "Makefile",
             "study/results/analysis/development/summary.json",
             "study/results/analysis/development/models.csv",
             "study/results/analysis/development/reference-check.json"]
    files += sorted(str(p.relative_to(ROOT)) for p in (STUDY / "code").glob("*.py"))
    save_json(path, {"utc": utc_now(), "kind": "local analysis freeze; not external preregistration",
                     "development_models": DEVELOPMENT, "confirmation_models": CONFIRMATION,
                     "confirmation_outputs_exist": False,
                     "files": {rel: sha256(ROOT / rel) for rel in files},
                     "ai06_inputs": {rel: sha256(AI06 / rel) for rel in AI06_INPUTS}})


def freeze2() -> None:
    path = STUDY / "freeze-2.json"
    if path.exists():
        raise SystemExit(f"{path} exists; Amendment 2 is frozen once")
    first = json.loads((STUDY / "freeze.json").read_text())
    if (RESULTS / "analysis/confirmation").exists():
        raise SystemExit("A confirmation analysis exists; Amendment 2 must be frozen before any confirmation outcome")
    files = sorted(set(first["files"]) | {"study/freeze.json", "study/results/GATE-FAILURE-009.md",
                                          "study/amendment-2/apply_amendment2.py"}
                   | {str(p.relative_to(ROOT)) for p in (STUDY / "code").glob("*.py")})
    records = sorted(m for m in CONFIRMATION if (RESULTS / "runs" / m / "dev_fast" / "complete.json").exists()
                     or (RESULTS / "runs" / m / "final_slow" / "complete.json").exists())
    save_json(path, {"utc": utc_now(), "kind": "second local analysis freeze, after Amendment 2 (PROTOCOL.md Section 14)",
                     "previous_freeze_sha256": sha256(STUDY / "freeze.json"),
                     "development_models": DEVELOPMENT, "confirmation_models": CONFIRMATION,
                     "confirmation_outcomes_computed": False,
                     "confirmation_models_with_run_records": records,
                     "files": {rel: sha256(ROOT / rel) for rel in files},
                     "ai06_inputs": {rel: sha256(AI06 / rel) for rel in AI06_INPUTS}})


if __name__ == "__main__":
    {"predictions": record_predictions, "addendum": record_addendum, "freeze": freeze,
     "freeze2": freeze2}[sys.argv[1]]()
