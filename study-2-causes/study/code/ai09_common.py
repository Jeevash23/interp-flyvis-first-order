"""Paths, read-only access to AI-06, guards and record helpers for AI-09.

AI-06 is read, never written: its simulator (02-Code/stage1/core.py), its stimuli, checkpoints and saved records are
imported or loaded in place. Run every script with `python -B` so that importing AI-06 modules writes no bytecode into
its folders. AI-06's stage1 `common.py` points matplotlib and the XDG cache at AI-06's own `.cache`; they are pointed
back at AI-09's `.cache` here, before FlyVis (and matplotlib) are imported.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "study"
RESULTS = STUDY / "results"
CONFIG = json.loads((STUDY / "config.json").read_text())
AI06 = (ROOT / CONFIG["ai06_root"]).resolve()
AI06_RUNS = AI06 / "study/results/runs"

sys.path.append(str(AI06 / "study/code"))
sys.path.append(str(AI06 / "02-Code/stage1"))
import revision_common as ai06  # noqa: E402  (AI-06 paths, stimuli, model list; reads only)
from core import Core, metrics, TARGETS  # noqa: E402  (AI-06 batched simulator; imports stage1 common)

os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")

NAMES = ai06.NAMES
READOUTS = ai06.READOUTS
MODELS = ai06.MODELS
CLUSTERS = ai06.CLUSTERS
STIMULI = ai06.STIMULI
DEVELOPMENT = CONFIG["development_models"]
CONFIRMATION = [m for m in MODELS if m not in DEVELOPMENT]
assert TARGETS == READOUTS


def sha256(path: Path) -> str:
    return ai06.sha256(Path(path))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    return ai06.load_npz(Path(path))


def stimulus(key: str) -> dict[str, np.ndarray]:
    return ai06.stimulus(key)


def causal_frames(key: str) -> int:
    return ai06.causal_frames(key)


def ai06_record(model: str, key: str, name: str) -> dict[str, np.ndarray]:
    """Load one saved AI-06 record after checking its recorded checksum."""
    path = AI06_RUNS / model / key / name
    metadata = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != metadata["sha256"]:
        raise RuntimeError(f"AI-06 record checksum mismatch: {path}")
    return load_npz(path)


def require_development(model: str) -> None:
    """Refuse confirmation models until the AI-09 freeze exists and matches."""
    if model in DEVELOPMENT:
        return
    freeze = STUDY / "freeze-2.json"  # Amendment 2; the first freeze, freeze.json, is kept unchanged
    if not freeze.exists():
        freeze = STUDY / "freeze.json"
    if not freeze.exists():
        raise RuntimeError(f"Model {model} is a confirmation model; AI-09 is not frozen")
    record = json.loads(freeze.read_text())
    changed = [rel for rel, digest in record["files"].items() if sha256(ROOT / rel) != digest]
    if changed:
        raise RuntimeError(f"AI-09 frozen files changed: {changed}")


RECORDS_LOCATION = RESULTS / "runs"  # set by the runner to the folder it is writing (may be linked to another drive)


def free_gib(path: Path) -> float:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path.resolve()).free / 1024**3


def health() -> None:
    """Stop our own worker before host memory or disk runs out (PROTOCOL.md Sections 9 and 13.3)."""
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) / 1024**2
    if available < CONFIG["minimum_ram_gib"]:
        raise RuntimeError(f"Available RAM below guard: {available:.2f} GiB")
    project = free_gib(ROOT)
    if project < CONFIG["minimum_project_disk_gib"]:
        raise RuntimeError(f"Free disk of the project's filesystem below guard: {project:.2f} GiB")
    records = free_gib(RECORDS_LOCATION)
    if records < CONFIG["minimum_disk_gib"]:
        raise RuntimeError(f"Free disk where the run records are written below guard: {records:.2f} GiB")


_THREADS_SET = False


def configure_torch(device: str):
    """Thread limits (settable once per process), the GPU lock check and the GPU memory cap."""
    global _THREADS_SET
    import torch
    if not _THREADS_SET:
        torch.set_num_threads(CONFIG["cpu_threads"])
        torch.set_num_interop_threads(CONFIG["cpu_threads"])
        _THREADS_SET = True
    if device == "cuda":
        if not os.environ.get("AI09_GPU_LOCK_FD"):
            raise RuntimeError("Launch through study/code/gpu_lock.py")
        os.fstat(int(os.environ["AI09_GPU_LOCK_FD"]))
        torch.cuda.set_per_process_memory_fraction(CONFIG["max_gpu_memory_fraction"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()
