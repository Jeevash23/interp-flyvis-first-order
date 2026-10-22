"""Resumable confirmation experiments; original study files are never written."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from revision_common import (
    ROOT, STUDY, RESULTS, CONFIG, MODELS, NAMES, SELECTION, STIMULI,
    causal_frames, freeze_check, health, independent_metrics, legacy_check,
    legacy_panel, load_npz, original_freeze_check, save_json, sha256,
    stimulus, utc_now,
)

sys.path.insert(0, str(ROOT / "02-Code/stage1"))
import torch
from core import Core, metrics
from gradients import gradient_bank


def simulation_signature() -> str:
    import hashlib
    files = [Path(__file__), STUDY / "code/revision_common.py", STUDY / "config.json",
             ROOT / "02-Code/stage1/core.py", ROOT / "02-Code/stage1/common.py",
             ROOT / "02-Code/stage1/gradients.py"]
    return hashlib.sha256("\n".join(sha256(path) for path in files).encode()).hexdigest()


def valid_record(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["sha256"] != sha256(path) or metadata["simulation_signature"] != signature:
        raise RuntimeError(f"Existing revision record differs: {path}")
    return True


def write_record(path: Path, arrays: dict, metadata: dict, signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    save_json(path.with_suffix(".json"), {
        **metadata, "sha256": sha256(path), "simulation_signature": signature,
        "utc": utc_now(),
    })


def checked_arrays(trace: np.ndarray, stim: dict) -> dict:
    original = metrics(torch.as_tensor(trace), stim["angle"], stim["intensity"], float(stim["speed"]), time_start=0)
    arrays = {key: value.detach().cpu().numpy() for key, value in original.items() if key != "window"}
    independent = independent_metrics(trace, stim)
    for key in ("dsi", "denominator", "peaks", "raw_mean", "raw_integral"):
        if not np.allclose(arrays[key], independent[key], atol=2e-6, rtol=2e-5):
            raise RuntimeError(f"Independent readout disagreement: {key}")
    arrays.update(trace=trace, max_abs_voltage=independent["max_abs_voltage"])
    return arrays


def reuse(path: Path, original: Path, key: str, signature: str, metadata: dict) -> None:
    legacy_check(original)
    arrays = load_npz(original)
    if "trace" in arrays:
        prefix = arrays["trace"][:, :causal_frames(key)]
        checked = checked_arrays(prefix, stimulus(key))
        for field in ("dsi", "denominator", "peaks", "raw_mean", "raw_integral"):
            if not np.allclose(checked[field], arrays[field], atol=2e-6, rtol=2e-5):
                raise RuntimeError(f"Causal-prefix reuse mismatch: {original}, {field}")
        arrays.update(checked)
    write_record(path, arrays, {**metadata, "source": str(original.relative_to(ROOT)),
                               "source_sha256": sha256(original), "reused": True}, signature)


def run_model(model: str, device: str, signature: str) -> None:
    health()
    model_folder = ROOT / f"03-Data/flyvis-root/results/flow/0000/{model}"
    checkpoint = model_folder / "best_chkpt"
    expected = next(item["checkpoint_sha256"] for item in SELECTION["models"] if item["model"] == model)
    if sha256(checkpoint) != expected:
        raise RuntimeError(f"Checkpoint changed: {model}")
    # Stage 0 reserialized the released checkpoint into the native loader's
    # chkpts layout. Container bytes differ; network tensors must be identical.
    released = torch.load(checkpoint, map_location="cpu", weights_only=False)["network"]
    native_copy = torch.load(model_folder / "chkpts/chkpt_00000", map_location="cpu", weights_only=False)["network"]
    if set(released) != set(native_copy) or any(not torch.equal(released[k], native_copy[k]) for k in released):
        raise RuntimeError(f"Native checkpoint tensor mismatch: {model}")
    is_development = model in CONFIG["development_models"]
    model_start = time.perf_counter()
    core = Core(model, device)
    gray = core.gray()
    for key in CONFIG["common_stimuli"]:
        health()
        stim = stimulus(key)
        if sha256(ROOT / STIMULI[key]["path"]) != STIMULI[key]["sha256"]:
            raise RuntimeError(f"Stimulus changed: {key}")
        frames = stim["stimulus"]
        if frames.ndim == 4:
            frames = frames[:, :, 0]
        frame_count = causal_frames(key)
        causal = frames[:, 200:200 + frame_count]
        dest = RESULTS / "runs" / model / key
        dest.mkdir(parents=True, exist_ok=True)
        previous = legacy_panel(model, key) if is_development else None
        _, state, _ = core.run(frames[:, :200], state=gray.expand(24, -1).clone())
        state = state.detach()
        base_path = dest / "baseline.npz"
        if not valid_record(base_path, signature):
            metadata = {"model": model, "stimulus": key, "kind": "baseline"}
            if previous is not None:
                reuse(base_path, previous / "baseline.npz", key, signature, metadata)
            else:
                trace, _, current = core.run(causal, state=state.clone(), start=0, currents=True)
                arrays = checked_arrays(trace.cpu().numpy(), stim)
                native = metrics(trace, stim["angle"], stim["intensity"], float(stim["speed"]), time_start=0)
                integrated = (current[:, native["window"]].sum(1) * 0.005).cpu().numpy()
                scores = np.stack([integrated[stim["intensity"] == (1 if r < 4 else 0), r].mean(0) for r in range(8)])
                arrays.update(current_score=scores, synapse_count=core.counts)
                write_record(base_path, arrays, metadata, signature)
                del trace, current
        grad_path = dest / "gradient.npz"
        if not valid_record(grad_path, signature):
            metadata = {"model": model, "stimulus": key, "kind": "first_order"}
            if previous is not None and (previous / "gradient.npz").exists():
                reuse(grad_path, previous / "gradient.npz", key, signature, metadata)
            else:
                gradient, details = gradient_bank(core, causal, state, stim["angle"], stim["intensity"], float(stim["speed"]), announce=lambda _: None)
                if not np.isfinite(gradient).all():
                    raise RuntimeError(f"Nonfinite gradient: {model}, {key}")
                write_record(grad_path, {"gradient": gradient}, {**metadata, **details}, signature)
        strengths = sorted(CONFIG["main_strengths"] + (CONFIG["development_extra_strengths"] if is_development else []))
        conditions = [(strength, [name], f"s{round(100*strength):03d}-cell{i:02d}")
                      for strength in strengths for i, name in enumerate(NAMES)]
        if is_development and key == "dev_fast":
            conditions.append((1.0, ["CT1(Lo1)", "CT1(M10)"], "s100-ct1_joint"))
        pending_conditions = []
        for strength, names, code in conditions:
            path = dest / (code + ".npz")
            if valid_record(path, signature):
                continue
            metadata = {"model": model, "stimulus": key, "strength": strength, "names": names,
                        "kind": "cell" if len(names) == 1 else "ct1_joint"}
            if previous is not None and (previous / path.name).exists():
                reuse(path, previous / path.name, key, signature, metadata)
            elif all(name in core.zeros for name in names):
                arrays = {k: v for k, v in load_npz(base_path).items() if k not in ("current_score", "synapse_count")}
                write_record(path, arrays, {**metadata, "structural_zero": True}, signature)
            else:
                pending_conditions.append((path, metadata))
        queue = [(condition, trajectory) for condition in range(len(pending_conditions)) for trajectory in range(24)]
        pending = {}
        batch_size = CONFIG["batch_size"]
        for begin in range(0, len(queue), batch_size):
            health()
            jobs = queue[begin:begin + batch_size]
            indices = [trajectory for _, trajectory in jobs]
            attenuation = torch.zeros((len(jobs), 65), device=device)
            for row, (condition, _) in enumerate(jobs):
                info = pending_conditions[condition][1]
                for name in info["names"]:
                    attenuation[row, NAMES.index(name)] = info["strength"]
            started = time.perf_counter()
            trace, _, _ = core.run(causal[indices], attenuation, state=state[indices].clone(), start=0)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            bank = trace.cpu().numpy()
            del trace
            for row, (condition, trajectory) in enumerate(jobs):
                if condition not in pending:
                    pending[condition] = [np.empty((24, frame_count, 8), np.float32), set(), 0.0]
                record = pending[condition]
                record[0][trajectory] = bank[row]
                record[1].add(trajectory)
                record[2] += elapsed / len(jobs)
                if len(record[1]) == 24:
                    completed, _, wall = pending.pop(condition)
                    path, info = pending_conditions[condition]
                    write_record(path, checked_arrays(completed, stim), {**info, "wall_seconds": wall}, signature)
            save_json(RESULTS / "progress.json", {
                "state": "running", "model": model, "stimulus": key,
                "completed_new_trajectories_in_panel": min(begin + batch_size, len(queue)),
                "new_trajectories_in_panel": len(queue), "utc": utc_now(),
            })
        if pending:
            raise RuntimeError("Incomplete trajectory assembly")
        save_json(dest / "complete.json", {"model": model, "stimulus": key,
                  "conditions": len(conditions), "causal_frames": frame_count,
                  "simulation_signature": signature, "utc": utc_now()})
        print(f"Panel complete: model {model}, {key}", flush=True)
        del state
        gc.collect()
    save_json(RESULTS / "runs" / model / "complete.json", {
        "model": model, "development": is_development,
        "elapsed_seconds_this_invocation": time.perf_counter() - model_start,
        "simulation_signature": signature, "utc": utc_now(),
    })
    print(f"Model {model} complete ({time.perf_counter()-model_start:.1f} seconds)", flush=True)
    del core, gray
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation", "all"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    original_freeze_check()
    signature = simulation_signature()
    if args.phase != "development":
        freeze_check()
        gate = json.loads((RESULTS / "verification/development-gate.json").read_text())
        if gate["status"] != "pass" or gate["simulation_signature"] != signature:
            raise RuntimeError("Development numerical gate must pass before confirmation")
    torch.set_num_threads(CONFIG["cpu_threads"])
    torch.set_num_interop_threads(CONFIG["cpu_threads"])
    if args.device == "cuda":
        if not os.environ.get("AI06_GPU_LOCK_FD"):
            raise RuntimeError("Launch through the portfolio GPU lock")
        os.fstat(int(os.environ["AI06_GPU_LOCK_FD"]))
        torch.cuda.set_per_process_memory_fraction(CONFIG["max_gpu_memory_fraction"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    models = CONFIG["development_models"] if args.phase == "development" else [m for m in MODELS if m not in CONFIG["development_models"]]
    if args.phase == "all":
        models = CONFIG["development_models"] + models
    started = time.perf_counter()
    save_json(RESULTS / f"{args.phase}-started.json", {"utc": utc_now(), "models": models,
              "simulation_signature": signature, "pid": os.getpid(), "device": args.device})
    try:
        for model in models:
            run_model(model, args.device, signature)
        save_json(RESULTS / f"{args.phase}-complete.json", {
            "status": "simulations_complete_pending_verification", "models": models,
            "utc": utc_now(), "elapsed_seconds": time.perf_counter() - started,
            "simulation_signature": signature,
        })
    except Exception as exc:
        save_json(RESULTS / f"{args.phase}-failure-{time.time_ns()}.json", {
            "error": repr(exc), "utc": utc_now(), "simulation_signature": signature,
        })
        raise


if __name__ == "__main__":
    main()
