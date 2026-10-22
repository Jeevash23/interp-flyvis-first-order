"""Per-model GPU computations of AI-09 (resumable; refuses confirmation models until the freeze exists).

For each model and stimulus it writes, under study/results/runs/<model>/<stimulus>/:
  tangents-zero.npz   J and K of the eight readout traces for the 63 simulated cell types under silencing
  means.npz           per-cell average output used by mean replacement
  tangents-mean.npz   J and K under mean replacement
  exact-mean-sXXX.npz exact mean-replacement traces of the 63 types at each strength
  gates.json          the per-model gates of PROTOCOL.md Section 10
Each .npz has a .json sidecar with its SHA-256, the code signature and timings. Usage (from the project folder):
  python -B study/code/gpu_lock.py <python> -B study/code/run_arms.py --phase development
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import ai09_common
from ai09_common import (AI06, CONFIG, CONFIRMATION, DEVELOPMENT, NAMES, RESULTS, ROOT, STIMULI, ai06,
                         ai06_record, causal_frames, configure_torch, health, require_development, save_json,
                         sha256, stimulus, utc_now)

CODE_FILES = [ROOT / "study/code" / name for name in ("ai09_common.py", "tangents.py", "readouts.py", "run_arms.py")]
CODE_FILES += [ROOT / "study/config.json", AI06 / "02-Code/stage1/core.py", AI06 / "02-Code/stage1/common.py"]


def code_signature() -> str:
    return hashlib.sha256("\n".join(sha256(path) for path in CODE_FILES).encode()).hexdigest()


def valid(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["sha256"] != sha256(path) or metadata["code_signature"] != signature:
        raise RuntimeError(f"Existing AI-09 record differs from this code: {path}")
    return True


def write(path: Path, arrays: dict, metadata: dict, signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    save_json(path.with_suffix(".json"), {**metadata, "sha256": sha256(path), "code_signature": signature,
                                          "utc": utc_now()})


def gate(estimate, reference):
    return np.abs(estimate - reference) <= 1e-4 + 0.02 * np.abs(reference)


REPLAY_TOLERANCE = 1e-4  # Amendment 2: AI-06's fresh-replay rule (study/code/verify_revision.py)


def normalised_difference(trace, reference) -> float:
    """Largest absolute difference divided by max(1, largest absolute reference voltage), as AI-06's replay check."""
    return float(np.abs(trace - reference).max()) / max(1.0, float(np.abs(reference).max()))


def run_model(model: str, device: str, signature: str) -> None:
    torch = configure_torch(device)
    from torch.func import jvp
    from readouts import Readout
    from tangents import ReplacementCore, exact_bank, node_means, onset_state, tangent_bank

    require_development(model)
    ai09_common.RECORDS_LOCATION = RESULTS / "runs" / model
    health()
    folder = AI06 / f"03-Data/flyvis-root/results/flow/0000/{model}"
    expected = next(item["checkpoint_sha256"] for item in ai06.SELECTION["models"] if item["model"] == model)
    if sha256(folder / "best_chkpt") != expected:
        raise RuntimeError(f"Checkpoint changed: {model}")
    strengths = sorted(CONFIG["strengths"] + (CONFIG["development_extra_strengths"] if model in DEVELOPMENT else []))
    core = ReplacementCore(model, device)
    gray = core.gray()
    types = [i for i, name in enumerate(NAMES) if name not in core.zeros]
    for key in CONFIG["stimuli"]:
        if sha256(AI06 / STIMULI[key]["path"]) != STIMULI[key]["sha256"]:
            raise RuntimeError(f"Stimulus changed: {key}")
        dest = RESULTS / "runs" / model / key
        if (dest / "complete.json").exists():
            continue
        ai09_common.RECORDS_LOCATION = dest
        health()
        stim = stimulus(key)
        frames = stim["stimulus"][:, :, 0] if stim["stimulus"].ndim == 4 else stim["stimulus"]
        count = causal_frames(key)
        causal = torch.as_tensor(frames[:, 200:200 + count], device=device, dtype=torch.float32)
        state = onset_state(core, frames, gray)
        gates = {"model": model, "stimulus": key}

        core.mean = None
        baseline = core.run(causal, state=state.clone(), start=0)[0].cpu().numpy()
        saved_baseline = ai06_record(model, key, "baseline.npz")["trace"]
        gates["baseline_max_abs_difference"] = float(np.abs(baseline - saved_baseline).max())
        gates["baseline_normalised_difference"] = normalised_difference(baseline, saved_baseline)
        if gates["baseline_normalised_difference"] > REPLAY_TOLERANCE:
            raise RuntimeError(f"Baseline differs from AI-06: {model} {key}")

        path = dest / "tangents-zero.npz"
        if not valid(path, signature):
            J, K, seconds = tangent_bank(core, causal, state, types, "zero", batch_types=CONFIG["tangent_batch_types"])
            write(path, {"J": J, "K": K, "types": np.array(types)},
                  {"model": model, "stimulus": key, "kind": "tangents", "intervention": "zero", "seconds": seconds},
                  signature)
        J = np.load(path)["J"]
        readout = Readout(stim, count)
        base = torch.as_tensor(baseline, dtype=torch.float64)
        slope = jvp(lambda s: readout.dsi(base + s * torch.as_tensor(J, dtype=torch.float64), "peak_dsi"),
                    (torch.zeros((), dtype=torch.float64),), (torch.ones((), dtype=torch.float64),))[1].detach().numpy()
        reference = ai06_record(model, key, "gradient.npz")["gradient"][:, types].T
        gates["gradient_max_abs_difference"] = float(np.abs(slope - reference).max())
        gates["gradient_gate_pass"] = bool(gate(slope, reference).all())
        if not gates["gradient_gate_pass"]:
            raise RuntimeError(f"Forward derivatives disagree with AI-06: {model} {key}")
        del J

        path = dest / "means.npz"
        if not valid(path, signature):
            mean = node_means(core, causal, state)
            write(path, {"mean": mean.cpu().numpy()}, {"model": model, "stimulus": key, "kind": "means"}, signature)
        mean = torch.as_tensor(np.load(path)["mean"], device=device)

        core.mean = torch.zeros_like(mean)
        for name in ("Mi1", "Tm3"):
            i = NAMES.index(name)
            trace, _ = exact_bank(core, causal, state, [(i, 1.0)], CONFIG["exact_batch_rows"])
            saved = ai06_record(model, key, f"s100-cell{i:02d}.npz")["trace"]
            gates[f"zero_replacement_{name}_max_abs_difference"] = float(np.abs(trace[0] - saved).max())
            gates[f"zero_replacement_{name}_normalised_difference"] = normalised_difference(trace[0], saved)
            if gates[f"zero_replacement_{name}_normalised_difference"] > REPLAY_TOLERANCE:
                raise RuntimeError(f"Replacement by zero differs from AI-06: {model} {key} {name}")

        core.mean = mean
        path = dest / "tangents-mean.npz"
        if not valid(path, signature):
            J, K, seconds = tangent_bank(core, causal, state, types, "mean", mean=mean,
                                         batch_types=CONFIG["tangent_batch_types"])
            write(path, {"J": J, "K": K, "types": np.array(types)},
                  {"model": model, "stimulus": key, "kind": "tangents", "intervention": "mean", "seconds": seconds},
                  signature)
        for strength in strengths:
            path = dest / f"exact-mean-s{round(100 * strength):03d}.npz"
            if valid(path, signature):
                continue
            traces, seconds = exact_bank(core, causal, state, [(i, strength) for i in types], CONFIG["exact_batch_rows"])
            write(path, {"trace": traces, "types": np.array(types)},
                  {"model": model, "stimulus": key, "kind": "exact", "intervention": "mean", "strength": strength,
                   "seconds": seconds}, signature)
        core.mean = None
        save_json(dest / "gates.json", gates)
        save_json(dest / "complete.json", {"model": model, "stimulus": key, "code_signature": signature,
                                           "utc": utc_now()})
        print(f"{model} {key} complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    models = DEVELOPMENT if args.phase == "development" else CONFIRMATION
    signature = code_signature()
    started = time.perf_counter()
    for model in models:
        run_model(model, args.device, signature)
    save_json(RESULTS / f"{args.phase}-complete.json", {"models": models, "code_signature": signature,
              "elapsed_seconds": time.perf_counter() - started, "utc": utc_now()})


if __name__ == "__main__":
    main()
