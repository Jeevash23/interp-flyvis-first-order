"""Development-only pilot on model 000: numerical checks of the AI-09 code and timings for the runtime estimates.

No outcome of the study (error, bias, sign, precision) is computed here. The pilot checks that
  1. the batched baseline reproduces AI-06's saved baseline traces;
  2. the readout code reproduces AI-06's saved DSI;
  3. the forward tangent recursion reproduces AI-06's saved reverse-mode gradients through the peak DSI;
  4. replacement by zero reproduces AI-06's saved silencing traces, and mean replacement at strength 0 the baseline;
  5. (CPU, float64) the tangents J and K match central finite differences of the exact simulation;
and it times the GPU parts: exact mean replacement of 63 cell types at one strength, and the J+K tangent banks for
silencing and for mean replacement, on both stimuli. Usage:
  python -B pilot/pilot_checks.py cpu          (checks 2 and 5; CPU only)
  python -B study/code/gpu_lock.py <python> -B pilot/pilot_checks.py gpu   (checks 1, 3, 4 and timings)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "study/code"))

import numpy as np  # noqa: E402

from ai09_common import (CONFIG, NAMES, READOUTS, ROOT, ai06_record, causal_frames, configure_torch,  # noqa: E402
                         save_json, stimulus, utc_now)

MODEL = "000"
OUT = ROOT / "pilot/results"


def gate(estimate, reference):
    """AI-06's gradient gate: |difference| <= 1e-4 + 0.02 |reference|."""
    return np.abs(estimate - reference) <= 1e-4 + 0.02 * np.abs(reference)


def gpu_part():
    torch = configure_torch("cuda")
    from readouts import Readout
    from tangents import ReplacementCore, exact_bank, node_means, onset_state, tangent_bank
    torch.cuda.reset_peak_memory_stats()
    core = ReplacementCore(MODEL, "cuda")
    gray = core.gray()
    types = [i for i, name in enumerate(NAMES) if name not in core.zeros]
    record = {"model": MODEL, "utc_start": utc_now(), "nodes": core.n, "edges": core.e,
              "structural_zero_types": core.zeros, "types_simulated": len(types), "stimuli": {}}
    for key in CONFIG["stimuli"]:
        stim = stimulus(key)
        frames = stim["stimulus"][:, :, 0] if stim["stimulus"].ndim == 4 else stim["stimulus"]
        count = causal_frames(key)
        causal = torch.as_tensor(frames[:, 200:200 + count], device="cuda", dtype=torch.float32)
        state = onset_state(core, frames, gray)
        result = {"causal_frames": count}

        # 1. Baseline against AI-06's saved baseline trace.
        baseline, _, _ = core.run(causal, state=state.clone(), start=0)
        baseline = baseline.cpu().numpy()
        saved = ai06_record(MODEL, key, "baseline.npz")
        result["baseline_max_abs_difference"] = float(np.abs(baseline - saved["trace"]).max())

        # 3. Forward tangents through the peak DSI against AI-06's saved reverse-mode gradient.
        J, K, seconds = tangent_bank(core, causal, state, types, "zero", second=True,
                                     batch_types=CONFIG["tangent_batch_types"])
        result["zero_tangent_seconds"] = seconds
        readout = Readout(stim, count)
        base64 = torch.as_tensor(baseline, dtype=torch.float64)
        from torch.func import jvp
        slope = jvp(lambda s: readout.dsi(base64 + s * torch.as_tensor(J, dtype=torch.float64), "peak_dsi"),
                    (torch.zeros((), dtype=torch.float64),), (torch.ones((), dtype=torch.float64),))[1].detach().numpy()
        reference = ai06_record(MODEL, key, "gradient.npz")["gradient"][:, types].T  # (types, 8)
        result["gradient_max_abs_difference"] = float(np.abs(slope - reference).max())
        result["gradient_gate_pass_fraction"] = float(gate(slope, reference).mean())
        result["gradient_gate_failures"] = [
            {"cell": NAMES[types[i]], "readout": READOUTS[r], "forward": float(slope[i, r]),
             "ai06": float(reference[i, r])}
            for i, r in zip(*np.nonzero(~gate(slope, reference)))][:20]
        del J, K

        # 4. Replacement by zero equals AI-06's silencing; mean replacement at strength 0 equals the baseline.
        mean = node_means(core, causal, state)
        result["mean_output_summary"] = {"min": float(mean.min()), "max": float(mean.max()),
                                         "fraction_zero": float((mean == 0).float().mean())}
        checks = {}
        core.mean = torch.zeros_like(mean)
        for name in ("Mi1", "Tm3"):
            i = NAMES.index(name)
            traces, _ = exact_bank(core, causal, state, [(i, 1.0)], CONFIG["exact_batch_rows"])
            saved_trace = ai06_record(MODEL, key, f"s100-cell{i:02d}.npz")["trace"]
            checks[f"zero_replacement_{name}_max_abs_difference_vs_ai06"] = float(np.abs(traces[0] - saved_trace).max())
        core.mean = mean
        traces, _ = exact_bank(core, causal, state, [(NAMES.index("Mi1"), 0.0)], CONFIG["exact_batch_rows"])
        checks["mean_replacement_strength0_max_abs_difference_vs_baseline"] = float(np.abs(traces[0] - baseline).max())
        result["replacement_checks"] = checks

        # Timings: mean-replacement tangents and exact mean replacement of every type at one strength.
        J, K, seconds = tangent_bank(core, causal, state, types, "mean", mean=mean, second=True,
                                     batch_types=CONFIG["tangent_batch_types"])
        result["mean_tangent_seconds"] = seconds
        del J, K
        traces, seconds = exact_bank(core, causal, state, [(i, 1.0) for i in types], CONFIG["exact_batch_rows"])
        result["mean_exact_one_strength_seconds"] = seconds
        del traces
        core.mean = None
        record["stimuli"][key] = result
        print(key, {k: v for k, v in result.items() if k != "gradient_gate_failures"}, flush=True)
    record["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    record["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    record["utc_end"] = utc_now()
    save_json(OUT / "pilot-gpu.json", record)


def cpu_part():
    torch = configure_torch("cpu")
    torch.set_default_dtype(torch.float32)
    from readouts import Readout
    from tangents import ReplacementCore, node_means, onset_state, tangent_bank
    key = "dev_fast"
    stim = stimulus(key)
    frames = stim["stimulus"][:, :, 0] if stim["stimulus"].ndim == 4 else stim["stimulus"]
    count = causal_frames(key)
    record = {"model": MODEL, "stimulus": key, "utc_start": utc_now()}

    # 2. Readouts against AI-06's saved DSI (baseline and two silencing records).
    readout = Readout(stim, count)
    readout_checks = {}
    for name in ("baseline.npz", "s100-cell05.npz", "s050-cell20.npz"):
        saved = ai06_record(MODEL, key, name)
        mine = readout(torch.as_tensor(saved["trace"], dtype=torch.float64), "peak_dsi")
        readout_checks[name] = {"dsi_max_abs_difference": float(np.abs(mine["dsi"].numpy() - saved["dsi"]).max()),
                                "denominator_max_abs_difference": float(
                                    np.abs(mine["denominator"].numpy() - saved["denominator"]).max())}
    record["readout_checks"] = readout_checks

    # 5. Tangents against central finite differences, in float64 on the CPU.
    core = ReplacementCore(MODEL, "cpu")
    core.weight, core.bias, core.tau = core.weight.double(), core.bias.double(), core.tau.double()
    gray = core.gray().double()
    state = onset_state(core, frames.astype(np.float64), gray)
    rows = [0, 3, 6, 12, 15, 18]
    causal = torch.as_tensor(frames[rows, 200:200 + count], dtype=torch.float64)
    state = state[rows]
    mean = node_means(core, causal, state).double()
    from torch.func import jvp

    def compare(estimate, reference):
        scale = max(float(np.abs(reference).max()), 1e-12)
        difference = np.abs(estimate - reference)
        return {"reference_max_abs": scale, "max_abs_difference": float(difference.max()),
                "median_abs_difference": float(np.median(difference)),
                "fraction_within_1e-3_of_max": float(np.mean(difference <= 1e-3 * scale))}

    checks = {}
    for mode in ("zero", "mean"):
        core.mean = None if mode == "zero" else mean
        names = ("Mi1", "Tm3", "Mi4") if mode == "zero" else ("Mi1", "Tm1")
        indices = [NAMES.index(n) for n in names]
        J, K, _ = tangent_bank(core, causal, state, indices, mode, mean=mean if mode == "mean" else None,
                               second=True, batch_types=len(indices), dtype=torch.float64)
        for position, (name, i) in enumerate(zip(names, indices)):
            def trace_at(a, i=i):
                attenuation = a * torch.nn.functional.one_hot(torch.tensor(i), core.c).to(torch.float64)
                return core.run(causal, attenuation[None].expand(len(rows), -1), state=state.clone(), start=0)[0]

            # Automatic differentiation of AI-06's own update (nested forward mode, rectifier'' = 0).
            zero = torch.zeros((), dtype=torch.float64)
            one = torch.ones((), dtype=torch.float64)
            first_ad = jvp(trace_at, (zero,), (one,))[1].detach().numpy()
            second_ad = jvp(lambda a: jvp(trace_at, (a,), (one,))[1], (zero,), (one,))[1].detach().numpy()
            entry = {"J_vs_autodiff": compare(J[position], first_ad), "K_vs_autodiff": compare(K[position], second_ad)}
            # Central differences of the exact simulation at two steps; rectifier crossings inside +-h add
            # curvature that automatic differentiation does not see.
            for step in (1e-3, 1e-6):
                runs = {a: trace_at(torch.tensor(a, dtype=torch.float64)).numpy() for a in (-step, 0.0, step)}
                entry[f"J_vs_central_difference_h{step:g}"] = compare(J[position], (runs[step] - runs[-step]) / (2 * step))
                entry[f"K_vs_central_difference_h{step:g}"] = compare(
                    K[position], (runs[step] - 2 * runs[0.0] + runs[-step]) / step**2)
            checks[f"{mode}_{name}"] = entry
            print(mode, name, {k: (v["max_abs_difference"], v["fraction_within_1e-3_of_max"]) for k, v in entry.items()},
                  flush=True)
    record["tangent_checks"] = {"trajectories": rows, "dtype": "float64", "results": checks}
    record["utc_end"] = utc_now()
    save_json(OUT / "pilot-cpu.json", record)


if __name__ == "__main__":
    {"gpu": gpu_part, "cpu": cpu_part}[sys.argv[1]]()
