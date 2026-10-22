"""Post hoc summaries for the AI-09 manuscript (written 17 October 2026, after the confirmation results were read).

Reads the saved records with the frozen analysis code (`study/code/analyse_arms.py`, imported, never changed) and
writes `analysis/results/posthoc-summary.json` and `analysis/results/cases-silencing-peak-s100.npz`:
- the scored cases of AI-06's setting (silencing, peak index) at full strength, with the exact change, AI-06's
  first-order estimate, the estimate from the exact readout on the linearised trajectory, the second-order estimate and
  the two parts of its second-order term (the network's, ½·∇F·K, and the readout's curvature, ½·J'(∇²F)J), and the
  linearised relative change of the index's denominator (for the scatter figure and Section 3.3 of the paper);
- per-model checks that these cases reproduce the frozen model-level table exactly;
- the largest gate values over the 46 confirmation models (Amendment 2).
The confirmation records are on the external drive (study/results/RUNS-LOCATION.md), which must be mounted.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "study/code"))
import torch  # noqa: E402
import analyse_arms as arms  # noqa: E402
from ai09_common import CONFIRMATION, RESULTS, READOUTS, NAMES, save_json, utc_now  # noqa: E402

OUT = ROOT / "analysis/results"
F64 = torch.float64
STRENGTH = 1.0


def denominator_slope(panel, J):
    """d(denominator)/da along each type's tangent, (types, 8): the linearised change of max(ON sum, OFF sum)."""
    from torch.func import jvp
    v0 = torch.as_tensor(panel.v0, dtype=F64)
    zero, one = torch.zeros((), dtype=F64), torch.ones((), dtype=F64)
    return jvp(lambda s: panel.readout(v0 + s * J, "peak_dsi")["denominator"], (zero,), (one,))[1].detach().numpy()


def model_cases(model, anatomy):
    columns = {k: [] for k in ("stimulus", "readout", "cell", "exact", "first", "linear_trajectory_exact_readout",
                               "second", "second_network", "second_readout", "relative_denominator_change")}
    for key in ("dev_fast", "final_slow"):
        panel = arms.Panel(model, key, anatomy)
        exact = panel.exact("zero", STRENGTH)
        cf = panel.counterfactuals("zero", "peak_dsi", STRENGTH, exact["traces"])
        types, J, K = panel.tangents["zero"]
        v0 = torch.as_tensor(panel.v0, dtype=F64)
        network = arms.full(arms.slope(panel.readout, v0, K, "peak_dsi").numpy(), types).T  # ∇F·K, (8, 65)
        second = panel.second["zero", "peak_dsi"]  # F''(0), (8, 65)
        first = STRENGTH * panel.gradient
        second_estimate = first + 0.5 * STRENGTH**2 * second
        relative = arms.full(denominator_slope(panel, J) / panel.base_denominator[None, :], types).T
        change = exact["dsi"]["peak_dsi"] - panel.base_peak_dsi[:, None]
        for r in range(8):
            if not panel.base_denominator[r] > arms.CUT or panel.base_peak_dsi[r] < arms.SELECTIVITY:
                continue
            eligible = ((exact["peak_denominator"][r] > arms.CUT) & np.isfinite(change[r]) & np.isfinite(first[r])
                        & (exact["amplification"] <= arms.AMPLIFICATION))
            for cell in np.flatnonzero(eligible):
                columns["stimulus"].append(key)
                columns["readout"].append(READOUTS[r])
                columns["cell"].append(NAMES[cell])
                columns["exact"].append(change[r, cell])
                columns["first"].append(first[r, cell])
                columns["linear_trajectory_exact_readout"].append(cf["linear_trajectory_exact_readout"][r, cell])
                columns["second"].append(second_estimate[r, cell])
                columns["second_network"].append(0.5 * STRENGTH**2 * network[r, cell])
                columns["second_readout"].append(0.5 * STRENGTH**2 * (second[r, cell] - network[r, cell]))
                columns["relative_denominator_change"].append(STRENGTH * relative[r, cell])
        del panel, exact
    return {k: np.array(v) for k, v in columns.items()}


def main():
    torch.set_num_threads(2)
    anatomy, _ = arms.ai06_scoring.structural_scores()
    frozen = {(r["model"], r["estimator"]): r for r in csv.DictReader(
        (RESULTS / "analysis/confirmation/models.csv").open())
        if r["strength"] == "1.0" and r["intervention"] == "zero" and r["readout"] == "peak_dsi"}
    pooled, per_model, worst = {}, [], 0.0
    for model in CONFIRMATION:
        cases = model_cases(model, anatomy)
        for estimator, column in (("first", "first"), ("linear_trajectory_exact_readout", "linear_trajectory_exact_readout"),
                                  ("second", "second")):
            mae = float(np.abs(cases[column] - cases["exact"]).mean())
            worst = max(worst, abs(mae - float(frozen[model, estimator]["mae"])))
        if len(cases["exact"]) != int(frozen[model, "first"]["effects"]):
            raise RuntimeError(f"Case count differs from the frozen table: {model}")
        network, readout = np.abs(cases["second_network"]), np.abs(cases["second_readout"])
        large = np.abs(cases["relative_denominator_change"]) > 1
        term = np.abs(cases["second_network"] + cases["second_readout"])
        per_model.append({
            "model": model, "cases": int(len(cases["exact"])),
            "readout_share_of_second_order_term": float(readout.sum() / (readout.sum() + network.sum())),
            "large_denominator_change_fraction": float(large.mean()),
            "large_denominator_change_share_of_term": float(term[large].sum() / term.sum()),
            "second_order_error_large": float(np.abs(cases["second"] - cases["exact"])[large].mean()) if large.any() else None,
            "second_order_error_rest": float(np.abs(cases["second"] - cases["exact"])[~large].mean()),
        })
        for k, v in cases.items():
            pooled.setdefault(k, []).append(v)
        pooled.setdefault("model", []).append(np.full(len(cases["exact"]), model))
        print(f"{model}: {len(cases['exact'])} cases", flush=True)
    if worst > 1e-12:
        raise RuntimeError(f"Cases do not reproduce the frozen model-level errors (largest difference {worst})")
    arrays = {k: np.concatenate(v) for k, v in pooled.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "cases-silencing-peak-s100.npz", **{k: v.astype(str) if v.dtype.kind in "OU" else v
                                                                    for k, v in arrays.items()})

    def ensemble(field):
        values = [m[field] for m in per_model if m[field] is not None]
        return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values)),
                "models": len(values)}

    gates, gate_files = {}, sorted((RESULTS / "runs").glob("*/*/gates.json"))
    for path in gate_files:
        record = json.loads(path.read_text())
        if record["model"] not in CONFIRMATION:
            continue
        for field, value in record.items():
            if isinstance(value, float) and (field not in gates or value > gates[field]["value"]):
                gates[field] = {"value": value, "model": record["model"], "stimulus": record["stimulus"]}
    save_json(OUT / "posthoc-summary.json", {
        "created": utc_now(), "label": "post hoc (written after the confirmation results were read)",
        "strength": STRENGTH, "cases": int(len(arrays["exact"])),
        "check_cases_reproduce_frozen_mae_largest_difference": worst,
        "readout_share_of_second_order_term": ensemble("readout_share_of_second_order_term"),
        "large_denominator_change_fraction": ensemble("large_denominator_change_fraction"),
        "large_denominator_change_share_of_term": ensemble("large_denominator_change_share_of_term"),
        "second_order_error_large": ensemble("second_order_error_large"),
        "second_order_error_rest": ensemble("second_order_error_rest"),
        "pooled_wrong_sign": {
            name: float(np.mean((arrays[name] * arrays["exact"] < 0)[np.abs(arrays["exact"]) >= arms.LARGE]))
            for name in ("first", "linear_trajectory_exact_readout")},
        "gates_confirmation": {"files": len([p for p in gate_files if json.loads(p.read_text())["model"] in CONFIRMATION]),
                               "largest": gates},
        "per_model": per_model,
    })


if __name__ == "__main__":
    main()
