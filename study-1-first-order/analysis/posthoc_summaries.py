"""Post hoc descriptive summaries of the saved confirmation and development tables.

Written on 9 October 2026, after the confirmation results had been read. Nothing here changes a frozen
result: the script only reads study/results/analysis/** (and, for the twelve responsive readout conditions whose
first-order estimates are all zero, their saved intact records and the stimulus files) and writes
analysis/results/posthoc-summary.json (`--check` compares without writing). Every quantity it adds is labelled post hoc
in the paper.

What it adds to the frozen analysis:
- counts of models in which first-order estimates are higher, equal (within 1e-12) or lower than each
  comparator, for the main screen, the inclusive screen and ranking among large effects;
- the paired inclusive-screen difference and the signed bias with the frozen model-level bootstrap
  (seed 20261007, 5,000 resamples of the 46 confirmation models); the protocol lists signed bias as a primary
  outcome, and the frozen analysis saved it per model but left it out of its ensemble summary;
- effect-level descriptions behind the predicted-versus-exact figure (pooled over models);
- response-amplitude ratios, abolition counts and readouts with all-zero first-order estimates;
- quartiles of the original pair-intervention contrasts (the frozen supplement saved medians only);
- medians of the frozen development-model table on silencing known T4/T5 input types (biological context only);
- (added on the evening of 9 October) wrong-sign counts split by the direction of the large change, amplitude
  ratios among large changes, the response pattern of readout conditions whose first-order estimates are all zero
  (read from the saved baseline records), and the inclusive random-ranking precision split by baseline DSI.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "study/results"
CONF = RES / "analysis/confirmation"
DEV = RES / "analysis/development"
OUT = ROOT / "analysis/results/posthoc-summary.json"
CONFIG = json.loads((ROOT / "study/config.json").read_text())
SEED, REPLICATES = CONFIG["bootstrap_seed"], CONFIG["bootstrap_replicates"]
TOL = 1e-12
MAIN_DSI, MAIN_AMP = CONFIG["primary_direction_selectivity_threshold"], CONFIG["primary_amplification_threshold"]
LARGE = CONFIG["important_effect_threshold"]
COMPARATORS = ["anatomy_hop1", "anatomy_hop2", "anatomy_hop3", "anatomy_hop4", "graph_distance",
               "trained_hop3", "input_current", "direct_count"]


def rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def num(value):
    return None if value in ("", None) else float(value)


def bootstrap(values):
    """The frozen analysis's model-level bootstrap (analyse_revision.estimate)."""
    values = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    rng = np.random.default_rng(SEED)
    means = values[rng.integers(0, len(values), (REPLICATES, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95": [float(low), float(high)], "min": float(values.min()),
            "max": float(values.max()), "models": int(len(values))}


def tally(differences):
    d = [x for x in differences if x is not None]
    return {"higher": sum(x > TOL for x in d), "equal": sum(abs(x) <= TOL for x in d),
            "lower": sum(x < -TOL for x in d), "models": len(d)}


def is_main(r):
    return float(r["dsi_threshold"]) == MAIN_DSI and r["amplification_limit"] != "" and float(r["amplification_limit"]) == MAIN_AMP


def is_inclusive(r):
    return float(r["dsi_threshold"]) == 0.0 and r["amplification_limit"] == ""


def model_comparisons(models, paired):
    out = {}
    for strength in CONFIG["main_strengths"]:
        key = f"{strength:g}"
        out[key] = {}
        main = [r for r in models if is_main(r) and float(r["strength"]) == strength]
        incl = [r for r in models if is_inclusive(r) and float(r["strength"]) == strength]
        grad_main = {r["model"]: r for r in main if r["method"] == "gradient"}
        grad_incl = {r["model"]: r for r in incl if r["method"] == "gradient"}
        for comparator in COMPARATORS:
            pairs = [r for r in paired if is_main(r) and float(r["strength"]) == strength and r["baseline"] == comparator]
            other_incl = {r["model"]: r for r in incl if r["method"] == comparator}
            inclusive_diff = [num(grad_incl[m]["composite_top10_precision"]) - num(other_incl[m]["composite_top10_precision"])
                              for m in sorted(grad_incl)]
            out[key][comparator] = {
                "main_precision": tally([num(r["top10_precision_difference"]) for r in pairs]),
                "large_effect_rho": tally([num(r["rho_important_difference"]) for r in pairs]),
                "inclusive_precision": tally(inclusive_diff),
                "inclusive_precision_difference": bootstrap(inclusive_diff),
            }
        g = [grad_main[m] for m in sorted(grad_main)]
        out[key]["gradient"] = {
            "bias": bootstrap([num(r["bias"]) for r in g]),
            "bias_positive_models": sum(num(r["bias"]) > 0 for r in g),
            "mae_vs_zero": tally([num(r["mae"]) - num(r["zero_mae"]) for r in g]),
            "clipped_vs_zero": tally([num(r["bounded_mae"]) - num(r["zero_mae"]) for r in g]),
            "large_effect_mae_vs_zero": tally([num(r["important_mae"]) - num(r["important_zero_mae"]) for r in g]),
            "opposite_sign_rate_range": [min(num(r["opposite_sign_rate"]) for r in g), max(num(r["opposite_sign_rate"]) for r in g)],
            "inclusive_mae": {f: float(np.mean([num(grad_incl[m][f]) for m in sorted(grad_incl)])) for f in ("mae", "zero_mae", "bounded_mae", "bias")},
        }
    return out


def effect_level(effects):
    out = {}
    for strength in CONFIG["main_strengths"]:
        sel = [r for r in effects if float(r["strength"]) == strength]
        responsive = [r for r in sel if r["defined"] == "True" or r["abolished"] == "True"]
        main = [r for r in sel if r["defined"] == "True" and float(r["baseline_dsi"]) >= MAIN_DSI
                and float(r["amplification_ratio"]) <= MAIN_AMP]
        x = np.array([float(r["exact"]) for r in main])
        y = np.array([float(r["prediction"]) for r in main])
        base = np.array([float(r["baseline_dsi"]) for r in main])
        large = np.abs(x) >= LARGE
        decrease, increase = large & (x < 0), large & (x > 0)
        main_ratio = np.array([float(r["preferred_amplitude_ratio"]) for r in main])
        ratio = np.array([float(r["preferred_amplitude_ratio"]) for r in responsive
                          if r["defined"] == "True" and r["preferred_amplitude_ratio"] != ""])
        abolished = [r for r in responsive if r["abolished"] == "True"]
        selective = [r for r in responsive if float(r["baseline_dsi"]) >= MAIN_DSI]
        amplified = [r for r in responsive if float(r["amplification_ratio"]) > MAIN_AMP]
        by_cell = {}
        for r in abolished:
            by_cell[r["cell"]] = by_cell.get(r["cell"], 0) + 1
        top = sorted(by_cell.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        out[f"{strength:g}"] = {
            "main_cases": int(len(main)), "large": int(large.sum()),
            "large_decreases": int(np.sum(large & (x < 0))),
            "large_wrong_sign": int(np.sum(large & (x * y < 0))),
            "outside_unit_interval": int(np.sum((base + y < 0) | (base + y > 1))),
            "abs_prediction_above_1": int(np.sum(np.abs(y) > 1)),
            "max_abs_prediction": float(np.abs(y).max()),
            "exact_range": [float(x.min()), float(x.max())],
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "responsive_cases": int(len(responsive)),
            "abolished": int(len(abolished)),
            "amplified_cases": int(len(amplified)),
            "amplified_interventions": len({(r["model"], r["stimulus"], r["cell"]) for r in amplified}),
            "selective_cases": int(len(selective)),
            "selective_abolished": sum(r["abolished"] == "True" for r in selective),
            "selective_amplified_defined": sum(r["defined"] == "True" and float(r["amplification_ratio"]) > MAIN_AMP for r in selective),
            "abolished_without_preferred_baseline": sum(r["preferred_amplitude_ratio"] == "" for r in abolished),
            "abolitions_by_cell_top": top,
            "defined_amplitude_ratio_quantiles": {str(q): float(np.quantile(ratio, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "defined_amplitude_ratio_below_half": float(np.mean(ratio < 0.5)),
            "defined_amplitude_ratio_above_two": float(np.mean(ratio > 2)),
            "defined_cases_with_ratio": int(len(ratio)),
            "large_increases": int(increase.sum()),
            "large_decreases_predicted_increase": int(np.sum(decrease & (y > 0))),
            "large_increases_predicted_decrease": int(np.sum(increase & (y < 0))),
            "large_amplitude_ratio_quartiles": [float(np.quantile(main_ratio[large], q)) for q in (0.25, 0.5, 0.75)],
            "large_decrease_amplitude_ratio_median": float(np.median(main_ratio[decrease])),
            "large_increase_amplitude_ratio_median": float(np.median(main_ratio[increase])),
        }
    return out


def response_pattern(r):
    """Directions of each polarity with a positive intact peak (saved baseline record of the readout condition)."""
    manifest = {s["key"]: s for s in json.loads((ROOT / "03-Data/stage1/stimulus-manifest.json").read_text())}
    with np.load(ROOT / manifest[r["stimulus"]]["path"]) as stim:
        intensity = stim["intensity"]
    with np.load(RES / "runs" / r["model"] / r["stimulus"] / "baseline.npz") as base:
        peaks = base["peaks"][:, CONFIG["readouts"].index(r["readout"])]
    preferred = 1 if r["readout"].startswith("T4") else 0
    return {"preferred_directions_responding": int(np.sum(peaks[intensity == preferred] > 0)),
            "other_directions_responding": int(np.sum(peaks[intensity != preferred] > 0))}


def zero_gradient(failures):
    sel = [r for r in failures if float(r["strength"]) == 1.0 and r["gradient_zero"] == "True"]
    responsive = [r for r in sel if r["baseline_responsive"] == "True"]
    patterns = [{"model": r["model"], "stimulus": r["stimulus"], "readout": r["readout"], "baseline_dsi": float(r["baseline_dsi"]),
                 **response_pattern(r)} for r in responsive]
    return {"readout_conditions": len(sel), "unresponsive": len(sel) - len(responsive), "responsive": len(responsive),
            "responsive_patterns": patterns,
            "responsive_dsi_zero": sum(float(r["baseline_dsi"]) < 1e-9 for r in responsive),
            "main_set": [{"model": r["model"], "stimulus": r["stimulus"], "readout": r["readout"],
                          "baseline_dsi": float(r["baseline_dsi"]), "abolitions_full": int(r["response_abolitions"])}
                         for r in responsive if float(r["baseline_dsi"]) >= MAIN_DSI]}


def pairs():
    data = [r for r in rows(RES / "analysis/supplement/pairs.csv") if float(r["strength"]) == 1.0 and r["primary_eligible"] == "True"]
    out = []
    for name in dict.fromkeys(r["pair"] for r in data):
        g = [r for r in data if r["pair"] == name]
        q = lambda field, f=abs: [float(np.quantile([f(float(r[field])) for r in g], p)) for p in (0.25, 0.5, 0.75)]
        out.append({"pair": name, "readout_conditions": len(g), "models": len({r["model"] for r in g}),
                    "abs_dsi_interaction_quartiles": q("dsi_interaction"),
                    "abs_dsi_interaction_max": max(abs(float(r["dsi_interaction"])) for r in g),
                    "voltage_rms_quartiles": q("voltage_interaction_rms", float),
                    "relative_voltage_quartiles": q("relative_voltage_interaction", float),
                    "same_direction_singles": sum(r["same_direction_singles"] == "True" for r in g),
                    "subadditive_dsi": sum(r["subadditive_dsi"] == "True" for r in g)})
    return out


def biology_context():
    """Development models, faster common stimulus: silencing known T4/T5 input types (frozen supplement table)."""
    data = rows(RES / "analysis/supplement/common-stimulus-biology-context.csv")
    out = []
    for cell in dict.fromkeys(r["cell"] for r in data):
        for family in ("T4", "T5"):
            g = [r for r in data if r["cell"] == cell and r["readout"].startswith(family)]
            ratio = [float(r["preferred_polarity_amplitude_ratio"]) for r in g if r["preferred_polarity_amplitude_ratio"] != ""]
            change = [float(r["dsi_change"]) for r in g if r["dsi_change"] != ""]
            out.append({"cell": cell, "readouts": family, "readout_conditions": len(g), "models": len({r["model"] for r in g}),
                        "amplitude_ratio_median": float(np.median(ratio)) if ratio else None,
                        "amplitude_ratio_range": [min(ratio), max(ratio)] if ratio else None,
                        "dsi_defined": len(change), "dsi_change_median": float(np.median(change)) if change else None})
    return out


def inclusive_random(units):
    """Expected inclusive-screen precision of a random ranking: readout-condition mean within model, then model mean."""
    out = {}
    for strength in CONFIG["main_strengths"]:
        per_model = {}
        for r in units:
            if r["method"] == "gradient" and float(r["strength"]) == strength and is_inclusive(r):
                per_model.setdefault(r["model"], []).append(float(r["composite_top10_random_precision"]))
        out[f"{strength:g}"] = float(np.mean([np.mean(v) for _, v in sorted(per_model.items())]))
        for label, keep in (("below", lambda d: d < MAIN_DSI), ("at_least", lambda d: d >= MAIN_DSI)):
            split = {}
            for r in units:
                if (r["method"] == "gradient" and float(r["strength"]) == strength and is_inclusive(r)
                        and keep(float(r["baseline_dsi"]))):
                    split.setdefault(r["model"], []).append(float(r["composite_top10_random_precision"]))
            out[f"{strength:g}_baseline_dsi_{label}_{MAIN_DSI:g}"] = {
                "precision": float(np.mean([np.mean(v) for _, v in sorted(split.items())])),
                "models": len(split), "readout_conditions": sum(len(v) for v in split.values())}
    return out


def development(models):
    out = {}
    main = [r for r in models if is_main(r) and r["method"] == "gradient"]
    for strength in sorted({float(r["strength"]) for r in main}):
        g = [r for r in main if float(r["strength"]) == strength]
        out[f"{strength:g}"] = {"gradient_better_than_zero": sum(num(r["mae"]) < num(r["zero_mae"]) - TOL for r in g),
                                "models": len(g)}
    return out


def main():
    models, paired = rows(CONF / "models.csv"), rows(CONF / "paired-models.csv")
    result = {
        "status": "post hoc; descriptive summaries of saved outputs, written after the confirmation results were read",
        "sources": ["study/results/analysis/confirmation/models.csv", "study/results/analysis/confirmation/paired-models.csv",
                    "study/results/analysis/confirmation/effects.csv", "study/results/analysis/confirmation/failures.csv",
                    "study/results/analysis/confirmation/units.csv", "study/results/analysis/development/models.csv",
                    "study/results/analysis/supplement/pairs.csv",
                    "study/results/analysis/supplement/common-stimulus-biology-context.csv",
                    "study/results/runs/<model>/<stimulus>/baseline.npz (readout conditions with all-zero estimates)",
                    "03-Data/stage1/stimulus-manifest.json and the stimulus files it lists"],
        "tie_tolerance": TOL, "bootstrap": {"seed": SEED, "replicates": REPLICATES, "unit": "model"},
        "comparisons": model_comparisons(models, paired),
        "effects": effect_level(rows(CONF / "effects.csv")),
        "zero_gradient": zero_gradient(rows(CONF / "failures.csv")),
        "inclusive_random_precision": inclusive_random(rows(CONF / "units.csv")),
        "pairs": pairs(),
        "biology_context": biology_context(),
        "development": development(rows(DEV / "models.csv")),
    }
    text = json.dumps(result, indent=1) + "\n"
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT.relative_to(ROOT)} is out of date")
        print(f"ok  {OUT.relative_to(ROOT)} matches the saved outputs")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists() or OUT.read_text() != text:
        OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
