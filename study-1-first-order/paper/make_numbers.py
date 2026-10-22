"""Write paper/generated/numbers.tex and the generated tables from saved study outputs.

Every number in the manuscript that comes from the study is a macro defined here, computed from
study/results/** (the frozen confirmation and development analyses, verification records and run records),
analysis/results/posthoc-summary.json (post hoc summaries of those tables) and the study's input records.
`--check` recomputes everything in memory and fails if a committed file differs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RES = ROOT / "study/results"
CONF = RES / "analysis/confirmation"
DEV = RES / "analysis/development"
SUPP = RES / "analysis/supplement"
CONFIG = json.loads((ROOT / "study/config.json").read_text())
POST = json.loads((ROOT / "analysis/results/posthoc-summary.json").read_text())
MAIN_DSI, MAIN_AMP = CONFIG["primary_direction_selectivity_threshold"], CONFIG["primary_amplification_threshold"]
CLUSTER_NAMES = {0: "Weakly tuned", 1: "Tuned upward", 2: "Tuned downward"}
CLUSTER_SHORT = {0: "weak", 1: "up", 2: "down"}
METHODS = [("gradient", "First-order estimate"), ("anatomy_hop1", "Anatomy, 1 step"), ("anatomy_hop2", "Anatomy, 2 steps"),
           ("anatomy_hop3", "Anatomy, 3 steps"), ("anatomy_hop4", "Anatomy, 4 steps"), ("graph_distance", "Graph distance"),
           ("trained_hop3", "Trained weights, 3 steps"), ("input_current", "Input current"), ("direct_count", "Direct synapse counts")]
COMPARATORS = [("anatomy_hop3", "Anatomy, 3 steps"), ("anatomy_hop4", "Anatomy, 4 steps"), ("anatomy_hop2", "Anatomy, 2 steps"),
               ("graph_distance", "Graph distance"), ("trained_hop3", "Trained weights, 3 steps"),
               ("input_current", "Input current"), ("direct_count", "Direct synapse counts")]


# ---------------------------------------------------------------- formatting
def f(x, d=3):
    text = f"{x:.{d}f}"
    if float(text) == 0:
        text = f"{0:.{d}f}"
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def signed(x, d=3):
    text = f(x, d)
    return text if text.startswith("\\ensuremath{-}") or float(f"{x:.{d}f}") == 0 else "\\ensuremath{+}" + text


def pct(x, d=1):
    return f(100 * x, d)


def count(n):
    return f"{int(n):,}"


def ci(pair, d=3):
    return f"[{f(pair[0], d)}, {f(pair[1], d)}]"


def sci(x, d=1):
    mant, exp = f"{x:.{d}e}".split("e")
    return f"\\ensuremath{{{mant}\\times 10^{{{int(exp)}}}}}"


# ---------------------------------------------------------------- loading
def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def num(v):
    return None if v in ("", None) else float(v)


def is_main(r):
    return float(r["dsi_threshold"]) == MAIN_DSI and r["amplification_limit"] != "" and float(r["amplification_limit"]) == MAIN_AMP


SUMMARY = {(r["strength"], r["method"]): r for r in read(CONF / "summary.json")["summary"]}
DEV_SUMMARY = {(r["strength"], r["method"]): r for r in read(DEV / "summary.json")["summary"]}
SENS = read(CONF / "sensitivity.json")
PAIRED = {(r["strength"], r["baseline"]): r for r in read(CONF / "paired-summary.json")}
MAGDIFF = {r["strength"]: r for r in read(CONF / "magnitude-differences.json")}
DEV_MAGDIFF = {r["strength"]: r for r in read(DEV / "magnitude-differences.json")}
TIMING = read(CONF / "timing.json")
CLUSTERS = read(CONF / "cluster-summary.json")
MODELS = rows(CONF / "models.csv")
DEV_MODELS = rows(DEV / "models.csv")
FAILURES = rows(CONF / "failures.csv")
DIAG = rows(CONF / "diagnostics.csv")


def mean(strength, method, field, source=SUMMARY):
    return source[(strength, method)][field]["mean"]


def interval(strength, method, field):
    return SUMMARY[(strength, method)][field]["ensemble_bootstrap_95"]


def sens(strength, threshold, amp, method):
    return next(r for r in SENS if r["strength"] == strength and r["dsi_threshold"] == threshold
                and r["amplification_limit"] == amp and r["method"] == method)


def diag_mean(strength, stratum, diagnostic, category, field):
    sel = [r for r in DIAG if float(r["strength"]) == strength and r["stratum"] == stratum
           and r["diagnostic"] == diagnostic and r["category"] == category]
    return float(np.mean([float(r[field]) for r in sel])), sum(int(r["effects"]) for r in sel), len(sel)


# ---------------------------------------------------------------- macros
def stimuli_path(key):
    return next(s["path"] for s in read(ROOT / "03-Data/stage1/stimulus-manifest.json") if s["key"] == key)


def numbers():
    n = {}
    net = read(ROOT / "03-Data/stage1/network-structure.json")
    stimuli = {s["key"]: s for s in read(ROOT / "03-Data/stage1/stimulus-manifest.json")}
    freeze = read(ROOT / "study/freeze.json")
    # Design constants, read from the frozen configuration and input records
    selection = read(ROOT / "03-Data/stage1/model-selection.json")["models"]
    n["NModels"] = str(len(selection))
    n["NDevModels"] = str(len(CONFIG["development_models"]))
    n["NConfModels"] = str(len(selection) - len(CONFIG["development_models"]))
    n["NReadoutTypes"] = str(len(CONFIG["readouts"]))
    n["Budget"] = str(CONFIG["screening_budget"])
    n["LargeThreshold"] = f"{CONFIG['important_effect_threshold']:g}"
    n["MainDSI"] = f"{MAIN_DSI:g}"
    n["AmpLimit"] = f"{MAIN_AMP:g}"
    n["AmpLimitLow"] = f"{min(CONFIG['amplification_thresholds']):g}"
    n["AmpLimitHigh"] = f"{max(CONFIG['amplification_thresholds']):g}"
    n["DSIThresholdsOther"] = ", ".join(f"{x:g}" for x in CONFIG["direction_selectivity_thresholds"] if x != MAIN_DSI)
    n["DenomThreshold"] = sci(CONFIG["denominator_threshold"], 0).replace("1\\times ", "")
    n["BootstrapReps"] = count(CONFIG["bootstrap_replicates"])
    n["BootstrapSeed"] = str(CONFIG["bootstrap_seed"])
    n["ExtraStrengths"] = ", ".join(f"{x:g}" for x in CONFIG["development_extra_strengths"])
    dev = sorted(CONFIG["development_models"])
    n["DevModelList"] = ", ".join(dev[:-1]) + " and " + dev[-1]
    n["CPUThreads"] = str(CONFIG["cpu_threads"])
    n["BatchSize"] = str(CONFIG["batch_size"])
    n["TieTolerance"] = sci(POST["tie_tolerance"], 0).replace("1\\times ", "")
    stim = np.load(ROOT / stimuli_path("dev_fast"))
    n["NDirections"] = str(len(set(stim["angle"].tolist())))
    n["NTrajectories"] = str(len(stim["angle"]))
    n["NetNodes"] = count(net["nodes"])
    n["NetEdges"] = count(net["edges"])
    n["NetTypes"] = str(len(net["model_types"]))
    n["NetTypePairs"] = str(len(net["realized_pairs"]))
    fast, slow = stimuli["dev_fast"], stimuli["final_slow"]
    n["SpeedFast"] = f"{fast['degrees_per_second']:.1f}"
    n["SpeedSlow"] = f"{slow['degrees_per_second']:.2f}"
    n["WindowFast"] = f"{fast['motion_window_end_s']:.2f}"
    n["WindowSlow"] = f"{slow['motion_window_end_s']:.2f}"
    n["FramesFast"] = str(int(np.floor(fast["motion_window_end_s"] / 0.005 + 1e-10)) + 1)
    n["FramesSlow"] = str(int(np.floor(slow["motion_window_end_s"] / 0.005 + 1e-10)) + 1)
    n["FrozenFiles"] = str(len(freeze["files"]))
    n["FreezeTime"] = freeze["utc"][11:16]
    n["OriginalFiles"] = count(len(read(ROOT / "provenance/original-records.json")["files"]))
    completed = read(RES / "pipeline.json")["completed"]
    n["ConfirmationDoneTime"] = completed[11:16]

    # Coverage (confirmation, both strengths share the same readout conditions)
    full = [r for r in FAILURES if float(r["strength"]) == 1.0]
    responsive = [r for r in full if r["baseline_responsive"] == "True"]
    n["ReadoutConditions"] = str(len(full))
    n["Responsive"] = str(len(responsive))
    n["Unresponsive"] = str(len(full) - len(responsive))
    n["LowDSI"] = str(sum(float(r["baseline_dsi"]) < MAIN_DSI for r in responsive))
    n["MainReadouts"] = str(sum(float(r["baseline_dsi"]) >= MAIN_DSI for r in responsive))
    e1, e5 = POST["effects"]["1"], POST["effects"]["0.5"]
    n["ResponsiveCases"] = count(e1["responsive_cases"])
    n["SelectiveCases"] = count(e1["selective_cases"])
    n["AbolishedFull"] = str(e1["abolished"])
    n["AbolishedHalf"] = str(e5["abolished"])
    n["AbolishedFullPct"] = pct(e1["abolished"] / e1["responsive_cases"])
    n["AbolishedSelectiveFull"] = str(e1["selective_abolished"])
    n["AbolishedSelectiveHalf"] = str(e5["selective_abolished"])
    n["AbolishedNoPreferred"] = str(e1["abolished_without_preferred_baseline"])
    n["AbolitionReadoutsFull"] = str(sum(int(r["response_abolitions"]) > 0 for r in full))
    top = dict(e1["abolitions_by_cell_top"])
    n["AbolishedByMiOne"] = str(top["Mi1"])
    n["AbolishedByLOne"] = str(top["L1"])
    n["AbolishedByMiFour"] = str(top["Mi4"])
    n["AbolishedByTmThree"] = str(top["Tm3"])
    n["AmplifiedInterventionsFull"] = str(e1["amplified_interventions"])
    n["AmplifiedInterventionsHalf"] = str(e5["amplified_interventions"])
    n["AmplifiedCasesFull"] = str(e1["amplified_cases"])
    n["AmplifiedCasesHalf"] = str(e5["amplified_cases"])
    n["AmplifiedSelectiveFull"] = str(e1["selective_amplified_defined"])
    n["AmplifiedSelectiveHalf"] = str(e5["selective_amplified_defined"])
    n["MainCasesFull"] = count(e1["main_cases"])
    n["MainCasesHalf"] = count(e5["main_cases"])
    n["LargeFull"] = count(e1["large"])
    n["LargeHalf"] = count(e5["large"])
    n["LargeFullPct"] = pct(e1["large"] / e1["main_cases"])
    n["LargeDecreasePct"] = pct(e1["large_decreases"] / e1["large"], 0)
    n["AmpRatioLowQ"] = f(e1["defined_amplitude_ratio_quantiles"]["0.05"], 2)
    n["AmpRatioHighQ"] = f(e1["defined_amplitude_ratio_quantiles"]["0.95"], 2)
    n["AmpRatioMedian"] = f(e1["defined_amplitude_ratio_quantiles"]["0.5"], 2)
    n["AmpRatioQOne"] = f(e1["defined_amplitude_ratio_quantiles"]["0.25"], 2)
    n["AmpRatioQThree"] = f(e1["defined_amplitude_ratio_quantiles"]["0.75"], 2)
    n["AmpRatioBelowHalfPct"] = pct(e1["defined_amplitude_ratio_below_half"])
    n["AmpRatioAboveTwoPct"] = pct(e1["defined_amplitude_ratio_above_two"])
    zg = POST["zero_gradient"]
    n["ZeroGradReadouts"] = str(zg["readout_conditions"])
    n["ZeroGradResponsiveSilent"] = str(zg["responsive_dsi_zero"])
    n["ZeroGradSelective"] = str(len(zg["main_set"]))
    abol = sorted(r["abolitions_full"] for r in zg["main_set"])
    n["ZeroGradAbolitionsRange"] = f"{abol[0]}--{abol[-1]}"
    # The text describes the response pattern of these readout conditions; check it against the saved baselines.
    silent = [z for z in zg["responsive_patterns"] if z["baseline_dsi"] < 1e-9]
    single = [z for z in zg["responsive_patterns"] if z["baseline_dsi"] >= MAIN_DSI]
    assert len(silent) == zg["responsive_dsi_zero"] and all(z["preferred_directions_responding"] == 0 for z in silent)
    assert len(single) == len(zg["main_set"]) and all(z["preferred_directions_responding"] == 1 and z["baseline_dsi"] > 1 - 1e-9
                                                       for z in single)
    assert len(silent) + len(single) == zg["responsive"]
    n["LargeDecFull"] = count(e1["large_decreases"])
    n["LargeIncFull"] = count(e1["large_increases"])
    n["LargeDecPredIncPct"] = pct(e1["large_decreases_predicted_increase"] / e1["large_decreases"])
    n["LargeIncPredDecPct"] = pct(e1["large_increases_predicted_decrease"] / e1["large_increases"])
    q1, q2, q3 = e1["large_amplitude_ratio_quartiles"]
    n["AmpRatioLargeQOne"], n["AmpRatioLargeMedian"], n["AmpRatioLargeQThree"] = f(q1, 2), f(q2, 2), f(q3, 2)
    n["AmpRatioLargeDecMedian"] = f(e1["large_decrease_amplitude_ratio_median"], 2)
    n["AmpRatioLargeIncMedian"] = f(e1["large_increase_amplitude_ratio_median"], 2)

    # Screening, full and half strength
    for s, tag in [(1.0, "Full"), (0.5, "Half")]:
        for method, short in [("gradient", "FO"), ("anatomy_hop1", "AnatOne"), ("anatomy_hop2", "AnatTwo"),
                              ("anatomy_hop3", "Anat"), ("anatomy_hop4", "AnatFour"), ("graph_distance", "Dist"),
                              ("trained_hop3", "Trained"), ("input_current", "Current"), ("direct_count", "Direct")]:
            n[f"Prec{short}{tag}"] = f(mean(s, method, "top10_precision"))
            n[f"Rho{short}{tag}"] = f(mean(s, method, "rho"))
            n[f"RhoLarge{short}{tag}"] = f(mean(s, method, "rho_important"))
            n[f"Recall{short}{tag}"] = f(mean(s, method, "top10_recall"))
        n[f"PrecRandom{tag}"] = f(mean(s, "gradient", "top10_random_precision"))
        n[f"RecallRandom{tag}"] = f(mean(s, "gradient", "top10_random_recall"))
        n[f"PrecFOCI{tag}"] = ci(interval(s, "gradient", "top10_precision"))
        n[f"PrecAnatCI{tag}"] = ci(interval(s, "anatomy_hop3", "top10_precision"))
        pa = PAIRED[(s, "anatomy_hop3")]
        n[f"PrecDiffAnat{tag}"] = f(pa["top10_precision_difference"]["mean"])
        n[f"PrecDiffAnatCI{tag}"] = ci(pa["top10_precision_difference"]["ensemble_bootstrap_95"])
        n[f"RhoLargeDiffAnat{tag}"] = f(pa["rho_important_difference"]["mean"])
        n[f"RhoLargeDiffAnatCI{tag}"] = ci(pa["rho_important_difference"]["ensemble_bootstrap_95"])
        n[f"RhoDiffAnat{tag}"] = f(pa["rho_difference"]["mean"])
        comp = POST["comparisons"][f"{s:g}"]
        t = comp["anatomy_hop3"]
        n[f"PrecWinsAnat{tag}"] = str(t["main_precision"]["higher"])
        n[f"PrecTiesAnat{tag}"] = str(t["main_precision"]["equal"])
        n[f"PrecLossesAnat{tag}"] = str(t["main_precision"]["lower"])
        n[f"RhoLargeWinsAnat{tag}"] = str(t["large_effect_rho"]["higher"])
        n[f"RhoLargeModelsAnat{tag}"] = str(t["large_effect_rho"]["models"])
    pa4 = PAIRED[(1.0, "anatomy_hop4")]
    n["PrecDiffAnatFourFull"] = f(pa4["top10_precision_difference"]["mean"])
    n["PrecDiffAnatFourCIFull"] = ci(pa4["top10_precision_difference"]["ensemble_bootstrap_95"])
    pt = PAIRED[(1.0, "trained_hop3")]
    n["RhoLargeDiffTrainedFull"] = f(pt["rho_important_difference"]["mean"])
    n["RhoLargeDiffTrainedCIFull"] = ci(pt["rho_important_difference"]["ensemble_bootstrap_95"])
    n["RhoLargeWinsTrainedFull"] = str(POST["comparisons"]["1"]["trained_hop3"]["large_effect_rho"]["higher"])
    n["PrecWinsAnatFourFull"] = str(POST["comparisons"]["1"]["anatomy_hop4"]["main_precision"]["higher"])
    n["RhoUnitsFull"] = str(PAIRED[(1.0, "anatomy_hop3")]["rho_important_difference_units"])
    # Inclusive screen (all responsive readout conditions, abolitions count as large, no amplification exclusion)
    for s, tag in [(1.0, "Full"), (0.5, "Half")]:
        for method, short in [("gradient", "FO"), ("anatomy_hop3", "Anat"), ("anatomy_hop4", "AnatFour"),
                              ("trained_hop3", "Trained"), ("direct_count", "Direct"), ("input_current", "Current")]:
            n[f"Incl{short}{tag}"] = f(sens(s, 0.0, None, method)["composite_top10_precision"]["mean"])
        n[f"InclFOCI{tag}"] = ci(sens(s, 0.0, None, "gradient")["composite_top10_precision"]["ensemble_bootstrap_95"])
        n[f"InclAnatCI{tag}"] = ci(sens(s, 0.0, None, "anatomy_hop3")["composite_top10_precision"]["ensemble_bootstrap_95"])
        t = POST["comparisons"][f"{s:g}"]["anatomy_hop3"]
        n[f"InclDiffAnat{tag}"] = f(t["inclusive_precision_difference"]["mean"])
        n[f"InclDiffAnatCI{tag}"] = ci(t["inclusive_precision_difference"]["ci95"])
        n[f"InclWinsAnat{tag}"] = str(t["inclusive_precision"]["higher"])
        n[f"InclTiesAnat{tag}"] = str(t["inclusive_precision"]["equal"])
        n[f"InclLossesAnat{tag}"] = str(t["inclusive_precision"]["lower"])
        n[f"InclRandom{tag}"] = f(POST["inclusive_random_precision"][f"{s:g}"])
        n[f"InclRandomLowDSI{tag}"] = f(POST["inclusive_random_precision"][f"{s:g}_baseline_dsi_below_{MAIN_DSI:g}"]["precision"])
        n[f"InclRandomMainDSI{tag}"] = f(POST["inclusive_random_precision"][f"{s:g}_baseline_dsi_at_least_{MAIN_DSI:g}"]["precision"])
        n[f"InclReadouts{tag}"] = str(sens(s, 0.0, None, "gradient")["units"])
    n["CompositeFOFull"] = f(mean(1.0, "gradient", "composite_top10_precision"))

    # Effect sizes
    for s, tag in [(1.0, "Full"), (0.5, "Half")]:
        n[f"MaeFO{tag}"] = f(mean(s, "gradient", "mae"))
        n[f"MaeZero{tag}"] = f(mean(s, "gradient", "zero_mae"))
        n[f"MaeClip{tag}"] = f(mean(s, "gradient", "bounded_mae"))
        n[f"MaeFOCI{tag}"] = ci(interval(s, "gradient", "mae"))
        n[f"MaeZeroCI{tag}"] = ci(interval(s, "gradient", "zero_mae"))
        n[f"MaeLargeFO{tag}"] = f(mean(s, "gradient", "important_mae"))
        n[f"MaeLargeZero{tag}"] = f(mean(s, "gradient", "important_zero_mae"))
        n[f"WrongSign{tag}"] = pct(mean(s, "gradient", "opposite_sign_rate"))
        n[f"WrongSignCI{tag}"] = ci([100 * x for x in interval(s, "gradient", "opposite_sign_rate")], 1)
        n[f"ZeroPred{tag}"] = pct(mean(s, "gradient", "zero_prediction_rate"))
        md = MAGDIFF[s]
        n[f"MaeDiff{tag}"] = f(md["gradient_minus_zero"]["mean"])
        n[f"MaeDiffCI{tag}"] = ci(md["gradient_minus_zero"]["ensemble_bootstrap_95"])
        n[f"ClipDiff{tag}"] = f(md["bounded_minus_zero"]["mean"])
        n[f"ClipDiffCI{tag}"] = ci(md["bounded_minus_zero"]["ensemble_bootstrap_95"])
        g = POST["comparisons"][f"{s:g}"]["gradient"]
        n[f"MaeWorse{tag}"] = str(g["mae_vs_zero"]["higher"])
        n[f"MaeBetter{tag}"] = str(g["mae_vs_zero"]["lower"])
        n[f"ClipBetter{tag}"] = str(g["clipped_vs_zero"]["lower"])
        n[f"MaeLargeWorse{tag}"] = str(g["large_effect_mae_vs_zero"]["higher"])
        n[f"Bias{tag}"] = f(g["bias"]["mean"])
        n[f"BiasCI{tag}"] = ci(g["bias"]["ci95"])
        n[f"BiasMin{tag}"] = f(g["bias"]["min"])
        n[f"BiasMax{tag}"] = f(g["bias"]["max"])
        n[f"BiasPositive{tag}"] = str(g["bias_positive_models"])
        n[f"WrongSignMin{tag}"] = pct(g["opposite_sign_rate_range"][0])
        n[f"WrongSignMax{tag}"] = pct(g["opposite_sign_rate_range"][1])
        n[f"Outside{tag}"] = pct(diag_mean(s, "primary", "baseline_selectivity", "at_least_0.2", "infeasible_prediction_fraction")[0])
        e = POST["effects"][f"{s:g}"]
        n[f"AbsPredAboveOne{tag}"] = str(e["abs_prediction_above_1"])
        n[f"MaxAbsPred{tag}"] = f"{e['max_abs_prediction']:.1f}"
        n[f"PearsonPooled{tag}"] = f(e["pearson_r"], 2)
        n[f"WrongSignPooled{tag}"] = count(e["large_wrong_sign"])
    gi = POST["comparisons"]["1"]["gradient"]["inclusive_mae"]
    n["InclMaeFOFull"] = f(gi["mae"])
    n["InclMaeZeroFull"] = f(gi["zero_mae"])
    n["InclMaeClipFull"] = f(gi["bounded_mae"])
    for cat, tag in [("important", "Large"), ("smaller", "Small")]:
        m, eff, _ = diag_mean(1.0, "primary", "effect_size", cat, "median_absolute_error")
        n[f"MedianErr{tag}Full"] = f(m, 3)
    m, _, _ = diag_mean(1.0, "primary", "direct_input", "direct", "mae")
    n["MaeDirectFull"] = f(m)
    m, _, _ = diag_mean(1.0, "primary", "direct_input", "indirect", "mae")
    n["MaeIndirectFull"] = f(m)
    m, _, _ = diag_mean(1.0, "primary", "zero_peak_pattern", "changed", "mae")
    n["MaeZeroPeakChangedFull"] = f(m)
    m, _, _ = diag_mean(1.0, "primary", "zero_peak_pattern", "unchanged", "mae")
    n["MaeZeroPeakSameFull"] = f(m)
    for cat, tag in [("changed", "Changed"), ("unchanged", "Same")]:
        n[f"ZeroPeak{tag}Cases"] = count(diag_mean(1.0, "primary", "zero_peak_pattern", cat, "mae")[1])
    for cat, tag in [("direct", "Direct"), ("indirect", "Indirect")]:
        n[f"Input{tag}Cases"] = count(diag_mean(1.0, "primary", "direct_input", cat, "mae")[1])
    m, eff, _ = diag_mean(1.0, "all_responsive", "baseline_selectivity", "below_0.2", "mae")
    n["MaeLowDSIFull"] = f(m)
    n["MaeZeroLowDSIFull"] = f(diag_mean(1.0, "all_responsive", "baseline_selectivity", "below_0.2", "zero_mae")[0])
    n["LowDSICasesFull"] = count(eff)

    # Development strength panel (means and ranges over four models)
    for s, tag in [(0.1, "OneTenth"), (0.25, "Quarter"), (0.5, "Half"), (0.75, "ThreeQuarter"), (1.0, "Full")]:
        n[f"DevMaeFO{tag}"] = f(mean(s, "gradient", "mae", DEV_SUMMARY))
        n[f"DevMaeZero{tag}"] = f(mean(s, "gradient", "zero_mae", DEV_SUMMARY))
        n[f"DevMaeClip{tag}"] = f(mean(s, "gradient", "bounded_mae", DEV_SUMMARY))
        n[f"DevPrecFO{tag}"] = f(mean(s, "gradient", "top10_precision", DEV_SUMMARY))
        n[f"DevPrecAnat{tag}"] = f(mean(s, "anatomy_hop3", "top10_precision", DEV_SUMMARY))
        n[f"DevBetter{tag}"] = str(POST["development"][f"{s:g}"]["gradient_better_than_zero"])

    dev_strengths = sorted({k[0] for k in DEV_SUMMARY})
    better = [s for s in dev_strengths if mean(s, "gradient", "mae", DEV_SUMMARY) < mean(s, "gradient", "zero_mae", DEV_SUMMARY)]
    assert better == dev_strengths[:len(better)], "first-order better than zero change at a non-contiguous set of strengths"
    n["DevBetterMaxStrength"] = f"{better[-1]:g}"
    devb = POST["development"]
    # the text: better in every development model at 0.1 and in none at 0.75 or 1
    assert devb["0.1"]["gradient_better_than_zero"] == devb["0.1"]["models"] == len(CONFIG["development_models"])
    assert devb["0.75"]["gradient_better_than_zero"] == 0 and devb["1"]["gradient_better_than_zero"] == 0

    # Threshold sensitivity (full silencing): DSI thresholds >= 0.1 with every amplification limit, and all combinations
    combos = [(th, amp) for th in CONFIG["direction_selectivity_thresholds"] for amp in CONFIG["amplification_thresholds"] + [None]]
    selective = [(th, amp) for th, amp in combos if th >= 0.1]
    fo = [sens(1.0, th, amp, "gradient")["top10_precision"]["mean"] for th, amp in selective]
    an = [sens(1.0, th, amp, "anatomy_hop3")["top10_precision"]["mean"] for th, amp in selective]
    n["SensCombos"] = str(len(selective))
    n["SensCombosAll"] = str(len(combos))
    n["SensPrecFOMin"], n["SensPrecFOMax"] = f(min(fo)), f(max(fo))
    n["SensPrecAnatMin"], n["SensPrecAnatMax"] = f(min(an)), f(max(an))
    worse = [sens(1.0, th, amp, "gradient")["mae"]["mean"] > sens(1.0, th, amp, "gradient")["zero_mae"]["mean"] for th, amp in combos]
    assert all(worse), "the text states that the first-order error exceeds the zero-change error in every combination"
    ordered = [sens(1.0, th, amp, "gradient")["bounded_mae"]["mean"] < sens(1.0, th, amp, "gradient")["zero_mae"]["mean"]
               and sens(1.0, th, amp, "gradient")["top10_precision"]["mean"] > sens(1.0, th, amp, "anatomy_hop3")["top10_precision"]["mean"]
               for th, amp in combos]
    assert all(ordered), "the appendix states that the clipped estimate beats zero change and first-order beats anatomy in every row"

    # Clusters (confirmation): at half strength the text names the groups in which first-order beats zero change
    half = {c["cluster"]: c for c in CLUSTERS if c["strength"] == 0.5}
    n["ClusterUpMaeFOHalf"] = f(half[1]["mae"]["mean"])
    n["ClusterUpMaeZeroHalf"] = f(half[1]["zero_mae"]["mean"])
    assert half[1]["mae"]["mean"] < half[1]["zero_mae"]["mean"]
    assert all(half[k]["mae"]["mean"] > half[k]["zero_mae"]["mean"] for k in (0, 2))
    assert all(c["mae"]["mean"] > c["zero_mae"]["mean"] for c in CLUSTERS if c["strength"] == 1.0)
    for c in CLUSTERS:
        if c["strength"] != 1.0:
            continue
        tag = {0: "Weak", 1: "Up", 2: "Down"}[c["cluster"]]
        n[f"Cluster{tag}Models"] = str(len(c["models"]))
        n[f"Cluster{tag}MaeFO"] = f(c["mae"]["mean"])
        n[f"Cluster{tag}MaeZero"] = f(c["zero_mae"]["mean"])
        n[f"Cluster{tag}Prec"] = f(c["top10_precision"]["mean"])

    # Pair interactions (development models, full silencing)
    pair = {p["pair"]: p for p in POST["pairs"]}
    t4_pairs = [k for k in pair if k.split("+")[0] in ("Mi1", "Tm3", "Mi4", "Mi9")]
    t5_pairs = [k for k in pair if k not in t4_pairs]
    majority = lambda k: pair[k]["subadditive_dsi"] > pair[k]["same_direction_singles"] / 2
    assert all(majority(k) for k in t4_pairs)
    n["PairMiOneMiFourDSI"] = f(pair["Mi1+Mi4"]["abs_dsi_interaction_quartiles"][1])
    n["PairTmMaxDSI"] = f(max(pair[k]["abs_dsi_interaction_quartiles"][1] for k in t5_pairs))
    n["PairTmSubadditiveMajority"] = str(sum(majority(k) for k in t5_pairs))
    n["PairTmCount"] = str(len(t5_pairs))

    bio = {(r["cell"], r["readouts"]): r for r in POST["biology_context"]}
    n["BioMiFourTFourDSI"] = f(bio[("Mi4", "T4")]["dsi_change_median"], 2)
    n["BioMiNineTFourDSI"] = f(bio[("Mi9", "T4")]["dsi_change_median"], 2)
    n["BioCTOneTFourRatio"] = f(bio[("CT1_joint", "T4")]["amplitude_ratio_median"], 2)
    n["BioCTOneTFiveRatio"] = f(bio[("CT1_joint", "T5")]["amplitude_ratio_median"], 2)
    n["BioCTOneTFourDSI"] = f(bio[("CT1_joint", "T4")]["dsi_change_median"], 2)
    n["BioCTOneTFiveDSI"] = f(bio[("CT1_joint", "T5")]["dsi_change_median"], 2)
    mi1 = bio[("Mi1", "T4")]
    n["BioMiOneTFourDefined"] = str(mi1["dsi_defined"])
    n["BioMiOneTFourConditions"] = str(mi1["readout_conditions"])
    n["BioTmThreeTFourRatio"] = f(bio[("Tm3", "T4")]["amplitude_ratio_median"], 2)
    n["BioMiFourTFourRatio"] = f(bio[("Mi4", "T4")]["amplitude_ratio_median"], 2)
    n["BioMiNineTFourRatio"] = f(bio[("Mi9", "T4")]["amplitude_ratio_median"], 2)

    # Cost
    t = TIMING["summary"]
    n["TimeGradient"] = f(t["gradient_seconds"]["mean"], 1)
    n["TimeExactFull"] = f(t["full_exact_seconds"]["mean"], 1)
    n["TimeExactBoth"] = f(t["both_exact_seconds"]["mean"], 1)
    n["TimeRatioFull"] = f(t["full_exact_over_gradient"]["mean"], 2)
    n["TimeRatioBoth"] = f(t["both_exact_over_gradient"]["mean"], 2)
    n["TimeModels"] = str(len(TIMING["models"]))
    conf_models = sorted({r["model"] for r in MODELS})
    forward, backward, simulated = [], [], []
    for model in conf_models:
        fw = bw = 0.0
        for key in CONFIG["common_stimuli"]:
            g = read(RES / "runs" / model / key / "gradient.json")
            fw += g["forward_s"]
            bw += sum(g["backward_s"])
            if model == conf_models[0] and key == CONFIG["common_stimuli"][0]:
                details = [read(RES / "runs" / model / key / f"s100-cell{i:02d}.json") for i in range(65)]
                simulated.append(sum("wall_seconds" in d for d in details))
        forward.append(fw)
        backward.append(bw / len(CONFIG["readouts"]))
    n["TimeForward"] = f(float(np.mean(forward)), 1)
    n["TimeBackwardEach"] = f(float(np.mean(backward)), 1)
    n["SimulatedTypes"] = str(simulated[0])
    n["TimePerType"] = f(t["full_exact_seconds"]["mean"] / simulated[0], 1)
    n["BackwardOverType"] = f(float(np.mean(backward)) / (t["full_exact_seconds"]["mean"] / simulated[0]), 0)
    ties = near = 0
    for model in conf_models:
        for key in CONFIG["common_stimuli"]:
            g = read(RES / "runs" / model / key / "gradient.json")
            near += sum(x > 0 for x in g["near_peak_counts_per_readout"])
            ties += sum(x > 0 for x in g["peak_tie_counts_per_readout"])
    n["NearTieReadouts"] = str(near)
    n["ExactTieReadouts"] = str(ties)

    # Verification
    gate, ver = read(RES / "verification/development-gate.json"), read(RES / "verification/confirmation-verification.json")
    n["CheckedFilesDev"] = count(gate["checked_files"])
    n["CheckedFilesConf"] = count(ver["checked_files"])
    n["ReplayMaxErr"] = sci(max(r["normalized_error"] for r in gate["fresh_cpu_replays"] + ver["fresh_cpu_replays"]))
    n["ReplayCount"] = str(len(gate["fresh_cpu_replays"]) + len(ver["fresh_cpu_replays"]))
    n["DsiRecomputeMaxErr"] = sci(max(gate["independent_max_errors"]["dsi"], ver["independent_max_errors"]["dsi"]))
    tangent = read(ROOT / "04-Results/stage1/benchmark/cuda/gradient-gate-resolution.json")
    n["TangentMaxErr"] = sci(max(r["autograd_tangent_max_error"] for r in tangent["records"]))
    idx = read(RES / "verification/baseline-indexing.json")
    n["IndexedTypePairs"] = str(idx["ordered_type_pairs"])
    return n


# ---------------------------------------------------------------- tables
def table_screen():
    lines = []
    for method, label in METHODS:
        cells = [f(mean(0.5, method, "top10_precision")), f(mean(1.0, method, "top10_precision")),
                 f(mean(1.0, method, "top10_recall")), f(mean(1.0, method, "rho")), f(mean(1.0, method, "rho_important")),
                 f(sens(1.0, 0.0, None, method)["composite_top10_precision"]["mean"])]
        if method == "gradient":
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if method == "gradient":
            lines.append("\\addlinespace")
    lines.append("\\addlinespace")
    lines.append("Random ranking (expected) & " + " & ".join([
        f(mean(0.5, "gradient", "top10_random_precision")), f(mean(1.0, "gradient", "top10_random_precision")),
        f(mean(1.0, "gradient", "top10_random_recall")), "--", "--", f(POST["inclusive_random_precision"]["1"])]) + " \\\\")
    return "\n".join(lines) + "\n"


def table_paired():
    lines = []
    comp = POST["comparisons"]["1"]
    for method, label in COMPARATORS:
        p = PAIRED[(1.0, method)]
        t = comp[method]
        mp, lr = t["main_precision"], t["large_effect_rho"]
        lines.append(f"{label} & {signed(p['top10_precision_difference']['mean'])} {ci(p['top10_precision_difference']['ensemble_bootstrap_95'])}"
                     f" & {mp['higher']}/{mp['equal']}/{mp['lower']}"
                     f" & {signed(p['rho_important_difference']['mean'])} {ci(p['rho_important_difference']['ensemble_bootstrap_95'])}"
                     f" & {lr['higher']}/{lr['lower']} \\\\")
    return "\n".join(lines) + "\n"


def table_magnitude():
    e = {s: POST["effects"][f"{s:g}"] for s in (0.5, 1.0)}
    g = {s: POST["comparisons"][f"{s:g}"]["gradient"] for s in (0.5, 1.0)}
    both = lambda fn: " & ".join(fn(s) for s in (0.5, 1.0))
    body = [
        ("Silencing cases scored", both(lambda s: count(e[s]["main_cases"]))),
        ("Mean absolute error: first-order", both(lambda s: f(mean(s, "gradient", "mae")))),
        ("\\quad predicting zero change", both(lambda s: f(mean(s, "gradient", "zero_mae")))),
        ("\\quad first-order clipped to $[0,1]$", both(lambda s: f(mean(s, "gradient", "bounded_mae")))),
        ("First-order minus zero change", both(lambda s: f"{signed(MAGDIFF[s]['gradient_minus_zero']['mean'])} {ci(MAGDIFF[s]['gradient_minus_zero']['ensemble_bootstrap_95'])}")),
        ("\\quad models with larger first-order error", both(lambda s: f"{g[s]['mae_vs_zero']['higher']} of {g[s]['mae_vs_zero']['models']}")),
        ("Clipped minus zero change", both(lambda s: f"{signed(MAGDIFF[s]['bounded_minus_zero']['mean'])} {ci(MAGDIFF[s]['bounded_minus_zero']['ensemble_bootstrap_95'])}")),
        ("Mean signed error of first-order", both(lambda s: f"{signed(g[s]['bias']['mean'])} {ci(g[s]['bias']['ci95'])}")),
        ("Predicted DSI outside $[0,1]$ (\\%)", both(lambda s: pct(diag_mean(s, 'primary', 'baseline_selectivity', 'at_least_0.2', 'infeasible_prediction_fraction')[0]))),
        ("\\addlinespace Large changes ($|\\Delta|\\ge 0.1$)", both(lambda s: count(e[s]["large"]))),
        ("Mean absolute error: first-order", both(lambda s: f(mean(s, "gradient", "important_mae")))),
        ("\\quad predicting zero change", both(lambda s: f(mean(s, "gradient", "important_zero_mae")))),
        ("Predicted with the wrong sign (\\%)", both(lambda s: pct(mean(s, "gradient", "opposite_sign_rate")))),
        ("Predicted as exactly zero (\\%)", both(lambda s: pct(mean(s, "gradient", "zero_prediction_rate")))),
    ]
    return "\n".join(f"{a} & {b} \\\\" for a, b in body) + "\n"


def table_coverage():
    e = {s: POST["effects"][f"{s:g}"] for s in (0.5, 1.0)}
    full = [r for r in FAILURES if float(r["strength"]) == 1.0]
    resp = [r for r in full if r["baseline_responsive"] == "True"]
    lo = sum(float(r["baseline_dsi"]) < MAIN_DSI for r in resp)
    zg = POST["zero_gradient"]
    both = lambda fn: " & ".join(fn(s) for s in (0.5, 1.0))
    body = [
        ("Readout conditions", f"\\multicolumn{{2}}{{c}}{{{len(full)}}}"),
        ("\\quad no response to either polarity", f"\\multicolumn{{2}}{{c}}{{{len(full) - len(resp)}}}"),
        ("\\quad responsive, DSI below 0.2", f"\\multicolumn{{2}}{{c}}{{{lo}}}"),
        ("\\quad responsive, DSI at least 0.2 (main analysis)", f"\\multicolumn{{2}}{{c}}{{{len(resp) - lo}}}"),
        ("\\quad all first-order estimates exactly zero", f"\\multicolumn{{2}}{{c}}{{{zg['readout_conditions']}}}"),
        ("\\addlinespace Silencing cases in responsive readout conditions", both(lambda s: count(e[s]["responsive_cases"]))),
        ("\\quad response abolished", both(lambda s: count(e[s]["abolished"]))),
        ("\\quad readout voltage amplified more than 100-fold", both(lambda s: count(e[s]["amplified_cases"]))),
        ("\\addlinespace Silencing cases in direction-selective readout conditions", both(lambda s: count(e[s]["selective_cases"]))),
        ("\\quad response abolished", both(lambda s: count(e[s]["selective_abolished"]))),
        ("\\quad amplified more than 100-fold", both(lambda s: count(e[s]["selective_amplified_defined"]))),
        ("\\quad scored in the main analysis", both(lambda s: count(e[s]["main_cases"]))),
    ]
    return "\n".join(f"{a} & {b} \\\\" for a, b in body) + "\n"


def table_diagnostics():
    cats = [("effect_size", "important", "Large change ($|\\Delta|\\ge0.1$)"), ("effect_size", "smaller", "Smaller change"),
            ("direct_input", "direct", "Direct input to readout"), ("direct_input", "indirect", "Indirect input only"),
            ("zero_peak_pattern", "changed", "Zero-response directions changed"),
            ("zero_peak_pattern", "unchanged", "Zero-response directions unchanged"),
            ("baseline_selectivity", "at_least_0.2", "Baseline DSI at least 0.2"), ("baseline_selectivity", "below_0.2", "Baseline DSI below 0.2")]
    lines = []
    for diagnostic, category, label in cats:
        main = [r for r in DIAG if float(r["strength"]) == 1.0 and r["stratum"] == "primary" and r["diagnostic"] == diagnostic and r["category"] == category]
        allr = [r for r in DIAG if float(r["strength"]) == 1.0 and r["stratum"] == "all_responsive" and r["diagnostic"] == diagnostic and r["category"] == category]
        mm = lambda sel, field: f(float(np.mean([float(r[field]) for r in sel])))
        left = (f"{count(sum(int(r['effects']) for r in main))} & {mm(main, 'mae')} & {mm(main, 'zero_mae')} & "
                f"{pct(float(np.mean([float(r['infeasible_prediction_fraction']) for r in main])))}") if main else "\\multicolumn{4}{c}{--}"
        right = f"{count(sum(int(r['effects']) for r in allr))} & {mm(allr, 'mae')} & {mm(allr, 'zero_mae')}"
        lines.append(f"{label} & {left} & {right} \\\\")
        if category in ("smaller", "indirect", "unchanged"):
            lines.append("\\addlinespace")
    return "\n".join(lines) + "\n"


def table_sensitivity():
    lines = []
    for threshold in CONFIG["direction_selectivity_thresholds"]:
        for amp in [10.0, 100.0, 1000.0, None]:
            g, a = sens(1.0, threshold, amp, "gradient"), sens(1.0, threshold, amp, "anatomy_hop3")
            label = "none" if amp is None else f"{amp:g}"
            row = [f"{threshold:g}", label, str(g["units"]), count(g["effects"]), f(g["top10_precision"]["mean"]),
                   f(a["top10_precision"]["mean"]), f(g["composite_top10_precision"]["mean"]), f(a["composite_top10_precision"]["mean"]),
                   f(g["mae"]["mean"]), f(g["zero_mae"]["mean"]), f(g["bounded_mae"]["mean"])]
            if threshold == MAIN_DSI and amp == MAIN_AMP:
                row = [f"\\textbf{{{x}}}" for x in row]
            lines.append(" & ".join(row) + " \\\\")
        if threshold != CONFIG["direction_selectivity_thresholds"][-1]:
            lines.append("\\addlinespace")
    return "\n".join(lines) + "\n"


def table_clusters():
    lines = []
    for s in (0.5, 1.0):
        for c in sorted((c for c in CLUSTERS if c["strength"] == s), key=lambda c: [1, 2, 0].index(c["cluster"])):
            lines.append(f"{s:g} & {CLUSTER_NAMES[c['cluster']]} & {len(c['models'])} & {f(c['top10_precision']['mean'])} & "
                         f"{f(c['mae']['mean'])} & {f(c['zero_mae']['mean'])} & {f(c['bounded_mae']['mean'])} \\\\")
        if s == 0.5:
            lines.append("\\addlinespace")
    return "\n".join(lines) + "\n"


def table_pairs():
    lines = []
    q = lambda v: f"{f(v[1])} [{f(v[0])}, {f(v[2])}]"
    for p in POST["pairs"]:
        lines.append(f"{p['pair'].replace('+', ' + ')} & {p['readout_conditions']} ({p['models']}) & {q(p['abs_dsi_interaction_quartiles'])} & "
                     f"{q(p['voltage_rms_quartiles'])} & {q(p['relative_voltage_quartiles'])} & {p['subadditive_dsi']}/{p['same_direction_singles']} \\\\")
    return "\n".join(lines) + "\n"


def table_development():
    lines = []
    for s in sorted({k[0] for k in DEV_SUMMARY}):
        md = DEV_MAGDIFF[s]["gradient_minus_zero"]
        lines.append(f"{s:g} & {f(mean(s, 'gradient', 'top10_precision', DEV_SUMMARY))} & {f(mean(s, 'anatomy_hop3', 'top10_precision', DEV_SUMMARY))} & "
                     f"{f(mean(s, 'gradient', 'rho_important', DEV_SUMMARY))} & {f(mean(s, 'gradient', 'mae', DEV_SUMMARY))} & "
                     f"{f(mean(s, 'gradient', 'zero_mae', DEV_SUMMARY))} & {f(mean(s, 'gradient', 'bounded_mae', DEV_SUMMARY))} & "
                     f"{signed(md['mean'])} [{f(md['min'])}, {f(md['max'])}] & {pct(mean(s, 'gradient', 'opposite_sign_rate', DEV_SUMMARY))} \\\\")
    return "\n".join(lines) + "\n"


def table_models():
    lines = []
    for label, source in [("confirmation", MODELS), ("development", DEV_MODELS)]:
        main = [r for r in source if is_main(r) and float(r["strength"]) == 1.0]
        by = {}
        for r in main:
            by.setdefault(r["model"], {})[r["method"]] = r
        if label == "development":
            lines.append("\\midrule\\multicolumn{13}{l}{\\textit{Development models}}\\\\")
        for model in sorted(by):
            g, a = by[model]["gradient"], by[model]["anatomy_hop3"]
            rl = lambda r: f(float(r["rho_important"]), 2) if r["rho_important"] else "--"
            lines.append(" & ".join([model, CLUSTER_SHORT[int(g["cluster"])],
                                     g["units"], count(int(g["effects"])), f(float(g["top10_precision"]), 2), f(float(a["top10_precision"]), 2),
                                     rl(g), rl(a), f(float(g["mae"])), f(float(g["zero_mae"])), f(float(g["bounded_mae"])),
                                     f(float(g["bias"])), pct(float(g["opposite_sign_rate"]), 0)]) + " \\\\")
    return "\n".join(lines) + "\n"


def table_biology():
    names = {"Mi1": "Mi1", "Tm3": "Tm3", "Mi4": "Mi4", "Mi9": "Mi9", "C2": "C2", "C3": "C3", "CT1_joint": "CT1 (both compartments)"}
    lines = []
    by = {(r["cell"], r["readouts"]): r for r in POST["biology_context"]}
    for cell, label in names.items():
        cells = []
        for family in ("T4", "T5"):
            r = by[(cell, family)]
            cells += [f(r["amplitude_ratio_median"], 2), f"{r['dsi_defined']}/{r['readout_conditions']}",
                      f(r["dsi_change_median"], 2) if r["dsi_change_median"] is not None else "--"]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def tables():
    return {"table-screen.tex": table_screen(), "table-paired.tex": table_paired(), "table-magnitude.tex": table_magnitude(),
            "table-coverage.tex": table_coverage(), "table-diagnostics.tex": table_diagnostics(),
            "table-sensitivity.tex": table_sensitivity(), "table-clusters.tex": table_clusters(), "table-pairs.tex": table_pairs(),
            "table-development.tex": table_development(), "table-models.tex": table_models(),
            "table-biology.tex": table_biology()}


def render(n):
    lines = ["% Generated by paper/make_numbers.py from saved study outputs. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{n[k]}}}" for k in sorted(n)]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    # Each table body ends with its \bottomrule: a rule placed after \input in the section would be misplaced.
    files = {"numbers.tex": render(numbers()), **{k: v + "\\bottomrule\n" for k, v in tables().items()}}
    if args.check:
        stale = [k for k, v in files.items() if not (PAPER / "generated" / k).exists() or (PAPER / "generated" / k).read_text() != v]
        if stale:
            print("out of date: " + ", ".join(stale), file=sys.stderr)
            sys.exit(1)
        return
    (PAPER / "generated").mkdir(exist_ok=True)
    for name, text in files.items():
        target = PAPER / "generated" / name
        if not target.exists() or target.read_text() != text:   # unchanged files keep their bytes and dates
            target.write_text(text)
    print(f"wrote {len(files['numbers.tex'].splitlines()) - 1} macros and {len(files) - 1} tables")


if __name__ == "__main__":
    main()
