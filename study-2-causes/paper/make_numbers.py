"""Write paper/generated/numbers.tex and the generated tables from saved study outputs.

Every study number in the manuscript is a macro defined here, computed from the frozen confirmation and development
analyses (study/results/analysis/*/summary.json and models.csv), the study's records (configuration, freezes,
prediction records, pilot records) and the post hoc summary analysis/results/posthoc-summary.json.
`--check` recomputes everything in memory and fails if a written file differs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
STUDY = ROOT / "study"
RES = STUDY / "results"
CONFIG = json.loads((STUDY / "config.json").read_text())
CONF = json.loads((RES / "analysis/confirmation/summary.json").read_text())
DEV = json.loads((RES / "analysis/development/summary.json").read_text())
POST = json.loads((ROOT / "analysis/results/posthoc-summary.json").read_text())
POSTD = json.loads((ROOT / "analysis/results/posthoc-denominator.json").read_text())
FULL, HALF = CONF["strengths"]["1.0"], CONF["strengths"]["0.5"]
STRENGTHS = {"Full": FULL, "Half": HALF}

CELL = {
    "Ref": "zero/peak_dsi/first", "A": "mean/peak_dsi/first", "B": "zero/mean_dsi/first", "C": "zero/peak_dsi/second",
    "CfNet": "zero/peak_dsi/exact_trajectory_linear_readout", "CfRead": "zero/peak_dsi/linear_trajectory_exact_readout",
    "SplitResp": "zero/peak_dsi/linear_trajectory_exact_responses", "SplitRatio": "zero/peak_dsi/linear_trajectory_exact_ratio",
    "MisMean": "mean/peak_dsi/mismatch", "MisZero": "zero/peak_dsi/mismatch",
    "AB": "mean/mean_dsi/first", "BC": "zero/mean_dsi/second", "AC": "mean/peak_dsi/second", "ABC": "mean/mean_dsi/second",
}
ARM_LETTER = {"A": "a", "B": "b", "C": "c"}


# ---------------------------------------------------------------- formatting
def f(x, d=3):
    text = f"{x:.{d}f}"
    if float(text) == 0:
        text = f"{0:.{d}f}"
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def signed(x, d=3):
    text = f(x, d)
    return text if text.startswith("\\ensuremath{-}") or float(f"{x:.{d}f}") == 0 else "\\ensuremath{+}" + text


def pct(x, d=0):
    return f(100 * x, d)


def count(n):
    return f"{int(n):,}"


def ci(pair, d=3):
    return f"[{f(pair[0], d)}, {f(pair[1], d)}]"


def ci_adaptive(pair, d=3):
    """An interval with enough decimals that neither end rounds to zero (decisions rest on the sign of each end)."""
    while d < 5 and any(float(f"{abs(x):.{d}f}") == 0 for x in pair):
        d += 1
    return ci(pair, d)


def sci(x, d=1):
    mant, exp = f"{x:.{d}e}".split("e")
    return f"\\ensuremath{{{mant}\\times 10^{{{int(exp)}}}}}"


def rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def cell(block, name):
    return block["cells"][CELL[name]]


# ---------------------------------------------------------------- numbers
def numbers():
    n = {}
    conf_models = rows(RES / "analysis/confirmation/models.csv")
    reference_rows = [r for r in conf_models if r["strength"] == "1.0" and (r["intervention"], r["readout"], r["estimator"])
                      == ("zero", "peak_dsi", "first")]
    n["NModels"] = count(len(CONF["models"]) + len(CONFIG["development_models"]))
    n["NConfModels"] = count(len(CONF["models"]))
    n["NDevModels"] = count(len(CONFIG["development_models"]))
    n["NTypes"] = "65"
    n["NSimTypes"] = "63"
    n["NReadouts"] = count(len(CONFIG["readouts"]))
    n["NDirections"] = "12"
    n["NUnitsEligible"] = count(sum(int(r["units"]) for r in reference_rows))
    n["NUnitsAll"] = count(len(CONF["models"]) * len(CONFIG["stimuli"]) * len(CONFIG["readouts"]))
    n["NCasesFull"] = count(sum(int(r["effects"]) for r in reference_rows))
    n["NLargeFull"] = count(sum(int(r["large_effects"]) for r in reference_rows))
    n["LargeThreshold"] = f(CONFIG["important_effect_threshold"], 1)
    n["SelThreshold"] = f(CONFIG["primary_dsi_threshold"], 1)
    n["AmpLimit"] = count(CONFIG["primary_amplification_threshold"])
    n["BootReps"] = count(CONFIG["bootstrap_replicates"])
    n["ScreenBudget"] = count(CONFIG["screening_budget"])
    n["FlagBudgetPrimary"] = count(CONFIG["primary_flag_budget"])
    n["StrengthsDev"] = ", ".join(f(s, 2).rstrip("0").rstrip(".") for s in sorted(CONFIG["strengths"] + CONFIG["development_extra_strengths"]))

    # The arms and every estimator, at full and half strength.
    for suffix, block in STRENGTHS.items():
        for name in CELL:
            c = cell(block, name)
            n[f"Ratio{name}{suffix}"] = f(c["ratio"]["mean"], 2)
            n[f"RatioCI{name}{suffix}"] = ci(c["ratio"]["interval"], 2)
            n[f"Mae{name}{suffix}"] = f(c["mae"]["mean"])
            n[f"MaeZero{name}{suffix}"] = f(c["zero_mae"]["mean"])
            n[f"MaeDiff{name}{suffix}"] = signed(c["mae_minus_zero"]["mean"])
            n[f"MaeDiffCI{name}{suffix}"] = ci_adaptive(c["mae_minus_zero"]["interval"])
            n[f"Bias{name}{suffix}"] = signed(c["bias"]["mean"])
            n[f"BiasCI{name}{suffix}"] = ci(c["bias"]["interval"])
            n[f"Wrong{name}{suffix}"] = pct(c["wrong_sign_rate"]["mean"], 1)
            n[f"WrongFrac{name}{suffix}"] = f(c["wrong_sign_rate"]["mean"])
            n[f"TopTen{name}{suffix}"] = f(c["top10"]["mean"])
            n[f"TopTenCI{name}{suffix}"] = ci(c["top10"]["interval"])
            n[f"Anat{name}{suffix}"] = f(c["anatomy_top10"]["mean"])
            n[f"TopAnat{name}{suffix}"] = signed(c["top10_minus_anatomy"]["mean"])
            n[f"TopAnatCI{name}{suffix}"] = ci(c["top10_minus_anatomy"]["interval"])
            n[f"Clip{name}{suffix}"] = f(c["clipped_ratio"]["mean"], 2)
        for arm, letter in ARM_LETTER.items():
            a = block["arms"][letter]
            n[f"Change{arm}{suffix}"] = signed(a["change_from_reference"]["mean"], 2)
            n[f"ChangeCI{arm}{suffix}"] = ci(a["change_from_reference"]["interval"], 2)
            n[f"WrongChange{arm}{suffix}"] = signed(a["wrong_sign_change"]["mean"])
            n[f"WrongChangeCI{arm}{suffix}"] = ci(a["wrong_sign_change"]["interval"])
            n[f"DecZero{arm}{suffix}"] = a["zero"]
            n[f"DecChange{arm}{suffix}"] = unlabel(a["change_decision"])
            n[f"DecBias{arm}{suffix}"] = a["bias_decision"]
            n[f"DecWrong{arm}{suffix}"] = a["wrong_sign_decision"]
            n[f"DecScreen{arm}{suffix}"] = a["screen_decision"]
        n[f"TopFirstC{suffix}"] = signed(block["arms"]["c"]["top10_minus_first_order"]["mean"])
        n[f"TopFirstCIC{suffix}"] = ci(block["arms"]["c"]["top10_minus_first_order"]["interval"])
        n[f"Dominance{suffix}"] = block["dominance"]["decision"]
        # Mechanism (Section 8.1) and readout split (Section 13.1), recoveries as shares of AI-06's error removed.
        for key, label in (("zero/peak_dsi", "ZP"), ("zero/mean_dsi", "ZM"), ("mean/peak_dsi", "MP"), ("mean/mean_dsi", "MM")):
            m = block["mechanism"][key]
            n[f"RecNet{label}{suffix}"] = pct(m["recovery"]["exact_trajectory_linear_readout"]["mean"])
            n[f"RecRead{label}{suffix}"] = pct(m["recovery"]["linear_trajectory_exact_readout"]["mean"])
            n[f"ReadMinusNet{label}{suffix}"] = f(m["readout_minus_network"]["mean"], 2)
            n[f"ReadMinusNetCI{label}{suffix}"] = ci(m["readout_minus_network"]["interval"], 2)
            n[f"DecMech{label}{suffix}"] = m["decision"]
            s = block["readout_split"][key]
            n[f"RecResp{label}{suffix}"] = pct(s["recovery"]["linear_trajectory_exact_responses"]["mean"])
            n[f"RecRatio{label}{suffix}"] = pct(s["recovery"]["linear_trajectory_exact_ratio"]["mean"])
            n[f"RatioMinusResp{label}{suffix}"] = f(s["ratio_minus_responses"]["mean"], 2)
            n[f"RatioMinusRespCI{label}{suffix}"] = ci(s["ratio_minus_responses"]["interval"], 2)
            n[f"Interaction{label}{suffix}"] = f(s["interaction"]["mean"], 2)
            n[f"InteractionCI{label}{suffix}"] = ci(s["interaction"]["interval"], 2)
            n[f"DecSplit{label}{suffix}"] = s["decision"]
        mis = block["mismatch"]
        n[f"MisMeanMinusMatched{suffix}"] = signed(mis["zero_derivative_on_mean_target_minus_matched_ratio"]["mean"], 2)
        n[f"MisMeanMinusMatchedCI{suffix}"] = ci(mis["zero_derivative_on_mean_target_minus_matched_ratio"]["interval"], 2)
        n[f"MisMeanTopAnat{suffix}"] = signed(mis["zero_derivative_on_mean_target_top10_minus_anatomy"]["mean"])
        n[f"MisMeanTopAnatCI{suffix}"] = ci(mis["zero_derivative_on_mean_target_top10_minus_anatomy"]["interval"])
        n[f"MisZeroMinusRef{suffix}"] = signed(mis["mean_derivative_on_zero_target_minus_reference_ratio"]["mean"], 2)
        n[f"MisZeroMinusRefCI{suffix}"] = ci(mis["mean_derivative_on_zero_target_minus_reference_ratio"]["interval"], 2)

    # Per-model counts at full strength.
    def per_model(estimator_cell, field="ratio"):
        i, r, e = CELL[estimator_cell].split("/")
        return {m["model"]: float(m[field]) for m in conf_models
                if m["strength"] == "1.0" and m["intervention"] == i and m["readout"] == r and m["estimator"] == e}
    ref = per_model("Ref")
    for name in ("A", "B", "CfRead", "SplitRatio"):
        values = per_model(name)
        n[f"LowerThanRef{name}"] = count(sum(values[m] < ref[m] for m in ref))
        n[f"BelowOne{name}"] = count(sum(values[m] < 1 for m in values))
    n["BelowOneRef"] = count(sum(v < 1 for v in ref.values()))

    # Trajectories.
    trace = CONF["trace_level"]
    n["TraceLinZero"] = pct(trace["zero/1.0"]["linear"]["mean"])
    n["TraceQuadZero"] = pct(trace["zero/1.0"]["quadratic"]["mean"])
    n["TraceRatioZero"] = f(trace["zero/1.0"]["quadratic_over_linear"]["mean"], 2)
    n["TraceLinMean"] = pct(trace["mean/1.0"]["linear"]["mean"])
    n["TraceQuadMean"] = pct(trace["mean/1.0"]["quadratic"]["mean"])

    # Screen, flag, then simulate; cost.
    rule = CONF["flag_rule"]
    for k in CONFIG["flag_budgets"]:
        for fallback, label in (("zero", "Zero"), ("clipped", "Clip")):
            r = rule[f"k{k}/{fallback}"]
            word = {5: "Five", 10: "Ten", 20: "Twenty"}[k]
            n[f"FlagRec{word}{label}"] = pct(r["recovered"]["mean"])
            n[f"FlagSec{word}{label}"] = f(r["seconds"]["mean"], 0)
            n[f"FlagPerSec{word}{label}"] = f(r["recovered_per_second"]["mean"], 4)
            n[f"FlagPays{word}{label}"] = "yes" if r["rule_pays"] else "no"
    n["ExhaustivePerSec"] = f(rule["k10/zero"]["exhaustive_per_second"]["mean"], 4)
    timing = CONF["timing"]
    n["TimeTangentsZero"] = f(timing["tangents_zero"]["mean"], 0)
    n["TimeTangentsMean"] = f(timing["tangents_mean"]["mean"], 0)
    n["TimeExactMean"] = f(timing["exact_mean_s100"]["mean"], 0)
    n["TimeExactAIsix"] = f(timing["ai06_exact_s100"]["mean"], 1)
    n["TimeDerivAIsix"] = f(timing["ai06_derivative"]["mean"], 1)
    n["TimeRatioTangents"] = f(timing["ratio/c"]["mean"], 2)
    n["TimeRatioExactMean"] = f(timing["ratio/a_exact_over_ai06_exact"]["mean"], 2)

    # Predictions.
    for key, label in (("predictions", "Primary"), ("addendum_predictions", "Addendum")):
        checks = CONF[key]
        n[f"Pred{label}Held"] = count(sum(c["holds"] for c in checks))
        n[f"Pred{label}Total"] = count(len(checks))

    # Development models.
    for s, label in (("0.1", "PointOne"), ("0.25", "Quarter"), ("0.5", "Half"), ("0.75", "ThreeQ"), ("1.0", "Full")):
        block = DEV["strengths"][s]
        for name in ("Ref", "A", "B", "C", "CfRead", "SplitRatio"):
            n[f"Dev{name}{label}"] = f(cell(block, name)["ratio"]["mean"], 2)

    # Post hoc (Section 3.3 of the paper).
    n["PostShareReadout"] = pct(POST["readout_share_of_second_order_term"]["mean"])
    n["PostLargeDenFrac"] = pct(POST["large_denominator_change_fraction"]["mean"])
    n["PostLargeDenShare"] = pct(POST["large_denominator_change_share_of_term"]["mean"])
    n["PostSecondErrLarge"] = f(POST["second_order_error_large"]["mean"], 2)
    n["PostSecondErrRest"] = f(POST["second_order_error_rest"]["mean"], 3)
    n["PostCases"] = count(POST["cases"])
    n["PostWrongFirst"] = pct(POST["pooled_wrong_sign"]["first"], 1)
    n["PostWrongRead"] = pct(POST["pooled_wrong_sign"]["linear_trajectory_exact_readout"], 1)
    n["PostLargeFirst"] = f(POSTD["large_first"]["mean"], 2)
    n["PostLargeZero"] = f(POSTD["large_zero"]["mean"], 2)
    n["PostLargeSecond"] = f(POSTD["large_second"]["mean"], 1)
    n["PostRestFirst"] = f(POSTD["rest_first"]["mean"], 3)
    n["PostRestZero"] = f(POSTD["rest_zero"]["mean"], 3)
    n["PostRestSecond"] = f(POSTD["rest_second"]["mean"], 3)
    n["PostRestSecondBelowFirst"] = count(POSTD["rest_second_below_first_models"])
    gates = POST["gates_confirmation"]["largest"]
    n["GateNormMax"] = sci(max(gates[k]["value"] for k in gates if k.endswith("normalised_difference")))
    n["GateAbsTmThreeMax"] = sci(gates["zero_replacement_Tm3_max_abs_difference"]["value"])
    n["GateAbsTmThreeModel"] = gates["zero_replacement_Tm3_max_abs_difference"]["model"]

    # Pilot checks (development model 000).
    pilot_gpu = json.loads((ROOT / "pilot/results/pilot-gpu.json").read_text())
    pilot_cpu = json.loads((ROOT / "pilot/results/pilot-cpu.json").read_text())
    n["PilotGradMax"] = sci(max(v["gradient_max_abs_difference"] for v in pilot_gpu["stimuli"].values()))
    results = pilot_cpu["tangent_checks"]["results"]
    n["PilotAutodiffMax"] = sci(max(max(v["J_vs_autodiff"]["max_abs_difference"], v["K_vs_autodiff"]["max_abs_difference"])
                                    for v in results.values()))
    zero_cases = [v for k, v in results.items() if k.startswith("zero_")]
    n["PilotFDJMax"] = sci(max(v["J_vs_central_difference_h1e-06"]["max_abs_difference"] for v in zero_cases))
    fd_k = [max(v["K_vs_central_difference_h0.001"]["reference_max_abs"], v["K_vs_central_difference_h1e-06"]["reference_max_abs"])
            for v in zero_cases]
    n["PilotFDKMin"] = f(min(fd_k), 0)
    n["PilotFDKMax"] = f(max(fd_k), 0)
    n["PilotAutoKMax"] = f(max(v["K_vs_autodiff"]["reference_max_abs"] for v in zero_cases), 1)
    n["NodesFlyVis"] = count(pilot_gpu["nodes"])
    n["EdgesFlyVis"] = count(pilot_gpu["edges"])

    # The reference reproduces the companion study's saved model-level values.
    check = json.loads((RES / "analysis/confirmation/reference-check.json").read_text())
    n["RefCheckValues"] = count(check["values_compared"])
    n["RefCheckMax"] = sci(check["largest_difference"], 0)
    n["RefCheckStatus"] = check["status"]

    # Records.
    first = json.loads((STUDY / "freeze.json").read_text())
    second = json.loads((STUDY / "freeze-2.json").read_text())
    n["FreezeFiles"] = count(len(first["files"]))
    n["FreezeAISixInputs"] = count(len(first["ai06_inputs"]))
    n["FreezeTwoFiles"] = count(len(second["files"]))
    n["FreezeTwoRecordModels"] = count(len(second["confirmation_models_with_run_records"]))
    return n


# ---------------------------------------------------------------- tables
def stack(value, pair, d):
    """A value with its interval on the line below (the braces stop \\\\ reading the bracket as a length)."""
    return f"\\makecell[l]{{{value}\\\\{{[}}{f(pair[0], d)}, {f(pair[1], d)}{{]}}}}"


def arm_table():
    lines = []
    for name, label in (("Ref", "Reference"), ("A", "(a) mean replacement"), ("B", "(b) mean-response index"),
                        ("C", "(c) second order")):
        for suffix, strength in (("Full", "1"), ("Half", "0.5")):
            c = cell(STRENGTHS[suffix], name)
            if name == "Ref":
                change, decision = "", "worse" if c["mae_minus_zero"]["interval"][0] > 0 else (
                    "better" if c["mae_minus_zero"]["interval"][1] < 0 else "not distinguishable")
            else:
                a = STRENGTHS[suffix]["arms"][ARM_LETTER[name]]
                change = stack(signed(a['change_from_reference']['mean'], 2), a['change_from_reference']['interval'], 2)
                decision = {"beats zero": "better", "worse than zero": "worse",
                            "not distinguishable from zero": "not distinguishable"}[a["zero"]]
            lines.append(f"{label if suffix == 'Full' else ''} & {strength} & {stack(f(c['ratio']['mean'], 2), c['ratio']['interval'], 2)} "
                         f"& {change} & {decision} & {signed(c['bias']['mean'])} & "
                         f"{f(c['wrong_sign_rate']['mean'])} & {f(c['top10']['mean'])} ({f(c['anatomy_top10']['mean'])}) \\\\")
        lines.append("\\addlinespace")
    return "\n".join(lines[:-1]) + "\n"


def mechanism_table():
    lines = []
    for name, label in (("Ref", "First order (reference)"), ("CfNet", "Exact trajectory, linearised readout"),
                        ("CfRead", "Linearised trajectory, exact readout"),
                        ("SplitResp", "Linearised trajectory, exact responses, linearised ratio"),
                        ("SplitRatio", "Linearised trajectory, floored linear responses, exact ratio")):
        c = cell(FULL, name)
        if name == "Ref":
            removed = ""
        elif name in ("CfNet", "CfRead"):
            key = {"CfNet": "exact_trajectory_linear_readout", "CfRead": "linear_trajectory_exact_readout"}[name]
            r = FULL["mechanism"]["zero/peak_dsi"]["recovery"][key]
            removed = f"{pct(r['mean'])}\\% [{pct(r['interval'][0])}, {pct(r['interval'][1])}]"
        else:
            key = {"SplitResp": "linear_trajectory_exact_responses", "SplitRatio": "linear_trajectory_exact_ratio"}[name]
            r = FULL["readout_split"]["zero/peak_dsi"]["recovery"][key]
            removed = f"{pct(r['mean'])}\\% [{pct(r['interval'][0])}, {pct(r['interval'][1])}]"
        lines.append(f"{label} & {f(c['ratio']['mean'], 2)} {ci(c['ratio']['interval'], 2)} & {removed} & "
                     f"{signed(c['mae_minus_zero']['mean'])} {ci(c['mae_minus_zero']['interval'])} & "
                     f"{f(c['wrong_sign_rate']['mean'])} \\\\")
    return "\n".join(lines) + "\n"


def factorial_table():
    lines = []
    names = [("Ref", "silencing", "peak", "first order"), ("C", "silencing", "peak", "second order"),
             ("B", "silencing", "mean response", "first order"), ("BC", "silencing", "mean response", "second order"),
             ("A", "mean replacement", "peak", "first order"), ("AC", "mean replacement", "peak", "second order"),
             ("AB", "mean replacement", "mean response", "first order"),
             ("ABC", "mean replacement", "mean response", "second order")]
    for name, intervention, readout, estimator in names:
        full, half = cell(FULL, name), cell(HALF, name)
        lines.append(f"{intervention} & {readout} & {estimator} & {f(full['ratio']['mean'], 2)} {ci(full['ratio']['interval'], 2)} & "
                     f"{f(half['ratio']['mean'], 2)} {ci(half['ratio']['interval'], 2)} & {f(full['wrong_sign_rate']['mean'])} & "
                     f"{f(full['top10']['mean'])} \\\\")
    return "\n".join(lines) + "\n"


def mechanism_all_table():
    lines = []
    for key, intervention, readout in (("zero/peak_dsi", "silencing", "peak"), ("zero/mean_dsi", "silencing", "mean response"),
                                       ("mean/peak_dsi", "mean replacement", "peak"),
                                       ("mean/mean_dsi", "mean replacement", "mean response")):
        m, s = FULL["mechanism"][key], FULL["readout_split"][key]
        rec = lambda e: pct(e["mean"])  # noqa: E731
        lines.append(f"{intervention} & {readout} & {rec(m['recovery']['exact_trajectory_linear_readout'])}\\% & "
                     f"{rec(m['recovery']['linear_trajectory_exact_readout'])}\\% & "
                     f"{rec(s['recovery']['linear_trajectory_exact_responses'])}\\% & "
                     f"{rec(s['recovery']['linear_trajectory_exact_ratio'])}\\% & {short(m['decision'])} & {short(s['decision'])} \\\\")
    return "\n".join(lines) + "\n"


def short(decision):
    return decision.replace(" dominates", "").replace("response stage", "responses")


def development_table():
    lines = []
    for name, label in (("Ref", "Reference"), ("A", "(a) mean replacement"), ("B", "(b) mean-response index"),
                        ("C", "(c) second order"), ("CfRead", "Linearised trajectory, exact readout"),
                        ("SplitRatio", "Floored linear responses, exact ratio")):
        values = " & ".join(f(cell(DEV["strengths"][s], name)["ratio"]["mean"], 2) for s in ("0.1", "0.25", "0.5", "0.75", "1.0"))
        lines.append(f"{label} & {values} \\\\")
    return "\n".join(lines) + "\n"


def flag_table():
    lines = []
    for k in CONFIG["flag_budgets"]:
        for fallback, label in (("zero", "zero"), ("clipped", "clipped first order")):
            r = CONF["flag_rule"][f"k{k}/{fallback}"]
            lines.append(f"{k} & {label} & {f(r['mae']['mean'])} & {pct(r['recovered']['mean'])}\\% & {f(r['seconds']['mean'], 0)} & "
                         f"{f(r['recovered_per_second']['mean'], 4)} & {'yes' if r['rule_pays'] else 'no'} \\\\")
    return "\n".join(lines) + "\n"


def tex_escape(text):
    return (str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_\\allowbreak{}").replace("%", "\\%").replace("&", "\\&")
            .replace("#", "\\#").replace("|", "$|$").replace("<=", "$\\le$").replace(">=", "$\\ge$").replace("~", "--"))


def minus(text):
    return re.sub(r"(?<![\w.])-(?=\d)", r"\\ensuremath{-}", text)


def unlabel(text):
    """The saved records name the companion study by its project label; the manuscript calls its estimate the reference."""
    return text.replace("AI-06", "the reference").replace("ai06", "reference")


def observed(value):
    if isinstance(value, float):
        return f(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        if "mean" in value and isinstance(value["mean"], float):
            return f(value["mean"])
        parts = [f"{tex_escape(k)}: {observed(v)}" for k, v in value.items()
                 if not isinstance(v, (dict, list)) and k not in ("models",)]
        return "; ".join(parts)
    return tex_escape(value)


def prediction_table(key):
    lines = []
    for p in CONF[key]:
        quantity = tex_escape(unlabel(p["quantity"]).replace("addendum.", "")).replace(".", ".\\allowbreak{}")
        lines.append(f"\\texttt{{{quantity}}} & {minus(tex_escape(unlabel(p['prediction'])))} & {unlabel(observed(p['observed']))} & "
                     f"{'holds' if p['holds'] else '\\textbf{fails}'} \\\\")
    return "\n".join(lines) + "\n"


def tables():
    return {"table-arms.tex": arm_table(), "table-mechanism.tex": mechanism_table(),
            "table-factorial.tex": factorial_table(), "table-mechanism-all.tex": mechanism_all_table(),
            "table-development.tex": development_table(), "table-flag.tex": flag_table(),
            "table-predictions.tex": prediction_table("predictions"),
            "table-addendum.tex": prediction_table("addendum_predictions")}


def render(n):
    lines = ["% Generated by paper/make_numbers.py from saved study outputs. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(n.items())]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = {"numbers.tex": render(numbers()), **tables()}
    dest = PAPER / "generated"
    if args.check:
        differ = [name for name, text in outputs.items() if not (dest / name).exists() or (dest / name).read_text() != text]
        if differ:
            sys.exit(f"Generated files differ from the saved outputs: {differ}")
        print(f"{len(outputs)} generated files match")
        return
    dest.mkdir(exist_ok=True)
    for name, text in outputs.items():
        (dest / name).write_text(text)
    print(f"wrote {len(outputs)} files, {len(numbers())} macros")


if __name__ == "__main__":
    main()
