"""AI-09 analysis: the scoring of PROTOCOL.md Sections 5-8, from AI-06's saved records and AI-09's runs.

  python -B study/code/analyse_arms.py --source development
  python -B study/code/analyse_arms.py --source confirmation     (refused until study/freeze.json exists)

Writes study/results/analysis/<source>/: models.csv (one row per model, strength and estimator cell), traces.csv
(trace-level linearisation errors), flag-rule.csv, timing.csv, reference-check.json and summary.json (ensemble
estimates; for confirmation also the bootstrap intervals, the decisions and the scored predictions).

Exact peak-DSI values use AI-06's independent readout (`revision_common.independent_metrics`) and the reference uses
AI-06's saved derivatives, so the reference reproduces AI-06's saved model-level values. Everything else uses
readouts.Readout on the stored traces and tangents. Every AI-06 and AI-09 record is checked against its SHA-256.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ai09_common import (AI06, AI06_RUNS, CONFIG, CONFIRMATION, DEVELOPMENT, NAMES, READOUTS, RESULTS, STUDY, ai06,
                         ai06_record, causal_frames, load_npz, require_development, save_json, sha256, stimulus,
                         utc_now)
import analyse_revision as ai06_scoring  # AI-06: top_budget, correlation, bounded_prediction, structural_scores

CUT = CONFIG["denominator_threshold"]
LARGE = CONFIG["important_effect_threshold"]
SELECTIVITY = CONFIG["primary_dsi_threshold"]
AMPLIFICATION = CONFIG["primary_amplification_threshold"]
BUDGET = CONFIG["screening_budget"]
INTERVENTIONS = ("zero", "mean")
KINDS = ("peak_dsi", "mean_dsi")
ARMS = {
    # name: (intervention, readout, estimator)
    "reference": ("zero", "peak_dsi", "first"),
    "a": ("mean", "peak_dsi", "first"),
    "b": ("zero", "mean_dsi", "first"),
    "c": ("zero", "peak_dsi", "second"),
}
FACTORIAL = [(i, r, e) for i in INTERVENTIONS for r in KINDS for e in ("first", "second")]
COUNTERFACTUALS = ("exact_trajectory_linear_readout", "linear_trajectory_exact_readout")
# Amendment 1 (PROTOCOL.md Section 13.1): on the linearised trajectory, one readout stage exact and the other linear.
SPLIT = ("linear_trajectory_exact_responses", "linear_trajectory_exact_ratio")
F64 = torch.float64


# ----------------------------------------------------------------------------------------------- loading


def ai09_record(model: str, key: str, name: str) -> dict[str, np.ndarray]:
    path = RESULTS / "runs" / model / key / name
    metadata = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != metadata["sha256"]:
        raise RuntimeError(f"AI-09 record checksum mismatch: {path}")
    return {**load_npz(path), "_meta": metadata}


def full(values: np.ndarray, types: np.ndarray, fill: np.ndarray | float = 0.0) -> np.ndarray:
    """Expand an array over the simulated types to all 65 types (structural zeros get `fill`)."""
    out = np.empty((65,) + values.shape[1:], values.dtype)
    out[:] = fill
    out[types] = values
    return out


def strengths_for(model: str) -> list[float]:
    return sorted(CONFIG["strengths"] + (CONFIG["development_extra_strengths"] if model in DEVELOPMENT else []))


def code(strength: float) -> str:
    return f"s{round(100 * strength):03d}"


def slope(readout, base: torch.Tensor, direction: torch.Tensor, kind: str) -> torch.Tensor:
    """First derivative at s = 0 of s -> DSI(base + s * direction)."""
    from torch.func import jvp
    zero, one = torch.zeros((), dtype=F64), torch.ones((), dtype=F64)
    return jvp(lambda s: readout.dsi(base + s * direction, kind), (zero,), (one,))[1].detach()


class Panel:
    """Everything one model and stimulus contributes, at every strength."""

    def __init__(self, model: str, key: str, anatomy: dict[str, np.ndarray]):
        from readouts import Readout
        self.model, self.key = model, key
        self.stim = stimulus(key)
        self.count = causal_frames(key)
        self.readout = Readout(self.stim, self.count)
        base = ai06_record(model, key, "baseline.npz")
        self.v0 = base["trace"]
        measured = ai06.independent_metrics(self.v0, self.stim)
        self.base_peak_dsi = measured["dsi"]
        self.base_denominator = measured["denominator"]
        self.base_max = max(float(measured["max_abs_voltage"]), CUT)
        self.anatomy = anatomy["anatomy_hop3"]
        self.gradient = ai06_record(model, key, "gradient.npz")["gradient"].astype(np.float64)  # (8, 65)
        v0 = torch.as_tensor(self.v0, dtype=F64)
        self.base_dsi = {"peak_dsi": self.base_peak_dsi,
                         "mean_dsi": self.readout.dsi(v0, "mean_dsi").numpy()}
        self.base_mean_denominator = self.readout(v0, "mean_dsi")["denominator"].numpy()
        # Derivatives F'(0) and F''(0), (8, 65), for each intervention and readout.
        self.tangents, self.first, self.second, self.ai09_seconds = {}, {}, {}, {}
        for intervention in INTERVENTIONS:
            record = ai09_record(model, key, f"tangents-{intervention}.npz")
            self.ai09_seconds[f"tangents_{intervention}"] = record["_meta"]["seconds"]
            types = record["types"]
            J = torch.as_tensor(record["J"], dtype=F64)
            K = torch.as_tensor(record["K"], dtype=F64)
            self.tangents[intervention] = (types, J, K)
            for kind in KINDS:
                g, h = self.readout.path_derivatives(v0, J, K, kind)
                self.first[intervention, kind] = full(g.detach().numpy(), types).T
                self.second[intervention, kind] = full(h.detach().numpy(), types).T

    def exact(self, intervention: str, strength: float) -> dict:
        """Exact traces of all 65 types and their readouts: dsi per kind (8, 65), denominators, amplification."""
        if intervention == "zero":
            traces = np.stack([ai06_record(self.model, self.key, f"{code(strength)}-cell{i:02d}.npz")["trace"]
                               for i in range(65)])
            seconds = None
        else:
            record = ai09_record(self.model, self.key, f"exact-mean-{code(strength)}.npz")
            traces = full(record["trace"], record["types"], self.v0)
            seconds = record["_meta"]["seconds"]
        measured = [ai06.independent_metrics(trace, self.stim) for trace in traces]
        values = torch.as_tensor(traces, dtype=F64)
        mean_readout = self.readout(values, "mean_dsi")
        return {
            "traces": traces,
            "dsi": {"peak_dsi": np.stack([m["dsi"] for m in measured], 1),
                    "mean_dsi": mean_readout["dsi"].numpy().T},
            "peak_denominator": np.stack([m["denominator"] for m in measured], 1),
            "mean_denominator": mean_readout["denominator"].numpy().T,
            "amplification": np.array([float(m["max_abs_voltage"]) / self.base_max for m in measured]),
            "seconds": seconds,
        }

    def counterfactuals(self, intervention: str, kind: str, strength: float, traces: np.ndarray) -> dict:
        """Section 8.1 estimators and the Section 13.1 readout split, each (8, 65)."""
        from torch.func import jvp
        types, J, _ = self.tangents[intervention]
        v0 = torch.as_tensor(self.v0, dtype=F64)
        change = torch.as_tensor(traces, dtype=F64) - v0
        linear_readout = slope(self.readout, v0, change, kind).numpy().T
        linear_trajectory = (self.readout.dsi(v0 + strength * J, kind) - self.readout.dsi(v0, kind)).numpy()
        # Section 13.1: response stage P and ratio stage G, one exact and one linear, on v0 + s J.
        zero, one = torch.zeros((), dtype=F64), torch.ones((), dtype=F64)
        ratio = lambda response: self.readout.ratio(response)["dsi"]  # noqa: E731
        p0 = self.readout.responses(v0, kind)
        exact_responses = self.readout.responses(v0 + strength * J, kind)
        response_slope = jvp(lambda t: self.readout.responses(v0 + t * J, kind), (zero,), (one,))[1].detach()
        exact_responses_linear_ratio = jvp(lambda t: ratio(p0 + t * (exact_responses - p0)), (zero,), (one,))[1]
        # Linear responses can fall below zero, where the index is undefined; they are floored at zero (13.1).
        linear_responses_exact_ratio = ratio((p0 + strength * response_slope).clamp_min(0)) - ratio(p0)
        return {"exact_trajectory_linear_readout": linear_readout,
                "linear_trajectory_exact_readout": full(linear_trajectory, types).T,
                "linear_trajectory_exact_responses": full(exact_responses_linear_ratio.detach().numpy(), types).T,
                "linear_trajectory_exact_ratio": full(linear_responses_exact_ratio.detach().numpy(), types).T}

    def trace_errors(self, intervention: str, strength: float, traces: np.ndarray, amplification: np.ndarray):
        """Section 8.1: relative errors of the linear and quadratic trajectory approximations, per simulated type."""
        types, J, K = self.tangents[intervention]
        change = torch.as_tensor(traces[types], dtype=F64) - torch.as_tensor(self.v0, dtype=F64)
        linear = change - strength * J
        quadratic = linear - 0.5 * strength**2 * K
        size = change.flatten(1).norm(dim=1)
        rows = []
        for position, cell in enumerate(types):
            if float(size[position]) <= 0 or amplification[cell] > AMPLIFICATION:
                continue
            rows.append({"model": self.model, "stimulus": self.key, "intervention": intervention,
                         "strength": strength, "cell": NAMES[cell],
                         "linear_relative_error": float(linear[position].norm() / size[position]),
                         "quadratic_relative_error": float(quadratic[position].norm() / size[position])})
        return rows


# ----------------------------------------------------------------------------------------------- scoring


def estimate_for(panel: Panel, intervention: str, kind: str, estimator: str, strength: float, cf: dict) -> np.ndarray:
    """Estimates (8, 65) for one factorial cell, counterfactual or mismatch."""
    if (intervention, kind) == ("zero", "peak_dsi"):
        first = panel.gradient  # AI-06's saved derivatives: the reference
    else:
        first = panel.first[intervention, kind]
    if estimator == "first":
        return strength * first
    if estimator == "second":
        return strength * first + 0.5 * strength**2 * panel.second[intervention, kind]
    if estimator in COUNTERFACTUALS + SPLIT:
        return cf[estimator]
    if estimator == "mismatch":
        other = "mean" if intervention == "zero" else "zero"
        slope_other = panel.gradient if other == "zero" else panel.first["mean", "peak_dsi"]
        return strength * slope_other
    raise ValueError(estimator)


def score_units(panel: Panel, exact: dict, kind: str, estimate: np.ndarray, strength: float) -> list[dict]:
    """AI-06's primary-stratum scoring of one estimator, one row per eligible readout condition."""
    rows = []
    change = exact["dsi"][kind] - panel.base_dsi[kind][:, None]
    for r, name in enumerate(READOUTS):
        responsive = bool(panel.base_denominator[r] > CUT)
        if not responsive or panel.base_peak_dsi[r] < SELECTIVITY:
            continue
        defined = (exact["peak_denominator"][r] > CUT) & np.isfinite(change[r]) & np.isfinite(estimate[r])
        if kind == "mean_dsi":
            mean_dropped = defined & ~(exact["mean_denominator"][r] > CUT)
            defined = defined & (exact["mean_denominator"][r] > CUT)
        else:
            mean_dropped = np.zeros(65, bool)
        eligible = defined & (exact["amplification"] <= AMPLIFICATION)
        if not eligible.any():
            continue
        ex, est = change[r, eligible], estimate[r, eligible]
        truth = np.abs(ex) >= LARGE
        clipped = ai06_scoring.bounded_prediction(est, panel.base_dsi[kind][r])
        screen = ai06_scoring.top_budget(np.abs(est), truth, BUDGET)
        anatomy = ai06_scoring.top_budget(panel.anatomy[r, eligible], truth, BUDGET)
        rows.append({
            "stimulus": panel.key, "readout": name, "n": int(eligible.sum()), "n_large": int(truth.sum()),
            "abs_error_sum": float(np.abs(est - ex).sum()), "abs_exact_sum": float(np.abs(ex).sum()),
            "clipped_abs_error_sum": float(np.abs(clipped - ex).sum()), "signed_error_sum": float((est - ex).sum()),
            "wrong_sign": int(np.sum(est[truth] * ex[truth] < 0)), "zero_estimate": int(np.sum(est[truth] == 0)),
            "top10": screen["precision"], "anatomy_top10": anatomy["precision"],
            "rho": ai06_scoring.correlation(np.abs(est), np.abs(ex)),
            "rho_large": ai06_scoring.correlation(np.abs(est[truth]), np.abs(ex[truth])),
            "mean_denominator_dropped": int(mean_dropped.sum()),
        })
    return rows


def pool(units: list[dict]) -> dict:
    """Model-level outcomes: cases pooled within the model, precisions averaged over readout conditions."""
    n = sum(u["n"] for u in units)
    large = sum(u["n_large"] for u in units)
    mae = sum(u["abs_error_sum"] for u in units) / n
    zero = sum(u["abs_exact_sum"] for u in units) / n
    clipped = sum(u["clipped_abs_error_sum"] for u in units) / n
    rho = [u["rho"] for u in units if u["rho"] is not None]
    rho_large = [u["rho_large"] for u in units if u["rho_large"] is not None]
    return {
        "units": len(units), "effects": n, "large_effects": large, "mae": mae, "zero_mae": zero,
        "mae_minus_zero": mae - zero, "ratio": mae / zero if zero > 0 else None,
        "clipped_mae": clipped, "clipped_ratio": clipped / zero if zero > 0 else None,
        "bias": sum(u["signed_error_sum"] for u in units) / n,
        "wrong_sign_rate": sum(u["wrong_sign"] for u in units) / large if large else None,
        "zero_estimate_rate": sum(u["zero_estimate"] for u in units) / large if large else None,
        "top10": float(np.mean([u["top10"] for u in units])),
        "anatomy_top10": float(np.mean([u["anatomy_top10"] for u in units])),
        "top10_minus_anatomy": float(np.mean([u["top10"] - u["anatomy_top10"] for u in units])),
        "rho": float(np.mean(rho)) if rho else None, "rho_large": float(np.mean(rho_large)) if rho_large else None,
        "mean_denominator_dropped": sum(u["mean_denominator_dropped"] for u in units),
    }


def flag_rule(cases: list[tuple], derivative_seconds: float, wall: dict, k: int, fallback: str) -> dict:
    """Section 8.6 for one model at one strength.

    cases: (stimulus, cell, exact, first-order estimate, clipped first-order estimate, flag score) for every scored
    case of the reference; wall: {(stimulus, cell): exact simulation seconds}. Within each stimulus, the k types with
    the largest flag score (the largest absolute first-order estimate over the eligible readouts) are simulated; the
    rest are predicted by zero or by the clipped first-order estimate.
    """
    flagged = set()
    for key in sorted({c[0] for c in cases}):
        score = np.zeros(65)
        for stim, cell, _, _, _, value in cases:
            if stim == key:
                score[cell] = max(score[cell], value)
        flagged |= {(key, int(cell)) for cell in np.argsort(-score, kind="stable")[:k]}
    errors, zero = [], []
    for stim, cell, exact, first, clipped, _ in cases:
        zero.append(abs(exact))
        if (stim, cell) in flagged:
            errors.append(0.0)
        else:
            errors.append(abs(exact) if fallback == "zero" else abs(clipped - exact))
    mae, zero_mae = float(np.mean(errors)), float(np.mean(zero))
    cost = derivative_seconds + sum(wall.get(item, 0.0) for item in flagged)
    exhaustive = sum(wall.values())
    recovered = 1 - mae / zero_mae
    return {"k": k, "fallback": fallback, "mae": mae, "zero_mae": zero_mae, "recovered": recovered,
            "seconds": cost, "recovered_per_second": recovered / cost if cost > 0 else None,
            "exhaustive_seconds": exhaustive, "exhaustive_per_second": 1 / exhaustive if exhaustive > 0 else None}


def ai06_timing(model: str, key: str, strength: float):
    """AI-06's recorded derivative time and per-type simulation times (None where the records were reused)."""
    gradient = json.loads((AI06_RUNS / model / key / "gradient.json").read_text())
    if gradient.get("reused"):
        return None, {}
    derivative = gradient["forward_s"] + sum(gradient["backward_s"])
    wall = {}
    for cell in range(65):
        record = json.loads((AI06_RUNS / model / key / f"{code(strength)}-cell{cell:02d}.json").read_text())
        if record.get("reused"):
            return None, {}
        wall[key, cell] = record.get("wall_seconds", 0.0)
    return derivative, wall


def analyse_model(model: str, anatomy: dict) -> dict:
    rows, traces, flags, timing = [], [], [], {}
    units = {}  # (strength, intervention, kind, estimator) -> unit rows
    flag_cases = {}  # strength -> cases
    ai06_times, walls, available = {"derivative": 0.0, "exact_s100": 0.0, "exact_s050": 0.0}, {}, True
    for key in CONFIG["stimuli"]:
        panel = Panel(model, key, anatomy)
        for name, seconds in panel.ai09_seconds.items():
            timing[name] = timing.get(name, 0.0) + seconds
        for strength in strengths_for(model):
            for intervention in INTERVENTIONS:
                exact = panel.exact(intervention, strength)
                if exact["seconds"] is not None:
                    timing[f"exact_mean_{code(strength)}"] = timing.get(f"exact_mean_{code(strength)}", 0.0) + exact["seconds"]
                traces += panel.trace_errors(intervention, strength, exact["traces"], exact["amplification"])
                for kind in KINDS:
                    cf = panel.counterfactuals(intervention, kind, strength, exact["traces"])
                    estimators = ["first", "second", *COUNTERFACTUALS, *SPLIT] + (["mismatch"] if kind == "peak_dsi" else [])
                    for estimator in estimators:
                        estimate = estimate_for(panel, intervention, kind, estimator, strength, cf)
                        units.setdefault((strength, intervention, kind, estimator), []).extend(
                            score_units(panel, exact, kind, estimate, strength))
                    if (intervention, kind) == ("zero", "peak_dsi"):
                        flag_cases.setdefault(strength, []).extend(rule_cases(panel, exact, strength))
                del exact
        derivative, wall = ai06_timing(model, key, 1.0)
        _, wall_half = ai06_timing(model, key, 0.5)
        if derivative is None or not wall or not wall_half:
            available = False
        else:
            ai06_times["derivative"] += derivative
            ai06_times["exact_s100"] += sum(wall.values())
            ai06_times["exact_s050"] += sum(wall_half.values())
            walls.update(wall)
        del panel
    timing.update({f"ai06_{name}": value if available else None for name, value in ai06_times.items()})
    for (strength, intervention, kind, estimator), unit_rows in units.items():
        rows.append({"model": model, "cluster": ai06.CLUSTERS[model], "strength": strength,
                     "intervention": intervention, "readout": kind, "estimator": estimator, **pool(unit_rows)})
    if available:
        for k in CONFIG["flag_budgets"]:
            for fallback in ("zero", "clipped"):
                flags.append({"model": model, "strength": 1.0,
                              **flag_rule(flag_cases[1.0], ai06_times["derivative"], walls, k, fallback)})
    return {"rows": rows, "traces": traces, "flags": flags, "timing": {"model": model, **timing}}


def rule_cases(panel: Panel, exact: dict, strength: float) -> list[tuple]:
    """Reference cases for the flag rule: (stimulus, cell, exact, first, clipped first, flag score)."""
    cases = []
    change = exact["dsi"]["peak_dsi"] - panel.base_peak_dsi[:, None]
    estimate = strength * panel.gradient
    for r in range(8):
        if not panel.base_denominator[r] > CUT or panel.base_peak_dsi[r] < SELECTIVITY:
            continue
        eligible = ((exact["peak_denominator"][r] > CUT) & np.isfinite(change[r])
                    & (exact["amplification"] <= AMPLIFICATION))
        clipped = ai06_scoring.bounded_prediction(estimate[r], panel.base_peak_dsi[r])
        for cell in np.flatnonzero(eligible):
            cases.append((panel.key, int(cell), float(change[r, cell]), float(estimate[r, cell]),
                          float(clipped[cell]), float(abs(estimate[r, cell]))))
    return cases


# ----------------------------------------------------------------------------------------------- reference check


def reference_check(rows: list[dict], source: str) -> dict:
    """The reference must reproduce AI-06's saved model-level values (PROTOCOL.md Section 5)."""
    table = list(csv.DictReader((AI06 / f"study/results/analysis/{source}/models.csv").open()))
    fields = {"effects": "effects", "large_effects": "important_effects", "mae": "mae", "zero_mae": "zero_mae",
              "clipped_mae": "bounded_mae", "bias": "bias", "top10": "top10_precision",
              "wrong_sign_rate": "opposite_sign_rate", "zero_estimate_rate": "zero_prediction_rate"}
    worst, compared, missing = 0.0, 0, []
    for row in rows:
        if (row["intervention"], row["readout"], row["estimator"]) != ARMS["reference"]:
            continue
        match = [t for t in table if t["model"] == row["model"] and float(t["strength"]) == row["strength"]
                 and t["dsi_threshold"] == "0.2" and t["amplification_limit"] == "100.0"]
        gradient = [t for t in match if t["method"] == "gradient"]
        anatomy = [t for t in match if t["method"] == "anatomy_hop3"]
        if len(gradient) != 1 or len(anatomy) != 1:
            missing.append((row["model"], row["strength"]))
            continue
        for ours, theirs in fields.items():
            if gradient[0][theirs] == "":
                continue
            worst = max(worst, abs(float(row[ours]) - float(gradient[0][theirs])))
            compared += 1
        worst = max(worst, abs(row["anatomy_top10"] - float(anatomy[0]["top10_precision"])))
        compared += 1
    return {"status": "pass" if worst <= 1e-9 and not missing and compared else "fail",
            "largest_difference": worst, "values_compared": compared, "missing": missing}


# ----------------------------------------------------------------------------------------------- summaries


def bootstrap_indices(models: int) -> np.ndarray:
    """One shared set of resamples, so that every paired difference is resampled together."""
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    return rng.integers(0, models, (CONFIG["bootstrap_replicates"], models))


def interval(values, indices=None) -> dict:
    """Mean, range and (with shared resamples) the 95% percentile interval over models.

    A model without a value (for example no large effect, so no wrong-sign rate) is left out of the mean and of each
    resample's mean, as AI-06 left out undefined values; the count of models with a value is reported.
    """
    values = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    present = ~np.isnan(values)
    if not present.any():
        raise ValueError("No model-level value in an ensemble summary")
    result = {"mean": float(values[present].mean()), "min": float(values[present].min()),
              "max": float(values[present].max()), "models": int(present.sum())}
    if indices is not None:
        resampled = values[indices]
        means = np.nanmean(np.where(np.isnan(resampled), np.nan, resampled), axis=1)
        result["interval"] = [float(x) for x in np.nanquantile(means, [0.025, 0.975])]
    return result


def sign_decision(summary: dict, below: str, above: str, neither: str) -> str:
    low, high = summary["interval"]
    return below if high < 0 else above if low > 0 else neither


def dominance(reductions: dict[str, np.ndarray], improves: dict[str, bool], indices: np.ndarray) -> dict:
    """PROTOCOL.md Section 7: the largest reduction in error ratio among improving arms, separated from the next."""
    candidates = sorted((arm for arm in reductions if improves[arm]), key=lambda arm: -float(np.mean(reductions[arm])))
    if not candidates:
        return {"decision": "no arm improves", "order": []}
    if len(candidates) == 1:
        return {"decision": f"({candidates[0]}) dominates", "order": candidates}
    gap = interval(np.asarray(reductions[candidates[0]]) - np.asarray(reductions[candidates[1]]), indices)
    if gap["interval"][0] > 0:
        return {"decision": f"({candidates[0]}) dominates", "order": candidates, "gap": gap}
    return {"decision": "no single arm dominates", "order": candidates, "gap": gap}


def series(rows, models, strength, cell, field):
    lookup = {(r["model"], r["strength"], r["intervention"], r["readout"], r["estimator"]): r for r in rows}
    return np.array([np.nan if lookup[m, strength, *cell][field] is None else lookup[m, strength, *cell][field]
                     for m in models], dtype=float)


def summarise(rows, traces, flags, timing, models, confirmation: bool) -> dict:
    indices = bootstrap_indices(len(models)) if confirmation else None
    out = {"models": models, "strengths": {}}
    strengths = sorted({r["strength"] for r in rows})
    fields = ("ratio", "mae", "zero_mae", "mae_minus_zero", "clipped_ratio", "bias", "wrong_sign_rate",
              "zero_estimate_rate", "top10", "anatomy_top10", "top10_minus_anatomy", "rho", "rho_large")
    for strength in strengths:
        block = {"cells": {}}
        for cell in sorted({(r["intervention"], r["readout"], r["estimator"]) for r in rows}):
            entry = {}
            for field in fields:
                values = series(rows, models, strength, cell, field)
                if np.isnan(values).all():
                    continue
                entry[field] = interval(values, indices)
            block["cells"]["/".join(cell)] = entry
        if confirmation:
            block.update(decisions(rows, models, strength, indices))
        out["strengths"][str(strength)] = block
    trace_summary = {}
    for intervention in INTERVENTIONS:
        for strength in strengths:
            per_model = []
            for model in models:
                selected = [t for t in traces if t["model"] == model and t["intervention"] == intervention
                            and t["strength"] == strength]
                if not selected:
                    continue
                linear = np.array([t["linear_relative_error"] for t in selected])
                quadratic = np.array([t["quadratic_relative_error"] for t in selected])
                per_model.append({"linear": float(np.median(linear)), "quadratic": float(np.median(quadratic)),
                                  "quadratic_over_linear": float(np.median(quadratic / linear))})
            if len(per_model) == len(models):
                trace_summary[f"{intervention}/{strength}"] = {
                    name: interval([p[name] for p in per_model], indices)
                    for name in ("linear", "quadratic", "quadratic_over_linear")}
    out["trace_level"] = trace_summary
    if flags:
        rule = {}
        for k in CONFIG["flag_budgets"]:
            for fallback in ("zero", "clipped"):
                selected = [f for f in flags if f["k"] == k and f["fallback"] == fallback]
                if len(selected) != len(models):
                    continue
                entry = {name: interval([f[name] for f in selected], indices)
                         for name in ("mae", "recovered", "seconds", "recovered_per_second", "exhaustive_per_second")}
                difference = [f["recovered_per_second"] - f["exhaustive_per_second"] for f in selected]
                entry["minus_exhaustive_per_second"] = interval(difference, indices)
                if confirmation:
                    entry["rule_pays"] = entry["minus_exhaustive_per_second"]["interval"][0] > 0
                rule[f"k{k}/{fallback}"] = entry
        out["flag_rule"] = rule
    out["timing"] = {}
    for name in sorted({k for t in timing for k in t if k != "model"}):
        values = [t.get(name) for t in timing]
        if all(v is not None for v in values):
            out["timing"][name] = interval(values)
    ratios = {"reference": ("ai06_derivative", "ai06_exact_s100"), "a": ("tangents_mean", "exact_mean_s100"),
              "b": ("tangents_zero", "ai06_exact_s100"), "c": ("tangents_zero", "ai06_exact_s100"),
              "a_exact_over_ai06_exact": ("exact_mean_s100", "ai06_exact_s100")}
    for arm, (top, bottom) in ratios.items():
        values = [t.get(top) / t.get(bottom) if t.get(top) and t.get(bottom) else None for t in timing]
        if all(v is not None for v in values):
            out["timing"][f"ratio/{arm}"] = {**interval(values), "all_above_1": bool(min(values) > 1)}
    return out


def decisions(rows, models, strength, indices) -> dict:
    """PROTOCOL.md Section 7 at one strength (confirmation only)."""
    get = lambda cell, field: series(rows, models, strength, cell, field)  # noqa: E731
    reference = ARMS["reference"]
    result, reductions, improves = {"arms": {}}, {}, {}
    for arm in ("a", "b", "c"):
        cell = ARMS[arm]
        entry = {}
        entry["zero"] = sign_decision(interval(get(cell, "mae_minus_zero"), indices),
                                      "beats zero", "worse than zero", "not distinguishable from zero")
        change = get(cell, "ratio") - get(reference, "ratio")
        entry["change_from_reference"] = interval(change, indices)
        entry["change_decision"] = sign_decision(entry["change_from_reference"], "improves on AI-06",
                                                 "worse than AI-06", "no detectable change")
        bias = interval(get(cell, "bias"), indices)
        entry["bias_decision"] = ("biased upwards" if bias["interval"][0] > 0 else
                                  "biased downwards" if bias["interval"][1] < 0 else "no detectable bias")
        sign_change = interval(get(cell, "wrong_sign_rate") - get(reference, "wrong_sign_rate"), indices)
        entry["wrong_sign_change"] = sign_change
        entry["wrong_sign_decision"] = sign_decision(sign_change, "fewer wrong signs", "more wrong signs",
                                                     "no detectable change")
        screen = interval(get(cell, "top10_minus_anatomy"), indices)
        entry["screen_decision"] = "screen kept" if screen["interval"][0] > 0 else "screen not kept"
        if arm == "c":
            entry["top10_minus_first_order"] = interval(get(cell, "top10") - get(reference, "top10"), indices)
        result["arms"][arm] = entry
        reductions[arm] = -change
        improves[arm] = entry["change_decision"] == "improves on AI-06"
    result["dominance"] = dominance(reductions, improves, indices)
    mechanism = {}
    for intervention in INTERVENTIONS:
        for kind in KINDS:
            first = get((intervention, kind, "first"), "mae")
            recovery = {name: 1 - get((intervention, kind, name), "mae") / first for name in COUNTERFACTUALS}
            difference = interval(recovery["linear_trajectory_exact_readout"]
                                  - recovery["exact_trajectory_linear_readout"], indices)
            mechanism[f"{intervention}/{kind}"] = {
                "recovery": {name: interval(values, indices) for name, values in recovery.items()},
                "readout_minus_network": difference,
                "decision": sign_decision(difference, "network dominates", "readout dominates", "neither dominates"),
            }
    result["mechanism"] = mechanism
    split = {}
    for intervention in INTERVENTIONS:
        for kind in KINDS:
            first = get((intervention, kind, "first"), "mae")
            recovery = {name: 1 - get((intervention, kind, name), "mae") / first
                        for name in SPLIT + ("linear_trajectory_exact_readout",)}
            difference = interval(recovery["linear_trajectory_exact_ratio"]
                                  - recovery["linear_trajectory_exact_responses"], indices)
            split[f"{intervention}/{kind}"] = {
                "recovery": {name: interval(values, indices) for name, values in recovery.items()},
                "ratio_minus_responses": difference,
                "interaction": interval(recovery["linear_trajectory_exact_readout"]
                                        - recovery["linear_trajectory_exact_ratio"]
                                        - recovery["linear_trajectory_exact_responses"], indices),
                "decision": sign_decision(difference, "response stage dominates", "ratio dominates",
                                          "neither dominates"),
            }
    result["readout_split"] = split
    mismatch = interval(get(("mean", "peak_dsi", "mismatch"), "ratio") - get(ARMS["a"], "ratio"), indices)
    result["mismatch"] = {
        "zero_derivative_on_mean_target_minus_matched_ratio": mismatch,
        "zero_derivative_on_mean_target_top10_minus_anatomy": interval(
            get(("mean", "peak_dsi", "mismatch"), "top10_minus_anatomy"), indices),
        "mean_derivative_on_zero_target_minus_reference_ratio": interval(
            get(("zero", "peak_dsi", "mismatch"), "ratio") - get(reference, "ratio"), indices)}
    return result


# ----------------------------------------------------------------------------------------------- predictions


def score_predictions(summary: dict) -> list[dict]:
    """The predictions of PREDICTIONS.md (hashed in PREDICTIONS-RECORD.json), evaluated at strength 1."""
    block = summary["strengths"]["1.0"]
    cell = lambda arm: block["cells"]["/".join(ARMS[arm])]  # noqa: E731
    arms = block["arms"]
    within = lambda value, low, high: low <= value <= high  # noqa: E731
    checks = []

    def add(key, prediction, observed, holds):
        checks.append({"quantity": key, "prediction": prediction, "observed": observed, "holds": bool(holds)})

    add("a.change_from_reference.decision", "= improves on AI-06", arms["a"]["change_decision"],
        arms["a"]["change_decision"] == "improves on AI-06")
    add("a.ratio.mean", "0.8 to 1.3", cell("a")["ratio"]["mean"], within(cell("a")["ratio"]["mean"], 0.8, 1.3))
    add("a.zero.decision", "not beats zero", arms["a"]["zero"], arms["a"]["zero"] != "beats zero")
    add("a.wrong_sign.rate", "0.08 to 0.20", cell("a")["wrong_sign_rate"]["mean"],
        within(cell("a")["wrong_sign_rate"]["mean"], 0.08, 0.20))
    add("a.top10.precision", "0.55 to 0.85", cell("a")["top10"]["mean"], within(cell("a")["top10"]["mean"], 0.55, 0.85))
    add("a.screen.decision", "= screen kept", arms["a"]["screen_decision"], arms["a"]["screen_decision"] == "screen kept")
    add("b.change_from_reference.decision", "not improves on AI-06", arms["b"]["change_decision"],
        arms["b"]["change_decision"] != "improves on AI-06")
    add("b.ratio.mean", "1.2 to 1.7", cell("b")["ratio"]["mean"], within(cell("b")["ratio"]["mean"], 1.2, 1.7))
    add("b.zero.decision", "= worse than zero", arms["b"]["zero"], arms["b"]["zero"] == "worse than zero")
    add("b.wrong_sign.rate", "0.15 to 0.30", cell("b")["wrong_sign_rate"]["mean"],
        within(cell("b")["wrong_sign_rate"]["mean"], 0.15, 0.30))
    add("b.top10.precision", "0.6 to 0.85", cell("b")["top10"]["mean"], within(cell("b")["top10"]["mean"], 0.6, 0.85))
    add("b.screen.decision", "= screen kept", arms["b"]["screen_decision"], arms["b"]["screen_decision"] == "screen kept")
    add("c.change_from_reference.decision", "not improves on AI-06", arms["c"]["change_decision"],
        arms["c"]["change_decision"] != "improves on AI-06")
    add("c.ratio.mean", "1.2 to 2.2", cell("c")["ratio"]["mean"], within(cell("c")["ratio"]["mean"], 1.2, 2.2))
    add("c.zero.decision", "= worse than zero", arms["c"]["zero"], arms["c"]["zero"] == "worse than zero")
    add("c.wrong_sign.rate", "0.15 to 0.35", cell("c")["wrong_sign_rate"]["mean"],
        within(cell("c")["wrong_sign_rate"]["mean"], 0.15, 0.35))
    c_top = arms["c"]["top10_minus_first_order"]["mean"]
    add("c.top10.minus_first_order", "-0.05 to 0.05", c_top, within(c_top, -0.05, 0.05))
    add("c.screen.decision", "= screen kept", arms["c"]["screen_decision"], arms["c"]["screen_decision"] == "screen kept")
    biases = {arm: cell(arm)["bias"]["mean"] for arm in ("a", "b", "c")}
    add("bias.all_arms", "|mean signed error| <= 0.03 in arms a, b and c", biases,
        all(abs(v) <= 0.03 for v in biases.values()))
    add("dominance.decision", "= (a) dominates", block["dominance"]["decision"],
        block["dominance"]["decision"] == "(a) dominates")
    mechanism = block["mechanism"]["zero/peak_dsi"]
    add("mechanism.decision", "= readout dominates", mechanism["decision"], mechanism["decision"] == "readout dominates")
    cf_ratio = block["cells"]["zero/peak_dsi/linear_trajectory_exact_readout"]["ratio"]["mean"]
    add("mechanism.exact_readout_on_linear_trajectory.ratio", "0.5 to 1.2", cf_ratio, within(cf_ratio, 0.5, 1.2))
    trace = summary["trace_level"]["zero/1.0"]["quadratic_over_linear"]["mean"]
    add("trace.quadratic_over_linear_error",
        "ensemble mean of per-model median (quadratic / linear relative trajectory error) above 0.75", trace, trace > 0.75)
    mismatch = block["mismatch"]
    holds = (mismatch["zero_derivative_on_mean_target_minus_matched_ratio"]["interval"][0] > 0
             and mismatch["zero_derivative_on_mean_target_top10_minus_anatomy"]["interval"][0] > 0)
    add("mismatch.zero_derivative_for_mean_effects", "ratio above the matched estimate's and top-10 above anatomy "
        "(both intervals above 0)", {k: v["mean"] for k, v in mismatch.items()}, holds)
    runtime_c = summary["timing"].get("ratio/c", {})
    add("runtime.c_estimate_over_exact", "above 1 in every model", runtime_c, runtime_c.get("all_above_1", False))
    runtime_a = summary["timing"].get("ratio/a_exact_over_ai06_exact", {})
    add("runtime.a_exact_over_ai06_exact", "0.8 to 1.25", runtime_a.get("mean"),
        runtime_a.get("mean") is not None and within(runtime_a["mean"], 0.8, 1.25))
    pays = {k: summary["flag_rule"][f"k{k}/zero"]["rule_pays"] for k in CONFIG["flag_budgets"]}
    add("d.rule_pays", "= false at k = 5, 10 and 20", pays, not any(pays.values()))
    return checks


def score_addendum(summary: dict) -> list[dict]:
    """The development-informed predictions of PREDICTIONS-ADDENDUM.md (PROTOCOL.md Section 13.2), scored apart."""
    block, half = summary["strengths"]["1.0"], summary["strengths"]["0.5"]
    cells = block["cells"]
    cell = lambda arm: cells["/".join(ARMS[arm])]  # noqa: E731
    arms = block["arms"]
    within = lambda value, low, high: value is not None and low <= value <= high  # noqa: E731
    checks = []

    def add(key, prediction, observed, holds):
        checks.append({"quantity": f"addendum.{key}", "prediction": prediction, "observed": observed,
                       "holds": bool(holds)})

    def mean(entry, field):
        return entry[field]["mean"]

    add("a.ratio.mean", "0.9 to 1.3", mean(cell("a"), "ratio"), within(mean(cell("a"), "ratio"), 0.9, 1.3))
    half_a = half["cells"]["/".join(ARMS["a"])]["ratio"]["mean"]
    add("a.ratio_half.mean", "0.6 to 0.95 at strength 0.5", half_a, within(half_a, 0.6, 0.95))
    add("a.change_decision", "= improves on AI-06", arms["a"]["change_decision"],
        arms["a"]["change_decision"] == "improves on AI-06")
    add("a.wrong_sign.rate", "0.20 to 0.35", mean(cell("a"), "wrong_sign_rate"),
        within(mean(cell("a"), "wrong_sign_rate"), 0.20, 0.35))
    add("a.top10.precision", "0.50 to 0.70", mean(cell("a"), "top10"), within(mean(cell("a"), "top10"), 0.50, 0.70))
    add("b.ratio.mean", "1.5 to 2.2", mean(cell("b"), "ratio"), within(mean(cell("b"), "ratio"), 1.5, 2.2))
    add("b.change_decision", "= worse than AI-06", arms["b"]["change_decision"],
        arms["b"]["change_decision"] == "worse than AI-06")
    add("b.wrong_sign.rate", "0.10 to 0.25", mean(cell("b"), "wrong_sign_rate"),
        within(mean(cell("b"), "wrong_sign_rate"), 0.10, 0.25))
    add("b.top10.precision", "0.80 to 0.93", mean(cell("b"), "top10"), within(mean(cell("b"), "top10"), 0.80, 0.93))
    add("c.ratio.mean", "above 3", mean(cell("c"), "ratio"), mean(cell("c"), "ratio") > 3)
    add("c.change_decision", "= worse than AI-06", arms["c"]["change_decision"],
        arms["c"]["change_decision"] == "worse than AI-06")
    add("c.bias", "biased upwards, mean above +0.3", {"decision": arms["c"]["bias_decision"],
        "mean": mean(cell("c"), "bias")},
        arms["c"]["bias_decision"] == "biased upwards" and mean(cell("c"), "bias") > 0.3)
    add("c.wrong_sign.rate", "0.30 to 0.55", mean(cell("c"), "wrong_sign_rate"),
        within(mean(cell("c"), "wrong_sign_rate"), 0.30, 0.55))
    c_top = arms["c"]["top10_minus_first_order"]["mean"]
    add("c.top10.minus_first_order", "-0.06 to 0.02", c_top, within(c_top, -0.06, 0.02))
    biases = {arm: mean(cell(arm), "bias") for arm in ("a", "b")}
    add("bias.a_and_b", "|mean signed error| <= 0.04 in arms a and b", biases, all(abs(v) <= 0.04 for v in biases.values()))
    add("dominance.decision", "= (a) dominates", block["dominance"]["decision"],
        block["dominance"]["decision"] == "(a) dominates")
    for kind, label in (("peak_dsi", "zero_peak"), ("mean_dsi", "zero_mean")):
        decision = block["mechanism"][f"zero/{kind}"]["decision"]
        add(f"mechanism.{label}.decision", "= readout dominates", decision, decision == "readout dominates")
    cf = cells["zero/peak_dsi/linear_trajectory_exact_readout"]
    add("mechanism.exact_readout_on_linear_trajectory.ratio", "0.35 to 0.65", mean(cf, "ratio"),
        within(mean(cf, "ratio"), 0.35, 0.65))
    cf_zero = cf["mae_minus_zero"]["interval"][1] < 0
    add("mechanism.exact_readout_on_linear_trajectory.zero_decision", "= beats zero", cf["mae_minus_zero"], cf_zero)
    for kind, label in (("peak_dsi", "zero_peak"), ("mean_dsi", "zero_mean")):
        decision = block["readout_split"][f"zero/{kind}"]["decision"]
        add(f"split.{label}.decision", "= ratio dominates", decision, decision == "ratio dominates")
    exact_ratio = mean(cells["zero/peak_dsi/linear_trajectory_exact_ratio"], "ratio")
    add("split.exact_ratio.ratio", "0.5 to 0.85", exact_ratio, within(exact_ratio, 0.5, 0.85))
    exact_responses = mean(cells["zero/peak_dsi/linear_trajectory_exact_responses"], "ratio")
    add("split.exact_responses.ratio", "above 2", exact_responses, exact_responses > 2)
    trace = summary["trace_level"]["zero/1.0"]["quadratic_over_linear"]["mean"]
    add("trace.quadratic_over_linear_error", "above 0.9", trace, trace > 0.9)
    on_mean = mean(cells["mean/peak_dsi/mismatch"], "ratio")
    add("mismatch.zero_derivative_on_mean_target.ratio", "above 2.5", on_mean, on_mean > 2.5)
    on_zero = mean(cells["zero/peak_dsi/mismatch"], "ratio")
    add("mismatch.mean_derivative_on_zero_target.ratio", "1.0 to 1.3", on_zero, within(on_zero, 1.0, 1.3))
    runtime_c = summary["timing"].get("ratio/c", {}).get("mean")
    add("runtime.c_estimate_over_exact", "2.0 to 2.8", runtime_c, within(runtime_c, 2.0, 2.8))
    runtime_a = summary["timing"].get("ratio/a_exact_over_ai06_exact", {}).get("mean")
    add("runtime.a_exact_over_ai06_exact", "0.9 to 1.15", runtime_a, within(runtime_a, 0.9, 1.15))
    return checks


# ----------------------------------------------------------------------------------------------- main


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    models = DEVELOPMENT if args.source == "development" else CONFIRMATION
    for model in models:
        require_development(model)
        for key in CONFIG["stimuli"]:
            if not (RESULTS / "runs" / model / key / "complete.json").exists():
                raise RuntimeError(f"Missing completed AI-09 run: {model} {key}")
    torch.set_num_threads(CONFIG["cpu_threads"])
    anatomy, _ = ai06_scoring.structural_scores()
    rows, traces, flags, timing = [], [], [], []
    for model in models:
        result = analyse_model(model, anatomy)
        rows += result["rows"]
        traces += result["traces"]
        flags += result["flags"]
        timing.append(result["timing"])
        print(f"analysed {model}", flush=True)
    check = reference_check(rows, args.source)
    dest = RESULTS / "analysis" / args.source
    save_json(dest / "reference-check.json", check)
    if check["status"] != "pass":
        raise RuntimeError(f"The reference does not reproduce AI-06: {check}")
    save_csv(dest / "models.csv", rows)
    save_csv(dest / "traces.csv", traces)
    if flags:
        save_csv(dest / "flag-rule.csv", flags)
    save_csv(dest / "timing.csv", timing)
    confirmation = args.source == "confirmation"
    summary = summarise(rows, traces, flags, timing, models, confirmation)
    if confirmation:
        summary["predictions"] = score_predictions(summary)
        summary["addendum_predictions"] = score_addendum(summary)
    save_json(dest / "summary.json", {"source": args.source, "created": utc_now(), "models": models,
                                      "interpretation": ("Model-level estimates; development models are summarised "
                                                         "by means and ranges only, with no decisions."
                                                         if not confirmation else
                                                         "Means over the 46 confirmation models with 95% bootstrap "
                                                         "intervals over models; decisions as in PROTOCOL.md Section 7."),
                                      **summary})
    print(json.dumps({"source": args.source, "models": len(models), "rows": len(rows),
                      "reference_check": check["status"]}))


if __name__ == "__main__":
    main()
