"""Draw the manuscript figures (paper/figures/*.pdf) from the saved study outputs.

Figure 1  screening precision per model and method (main and inclusive screens, full silencing)
Figure 2  first-order estimate against exact DSI change for every scored silencing case (half and full strength)
Figure 3  per-model error of the first-order and clipped estimates against predicting zero change
Figure 4  development models: error and screening precision across five silencing strengths

Colours: first-order estimate blue, predicting zero change orange, clipped estimate aqua, structural
comparators grey (validated categorical slots 1-3 of the portfolio's reference palette).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
CONF = ROOT / "study/results/analysis/confirmation"
DEV = ROOT / "study/results/analysis/development"
FIG = PAPER / "figures"
CONFIG = json.loads((ROOT / "study/config.json").read_text())
MAIN_DSI, MAIN_AMP = CONFIG["primary_direction_selectivity_threshold"], CONFIG["primary_amplification_threshold"]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
WIDTH = 6.5

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def is_main(r):
    return float(r["dsi_threshold"]) == MAIN_DSI and r["amplification_limit"] != "" and float(r["amplification_limit"]) == MAIN_AMP


def is_inclusive(r):
    return float(r["dsi_threshold"]) == 0.0 and r["amplification_limit"] == ""


def jitter(n, width=0.16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


OUTDIR = FIG  # replaced by a temporary folder under --check


def save(fig, name):
    OUTDIR.mkdir(exist_ok=True)
    fig.savefig(OUTDIR / name, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def figure_screening(models):
    methods = [("gradient", "First-order estimate"), ("anatomy_hop3", "Anatomy, 3 steps"), ("anatomy_hop4", "Anatomy, 4 steps"),
               ("trained_hop3", "Trained weights, 3 steps"), ("input_current", "Input current"), ("direct_count", "Direct synapse counts")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.45), sharey=True)
    panels = [("a  Main screen: selective readouts", is_main, "top10_precision"),
              ("b  Inclusive screen: all responsive readouts", is_inclusive, "composite_top10_precision")]
    for ax, (title, select, field) in zip(axes, panels):
        sel = [r for r in models if select(r) and float(r["strength"]) == 1.0]
        for k, (method, _) in enumerate(methods):
            values = np.array([float(r[field]) for r in sel if r["method"] == method])
            colour = BLUE if method == "gradient" else (INK2 if method == "anatomy_hop3" else MUTED)
            ax.scatter(values, k + jitter(len(values), 0.2, seed=k), s=8, color=colour, alpha=0.55, linewidths=0, zorder=2)
            ax.plot([values.mean()] * 2, [k - 0.32, k + 0.32], color=INK, lw=1.6, solid_capstyle="round", zorder=3)
            ax.text(1.03, k, f"{values.mean():.3f}", va="center", ha="left", fontsize=7, color=INK,
                    transform=ax.get_yaxis_transform(), clip_on=False)
        if field == "top10_precision":
            chance = np.mean([float(r["top10_random_precision"]) for r in sel if r["method"] == "gradient"])
        else:
            chance = json.loads((ROOT / "analysis/results/posthoc-summary.json").read_text())["inclusive_random_precision"]["1"]
        ax.axvline(chance, color=MUTED, lw=0.9, zorder=1)
        ax.text(chance + 0.012, -0.8, "random ranking", ha="left", va="center", fontsize=6.6, color=INK2)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([m[1] for m in methods])
        ax.set_ylim(len(methods) - 0.5, -1.1)
        ax.set_xlim(0, 1.0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Top-10 precision (dots: models; bar: mean)")
        ax.set_title(title, pad=6)
        ax.text(1.03, -0.8, "mean", va="center", ha="left", fontsize=6.6, color=INK2, transform=ax.get_yaxis_transform(), clip_on=False)
    fig.tight_layout(w_pad=3.0)
    save(fig, "fig1-screening.pdf")


def figure_scatter(effects):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    lim = 1.5
    for ax, (strength, title) in zip(axes, [(0.5, "a  Half-strength silencing (s = 0.5)"), (1.0, "b  Full silencing (s = 1)")]):
        sel = [r for r in effects if float(r["strength"]) == strength and r["defined"] == "True"
               and float(r["baseline_dsi"]) >= MAIN_DSI and float(r["amplification_ratio"]) <= MAIN_AMP]
        x = np.array([float(r["exact"]) for r in sel])
        y = np.array([float(r["prediction"]) for r in sel])
        inside = np.abs(y) <= lim
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        for edge in (-0.1, 0.1):
            ax.axvline(edge, color=GRID, lw=0.8, zorder=1)
        ax.plot([-1, 1], [-1, 1], color=INK2, lw=0.9, zorder=3)
        ax.scatter(x[inside], y[inside], s=2.2, color=BLUE, alpha=0.18, linewidths=0, rasterized=True, zorder=2)
        above, below = y > lim, y < -lim
        ax.scatter(x[above], np.full(above.sum(), lim), marker="^", s=7, color=ORANGE, alpha=0.6, linewidths=0, zorder=4)
        ax.scatter(x[below], np.full(below.sum(), -lim), marker="v", s=7, color=ORANGE, alpha=0.6, linewidths=0, zorder=4)
        ax.text(-1.02, lim - 0.06, f"{above.sum()} above {lim:g}", fontsize=6.8, color=INK2, va="top")
        ax.text(-1.02, -lim + 0.06, f"{below.sum()} below −{lim:g}", fontsize=6.8, color=INK2, va="bottom")
        ax.text(0.98, 1.06, "estimate = exact", fontsize=6.8, color=INK2, ha="right", va="bottom")
        ax.set_xlim(-1.05, 1.0)
        ax.set_ylim(-lim - 0.08, lim + 0.08)
        ax.set_xlabel("Exact DSI change after silencing")
        ax.set_title(title, pad=6)
    axes[0].set_ylabel("First-order estimate of the change")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig2-predicted-vs-exact.pdf")


def figure_model_errors(models):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=False)
    for ax, (strength, title) in zip(axes, [(0.5, "a  Half-strength silencing (s = 0.5)"), (1.0, "b  Full silencing (s = 1)")]):
        g = [r for r in models if is_main(r) and r["method"] == "gradient" and float(r["strength"]) == strength]
        zero = np.array([float(r["zero_mae"]) for r in g])
        first = np.array([float(r["mae"]) for r in g])
        clipped = np.array([float(r["bounded_mae"]) for r in g])
        xmax, ymax = zero.max() * 1.12, max(first.max(), clipped.max()) * 1.08
        ax.plot([0, xmax], [0, xmax], color=INK2, lw=0.9, zorder=1)
        ax.scatter(zero, first, s=14, color=BLUE, alpha=0.85, linewidths=0.6, edgecolors="white", zorder=3, label="First-order estimate")
        ax.scatter(zero, clipped, s=14, color=AQUA, alpha=0.85, linewidths=0.6, edgecolors="white", zorder=2, marker="s",
                   label="First-order, clipped to [0, 1]")
        worse = int(np.sum(first > zero + 1e-12))
        ax.text(0.03, 0.97, f"First-order error larger than\nzero-change error in {worse} of {len(g)} models",
                transform=ax.transAxes, fontsize=6.8, color=INK2, va="top")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Mean absolute error, zero change")
        ax.set_title(title, pad=6)
    axes[0].set_ylabel("Mean absolute error, estimate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.0)
    save(fig, "fig3-model-errors.pdf")


def figure_strength(dev_models):
    main = [r for r in dev_models if is_main(r)]
    strengths = sorted({float(r["strength"]) for r in main})
    models = sorted({r["model"] for r in main})
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35))
    series = [("mae", "gradient", BLUE, "First-order estimate"), ("zero_mae", "gradient", ORANGE, "Predicting zero change"),
              ("bounded_mae", "gradient", AQUA, "Clipped to [0, 1]")]
    ax = axes[0]
    for field, method, colour, label in series:
        for model in models:
            vals = [float(next(r[field] for r in main if r["model"] == model and r["method"] == method and float(r["strength"]) == s)) for s in strengths]
            ax.plot(strengths, vals, color=colour, lw=0.7, alpha=0.3, zorder=1)
        means = [np.mean([float(r[field]) for r in main if r["method"] == method and float(r["strength"]) == s]) for s in strengths]
        ax.plot(strengths, means, color=colour, lw=2, marker="o", ms=4, zorder=3, label=label)
    ax.set_xlabel("Silencing strength s")
    ax.set_ylabel("Mean absolute DSI error")
    ax.set_title("a  Effect-size error", pad=6)
    ax.legend(loc="upper left", handlelength=1.6)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(bottom=0)
    ax = axes[1]
    for method, colour, label in [("gradient", BLUE, "First-order estimate"), ("anatomy_hop3", INK2, "Anatomy, 3 steps")]:
        for model in models:
            vals = [float(next(r["top10_precision"] for r in main if r["model"] == model and r["method"] == method and float(r["strength"]) == s)) for s in strengths]
            ax.plot(strengths, vals, color=colour, lw=0.7, alpha=0.3, zorder=1)
        means = [np.mean([float(r["top10_precision"]) for r in main if r["method"] == method and float(r["strength"]) == s]) for s in strengths]
        ax.plot(strengths, means, color=colour, lw=2, marker="o", ms=4, zorder=3, label=label)
    ax.set_xlabel("Silencing strength s")
    ax.set_ylabel("Top-10 precision")
    ax.set_title("b  Screening precision", pad=6)
    ax.legend(loc="lower right", handlelength=1.6)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.02)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4-strength.pdf")


def draw():
    models = rows(CONF / "models.csv")
    figure_screening(models)
    figure_scatter(rows(CONF / "effects.csv"))
    figure_model_errors(models)
    figure_strength(rows(DEV / "models.csv"))


def main():
    """Draw into a temporary folder; with --check compare with paper/figures, otherwise replace changed files only.

    The PDFs carry no dates, so unchanged inputs give identical bytes and an unchanged figure keeps its file (and the
    manuscript PDF built from it stays identical)."""
    import filecmp, shutil, sys, tempfile
    global OUTDIR
    with tempfile.TemporaryDirectory() as tmp:
        OUTDIR = Path(tmp)
        draw()
        new = sorted(OUTDIR.glob("*.pdf"))
        stale = [p.name for p in new if not (FIG / p.name).exists() or not filecmp.cmp(p, FIG / p.name, shallow=False)]
        if "--check" in sys.argv:
            if stale:
                sys.exit("figures out of date: " + ", ".join(stale))
            print(f"ok  {len(new)} figures match the saved outputs")
            return
        FIG.mkdir(exist_ok=True)
        for name in stale:
            shutil.copyfile(OUTDIR / name, FIG / name)
        print(f"figures: {len(stale)} rewritten, {len(new) - len(stale)} unchanged")


if __name__ == "__main__":
    main()
