"""Draw the manuscript figures (paper/figures/*.pdf) from the saved study outputs.

Figure 1  error ratio of every estimator at full and half strength (confirmation models, 95% intervals)
Figure 2  AI-06's first-order estimate and the exact readout on the linearised trajectory against the exact change
          (every scored silencing case at full strength; post hoc case file)
Figure 3  per-model error ratios of arms (a), (b) and the linearised-trajectory exact-readout estimate against AI-06's
Figure 4  development models: error ratio across five strengths

Colours follow AI-06's figures (the portfolio's reference palette).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RES = ROOT / "study/results/analysis"
CONF = json.loads((RES / "confirmation/summary.json").read_text())
DEV = json.loads((RES / "development/summary.json").read_text())
CASES = ROOT / "analysis/results/cases-silencing-peak-s100.npz"
FIG = Path(os.environ.get("AI09_FIGURE_DIR", PAPER / "figures"))  # the paper check redraws into a scratch folder

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
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

ESTIMATORS = [
    ("zero/peak_dsi/first", "Reference: first-order estimate", INK),
    ("mean/peak_dsi/first", "(a) mean replacement", BLUE),
    ("zero/mean_dsi/first", "(b) mean-response index", BLUE),
    ("zero/peak_dsi/second", "(c) second-order estimate", BLUE),
    ("zero/peak_dsi/exact_trajectory_linear_readout", "Exact trajectory, linearised readout", ORANGE),
    ("zero/peak_dsi/linear_trajectory_exact_readout", "Linearised trajectory, exact readout", ORANGE),
    ("zero/peak_dsi/linear_trajectory_exact_responses", "Exact responses, linearised ratio", AQUA),
    ("zero/peak_dsi/linear_trajectory_exact_ratio", "Floored linear responses, exact ratio", AQUA),
]


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None, "Producer": None})
    plt.close(fig)


def figure_ratios():
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    labels = []
    for row, (key, label, colour) in enumerate(ESTIMATORS):
        y = len(ESTIMATORS) - 1 - row
        for strength, filled, offset in (("1.0", True, 0.13), ("0.5", False, -0.13)):
            ratio = CONF["strengths"][strength]["cells"][key]["ratio"]
            low, high = ratio["interval"]
            ax.plot([low, high], [y + offset] * 2, color=colour, lw=1.2, solid_capstyle="butt")
            ax.plot(ratio["mean"], y + offset, "o", ms=4.2, color=colour, mfc=colour if filled else "white", mew=1.1)
        labels.append(label)
    ax.axvline(1, color=INK2, lw=0.9, ls="--")
    ax.text(1.03, len(ESTIMATORS) - 0.05, "predicting zero change", color=INK2, fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "5", "10", "20"])
    ax.set_xlim(0.3, 25)
    ax.set_yticks(range(len(ESTIMATORS)))
    ax.set_yticklabels(labels[::-1])
    ax.set_ylim(-0.6, len(ESTIMATORS) + 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("error ratio: mean absolute error / that of predicting zero change (log scale)")
    ax.plot([], [], "o", color=INK2, mfc=INK2, label="full strength")
    ax.plot([], [], "o", color=INK2, mfc="white", label="half strength")
    ax.legend(loc="lower right")
    save(fig, "fig1-error-ratios.pdf")


def figure_scatter():
    data = np.load(CASES)
    exact = data["exact"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharex=True, sharey=True)
    for ax, key, title, colour in ((axes[0], "first", "Reference: first-order estimate", BLUE),
                                   (axes[1], "linear_trajectory_exact_readout", "Linearised trajectory, exact readout", ORANGE)):
        estimate = np.clip(data[key], -1.5, 1.5)
        ax.scatter(exact, estimate, s=2, color=colour, alpha=0.18, lw=0, rasterized=True)
        ax.plot([-1, 1], [-1, 1], color=INK2, lw=0.8)
        ax.axhline(0, color=AXIS, lw=0.6)
        ax.axvline(0, color=AXIS, lw=0.6)
        ax.set_title(title)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("exact change in direction selectivity")
    axes[0].set_ylabel("estimated change (clipped to ±1.5)")
    save(fig, "fig2-predicted-vs-exact.pdf")


def model_rows():
    with open(RES / "confirmation/models.csv", newline="") as stream:
        return [r for r in csv.DictReader(stream) if r["strength"] == "1.0"]


def figure_models():
    table = model_rows()
    def ratios(cell):
        i, r, e = cell.split("/")
        return {m["model"]: float(m["ratio"]) for m in table if (m["intervention"], m["readout"], m["estimator"]) == (i, r, e)}
    reference = ratios("zero/peak_dsi/first")
    models = sorted(reference)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    plotted = list(reference.values())
    for cell, label, colour, marker in (("mean/peak_dsi/first", "(a) mean replacement", BLUE, "o"),
                                        ("zero/mean_dsi/first", "(b) mean-response index", MUTED, "s"),
                                        ("zero/peak_dsi/linear_trajectory_exact_readout",
                                         "linearised trajectory, exact readout", ORANGE, "^")):
        values = ratios(cell)
        plotted += list(values.values())
        ax.scatter([reference[m] for m in models], [values[m] for m in models], s=14, marker=marker, color=colour,
                   alpha=0.85, lw=0, label=label)
    top = max(plotted) * 1.05
    ax.plot([0, top], [0, top], color=INK2, lw=0.8)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xlim(0.5, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("error ratio of the reference (per model)")
    ax.set_ylabel("error ratio of the alternative (per model)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=6.8)
    save(fig, "fig3-per-model.pdf")


def figure_strength():
    strengths = ["0.1", "0.25", "0.5", "0.75", "1.0"]
    x = [float(s) for s in strengths]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for cell, label, colour, style in (("zero/peak_dsi/first", "Reference: first order", INK, "-"),
                                       ("mean/peak_dsi/first", "(a) mean replacement", BLUE, "-"),
                                       ("zero/mean_dsi/first", "(b) mean-response index", MUTED, "-"),
                                       ("zero/peak_dsi/second", "(c) second order", BLUE, ":"),
                                       ("zero/peak_dsi/linear_trajectory_exact_readout", "linearised trajectory, exact readout",
                                        ORANGE, "-"),
                                       ("zero/peak_dsi/linear_trajectory_exact_ratio", "floored linear responses, exact ratio",
                                        AQUA, "-")):
        y = [DEV["strengths"][s]["cells"][cell]["ratio"]["mean"] for s in strengths]
        ax.plot(x, y, style, marker="o", ms=3, color=colour, lw=1.2, label=label)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.3", "0.5", "1", "2", "5", "10"])
    ax.set_xticks(x)
    ax.set_xticklabels(["0.1", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("silencing or replacement strength")
    ax.set_ylabel("error ratio (log scale)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    save(fig, "fig4-strength.pdf")


if __name__ == "__main__":
    figure_ratios()
    figure_scatter()
    figure_models()
    figure_strength()
    print("wrote 4 figures")
