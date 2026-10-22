"""Rerun the saved-output analysis and the figure script, and compare their outputs with the shipped files.

    python tools/replay_analyses.py          # after tools/unpack.py; NumPy and Matplotlib 3.11.2; exit status 1 on a difference

Reruns the post hoc denominator analysis (from the saved case table) and draws the four figures (from the study's
saved summaries), each in a temporary copy, and compares the outputs with the shipped files byte for byte. The
figure PDFs carry no dates or producer, so identical inputs give identical bytes with the same Matplotlib version.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
FIGURES = ["fig1-error-ratios.pdf", "fig2-predicted-vs-exact.pdf", "fig3-per-model.pdf", "fig4-strength.pdf"]
STEPS = [  # (command run from the copy's root, outputs)
    (["analysis/posthoc_denominator.py"], ["analysis/results/posthoc-denominator.json"]),
    (["paper/make_figures.py"], [f"paper/figures/{f}" for f in FIGURES]),
]
COPY = ["analysis", "paper", "study", "pilot", "reference"]


def main():
    if not (RELEASE / "study/results/analysis/confirmation/summary.json").exists():
        sys.exit("saved outputs missing; run tools/unpack.py (make unpack) first")
    failed = 0
    with tempfile.TemporaryDirectory(prefix="replay-") as tmp:
        work = Path(tmp)
        for rel in COPY:
            shutil.copytree(RELEASE / rel, work / rel, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        for cmd, outs in STEPS:
            run = subprocess.run([sys.executable, "-B", *cmd], cwd=work, capture_output=True, text=True)
            if run.returncode != 0:
                failed += 1
                print(f"FAIL  {cmd[0]} exited with {run.returncode}")
                print("      " + (run.stdout + run.stderr).strip().replace("\n", "\n      ")[-2000:])
                continue
            for o in outs:
                same = (work / o).read_bytes() == (RELEASE / o).read_bytes()
                failed += not same
                print(f"{'identical' if same else 'DIFFERENT'}  {o}  ({cmd[0]})")
    print(f"{len(STEPS)} scripts rerun from the saved outputs: "
          + ("every output identical to the shipped file" if not failed else f"{failed} problem(s)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
