# Why first-order estimates misjudge cell-type silencing in the FlyVis models: code release

This folder holds the code, frozen inputs, study records and saved outputs behind the manuscript *Why first-order
estimates misjudge cell-type silencing in the FlyVis models: the nonlinearity is in the readout*. Every number,
generated table and figure in the manuscript can be recomputed from the saved outputs on a CPU in under a minute,
without a model or network.

## What the paper asks

A reference study found that first-order (gradient) estimates of cell-type silencing in the 50 released FlyVis models
find the influential cell types well but misestimate their effects. This study asks which of three causes accounts
for most of that failure, changing one thing at a time on the same models, stimuli, readouts and cell types: (a) the
baseline (mean replacement instead of silencing to zero), (b) the readout (a direction-selectivity index built from
the mean rectified response instead of its peak), and (c) curvature (the second-order term in the silencing strength).
A mechanism analysis splits the reference study's error into the network's and the readout's share, and an optional
rule (screen, flag, then simulate) reports error recovered per second of computation.

## The study

`study/` holds the protocol (`PROTOCOL.md`, with the approved version of 16 October 2026 in
`PROTOCOL-approved-2026-10-16.md`), the predictions hashed before any run (`PREDICTIONS.md`, `PREDICTIONS-RECORD.json`),
the development-informed addendum of Amendment 1 (`PREDICTIONS-ADDENDUM.md` and its record), the configuration, the
development notes, the code, and two freeze records: `freeze.json` (after Amendment 1, before the confirmation run)
and `freeze-2.json` (after Amendment 2, which brought two numerical gates into line with the reference study's
tolerance, before any confirmation outcome was computed; `study/results/GATE-FAILURE-009.md` explains why). Four
development models and 46 confirmation models were run. The freezes are local hash records written by the study code,
not public preregistrations.

## What is in this folder

| Path | Contents |
|---|---|
| `study/` | The study (above); after `make unpack`, also its `results/` |
| `reference/flyvis-first-order/` | The nine files of the reference study that the code imports or the freeze names (see its README) |
| `analysis/` | Post hoc case-level summaries (`posthoc_paper.py`, which reads the run records, and `posthoc_denominator.py`) and their results |
| `paper/make_numbers.py`, `paper/make_figures.py` | Every macro and generated table of the manuscript and its four figures, from the saved outputs; `paper/generated/` and `paper/figures/` hold what the manuscript was built with (the figure PDFs carry no dates or producer) |
| `pilot/` | The development pilot on one model: numerical checks and timings |
| `RESULTS.md` | The findings with their numbers |
| `provenance/Makefile-at-freeze` | The study's own Makefile, which the freezes list (this release's Makefile replaces it) |
| `data/saved-outputs/` | Saved outputs as a compressed archive (see its README) |
| `tools/` | Unpacking, freeze checking and the replay of the saved-output scripts |
| `requirements.txt`, `ENVIRONMENT.md` | The exact environment, and how to obtain the FlyVis models |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the original project, with the original's SHA-256, and every frozen file not shipped |

## Install

```sh
make setup        # .venv with CPython 3.12.14 and the 102 pinned packages (see ENVIRONMENT.md)
```

This downloads the packages in `requirements.txt`, including PyTorch's CUDA 12.4 build and its CUDA libraries (several
GB). The quick check needs only NumPy and Matplotlib 3.11.2; `ENVIRONMENT.md` gives a small route. Pass
`PYTHON=/path/to/python` to any `make` target to use another environment.

## Quick check (CPU, under a minute, no model, no network)

```sh
make quick-check
```

It unpacks the saved outputs (checking every SHA-256), checks both freeze records, for the study's files and for the
reference study's inputs, against the files they name, recomputes every macro and generated table
(`paper/make_numbers.py --check`), and reruns the post hoc denominator analysis and the figure script in a temporary
copy, comparing their outputs with the shipped files byte for byte. Expected end of the output:

```text
total: 13 changed for the release (documented), 6 superseded by a recorded amendment, 41 unchanged
9 generated files match
...
2 scripts rerun from the saved outputs: every output identical to the shipped file
```

"Superseded by a recorded amendment" are the six files that Amendment 2 changed after the first freeze; the second
freeze records their new digests, which the check uses instead.

```sh
make test         # the unit tests of the tangents, readouts and scoring (PyTorch and flyvis; make setup)
make numbers      # rewrite paper/generated/
make figures      # redraw paper/figures/
```

## What needs the run records

Each model's run records (the readout trajectories of every arm, with a JSON sidecar holding each array's checksum)
take about 160 MB per model, 5.2 GB for the confirmation models. The release keeps the analyses built from them and the
JSON sidecars of the four development models, but not the arrays. The study's analysis (`study/code/analyse_arms.py`)
and the post hoc case summaries (`analysis/posthoc_paper.py`) read them; their saved outputs are in
`study/results/analysis/` and `analysis/results/`.

Rerunning the model stages also needs the FlyVis models and the reference study's stimuli and saved response records,
against which the study's per-model gates compare its own baseline and zero-replacement runs; those come from the
reference study's pipeline. The runs used one 6 GB NVIDIA GPU under a GPU lock file, now named by
`GPU_LOCK` (create it first, for example `touch gpu.lock`); the protocol estimated about 16 minutes per confirmation
model, and the final confirmation run took 9.5 hours (`study/results/confirmation-continue-complete.json`). The stages
are listed in `provenance/Makefile-at-freeze` and `study/PROTOCOL.md` Section 10.

## Tables and figures

| Manuscript | Source |
|---|---|
| All numbers in the text | macros in `paper/generated/numbers.tex`, written by `paper/make_numbers.py` |
| Table 1 (the three arms) | `paper/generated/table-arms.tex` |
| Table 2 (mechanism) | `paper/generated/table-mechanism.tex` |
| Tables 3-8 (appendix: factorial, mechanism by readout, development, flag rule, predictions, addendum) | `paper/generated/table-factorial.tex`, `table-mechanism-all.tex`, `table-development.tex`, `table-flag.tex`, `table-predictions.tex`, `table-addendum.tex` |
| Figure 1 | `paper/figures/fig1-error-ratios.pdf` |
| Figure 2 | `paper/figures/fig3-per-model.pdf` |
| Figure 3 | `paper/figures/fig2-predicted-vs-exact.pdf` |
| Figure 4 | `paper/figures/fig4-strength.pdf` |

All four figures are drawn by `paper/make_figures.py` (the file names follow the order in which they were planned).
The post hoc analyses are labelled post hoc in the manuscript.

## What is not included

- The run records' arrays (above), the FlyVis models, and the rest of the reference study (released separately).
- Logs, the development pilot's GPU log, and the original project's other provenance records, manuscript source and
  checks. Frozen files not shipped are listed with their SHA-256 in `RELEASE-CHANGES.md`.

## Changes from the original project

Files are copied byte for byte, except those listed in `RELEASE-CHANGES.md` with the original's SHA-256: records had
personal paths, the external drive, other project folders' names and the names of software tools in working records replaced by placeholders (the assistants used are named in the top-level README.md, under "Use of AI assistants"), and local
timestamps given in UTC; `study/config.json` points at `reference/flyvis-first-order`; the GPU lock script takes its
lock file from `GPU_LOCK`; and this release's Makefile replaces the study's.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs. The FlyVis models and package are distributed under their own licences.
