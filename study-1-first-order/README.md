# First-order estimates of cell-type silencing in the FlyVis models: code release

This folder holds the code, frozen inputs, study records and saved outputs behind the manuscript *First-order
estimates identify influential cell types in the FlyVis models but misestimate their silencing effects*. Every number,
generated table and figure in the manuscript can be recomputed from the saved outputs on a CPU in under a minute,
without a model or network.

## What the paper asks

First-order (gradient) estimates predict what removing a network component does from the derivative of an output at
the intact network. The study tests them against exact silencing of each of the 65 cell types in the 50 released FlyVis
models, connectome-constrained recurrent networks of the fly visual system, using the direction selectivity of the
eight T4 and T5 motion detectors for ON and OFF edges at two speeds. The analysis was fixed on four development models
before the other 46 were analysed.

## What is in this folder

| Path | Contents |
|---|---|
| `study/` | The frozen study: `PROTOCOL.md`, `config.json`, `freeze.json` (SHA-256 of 182 files, written before any confirmation response existed), `LITERATURE.md`, `AMENDMENTS.md`, `STATUS.md`, `REPRODUCE.md`, `code/`, and after `make unpack` its `results/` |
| `analysis/` | Post hoc summaries of the saved tables (`posthoc_summaries.py`, `results/`) and the original project's full verifier (`verify_saved.py`, which needs the response arrays, see below) |
| `paper/make_numbers.py`, `paper/make_figures.py` | Every macro and generated table of the manuscript and its four figures, from the saved outputs; `paper/generated/` and `paper/figures/` hold what the manuscript was built with (the figure PDFs carry no dates) |
| `02-Code/`, `03-Data/`, `04-Results/`, `06-Study-History/` | The earlier stages the study built on: their code (the study imports part of it), the stimuli and model selection, the stage 1 results records, and the frozen protocols and freeze manifest of those stages |
| `provenance/original-records.json`, `provenance/revision-environment.json` | The hash inventory of the 3,228 original evidence files (the manuscript counts them) and the environment recorded at the freeze |
| `RESULTS.md` | The findings with their numbers |
| `data/saved-outputs/` | Saved outputs as compressed archives (see its README) |
| `tools/` | Unpacking and freeze checking |
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

It unpacks the saved outputs and compressed inputs (checking every SHA-256), checks the study's freeze record and the
earlier stage's freeze manifest against the files they name, recomputes every macro and generated table
(`paper/make_numbers.py --check`), redraws the four figures and compares them with the shipped PDFs byte for byte
(`paper/make_figures.py --check`), and recomputes the post hoc summaries (`analysis/posthoc_summaries.py --check`).
Expected end of the output:

```text
study/freeze.json: 182 files: 2 changed for the release (documented), 154 not shipped (documented), 26 unchanged
06-Study-History/stage1/freeze-manifest.json: 33 files: 1 changed for the release (documented), 32 unchanged
total: 3 changed for the release (documented), 154 not shipped (documented), 58 unchanged
paper/make_numbers.py --check: every macro and table matches
ok  4 figures match the saved outputs
ok  analysis/results/posthoc-summary.json matches the saved outputs
```

```sh
make test         # the frozen unit tests of the scoring code (PyTorch; make setup)
make numbers      # rewrite paper/generated/
make figures      # redraw paper/figures/
```

## What needs the response arrays and the models

The study saved, for each model and stimulus, the intact network's responses, the derivatives and one response array
per silenced cell type and strength (14,764 arrays, 2.3 GB). The release keeps every JSON record (with each array's
checksum and the metrics computed from it), the intact-network and derivative arrays, and the analysis tables built
from all of them, but not the 14,560 silencing arrays. The frozen analysis (`study/code/analyse_revision.py`) and the
original project's verifier (`analysis/verify_saved.py`, which rechecks every array against its checksum and reruns the
frozen analysis in a temporary copy) need them; their saved results are the tables and verification records in
`study/results/analysis/` and `study/results/verification/`.

The arrays can be computed again with the frozen pipeline and the FlyVis models:

```sh
make flyvis-models    # the released checkpoints of the 50 models, from the FlyVis authors (needs network access)
```

The pipeline ran on 7 October 2026 on one 6 GB NVIDIA GPU, one model at a time: exact silencing of
the 63 cell types with a path to the readouts took 145.5 s per model at one strength and 291.1 s at both, and the
derivatives for the eight readouts 158.2 s. `study/REPRODUCE.md` gives the stages; `study/code/finish_study.py` ran
them, holding a GPU lock file named by `GPU_LOCK` (create it first, for example `touch gpu.lock`). It refuses to write
over existing results, so run it in a copy with `study/results/` removed.

## Tables and figures

| Manuscript | Source |
|---|---|
| All numbers in the text | macros in `paper/generated/numbers.tex`, written by `paper/make_numbers.py` |
| Table 1 (screening) | `paper/generated/table-screen.tex` |
| Table 2 (paired comparison) | `paper/generated/table-paired.tex` |
| Table 3 (effect sizes) | `paper/generated/table-magnitude.tex` |
| Table 4 (diagnostics) | `paper/generated/table-diagnostics.tex` |
| Tables 5-11 (appendix: coverage, sensitivity, clusters, development, pairs, biology, models) | `paper/generated/table-coverage.tex`, `table-sensitivity.tex`, `table-clusters.tex`, `table-development.tex`, `table-pairs.tex`, `table-biology.tex`, `table-models.tex` |
| Figures 1-4 | `paper/make_figures.py` → `paper/figures/fig1-screening.pdf`, `fig2-predicted-vs-exact.pdf`, `fig3-model-errors.pdf`, `fig4-strength.pdf` |

The values come from the frozen analysis tables and from the post hoc summaries, which the manuscript labels post hoc.

## What is not included

- The FlyVis models and the files the flyvis package ships or builds (`make flyvis-models`; see `ENVIRONMENT.md`),
  and the silencing response arrays (above).
- The earlier stages' response arrays (329 MB; the two intact-network responses the unit tests read are included),
  the papers and the FlyVis source copy the earlier stages collected, their notes, briefs, logs and rendered figures,
  the original project's other provenance records, and the manuscript and its source. Frozen files not shipped are
  listed with their SHA-256 in `RELEASE-CHANGES.md`.

## Changes from the original project

Files are copied byte for byte, except those listed in `RELEASE-CHANGES.md` with the original's SHA-256: records had
personal paths, other project folders' names and the names of software tools in working records replaced by placeholders (the assistants used are named in the top-level README.md, under "Use of AI assistants"), local timestamps were given in
UTC, and two scripts take the GPU lock file from `GPU_LOCK`.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs. The FlyVis models and package are distributed under their own licences.
