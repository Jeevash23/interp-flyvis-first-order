# Reproducing the AI-06 study

The commands, environment and checks are in [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md). This page describes the
frozen pipeline that produced `results/`; its 7 October version is preserved in
`../provenance/superseded-manuscript-2026-10-07/copies/study/REPRODUCE.md`.

## The pipeline (ran on 7 October 2026; not to be repeated)

`code/finish_study.py` ran the study as one resumable, single-worker pipeline holding the portfolio GPU lock:

1. Resume the four development models, verify every stored metric and run one fresh CPU replay per model.
2. Analyse the development results with the frozen rules.
3. Run the other 46 models on both common stimuli at strengths 0.5 and 1.
4. Verify every saved result's checksum and metrics, then replay one seeded silencing case per confirmation model
   with the FlyVis package's own update step on the CPU.
5. Produce the confirmation, all-model and supplementary analyses, then the 7 October manuscript.

Progress records are `results/pipeline.json` and `results/progress.json`; logs are in `results/logs/`. Completed
records are reused only when their checksums and simulation signature match. `make run`, which started this pipeline,
is retired: running it again would rebuild the superseded manuscript over the current one.

## Saved outputs

`results/runs/<model>/<stimulus>/` holds the intact traces, the first-order derivatives with their timing records, and
every silencing condition with its metadata. `results/analysis/<source>/` holds the tables of the frozen analysis:
per readout condition (`units.csv`), per silencing case (`effects.csv`), coverage and failure counts (`failures.csv`),
per model (`models.csv`), paired comparisons (`paired-models.csv`, `paired-summary.json`), threshold sensitivity,
tuning groups, diagnostics, error differences and timing. `results/analysis/supplement/` holds the pair interactions
and the biological-context tables of the development models. `make verify` reruns this analysis in a temporary mirror
and confirms that it reproduces every file.

## Limits

The main screen uses the exact outcome to decide which cases are scored (abolished and amplified responses are
excluded), so its precision describes the scored cases; the inclusive screen keeps every responsive readout condition.
Bootstrap intervals resample models of one released ensemble. GPU scatter operations can differ slightly between
executions; the independent double-precision checks and the replay tolerance are part of the frozen code. Derivatives
at tied peaks follow the FlyVis subgradient convention, and ties are recorded with the gradients.
