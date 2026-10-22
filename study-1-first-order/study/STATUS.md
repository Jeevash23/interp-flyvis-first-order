# AI-06 study status

Updated 9 October 2026. **Complete.**

## The study

- Protocol, configuration, code and inputs frozen at 04:48 UTC on 7 October 2026 (`freeze.json`, 182 files), before any
  confirmation response was computed. The freeze is a local record, not an external registration.
- All 50 released FlyVis models simulated on both common stimuli: 4 development models (000, 001, 004, 005; strengths
  0.1, 0.25, 0.5, 0.75 and 1, plus the joint CT1 silencing) and 46 confirmation models (strengths 0.5 and 1). The
  pipeline finished at 11:35 UTC on 7 October; one interruption at 11:13 UTC was resumed from checksummed records
  (`../provenance/run-interruptions/20261007T111510Z/`).
- Verification (`results/verification/`): every saved record checked (2,620 development and 12,144 confirmation files)
  and one fresh CPU replay with the FlyVis package's own update step per model (50 replays, largest normalised
  difference 8.3e-7); the indexing of the trained strengths checked for all 604 type pairs in all 50 checkpoints.
- On 9 October `make verify` repeated the hash checks (182 frozen files, 3,228 original evidence files, 33 files of the
  earlier freeze) and the record checks (14,764 files), and reran the frozen analysis in a temporary mirror: it
  reproduces all 56 saved analysis files.

## The manuscript

The manuscript generated on 7 October (`code/build_paper.py`) used template prose written before the confirmation data
existed. Under amendment 1 of 9 October (`AMENDMENTS.md`) it was replaced by a hand-written manuscript built from the
results: `../paper/manuscript.pdf` (19 pages since the literature update of 17 October 2026; 18 before), source `../paper/main.tex` and `../paper/sections/`. Every number, table
and figure comes from the saved outputs; analyses added after the confirmation results were read are labelled post hoc
(`../analysis/posthoc_summaries.py`). The superseded files are in
`../provenance/superseded-manuscript-2026-10-07/`. Every page of the final PDF was inspected
(`../paper/visual-review.json`); `make check-paper`, `make check-delivery-frozen`, `make check-analysis`,
`make verify` and `make references` pass.

## Main result

First-order estimates screen well and estimate effect sizes poorly. At full silencing, top-10 precision is 0.815
against 0.646 for three-step anatomical path weights (difference 0.169 [0.140, 0.199]), while the mean absolute error
of the estimated change is 0.155 against 0.101 for predicting zero change (difference 0.054 [0.034, 0.074]), with 21.6%
of large changes predicted with the wrong sign. See `../RESULTS.md`.

## Scope decisions

Integrated gradients was left out: along each cell type's own silencing path it converges to the exact change, so it
needs several simulations and backward passes per cell type and answers the exact question rather than approximating
it from one backward pass per readout. Second-order corrections and connection-level silencing are outside the first-order question of this study.
Biological experiments are context, not a scoring key (manuscript Appendix F).

## Left to the author

The author line of the manuscript (`\author{}` in `../paper/main.tex`) and the choice of venue. When network access
is available, run `make references` again and recheck two details by hand: the result attributed to Dhiman and
Panwar (2026; a refitted decoder still recovers direction after T4/T5 silencing, as their PubMed abstract read on
9 October states) and the page numbers 17 and 46 cited for the FlyVis peer review file (taken from the text of the
PDF downloaded on 9 October).
