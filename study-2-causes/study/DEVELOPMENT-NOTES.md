# AI-09 development phase: record

16 October 2026. The protocol (`PROTOCOL.md`, SHA-256 b2f5c5b4…) was approved by the author as drafted. The predictions
were hashed at 09:51:49 UTC (`PREDICTIONS-RECORD.json`, SHA-256 of `PREDICTIONS.md` 9f3436ba…), before any AI-09 run
existed. This file records what happened in the development phase on models 000, 001, 004 and 005, and every change
to the code. None of these changes alters an arm, outcome, threshold or decision rule.

## Code changes during the development phase

1. **Thread set-up once per process** (`ai09_common.configure_torch`). The runner stopped after model 000 when it set
   PyTorch's thread counts again for model 001, which PyTorch allows once per process. The counts are now set once.
   Model 000's records were written before this change, so their code signature differs from the other three models';
   the change does not touch any computed quantity. The first run's log is `results/logs/development-attempt1.log`.
2. **The analysis** (`analyse_arms.py`) was completed from the skeleton, and unit tests were added (`test_arms.py`, 7
   tests: the peak readout against AI-06's independent readout; readout derivatives against finite differences; the
   tangent recursion against nested automatic differentiation on random small networks under both interventions;
   replacement by zero against AI-06's silencing; estimators and pooling; eligibility; the flag rule; intervals and
   the dominance rule).

3. **Amendment 1** (PROTOCOL.md Section 13), after the author read the development results and asked for it:
   - `analyse_arms.py`: the readout split (Section 13.1), its decision, and the scoring of the addendum predictions;
   - `readouts.py`: the ratio stage is now a method of its own (`Readout.ratio`), used by the split; the index is
     computed exactly as before (unit test against AI-06's readout, and the development analysis reproduced all 360
     earlier rows exactly);
   - `ai09_common.health` and `run_arms.py`: the 8 GB disk guard applies to the filesystem holding the run records,
     with a 2 GB guard on the project's own filesystem (`config.json`: `minimum_project_disk_gib`);
   - `test_arms.py`: a test of the split (8 tests).
   The runner's code signature therefore differs between the development records and the confirmation records; no
   computed quantity depends on these changes.
4. **The split as first written was undefined.** Applying the exact ratio to linearised responses let some responses
   fall below zero, the index's denominator reached zero or below, and the estimate grew without bound (error ratios
   near 1e13 on the development models). The linear responses are therefore floored at zero (Section 13.1 says why).
   This was decided on development data, before the addendum was written and before the freeze.

## Definitions fixed in the analysis code

These settle details the protocol leaves implicit; each follows AI-06 where AI-06 has a convention.

- **Exact peak-DSI effects** use AI-06's independent readout (`revision_common.independent_metrics`) on the stored
  traces, for both interventions, as AI-06's analysis did. The mean-response DSI and every derivative use
  `readouts.Readout` in float64.
- **The reference** uses AI-06's saved derivatives; arm (c) adds ½·s²·F″(0) from AI-09's tangents to them. All other
  first-order estimates use AI-09's tangents.
- **Missing model-level values.** A model with no large effect in a cell has no wrong-sign rate there; it is left out
  of that mean and of each bootstrap resample's mean (the resamples stay shared), and the number of models with a
  value is reported.
- **Trace-level prediction** (`trace.quadratic_over_linear_error`): for each model, the median over its scored cases
  (silencing, strength 1, all simulated types not amplified more than 100-fold) of the ratio of the quadratic to the
  linear relative trajectory error; the prediction holds if the ensemble mean of these medians exceeds 0.75.
- **Flag rule ties** are broken by cell-type index. Its derivative cost is AI-06's recorded reverse-mode time and its
  simulation costs are AI-06's recorded per-type times; the development models have neither (their AI-06 records were
  reused from AI-06's Stage 1), so rule (d) is evaluated on the confirmation models only.

## Development run

- **When.** Model 000 ran from 09:51 to 10:20 UTC. The run then stopped (change 1 above). It resumed at 10:51 UTC,
  after waiting about 20 minutes for another portfolio session's GPU job (AI-01 Study 6) to finish, and ended at
  12:03 UTC. Models 001, 004 and 005 took 24 minutes each.
- **Gates.** All passed on both stimuli of all four models:
  - forward derivatives through the peak DSI against AI-06's saved derivatives: largest difference 6.7e-5, every
    value within AI-06's tolerance;
  - baselines against AI-06's saved traces: largest difference 1.6e-6;
  - replacement by zero against AI-06's saved silencing of Mi1 and Tm3: largest difference 1.9e-6.
- **Reference.** The recomputed reference equals AI-06's saved development values (200 values, largest difference
  7e-18; `results/analysis/development/reference-check.json`).
- **Timing.** On models 001, 004 and 005, per model and both stimuli: silencing tangents 337 s, mean-replacement
  tangents 338–340 s, and exact mean replacement 147 s per strength, the same as the pilot. Model 000's GPU
  sections ran 11–40% slower; the cause is unknown. At these times the 46 confirmation models need about 12.4 GPU hours.
- **Storage.** 163–166 MB per development model (five strengths), so about 110 MB per confirmation model and 5 GB in
  all. Free disk was 22 GB after the run.
- **Rule (d)** is not evaluated on these models: AI-06's records for them were reused from its Stage 1 and carry no
  timings.

## Development results (four models; means and ranges, no decisions)

These are the files `results/analysis/development/summary.json` and `models.csv`. They are development results:
they set no decision and are not used to score the predictions.

At strength 1:

| | Error ratio to zero | MAE (zero) | Signed error | Wrong signs | Top-10 (anatomy) |
|---|---|---|---|---|---|
| Reference (AI-06) | 1.56 [1.21, 2.36] | 0.174 (0.105) | +0.028 | 0.256 | 0.820 (0.642) |
| (a) mean replacement | 1.09 [0.90, 1.56] | 0.056 (0.049) | +0.026 | 0.298 | 0.586 (0.436) |
| (b) mean-response DSI | 1.84 [1.43, 2.65] | 0.204 (0.115) | +0.025 | 0.171 | 0.888 (0.641) |
| (c) second order | 11.65 [2.10, 32.69] | 1.501 (0.105) | +1.490 | 0.449 | 0.800 (0.642) |
| Exact trajectory, linearised readout | 1.50 [0.99, 2.55] | 0.172 (0.105) | +0.064 | 0.270 | 0.865 (0.642) |
| Linearised trajectory, exact readout | 0.49 [0.28, 0.70] | 0.054 (0.105) | −0.006 | 0.062 | 0.903 (0.642) |

Error ratio by strength (0.1, 0.25, 0.5, 0.75, 1):

| | 0.1 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| Reference | 0.57 | 0.81 | 1.07 | 1.33 | 1.56 |
| (a) mean replacement | 0.31 | 0.50 | 0.73 | 0.93 | 1.09 |
| (b) mean-response DSI | 0.51 | 0.83 | 1.20 | 1.58 | 1.84 |
| (c) second order | 0.90 | 2.03 | 4.21 | 7.68 | 11.65 |
| Linearised trajectory, exact readout | 0.31 | 0.35 | 0.38 | 0.47 | 0.49 |

- **Mechanism.** Applying the exact readout to the linearised trajectory removes 68% of the reference's error at
  strength 1 (range 60–80% over the four models); applying the linearised readout to the exact trajectory removes 5%
  (−10 to 30%). The pattern is the same for the mean-response readout (72% against −4%) and under mean replacement
  (49% against 10%).
- **Trajectories.** At strength 1 the linear trajectory approximation misses 48% of the change in the readout traces
  (median per model), and the quadratic one 41%; the per-case ratio of the two errors has a median of 0.97.
- **Mismatch.** The silencing derivative used for mean-replacement effects has error ratio 3.75; the mean-replacement
  derivative used for silencing effects, 1.10.
- **Readout split (Amendment 1), silencing, strength 1.** With the peak DSI, the exact ratio applied to the floored
  linear responses removes 54% of the first-order error (error ratio 0.68), the exact responses with a linear ratio
  make it worse (recovery −237%, error ratio 6.90), and both stages exact remove 68% (0.49). The difference between the
  first two recoveries is positive in all four models (0.51 to 9.24). With the mean-response DSI, where the response
  stage is the rectification alone: 59%, −1112% and 72%. The linear ratio fails on exact responses for the same reason
  as the second-order term: applied to large changes of the responses, the linearised ratio overshoots.

**The second-order result was checked for errors.** In models 000 (fast stimulus) and 004 (slow stimulus):
- the computed second derivatives agree with numerical second derivatives of the same readout path on the real traces
  (median difference 1e-10 to 1e-6);
- almost all of the term is the readout's own curvature, not the network's (in model 004, slow stimulus: mean 17.4 of
  18.0);
- it is heavy-tailed (median near 0, largest 804). It is large where the linearised trajectory changes the index's
  denominator by a large fraction of its own size (median predicted change at strength 1 among the largest tenth of
  terms: +416% in model 004 on the slow stimulus and +40% in model 000 on the fast one, against 0% among the rest).

The Taylor series of a ratio converges only while the denominator's predicted change stays below its own size, so at
strengths 0.5 and 1 the second-order term overshoots in those cases. This is a result about the estimator, not a
coding error. It agrees in direction with the hashed expectation that arm (c) "may overshoot", but at a much larger
size than the predicted range.

**The hashed predictions on these four models (for information only; scored on confirmation).** Each holds or fails
on the development means as follows:
- arm (a): improves on AI-06 (holds), ratio 0.8–1.3 (holds), does not beat zero (holds), wrong signs 0.08–0.20
  (fails: 0.30), top-10 0.55–0.85 (holds);
- arm (b): does not improve on AI-06 (holds), ratio 1.2–1.7 (fails: 1.84), wrong signs 0.15–0.30 (holds), top-10
  0.6–0.85 (fails: 0.89);
- arm (c): does not improve (holds), ratio 1.2–2.2 (fails: 11.7), wrong signs 0.15–0.35 (fails: 0.45), top-10 within
  0.05 of first order (holds: −0.02);
- signed error within ±0.03: holds for (a) and (b), fails for (c) (+1.49);
- readout dominates the mechanism split (holds); linearised-trajectory exact-readout ratio 0.5–1.2 (fails narrowly:
  0.49); quadratic-to-linear trajectory error above 0.75 (holds: 0.97); mismatch pattern (holds).
