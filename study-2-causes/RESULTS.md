# Results

Short account of what AI-09 found in the 46 confirmation models. Every number comes from
`study/results/analysis/confirmation/summary.json` (written by `study/code/analyse_arms.py`, 17 October 2026). Unless
stated, numbers are means over the 46 models at full strength (strength 1), with 95% bootstrap intervals over models.
The error ratio is a model's mean absolute error divided by that of predicting zero change on the same cases: below 1
beats predicting zero change.

## The question

AI-06 found that first-order estimates of cell-type silencing in the FlyVis models find the influential cell types
but misestimate their effects. AI-09 changed one thing at a time to find out why: (a) the baseline (each cell's
average output instead of zero), (b) the readout (direction selectivity from the mean instead of the peak response),
and (c) curvature (the second-order term of the Taylor expansion). A mechanism analysis applied, on AI-06's own target,
either the exact readout to the linearised trajectory or the linearised readout to the exact trajectory.

The recomputed AI-06 reference equals AI-06's saved values (920 values, largest difference 6e-17): error ratio 1.51
[1.34, 1.71], mean absolute error 0.155 against 0.101 for predicting zero change, 21.6% wrong signs among large effects,
top-10 precision 0.815 against 0.646 for anatomy.

## The three arms

| | Error ratio | Change from AI-06 | Against predicting zero change | Wrong signs | Top-10 (anatomy) |
|---|---|---|---|---|---|
| AI-06 reference | 1.51 [1.34, 1.71] | | worse | 0.216 | 0.815 (0.646) |
| (a) mean replacement | 1.09 [0.97, 1.26] | −0.42 [−0.59, −0.25] | not distinguishable | 0.275 | 0.590 (0.468) |
| (b) mean-response index | 1.52 [1.34, 1.73] | +0.01 [−0.14, 0.15] | worse | 0.192 | 0.820 (0.620) |
| (c) second order | 10.14 [5.93, 16.34] | +8.63 [4.54, 14.69] | worse | 0.461 | 0.792 (0.646) |

- **(a) Mean replacement is the only change that improves on AI-06**, so by the prespecified rule it dominates. At
  full strength its error is still not distinguishable from predicting zero change (difference 0.008 [−0.0003, 0.0199]);
  at half strength it beats it (ratio 0.76 [0.67, 0.87]). Its estimates are biased upwards (+0.022 [0.014, 0.029])
  and have more wrong signs than AI-06's (+0.059 [0.025, 0.096]). It still finds the influential cell types better
  than anatomy.
- **(b) Replacing the peak by the mean response changes nothing detectable.** Fewer wrong signs (−0.024
  [−0.041, −0.008]); the screen is kept.
- **(c) The second-order term makes the estimates far worse,** with a large upward bias (+1.07 [0.53, 1.92]). In
  the development models the term came almost entirely from the curvature of the index's ratio
  (`study/DEVELOPMENT-NOTES.md`): where silencing changes the responses by a large fraction, the Taylor series of the
  ratio diverges. The term changes the ranking little (top-10 −0.023
  [−0.040, −0.007] against first order).

## Why: the readout's ratio

| Estimator (silencing, peak index) | Error ratio | Share of AI-06's error removed |
|---|---|---|
| AI-06 first order | 1.51 | |
| Exact trajectory, linearised readout | 1.78 [1.50, 2.09] | −24% |
| Linearised trajectory, exact readout | **0.55 [0.51, 0.60]** | 58% |
| Linearised trajectory, exact responses, linearised ratio | 7.39 [3.81, 12.02] | −418% |
| Linearised trajectory, floored linear responses, exact ratio | **0.76 [0.73, 0.80]** | 43% |

- **The readout's nonlinearity, not the network's, causes AI-06's failure** (difference in removed error 0.82
  [0.65, 1.02]). Keeping the network linear but computing the index exactly beats predicting zero change (difference in
  absolute error −0.047 [−0.054, −0.041]) with 9.9% wrong signs. The same holds for the mean-response index and under
  mean replacement.
- **Within the readout, the ratio dominates** (difference 4.61 [2.05, 8.05]; Amendment 1). Applying the exact ratio to
  the linear responses, floored at zero, beats predicting zero change on its own. Making only the responses exact makes
  the estimate far worse, because a linearised ratio applied to large response changes overshoots, as in arm (c). The
  two stages interact strongly.
- **The network's trajectories are not linear either:** the linear approximation misses 42% of the change in the
  readout traces (median per model) and the quadratic one 33%. These errors matter less because the index is computed
  from the responses through the ratio.

## Other prespecified analyses

- **Baseline mismatch.** The silencing derivative used for mean-replacement effects has error ratio 3.30 [2.92, 3.71]
  and loses its screening advantage (top-10 equal to anatomy: +0.001 [−0.022, 0.025]). The mean-replacement derivative
  used for silencing effects has ratio 1.22 [1.09, 1.38], lower than AI-06's own estimate (−0.29 [−0.45, −0.14]).
- **Screen, flag, then simulate.** It never pays here. Flagging the ten largest estimates per model and stimulus and
  simulating them removes 44% of the error of predicting zero change in 181 s per model, 0.0024 per second, against
  0.0069 per second for simulating all 63 cell types (145.5 s).
- **Cost.** The first- and second-order tangents took 344 s per model, 2.37 times exact silencing at one strength;
  exact mean replacement took 1.07 times as long as AI-06's exact silencing.

## The predictions

- **Hashed before the development phase (primary): 22 of 27 hold.** Failed: arm (a)'s wrong-sign rate (0.275, predicted
  0.08–0.20); arm (c)'s error ratio (10.1, predicted 1.2–2.2) and wrong-sign rate (0.461, predicted 0.15–0.35); the
  bias bound (arm (c) +1.07, predicted within ±0.03); and that the mismatched silencing derivative keeps its screen.
- **Development-informed addendum: 28 of 29 hold.** Failed: arm (b) "worse than AI-06" (no detectable change).

## Records and limits

- Protocol approved and predictions hashed on 16 October 2026 before any run; development on models 000, 001, 004 and
  005; Amendment 1 (readout split, prediction addendum, disk guard) and freeze before any confirmation run; Amendment 2
  (AI-06's normalised tolerance for two numerical gates, `study/results/GATE-FAILURE-009.md`) and a second freeze
  during the run, before any confirmation outcome was computed. The largest normalised gate difference over the 46
  models was 1.0e-6 (tolerance 1e-4).
- Arms (b) and (c), the mechanism analysis and the flag rule are new quantities computed from AI-06's saved
  confirmation traces, which AI-06 had already analysed with the peak index; they are prespecified analyses of
  existing simulations, not blind confirmations. Arm (a)'s exact effects and all derivatives are new simulations.
- One released ensemble with a shared connectome and task; two moving-edge stimuli and eight readouts. The
  confirmation run records (5.2 GB) are on the external drive (`study/results/RUNS-LOCATION.md`).
