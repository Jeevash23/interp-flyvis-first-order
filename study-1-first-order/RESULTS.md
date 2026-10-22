# Results

Short account of what AI-06 found. The manuscript (`paper/manuscript.pdf`) has the details. Every number below is one
of the macros that `paper/make_numbers.py` generates from the saved outputs, and `make check-paper` checks that each
number in this file appears among them. Unless stated otherwise, results are means over the 46 confirmation models
with 95% bootstrap intervals over models.

## The question

A first-order estimate predicts the effect of silencing a cell type as the silencing strength times the derivative of
the output at the intact network. It can serve two purposes: screening (finding the cell types whose silencing
matters) and estimation (the size and sign of each effect). We evaluate both against exact silencing in the 50
released FlyVis models: each of the 65 cell types silenced at full and half strength, the outcome being the change in
direction selectivity (DSI) of the central T4 and T5 cells for ON and OFF edges moving in 12 directions at 110.2 and
27.55 degrees per second. Readout conditions (one readout at one speed in one model) enter the main analysis when the
readout responds and its DSI is at least 0.2: 314 of 736 (39 have no response; 383 respond with DSI below 0.2). At full
silencing this leaves 19,825 scored silencing cases.

## Screening

- Top-10 precision (fraction of the ten highest-ranked cell types with |ΔDSI| ≥ 0.1) at full silencing: 0.815 for the
  first-order estimate, 0.646 for three-step anatomical path weights and 0.277 for a random ranking. Paired difference
  0.169 [0.140, 0.199]; the first-order estimate is higher in 43 models, equal in 1 and lower in 2.
- The anatomical rankings improve up to four steps (0.672), which still trails the first-order estimate by 0.143
  [0.116, 0.172]. Trained-weight paths reach 0.592, input currents 0.539, direct synapse counts 0.522.
- Half strength: 0.722 against 0.507 (difference 0.214 [0.188, 0.243]; higher in all 46 models).
- Inclusive screen (every responsive readout condition, all 65 types, abolished responses counted as large effects):
  0.449 against 0.367 (difference 0.082 [0.066, 0.099], post hoc; higher in 42, equal in 2, lower in 2).

## Effect sizes

- Full silencing: mean absolute error 0.155 for the first-order estimate against 0.101 for predicting zero change
  (difference 0.054 [0.034, 0.074]); the first-order error is larger in 39 of 46 models.
- The errors run both ways: mean signed error −0.002 [−0.019, 0.011] (summarised post hoc).
- Among the 5,501 large changes (27.7% of scored cases): error 0.466 against 0.316, and 21.6% of estimates have the
  wrong sign.
- Clipping each estimate so that the predicted DSI stays in [0, 1] gives 0.082, below zero change by −0.019
  [−0.024, −0.015] and in 40 of 46 models.
- Half strength: 0.070 against 0.065 (difference 0.005 [−0.003, 0.013]); larger in 18 models; 15.9% wrong signs.
- Development models across five strengths: the first-order estimate is the better predictor only up to strength 0.25
  (0.011 against 0.017 at 0.1, 0.036 against 0.040 at 0.25, 0.174 against 0.105 at full silencing).

## Ranking among large effects

Spearman correlation among large effects at full silencing: 0.322 for the first-order estimate, 0.374 for anatomy
(difference −0.053 [−0.110, 0.003]; higher in 21 of 46 models). Over all scored cell types the first-order ranking is
far better (0.894 against 0.566): its advantage lies in separating large effects from small ones.

## Where the errors arise

- Errors are largest when silencing changes which directions evoke any response (0.378 against 0.088), because a zero
  response contributes zero derivative.
- They are larger for cell types that synapse directly onto the readout (0.331) than for indirect inputs (0.122).
- Full silencing abolished the readout's response in 723 of 45,305 cases in responsive readout conditions (1.6%).

## Cost

Per model, both stimuli: exact silencing of the 63 cell types with a path to the readouts took 145.5 s at one
strength and 291.1 s at two; the first-order estimates for the eight readouts took 158.2 s (time ratios 0.92 and
1.84). Exact silencing costs one simulation per intervention; the first-order estimate costs one backward pass per
readout and serves every strength.

## What this means

In these models, first-order estimates are a useful screen for the cell types that matter, but the size and sign of
the effects of silencing them towards zero output have to be simulated. A follow-up study of the same models and cases
(AI-09, `../<sibling-project>/`) located the cause: the error of first-order estimates comes from
linearising the readout (the direction-selectivity index), not the network. The main limits: one released ensemble
with a shared connectome and task; two moving-edge stimuli and eight readouts; a peak-based tuning index; and a main
screen scored on cases selected after silencing (the inclusive screen removes that selection and keeps the same order
of methods).
