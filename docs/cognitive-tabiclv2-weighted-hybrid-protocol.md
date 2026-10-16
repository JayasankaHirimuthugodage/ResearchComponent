# Cognitive TabICLv2 + Logistic Regression weighted-hybrid protocol

This is one development-only, post hoc exploratory experiment. It uses only the
500-person development partition and must not load the 125-person final-test
partition.

## Predeclared candidates

For calibrated probabilities `p_tab` and `p_lr`, evaluate exactly:

`p_w = w * p_tab + (1 - w) * p_lr`, for `w` in `{0.00, 0.25, 0.50, 0.75, 1.00}`.

The endpoints are individual benchmarks. `w=0.50` is the fixed equal-weight
reference. No other algorithm, stacking method, decision fusion, weight, or
post-result expansion is allowed. Inner-development PR-AUC selects the weight.
Ties are resolved by: 0.50 first, then the weight nearest 0.50, then the lower
weight. This tie rule is fixed before execution.

## Leakage control

The historical five-repeat, five-fold participant-safe outer folds are recreated
from the frozen fold seed and fingerprint. In each outer fold, both components are
fit only on the 400 outer-training participants. Three-fold inner OOF raw
probabilities fit separate component sigmoid calibrators. The same inner calibrated
probabilities select all thresholds by balanced accuracy/Youden's J and select the
weighted candidate by PR-AUC. Only then are the fitted components and calibrators
applied to the untouched 100-person outer-validation fold.

## Reporting and limitations

The final report contains aggregates only. Participant identifiers, row-level
predictions, and labelled contexts remain in memory and are not serialized.
Participant-averaged probabilities across the five repeat-level OOF predictions
support paired stratified bootstrap comparisons. The evaluation is exploratory,
does not establish superiority or clinical diagnostic accuracy, and cannot use the
previously examined test cohort for selection or validation.
