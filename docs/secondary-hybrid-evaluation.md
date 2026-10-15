# Secondary exploratory hybrid evaluation

This study compares the original frozen individual models with separately frozen
50/50 calibrated Logistic Regression–XGBoost hybrids on the same 125-person
cohort. That cohort was evaluated previously. Consequently, this analysis is not
a fresh independent test, external validation, or confirmatory evidence.

All hybrid training, sigmoid calibration, equal weights and domain thresholds
are finalized using only the 500-person development partition. The secondary
evaluator can load and score frozen artifacts but contains no fitting,
calibration, weight-selection or threshold-selection path. Its aggregate report
is write-once and contains no participant identifiers or row-level predictions.

The historical locked-test report and both earlier notebooks remain unchanged.
Application inference and deployment continue using the original individual
models unless a later change receives explicit approval.
