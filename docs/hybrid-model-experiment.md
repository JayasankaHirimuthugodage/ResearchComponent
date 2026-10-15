# Development-only hybrid model experiment

This experiment evaluates whether calibrated soft voting between the existing
Logistic Regression and XGBoost candidates improves development validation.
It does not replace, refit, or modify the frozen `slc-development-models-v1`
release.

## Boundary

- Only the frozen 500-participant development partition is available.
- The existing 125-participant test partition has already been examined and is
  not valid as a new confirmatory test for a hybrid.
- The inputs remain `baseline_checkpoint_id` plus the 15 baseline questionnaire
  responses. The three domain targets remain separate, and overall concern is
  never trained as a fourth model.
- Reports contain aggregate evidence only. They contain no participant IDs or
  row-level predictions.

## Nested procedure

The outer evaluation reuses the shared five-fold, five-repeat participant-safe
development folds. Within each outer-training partition, an inner stratified
CV produces out-of-fold probabilities for both candidates. Separate regularized
sigmoid calibrators are fitted from those predictions. The inner calibrated
predictions select a weight from the predefined Logistic Regression grid
`0.25`, `0.50`, `0.75` and select a balanced-accuracy threshold. The fitted base
models, calibrators, selected weight, and threshold are then applied once to the
outer validation fold.

The report compares both calibrated members, equal-weight voting, and selected-
weight voting on identical outer observations. Average Precision is primary;
AUROC, balanced accuracy, sensitivity, specificity, precision, NPV, Brier score,
calibration bins, threshold stability, complementarity measures, and paired
bootstrap uncertainty are supporting evidence.

## Interpretation

A hybrid is not accepted merely because its pooled point estimate is higher.
Improvement must be repeat-stable and must not cause an unacceptable loss in
sensitivity, specificity, or calibration. A selected hybrid remains an internal
development candidate until evaluated on a new untouched cohort or external
validation set. Its threshold is experimental and requires separate governance
review before operational use.

Run the isolated experiment with:

```powershell
python -B -m training.experiments.run_hybrid_development_experiment
```

The command refuses to overwrite an existing report unless `--force` is supplied.
It writes `reports/generated/development-hybrid-model-comparison-report.json`.
