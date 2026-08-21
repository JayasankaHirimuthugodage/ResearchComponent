# Development-only model comparison

This repository now contains a portable offline experiment boundary for comparing small, classical tabular classifiers. It is deliberately independent of an HTTP framework, database, message broker and external monorepo layout. A future service can consume a separately frozen inference artifact and schema without importing research evaluation code.

## Current experiment

The first candidate is `logistic_l2_balanced`, configuration version `logistic-l2-balanced-v1`. For every shared cross-validation fold, it fits three regularized binary logistic classifiers:

1. communication concern;
2. cognitive concern; and
3. social-emotional concern.

This is three domain models per candidate configuration, not three datasets and not a fourth overall-concern model. The derived overall decision is the logical OR of the three domain decisions.

Every domain model receives the same 16 conceptual inputs:

- `baseline_checkpoint_id`, encoded inside the fitted pipeline; and
- the 15 scoreable baseline questionnaire answers.

The logistic pipeline uses `CP2M` as the explicit reference checkpoint and emits coefficients for the other checkpoint categories. This avoids redundant dummy columns and gives checkpoint coefficients a defined comparison point.

No identifier, profile context, supplied one-hot checkpoint column, follow-up field, score or target can enter the feature matrix. The data loader projects each selected row onto the allowlist before it reaches scikit-learn.

## Locked-test boundary

Model comparison uses only the frozen 405-participant `development` partition. The loader validates the dataset checksum, private manifest checksum and split fingerprint, then returns development rows only. The experiment CLI has no final-test option and records `final_test_evaluated: false`.

The 102-participant final-test partition remains locked until the candidate, preprocessing configuration and calibration approach have been selected and frozen using development evidence only. The approved `0.50` threshold is versioned for the current logistic configuration and does not authorize final-test access by itself.

## Shared evaluation

The default comparison uses five folds repeated five times. It stratifies on the observed three-label combination, then deterministically searches candidate allocations to improve checkpoint balance. This gives every domain and estimator candidate the same participant allocation. Each participant is in validation exactly once per repeat, and every fold must contain both target classes for every domain.

The report contains, separately for each domain:

- confusion counts, sensitivity, specificity, precision, F1, balanced accuracy and accuracy;
- average precision (the reported PR-AUC summary), ROC-AUC and Brier score;
- aggregate calibration bins;
- per-fold aggregate counts and metrics; and
- logistic coefficient sign and magnitude stability across fitted folds.

The report contains no participant keys or row-level predictions.

## Two different `50` thresholds

The observed-outcome definition and the model decision cutoff are separate:

- follow-up domain score `<= 50` creates the stored observed target (`0` or `1`);
- predicted probability `>= 0.50` creates the predicted class for the current model configuration.

The project owner supplied the following authoritative approval wording:

> The operational predicted-probability threshold of 0.50 was reviewed and approved by the consultant pediatrician for this model version.

The repository records this as `domain-probability-threshold-v1`, scoped to `logistic-l2-balanced-v1`. The consultant pediatrician's identity, approval date and authoritative document reference have not been supplied and are not fabricated here. A different estimator, calibration method, retraining cycle or model version does not automatically inherit this approval.

## Local commands

Run the complete local test suite:

```powershell
python -m pip install -r requirements.lock
python -B -m unittest discover -v
```

Run the default development-only baseline experiment:

```powershell
python -B -m training.run_development_experiment
```

The generated report is written to `reports/generated/development-logistic-report.json`, which is excluded from Git. The command refuses to overwrite an existing report unless `--force` is supplied intentionally.

For a human-readable view, open `notebooks/logistic-baseline-analysis.ipynb`. It reads only the aggregate report by default and presents performance, calibration and coefficient-stability plots without exposing participant rows.

Install the optional presentation environment with `python -m pip install -r requirements-notebook.txt`, then launch `python -m jupyter lab`. Notebook checkpoint directories are excluded from Git.

The synthetic unit fixtures validate code contracts only. They are not added to model training and do not increase the effective research sample size. A GitHub Actions workflow is not required at this stage; the same local command can be added to repository CI later without changing the modelling modules.

## Adding later candidates

A later random forest, Extra Trees or another justified estimator should be represented as another `CandidateSpec` and passed to the same development evaluation function. It must use the same frozen feature projection and shared fold objects. This makes comparisons paired and prevents a candidate from receiving an easier split.

Candidate expansion, hyperparameter search, calibration, threshold revalidation for changed model configurations, final-test evaluation and model serialization are separate approval gates. No model produced inside a cross-validation fold is a production artifact.
