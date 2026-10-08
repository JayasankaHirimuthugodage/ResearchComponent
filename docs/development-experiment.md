# Development-only model comparison

This repository contains a portable offline experiment boundary for comparing small, classical tabular classifiers. It is deliberately independent of an HTTP framework, database, message broker and external monorepo layout. A future service can consume a separately frozen inference artifact and schema without importing research evaluation code.

## Current candidates

The shared registry contains two predeclared configurations:

- `logistic_l2_balanced`, version `logistic-l2-balanced-v1`: class-weighted, L2-regularized logistic regression with standardized questionnaire values;
- `extra_trees_balanced_leaf5`, version `extra-trees-balanced-leaf5-v1`: 300 extremely randomized trees with balanced class weights, minimum leaf size 5 and unscaled questionnaire values.

For every shared cross-validation fold, each candidate fits three binary classifiers:

1. communication concern;
2. cognitive concern; and
3. social-emotional concern.

This is three domain models per candidate configuration, not three datasets and not a fourth overall-concern model. With two candidates, each fold fits six classifiers. The derived overall decision is the logical OR of each candidate's three domain decisions.

Every domain model receives the same 16 conceptual inputs:

- `baseline_checkpoint_id`, encoded inside the fitted pipeline; and
- the 15 scoreable baseline questionnaire answers.

Both pipelines encode checkpoint categories inside each training fold. The logistic pipeline uses `CP2M` as the explicit reference checkpoint and emits coefficients for the other checkpoint categories. Extra Trees uses the same encoded input boundary but does not scale questionnaire values.

No identifier, profile context, supplied one-hot checkpoint column, follow-up field, score or target can enter the feature matrix. The data loader projects each selected row onto the allowlist before it reaches scikit-learn.

## Locked-test boundary

Model comparison uses only the frozen 500-participant `development` partition. The loader validates the dataset checksum, private manifest checksum and split fingerprint, then returns development rows only. The experiment CLI has no final-test option and records `final_test_evaluated: false`.

The 125-participant final-test partition remains locked until the candidate, preprocessing configuration and calibration approach have been selected and frozen using development evidence only. The approved `0.50` threshold is versioned only for the logistic configuration and does not authorize final-test access by itself. Extra Trees uses the same numeric cutoff for development comparison, but its candidate metadata marks that cutoff `experimental_not_approved_for_model`.

## Shared evaluation

The default comparison uses five folds repeated five times. It stratifies on the observed three-label combination, then deterministically searches candidate allocations to improve checkpoint balance. This gives every domain and candidate the same participant allocation. Each participant is in validation exactly once per repeat, and every fold must contain both target classes for every domain.

The report contains, separately for each candidate and domain:

- confusion counts, sensitivity, specificity, precision/PPV, NPV, F1, balanced accuracy and accuracy;
- average precision (the reported PR-AUC summary), ROC-AUC and Brier score;
- aggregate calibration bins;
- per-fold aggregate counts and metrics; and
- candidate-appropriate explanation evidence.

The report contains no participant keys or row-level predictions.

## Explainability

Logistic regression uses standardized coefficient sign, magnitude and fold-stability evidence. Coefficients are model associations, not causal or clinical effects, and checkpoint coefficients are relative to the `CP2M` reference.

Extra Trees uses permutation importance calculated only on each held-out validation fold and scored by average precision. It reports the 16 original conceptual inputs, including checkpoint as one input. Explanation values are aggregated across folds and permutation repeats. Negative importance can occur when permutation happens to improve a validation score; importance is not causation or clinical importance.

Permutation importance is evaluation-only. It is not used to tune the candidate or select features.

## Two different `50` thresholds

The observed-outcome definition and model decision cutoff are separate:

- follow-up domain score `<= 50` creates the stored observed target (`0` or `1`);
- predicted probability `>= 0.50` creates the predicted class for each candidate in this development comparison.

The project owner supplied the following authoritative approval wording:

> The operational predicted-probability threshold of 0.50 was reviewed and approved by the consultant pediatrician for this model version.

The repository records this as `domain-probability-threshold-v1`, scoped to `logistic-l2-balanced-v1`. The consultant pediatrician's identity, approval date and authoritative document reference have not been supplied and are not fabricated here. Extra Trees does not inherit this approval: its `0.50` cutoff is experimental until that exact model version is reviewed and recorded.

## Local commands

Run the complete local test suite:

```powershell
python -m pip install -r requirements.lock
python -B -m unittest discover -v
```

Run the default development-only candidate comparison:

```powershell
python -B -m training.experiments.run_development_experiment
```

The generated report is written to `reports/generated/development-model-comparison-report.json`, which is excluded from Git. The command refuses to overwrite an existing report unless `--force` is supplied intentionally.

For a human-readable view, open `notebooks/development-model-comparison.ipynb`. It reads only the aggregate comparison report by default and presents paired performance, calibration, fold variation and candidate-appropriate explanation plots without exposing participant rows.

Install the optional presentation environment with `python -m pip install -r requirements-notebook.txt`, then launch `python -m jupyter lab`. Notebook checkpoint directories are excluded from Git.

The synthetic unit fixtures validate code contracts only. They are not added to model training and do not increase the effective research sample size. A GitHub Actions workflow is not required at this stage; the same local command can be added to repository CI later without changing the modelling modules.

## Candidate boundary

Any later justified estimator must be represented as another `CandidateSpec` and passed to the same development evaluation function. It must use the same frozen feature projection and shared fold objects. This makes comparisons paired and prevents a candidate from receiving an easier split.

The current Extra Trees settings are a single predeclared, regularized comparison configuration, not the result of a hyperparameter search. Candidate expansion, tuning, calibration, threshold validation for changed model configurations, final-test evaluation and model serialization are separate approval gates. No model produced inside a cross-validation fold is a production artifact.
