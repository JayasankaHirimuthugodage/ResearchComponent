# Development-only model comparison

This repository contains a portable offline experiment boundary for comparing small, classical tabular classifiers. It is deliberately independent of an HTTP framework, database, message broker and external monorepo layout. A future service can consume a separately frozen inference artifact and schema without importing research evaluation code.

## Current candidates

The shared registry contains five predefined, regularized configurations:

- `logistic_l2_balanced`, version `logistic-l2-balanced-v1`: class-weighted, L2-regularized logistic regression with standardized questionnaire values;
- `svm_rbf_balanced`, version `svm-rbf-balanced-v1`: class-weighted RBF support-vector classification;
- `random_forest_balanced_leaf5`, version `random-forest-balanced-leaf5-v1`: class-weighted random forest with minimum leaf size 5;
- `extra_trees_balanced_leaf5`, version `extra-trees-balanced-leaf5-v1`: 300 extremely randomized trees with balanced class weights, minimum leaf size 5 and unscaled questionnaire values.
- `xgboost_regularized_balanced`, version `xgboost-regularized-balanced-v1`: regularized gradient-boosted trees with training-fold-derived class weights.

For every shared cross-validation fold, each candidate fits three binary classifiers:

1. communication concern;
2. cognitive concern; and
3. social-emotional concern.

This is three domain models per candidate configuration, not three datasets and not a fourth overall-concern model. With five candidates, each fold fits 15 classifiers. The derived overall decision is the logical OR of the three selected domain decisions.

Every domain model receives the same 16 conceptual inputs:

- `baseline_checkpoint_id`, encoded inside the fitted pipeline; and
- the 15 scoreable baseline questionnaire answers.

All pipelines encode checkpoint categories inside each training fold. Logistic Regression uses standardized values and signed coefficient evidence. The other candidates use held-out permutation importance; tree pipelines do not scale questionnaire values.

No identifier, profile context, supplied one-hot checkpoint column, follow-up field, score or target can enter the feature matrix. The data loader projects each selected row onto the allowlist before it reaches scikit-learn.

## Locked-test boundary

Model comparison uses only the frozen 500-participant `development` partition. The loader validates the dataset checksum, private manifest checksum and split fingerprint, then returns development rows only. The experiment CLI has no final-test option and records `final_test_evaluated: false`.

The 125-participant final-test partition remained locked while candidates, preprocessing, sigmoid calibration and operating thresholds were selected using the 500-participant development partition. After these configurations were frozen, the final partition was evaluated once. The evaluator records that models were not refitted and thresholds were not adjusted on final-test data, and it refuses to overwrite the existing result.

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

## Questionnaire cutoff and ML operating thresholds

The observed-outcome definition and model decision cutoff are separate:

- follow-up domain score `<= 50` creates the stored observed target (`0` or `1`);
- the final calibrated ML thresholds convert predicted probabilities into domain decisions.

The final thresholds were selected separately from averaged nested out-of-fold development predictions by maximizing balanced accuracy, equivalently Youden's J. Ties prefer higher sensitivity, then higher specificity, then the lower threshold. The frozen values are:

| Domain | Selected configuration | Threshold |
|---|---|---:|
| Communication | `xgboost-regularized-balanced-v1` | 0.130882 |
| Cognitive | `logistic-l2-balanced-v1` | 0.109680 |
| Social-emotional | `xgboost-regularized-balanced-v1` | 0.126137 |

These thresholds were frozen before the one-time locked-test evaluation and were not revised afterward. The earlier numeric `0.50` cutoff remains only an intermediate comparison value; it is not the final operating policy. A questionnaire score of 50% is not a model probability of 0.50.

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

For a human-readable view, open `notebooks/development-model-comparison.ipynb`. It reads the aggregate development, calibration, decision and locked-test reports. It presents five-candidate rankings, frozen thresholds, final metrics, confidence intervals and confusion matrices without exposing participant rows or rerunning the locked evaluation.

Install the optional presentation environment with `python -m pip install -r requirements-notebook.txt`, then launch `python -m jupyter lab`. Notebook checkpoint directories are excluded from Git.

The synthetic unit fixtures validate code contracts only. They are not added to model training and do not increase the effective research sample size. A GitHub Actions workflow is not required at this stage; the same local command can be added to repository CI later without changing the modelling modules.

## Candidate boundary

Any later justified estimator must be represented as another `CandidateSpec` and passed to the same development evaluation function. It must use the same frozen feature projection and shared fold objects. This makes comparisons paired and prevents a candidate from receiving an easier split.

All five configurations were predefined and regularized rather than selected through a hyperparameter search. They are not claimed to be optimal. Any future change to data, features, configuration, calibration or thresholds begins a new model-development cycle and cannot reuse the existing locked result as a fresh unbiased test. A model fitted inside a cross-validation fold is not a deployable artifact.
