# Training package map

The training package separates dataset controls, model definitions and experiment orchestration so additional candidates can be compared without duplicating the research protocol.

```text
training/
|-- contracts.py
|-- data/
|   |-- dataset_validation.py
|   |-- development_data.py
|   |-- prepare_data.py
|   `-- splitting.py
|-- models/
|   |-- common.py
|   |-- extra_trees.py
|   |-- logistic.py
|   |-- random_forest.py
|   |-- svm.py
|   |-- xgboost.py
|   |-- registry.py
|   `-- demo_fitting.py
`-- experiments/
    |-- calibration.py
    |-- cross_validation.py
    |-- development_decision_report.py
    |-- evaluation.py
    |-- run_development_experiment.py
    `-- run_locked_test_evaluation.py
```

## Responsibility flow

1. `contracts.py` freezes features, targets, versions and dataset checksums.
2. `data/dataset_validation.py` checks the private processed snapshot without exposing participant rows.
3. `data/splitting.py` creates the participant-level development/final-test boundary.
4. `data/development_data.py` loads only the frozen development partition for modelling.
5. `models/common.py` owns the shared 16-input projection, checkpoint encoding, questionnaire preprocessing and fresh pipeline construction.
6. The model modules declare the five predefined Logistic Regression, RBF SVM, Random Forest, Extra Trees and XGBoost configurations.
7. `models/registry.py` explicitly lists the five candidates included in the comparison.
8. `experiments/cross_validation.py` creates the shared participant folds used by every candidate and domain.
9. `experiments/evaluation.py` fits and evaluates candidates on identical folds using aggregate metrics and candidate-appropriate explanations.
10. `experiments/calibration.py` performs nested development-only sigmoid calibration and deterministic domain threshold selection.
11. `experiments/development_decision_report.py` links model rankings, calibration and thresholds into the frozen development decision.
12. `experiments/run_locked_test_evaluation.py` verifies frozen artifacts and performs the guarded one-time aggregate final evaluation without fitting.
13. `models/demo_fitting.py` supports the older local two-candidate research-comparison adapter; it cannot evaluate the locked partition or persist an artifact.

The three targets still produce three separately fitted binary classifiers. The overall concern remains a deterministic OR of their decisions, not a fourth model.

Run the current development experiment from the repository root:

```powershell
python -B -m training.experiments.run_development_experiment
```

The locked-test evaluator has already completed and is intentionally not a routine rerun command. Its saved aggregate result is `reports/generated/final-locked-test-report.json`.
