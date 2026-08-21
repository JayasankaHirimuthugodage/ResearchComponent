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
|   |-- logistic.py
|   |-- registry.py
|   `-- demo_fitting.py
`-- experiments/
    |-- cross_validation.py
    |-- evaluation.py
    `-- run_development_experiment.py
```

## Responsibility flow

1. `contracts.py` freezes features, targets, versions, checksums and the approved probability threshold.
2. `data/dataset_validation.py` checks the private processed snapshot without exposing participant rows.
3. `data/splitting.py` creates the participant-level development/final-test boundary.
4. `data/development_data.py` loads only the frozen development partition for modelling.
5. `models/common.py` owns the shared 16-input projection, checkpoint encoding, questionnaire preprocessing and fresh pipeline construction.
6. `models/logistic.py` declares only the regularized-logistic candidate and its parameters.
7. `models/registry.py` explicitly lists candidates included in the default comparison. A future candidate is added as its own module and registered only after its experiment contract is approved.
8. `experiments/cross_validation.py` creates the shared participant folds used by every candidate and domain.
9. `experiments/evaluation.py` fits and evaluates each candidate on those same folds using aggregate metrics.
10. `experiments/run_development_experiment.py` is the development-only command-line entry point.
11. `models/demo_fitting.py` fits the selected provisional logistic configuration in memory for the local Streamlit review adapter; it cannot evaluate the locked final-test partition or persist an artifact.

The three targets still produce three separately fitted binary classifiers. The overall concern remains a deterministic OR of their decisions, not a fourth model.

Run the current development experiment from the repository root:

```powershell
python -B -m training.experiments.run_development_experiment
```
