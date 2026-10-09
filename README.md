# SLC Development Assessment Service

This repository is the research workspace for an explainable, safety-constrained early-childhood **Socio-Emotional, Language/Communication and Cognitive/Problem-Solving (SLC)** developmental assessment and three-month outcome-prediction component. The intended technical service name is `slc-development-assessment-service`.

The component is for developmental monitoring and research for children approximately 0–5 years old. It is non-diagnostic: it must not diagnose autism, screen specifically for autism, diagnose developmental delay or another medical condition, replace pediatric assessment, or make unsupported clinical conclusions.

## Current status

Data validation, development-only model selection, frozen artifact creation and the one-time locked-test evaluation are complete:

- the locally supplied modelling CSV is stored under `data/processed/` and excluded from Git by a narrow ignore rule;
- the dataset has been audited without transforming participant records;
- the research objective, target and leakage boundaries are specified;
- a development-only model-comparison framework uses the frozen 500-participant partition, shared repeated folds and fold-local preprocessing;
- five predefined model families are compared on shared development folds: Logistic Regression, RBF SVM, Random Forest, Extra Trees and XGBoost;
- Average Precision selects a candidate separately for each domain; sigmoid calibration and threshold selection use nested development-only out-of-fold predictions;
- each final threshold maximizes balanced accuracy (equivalently Youden's J), with deterministic tie-breaking by sensitivity, specificity and then the lower threshold;
- the frozen selections are XGBoost for communication, Logistic Regression for cognitive and XGBoost for social-emotional;
- the frozen ML probability thresholds are `0.130882`, `0.109680` and `0.126137`, respectively;
- a versioned questionnaire catalogue contains the supplied wording for 12 checkpoints and maps each checkpoint's 15 questions to the frozen model inputs;
- framework-neutral inference contracts enforce required fields, unscorable-response handling, three domain decisions and the derived OR decision;
- a local Streamlit research-review demo accepts manual answers or synthetic quick-fill examples and runs the three checksum-verified frozen domain artifacts without retraining;
- the 125-participant locked-test partition was evaluated once after the complete configurations were frozen; no model was refitted and no threshold was adjusted on final-test data;
- a future ports-and-adapters service architecture is documented;
- no production database, application source, API, frontend, recommendation engine or LLM integration has been created.

Versioned, serialized research model bundles are stored locally under the ignored `models/slc-development-models-v1/` directory. They include preprocessing, the fitted classifier, sigmoid calibrator and frozen threshold. They are inference artifacts, not evidence of clinical or production readiness.

## Dataset

Local path:

`data/processed/survey-ml-cross-checkpoint-classification.csv`

The CSV is privacy-sensitive and must not be committed, transmitted or published. The original raw survey export and a raw-to-processed feature-generation pipeline are not present. See [data/processed/README.md](data/processed/README.md) and [docs/data-audit.md](docs/data-audit.md).

## Documentation

- [Dataset audit](docs/data-audit.md)
- [Research and ML specification](docs/research-ml-specification.md)
- [Future ML training plan](docs/ml-training-plan.md)
- [Development-only model comparison](docs/development-experiment.md)
- [Explainable AI boundary](docs/explainable-ai.md)
- [Training package map](training/README.md)
- [Local development prediction demo](docs/development-prediction-demo.md)
- [Future service architecture](docs/service-architecture-plan.md)
- [Decision log](docs/decision-log.md)

## Local development experiment

The tests use privacy-safe synthetic fixtures for the new modelling contracts. They do not augment the research dataset:

```powershell
python -m pip install -r requirements.lock
python -B -m unittest discover -v
```

All five registered candidates can be evaluated on the development partition only:

```powershell
python -B -m training.experiments.run_development_experiment
```

The generated aggregate comparison report is ignored by Git. See [development-experiment.md](docs/development-experiment.md) for the feature boundary, three-model-per-candidate interpretation, metrics and explainability methods.

For human review, `notebooks/development-model-comparison.ipynb` presents the five-candidate development comparison, frozen domain decisions and saved one-time locked-test results as aggregate tables and plots. It does not duplicate training logic, rerun the locked evaluation or display participant-level records. Install the optional notebook environment and launch it locally with:

```powershell
python -m pip install -r requirements-notebook.txt
python -m jupyter lab
```

For a local question-and-answer prediction demonstration using the frozen models, install the separate UI dependency and run:

```powershell
python -m pip install -r requirements-demo.txt
python -m streamlit run demo/development_prediction_demo.py
```

Use only the committed synthetic quick-fill examples during a shared demo unless handling real answers has separately been approved. The app writes no answers, does not fit models and does not access the research dataset. See [development-prediction-demo.md](docs/development-prediction-demo.md) for the exact boundary and limitations.

## Remaining engineering phases

1. Record authoritative versions of the reviewed scoring, safety and age-correction rules without inferring missing clinical wording.
2. Complete production inference, deterministic safety rules and service contracts around the frozen research artifacts.
3. Add the approved activity catalogue and deterministic recommendation selection.
4. Add constrained LLM personalization only after privacy and safety review.
5. Integrate the bounded service into the external monorepo and its approved platform standards.

No later phase should begin merely because it appears in this list; each requires explicit approval and the relevant validation evidence.
