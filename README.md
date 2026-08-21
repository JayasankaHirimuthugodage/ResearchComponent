# SLC Development Assessment Service

This repository is the research workspace for an explainable, safety-constrained early-childhood **Socio-Emotional, Language/Communication and Cognitive/Problem-Solving (SLC)** developmental assessment and three-month outcome-prediction component. The intended technical service name is `slc-development-assessment-service`.

The component is for developmental monitoring and research for children approximately 0–5 years old. It is non-diagnostic: it must not diagnose autism, screen specifically for autism, diagnose developmental delay or another medical condition, replace pediatric assessment, or make unsupported clinical conclusions.

## Current status

Phase 0 data validation and splitting are complete. The first approved Phase 1 development experiment is now implemented:

- the locally supplied modelling CSV is stored under `data/processed/` and excluded from Git by a narrow ignore rule;
- the dataset has been audited without transforming participant records;
- the research objective, target and leakage boundaries are specified;
- a development-only model-comparison framework uses the frozen 405-participant partition, shared repeated folds and fold-local preprocessing;
- the first candidate fits three class-weighted, L2-regularized logistic classifiers, one per domain;
- aggregate metrics, calibration bins and coefficient-stability evidence can be generated without row-level results;
- the 102-participant final-test partition remains locked and unavailable to the experiment CLI;
- a future ports-and-adapters service architecture is documented;
- no production database, application source, API, frontend, recommendation engine or LLM integration has been created.

There is **no selected, serialized or production model artifact** in this repository. Cross-validation fits are temporary research objects only.

## Dataset

Local path:

`data/processed/survey-ml-cross-checkpoint-classification (2).csv`

The CSV is privacy-sensitive and must not be committed, transmitted or published. The original raw survey export and a raw-to-processed feature-generation pipeline are not present. See [data/processed/README.md](data/processed/README.md) and [docs/data-audit.md](docs/data-audit.md).

## Documentation

- [Dataset audit](docs/data-audit.md)
- [Research and ML specification](docs/research-ml-specification.md)
- [Future ML training plan](docs/ml-training-plan.md)
- [Development-only model comparison](docs/development-experiment.md)
- [Future service architecture](docs/service-architecture-plan.md)
- [Decision log](docs/decision-log.md)

## Local development experiment

The tests use privacy-safe synthetic fixtures for the new modelling contracts. They do not augment the research dataset:

```powershell
python -m pip install -r requirements.lock
python -B -m unittest discover -v
```

The first candidate can be evaluated on the development partition only:

```powershell
python -B -m training.run_development_experiment
```

The generated aggregate report is ignored by Git. See [development-experiment.md](docs/development-experiment.md) for the feature boundary, three-model interpretation, metrics and future candidate-extension point.

For human review, `notebooks/logistic-baseline-analysis.ipynb` presents the aggregate report as tables and plots. It does not duplicate the training implementation or display participant-level records. Install the optional notebook environment and launch it locally with:

```powershell
python -m pip install -r requirements-notebook.txt
python -m jupyter lab
```

## Future phases

1. Record authoritative versions of the reviewed scoring, safety and age-correction rules without inferring missing clinical wording.
2. Compare a compact, predeclared candidate set on the same development folds.
3. Review development evidence, freeze the feature schema/model/calibration, and confirm threshold approval still applies to the selected model version.
4. Implement production inference, deterministic safety rules and service contracts.
5. Add the approved activity catalogue and deterministic recommendation selection.
6. Add constrained LLM personalization only after privacy and safety review.
7. Integrate the bounded service into the external monorepo and its approved platform standards.

No later phase should begin merely because it appears in this list; each requires explicit approval and the relevant validation evidence.
