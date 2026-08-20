# SLC Development Assessment Service

This repository is the research workspace for an explainable, safety-constrained early-childhood **Socio-Emotional, Language/Communication and Cognitive/Problem-Solving (SLC)** developmental assessment and three-month outcome-prediction component. The intended technical service name is `slc-development-assessment-service`.

The component is for developmental monitoring and research for children approximately 0–5 years old. It is non-diagnostic: it must not diagnose autism, screen specifically for autism, diagnose developmental delay or another medical condition, replace pediatric assessment, or make unsupported clinical conclusions.

## Current status

Phase 0 is complete at the documentation and data-organisation level:

- the locally supplied modelling CSV is stored under `data/processed/` and excluded from Git by a narrow ignore rule;
- the dataset has been audited without transforming participant records;
- the research objective, target and leakage boundaries are specified;
- future ML training and evaluation are planned but have not been executed;
- a future ports-and-adapters service architecture is documented;
- no production database, application source, API, frontend, recommendation engine or LLM integration has been created.

There is **no trained model or model artifact** in this repository.

## Dataset

Local path:

`data/processed/survey-ml-cross-checkpoint-classification (2).csv`

The CSV is privacy-sensitive and must not be committed, transmitted or published. The original raw survey export and a raw-to-processed feature-generation pipeline are not present. See [data/processed/README.md](data/processed/README.md) and [docs/data-audit.md](docs/data-audit.md).

## Documentation

- [Dataset audit](docs/data-audit.md)
- [Research and ML specification](docs/research-ml-specification.md)
- [Future ML training plan](docs/ml-training-plan.md)
- [Future service architecture](docs/service-architecture-plan.md)
- [Decision log](docs/decision-log.md)

## Future phases

1. Obtain supervisor and pediatric review of unresolved research, scoring, safety and age-correction decisions.
2. Implement schema validation and a reproducible offline training/evaluation pipeline.
3. Review results and freeze the feature schema, model, calibration and domain thresholds.
4. Implement production inference, deterministic safety rules and service contracts.
5. Add the approved activity catalogue and deterministic recommendation selection.
6. Add constrained LLM personalization only after privacy and safety review.
7. Integrate the bounded service into the external monorepo and its approved platform standards.

No later phase should begin merely because it appears in this list; each requires explicit approval and the relevant validation evidence.

