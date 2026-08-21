# Local development prediction demo

## Purpose

`demo/development_prediction_demo.py` is a local research-review adapter. It lets a reviewer see the real checkpoint-specific question wording, enter answers manually, and compare how the provisional Logistic Regression and Extra Trees configurations produce future-domain probabilities and decisions from the same answers.

It is not a parent-facing application, a diagnostic tool, a deployed microservice, or final-test evidence.

## End-to-end boundary

The demonstration uses only the approved baseline inputs:

1. one already-determined `baseline_checkpoint_id`; and
2. the 15 answers mapped to `communication_q1`–`communication_q5`, `cognitive_q1`–`cognitive_q5`, and `social_emotional_q1`–`social_emotional_q5`.

It does not accept identifiers, profile fields, follow-up checkpoint values, follow-up answers, follow-up scores or targets. It does not calculate age or apply checkpoint-routing, prematurity, gestational-age, birth-weight, safety/referral or recommendation rules.

All 15 fields are required. `NOT_SURE` and `NOT_OBSERVED` remain valid but unscorable responses. Either response produces `INCOMPLETE` and no ML prediction; neither is silently converted to `0`.

## Targets and decisions

The observed training labels and predicted decisions are different operations:

- In the supplied modelling export, an observed follow-up domain score `<= 50` created that domain's positive training target (`1`).
- At inference, a predicted positive-class probability `>= 0.50` creates that domain's positive concern decision.

The operational predicted-probability threshold of 0.50 was reviewed and approved by the consultant pediatrician for `logistic-l2-balanced-v1`. The repository versions it as `domain-probability-threshold-v1` for that configuration. The same numeric cutoff is shown for `extra-trees-balanced-leaf5-v1` only as an experimental comparison threshold; it has no assigned threshold version or recorded pediatrician approval for that model configuration.

The three outputs are:

- `communication_probability` and its decision;
- `cognitive_probability` and its decision; and
- `social_emotional_probability` and its decision.

A positive decision means the model probability is at or above that candidate's displayed 0.50 cutoff. The cutoff is approved for the current Logistic Regression version and experimental for the current Extra Trees version. A positive decision does not mean the child has a diagnosis. Each candidate's displayed overall concern is a deterministic OR of its three domain decisions; there is no fourth model.

## Model status

When a complete answer set is submitted, the demo fits six classifiers in memory using all 405 records in the frozen development partition: three L2-regularized, class-weighted logistic classifiers and three regularized Extra Trees classifiers. Each candidate has one classifier per domain. The 102-record final-test partition remains inaccessible to this path. The fits are cached for the local Streamlit process and are not serialized.

This development-only fit is provisional. It allows an input-to-output demonstration, but neither candidate is a selected final artifact and a synthetic comparison is not evaluation evidence. Formal candidate comparison still uses the shared development folds and aggregate metrics. Final-test evaluation and model serialization remain later approval gates.

## Questionnaire catalogue

The committed catalogue is `questionnaire_catalog/developmental-questionnaire-v3/questionnaire.json`. It contains 12 checkpoints × 3 domains × 5 questions, the response labels/codes, and the exact feature mapping. Runtime validation fails closed if the 180 unique IDs, versions, checkpoints, response mapping or 15-feature mapping drift.

Repository inclusion was authorized by the project owner on 2026-08-22. The supplied reference did not state an underlying third-party licence or an authoritative external-document identifier, so those values are recorded as `not supplied` rather than inferred.

## Privacy and synthetic examples

Manual answers are held only in the local Streamlit session. The adapter contains no file write, database, HTTP, analytics or model-artifact operation. Streamlit secret/credential files are ignored by Git.

The two committed quick-fill files are explicitly synthetic demonstration fixtures. They are not participant records, are never added to training, and exist only to make a repeatable demo quick. One is complete; the other contains `NOT_SURE` to demonstrate the no-prediction boundary.

## Local command

From the repository root in the activated `.venv`:

```powershell
python -m pip install -r requirements-demo.txt
python -m streamlit run demo/development_prediction_demo.py
```

The first complete comparison fits six models and may take a few seconds. Later predictions in the same Streamlit process reuse the in-memory fits.

## Future microservice integration

The reusable code is outside Streamlit:

- `inference/contracts.py` defines transport-independent request/result objects;
- `inference/questionnaire.py` validates and encodes the versioned catalogue;
- `inference/predictor.py` owns the three-domain threshold use case through a small model-runtime port; and
- `training/models/demo_fitting.py` is the development-only scikit-learn adapter.

A future HTTP adapter can construct the same `QuestionnaireSubmission` and call the same application service. Transport, authentication, storage and production model loading can therefore be added without embedding Streamlit, HTTP or database types in the inference boundary.
