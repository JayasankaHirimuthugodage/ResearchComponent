# Local development prediction demo

## Purpose

`demo/development_prediction_demo.py` is a local research-review adapter. It lets a reviewer see the checkpoint-specific question wording, enter answers manually, and run the three frozen domain models from the same answers.

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
- At inference, each calibrated probability is compared with its own frozen domain threshold.

The frozen thresholds are `0.130882` for communication, `0.109680` for cognitive and `0.126137` for social-emotional. They were selected using development-only out-of-fold predictions and were fixed before locked-test evaluation.

The three outputs are:

- `communication_probability` and its decision;
- `cognitive_probability` and its decision; and
- `social_emotional_probability` and its decision.

A positive decision means the domain probability is at or above that domain's frozen threshold. A positive decision does not mean the child has a diagnosis. The displayed overall concern is a deterministic OR of the three domain decisions; there is no fourth model.

## Model status

When a complete answer set is submitted, the demo loads three checksum-verified artifacts: XGBoost for communication, Logistic Regression for cognitive and XGBoost for social-emotional. It does not load the development or locked-test datasets and does not fit or update a model. The verified service is cached for the local Streamlit process.

The artifacts are the final frozen research selections. Using them in this demonstration does not add evaluation evidence, and predictions made from synthetic examples must not be reported as model-performance results.

## Questionnaire catalogue

The committed catalogue is `questionnaire_catalog/developmental-questionnaire-v3/questionnaire.json`. It contains 12 checkpoints × 3 domains × 5 questions, the response labels/codes, and the exact feature mapping. Runtime validation fails closed if the 180 unique IDs, versions, checkpoints, response mapping or 15-feature mapping drift.

Repository inclusion was authorized by the project owner on 2026-08-22. The supplied reference did not state an underlying third-party licence or an authoritative external-document identifier, so those values are recorded as `not supplied` rather than inferred.

## Explanation display

For each domain, the main result gives a plain-language summary of the strongest submitted answer supporting the final decision. It shows the checkpoint-specific question, response label and direction, then explains whether the combined result met the threshold. One shared notice explains that all answers and the checkpoint contribute and that the explanation is non-diagnostic. It does not expose probabilities or numerical contribution values. Opposing and supporting contributions remain in the research/reviewer trace, using coefficient contributions for Logistic Regression and model-native TreeSHAP contributions for XGBoost.

## Privacy and synthetic examples

Manual answers are held only in the local Streamlit session. The adapter reads the local frozen artifacts but contains no answer-file write, database, HTTP or analytics operation. Streamlit secret/credential files are ignored by Git.

The quick-fill controls generate repeatable synthetic responses for whichever of the 12 checkpoints is currently selected. The complete control fills all 15 checkpoint-specific question IDs. The incomplete control sets one of those answers to `NOT_SURE` to demonstrate the no-prediction boundary. These values are not participant records and are never added to training. The original CP24M fixture files remain as internal compatibility-test inputs.

## Local command

From the repository root in the activated `.venv`:

```powershell
python -m pip install -r requirements-demo.txt
python -m streamlit run demo/development_prediction_demo.py
```

The first complete prediction verifies and loads the three artifacts and may take a few seconds. Later predictions in the same Streamlit process reuse the cached frozen service.

## Future microservice integration

The reusable code is outside Streamlit:

- `inference/contracts.py` defines transport-independent request/result objects;
- `inference/questionnaire.py` validates and encodes the versioned catalogue;
- `inference/predictor.py` retains the earlier shared-threshold development comparison use case;
- `inference/frozen_models.py` verifies the release and applies each frozen model's own threshold; and
- `inference/explanations.py` produces model-specific local explanations and caregiver-safe summaries.

A future HTTP adapter can construct the same `QuestionnaireSubmission` and call the frozen prediction service. Transport, authentication and storage can therefore be added without embedding Streamlit, HTTP or database types in the inference boundary.
