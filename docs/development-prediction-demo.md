# Local development prediction demo

## Purpose

`demo/development_prediction_demo.py` is a local research-review adapter. It lets a reviewer see checkpoint-specific question wording, enter answers manually, and run either the default development-only experimental three-domain release or the historical frozen rollback release.

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

Each selected release uses its own frozen domain thresholds. The default experimental Cognitive TabICLv2 release uses calibrated probability threshold `0.1503614235173848` with inclusive comparison. The historical rollback release retains its original frozen Logistic Regression threshold. Thresholds are not recalculated by the demo.

The three outputs are:

- `communication_probability` and its decision;
- `cognitive_probability` and its decision; and
- `social_emotional_probability` and its decision.

A positive decision means the domain probability is at or above that domain's frozen threshold. A positive decision does not mean the child has a diagnosis. The displayed overall concern is a deterministic OR of the three domain decisions; there is no fourth model.

The caregiver-facing overall summary names the domain or domains identified for further attention. When no domain is positive, it states that no developmental area was flagged by the current model results.

## Model status

When a complete answer set is submitted, the default release lazily loads checksum-verified Communication/Social-emotional XGBoost artifacts and the verified Cognitive TabICLv2 KV-cache. The historical rollback release instead uses Cognitive Logistic Regression. Neither path loads the development or locked-test datasets, fits, or updates a model. The selected service is cached for the local Streamlit process.

The artifacts are the final frozen research selections. Using them in this demonstration does not add evaluation evidence, and predictions made from synthetic examples must not be reported as model-performance results.

## Questionnaire catalogue

The committed catalogue is `questionnaire_catalog/developmental-questionnaire-v3/questionnaire.json`. It contains 12 checkpoints × 3 domains × 5 questions, the response labels/codes, and the exact feature mapping. Runtime validation fails closed if the 180 unique IDs, versions, checkpoints, response mapping or 15-feature mapping drift.

Repository inclusion was authorized by the project owner on 2026-08-22. The supplied reference did not state an underlying third-party licence or an authoritative external-document identifier, so those values are recorded as `not supplied` rather than inferred.

## Explanation display

The historical release retains its existing model-specific explanation display. For the experimental default, the Communication and Social-emotional XGBoost artifacts use the existing TreeSHAP contribution method in calibrated-logit space. Those contributions are model calculations, not causal effects, clinical importance, or direct additions to calibrated probabilities.

The experimental Cognitive TabICLv2 path never reuses Logistic Regression coefficients or XGBoost contributions. After an explicit reviewer action, it runs a bounded local model-behaviour analysis: for each of the 15 valid submitted answers it evaluates the other two valid response codes while holding the checkpoint and other 14 answers fixed. This creates exactly 30 alternate predictions using the already initialized verified KV-cache-backed Cognitive model and its frozen calibration. It reports calibrated probability differences relative to the actual baseline prediction. It is not SHAP, feature attribution, population importance, causal evidence, clinical advice, or independent validation. The operation is not performed automatically at page rendering or prediction submission.

The caregiver view does not display question-by-question alternatives. It presents only a plain-language Cognitive screening summary, the non-diagnostic limitation, and guidance to discuss concerns with an appropriate healthcare professional. The optional 30-alternative analysis and its complete table are reserved for the separate researcher expander. This presentation performs no additional Cognitive model calls and does not persist questionnaire answers or explanation results.

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
