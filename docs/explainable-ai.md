# Explainable AI boundary

## Purpose

The explanation layer describes how the frozen research models use their inputs. It does not change a probability, threshold or decision, and it must not be interpreted as a diagnosis, cause or measure of clinical importance.

## Global explanations

Global evidence is calculated during development validation:

- Logistic Regression uses standardised coefficient magnitude, direction and sign agreement across the shared development folds.
- XGBoost uses permutation importance calculated on held-out validation folds and scored with Average Precision.

The notebook displays the five leading features for each selected domain model together with their mean value, standard deviation and consistency fraction. For Logistic Regression, consistency is coefficient sign agreement. For XGBoost, it is the fraction of held-out permutation values that are positive. Variation across folds is reported rather than hidden.

Global importance describes the fitted model across the development validation process. It does not establish causation and is not recalculated from locked-test data.

## Local explanations

One complete assessment can receive a local explanation without refitting a model:

- Logistic Regression uses the transformed feature value multiplied by its fitted coefficient.
- XGBoost uses model-native TreeSHAP contributions on the transformed input.
- The fitted sigmoid-calibration coefficient is applied so contributions reproduce the calibrated model logit.
- Checkpoint one-hot contributions are combined into the single conceptual input `baseline_checkpoint_id`.
- Other contributions map back to the 15 questionnaire features and checkpoint-specific question wording.

The implementation verifies that the base value plus all feature contributions reproduces the calibrated output. This is a technical consistency check, not evidence that an input caused an outcome.

## Caregiver and reviewer separation

For each domain, the main Streamlit result identifies the strongest submitted questionnaire answer supporting the final decision. It shows the checkpoint-specific question wording, submitted response label and contribution direction, then explains whether the combined result met the domain threshold. One shared notice after the three domains states that all 15 answers and the checkpoint determine each result and that other answers may influence the calculation differently. This avoids repeated wording and keeps opposing contributions in the reviewer trace. The caregiver wording contains no raw probability, numerical contribution value or causal claim and states that the explanations are not diagnoses.

The separately labelled research/reviewer section may show:

- the model-specific explanation method;
- the leading feature descriptions;
- whether each contribution moves the model output toward or away from a concern flag; and
- the numerical contribution values.

No participant identifier, follow-up value, target or future-derived field enters an explanation.

## Incomplete assessments

An assessment containing `NOT_SURE` or `NOT_OBSERVED` remains `INCOMPLETE`. No model prediction or local model explanation is produced.

## Limitations

- Feature importance and contribution values describe model behaviour, not child development mechanisms.
- Correlated questionnaire items can share or redistribute apparent importance.
- Global and local explanations do not validate clinical safety or generalisability.
- Explanation stability on development folds does not replace independent external validation.
