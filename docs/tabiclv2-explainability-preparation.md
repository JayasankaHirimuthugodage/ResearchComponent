# Experimental TabICLv2 explainability preparation

`inference.tabiclv2_explanations` provides only a structured unavailable status for the development-only Cognitive TabICLv2 candidate. It contains no SHAP values, feature coefficients, contributions, model calls, logging, calibration refit, or threshold selection. Existing Logistic Regression coefficient and XGBoost TreeSHAP explanations remain separate and unchanged.

## Future bounded model-behaviour method

Subject to separate approval and real-runtime authorization, an explanation request may perform bounded local response analysis against the fixed verified 500-person in-context reference. Starting from a completed synthetic or permitted live questionnaire assessment, retain the checkpoint and all other answers, replace one of the 15 questionnaire responses with each other valid ordinal alternative, and obtain the actual calibrated TabICLv2 probability using the frozen sigmoid calibration and inclusive threshold.

The proposed maximum budget is 30 counterfactual predictions (two alternatives for each of 15 responses), in addition to the already-generated original prediction. The analysis must not alter the original prediction, cached context, checkpoint, calibration, or threshold. It describes only the model response to specified input changes; it is not causal feature importance, clinical significance, a diagnosis, or an explanation of an individual participant in the protected context.

## Future Streamlit changes

The current Streamlit demo imports `load_frozen_prediction_service()` directly and therefore remains historical-default only. A future separately approved integration should add an explicit experimental selection control, development-only/not-deployed status label, error display with no fallback, and a Cognitive explanation-unavailable state alongside unchanged Communication and Social-emotional XGBoost explanations. It must preserve complete/incomplete handling, the three-domain OR result, and the historical default path.
