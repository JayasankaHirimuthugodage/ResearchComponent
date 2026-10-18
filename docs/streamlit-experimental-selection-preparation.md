# Synthetic-only Streamlit experimental-selection preparation

## Phase 3M local-demo default

The local research demonstration now defaults to the explicitly labelled **Experimental, development-only TabICLv2 KV-cache release — not independently validated or clinically deployed.** This changes only the preselected UI release. It does not load the cache until a complete assessment is submitted, and it does not change the historical release, model artifacts, calibration, thresholds, or evidence.

The separate **Historical frozen release** remains selectable as the rollback option and continues to use Communication XGBoost, Cognitive Logistic Regression, Social-emotional XGBoost, and its existing explanation path. The default is a local-demo presentation choice, not a clinical, production, or independent-validation claim.
The local Streamlit research-review demo defaults to the explicit TabICLv2 KV-cache experimental release. Selecting the option alone does not load a model. A complete submission invokes the separately verified cache-backed experimental runtime; an error fails visibly and never falls back to Logistic Regression. The historical frozen release remains an explicitly selectable rollback path.

For complete questionnaires, the UI displays `unavailable_pending_model_specific_validation`; for incomplete questionnaires, it displays `not_applicable_incomplete_assessment`. It never displays invented TabICLv2 feature contributions. Historical XGBoost and Logistic Regression explanation rendering remains unchanged.

The activated cache-backed path verifies the approved cache before deserialization and does not read the development-context CSV. It retains development-only wording, explicit failure, incomplete blocking, three-domain order, and OR aggregation. A validated TabICLv2 explanation method remains future work.
