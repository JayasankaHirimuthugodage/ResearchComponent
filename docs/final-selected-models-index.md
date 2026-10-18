# Final selected research models index

The local [final-selected-models](../models/final-selected-models/README.md) directory is a documentation-only view of the proposed research-prototype architecture. It preserves canonical artifact locations and does not make a new executable release.

| Domain | Proposed research-prototype model | Canonical reference | Current integration status |
|---|---|---|---|
| Communication | XGBoost | [historical Communication manifest](../models/slc-development-models-v1/communication/manifest.json) | Historical/default release remains active. |
| Cognitive | TabICLv2 | [experimental Cognitive release manifest](../models/experimental-cognitive-tabiclv2-v1/release-manifest.json) and [integrity index](../models/experimental-cognitive-tabiclv2-v1/integrity-index.json) | Finalized experimental development-only candidate; not integrated, deployed, independently validated, or clinically validated. |
| Social-emotional | XGBoost | [historical Social-emotional manifest](../models/slc-development-models-v1/social_emotional/manifest.json) | Historical/default release remains active. |

The currently active historical [release](../models/slc-development-models-v1/release-manifest.json) remains XGBoost / Logistic Regression / XGBoost. Its Cognitive Logistic Regression [artifact](../models/slc-development-models-v1/cognitive/model.joblib) and [manifest](../models/slc-development-models-v1/cognitive/manifest.json) remain the benchmark and rollback option.

The TabICLv2 finalization report is [development-cognitive-tabiclv2-finalization-report.json](../reports/generated/development-cognitive-tabiclv2-finalization-report.json). It is development-only research evidence and must not be described as independent validation, clinical validation, or deployment authorization.

No model artifact, release manifest, integrity index, checkpoint, calibration parameter, threshold, participant context, report, or notebook has been moved, copied, or changed by this index.
