# Development-only additional-model comparison

This isolated experiment compares CatBoost 1.2.8 and InterpretML EBM 0.7.8
against the five original fixed candidate configurations. It uses the same
500-person development partition and shared five-fold, five-repeat outer folds.
The previously evaluated 125-person cohort is unavailable to the runner.

CatBoost uses native categorical handling for the checkpoint and ordered numeric
questionnaire responses. EBM uses a nominal checkpoint, explicit ordinal
response levels and no interaction terms. Each new family has two predeclared,
small configurations. Configuration selection, sigmoid calibration and threshold
selection occur inside each outer-training partition.

The report is aggregate-only. No model is frozen or deployed, and the original
models, reports, notebooks and inference application remain unchanged.
