# FIGS development-only experiment

This isolated exploratory evaluation uses the frozen 500-participant development cohort, exact 16-feature contract, and participant-safe shared 5 repeats by 5 outer folds for the three binary developmental targets. It does not load the locked 125-participant test cohort.

The fixed candidate is `FIGSClassifier(max_rules=8, max_trees=3, max_depth=3, min_impurity_decrease=0.0, class_weight="balanced", random_state=42, n_jobs=2, verbose=0)`. No tuning, model selection, or configuration changes occur after results are observed.

Within every outer-training partition, FIGS receives checkpoint as an internally encoded categorical feature and questionnaire responses as numeric ordinal inputs. Three inner out-of-fold FIGS fits supply probabilities for logistic calibration and balanced-accuracy threshold selection. The outer validation partition is never used for fitting, calibration, or threshold selection. Aggregate metrics and rule-count summaries only are written; no participant identifiers, predictions, learned rule text, or test data are stored.
