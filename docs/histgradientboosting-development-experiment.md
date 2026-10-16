# HistGradientBoostingClassifier development-only evaluation

This single exploratory experiment evaluates scikit-learn's histogram gradient
boosting classifier on the existing 500-person development partition only. It
uses the established five-repeat, five-fold participant-safe outer plan, three
inner folds for calibration and threshold selection, and never loads the
125-person cohort.

Configuration is fixed before execution: 150 iterations, learning rate 0.05,
15 maximum leaf nodes, minimum leaf size 10, L2 regularization 1.0, balanced
classes, fixed fold seed, and early stopping disabled. Checkpoint ID is one-hot
encoded; the 15 questionnaire responses retain their ordinal values. No tuning
or hyperparameter search is permitted.
