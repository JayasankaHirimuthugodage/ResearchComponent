# LightGBM development-only experiment

This isolated exploratory comparison evaluates one fixed LightGBM candidate on the frozen 500-participant development cohort for Communication, Cognitive, and Social-emotional outcomes. It does not load the locked 125-participant test cohort.

The candidate uses the pre-authorized configuration: binary objective, 100 estimators, learning rate 0.05, maximum depth 3, seven leaves, minimum child samples 20, L2 regularization 1.0, two threads, seed 42, and silent verbosity. No early stopping, tuning, or class weighting is added.

Evaluation reuses the shared participant-safe 5 repeats by 5 outer folds and exact 16-feature contract. Checkpoint one-hot encoding and questionnaire passthrough are fitted within each fold. Platt calibration and a balanced-accuracy threshold are learned from inner out-of-fold predictions within the outer training partition only. The report stores aggregate metrics only; it stores no participant identifiers or row-level predictions.
