# TabICLv2 sensitivity-constrained threshold study

This isolated, post-hoc, development-only study evaluates Cognitive and Social-emotional TabICLv2 under two fold-local operating-point rules: the existing balanced-accuracy/Youden's-J rule and a fixed illustrative 85% minimum inner-development sensitivity constraint. It uses the frozen 500-participant development partition, the exact 16-feature contract, the shared 5x5 participant-safe folds, `tabicl==2.2.0`, and the checksum-verified local checkpoint.

For every outer fold, each strategy uses the same calibrated outer predictions. Calibration and both thresholds are fitted only from three-fold inner OOF predictions derived from the outer-training participants. The sensitivity constraint maximizes specificity among thresholds meeting 85% inner sensitivity. The target is an exploratory assumption, not a clinical requirement.

The study never loads the 125-participant cohort, changes historical artifacts, persists identifiers or row-level probabilities, refits model weights, or changes deployment. PR-AUC, AUROC, and Brier score are probability metrics and are unchanged by the threshold strategy. Repeated-CV estimates share participants; paired bootstrap intervals are descriptive and do not establish clinical validity.
