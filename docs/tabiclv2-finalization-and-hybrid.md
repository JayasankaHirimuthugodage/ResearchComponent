# TabICLv2 development finalization and narrow hybrid assessment

This experiment is development-only. It regenerates leakage-controlled predictions on
the 500-person development partition, fits domain-specific final sigmoid calibrators,
selects thresholds from averaged out-of-fold predictions, and prepares the full
500-person TabICLv2 context in memory. No labelled context or row-level prediction is
persisted in the aggregate report or release manifest.

The hybrid assessment compares TabICLv2 with the historically selected configuration
for each domain. An equal-weight hybrid is eligible for freezing only when development
OOF probability correlation is below 0.95 and each member is uniquely correct for at
least 2% of participants. This rule is fixed before results are inspected. PR-AUC is the
primary metric; numerical differences are accompanied by participant-level paired
bootstrap intervals and are not automatically interpreted as superiority.

The resulting release remains experimental and undeployed. The previously evaluated
125-person cohort is not loaded by these scripts. A separate explicit approval is
required before secondary exploratory test evaluation.
