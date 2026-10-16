# Development-only TabICLv2 experiment

This isolated experiment evaluates `tabicl==2.2.0` as one exploratory eighth
individual candidate for the three developmental-domain targets. It does not
implement a hybrid, modify frozen models, or evaluate the 125-person test
partition.

The approved checkpoint is `tabicl-classifier-v2-20260212.ckpt` from the
official `jingang/TabICL` repository at revision
`4dcd344ece2c00be9e831fdd35bed57b5ad83e19`. Its verified SHA-256 is
`BDC7DBD5E4FF21F8F0456FCF90C6B7CDF72DBEA960F2D05B19BEC19F9B3D4ED0`.
The software and model card declare BSD 3-Clause licensing. Inference uses the
local checkpoint with automatic downloads disabled.

The experiment preserves the exact 16-feature contract. The checkpoint ID is
categorical and the 15 scoreable questionnaire answers remain ordinal integer
values. Incomplete assessments are excluded by the existing fail-closed data
loader; TabICLv2 missing-value handling does not change questionnaire
eligibility.

Evaluation uses the historical participant-safe five-fold, five-repeat outer
fold plan. Within each outer-training partition, three-fold out-of-fold raw
probabilities fit a regularized sigmoid calibrator and select the experimental
balanced-accuracy threshold. Each outer-validation participant is absent from
the labelled context used for their prediction. The approximately 266-person
inner contexts are below TabICLv2's reported pretraining range and remain an
explicit limitation.

The progress file stores completed-repeat aggregate metrics only. It contains
no identifiers or row-level predictions. An interrupted partial repeat is
rerun. The final report is written atomically and cannot overwrite an existing
completed report.

The historical seven-model aggregate report supplies comparison values and the
fold fingerprint. Because it contains no aligned participant-level
predictions, the experiment reports numerical differences but does not invent
paired confidence intervals. Five-repeat intervals are descriptive because
the repeated estimates share participants.

The notebook reads aggregate JSON reports only and performs no model fitting.
TabICLv2 remains an exploratory development candidate and is not automatically
frozen, promoted, deployed, or integrated with application inference.

