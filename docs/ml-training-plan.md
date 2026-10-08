# Future ML training and evaluation plan

This is a design document only. No model was trained, tuned, calibrated or serialized during Phase 0.

## Feasibility and constraints

The current dataset has 625 records, one record per participant, and 16 conceptual primary features (15 ordinal questionnaire responses plus one categorical checkpoint). Classical tabular experiments are readily feasible on an Intel i5 CPU with 20 GB RAM. Expected memory use is small, and disciplined cross-validation across a compact candidate set should complete locally without GPUs.

The limiting resource is statistical information, not hardware: domain-positive counts are only 61, 53 and 58. Deep learning and large hyperparameter searches are not justified by the current sample.

## Reproducible inputs and schema gates

Before a training run:

1. Verify the dataset path, size and SHA-256 against [data-audit.md](data-audit.md).
2. Validate the exact 66-column source schema and stored domains.
3. Build `X` from the explicit primary allowlist in [research-ml-specification.md](research-ml-specification.md).
4. Build `Y` from exactly the three primary target columns.
5. Retain `participant_number` only for group allocation and traceability outside estimator inputs.
6. Apply a prohibited-column denylist that includes identifiers, T1 fields, scores, targets and contextual exclusions.
7. Fail on unexpected columns rather than silently accepting them.
8. Confirm that all 15 baseline item fields in this modelling snapshot are scoreable values in `{0,1,2}`; fail rather than impute.
9. Record dataset checksum, feature-schema version, target-definition version and split seed.

Training code and configuration must be separate from production inference code. A training package may depend on evaluation/tuning libraries; the inference adapter should depend only on the frozen schema, preprocessing/model artifact and integrity checks it needs to serve predictions.

## Splitting strategy

### Untouched final test set

Reserve approximately 20% (about 100 participants) as a final test set before model comparison. Split by `participant_number`, even though the current export has one row per participant, so the invariant survives future longitudinal exports.

Because exact eight-way label combinations include groups with only 7–21 records, do not rely blindly on a single exact-combination stratifier. Use a deterministic, participant-group-aware allocation that searches a small number of seeded candidate splits and selects one that best preserves:

- marginal positive rates for all three targets;
- representation of baseline checkpoints;
- the observed multi-label combinations where practically possible;
- adequate positive counts in both development and test partitions.

Freeze and version the chosen participant lists before fitting anything. Do not inspect final-test model performance during selection.

### Development resampling

On the remaining approximately 80%, use repeated group-aware cross-validation to reduce dependence on one small validation split. A practical starting design is five folds repeated five times, subject to verifying sufficient positives per fold for every domain. Use the same participant fold assignments across candidate methods so comparisons are paired.

If a validated iterative multi-label group-stratification implementation is unavailable, create and test a deterministic allocator that balances the three marginal labels and checkpoint counts at participant level. Store its tests and assignments. For three separate estimators, a shared fold allocation avoids giving any domain a selectively easier validation set.

Nested cross-validation is not mandatory when an untouched final test set is preserved. It may be used as a sensitivity analysis if model-selection uncertainty must be quantified, but a large nested search would be wasteful. Repeated development CV plus a single final locked test evaluation is the recommended default.

Random seeds must be explicit for split allocation, estimator randomness, resampling and bootstrap intervals.

## Preprocessing pipeline

All learned preprocessing must be fit inside each training fold.

- Preserve item responses as ordered numeric values `0`, `1`, `2` under the documented mapping.
- Treat an absent submitted item field as a schema validation error.
- Preserve `NOT_SURE` and `NOT_OBSERVED` as explicit unscorable response states. They make the production assessment `INCOMPLETE`; never convert or impute them to zero or another score.
- Invoke the current ML inference contract only for `COMPLETE` assessments with all 15 scoreable responses. Validated safety/referral rules remain active for incomplete assessments.
- Encode `baseline_checkpoint_id` with an unknown-category-safe categorical encoder inside the pipeline.
- Do not pass the supplied baseline checkpoint one-hot fields when using the ID.
- Validate feature names, order, types, allowed categories and missingness at inference.
- Fit any imputer, scaler, encoder, feature selector or resampler on training-fold data only.
- Keep any resampling inside the training portion of each cross-validation fold.
- Preserve a direct mapping from transformed features back to source concepts for explanations.

Logistic regression should scale continuous/ordinal inputs when appropriate; tree models generally do not require scaling. Use pipeline objects so transformations cannot be applied globally by accident.

The present modelling CSV requires no missing-response transformation: all 9,375 baseline questionnaire cells are already in `{0,1,2}`. Future research on partial or missing-response modelling would be a separate, explicitly approved experiment and must not alter the primary complete-response contract.

## Candidate model set

Keep the comparison compact and CPU-friendly.

1. **Interpretable baseline:** regularized logistic regression with class weighting; fit one estimator per domain.
2. **Tree ensemble:** class-weighted random forest or Extra Trees with restrained depth/leaf-size search and fixed seeds.
3. **Gradient boosting:** histogram-based gradient boosting only if its imbalance strategy, calibration and explanation path are justified and dependency cost remains small.

Compare:

- three separately configured binary classifiers; and
- a `MultiOutputClassifier`-style wrapper around a suitable base estimator using the same folds and preprocessing contract.

A standard multi-output wrapper still fits independent binary estimators; its value is orchestration consistency, not automatic modelling of domain correlation. More complex classifier chains or joint methods should be exploratory only because they add ordering, leakage-like training dependencies and explanation complexity.

Avoid deep networks, large automated searches, hundreds of weakly motivated configurations and final-test-driven model selection.

### Hyperparameter search

Use a small, predeclared randomized or grid search around parameters that control regularization, tree depth, minimum leaf size, number of trees and class weighting. Prefer learning curves and stability analysis to search breadth. Stop expanding the search when differences are within uncertainty.

## Class imbalance and thresholds

Observed domain positive rates are 10.45%–12.03%.

- Start with class weighting.
- Consider fold-safe random over/under-sampling only if weighting is inadequate.
- Do not apply SMOTE or any resampler before splitting or outside training folds.
- If synthetic resampling is considered later, verify that ordinal questionnaire/checkpoint semantics are not violated.
- Compare calibration before relying on probabilities internally.

The project owner reports that the operational predicted-probability threshold of `0.50` was reviewed and approved by the consultant pediatrician for the current logistic model version. It is recorded as `domain-probability-threshold-v1` for `logistic-l2-balanced-v1`. This approval is distinct from the follow-up score `<= 50` target definition and does not automatically transfer to a different estimator, calibration method, retraining cycle or model version. The reviewer identity, approval date and authoritative document reference remain unrecorded rather than inferred.

Any alternative threshold must be selected using development data only and treated as unapproved until it receives the required research/clinical governance. Never select or revise a threshold from final-test performance.

### Implemented research threshold policy

The proposal defines the development-only selection boundary and requires thresholds to be frozen before locked-test evaluation; it does not prescribe Balanced Accuracy or Youden's J. The completed research cycle selected a separate calibrated threshold for each domain using averaged nested out-of-fold predictions from the development partition. As an implementation decision, the objective maximized balanced accuracy, equivalently Youden's J; deterministic ties preferred higher sensitivity, then higher specificity, then the lower threshold. The frozen values are communication `0.130882`, cognitive `0.109680`, and social-emotional `0.126137`. They were fixed before the one-time locked-test evaluation and were not revised afterward. These ML thresholds are separate from the questionnaire score cutoff of `<= 50%`.

## Evaluation

Report each domain separately for every candidate and the final locked model:

- confusion matrix;
- sensitivity/recall;
- specificity;
- precision/positive predictive value;
- F1 score;
- balanced accuracy;
- PR-AUC as a primary ranking summary under imbalance;
- ROC-AUC with cautious interpretation;
- Brier score and calibration curve/reliability summary when probabilities are retained;
- positive/negative counts and the selected operational threshold.

Accuracy may be included but cannot be the principal metric.

Use participant-level bootstrap confidence intervals on the untouched test set where feasible. Given the low positive counts, report interval width honestly and avoid overinterpreting small metric differences. For repeated development CV, report the distribution across repeats/folds and, where appropriate, paired differences between candidates.

After the three domain decisions, derive `predicted_any_concern` by OR and report its confusion matrix only as a derived system-level check. Do not use it to conceal weak performance in an individual domain.

### Fairness and subgroup evaluation

Sex is excluded from primary model inputs but may support subgroup evaluation. Prematurity, gestational-age availability, checkpoint and birth-weight categories may support carefully aggregated error analysis if minimum group sizes and privacy constraints allow.

Do not publish unstable metrics for very small subgroups. Predefine suppression/minimum-count rules and report uncertainty. Fairness analysis is descriptive; it does not authorize sex-, prematurity- or birth-weight-based concern rules.

## Calibration and uncertainty

If decision thresholds use predicted probabilities, compare uncalibrated results with fold-safe sigmoid or isotonic calibration. Isotonic calibration can overfit a small sample and should be used only with sufficient fold data. Calibration must be learned from development data, never the final test set.

Model uncertainty is not the same as clinical uncertainty. Parent-facing output should use approved qualitative wording rather than probabilities or confidence percentages.

## Explainability evaluation

- For logistic regression, inspect coefficient stability across resamples and map contributions to source items/checkpoint categories.
- For tree models, use SHAP only with a compatible, pinned implementation and validate that contribution sums and class orientation are correct.
- Use held-out permutation importance as supplementary global evidence.
- Evaluate explanation stability across folds and whether parent-facing mappings remain faithful without revealing technical values.
- Never describe an association as causal.

## Model-selection rule

Before running experiments, document:

- the primary metric or multi-metric decision rule;
- minimum domain sensitivity or safety constraints approved by appropriate reviewers;
- calibration and complexity considerations;
- how ties/overlapping uncertainty will favor the simpler, more interpretable model.

Select using development results only. Freeze preprocessing, estimator configuration, calibration and thresholds, then evaluate the untouched final test set once. This cycle has now been completed for the recorded frozen artifacts. Any subsequent change creates a new model-selection cycle and requires threshold revalidation plus a new untouched test strategy or explicit acknowledgement that the prior test has become development data.

## Required outputs of a future approved training phase

- immutable dataset checksum and participant split manifests;
- schema and target-definition versions;
- locked dependency/environment file;
- machine-readable training configuration and seeds;
- fold-level and aggregate evaluation results;
- threshold and calibration reports;
- subgroup/privacy-reviewed evaluation;
- model card covering intended use, exclusions, data limitations and non-diagnostic scope;
- serialized preprocessing/model artifact owned by the service;
- artifact checksum/signature and compatibility metadata;
- separate inference contract tests using representative non-sensitive fixtures.

No artifact should be promoted merely because it outperforms another on a single split.
