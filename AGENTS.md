# Durable repository instructions

This is a privacy-sensitive, non-diagnostic early-childhood SLC developmental monitoring research component. Preserve user data and existing changes.

- Only information available at T0 may enter ML features.
- Primary features are `baseline_checkpoint_id` and exactly the 15 baseline items: `communication_q1`–`communication_q5`, `cognitive_q1`–`cognitive_q5`, and `social_emotional_q1`–`social_emotional_q5`.
- Do not also use baseline checkpoint one-hot columns in the primary model. Encode the categorical ID inside the training pipeline.
- Prohibit T1 answers, T1 scores, T1 checkpoint fields, post-follow-up fields, targets and future-derived values from ML inputs.
- Primary targets are `target_communication_concern`, `target_cognitive_concern`, and `target_social_emotional_concern`; evaluate them separately.
- Derive overall concern by OR of the three domain decisions. Do not train a fourth primary model for `target_any_developmental_concern`.
- Every submitted questionnaire field is required. `NOT_SURE` and `NOT_OBSERVED` are explicit valid responses but are unscorable; an absent field is a validation error. Never convert an unscorable response to zero.
- Mark an assessment `INCOMPLETE` if any of the 15 responses is `NOT_SURE` or `NOT_OBSERVED`. Calculate final scores and run the current ML contract only when all 15 responses are scoreable. Validated safety/referral rules still run for incomplete assessments.
- Split by participant. Fit preprocessing, feature selection, resampling, calibration and threshold selection using training/development data only.
- Keep offline training separate from production inference.
- Keep ML predictions, deterministic safety/checkpoint rules, approved-activity selection and LLM personalization as separate boundaries. An LLM cannot alter predictions or rules.
- Use plain, non-diagnostic parent wording; do not expose raw probabilities or technical explanation values to parents.
- Never commit, transmit, publish, log or place the processed CSV or participant identifiers in LLM prompts. Minimize sensitive fields in APIs and errors.
- Preserve ports-and-adapters boundaries: domain/application code must not depend on HTTP, ORM/ODM, database, broker, cloud or provider SDK types.
- Do not access another service's database or perform cross-service database joins. Integrate through documented APIs or events.
- Before changing data or feature definitions, verify schema, dimensions, missingness, leakage boundaries, participant grouping, target derivation and artifact checksums.

Detailed requirements and evidence are in [`docs/`](docs/), especially [data-audit.md](docs/data-audit.md), [research-ml-specification.md](docs/research-ml-specification.md), [ml-training-plan.md](docs/ml-training-plan.md) and [service-architecture-plan.md](docs/service-architecture-plan.md).
