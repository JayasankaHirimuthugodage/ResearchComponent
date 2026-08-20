# Research and ML specification

## Purpose and scope

The research component is an explainable, safety-constrained early-childhood **Socio-Emotional, Language/Communication and Cognitive/Problem-Solving (SLC)** developmental monitoring and three-month outcome-prediction system for children approximately 0–5 years old.

It is not a diagnostic or autism-specific screening system. It must not diagnose autism, developmental delay or any medical condition; replace pediatric assessment; or present an ML result as a clinical conclusion.

## Research objective

Primary question: using only information available when a baseline developmental assessment is completed, can the system predict each of three domain-specific concern outcomes at a follow-up assessment approximately three months later and explain those decisions without crossing clinical or safety boundaries?

- **T0:** baseline assessment and context available at prediction time.
- **T1:** subsequent assessment approximately three months later.
- **Unit represented by the supplied file:** one baseline/follow-up modelling record per participant; 507 participants and 507 records were observed.
- **Primary task:** three related but separately evaluated binary classification targets.

## Questionnaire and scoring definition

Each checkpoint contains five items for each domain: communication/language, cognitive/problem-solving and socio-emotional.

```text
YES_USUALLY = 2
SOMETIMES = 1
NOT_YET = 0
NOT_SURE = unscorable/missing
NOT_OBSERVED = unscorable/missing
Unanswered/absent field = invalid submitted assessment
```

For a complete five-item domain:

```text
maximum_points = 5 × 2 = 10
domain_score_percentage = obtained_points / 10 × 100
```

Complete-domain scores can be `0, 10, 20, …, 100`.

The supplied modelling export contains only `0`, `1` and `2` across all 15 baseline item columns. It does not retain unscorable baseline states, T1 item responses, questionnaire version or scoring-rule version. This limits independent reconstruction. A future source-to-feature pipeline must keep unscorable states distinct from `NOT_YET = 0`.

### Required-field and unscorable-response policy

The project owner reports that the medical scoring and deterministic rule design has been validated by a pediatrician. The following required-field policy is therefore fixed for implementation:

- every submitted assessment must contain exactly one response for each of the 15 questionnaire items;
- a missing item field is a request/schema validation error and the submission is not accepted as a submitted assessment;
- `NOT_SURE` and `NOT_OBSERVED` are explicit permitted responses, not blanks, but they remain unscorable;
- an assessment containing either unscorable response has status `INCOMPLETE`;
- never map, impute or coerce an unscorable response to `NOT_YET = 0`, `SOMETIMES = 1` or `YES_USUALLY = 2`;
- calculate final domain scores, concern labels, the overall result and the current 16-feature ML prediction only when all 15 responses are scoreable;
- do not issue ML-dependent recommendations for an incomplete assessment;
- run validated caregiver-concern, skill-loss/regression and other safety/referral rules even when the assessment is incomplete;
- allow the caregiver to return after observation and replace `NOT_SURE` or `NOT_OBSERVED` with an evidence-based scoreable response.

Assessment statuses are:

```text
DRAFT       = assessment not submitted; some fields may not yet have responses
INCOMPLETE  = all required fields are present, but at least one is NOT_SURE or NOT_OBSERVED
COMPLETE    = all 15 responses are scoreable
```

Suggested non-diagnostic message for an incomplete assessment:

> Some items are marked “Not sure” or “Not observed.” We need a response based on observation before providing the developmental result. Please observe the relevant activities and update these responses when possible.

This policy does not change the supplied CSV: all 7,605 baseline questionnaire cells are already scoreable.

## Primary feature contract

The confirmatory model has exactly 16 conceptual inputs:

```text
baseline_checkpoint_id
communication_q1
communication_q2
communication_q3
communication_q4
communication_q5
cognitive_q1
cognitive_q2
cognitive_q3
cognitive_q4
cognitive_q5
social_emotional_q1
social_emotional_q2
social_emotional_q3
social_emotional_q4
social_emotional_q5
```

`baseline_checkpoint_id` is required because the meaning of an item such as `communication_q1` changes across age-appropriate checkpoints. It should be encoded categorically inside the fold-fitted training pipeline.

The supplied 12 `baseline_checkpoint_cp*` one-hot columns are valid but redundant. The primary experiment must not use both the categorical ID and its one-hot representation. Prefer the ID.

Baseline domain scores are not present. If later generated, they must not be included alongside all 15 component answers in the primary model. A score-only or answers-plus-checkpoint ablation can be proposed separately but requires approval.

## Target contract

Primary targets:

```text
target_communication_concern
target_cognitive_concern
target_social_emotional_concern
```

For each domain in the supplied data:

```text
domain_concern = 1 when followup_domain_score <= 50
domain_concern = 0 when followup_domain_score > 50
```

The audit found zero score/label contradictions. These three targets must be evaluated and explained separately regardless of whether implementation uses three estimators or a multi-output wrapper.

Overall concern is a derived reporting result:

```text
predicted_any_concern =
    predicted_communication_concern
    OR predicted_cognitive_concern
    OR predicted_social_emotional_concern
```

Do not train a fourth primary model for `target_any_developmental_concern`. The supplied field agrees with the OR rule in all 507 rows and may be retained for traceability and consistency testing.

## Explicit feature denylist

The following are prohibited from the primary feature matrix:

- `participant_number` and `baseline_assessment_id`;
- `reporting_method` and any questionnaire/scoring/version metadata;
- `profile_sex_male`, `profile_sex_female`;
- all prematurity, gestational-age, gestational-missingness and birth-weight fields;
- all 12 baseline checkpoint one-hot columns when `baseline_checkpoint_id` is used;
- `followup_checkpoint_id` and all 12 follow-up checkpoint one-hot columns;
- all T1 questionnaire responses if they become available;
- `followup_communication_score`, `followup_cognitive_score`, `followup_social_emotional_score`;
- all four `target_*` columns when constructing `X`;
- post-follow-up variables and transformations using T1 information;
- baseline domain totals when all 15 baseline item responses are used in the primary model;
- identifiers, row order, or derived values whose provenance cannot prove T0 availability.

Sex is reserved for fairness/subgroup evaluation. Prematurity and gestational age belong to deterministic age-correction/checkpoint routing. Birth weight is contextual metadata unless separately clinically validated. None of these exclusions implies a deterministic developmental-concern rule.

A scientifically justified T0 contextual sensitivity analysis may be designed later. It must be clearly labelled exploratory, use a separately approved allowlist, and never replace the confirmatory primary analysis.

## Leakage boundary and validation invariants

Only T0 information may become an input. The schema validator must fail closed on prohibited or unknown columns and verify:

- exact primary feature and target allowlists;
- participant groups before any split;
- no T1 or post-follow-up fields in feature preparation;
- no target copy, outcome proxy or future checkpoint field;
- checkpoint representation is singular;
- preprocessing, feature selection and resampling are fit inside training folds only;
- calibration and threshold selection use development data only;
- inference accepts the frozen feature schema and version.

The follow-up scores are legitimate evidence for constructing/verifying targets, but supplying them to a model would be direct leakage. The actual follow-up checkpoint is future timing and must also remain outside the model.

## Age correction and checkpoint routing

Age correction is deterministic application logic, not a primary ML predictor:

```text
prematurity_weeks = 40 - gestational_age_weeks
corrected_age_days = chronological_age_days - (prematurity_weeks × 7)
```

Correction is considered only when prematurity is `YES`, gestational age is known and below 40 weeks, the child was at least three weeks premature, chronological age is under approximately 730.5 days, the chronological checkpoint permits correction, and correction is enabled from CP2M through CP24M. It is disabled from CP30M onward.

The reported form range is 20–36 weeks, while a 40-week reference and “at least three weeks” criterion could mathematically include 37 weeks. The project owner reports pediatric validation of the medical scoring/rule design. Before implementation, the exact validated resolution of this form-boundary wording must be transcribed from the authoritative validation record and versioned; it must not be inferred from the modelling CSV.

The dataset contains no date of birth, chronological baseline age, corrected baseline age, chronological age days or corrected age days. These must not be reverse-engineered. `baseline_checkpoint_id` is the available ML age context; a future production service must obtain chronological and correction inputs through child-profile or survey context.

## ML, deterministic rules and recommendations

These boundaries must remain distinct:

1. **ML inference** returns three domain decisions plus technical evidence needed for approved explanation processes.
2. **Derived reporting** calculates overall concern by OR.
3. **Deterministic rules** enforce the required-field/completeness policy, handle age/checkpoint routing, reported skill loss/regression, serious caregiver concern, hearing/vision and supporting motor safety checks, referral guidance, activity suitability and recommendation-withholding conditions. Safety/referral evaluation still runs for `INCOMPLETE` assessments.
4. **Activity selection** chooses only approved, age-appropriate catalogue entries for relevant domains.
5. **Optional LLM personalization** may rephrase or personalize already-approved activities using minimal permitted context.

Rules for hearing, vision, gross motor and fine motor are supporting safety checks, not SLC ML targets and not a replacement for the separate physical-development service. No new clinical threshold is authorized here.

## Explainability specification

Every domain requires its own explanation.

- Logistic regression: signed, scaled coefficients for global audit and per-record feature contributions with suitable preprocessing traceability.
- Compatible tree models: SHAP values for technical research/audit views after version and correctness checks.
- Permutation importance: supplementary global analysis performed on held-out/development folds, not a causal explanation.

Explanation language must be associational, not causal. An influential feature does not prove it caused a concern outcome.

Technical explanations may be retained for research evaluation, model audit and appropriately designed professional views. Parent-facing text must omit raw SHAP values, coefficients, technical probabilities and diagnostic wording.

Example parent-facing wording:

> Your child’s communication development may benefit from additional age-appropriate activities and continued observation.

Do not display “predicted score” or raw concern probability as though it were a developmental measurement.

## LLM boundary

An LLM may explain or personalize only activities already approved and selected by deterministic logic. It must not:

- diagnose or generate a new clinical conclusion;
- change an ML decision or overall OR result;
- override, weaken or omit a safety/referral rule;
- invent an intervention or activity outside the approved catalogue;
- provide unsupported medical advice;
- receive participant numbers, assessment IDs, raw research rows or unnecessary sensitive attributes.

## Versioning and traceability requirements

Future results must record dataset checksum, source/export provenance, feature-schema version, target definition, questionnaire version if known, scoring-rule version if known, split seed and group allocation, training configuration, dependency lockfile, model version, calibration version, threshold version, evaluation results, explanation implementation version and artifact checksums. Unknown questionnaire or scoring versions must remain explicitly `unknown`, not fabricated.
