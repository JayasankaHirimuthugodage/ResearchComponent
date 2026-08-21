# Research decision log

Date: 2026-08-20

## Confirmed decisions

| Decision | Rationale / evidence |
|---|---|
| Service name is `slc-development-assessment-service` | Defined research-component boundary |
| Scope is developmental monitoring and research, not diagnosis | Safety and claims boundary in the research brief |
| Processed CSV belongs in `data/processed/` | It is a supplied modelling export rather than a verified untouched source-system export |
| Preserve the CSV byte-for-byte and exclude it narrowly from Git | Privacy and research reproducibility; pre/post SHA-256 is identical |
| Primary ML task has three domain-specific binary targets | Communication, cognitive and socio-emotional outcomes require separate evaluation/explanation |
| `target_any_developmental_concern` is derived by OR | It agrees with the three-domain OR for all 507 records and should not become a fourth primary model |
| Primary features are 15 T0 item responses plus `baseline_checkpoint_id` | Checkpoint supplies age/item meaning; intentionally restrictive confirmatory design |
| Use exactly one baseline checkpoint representation | Supplied ID and one-hot group agree for all records; dual use is redundant |
| All T1 fields are prohibited inputs | Follow-up scores deterministically yield targets; actual follow-up checkpoint is future timing |
| Every submitted questionnaire field is required | Prevents accidental blank responses; missing fields are rejected by schema validation |
| `NOT_SURE` and `NOT_OBSERVED` are explicit unscorable responses | They produce `INCOMPLETE`, are never converted to zero, and may be updated after observation |
| Final scoring and the current ML contract require all 15 responses to be scoreable | The current feature contract has no validated partial-response model or imputation path |
| Validated safety/referral rules run for incomplete assessments | Completeness must not suppress serious caregiver concern, skill loss/regression or another safety condition |
| Participant-level splitting is mandatory | Prevents future longitudinal records for a child crossing partitions; current file has one record per participant |
| Training-only preprocessing, resampling, calibration and threshold selection | Prevents optimistic leakage |
| No deep-learning default | 507 records and 53–61 domain positives favour classical, interpretable tabular models |
| ML, deterministic rules, approved activity selection and LLM personalization are separate boundaries | Prevents probabilistic inference or free-form generation from overriding safety logic |
| Ports-and-adapters architecture for one bounded service | Preserves domain/application logic across transport, model, provider and database changes |
| No production database decision in Phase 0 | External monorepo standards and operational requirements are unknown |
| No cross-service database access | Service autonomy, privacy and contract ownership |
| Pediatric validation status is confirmed by the project owner | Medical scoring and deterministic rule-based elements were reported as pediatrician-validated; reviewer identity, date and signed/versioned validation artifact were not supplied to this repository and must not be fabricated |

## Evidence-based observations, not policy decisions

- The dataset contains 507 rows and 66 columns with no duplicate participant or assessment identifier.
- All 15 baseline item fields contain only `0`, `1`, `2` and no blanks.
- The export contains no T1 item responses, dates of birth, chronological baseline ages or corrected ages.
- Four `reporting_method` values are blank.
- Gestational-age missingness is structurally consistent: 44 premature records have ages 32–36; 463 non-premature records are blank with the missing flag set.
- All one-hot groups are valid; three never-active categories are documented in the audit.
- Domain positive rates are 10.45%–12.03%; results will have substantial sampling uncertainty.
- The UUID-shaped assessment identifiers are unique, but their generation namespace/inputs are not documented.

## Decisions requiring research-supervisor validation

- Confirm the questionnaire identity/version and whether checkpoint item wording is final.
- Confirm the scoring-rule version and that `0/1/2` semantically maps to `NOT_YET/SOMETIMES/YES_USUALLY` in the source system.
- Confirm T0/T1 timing windows and rules for records that retain the same checkpoint at follow-up.
- Approve the primary performance-selection rule, acceptable uncertainty, subgroup reporting minimums and test-set allocation.
- Decide whether any T0 contextual sensitivity analysis is scientifically justified; it must remain separate from the primary confirmatory analysis.

## Pediatric validation status and implementation evidence

The project owner reports that all medical scoring and deterministic rule-based elements have been validated by a pediatrician. This closes the Phase-0 clinical-review status for the required-field/unscorable-response policy, scoring boundary, age/checkpoint rules, safety/referral rules and prohibited demographic concern rules.

Before implementing those rules, the project must reference or archive the authoritative validated rule version and convert it into explicit acceptance tests. The repository must not fabricate the pediatrician's identity, validation date, document identifier or the exact resolution of wording that was not supplied here. In particular, the validated resolution of the reported 20–36-week form range versus the mathematical 37-week boundary must be transcribed rather than inferred.

Operational ML sensitivity/specificity constraints and threshold utility remain research/model-governance decisions unless they are explicitly included in the pediatric validation artifact.

## Unresolved architecture and technology choices

- External monorepo language, framework, package/build tooling and shared conventions.
- Production database product, schema strategy and migration tooling.
- Child/profile identity and context API/event contracts.
- Authentication, authorization, consent, retention, deletion and data-residency standards.
- Synchronous endpoint granularity versus a single orchestrated assessment workflow.
- Event broker/product and whether an outbox is required.
- Model artifact registry/storage, signing and deployment mechanism.
- Approved ML/runtime libraries and environment-lock format.
- Observability stack and redaction implementation.
- Approved activity catalogue ownership and review workflow.
- LLM provider, privacy/retention terms, prompt versioning and safety evaluation.

## Deliberately deferred

- Data cleaning or transformation.
- Raw-to-processed feature-generation implementation.
- Model training, tuning, calibration, threshold selection and artifacts.
- Production inference and API code.
- Database initialization and migrations.
- Deterministic clinical/safety rule implementation.
- Recommendation engine and activity catalogue implementation.
- LLM connection or prompt implementation.
- Frontend, container and deployment configuration.
- Event-broker implementation.
- Moving the project into an external monorepo.
- Git initialization, commit or push.

## Next approval gate

Recommended next phase: implement schema/leakage validation and create the reproducible participant-level development/final-test split described in [ml-training-plan.md](ml-training-plan.md). Verify those gates before fitting a model. Model training follows only after feature, target, group and untouched-test boundaries pass their checks.

## Phase-1 development baseline decisions

Date: 2026-08-21

| Decision | Rationale / evidence |
|---|---|
| Fit one binary classifier per domain for each candidate | Preserves domain-specific evaluation and explanation; three targets do not require three datasets |
| Start with class-weighted, L2-regularized logistic regression | Appropriate interpretable baseline for 405 development records and 42–48 positives per domain |
| Use one shared five-fold, five-repeat allocation | Enables paired candidate/domain comparison and prevents selectively easier validation partitions |
| Stratify on the three-label combination, then search for checkpoint balance | Every observed combination has at least five development records; deterministic secondary balancing reduces checkpoint skew without changing the label-first design |
| Fit checkpoint encoding and questionnaire scaling inside each training fold | Prevents preprocessing leakage and keeps the complete inference transformation together |
| Version probability `0.50` as approved for the current logistic configuration | The project owner supplied the exact statement: “The operational predicted-probability threshold of 0.50 was reviewed and approved by the consultant pediatrician for this model version.” This remains distinct from the observed score `<= 50` target rule |
| Keep final-test evaluation unavailable from the development CLI | Prevents iterative model comparison from consuming the locked 102-participant evaluation set |
| Emit only aggregate evaluation and coefficient-stability evidence | Supports explainability research without writing participant predictions or identifiers |
| Use synthetic fixtures for local contract tests but not training | Tests leakage/fold/report behavior without augmenting the small research dataset |
| Keep the experiment framework-neutral | Later candidates and a future service/monorepo adapter can reuse contracts without importing HTTP or database concerns |

The repository assigns `domain-probability-threshold-v1` to the approved cutoff and scopes it to `logistic-l2-balanced-v1`. These are repository traceability identifiers, not fabricated clinical document identifiers. The consultant pediatrician's identity, approval date and authoritative document reference remain unrecorded.

Implementation and local commands are documented in [development-experiment.md](development-experiment.md). Candidate expansion, tuning, calibration, threshold revalidation for a changed model configuration, final-test evaluation and model serialization remain separate approval gates.
