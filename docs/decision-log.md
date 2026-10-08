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
| `target_any_developmental_concern` is derived by OR | It agrees with the three-domain OR for all 625 records and should not become a fourth primary model |
| Primary features are 15 T0 item responses plus `baseline_checkpoint_id` | Checkpoint supplies age/item meaning; intentionally restrictive confirmatory design |
| Use exactly one baseline checkpoint representation | Supplied ID and one-hot group agree for all records; dual use is redundant |
| All T1 fields are prohibited inputs | Follow-up scores deterministically yield targets; actual follow-up checkpoint is future timing |
| Every submitted questionnaire field is required | Prevents accidental blank responses; missing fields are rejected by schema validation |
| `NOT_SURE` and `NOT_OBSERVED` are explicit unscorable responses | They produce `INCOMPLETE`, are never converted to zero, and may be updated after observation |
| Final scoring and the current ML contract require all 15 responses to be scoreable | The current feature contract has no validated partial-response model or imputation path |
| Validated safety/referral rules run for incomplete assessments | Completeness must not suppress serious caregiver concern, skill loss/regression or another safety condition |
| Participant-level splitting is mandatory | Prevents future longitudinal records for a child crossing partitions; current file has one record per participant |
| Training-only preprocessing, resampling, calibration and threshold selection | Prevents optimistic leakage |
| No deep-learning default | 625 records and 67–76 domain positives favour classical, interpretable tabular models |
| ML, deterministic rules, approved activity selection and LLM personalization are separate boundaries | Prevents probabilistic inference or free-form generation from overriding safety logic |
| Ports-and-adapters architecture for one bounded service | Preserves domain/application logic across transport, model, provider and database changes |
| No production database decision in Phase 0 | External monorepo standards and operational requirements are unknown |
| No cross-service database access | Service autonomy, privacy and contract ownership |
| Pediatric validation status is confirmed by the project owner | Medical scoring and deterministic rule-based elements were reported as pediatrician-validated; reviewer identity, date and signed/versioned validation artifact were not supplied to this repository and must not be fabricated |

## Evidence-based observations, not policy decisions

- The dataset contains 625 rows and 66 columns with no duplicate participant or assessment identifier.
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
| Start with class-weighted, L2-regularized logistic regression | Appropriate interpretable baseline for 500 development records and 53–61 positives per domain |
| Use one shared five-fold, five-repeat allocation | Enables paired candidate/domain comparison and prevents selectively easier validation partitions |
| Stratify on the three-label combination, then search for checkpoint balance | Every observed combination has at least five development records; deterministic secondary balancing reduces checkpoint skew without changing the label-first design |
| Fit checkpoint encoding and questionnaire scaling inside each training fold | Prevents preprocessing leakage and keeps the complete inference transformation together |
| Version probability `0.50` as approved for the current logistic configuration | The project owner supplied the exact statement: “The operational predicted-probability threshold of 0.50 was reviewed and approved by the consultant pediatrician for this model version.” This remains distinct from the observed score `<= 50` target rule |
| Keep final-test evaluation unavailable from the development CLI | Prevents iterative model comparison from consuming the locked 125-participant evaluation set |
| Emit only aggregate evaluation and coefficient-stability evidence | Supports explainability research without writing participant predictions or identifiers |
| Use synthetic fixtures for local contract tests but not training | Tests leakage/fold/report behavior without augmenting the small research dataset |
| Keep the experiment framework-neutral | Later candidates and a future service/monorepo adapter can reuse contracts without importing HTTP or database concerns |

The repository assigns `domain-probability-threshold-v1` to the approved cutoff and scopes it to `logistic-l2-balanced-v1`. These are repository traceability identifiers, not fabricated clinical document identifiers. The consultant pediatrician's identity, approval date and authoritative document reference remain unrecorded.

Implementation and local commands are documented in [development-experiment.md](development-experiment.md). This entry records the earlier logistic-only approval context; it is not the final three-domain operating-threshold policy.

## Phase-1 local prediction demonstration decisions

Date: 2026-08-22

| Decision | Rationale / evidence |
|---|---|
| Commit `developmental-questionnaire-v3` as a versioned catalogue | The project owner supplied and authorized the wording; 180 unique question IDs map 12 checkpoints onto the frozen 15-item feature schema |
| Record missing provenance as `not supplied` | The supplied reference did not include an underlying licence or authoritative external document identifier; neither may be fabricated |
| Keep Streamlit as a thin local adapter | Questionnaire validation, encoding and prediction contracts remain reusable by a future HTTP/microservice adapter |
| Accept manual entry and optional synthetic quick-fill | Manual entry shows the real question-and-answer workflow; synthetic fixtures make the demo repeatable without presenting participant records |
| Store no entered answers and no fitted artifact | The demo is local research review; responses remain in the Streamlit session and the provisional logistic fits remain in process memory |
| Fit from the 500-participant development partition only | The input-to-output flow can run while the 125-participant final-test partition stays locked |
| Do not calculate age or route checkpoints in the demo | Age correction and deterministic routing are separate validated-rule concerns; the demo receives an already-determined baseline checkpoint |
| Display probabilities only in the technical research-review view | Raw probabilities are useful for model review but remain unsuitable for the future parent-facing interface |
| Keep `NOT_SURE` and `NOT_OBSERVED` unscorable | The inference boundary returns `INCOMPLETE` and invokes no model rather than silently converting either response to zero |

The demo's in-memory full-development fit is provisional and is not additional performance evidence. It remains separate from the subsequently frozen research artifacts and does not represent the final three-domain selection.

## Phase-1 Extra Trees comparison decisions

Date: 2026-08-22

| Decision | Rationale / evidence |
|---|---|
| Add `extra_trees_balanced_leaf5` as the second candidate | Provides a nonlinear classical comparison suitable for the small tabular dataset without introducing deep learning |
| Use 300 trees, balanced class weights and minimum leaf size 5 | Predeclared deterministic configuration balances stability, class imbalance and regularization; it is not a tuned optimum |
| Do not scale questionnaire values for Extra Trees | Tree split decisions do not require feature scaling; checkpoint encoding remains inside each fitted fold |
| Use the exact same participant folds as logistic | Enables paired comparison and prevents the second candidate from receiving an easier allocation |
| Use held-out permutation importance scored by average precision | Produces model-appropriate, conceptual-input explanations on validation data rather than misusing logistic coefficients or impurity-only importance |
| Treat Extra Trees probability `0.50` as experimental | The logistic approval does not transfer to a different model configuration even when the numeric cutoff is identical |
| Expose both candidates in a separately labelled Streamlit research comparison | The same answers can demonstrate both outputs without implying model selection; Logistic retains its approved threshold metadata and the Extra Trees cutoff remains experimental |

The Extra Trees configuration is versioned as `extra-trees-balanced-leaf5-v1`. It remains a development candidate: no final-test evaluation, calibration selection, model serialization or production routing change is authorized by this comparison.

## Final development selection and locked-test decisions

The following later decisions supersede the earlier two-candidate comparison for the frozen research artifacts while preserving it as historical evidence:

| Decision | Rationale |
|---|---|
| Compare five predefined configurations | Logistic Regression, RBF SVM, Random Forest, Extra Trees and XGBoost receive the same development folds and feature contract |
| Select separately by domain | Mean repeated-cross-validation Average Precision determines the selected candidate for each target |
| Use nested sigmoid calibration | Calibration is learned only from development out-of-fold predictions |
| Select thresholds by balanced accuracy/Youden's J | Domain thresholds use averaged calibrated out-of-fold development predictions; ties prefer sensitivity, then specificity, then the lower threshold |
| Freeze XGBoost at `0.130882` for communication | Highest development mean Average Precision for that domain |
| Freeze Logistic Regression at `0.109680` for cognitive | Highest development mean Average Precision for that domain; estimator age is not a rejection criterion |
| Freeze XGBoost at `0.126137` for social-emotional | Highest development mean Average Precision for that domain |
| Evaluate the locked partition once | The 125-participant test evaluation occurred after freezing; models and thresholds were not changed on test data |

The questionnaire outcome cutoff of `<= 50%` remains distinct from these calibrated ML probability thresholds. The historical `0.50` comparison cutoff is not the final operating threshold for any of the three frozen domain artifacts.
