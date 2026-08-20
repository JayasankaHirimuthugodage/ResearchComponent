# Dataset audit

Audit date: 2026-08-20  
Audit scope: aggregate, read-only inspection of the supplied modelling CSV; no participant records are reproduced here.

## Provenance and file-integrity evidence

| Property | Before organisation | After organisation |
|---|---|---|
| Path | `survey-ml-cross-checkpoint-classification (2).csv` | `data/processed/survey-ml-cross-checkpoint-classification (2).csv` |
| Filename | `survey-ml-cross-checkpoint-classification (2).csv` | unchanged |
| Size | 99,794 bytes | 99,794 bytes |
| SHA-256 | `96521381C630E7AF4A551007C55F5703DDAAAEEDFCA3C3A09435A8082688F4D5` | `96521381C630E7AF4A551007C55F5703DDAAAEEDFCA3C3A09435A8082688F4D5` |
| Data rows | 507 | 507 |
| Columns | 66 | 66 |

The matching checksum, size and dimensions demonstrate that organisation was a byte-preserving move. The CSV was not re-saved, cleaned, sorted, encoded, imputed or otherwise transformed.

The repository contained exactly one CSV candidate. Before Phase 0 it contained no other files or directories, no `AGENTS.md`, and no Git repository. Git was not initialized.

## Structure

All 507 parsed data rows had exactly 66 fields. No unexpected index column or empty column was found.

Columns, in source order:

```text
participant_number
reporting_method
baseline_assessment_id
baseline_checkpoint_id
followup_checkpoint_id
communication_q1, communication_q2, communication_q3, communication_q4, communication_q5
cognitive_q1, cognitive_q2, cognitive_q3, cognitive_q4, cognitive_q5
social_emotional_q1, social_emotional_q2, social_emotional_q3, social_emotional_q4, social_emotional_q5
baseline_checkpoint_cp2m, baseline_checkpoint_cp4m, baseline_checkpoint_cp6m,
baseline_checkpoint_cp9m, baseline_checkpoint_cp12m, baseline_checkpoint_cp15m,
baseline_checkpoint_cp18m, baseline_checkpoint_cp24m, baseline_checkpoint_cp30m,
baseline_checkpoint_cp36m, baseline_checkpoint_cp48m, baseline_checkpoint_cp60m
followup_checkpoint_cp2m, followup_checkpoint_cp4m, followup_checkpoint_cp6m,
followup_checkpoint_cp9m, followup_checkpoint_cp12m, followup_checkpoint_cp15m,
followup_checkpoint_cp18m, followup_checkpoint_cp24m, followup_checkpoint_cp30m,
followup_checkpoint_cp36m, followup_checkpoint_cp48m, followup_checkpoint_cp60m
profile_sex_male, profile_sex_female
profile_premature_yes, profile_premature_no, profile_premature_not_sure
profile_gestational_weeks, profile_gestational_weeks_missing
profile_birth_weight_under_1_5_kg, profile_birth_weight_1_5_to_1_99,
profile_birth_weight_2_0_to_2_49, profile_birth_weight_2_5_to_2_99,
profile_birth_weight_3_0_to_3_49, profile_birth_weight_3_5_to_3_99,
profile_birth_weight_4_or_more, profile_birth_weight_not_sure
followup_communication_score, followup_cognitive_score, followup_social_emotional_score
target_communication_concern, target_cognitive_concern,
target_social_emotional_concern, target_any_developmental_concern
```

### Inferred types, missingness and intended roles

Ranges below apply to every explicitly named member. Blank percentages use 507 as the denominator. No placeholder strings such as `NA`, `N/A`, `NULL`, `UNKNOWN`, `NOT_SURE`, `NOT_OBSERVED` or `UNANSWERED` were present in any column.

| Column(s) | Inferred stored type | Blank count (%) | Intended role / primary-ML treatment |
|---|---:|---:|---|
| `participant_number` | integer | 0 (0%) | Pseudonymous participant link and mandatory group-splitting metadata; never a feature |
| `reporting_method` | categorical string | 4 (0.79%) | Provenance/validation metadata; never a feature |
| `baseline_assessment_id` | string UUID | 0 (0%) | Traceability metadata; never a feature |
| `baseline_checkpoint_id` | categorical string | 0 (0%) | Primary age/checkpoint feature; encode inside the training pipeline |
| `followup_checkpoint_id` | categorical string | 0 (0%) | Actual T1 checkpoint; prohibited future-information feature |
| `communication_q1`–`communication_q5` | integer in {0,1,2} | 0 each (0%) | Five primary T0 communication/language features |
| `cognitive_q1`–`cognitive_q5` | integer in {0,1,2} | 0 each (0%) | Five primary T0 cognitive/problem-solving features |
| `social_emotional_q1`–`social_emotional_q5` | integer in {0,1,2} | 0 each (0%) | Five primary T0 socio-emotional features |
| `baseline_checkpoint_cp2m`, `baseline_checkpoint_cp4m`, `baseline_checkpoint_cp6m`, `baseline_checkpoint_cp9m`, `baseline_checkpoint_cp12m`, `baseline_checkpoint_cp15m`, `baseline_checkpoint_cp18m`, `baseline_checkpoint_cp24m`, `baseline_checkpoint_cp30m`, `baseline_checkpoint_cp36m`, `baseline_checkpoint_cp48m`, `baseline_checkpoint_cp60m` | binary integer | 0 each (0%) | Valid redundant representation of `baseline_checkpoint_id`; exclude from the primary model to avoid duplicate representations |
| `followup_checkpoint_cp2m`, `followup_checkpoint_cp4m`, `followup_checkpoint_cp6m`, `followup_checkpoint_cp9m`, `followup_checkpoint_cp12m`, `followup_checkpoint_cp15m`, `followup_checkpoint_cp18m`, `followup_checkpoint_cp24m`, `followup_checkpoint_cp30m`, `followup_checkpoint_cp36m`, `followup_checkpoint_cp48m`, `followup_checkpoint_cp60m` | binary integer | 0 each (0%) | Actual T1 checkpoint one-hot group; prohibited future information |
| `profile_sex_male`, `profile_sex_female` | binary integer | 0 each (0%) | Fairness/subgroup evaluation only; exclude from the primary feature matrix |
| `profile_premature_yes`, `profile_premature_no`, `profile_premature_not_sure` | binary integer | 0 each (0%) | Deterministic age-correction/checkpoint context; not an SLC concern rule or primary feature |
| `profile_gestational_weeks` | integer when applicable | 463 (91.32%) | Deterministic age-correction context; structural missingness for all non-premature records; exclude from primary ML |
| `profile_gestational_weeks_missing` | binary integer | 0 (0%) | Structural-missingness validation; exclude from primary ML |
| `profile_birth_weight_under_1_5_kg`, `profile_birth_weight_1_5_to_1_99`, `profile_birth_weight_2_0_to_2_49`, `profile_birth_weight_2_5_to_2_99`, `profile_birth_weight_3_0_to_3_49`, `profile_birth_weight_3_5_to_3_99`, `profile_birth_weight_4_or_more`, `profile_birth_weight_not_sure` | binary integer | 0 each (0%) | Contextual metadata; not a concern rule or primary feature without clinical validation |
| `followup_communication_score` | integer percentage | 0 (0%) | T1 target-construction verification; direct leakage if used as input |
| `followup_cognitive_score` | integer percentage | 0 (0%) | T1 target-construction verification; direct leakage if used as input |
| `followup_social_emotional_score` | integer percentage | 0 (0%) | T1 target-construction verification; direct leakage if used as input |
| `target_communication_concern` | binary integer | 0 (0%) | Primary communication/language target |
| `target_cognitive_concern` | binary integer | 0 (0%) | Primary cognitive/problem-solving target |
| `target_social_emotional_concern` | binary integer | 0 (0%) | Primary socio-emotional target |
| `target_any_developmental_concern` | binary integer | 0 (0%) | Derived OR consistency/reporting field; not a fourth independently trained primary target |

The only blank fields are the 4 `reporting_method` values and 463 structurally inapplicable `profile_gestational_weeks` values. All remaining columns have zero blanks.

### Constant and near-constant columns

Constant columns:

- `followup_checkpoint_cp2m = 0` for all 507 records;
- `profile_premature_not_sure = 0` for all records;
- `profile_birth_weight_not_sure = 0` for all records.

Near-constant at a 95% threshold:

| Column | Most frequent value | Share |
|---|---:|---:|
| `reporting_method` | `PARENT_REPORTED` | 99.21% (503/507; the other 4 are blank) |
| `followup_checkpoint_cp6m` | 0 | 96.45% |
| `profile_birth_weight_under_1_5_kg` | 0 | 99.61% |
| `profile_birth_weight_1_5_to_1_99` | 0 | 98.82% |
| `profile_birth_weight_4_or_more` | 0 | 96.06% |

Constant/near-constant fields remain useful for validation or provenance in some cases, but provide little or no training signal and are already outside the primary feature allowlist.

## Record and identifier integrity

- Exact duplicate rows: 0.
- Duplicate rows after excluding `participant_number` and `baseline_assessment_id`: 0.
- `participant_number`: 507 unique values; no duplicates; one record per participant. Values are positive integers from 10 to 644 with 104 gaps, so they are not a contiguous row index.
- `baseline_assessment_id`: 507 unique 36-character UUID-shaped values; 503 carry UUID version nibble 5 and 4 carry version nibble 4.
- Participants linked to multiple baseline assessment IDs: 0.
- Baseline assessment IDs linked to multiple participants: 0.
- Simple Pearson correlations between `participant_number` and the four target fields range from −0.0341 to −0.0150; no simple ordering association was observed. Identifiers remain prohibited because uniqueness can enable memorisation and their generation semantics are not documented.

The majority-version-5 UUID pattern suggests deterministic name-based generation, but its namespace and input are unknown. Therefore the audit cannot prove that `baseline_assessment_id` is independent of all source-system attributes. Treat it only as a pseudonymous trace identifier.

## Questionnaire-response audit

All 15 baseline item columns contain only numeric values `0`, `1` and `2`, with no blanks or invalid categories. This is structurally consistent with the specified mapping:

```text
0 = NOT_YET
1 = SOMETIMES
2 = YES_USUALLY
```

The modelling export contains no distinguishable `NOT_SURE`, `NOT_OBSERVED`, unanswered or other unscorable baseline response. It also contains no follow-up item responses. Consequently:

- the audit can verify the stored numeric domain but cannot independently prove the semantic mapping from the absent raw survey export;
- missing/unscorable handling cannot be reconstructed from this file;
- follow-up scores cannot be recomputed independently from T1 answers;
- production and future feature generation must preserve unscorable responses distinctly from `NOT_YET = 0` under a supervisor-approved policy.

### Response distribution per baseline item

| Item | 0 | 1 | 2 |
|---|---:|---:|---:|
| `communication_q1` | 89 | 151 | 267 |
| `communication_q2` | 88 | 148 | 271 |
| `communication_q3` | 60 | 174 | 273 |
| `communication_q4` | 72 | 148 | 287 |
| `communication_q5` | 85 | 159 | 263 |
| `cognitive_q1` | 84 | 155 | 268 |
| `cognitive_q2` | 83 | 173 | 251 |
| `cognitive_q3` | 95 | 153 | 259 |
| `cognitive_q4` | 100 | 144 | 263 |
| `cognitive_q5` | 91 | 141 | 275 |
| `social_emotional_q1` | 79 | 167 | 261 |
| `social_emotional_q2` | 76 | 155 | 276 |
| `social_emotional_q3` | 85 | 143 | 279 |
| `social_emotional_q4` | 86 | 164 | 257 |
| `social_emotional_q5` | 88 | 156 | 263 |

Every item contains all three response categories within every baseline checkpoint; no item is constant within any checkpoint. There are 500 unique 15-item numeric patterns overall. One pattern occurs in 8 records across checkpoints; equal numeric patterns across checkpoints do not imply equal item content because checkpoint questions represent different behaviours. Within checkpoints, CP15M has 46 patterns among 47 records, CP36M has 40 among 41, and CP48M has 37 among 38; every other checkpoint has one unique pattern per record. No full-row or non-identifier duplicate was found.

## One-hot validation

| Group | Members | Binary violations | Exactly one active | All-zero | Multiple active | Never-active columns |
|---|---:|---:|---:|---:|---:|---|
| Baseline checkpoint | 12 | 0 | 507 | 0 | 0 | none |
| Follow-up checkpoint | 12 | 0 | 507 | 0 | 0 | `followup_checkpoint_cp2m` |
| Sex | 2 | 0 | 507 | 0 | 0 | none |
| Prematurity | 3 | 0 | 507 | 0 | 0 | `profile_premature_not_sure` |
| Birth weight | 8 | 0 | 507 | 0 | 0 | `profile_birth_weight_not_sure` |

The categorical checkpoint IDs agree with their one-hot groups for all 507 rows. The primary model should nevertheless use only `baseline_checkpoint_id` and encode it within training folds.

Supporting distributions:

- sex: 262 male, 245 female;
- prematurity: 44 yes, 463 no, 0 not sure;
- birth weight categories, in ascending order plus not-sure: 2, 6, 61, 190, 169, 59, 20, 0.

## Gestational-age consistency

All 44 premature records have known gestational age and `profile_gestational_weeks_missing = 0`. All 463 non-premature records have a blank gestational-age value and the missing flag set to 1. There are no contradictions.

Known gestational-age distribution: 32 weeks (3), 33 (2), 34 (11), 35 (13), 36 (15). All lie within the reported form range of 20–36 weeks.

This does not resolve the design discrepancy between a form maximum of 36 weeks and a 40-week reference with an “at least three weeks premature” rule that could mathematically include 37 weeks. Pediatrician or research-supervisor validation is required.

## Checkpoints and longitudinal timing

### Checkpoint distributions

| Checkpoint | Baseline | Follow-up |
|---|---:|---:|
| CP2M | 38 | 0 |
| CP4M | 39 | 38 |
| CP6M | 44 | 18 |
| CP9M | 45 | 65 |
| CP12M | 47 | 45 |
| CP15M | 47 | 47 |
| CP18M | 46 | 48 |
| CP24M | 45 | 65 |
| CP30M | 42 | 50 |
| CP36M | 41 | 58 |
| CP48M | 38 | 38 |
| CP60M | 35 | 35 |

Observed baseline-to-follow-up transitions:

| Transition | Count | Transition | Count |
|---|---:|---|---:|
| CP2M → CP4M | 38 | CP24M → CP24M | 20 |
| CP4M → CP6M | 18 | CP24M → CP30M | 25 |
| CP4M → CP9M | 21 | CP30M → CP30M | 25 |
| CP6M → CP9M | 44 | CP30M → CP36M | 17 |
| CP9M → CP12M | 45 | CP36M → CP36M | 41 |
| CP12M → CP15M | 47 | CP48M → CP48M | 38 |
| CP15M → CP18M | 47 | CP60M → CP60M | 35 |
| CP18M → CP18M | 1 | CP18M → CP24M | 45 |

Checkpoint-label deltas are 0 months (160), 2 (56), 3 (183), 5 (21) and 6 (87). No follow-up checkpoint is earlier than baseline and no jump exceeds 12 checkpoint months. Same-label transitions can be plausible for wider older-age checkpoint windows, but exact chronological plausibility cannot be verified because date of birth, chronological baseline age, corrected baseline age and assessment dates are absent. The single CP18M → CP18M record and all same-checkpoint cases require source-age/date validation rather than being declared erroneous.

`followup_checkpoint_id` and its one-hot group describe realised T1 timing and are therefore future-information leakage risks. They must not be primary ML inputs.

## Scores, targets and class balance

All observed follow-up scores lie between 30 and 100 and use permitted 10-point increments. Scores below 30 do not occur; this is an observed sample property, not a scoring-rule restriction.

| Score | Communication | Cognitive | Socio-emotional |
|---:|---:|---:|---:|
| 30 | 11 | 5 | 8 |
| 40 | 18 | 25 | 20 |
| 50 | 32 | 23 | 30 |
| 60 | 56 | 61 | 42 |
| 70 | 90 | 112 | 100 |
| 80 | 131 | 120 | 133 |
| 90 | 116 | 109 | 114 |
| 100 | 53 | 52 | 60 |

For all three domains, every label equals `1` exactly when its follow-up score is `<= 50`; contradictions: 0.

| Target | Negative | Positive | Positive rate | Negative:positive |
|---|---:|---:|---:|---:|
| `target_communication_concern` | 446 | 61 | 12.03% | 7.31:1 |
| `target_cognitive_concern` | 454 | 53 | 10.45% | 8.57:1 |
| `target_social_emotional_concern` | 449 | 58 | 11.44% | 7.74:1 |
| `target_any_developmental_concern` | 406 | 101 | 19.92% | 4.02:1 |

The three primary tasks are materially imbalanced. Accuracy alone would be misleading.

All eight three-domain label combinations are present:

| Communication, cognitive, socio-emotional | Count |
|---|---:|
| 000 | 406 |
| 001 | 10 |
| 010 | 15 |
| 011 | 15 |
| 100 | 21 |
| 101 | 17 |
| 110 | 7 |
| 111 | 16 |

`target_any_developmental_concern` equals the OR/max of the three domain targets for every record; contradictions: 0. It is suitable for consistency checking and derived reporting, not as a fourth independent primary target.

## Leakage assessment

Direct prohibited outcome information:

- `followup_communication_score`, `followup_cognitive_score`, `followup_social_emotional_score` deterministically map to their corresponding targets at the `<= 50` threshold;
- all four `target_*` fields are outcomes, not predictors.

Future-timing information:

- `followup_checkpoint_id` and all 12 `followup_checkpoint_*` columns describe actual T1 routing/timing and must be excluded.

Redundant but not future-derived:

- the 12 `baseline_checkpoint_*` columns duplicate `baseline_checkpoint_id`; use exactly one representation in the primary model;
- future creation of baseline domain totals alongside all 15 answers would add deterministic redundancy and is excluded from the primary specification.

No low-cardinality T0 or contextual column outside the target/follow-up fields deterministically mapped to any domain target. No exact or binary-complement target copies were found. Identifiers remain denied because their source-generation semantics are not established. A future pipeline must also prevent preprocessing, resampling, feature selection, calibration or threshold fitting on the complete dataset before splitting.

## Privacy assessment

No names, addresses, phone numbers, email addresses, dates of birth or raw dates are present.

- Pseudonymous identifiers: `participant_number`, `baseline_assessment_id`.
- Quasi-identifiers/context: checkpoint, sex, prematurity status, gestational age and birth-weight category, especially in combination or in rare categories.
- Sensitive research observations: the 15 questionnaire responses, follow-up scores and concern labels.
- Provenance metadata: `reporting_method`.

Public APIs should not expose internal assessment UUIDs or participant numbers unnecessarily. Logs, metrics labels, traces, error messages and LLM prompts must not contain identifiers, raw item responses, gestational age, birth weight, scores or target labels. An LLM should receive only the minimum approved context needed to explain already-selected activities—normally domain, age/checkpoint band, approved activity text and allowed personalization fields—with correlation/request IDs that are not participant identifiers.

## Data-quality conclusions and limitations

Strengths:

- stable rectangular structure;
- no duplicate records or contradictory ID mappings;
- valid baseline item domain and one-hot encodings;
- internally consistent gestational missingness;
- internally consistent score-derived targets and overall OR field;
- all three positive classes and all eight label combinations occur.

Limitations and risks:

- the raw survey export, item wording/version, scoring-rule version and feature-generation pipeline are absent;
- all baseline responses are scoreable, so unscorable-response handling cannot be evaluated;
- T1 answers and assessment dates/ages are absent, preventing independent score and exact timing verification;
- four reporting-method values are missing;
- several fields/categories are constant or rare;
- domain-positive counts are only 53–61, limiting model and subgroup precision;
- assessment UUID generation semantics and monorepo identifier standards are unknown;
- follow-up score, target and checkpoint columns create a high-risk leakage boundary that requires an explicit schema denylist.

## Audit and verification commands

The following commands or equivalent in-memory checks were run from the repository root. Aggregate scripts did not write participant data or emit complete rows.

```powershell
Get-ChildItem -LiteralPath . -Force
rg --files -uu -g '!.git/**'
Get-ChildItem -LiteralPath . -Filter 'AGENTS.md' -File -Force -Recurse
Get-ChildItem -LiteralPath . -Filter '*.csv' -File -Force -Recurse
git rev-parse --is-inside-work-tree

Get-FileHash -LiteralPath '<dataset-path>' -Algorithm SHA256
$rows = @(Import-Csv -LiteralPath '<dataset-path>')
$rows.Count
$rows[0].PSObject.Properties.Name.Count

Move-Item -LiteralPath '<root-dataset-path>' -Destination '<data/processed/dataset-path>'

Get-FileHash -LiteralPath '<data/processed/dataset-path>' -Algorithm SHA256
rg --files -uu
```

Read-only in-memory JavaScript checks used an RFC-4180-style parser and calculated per-column inferred types, blanks/placeholders, unique and frequent values, duplicates, identifier mappings, response distributions, one-hot sums, checkpoint transitions, gestational consistency, score/target thresholds, target combinations, deterministic proxies and OR consistency. A final PowerShell check resolved every local Markdown link and confirmed the absence of `src/`, `models/`, `artifacts/`, `reports/` and `.git` directories.
