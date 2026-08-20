# Future service architecture plan

This is a Phase-0 design only. No production source tree, API, database, container, event broker, recommendation engine or LLM integration has been created.

## Bounded-service ownership

The future independently deployable bounded service is:

`slc-development-assessment-service`

Its eventual monorepo mapping is:

`services/slc-development-assessment-service/`

Do not move the current research repository there until the external monorepo's language, framework, deployment, identity, observability and data standards are known.

The service may own:

- SLC questionnaire/checkpoint definitions and assessment records;
- deterministic SLC scoring and score provenance;
- longitudinal T0/T1 linkage and follow-up status;
- validated ML feature preparation and model inference;
- domain predictions, explanations, thresholds and model versions;
- SLC safety/referral-rule outcomes within approved scope;
- the approved SLC activity catalogue and deterministic selection;
- constrained LLM personalization records;
- audit/version records for its own decisions and artifacts.

It must not own:

- physical or motor-development assessment;
- nutrition assessment, growth-faltering prediction, vaccination or dental/caries assessment;
- organization-wide child identity or authentication;
- another service's tables, migrations or model artifacts;
- cross-service joins or direct reads of another service's database.

## Ports-and-adapters boundaries

```text
API / transport adapters
        |
Application use cases and orchestration
        |
Domain concepts, scoring invariants and approved rules
        |
Ports (interfaces owned inward)
        |
Infrastructure adapters: persistence, ML runtime, LLM, events, external APIs
```

### Domain

Pure concepts such as checkpoint, response state, domain score, domain concern decision, safety outcome, approved activity, recommendation and version identifiers. Domain code contains no HTTP, database, broker, cloud, environment-variable or vendor SDK types.

Only clinically/research-approved deterministic rules belong here. Unvalidated thresholds remain configuration proposals or decision-log items, not executable defaults.

### Application

Use cases coordinate domain logic and ports, for example:

- record/validate an SLC assessment;
- assign and transition assessment status (`DRAFT`, `INCOMPLETE`, `COMPLETE`);
- calculate versioned scores;
- prepare a T0 feature request;
- obtain and persist three domain predictions;
- derive overall concern by OR;
- evaluate deterministic safety/referral rules;
- select approved activities;
- request optional constrained personalization;
- retrieve an audit history.

Application code depends on domain types and ports, not concrete database, HTTP, model or LLM implementations.

### Ports

Persistence ports:

```text
AssessmentRepository
PredictionRepository
RecommendationRepository
ActivityCatalogueRepository
AuditRepository
```

Capability ports:

```text
ChildContextPort
DevelopmentModelPort
ExplanationPort
PersonalizationPort
EventPublisherPort
ClockPort
IdentifierPort
```

`DevelopmentModelPort` accepts a versioned, validated internal feature object and returns three domain results plus technical explanation references. Scoring and recommendation code must never load a model file directly.

### Infrastructure

Adapters implement ports for a selected database, model runtime, LLM provider, message system or external child-context service. Environment configuration, transactions, retries, timeouts, SDK types and provider errors are contained here and mapped to stable application errors.

Possible future adapters include `PostgresAssessmentRepository` or `MongoAssessmentRepository`; these names illustrate portability, not a Phase-0 database choice.

### API

External request/response DTOs are versioned and mapped to internal commands/entities. ORM/ODM documents are never returned as API models. Database-specific object IDs must not be the only public identifiers.

## Future internal modules

Keep these as modules of one deployable service unless later evidence establishes an independent deployment/scaling/technology boundary:

- child-context integration;
- questionnaire/checkpoint management;
- assessment collection;
- scoring;
- longitudinal follow-up management;
- dataset and inference-schema validation;
- ML feature preparation;
- offline model training and evaluation;
- threshold and calibration management;
- production inference;
- prediction explainability;
- deterministic safety rules;
- activity catalogue and recommendation selection;
- LLM personalization;
- audit/version management;
- API contracts;
- persistence and external-service adapters.

Offline training may live in a clearly separated package or workflow but remains governed by the same feature/target specification. It is not exposed as a production endpoint.

## ML, rules and LLM orchestration

Every submitted assessment request must include exactly one response for each of the 15 items. An absent field is rejected as a validation error. `NOT_SURE` and `NOT_OBSERVED` are permitted explicit responses, but either makes the assessment `INCOMPLETE`. The caregiver may update those responses after observation.

```text
Validated T0 assessment
        |
        +--> deterministic completeness gate
        |       |
        |       +--> INCOMPLETE: no score or ML inference
        |       |                 safety/referral rules still run
        |       |
        |       +--> COMPLETE: continue to scoring and inference
        |
        +--> versioned ML inference --> three domain decisions
                                      |
                                      +--> overall OR result
                                      +--> technical explanation records
        |
        +--> deterministic safety/referral outcome
        |
        +--> approved activity selector
                    |
                    +--> optional constrained LLM wording
```

An incomplete assessment does not produce a developmental score, concern label, overall result or ML-dependent recommendation. It receives the approved incomplete-assessment message and remains updateable. Validated caregiver-concern, skill-loss/regression and other safety/referral rules still execute and may provide appropriate guidance. A rule can withhold prediction/recommendation when a safety/referral condition takes priority, but it does not rewrite a model output. The LLM can only personalize approved material and cannot change either layer.

## API contract considerations

Potential versioned endpoints:

```text
POST /api/v1/assessments
POST /api/v1/scores
POST /api/v1/predictions
GET  /api/v1/predictions/{prediction_id}/explanations
POST /api/v1/recommendations
GET  /health
GET  /ready
```

This list is conceptual. Before implementation, decide whether scoring/prediction/recommendation should be internal steps of one assessment workflow rather than separately writable public endpoints. Avoid exposing internal orchestration merely because it has a module boundary.

Contracts should provide:

- stable service-generated public IDs for child references, assessments, predictions and recommendations;
- explicit DTOs and enums with schema versions;
- required item fields, explicit `NOT_SURE`/`NOT_OBSERVED` enum values and assessment-status transitions;
- structured errors with stable codes and safe messages;
- correlation/request IDs that do not contain participant identifiers;
- idempotency keys for retried write operations;
- authentication and authorization at service boundaries;
- concurrency/version checks where updates are allowed;
- OpenAPI documentation and compatibility tests;
- backward-compatible additions or documented/versioned breaking changes.

Do not hard-code frontend URLs, service URLs, ports, credentials, model paths or secrets. Use environment-based configuration validated at startup.

## Synchronous versus event interactions

Keep scoring, model inference, safety evaluation and recommendation selection internal synchronous calls while they are part of one service transaction/use case.

Likely synchronous external calls:

- obtain minimal child/profile context needed for age/checkpoint routing, subject to availability and timeout policy;
- submit an assessment and receive the current result when user experience requires immediate feedback;
- retrieve a prediction/explanation/recommendation by stable ID.

Potential future published events:

- `SlcAssessmentCompleted`
- `SlcScoresCalculated`
- `SlcPredictionGenerated`
- `SlcSafetyFlagRaised`
- `SlcRecommendationGenerated`

Events are justified when other bounded services require asynchronous notification without accessing this service's database. They should carry stable identifiers and minimal non-sensitive summaries, not raw responses. Consumers must tolerate duplicates and versioned schemas. An outbox may later support reliable publication, but Phase 0 does not choose or implement a broker.

## Database portability

No production database is selected or initialized. Application use cases depend on repository ports. A database change should primarily affect infrastructure adapters, schemas, migrations, database configuration and database-specific integration tests.

It should not require rewriting scoring logic, ML feature rules, model inference contracts, explanation policy, safety rules, recommendation selection or LLM boundaries. Portability does not mean zero change: a new database can require new schemas, migrations, query implementations, transaction handling, concurrency controls and tests.

### Storage requirements by data area

| Data area | Likely requirements |
|---|---|
| Longitudinal assessments | Stable child/assessment IDs, T0/T1 links, ordering, uniqueness, idempotency and transactional writes |
| Questionnaire responses | Versioned questionnaire/item IDs, explicit response states including unscorable values, validation and immutable history |
| Domain scores | Score and rule versions, inputs/provenance, recomputation trace and immutability after finalization |
| Predictions | Three domain decisions, internal probabilities if retained, model/schema/threshold/calibration versions and artifact checksum |
| Explanations | Domain-specific technical evidence, parent-facing text version and access controls |
| Recommendations | Selected catalogue/activity versions, domain rationale, safety state and personalization provenance |
| Audit/version records | Append-oriented events, actor/service identity, correlation ID, timestamps, before/after references and retention policy |

### Relational versus document considerations

| Criterion | Relational store | Document store |
|---|---|---|
| Longitudinal integrity and cross-record constraints | Strong fit with keys, uniqueness and transactions | Requires deliberate modelling and application/schema validation |
| Versioned questionnaire response sets | Normalized rows or JSON columns; flexible hybrid possible | Natural document shape but version/query discipline is essential |
| Audit/reporting queries | Strong ad hoc joins inside this service boundary | May require denormalization or aggregation pipelines |
| Schema evolution | Explicit migrations and constraints | Flexible storage, but application-level migrations still required |
| Transactional assessment→score→prediction write | Mature multi-table transactions | Depends on product and document/transaction design |
| External monorepo standards | Must align with approved platform | Must align with approved platform |

Selection criteria should include external monorepo standards, operational expertise, managed-service availability, backup/restore, encryption, transaction needs, reporting/query patterns, data residency, cost and migration tooling. A relational database may be a natural candidate for longitudinal integrity, but no irreversible recommendation is made until those constraints are known.

Transactions stay within infrastructure adapters. Cross-service workflows use APIs/events and appropriate consistency patterns; they do not create cross-service database transactions or joins.

## Training versus inference boundary

Offline training reads an approved research snapshot, performs group-aware evaluation and emits a reviewed artifact bundle. Production inference loads only an approved immutable bundle through `DevelopmentModelPort` and validates:

- artifact checksum/signature;
- model, feature-schema, target-definition, calibration and threshold versions;
- runtime compatibility;
- exact request schema and allowed categories;
- readiness before serving traffic.

The service owns its model artifacts and associated validation. Another service must not import its training package or model files directly; consumers use documented APIs/events.

## Security and privacy design

- Minimize personal/context data at collection and at every adapter boundary.
- Encrypt data in transit and at rest using platform standards.
- Apply least-privilege service identities and role-based access.
- Keep secrets in the approved secret store, never files or images.
- Separate research/training access from production serving access.
- Do not log identifiers, raw answers, gestational age, birth weight, scores, predictions or LLM prompt content.
- Use redaction-aware structured logging and safe error mapping.
- Define retention, deletion, backup, consent and research-governance policies before production.
- Record model/rule/catalogue/prompt versions for audit without storing unnecessary prompt context.
- Send the LLM only minimal domain, checkpoint/age band and approved activity context; prevent provider training/retention where required by policy.
- Threat-model prompt injection through free text, catalogue tampering, artifact substitution, replayed writes and unauthorized explanation access.

## Health and deployment

`/health` should report process liveness without external calls. `/ready` may check required configuration, database connectivity and successful loading/validation of the approved model artifact and catalogue. Neither endpoint should expose secrets, paths, participant data or detailed dependency errors.

The future service should be independently containerized and deployable with service-owned migrations. Container/runtime, language and framework choices remain deferred to monorepo standards.

## Monorepo integration assessment

The proposed boundaries support limited-change integration because domain/application code does not depend on monorepo imports, a database product or transport framework. Expected integration work still includes:

- mapping repository layout and build tooling;
- adopting shared authentication, authorization, observability and configuration conventions;
- implementing child-context and event/API adapters;
- selecting the approved database and migration tooling;
- packaging the reviewed model artifact;
- allocating routes/ports and deployment manifests;
- contract, security and integration testing.

No direct imports from other services and no access to their databases are permitted. Shared libraries should be limited to stable platform concerns, not another service's domain entities.
