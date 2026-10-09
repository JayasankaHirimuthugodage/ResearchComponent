# Personalised activity retrieval integration

## Scope

This research-demo workflow connects the existing frozen three-domain prediction service to the v0.4 CDC and Vroom activity catalogue. It does not alter model preprocessing, calibration, thresholds, questionnaire scoring or XAI. It is safety-filtered semantic retrieval, not complete retrieval-augmented generation (RAG), and no generative LLM is used.

## Sequence

1. Validate all 15 checkpoint-specific developmental answers.
2. Run the checksum-verified frozen models and existing XAI.
3. Use positive domain decisions as independent retrieval gates.
4. Within each gated domain, use exact questionnaire wording for `NOT_YET` items followed by `SOMETIMES` items to form a traceable skill query.
5. Collect the four separate activity-suitability observations.
6. Filter the activity catalogue by checkpoint, domain, research-demo status and safety requirements.
7. Rank only eligible records using the pinned local Sentence Transformer and NumPy cosine similarity.
8. Return up to three records for each domain. Fewer are returned when fewer are eligible.

The overall OR result is not a fourth model and is not a retrieval query. Model probabilities do not rank activities. XAI remains an explanation of model calculations and is not used as developmental skill evidence.

## Boundaries

- A zero-concern result produces no targeted activity retrieval.
- An incomplete assessment produces no ML-dependent retrieval.
- A flagged domain without a `NOT_YET` or `SOMETIMES` item produces no unsupported skill query.
- `NOT_SURE`, `NOT_OBSERVED` and missing developmental responses never support a skill claim.
- Missing safety answers block retrieval. Explicit `UNKNOWN` is handled conservatively.
- `NO_CONCERN_REPORTED` is not confirmed ability and cannot satisfy an activity marked as requiring that ability.
- Safety and approval restrictions are never relaxed to reach three results.
- All returned records remain internal research-demo candidates; none is marked caregiver-deployment eligible.

## Safety-question review provenance

The researcher reports that a pediatrician reviewed the four question wordings, response mapping, checkpoint examples and filtering rules on 2026-10-10. Reviewer identity is not stored and written evidence was not independently inspected by this repository workflow.

The four dimensions do not cover all hazards. Choking, water, stairs, falls, heat, tools, appliances, allergies, hygiene, material condition and supervision remain separate activity-level review matters.
