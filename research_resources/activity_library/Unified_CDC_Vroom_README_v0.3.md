# Unified CDC + Vroom Activity Library — v0.3 (research draft)

Generated: 2026-10-09.  **This is one library, not two.**

## Contents
- **180 CDC-based draft activity records**, carried forward from v0.2 (source page linked; activity-to-source matching has not been independently audited item by item).
- **96 Vroom source-verified activity records**: official published title/concept, stated age range and source page, with short original research descriptions.
- **3 older Vroom reference-only IDs excluded from the activity records**, identified in JSON metadata for reconciliation.
- **276 total records**. Records ≠ fully reviewed, ready-to-deploy interventions.

## Five official Vroom sources
- 0_1: https://assets.ctfassets.net/khbu829tujdl/LyHni4h8ESCUgSOaGoWKw/698282000d890a095c55f564a92911d2/Vroom_TipSheet_Age_0_1_ENG.pdf
- 1_2: https://assets.ctfassets.net/khbu829tujdl/2oHlhQZZBWU8yq8acmwUy6/1c24e1c4340e3ece406d76babc3b1f11/Vroom_TipSheet_Age_1-2_ENG.pdf
- 2_3: https://assets.ctfassets.net/khbu829tujdl/S3XP7JV0MSgu4qOecywSw/e2dbd11a380df10f0208a746f884f0d6/Vroom_TipSheet_Age_2_3_ENG.pdf
- 3_4: https://assets.ctfassets.net/khbu829tujdl/coGsAM5mvKmay0qmsEICq/2eda9bb17fb61a13e587751b3ce57512/Vroom_Tips-Age_3-4-English-Low_Ink.pdf
- 4_5: https://assets.ctfassets.net/khbu829tujdl/39B0taLOtWwKkKGMo42Eig/c2d95c37a3ba74297712fd30ab7ad9c6/Vroom_TipSheet_Age_4_5_ENG.pdf


## Key structural fields
- `activity_id`: stable internal ID; preserve when content is revised; assign a new ID for a different activity.
- `source_tip_id`: Vroom tip identifier where confirmed; empty for the older 3–4 years low-ink document when page layout makes ID alignment uncertain.
- `source_url`, `source_page`: original Vroom PDF and 1-indexed page, or CDC age-page URL.
- `activity_summary`: short new *research descriptor*, NOT a copy of the full Vroom Tip.
- `instruction_type`: distinguishes research paraphrases from source-verifiable brief descriptors and reference-only records.
- `checkpoint_ids`, `domains`: research mappings, not provided as these three-domain categories by CDC/Vroom.
- `safety_demands`: preliminary cues only (`review_if_concern` or `not_established_as_essential_in_brief_descriptor`). **Neither means clinically safe or unsafe.**
- `retrieval_eligible_for_internal_research_demo`: permits controlled testing of source-linked descriptors only.
- `retrieval_eligible_for_caregiver_deployment`: all FALSE pending item-level suitability and provenance sign-off.

## Why not copy Vroom's complete instructions here?
The actual Vroom activity instructions and Brainy Backgrounds reside in the linked official PDFs. These records are concise pointers that preserve original title, suggested age, and source. The supervisor/pediatrician's permission is user-reported, but its legal scope has not been independently reviewed; Vroom's website distinguishes verbatim use and adaptation. Do not mistake the research descriptor for Vroom's canonical text or silently expand it into invented guidance.

## Import and safety-filtering recommendations
1. Keep all legacy files intact; import v0.3 as a new snapshot.
2. Validate all IDs and foreign keys before switching the active snapshot.
3. Never overwrite old approved versions in historical recommendations.
4. Require exact checkpoint/domain compatibility and check all 4 safety dimensions. When sensory/motor concern exists and the relevant `safety_demands` is `review_if_concern`, withhold or require documented professional guidance. Do not assume unflagged means automatically accessible.
5. Do not call an activity 'reviewed by a clinician' solely because the source-selection policy was approved; obtain and record exactly what the pediatrician approved.
6. Rebuild semantic embeddings atomically after approved content changes; use a versioned index and evaluate source consistency.
7. For caregiver deployment, only mark explicit individual records eligible when content, source terms, age and suitability evidence are verified and approved.
8. Keep model feature contract (checkpoint + 15 answers), thresholds, locked test and trained model artifacts unchanged.

## Limitations
- CDC list was made earlier and not independently verified idea-by-idea against individual CDC tips in this merge.
- Vroom summaries are intentionally short and not full caregiver-facing tip text. Full source tip plus Vroom Brainy Background must be consulted for detail.
- Preliminary safety-demand labels were produced from concise descriptors and can miss requirements. **Manual suitability verification is required.**
- A source age interval does not guarantee appropriateness for every child in it; boundary checkpoints need careful review.
- This is suitable as a research catalogue and as a starting point for a safety-filtered semantic retrieval prototype, not as a validated clinical recommendation system.
