# Unified CDC + Vroom Activity Library — v0.4 (232-record research snapshot)

Generated: 2026-10-14. `Unified_CDC_Vroom_ActivityLibrary_v0.4.json` is the single authoritative activity-library snapshot.

## Contents

- 179 active CDC research-draft activities.
- 53 active Vroom research-candidate activities.
- 232 total active records across all 12 checkpoints and all three developmental domains.
- 44 reserved activity IDs in the JSON field `all_active_catalogue_exclusion_ids`; none appears in the active catalogue.
- All 232 records are eligible only for the controlled internal research demo.
- All caregiver-deployment flags remain false and all pediatrician validation states remain pending independent review.

The embedded 44-ID list is authoritative for preventing excluded activities from entering retrieval. It does not independently document the complete historical rationale for every exclusion. `Vroom_Activity_Exclusion_Log_v0.4.csv` is retained unchanged as the historical record for the three earlier permanent Vroom exclusions.

## Internal record order and presentation

Records use a deterministic internal order: earliest eligible checkpoint, developmental domain (`communication`, `cognitive`, `social_emotional`), source, case-insensitive title and activity ID. Source ordering is internal implementation metadata and is not caregiver-facing.

The main activity presentation contains only the activity title and approved instruction text. Source organization, source link, activity ID and similarity score remain available only in the research-reviewer trace.

## Content and review status

The caregiver-facing instructions, source references, activity order, IDs and activity-specific safety metadata are preserved from the approved `EarlyChildhood_232_ActivityLibrary.json` handoff. They must not be rewritten during integration or retrieval.

The content is intended for controlled internal research evaluation and subsequent pediatrician review. It is not clinically validated and must not be enabled for caregiver deployment. Source-use or source-selection permission is not individual clinical approval.

## Safety and retrieval boundaries

Eligibility filtering checks checkpoint, developmental domain, internal-research status, and hearing, vision, fine-motor and gross-motor requirements before semantic ranking. Missing or unrecognized requirements fail closed. Safety and approval restrictions are not relaxed to obtain a requested result count.

The four suitability dimensions are not a complete safety assessment. Additional hazard tags and review notes are retained as activity metadata, but they are not treated as proof that choking, food/allergy, water, falls, tools, appliances, hygiene, public-space, privacy or supervision risks have been resolved.

## Source and fidelity limits

Every active record retains an official source URL. The supplied page-11 references for `ACT-VROOM-TIP-00189` (Vroom tip #189, “Did You Hear That?”) and `ACT-VROOM-TIP-00021` (Vroom tip #21, “Share a Stare”) were checked against the official Vroom 0–1 tip sheet on 2026-10-14. The handoff does not claim a complete independent source-fidelity, licensing or clinical audit.

## Generated companion files

- `Vroom_Activity_Exclusion_Log_v0.4.csv`: unchanged historical record for three earlier Vroom exclusions.

## Embeddings

The active embedding artifacts must contain exactly 232 rows in the same order as the JSON records. They use the pinned local `sentence-transformers/all-MiniLM-L6-v2` model and are checksum-bound to this catalogue. Embeddings from the earlier 273-record snapshot are stale and must fail validation.

Rebuilding activity embeddings does not retrain or change the frozen developmental prediction models, calibration, thresholds, questionnaire, XAI or evaluation artifacts.
