# Unified CDC + Vroom Activity Library — v0.4 (research draft)

Generated: 2026-10-10. The JSON file is the authoritative library snapshot.

## Contents

- 180 CDC-based draft activity records retained from v0.3.
- 93 active Vroom research-candidate records.
- 3 Vroom records permanently excluded and recorded in `Vroom_Activity_Exclusion_Log_v0.4.csv`. Their IDs are reserved and must not be reused.
- 273 total active records.
- All 12 checkpoints and all three developmental domains remain covered. The minimum is 10 records in a checkpoint/domain combination.

## v0.4 review changes

- Preserved stable IDs and source references for every retained record.
- Corrected source-supported descriptions or conservative safety metadata for 23 Vroom records.
- Permanently excluded activities involving appliance, water/small-object, or meal-preparation hazards that the current four-dimensional safety contract cannot represent reliably.
- Kept every caregiver-deployment flag false. Source-selection permission is not treated as individual clinical review.

## Safety and retrieval boundaries

Eligibility filtering checks checkpoint, developmental domain and hearing, vision, fine-motor and gross-motor requirements before semantic ranking. Missing safety information is not evidence of safety. `review_if_concern` records are withheld when the corresponding concern is reported or the required context is unknown. Activities remain internal research-demo candidates only.

## Source and licensing limits

The five official Vroom source URLs remain in the JSON metadata and on each Vroom record. Seventeen records from the 3–4 years low-ink document retain a null `source_tip_id` because the page layout does not establish a reliable ID alignment. The official source must be consulted for complete instructions. Permission scope for adapted brief descriptors remains to be documented.

CDC records remain earlier research paraphrases and were not re-audited item by item during this Vroom-focused update.

## Generated companion files

- `Unified_CDC_Vroom_ActivityLibrary_v0.4.csv`: flat export generated from the JSON.
- `Unified_CDC_Vroom_Coverage_v0.4.csv`: checkpoint/domain counts generated from the JSON.
- `Unified_CDC_Vroom_Validation_v0.4.json`: structural and count checks generated from the JSON.
- `Vroom_Activity_Exclusion_Log_v0.4.csv`: permanent-exclusion history for the three removed records.

## Embeddings

Any v0.3 embedding matrix is stale for this snapshot because the JSON checksum, record count, ordered IDs and some descriptions changed. Rebuild the versioned v0.4 embedding artifacts with the pinned local model. This does not retrain or change the developmental prediction models.
