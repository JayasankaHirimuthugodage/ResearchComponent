# Execution and verification report

SYNTHETIC DEMONSTRATION OUTPUT - FOR PAEDIATRICIAN CONTENT AND OUTPUT REVIEW ONLY. NOT CLINICAL VALIDATION, DIAGNOSTIC EVIDENCE, OR PROOF OF PREDICTIVE ACCURACY.

Generated: 2026-10-18T21:13:51.759381+00:00 UTC

## Completion summary

- Synthetic assessments accounted for: 36
- Successfully predicted: 36 (verified prior genuine model outputs reused; no unnecessary ML rerun)
- Assessments with eligible returned activity recommendations: 24
- Concern-flagged assessments with no returned recommendation: 0
- Assessments with no flagged domain, so retrieval was not applicable: 12
- Total returned activity recommendations across cases and targeted domains: 198
- Failed assessments: 0

These counts describe synthetic test execution only. They are not prevalence, sensitivity, specificity, predictive accuracy, or clinical-effectiveness estimates.

## Authorized synthetic suitability profile

Fictional activity-suitability profile used solely for synthetic software-output demonstration.

All four exact safety-questionnaire fields used response code `NO_CONCERN_REPORTED` (label: `No concern noticed`; mapped state: `no_concern_reported`). The profile was recorded for every case and remained separate from the frozen 16-feature ML input.

## Execution verification

- Reused the 36 previously verified genuine XGBoost/TabICLv2 KV-cache prediction outputs; probabilities, decisions, thresholds, explanations, and OR decisions were not changed.
- Executed the existing `ActivityRecommendationService` in `caregiver_deployment` mode with the approved v0.5 catalogue, checksum-verified v0.5 embeddings, pinned local semantic encoder, checkpoint/domain gates, caregiver-deployment approval filter, and existing suitability rules.
- Preserved the maximum of three returned activities per flagged domain.
- No recommendation was manually selected, generated, or rewritten.
- The `NO_CONCERN_REPORTED` state was not promoted to `AVAILABLE`; therefore strict ability-required metadata remained conservatively blocking where applicable.
- No protected participant dataset or locked-test cohort was accessed. No model, threshold, calibration, KV-cache, questionnaire, catalogue, embedding artifact, or production inference behavior was modified.
- All reviewer identity, decision, date, signature, and seal fields remain blank.

## Aggregate withholding reasons exposed by the existing filter

- `checkpoint_not_eligible`: 13341
- `domain_not_eligible`: 9306

## File integrity

- Detailed JSON SHA-256: `26677FEA3CB355B85CC2894A8FBE0FF4C528D4C71A0C4D27C00537CB2E71DF66`
- Results CSV SHA-256: `C74458B337CC8EF00A58D858DD64FD7E31350A8371105BB9B59DC6185B640722`
- Review PDF SHA-256: `06FF18D05372FC238A02F0EDBA0DAAF20CCF849D43164BD5D5C67826F07D4F53`
