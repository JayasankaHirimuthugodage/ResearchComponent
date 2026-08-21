# Developmental questionnaire v3

This directory is the committed, versioned input catalogue used by the local development prediction demonstration. `questionnaire.json` contains the exact project-owner-supplied English question wording for 12 checkpoints, three domains and five questions per domain.

The catalogue maps the checkpoint-specific wording onto the frozen ML inputs `communication_q1`–`communication_q5`, `cognitive_q1`–`cognitive_q5` and `social_emotional_q1`–`social_emotional_q5`. The checkpoint itself maps to `baseline_checkpoint_id`.

`NOT_SURE` and `NOT_OBSERVED` are valid responses but are deliberately unscorable. If either is selected, the current ML contract marks the assessment `INCOMPLETE` and does not issue predictions. They must never be converted to zero.

The repository inclusion was authorized by the project owner on 2026-08-22. The supplied reference did not include an underlying third-party licence or an authoritative external document identifier. Those provenance fields remain explicitly `not supplied`; this repository does not infer them.

The checkpoint age ranges are catalogue metadata only. This demonstration asks the reviewer to select an already-determined checkpoint and does not implement or validate age-correction or routing rules.
