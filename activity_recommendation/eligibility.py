"""Deterministic eligibility and conservative safety filtering."""

from __future__ import annotations

from .contracts import (
    CHECKPOINT_IDS,
    DOMAIN_IDS,
    SAFETY_DIMENSIONS,
    AbilityState,
    ActivityRecord,
    EligibilityRequest,
    EligibilityResult,
    RetrievalMode,
    WithheldActivity,
)


REVIEW_IF_CONCERN = "review_if_concern"
NOT_ESTABLISHED = {
    "not_obviously_required",
    "not_established_as_essential_in_brief_descriptor",
}
REQUIRED = {"required", "ability_required"}
UNKNOWN = {"unknown", "not_assessed", "not_documented"}


def filter_eligible_activities(
    records: tuple[ActivityRecord, ...], request: EligibilityRequest
) -> EligibilityResult:
    if request.checkpoint_id not in CHECKPOINT_IDS:
        raise ValueError(f"Unknown checkpoint: {request.checkpoint_id}")
    if request.domain not in DOMAIN_IDS:
        raise ValueError(f"Unknown developmental domain: {request.domain}")

    eligible: list[ActivityRecord] = []
    withheld: list[WithheldActivity] = []
    cautions: dict[str, tuple[str, ...]] = {}
    ability_states = request.safety_context.as_mapping()

    for record in records:
        reasons: list[str] = []
        record_cautions: list[str] = []

        if request.checkpoint_id not in record.checkpoint_ids:
            reasons.append("checkpoint_not_eligible")
        if request.domain not in record.domains:
            reasons.append("domain_not_eligible")

        if request.mode is RetrievalMode.INTERNAL_RESEARCH_DEMO:
            if not record.research_demo_eligible:
                reasons.append("not_eligible_for_internal_research_demo")
        elif not record.caregiver_deployment_eligible:
            reasons.append("not_eligible_for_caregiver_deployment")

        for dimension in SAFETY_DIMENSIONS:
            demand = record.safety_demands.get(dimension)
            state = ability_states[dimension]
            if demand == REVIEW_IF_CONCERN:
                if state is AbilityState.CONCERN_REPORTED:
                    reasons.append(f"{dimension}_review_required_for_reported_concern")
                elif state is AbilityState.UNKNOWN:
                    reasons.append(f"{dimension}_review_required_but_context_unknown")
            elif demand in REQUIRED:
                if state is not AbilityState.AVAILABLE:
                    reasons.append(f"{dimension}_required_but_not_confirmed")
            elif demand in NOT_ESTABLISHED:
                if state is not AbilityState.AVAILABLE:
                    record_cautions.append(f"{dimension}_requirement_not_established")
            elif demand in UNKNOWN or demand is None:
                reasons.append(f"{dimension}_requirement_unknown")
            else:
                reasons.append(f"{dimension}_requirement_unrecognised")

        if reasons:
            withheld.append(WithheldActivity(record.activity_id, tuple(reasons)))
        else:
            eligible.append(record)
            if record_cautions:
                cautions[record.activity_id] = tuple(record_cautions)

    return EligibilityResult(tuple(eligible), tuple(withheld), cautions)
