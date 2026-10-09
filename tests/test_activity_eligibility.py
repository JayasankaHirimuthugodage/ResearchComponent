from __future__ import annotations

import unittest

from activity_recommendation.catalogue import load_activity_catalogue
from activity_recommendation.contracts import (
    AbilityState,
    ActivityRecord,
    EligibilityRequest,
    RetrievalMode,
    SafetyContext,
)
from activity_recommendation.eligibility import filter_eligible_activities


def make_record(activity_id: str, safety_demands: dict[str, str]) -> ActivityRecord:
    return ActivityRecord(
        activity_id=activity_id,
        record_version="1.0",
        library_version="0.4.0",
        title="Test activity",
        activity_summary="A test activity.",
        primary_skill="turn-taking",
        domains=("communication",),
        checkpoint_ids=("CP2M",),
        source_org="CDC",
        source_url="https://example.test/activity",
        source_page=None,
        source_tip_id=None,
        status="draft_for_validation",
        safety_demands=safety_demands,
        research_demo_eligible=True,
        caregiver_deployment_eligible=False,
        raw_record={},
    )


def context(state: AbilityState = AbilityState.AVAILABLE) -> SafetyContext:
    return SafetyContext(state, state, state, state)


class ActivityEligibilityTests(unittest.TestCase):
    def test_canonical_records_filter_by_checkpoint_and_domain(self) -> None:
        catalogue = load_activity_catalogue()
        result = filter_eligible_activities(
            catalogue.records,
            EligibilityRequest("CP2M", "communication", context()),
        )
        self.assertTrue(result.eligible)
        self.assertTrue(
            all("CP2M" in record.checkpoint_ids for record in result.eligible)
        )
        self.assertTrue(
            all("communication" in record.domains for record in result.eligible)
        )

    def test_deployment_mode_withholds_every_current_record(self) -> None:
        catalogue = load_activity_catalogue()
        result = filter_eligible_activities(
            catalogue.records,
            EligibilityRequest(
                "CP2M",
                "communication",
                context(),
                RetrievalMode.CAREGIVER_DEPLOYMENT,
            ),
        )
        self.assertEqual((), result.eligible)

    def test_review_if_concern_is_withheld_for_concern(self) -> None:
        record = make_record(
            "ACT-TEST-1",
            {
                "hearing": "review_if_concern",
                "vision": "not_obviously_required",
                "fine_motor": "not_obviously_required",
                "gross_motor": "not_obviously_required",
            },
        )
        safety = SafetyContext(
            hearing=AbilityState.CONCERN_REPORTED,
            vision=AbilityState.AVAILABLE,
            fine_motor=AbilityState.AVAILABLE,
            gross_motor=AbilityState.AVAILABLE,
        )
        result = filter_eligible_activities(
            (record,), EligibilityRequest("CP2M", "communication", safety)
        )
        self.assertEqual((), result.eligible)
        self.assertIn("hearing_review_required_for_reported_concern", result.withheld[0].reasons)

    def test_review_if_concern_is_withheld_when_context_unknown(self) -> None:
        record = make_record(
            "ACT-TEST-2",
            {
                "hearing": "review_if_concern",
                "vision": "not_obviously_required",
                "fine_motor": "not_obviously_required",
                "gross_motor": "not_obviously_required",
            },
        )
        result = filter_eligible_activities(
            (record,), EligibilityRequest("CP2M", "communication", context(AbilityState.UNKNOWN))
        )
        self.assertEqual((), result.eligible)

    def test_not_established_requirement_allows_with_caution(self) -> None:
        record = make_record(
            "ACT-TEST-3",
            {
                "hearing": "not_established_as_essential_in_brief_descriptor",
                "vision": "not_obviously_required",
                "fine_motor": "not_obviously_required",
                "gross_motor": "not_obviously_required",
            },
        )
        result = filter_eligible_activities(
            (record,), EligibilityRequest("CP2M", "communication", context(AbilityState.UNKNOWN))
        )
        self.assertEqual((record,), result.eligible)
        self.assertEqual(4, len(result.cautions[record.activity_id]))

    def test_no_concern_reported_is_not_confirmed_ability(self) -> None:
        record = make_record(
            "ACT-TEST-NO-CONCERN",
            {
                "hearing": "required",
                "vision": "not_obviously_required",
                "fine_motor": "not_obviously_required",
                "gross_motor": "not_obviously_required",
            },
        )
        result = filter_eligible_activities(
            (record,),
            EligibilityRequest(
                "CP2M",
                "communication",
                context(AbilityState.NO_CONCERN_REPORTED),
            ),
        )
        self.assertEqual((), result.eligible)
        self.assertIn("hearing_required_but_not_confirmed", result.withheld[0].reasons)

    def test_no_concern_reported_allows_review_if_concern(self) -> None:
        record = make_record(
            "ACT-TEST-NO-CONCERN-REVIEW",
            {
                "hearing": "review_if_concern",
                "vision": "not_obviously_required",
                "fine_motor": "not_obviously_required",
                "gross_motor": "not_obviously_required",
            },
        )
        result = filter_eligible_activities(
            (record,),
            EligibilityRequest(
                "CP2M",
                "communication",
                context(AbilityState.NO_CONCERN_REPORTED),
            ),
        )
        self.assertEqual((record,), result.eligible)

    def test_required_ability_must_be_confirmed(self) -> None:
        record = make_record(
            "ACT-TEST-4",
            {
                "hearing": "required",
                "vision": "not_obviously_required",
                "fine_motor": "not_obviously_required",
                "gross_motor": "not_obviously_required",
            },
        )
        result = filter_eligible_activities(
            (record,), EligibilityRequest("CP2M", "communication", context(AbilityState.UNKNOWN))
        )
        self.assertEqual((), result.eligible)
        self.assertIn("hearing_required_but_not_confirmed", result.withheld[0].reasons)

    def test_invalid_request_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            filter_eligible_activities((), EligibilityRequest("CP3M", "communication", context()))
        with self.assertRaises(ValueError):
            filter_eligible_activities((), EligibilityRequest("CP2M", "physical", context()))


if __name__ == "__main__":
    unittest.main()
