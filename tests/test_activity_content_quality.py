from __future__ import annotations

import unittest

from activity_recommendation.catalogue import load_activity_catalogue


CHECKPOINT_MONTHS = {
    "CP2M": 2,
    "CP4M": 4,
    "CP6M": 6,
    "CP9M": 9,
    "CP12M": 12,
    "CP15M": 15,
    "CP18M": 18,
    "CP24M": 24,
    "CP30M": 30,
    "CP36M": 36,
    "CP48M": 48,
    "CP60M": 60,
}


class ActivityContentQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogue = load_activity_catalogue()

    def record(self, activity_id: str):
        return self.catalogue.by_id[activity_id].raw_record

    def test_assigned_checkpoints_are_within_documented_source_age_ranges(self) -> None:
        for activity in self.catalogue.records:
            with self.subTest(activity_id=activity.activity_id):
                minimum = activity.raw_record["source_age_min_months"]
                maximum = activity.raw_record["source_age_max_months"]
                self.assertIsNotNone(minimum)
                self.assertIsNotNone(maximum)
                for checkpoint_id in activity.checkpoint_ids:
                    self.assertLessEqual(minimum, CHECKPOINT_MONTHS[checkpoint_id])
                    self.assertGreaterEqual(maximum, CHECKPOINT_MONTHS[checkpoint_id])

    def test_identified_material_mismatches_are_corrected(self) -> None:
        expected_terms = {
            "ACT-VROOM-TIP-00822": ("no mirror",),
            "ACT-VROOM-TIP-00272": ("paper", "drawing tool"),
            "ACT-CDC-36-COG-05": ("paper", "crayon"),
            "ACT-CDC-60-COG-01": ("puzzle",),
            "ACT-VROOM-TIP-00439": ("clean", "dressing or changing"),
        }
        for activity_id, terms in expected_terms.items():
            with self.subTest(activity_id=activity_id):
                text = self.record(activity_id)["caregiver_guide"]["what_you_need"].lower()
                for term in terms:
                    self.assertIn(term, text)

    def test_infant_corrections_do_not_demand_advanced_responses(self) -> None:
        body = self.record("ACT-VROOM-TIP-00073")
        shared_book = self.record("ACT-CDC-06-SOC-04")
        daily_care = self.record("ACT-CDC-02-COM-02")

        self.assertIn("there is no need for the child to copy", body["activity_summary"].lower())
        self.assertNotIn("sounds or pointing", shared_book["activity_summary"].lower())
        self.assertNotIn("disclosure", daily_care["activity_summary"].lower())
        self.assertNotIn(
            "apology", daily_care["caregiver_guide"]["supervision_and_safety"].lower()
        )

    def test_bat_it_preserves_adult_held_toy_interaction(self) -> None:
        activity = self.record("ACT-VROOM-TIP-00428")
        text = activity["caregiver_guide"]["what_to_do"].lower()
        safety = activity["caregiver_guide"]["supervision_and_safety"].lower()

        self.assertIn("hold", text)
        self.assertIn("toy", text)
        self.assertIn("hand movement or kick", text)
        self.assertIn("adult must hold the toy", safety)
        self.assertIn("never attach or suspend", safety)


if __name__ == "__main__":
    unittest.main()
