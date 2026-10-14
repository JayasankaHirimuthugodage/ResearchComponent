from __future__ import annotations

import unittest

from activity_recommendation.catalogue import load_activity_catalogue
from demo.activity_presentation import (
    caregiver_activity_content,
    reviewer_activity_provenance,
)


class ActivityPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.activities = load_activity_catalogue().records
        cls.activity = cls.activities[0]

    def test_caregiver_content_excludes_source_and_technical_metadata(self) -> None:
        for activity in self.activities:
            with self.subTest(activity_id=activity.activity_id):
                title, instructions = caregiver_activity_content(activity)
                combined = f"{title}\n{instructions}"

                self.assertEqual(activity.title, title)
                self.assertEqual(activity.activity_summary, instructions)
                self.assertNotIn(activity.source_org, combined)
                self.assertNotIn(activity.activity_id, combined)
                self.assertNotIn(activity.source_url, combined)

    def test_reviewer_provenance_preserves_source_metadata(self) -> None:
        provenance = reviewer_activity_provenance(self.activity)

        self.assertEqual(self.activity.activity_id, provenance["activity_id"])
        self.assertEqual(self.activity.source_org, provenance["source_org"])
        self.assertEqual(self.activity.source_url, provenance["source_url"])


if __name__ == "__main__":
    unittest.main()
