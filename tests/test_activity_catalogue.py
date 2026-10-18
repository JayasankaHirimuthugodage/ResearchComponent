from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from activity_recommendation.catalogue import (
    CAREGIVER_APPROVED_LIBRARY_PATH,
    CAREGIVER_APPROVED_SCHEMA_PATH,
    DEFAULT_LIBRARY_PATH,
    activity_order_key,
    load_activity_catalogue,
)
from activity_recommendation.contracts import (
    CHECKPOINT_IDS,
    DOMAIN_IDS,
    AbilityState,
    SAFETY_DIMENSIONS,
    ActivityLibraryValidationError,
    EligibilityRequest,
    RetrievalMode,
    SafetyContext,
)
from activity_recommendation.embeddings import load_embedding_artifacts
from activity_recommendation.eligibility import filter_eligible_activities


class ActivityCatalogueTests(unittest.TestCase):
    def test_versioned_caregiver_approved_successor_covers_all_active_records(self) -> None:
        catalogue = load_activity_catalogue(
            CAREGIVER_APPROVED_LIBRARY_PATH,
            CAREGIVER_APPROVED_SCHEMA_PATH,
        )
        self.assertEqual("0.5.0", catalogue.library_version)
        self.assertEqual(232, len(catalogue.records))
        self.assertTrue(all(record.caregiver_deployment_eligible for record in catalogue.records))
        document = json.loads(CAREGIVER_APPROVED_LIBRARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(232, document["counts"]["records_eligible_for_caregiver_deployment"])
        self.assertEqual("Pediatrician", document["caregiver_deployment_approval"]["approver_role"])
        self.assertEqual("2026-10-18", document["caregiver_deployment_approval"]["approval_date"])

    def test_approved_successor_has_a_matching_embedding_manifest_and_caregiver_filter(self) -> None:
        catalogue = load_activity_catalogue(
            CAREGIVER_APPROVED_LIBRARY_PATH,
            CAREGIVER_APPROVED_SCHEMA_PATH,
        )
        artifacts = load_embedding_artifacts(catalogue, Path("artifacts/activity-embeddings/0.5.0"))
        self.assertEqual(232, len(artifacts.activity_ids))
        eligible = filter_eligible_activities(
            catalogue.records,
            EligibilityRequest(
                "CP2M",
                "communication",
                SafetyContext(*(AbilityState.AVAILABLE,) * 4),
                RetrievalMode.CAREGIVER_DEPLOYMENT,
            ),
        )
        self.assertTrue(eligible.eligible)
        self.assertTrue(all(item.caregiver_deployment_eligible for item in eligible.eligible))

    def test_canonical_library_loads_with_expected_contract(self) -> None:
        catalogue = load_activity_catalogue()

        self.assertEqual("0.4.0", catalogue.library_version)
        self.assertEqual(232, len(catalogue.records))
        self.assertEqual(232, len({record.activity_id for record in catalogue.records}))
        self.assertEqual(set(CHECKPOINT_IDS), {cp for record in catalogue.records for cp in record.checkpoint_ids})
        self.assertEqual(set(DOMAIN_IDS), {domain for record in catalogue.records for domain in record.domains})
        for record in catalogue.records:
            self.assertEqual(set(SAFETY_DIMENSIONS), set(record.safety_demands))
            self.assertTrue(record.research_demo_eligible)
            self.assertFalse(record.caregiver_deployment_eligible)
            self.assertTrue(record.source_url)
            self.assertTrue(record.record_version)

        document = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        excluded = set(document["all_active_catalogue_exclusion_ids"])
        self.assertEqual(44, len(excluded))
        self.assertTrue(excluded.isdisjoint(catalogue.by_id))
        self.assertEqual(
            {"CDC": 179, "Vroom": 53},
            {
                source: sum(record.source_org == source for record in catalogue.records)
                for source in ("CDC", "Vroom")
            },
        )
        self.assertTrue(all(record.research_demo_eligible for record in catalogue.records))
        self.assertTrue(
            all(not record.caregiver_deployment_eligible for record in catalogue.records)
        )
        self.assertEqual(
            document["activities"],
            sorted(document["activities"], key=activity_order_key),
        )

    def test_duplicate_activity_id_is_rejected(self) -> None:
        document = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        document["activities"][1]["activity_id"] = document["activities"][0]["activity_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ActivityLibraryValidationError, "unique"):
                load_activity_catalogue(path)

    def test_missing_safety_dimension_is_rejected(self) -> None:
        document = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        del document["activities"][0]["safety_demands"]["hearing"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ActivityLibraryValidationError):
                load_activity_catalogue(path)

    def test_unknown_domain_is_rejected(self) -> None:
        document = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        document["activities"][0]["domains"] = ["physical"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ActivityLibraryValidationError):
                load_activity_catalogue(path)

    def test_invalid_source_url_is_rejected(self) -> None:
        document = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        document["activities"][0]["source_url"] = "not a URL"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ActivityLibraryValidationError):
                load_activity_catalogue(path)

    def test_active_excluded_overlap_is_rejected(self) -> None:
        document = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        document["all_active_catalogue_exclusion_ids"][0] = document["activities"][0][
            "activity_id"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ActivityLibraryValidationError, "must not appear"):
                load_activity_catalogue(path)

    def test_declared_source_counts_are_enforced(self) -> None:
        document = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        document["activities"][0]["source_org"] = "Vroom"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ActivityLibraryValidationError, "source counts"):
                load_activity_catalogue(path)

    def test_out_of_order_activities_are_rejected(self) -> None:
        document = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        document["activities"][0], document["activities"][1] = (
            document["activities"][1],
            document["activities"][0],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ActivityLibraryValidationError, "approved.*order"):
                load_activity_catalogue(path)


if __name__ == "__main__":
    unittest.main()
