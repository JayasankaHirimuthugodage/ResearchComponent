from __future__ import annotations

import json
import unittest

from training.contracts import PARTICIPANT_COLUMN, default_dataset_path
from training.dataset_validation import read_csv_records, sha256_file
from training.splitting import create_participant_split, participant_labels


class ParticipantSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.records, _ = read_csv_records(default_dataset_path())
        cls.dataset_hash = sha256_file(default_dataset_path())
        cls.manifest, cls.summary = create_participant_split(
            cls.records,
            dataset_sha256=cls.dataset_hash,
            search_iterations=5_000,
        )

    def test_split_has_expected_size_and_no_overlap(self) -> None:
        partition_counts: dict[str, int] = {}
        for row in self.manifest:
            partition_counts[row["partition"]] = partition_counts.get(row["partition"], 0) + 1

        self.assertEqual(405, partition_counts["development"])
        self.assertEqual(102, partition_counts["final_test"])
        self.assertEqual(0, self.summary["constraints"]["participant_overlap"])
        self.assertTrue(self.summary["constraints"]["all_participants_assigned_once"])
        self.assertEqual(
            507, len({row[PARTICIPANT_COLUMN] for row in self.manifest})
        )

    def test_split_preserves_required_categories_in_both_partitions(self) -> None:
        constraints = self.summary["constraints"]

        self.assertTrue(constraints["all_target_classes_in_both_partitions"])
        self.assertTrue(constraints["all_checkpoints_in_both_partitions"])
        self.assertTrue(constraints["all_label_combinations_in_both_partitions"])

    def test_split_is_deterministic(self) -> None:
        second_manifest, second_summary = create_participant_split(
            self.records,
            dataset_sha256=self.dataset_hash,
            search_iterations=5_000,
        )

        self.assertEqual(self.manifest, second_manifest)
        self.assertEqual(
            self.summary["split_fingerprint_sha256"],
            second_summary["split_fingerprint_sha256"],
        )

    def test_aggregate_summary_does_not_contain_manifest_identifiers(self) -> None:
        serialized = json.dumps(self.summary, sort_keys=True)

        self.assertNotIn(f'"{PARTICIPANT_COLUMN}"', serialized)

    def test_duplicate_participant_is_rejected(self) -> None:
        duplicated_records = [*self.records, self.records[0].copy()]

        with self.assertRaisesRegex(ValueError, "one validated modelling row"):
            participant_labels(duplicated_records)


if __name__ == "__main__":
    unittest.main()

