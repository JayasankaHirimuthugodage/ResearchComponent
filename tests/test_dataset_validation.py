from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from training.contracts import (
    ASSESSMENT_ID_COLUMN,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SOURCE_COLUMNS,
    FEATURE_COLUMNS,
    FOLLOWUP_SCORE_COLUMNS,
    PARTICIPANT_COLUMN,
    PROHIBITED_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    default_dataset_path,
)
from training.dataset_validation import (
    read_csv_records,
    sha256_file,
    validate_dataset,
    validate_feature_contract,
)


class DatasetValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_path = default_dataset_path()
        cls.headers, cls.records, _ = read_csv_records(cls.dataset_path)

    def _write_modified_dataset(self, mutate) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary_directory = tempfile.TemporaryDirectory()
        destination = Path(temporary_directory.name) / "modified.csv"
        records = [record.copy() for record in self.records]
        mutate(records)
        with destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(records)
        return temporary_directory, destination

    def _issue_codes(self, report: dict[str, object]) -> set[str]:
        return {issue["code"] for issue in report["issues"]}

    def test_audited_dataset_passes_with_only_known_provenance_warning(self) -> None:
        report = validate_dataset(self.dataset_path)

        self.assertTrue(report["valid"])
        self.assertEqual(EXPECTED_DATASET_SHA256, report["dataset"]["sha256"])
        self.assertEqual(507, report["dataset"]["row_count"])
        self.assertEqual(66, report["dataset"]["column_count"])
        self.assertEqual(7_605, report["questionnaire"]["scoreable_cells"])
        self.assertEqual(0, report["questionnaire"]["invalid_or_unscorable_cells"])
        self.assertEqual(4, report["missingness"]["reporting_method"])
        self.assertEqual(0, report["integrity"]["exact_duplicate_records"])
        self.assertEqual(0, report["summary"]["error_count"])
        self.assertEqual(1, report["summary"]["warning_count"])
        self.assertEqual({"missing_reporting_method"}, self._issue_codes(report))

    def test_feature_contract_rejects_followup_score_leakage(self) -> None:
        issues = validate_feature_contract((*FEATURE_COLUMNS, FOLLOWUP_SCORE_COLUMNS[0]))

        self.assertIn("prohibited_feature_columns", {issue.code for issue in issues})
        self.assertIn("unexpected_feature_columns", {issue.code for issue in issues})

    def test_feature_contract_partitions_all_source_columns(self) -> None:
        self.assertEqual(66, len(EXPECTED_SOURCE_COLUMNS))
        self.assertEqual(16, len(FEATURE_COLUMNS))
        self.assertEqual(50, len(PROHIBITED_FEATURE_COLUMNS))
        self.assertFalse(set(FEATURE_COLUMNS).intersection(PROHIBITED_FEATURE_COLUMNS))
        self.assertEqual(
            set(EXPECTED_SOURCE_COLUMNS),
            set(FEATURE_COLUMNS).union(PROHIBITED_FEATURE_COLUMNS),
        )

    def test_representative_identifier_future_context_and_target_columns_are_denied(self) -> None:
        representative_denied_columns = (
            PARTICIPANT_COLUMN,
            ASSESSMENT_ID_COLUMN,
            "baseline_checkpoint_cp2m",
            "followup_checkpoint_id",
            FOLLOWUP_SCORE_COLUMNS[0],
            "profile_sex_male",
            "profile_premature_yes",
            TARGET_COLUMNS[0],
        )
        for denied_column in representative_denied_columns:
            with self.subTest(column=denied_column):
                issues = validate_feature_contract((*FEATURE_COLUMNS, denied_column))
                self.assertIn(
                    "prohibited_feature_columns", {issue.code for issue in issues}
                )

    def test_unscorable_value_is_rejected_in_frozen_modelling_snapshot(self) -> None:
        temporary, path = self._write_modified_dataset(
            lambda records: records[0].__setitem__("communication_q1", "NOT_SURE")
        )
        self.addCleanup(temporary.cleanup)

        report = validate_dataset(path, expected_sha256=sha256_file(path))

        self.assertFalse(report["valid"])
        self.assertIn("unscorable_modelling_response", self._issue_codes(report))

    def test_score_target_contradiction_is_rejected(self) -> None:
        def mutate(records: list[dict[str, str]]) -> None:
            current = records[0]["target_communication_concern"]
            records[0]["target_communication_concern"] = "0" if current == "1" else "1"

        temporary, path = self._write_modified_dataset(mutate)
        self.addCleanup(temporary.cleanup)

        report = validate_dataset(path, expected_sha256=sha256_file(path))

        self.assertFalse(report["valid"])
        self.assertIn("score_target_contradiction", self._issue_codes(report))

    def test_multiple_active_one_hot_values_are_rejected(self) -> None:
        def mutate(records: list[dict[str, str]]) -> None:
            records[0]["profile_sex_male"] = "1"
            records[0]["profile_sex_female"] = "1"

        temporary, path = self._write_modified_dataset(mutate)
        self.addCleanup(temporary.cleanup)

        report = validate_dataset(path, expected_sha256=sha256_file(path))

        self.assertFalse(report["valid"])
        self.assertIn("invalid_sex_one_hot", self._issue_codes(report))

    def test_checksum_change_is_rejected(self) -> None:
        report = validate_dataset(self.dataset_path, expected_sha256="0" * 64)

        self.assertFalse(report["valid"])
        self.assertIn("dataset_checksum_mismatch", self._issue_codes(report))


if __name__ == "__main__":
    unittest.main()
