"""Fail-closed loading of the frozen development partition for modelling."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from training.contracts import (
    BASELINE_CHECKPOINT_COLUMN,
    BINARY_VALUES,
    CHECKPOINT_IDS,
    EXPECTED_DEVELOPMENT_PARTICIPANT_COUNT,
    EXPECTED_FINAL_TEST_PARTICIPANT_COUNT,
    EXPECTED_SPLIT_FINGERPRINT_SHA256,
    FEATURE_COLUMNS,
    PARTICIPANT_COLUMN,
    QUESTION_COLUMNS,
    SCOREABLE_RESPONSE_VALUES,
    TARGET_COLUMNS,
    default_dataset_path,
    repository_root,
)
from training.data.dataset_validation import (
    read_csv_records,
    require_valid,
    sha256_file,
    validate_dataset,
)


DEVELOPMENT_PARTITION = "development"
FINAL_TEST_PARTITION = "final_test"
ALLOWED_PARTITIONS = frozenset({DEVELOPMENT_PARTITION, FINAL_TEST_PARTITION})


@dataclass(frozen=True)
class ModelingRecord:
    """One participant represented only by approved inputs and three targets."""

    participant_key: str
    feature_values: tuple[str, ...]
    target_values: tuple[int, int, int]


@dataclass(frozen=True)
class DevelopmentDataset:
    """Development-only records plus non-identifying provenance metadata."""

    records: tuple[ModelingRecord, ...]
    dataset_sha256: str
    split_fingerprint_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class FinalTestDataset:
    """Locked final-test records plus non-identifying provenance metadata."""

    records: tuple[ModelingRecord, ...]
    dataset_sha256: str
    split_fingerprint_sha256: str
    manifest_sha256: str


def default_manifest_path() -> Path:
    return repository_root() / "artifacts" / "splits" / "split_manifest.csv"


def default_split_summary_path() -> Path:
    return repository_root() / "artifacts" / "splits" / "split_summary.json"


def _split_fingerprint(manifest: Sequence[Mapping[str, str]]) -> str:
    source = "\n".join(
        f"{row[PARTICIPANT_COLUMN]}:{row['partition']}" for row in manifest
    ).encode("utf-8")
    return hashlib.sha256(source).hexdigest().upper()


def read_split_manifest(path: Path) -> list[dict[str, str]]:
    """Read and validate the private split manifest without logging identifiers."""

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != [PARTICIPANT_COLUMN, "partition"]:
            raise ValueError("Split manifest schema does not match the required two columns.")
        rows = [
            {
                PARTICIPANT_COLUMN: row[PARTICIPANT_COLUMN].strip(),
                "partition": row["partition"].strip(),
            }
            for row in reader
        ]

    participants = [row[PARTICIPANT_COLUMN] for row in rows]
    if not rows or any(not participant for participant in participants):
        raise ValueError("Split manifest is empty or contains a blank participant key.")
    if len(participants) != len(set(participants)):
        raise ValueError("Split manifest must assign every participant exactly once.")
    if any(row["partition"] not in ALLOWED_PARTITIONS for row in rows):
        raise ValueError("Split manifest contains an unsupported partition value.")
    if not all(any(row["partition"] == value for row in rows) for value in ALLOWED_PARTITIONS):
        raise ValueError("Split manifest must contain both development and final-test partitions.")
    return rows


def to_modeling_record(record: Mapping[str, str]) -> ModelingRecord:
    """Project a validated source row onto the strict model allowlist."""

    participant = record.get(PARTICIPANT_COLUMN, "").strip()
    checkpoint = record.get(BASELINE_CHECKPOINT_COLUMN, "").strip()
    question_values = tuple(record.get(column, "").strip() for column in QUESTION_COLUMNS)
    target_values = tuple(record.get(column, "").strip() for column in TARGET_COLUMNS)

    if not participant:
        raise ValueError("A participant key is required outside estimator inputs.")
    if checkpoint not in CHECKPOINT_IDS:
        raise ValueError("A baseline checkpoint is outside the frozen feature schema.")
    if any(value not in SCOREABLE_RESPONSE_VALUES for value in question_values):
        raise ValueError("All 15 model inputs must be scoreable values in {0,1,2}.")
    if any(value not in BINARY_VALUES for value in target_values):
        raise ValueError("All three development targets must be binary.")

    features = (checkpoint, *question_values)
    if len(features) != len(FEATURE_COLUMNS):
        raise RuntimeError("Internal feature projection does not match the frozen schema.")
    return ModelingRecord(
        participant_key=participant,
        feature_values=features,
        target_values=tuple(int(value) for value in target_values),
    )


def select_development_records(
    source_records: Sequence[Mapping[str, str]],
    manifest: Sequence[Mapping[str, str]],
) -> tuple[ModelingRecord, ...]:
    """Return only development rows and reject incomplete or overlapping assignments."""

    source_participants = [record.get(PARTICIPANT_COLUMN, "").strip() for record in source_records]
    if len(source_participants) != len(set(source_participants)):
        raise ValueError("The modelling snapshot must have one row per participant.")

    partition_by_participant = {
        row[PARTICIPANT_COLUMN]: row["partition"] for row in manifest
    }
    if set(source_participants) != set(partition_by_participant):
        raise ValueError("Dataset and split manifest participant sets do not match.")

    selected = tuple(
        to_modeling_record(record)
        for record in source_records
        if partition_by_participant[record[PARTICIPANT_COLUMN].strip()]
        == DEVELOPMENT_PARTITION
    )
    if not selected:
        raise ValueError("The development partition is empty.")
    return selected


def select_final_test_records(
    source_records: Sequence[Mapping[str, str]],
    manifest: Sequence[Mapping[str, str]],
) -> tuple[ModelingRecord, ...]:
    """Return only locked final-test rows after verifying exact manifest coverage."""

    source_participants = [record.get(PARTICIPANT_COLUMN, "").strip() for record in source_records]
    if len(source_participants) != len(set(source_participants)):
        raise ValueError("The modelling snapshot must have one row per participant.")
    partition_by_participant = {
        row[PARTICIPANT_COLUMN]: row["partition"] for row in manifest
    }
    if set(source_participants) != set(partition_by_participant):
        raise ValueError("Dataset and split manifest participant sets do not match.")
    selected = tuple(
        to_modeling_record(record)
        for record in source_records
        if partition_by_participant[record[PARTICIPANT_COLUMN].strip()]
        == FINAL_TEST_PARTITION
    )
    if not selected:
        raise ValueError("The final-test partition is empty.")
    return selected


def load_development_dataset(
    dataset_path: Path | None = None,
    manifest_path: Path | None = None,
    summary_path: Path | None = None,
) -> DevelopmentDataset:
    """Load the audited development partition; never return final-test rows."""

    dataset_path = (dataset_path or default_dataset_path()).resolve()
    manifest_path = (manifest_path or default_manifest_path()).resolve()
    summary_path = (summary_path or default_split_summary_path()).resolve()

    validation_report = validate_dataset(dataset_path)
    require_valid(validation_report)
    dataset_sha256 = str(validation_report["dataset"]["sha256"])

    with summary_path.open("r", encoding="utf-8") as source:
        summary = json.load(source)
    if not isinstance(summary, dict):
        raise ValueError("Split summary must be a JSON object.")

    manifest = read_split_manifest(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    if summary.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Split summary does not reference the audited dataset checksum.")
    if summary.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Private split manifest checksum does not match the split summary.")
    split_fingerprint = _split_fingerprint(manifest)
    if summary.get("split_fingerprint_sha256") != split_fingerprint:
        raise ValueError("Split allocation fingerprint does not match the split summary.")
    if split_fingerprint != EXPECTED_SPLIT_FINGERPRINT_SHA256:
        raise ValueError("Split allocation differs from the frozen development/test boundary.")

    _, source_records, _ = read_csv_records(dataset_path)
    records = select_development_records(source_records, manifest)
    expected_development_count = summary.get("development", {}).get("participants")
    expected_final_test_count = summary.get("final_test", {}).get("participants")
    if (
        expected_development_count != EXPECTED_DEVELOPMENT_PARTICIPANT_COUNT
        or expected_development_count != len(records)
        or expected_final_test_count != EXPECTED_FINAL_TEST_PARTICIPANT_COUNT
    ):
        raise ValueError("Frozen partition counts do not match the aggregate split summary.")

    return DevelopmentDataset(
        records=records,
        dataset_sha256=dataset_sha256,
        split_fingerprint_sha256=split_fingerprint,
        manifest_sha256=manifest_sha256,
    )


def load_final_test_dataset(
    dataset_path: Path | None = None,
    manifest_path: Path | None = None,
    summary_path: Path | None = None,
) -> FinalTestDataset:
    """Load the locked final-test partition for the authorized final evaluation."""

    dataset_path = (dataset_path or default_dataset_path()).resolve()
    manifest_path = (manifest_path or default_manifest_path()).resolve()
    summary_path = (summary_path or default_split_summary_path()).resolve()

    validation_report = validate_dataset(dataset_path)
    require_valid(validation_report)
    dataset_sha256 = str(validation_report["dataset"]["sha256"])
    with summary_path.open("r", encoding="utf-8") as source:
        summary = json.load(source)
    if not isinstance(summary, dict):
        raise ValueError("Split summary must be a JSON object.")

    manifest = read_split_manifest(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    split_fingerprint = _split_fingerprint(manifest)
    if summary.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Split summary does not reference the audited dataset checksum.")
    if summary.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Private split manifest checksum does not match the split summary.")
    if summary.get("split_fingerprint_sha256") != split_fingerprint:
        raise ValueError("Split allocation fingerprint does not match the split summary.")
    if split_fingerprint != EXPECTED_SPLIT_FINGERPRINT_SHA256:
        raise ValueError("Split allocation differs from the frozen development/test boundary.")

    _, source_records, _ = read_csv_records(dataset_path)
    records = select_final_test_records(source_records, manifest)
    if (
        summary.get("development", {}).get("participants")
        != EXPECTED_DEVELOPMENT_PARTICIPANT_COUNT
        or summary.get("final_test", {}).get("participants")
        != EXPECTED_FINAL_TEST_PARTICIPANT_COUNT
        or len(records) != EXPECTED_FINAL_TEST_PARTICIPANT_COUNT
    ):
        raise ValueError("Frozen partition counts do not match the aggregate split summary.")

    return FinalTestDataset(
        records=records,
        dataset_sha256=dataset_sha256,
        split_fingerprint_sha256=split_fingerprint,
        manifest_sha256=manifest_sha256,
    )
