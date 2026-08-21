"""Aggregate validation for the privacy-sensitive processed modelling CSV."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from training.contracts import (
    ASSESSMENT_ID_COLUMN,
    BASELINE_CHECKPOINT_COLUMN,
    BASELINE_CHECKPOINT_ONE_HOT_COLUMNS,
    BINARY_VALUES,
    CHECKPOINT_IDS,
    DERIVED_TARGET_COLUMN,
    EXPECTED_DATASET_SHA256,
    EXPECTED_ROW_COUNT,
    EXPECTED_SOURCE_COLUMNS,
    FEATURE_COLUMNS,
    FOLLOWUP_CHECKPOINT_COLUMN,
    FOLLOWUP_CHECKPOINT_ONE_HOT_COLUMNS,
    ONE_HOT_GROUPS,
    PARTICIPANT_COLUMN,
    PREMATURITY_ONE_HOT_COLUMNS,
    PROHIBITED_FEATURE_COLUMNS,
    QUESTION_COLUMNS,
    SCORE_TARGET_PAIRS,
    SCOREABLE_RESPONSE_VALUES,
    TARGET_COLUMNS,
    repository_root,
)


UUID_SHAPE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
CHECKPOINT_MONTHS = {
    checkpoint: int(checkpoint.removeprefix("CP").removesuffix("M"))
    for checkpoint in CHECKPOINT_IDS
}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    count: int | None = None
    columns: tuple[str, ...] = ()


class DatasetValidationError(RuntimeError):
    """Raised when a validation report contains one or more errors."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv_records(path: Path) -> tuple[list[str], list[dict[str, str]], list[int]]:
    """Read a CSV without modifying it and return headers, records and row widths."""

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))

    if not rows:
        return [], [], []

    headers = rows[0]
    data_rows = [row for row in rows[1:] if row]
    widths = [len(row) for row in data_rows]
    records: list[dict[str, str]] = []
    for row in data_rows:
        padded = row[: len(headers)] + [""] * max(0, len(headers) - len(row))
        records.append(dict(zip(headers, padded, strict=True)))
    return headers, records, widths


def _distribution(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def validate_feature_contract(
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> list[ValidationIssue]:
    """Fail closed unless the feature contract is the exact approved ordered allowlist."""

    issues: list[ValidationIssue] = []
    provided = tuple(feature_columns)
    duplicates = tuple(
        column for column, count in Counter(provided).items() if count > 1
    )
    prohibited = tuple(
        column for column in provided if column in PROHIBITED_FEATURE_COLUMNS
    )
    missing = tuple(column for column in FEATURE_COLUMNS if column not in provided)
    unexpected = tuple(column for column in provided if column not in FEATURE_COLUMNS)

    if duplicates:
        issues.append(
            ValidationIssue(
                "error",
                "duplicate_feature_columns",
                "Feature contract contains duplicate columns.",
                len(duplicates),
                duplicates,
            )
        )
    if prohibited:
        issues.append(
            ValidationIssue(
                "error",
                "prohibited_feature_columns",
                "Feature contract contains metadata, redundant, future or outcome columns.",
                len(prohibited),
                prohibited,
            )
        )
    if missing:
        issues.append(
            ValidationIssue(
                "error",
                "missing_feature_columns",
                "Feature contract is missing approved primary features.",
                len(missing),
                missing,
            )
        )
    if unexpected:
        issues.append(
            ValidationIssue(
                "error",
                "unexpected_feature_columns",
                "Feature contract contains columns outside the approved allowlist.",
                len(unexpected),
                unexpected,
            )
        )
    if not issues and provided != FEATURE_COLUMNS:
        issues.append(
            ValidationIssue(
                "error",
                "feature_order_mismatch",
                "Feature contract order differs from the frozen primary schema.",
            )
        )
    return issues


def validate_dataset(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_DATASET_SHA256,
) -> dict[str, object]:
    """Validate the frozen modelling snapshot and return an aggregate-only report."""

    path = path.resolve()
    issues: list[ValidationIssue] = []
    if not path.is_file():
        raise FileNotFoundError(path)

    headers, records, widths = read_csv_records(path)
    row_count = len(records)
    file_hash = sha256_file(path)

    def add(
        severity: str,
        code: str,
        message: str,
        *,
        count: int | None = None,
        columns: Sequence[str] = (),
    ) -> None:
        issues.append(
            ValidationIssue(severity, code, message, count, tuple(columns))
        )

    if headers != list(EXPECTED_SOURCE_COLUMNS):
        missing_headers = tuple(
            column for column in EXPECTED_SOURCE_COLUMNS if column not in headers
        )
        unexpected_headers = tuple(
            column for column in headers if column not in EXPECTED_SOURCE_COLUMNS
        )
        add(
            "error",
            "source_schema_mismatch",
            "CSV headers or their order differ from the frozen 66-column schema.",
            columns=(*missing_headers, *unexpected_headers),
        )
    if len(headers) != len(set(headers)):
        add("error", "duplicate_headers", "CSV contains duplicate header names.")
    malformed_rows = sum(width != len(headers) for width in widths)
    if malformed_rows:
        add(
            "error",
            "malformed_row_width",
            "One or more CSV rows do not match the header width.",
            count=malformed_rows,
        )
    if row_count != EXPECTED_ROW_COUNT:
        add(
            "error",
            "row_count_mismatch",
            f"Expected {EXPECTED_ROW_COUNT} data rows.",
            count=row_count,
        )
    if file_hash != expected_sha256.upper():
        add(
            "error",
            "dataset_checksum_mismatch",
            "Dataset bytes differ from the audited Phase-0 snapshot.",
        )

    available = set(headers)
    expected_available = set(EXPECTED_SOURCE_COLUMNS).issubset(available)
    missingness = {
        column: sum(record.get(column, "").strip() == "" for record in records)
        for column in headers
    }

    feature_contract_issues = validate_feature_contract()
    issues.extend(feature_contract_issues)

    identifier_summary: dict[str, object] = {}
    duplicate_row_count = 0
    if records:
        duplicate_row_count = sum(
            count for count in Counter(tuple(record.get(h, "") for h in headers) for record in records).values() if count > 1
        )
        if duplicate_row_count:
            add(
                "error",
                "duplicate_rows",
                "Exact duplicate rows were found.",
                count=duplicate_row_count,
            )

    if {PARTICIPANT_COLUMN, ASSESSMENT_ID_COLUMN}.issubset(available):
        participants = [record[PARTICIPANT_COLUMN].strip() for record in records]
        assessment_ids = [record[ASSESSMENT_ID_COLUMN].strip() for record in records]
        blank_participants = sum(value == "" for value in participants)
        blank_assessments = sum(value == "" for value in assessment_ids)
        duplicate_participants = sum(
            count for value, count in Counter(participants).items() if value and count > 1
        )
        duplicate_assessments = sum(
            count for value, count in Counter(assessment_ids).items() if value and count > 1
        )
        non_uuid_assessments = sum(
            bool(value) and not UUID_SHAPE.fullmatch(value) for value in assessment_ids
        )
        identifier_summary = {
            "unique_participants": len({value for value in participants if value}),
            "blank_participants": blank_participants,
            "duplicate_participant_records": duplicate_participants,
            "unique_assessment_ids": len({value for value in assessment_ids if value}),
            "blank_assessment_ids": blank_assessments,
            "duplicate_assessment_records": duplicate_assessments,
            "non_uuid_shaped_assessment_ids": non_uuid_assessments,
        }
        for code, count, message in (
            ("blank_participant", blank_participants, "Participant number is blank."),
            (
                "duplicate_participant",
                duplicate_participants,
                "A participant appears in more than one modelling row.",
            ),
            ("blank_assessment_id", blank_assessments, "Assessment ID is blank."),
            (
                "duplicate_assessment_id",
                duplicate_assessments,
                "Assessment ID is duplicated.",
            ),
            (
                "assessment_id_shape",
                non_uuid_assessments,
                "Assessment ID does not have the audited UUID shape.",
            ),
        ):
            if count:
                add("error", code, message, count=count)

    questionnaire_summary: dict[str, object] = {}
    if set(QUESTION_COLUMNS).issubset(available):
        response_distributions: dict[str, dict[str, int]] = {}
        invalid_question_cells = 0
        for column in QUESTION_COLUMNS:
            values = [record[column].strip() for record in records]
            response_distributions[column] = _distribution(values)
            invalid_question_cells += sum(
                value not in SCOREABLE_RESPONSE_VALUES for value in values
            )
        questionnaire_summary = {
            "scoreable_cells": row_count * len(QUESTION_COLUMNS) - invalid_question_cells,
            "invalid_or_unscorable_cells": invalid_question_cells,
            "response_distributions": response_distributions,
        }
        if invalid_question_cells:
            add(
                "error",
                "unscorable_modelling_response",
                "The frozen modelling snapshot must contain only scoreable 0/1/2 baseline responses.",
                count=invalid_question_cells,
                columns=QUESTION_COLUMNS,
            )

    one_hot_summary: dict[str, object] = {}
    for group_name, columns in ONE_HOT_GROUPS.items():
        if not set(columns).issubset(available):
            continue
        non_binary = 0
        all_zero = 0
        multiple_active = 0
        exactly_one = 0
        never_active = []
        for column in columns:
            if all(record[column].strip() != "1" for record in records):
                never_active.append(column)
        for record in records:
            values = [record[column].strip() for column in columns]
            non_binary += any(value not in BINARY_VALUES for value in values)
            active = sum(value == "1" for value in values)
            all_zero += active == 0
            multiple_active += active > 1
            exactly_one += active == 1
        one_hot_summary[group_name] = {
            "members": list(columns),
            "non_binary_rows": non_binary,
            "exactly_one_active_rows": exactly_one,
            "all_zero_rows": all_zero,
            "multiple_active_rows": multiple_active,
            "never_active_columns": never_active,
        }
        if non_binary or all_zero or multiple_active:
            add(
                "error",
                f"invalid_{group_name}_one_hot",
                f"{group_name} one-hot group is not exactly-one binary.",
                count=non_binary + all_zero + multiple_active,
                columns=columns,
            )

    checkpoint_summary: dict[str, object] = {}
    checkpoint_columns = {
        BASELINE_CHECKPOINT_COLUMN,
        FOLLOWUP_CHECKPOINT_COLUMN,
        *BASELINE_CHECKPOINT_ONE_HOT_COLUMNS,
        *FOLLOWUP_CHECKPOINT_ONE_HOT_COLUMNS,
    }
    if checkpoint_columns.issubset(available):
        baseline_mismatches = 0
        followup_mismatches = 0
        invalid_checkpoint_ids = 0
        earlier_followups = 0
        for record in records:
            baseline = record[BASELINE_CHECKPOINT_COLUMN].strip()
            followup = record[FOLLOWUP_CHECKPOINT_COLUMN].strip()
            if baseline not in CHECKPOINT_MONTHS or followup not in CHECKPOINT_MONTHS:
                invalid_checkpoint_ids += 1
                continue
            baseline_expected = f"baseline_checkpoint_{baseline.lower()}"
            followup_expected = f"followup_checkpoint_{followup.lower()}"
            baseline_mismatches += record[baseline_expected].strip() != "1"
            followup_mismatches += record[followup_expected].strip() != "1"
            earlier_followups += CHECKPOINT_MONTHS[followup] < CHECKPOINT_MONTHS[baseline]
        checkpoint_summary = {
            "baseline_distribution": _distribution(
                record[BASELINE_CHECKPOINT_COLUMN].strip() for record in records
            ),
            "followup_distribution": _distribution(
                record[FOLLOWUP_CHECKPOINT_COLUMN].strip() for record in records
            ),
            "invalid_checkpoint_ids": invalid_checkpoint_ids,
            "baseline_id_one_hot_mismatches": baseline_mismatches,
            "followup_id_one_hot_mismatches": followup_mismatches,
            "earlier_followup_checkpoints": earlier_followups,
        }
        if invalid_checkpoint_ids or baseline_mismatches or followup_mismatches:
            add(
                "error",
                "checkpoint_encoding_mismatch",
                "Checkpoint IDs and one-hot representations are inconsistent.",
                count=invalid_checkpoint_ids + baseline_mismatches + followup_mismatches,
            )
        if earlier_followups:
            add(
                "error",
                "earlier_followup_checkpoint",
                "A follow-up checkpoint is earlier than its baseline checkpoint.",
                count=earlier_followups,
            )

    target_summary: dict[str, object] = {}
    score_columns = {score for score, _ in SCORE_TARGET_PAIRS}
    if {*TARGET_COLUMNS, DERIVED_TARGET_COLUMN, *score_columns}.issubset(available):
        target_distributions: dict[str, dict[str, object]] = {}
        invalid_target_cells = 0
        score_label_contradictions: dict[str, int] = {}
        invalid_score_cells = 0
        for target in (*TARGET_COLUMNS, DERIVED_TARGET_COLUMN):
            values = [record[target].strip() for record in records]
            invalid_target_cells += sum(value not in BINARY_VALUES for value in values)
            positives = sum(value == "1" for value in values)
            target_distributions[target] = {
                "negative": row_count - positives,
                "positive": positives,
                "positive_rate": _rate(positives, row_count),
            }
        for score_column, target_column in SCORE_TARGET_PAIRS:
            contradictions = 0
            for record in records:
                try:
                    score = int(record[score_column].strip())
                except ValueError:
                    invalid_score_cells += 1
                    continue
                if score < 0 or score > 100 or score % 10:
                    invalid_score_cells += 1
                    continue
                contradictions += int(record[target_column].strip()) != int(score <= 50)
            score_label_contradictions[target_column] = contradictions
        overall_or_contradictions = 0
        for record in records:
            if any(record[target].strip() not in BINARY_VALUES for target in TARGET_COLUMNS):
                continue
            expected_overall = max(int(record[target]) for target in TARGET_COLUMNS)
            if record[DERIVED_TARGET_COLUMN].strip() not in BINARY_VALUES:
                continue
            overall_or_contradictions += (
                int(record[DERIVED_TARGET_COLUMN]) != expected_overall
            )
        combinations = _distribution(
            "".join(record[target].strip() for target in TARGET_COLUMNS)
            for record in records
        )
        target_summary = {
            "distributions": target_distributions,
            "score_label_contradictions": score_label_contradictions,
            "overall_or_contradictions": overall_or_contradictions,
            "three_domain_label_combinations": combinations,
            "invalid_target_cells": invalid_target_cells,
            "invalid_score_cells": invalid_score_cells,
        }
        if invalid_target_cells:
            add(
                "error",
                "invalid_target_value",
                "One or more target values are not binary.",
                count=invalid_target_cells,
            )
        if invalid_score_cells:
            add(
                "error",
                "invalid_followup_score",
                "A follow-up score is outside 0..100 in 10-point increments.",
                count=invalid_score_cells,
            )
        contradiction_count = sum(score_label_contradictions.values())
        if contradiction_count:
            add(
                "error",
                "score_target_contradiction",
                "A domain target disagrees with the score <= 50 definition.",
                count=contradiction_count,
            )
        if overall_or_contradictions:
            add(
                "error",
                "overall_target_contradiction",
                "Overall concern disagrees with the OR of the domain targets.",
                count=overall_or_contradictions,
            )

    gestational_summary: dict[str, int] = {}
    gestational_columns = {
        *PREMATURITY_ONE_HOT_COLUMNS,
        "profile_gestational_weeks",
        "profile_gestational_weeks_missing",
    }
    if gestational_columns.issubset(available):
        contradictions = 0
        invalid_known_values = 0
        for record in records:
            yes = record["profile_premature_yes"].strip() == "1"
            no = record["profile_premature_no"].strip() == "1"
            not_sure = record["profile_premature_not_sure"].strip() == "1"
            weeks = record["profile_gestational_weeks"].strip()
            missing_flag = record["profile_gestational_weeks_missing"].strip()
            if yes:
                if not weeks or missing_flag != "0":
                    contradictions += 1
                else:
                    try:
                        week_value = int(weeks)
                    except ValueError:
                        invalid_known_values += 1
                    else:
                        invalid_known_values += not 20 <= week_value <= 36
            elif no or not_sure:
                contradictions += bool(weeks) or missing_flag != "1"
        gestational_summary = {
            "structural_missingness_contradictions": contradictions,
            "invalid_known_gestational_values": invalid_known_values,
        }
        if contradictions or invalid_known_values:
            add(
                "error",
                "gestational_consistency",
                "Prematurity, gestational age and missingness flag are inconsistent.",
                count=contradictions + invalid_known_values,
            )

    reporting_method_blanks = missingness.get("reporting_method", 0)
    if reporting_method_blanks:
        add(
            "warning",
            "missing_reporting_method",
            "Reporting-method provenance is blank for some records.",
            count=reporting_method_blanks,
            columns=("reporting_method",),
        )

    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    try:
        display_path = str(path.relative_to(repository_root()))
    except ValueError:
        display_path = path.name

    return {
        "valid": error_count == 0,
        "dataset": {
            "path": display_path.replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": file_hash,
            "expected_sha256_match": file_hash == expected_sha256.upper(),
            "row_count": row_count,
            "column_count": len(headers),
            "all_row_widths_match": malformed_rows == 0,
        },
        "feature_contract": {
            "features": list(FEATURE_COLUMNS),
            "targets": list(TARGET_COLUMNS),
            "derived_target": DERIVED_TARGET_COLUMN,
            "prohibited_feature_count": len(PROHIBITED_FEATURE_COLUMNS),
            "prohibited_overlap": sorted(
                set(FEATURE_COLUMNS).intersection(PROHIBITED_FEATURE_COLUMNS)
            ),
            "valid": not feature_contract_issues,
        },
        "missingness": missingness,
        "integrity": {
            "exact_duplicate_records": duplicate_row_count,
            **identifier_summary,
        },
        "questionnaire": questionnaire_summary,
        "one_hot_groups": one_hot_summary,
        "checkpoints": checkpoint_summary,
        "targets": target_summary,
        "gestational_age": gestational_summary,
        "summary": {
            "error_count": error_count,
            "warning_count": warning_count,
        },
        "issues": [asdict(issue) for issue in issues],
        "all_expected_columns_available": expected_available,
    }


def require_valid(report: dict[str, object]) -> None:
    """Raise without exposing rows when aggregate validation fails."""

    if not report.get("valid", False):
        summary = report.get("summary", {})
        raise DatasetValidationError(
            f"Dataset validation failed with {summary.get('error_count', 'unknown')} error(s)."
        )
