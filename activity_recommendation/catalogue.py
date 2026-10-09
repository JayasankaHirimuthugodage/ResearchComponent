"""Load and validate an immutable activity-library JSON snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from .contracts import ActivityLibraryValidationError, ActivityRecord


DEFAULT_LIBRARY_PATH = Path(
    "research_resources/activity_library/Unified_CDC_Vroom_ActivityLibrary_v0.4.json"
)
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("schemas") / "activity-library-v0.4.schema.json"


@dataclass(frozen=True)
class ActivityCatalogue:
    library_name: str
    library_version: str
    library_sha256: str
    records: tuple[ActivityRecord, ...]

    @property
    def by_id(self) -> dict[str, ActivityRecord]:
        return {record.activity_id: record for record in self.records}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _format_schema_errors(validator: Draft202012Validator, document: object) -> str:
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))
    details = []
    for error in errors[:20]:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        details.append(f"{location}: {error.message}")
    if len(errors) > 20:
        details.append(f"... and {len(errors) - 20} more errors")
    return "; ".join(details)


def load_activity_catalogue(
    library_path: Path = DEFAULT_LIBRARY_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> ActivityCatalogue:
    """Load the canonical JSON snapshot and fail closed on contract violations."""

    try:
        library_bytes = library_path.read_bytes()
        document = json.loads(library_bytes.decode("utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActivityLibraryValidationError(f"Unable to load activity library: {exc}") from exc

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validation_message = _format_schema_errors(validator, document)
    if validation_message:
        raise ActivityLibraryValidationError(validation_message)

    activities = document["activities"]
    declared_total = document["counts"]["total_records"]
    if declared_total != len(activities):
        raise ActivityLibraryValidationError(
            f"Declared total {declared_total} does not match {len(activities)} records."
        )

    identifiers = [record["activity_id"] for record in activities]
    if len(identifiers) != len(set(identifiers)):
        raise ActivityLibraryValidationError("Activity IDs must be unique.")

    records = tuple(_to_activity_record(record) for record in activities)
    return ActivityCatalogue(
        library_name=document["library_name"],
        library_version=document["library_version"],
        library_sha256=_sha256(library_bytes),
        records=records,
    )


def _to_activity_record(record: dict[str, object]) -> ActivityRecord:
    return ActivityRecord(
        activity_id=str(record["activity_id"]),
        record_version=str(record["record_version"]),
        library_version=str(record["library_version"]),
        title=str(record["title"]),
        activity_summary=str(record["activity_summary"]),
        primary_skill=(str(record["primary_skill"]) if record.get("primary_skill") else None),
        domains=tuple(str(value) for value in record["domains"]),
        checkpoint_ids=tuple(str(value) for value in record["checkpoint_ids"]),
        source_org=str(record["source_org"]),
        source_url=str(record["source_url"]),
        source_page=(int(record["source_page"]) if record.get("source_page") is not None else None),
        source_tip_id=(str(record["source_tip_id"]) if record.get("source_tip_id") else None),
        status=str(record["status"]),
        safety_demands=dict(record["safety_demands"]),
        research_demo_eligible=bool(record["retrieval_eligible_for_internal_research_demo"]),
        caregiver_deployment_eligible=bool(record["retrieval_eligible_for_caregiver_deployment"]),
        raw_record=dict(record),
    )
