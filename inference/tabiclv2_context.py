"""Fail-closed, development-only context access for experimental TabICLv2.

This module intentionally does not know the project combined dataset path beyond
rejecting it.  A separately provisioned, access-controlled development-only
source is required before a real context can be constructed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from training.contracts import FEATURE_COLUMNS, PARTICIPANT_COLUMN

COGNITIVE_TARGET = "target_cognitive_concern"
EXPECTED_PARTICIPANTS = 500
EXPECTED_POSITIVES = 53
SOURCE_SCHEMA = "tabiclv2-cognitive-development-context-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class ContextSourceAuthorization:
    """Trusted configuration supplied by a future service deployment, not the CSV."""

    source_path: Path
    provenance_path: Path
    approved_source_sha256: str
    approved_membership_sha256: str
    approved_dataset_sha256: str
    approved_split_fingerprint_sha256: str
    forbidden_combined_source: Path


@dataclass(frozen=True)
class VerifiedCognitiveContext:
    """Sensitive development context retained only by its owning process."""

    feature_rows: np.ndarray = field(repr=False)
    cognitive_labels: np.ndarray = field(repr=False)
    source_sha256: str


class InMemoryContextCache:
    """Create one verified context/model per process; never serializes its value."""

    def __init__(self, factory: Callable[[], object]) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._value: object | None = None

    def get(self) -> object:
        with self._lock:
            if self._value is None:
                self._value = self._factory()
            return self._value


def _load_provenance(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("Context provenance must be a JSON object.")
    return value


def load_authorized_cognitive_context(
    authorization: ContextSourceAuthorization,
) -> VerifiedCognitiveContext:
    """Load only a separately provisioned development-only source after validation."""

    source = authorization.source_path.resolve()
    if source == authorization.forbidden_combined_source.resolve():
        raise ValueError("The combined development/test source is forbidden at runtime.")
    if not source.is_file() or not authorization.provenance_path.is_file():
        raise FileNotFoundError("Authorized development-only context source is unavailable.")
    source_hash = sha256_file(source)
    if source_hash != authorization.approved_source_sha256:
        raise ValueError("Development-only source checksum does not match authorization.")
    provenance = _load_provenance(authorization.provenance_path)
    expected = {
        "schema": SOURCE_SCHEMA,
        "source_sha256": authorization.approved_source_sha256,
        "membership_sha256": authorization.approved_membership_sha256,
        "dataset_sha256": authorization.approved_dataset_sha256,
        "split_fingerprint_sha256": authorization.approved_split_fingerprint_sha256,
        "participants": EXPECTED_PARTICIPANTS,
        "cognitive_positive_labels": EXPECTED_POSITIVES,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Development-only source provenance does not match authorization.")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [PARTICIPANT_COLUMN, *FEATURE_COLUMNS, COGNITIVE_TARGET]
        if reader.fieldnames != expected_fields:
            raise ValueError("Development-only context schema does not match the strict allowlist.")
        rows = list(reader)
    participants = [row[PARTICIPANT_COLUMN].strip() for row in rows]
    if len(rows) != EXPECTED_PARTICIPANTS or not all(participants) or len(set(participants)) != len(participants):
        raise ValueError("Development-only context participant contract is invalid.")
    membership = hashlib.sha256("\n".join(sorted(participants)).encode("utf-8")).hexdigest().upper()
    if membership != authorization.approved_membership_sha256:
        raise ValueError("Development-only source membership does not match authorization.")
    features = np.asarray([[row[name].strip() for name in FEATURE_COLUMNS] for row in rows], dtype=object)
    labels = np.asarray([row[COGNITIVE_TARGET].strip() for row in rows], dtype=object)
    if features.shape != (EXPECTED_PARTICIPANTS, len(FEATURE_COLUMNS)) or not np.isin(labels, ("0", "1")).all():
        raise ValueError("Development-only context values are invalid.")
    numeric_labels = labels.astype(np.int8)
    if int(numeric_labels.sum()) != EXPECTED_POSITIVES:
        raise ValueError("Cognitive positive-label count does not match authorization.")
    features.setflags(write=False)
    numeric_labels.setflags(write=False)
    return VerifiedCognitiveContext(features, numeric_labels, source_hash)
