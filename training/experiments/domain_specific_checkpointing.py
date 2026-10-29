"""Fail-closed, privacy-minimised checkpoints for domain-specific experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np


SCHEMA = "domain-specific-training-checkpoint-v1"
FORBIDDEN_KEYS = {
    "participant_number", "participant_id", "participant_key", "assessment_id",
    "responses", "questionnaire_responses", "raw_rows", "feature_matrix",
    "target_values", "locked_test", "final_test",
}


class CheckpointError(RuntimeError):
    """Raised when persisted progress cannot be trusted."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _validate_privacy(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_KEYS or normalized.endswith("_participant_id"):
                raise CheckpointError(f"Prohibited checkpoint field at {path}.{key}")
            _validate_privacy(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_privacy(child, f"{path}[{index}]")


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _atomic_bytes(path: Path, content: bytes, *, overwrite: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    if temporary.exists():
        raise CheckpointError(f"Abandoned temporary checkpoint requires review: {temporary}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    expected = sha256_bytes(content)
    try:
        with temporary.open("xb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        if sha256_file(temporary) != expected:
            raise CheckpointError(f"Temporary checkpoint verification failed: {temporary}")
        os.replace(temporary, path)
        if sha256_file(path) != expected:
            raise CheckpointError(f"Checkpoint verification failed after replacement: {path}")
    finally:
        if temporary.exists():
            temporary.unlink()
    return expected


def atomic_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    _validate_privacy(value)
    return _atomic_bytes(path, canonical_bytes(value), overwrite=overwrite)


def load_json(path: Path, *, expected_sha256: str | None = None) -> Any:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint is missing: {path}")
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise CheckpointError(f"Checkpoint checksum mismatch: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CheckpointError(f"Checkpoint JSON is invalid: {path}") from error
    _validate_privacy(value)
    return value


def write_result(directory: Path, probabilities: np.ndarray, metadata: Mapping[str, Any]) -> None:
    """Write one completed fold and its integrity manifest exactly once."""
    _validate_privacy(metadata)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise CheckpointError("Fold probabilities must be a finite one-dimensional probability vector.")
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / "result.npy"
    manifest = directory / "manifest.json"
    if result.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite completed fold: {directory}")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
        scratch = Path(handle.name)
    try:
        np.save(scratch, probabilities, allow_pickle=False)
        content = scratch.read_bytes()
    finally:
        scratch.unlink(missing_ok=True)
    result_hash = _atomic_bytes(result, content)
    atomic_json(manifest, {"schema": SCHEMA, "kind": "fold", **dict(metadata),
                           "prediction_count": int(probabilities.size), "result_sha256": result_hash})


def load_result(directory: Path, expected: Mapping[str, Any]) -> np.ndarray | None:
    result, manifest = directory / "result.npy", directory / "manifest.json"
    if not result.exists() and not manifest.exists():
        return None
    if not result.is_file() or not manifest.is_file():
        raise CheckpointError(f"Incomplete fold checkpoint: {directory}")
    metadata = load_json(manifest)
    if metadata.get("schema") != SCHEMA or metadata.get("kind") != "fold":
        raise CheckpointError(f"Unsupported fold checkpoint schema: {directory}")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise CheckpointError(f"Fold checkpoint {key} mismatch: {directory}")
    if sha256_file(result) != metadata.get("result_sha256"):
        raise CheckpointError(f"Fold result checksum mismatch: {directory}")
    try:
        probabilities = np.load(result, allow_pickle=False)
    except Exception as error:
        raise CheckpointError(f"Fold result is unreadable: {directory}") from error
    if probabilities.shape != (metadata.get("prediction_count"),) or not np.isfinite(probabilities).all():
        raise CheckpointError(f"Fold result shape or values are invalid: {directory}")
    return probabilities.astype(float, copy=False)


def validate_provenance(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    _validate_privacy(actual)
    if fingerprint(actual) != fingerprint(expected):
        differing = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise CheckpointError(f"Experiment provenance mismatch: {', '.join(differing)}")


def atomic_joblib(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    if temporary.exists() or path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact or temporary file: {path}")
    try:
        joblib.dump(value, temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        if sha256_file(path) != digest:
            raise CheckpointError(f"Artifact checksum mismatch after replacement: {path}")
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()
