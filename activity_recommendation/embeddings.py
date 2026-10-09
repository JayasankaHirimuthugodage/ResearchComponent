"""Build, store, and verify local Sentence Transformer embedding artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .catalogue import ActivityCatalogue
from .contracts import ActivityRecord, EmbeddingArtifactError


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_DIMENSION = 384
TEXT_TEMPLATE_VERSION = "1.0"
ARTIFACT_SCHEMA_VERSION = "1.0"


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Local encoder pinned to an immutable Hugging Face model revision."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_revision: str = MODEL_REVISION,
        *,
        local_files_only: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-activity-retrieval.txt to generate embeddings."
            ) from exc
        self.model_name = model_name
        self.model_revision = model_revision
        self._model = SentenceTransformer(
            model_name,
            revision=model_revision,
            device="cpu",
            local_files_only=local_files_only,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


@dataclass(frozen=True)
class EmbeddingArtifacts:
    embeddings: np.ndarray
    activity_ids: tuple[str, ...]
    id_to_row: dict[str, int]
    manifest: dict[str, object]


def activity_embedding_text(record: ActivityRecord) -> str:
    parts = [f"Title: {record.title}", f"Activity: {record.activity_summary}"]
    if record.primary_skill:
        parts.append(f"Developmental skill: {record.primary_skill}")
    parts.append(f"Domains: {', '.join(record.domains)}")
    return "\n".join(parts)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise EmbeddingArtifactError("Embeddings must be a two-dimensional matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise EmbeddingArtifactError("Embeddings must contain finite, non-zero rows.")
    return np.asarray(matrix / norms, dtype=np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(activity_ids: Sequence[str]) -> str:
    encoded = json.dumps(list(activity_ids), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_embedding_artifacts(
    catalogue: ActivityCatalogue,
    output_dir: Path,
    encoder: TextEncoder,
    *,
    model_name: str = MODEL_NAME,
    model_revision: str = MODEL_REVISION,
) -> dict[str, object]:
    """Build into a temporary sibling directory, validate, then replace atomically."""

    activity_ids = tuple(record.activity_id for record in catalogue.records)
    texts = [activity_embedding_text(record) for record in catalogue.records]
    embeddings = normalize_rows(encoder.encode(texts))
    if embeddings.shape != (len(activity_ids), EMBEDDING_DIMENSION):
        raise EmbeddingArtifactError(
            f"Expected {(len(activity_ids), EMBEDDING_DIMENSION)}, got {embeddings.shape}."
        )

    temporary = output_dir.with_name(f"{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        embeddings_path = temporary / "embeddings.npy"
        ids_path = temporary / "activity_ids.json"
        np.save(embeddings_path, embeddings, allow_pickle=False)
        ids_path.write_text(
            json.dumps(activity_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest: dict[str, object] = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "library_version": catalogue.library_version,
            "library_sha256": catalogue.library_sha256,
            "record_count": len(activity_ids),
            "embedding_model": model_name,
            "embedding_model_revision": model_revision,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "normalized": True,
            "dtype": "float32",
            "embedding_text_template_version": TEXT_TEMPLATE_VERSION,
            "activity_ids_sha256": _ids_sha256(activity_ids),
            "embeddings_sha256": _file_sha256(embeddings_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        load_embedding_artifacts(catalogue, temporary)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_embedding_artifacts(
    catalogue: ActivityCatalogue,
    artifact_dir: Path,
    *,
    model_name: str = MODEL_NAME,
    model_revision: str = MODEL_REVISION,
) -> EmbeddingArtifacts:
    try:
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        activity_ids = tuple(
            json.loads((artifact_dir / "activity_ids.json").read_text(encoding="utf-8"))
        )
        embeddings = np.load(artifact_dir / "embeddings.npy", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise EmbeddingArtifactError(f"Unable to load embedding artifacts: {exc}") from exc

    expected_ids = tuple(record.activity_id for record in catalogue.records)
    expected = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "library_version": catalogue.library_version,
        "library_sha256": catalogue.library_sha256,
        "record_count": len(expected_ids),
        "embedding_model": model_name,
        "embedding_model_revision": model_revision,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "normalized": True,
        "dtype": "float32",
        "embedding_text_template_version": TEXT_TEMPLATE_VERSION,
        "activity_ids_sha256": _ids_sha256(expected_ids),
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if activity_ids != expected_ids:
        mismatches.append("activity_id_order")
    if embeddings.shape != (len(expected_ids), EMBEDDING_DIMENSION):
        mismatches.append("embedding_shape")
    if embeddings.dtype != np.float32:
        mismatches.append("embedding_dtype")
    embeddings_path = artifact_dir / "embeddings.npy"
    if manifest.get("embeddings_sha256") != _file_sha256(embeddings_path):
        mismatches.append("embeddings_sha256")
    if mismatches:
        raise EmbeddingArtifactError(
            "Embedding artifacts are stale or inconsistent: " + ", ".join(sorted(set(mismatches)))
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(embeddings)) or not np.allclose(norms, 1.0, atol=1e-5):
        raise EmbeddingArtifactError("Stored embeddings are not finite normalized vectors.")

    return EmbeddingArtifacts(
        embeddings=embeddings,
        activity_ids=activity_ids,
        id_to_row={activity_id: row for row, activity_id in enumerate(activity_ids)},
        manifest=manifest,
    )
