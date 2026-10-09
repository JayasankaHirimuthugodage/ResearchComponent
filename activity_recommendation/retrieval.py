"""Eligibility-first semantic ranking with NumPy cosine similarity."""

from __future__ import annotations

import numpy as np

from .contracts import EligibilityResult, RankedActivity
from .embeddings import EmbeddingArtifacts, TextEncoder


def retrieve_top_k(
    skill_query: str,
    eligibility: EligibilityResult,
    artifacts: EmbeddingArtifacts,
    encoder: TextEncoder,
    *,
    top_k: int = 5,
) -> tuple[RankedActivity, ...]:
    query = skill_query.strip()
    if not query:
        raise ValueError("A non-empty developmental skill query is required.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not eligibility.eligible:
        return ()

    eligible_records = tuple(
        record for record in eligibility.eligible if record.activity_id in artifacts.id_to_row
    )
    if len(eligible_records) != len(eligibility.eligible):
        raise ValueError("An eligible activity is missing from the embedding artifacts.")

    query_vector = np.asarray(encoder.encode([query]), dtype=np.float32)
    if query_vector.shape != (1, artifacts.embeddings.shape[1]):
        raise ValueError("Query embedding has an unexpected shape.")
    norm = float(np.linalg.norm(query_vector[0]))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Query embedding must be finite and non-zero.")
    normalized_query = query_vector[0] / norm

    rows = np.asarray(
        [artifacts.id_to_row[record.activity_id] for record in eligible_records], dtype=np.int64
    )
    scores = artifacts.embeddings[rows] @ normalized_query
    ranked = sorted(
        zip(eligible_records, scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].activity_id),
    )
    return tuple(
        RankedActivity(activity=record, similarity_score=float(score))
        for record, score in ranked[: min(top_k, len(ranked))]
    )
