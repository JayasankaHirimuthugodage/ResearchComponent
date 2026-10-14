from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from activity_recommendation.catalogue import ActivityCatalogue, load_activity_catalogue
from activity_recommendation.contracts import (
    AbilityState,
    EligibilityRequest,
    EmbeddingArtifactError,
    RetrievalMode,
    SafetyContext,
)
from activity_recommendation.eligibility import filter_eligible_activities
from activity_recommendation.embeddings import (
    EMBEDDING_DIMENSION,
    EmbeddingArtifacts,
    build_embedding_artifacts,
    load_embedding_artifacts,
)
from activity_recommendation.retrieval import retrieve_top_k


class DeterministicEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = np.resize(np.frombuffer(digest, dtype=np.uint8), EMBEDDING_DIMENSION)
            rows.append(values.astype(np.float32) + 1.0)
        return np.asarray(rows, dtype=np.float32)


class FixedQueryEncoder:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector.astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.repeat(self.vector[np.newaxis, :], len(texts), axis=0)


def all_available() -> SafetyContext:
    return SafetyContext(
        AbilityState.AVAILABLE,
        AbilityState.AVAILABLE,
        AbilityState.AVAILABLE,
        AbilityState.AVAILABLE,
    )


class ActivityRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogue = load_activity_catalogue()

    def test_artifact_round_trip_preserves_ids_and_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "embeddings"
            manifest = build_embedding_artifacts(
                self.catalogue, target, DeterministicEncoder()
            )
            artifacts = load_embedding_artifacts(self.catalogue, target)

            self.assertEqual(232, artifacts.embeddings.shape[0])
            self.assertEqual(EMBEDDING_DIMENSION, artifacts.embeddings.shape[1])
            self.assertEqual(
                tuple(record.activity_id for record in self.catalogue.records),
                artifacts.activity_ids,
            )
            self.assertEqual(self.catalogue.library_sha256, manifest["library_sha256"])

    def test_library_checksum_mismatch_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "embeddings"
            build_embedding_artifacts(self.catalogue, target, DeterministicEncoder())
            altered = ActivityCatalogue(
                self.catalogue.library_name,
                self.catalogue.library_version,
                "0" * 64,
                self.catalogue.records,
            )
            with self.assertRaisesRegex(EmbeddingArtifactError, "library_sha256"):
                load_embedding_artifacts(altered, target)

    def test_activity_id_order_mismatch_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "embeddings"
            build_embedding_artifacts(self.catalogue, target, DeterministicEncoder())
            ids_path = target / "activity_ids.json"
            activity_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            activity_ids[0], activity_ids[1] = activity_ids[1], activity_ids[0]
            ids_path.write_text(json.dumps(activity_ids), encoding="utf-8")
            with self.assertRaisesRegex(EmbeddingArtifactError, "activity_id_order"):
                load_embedding_artifacts(self.catalogue, target)

    def test_tampered_embedding_file_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "embeddings"
            build_embedding_artifacts(self.catalogue, target, DeterministicEncoder())
            matrix = np.load(target / "embeddings.npy", allow_pickle=False)
            matrix[0, 0] += 0.25
            np.save(target / "embeddings.npy", matrix, allow_pickle=False)
            with self.assertRaisesRegex(EmbeddingArtifactError, "embeddings_sha256"):
                load_embedding_artifacts(self.catalogue, target)

    def test_retrieval_ranks_only_eligible_records(self) -> None:
        eligibility = filter_eligible_activities(
            self.catalogue.records,
            EligibilityRequest("CP2M", "communication", all_available()),
        )
        eligible = eligibility.eligible[:2]
        restricted = type(eligibility)(eligible, eligibility.withheld, {})
        activity_ids = tuple(record.activity_id for record in self.catalogue.records)
        matrix = np.zeros((len(activity_ids), EMBEDDING_DIMENSION), dtype=np.float32)
        matrix[:, 2] = 1.0
        matrix[activity_ids.index(eligible[0].activity_id)] = np.eye(1, EMBEDDING_DIMENSION, 0)[0]
        matrix[activity_ids.index(eligible[1].activity_id)] = np.eye(1, EMBEDDING_DIMENSION, 1)[0]
        artifacts = EmbeddingArtifacts(
            matrix,
            activity_ids,
            {activity_id: index for index, activity_id in enumerate(activity_ids)},
            {},
        )
        query = np.eye(1, EMBEDDING_DIMENSION, 1)[0]
        results = retrieve_top_k(
            "verbal turn-taking",
            restricted,
            artifacts,
            FixedQueryEncoder(query),
            top_k=2,
        )
        self.assertEqual(eligible[1].activity_id, results[0].activity.activity_id)
        self.assertEqual({item.activity.activity_id for item in results}, {r.activity_id for r in eligible})

    def test_equal_scores_use_activity_id_as_tie_breaker(self) -> None:
        eligibility = filter_eligible_activities(
            self.catalogue.records,
            EligibilityRequest("CP2M", "communication", all_available()),
        )
        eligible = tuple(reversed(eligibility.eligible[:2]))
        restricted = type(eligibility)(eligible, (), {})
        activity_ids = tuple(record.activity_id for record in self.catalogue.records)
        matrix = np.zeros((len(activity_ids), EMBEDDING_DIMENSION), dtype=np.float32)
        matrix[:, 0] = 1.0
        artifacts = EmbeddingArtifacts(
            matrix,
            activity_ids,
            {activity_id: index for index, activity_id in enumerate(activity_ids)},
            {},
        )
        results = retrieve_top_k(
            "shared score",
            restricted,
            artifacts,
            FixedQueryEncoder(np.eye(1, EMBEDDING_DIMENSION, 0)[0]),
            top_k=2,
        )
        self.assertEqual(sorted(record.activity_id for record in eligible), [r.activity.activity_id for r in results])

    def test_empty_eligibility_returns_no_results_without_encoding(self) -> None:
        eligibility = filter_eligible_activities(
            self.catalogue.records,
            EligibilityRequest(
                "CP2M",
                "communication",
                all_available(),
                mode=RetrievalMode.CAREGIVER_DEPLOYMENT,
            ),
        )
        results = retrieve_top_k(
            "communication turn-taking",
            eligibility,
            EmbeddingArtifacts(
                np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), (), {}, {}
            ),
            FixedQueryEncoder(np.ones(EMBEDDING_DIMENSION, dtype=np.float32)),
        )
        self.assertEqual((), results)

    def test_invalid_query_and_top_k_are_rejected(self) -> None:
        eligibility = filter_eligible_activities(
            self.catalogue.records,
            EligibilityRequest("CP2M", "communication", all_available()),
        )
        artifacts = EmbeddingArtifacts(
            np.zeros((len(self.catalogue.records), EMBEDDING_DIMENSION), dtype=np.float32),
            tuple(record.activity_id for record in self.catalogue.records),
            {record.activity_id: i for i, record in enumerate(self.catalogue.records)},
            {},
        )
        encoder = FixedQueryEncoder(np.ones(EMBEDDING_DIMENSION, dtype=np.float32))
        with self.assertRaises(ValueError):
            retrieve_top_k(" ", eligibility, artifacts, encoder)
        with self.assertRaises(ValueError):
            retrieve_top_k("skill", eligibility, artifacts, encoder, top_k=0)


if __name__ == "__main__":
    unittest.main()
