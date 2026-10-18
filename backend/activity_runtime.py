"""Lazy activity-retrieval bridge for the explicitly approved caregiver catalogue."""
from __future__ import annotations

from pathlib import Path

from activity_recommendation.catalogue import (
    CAREGIVER_APPROVED_LIBRARY_PATH,
    CAREGIVER_APPROVED_SCHEMA_PATH,
    load_activity_catalogue,
)
from activity_recommendation.embeddings import (
    SentenceTransformerEncoder,
    load_embedding_artifacts,
)
from activity_recommendation.recommendation_service import ActivityRecommendationService


class ExistingCaregiverActivityRuntime:
    """Load local retrieval resources only on an explicitly enabled request."""

    def __init__(self, questionnaire_catalog) -> None:
        self.questionnaire_catalog = questionnaire_catalog
        self._service = None

    def __call__(self, prediction, checkpoint_id, responses, safety_context):
        if self._service is None:
            catalogue = load_activity_catalogue(
                CAREGIVER_APPROVED_LIBRARY_PATH,
                CAREGIVER_APPROVED_SCHEMA_PATH,
            )
            artifacts = load_embedding_artifacts(
                catalogue,
                Path("artifacts/activity-embeddings/0.5.0"),
            )
            self._service = ActivityRecommendationService(
                self.questionnaire_catalog,
                catalogue,
                artifacts,
                SentenceTransformerEncoder(),
            )
        from activity_recommendation.contracts import RetrievalMode
        return self._service.recommend(
            prediction,
            checkpoint_id,
            responses,
            safety_context,
            mode=RetrievalMode.CAREGIVER_DEPLOYMENT,
        )
