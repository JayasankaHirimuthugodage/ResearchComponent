from __future__ import annotations

import unittest

import numpy as np

from activity_recommendation.catalogue import ActivityCatalogue
from activity_recommendation.contracts import (
    AbilityState,
    ActivityRecord,
    SafetyContext,
)
from activity_recommendation.embeddings import EMBEDDING_DIMENSION, EmbeddingArtifacts
from activity_recommendation.recommendation_service import (
    ActivityRecommendationService,
    RecommendationStatus,
)
from inference.contracts import AssessmentStatus, DOMAIN_IDS, DomainPrediction, PredictionResult
from inference.questionnaire import load_questionnaire_catalog


class QueryEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
            lowered = text.lower()
            if "rhyme" in lowered or "rhyming" in lowered:
                vector[0] = 1.0
            elif "conversation" in lowered or "several turns" in lowered:
                vector[1] = 1.0
            elif "count" in lowered:
                vector[2] = 1.0
            else:
                vector[3] = 1.0
            rows.append(vector)
        return np.asarray(rows)


def record(
    activity_id: str,
    domain: str,
    vector_index: int,
    *,
    hearing: str = "not_obviously_required",
    research_demo_eligible: bool = True,
) -> tuple[ActivityRecord, np.ndarray]:
    item = ActivityRecord(
        activity_id=activity_id,
        record_version="1.0",
        library_version="0.4.0",
        title=activity_id,
        activity_summary="Test source-linked activity.",
        primary_skill=None,
        domains=(domain,),
        checkpoint_ids=("CP60M",),
        source_org="CDC",
        source_url=f"https://example.test/{activity_id}",
        source_page=None,
        source_tip_id=None,
        status="draft_for_validation",
        safety_demands={
            "hearing": hearing,
            "vision": "not_obviously_required",
            "fine_motor": "not_obviously_required",
            "gross_motor": "not_obviously_required",
        },
        research_demo_eligible=research_demo_eligible,
        caregiver_deployment_eligible=False,
        raw_record={"activity_id": activity_id},
    )
    vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    vector[vector_index] = 1.0
    return item, vector


def prediction(*positive_domains: str, incomplete: bool = False) -> PredictionResult:
    if incomplete:
        return PredictionResult(
            AssessmentStatus.INCOMPLETE, (), None, ("CP60M_COMM_Q1",),
            "test-model", "test-features", "developmental-questionnaire-v3", None, None,
        )
    decisions = tuple(
        DomainPrediction(domain, 0.8 if domain in positive_domains else 0.1, domain in positive_domains)
        for domain in DOMAIN_IDS
    )
    return PredictionResult(
        AssessmentStatus.COMPLETE, decisions, bool(positive_domains), (),
        "test-model", "test-features", "developmental-questionnaire-v3", None, None,
    )


class RecommendationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questionnaire = load_questionnaire_catalog()
        pairs = [
            record("ACT-RHYME", "communication", 0),
            record("ACT-CONVERSATION", "communication", 1),
            record("ACT-UNSAFE-HEARING", "communication", 0, hearing="review_if_concern"),
            record("ACT-NOT-APPROVED", "communication", 0, research_demo_eligible=False),
            record("ACT-COUNT", "cognitive", 2),
            record("ACT-SOCIAL", "social_emotional", 3),
        ]
        records = tuple(item for item, _ in pairs)
        matrix = np.asarray([vector for _, vector in pairs], dtype=np.float32)
        ids = tuple(item.activity_id for item in records)
        cls.service = ActivityRecommendationService(
            cls.questionnaire,
            ActivityCatalogue("test", "0.4.0", "0" * 64, records),
            EmbeddingArtifacts(matrix, ids, {value: index for index, value in enumerate(ids)}, {}),
            QueryEncoder(),
        )

    def responses(self, **overrides: str) -> dict[str, str]:
        values = {
            question.question_id: "YES_USUALLY"
            for question in self.questionnaire.questions_for_checkpoint("CP60M")
        }
        values.update(overrides)
        return values

    @staticmethod
    def safety(state: AbilityState = AbilityState.NO_CONCERN_REPORTED) -> SafetyContext:
        return SafetyContext(state, state, state, state)

    def test_one_two_and_three_concern_domains_are_independently_targeted(self) -> None:
        responses = self.responses(
            CP60M_COMM_Q5="NOT_YET",
            CP60M_COG_Q4="NOT_YET",
            CP60M_SOC_Q1="SOMETIMES",
        )
        for domains in (("communication",), ("communication", "cognitive"), DOMAIN_IDS):
            with self.subTest(domains=domains):
                result = self.service.recommend(
                    prediction(*domains), "CP60M", responses, self.safety()
                )
                self.assertEqual(domains, result.targeted_domains)
                self.assertEqual(domains, tuple(item.domain for item in result.domains))

    def test_zero_concern_and_incomplete_cases_do_not_retrieve(self) -> None:
        responses = self.responses(CP60M_COMM_Q5="NOT_YET")
        no_concern = self.service.recommend(prediction(), "CP60M", responses, self.safety())
        incomplete = self.service.recommend(
            prediction(incomplete=True), "CP60M", responses, self.safety()
        )
        self.assertIs(RecommendationStatus.NO_CONCERN, no_concern.status)
        self.assertEqual((), no_concern.domains)
        self.assertIs(RecommendationStatus.INCOMPLETE_ASSESSMENT, incomplete.status)
        self.assertEqual((), incomplete.domains)

    def test_only_not_yet_and_sometimes_create_skill_context(self) -> None:
        no_need = self.service.recommend(
            prediction("communication"), "CP60M", self.responses(), self.safety()
        )
        self.assertIs(
            RecommendationStatus.NO_SUPPORTED_SKILL_CONTEXT,
            no_need.domains[0].status,
        )
        for unscorable in ("NOT_SURE", "NOT_OBSERVED"):
            with self.subTest(unscorable=unscorable), self.assertRaises(ValueError):
                self.service.recommend(
                    prediction("communication"),
                    "CP60M",
                    self.responses(CP60M_COMM_Q5=unscorable),
                    self.safety(),
                )

    def test_different_observed_needs_change_ranking(self) -> None:
        rhyme = self.service.recommend(
            prediction("communication"),
            "CP60M",
            self.responses(CP60M_COMM_Q5="NOT_YET"),
            self.safety(),
        )
        conversation = self.service.recommend(
            prediction("communication"),
            "CP60M",
            self.responses(CP60M_COMM_Q3="NOT_YET"),
            self.safety(),
        )
        self.assertEqual("ACT-RHYME", rhyme.domains[0].activities[0].activity.activity_id)
        self.assertEqual(
            "ACT-CONVERSATION",
            conversation.domains[0].activities[0].activity.activity_id,
        )

    def test_safety_filter_precedes_ranking_and_result_count_is_not_padded(self) -> None:
        result = self.service.recommend(
            prediction("communication"),
            "CP60M",
            self.responses(CP60M_COMM_Q5="NOT_YET"),
            SafetyContext(
                AbilityState.CONCERN_REPORTED,
                AbilityState.NO_CONCERN_REPORTED,
                AbilityState.NO_CONCERN_REPORTED,
                AbilityState.NO_CONCERN_REPORTED,
            ),
        )
        ids = tuple(item.activity.activity_id for item in result.domains[0].activities)
        self.assertNotIn("ACT-UNSAFE-HEARING", ids)
        self.assertNotIn("ACT-NOT-APPROVED", ids)
        self.assertEqual(2, len(ids))

    def test_missing_or_unscorable_questionnaire_context_is_rejected(self) -> None:
        responses = self.responses(CP60M_COMM_Q5="NOT_YET")
        responses.pop("CP60M_COMM_Q1")
        with self.assertRaisesRegex(ValueError, "exact checkpoint questions"):
            self.service.recommend(
                prediction("communication"), "CP60M", responses, self.safety()
            )


if __name__ == "__main__":
    unittest.main()
