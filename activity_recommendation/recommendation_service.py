"""Coordinate frozen domain decisions with safe, traceable activity retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from inference.contracts import AssessmentStatus, PredictionResult
from inference.questionnaire import QuestionnaireCatalog

from .catalogue import ActivityCatalogue
from .contracts import EligibilityRequest, RankedActivity, RetrievalMode, SafetyContext
from .eligibility import filter_eligible_activities
from .embeddings import EmbeddingArtifacts, TextEncoder
from .retrieval import retrieve_top_k
from .skill_queries import SkillQuery, build_skill_query


class RecommendationStatus(str, Enum):
    READY = "ready"
    NO_CONCERN = "no_concern"
    INCOMPLETE_ASSESSMENT = "incomplete_assessment"
    NO_SUPPORTED_SKILL_CONTEXT = "no_supported_skill_context"
    NO_ELIGIBLE_ACTIVITIES = "no_eligible_activities"


@dataclass(frozen=True)
class DomainActivityRecommendations:
    domain: str
    status: RecommendationStatus
    skill_query: SkillQuery | None
    activities: tuple[RankedActivity, ...]
    eligible_count: int
    withheld_count: int


@dataclass(frozen=True)
class ActivityRecommendationResult:
    status: RecommendationStatus
    checkpoint_id: str
    targeted_domains: tuple[str, ...]
    domains: tuple[DomainActivityRecommendations, ...]


@dataclass(frozen=True)
class ActivityRecommendationService:
    questionnaire_catalog: QuestionnaireCatalog
    activity_catalogue: ActivityCatalogue
    embedding_artifacts: EmbeddingArtifacts
    encoder: TextEncoder

    def recommend(
        self,
        prediction: PredictionResult,
        checkpoint_id: str,
        questionnaire_responses: Mapping[str, str],
        safety_context: SafetyContext,
        *,
        top_k: int = 3,
    ) -> ActivityRecommendationResult:
        if top_k < 1 or top_k > 3:
            raise ValueError("The research workflow supports between one and three results.")
        if prediction.status is AssessmentStatus.INCOMPLETE:
            return ActivityRecommendationResult(
                RecommendationStatus.INCOMPLETE_ASSESSMENT,
                checkpoint_id,
                (),
                (),
            )

        expected_ids = {
            question.question_id
            for question in self.questionnaire_catalog.questions_for_checkpoint(checkpoint_id)
        }
        if set(questionnaire_responses) != expected_ids:
            raise ValueError("Recommendation responses must contain the exact checkpoint questions.")
        if any(
            code not in {"YES_USUALLY", "SOMETIMES", "NOT_YET"}
            for code in questionnaire_responses.values()
        ):
            raise ValueError("A complete prediction requires scoreable questionnaire responses.")

        targeted_domains = tuple(
            item.domain for item in prediction.domain_predictions if item.decision
        )
        if not targeted_domains:
            return ActivityRecommendationResult(
                RecommendationStatus.NO_CONCERN,
                checkpoint_id,
                (),
                (),
            )

        domain_results: list[DomainActivityRecommendations] = []
        for domain in targeted_domains:
            skill_query = build_skill_query(
                self.questionnaire_catalog,
                checkpoint_id,
                domain,
                questionnaire_responses,
            )
            if skill_query is None:
                domain_results.append(
                    DomainActivityRecommendations(
                        domain,
                        RecommendationStatus.NO_SUPPORTED_SKILL_CONTEXT,
                        None,
                        (),
                        0,
                        0,
                    )
                )
                continue

            eligibility = filter_eligible_activities(
                self.activity_catalogue.records,
                EligibilityRequest(
                    checkpoint_id,
                    domain,
                    safety_context,
                    RetrievalMode.INTERNAL_RESEARCH_DEMO,
                ),
            )
            activities = retrieve_top_k(
                skill_query.text,
                eligibility,
                self.embedding_artifacts,
                self.encoder,
                top_k=min(top_k, len(eligibility.eligible)),
            ) if eligibility.eligible else ()
            status = (
                RecommendationStatus.READY
                if activities
                else RecommendationStatus.NO_ELIGIBLE_ACTIVITIES
            )
            domain_results.append(
                DomainActivityRecommendations(
                    domain,
                    status,
                    skill_query,
                    activities,
                    len(eligibility.eligible),
                    len(eligibility.withheld),
                )
            )

        if any(item.status is RecommendationStatus.READY for item in domain_results):
            overall_status = RecommendationStatus.READY
        elif any(
            item.status is RecommendationStatus.NO_ELIGIBLE_ACTIVITIES
            for item in domain_results
        ):
            overall_status = RecommendationStatus.NO_ELIGIBLE_ACTIVITIES
        else:
            overall_status = RecommendationStatus.NO_SUPPORTED_SKILL_CONTEXT
        return ActivityRecommendationResult(
            overall_status,
            checkpoint_id,
            targeted_domains,
            tuple(domain_results),
        )
