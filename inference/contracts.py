"""Transport- and storage-independent prediction request/response contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


DOMAIN_IDS = ("communication", "cognitive", "social_emotional")
DOMAIN_DISPLAY_NAMES = {
    "communication": "Communication",
    "cognitive": "Cognitive",
    "social_emotional": "Social / Emotional",
}


class AssessmentStatus(str, Enum):
    """Whether the submitted answers satisfy the current ML feature contract."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class SubmissionValidationError(ValueError):
    """Raised when a submission violates the versioned input contract."""


@dataclass(frozen=True)
class QuestionnaireSubmission:
    """The minimum assessment data accepted by the prediction application."""

    schema_version: str
    checkpoint_id: str
    responses: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "responses", MappingProxyType(dict(self.responses)))


@dataclass(frozen=True)
class DomainPrediction:
    """One domain's probability and threshold-derived positive-concern flag."""

    domain: str
    probability: float
    decision: bool


@dataclass(frozen=True)
class PredictionResult:
    """Non-diagnostic result returned by the application boundary."""

    status: AssessmentStatus
    domain_predictions: tuple[DomainPrediction, ...]
    overall_concern: bool | None
    incomplete_question_ids: tuple[str, ...]
    model_version: str
    feature_schema_version: str
    questionnaire_schema_version: str
    threshold: float
    threshold_version: str

    def __post_init__(self) -> None:
        if self.status is AssessmentStatus.COMPLETE:
            if tuple(item.domain for item in self.domain_predictions) != DOMAIN_IDS:
                raise ValueError("A complete result requires all three domains in order.")
            if self.overall_concern is None or self.incomplete_question_ids:
                raise ValueError("A complete result has a decision and no incomplete items.")
        elif self.domain_predictions or self.overall_concern is not None:
            raise ValueError("An incomplete result cannot contain ML predictions.")
