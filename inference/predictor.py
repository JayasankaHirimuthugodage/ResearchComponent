"""Application service that turns validated answers into three domain predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from inference.contracts import (
    AssessmentStatus,
    DOMAIN_IDS,
    DomainPrediction,
    PredictionResult,
    QuestionnaireSubmission,
)
from inference.questionnaire import EncodedAssessment, QuestionnaireCatalog, encode_submission


class PositiveProbabilityModel(Protocol):
    """Port implemented by any model-runtime adapter."""

    def positive_probability(self, feature_values: tuple[str, ...]) -> float:
        """Return the probability of the positive-concern class."""


@dataclass(frozen=True)
class DevelopmentPredictionService:
    """Framework-neutral three-domain prediction use case."""

    catalog: QuestionnaireCatalog
    models: Mapping[str, PositiveProbabilityModel]
    model_version: str
    feature_schema_version: str
    threshold: float
    threshold_version: str | None

    def __post_init__(self) -> None:
        model_copy = dict(self.models)
        if set(model_copy) != set(DOMAIN_IDS):
            raise ValueError("Exactly one probability model is required for each domain.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("The probability threshold must be between zero and one.")
        for value, name in (
            (self.model_version, "model_version"),
            (self.feature_schema_version, "feature_schema_version"),
        ):
            if not value:
                raise ValueError(f"{name} is required for prediction traceability.")
        if self.threshold_version is not None and not self.threshold_version:
            raise ValueError("threshold_version must be non-empty when supplied.")
        object.__setattr__(self, "models", MappingProxyType(model_copy))

    def predict(self, submission: QuestionnaireSubmission) -> PredictionResult:
        """Validate, encode and predict one questionnaire submission."""

        return self.predict_encoded(encode_submission(self.catalog, submission))

    def predict_encoded(self, encoded: EncodedAssessment) -> PredictionResult:
        """Predict from one already-validated encoding shared across candidates."""

        common = {
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "questionnaire_schema_version": self.catalog.questionnaire_schema_version,
            "threshold": self.threshold,
            "threshold_version": self.threshold_version,
        }
        if encoded.status is AssessmentStatus.INCOMPLETE:
            return PredictionResult(
                status=AssessmentStatus.INCOMPLETE,
                domain_predictions=(),
                overall_concern=None,
                incomplete_question_ids=encoded.incomplete_question_ids,
                **common,
            )

        if encoded.feature_values is None:
            raise RuntimeError("A complete assessment has no encoded model inputs.")
        predictions: list[DomainPrediction] = []
        for domain in DOMAIN_IDS:
            probability = float(
                self.models[domain].positive_probability(encoded.feature_values)
            )
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeError("A model returned an invalid positive-class probability.")
            predictions.append(
                DomainPrediction(
                    domain=domain,
                    probability=probability,
                    decision=probability >= self.threshold,
                    threshold=self.threshold,
                    model_version=self.model_version,
                )
            )

        return PredictionResult(
            status=AssessmentStatus.COMPLETE,
            domain_predictions=tuple(predictions),
            overall_concern=any(item.decision for item in predictions),
            incomplete_question_ids=(),
            **common,
        )
