"""Bounded, non-causal local TabICLv2 model-behaviour analysis.

This is not SHAP, feature attribution, population importance, or a causal
explanation.  It reports the selected fitted model's calibrated probability
changes when one valid submitted response is varied at a time.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from inference.contracts import AssessmentStatus, DomainPrediction
from inference.questionnaire import EncodedAssessment, QuestionnaireCatalog
from training.contracts import FEATURE_COLUMNS


class TabICLv2ExplanationStatus(str, Enum):
    AVAILABLE_LOCAL_MODEL_BEHAVIOUR = "available_local_model_behaviour_analysis"
    UNAVAILABLE_PENDING_VALIDATION = "unavailable_pending_model_specific_validation"
    NOT_APPLICABLE_INCOMPLETE = "not_applicable_incomplete_assessment"


@dataclass(frozen=True)
class TabICLv2Explanation:
    """Safe, non-causal status response without inputs or contribution values."""

    domain: str
    status: TabICLv2ExplanationStatus
    method: str
    is_available: bool
    feature_contributions: tuple[()] = ()
    limitation: str = (
        "No model-specific TabICLv2 feature explanation has been implemented or "
        "validated. This status is not evidence of causal, clinical, or feature importance."
    )
    development_only: bool = True


@dataclass(frozen=True)
class TabICLv2ResponseVariation:
    """One actual alternate-model calculation, never an attribution value."""

    feature: str
    question_id: str
    question_text: str
    question_domain: str
    original_response: int
    alternative_response: int
    baseline_probability: float
    alternative_probability: float
    probability_change: float


@dataclass(frozen=True)
class TabICLv2LocalModelBehaviour:
    domain: str
    status: TabICLv2ExplanationStatus
    method: str
    is_available: bool
    baseline_probability: float
    threshold: float
    baseline_decision: bool
    variations: tuple[TabICLv2ResponseVariation, ...]
    limitation: str
    development_only: bool = True


def _complete_values(encoded: EncodedAssessment) -> tuple[str, ...]:
    if encoded.status is not AssessmentStatus.COMPLETE or encoded.feature_values is None:
        raise ValueError("A complete encoded assessment is required for a Cognitive explanation request.")
    if len(encoded.feature_values) != len(FEATURE_COLUMNS):
        raise ValueError("Cognitive explanation requires the exact 16-feature contract.")
    return encoded.feature_values


def analyze_tabiclv2_cognitive_model_behaviour(
    catalog: QuestionnaireCatalog,
    encoded: EncodedAssessment,
    prediction: DomainPrediction,
    cognitive_model,
) -> TabICLv2LocalModelBehaviour:
    """Evaluate exactly two valid alternatives for each of the 15 responses.

    ``cognitive_model`` is intentionally a narrow probability interface so the
    production adapter and deterministic synthetic test doubles use the same
    calculation path.  The baseline is reused from the already displayed
    selected-model prediction; only altered inputs invoke the model.
    """
    if prediction.domain != "cognitive":
        raise ValueError("TabICLv2 explanation is defined only for the Cognitive domain.")
    values = _complete_values(encoded)
    baseline = float(prediction.probability)
    if not 0.0 <= baseline <= 1.0:
        raise ValueError("Cognitive baseline probability is invalid.")
    questions = catalog.questions_for_checkpoint(values[0])
    if len(questions) != 15:
        raise ValueError("Cognitive explanation requires exactly 15 checkpoint questions.")

    variations: list[TabICLv2ResponseVariation] = []
    for offset, question in enumerate(questions, start=1):
        original = int(values[offset])
        if original not in (0, 1, 2):
            raise ValueError("Cognitive explanation received an invalid scoreable response.")
        for alternative in (0, 1, 2):
            if alternative == original:
                continue
            altered = list(values)
            altered[offset] = str(alternative)
            probability = float(cognitive_model.positive_probability(tuple(altered)))
            if not 0.0 <= probability <= 1.0:
                raise RuntimeError("TabICLv2 returned an invalid altered-input probability.")
            variations.append(
                TabICLv2ResponseVariation(
                    feature=FEATURE_COLUMNS[offset],
                    question_id=question.question_id,
                    question_text=question.text,
                    question_domain=question.domain,
                    original_response=original,
                    alternative_response=alternative,
                    baseline_probability=baseline,
                    alternative_probability=probability,
                    probability_change=probability - baseline,
                )
            )
    if len(variations) != 30:
        raise RuntimeError("Cognitive explanation exceeded its fixed 30-alternative bound.")
    return TabICLv2LocalModelBehaviour(
        domain="cognitive",
        status=TabICLv2ExplanationStatus.AVAILABLE_LOCAL_MODEL_BEHAVIOUR,
        method="one_at_a_time_calibrated_probability_response_analysis",
        is_available=True,
        baseline_probability=baseline,
        threshold=float(prediction.threshold),
        baseline_decision=bool(prediction.decision),
        variations=tuple(variations),
        limitation=(
            "These results show local model behaviour for hypothetical one-answer changes while all "
            "other submitted answers remain fixed. They are not feature attributions, causal effects, "
            "clinical importance, population-level importance, or evidence that changing a child's "
            "behaviour changes developmental outcomes."
        ),
    )


def explain_tabiclv2_cognitive_prediction(
    encoded: EncodedAssessment,
    prediction: DomainPrediction,
) -> TabICLv2Explanation:
    """Return only an explicit unavailable status; never invokes a model."""
    if prediction.domain != "cognitive":
        raise ValueError("TabICLv2 explanation is defined only for the Cognitive domain.")
    if encoded.status is AssessmentStatus.INCOMPLETE:
        return TabICLv2Explanation(
            domain="cognitive",
            status=TabICLv2ExplanationStatus.NOT_APPLICABLE_INCOMPLETE,
            method="none_incomplete_assessment",
            is_available=False,
            limitation=(
                "The assessment is incomplete, so no Cognitive model prediction or "
                "model-behaviour explanation is available."
            ),
        )
    if encoded.status is not AssessmentStatus.COMPLETE or encoded.feature_values is None:
        raise ValueError("A complete encoded assessment is required for a Cognitive explanation request.")
    return TabICLv2Explanation(
        domain="cognitive",
        status=TabICLv2ExplanationStatus.UNAVAILABLE_PENDING_VALIDATION,
        method="none_pending_tabiclv2_model_behaviour_validation",
        is_available=False,
    )
