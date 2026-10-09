"""Model-specific local explanations with a separate caregiver-safe summary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import DMatrix, XGBClassifier

from inference.contracts import AssessmentStatus, DomainPrediction
from inference.model_artifact import CalibratedDomainModel
from inference.questionnaire import EncodedAssessment, QuestionnaireCatalog
from training.contracts import FEATURE_COLUMNS
from training.models.common import transformed_feature_names


CAREGIVER_EXPLANATION_NOTICE = (
    "The models considered all 15 answers and the selected checkpoint when producing "
    "these results. The highlighted observations are examples of influential answers "
    "and did not determine the results alone. These results support developmental "
    "monitoring and are not diagnoses."
)


@dataclass(frozen=True)
class LocalFeatureContribution:
    feature: str
    display_name: str
    input_value_label: str
    direction: str
    contribution: float


@dataclass(frozen=True)
class DomainLocalExplanation:
    domain: str
    method: str
    model_output_logit: float
    base_value: float
    contributions: tuple[LocalFeatureContribution, ...]
    caregiver_message: str
    limitation: str


def _transformed_contributions(
    model: CalibratedDomainModel,
    feature_values: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray, float, str]:
    pipeline = model.pipeline
    matrix = np.asarray([feature_values], dtype=object)
    transformed = pipeline.named_steps["preprocess"].transform(matrix)
    names = transformed_feature_names(pipeline)
    estimator = pipeline.named_steps["estimator"]

    if isinstance(estimator, LogisticRegression):
        raw_contributions = estimator.coef_[0] * transformed[0]
        raw_base = float(estimator.intercept_[0])
        method = "logistic_coefficient_contribution"
    elif isinstance(estimator, XGBClassifier):
        contribution_matrix = estimator.get_booster().predict(
            DMatrix(transformed),
            pred_contribs=True,
            validate_features=False,
        )
        if contribution_matrix.shape != (1, len(names) + 1):
            raise RuntimeError("XGBoost returned an invalid contribution matrix.")
        raw_contributions = contribution_matrix[0, :-1]
        raw_base = float(contribution_matrix[0, -1])
        method = "xgboost_tree_shap_contribution"
    else:
        raise TypeError("The frozen estimator has no supported local explanation method.")

    calibration_coefficient = float(model.calibrator.coef_[0, 0])
    calibration_intercept = float(model.calibrator.intercept_[0])
    if calibration_coefficient <= 0.0:
        raise RuntimeError("The fitted calibrator is not monotonic increasing.")
    calibrated_contributions = np.asarray(raw_contributions) * calibration_coefficient
    calibrated_base = raw_base * calibration_coefficient + calibration_intercept
    return names, calibrated_contributions, calibrated_base, method


def _conceptual_feature(name: str) -> str:
    if name.startswith("baseline_checkpoint_id_"):
        return "baseline_checkpoint_id"
    return name


def _aggregate_contributions(
    names: Sequence[str], values: Sequence[float]
) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for name, value in zip(names, values, strict=True):
        feature = _conceptual_feature(name)
        aggregated[feature] = aggregated.get(feature, 0.0) + float(value)
    return aggregated


def _display_names(
    catalog: QuestionnaireCatalog,
    checkpoint_id: str,
) -> dict[str, str]:
    names = {"baseline_checkpoint_id": "Selected age checkpoint"}
    for question in catalog.questions_for_checkpoint(checkpoint_id):
        names[question.feature_column] = question.text
    return names


def _input_value_labels(feature_values: tuple[str, ...]) -> dict[str, str]:
    response_labels = {
        "0": "Not yet",
        "1": "Sometimes",
        "2": "Yes, usually",
    }
    labels = {"baseline_checkpoint_id": feature_values[0]}
    for feature, value in zip(FEATURE_COLUMNS[1:], feature_values[1:], strict=True):
        labels[feature] = response_labels[value]
    return labels


def _caregiver_message(
    contributions: Sequence[LocalFeatureContribution],
    *,
    decision: bool,
) -> str:
    questionnaire_items = [
        item
        for item in contributions
        if item.feature != "baseline_checkpoint_id" and item.direction != "neutral"
    ]
    supporting_direction = "toward_concern" if decision else "away_from_concern"
    supporting = next(
        (item for item in questionnaire_items if item.direction == supporting_direction),
        None,
    )

    lines = ["**Observation contributing most to this result**"]
    if supporting is None:
        lines.append(
            "- No individual questionnaire answer moved the calculation in the same "
            "direction as the final result."
        )
    else:
        supporting_text = "toward" if decision else "away from"
        lines.append(
            f'- **{supporting.input_value_label}** for “{supporting.display_name}” moved '
            f"the model calculation {supporting_text} a concern flag."
        )

    return "\n".join(lines)


def explain_complete_assessment(
    catalog: QuestionnaireCatalog,
    models: Mapping[str, CalibratedDomainModel],
    encoded: EncodedAssessment,
    predictions: Sequence[DomainPrediction],
) -> tuple[DomainLocalExplanation, ...]:
    """Explain one complete assessment without changing its predictions."""

    if encoded.status is not AssessmentStatus.COMPLETE or encoded.feature_values is None:
        return ()
    prediction_by_domain = {item.domain: item for item in predictions}
    if set(prediction_by_domain) != set(models):
        raise ValueError("Predictions and frozen models do not cover the same domains.")

    display_names = _display_names(catalog, encoded.feature_values[0])
    input_value_labels = _input_value_labels(encoded.feature_values)
    explanations: list[DomainLocalExplanation] = []
    for domain, model in models.items():
        names, values, base_value, method = _transformed_contributions(
            model,
            encoded.feature_values,
        )
        aggregated = _aggregate_contributions(names, values)
        ordered = sorted(aggregated.items(), key=lambda item: -abs(item[1]))
        contributions = tuple(
            LocalFeatureContribution(
                feature=feature,
                display_name=display_names[feature],
                input_value_label=input_value_labels[feature],
                direction=(
                    "toward_concern"
                    if value > 0
                    else "away_from_concern"
                    if value < 0
                    else "neutral"
                ),
                contribution=value,
            )
            for feature, value in ordered
        )
        calibrated_logit = base_value + sum(item.contribution for item in contributions)
        probability = prediction_by_domain[domain].probability
        expected_logit = math.log(np.clip(probability, 1e-12, 1.0 - 1e-12)) - math.log1p(
            -np.clip(probability, 1e-12, 1.0 - 1e-12)
        )
        if not math.isclose(calibrated_logit, expected_logit, rel_tol=1e-5, abs_tol=1e-5):
            raise RuntimeError("Local contributions do not reproduce the calibrated output.")
        explanations.append(
            DomainLocalExplanation(
                domain=domain,
                method=method,
                model_output_logit=calibrated_logit,
                base_value=base_value,
                contributions=contributions,
                caregiver_message=_caregiver_message(
                    contributions,
                    decision=prediction_by_domain[domain].decision,
                ),
                limitation=(
                    "Feature contributions describe this fitted model, not causation, "
                    "clinical importance, or a diagnosis."
                ),
            )
        )
    return tuple(explanations)
