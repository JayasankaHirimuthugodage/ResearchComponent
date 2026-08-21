"""Predeclared regularized-logistic candidate configuration."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from training.contracts import LOGISTIC_MODEL_CONFIGURATION_VERSION
from training.models.common import CandidateSpec


def _balanced_logistic(seed: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2_000,
        random_state=seed,
    )


REGULARIZED_LOGISTIC = CandidateSpec(
    name="logistic_l2_balanced",
    version=LOGISTIC_MODEL_CONFIGURATION_VERSION,
    estimator_factory=_balanced_logistic,
    explanation_method="standardized_logistic_coefficient_stability",
    scale_questionnaire=True,
    public_parameters={
        "estimator": "LogisticRegression",
        "penalty": "l2",
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "liblinear",
        "max_iter": 2_000,
    },
)
