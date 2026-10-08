"""Predeclared regularized random-forest candidate."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from training.contracts import RANDOM_FOREST_MODEL_CONFIGURATION_VERSION
from training.models.common import CandidateSpec


def _balanced_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


REGULARIZED_RANDOM_FOREST = CandidateSpec(
    name="random_forest_balanced_leaf5",
    version=RANDOM_FOREST_MODEL_CONFIGURATION_VERSION,
    estimator_factory=_balanced_random_forest,
    explanation_method="held_out_permutation_importance_average_precision",
    scale_questionnaire=False,
    public_parameters={
        "estimator": "RandomForestClassifier",
        "n_estimators": 300,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": "balanced",
        "n_jobs": 1,
    },
)
