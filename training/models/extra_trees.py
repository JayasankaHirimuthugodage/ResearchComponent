"""Predeclared regularized Extra Trees candidate configuration."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier

from training.contracts import EXTRA_TREES_MODEL_CONFIGURATION_VERSION
from training.models.common import CandidateSpec


def _balanced_extra_trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=False,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


REGULARIZED_EXTRA_TREES = CandidateSpec(
    name="extra_trees_balanced_leaf5",
    version=EXTRA_TREES_MODEL_CONFIGURATION_VERSION,
    estimator_factory=_balanced_extra_trees,
    explanation_method="held_out_permutation_importance_average_precision",
    scale_questionnaire=False,
    public_parameters={
        "estimator": "ExtraTreesClassifier",
        "n_estimators": 300,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "bootstrap": False,
        "class_weight": "balanced",
        "n_jobs": 1,
    },
)
