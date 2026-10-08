"""Predeclared regularized XGBoost candidate."""

from __future__ import annotations

from xgboost import XGBClassifier

from training.contracts import XGBOOST_MODEL_CONFIGURATION_VERSION
from training.models.common import CandidateSpec


def _regularized_xgboost(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


REGULARIZED_XGBOOST = CandidateSpec(
    name="xgboost_regularized_balanced",
    version=XGBOOST_MODEL_CONFIGURATION_VERSION,
    estimator_factory=_regularized_xgboost,
    explanation_method="held_out_permutation_importance_average_precision",
    scale_questionnaire=False,
    use_balanced_sample_weight=True,
    public_parameters={
        "estimator": "XGBClassifier",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "n_jobs": 1,
        "sample_weight": "balanced_from_training_fold",
    },
)
