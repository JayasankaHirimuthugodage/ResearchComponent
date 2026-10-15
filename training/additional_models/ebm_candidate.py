"""Predeclared main-effects-only EBM configurations."""

from __future__ import annotations

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from training.additional_models.contracts import AdditionalCandidateConfiguration
from training.contracts import FEATURE_COLUMNS


EBM_CONFIGURATIONS = (
    AdditionalCandidateConfiguration(
        candidate_name="ebm_main_effects_balanced",
        configuration_name="leaf8",
        configuration_version="ebm-main-effects-balanced-leaf8-v1",
        public_parameters={"min_samples_leaf": 8},
    ),
    AdditionalCandidateConfiguration(
        candidate_name="ebm_main_effects_balanced",
        configuration_name="leaf16",
        configuration_version="ebm-main-effects-balanced-leaf16-v1",
        public_parameters={"min_samples_leaf": 16},
    ),
)


EBM_FEATURE_TYPES: tuple[object, ...] = (
    "nominal",
    *(("0", "1", "2") for _ in FEATURE_COLUMNS[1:]),
)


def fit_ebm(
    configuration: AdditionalCandidateConfiguration,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> ExplainableBoostingClassifier:
    model = ExplainableBoostingClassifier(
        feature_names=list(FEATURE_COLUMNS),
        feature_types=list(EBM_FEATURE_TYPES),
        interactions=0,
        max_bins=16,
        outer_bags=8,
        inner_bags=0,
        learning_rate=0.015,
        max_rounds=5_000,
        early_stopping_rounds=100,
        min_samples_leaf=int(configuration.public_parameters["min_samples_leaf"]),
        max_leaves=2,
        n_jobs=1,
        random_state=seed,
    )
    weights = compute_sample_weight(class_weight="balanced", y=targets)
    model.fit(features, targets, sample_weight=weights)
    return model


def ebm_probabilities(
    model: ExplainableBoostingClassifier, features: np.ndarray
) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
