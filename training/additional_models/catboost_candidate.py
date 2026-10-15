"""Predeclared CatBoost configurations for development-only comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from training.additional_models.contracts import AdditionalCandidateConfiguration
from training.contracts import FEATURE_COLUMNS


CATBOOST_CONFIGURATIONS = (
    AdditionalCandidateConfiguration(
        candidate_name="catboost_ordered_balanced",
        configuration_name="depth3_l2_10",
        configuration_version="catboost-ordered-balanced-depth3-v1",
        public_parameters={"depth": 3, "l2_leaf_reg": 10.0},
    ),
    AdditionalCandidateConfiguration(
        candidate_name="catboost_ordered_balanced",
        configuration_name="depth4_l2_10",
        configuration_version="catboost-ordered-balanced-depth4-v1",
        public_parameters={"depth": 4, "l2_leaf_reg": 10.0},
    ),
)


def catboost_frame(features: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    frame[FEATURE_COLUMNS[0]] = frame[FEATURE_COLUMNS[0]].astype(str)
    for column in FEATURE_COLUMNS[1:]:
        frame[column] = frame[column].astype(np.int8)
    return frame


def fit_catboost(
    configuration: AdditionalCandidateConfiguration,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.03,
        depth=int(configuration.public_parameters["depth"]),
        l2_leaf_reg=float(configuration.public_parameters["l2_leaf_reg"]),
        boosting_type="Ordered",
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        cat_features=[FEATURE_COLUMNS[0]],
    )
    weights = compute_sample_weight(class_weight="balanced", y=targets)
    model.fit(catboost_frame(features), targets, sample_weight=weights)
    return model


def catboost_probabilities(model: CatBoostClassifier, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(catboost_frame(features))[:, 1], dtype=float)
