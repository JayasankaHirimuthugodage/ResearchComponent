"""Reusable preprocessing and estimator candidates for offline comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.contracts import (
    BINARY_VALUES,
    CHECKPOINT_IDS,
    FEATURE_COLUMNS,
    LOGISTIC_MODEL_CONFIGURATION_VERSION,
    QUESTION_COLUMNS,
    SCOREABLE_RESPONSE_VALUES,
)
from training.development_data import ModelingRecord


DEFAULT_MODEL_SEED = 20260822


@dataclass(frozen=True)
class CandidateSpec:
    """One comparable estimator configuration using the shared experiment contract."""

    name: str
    version: str
    estimator_factory: Callable[[int], BaseEstimator] = field(repr=False, compare=False)
    scale_questionnaire: bool = True
    public_parameters: Mapping[str, object] = field(default_factory=dict)


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

DEFAULT_CANDIDATES = (REGULARIZED_LOGISTIC,)


def modeling_arrays(
    records: Sequence[ModelingRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact 16-column X matrix and three-column binary Y matrix."""

    if not records:
        raise ValueError("At least one development record is required.")
    if any(len(record.feature_values) != len(FEATURE_COLUMNS) for record in records):
        raise ValueError("Every modelling row must match the frozen 16-input schema.")
    if any(len(record.target_values) != 3 for record in records):
        raise ValueError("Every modelling row must contain exactly three domain targets.")
    participant_keys = [record.participant_key for record in records]
    if any(not value for value in participant_keys):
        raise ValueError("Each modelling record requires a participant key for grouping.")
    if len(participant_keys) != len(set(participant_keys)):
        raise ValueError("The modelling input must contain one row per participant.")
    if any(record.feature_values[0] not in CHECKPOINT_IDS for record in records):
        raise ValueError("A baseline checkpoint is outside the frozen feature schema.")
    if any(
        value not in SCOREABLE_RESPONSE_VALUES
        for record in records
        for value in record.feature_values[1:]
    ):
        raise ValueError("Questionnaire inputs must be scoreable values in {0,1,2}.")
    if any(
        str(value) not in BINARY_VALUES
        for record in records
        for value in record.target_values
    ):
        raise ValueError("Domain targets must contain only binary values.")
    features = np.asarray([record.feature_values for record in records], dtype=object)
    targets = np.asarray([record.target_values for record in records], dtype=np.int8)
    if features.shape != (len(records), len(FEATURE_COLUMNS)):
        raise ValueError("Feature matrix does not match the frozen 16-input schema.")
    if targets.shape != (len(records), 3):
        raise ValueError("Target matrix must contain exactly three binary domains.")
    return features, targets


def build_preprocessor(*, scale_questionnaire: bool) -> ColumnTransformer:
    """Encode checkpoint and transform questionnaire values within each fitted fold."""

    questionnaire_transformer: str | StandardScaler
    questionnaire_transformer = StandardScaler() if scale_questionnaire else "passthrough"
    return ColumnTransformer(
        transformers=(
            (
                "checkpoint",
                OneHotEncoder(
                    categories=[list(CHECKPOINT_IDS)],
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                [0],
            ),
            (
                "questionnaire",
                questionnaire_transformer,
                list(range(1, len(FEATURE_COLUMNS))),
            ),
        ),
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_candidate_pipeline(candidate: CandidateSpec, *, seed: int) -> Pipeline:
    """Build a fresh pipeline so every fold learns preprocessing independently."""

    return Pipeline(
        steps=(
            (
                "preprocess",
                build_preprocessor(
                    scale_questionnaire=candidate.scale_questionnaire
                ),
            ),
            ("estimator", candidate.estimator_factory(seed)),
        )
    )


def transformed_feature_names(fitted_pipeline: Pipeline) -> tuple[str, ...]:
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out(FEATURE_COLUMNS)
    expected_count = len(CHECKPOINT_IDS) - 1 + len(QUESTION_COLUMNS)
    if len(names) != expected_count:
        raise RuntimeError("Transformed feature mapping is incomplete.")
    return tuple(str(name) for name in names)
