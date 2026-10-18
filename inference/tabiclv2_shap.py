"""Synthetic-only helpers for future checkpoint-conditional TabICLv2 SHAP work.

This module deliberately does not construct a TabICLv2 model, load a checkpoint,
load the persisted context cache, or call an application prediction service.  A
caller must inject an already constructed classifier.  Stage B uses only test
doubles; use with a real model requires separate authorization and validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS, QUESTION_COLUMNS


COGNITIVE_CALIBRATION_COEFFICIENT = 1.069105693175016
COGNITIVE_CALIBRATION_INTERCEPT = 0.255897163477147
COGNITIVE_FROZEN_THRESHOLD = 0.1503614235173848
POSITIVE_COGNITIVE_CONCERN_CLASS = 1
EXPLAINABLE_RESPONSE_FEATURES = QUESTION_COLUMNS


class ProbabilityClassifier(Protocol):
    """The narrow classifier surface required by the synthetic wrapper."""

    classes_: object

    def predict_proba(self, values: pd.DataFrame) -> np.ndarray: ...


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Stable vector sigmoid matching the adapter's scalar branch behaviour."""

    output = np.empty_like(values, dtype=float)
    nonnegative = values >= 0.0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    output[~nonnegative] = exp_values / (1.0 + exp_values)
    return output


def calibrated_cognitive_probability(raw_positive_probability: np.ndarray) -> np.ndarray:
    """Apply the frozen release calibration once, without thresholding."""

    raw = np.asarray(raw_positive_probability, dtype=float)
    if raw.ndim != 1 or not np.all(np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("Raw positive-class probabilities must be finite values in [0, 1].")
    clipped = np.clip(raw, 1e-12, 1.0 - 1e-12)
    logit = np.log(clipped) - np.log1p(-clipped)
    calibrated = _sigmoid(COGNITIVE_CALIBRATION_COEFFICIENT * logit + COGNITIVE_CALIBRATION_INTERCEPT)
    if not np.all(np.isfinite(calibrated)) or np.any((calibrated < 0.0) | (calibrated > 1.0)):
        raise RuntimeError("Calibration produced invalid probabilities.")
    return calibrated


@dataclass(frozen=True)
class CheckpointConditionalCalibratedProbabilityWrapper:
    """Callable SHAP prediction function in calibrated probability units.

    The 15-column input matrix is intentionally limited to scoreable response
    values or NaNs.  NaNs are an *explanation-only* missingness/imputation
    baseline.  They never represent a production submission and are passed to
    the injected classifier's preprocessing boundary unchanged.
    """

    classifier: ProbabilityClassifier
    checkpoint_id: str
    positive_class: object = POSITIVE_COGNITIVE_CONCERN_CLASS

    def __post_init__(self) -> None:
        if self.checkpoint_id not in CHECKPOINT_IDS:
            raise ValueError("A valid fixed baseline checkpoint is required.")
        self._positive_class_index()

    def _positive_class_index(self) -> int:
        classes = np.asarray(getattr(self.classifier, "classes_", ()), dtype=object)
        if classes.ndim != 1 or len(classes) != 2:
            raise ValueError("Classifier classes_ must contain exactly two ordered labels.")
        matches = np.flatnonzero(classes == self.positive_class)
        if len(matches) != 1:
            raise ValueError("Classifier classes_ must contain the positive Cognitive-concern class exactly once.")
        if len(set(classes.tolist())) != 2:
            raise ValueError("Classifier classes_ labels must be distinct.")
        return int(matches[0])

    @staticmethod
    def _response_matrix(response_values: object) -> np.ndarray:
        values = np.asarray(response_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(EXPLAINABLE_RESPONSE_FEATURES):
            raise ValueError("SHAP requires a two-dimensional matrix with exactly 15 ordered response features.")
        valid = np.isnan(values) | np.isin(values, (0.0, 1.0, 2.0))
        if not np.all(valid):
            raise ValueError("Response features must be scoreable values {0, 1, 2} or explanation-only NaN masks.")
        return values

    def reconstruct_features(self, response_values: object) -> pd.DataFrame:
        """Reconstruct the categorical checkpoint plus 15 ordered responses."""

        responses = self._response_matrix(response_values)
        frame = pd.DataFrame(responses, columns=EXPLAINABLE_RESPONSE_FEATURES)
        frame.insert(0, FEATURE_COLUMNS[0], pd.Categorical([self.checkpoint_id] * len(frame), categories=CHECKPOINT_IDS))
        if tuple(frame.columns) != FEATURE_COLUMNS:
            raise RuntimeError("The reconstructed TabICLv2 feature contract is invalid.")
        return frame

    def __call__(self, response_values: object) -> np.ndarray:
        frame = self.reconstruct_features(response_values)
        probabilities = np.asarray(self.classifier.predict_proba(frame), dtype=float)
        positive_index = self._positive_class_index()
        if probabilities.ndim != 2 or probabilities.shape != (len(frame), 2):
            raise RuntimeError("Classifier predict_proba must return one two-class row per input.")
        raw_positive = probabilities[:, positive_index]
        return calibrated_cognitive_probability(raw_positive)

