"""Framework-neutral runtime wrapper for a frozen calibrated domain model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from training.experiments.calibration import _logit_feature


@dataclass(frozen=True)
class CalibratedDomainModel:
    domain: str
    pipeline: Pipeline
    calibrator: LogisticRegression
    threshold: float
    model_configuration_version: str
    feature_schema_version: str
    calibration_method: str

    def positive_probability(self, feature_values: tuple[str, ...]) -> float:
        matrix = np.asarray([feature_values], dtype=object)
        raw_probability = self.pipeline.predict_proba(matrix)[:, 1]
        calibrated = self.calibrator.predict_proba(_logit_feature(raw_probability))[:, 1]
        probability = float(calibrated[0])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("The calibrated model returned an invalid probability.")
        return probability

    def decision(self, feature_values: tuple[str, ...]) -> bool:
        return self.positive_probability(feature_values) >= self.threshold
