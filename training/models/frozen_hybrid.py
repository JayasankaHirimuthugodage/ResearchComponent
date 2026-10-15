"""Evaluation-only artifact for a frozen calibrated soft-voting hybrid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from training.experiments.calibration import _logit_feature


@dataclass(frozen=True)
class FrozenHybridDomainModel:
    domain: str
    logistic_pipeline: Pipeline
    logistic_calibrator: LogisticRegression
    xgboost_pipeline: Pipeline
    xgboost_calibrator: LogisticRegression
    logistic_weight: float
    xgboost_weight: float
    threshold: float
    feature_schema_version: str
    calibration_method: str
    hybrid_configuration_version: str

    def positive_probability(self, feature_values: tuple[str, ...]) -> float:
        matrix = np.asarray([feature_values], dtype=object)
        logistic_raw = self.logistic_pipeline.predict_proba(matrix)[:, 1]
        xgboost_raw = self.xgboost_pipeline.predict_proba(matrix)[:, 1]
        logistic_probability = self.logistic_calibrator.predict_proba(
            _logit_feature(logistic_raw)
        )[:, 1]
        xgboost_probability = self.xgboost_calibrator.predict_proba(
            _logit_feature(xgboost_raw)
        )[:, 1]
        probability = float(
            self.logistic_weight * logistic_probability[0]
            + self.xgboost_weight * xgboost_probability[0]
        )
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("The frozen hybrid returned an invalid probability.")
        return probability

    def decision(self, feature_values: tuple[str, ...]) -> bool:
        return self.positive_probability(feature_values) >= self.threshold
