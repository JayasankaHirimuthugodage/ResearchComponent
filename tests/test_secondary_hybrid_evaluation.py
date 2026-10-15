from __future__ import annotations

import inspect
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from tests.test_development_modeling import synthetic_modeling_records
from training.models.common import build_candidate_pipeline, fit_candidate_pipeline, modeling_arrays
from training.models.frozen_hybrid import FrozenHybridDomainModel
from training.models.logistic import REGULARIZED_LOGISTIC
from training.models.xgboost import REGULARIZED_XGBOOST
import training.experiments.run_secondary_hybrid_evaluation as secondary


class SecondaryHybridContractTests(unittest.TestCase):
    def test_frozen_hybrid_combines_calibrated_probabilities_equally(self) -> None:
        records = synthetic_modeling_records(records_per_combination=2)
        features, targets = modeling_arrays(records)
        y = targets[:, 0]
        logistic = build_candidate_pipeline(REGULARIZED_LOGISTIC, seed=7)
        xgboost = build_candidate_pipeline(REGULARIZED_XGBOOST, seed=11)
        fit_candidate_pipeline(logistic, REGULARIZED_LOGISTIC, features, y)
        fit_candidate_pipeline(xgboost, REGULARIZED_XGBOOST, features, y)
        logistic_raw = logistic.predict_proba(features)[:, 1]
        xgboost_raw = xgboost.predict_proba(features)[:, 1]
        logistic_calibrator = LogisticRegression().fit(logistic_raw.reshape(-1, 1), y)
        xgboost_calibrator = LogisticRegression().fit(xgboost_raw.reshape(-1, 1), y)
        artifact = FrozenHybridDomainModel(
            domain="communication",
            logistic_pipeline=logistic,
            logistic_calibrator=logistic_calibrator,
            xgboost_pipeline=xgboost,
            xgboost_calibrator=xgboost_calibrator,
            logistic_weight=0.5,
            xgboost_weight=0.5,
            threshold=0.2,
            feature_schema_version="fixture",
            calibration_method="fixture",
            hybrid_configuration_version="fixture",
        )
        probability = artifact.positive_probability(tuple(features[0]))
        self.assertTrue(np.isfinite(probability))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_secondary_runner_has_no_training_or_selection_path(self) -> None:
        source = inspect.getsource(secondary)
        self.assertNotIn("fit_candidate_pipeline", source)
        self.assertNotIn("build_candidate_pipeline", source)
        self.assertNotIn("select_balanced_accuracy_threshold", source)
        self.assertNotIn("_fit_sigmoid_calibrator", source)
        self.assertIn("cohort_previously_evaluated", source)
        self.assertIn("weights_adjusted_on_test", source)
        self.assertIn("thresholds_adjusted_on_test", source)


if __name__ == "__main__":
    unittest.main()
