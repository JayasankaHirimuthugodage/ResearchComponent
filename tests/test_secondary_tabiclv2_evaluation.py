from __future__ import annotations

import inspect
import unittest

import numpy as np

import training.experiments.run_secondary_tabiclv2_test_evaluation as secondary


class SecondaryTabICLv2EvaluationContractTests(unittest.TestCase):
    def test_frozen_calibration_matches_sigmoid_on_logit(self) -> None:
        raw = np.array([0.1, 0.5, 0.9])
        actual = secondary._calibrate(raw, 1.0, 0.0)
        np.testing.assert_allclose(raw, actual, atol=1e-12)

    def test_runner_is_one_time_aggregate_only_and_has_no_selection_path(self) -> None:
        source = inspect.getsource(secondary)
        self.assertNotIn("select_balanced_accuracy_threshold", source)
        self.assertNotIn("fit_candidate_pipeline", source)
        self.assertNotIn("equal_weight_hybrid", source)
        self.assertIn("preflight-only", source)
        self.assertIn("participant_identifiers_written", source)
        self.assertIn("row_level_predictions_written", source)
        self.assertIn("cohort_previously_evaluated", source)
        self.assertIn("thresholds_adjusted_on_test", source)

    def test_report_path_is_evaluation_specific(self) -> None:
        self.assertEqual(
            "reports/generated/secondary-tabiclv2-test-evaluation-report.json",
            secondary.REPORT_RELATIVE_PATH.as_posix(),
        )


if __name__ == "__main__":
    unittest.main()
