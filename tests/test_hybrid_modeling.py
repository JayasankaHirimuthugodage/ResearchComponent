from __future__ import annotations

import json
import unittest

import numpy as np

from tests.test_development_modeling import synthetic_modeling_records
from training.contracts import TARGET_COLUMNS
from training.experiments.hybrid_evaluation import run_hybrid_development_evaluation
from training.models.logistic import REGULARIZED_LOGISTIC
from training.models.soft_voting import soft_vote, validate_weight_grid
from training.models.xgboost import REGULARIZED_XGBOOST


class HybridModelingContractTests(unittest.TestCase):
    def test_soft_vote_validates_alignment_and_equal_weight_benchmark(self) -> None:
        first = np.asarray([0.2, 0.8])
        second = np.asarray([0.4, 0.6])
        self.assertTrue(
            np.allclose(
                np.asarray([0.3, 0.7]),
                soft_vote(first, second, first_weight=0.50),
            )
        )
        self.assertEqual((0.25, 0.50, 0.75), validate_weight_grid((0.75, 0.25, 0.50)))
        with self.assertRaisesRegex(ValueError, "equal-weight"):
            validate_weight_grid((0.25, 0.75))

    def test_nested_hybrid_report_is_aggregate_and_development_only(self) -> None:
        records = synthetic_modeling_records(records_per_combination=3)
        report = run_hybrid_development_evaluation(
            records,
            n_splits=2,
            n_repeats=1,
            inner_splits=2,
            cv_seed=101,
            fold_search_iterations=5,
            model_seed=103,
            bootstrap_replicates=100,
        )

        self.assertEqual("development_only_nested_hybrid_evaluation", report["scope"])
        self.assertFalse(report["final_test_evaluated"])
        self.assertEqual(16, report["feature_count"])
        self.assertEqual(set(TARGET_COLUMNS), set(report["domains"]))
        self.assertEqual(
            [REGULARIZED_LOGISTIC.name, REGULARIZED_XGBOOST.name],
            report["base_candidates"],
        )
        self.assertTrue(report["privacy"]["report_contains_aggregate_results_only"])
        for target in TARGET_COLUMNS:
            domain = report["domains"][target]
            self.assertEqual(2, len(domain["fold_results"]))
            self.assertIn("equal_weight_soft_voting", domain["methods"])
            self.assertIn("selected_weight_soft_voting", domain["methods"])
            self.assertIn(
                "decision_disagreement_rate", domain["complementarity"]
            )
            self.assertEqual(
                100,
                domain[
                    "paired_uncertainty_selected_hybrid_vs_existing_reference"
                ]["replicates"],
            )

        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("synthetic-", serialized)
        self.assertNotIn("participant_key", serialized)


if __name__ == "__main__":
    unittest.main()
