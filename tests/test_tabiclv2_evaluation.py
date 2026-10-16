from __future__ import annotations

import inspect
import json
import unittest

import numpy as np

from tests.test_development_modeling import synthetic_modeling_records
from training.contracts import TARGET_COLUMNS
from training.experiments.cross_validation import create_shared_folds
from training.experiments.tabiclv2_evaluation import (
    fold_plan_fingerprint,
    run_tabiclv2_evaluation,
)
import training.experiments.run_tabiclv2_development_experiment as runner


def deterministic_probabilities(train_x, train_y, validation_x, seed):
    del train_x, train_y, seed
    numeric = validation_x[:, 1:].astype(float)
    score = np.sum(numeric, axis=1) / (2.0 * numeric.shape[1])
    return np.clip(0.05 + 0.9 * score, 0.0, 1.0)


def historical_stub(records, *, n_splits, n_repeats, seed):
    folds = create_shared_folds(
        records,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        search_iterations=5,
    )
    metric_names = (
        "average_precision_pr_auc",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "negative_predictive_value",
        "f1",
        "brier_score",
    )
    summary = {
        name: {"mean": 0.5, "standard_deviation": 0.0, "minimum": 0.5, "maximum": 0.5}
        for name in metric_names
    }
    references = {
        TARGET_COLUMNS[0]: "xgboost_regularized_balanced",
        TARGET_COLUMNS[1]: "logistic_l2_balanced",
        TARGET_COLUMNS[2]: "xgboost_regularized_balanced",
    }
    return {
        "development_participants": len(records),
        "cross_validation": {"shared_fold_plan_sha256": fold_plan_fingerprint(folds)},
        "domains": {
            target: {
                "historical_selected_reference": reference,
                "candidates": {reference: {"repeat_metric_summary": summary}},
            }
            for target, reference in references.items()
        },
    }


class TabICLv2EvaluationTests(unittest.TestCase):
    def test_development_only_nested_contract_is_aggregate_and_leakage_controlled(self) -> None:
        records = synthetic_modeling_records(records_per_combination=6)
        report = run_tabiclv2_evaluation(
            records,
            fit_predict=deterministic_probabilities,
            historical_report=historical_stub(records, n_splits=2, n_repeats=1, seed=71),
            n_splits=2,
            n_repeats=1,
            cv_seed=71,
            fold_search_iterations=5,
            model_seed=73,
        )
        self.assertEqual("development_only_exploratory_tabiclv2_evaluation", report["scope"])
        self.assertFalse(report["final_test_evaluated"])
        self.assertEqual(16, report["feature_count"])
        self.assertEqual(3, report["cross_validation"]["inner_splits"])
        self.assertTrue(report["privacy"]["aggregate_results_only"])
        for domain in report["domains"].values():
            self.assertEqual(2, len(domain["fold_results"]))
            self.assertFalse(domain["paired_uncertainty_available"])
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("synthetic-", serialized)
        self.assertNotIn("participant_key", serialized)

    def test_runner_has_no_locked_test_loader_or_hybrid_dependency(self) -> None:
        source = inspect.getsource(runner)
        self.assertNotIn("load_final_test_dataset", source)
        self.assertNotIn("hybrid_evaluation", source)
        self.assertNotIn("frozen_hybrid", source)


if __name__ == "__main__":
    unittest.main()

