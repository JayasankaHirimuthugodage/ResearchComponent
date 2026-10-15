from __future__ import annotations

import inspect
import json
import unittest

from tests.test_development_modeling import synthetic_modeling_records
from training.contracts import TARGET_COLUMNS
from training.experiments.additional_model_evaluation import (
    CandidateFamily,
    _fit_catboost_adapter,
    _fit_ebm_adapter,
    run_additional_model_evaluation,
)
from training.additional_models.catboost_candidate import CATBOOST_CONFIGURATIONS, catboost_probabilities
from training.additional_models.ebm_candidate import EBM_CONFIGURATIONS, ebm_probabilities
import training.experiments.run_additional_model_experiment as runner


class AdditionalModelEvaluationTests(unittest.TestCase):
    def test_new_candidates_use_nested_development_only_contract(self) -> None:
        families = (
            CandidateFamily(
                "catboost_ordered_balanced",
                (CATBOOST_CONFIGURATIONS[0],),
                _fit_catboost_adapter,
                catboost_probabilities,
                "catboost_tree_shap_raw_logit_then_sigmoid_calibration",
            ),
            CandidateFamily(
                "ebm_main_effects_balanced",
                (EBM_CONFIGURATIONS[0],),
                _fit_ebm_adapter,
                ebm_probabilities,
                "ebm_additive_terms_raw_logit_then_sigmoid_calibration",
            ),
        )
        report = run_additional_model_evaluation(
            synthetic_modeling_records(records_per_combination=3),
            families=families,
            n_splits=2,
            n_repeats=1,
            cv_seed=71,
            fold_search_iterations=5,
            model_seed=73,
        )
        self.assertEqual("development_only_nested_additional_model_comparison", report["scope"])
        self.assertFalse(report["final_test_evaluated"])
        self.assertEqual(16, report["feature_count"])
        self.assertEqual(set(TARGET_COLUMNS), set(report["domains"]))
        self.assertTrue(report["privacy"]["aggregate_results_only"])
        for domain in report["domains"].values():
            for candidate in domain["candidates"].values():
                error = candidate["maximum_calibrated_logit_reconstruction_error"]
                self.assertIsNotNone(error)
                self.assertLess(float(error), 1e-8)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("synthetic-", serialized)
        self.assertNotIn("participant_key", serialized)

    def test_runner_has_no_locked_test_loader(self) -> None:
        source = inspect.getsource(runner)
        self.assertNotIn("load_final_test_dataset", source)
        self.assertNotIn("final-locked-test-report", source)


if __name__ == "__main__":
    unittest.main()
