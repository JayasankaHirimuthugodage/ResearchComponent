"""Synthetic-only contracts for checkpoint-conditional TabICLv2 SHAP helpers."""
from __future__ import annotations

import inspect
import math
import unittest

import numpy as np
import pandas as pd

from inference.contracts import AssessmentStatus, QuestionnaireSubmission
from inference.questionnaire import encode_submission, load_questionnaire_catalog
from inference.tabiclv2_shap import (
    COGNITIVE_CALIBRATION_COEFFICIENT,
    COGNITIVE_CALIBRATION_INTERCEPT,
    COGNITIVE_FROZEN_THRESHOLD,
    EXPLAINABLE_RESPONSE_FEATURES,
    CheckpointConditionalCalibratedProbabilityWrapper,
    calibrated_cognitive_probability,
)
from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS, QUESTION_COLUMNS
from training.experiments.run_tabiclv2_shap_synthetic_pilot import MAX_EVALS, run_synthetic_pilot


class RecordingSyntheticClassifier:
    """Transparent deterministic stand-in; it accepts explanation-only NaNs."""

    def __init__(self, classes=(1, 0)):
        self.classes_ = np.asarray(classes, dtype=object)
        self.frames: list[pd.DataFrame] = []

    def predict_proba(self, frame):
        self.frames.append(frame.copy(deep=True))
        responses = frame.loc[:, QUESTION_COLUMNS].to_numpy(dtype=float)
        values = np.where(np.isnan(responses), 1.0, responses)
        raw_positive = np.clip(0.2 + 0.01 * values.sum(axis=1), 0.001, 0.999)
        # The class-1 column is deliberately first, testing non-default indexing.
        return np.column_stack((raw_positive, 1.0 - raw_positive))


class TabICLv2ShapWrapperTests(unittest.TestCase):
    def setUp(self):
        self.classifier = RecordingSyntheticClassifier()
        self.wrapper = CheckpointConditionalCalibratedProbabilityWrapper(self.classifier, "CP24M")
        self.responses = np.asarray([[0, 1, 2] * 5], dtype=float)

    def test_exactly_fifteen_ordered_explainable_features(self):
        self.assertEqual(15, len(EXPLAINABLE_RESPONSE_FEATURES))
        self.assertEqual(QUESTION_COLUMNS, EXPLAINABLE_RESPONSE_FEATURES)
        self.assertEqual(16, len(FEATURE_COLUMNS))

    def test_reconstruction_preserves_feature_order_and_categorical_checkpoint(self):
        frame = self.wrapper.reconstruct_features(self.responses)
        self.assertEqual(FEATURE_COLUMNS, tuple(frame.columns))
        self.assertIsInstance(frame[FEATURE_COLUMNS[0]].dtype, pd.CategoricalDtype)
        self.assertEqual(["CP24M"], frame[FEATURE_COLUMNS[0]].astype(str).tolist())
        self.assertEqual(list(range(15)), list(range(len(frame.columns) - 1)))
        np.testing.assert_allclose(frame.loc[:, QUESTION_COLUMNS].to_numpy(dtype=float), self.responses)

    def test_fixed_checkpoint_never_changes_for_masked_coalition(self):
        masked = self.responses.copy(); masked[0, [0, 4, 14]] = np.nan
        self.wrapper(masked)
        self.assertEqual({"CP24M"}, set(self.classifier.frames[-1][FEATURE_COLUMNS[0]].astype(str)))
        self.assertTrue(np.isnan(self.classifier.frames[-1].loc[:, QUESTION_COLUMNS].to_numpy(dtype=float)[0, [0, 4, 14]]).all())

    def test_positive_class_uses_fitted_label_not_first_column_assumption(self):
        output = self.wrapper(self.responses)
        raw = 0.2 + 0.01 * self.responses.sum(axis=1)
        np.testing.assert_allclose(output, calibrated_cognitive_probability(raw), atol=1e-15, rtol=0.0)

    def test_positive_class_can_be_second_column_when_classes_require_it(self):
        classifier = RecordingSyntheticClassifier(classes=(0, 1))
        wrapper = CheckpointConditionalCalibratedProbabilityWrapper(classifier, "CP24M")
        # The synthetic classifier's first column is labelled 0 in this case.
        output = wrapper(self.responses)
        raw_positive = 1.0 - (0.2 + 0.01 * self.responses.sum(axis=1))
        np.testing.assert_allclose(output, calibrated_cognitive_probability(raw_positive), atol=1e-15, rtol=0.0)

    def test_missing_ambiguous_or_duplicate_classes_are_rejected(self):
        for classes in ((0,), (0, 0), (2, 3), (1, 1)):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                CheckpointConditionalCalibratedProbabilityWrapper(RecordingSyntheticClassifier(classes), "CP24M")

    def test_malformed_inputs_are_rejected(self):
        for values in (np.zeros(15), np.zeros((1, 14)), np.asarray([[3] * 15], dtype=float), np.asarray([[np.inf] * 15])):
            with self.subTest(shape=np.asarray(values).shape), self.assertRaises(ValueError):
                self.wrapper(values)

    def test_nan_masked_outputs_are_finite_probabilities(self):
        masked = np.full((2, 15), np.nan)
        output = self.wrapper(masked)
        self.assertEqual((2,), output.shape)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertTrue(np.all((0.0 <= output) & (output <= 1.0)))

    def test_calibration_matches_independent_formula_and_is_not_thresholded(self):
        raw = np.asarray([0.0, 0.25, 0.5, 1.0])
        clipped = np.clip(raw, 1e-12, 1.0 - 1e-12)
        expected = 1.0 / (1.0 + np.exp(-(COGNITIVE_CALIBRATION_COEFFICIENT * (np.log(clipped) - np.log1p(-clipped)) + COGNITIVE_CALIBRATION_INTERCEPT)))
        np.testing.assert_allclose(calibrated_cognitive_probability(raw), expected, atol=1e-15, rtol=0.0)
        self.assertTrue(0.0 < COGNITIVE_FROZEN_THRESHOLD < 1.0)
        self.assertTrue(np.any(np.abs(calibrated_cognitive_probability(raw) - COGNITIVE_FROZEN_THRESHOLD) > 0.0))

    def test_unmasked_fidelity_and_batch_invariance_are_deterministic(self):
        single = self.wrapper(self.responses)[0]
        batched = self.wrapper(np.vstack((self.responses, self.responses)))[0]
        self.assertAlmostEqual(single, batched, places=15)
        self.assertEqual(2, len(self.classifier.frames))

    def test_incomplete_assessment_is_blocked_before_a_wrapper_is_requested(self):
        catalog = load_questionnaire_catalog(); checkpoint = "CP24M"
        responses = {question.question_id: "YES_USUALLY" for question in catalog.questions_for_checkpoint(checkpoint)}
        responses[next(iter(responses))] = "NOT_SURE"
        encoded = encode_submission(catalog, QuestionnaireSubmission(catalog.questionnaire_schema_version, checkpoint, responses))
        self.assertIs(encoded.status, AssessmentStatus.INCOMPLETE)
        self.assertIsNone(encoded.feature_values)
        self.assertEqual([], self.classifier.frames)

    def test_module_has_no_runtime_cache_or_threshold_decision_dependency(self):
        import inference.tabiclv2_shap as module
        source = inspect.getsource(module)
        self.assertNotIn("tabiclv2_kv_cache_runtime", source)
        self.assertNotIn("context_cache.get", source)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("load_cognitive_release_contract", source)
        self.assertNotIn("decision(", source)


class TabICLv2ShapPermutationSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_synthetic_pilot()

    def test_shap_output_budget_is_observed_not_assumed(self):
        self.assertEqual("shap.PermutationExplainer", self.result["algorithm"])
        self.assertEqual(15, self.result["feature_count"])
        self.assertEqual(6, self.result["permutation_cycles_planned"])
        self.assertEqual(MAX_EVALS, self.result["max_evals_requested"])
        self.assertLessEqual(self.result["max_shap_evaluation_rows"], MAX_EVALS)
        self.assertEqual(MAX_EVALS, self.result["max_shap_evaluation_rows"])
        self.assertGreater(self.result["max_shap_nan_masked_rows"], 0)

    def test_shap_probability_additivity_is_exact_for_deterministic_mock(self):
        self.assertLessEqual(self.result["max_reconstruction_error"], 1e-12)

    def test_seeded_repeat_stability_and_batch_invariance_are_exact_for_mock(self):
        self.assertLessEqual(self.result["max_repeat_attribution_abs_difference"], 1e-12)
        self.assertLessEqual(self.result["max_batch_invariance_abs_difference"], 1e-12)

    def test_all_three_fixed_checkpoints_are_preserved_and_bounded(self):
        self.assertEqual(3, self.result["assessments"])
        self.assertTrue(self.result["all_fixed_checkpoints_preserved"])
        self.assertLess(self.result["total_runtime_seconds"], 300.0)
        self.assertGreaterEqual(self.result["max_total_rows_including_controls"], MAX_EVALS)


if __name__ == "__main__":
    unittest.main()
