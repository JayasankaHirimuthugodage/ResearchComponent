"""No-data contracts for the isolated Stage C runner."""
from __future__ import annotations

import inspect
import json
import tempfile
import unittest

import numpy as np
import pandas as pd
import shap

from training.contracts import FEATURE_COLUMNS, QUESTION_COLUMNS
from training.experiments.run_tabiclv2_shap_stage_c_verification import (
    CACHE_BYTES,
    CACHE_SHA256,
    MAX_EVALS,
    TOLERANCE,
    RecordingRealClassifier,
    RecordingShapCallable,
    StageCVerificationError,
    _diagnostic,
    _success_diagnostic,
    persist_diagnostic,
    shap_sequence_accounting,
    validate_shap_acceptance,
    validate_shap_acceptance,
)


class _FakeRealModel:
    classes_ = np.asarray([0, 1])

    def __init__(self):
        self.frames = []

    def predict_proba(self, frame):
        self.frames.append(frame.copy(deep=True))
        return np.tile(np.asarray([[0.7, 0.3]]), (len(frame), 1))


class TabICLv2ShapStageCContractTests(unittest.TestCase):
    class _Explanation:
        def __init__(self, values, base_values):
            self.values = values
            self.base_values = base_values

    def _failure_diagnostic(self, explanation, *, prediction=0.4, rows=186):
        with self.assertRaises(StageCVerificationError) as caught:
            validate_shap_acceptance(
                explanation=explanation,
                calibrated_prediction=prediction,
                shap_rows=rows,
                wrapper_calls=6,
                unmasked_fidelity_error=0.0,
                batch_invariance_error=0.0,
                runtime_seconds=1.0,
                rss_delta_bytes=0,
                completed_assessments=0,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = __import__('pathlib').Path(directory) / 'diagnostic.json'
            persist_diagnostic(path, caught.exception.diagnostic)
            payload = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual('failed', payload['status'])
        self.assertNotIn('responses', json.dumps(payload).lower())
        self.assertNotIn('cache_path', json.dumps(payload).lower())
        return payload

    def test_stage_c_constants_preserve_predeclared_safeguards(self):
        self.assertEqual(259554811, CACHE_BYTES)
        self.assertEqual("96E6D812AB977B0E065329C33563A91F02856EBF3BC2D3F0B2DE2020CF9682F0", CACHE_SHA256)
        self.assertEqual(186, MAX_EVALS)
        self.assertEqual(1e-6, TOLERANCE)

    def test_recording_adapter_delegates_without_feature_reordering(self):
        model = _FakeRealModel(); recorder = RecordingRealClassifier(model)
        frame = pd.DataFrame([["CP24M", *([1.0] * 15)]], columns=FEATURE_COLUMNS)
        frame[FEATURE_COLUMNS[0]] = pd.Categorical(frame[FEATURE_COLUMNS[0]], categories=("CP24M",))
        frame.loc[0, QUESTION_COLUMNS[2]] = np.nan
        probabilities = recorder.predict_proba(frame)
        self.assertEqual((1, 2), probabilities.shape)
        self.assertEqual(1, recorder.calls); self.assertEqual(1, recorder.rows); self.assertEqual(1, recorder.nan_rows)
        self.assertEqual({"CP24M"}, recorder.checkpoints)
        self.assertEqual(FEATURE_COLUMNS, tuple(model.frames[0].columns))
        self.assertTrue(np.isnan(model.frames[0].loc[0, QUESTION_COLUMNS[2]]))

    def test_runner_has_no_fitting_csv_or_application_activation_path(self):
        import training.experiments.run_tabiclv2_shap_stage_c_verification as module
        source = inspect.getsource(module)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("load_authorized_cognitive_context", source)
        self.assertNotIn("build_kv_cache_experimental_runtime_service", source)
        self.assertNotIn("Streamlit", source)

    def test_incorrect_shap_values_shape_persists_diagnostic(self):
        payload = self._failure_diagnostic(self._Explanation(np.zeros((1, 14)), np.asarray([0.1])))
        self.assertEqual([1, 14], payload['shap_values_shape'])
        self.assertFalse(payload['conditions']['values_shape_is_one_by_fifteen'])

    def test_incorrect_base_value_shape_persists_diagnostic(self):
        payload = self._failure_diagnostic(self._Explanation(np.zeros((1, 15)), np.asarray([0.1, 0.2])))
        self.assertEqual([2], payload['shap_base_values_shape'])
        self.assertFalse(payload['conditions']['base_value_is_scalar_per_assessment'])

    def test_excessive_rows_persists_budget_diagnostic(self):
        payload = self._failure_diagnostic(self._Explanation(np.zeros((1, 15)), np.asarray([0.4])), rows=187)
        self.assertEqual(187, payload['shap_evaluation_rows'])
        self.assertFalse(payload['conditions']['evaluation_rows_within_budget'])

    def test_additivity_failure_persists_reconstruction_error(self):
        payload = self._failure_diagnostic(self._Explanation(np.ones((1, 15)), np.asarray([0.0])))
        self.assertGreater(payload['maximum_reconstruction_error'], TOLERANCE)
        self.assertFalse(payload['conditions']['additivity_within_tolerance'])

    def test_invalid_numerical_output_persists_diagnostic(self):
        payload = self._failure_diagnostic(self._Explanation(np.full((1, 15), np.nan), np.asarray([0.1])))
        self.assertFalse(payload['conditions']['values_are_finite'])

    def test_unexpected_exception_category_can_be_persisted_without_inputs(self):
        diagnostic = _diagnostic('unexpected_exception', error_category='RuntimeError', conditions={'unexpected_exception': False})
        with tempfile.TemporaryDirectory() as directory:
            path = __import__('pathlib').Path(directory) / 'unexpected.json'
            persist_diagnostic(path, diagnostic)
            payload = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual('RuntimeError', payload['error_category'])
        self.assertEqual('unavailable', payload['shap_values_shape'])
        self.assertEqual('unavailable', payload['maximum_same_seed_attribution_difference'])

    def test_success_diagnostic_is_aggregate_only_and_persistable(self):
        maximums = {'additivity_error': 0.0, 'internal_shap_reconstruction_error': 0.0, 'shap_sequence_vs_standalone_error': 0.0, 'unmasked_fidelity_error': 0.0, 'batch_invariance_error': 0.0, 'repeatability_error': 0.0, 'shap_evaluation_rows': 186, 'shap_wrapper_calls': 6, 'sequence_baseline_start_range': 0.0, 'sequence_baseline_end_range': 0.0, 'runtime_seconds': 1.0, 'rss_delta_bytes': 0}
        diagnostic = _success_diagnostic([{}], maximums)
        with tempfile.TemporaryDirectory() as directory:
            path = __import__('pathlib').Path(directory) / 'success.json'
            persist_diagnostic(path, diagnostic)
            self.assertEqual([1, 15], diagnostic['shap_values_shape'])
            self.assertEqual([1], diagnostic['shap_base_values_shape'])
            self.assertEqual(6, diagnostic['maximum_shap_wrapper_calls'])
            payload = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual('passed', payload['status'])
        self.assertEqual(0.0, payload['maximum_internal_shap_reconstruction_error'])
        self.assertNotIn('responses', json.dumps(payload).lower())

    def test_shap_sequence_accounting_matches_exact_permutation_reconstruction(self):
        feature_count = 15
        outputs = [np.linspace(0.1 + repeat * 0.001, 0.4 + repeat * 0.001, 2 * feature_count + 1) for repeat in range(6)]
        accounting = shap_sequence_accounting(outputs, feature_count)
        expected = outputs[-1][0]
        contribution_sum = np.mean([2 * row[feature_count] - row[0] - row[-1] for row in outputs]) / 2
        self.assertEqual('available', accounting['status'])
        self.assertAlmostEqual(expected, accounting['sequence_expected_value'])
        self.assertAlmostEqual(contribution_sum, accounting['sequence_contribution_sum'])

    def test_batch_dependent_rounding_can_separate_shap_sequence_from_standalone(self):
        class BatchDependentCallable:
            def __call__(self, values):
                values = np.asarray(values, dtype=np.float64)
                base = np.nan_to_num(values, nan=1.0).sum(axis=1) / 100.0
                return base + len(values) * 5e-8

        sample = np.asarray([[0, 1, 2] * 5], dtype=np.float64)
        recording = RecordingShapCallable(BatchDependentCallable())
        explainer = shap.PermutationExplainer(recording, shap.maskers.Independent(np.full((1, 15), np.nan)), seed=20260822)
        explanation = explainer(sample, max_evals=186)
        accounting = shap_sequence_accounting(recording.outputs)
        shap_reconstruction = float(np.asarray(explanation.base_values).reshape(-1)[0] + np.asarray(explanation.values).sum())
        standalone = float(BatchDependentCallable()(sample)[0])
        self.assertEqual('available', accounting['status'])
        self.assertAlmostEqual(shap_reconstruction, accounting['sequence_reconstruction'], places=12)
        self.assertGreater(abs(shap_reconstruction - standalone), 1e-6)

    def test_diagnostic_distinguishes_sequence_accounting_from_standalone_failure(self):
        class BatchDependentCallable:
            def __call__(self, values):
                values = np.asarray(values, dtype=np.float64)
                return np.nan_to_num(values, nan=1.0).sum(axis=1) / 100.0 + len(values) * 5e-8

        sample = np.asarray([[0, 1, 2] * 5], dtype=np.float64)
        recording = RecordingShapCallable(BatchDependentCallable())
        explanation = shap.PermutationExplainer(recording, shap.maskers.Independent(np.full((1, 15), np.nan)), seed=20260822)(sample, max_evals=186)
        accounting = shap_sequence_accounting(recording.outputs)
        standalone = float(BatchDependentCallable()(sample)[0])
        with self.assertRaises(StageCVerificationError) as caught:
            validate_shap_acceptance(
                explanation=explanation,
                calibrated_prediction=standalone,
                shap_rows=186,
                wrapper_calls=6,
                unmasked_fidelity_error=0.0,
                batch_invariance_error=0.0,
                runtime_seconds=1.0,
                rss_delta_bytes=0,
                completed_assessments=0,
                sequence_accounting=accounting,
            )
        diagnostic = caught.exception.diagnostic
        self.assertLessEqual(diagnostic['sequence_reconstruction_error'], 1e-12)
        self.assertGreater(diagnostic['sequence_vs_independent_prediction_error'], 1e-6)
        self.assertFalse(diagnostic['conditions']['additivity_within_tolerance'])

    def test_sequence_accounting_handles_float32_outputs_without_structural_failure(self):
        outputs = [np.linspace(0.1, 0.4, 31, dtype=np.float32) for _ in range(6)]
        accounting = shap_sequence_accounting(outputs)
        self.assertEqual('available', accounting['status'])
        self.assertTrue(np.isfinite(accounting['sequence_reconstruction']))

    def test_sequence_accounting_marks_unexpected_batch_shape_unavailable(self):
        self.assertEqual('unavailable', shap_sequence_accounting([np.zeros(30)])['status'])


if __name__ == "__main__":
    unittest.main()
