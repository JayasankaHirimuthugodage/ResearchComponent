"""Bounded synthetic-only pilot for the TabICLv2 SHAP methodology.

This runner never imports the KV-cache runtime, checkpoint loader, data loaders,
or application services.  It prints aggregate diagnostics only and writes no
assessment vectors or participant information.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import psutil
import shap

from inference.tabiclv2_shap import CheckpointConditionalCalibratedProbabilityWrapper
from training.contracts import CHECKPOINT_IDS, QUESTION_COLUMNS


SHAP_SEED = 20260822
PERMUTATION_CYCLES = 6
FEATURE_COUNT = len(QUESTION_COLUMNS)
MAX_EVALS = PERMUTATION_CYCLES * (2 * FEATURE_COUNT + 1)
MAX_RUNTIME_SECONDS = 300.0


@dataclass
class SyntheticMissingnessClassifier:
    """Deterministic mock that deliberately receives NaNs at its boundary."""

    classes_: np.ndarray = field(default_factory=lambda: np.asarray([1, 0]))
    calls: int = 0
    rows: int = 0
    nan_rows: int = 0
    observed_checkpoints: set[str] = field(default_factory=set)

    def predict_proba(self, frame):
        self.calls += 1
        self.rows += len(frame)
        self.observed_checkpoints.update(str(value) for value in frame["baseline_checkpoint_id"].astype(str))
        responses = frame.loc[:, QUESTION_COLUMNS].to_numpy(dtype=float)
        self.nan_rows += int(np.isnan(responses).any(axis=1).sum())
        # Fixed synthetic preprocessing: each NaN is imputed to 1.0.  This is
        # only a transparent mock assumption, not fitted TabICLv2 behaviour.
        imputed = np.where(np.isnan(responses), 1.0, responses)
        checkpoint_offsets = np.asarray([CHECKPOINT_IDS.index(str(value)) / 100.0 for value in frame["baseline_checkpoint_id"].astype(str)])
        raw_positive = np.clip(0.12 + 0.018 * imputed.sum(axis=1) + checkpoint_offsets, 0.001, 0.999)
        return np.column_stack((raw_positive, 1.0 - raw_positive))


def _single_run(checkpoint_id: str, assessment: np.ndarray) -> dict[str, float | int | bool]:
    classifier = SyntheticMissingnessClassifier()
    wrapper = CheckpointConditionalCalibratedProbabilityWrapper(classifier, checkpoint_id)
    background = np.full((1, FEATURE_COUNT), np.nan, dtype=float)
    start = time.monotonic()
    explainer = shap.PermutationExplainer(wrapper, shap.maskers.Independent(background, max_samples=1), seed=SHAP_SEED)
    result = explainer(assessment.reshape(1, FEATURE_COUNT), max_evals=MAX_EVALS)
    shap_wrapper_calls = classifier.calls
    shap_evaluation_rows = classifier.rows
    shap_nan_masked_rows = classifier.nan_rows
    elapsed = time.monotonic() - start
    if elapsed > MAX_RUNTIME_SECONDS:
        raise TimeoutError("Synthetic SHAP pilot exceeded its five-minute runtime bound.")
    values = np.asarray(result.values, dtype=float)
    base = np.asarray(result.base_values, dtype=float).reshape(-1)
    prediction = float(wrapper(assessment.reshape(1, FEATURE_COUNT))[0])
    reconstruction = float(base[0] + values.reshape(1, FEATURE_COUNT).sum(axis=1)[0])
    if values.shape != (1, FEATURE_COUNT) or not np.all(np.isfinite(values)):
        raise RuntimeError("Unexpected non-finite synthetic SHAP output shape.")
    if not np.isfinite(reconstruction) or abs(prediction - reconstruction) > 1e-12:
        raise RuntimeError("Synthetic SHAP probability reconstruction failed.")
    if shap_evaluation_rows > MAX_EVALS:
        raise RuntimeError("SHAP evaluation-row bound was exceeded.")
    if shap_nan_masked_rows == 0 or classifier.observed_checkpoints != {checkpoint_id}:
        raise RuntimeError("Synthetic missingness or fixed-checkpoint contract was not observed.")
    # Repeat the same seeded computation with fresh instrumentation.
    repeated_classifier = SyntheticMissingnessClassifier()
    repeated_wrapper = CheckpointConditionalCalibratedProbabilityWrapper(repeated_classifier, checkpoint_id)
    repeated = shap.PermutationExplainer(repeated_wrapper, shap.maskers.Independent(background, max_samples=1), seed=SHAP_SEED)(assessment.reshape(1, FEATURE_COUNT), max_evals=MAX_EVALS)
    repeat_difference = float(np.max(np.abs(values - np.asarray(repeated.values, dtype=float))))
    batch = np.vstack((assessment, assessment))
    batch_output = wrapper(batch)
    batch_difference = float(abs(batch_output[0] - prediction))
    return {
        "shap_wrapper_calls": shap_wrapper_calls,
        "shap_evaluation_rows": shap_evaluation_rows,
        "shap_nan_masked_rows": shap_nan_masked_rows,
        "total_wrapper_calls_including_controls": classifier.calls,
        "total_rows_including_controls": classifier.rows,
        "reconstruction_error": abs(prediction - reconstruction),
        "repeat_attribution_max_abs_difference": repeat_difference,
        "batch_invariance_abs_difference": batch_difference,
        "runtime_seconds": elapsed,
        "fixed_checkpoint_preserved": classifier.observed_checkpoints == {checkpoint_id},
    }


def run_synthetic_pilot() -> dict[str, object]:
    """Run three artificial complete assessments; return aggregate diagnostics."""

    assessments = (
        ("CP6M", np.asarray([0, 1, 2] * 5, dtype=float)),
        ("CP24M", np.asarray([2, 1, 0] * 5, dtype=float)),
        ("CP60M", np.asarray([1] * FEATURE_COUNT, dtype=float)),
    )
    process = psutil.Process()
    before = process.memory_info().rss
    started = time.monotonic()
    runs = [_single_run(checkpoint, assessment) for checkpoint, assessment in assessments]
    total_seconds = time.monotonic() - started
    if total_seconds > MAX_RUNTIME_SECONDS:
        raise TimeoutError("Synthetic SHAP pilot exceeded its five-minute total runtime bound.")
    after = process.memory_info().rss
    return {
        "algorithm": "shap.PermutationExplainer",
        "shap_version": shap.__version__,
        "seed": SHAP_SEED,
        "feature_count": FEATURE_COUNT,
        "permutation_cycles_planned": PERMUTATION_CYCLES,
        "max_evals_requested": MAX_EVALS,
        "assessments": len(runs),
        "max_shap_wrapper_calls": max(int(run["shap_wrapper_calls"]) for run in runs),
        "max_shap_evaluation_rows": max(int(run["shap_evaluation_rows"]) for run in runs),
        "max_shap_nan_masked_rows": max(int(run["shap_nan_masked_rows"]) for run in runs),
        "max_total_rows_including_controls": max(int(run["total_rows_including_controls"]) for run in runs),
        "max_reconstruction_error": max(float(run["reconstruction_error"]) for run in runs),
        "max_repeat_attribution_abs_difference": max(float(run["repeat_attribution_max_abs_difference"]) for run in runs),
        "max_batch_invariance_abs_difference": max(float(run["batch_invariance_abs_difference"]) for run in runs),
        "all_fixed_checkpoints_preserved": all(bool(run["fixed_checkpoint_preserved"]) for run in runs),
        "total_runtime_seconds": total_seconds,
        "rss_delta_bytes": max(0, after - before),
    }


if __name__ == "__main__":
    print(json.dumps(run_synthetic_pilot(), sort_keys=True))
