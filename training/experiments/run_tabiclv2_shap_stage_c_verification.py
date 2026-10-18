"""Bounded real-cache Stage C verification for TabICLv2 SHAP.

This runner is deliberately independent of the three-domain service because the
isolated XAI environment does not install its unrelated XGBoost stack. It reads
only the approved cache and its non-identifying release metadata, verifies every
contract before deserialization, and evaluates fixed synthetic questionnaires.
It writes only an aggregate diagnostic on controlled failure.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil
import shap
from tabicl import TabICLClassifier

from inference.tabiclv2_adapter import CognitiveReleaseContract, TabICLv2CognitiveAdapter
from inference.tabiclv2_context import InMemoryContextCache
from inference.tabiclv2_shap import CheckpointConditionalCalibratedProbabilityWrapper
from training.contracts import FEATURE_COLUMNS, QUESTION_COLUMNS
from training.tabiclv2.candidate import (
    TABICLV2_CHECKPOINT_NAME,
    TABICLV2_CHECKPOINT_SHA256,
    TABICLV2_PACKAGE_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = Path(r"F:\4thYear\2026July\Research_Protected\TabICLv2_Cognitive_KV_Cache_Engineering_Verification_20261017\cognitive-tabiclv2-kv-cache.pkl")
CACHE_SHA256 = "96E6D812AB977B0E065329C33563A91F02856EBF3BC2D3F0B2DE2020CF9682F0"
CACHE_BYTES = 259554811
CHECKPOINT = ROOT / ".venv-tabiclv2/hf-cache/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt"
CACHE_MANIFEST = ROOT / "models/experimental-cognitive-tabiclv2-kv-cache-v1/release-manifest.json"
COGNITIVE_MANIFEST = ROOT / "models/experimental-cognitive-tabiclv2-v1/release-manifest.json"
INTEGRITY_INDEX = ROOT / "models/experimental-cognitive-tabiclv2-v1/integrity-index.json"
FINALIZATION_REPORT = ROOT / "reports/generated/development-cognitive-tabiclv2-finalization-report.json"
MAX_EVALS = 186
SEED = 20260822
MAX_RUNTIME_SECONDS = 300.0
TOLERANCE = 1e-6
DEFAULT_DIAGNOSTIC_PATH = ROOT / "reports/generated/tabiclv2-shap-stage-c-diagnostic.json"
DIAGNOSTIC_SCHEMA = "tabiclv2-shap-stage-c-diagnostic-v1"


class StageCVerificationError(RuntimeError):
    """Failure carrying only aggregate, non-sensitive diagnostic fields."""

    def __init__(self, message: str, diagnostic: dict[str, object]):
        super().__init__(message)
        self.diagnostic = diagnostic


def _versions() -> dict[str, str]:
    import numpy
    import torch

    return {
        "shap": shap.__version__,
        "tabicl": importlib.metadata.version("tabicl"),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }


def _diagnostic(
    stage: str,
    *,
    values_shape: tuple[int, ...] | None = None,
    base_values_shape: tuple[int, ...] | None = None,
    actual_feature_count: int | None = None,
    shap_rows: int | None = None,
    wrapper_calls: int | None = None,
    completed_assessments: int = 0,
    reconstruction_error: float | None = None,
    unmasked_fidelity_error: float | None = None,
    batch_invariance_error: float | None = None,
    repeatability_error: float | None = None,
    sequence_reconstruction_error: float | None = None,
    sequence_vs_independent_error: float | None = None,
    sequence_accounting: dict[str, float | int | str] | None = None,
    runtime_seconds: float | None = None,
    rss_delta_bytes: int | None = None,
    conditions: dict[str, bool] | None = None,
    error_category: str | None = None,
) -> dict[str, object]:
    """Build a JSON-safe report without paths, inputs, predictions, or matrices."""

    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "failed",
        "stage": stage,
        "error_category": error_category or "acceptance_failure",
        "shap_values_shape": list(values_shape) if values_shape is not None else "unavailable",
        "shap_base_values_shape": list(base_values_shape) if base_values_shape is not None else "unavailable",
        "expected_feature_count": 15,
        "actual_feature_count": actual_feature_count if actual_feature_count is not None else "unavailable",
        "shap_evaluation_rows": shap_rows if shap_rows is not None else "unavailable",
        "wrapper_calls": wrapper_calls if wrapper_calls is not None else "unavailable",
        "synthetic_assessments_completed": completed_assessments,
        "maximum_reconstruction_error": reconstruction_error if reconstruction_error is not None else "unavailable",
        "maximum_unmasked_fidelity_error": unmasked_fidelity_error if unmasked_fidelity_error is not None else "unavailable",
        "maximum_batch_invariance_error": batch_invariance_error if batch_invariance_error is not None else "unavailable",
        "maximum_same_seed_attribution_difference": repeatability_error if repeatability_error is not None else "unavailable",
        "sequence_reconstruction_error": sequence_reconstruction_error if sequence_reconstruction_error is not None else "unavailable",
        "sequence_vs_independent_prediction_error": sequence_vs_independent_error if sequence_vs_independent_error is not None else "unavailable",
        "sequence_permutation_batches": sequence_accounting.get("permutation_batches", "unavailable") if sequence_accounting else "unavailable",
        "sequence_baseline_start_range": sequence_accounting.get("sequence_baseline_start_range", "unavailable") if sequence_accounting else "unavailable",
        "sequence_baseline_end_range": sequence_accounting.get("sequence_baseline_end_range", "unavailable") if sequence_accounting else "unavailable",
        "requested_max_evals": MAX_EVALS,
        "actual_evaluation_budget_rows": shap_rows if shap_rows is not None else "unavailable",
        "tolerance": TOLERANCE,
        "conditions": conditions or {},
        "versions": _versions(),
        "runtime_seconds": runtime_seconds if runtime_seconds is not None else "unavailable",
        "rss_delta_bytes": rss_delta_bytes if rss_delta_bytes is not None else "unavailable",
        "privacy": {"raw_questionnaire_vectors_persisted": False, "participant_data_accessed": False, "sensitive_paths_persisted": False},
    }


def persist_diagnostic(path: Path, diagnostic: dict[str, object]) -> None:
    """Atomically persist only the allowlisted aggregate diagnostic schema."""

    if diagnostic.get("schema") != DIAGNOSTIC_SCHEMA:
        raise ValueError("Refusing to persist an invalid Stage C diagnostic.")
    forbidden = ("cache", "checkpoint", "response", "prediction", "path")
    serialized = json.dumps(diagnostic, sort_keys=True, indent=2)
    if any(token in serialized.lower() for token in forbidden if token not in {"prediction", "path"}):
        # The schema intentionally contains neither cache/checkpoint identifiers
        # nor raw response data. Prediction/path are permitted only in fixed key
        # names above and are not emitted as values.
        raise ValueError("Diagnostic contains forbidden sensitive content.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _fail(message: str, diagnostic: dict[str, object]) -> None:
    raise StageCVerificationError(message, diagnostic)


def _success_diagnostic(results: list[dict[str, object]], maximums: dict[str, float | int]) -> dict[str, object]:
    """Create a sanitized, aggregate-only success record for a future run."""

    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "passed",
        "stage": "completed",
        "error_category": "none",
        "synthetic_assessments_completed": len(results),
        "expected_feature_count": 15,
        "actual_feature_count": 15,
        "shap_values_shape": [1, 15],
        "shap_base_values_shape": [1],
        "requested_max_evals": MAX_EVALS,
        "actual_evaluation_budget_rows": maximums["shap_evaluation_rows"],
        "tolerance": TOLERANCE,
        "maximum_independent_reconstruction_error": maximums["additivity_error"],
        "maximum_internal_shap_reconstruction_error": maximums["internal_shap_reconstruction_error"],
        "maximum_shap_sequence_vs_standalone_error": maximums["shap_sequence_vs_standalone_error"],
        "maximum_unmasked_fidelity_error": maximums["unmasked_fidelity_error"],
        "maximum_batch_invariance_error": maximums["batch_invariance_error"],
        "maximum_same_seed_attribution_difference": maximums["repeatability_error"],
        "maximum_shap_evaluation_rows": maximums["shap_evaluation_rows"],
        "maximum_shap_wrapper_calls": maximums["shap_wrapper_calls"],
        "maximum_sequence_baseline_start_range": maximums["sequence_baseline_start_range"],
        "maximum_sequence_baseline_end_range": maximums["sequence_baseline_end_range"],
        "runtime_seconds": maximums["runtime_seconds"],
        "rss_delta_bytes": maximums["rss_delta_bytes"],
        "conditions": {"all_original_stage_c_acceptance_criteria_passed": True},
        "versions": _versions(),
        "interpretation": "Bounded development-only numerical compatibility evidence; not clinical validation.",
        "privacy": {"raw_questionnaire_vectors_persisted": False, "participant_data_accessed": False, "sensitive_paths_persisted": False},
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def preflight() -> CognitiveReleaseContract:
    """Verify all trusted artifacts before the cache is deserialized."""

    if not CACHE_PATH.is_file() or CACHE_PATH.resolve() != Path(r"F:\4thYear\2026July\Research_Protected\TabICLv2_Cognitive_KV_Cache_Engineering_Verification_20261017\cognitive-tabiclv2-kv-cache.pkl").resolve():
        raise ValueError("Stage C cache path is not the approved protected artifact.")
    if CACHE_PATH.stat().st_size != CACHE_BYTES or sha256(CACHE_PATH) != CACHE_SHA256:
        raise ValueError("Stage C cache integrity verification failed.")
    if CHECKPOINT.name != TABICLV2_CHECKPOINT_NAME or not CHECKPOINT.is_file() or sha256(CHECKPOINT) != TABICLV2_CHECKPOINT_SHA256:
        raise ValueError("Stage C checkpoint verification failed.")
    if importlib.metadata.version("tabicl") != TABICLV2_PACKAGE_VERSION:
        raise ValueError("Stage C TabICL version differs from the frozen release.")
    cache_manifest = _read_json(CACHE_MANIFEST)
    cognitive_manifest = _read_json(COGNITIVE_MANIFEST)
    index = _read_json(INTEGRITY_INDEX)
    if cache_manifest.get("schema") != "experimental-cognitive-tabiclv2-kv-cache-release-v1" or cache_manifest.get("status") != "development_only_experimental_not_deployed":
        raise ValueError("Stage C cache release manifest is invalid.")
    artifact = cache_manifest.get("cache_artifact")
    if not isinstance(artifact, dict) or artifact != {"path": str(CACHE_PATH), "sha256": CACHE_SHA256, "bytes": CACHE_BYTES}:
        raise ValueError("Stage C cache manifest does not bind the approved artifact.")
    if tuple(cache_manifest.get("feature_contract", ())) != FEATURE_COLUMNS:
        raise ValueError("Stage C cache manifest feature contract is invalid.")
    config = cache_manifest.get("configuration")
    expected_config = {"n_estimators": 8, "device": "cpu", "use_amp": False, "use_fa3": False, "offload_mode": False, "random_state": 20260822, "n_jobs": 4, "allow_auto_download": False}
    if config != expected_config:
        raise ValueError("Stage C cache configuration differs from the approved release.")
    calibration, threshold = cognitive_manifest.get("calibration"), cognitive_manifest.get("threshold")
    if not isinstance(calibration, dict) or not isinstance(threshold, dict) or cognitive_manifest.get("checkpoint_sha256") != TABICLV2_CHECKPOINT_SHA256:
        raise ValueError("Stage C Cognitive release manifest is incomplete.")
    release = CognitiveReleaseContract(TABICLV2_CHECKPOINT_SHA256, float(calibration["coefficient"]), float(calibration["intercept"]), float(threshold["value"]))
    if (release.coefficient, release.intercept, release.threshold) != (1.069105693175016, 0.255897163477147, 0.1503614235173848):
        raise ValueError("Stage C frozen calibration/threshold differs from the approved release.")
    artifacts = index.get("artifacts")
    expected_index_artifacts = {
        "release_manifest": COGNITIVE_MANIFEST,
        "aggregate_report": FINALIZATION_REPORT,
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_index_artifacts):
        raise ValueError("Stage C detached integrity index does not validate the Cognitive release.")
    for name, artifact_path in expected_index_artifacts.items():
        entry = artifacts[name]
        if not isinstance(entry, dict) or entry.get("sha256") != sha256(artifact_path):
            raise ValueError("Stage C detached integrity index checksum mismatch.")
    if cache_manifest.get("original_cognitive_manifest_sha256") != sha256(COGNITIVE_MANIFEST):
        raise ValueError("Stage C cache release is not bound to the finalized Cognitive manifest.")
    return release


def load_verified_model(release: CognitiveReleaseContract) -> object:
    """Deserialize only after ``preflight``; reject retained training arrays."""

    model = TabICLClassifier.load(CACHE_PATH, device="cpu")
    generator = getattr(model, "ensemble_generator_", None)
    preprocessors = getattr(generator, "preprocessors_", {})
    arrays_absent = getattr(generator, "X_", None) is None and getattr(generator, "y_", None) is None and all(getattr(item, "X_transformed_", None) is None for item in preprocessors.values())
    if not arrays_absent or getattr(model, "model_kv_cache_", None) is None:
        raise ValueError("Loaded cache retains training arrays or lacks KV-cache data.")
    if Path(str(getattr(model, "model_path_", ""))).resolve() != CHECKPOINT.resolve():
        raise ValueError("Loaded cache points at a different checkpoint.")
    classes = np.asarray(getattr(model, "classes_", ()), dtype=object)
    if classes.ndim != 1 or len(classes) != 2 or len(np.flatnonzero(classes == 1)) != 1 or len(set(classes.tolist())) != 2:
        raise ValueError("Loaded cache has ambiguous or invalid positive-class labels.")
    return model


@dataclass
class RecordingRealClassifier:
    """Records aggregate call information while delegating to the real cache."""

    model: object
    calls: int = 0
    rows: int = 0
    nan_rows: int = 0
    checkpoints: set[str] = field(default_factory=set)

    @property
    def classes_(self):
        return self.model.classes_

    def predict_proba(self, frame):
        self.calls += 1
        self.rows += len(frame)
        self.nan_rows += int(frame.loc[:, QUESTION_COLUMNS].isna().any(axis=1).sum())
        self.checkpoints.update(str(value) for value in frame[FEATURE_COLUMNS[0]].astype(str))
        return self.model.predict_proba(frame)


@dataclass
class RecordingShapCallable:
    """Records only in-memory prediction endpoints for SHAP sequence accounting."""

    wrapped: object
    outputs: list[np.ndarray] = field(default_factory=list)

    def __call__(self, values: object) -> np.ndarray:
        output = np.asarray(self.wrapped(values), dtype=float)
        # Each entry is a synthetic-request probability vector. It is retained
        # only for the current calculation and is never persisted.
        self.outputs.append(output.copy())
        return output


def shap_sequence_accounting(outputs: list[np.ndarray], feature_count: int = 15) -> dict[str, float | int | str]:
    """Derive the permutation-path reconstruction from SHAP's own batches.

    With the Independent masker used here, each full permutation batch has
    ``2 * feature_count + 1`` outputs: baseline, fully unmasked midpoint, and
    return-to-baseline endpoint. This function deliberately returns
    ``unavailable`` rather than guessing when that structure is absent.
    """

    width = 2 * feature_count + 1
    if not outputs or any(np.asarray(item).ndim != 1 or len(item) != width for item in outputs):
        return {"status": "unavailable", "permutation_batches": len(outputs)}
    batches = np.asarray(outputs, dtype=float)
    if not np.all(np.isfinite(batches)):
        return {"status": "unavailable", "permutation_batches": len(outputs)}
    starts, middles, ends = batches[:, 0], batches[:, feature_count], batches[:, -1]
    contribution_sum = float(np.mean(2.0 * middles - starts - ends) / 2.0)
    expected_value = float(starts[-1])
    return {
        "status": "available",
        "permutation_batches": int(len(batches)),
        "sequence_expected_value": expected_value,
        "sequence_contribution_sum": contribution_sum,
        "sequence_reconstruction": expected_value + contribution_sum,
        "sequence_full_midpoint_mean": float(np.mean(middles)),
        "sequence_baseline_start_range": float(np.max(starts) - np.min(starts)),
        "sequence_baseline_end_range": float(np.max(ends) - np.min(ends)),
    }


def validate_shap_acceptance(
    *,
    explanation: object,
    calibrated_prediction: float,
    shap_rows: int,
    wrapper_calls: int,
    unmasked_fidelity_error: float,
    batch_invariance_error: float,
    runtime_seconds: float,
    rss_delta_bytes: int,
    completed_assessments: int,
    sequence_accounting: dict[str, float | int | str] | None = None,
) -> tuple[float | None, dict[str, object]]:
    """Validate post-SHAP output and return safe metrics or raise with them."""

    raw_values = getattr(explanation, "values", None)
    raw_base = getattr(explanation, "base_values", None)
    values = np.asarray(raw_values, dtype=float)
    bases = np.asarray(raw_base, dtype=float)
    values_shape, base_shape = tuple(values.shape), tuple(bases.shape)
    actual_features = values_shape[1] if len(values_shape) == 2 else None
    reconstruction_error: float | None = None
    if values_shape == (1, 15) and bases.size == 1 and np.isfinite(calibrated_prediction) and np.all(np.isfinite(values)) and np.all(np.isfinite(bases)):
        reconstruction_error = abs(float(calibrated_prediction) - (float(bases.reshape(-1)[0]) + float(values.sum())))
    sequence_reconstruction_error: float | None = None
    sequence_vs_independent_error: float | None = None
    if sequence_accounting and sequence_accounting.get("status") == "available" and reconstruction_error is not None:
        sequence_reconstruction = float(sequence_accounting["sequence_reconstruction"])
        shap_reconstruction = float(bases.reshape(-1)[0]) + float(values.sum())
        sequence_reconstruction_error = abs(shap_reconstruction - sequence_reconstruction)
        sequence_vs_independent_error = abs(sequence_reconstruction - float(calibrated_prediction))
    conditions = {
        "values_shape_is_one_by_fifteen": values_shape == (1, 15),
        "base_value_is_scalar_per_assessment": bases.size == 1,
        "feature_count_matches": actual_features == 15,
        "values_are_finite": bool(np.all(np.isfinite(values))),
        "base_values_are_finite": bool(np.all(np.isfinite(bases))),
        "calibrated_prediction_is_finite": bool(np.isfinite(calibrated_prediction) and 0.0 <= calibrated_prediction <= 1.0),
        "evaluation_rows_within_budget": shap_rows <= MAX_EVALS,
        "unmasked_fidelity_within_tolerance": unmasked_fidelity_error <= TOLERANCE,
        "batch_invariance_within_tolerance": batch_invariance_error <= TOLERANCE,
        "additivity_within_tolerance": reconstruction_error is not None and reconstruction_error <= TOLERANCE,
        "runtime_within_bound": runtime_seconds <= MAX_RUNTIME_SECONDS,
    }
    diagnostic = _diagnostic(
        "post_shap_acceptance",
        values_shape=values_shape,
        base_values_shape=base_shape,
        actual_feature_count=actual_features,
        shap_rows=shap_rows,
        wrapper_calls=wrapper_calls,
        completed_assessments=completed_assessments,
        reconstruction_error=reconstruction_error,
        unmasked_fidelity_error=unmasked_fidelity_error,
        batch_invariance_error=batch_invariance_error,
        sequence_reconstruction_error=sequence_reconstruction_error,
        sequence_vs_independent_error=sequence_vs_independent_error,
        sequence_accounting=sequence_accounting,
        runtime_seconds=runtime_seconds,
        rss_delta_bytes=rss_delta_bytes,
        conditions=conditions,
    )
    if not all(conditions.values()):
        _fail("Real-model SHAP output or bounded-workload verification failed.", diagnostic)
    return reconstruction_error, diagnostic


def _one_assessment(model: object, release: CognitiveReleaseContract, checkpoint: str, responses: np.ndarray) -> dict[str, object]:
    recorder = RecordingRealClassifier(model)
    wrapper = CheckpointConditionalCalibratedProbabilityWrapper(recorder, checkpoint)
    tuple_features = tuple([checkpoint, *(str(int(value)) for value in responses)])
    deployed = TabICLv2CognitiveAdapter(InMemoryContextCache(lambda: model), release).positive_probability(tuple_features)
    unmasked = float(wrapper(responses.reshape(1, 15))[0])
    if abs(deployed - unmasked) > TOLERANCE or (deployed >= release.threshold) != (unmasked >= release.threshold):
        raise RuntimeError("Unmasked wrapper fidelity failed before SHAP execution.")
    before_cache_id = id(model.model_kv_cache_)
    masked = responses.reshape(1, 15).astype(float).copy(); masked[0, [0, 7, 14]] = np.nan
    masked_probability = float(wrapper(masked)[0])
    if not np.isfinite(masked_probability) or not 0.0 <= masked_probability <= 1.0 or id(model.model_kv_cache_) != before_cache_id:
        raise RuntimeError("Controlled real-model NaN masking failed before SHAP execution.")
    individual = float(wrapper(responses.reshape(1, 15))[0])
    ordinary_batch = float(wrapper(np.vstack((responses, responses)).astype(float))[0])
    alongside_mask = float(wrapper(np.vstack((responses, np.full(15, np.nan)) ))[0])
    batch_difference = max(abs(individual - ordinary_batch), abs(individual - alongside_mask))
    if batch_difference > TOLERANCE:
        raise RuntimeError("Real-model batch invariance failed before SHAP execution.")
    background = np.full((1, 15), np.nan, dtype=float)
    started = time.monotonic()
    rss_before_shap = psutil.Process().memory_info().rss
    calls_before_shap, rows_before_shap, nan_rows_before_shap = recorder.calls, recorder.rows, recorder.nan_rows
    shap_callable = RecordingShapCallable(wrapper)
    explainer = shap.PermutationExplainer(shap_callable, shap.maskers.Independent(background, max_samples=1), seed=SEED)
    explanation = explainer(responses.reshape(1, 15), max_evals=MAX_EVALS)
    shap_calls = recorder.calls - calls_before_shap
    shap_rows = recorder.rows - rows_before_shap
    shap_nan_rows = recorder.nan_rows - nan_rows_before_shap
    elapsed = time.monotonic() - started
    prediction = float(wrapper(responses.reshape(1, 15))[0])
    sequence = shap_sequence_accounting(shap_callable.outputs)
    reconstruction_error, validation = validate_shap_acceptance(
        explanation=explanation,
        calibrated_prediction=prediction,
        shap_rows=shap_rows,
        wrapper_calls=shap_calls,
        unmasked_fidelity_error=abs(deployed-unmasked),
        batch_invariance_error=batch_difference,
        runtime_seconds=elapsed,
        rss_delta_bytes=max(0, psutil.Process().memory_info().rss-rss_before_shap),
        completed_assessments=0,
        sequence_accounting=sequence,
    )
    values = np.asarray(explanation.values, dtype=float)
    base = float(np.asarray(explanation.base_values, dtype=float).reshape(-1)[0])
    contribution_sum = float(values.reshape(1, 15).sum(axis=1)[0])
    repeat_recorder = RecordingRealClassifier(model)
    repeated = shap.PermutationExplainer(CheckpointConditionalCalibratedProbabilityWrapper(repeat_recorder, checkpoint), shap.maskers.Independent(background, max_samples=1), seed=SEED)(responses.reshape(1, 15), max_evals=MAX_EVALS)
    repeat_difference = float(np.max(np.abs(values - np.asarray(repeated.values, dtype=float))))
    if repeat_difference > TOLERANCE:
        _fail("Real-model SHAP same-seed repeatability failed.", _diagnostic(
            "repeatability_acceptance",
            values_shape=tuple(values.shape),
            base_values_shape=tuple(np.asarray(explanation.base_values).shape),
            actual_feature_count=15,
            shap_rows=shap_rows,
            wrapper_calls=shap_calls,
            reconstruction_error=reconstruction_error,
            unmasked_fidelity_error=abs(deployed-unmasked),
            batch_invariance_error=batch_difference,
            repeatability_error=repeat_difference,
            runtime_seconds=elapsed,
            rss_delta_bytes=max(0, psutil.Process().memory_info().rss-rss_before_shap),
            conditions={"same_seed_repeatability_within_tolerance": False},
        ))
    if elapsed > MAX_RUNTIME_SECONDS or recorder.checkpoints != {checkpoint} or shap_nan_rows == 0:
        raise RuntimeError("Real-model Stage C fixed-checkpoint, masking, or runtime bound failed.")
    return {"checkpoint": checkpoint, "feature_count": 15, "shap_wrapper_calls": shap_calls, "shap_evaluation_rows": shap_rows, "shap_nan_rows": shap_nan_rows, "base_value": base, "contribution_sum": contribution_sum, "calibrated_prediction": prediction, "unmasked_fidelity_error": abs(deployed-unmasked), "batch_invariance_error": batch_difference, "additivity_error": reconstruction_error, "internal_shap_reconstruction_error": validation["sequence_reconstruction_error"], "shap_sequence_vs_standalone_error": validation["sequence_vs_independent_prediction_error"], "sequence_baseline_start_range": validation["sequence_baseline_start_range"], "sequence_baseline_end_range": validation["sequence_baseline_end_range"], "repeatability_error": repeat_difference, "runtime_seconds": elapsed, "threshold_decision_preserved": (deployed >= release.threshold) == (unmasked >= release.threshold)}


def run_stage_c(diagnostic_output: Path = DEFAULT_DIAGNOSTIC_PATH) -> dict[str, object]:
    """Execute exactly three frozen synthetic requests and emit aggregate diagnostics."""

    process = psutil.Process(); before_rss = process.memory_info().rss; started = time.monotonic(); completed = 0
    try:
        release = preflight()
        model = load_verified_model(release)
        requests = (("CP6M", np.asarray([0, 1, 2] * 5)), ("CP24M", np.asarray([2, 1, 0] * 5)), ("CP60M", np.asarray([1] * 15)))
        results = []
        for checkpoint, responses in requests:
            if time.monotonic() - started > MAX_RUNTIME_SECONDS:
                raise TimeoutError("Stage C exceeded its five-minute bounded runtime before a new explanation.")
            results.append(_one_assessment(model, release, checkpoint, responses))
            completed += 1
        elapsed = time.monotonic() - started
        if elapsed > MAX_RUNTIME_SECONDS:
            raise TimeoutError("Stage C exceeded its five-minute bounded runtime.")
        maximums = {"unmasked_fidelity_error": max(float(item["unmasked_fidelity_error"]) for item in results), "batch_invariance_error": max(float(item["batch_invariance_error"]) for item in results), "additivity_error": max(float(item["additivity_error"]) for item in results), "internal_shap_reconstruction_error": max(float(item["internal_shap_reconstruction_error"]) for item in results), "shap_sequence_vs_standalone_error": max(float(item["shap_sequence_vs_standalone_error"]) for item in results), "repeatability_error": max(float(item["repeatability_error"]) for item in results), "shap_evaluation_rows": max(int(item["shap_evaluation_rows"]) for item in results), "shap_wrapper_calls": max(int(item["shap_wrapper_calls"]) for item in results), "sequence_baseline_start_range": max(float(item["sequence_baseline_start_range"]) for item in results), "sequence_baseline_end_range": max(float(item["sequence_baseline_end_range"]) for item in results), "runtime_seconds": elapsed, "rss_delta_bytes": max(0, process.memory_info().rss-before_rss)}
        result = {"status": "passed", "algorithm": "shap.PermutationExplainer", "shap_version": shap.__version__, "seed": SEED, "tolerance": TOLERANCE, "max_evals": MAX_EVALS, "assessments": results, "maximums": maximums, "privacy": {"participant_rows_read": False, "raw_questionnaire_vectors_persisted": False, "cache_rebuilt": False}}
        persist_diagnostic(diagnostic_output, _success_diagnostic(results, maximums))
        return result
    except StageCVerificationError as error:
        error.diagnostic["synthetic_assessments_completed"] = completed
        persist_diagnostic(diagnostic_output, error.diagnostic)
        raise
    except Exception as error:
        diagnostic = _diagnostic("unexpected_exception", completed_assessments=completed, runtime_seconds=time.monotonic()-started, rss_delta_bytes=max(0, process.memory_info().rss-before_rss), error_category=type(error).__name__, conditions={"unexpected_exception": False})
        persist_diagnostic(diagnostic_output, diagnostic)
        raise


if __name__ == "__main__":
    print(json.dumps(run_stage_c(), sort_keys=True))
