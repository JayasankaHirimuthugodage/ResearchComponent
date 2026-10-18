"""Development-only exploratory TabICLv2 sensitivity-threshold study.

This runner intentionally evaluates only Cognitive and Social-emotional targets.
It never loads the locked test partition and retains no row-level predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from training.contracts import FEATURE_COLUMNS, TARGET_COLUMNS, default_dataset_path, repository_root
from training.data.development_data import default_manifest_path, default_split_summary_path, load_development_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.tabiclv2_evaluation import (
    _fit_calibrator, _logit, _metric_summary, _metrics, fold_plan_fingerprint,
    modeling_arrays, select_balanced_accuracy_threshold,
)
from training.tabiclv2.candidate import (
    TABICLV2_CHECKPOINT_NAME, TABICLV2_CHECKPOINT_SHA256,
    TABICLV2_CONFIGURATION_VERSION, TABICLV2_PACKAGE_VERSION, tabiclv2_probabilities,
)

MODEL_SEED = 20260822
INNER_SPLITS = 3
SENSITIVITY_TARGET = 0.85
STUDY_TARGETS = ("target_cognitive_concern", "target_social_emotional_concern")
REPORT_NAME = "development-tabiclv2-sensitivity-threshold-report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def select_sensitivity_constrained_threshold(targets: np.ndarray, probabilities: np.ndarray, *, target: float = SENSITIVITY_TARGET) -> dict[str, object]:
    """Maximize specificity subject to fixed, inner-development sensitivity target."""
    if not 0.0 < target <= 1.0:
        raise ValueError("Sensitivity target must be in (0, 1].")
    candidates = np.unique(np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6))
    eligible: list[tuple[tuple[float, float, float], dict[str, object]]] = []
    for threshold in candidates:
        decisions = (probabilities >= threshold).astype(np.int8)
        metrics = _metrics(targets, probabilities, decisions)
        if float(metrics["sensitivity"]) >= target:
            # specificity first; then sensitivity; then lower threshold (predeclared deterministic ties).
            key = (float(metrics["specificity"]), float(metrics["sensitivity"]), -float(threshold))
            eligible.append((key, {"threshold": float(threshold), "metrics": metrics}))
    if not eligible:
        raise RuntimeError("No inner-development threshold met the fixed sensitivity target.")
    _, selected = max(eligible, key=lambda row: row[0])
    return {
        **selected,
        "objective": "maximize_specificity_subject_to_minimum_sensitivity",
        "sensitivity_target": float(target),
        "candidate_threshold_count": int(len(candidates)),
        "eligible_threshold_count": int(len(eligible)),
        "tie_breaking": "higher_specificity_then_higher_sensitivity_then_lower_threshold",
    }


def _inner_probabilities(train_x: np.ndarray, train_y: np.ndarray, checkpoint: Path, seed: int) -> np.ndarray:
    probabilities = np.full(len(train_y), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    for inner_index, (fit_indices, validation_indices) in enumerate(splitter.split(np.zeros(len(train_y)), train_y)):
        probabilities[validation_indices] = tabiclv2_probabilities(
            train_x[fit_indices], train_y[fit_indices], train_x[validation_indices],
            checkpoint_path=checkpoint, seed=seed + inner_index,
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Inner OOF probabilities are incomplete.")
    return probabilities


def _threshold_stability(rows: list[dict[str, object]], strategy: str) -> dict[str, float]:
    values = np.asarray([float(row["thresholds"][strategy]) for row in rows], dtype=float)
    return {"mean": float(values.mean()), "standard_deviation": float(values.std(ddof=1)), "minimum": float(values.min()), "maximum": float(values.max())}


def _paired_bootstrap(targets: np.ndarray, probability_a: np.ndarray, decision_a: np.ndarray, probability_b: np.ndarray, decision_b: np.ndarray) -> dict[str, object]:
    """Paired participant bootstrap on repeat-averaged OOF outputs; no predictions persisted."""
    rng = np.random.default_rng(20261017)
    metrics = ("sensitivity", "specificity", "accuracy", "balanced_accuracy", "precision", "negative_predictive_value", "f1", "brier_score", "average_precision_pr_auc", "roc_auc")
    def one(ix: np.ndarray, probabilities: np.ndarray, decisions: np.ndarray) -> dict[str, object]:
        return _metrics(targets[ix], probabilities[ix], decisions[ix])
    point_a, point_b = one(np.arange(len(targets)), probability_a, decision_a), one(np.arange(len(targets)), probability_b, decision_b)
    values = {name: [] for name in metrics}
    positive, negative = np.flatnonzero(targets == 1), np.flatnonzero(targets == 0)
    for _ in range(2000):
        indices = np.concatenate((rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)))
        left, right = one(indices, probability_a, decision_a), one(indices, probability_b, decision_b)
        for name in metrics:
            values[name].append(float(right[name]) - float(left[name]))
    return {"method": "stratified_participant_level_paired_bootstrap_on_repeat_averaged_oof_outputs", "replicates": 2000, "comparison": "sensitivity_constrained_minus_youdens_j", "metrics": {name: {"point_difference": float(point_b[name]) - float(point_a[name]), "confidence_interval_95": [float(np.percentile(values[name], 2.5)), float(np.percentile(values[name], 97.5))]} for name in metrics}}


def _atomic_write_once(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write((json.dumps(value, indent=2) + "\n").encode("utf-8"))
            destination.flush(); os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description="Development-only TabICLv2 threshold comparison.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--split-summary", type=Path, default=default_split_summary_path())
    parser.add_argument("--checkpoint", type=Path, default=Path(sys.prefix) / "hf-cache/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--output", type=Path, default=root / "reports/generated" / REPORT_NAME)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args(); root = repository_root().resolve(); output = args.output.resolve(); checkpoint = args.checkpoint.absolute()
    if not output.is_relative_to(root): raise ValueError("Output must remain in repository.")
    if checkpoint.name != TABICLV2_CHECKPOINT_NAME or not checkpoint.is_file() or _sha256(checkpoint) != TABICLV2_CHECKPOINT_SHA256: raise ValueError("Approved checkpoint verification failed.")
    if importlib.metadata.version("tabicl") != TABICLV2_PACKAGE_VERSION: raise ValueError("tabicl version mismatch.")
    development = load_development_dataset(dataset_path=args.dataset, manifest_path=args.manifest, summary_path=args.split_summary)
    if len(development.records) != 500: raise ValueError("Expected exactly 500 development participants.")
    features, targets = modeling_arrays(development.records)
    if tuple(FEATURE_COLUMNS) != tuple(development.records[0].feature_names) if hasattr(development.records[0], "feature_names") else False: raise ValueError("Feature contract mismatch.")
    folds = create_shared_folds(development.records); fingerprint = fold_plan_fingerprint(folds)
    if fingerprint != "7AB8604BB0DD59DA33420D2743730A468D22C06FFD7EE1E78395EB5050B92C39": raise ValueError("Shared-fold fingerprint mismatch.")
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "development_participants": len(development.records), "feature_count": int(features.shape[1]), "outer_evaluations": 50, "checkpoint_sha256": _sha256(checkpoint), "shared_fold_plan_sha256": fingerprint, "test_cohort_loaded": False}, indent=2)); return 0
    if output.exists(): raise FileExistsError("Output already exists.")
    start = time.perf_counter(); domains: dict[str, object] = {}
    for target_name in STUDY_TARGETS:
        position = TARGET_COLUMNS.index(target_name); y = targets[:, position]
        repeat_probabilities = {strategy: [] for strategy in ("youdens_j", "sensitivity_constrained")}; repeat_decisions = {strategy: [] for strategy in repeat_probabilities}; fold_rows: list[dict[str, object]] = []
        for repeat in range(5):
            probabilities = {strategy: np.full(len(y), np.nan) for strategy in repeat_probabilities}; decisions = {strategy: np.zeros(len(y), dtype=np.int8) for strategy in repeat_probabilities}
            for fold in (item for item in folds if item.repeat_index == repeat):
                seed = MODEL_SEED + position * 1_000_000 + fold.repeat_index * 10_000 + fold.fold_index * 100
                train_x, train_y, validation_x = features[fold.train_indices], y[fold.train_indices], features[fold.validation_indices]
                inner_raw = _inner_probabilities(train_x, train_y, checkpoint, seed + 1)
                calibrator = _fit_calibrator(inner_raw, train_y, seed=seed + 50); inner_calibrated = calibrator.predict_proba(_logit(inner_raw))[:, 1]
                selected = {"youdens_j": select_balanced_accuracy_threshold(train_y, inner_calibrated), "sensitivity_constrained": select_sensitivity_constrained_threshold(train_y, inner_calibrated)}
                outer_raw = tabiclv2_probabilities(train_x, train_y, validation_x, checkpoint_path=checkpoint, seed=seed + 75)
                outer_calibrated = calibrator.predict_proba(_logit(outer_raw))[:, 1]
                validation_y = y[fold.validation_indices]
                for strategy, value in selected.items():
                    decision = (outer_calibrated >= float(value["threshold"])).astype(np.int8); probabilities[strategy][fold.validation_indices] = outer_calibrated; decisions[strategy][fold.validation_indices] = decision
                fold_rows.append({"repeat": repeat + 1, "fold": fold.fold_index + 1, "training_participants": int(len(train_y)), "validation_participants": int(len(validation_y)), "thresholds": {name: float(value["threshold"]) for name, value in selected.items()}, "inner_sensitivity_target_met": bool(float(selected["sensitivity_constrained"]["metrics"]["sensitivity"]) >= SENSITIVITY_TARGET), "validation_metrics": {name: _metrics(validation_y, outer_calibrated, decisions[name][fold.validation_indices]) for name in decisions}})
                completed = (len(domains) * 25) + len(fold_rows); elapsed = time.perf_counter() - start; eta = elapsed / completed * (50 - completed)
                print(f"[{completed}/50] domain={target_name} repeat={repeat+1} fold={fold.fold_index+1} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
            for strategy in probabilities:
                repeat_probabilities[strategy].append(probabilities[strategy]); repeat_decisions[strategy].append(decisions[strategy])
        avg_probability = {name: np.mean(values, axis=0) for name, values in repeat_probabilities.items()}; majority_decision = {name: (np.mean(repeat_decisions[name], axis=0) >= 0.5).astype(np.int8) for name in repeat_decisions}
        methods = {}
        for strategy in probabilities:
            repeated_metrics = [_metrics(y, repeat_probabilities[strategy][i], repeat_decisions[strategy][i]) for i in range(5)]
            methods[strategy] = {"repeat_metric_summary": _metric_summary(repeated_metrics), "participant_averaged_oof_metrics": _metrics(y, avg_probability[strategy], majority_decision[strategy]), "threshold_independent_metric_note": "PR-AUC, AUROC, and Brier score are identical across strategies because calibrated probabilities are unchanged."}
        domains[target_name] = {"positive_count": int(y.sum()), "negative_count": int(len(y)-y.sum()), "methods": methods, "fold_results": fold_rows, "threshold_stability": {strategy: _threshold_stability(fold_rows, strategy) for strategy in methods}, "outer_folds_meeting_85_percent_sensitivity": {strategy: int(sum(float(row["validation_metrics"][strategy]["sensitivity"]) >= SENSITIVITY_TARGET for row in fold_rows)) for strategy in methods}, "paired_uncertainty": _paired_bootstrap(y, avg_probability["youdens_j"], majority_decision["youdens_j"], avg_probability["sensitivity_constrained"], majority_decision["sensitivity_constrained"])}
    report = {"artifact_schema_version": "tabiclv2-sensitivity-threshold-development-v1", "status": "completed_development_only_post_hoc_exploratory", "scope": "development_only", "development_participants": 500, "feature_contract": list(FEATURE_COLUMNS), "targets": list(STUDY_TARGETS), "test_cohort_loaded": False, "sensitivity_target": SENSITIVITY_TARGET, "target_interpretation": "illustrative_exploratory_assumption_not_clinical_standard", "cross_validation": {"outer_splits": 5, "outer_repeats": 5, "outer_evaluations_completed": 50, "inner_splits": INNER_SPLITS, "shared_fold_plan_sha256": fingerprint}, "calibration": "inner_oof_regularized_logistic_sigmoid_on_raw_probability_logit", "strategies": {"youdens_j": "maximize_balanced_accuracy_equivalent_to_youdens_j", "sensitivity_constrained": "maximize_specificity_subject_to_inner_sensitivity_at_least_0.85"}, "domains": domains, "runtime_seconds": time.perf_counter()-start, "checkpoint_sha256": _sha256(checkpoint), "tabiclv2_configuration_version": TABICLV2_CONFIGURATION_VERSION, "privacy": {"participant_identifiers_written": False, "row_level_predictions_written": False, "aggregate_results_only": True}, "limitations": ["Post-hoc exploratory development-only threshold study.", "The 85% target is illustrative and not clinically endorsed.", "Repeated CV estimates share participants.", "No locked-test or external validation was performed.", "Targets are questionnaire-defined concerns, not diagnoses."], "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "tabicl": importlib.metadata.version("tabicl")}}
    _atomic_write_once(output, report); print(json.dumps({"status": report["status"], "runtime_seconds": report["runtime_seconds"], "outer_evaluations_completed": 50, "test_cohort_loaded": False, "output": str(output.relative_to(root))}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
