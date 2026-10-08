"""Nested development-only probability calibration for selected domain candidates."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from training.contracts import APPROVED_DECISION_THRESHOLD, TARGET_COLUMNS
from training.data.development_data import ModelingRecord
from training.experiments.cross_validation import SharedFold, create_shared_folds
from training.experiments.evaluation import binary_probability_metrics, calibration_bins
from training.models.common import (
    CandidateSpec,
    build_candidate_pipeline,
    fit_candidate_pipeline,
    modeling_arrays,
)


CALIBRATION_METHOD = "regularized_logistic_sigmoid_on_out_of_fold_logit"
CALIBRATION_EPSILON = 1e-6
DEFAULT_INNER_SPLITS = 5


def _logit_feature(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _fit_sigmoid_calibrator(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> LogisticRegression:
    calibrator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        random_state=seed,
        max_iter=2_000,
    )
    calibrator.fit(_logit_feature(probabilities), targets)
    return calibrator


def select_balanced_accuracy_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    """Select a deterministic threshold by balanced accuracy on development predictions."""

    if len(targets) != len(probabilities) or not len(targets):
        raise ValueError("Threshold targets and probabilities must be non-empty and aligned.")
    if np.unique(targets).size != 2:
        raise ValueError("Threshold selection requires both target classes.")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Threshold probabilities must be finite values in [0,1].")

    candidates = np.unique(
        np.clip(
            probabilities.astype(float),
            CALIBRATION_EPSILON,
            1.0 - CALIBRATION_EPSILON,
        )
    )
    best: dict[str, object] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        metrics = binary_probability_metrics(
            targets,
            probabilities,
            threshold=float(threshold),
        )
        sensitivity = float(metrics["sensitivity"])
        specificity = float(metrics["specificity"])
        balanced_accuracy = float(metrics["balanced_accuracy"])
        key = (balanced_accuracy, sensitivity, specificity, -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "objective": "maximize_balanced_accuracy_equivalent_to_youdens_j",
                "tie_breaking": (
                    "higher_sensitivity_then_higher_specificity_then_lower_threshold"
                ),
                "candidate_threshold_count": int(len(candidates)),
                "youdens_j": float(sensitivity + specificity - 1.0),
                "metrics": metrics,
            }
    if best is None:
        raise RuntimeError("Threshold selection produced no candidate result.")
    return best


def _inner_oof_probabilities(
    features: np.ndarray,
    targets: np.ndarray,
    candidate: CandidateSpec,
    *,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    if np.min(np.bincount(targets)) < n_splits:
        raise ValueError("Each class must support every inner calibration fold.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities = np.full(len(targets), np.nan, dtype=float)
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(features, targets)
    ):
        pipeline = build_candidate_pipeline(candidate, seed=seed + fold_index + 1)
        fit_candidate_pipeline(
            pipeline,
            candidate,
            features[train_indices],
            targets[train_indices],
        )
        probabilities[validation_indices] = pipeline.predict_proba(
            features[validation_indices]
        )[:, 1]
    if np.any(~np.isfinite(probabilities)):
        raise RuntimeError("Inner calibration predictions are incomplete.")
    return probabilities


def _metric_summary(metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    names = (
        "sensitivity",
        "specificity",
        "precision",
        "negative_predictive_value",
        "f1",
        "balanced_accuracy",
        "accuracy",
        "average_precision_pr_auc",
        "roc_auc",
        "brier_score",
    )
    return {
        name: {
            "mean": float(np.mean([float(item[name]) for item in metrics])),
            "standard_deviation": float(
                np.std([float(item[name]) for item in metrics], ddof=1)
            )
            if len(metrics) > 1
            else 0.0,
            "minimum": float(np.min([float(item[name]) for item in metrics])),
            "maximum": float(np.max([float(item[name]) for item in metrics])),
        }
        for name in names
    }


def run_nested_calibration_evaluation(
    records: Sequence[ModelingRecord],
    selections: Mapping[str, CandidateSpec],
    *,
    outer_folds: Sequence[SharedFold] | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    inner_splits: int = DEFAULT_INNER_SPLITS,
    cv_seed: int = 20260821,
    model_seed: int = 20260822,
    threshold: float = APPROVED_DECISION_THRESHOLD,
) -> dict[str, object]:
    """Assess sigmoid calibration without using outer validation observations."""

    if set(selections) != set(TARGET_COLUMNS):
        raise ValueError("Calibration requires one selected candidate per domain.")
    features, targets = modeling_arrays(records)
    folds = tuple(
        outer_folds
        or create_shared_folds(
            records,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=cv_seed,
        )
    )
    raw_repeat_predictions = {
        target: {repeat: np.zeros(len(records)) for repeat in range(n_repeats)}
        for target in TARGET_COLUMNS
    }
    calibrated_repeat_predictions = {
        target: {repeat: np.zeros(len(records)) for repeat in range(n_repeats)}
        for target in TARGET_COLUMNS
    }
    fold_results: dict[str, list[dict[str, object]]] = defaultdict(list)
    calibration_parameters: dict[str, list[dict[str, float]]] = defaultdict(list)

    for fold in folds:
        for target_position, target_name in enumerate(TARGET_COLUMNS):
            candidate = selections[target_name]
            train_features = features[fold.train_indices]
            train_targets = targets[fold.train_indices, target_position]
            validation_features = features[fold.validation_indices]
            validation_targets = targets[fold.validation_indices, target_position]
            seed = (
                model_seed
                + fold.repeat_index * 10_000
                + fold.fold_index * 100
                + target_position
            )
            inner_probabilities = _inner_oof_probabilities(
                train_features,
                train_targets,
                candidate,
                n_splits=inner_splits,
                seed=seed + 1_000_000,
            )
            calibrator = _fit_sigmoid_calibrator(
                inner_probabilities,
                train_targets,
                seed=seed + 2_000_000,
            )
            pipeline = build_candidate_pipeline(candidate, seed=seed)
            fit_candidate_pipeline(
                pipeline,
                candidate,
                train_features,
                train_targets,
            )
            raw_probabilities = pipeline.predict_proba(validation_features)[:, 1]
            calibrated_probabilities = calibrator.predict_proba(
                _logit_feature(raw_probabilities)
            )[:, 1]
            raw_repeat_predictions[target_name][fold.repeat_index][
                fold.validation_indices
            ] = raw_probabilities
            calibrated_repeat_predictions[target_name][fold.repeat_index][
                fold.validation_indices
            ] = calibrated_probabilities
            calibration_parameters[target_name].append(
                {
                    "coefficient": float(calibrator.coef_[0, 0]),
                    "intercept": float(calibrator.intercept_[0]),
                }
            )
            fold_results[target_name].append(
                {
                    "repeat": fold.repeat_index + 1,
                    "fold": fold.fold_index + 1,
                    "training_participants": int(len(fold.train_indices)),
                    "validation_participants": int(len(fold.validation_indices)),
                    "raw_metrics": binary_probability_metrics(
                        validation_targets,
                        raw_probabilities,
                        threshold=threshold,
                    ),
                    "calibrated_metrics": binary_probability_metrics(
                        validation_targets,
                        calibrated_probabilities,
                        threshold=threshold,
                    ),
                }
            )

    domains: dict[str, object] = {}
    for target_position, target_name in enumerate(TARGET_COLUMNS):
        raw_repeats = [
            binary_probability_metrics(
                targets[:, target_position],
                raw_repeat_predictions[target_name][repeat],
                threshold=threshold,
            )
            for repeat in range(n_repeats)
        ]
        calibrated_repeats = [
            binary_probability_metrics(
                targets[:, target_position],
                calibrated_repeat_predictions[target_name][repeat],
                threshold=threshold,
            )
            for repeat in range(n_repeats)
        ]
        averaged_raw = np.mean(
            list(raw_repeat_predictions[target_name].values()), axis=0
        )
        averaged_calibrated = np.mean(
            list(calibrated_repeat_predictions[target_name].values()), axis=0
        )
        repeat_thresholds = [
            select_balanced_accuracy_threshold(
                targets[:, target_position],
                calibrated_repeat_predictions[target_name][repeat],
            )
            for repeat in range(n_repeats)
        ]
        selected_threshold = select_balanced_accuracy_threshold(
            targets[:, target_position],
            averaged_calibrated,
        )
        parameters = calibration_parameters[target_name]
        domains[target_name] = {
            "selected_candidate": selections[target_name].name,
            "model_configuration_version": selections[target_name].version,
            "calibration_method": CALIBRATION_METHOD,
            "inner_splits": inner_splits,
            "fixed_threshold": threshold,
            "fold_results": fold_results[target_name],
            "raw_repeat_metric_summary": _metric_summary(raw_repeats),
            "calibrated_repeat_metric_summary": _metric_summary(calibrated_repeats),
            "raw_averaged_oof_metrics": binary_probability_metrics(
                targets[:, target_position], averaged_raw, threshold=threshold
            ),
            "calibrated_averaged_oof_metrics": binary_probability_metrics(
                targets[:, target_position], averaged_calibrated, threshold=threshold
            ),
            "raw_calibration_bins": calibration_bins(
                targets[:, target_position], averaged_raw
            ),
            "calibrated_calibration_bins": calibration_bins(
                targets[:, target_position], averaged_calibrated
            ),
            "threshold_selection": {
                **selected_threshold,
                "selection_data": "averaged_nested_out_of_fold_development_predictions",
                "repeat_thresholds": [
                    {
                        "repeat": repeat + 1,
                        "threshold": float(result["threshold"]),
                        "balanced_accuracy": float(
                            result["metrics"]["balanced_accuracy"]
                        ),
                        "sensitivity": float(result["metrics"]["sensitivity"]),
                        "specificity": float(result["metrics"]["specificity"]),
                    }
                    for repeat, result in enumerate(repeat_thresholds)
                ],
                "threshold_stability": {
                    "mean": float(
                        np.mean([result["threshold"] for result in repeat_thresholds])
                    ),
                    "standard_deviation": float(
                        np.std(
                            [result["threshold"] for result in repeat_thresholds],
                            ddof=1,
                        )
                    )
                    if len(repeat_thresholds) > 1
                    else 0.0,
                    "minimum": float(
                        np.min([result["threshold"] for result in repeat_thresholds])
                    ),
                    "maximum": float(
                        np.max([result["threshold"] for result in repeat_thresholds])
                    ),
                },
            },
            "calibration_parameter_stability": {
                "coefficient_mean": float(
                    np.mean([item["coefficient"] for item in parameters])
                ),
                "coefficient_standard_deviation": float(
                    np.std([item["coefficient"] for item in parameters], ddof=1)
                ),
                "intercept_mean": float(
                    np.mean([item["intercept"] for item in parameters])
                ),
                "intercept_standard_deviation": float(
                    np.std([item["intercept"] for item in parameters], ddof=1)
                ),
            },
        }
    return {
        "scope": "development_only_nested_calibration",
        "development_participants": len(records),
        "outer_splits": n_splits,
        "outer_repeats": n_repeats,
        "inner_splits": inner_splits,
        "threshold_policy": "fixed_predeclared_probability_threshold",
        "threshold": threshold,
        "final_test_evaluated": False,
        "domains": domains,
    }
