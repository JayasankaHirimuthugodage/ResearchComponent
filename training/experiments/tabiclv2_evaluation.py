"""Leakage-controlled, development-only evaluation of individual TabICLv2 models."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from training.contracts import (
    BINARY_VALUES,
    CHECKPOINT_IDS,
    FEATURE_COLUMNS,
    SCOREABLE_RESPONSE_VALUES,
    TARGET_COLUMNS,
)
from training.data.development_data import ModelingRecord
from training.experiments.cross_validation import (
    DEFAULT_CV_SEED,
    DEFAULT_FOLD_SEARCH_ITERATIONS,
    DEFAULT_N_REPEATS,
    DEFAULT_N_SPLITS,
    SharedFold,
    create_shared_folds,
)


DEFAULT_MODEL_SEED = 20260822


INNER_SPLITS = 3
CALIBRATION_EPSILON = 1e-6
TABICLV2_CANDIDATE_NAME = "tabiclv2_default_eight_view"
FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]
ProgressCallback = Callable[[dict[str, object]], None]


def modeling_arrays(
    records: Sequence[ModelingRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Project records onto the exact local 16-feature, three-target contract."""

    if not records:
        raise ValueError("At least one development record is required.")
    participant_keys = [record.participant_key for record in records]
    if len(participant_keys) != len(set(participant_keys)):
        raise ValueError("Development modelling requires one row per participant.")
    if any(len(record.feature_values) != len(FEATURE_COLUMNS) for record in records):
        raise ValueError("Every record must match the exact 16-feature contract.")
    if any(record.feature_values[0] not in CHECKPOINT_IDS for record in records):
        raise ValueError("A checkpoint is outside the frozen feature schema.")
    if any(
        value not in SCOREABLE_RESPONSE_VALUES
        for record in records
        for value in record.feature_values[1:]
    ):
        raise ValueError("Only scoreable questionnaire values may enter TabICLv2.")
    if any(
        str(value) not in BINARY_VALUES
        for record in records
        for value in record.target_values
    ):
        raise ValueError("TabICLv2 domain targets must be binary.")
    features = np.asarray([record.feature_values for record in records], dtype=object)
    targets = np.asarray([record.target_values for record in records], dtype=np.int8)
    if features.shape != (len(records), len(FEATURE_COLUMNS)) or targets.shape != (
        len(records),
        len(TARGET_COLUMNS),
    ):
        raise ValueError("TabICLv2 modelling arrays have unexpected dimensions.")
    return features, targets


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def decision_metrics(targets: np.ndarray, decisions: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(targets, decisions, labels=[0, 1]).ravel()
    return {
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "sensitivity": float(recall_score(targets, decisions, zero_division=0)),
        "specificity": _safe_ratio(int(tn), int(tn + fp)),
        "precision": float(precision_score(targets, decisions, zero_division=0)),
        "negative_predictive_value": _safe_ratio(int(tn), int(tn + fn)),
        "f1": float(f1_score(targets, decisions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, decisions)),
        "accuracy": float(accuracy_score(targets, decisions)),
    }


def calibration_bins(
    targets: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, object]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.digitize(probabilities, edges[1:-1], right=False)
    rows: list[dict[str, object]] = []
    for index in range(n_bins):
        mask = assignments == index
        count = int(np.sum(mask))
        if count:
            rows.append(
                {
                    "lower_bound": float(edges[index]),
                    "upper_bound": float(edges[index + 1]),
                    "count": count,
                    "mean_predicted_probability": float(np.mean(probabilities[mask])),
                    "observed_positive_rate": float(np.mean(targets[mask])),
                }
            )
    return rows


def select_balanced_accuracy_threshold(
    targets: np.ndarray, probabilities: np.ndarray
) -> dict[str, object]:
    if len(targets) != len(probabilities) or not len(targets):
        raise ValueError("Threshold targets and probabilities must be non-empty and aligned.")
    if np.unique(targets).size != 2:
        raise ValueError("Threshold selection requires both target classes.")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Threshold probabilities must be finite values in [0,1].")
    candidates = np.unique(
        np.clip(probabilities.astype(float), CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
    )
    selected: dict[str, object] | None = None
    selected_key: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        metrics = decision_metrics(targets, (probabilities >= threshold).astype(np.int8))
        key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["sensitivity"]),
            float(metrics["specificity"]),
            -float(threshold),
        )
        if selected_key is None or key > selected_key:
            selected_key = key
            selected = {
                "threshold": float(threshold),
                "objective": "maximize_balanced_accuracy_equivalent_to_youdens_j",
                "tie_breaking": "higher_sensitivity_then_higher_specificity_then_lower_threshold",
                "candidate_threshold_count": int(len(candidates)),
                "youdens_j": float(metrics["sensitivity"] + metrics["specificity"] - 1.0),
                "metrics": metrics,
            }
    if selected is None:
        raise RuntimeError("Threshold selection did not produce a result.")
    return selected


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _fit_calibrator(
    probabilities: np.ndarray, targets: np.ndarray, *, seed: int
) -> LogisticRegression:
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000, random_state=seed)
    model.fit(_logit(probabilities), targets)
    return model


def _metrics(
    targets: np.ndarray, probabilities: np.ndarray, decisions: np.ndarray
) -> dict[str, object]:
    return {
        **decision_metrics(targets, decisions),
        "average_precision_pr_auc": float(average_precision_score(targets, probabilities)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "brier_score": float(brier_score_loss(targets, probabilities)),
    }


def _metric_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
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
    result: dict[str, object] = {}
    for name in names:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        standard_error = standard_deviation / math.sqrt(len(values)) if len(values) else 0.0
        # t(4) for the approved five repeated estimates; intervals are descriptive because
        # repeat estimates share participants and are not statistically independent.
        half_width = 2.776 * standard_error if len(values) == 5 else 1.96 * standard_error
        result[name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": standard_deviation,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "descriptive_confidence_interval_95": [
                float(np.mean(values) - half_width),
                float(np.mean(values) + half_width),
            ],
        }
    return result


def fold_plan_fingerprint(folds: Sequence[SharedFold]) -> str:
    source = "\n".join(
        f"{fold.repeat_index}:{fold.fold_index}:"
        f"{','.join(str(index) for index in fold.validation_indices.tolist())}"
        for fold in folds
    ).encode("utf-8")
    return hashlib.sha256(source).hexdigest().upper()


def _combined_calibration_bins(
    rows: Sequence[Sequence[dict[str, object]]], *, n_bins: int = 10
) -> list[dict[str, object]]:
    combined: list[dict[str, object]] = []
    for index in range(n_bins):
        selected = [
            bin_row
            for repeat_bins in rows
            for bin_row in repeat_bins
            if math.isclose(float(bin_row["lower_bound"]), index / n_bins)
        ]
        count = sum(int(row["count"]) for row in selected)
        if not count:
            continue
        combined.append(
            {
                "lower_bound": index / n_bins,
                "upper_bound": (index + 1) / n_bins,
                "count": count,
                "mean_predicted_probability": sum(
                    int(row["count"]) * float(row["mean_predicted_probability"])
                    for row in selected
                )
                / count,
                "observed_positive_rate": sum(
                    int(row["count"]) * float(row["observed_positive_rate"])
                    for row in selected
                )
                / count,
            }
        )
    return combined


def _inner_oof_probabilities(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    fit_predict: FitPredict,
    seed: int,
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    probabilities = np.full(len(targets), np.nan, dtype=float)
    for inner_index, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(targets)), targets)
    ):
        if np.intersect1d(train_indices, validation_indices).size:
            raise RuntimeError("Inner training and validation participants overlap.")
        probabilities[validation_indices] = fit_predict(
            features[train_indices],
            targets[train_indices],
            features[validation_indices],
            seed + inner_index,
        )
    if np.any(~np.isfinite(probabilities)):
        raise RuntimeError("Inner out-of-fold probabilities are incomplete.")
    return probabilities


def run_tabiclv2_evaluation(
    records: Sequence[ModelingRecord],
    *,
    fit_predict: FitPredict,
    historical_report: dict[str, object],
    completed_repeats: dict[str, dict[str, object]] | None = None,
    progress_callback: ProgressCallback | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    cv_seed: int = DEFAULT_CV_SEED,
    fold_search_iterations: int = DEFAULT_FOLD_SEARCH_ITERATIONS,
    model_seed: int = DEFAULT_MODEL_SEED,
) -> dict[str, object]:
    features, targets = modeling_arrays(records)
    folds = create_shared_folds(
        records,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=cv_seed,
        search_iterations=fold_search_iterations,
    )
    fingerprint = fold_plan_fingerprint(folds)
    historical_fingerprint = str(
        historical_report.get("cross_validation", {}).get("shared_fold_plan_sha256", "")
    )
    if historical_fingerprint != fingerprint:
        raise ValueError("TabICLv2 fold fingerprint does not match the seven-model experiment.")
    if historical_report.get("development_participants") != len(records):
        raise ValueError("Historical comparison participant count does not match development data.")

    cached = completed_repeats or {}
    domains: dict[str, object] = {}
    start = time.perf_counter()
    total_outer_folds = len(TARGET_COLUMNS) * len(folds)
    completed_outer_folds = 0

    for target_position, target_name in enumerate(TARGET_COLUMNS):
        domain_targets = targets[:, target_position]
        repeat_rows: list[dict[str, object]] = []
        all_fold_rows: list[dict[str, object]] = []
        for repeat_index in range(n_repeats):
            cache_key = f"{target_name}:repeat-{repeat_index + 1}"
            cached_repeat = cached.get(cache_key)
            if cached_repeat is not None:
                repeat_rows.append(cached_repeat)
                all_fold_rows.extend(cached_repeat["fold_results"])
                completed_outer_folds += n_splits
                continue

            repeat_probabilities = np.full(len(records), np.nan, dtype=float)
            repeat_decisions = np.zeros(len(records), dtype=np.int8)
            fold_rows: list[dict[str, object]] = []
            repeat_folds = [fold for fold in folds if fold.repeat_index == repeat_index]
            for fold in repeat_folds:
                seed = (
                    model_seed
                    + target_position * 1_000_000
                    + fold.repeat_index * 10_000
                    + fold.fold_index * 100
                )
                train_x = features[fold.train_indices]
                train_y = domain_targets[fold.train_indices]
                inner_raw = _inner_oof_probabilities(
                    train_x, train_y, fit_predict=fit_predict, seed=seed + 1
                )
                calibrator = _fit_calibrator(inner_raw, train_y, seed=seed + 50)
                inner_calibrated = calibrator.predict_proba(_logit(inner_raw))[:, 1]
                threshold_result = select_balanced_accuracy_threshold(train_y, inner_calibrated)
                threshold = float(threshold_result["threshold"])
                raw_validation = fit_predict(
                    train_x,
                    train_y,
                    features[fold.validation_indices],
                    seed + 75,
                )
                calibrated_validation = calibrator.predict_proba(
                    _logit(raw_validation)
                )[:, 1]
                validation_targets = domain_targets[fold.validation_indices]
                validation_decisions = (calibrated_validation >= threshold).astype(np.int8)
                repeat_probabilities[fold.validation_indices] = calibrated_validation
                repeat_decisions[fold.validation_indices] = validation_decisions
                fold_rows.append(
                    {
                        "repeat": repeat_index + 1,
                        "fold": fold.fold_index + 1,
                        "training_participants": int(len(fold.train_indices)),
                        "validation_participants": int(len(fold.validation_indices)),
                        "inner_context_minimum": int(math.floor(len(fold.train_indices) * 2 / 3)),
                        "training_positives": int(np.sum(train_y)),
                        "validation_positives": int(np.sum(validation_targets)),
                        "calibration_coefficient": float(calibrator.coef_[0, 0]),
                        "calibration_intercept": float(calibrator.intercept_[0]),
                        "threshold": threshold,
                        "validation_metrics": _metrics(
                            validation_targets, calibrated_validation, validation_decisions
                        ),
                    }
                )
                completed_outer_folds += 1
                if progress_callback is not None:
                    elapsed = time.perf_counter() - start
                    progress_callback(
                        {
                            "event": "fold_completed",
                            "domain": target_name,
                            "repeat": repeat_index + 1,
                            "fold": fold.fold_index + 1,
                            "completed_outer_folds": completed_outer_folds,
                            "total_outer_folds": total_outer_folds,
                            "elapsed_seconds": elapsed,
                            "estimated_remaining_seconds": (
                                elapsed / completed_outer_folds
                                * (total_outer_folds - completed_outer_folds)
                            ),
                        }
                    )
            if np.any(~np.isfinite(repeat_probabilities)):
                raise RuntimeError("A repeated outer evaluation is incomplete.")
            repeat_result = {
                "repeat": repeat_index + 1,
                "metrics": _metrics(
                    domain_targets, repeat_probabilities, repeat_decisions
                ),
                "calibration_bins": calibration_bins(domain_targets, repeat_probabilities),
                "fold_results": fold_rows,
            }
            repeat_rows.append(repeat_result)
            all_fold_rows.extend(fold_rows)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "repeat_completed",
                        "cache_key": cache_key,
                        "repeat_result": repeat_result,
                        "completed_outer_folds": completed_outer_folds,
                        "total_outer_folds": total_outer_folds,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )

        repeat_metrics = [row["metrics"] for row in repeat_rows]
        metric_summary = _metric_summary(repeat_metrics)
        historical_domain = historical_report["domains"][target_name]
        reference_name = historical_domain["historical_selected_reference"]
        reference = historical_domain["candidates"][reference_name]
        reference_summary = reference["repeat_metric_summary"]
        comparison_metrics = (
            "average_precision_pr_auc",
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "brier_score",
        )
        numerical_comparison = {
            name: {
                "tabiclv2_mean": metric_summary[name]["mean"],
                "historical_reference_mean": reference_summary[name]["mean"],
                "tabiclv2_minus_reference": (
                    float(metric_summary[name]["mean"])
                    - float(reference_summary[name]["mean"])
                ),
            }
            for name in comparison_metrics
        }
        existing_ranking = [
            {
                "candidate": name,
                "mean_average_precision": candidate["repeat_metric_summary"]
                ["average_precision_pr_auc"]["mean"],
                "source": "historical_seven_model_report",
            }
            for name, candidate in historical_domain["candidates"].items()
        ]
        ranking = sorted(
            [
                *existing_ranking,
                {
                    "candidate": TABICLV2_CANDIDATE_NAME,
                    "mean_average_precision": metric_summary["average_precision_pr_auc"]["mean"],
                    "source": "current_tabiclv2_evaluation",
                },
            ],
            key=lambda row: (-float(row["mean_average_precision"]), str(row["candidate"])),
        )
        domains[target_name] = {
            "positive_count": int(np.sum(domain_targets)),
            "negative_count": int(len(domain_targets) - np.sum(domain_targets)),
            "candidate": TABICLV2_CANDIDATE_NAME,
            "historical_selected_reference": reference_name,
            "repeat_results": repeat_rows,
            "repeat_metric_summary": metric_summary,
            "pooled_repeated_validation_calibration_bins": _combined_calibration_bins(
                [row["calibration_bins"] for row in repeat_rows]
            ),
            "fold_results": all_fold_rows,
            "threshold_stability": {
                "mean": float(np.mean([row["threshold"] for row in all_fold_rows])),
                "standard_deviation": float(
                    np.std([row["threshold"] for row in all_fold_rows], ddof=1)
                ),
                "minimum": float(np.min([row["threshold"] for row in all_fold_rows])),
                "maximum": float(np.max([row["threshold"] for row in all_fold_rows])),
            },
            "calibration_coefficient_stability": {
                "mean": float(
                    np.mean([row["calibration_coefficient"] for row in all_fold_rows])
                ),
                "standard_deviation": float(
                    np.std(
                        [row["calibration_coefficient"] for row in all_fold_rows], ddof=1
                    )
                ),
            },
            "numerical_comparison_to_historical_selected_model": numerical_comparison,
            "paired_uncertainty_available": False,
            "paired_uncertainty_limitation": (
                "The historical aggregate report contains no aligned participant-level predictions; "
                "a paired confidence interval cannot be reconstructed without retraining references."
            ),
            "eight_model_pr_auc_ranking": ranking,
        }

    return {
        "scope": "development_only_exploratory_tabiclv2_evaluation",
        "development_participants": len(records),
        "feature_count": int(features.shape[1]),
        "targets": list(TARGET_COLUMNS),
        "candidate_names": [TABICLV2_CANDIDATE_NAME],
        "domains": domains,
        "primary_selection_metric": "average_precision_pr_auc",
        "cross_validation": {
            "outer_splits": n_splits,
            "outer_repeats": n_repeats,
            "outer_evaluations_per_domain": len(folds),
            "inner_splits": INNER_SPLITS,
            "shared_fold_plan_sha256": fingerprint,
            "matches_historical_seven_model_fingerprint": True,
            "same_outer_folds_for_all_domains": True,
        },
        "calibration": "inner_oof_regularized_logistic_sigmoid_on_raw_probability_logit",
        "threshold_selection": "inner_oof_maximize_balanced_accuracy",
        "oof_interpretation": {
            "reported_primary_metrics": "mean_of_five_complete_repeat_level_oof_metrics",
            "participant_averaged_oof_predictions_retained": False,
            "single_final_fitted_model_evaluated": False,
        },
        "privacy": {
            "participant_identifiers_written": False,
            "row_level_predictions_written": False,
            "aggregate_results_only": True,
        },
        "final_test_evaluated": False,
        "limitations": [
            "Inner TabICLv2 contexts contain approximately 266 participants, below the reported pretraining range.",
            "The development cohort has been reused adaptively across research stages.",
            "Descriptive repeat intervals treat five correlated repeated-CV estimates cautiously and are not proof of superiority.",
            "Historical aggregate results permit numerical comparison but not paired participant-level confidence intervals.",
            "No single final TabICLv2 model was fitted, frozen, promoted, or evaluated on the locked test cohort.",
        ],
    }
