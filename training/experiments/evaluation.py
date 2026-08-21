"""Development-only, aggregate evaluation shared by all estimator candidates."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

import numpy as np
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
from sklearn.inspection import permutation_importance

from training.contracts import (
    APPROVED_DECISION_THRESHOLD,
    DECISION_THRESHOLD_APPROVAL_STATEMENT,
    DECISION_THRESHOLD_APPROVAL_STATUS,
    DECISION_THRESHOLD_VERSION,
    EXPERIMENTAL_DECISION_THRESHOLD_STATUS,
    FEATURE_COLUMNS,
    LOGISTIC_MODEL_CONFIGURATION_VERSION,
    TARGET_COLUMNS,
)
from training.experiments.cross_validation import (
    DEFAULT_CV_SEED,
    DEFAULT_FOLD_SEARCH_ITERATIONS,
    DEFAULT_N_REPEATS,
    DEFAULT_N_SPLITS,
    SharedFold,
    create_shared_folds,
)
from training.data.development_data import ModelingRecord
from training.models.common import (
    DEFAULT_MODEL_SEED,
    CandidateSpec,
    build_candidate_pipeline,
    modeling_arrays,
    transformed_feature_names,
)
from training.models.registry import DEFAULT_CANDIDATES


PERMUTATION_IMPORTANCE_REPEATS = 5
SUPPORTED_EXPLANATION_METHODS = frozenset(
    {
        "standardized_logistic_coefficient_stability",
        "held_out_permutation_importance_average_precision",
    }
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def decision_metrics(y_true: np.ndarray, decisions: np.ndarray) -> dict[str, object]:
    """Return class-decision metrics without interpreting a probability cutoff clinically."""

    tn, fp, fn, tp = confusion_matrix(y_true, decisions, labels=[0, 1]).ravel()
    return {
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "sensitivity": float(recall_score(y_true, decisions, zero_division=0)),
        "specificity": _safe_ratio(int(tn), int(tn + fp)),
        "precision": float(precision_score(y_true, decisions, zero_division=0)),
        "f1": float(f1_score(y_true, decisions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, decisions)),
        "accuracy": float(accuracy_score(y_true, decisions)),
    }


def binary_probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = APPROVED_DECISION_THRESHOLD,
) -> dict[str, object]:
    """Evaluate binary probabilities at the supplied operational cutoff."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("Probability threshold must be strictly between zero and one.")
    if len(y_true) != len(probabilities) or not len(y_true):
        raise ValueError("Targets and probabilities must be non-empty and equal length.")
    if np.unique(y_true).size != 2:
        raise ValueError("Probability metrics require both target classes.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Predicted probabilities must be in the closed interval [0,1].")

    decisions = (probabilities >= threshold).astype(np.int8)
    return {
        **decision_metrics(y_true, decisions),
        "average_precision_pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Return aggregate reliability bins; empty bins are omitted."""

    if n_bins < 2:
        raise ValueError("Calibration requires at least two bins.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.digitize(probabilities, edges[1:-1], right=False)
    bins: list[dict[str, float | int]] = []
    for index in range(n_bins):
        mask = assignments == index
        count = int(np.sum(mask))
        if not count:
            continue
        bins.append(
            {
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "count": count,
                "mean_predicted_probability": float(np.mean(probabilities[mask])),
                "observed_positive_rate": float(np.mean(y_true[mask])),
            }
        )
    return bins


def decision_threshold_metadata(
    threshold: float,
    *,
    model_configuration_version: str,
) -> dict[str, object]:
    """Attach approval only to the exact cutoff and reviewed model version."""

    cutoff_matches = bool(
        np.isclose(threshold, APPROVED_DECISION_THRESHOLD, rtol=0.0, atol=1e-12)
    )
    threshold_is_approved = bool(
        cutoff_matches
        and model_configuration_version == LOGISTIC_MODEL_CONFIGURATION_VERSION
    )
    if threshold_is_approved:
        status = DECISION_THRESHOLD_APPROVAL_STATUS
    elif cutoff_matches:
        status = EXPERIMENTAL_DECISION_THRESHOLD_STATUS
    else:
        status = "unapproved_experimental_override"
    return {
        "value": threshold,
        "version": DECISION_THRESHOLD_VERSION if threshold_is_approved else None,
        "status": status,
        "approval_statement": (
            DECISION_THRESHOLD_APPROVAL_STATEMENT if threshold_is_approved else None
        ),
        "model_configuration_version": model_configuration_version,
        "scope": f"model_configuration:{model_configuration_version}",
        "note": (
            "The probability cutoff creates the model decision and is separate "
            "from the follow-up score <= 50 observed-target definition. Approval "
            "does not automatically transfer to another model configuration. A "
            "numerically identical cutoff remains experimental when the model "
            "configuration has not been approved."
        ),
    }


def _numeric_metric_summary(repeat_metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    metric_names = (
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "accuracy",
        "average_precision_pr_auc",
        "roc_auc",
        "brier_score",
    )
    summary: dict[str, object] = {}
    for name in metric_names:
        values = np.asarray([float(metrics[name]) for metrics in repeat_metrics])
        summary[name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return summary


def _fold_plan_fingerprint(folds: Sequence[SharedFold]) -> str:
    source = "\n".join(
        f"{fold.repeat_index}:{fold.fold_index}:"
        f"{','.join(str(index) for index in fold.validation_indices.tolist())}"
        for fold in folds
    ).encode("utf-8")
    return hashlib.sha256(source).hexdigest().upper()


def _coefficient_stability(
    feature_names: tuple[str, ...], coefficient_rows: Sequence[np.ndarray]
) -> list[dict[str, float | str]]:
    if not coefficient_rows:
        return []
    matrix = np.vstack(coefficient_rows)
    if matrix.shape[1] != len(feature_names):
        raise RuntimeError("Coefficient columns do not match transformed feature names.")
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(feature_names):
        values = matrix[:, index]
        positive_fraction = float(np.mean(values > 0.0))
        negative_fraction = float(np.mean(values < 0.0))
        rows.append(
            {
                "transformed_feature": name,
                "mean_coefficient": float(np.mean(values)),
                "standard_deviation": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                "positive_sign_fraction": positive_fraction,
                "negative_sign_fraction": negative_fraction,
                "sign_agreement": max(positive_fraction, negative_fraction),
            }
        )
    return rows


def _permutation_importance_stability(
    feature_names: tuple[str, ...], importance_rows: Sequence[np.ndarray]
) -> list[dict[str, float | str]]:
    if not importance_rows:
        return []
    matrix = np.vstack(importance_rows)
    if matrix.shape[1] != len(feature_names):
        raise RuntimeError("Permutation-importance columns do not match input features.")
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(feature_names):
        values = matrix[:, index]
        rows.append(
            {
                "feature": name,
                "mean_importance": float(np.mean(values)),
                "standard_deviation": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                "positive_importance_fraction": float(np.mean(values > 0.0)),
            }
        )
    return rows


def run_development_cross_validation(
    records: Sequence[ModelingRecord],
    *,
    candidates: Sequence[CandidateSpec] = DEFAULT_CANDIDATES,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    cv_seed: int = DEFAULT_CV_SEED,
    fold_search_iterations: int = DEFAULT_FOLD_SEARCH_ITERATIONS,
    model_seed: int = DEFAULT_MODEL_SEED,
    threshold: float = APPROVED_DECISION_THRESHOLD,
) -> dict[str, object]:
    """Compare candidates on shared development folds and return aggregate results only."""

    if not candidates:
        raise ValueError("At least one estimator candidate is required.")
    candidate_names = [candidate.name for candidate in candidates]
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("Estimator candidate names must be unique.")
    if any(
        candidate.explanation_method not in SUPPORTED_EXPLANATION_METHODS
        for candidate in candidates
    ):
        raise ValueError("An estimator candidate has an unsupported explanation method.")

    features, targets = modeling_arrays(records)
    folds = create_shared_folds(
        records,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=cv_seed,
        search_iterations=fold_search_iterations,
    )
    candidate_reports: dict[str, object] = {}
    result: dict[str, object] = {
        "scope": "development_only",
        "development_participants": len(records),
        "target_positive_counts": {
            target: int(np.sum(targets[:, position]))
            for position, target in enumerate(TARGET_COLUMNS)
        },
        "cross_validation": {
            "strategy": "repeated_stratified_three-label-combination",
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "cv_seed": cv_seed,
            "checkpoint_balance_search_iterations_per_repeat": fold_search_iterations,
            "selected_allocation_seeds": sorted(
                {fold.allocation_seed for fold in folds}
            ),
            "shared_fold_plan_sha256": _fold_plan_fingerprint(folds),
            "same_folds_for_all_domains_and_candidates": True,
        },
        "model_seed": model_seed,
        "comparison_threshold": {
            "value": threshold,
            "approval_is_candidate_specific": True,
            "note": (
                "Candidate-level threshold metadata determines whether this shared "
                "numeric comparison cutoff is approved or experimental."
            ),
        },
        "candidates": candidate_reports,
    }

    for candidate in candidates:
        probability_sums = np.zeros_like(targets, dtype=float)
        domain_fold_results: dict[str, list[dict[str, object]]] = {
            target: [] for target in TARGET_COLUMNS
        }
        domain_repeat_predictions: dict[str, dict[int, np.ndarray]] = {
            target: {
                repeat: np.zeros(len(records), dtype=float)
                for repeat in range(n_repeats)
            }
            for target in TARGET_COLUMNS
        }
        coefficient_rows: dict[str, list[np.ndarray]] = defaultdict(list)
        coefficient_names: dict[str, tuple[str, ...]] = {}
        permutation_rows: dict[str, list[np.ndarray]] = defaultdict(list)

        for fold in folds:
            for target_position, target_name in enumerate(TARGET_COLUMNS):
                seed = (
                    model_seed
                    + fold.repeat_index * 10_000
                    + fold.fold_index * 100
                    + target_position
                )
                pipeline = build_candidate_pipeline(candidate, seed=seed)
                train_y = targets[fold.train_indices, target_position]
                validation_y = targets[fold.validation_indices, target_position]
                pipeline.fit(features[fold.train_indices], train_y)
                probabilities = pipeline.predict_proba(
                    features[fold.validation_indices]
                )[:, 1]
                probability_sums[fold.validation_indices, target_position] += probabilities
                domain_repeat_predictions[target_name][fold.repeat_index][
                    fold.validation_indices
                ] = probabilities

                estimator = pipeline.named_steps["estimator"]
                if (
                    candidate.explanation_method
                    == "standardized_logistic_coefficient_stability"
                ):
                    if not hasattr(estimator, "coef_"):
                        raise RuntimeError("The coefficient explanation requires coef_.")
                    names = transformed_feature_names(pipeline)
                    if target_name in coefficient_names and coefficient_names[target_name] != names:
                        raise RuntimeError("Transformed feature order changed across folds.")
                    coefficient_names[target_name] = names
                    coefficient_rows[target_name].append(
                        np.asarray(estimator.coef_[0], dtype=float)
                    )
                elif (
                    candidate.explanation_method
                    == "held_out_permutation_importance_average_precision"
                ):
                    importance = permutation_importance(
                        pipeline,
                        features[fold.validation_indices],
                        validation_y,
                        scoring="average_precision",
                        n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
                        random_state=seed + 1_000_000,
                        n_jobs=1,
                    )
                    if importance.importances.shape != (
                        len(FEATURE_COLUMNS),
                        PERMUTATION_IMPORTANCE_REPEATS,
                    ):
                        raise RuntimeError("Permutation importance has an invalid shape.")
                    permutation_rows[target_name].extend(
                        np.asarray(row, dtype=float)
                        for row in importance.importances.T
                    )

                metrics = binary_probability_metrics(
                    validation_y,
                    probabilities,
                    threshold=threshold,
                )
                domain_fold_results[target_name].append(
                    {
                        "repeat": fold.repeat_index + 1,
                        "fold": fold.fold_index + 1,
                        "training_participants": int(len(fold.train_indices)),
                        "validation_participants": int(len(fold.validation_indices)),
                        "training_positives": int(np.sum(train_y)),
                        "validation_positives": int(np.sum(validation_y)),
                        "metrics": metrics,
                    }
                )

        averaged_probabilities = probability_sums / n_repeats
        domain_reports: dict[str, object] = {}
        for target_position, target_name in enumerate(TARGET_COLUMNS):
            repeat_metrics = [
                binary_probability_metrics(
                    targets[:, target_position],
                    domain_repeat_predictions[target_name][repeat],
                    threshold=threshold,
                )
                for repeat in range(n_repeats)
            ]
            participant_oof_metrics = binary_probability_metrics(
                targets[:, target_position],
                averaged_probabilities[:, target_position],
                threshold=threshold,
            )
            if (
                candidate.explanation_method
                == "standardized_logistic_coefficient_stability"
            ):
                explanation: dict[str, object] = {
                    "method": candidate.explanation_method,
                    "feature_level": "transformed_pipeline_feature",
                    "scoring": None,
                    "values": _coefficient_stability(
                        coefficient_names.get(target_name, ()),
                        coefficient_rows[target_name],
                    ),
                }
            else:
                explanation = {
                    "method": candidate.explanation_method,
                    "feature_level": "original_conceptual_input",
                    "scoring": "average_precision",
                    "permutation_repeats_per_fold": PERMUTATION_IMPORTANCE_REPEATS,
                    "values": _permutation_importance_stability(
                        FEATURE_COLUMNS,
                        permutation_rows[target_name],
                    ),
                }
            domain_reports[target_name] = {
                "negative_count": int(len(records) - np.sum(targets[:, target_position])),
                "positive_count": int(np.sum(targets[:, target_position])),
                "fold_results": domain_fold_results[target_name],
                "repeat_metric_summary": _numeric_metric_summary(repeat_metrics),
                "averaged_participant_oof_metrics": participant_oof_metrics,
                "calibration_bins": calibration_bins(
                    targets[:, target_position],
                    averaged_probabilities[:, target_position],
                ),
                "explanation": explanation,
            }

        domain_decisions = averaged_probabilities >= threshold
        overall_true = np.max(targets, axis=1)
        overall_decision = np.any(domain_decisions, axis=1).astype(np.int8)
        candidate_reports[candidate.name] = {
            "model_configuration_version": candidate.version,
            "parameters": dict(candidate.public_parameters),
            "models_per_fold": len(TARGET_COLUMNS),
            "decision_threshold": decision_threshold_metadata(
                threshold,
                model_configuration_version=candidate.version,
            ),
            "explanation_method": candidate.explanation_method,
            "domains": domain_reports,
            "derived_any_concern": {
                "definition": "OR of the three domain decisions; no fourth model",
                "metrics": decision_metrics(overall_true, overall_decision),
            },
        }

    return result
