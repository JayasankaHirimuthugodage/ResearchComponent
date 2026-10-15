"""Leakage-safe, development-only evaluation of calibrated soft-voting hybrids."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from training.contracts import TARGET_COLUMNS
from training.data.development_data import ModelingRecord
from training.experiments.calibration import select_balanced_accuracy_threshold
from training.experiments.cross_validation import (
    DEFAULT_CV_SEED,
    DEFAULT_FOLD_SEARCH_ITERATIONS,
    DEFAULT_N_REPEATS,
    DEFAULT_N_SPLITS,
    SharedFold,
    create_shared_folds,
)
from training.experiments.evaluation import calibration_bins, decision_metrics
from training.models.common import (
    DEFAULT_MODEL_SEED,
    CandidateSpec,
    build_candidate_pipeline,
    fit_candidate_pipeline,
    modeling_arrays,
)
from training.models.logistic import REGULARIZED_LOGISTIC
from training.models.soft_voting import (
    DEFAULT_SOFT_VOTING_WEIGHTS,
    soft_vote,
    validate_weight_grid,
)
from training.models.xgboost import REGULARIZED_XGBOOST


CALIBRATION_EPSILON = 1e-6
HYBRID_METHODS = (
    REGULARIZED_LOGISTIC.name,
    REGULARIZED_XGBOOST.name,
    "equal_weight_soft_voting",
    "selected_weight_soft_voting",
)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _fit_calibrator(
    probabilities: np.ndarray, targets: np.ndarray, *, seed: int
) -> LogisticRegression:
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000, random_state=seed)
    model.fit(_logit(probabilities), targets)
    return model


def _inner_oof_raw_probabilities(
    features: np.ndarray,
    targets: np.ndarray,
    candidate: CandidateSpec,
    *,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    if int(np.min(np.bincount(targets))) < n_splits:
        raise ValueError("Each class must support every inner hybrid fold.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = np.full(len(targets), np.nan, dtype=float)
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(features, targets)
    ):
        pipeline = build_candidate_pipeline(candidate, seed=seed + fold_index + 1)
        fit_candidate_pipeline(
            pipeline, candidate, features[train_indices], targets[train_indices]
        )
        result[validation_indices] = pipeline.predict_proba(
            features[validation_indices]
        )[:, 1]
    if np.any(~np.isfinite(result)):
        raise RuntimeError("Inner hybrid predictions are incomplete.")
    return result


def _select_weight(
    targets: np.ndarray,
    logistic_probabilities: np.ndarray,
    xgboost_probabilities: np.ndarray,
    weights: Sequence[float],
) -> dict[str, object]:
    rows = []
    for logistic_weight in weights:
        probabilities = soft_vote(
            logistic_probabilities,
            xgboost_probabilities,
            first_weight=logistic_weight,
        )
        rows.append(
            {
                "logistic_weight": float(logistic_weight),
                "xgboost_weight": float(1.0 - logistic_weight),
                "average_precision_pr_auc": float(
                    average_precision_score(targets, probabilities)
                ),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            -float(row["average_precision_pr_auc"]),
            abs(float(row["logistic_weight"]) - 0.50),
            float(row["logistic_weight"]),
        ),
    )
    return {"selected": selected, "candidates": rows}


def _metrics_with_fold_decisions(
    targets: np.ndarray, probabilities: np.ndarray, decisions: np.ndarray
) -> dict[str, object]:
    return {
        **decision_metrics(targets, decisions),
        "average_precision_pr_auc": float(average_precision_score(targets, probabilities)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "brier_score": float(brier_score_loss(targets, probabilities)),
    }


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
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
            "mean": float(np.mean([float(row[name]) for row in rows])),
            "standard_deviation": float(
                np.std([float(row[name]) for row in rows], ddof=1)
            )
            if len(rows) > 1
            else 0.0,
            "minimum": float(np.min([float(row[name]) for row in rows])),
            "maximum": float(np.max([float(row[name]) for row in rows])),
        }
        for name in names
    }


def _complementarity(
    targets: np.ndarray,
    logistic_probabilities: np.ndarray,
    xgboost_probabilities: np.ndarray,
    logistic_decisions: np.ndarray,
    xgboost_decisions: np.ndarray,
) -> dict[str, object]:
    logistic_errors = logistic_decisions != targets
    xgboost_errors = xgboost_decisions != targets
    positives = targets == 1
    negatives = targets == 0
    if np.std(logistic_probabilities) == 0.0 or np.std(xgboost_probabilities) == 0.0:
        correlation: float | None = None
    else:
        correlation = float(
            np.corrcoef(logistic_probabilities, xgboost_probabilities)[0, 1]
        )
    return {
        "calibrated_probability_pearson_correlation": correlation,
        "decision_disagreement_count": int(np.sum(logistic_decisions != xgboost_decisions)),
        "decision_disagreement_rate": float(np.mean(logistic_decisions != xgboost_decisions)),
        "double_fault_count": int(np.sum(logistic_errors & xgboost_errors)),
        "double_fault_rate": float(np.mean(logistic_errors & xgboost_errors)),
        "logistic_correct_xgboost_wrong_count": int(
            np.sum(~logistic_errors & xgboost_errors)
        ),
        "xgboost_correct_logistic_wrong_count": int(
            np.sum(logistic_errors & ~xgboost_errors)
        ),
        "shared_false_negative_count": int(
            np.sum(positives & ~logistic_decisions & ~xgboost_decisions)
        ),
        "shared_false_positive_count": int(
            np.sum(negatives & logistic_decisions & xgboost_decisions)
        ),
    }


def _paired_bootstrap(
    targets: np.ndarray,
    reference: np.ndarray,
    hybrid: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if replicates < 100:
        raise ValueError("Paired bootstrap requires at least 100 replicates.")
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(targets == value) for value in (0, 1)]
    differences: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        rng.shuffle(sampled)
        sampled_targets = targets[sampled]
        reference_probabilities = reference[sampled]
        hybrid_probabilities = hybrid[sampled]
        differences["average_precision_pr_auc"].append(
            float(average_precision_score(sampled_targets, hybrid_probabilities))
            - float(average_precision_score(sampled_targets, reference_probabilities))
        )
        differences["brier_score"].append(
            float(brier_score_loss(sampled_targets, hybrid_probabilities))
            - float(brier_score_loss(sampled_targets, reference_probabilities))
        )
    return {
        "method": "stratified_participant_level_paired_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "differences": {
            name: {
                "mean": float(np.mean(values)),
                "confidence_interval_95": [
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                ],
            }
            for name, values in differences.items()
        },
    }


def _fold_fingerprint(folds: Sequence[SharedFold]) -> str:
    source = "\n".join(
        f"{fold.repeat_index}:{fold.fold_index}:"
        f"{','.join(str(index) for index in fold.validation_indices.tolist())}"
        for fold in folds
    ).encode("utf-8")
    return hashlib.sha256(source).hexdigest().upper()


def run_hybrid_development_evaluation(
    records: Sequence[ModelingRecord],
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    inner_splits: int = 5,
    cv_seed: int = DEFAULT_CV_SEED,
    fold_search_iterations: int = DEFAULT_FOLD_SEARCH_ITERATIONS,
    model_seed: int = DEFAULT_MODEL_SEED,
    weights: Sequence[float] = DEFAULT_SOFT_VOTING_WEIGHTS,
    bootstrap_replicates: int = 1_000,
) -> dict[str, object]:
    """Compare two calibrated members and soft-voting hybrids without final-test access."""

    weight_grid = validate_weight_grid(weights)
    features, targets = modeling_arrays(records)
    folds = create_shared_folds(
        records,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=cv_seed,
        search_iterations=fold_search_iterations,
    )
    candidates = (REGULARIZED_LOGISTIC, REGULARIZED_XGBOOST)
    domain_reports: dict[str, object] = {}

    for target_position, target_name in enumerate(TARGET_COLUMNS):
        repeat_probabilities = {
            method: {repeat: np.zeros(len(records)) for repeat in range(n_repeats)}
            for method in HYBRID_METHODS
        }
        repeat_decisions = {
            method: {repeat: np.zeros(len(records), dtype=np.int8) for repeat in range(n_repeats)}
            for method in HYBRID_METHODS
        }
        fold_rows: list[dict[str, object]] = []

        for fold in folds:
            outer_train_x = features[fold.train_indices]
            outer_train_y = targets[fold.train_indices, target_position]
            outer_validation_x = features[fold.validation_indices]
            seed = (
                model_seed
                + target_position * 1_000_000
                + fold.repeat_index * 10_000
                + fold.fold_index * 100
            )
            inner_calibrated: dict[str, np.ndarray] = {}
            validation_calibrated: dict[str, np.ndarray] = {}
            calibrators: dict[str, LogisticRegression] = {}

            for candidate_index, candidate in enumerate(candidates):
                candidate_seed = seed + candidate_index * 10
                raw_inner = _inner_oof_raw_probabilities(
                    outer_train_x,
                    outer_train_y,
                    candidate,
                    n_splits=inner_splits,
                    seed=candidate_seed + 1,
                )
                calibrator = _fit_calibrator(
                    raw_inner, outer_train_y, seed=candidate_seed + 2
                )
                calibrators[candidate.name] = calibrator
                inner_calibrated[candidate.name] = calibrator.predict_proba(
                    _logit(raw_inner)
                )[:, 1]
                pipeline = build_candidate_pipeline(candidate, seed=candidate_seed + 3)
                fit_candidate_pipeline(pipeline, candidate, outer_train_x, outer_train_y)
                raw_validation = pipeline.predict_proba(outer_validation_x)[:, 1]
                validation_calibrated[candidate.name] = calibrator.predict_proba(
                    _logit(raw_validation)
                )[:, 1]

            logistic_inner = inner_calibrated[REGULARIZED_LOGISTIC.name]
            xgboost_inner = inner_calibrated[REGULARIZED_XGBOOST.name]
            logistic_validation = validation_calibrated[REGULARIZED_LOGISTIC.name]
            xgboost_validation = validation_calibrated[REGULARIZED_XGBOOST.name]
            weight_selection = _select_weight(
                outer_train_y, logistic_inner, xgboost_inner, weight_grid
            )
            selected_weight = float(weight_selection["selected"]["logistic_weight"])
            inner_by_method = {
                REGULARIZED_LOGISTIC.name: logistic_inner,
                REGULARIZED_XGBOOST.name: xgboost_inner,
                "equal_weight_soft_voting": soft_vote(
                    logistic_inner, xgboost_inner, first_weight=0.50
                ),
                "selected_weight_soft_voting": soft_vote(
                    logistic_inner, xgboost_inner, first_weight=selected_weight
                ),
            }
            validation_by_method = {
                REGULARIZED_LOGISTIC.name: logistic_validation,
                REGULARIZED_XGBOOST.name: xgboost_validation,
                "equal_weight_soft_voting": soft_vote(
                    logistic_validation, xgboost_validation, first_weight=0.50
                ),
                "selected_weight_soft_voting": soft_vote(
                    logistic_validation,
                    xgboost_validation,
                    first_weight=selected_weight,
                ),
            }
            thresholds = {
                method: float(
                    select_balanced_accuracy_threshold(outer_train_y, probabilities)[
                        "threshold"
                    ]
                )
                for method, probabilities in inner_by_method.items()
            }
            for method, probabilities in validation_by_method.items():
                repeat_probabilities[method][fold.repeat_index][
                    fold.validation_indices
                ] = probabilities
                repeat_decisions[method][fold.repeat_index][fold.validation_indices] = (
                    probabilities >= thresholds[method]
                ).astype(np.int8)
            fold_rows.append(
                {
                    "repeat": fold.repeat_index + 1,
                    "fold": fold.fold_index + 1,
                    "training_participants": int(len(fold.train_indices)),
                    "validation_participants": int(len(fold.validation_indices)),
                    "training_positives": int(np.sum(outer_train_y)),
                    "validation_positives": int(
                        np.sum(targets[fold.validation_indices, target_position])
                    ),
                    "selected_logistic_weight": selected_weight,
                    "selected_xgboost_weight": float(1.0 - selected_weight),
                    "inner_weight_candidates": weight_selection["candidates"],
                    "thresholds": thresholds,
                }
            )

        domain_targets = targets[:, target_position]
        methods: dict[str, object] = {}
        averaged_probabilities: dict[str, np.ndarray] = {}
        averaged_decisions: dict[str, np.ndarray] = {}
        for method in HYBRID_METHODS:
            repeat_metrics = [
                _metrics_with_fold_decisions(
                    domain_targets,
                    repeat_probabilities[method][repeat],
                    repeat_decisions[method][repeat],
                )
                for repeat in range(n_repeats)
            ]
            averaged_probabilities[method] = np.mean(
                list(repeat_probabilities[method].values()), axis=0
            )
            averaged_decisions[method] = (
                np.mean(list(repeat_decisions[method].values()), axis=0) >= 0.50
            ).astype(np.int8)
            methods[method] = {
                "repeat_metric_summary": _summary(repeat_metrics),
                "averaged_oof_metrics": _metrics_with_fold_decisions(
                    domain_targets,
                    averaged_probabilities[method],
                    averaged_decisions[method],
                ),
                "calibration_bins": calibration_bins(
                    domain_targets, averaged_probabilities[method]
                ),
            }

        reference_method = (
            REGULARIZED_LOGISTIC.name
            if target_name == "target_cognitive_concern"
            else REGULARIZED_XGBOOST.name
        )
        domain_reports[target_name] = {
            "positive_count": int(np.sum(domain_targets)),
            "negative_count": int(len(domain_targets) - np.sum(domain_targets)),
            "reference_existing_selected_model": reference_method,
            "fold_results": fold_rows,
            "methods": methods,
            "selected_weight_stability": {
                "logistic_weight_mean": float(
                    np.mean([row["selected_logistic_weight"] for row in fold_rows])
                ),
                "logistic_weight_standard_deviation": float(
                    np.std(
                        [row["selected_logistic_weight"] for row in fold_rows], ddof=1
                    )
                )
                if len(fold_rows) > 1
                else 0.0,
                "selection_counts": {
                    str(weight): int(
                        sum(
                            bool(np.isclose(row["selected_logistic_weight"], weight))
                            for row in fold_rows
                        )
                    )
                    for weight in weight_grid
                },
            },
            "complementarity": _complementarity(
                domain_targets,
                averaged_probabilities[REGULARIZED_LOGISTIC.name],
                averaged_probabilities[REGULARIZED_XGBOOST.name],
                averaged_decisions[REGULARIZED_LOGISTIC.name],
                averaged_decisions[REGULARIZED_XGBOOST.name],
            ),
            "paired_uncertainty_selected_hybrid_vs_existing_reference": _paired_bootstrap(
                domain_targets,
                averaged_probabilities[reference_method],
                averaged_probabilities["selected_weight_soft_voting"],
                replicates=bootstrap_replicates,
                seed=cv_seed + target_position + 50_000,
            ),
        }

    return {
        "scope": "development_only_nested_hybrid_evaluation",
        "development_participants": len(records),
        "feature_count": int(features.shape[1]),
        "targets": list(TARGET_COLUMNS),
        "base_candidates": [candidate.name for candidate in candidates],
        "weight_grid_logistic": list(weight_grid),
        "calibration": "inner_oof_regularized_logistic_sigmoid_on_logit",
        "threshold_selection": "inner_oof_maximize_balanced_accuracy",
        "cross_validation": {
            "outer_splits": n_splits,
            "outer_repeats": n_repeats,
            "inner_splits": inner_splits,
            "cv_seed": cv_seed,
            "shared_fold_plan_sha256": _fold_fingerprint(folds),
            "same_outer_folds_for_all_domains_and_methods": True,
        },
        "domains": domain_reports,
        "privacy": {
            "participant_identifiers_written": False,
            "row_level_predictions_written": False,
            "report_contains_aggregate_results_only": True,
        },
        "final_test_evaluated": False,
        "limitations": [
            "This is internal development validation, not external or clinical validation.",
            "The previously examined 125-participant test set is not evidence for a new hybrid.",
            "Any selected hybrid requires a new untouched cohort or external validation.",
        ],
    }
