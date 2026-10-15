"""Nested development-only comparison of five baselines, CatBoost and EBM."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from catboost import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from training.additional_models.catboost_candidate import (
    CATBOOST_CONFIGURATIONS,
    catboost_frame,
    catboost_probabilities,
    fit_catboost,
)
from training.additional_models.contracts import AdditionalCandidateConfiguration
from training.additional_models.ebm_candidate import (
    EBM_CONFIGURATIONS,
    ebm_probabilities,
    fit_ebm,
)
from training.contracts import FEATURE_COLUMNS, TARGET_COLUMNS
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
from training.models.registry import DEFAULT_CANDIDATES


INNER_SPLITS = 3
BOOTSTRAP_REPLICATES = 1_000
CALIBRATION_EPSILON = 1e-6


@dataclass(frozen=True)
class CandidateFamily:
    name: str
    configurations: tuple[object, ...]
    fit: Callable[[object, np.ndarray, np.ndarray, int], object]
    predict: Callable[[object, np.ndarray], np.ndarray]
    explanation_kind: str


def _fit_existing(
    configuration: object, features: np.ndarray, targets: np.ndarray, seed: int
) -> object:
    candidate = configuration
    if not isinstance(candidate, CandidateSpec):
        raise TypeError("Existing candidate configuration is invalid.")
    pipeline = build_candidate_pipeline(candidate, seed=seed)
    return fit_candidate_pipeline(pipeline, candidate, features, targets)


def _predict_existing(model: object, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def _fit_catboost_adapter(
    configuration: object, features: np.ndarray, targets: np.ndarray, seed: int
) -> object:
    if not isinstance(configuration, AdditionalCandidateConfiguration):
        raise TypeError("CatBoost configuration is invalid.")
    return fit_catboost(configuration, features, targets, seed=seed)


def _fit_ebm_adapter(
    configuration: object, features: np.ndarray, targets: np.ndarray, seed: int
) -> object:
    if not isinstance(configuration, AdditionalCandidateConfiguration):
        raise TypeError("EBM configuration is invalid.")
    return fit_ebm(configuration, features, targets, seed=seed)


def default_candidate_families() -> tuple[CandidateFamily, ...]:
    existing = tuple(
        CandidateFamily(
            name=candidate.name,
            configurations=(candidate,),
            fit=_fit_existing,
            predict=_predict_existing,
            explanation_kind=candidate.explanation_method,
        )
        for candidate in DEFAULT_CANDIDATES
    )
    return (
        *existing,
        CandidateFamily(
            name="catboost_ordered_balanced",
            configurations=CATBOOST_CONFIGURATIONS,
            fit=_fit_catboost_adapter,
            predict=catboost_probabilities,
            explanation_kind="catboost_tree_shap_raw_logit_then_sigmoid_calibration",
        ),
        CandidateFamily(
            name="ebm_main_effects_balanced",
            configurations=EBM_CONFIGURATIONS,
            fit=_fit_ebm_adapter,
            predict=ebm_probabilities,
            explanation_kind="ebm_additive_terms_raw_logit_then_sigmoid_calibration",
        ),
    )


def _configuration_name(configuration: object) -> str:
    if isinstance(configuration, CandidateSpec):
        return configuration.version
    if isinstance(configuration, AdditionalCandidateConfiguration):
        return configuration.configuration_version
    raise TypeError("Unknown candidate configuration type.")


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _fit_calibrator(
    probabilities: np.ndarray, targets: np.ndarray, *, seed: int
) -> LogisticRegression:
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000, random_state=seed)
    calibrator.fit(_logit(probabilities), targets)
    return calibrator


def _inner_splits(targets: np.ndarray, *, seed: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    return tuple(splitter.split(np.zeros(len(targets)), targets))


def _configuration_oof(
    family: CandidateFamily,
    configuration: object,
    features: np.ndarray,
    targets: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> np.ndarray:
    probabilities = np.full(len(targets), np.nan, dtype=float)
    for fold_index, (train_indices, validation_indices) in enumerate(splits):
        model = family.fit(
            configuration,
            features[train_indices],
            targets[train_indices],
            seed + fold_index,
        )
        probabilities[validation_indices] = family.predict(
            model, features[validation_indices]
        )
    if np.any(~np.isfinite(probabilities)):
        raise RuntimeError("Inner OOF predictions are incomplete.")
    return probabilities


def _select_configuration(
    family: CandidateFamily,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> tuple[object, np.ndarray, list[dict[str, object]]]:
    splits = _inner_splits(targets, seed=seed)
    results: list[tuple[object, np.ndarray, float]] = []
    for config_index, configuration in enumerate(family.configurations):
        probabilities = _configuration_oof(
            family,
            configuration,
            features,
            targets,
            splits,
            seed=seed + config_index * 100,
        )
        results.append(
            (
                configuration,
                probabilities,
                float(average_precision_score(targets, probabilities)),
            )
        )
    selected = min(results, key=lambda row: (-row[2], _configuration_name(row[0])))
    summary = [
        {
            "configuration_version": _configuration_name(configuration),
            "inner_oof_average_precision": score,
        }
        for configuration, _, score in results
    ]
    return selected[0], selected[1], summary


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
        "sensitivity", "specificity", "precision", "negative_predictive_value",
        "f1", "balanced_accuracy", "accuracy", "average_precision_pr_auc",
        "roc_auc", "brier_score",
    )
    return {
        name: {
            "mean": float(np.mean([float(row[name]) for row in rows])),
            "standard_deviation": float(np.std([float(row[name]) for row in rows], ddof=1))
            if len(rows) > 1 else 0.0,
            "minimum": float(np.min([float(row[name]) for row in rows])),
            "maximum": float(np.max([float(row[name]) for row in rows])),
        }
        for name in names
    }


def _explanation_reconstruction(
    family: CandidateFamily,
    model: object,
    calibrator: LogisticRegression,
    features: np.ndarray,
) -> float | None:
    if not len(features):
        return None
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    sample = features[: min(5, len(features))]
    if family.name == "catboost_ordered_balanced":
        frame = catboost_frame(sample)
        raw = np.asarray(model.predict(frame, prediction_type="RawFormulaVal"), dtype=float)
        shap = np.asarray(
            model.get_feature_importance(Pool(frame, cat_features=[FEATURE_COLUMNS[0]]), type="ShapValues"),
            dtype=float,
        )
        raw_reconstructed = np.sum(shap[:, :-1], axis=1) + shap[:, -1]
    elif family.name == "ebm_main_effects_balanced":
        terms = np.asarray(model.eval_terms(sample), dtype=float)
        raw = _logit(np.asarray(model.predict_proba(sample)[:, 1], dtype=float))[:, 0]
        raw_reconstructed = np.sum(terms, axis=1) + float(model.intercept_[0])
    else:
        return None
    calibrated_direct = coefficient * raw + intercept
    calibrated_reconstructed = coefficient * raw_reconstructed + intercept
    return float(np.max(np.abs(calibrated_direct - calibrated_reconstructed)))


def _paired_bootstrap(
    targets: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(targets == value) for value in (0, 1)]
    differences: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in classes]
        )
        y = targets[sampled]
        differences["average_precision_pr_auc"].append(
            float(average_precision_score(y, candidate[sampled]))
            - float(average_precision_score(y, reference[sampled]))
        )
        differences["roc_auc"].append(
            float(roc_auc_score(y, candidate[sampled]))
            - float(roc_auc_score(y, reference[sampled]))
        )
        differences["brier_score"].append(
            float(brier_score_loss(y, candidate[sampled]))
            - float(brier_score_loss(y, reference[sampled]))
        )
    return {
        "method": "stratified_participant_level_paired_bootstrap_on_averaged_oof_predictions",
        "replicates": BOOTSTRAP_REPLICATES,
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


def run_additional_model_evaluation(
    records: Sequence[ModelingRecord],
    *,
    families: Sequence[CandidateFamily] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    cv_seed: int = DEFAULT_CV_SEED,
    fold_search_iterations: int = DEFAULT_FOLD_SEARCH_ITERATIONS,
    model_seed: int = DEFAULT_MODEL_SEED,
) -> dict[str, object]:
    features, targets = modeling_arrays(records)
    candidate_families = tuple(families or default_candidate_families())
    if len({family.name for family in candidate_families}) != len(candidate_families):
        raise ValueError("Candidate family names must be unique.")
    folds = create_shared_folds(
        records, n_splits=n_splits, n_repeats=n_repeats, seed=cv_seed,
        search_iterations=fold_search_iterations,
    )
    domains: dict[str, object] = {}
    historical_references = {
        "target_communication_concern": "xgboost_regularized_balanced",
        "target_cognitive_concern": "logistic_l2_balanced",
        "target_social_emotional_concern": "xgboost_regularized_balanced",
    }

    for target_position, target_name in enumerate(TARGET_COLUMNS):
        domain_targets = targets[:, target_position]
        candidate_reports: dict[str, object] = {}
        averaged_by_candidate: dict[str, np.ndarray] = {}
        for family_index, family in enumerate(candidate_families):
            repeat_probabilities = {
                repeat: np.zeros(len(records), dtype=float) for repeat in range(n_repeats)
            }
            repeat_decisions = {
                repeat: np.zeros(len(records), dtype=np.int8) for repeat in range(n_repeats)
            }
            fold_rows: list[dict[str, object]] = []
            reconstruction_errors: list[float] = []
            for fold in folds:
                seed = (
                    model_seed + target_position * 1_000_000 + family_index * 100_000
                    + fold.repeat_index * 10_000 + fold.fold_index * 100
                )
                train_x = features[fold.train_indices]
                train_y = domain_targets[fold.train_indices]
                selected, inner_raw, configuration_rows = _select_configuration(
                    family, train_x, train_y, seed=seed + 1
                )
                calibrator = _fit_calibrator(inner_raw, train_y, seed=seed + 50)
                inner_calibrated = calibrator.predict_proba(_logit(inner_raw))[:, 1]
                threshold_result = select_balanced_accuracy_threshold(
                    train_y, inner_calibrated
                )
                threshold = float(threshold_result["threshold"])
                model = family.fit(selected, train_x, train_y, seed + 75)
                raw_validation = family.predict(model, features[fold.validation_indices])
                calibrated_validation = calibrator.predict_proba(
                    _logit(raw_validation)
                )[:, 1]
                repeat_probabilities[fold.repeat_index][fold.validation_indices] = calibrated_validation
                repeat_decisions[fold.repeat_index][fold.validation_indices] = (
                    calibrated_validation >= threshold
                ).astype(np.int8)
                reconstruction = _explanation_reconstruction(
                    family, model, calibrator, features[fold.validation_indices]
                )
                if reconstruction is not None:
                    reconstruction_errors.append(reconstruction)
                fold_rows.append({
                    "repeat": fold.repeat_index + 1,
                    "fold": fold.fold_index + 1,
                    "training_participants": int(len(fold.train_indices)),
                    "validation_participants": int(len(fold.validation_indices)),
                    "training_positives": int(np.sum(train_y)),
                    "validation_positives": int(np.sum(domain_targets[fold.validation_indices])),
                    "selected_configuration_version": _configuration_name(selected),
                    "configuration_comparison": configuration_rows,
                    "calibration_coefficient": float(calibrator.coef_[0, 0]),
                    "calibration_intercept": float(calibrator.intercept_[0]),
                    "threshold": threshold,
                })
            repeat_metrics = [
                _metrics(
                    domain_targets,
                    repeat_probabilities[repeat],
                    repeat_decisions[repeat],
                )
                for repeat in range(n_repeats)
            ]
            averaged = np.mean(list(repeat_probabilities.values()), axis=0)
            majority = (
                np.mean(list(repeat_decisions.values()), axis=0) >= 0.5
            ).astype(np.int8)
            averaged_by_candidate[family.name] = averaged
            selection_counts = {
                version: sum(
                    row["selected_configuration_version"] == version for row in fold_rows
                )
                for version in sorted(
                    {row["selected_configuration_version"] for row in fold_rows}
                )
            }
            candidate_reports[family.name] = {
                "configuration_versions": [
                    _configuration_name(configuration)
                    for configuration in family.configurations
                ],
                "selected_configuration_counts": selection_counts,
                "explanation_method": family.explanation_kind,
                "maximum_calibrated_logit_reconstruction_error": (
                    max(reconstruction_errors) if reconstruction_errors else None
                ),
                "fold_results": fold_rows,
                "repeat_metric_summary": _metric_summary(repeat_metrics),
                "averaged_oof_metrics": _metrics(domain_targets, averaged, majority),
                "calibration_bins": calibration_bins(domain_targets, averaged),
                "threshold_stability": {
                    "mean": float(np.mean([row["threshold"] for row in fold_rows])),
                    "standard_deviation": float(np.std([row["threshold"] for row in fold_rows], ddof=1)),
                    "minimum": float(np.min([row["threshold"] for row in fold_rows])),
                    "maximum": float(np.max([row["threshold"] for row in fold_rows])),
                },
                "calibration_coefficient_stability": {
                    "mean": float(np.mean([row["calibration_coefficient"] for row in fold_rows])),
                    "standard_deviation": float(np.std([row["calibration_coefficient"] for row in fold_rows], ddof=1)),
                },
            }
        ranking = sorted(
            (
                {
                    "candidate": name,
                    "mean_average_precision": report["repeat_metric_summary"]["average_precision_pr_auc"]["mean"],
                    "average_precision_standard_deviation": report["repeat_metric_summary"]["average_precision_pr_auc"]["standard_deviation"],
                    "mean_roc_auc": report["repeat_metric_summary"]["roc_auc"]["mean"],
                    "mean_brier_score": report["repeat_metric_summary"]["brier_score"]["mean"],
                }
                for name, report in candidate_reports.items()
            ),
            key=lambda row: (-float(row["mean_average_precision"]), str(row["candidate"])),
        )
        reference = historical_references[target_name]
        comparisons = {}
        if reference in averaged_by_candidate:
            comparisons = {
                name: _paired_bootstrap(
                    domain_targets,
                    averaged_by_candidate[reference],
                    averaged_by_candidate[name],
                    seed=cv_seed + target_position * 100 + index,
                )
                for index, name in enumerate(
                    ("catboost_ordered_balanced", "ebm_main_effects_balanced")
                )
                if name in averaged_by_candidate
            }
        domains[target_name] = {
            "positive_count": int(np.sum(domain_targets)),
            "negative_count": int(len(domain_targets) - np.sum(domain_targets)),
            "historical_selected_reference": reference,
            "ranking": ranking,
            "candidates": candidate_reports,
            "paired_uncertainty_vs_historical_reference": comparisons,
        }

    return {
        "scope": "development_only_nested_additional_model_comparison",
        "development_participants": len(records),
        "feature_count": int(features.shape[1]),
        "targets": list(TARGET_COLUMNS),
        "candidate_names": [family.name for family in candidate_families],
        "domains": domains,
        "primary_selection_metric": "average_precision_pr_auc",
        "cross_validation": {
            "outer_splits": n_splits,
            "outer_repeats": n_repeats,
            "inner_splits": INNER_SPLITS,
            "shared_fold_plan_sha256": _fold_fingerprint(folds),
            "same_outer_folds_for_all_domains_and_candidates": True,
        },
        "calibration": "inner_oof_regularized_logistic_sigmoid_on_raw_probability_logit",
        "threshold_selection": "inner_oof_maximize_balanced_accuracy",
        "privacy": {
            "participant_identifiers_written": False,
            "row_level_predictions_written": False,
            "aggregate_results_only": True,
        },
        "final_test_evaluated": False,
        "limitations": [
            "The development set has been reused adaptively across research stages.",
            "Paired bootstrap intervals condition on averaged OOF predictions and do not refit the full pipeline.",
            "Any numerical improvement remains development evidence and does not justify deployment.",
        ],
    }
