"""Build an evidence-linked development decision report from aggregate experiment outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from training.contracts import FEATURE_COLUMNS, TARGET_COLUMNS


DOMAIN_LABELS = {
    "target_communication_concern": "Communication",
    "target_cognitive_concern": "Cognitive",
    "target_social_emotional_concern": "Social-emotional",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return value


def _validate_sources(
    comparison: Mapping[str, object],
    calibration: Mapping[str, object],
) -> None:
    if comparison.get("scope") != "development_only":
        raise ValueError("The comparison source is not development-only.")
    if calibration.get("scope") != "development_only_nested_calibration":
        raise ValueError("The calibration source is not development-only.")
    if comparison.get("final_test_evaluated") is not False:
        raise ValueError("The comparison source has accessed the final test partition.")
    if calibration.get("final_test_evaluated") is not False:
        raise ValueError("The calibration source has accessed the final test partition.")
    for field in (
        "dataset_sha256",
        "split_fingerprint_sha256",
        "development_participants",
    ):
        if comparison.get(field) != calibration.get(field):
            raise ValueError(f"Development sources disagree on {field}.")


def _candidate_ranking(
    comparison: Mapping[str, object],
    target: str,
) -> list[dict[str, object]]:
    candidates = comparison["candidates"]
    if not isinstance(candidates, dict):
        raise ValueError("Comparison candidates are invalid.")
    ranking: list[dict[str, object]] = []
    for candidate_name, candidate_value in candidates.items():
        domain = candidate_value["domains"][target]
        metrics = domain["repeat_metric_summary"]
        ranking.append(
            {
                "rank": 0,
                "candidate": candidate_name,
                "model_configuration_version": candidate_value[
                    "model_configuration_version"
                ],
                "average_precision_mean": metrics["average_precision_pr_auc"]["mean"],
                "average_precision_standard_deviation": metrics[
                    "average_precision_pr_auc"
                ]["standard_deviation"],
                "roc_auc_mean": metrics["roc_auc"]["mean"],
                "brier_score_mean": metrics["brier_score"]["mean"],
            }
        )
    ranking.sort(
        key=lambda item: (-float(item["average_precision_mean"]), str(item["candidate"]))
    )
    for position, item in enumerate(ranking, start=1):
        item["rank"] = position
    return ranking


def _leading_explanations(
    comparison: Mapping[str, object],
    *,
    target: str,
    candidate_name: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    candidate = comparison["candidates"][candidate_name]
    explanation = candidate["domains"][target]["explanation"]
    values = explanation["values"]
    if explanation["method"] == "standardized_logistic_coefficient_stability":
        ordered = sorted(
            values,
            key=lambda item: -abs(float(item["mean_coefficient"])),
        )
    else:
        ordered = sorted(
            values,
            key=lambda item: -abs(float(item["mean_importance"])),
        )
    return [dict(item) for item in ordered[:limit]]


def build_development_decision_report(
    comparison: Mapping[str, object],
    calibration: Mapping[str, object],
    *,
    comparison_sha256: str,
    calibration_sha256: str,
) -> dict[str, object]:
    """Combine aggregate sources while independently reproducing every selection."""

    _validate_sources(comparison, calibration)
    calibration_domains = calibration["domains"]
    if not isinstance(calibration_domains, dict):
        raise ValueError("Calibration domains are invalid.")

    domain_decisions: dict[str, object] = {}
    for target in TARGET_COLUMNS:
        ranking = _candidate_ranking(comparison, target)
        selected_name = str(ranking[0]["candidate"])
        calibrated_domain = calibration_domains[target]
        if calibrated_domain["selected_candidate"] != selected_name:
            raise ValueError(f"Calibration selection disagrees with AP ranking for {target}.")
        threshold = calibrated_domain["threshold_selection"]
        domain_decisions[target] = {
            "display_name": DOMAIN_LABELS[target],
            "candidate_ranking": ranking,
            "selected_candidate": selected_name,
            "selection_basis": "highest_mean_repeated_cross_validation_average_precision",
            "model_configuration_version": calibrated_domain[
                "model_configuration_version"
            ],
            "calibration": {
                "method": calibrated_domain["calibration_method"],
                "inner_splits": calibrated_domain["inner_splits"],
                "raw_brier_score_mean": calibrated_domain[
                    "raw_repeat_metric_summary"
                ]["brier_score"]["mean"],
                "calibrated_brier_score_mean": calibrated_domain[
                    "calibrated_repeat_metric_summary"
                ]["brier_score"]["mean"],
                "parameter_stability": calibrated_domain[
                    "calibration_parameter_stability"
                ],
            },
            "threshold_selection": threshold,
            "leading_development_explanations": _leading_explanations(
                comparison,
                target=target,
                candidate_name=selected_name,
            ),
        }

    return {
        "artifact_schema_version": "1.0",
        "status": "frozen_development_decision_pending_final_test",
        "scope": "development_only",
        "development_participants": comparison["development_participants"],
        "final_test_evaluated": False,
        "dataset_sha256": comparison["dataset_sha256"],
        "split_fingerprint_sha256": comparison["split_fingerprint_sha256"],
        "source_reports": {
            "development_model_comparison_sha256": comparison_sha256,
            "development_calibration_sha256": calibration_sha256,
        },
        "feature_contract": {
            "feature_count": len(FEATURE_COLUMNS),
            "features": list(FEATURE_COLUMNS),
            "checkpoint_encoded_inside_pipeline": True,
        },
        "model_development_policy": {
            "candidate_count": len(comparison["candidates"]),
            "candidate_selection_primary_metric": "average_precision",
            "hyperparameter_policy": "predefined_regularized_configurations_no_search",
            "hyperparameter_limitation": (
                "The evaluated configurations are not claimed to be optimal; systematic "
                "hyperparameter optimization remains future work."
            ),
            "calibration_policy": (
                "nested sigmoid calibration using development-only out-of-fold predictions"
            ),
            "threshold_policy": (
                "maximize balanced accuracy, equivalent to Youden's J, using averaged "
                "nested calibrated out-of-fold development predictions"
            ),
            "overall_concern_policy": (
                "logical OR of the three thresholded domain decisions; no fourth model"
            ),
        },
        "domains": domain_decisions,
        "limitations": [
            "Results are internal development evidence and are not final-test performance.",
            "The outcome groups are small, so metric and threshold estimates remain uncertain.",
            "Predefined regularized configurations were compared without hyperparameter search.",
            "Model associations and permutation importance do not establish causation.",
            "The system is non-diagnostic and is not evidence of clinical readiness.",
        ],
    }


def render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Development model decision report",
        "",
        "This report records development-only decisions. The locked final-test partition has not been evaluated.",
        "",
        "## Selected configurations",
        "",
        "| Domain | Selected candidate | Mean AP | Calibrated threshold | Sensitivity | Specificity | Balanced accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    domains = report["domains"]
    for target in TARGET_COLUMNS:
        domain = domains[target]
        leading = domain["candidate_ranking"][0]
        threshold = domain["threshold_selection"]
        metrics = threshold["metrics"]
        lines.append(
            "| {display} | `{candidate}` | {ap:.4f} | {cutoff:.6f} | {sensitivity:.2%} | {specificity:.2%} | {balanced:.2%} |".format(
                display=domain["display_name"],
                candidate=domain["selected_candidate"],
                ap=float(leading["average_precision_mean"]),
                cutoff=float(threshold["threshold"]),
                sensitivity=float(metrics["sensitivity"]),
                specificity=float(metrics["specificity"]),
                balanced=float(metrics["balanced_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Methodological decisions",
            "",
            "- Candidate selection uses mean repeated-cross-validation Average Precision.",
            "- Sigmoid calibration is fitted from nested out-of-fold development predictions.",
            "- Domain thresholds maximize balanced accuracy, equivalent to Youden's J.",
            "- Predefined regularized configurations are used without hyperparameter search.",
            "- Overall concern is the logical OR of the three domain decisions.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)
