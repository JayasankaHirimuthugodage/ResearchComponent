"""One-time aggregate evaluation of frozen models on the locked final-test partition."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost

from inference.contracts import DOMAIN_IDS
from inference.model_artifact import CalibratedDomainModel
from training.artifacts.build_frozen_models import MODEL_RELEASE_VERSION
from training.contracts import TARGET_COLUMNS, repository_root
from training.data.development_data import load_final_test_dataset
from training.experiments.development_decision_report import load_json, sha256_file
from training.experiments.evaluation import (
    binary_probability_metrics,
    calibration_bins,
    decision_metrics,
)
from training.models.common import modeling_arrays


FINAL_REPORT_SCHEMA_VERSION = "1.0"
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261008


def _atomic_write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"Refusing to repeat locked-test evaluation while {path.name} exists."
        )
    temporary = path.with_name(f".{path.name}.tmp")
    content = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stratified_bootstrap_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    negative = np.flatnonzero(targets == 0)
    positive = np.flatnonzero(targets == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Bootstrap intervals require both target classes.")
    rng = np.random.default_rng(seed)
    names = (
        "sensitivity",
        "specificity",
        "precision",
        "negative_predictive_value",
        "f1",
        "balanced_accuracy",
        "average_precision_pr_auc",
        "roc_auc",
        "brier_score",
    )
    values = {name: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for name in names}
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        metrics = binary_probability_metrics(
            targets[sampled],
            probabilities[sampled],
            threshold=threshold,
        )
        for name in names:
            values[name][index] = float(metrics[name])
    return {
        name: {
            "lower_95_percentile": float(np.percentile(metric_values, 2.5)),
            "upper_95_percentile": float(np.percentile(metric_values, 97.5)),
        }
        for name, metric_values in values.items()
    }


def _load_and_verify_artifacts(root: Path) -> tuple[dict[str, CalibratedDomainModel], dict[str, object]]:
    release_root = root / "models" / MODEL_RELEASE_VERSION
    release_manifest_path = release_root / "release-manifest.json"
    release_manifest = load_json(release_manifest_path)
    if release_manifest.get("status") != "frozen_development_models_pending_final_test":
        raise ValueError("Model release is not frozen for final evaluation.")
    if release_manifest.get("final_test_evaluated") is not False:
        raise ValueError("Model release already records final-test access.")

    artifacts: dict[str, CalibratedDomainModel] = {}
    for domain in DOMAIN_IDS:
        entry = release_manifest["domains"][domain]
        artifact_path = (root / entry["artifact"]).resolve()
        manifest_path = (root / entry["manifest"]).resolve()
        if not artifact_path.is_relative_to(release_root.resolve()):
            raise ValueError("Artifact path escaped the frozen release directory.")
        if sha256_file(artifact_path) != entry["artifact_sha256"]:
            raise ValueError(f"Artifact checksum mismatch for {domain}.")
        if sha256_file(manifest_path) != entry["manifest_sha256"]:
            raise ValueError(f"Manifest checksum mismatch for {domain}.")
        domain_manifest = load_json(manifest_path)
        if domain_manifest.get("final_test_evaluated") is not False:
            raise ValueError(f"Domain artifact already records final-test access for {domain}.")
        artifact = joblib.load(artifact_path)
        if not isinstance(artifact, CalibratedDomainModel) or artifact.domain != domain:
            raise ValueError(f"Loaded artifact is invalid for {domain}.")
        if float(domain_manifest["threshold"]["value"]) != artifact.threshold:
            raise ValueError(f"Artifact threshold differs from its manifest for {domain}.")
        artifacts[domain] = artifact
    release_manifest["release_manifest_sha256"] = sha256_file(release_manifest_path)
    return artifacts, release_manifest


def main() -> int:
    root = repository_root()
    output_path = root / "reports" / "generated" / "final-locked-test-report.json"
    if output_path.exists():
        raise FileExistsError(
            "The locked-test report already exists; repeated evaluation is prohibited."
        )

    artifacts, release_manifest = _load_and_verify_artifacts(root)
    final_test = load_final_test_dataset()
    if final_test.dataset_sha256 != release_manifest["dataset_sha256"]:
        raise ValueError("Frozen release and final-test dataset checksums differ.")
    if final_test.split_fingerprint_sha256 != release_manifest["split_fingerprint_sha256"]:
        raise ValueError("Frozen release and final-test split fingerprints differ.")

    features, targets = modeling_arrays(final_test.records)
    domains: dict[str, object] = {}
    decision_columns: list[np.ndarray] = []
    for target_position, (target_name, domain) in enumerate(
        zip(TARGET_COLUMNS, DOMAIN_IDS, strict=True)
    ):
        artifact = artifacts[domain]
        probabilities = np.asarray(
            [artifact.positive_probability(tuple(row)) for row in features],
            dtype=float,
        )
        metrics = binary_probability_metrics(
            targets[:, target_position],
            probabilities,
            threshold=artifact.threshold,
        )
        decisions = (probabilities >= artifact.threshold).astype(np.int8)
        decision_columns.append(decisions)
        domains[target_name] = {
            "domain": domain,
            "model_configuration_version": artifact.model_configuration_version,
            "calibration_method": artifact.calibration_method,
            "threshold": artifact.threshold,
            "participants": len(final_test.records),
            "negative_count": int(np.sum(targets[:, target_position] == 0)),
            "positive_count": int(np.sum(targets[:, target_position] == 1)),
            "metrics": metrics,
            "bootstrap_95_percentile_intervals": _stratified_bootstrap_intervals(
                targets[:, target_position],
                probabilities,
                threshold=artifact.threshold,
                seed=BOOTSTRAP_SEED + target_position,
            ),
            "calibration_bins": calibration_bins(
                targets[:, target_position], probabilities
            ),
        }

    overall_targets = np.max(targets, axis=1)
    overall_decisions = np.max(np.column_stack(decision_columns), axis=1)
    report = {
        "artifact_schema_version": FINAL_REPORT_SCHEMA_VERSION,
        "status": "completed_one_time_locked_test_evaluation",
        "scope": "locked_final_test_only",
        "final_test_participants": len(final_test.records),
        "dataset_sha256": final_test.dataset_sha256,
        "split_fingerprint_sha256": final_test.split_fingerprint_sha256,
        "manifest_sha256": final_test.manifest_sha256,
        "model_release_version": MODEL_RELEASE_VERSION,
        "model_release_manifest_sha256": release_manifest["release_manifest_sha256"],
        "models_refit_or_adjusted_on_final_test": False,
        "thresholds_adjusted_on_final_test": False,
        "final_test_evaluated": True,
        "domains": domains,
        "derived_any_concern": {
            "definition": "OR of the three frozen domain decisions; no fourth model",
            "metrics": decision_metrics(overall_targets, overall_decisions),
        },
        "uncertainty": {
            "method": "stratified_nonparametric_bootstrap_percentile_intervals",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "note": (
                "Intervals quantify sampling uncertainty in this locked partition and do "
                "not establish clinical validity."
            ),
        },
        "privacy": {
            "participant_identifiers_in_report": False,
            "row_level_predictions_in_report": False,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_write_once(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "final_test_participants": report["final_test_participants"],
                "models_refit_or_adjusted_on_final_test": False,
                "thresholds_adjusted_on_final_test": False,
                "report_path": str(output_path.relative_to(root)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
