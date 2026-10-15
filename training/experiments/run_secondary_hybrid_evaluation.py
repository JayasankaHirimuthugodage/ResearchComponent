"""One-time secondary exploratory evaluation on a previously evaluated cohort."""

from __future__ import annotations

import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost

from inference.contracts import DOMAIN_IDS
from inference.model_artifact import CalibratedDomainModel
from training.artifacts.build_frozen_hybrids import HYBRID_RELEASE_VERSION
from training.contracts import TARGET_COLUMNS, repository_root
from training.data.development_data import load_final_test_dataset
from training.experiments.development_decision_report import load_json, sha256_file
from training.experiments.evaluation import binary_probability_metrics, calibration_bins
from training.experiments.run_locked_test_evaluation import _load_and_verify_artifacts
from training.models.common import modeling_arrays
from training.models.frozen_hybrid import FrozenHybridDomainModel


REPORT_SCHEMA_VERSION = "secondary-exploratory-hybrid-evaluation-v1"
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261015
METRIC_NAMES = (
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


def _atomic_write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            "Refusing to repeat the secondary evaluation while its report exists."
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


def _load_hybrids(root: Path) -> tuple[dict[str, FrozenHybridDomainModel], dict[str, object]]:
    release_root = root / "models" / HYBRID_RELEASE_VERSION
    release_path = release_root / "release-manifest.json"
    release = load_json(release_path)
    if release.get("status") != "frozen_development_hybrids_pending_secondary_evaluation":
        raise ValueError("Hybrid release is not frozen for secondary evaluation.")
    if release.get("final_test_evaluated") is not False:
        raise ValueError("Hybrid release already records final-test access.")
    if release.get("deployment_authorized") is not False:
        raise ValueError("Exploratory hybrid release cannot be deployment-authorized.")
    if release.get("weights") != {"logistic": 0.5, "xgboost": 0.5}:
        raise ValueError("Hybrid release does not contain the frozen equal weights.")

    artifacts: dict[str, FrozenHybridDomainModel] = {}
    for domain in DOMAIN_IDS:
        entry = release["domains"][domain]
        artifact_path = (root / entry["artifact"]).resolve()
        manifest_path = (root / entry["manifest"]).resolve()
        if not artifact_path.is_relative_to(release_root.resolve()):
            raise ValueError("Hybrid artifact path escaped its release directory.")
        if sha256_file(artifact_path) != entry["artifact_sha256"]:
            raise ValueError(f"Hybrid artifact checksum mismatch for {domain}.")
        if sha256_file(manifest_path) != entry["manifest_sha256"]:
            raise ValueError(f"Hybrid manifest checksum mismatch for {domain}.")
        manifest = load_json(manifest_path)
        if manifest.get("final_test_evaluated") is not False:
            raise ValueError(f"Hybrid manifest already records test access for {domain}.")
        artifact = joblib.load(artifact_path)
        if not isinstance(artifact, FrozenHybridDomainModel) or artifact.domain != domain:
            raise ValueError(f"Loaded hybrid artifact is invalid for {domain}.")
        if artifact.logistic_weight != 0.5 or artifact.xgboost_weight != 0.5:
            raise ValueError(f"Frozen hybrid weights changed for {domain}.")
        if float(manifest["threshold"]["value"]) != artifact.threshold:
            raise ValueError(f"Hybrid threshold differs from its manifest for {domain}.")
        artifacts[domain] = artifact
    release["release_manifest_sha256"] = sha256_file(release_path)
    return artifacts, release


def _paired_bootstrap(
    targets: np.ndarray,
    original_probabilities: np.ndarray,
    hybrid_probabilities: np.ndarray,
    *,
    original_threshold: float,
    hybrid_threshold: float,
    seed: int,
) -> dict[str, object]:
    negative = np.flatnonzero(targets == 0)
    positive = np.flatnonzero(targets == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Paired bootstrap requires both target classes.")
    rng = np.random.default_rng(seed)
    original_values: dict[str, list[float]] = defaultdict(list)
    hybrid_values: dict[str, list[float]] = defaultdict(list)
    difference_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        original = binary_probability_metrics(
            targets[sampled],
            original_probabilities[sampled],
            threshold=original_threshold,
        )
        hybrid = binary_probability_metrics(
            targets[sampled], hybrid_probabilities[sampled], threshold=hybrid_threshold
        )
        for name in METRIC_NAMES:
            original_value = float(original[name])
            hybrid_value = float(hybrid[name])
            original_values[name].append(original_value)
            hybrid_values[name].append(hybrid_value)
            difference_values[name].append(hybrid_value - original_value)

    def intervals(values: dict[str, list[float]]) -> dict[str, object]:
        return {
            name: {
                "lower_95_percentile": float(np.percentile(metric_values, 2.5)),
                "upper_95_percentile": float(np.percentile(metric_values, 97.5)),
            }
            for name, metric_values in values.items()
        }

    return {
        "method": "stratified_participant_level_paired_bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "original_95_percentile_intervals": intervals(original_values),
        "hybrid_95_percentile_intervals": intervals(hybrid_values),
        "hybrid_minus_original_95_percentile_intervals": intervals(difference_values),
    }


def main() -> int:
    root = repository_root()
    output_path = (
        root
        / "reports"
        / "generated"
        / "secondary-exploratory-hybrid-evaluation-report.json"
    )
    if output_path.exists():
        raise FileExistsError("The secondary exploratory report already exists.")

    original_artifacts, original_release = _load_and_verify_artifacts(root)
    hybrid_artifacts, hybrid_release = _load_hybrids(root)
    if original_release["dataset_sha256"] != hybrid_release["dataset_sha256"]:
        raise ValueError("Original and hybrid releases use different datasets.")
    if original_release["split_fingerprint_sha256"] != hybrid_release["split_fingerprint_sha256"]:
        raise ValueError("Original and hybrid releases use different partitions.")

    previous_report_path = root / "reports" / "generated" / "final-locked-test-report.json"
    previous_report = load_json(previous_report_path)
    if previous_report.get("status") != "completed_one_time_locked_test_evaluation":
        raise ValueError("The historical locked-test result is unavailable.")

    final_test = load_final_test_dataset()
    if final_test.dataset_sha256 != hybrid_release["dataset_sha256"]:
        raise ValueError("Frozen releases and evaluation dataset checksums differ.")
    if final_test.split_fingerprint_sha256 != hybrid_release["split_fingerprint_sha256"]:
        raise ValueError("Frozen releases and evaluation split fingerprints differ.")
    features, targets = modeling_arrays(final_test.records)

    domains: dict[str, object] = {}
    for target_position, (target_name, domain) in enumerate(
        zip(TARGET_COLUMNS, DOMAIN_IDS, strict=True)
    ):
        original_artifact: CalibratedDomainModel = original_artifacts[domain]
        hybrid_artifact = hybrid_artifacts[domain]
        original_probabilities = np.asarray(
            [original_artifact.positive_probability(tuple(row)) for row in features],
            dtype=float,
        )
        hybrid_probabilities = np.asarray(
            [hybrid_artifact.positive_probability(tuple(row)) for row in features],
            dtype=float,
        )
        domain_targets = targets[:, target_position]
        original_metrics = binary_probability_metrics(
            domain_targets,
            original_probabilities,
            threshold=original_artifact.threshold,
        )
        hybrid_metrics = binary_probability_metrics(
            domain_targets,
            hybrid_probabilities,
            threshold=hybrid_artifact.threshold,
        )
        metric_differences = {
            name: float(hybrid_metrics[name]) - float(original_metrics[name])
            for name in METRIC_NAMES
        }
        domains[target_name] = {
            "domain": domain,
            "participants": len(final_test.records),
            "negative_count": int(np.sum(domain_targets == 0)),
            "positive_count": int(np.sum(domain_targets == 1)),
            "original_individual": {
                "model_configuration_version": original_artifact.model_configuration_version,
                "threshold": original_artifact.threshold,
                "metrics": original_metrics,
                "calibration_bins": calibration_bins(domain_targets, original_probabilities),
            },
            "equal_weight_hybrid": {
                "hybrid_configuration_version": hybrid_artifact.hybrid_configuration_version,
                "weights": {"logistic": 0.5, "xgboost": 0.5},
                "threshold": hybrid_artifact.threshold,
                "metrics": hybrid_metrics,
                "calibration_bins": calibration_bins(domain_targets, hybrid_probabilities),
            },
            "hybrid_minus_original_metric_differences": metric_differences,
            "paired_uncertainty": _paired_bootstrap(
                domain_targets,
                original_probabilities,
                hybrid_probabilities,
                original_threshold=original_artifact.threshold,
                hybrid_threshold=hybrid_artifact.threshold,
                seed=BOOTSTRAP_SEED + target_position,
            ),
        }

    report = {
        "artifact_schema_version": REPORT_SCHEMA_VERSION,
        "status": "completed_secondary_exploratory_evaluation",
        "scope": "secondary_exploratory_evaluation_on_previously_evaluated_test_cohort",
        "cohort_previously_evaluated": True,
        "fresh_independent_test": False,
        "external_validation": False,
        "confirmatory_evidence": False,
        "test_participants": len(final_test.records),
        "dataset_sha256": final_test.dataset_sha256,
        "split_fingerprint_sha256": final_test.split_fingerprint_sha256,
        "manifest_sha256": final_test.manifest_sha256,
        "historical_locked_test_report": str(previous_report_path.relative_to(root)),
        "historical_locked_test_report_sha256": sha256_file(previous_report_path),
        "original_model_release_manifest_sha256": original_release["release_manifest_sha256"],
        "hybrid_model_release_manifest_sha256": hybrid_release["release_manifest_sha256"],
        "models_refit_or_adjusted_on_test": False,
        "calibrators_adjusted_on_test": False,
        "weights_adjusted_on_test": False,
        "thresholds_adjusted_on_test": False,
        "deployment_changed": False,
        "domains": domains,
        "uncertainty": {
            "method": "stratified_participant_level_paired_bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "limitations": (
                "Intervals are descriptive because this cohort was evaluated previously; "
                "small positive counts and multiple metrics limit precision."
            ),
        },
        "privacy": {
            "participant_identifiers_in_report": False,
            "row_level_predictions_in_report": False,
        },
        "limitations": [
            "This cohort was previously evaluated for the original individual models.",
            "This is not a fresh independent test, external validation, or confirmatory evidence.",
            "No result may be used to revise the frozen hybrid without beginning a new study.",
        ],
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_write_once(output_path, report)
    print(json.dumps({
        "status": report["status"],
        "scope": report["scope"],
        "test_participants": report["test_participants"],
        "cohort_previously_evaluated": report["cohort_previously_evaluated"],
        "models_refit_or_adjusted_on_test": report["models_refit_or_adjusted_on_test"],
        "weights_adjusted_on_test": report["weights_adjusted_on_test"],
        "thresholds_adjusted_on_test": report["thresholds_adjusted_on_test"],
        "report_path": str(output_path.relative_to(root)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
