"""One authorized secondary exploratory TabICLv2 test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from tabicl import TabICLClassifier

from inference.contracts import DOMAIN_IDS
from training.contracts import FEATURE_COLUMNS, TARGET_COLUMNS, repository_root
from training.data.development_data import load_development_dataset, load_final_test_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.development_decision_report import load_json, sha256_file
from training.experiments.tabiclv2_evaluation import fold_plan_fingerprint, modeling_arrays
from training.tabiclv2.candidate import (
    TABICLV2_CHECKPOINT_NAME,
    TABICLV2_CHECKPOINT_SHA256,
    TABICLV2_CONFIGURATION_VERSION,
    TABICLV2_PACKAGE_VERSION,
    tabiclv2_frame,
)


REPORT_SCHEMA_VERSION = "secondary-exploratory-tabiclv2-test-evaluation-v1"
REPORT_RELATIVE_PATH = Path("reports/generated/secondary-tabiclv2-test-evaluation-report.json")
RELEASE_RELATIVE_PATH = Path("models/experimental-tabiclv2-secondary-v1/release-manifest.json")
FINALIZATION_RELATIVE_PATH = Path("reports/generated/development-tabiclv2-finalization-report.json")
HISTORICAL_TEST_RELATIVE_PATH = Path("reports/generated/final-locked-test-report.json")
MODEL_SEED = 20260822
BOOTSTRAP_SEED = 20261016
BOOTSTRAP_REPLICATES = 2_000
EPS = 1e-6
METRIC_NAMES = (
    "average_precision_pr_auc", "roc_auc", "accuracy", "balanced_accuracy",
    "sensitivity", "specificity", "precision", "negative_predictive_value",
    "f1", "brier_score",
)


def _metrics(targets: np.ndarray, probabilities: np.ndarray, *, threshold: float) -> dict[str, object]:
    decisions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(targets, decisions, labels=[0, 1]).ravel()
    return {
        "average_precision_pr_auc": float(average_precision_score(targets, probabilities)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "accuracy": float(accuracy_score(targets, decisions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, decisions)),
        "sensitivity": float(recall_score(targets, decisions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(targets, decisions, zero_division=0)),
        "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else 0.0,
        "f1": float(f1_score(targets, decisions, zero_division=0)),
        "brier_score": float(brier_score_loss(targets, probabilities)),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }


def _sha256_upper(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to repeat the one-time TabICLv2 evaluation.")
    temporary = path.with_name(f".{path.name}.tmp")
    content = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _calibrate(raw_probabilities: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    clipped = np.clip(np.asarray(raw_probabilities, dtype=float), EPS, 1.0 - EPS)
    logits = np.log(clipped / (1.0 - clipped))
    values = coefficient * logits + intercept
    calibrated = np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))
    if np.any(~np.isfinite(calibrated)):
        raise RuntimeError("Frozen calibration produced a non-finite probability.")
    return calibrated


def _paired_bootstrap(
    targets: np.ndarray,
    original_probabilities: np.ndarray,
    tabiclv2_probabilities: np.ndarray,
    *,
    original_threshold: float,
    tabiclv2_threshold: float,
    seed: int,
) -> dict[str, object]:
    negative = np.flatnonzero(targets == 0)
    positive = np.flatnonzero(targets == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Paired bootstrap requires both target classes.")
    rng = np.random.default_rng(seed)
    differences: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate((
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ))
        original = _metrics(
            targets[sampled], original_probabilities[sampled], threshold=original_threshold
        )
        tabiclv2 = _metrics(
            targets[sampled], tabiclv2_probabilities[sampled], threshold=tabiclv2_threshold
        )
        for name in METRIC_NAMES:
            differences[name].append(float(tabiclv2[name]) - float(original[name]))
    return {
        "method": "stratified_participant_level_paired_bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "tabiclv2_minus_original_95_percentile_intervals": {
            name: {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }
            for name, values in differences.items()
        },
    }


def _preflight(root: Path, checkpoint: Path) -> dict[str, object]:
    release_path = root / RELEASE_RELATIVE_PATH
    finalization_path = root / FINALIZATION_RELATIVE_PATH
    release = load_json(release_path)
    finalization = load_json(finalization_path)
    if release.get("status") != "frozen_experimental_candidates_pending_secondary_test_approval":
        raise ValueError("TabICLv2 release is not frozen pending secondary approval.")
    if release.get("deployment_status") != "not_deployed":
        raise ValueError("TabICLv2 release deployment state changed.")
    if release.get("test_cohort_evaluated_for_these_candidates") is not False:
        raise ValueError("TabICLv2 release already records test evaluation.")
    if finalization.get("status") != "development_finalization_complete_pending_test_approval":
        raise ValueError("Development finalization is not complete and frozen.")
    if finalization.get("final_test_evaluated") is not False:
        raise ValueError("Finalization report already records test evaluation.")
    if tuple(release.get("feature_contract", ())) != FEATURE_COLUMNS:
        raise ValueError("Frozen 16-feature contract mismatch.")
    if tuple(release.get("targets", ())) != TARGET_COLUMNS:
        raise ValueError("Frozen target contract mismatch.")
    tabicl = release.get("tabiclv2", {})
    if tabicl.get("configuration_version") != TABICLV2_CONFIGURATION_VERSION:
        raise ValueError("Frozen TabICLv2 configuration mismatch.")
    if tabicl.get("package_version") != TABICLV2_PACKAGE_VERSION:
        raise ValueError("Frozen TabICLv2 package version mismatch.")
    if importlib.metadata.version("tabicl") != TABICLV2_PACKAGE_VERSION:
        raise ValueError("Installed TabICLv2 version mismatch.")
    if checkpoint.name != TABICLV2_CHECKPOINT_NAME:
        raise ValueError("Checkpoint logical filename mismatch.")
    resolved_checkpoint = checkpoint.resolve()
    if _sha256_upper(resolved_checkpoint) != TABICLV2_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA-256 mismatch.")
    if tabicl.get("checkpoint_sha256") != TABICLV2_CHECKPOINT_SHA256:
        raise ValueError("Release checkpoint SHA-256 mismatch.")

    development = load_development_dataset()
    if development.dataset_sha256 != release.get("dataset_sha256"):
        raise ValueError("Development dataset checksum differs from frozen release.")
    if development.split_fingerprint_sha256 != release.get("split_fingerprint_sha256"):
        raise ValueError("Development split fingerprint differs from frozen release.")
    folds = create_shared_folds(development.records, seed=int(release["seeds"]["cv"]))
    if fold_plan_fingerprint(folds) != release.get("shared_fold_plan_sha256"):
        raise ValueError("Frozen shared fold-plan fingerprint mismatch.")

    for target in TARGET_COLUMNS:
        domain = release["domains"][target]
        if domain.get("candidate") != "tabiclv2_default_eight_view":
            raise ValueError(f"Unexpected frozen candidate for {target}.")
        if domain.get("hybrid_candidate_frozen") is not False or domain.get("hybrid") is not None:
            raise ValueError(f"A hybrid is unexpectedly frozen for {target}.")
        calibration = domain.get("calibration", {})
        threshold = domain.get("threshold", {})
        if calibration.get("method") != "sigmoid_on_logit_of_five_repeat_averaged_raw_oof_probability":
            raise ValueError(f"Frozen calibration method mismatch for {target}.")
        if not np.isfinite(float(calibration["coefficient"])) or not np.isfinite(float(calibration["intercept"])):
            raise ValueError(f"Invalid frozen calibrator for {target}.")
        if not 0.0 < float(threshold["value"]) < 1.0:
            raise ValueError(f"Invalid frozen threshold for {target}.")

    base_python = root / ".venv" / "Scripts" / "python.exe"
    helper_environment = os.environ.copy()
    helper_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    original_check = subprocess.run(
        [str(base_python), "-m", "training.experiments.secondary_tabiclv2_original_predictions", "--preflight"], cwd=root,
        env=helper_environment, capture_output=True, text=True, check=True,
    )
    original_verification = json.loads(original_check.stdout)
    original_release = load_json(root / "models/slc-development-models-v1/release-manifest.json")
    original_release["release_manifest_sha256"] = original_verification["release_manifest_sha256"]
    historical = load_json(root / HISTORICAL_TEST_RELATIVE_PATH)
    if historical.get("status") != "completed_one_time_locked_test_evaluation":
        raise ValueError("Historical original-model test report is unavailable.")
    if historical.get("model_release_manifest_sha256") != original_release["release_manifest_sha256"]:
        raise ValueError("Original frozen model release differs from its historical evaluation.")
    if release.get("dataset_sha256") != original_release.get("dataset_sha256"):
        raise ValueError("TabICLv2 and original releases reference different datasets.")
    if release.get("split_fingerprint_sha256") != original_release.get("split_fingerprint_sha256"):
        raise ValueError("TabICLv2 and original releases reference different splits.")
    return {
        "release": release,
        "finalization": finalization,
        "development": development,
        "original_release": original_release,
        "historical": historical,
        "verification": {
            "release_manifest_sha256": sha256_file(release_path),
            "finalization_report_sha256": sha256_file(finalization_path),
            "checkpoint_sha256": _sha256_upper(resolved_checkpoint),
            "feature_contract_verified": True,
            "development_dataset_and_split_verified": True,
            "calibrators_and_thresholds_frozen": True,
            "original_artifacts_checksum_verified": original_verification["artifacts_verified"],
            "test_outcomes_accessed_during_preflight": False,
        },
    }


def _progress(completed: int, total: int, started: float, target: str) -> None:
    elapsed = time.perf_counter() - started
    eta = (elapsed / completed) * (total - completed) if completed else 0.0
    print(f"[{completed}/{total}] {target} complete; elapsed={elapsed:.1f}s; ETA={eta:.1f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(os.environ.get(
            "TABICLV2_CHECKPOINT",
            Path(sys.prefix) / "hf-cache/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt",
        )),
    )
    args = parser.parse_args()
    root = repository_root()
    report_path = root / REPORT_RELATIVE_PATH
    if report_path.exists():
        raise FileExistsError("The one-time TabICLv2 test report already exists.")
    checkpoint = args.checkpoint.absolute()
    preflight = _preflight(root, checkpoint)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", **preflight["verification"]}, indent=2))
        return 0

    lock_path = report_path.with_suffix(report_path.suffix + ".lock")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(str(os.getpid()))
        started = time.perf_counter()
        development = preflight["development"]
        development_features, development_targets = modeling_arrays(development.records)
        final_test = load_final_test_dataset()
        release = preflight["release"]
        if final_test.dataset_sha256 != release["dataset_sha256"]:
            raise ValueError("Final-test dataset checksum differs from frozen release.")
        if final_test.split_fingerprint_sha256 != release["split_fingerprint_sha256"]:
            raise ValueError("Final-test split fingerprint differs from frozen release.")
        test_features, test_targets = modeling_arrays(final_test.records)
        helper_environment = os.environ.copy()
        helper_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        helper_result = subprocess.run(
            [str(root / ".venv/Scripts/python.exe"), "-m",
             "training.experiments.secondary_tabiclv2_original_predictions",
             "--emit-aggregate-inputs"],
            cwd=root, env=helper_environment, capture_output=True, text=True, check=True,
        )
        original_inputs = json.loads(helper_result.stdout)
        domains: dict[str, object] = {}
        for position, (target, domain_id) in enumerate(zip(TARGET_COLUMNS, DOMAIN_IDS, strict=True)):
            frozen = release["domains"][target]
            model = TabICLClassifier(
                n_estimators=8, model_path=str(checkpoint), allow_auto_download=False,
                device="cpu", use_amp=False, use_fa3=False, offload_mode=False,
                random_state=MODEL_SEED + position, n_jobs=4, verbose=False,
            )
            model.fit(tabiclv2_frame(development_features), development_targets[:, position])
            raw = np.asarray(model.predict_proba(tabiclv2_frame(test_features))[:, 1], dtype=float)
            del model
            calibration = frozen["calibration"]
            tabiclv2_probabilities = _calibrate(
                raw, float(calibration["coefficient"]), float(calibration["intercept"])
            )
            tabiclv2_threshold = float(frozen["threshold"]["value"])
            targets = test_targets[:, position]
            tabiclv2_metrics = _metrics(
                targets, tabiclv2_probabilities, threshold=tabiclv2_threshold
            )

            original_entry = original_inputs["domains"][target]
            if not np.array_equal(targets, np.asarray(original_entry["targets"], dtype=np.int8)):
                raise ValueError(f"Original and TabICLv2 target alignment mismatch for {target}.")
            original_probabilities = np.asarray(original_entry["probabilities"], dtype=float)
            original_threshold = float(original_entry["threshold"])
            original_metrics = _metrics(targets, original_probabilities, threshold=original_threshold)
            historical_metrics = preflight["historical"]["domains"][target]["metrics"]
            for metric in METRIC_NAMES:
                if not np.isclose(float(original_metrics[metric]), float(historical_metrics[metric]), atol=1e-12):
                    raise ValueError(f"Regenerated original metric mismatch for {target}: {metric}.")
            differences = {
                name: float(tabiclv2_metrics[name]) - float(original_metrics[name])
                for name in METRIC_NAMES
            }
            domains[target] = {
                "domain": domain_id,
                "participants": len(final_test.records),
                "actual_negative_count": int(np.sum(targets == 0)),
                "actual_positive_count": int(np.sum(targets == 1)),
                "original_selected": {
                    "model_configuration_version": original_entry["model_configuration_version"],
                    "threshold": original_threshold,
                    "metrics": original_metrics,
                },
                "tabiclv2": {
                    "candidate": frozen["candidate"],
                    "calibration": calibration,
                    "threshold": tabiclv2_threshold,
                    "metrics": tabiclv2_metrics,
                },
                "tabiclv2_minus_original_metric_differences": differences,
                "paired_uncertainty": _paired_bootstrap(
                    targets, original_probabilities, tabiclv2_probabilities,
                    original_threshold=original_threshold,
                    tabiclv2_threshold=tabiclv2_threshold,
                    seed=BOOTSTRAP_SEED + position,
                ),
            }
            _progress(position + 1, len(TARGET_COLUMNS), started, target)

        runtime_seconds = time.perf_counter() - started
        report = {
            "artifact_schema_version": REPORT_SCHEMA_VERSION,
            "status": "completed_secondary_exploratory_tabiclv2_evaluation",
            "scope": "secondary_exploratory_evaluation_on_previously_evaluated_test_cohort",
            "cohort_previously_evaluated": True,
            "fresh_independent_validation": False,
            "external_validation": False,
            "confirmatory_evidence": False,
            "test_participants": len(final_test.records),
            "dataset_sha256": final_test.dataset_sha256,
            "split_fingerprint_sha256": final_test.split_fingerprint_sha256,
            "split_manifest_sha256": final_test.manifest_sha256,
            "verification": preflight["verification"],
            "models_refit_or_adjusted_on_test": False,
            "calibrators_adjusted_on_test": False,
            "thresholds_adjusted_on_test": False,
            "hybrids_evaluated": False,
            "deployment_changed": False,
            "domains": domains,
            "uncertainty": {
                "method": "stratified_participant_level_paired_bootstrap",
                "replicates": BOOTSTRAP_REPLICATES,
                "limitations": "Descriptive intervals on a previously evaluated cohort; small positive counts and multiple metrics limit precision.",
            },
            "privacy": {
                "participant_identifiers_written": False,
                "row_level_predictions_written": False,
                "labelled_development_context_persisted": False,
            },
            "limitations": [
                "The cohort was previously evaluated using the original selected models.",
                "This is secondary exploratory evidence, not fresh independent, external, or confirmatory validation.",
                "Small positive counts make metric and difference estimates imprecise.",
                "The final 500-participant context differs from development cross-validation context sizes.",
                "Results must not be used to tune models, calibration, or thresholds on this cohort.",
                "Targets are questionnaire-defined developmental follow-up concerns, not clinical diagnoses.",
            ],
            "runtime_seconds": runtime_seconds,
            "runtime_versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "scikit_learn": sklearn.__version__, "torch": torch.__version__,
                "tabicl": importlib.metadata.version("tabicl"),
            },
        }
        _atomic_write_once(report_path, report)
        print(json.dumps({
            "status": report["status"], "scope": report["scope"],
            "test_participants": report["test_participants"],
            "runtime_seconds": runtime_seconds, "report": str(REPORT_RELATIVE_PATH),
        }, indent=2))
        return 0
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
