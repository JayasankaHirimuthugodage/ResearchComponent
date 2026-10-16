"""CLI for the isolated individual TabICLv2 development-only evaluation."""

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
import pandas
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold

from training.contracts import (
    DEVELOPMENT_SPLIT_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    TARGET_DEFINITION_VERSION,
    default_dataset_path,
    repository_root,
)
from training.data.development_data import (
    default_manifest_path,
    default_split_summary_path,
    load_development_dataset,
)
from training.data.dataset_validation import sha256_file
from training.experiments.cross_validation import (
    DEFAULT_CV_SEED,
    DEFAULT_FOLD_SEARCH_ITERATIONS,
    DEFAULT_N_REPEATS,
    DEFAULT_N_SPLITS,
    create_shared_folds,
)
from training.experiments.tabiclv2_evaluation import (
    fold_plan_fingerprint,
    modeling_arrays,
    run_tabiclv2_evaluation,
)


DEFAULT_MODEL_SEED = 20260822


REPORT_SCHEMA = "tabiclv2-development-report-v1"
PROGRESS_SCHEMA = "tabiclv2-development-progress-v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "BDC7DBD5E4FF21F8F0456FCF90C6B7CDF72DBEA960F2D05B19BEC19F9B3D4ED0"
)
EXPECTED_HISTORICAL_STATUS = "development_additional_candidates_not_final_models"


def _atomic_json_write(path: Path, value: object, *, refuse_overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
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


def _default_checkpoint_path() -> Path:
    return (
        Path(sys.prefix)
        / "hf-cache"
        / "hub"
        / "models--jingang--TabICL"
        / "snapshots"
        / "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
        / "tabicl-classifier-v2-20260212.ckpt"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Run individual TabICLv2 on the development partition only."
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--split-summary", type=Path, default=default_split_summary_path())
    parser.add_argument("--checkpoint", type=Path, default=_default_checkpoint_path())
    parser.add_argument(
        "--historical-report",
        type=Path,
        default=root / "reports" / "generated" / "development-additional-model-comparison-report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "generated" / "development-tabiclv2-model-comparison-report.json",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=root / "reports" / "generated" / "development-tabiclv2-progress.json",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--cv-seed", type=int, default=DEFAULT_CV_SEED)
    parser.add_argument(
        "--fold-search-iterations", type=int, default=DEFAULT_FOLD_SEARCH_ITERATIONS
    )
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--compatibility-check-only", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_checkpoint(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Verified TabICLv2 checkpoint is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("TabICLv2 checkpoint checksum does not match the approved artifact.")
    return digest


def _compatibility_check(records, checkpoint_path: Path) -> dict[str, object]:
    from training.tabiclv2.candidate import tabiclv2_probabilities

    features, targets = modeling_arrays(records)
    outer = create_shared_folds(records)[0]
    outer_x = features[outer.train_indices]
    outer_y = targets[outer.train_indices, 0]
    inner_train, inner_validation = next(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=DEFAULT_MODEL_SEED).split(
            np.zeros(len(outer_y)), outer_y
        )
    )
    probabilities = tabiclv2_probabilities(
        outer_x[inner_train],
        outer_y[inner_train],
        outer_x[inner_validation[:10]],
        checkpoint_path=checkpoint_path,
        seed=DEFAULT_MODEL_SEED,
    )
    return {
        "status": "development_partition_compatibility_check_passed",
        "training_context": int(len(inner_train)),
        "validation_examples": int(len(probabilities)),
        "finite_probabilities": bool(np.isfinite(probabilities).all()),
        "feature_count": int(features.shape[1]),
        "final_test_evaluated": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = repository_root().resolve()
    output = args.output.resolve()
    progress_path = args.progress.resolve()
    if not output.is_relative_to(root) or not progress_path.is_relative_to(root):
        raise ValueError("Experiment outputs must remain inside the repository workspace.")
    if output.exists() and not args.compatibility_check_only:
        raise FileExistsError(f"Completed report already exists: {output}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha256 = _validate_checkpoint(checkpoint_path)
    if importlib.metadata.version("tabicl") != "2.2.0":
        raise ValueError("The isolated environment does not contain approved tabicl==2.2.0.")

    development = load_development_dataset(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        summary_path=args.split_summary,
    )
    if args.compatibility_check_only:
        print(json.dumps(_compatibility_check(development.records, checkpoint_path), indent=2))
        return 0

    historical_path = args.historical_report.resolve()
    historical = _load_json(historical_path)
    if historical.get("status") != EXPECTED_HISTORICAL_STATUS:
        raise ValueError("Historical seven-model report has an unexpected status.")
    if historical.get("final_test_evaluated") is not False:
        raise ValueError("Historical comparison must be development-only.")
    folds = create_shared_folds(
        development.records,
        n_splits=args.folds,
        n_repeats=args.repeats,
        seed=args.cv_seed,
        search_iterations=args.fold_search_iterations,
    )
    fingerprint = fold_plan_fingerprint(folds)
    if historical.get("cross_validation", {}).get("shared_fold_plan_sha256") != fingerprint:
        raise ValueError("Historical and TabICLv2 fold fingerprints differ.")

    progress: dict[str, object] = {
        "schema": PROGRESS_SCHEMA,
        "status": "incomplete",
        "scope": "development_only_exploratory_tabiclv2_evaluation",
        "dataset_sha256": development.dataset_sha256,
        "split_fingerprint_sha256": development.split_fingerprint_sha256,
        "shared_fold_plan_sha256": fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "completed_repeats": {},
        "privacy": {
            "participant_identifiers_written": False,
            "row_level_predictions_written": False,
        },
    }
    if progress_path.exists():
        existing = _load_json(progress_path)
        required_matches = (
            existing.get("schema") == PROGRESS_SCHEMA
            and existing.get("dataset_sha256") == development.dataset_sha256
            and existing.get("shared_fold_plan_sha256") == fingerprint
            and existing.get("checkpoint_sha256") == checkpoint_sha256
        )
        if not required_matches:
            raise ValueError("Existing progress checkpoint does not match this experiment.")
        progress = existing
        if progress.get("status") == "complete":
            raise ValueError("Progress checkpoint already records a completed experiment.")

    from training.tabiclv2.candidate import tabiclv2_probabilities

    def fit_predict(train_x, train_y, validation_x, seed):
        return tabiclv2_probabilities(
            train_x,
            train_y,
            validation_x,
            checkpoint_path=checkpoint_path,
            seed=seed,
        )

    started = time.perf_counter()

    def update_progress(event: dict[str, object]) -> None:
        if event["event"] == "repeat_completed":
            progress["completed_repeats"][event["cache_key"]] = event["repeat_result"]
        progress["last_event"] = {key: value for key, value in event.items() if key != "repeat_result"}
        progress["status"] = "incomplete"
        _atomic_json_write(progress_path, progress)
        if event["event"] == "fold_completed":
            elapsed = float(event["elapsed_seconds"])
            remaining = float(event["estimated_remaining_seconds"])
            print(
                f"[{event['completed_outer_folds']}/{event['total_outer_folds']}] "
                f"{event['domain']} repeat={event['repeat']} fold={event['fold']} "
                f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m",
                flush=True,
            )

    report = run_tabiclv2_evaluation(
        development.records,
        fit_predict=fit_predict,
        historical_report=historical,
        completed_repeats=progress.get("completed_repeats", {}),
        progress_callback=update_progress,
        n_splits=args.folds,
        n_repeats=args.repeats,
        cv_seed=args.cv_seed,
        fold_search_iterations=args.fold_search_iterations,
        model_seed=args.model_seed,
    )
    runtime_seconds = time.perf_counter() - started
    report.update(
        {
            "artifact_schema_version": REPORT_SCHEMA,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "target_definition_version": TARGET_DEFINITION_VERSION,
            "development_split_schema_version": DEVELOPMENT_SPLIT_SCHEMA_VERSION,
            "dataset_sha256": development.dataset_sha256,
            "split_fingerprint_sha256": development.split_fingerprint_sha256,
            "manifest_sha256": development.manifest_sha256,
            "status": "development_only_exploratory_tabiclv2_completed",
            "deployment_changed": False,
            "historical_seven_model_report": str(historical_path.relative_to(root)),
            "historical_seven_model_report_sha256": sha256_file(historical_path),
            "runtime_seconds": runtime_seconds,
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pandas.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
                "tabicl": importlib.metadata.version("tabicl"),
            },
            "tabiclv2": {
                "package_version": "2.2.0",
                "configuration_version": "tabiclv2-default-eight-view-v1",
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": checkpoint_sha256,
                "repository": "jingang/TabICL",
                "repository_revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
                "software_license": "BSD-3-Clause",
                "checkpoint_license": "BSD-3-Clause",
                "offline_local_inference": True,
                "automatic_downloads": False,
                "n_estimators": 8,
                "device": "cpu",
                "fine_tuning": False,
            },
            "explainability": {
                "available_approach": "optional_tabicl_shap_with_all_nan_masking_background",
                "executed_across_folds": False,
                "reason": "Bounded evaluation excludes expensive all-fold SHAP and does not modify existing XAI.",
                "calibration_limitation": (
                    "Base-model SHAP values do not directly decompose the subsequently calibrated probability."
                ),
            },
        }
    )
    _atomic_json_write(output, report, refuse_overwrite=True)
    report_sha256 = sha256_file(output)
    progress["status"] = "complete"
    progress["final_report"] = str(output.relative_to(root))
    progress["final_report_sha256"] = report_sha256
    progress["runtime_seconds"] = runtime_seconds
    _atomic_json_write(progress_path, progress)
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_participants": report["development_participants"],
                "final_test_evaluated": report["final_test_evaluated"],
                "runtime_seconds": runtime_seconds,
                "report_path": str(output.relative_to(root)),
                "report_sha256": report_sha256,
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
