"""CLI for the development-only regularized-logistic baseline experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from training.contracts import (
    DEVELOPMENT_SPLIT_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    TARGET_DEFINITION_VERSION,
    default_dataset_path,
    repository_root,
)
import numpy
import sklearn

from training.cross_validation import (
    DEFAULT_CV_SEED,
    DEFAULT_FOLD_SEARCH_ITERATIONS,
    DEFAULT_N_REPEATS,
    DEFAULT_N_SPLITS,
)
from training.development_data import (
    default_manifest_path,
    default_split_summary_path,
    load_development_dataset,
)
from training.evaluation import run_development_cross_validation
from training.modeling import DEFAULT_MODEL_SEED


def _atomic_json_write(path: Path, value: object, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run shared-fold, development-only evaluation for the three-domain "
            "regularized-logistic baseline. The locked final-test partition is unavailable here."
        )
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--split-summary", type=Path, default=default_split_summary_path())
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root()
        / "reports"
        / "generated"
        / "development-logistic-report.json",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--cv-seed", type=int, default=DEFAULT_CV_SEED)
    parser.add_argument(
        "--fold-search-iterations",
        type=int,
        default=DEFAULT_FOLD_SEARCH_ITERATIONS,
    )
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_path = args.output.resolve()
    root = repository_root().resolve()
    if not output_path.is_relative_to(root):
        raise ValueError("Generated report must remain inside the repository workspace.")

    development = load_development_dataset(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        summary_path=args.split_summary,
    )
    report = run_development_cross_validation(
        development.records,
        n_splits=args.folds,
        n_repeats=args.repeats,
        cv_seed=args.cv_seed,
        fold_search_iterations=args.fold_search_iterations,
        model_seed=args.model_seed,
    )
    report.update(
        {
            "artifact_schema_version": "1.0",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "target_definition_version": TARGET_DEFINITION_VERSION,
            "development_split_schema_version": DEVELOPMENT_SPLIT_SCHEMA_VERSION,
            "dataset_sha256": development.dataset_sha256,
            "split_fingerprint_sha256": development.split_fingerprint_sha256,
            "manifest_sha256": development.manifest_sha256,
            "status": "development_baseline_not_final_model",
            "final_test_evaluated": False,
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
    )
    _atomic_json_write(output_path, report, force=args.force)

    console_summary = {
        "status": report["status"],
        "scope": report["scope"],
        "development_participants": report["development_participants"],
        "candidate_names": list(report["candidates"]),
        "domain_names": list(report["target_positive_counts"]),
        "shared_fold_plan_sha256": report["cross_validation"][
            "shared_fold_plan_sha256"
        ],
        "final_test_evaluated": report["final_test_evaluated"],
        "report_path": str(output_path.relative_to(root)),
    }
    print(json.dumps(console_summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # Never print rows or participant identifiers.
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
