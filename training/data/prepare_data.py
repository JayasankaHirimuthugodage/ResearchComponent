"""CLI for validating the frozen dataset and creating an ignored participant split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

from training.contracts import PARTICIPANT_COLUMN, default_dataset_path, repository_root
from training.data.dataset_validation import read_csv_records, require_valid, validate_dataset
from training.data.splitting import (
    DEFAULT_SEARCH_ITERATIONS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRACTION,
    create_participant_split,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _manifest_bytes(manifest: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=(PARTICIPANT_COLUMN, "partition"))
    writer.writeheader()
    writer.writerows(manifest)
    return buffer.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
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
            "Validate the audited modelling CSV and create a deterministic, "
            "participant-level development/final-test split."
        )
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=repository_root() / "artifacts",
    )
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--search-iterations", type=int, default=DEFAULT_SEARCH_ITERATIONS
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated artifacts after deterministic verification.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    root = repository_root().resolve()
    if not artifact_root.is_relative_to(root):
        raise ValueError("Artifact root must remain inside the repository workspace.")

    validation_path = artifact_root / "validation" / "dataset_validation_report.json"
    manifest_path = artifact_root / "splits" / "split_manifest.csv"
    summary_path = artifact_root / "splits" / "split_summary.json"
    for output_path in (validation_path, manifest_path, summary_path):
        if output_path.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite existing artifact: {output_path}. Use --force only intentionally."
            )

    validation_report = validate_dataset(args.dataset)
    validation_report["artifact_schema_version"] = "1.0"
    require_valid(validation_report)

    _, records, _ = read_csv_records(args.dataset.resolve())
    manifest, split_summary = create_participant_split(
        records,
        dataset_sha256=str(validation_report["dataset"]["sha256"]),
        test_fraction=args.test_fraction,
        seed=args.seed,
        search_iterations=args.search_iterations,
    )
    split_summary["artifact_schema_version"] = "1.0"
    manifest_content = _manifest_bytes(manifest)
    split_summary["manifest_sha256"] = hashlib.sha256(manifest_content).hexdigest().upper()

    _atomic_write(validation_path, _json_bytes(validation_report), force=args.force)
    _atomic_write(manifest_path, manifest_content, force=args.force)
    _atomic_write(summary_path, _json_bytes(split_summary), force=args.force)

    console_summary = {
        "validation_valid": validation_report["valid"],
        "validation_warnings": validation_report["summary"]["warning_count"],
        "development_participants": split_summary["development"]["participants"],
        "final_test_participants": split_summary["final_test"]["participants"],
        "participant_overlap": split_summary["constraints"]["participant_overlap"],
        "all_target_classes_in_both_partitions": split_summary["constraints"][
            "all_target_classes_in_both_partitions"
        ],
        "all_checkpoints_in_both_partitions": split_summary["constraints"][
            "all_checkpoints_in_both_partitions"
        ],
        "all_label_combinations_in_both_partitions": split_summary["constraints"][
            "all_label_combinations_in_both_partitions"
        ],
        "split_fingerprint_sha256": split_summary["split_fingerprint_sha256"],
    }
    print(json.dumps(console_summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # Aggregate-only CLI failure; never print participant rows.
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
