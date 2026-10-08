"""Select AP-leading candidates and run nested development-only calibration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from training.contracts import TARGET_COLUMNS, repository_root
from training.data.development_data import load_development_dataset
from training.experiments.calibration import run_nested_calibration_evaluation
from training.models.registry import DEFAULT_CANDIDATES


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_selections(report_path: Path) -> dict[str, object]:
    with report_path.open("r", encoding="utf-8") as source:
        report = json.load(source)
    if report.get("scope") != "development_only" or report.get("final_test_evaluated"):
        raise ValueError("Candidate selection requires a development-only report.")
    registry = {candidate.name: candidate for candidate in DEFAULT_CANDIDATES}
    selections: dict[str, object] = {}
    for target in TARGET_COLUMNS:
        candidate_name = max(
            report["candidates"],
            key=lambda name: report["candidates"][name]["domains"][target][
                "repeat_metric_summary"
            ]["average_precision_pr_auc"]["mean"],
        )
        selections[target] = registry[candidate_name]
    return selections


def main() -> int:
    root = repository_root()
    comparison_path = root / "reports" / "generated" / "development-model-comparison-report.json"
    output_path = root / "reports" / "generated" / "development-calibration-report.json"
    development = load_development_dataset()
    selections = _load_selections(comparison_path)
    report = run_nested_calibration_evaluation(development.records, selections)
    report.update(
        {
            "dataset_sha256": development.dataset_sha256,
            "split_fingerprint_sha256": development.split_fingerprint_sha256,
            "status": "development_calibration_not_final_model",
        }
    )
    _atomic_json_write(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_participants": report["development_participants"],
                "selections": {
                    target: candidate.name for target, candidate in selections.items()
                },
                "final_test_evaluated": report["final_test_evaluated"],
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
