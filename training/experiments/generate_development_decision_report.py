"""Generate JSON and Markdown development decision reports from aggregate evidence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from training.contracts import repository_root
from training.experiments.development_decision_report import (
    build_development_decision_report,
    load_json,
    render_markdown,
    sha256_file,
)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> int:
    root = repository_root()
    generated = root / "reports" / "generated"
    comparison_path = generated / "development-model-comparison-report.json"
    calibration_path = generated / "development-calibration-report.json"
    json_path = generated / "development-decision-report.json"
    markdown_path = generated / "development-decision-report.md"

    report = build_development_decision_report(
        load_json(comparison_path),
        load_json(calibration_path),
        comparison_sha256=sha256_file(comparison_path),
        calibration_sha256=sha256_file(calibration_path),
    )
    _atomic_write(
        json_path,
        (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    _atomic_write(markdown_path, render_markdown(report).encode("utf-8"))
    print(
        json.dumps(
            {
                "status": report["status"],
                "final_test_evaluated": report["final_test_evaluated"],
                "json_report": str(json_path.relative_to(root)),
                "markdown_report": str(markdown_path.relative_to(root)),
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
