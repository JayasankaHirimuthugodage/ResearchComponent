"""Opt-in, localhost-only command-line smoke test for the experimental release.

The command is intentionally inert unless a separately authorized operator
supplies ``--execute-real-inference`` and the required governance/configuration
files.  It creates no listener and persists no predictions or context.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.contracts import AssessmentStatus, QuestionnaireSubmission
from inference.experimental_runtime import RUNTIME_AUTHORIZATION, build_experimental_runtime_service
from inference.questionnaire import QuestionnaireCatalog, load_questionnaire_catalog
from inference.tabiclv2_context import ContextSourceAuthorization


COMPOSITE_MANIFEST = "models/experimental-three-domain-tabiclv2-v1/release-manifest.json"
RUNTIME_CONFIGURATION_SCHEMA = "experimental-tabiclv2-runtime-configuration-v1"
GOVERNANCE_ATTESTATION_SCHEMA = "experimental-tabiclv2-runtime-governance-v1"


@dataclass(frozen=True)
class OperationalSummary:
    initialized: bool
    complete_contract_ok: bool
    incomplete_contract_ok: bool
    cold_initialization_seconds: float
    warm_prediction_seconds: float
    total_seconds: float
    python_peak_memory_bytes: int

    def safe_dict(self) -> dict[str, object]:
        """Only operational facts; never inputs, predictions, or probabilities."""
        return {
            "initialized": self.initialized,
            "complete_contract_ok": self.complete_contract_ok,
            "incomplete_contract_ok": self.incomplete_contract_ok,
            "cold_initialization_seconds": self.cold_initialization_seconds,
            "warm_prediction_seconds": self.warm_prediction_seconds,
            "total_seconds": self.total_seconds,
            "python_peak_memory_bytes": self.python_peak_memory_bytes,
        }


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Runtime configuration must be a JSON object.")
    return value


def require_non_elevated_process(is_elevated: Callable[[], bool] | None = None) -> None:
    if is_elevated is None:
        if os.name != "nt":
            raise RuntimeError("Experimental smoke test is restricted to the approved Windows runtime.")
        is_elevated = lambda: bool(ctypes.windll.shell32.IsUserAnAdmin())
    if is_elevated():
        raise PermissionError("Experimental smoke test refuses an elevated process.")


def load_governance_attestation(path: Path) -> None:
    value = _load_json(path)
    required = {
        "schema": GOVERNANCE_ATTESTATION_SCHEMA,
        "custodian_runtime_permission": True,
        "interactive_account_authorized": True,
        "retention_and_backup_addressed": True,
        "localhost_only": True,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise PermissionError("Research-data governance attestation is incomplete.")


def load_context_authorization(path: Path) -> ContextSourceAuthorization:
    value = _load_json(path)
    if value.get("schema") != RUNTIME_CONFIGURATION_SCHEMA:
        raise ValueError("Runtime context configuration schema is invalid.")
    required = ("source_path", "provenance_path", "source_sha256", "membership_sha256", "dataset_sha256", "split_fingerprint_sha256", "forbidden_combined_source")
    if any(not isinstance(value.get(key), str) or not value[key] for key in required):
        raise ValueError("Runtime context configuration is incomplete.")
    authorization = ContextSourceAuthorization(
        Path(value["source_path"]), Path(value["provenance_path"]), str(value["source_sha256"]),
        str(value["membership_sha256"]), str(value["dataset_sha256"]),
        str(value["split_fingerprint_sha256"]), Path(value["forbidden_combined_source"]),
    )
    if authorization.source_path.resolve() == authorization.forbidden_combined_source.resolve():
        raise ValueError("Combined source is forbidden for the smoke test.")
    return authorization


def _submission(catalog: QuestionnaireCatalog, *, incomplete: bool) -> QuestionnaireSubmission:
    checkpoint = "CP24M"
    answers = {question.question_id: "YES_USUALLY" for question in catalog.questions_for_checkpoint(checkpoint)}
    if incomplete:
        first = next(iter(answers))
        answers[first] = "NOT_SURE"
    return QuestionnaireSubmission(catalog.questionnaire_schema_version, checkpoint, answers)


def run_real_smoke_test(
    *, repository_root: Path, checkpoint: Path, context_authorization: ContextSourceAuthorization,
    governance_attestation: Path, catalog: QuestionnaireCatalog | None = None,
    service_factory=build_experimental_runtime_service, is_elevated: Callable[[], bool] | None = None,
) -> OperationalSummary:
    """Real path, intentionally callable only from a separately approved command."""
    require_non_elevated_process(is_elevated)
    load_governance_attestation(governance_attestation)
    started = time.perf_counter(); tracemalloc.start()
    catalog = catalog or load_questionnaire_catalog()
    service = service_factory(catalog, repository_root=repository_root, composite_manifest=repository_root / COMPOSITE_MANIFEST, checkpoint=checkpoint, context_authorization=context_authorization, execution_authorization=RUNTIME_AUTHORIZATION)
    initialized = time.perf_counter()
    complete = service.predict(_submission(catalog, incomplete=False))
    warmed = time.perf_counter()
    incomplete = service.predict(_submission(catalog, incomplete=True))
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    if complete.status is not AssessmentStatus.COMPLETE or len(complete.domain_predictions) != 3 or complete.overall_concern is None:
        raise RuntimeError("Complete synthetic smoke-test contract failed.")
    if incomplete.status is not AssessmentStatus.INCOMPLETE or incomplete.domain_predictions or incomplete.overall_concern is not None:
        raise RuntimeError("Incomplete synthetic smoke-test contract failed.")
    return OperationalSummary(True, True, True, initialized-started, warmed-initialized, time.perf_counter()-started, peak)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicitly gated experimental TabICLv2 smoke test; no network listener.")
    parser.add_argument("--execute-real-inference", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--context-authorization", type=Path)
    parser.add_argument("--governance-attestation", type=Path)
    arguments = parser.parse_args(argv)
    if not arguments.execute_real_inference:
        print(json.dumps({"status": "blocked", "reason": "--execute-real-inference is required; no runtime I/O was performed."}))
        return 2
    if not all((arguments.checkpoint, arguments.context_authorization, arguments.governance_attestation)):
        parser.error("real execution requires checkpoint, context authorization, and governance attestation paths")
    authorization = load_context_authorization(arguments.context_authorization)
    summary = run_real_smoke_test(repository_root=PROJECT_ROOT, checkpoint=arguments.checkpoint, context_authorization=authorization, governance_attestation=arguments.governance_attestation)
    print(json.dumps(summary.safe_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
