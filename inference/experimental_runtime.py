"""Opt-in real-runtime factory for the experimental TabICLv2 composite release.

The public factory fails closed unless its caller deliberately supplies an
authorization token.  It is never imported by the historical frozen loader.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
from tabicl import TabICLClassifier

from inference.experimental_composite import (
    ExperimentalCompositeRelease,
    ExperimentalThreeDomainService,
    ProbabilityModel,
    build_experimental_service,
    load_experimental_composite_release,
)
from inference.model_artifact import CalibratedDomainModel
from inference.questionnaire import QuestionnaireCatalog
from inference.tabiclv2_adapter import (
    TabICLv2CognitiveAdapter,
    load_cognitive_release_contract,
    verify_tabiclv2_runtime,
)
from inference.tabiclv2_context import (
    ContextSourceAuthorization,
    InMemoryContextCache,
    VerifiedCognitiveContext,
    load_authorized_cognitive_context,
)
from training.tabiclv2.candidate import (
    TABICLV2_CONFIGURATION_VERSION,
    TABICLV2_PACKAGE_VERSION,
    tabiclv2_frame,
)


RUNTIME_AUTHORIZATION = "approved-experimental-runtime-only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _inside(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Integrity-index path is missing.")
    path = (root / value.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Integrity-index path is unavailable or outside the repository.")
    return path


def verify_cognitive_integrity_index(release: ExperimentalCompositeRelease, root: Path) -> None:
    """Verify every detached Cognitive release reference before any model work."""
    value = json.loads(release.cognitive_integrity_index.read_text(encoding="utf-8"))
    if value.get("schema") != "cognitive-tabiclv2-integrity-index-v1":
        raise ValueError("Cognitive detached integrity index is invalid.")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"aggregate_report", "release_manifest"}:
        raise ValueError("Cognitive detached integrity index is incomplete.")
    for name, entry in artifacts.items():
        if not isinstance(entry, dict):
            raise ValueError("Cognitive detached integrity entry is invalid.")
        artifact = _inside(root, entry.get("path"))
        if _sha256(artifact) != entry.get("sha256"):
            raise ValueError("Cognitive detached integrity checksum mismatch.")
        if name == "release_manifest" and artifact != release.cognitive_manifest.resolve():
            raise ValueError("Cognitive detached manifest reference is inconsistent.")


def verify_cognitive_runtime_contract(release: ExperimentalCompositeRelease, root: Path) -> None:
    contract = load_cognitive_release_contract(release.cognitive_manifest)
    raw = json.loads(release.cognitive_manifest.read_text(encoding="utf-8"))
    configuration = raw.get("configuration")
    if (
        raw.get("tabicl_version") != TABICLV2_PACKAGE_VERSION
        or not isinstance(configuration, dict)
        or configuration.get("n_estimators") != 8
        or configuration.get("device") != "cpu"
        or configuration.get("automatic_downloads") is not False
        or configuration.get("fine_tuning") is not False
        or contract.coefficient != 1.069105693175016
        or contract.intercept != 0.255897163477147
        or contract.threshold != 0.1503614235173848
    ):
        raise ValueError("Cognitive runtime contract differs from the finalized release.")
    verify_cognitive_integrity_index(release, root)


@dataclass(frozen=True)
class ExperimentalRuntimeDependencies:
    """Injectable seams make verification possible without sensitive data."""

    artifact_loader: Callable[[Path], ProbabilityModel]
    checkpoint_verifier: Callable[[Path], None]
    context_loader: Callable[[ContextSourceAuthorization], VerifiedCognitiveContext]
    classifier_factory: Callable[[VerifiedCognitiveContext, Path], object]


def _load_xgboost_domain(path: Path) -> ProbabilityModel:
    value = joblib.load(path)
    if not isinstance(value, CalibratedDomainModel) or value.domain not in {"communication", "social_emotional"}:
        raise ValueError("Experimental XGBoost artifact has an invalid type or domain.")
    return value


def _fit_tabicl_context(context: VerifiedCognitiveContext, checkpoint: Path) -> object:
    """The only real fitting path; called only after explicit future authorization."""
    model = TabICLClassifier(
        n_estimators=8, model_path=str(checkpoint), allow_auto_download=False,
        device="cpu", use_amp=False, use_fa3=False, offload_mode=False,
        random_state=20260822, n_jobs=4, verbose=False,
    )
    model.fit(tabiclv2_frame(context.feature_rows), context.cognitive_labels)
    return model


DEFAULT_RUNTIME_DEPENDENCIES = ExperimentalRuntimeDependencies(
    artifact_loader=_load_xgboost_domain,
    checkpoint_verifier=verify_tabiclv2_runtime,
    context_loader=load_authorized_cognitive_context,
    classifier_factory=_fit_tabicl_context,
)


def build_experimental_runtime_service(
    catalog: QuestionnaireCatalog,
    *,
    repository_root: Path,
    composite_manifest: Path,
    checkpoint: Path,
    context_authorization: ContextSourceAuthorization,
    execution_authorization: str | None = None,
    dependencies: ExperimentalRuntimeDependencies = DEFAULT_RUNTIME_DEPENDENCIES,
) -> ExperimentalThreeDomainService:
    """Build the experimental service only when a future operator authorizes it.

    Default invocation performs no I/O, model loading, context access, or fitting.
    There is intentionally no historical Logistic Regression fallback.
    """
    if execution_authorization != RUNTIME_AUTHORIZATION:
        raise PermissionError("Experimental TabICLv2 runtime execution is not authorized.")
    root = repository_root.resolve()
    release = load_experimental_composite_release(composite_manifest.resolve(), root)
    verify_cognitive_runtime_contract(release, root)
    dependencies.checkpoint_verifier(checkpoint)
    if importlib.metadata.version("tabicl") != TABICLV2_PACKAGE_VERSION:
        raise ValueError("Installed TabICLv2 version differs from the finalized release.")
    cognitive = TabICLv2CognitiveAdapter(
        InMemoryContextCache(lambda: dependencies.classifier_factory(dependencies.context_loader(context_authorization), checkpoint)),
        load_cognitive_release_contract(release.cognitive_manifest),
    )
    return build_experimental_service(catalog, release, cognitive, artifact_loader=dependencies.artifact_loader)
