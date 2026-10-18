"""Explicit, cache-backed experimental Cognitive runtime; never a default loader."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tabicl import TabICLClassifier

from inference.experimental_composite import (
    ExperimentalThreeDomainService, ProbabilityModel, build_experimental_service,
    load_experimental_composite_release,
)
from inference.experimental_runtime import _load_xgboost_domain, verify_cognitive_runtime_contract
from inference.tabiclv2_adapter import TabICLv2CognitiveAdapter, load_cognitive_release_contract, verify_tabiclv2_runtime
from inference.tabiclv2_context import InMemoryContextCache
from training.contracts import FEATURE_COLUMNS
from training.tabiclv2.candidate import TABICLV2_PACKAGE_VERSION

CACHE_RUNTIME_AUTHORIZATION = "approved-experimental-kv-cache-runtime-only"
CACHE_RELEASE_SCHEMA = "experimental-cognitive-tabiclv2-kv-cache-release-v1"
APPROVED_CACHE_PATH = Path(r"F:\4thYear\2026July\Research_Protected\TabICLv2_Cognitive_KV_Cache_Engineering_Verification_20261017\cognitive-tabiclv2-kv-cache.pkl")
APPROVED_CACHE_SHA256 = "96E6D812AB977B0E065329C33563A91F02856EBF3BC2D3F0B2DE2020CF9682F0"
APPROVED_CACHE_BYTES = 259554811
COGNITIVE_EXPLANATION_STATUS = "unavailable_pending_model_specific_validation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_cache_artifact(path: Path, expected_sha256: str, expected_bytes: int) -> None:
    if path.resolve() != APPROVED_CACHE_PATH.resolve() or not path.is_file():
        raise ValueError("KV-cache path is not the approved protected artifact.")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise ValueError("KV-cache artifact integrity verification failed.")


@dataclass(frozen=True)
class KVCacheReleaseContract:
    cache_path: Path
    cache_sha256: str
    cache_bytes: int
    original_cognitive_manifest_sha256: str


def load_kv_cache_release_contract(path: Path) -> KVCacheReleaseContract:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CACHE_RELEASE_SCHEMA or value.get("status") != "development_only_experimental_not_deployed":
        raise ValueError("KV-cache release manifest is invalid.")
    if tuple(value.get("feature_contract", ())) != tuple(FEATURE_COLUMNS):
        raise ValueError("KV-cache feature contract is invalid.")
    config = value.get("configuration")
    calibration, threshold = value.get("calibration"), value.get("threshold")
    if not isinstance(config, dict) or not isinstance(calibration, dict) or not isinstance(threshold, dict):
        raise ValueError("KV-cache release contract is incomplete.")
    if config != {"n_estimators": 8, "device": "cpu", "use_amp": False, "use_fa3": False, "offload_mode": False, "random_state": 20260822, "n_jobs": 4, "allow_auto_download": False}:
        raise ValueError("KV-cache runtime configuration is invalid.")
    if calibration.get("coefficient") != 1.069105693175016 or calibration.get("intercept") != 0.255897163477147 or threshold.get("value") != 0.1503614235173848 or threshold.get("comparison") != ">=":
        raise ValueError("KV-cache frozen calibration/threshold is invalid.")
    cache = value.get("cache_artifact")
    if not isinstance(cache, dict):
        raise ValueError("KV-cache artifact reference is missing.")
    contract = KVCacheReleaseContract(Path(str(cache.get("path", ""))), str(cache.get("sha256", "")), int(cache.get("bytes", -1)), str(value.get("original_cognitive_manifest_sha256", "")))
    if contract.cache_path.resolve() != APPROVED_CACHE_PATH.resolve() or contract.cache_sha256 != APPROVED_CACHE_SHA256 or contract.cache_bytes != APPROVED_CACHE_BYTES:
        raise ValueError("KV-cache release does not reference the approved artifact.")
    return contract


def validate_loaded_cache(model: object, checkpoint: Path) -> None:
    generator = getattr(model, "ensemble_generator_", None)
    preprocessors = getattr(generator, "preprocessors_", {})
    arrays_absent = getattr(generator, "X_", None) is None and getattr(generator, "y_", None) is None and all(getattr(item, "X_transformed_", None) is None for item in preprocessors.values())
    if not arrays_absent or getattr(model, "model_kv_cache_", None) is None:
        raise ValueError("KV-cache estimator retains training arrays or lacks its cache.")
    model_path = Path(str(getattr(model, "model_path_", ""))).resolve()
    if model_path != checkpoint.resolve():
        raise ValueError("KV-cache checkpoint path differs from the approved checkpoint.")


def load_verified_kv_cache(contract: KVCacheReleaseContract, checkpoint: Path) -> object:
    verify_cache_artifact(contract.cache_path, contract.cache_sha256, contract.cache_bytes)
    verify_tabiclv2_runtime(checkpoint)
    if importlib.metadata.version("tabicl") != TABICLV2_PACKAGE_VERSION:
        raise ValueError("TabICLv2 package version differs from the cache release.")
    model = TabICLClassifier.load(contract.cache_path, device="cpu")
    validate_loaded_cache(model, checkpoint)
    return model


@dataclass(frozen=True)
class KVCacheRuntimeDependencies:
    artifact_loader: Callable[[Path], ProbabilityModel]
    checkpoint_verifier: Callable[[Path], None]
    cache_loader: Callable[[KVCacheReleaseContract, Path], object]


DEFAULT_KV_CACHE_RUNTIME_DEPENDENCIES = KVCacheRuntimeDependencies(_load_xgboost_domain, verify_tabiclv2_runtime, load_verified_kv_cache)


def build_kv_cache_experimental_runtime_service(catalog, *, repository_root: Path, composite_manifest: Path, checkpoint: Path, execution_authorization: str | None = None, dependencies: KVCacheRuntimeDependencies = DEFAULT_KV_CACHE_RUNTIME_DEPENDENCIES) -> ExperimentalThreeDomainService:
    """Build only after explicit authorization; no context CSV is accepted or read."""
    if execution_authorization != CACHE_RUNTIME_AUTHORIZATION:
        raise PermissionError("Experimental KV-cache runtime execution is not authorized.")
    root = repository_root.resolve()
    release = load_experimental_composite_release(composite_manifest.resolve(), root)
    verify_cognitive_runtime_contract(release, root)
    raw = json.loads(composite_manifest.read_text(encoding="utf-8"))
    cognitive = raw.get("domains", {}).get("cognitive", {})
    if not isinstance(cognitive, dict):
        raise ValueError("KV-cache composite Cognitive reference is invalid.")
    cache_manifest = (root / str(cognitive.get("cache_release_manifest", "")).replace("\\", "/")).resolve()
    if not cache_manifest.is_relative_to(root) or not cache_manifest.is_file() or sha256_file(cache_manifest) != cognitive.get("cache_release_manifest_sha256"):
        raise ValueError("KV-cache release-manifest integrity check failed.")
    contract = load_kv_cache_release_contract(cache_manifest)
    if contract.original_cognitive_manifest_sha256 != sha256_file(release.cognitive_manifest):
        raise ValueError("KV-cache release is not bound to the finalized Cognitive manifest.")
    dependencies.checkpoint_verifier(checkpoint)
    adapter = TabICLv2CognitiveAdapter(InMemoryContextCache(lambda: dependencies.cache_loader(contract, checkpoint)), load_cognitive_release_contract(release.cognitive_manifest))
    return build_experimental_service(catalog, release, adapter, artifact_loader=dependencies.artifact_loader)
