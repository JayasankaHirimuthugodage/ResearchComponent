"""Experimental Cognitive TabICLv2 probability adapter; no default activation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from inference.tabiclv2_context import InMemoryContextCache
from training.contracts import FEATURE_COLUMNS
from training.tabiclv2.candidate import (
    TABICLV2_CHECKPOINT_NAME,
    TABICLV2_CHECKPOINT_SHA256,
    TABICLV2_PACKAGE_VERSION,
    tabiclv2_frame,
)


class RawProbabilityModel(Protocol):
    def predict_proba(self, values: object) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True)
class CognitiveReleaseContract:
    checkpoint_sha256: str
    coefficient: float
    intercept: float
    threshold: float


def load_cognitive_release_contract(manifest_path: Path) -> CognitiveReleaseContract:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("status") != "development_only_experimental_not_deployed" or value.get("domain") != "cognitive":
        raise ValueError("Experimental Cognitive release manifest is invalid.")
    if tuple(value.get("feature_contract", ())) != tuple(FEATURE_COLUMNS):
        raise ValueError("Experimental Cognitive feature contract is invalid.")
    calibration, threshold = value.get("calibration"), value.get("threshold")
    if not isinstance(calibration, dict) or not isinstance(threshold, dict):
        raise ValueError("Experimental Cognitive calibration/threshold is missing.")
    contract = CognitiveReleaseContract(str(value.get("checkpoint_sha256")), float(calibration["coefficient"]), float(calibration["intercept"]), float(threshold["value"]))
    if contract.checkpoint_sha256 != TABICLV2_CHECKPOINT_SHA256 or not all(math.isfinite(number) for number in (contract.coefficient, contract.intercept, contract.threshold)) or not 0.0 <= contract.threshold <= 1.0:
        raise ValueError("Experimental Cognitive release values are invalid.")
    return contract


def verify_tabiclv2_runtime(checkpoint: Path) -> None:
    if checkpoint.name != TABICLV2_CHECKPOINT_NAME or not checkpoint.is_file() or _sha256(checkpoint) != TABICLV2_CHECKPOINT_SHA256:
        raise ValueError("Approved TabICLv2 checkpoint verification failed.")
    if importlib.metadata.version("tabicl") != TABICLV2_PACKAGE_VERSION:
        raise ValueError("tabicl package version mismatch.")


@dataclass(frozen=True)
class TabICLv2CognitiveAdapter:
    """Applies only frozen release calibration; context/model construction is external."""

    context_cache: InMemoryContextCache
    release: CognitiveReleaseContract

    def positive_probability(self, feature_values: tuple[str, ...]) -> float:
        if len(feature_values) != len(FEATURE_COLUMNS):
            raise ValueError("TabICLv2 requires exactly 16 ordered baseline features.")
        raw_values = np.asarray(self.context_cache.get().predict_proba(tabiclv2_frame(np.asarray([feature_values], dtype=object)))[:, 1], dtype=float)
        if raw_values.shape != (1,) or not np.isfinite(raw_values[0]) or not 0.0 <= raw_values[0] <= 1.0:
            raise RuntimeError("TabICLv2 returned an invalid raw probability.")
        raw = float(np.clip(raw_values[0], 1e-12, 1.0 - 1e-12))
        logit = math.log(raw) - math.log1p(-raw)
        probability = _sigmoid(self.release.coefficient * logit + self.release.intercept)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("TabICLv2 returned an invalid calibrated probability.")
        return probability

    def decision(self, feature_values: tuple[str, ...]) -> bool:
        return self.positive_probability(feature_values) >= self.release.threshold
