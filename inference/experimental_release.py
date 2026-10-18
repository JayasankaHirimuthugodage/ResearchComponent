"""Explicit experimental-release selection; never changes the historical default."""
from __future__ import annotations

from dataclasses import dataclass


HISTORICAL_RELEASE = "slc-development-models-v1"
EXPERIMENTAL_COGNITIVE_RELEASE = "slc-development-models-tabiclv2-cognitive-v1"
EXPERIMENTAL_COMPOSITE_RELEASE = "slc-development-models-tabiclv2-cognitive-v1"
EXPERIMENTAL_KV_CACHE_COMPOSITE_RELEASE = "slc-development-models-tabiclv2-cognitive-kv-cache-v1"


@dataclass(frozen=True)
class ExperimentalReleaseSelection:
    release_id: str

    def require_experimental_cognitive(self) -> None:
        if self.release_id not in {EXPERIMENTAL_COGNITIVE_RELEASE, EXPERIMENTAL_COMPOSITE_RELEASE, EXPERIMENTAL_KV_CACHE_COMPOSITE_RELEASE}:
            raise ValueError("Experimental Cognitive release must be selected explicitly.")
