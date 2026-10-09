"""Framework-independent contracts for activity validation and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


CHECKPOINT_IDS = (
    "CP2M",
    "CP4M",
    "CP6M",
    "CP9M",
    "CP12M",
    "CP15M",
    "CP18M",
    "CP24M",
    "CP30M",
    "CP36M",
    "CP48M",
    "CP60M",
)
DOMAIN_IDS = ("communication", "cognitive", "social_emotional")
SAFETY_DIMENSIONS = ("hearing", "vision", "fine_motor", "gross_motor")


class RetrievalMode(str, Enum):
    INTERNAL_RESEARCH_DEMO = "internal_research_demo"
    CAREGIVER_DEPLOYMENT = "caregiver_deployment"


class AbilityState(str, Enum):
    AVAILABLE = "available"
    NO_CONCERN_REPORTED = "no_concern_reported"
    CONCERN_REPORTED = "concern_reported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SafetyContext:
    hearing: AbilityState
    vision: AbilityState
    fine_motor: AbilityState
    gross_motor: AbilityState

    def as_mapping(self) -> Mapping[str, AbilityState]:
        return MappingProxyType(
            {
                "hearing": self.hearing,
                "vision": self.vision,
                "fine_motor": self.fine_motor,
                "gross_motor": self.gross_motor,
            }
        )


@dataclass(frozen=True)
class ActivityRecord:
    activity_id: str
    record_version: str
    library_version: str
    title: str
    activity_summary: str
    primary_skill: str | None
    domains: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    source_org: str
    source_url: str
    source_page: int | None
    source_tip_id: str | None
    status: str
    safety_demands: Mapping[str, str]
    research_demo_eligible: bool
    caregiver_deployment_eligible: bool
    raw_record: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "safety_demands", MappingProxyType(dict(self.safety_demands)))
        object.__setattr__(self, "raw_record", MappingProxyType(dict(self.raw_record)))


@dataclass(frozen=True)
class EligibilityRequest:
    checkpoint_id: str
    domain: str
    safety_context: SafetyContext
    mode: RetrievalMode = RetrievalMode.INTERNAL_RESEARCH_DEMO


@dataclass(frozen=True)
class WithheldActivity:
    activity_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EligibilityResult:
    eligible: tuple[ActivityRecord, ...]
    withheld: tuple[WithheldActivity, ...]
    cautions: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cautions", MappingProxyType(dict(self.cautions)))


@dataclass(frozen=True)
class RankedActivity:
    activity: ActivityRecord
    similarity_score: float


class ActivityLibraryValidationError(ValueError):
    """Raised when the activity-library snapshot violates its contract."""


class EmbeddingArtifactError(ValueError):
    """Raised when local embedding artifacts are missing, stale, or inconsistent."""
