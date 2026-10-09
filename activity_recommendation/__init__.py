"""Safety-filtered activity retrieval for the internal research demonstration."""

from .catalogue import ActivityCatalogue, load_activity_catalogue
from .contracts import (
    AbilityState,
    ActivityRecord,
    EligibilityRequest,
    EligibilityResult,
    RetrievalMode,
    SafetyContext,
)
from .eligibility import filter_eligible_activities

__all__ = [
    "AbilityState",
    "ActivityCatalogue",
    "ActivityRecord",
    "EligibilityRequest",
    "EligibilityResult",
    "RetrievalMode",
    "SafetyContext",
    "filter_eligible_activities",
    "load_activity_catalogue",
]
