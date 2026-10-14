"""Separate caregiver-facing activity content from reviewer-only provenance."""

from __future__ import annotations

from activity_recommendation.contracts import ActivityRecord


def caregiver_activity_content(activity: ActivityRecord) -> tuple[str, str]:
    """Return only content intended for the main caregiver-facing activity card."""

    return activity.title, activity.activity_summary


def reviewer_activity_provenance(activity: ActivityRecord) -> dict[str, object]:
    """Return source and identity metadata for the restricted reviewer trace."""

    return {
        "activity_id": activity.activity_id,
        "source_org": activity.source_org,
        "source_url": activity.source_url,
        "source_page": activity.source_page,
        "source_tip_id": activity.source_tip_id,
    }
