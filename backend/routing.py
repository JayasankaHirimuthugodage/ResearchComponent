"""Deterministic child-profile age and checkpoint routing; no model calls."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from inference.questionnaire import QuestionnaireCatalog


class ProfileRoutingError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointRoute:
    chronological_age_days: int
    corrected_age_days: int | None
    checkpoint_id: str


def route_checkpoint(*, date_of_birth: date, as_of_date: date, premature_status: str, gestational_age_weeks: int | None, catalog: QuestionnaireCatalog) -> CheckpointRoute:
    if date_of_birth > as_of_date:
        raise ProfileRoutingError("Date of birth cannot be in the future.")
    chronological = (as_of_date - date_of_birth).days
    maximum = max(item.maximum_age_days for item in catalog.checkpoints)
    if chronological > maximum:
        raise ProfileRoutingError("This application supports children from birth through 60 months.")
    if premature_status not in {"yes", "no", "not_sure"}:
        raise ProfileRoutingError("Prematurity status is invalid.")
    if premature_status == "yes" and gestational_age_weeks is None:
        raise ProfileRoutingError("Gestational age is required when premature birth is reported.")
    if gestational_age_weeks is not None and not 20 <= gestational_age_weeks <= 36:
        raise ProfileRoutingError("Gestational age must be a whole number from 20 to 36 weeks.")
    if premature_status != "yes" and gestational_age_weeks is not None:
        raise ProfileRoutingError("Gestational age is collected only when premature birth is reported.")
    corrected: int | None = None
    routing_age = chronological
    if premature_status == "yes" and gestational_age_weeks is not None:
        prematurity_weeks = 40 - gestational_age_weeks
        if prematurity_weeks >= 3 and chronological < 730:
            corrected = max(0, chronological - prematurity_weeks * 7)
            routing_age = corrected
    checkpoint = next((item for item in catalog.checkpoints if item.minimum_age_days <= routing_age <= item.maximum_age_days), None)
    if checkpoint is None:
        raise ProfileRoutingError("No supported developmental checkpoint is available for this age.")
    if corrected is not None and not checkpoint.corrected_age_applies:
        checkpoint = next((item for item in catalog.checkpoints if item.minimum_age_days <= chronological <= item.maximum_age_days), None)
        corrected = None
    if checkpoint is None:
        raise ProfileRoutingError("No supported developmental checkpoint is available for this age.")
    return CheckpointRoute(chronological, corrected, checkpoint.checkpoint_id)
