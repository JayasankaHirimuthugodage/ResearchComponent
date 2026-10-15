"""Small, deterministic probability-voting helpers for hybrid research."""

from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_SOFT_VOTING_WEIGHTS = (0.25, 0.50, 0.75)


def validate_weight_grid(weights: Sequence[float]) -> tuple[float, ...]:
    """Return a unique ordered grid containing the equal-weight benchmark."""

    values = tuple(float(value) for value in weights)
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("Soft-voting weights must be strictly between zero and one.")
    if len(values) != len(set(values)):
        raise ValueError("Soft-voting weights must be unique.")
    if 0.50 not in values:
        raise ValueError("The weight grid must contain the equal-weight benchmark.")
    return tuple(sorted(values))


def soft_vote(
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
    *,
    first_weight: float,
) -> np.ndarray:
    """Combine aligned calibrated probabilities without learning from outcomes."""

    if first_probabilities.shape != second_probabilities.shape:
        raise ValueError("Soft-voting probability arrays must be aligned.")
    if not 0.0 < first_weight < 1.0:
        raise ValueError("The first-model weight must be strictly between zero and one.")
    if np.any(~np.isfinite(first_probabilities)) or np.any(
        ~np.isfinite(second_probabilities)
    ):
        raise ValueError("Soft-voting probabilities must be finite.")
    if np.any((first_probabilities < 0.0) | (first_probabilities > 1.0)) or np.any(
        (second_probabilities < 0.0) | (second_probabilities > 1.0)
    ):
        raise ValueError("Soft-voting probabilities must lie in [0,1].")
    return (
        first_weight * first_probabilities
        + (1.0 - first_weight) * second_probabilities
    )
