"""Contracts shared by isolated additional-model candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ProbabilityModel(Protocol):
    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class AdditionalCandidateConfiguration:
    candidate_name: str
    configuration_name: str
    configuration_version: str
    public_parameters: dict[str, object]


def numeric_questionnaire_matrix(features: np.ndarray) -> np.ndarray:
    """Preserve checkpoint as text and convert ordinal response strings to integers."""

    result = features.copy()
    result[:, 0] = np.asarray(features[:, 0], dtype=str)
    result[:, 1:] = np.asarray(features[:, 1:], dtype=np.int8)
    return result
