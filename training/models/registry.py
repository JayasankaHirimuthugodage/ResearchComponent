"""Explicit candidate registry used by the shared comparison runner."""

from training.models.extra_trees import REGULARIZED_EXTRA_TREES
from training.models.logistic import REGULARIZED_LOGISTIC


DEFAULT_CANDIDATES = (REGULARIZED_LOGISTIC, REGULARIZED_EXTRA_TREES)
