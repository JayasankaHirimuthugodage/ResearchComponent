"""Explicit candidate registry used by the shared comparison runner."""

from training.models.extra_trees import REGULARIZED_EXTRA_TREES
from training.models.logistic import REGULARIZED_LOGISTIC
from training.models.random_forest import REGULARIZED_RANDOM_FOREST
from training.models.svm import REGULARIZED_RBF_SVM
from training.models.xgboost import REGULARIZED_XGBOOST


DEFAULT_CANDIDATES = (
    REGULARIZED_LOGISTIC,
    REGULARIZED_RBF_SVM,
    REGULARIZED_RANDOM_FOREST,
    REGULARIZED_EXTRA_TREES,
    REGULARIZED_XGBOOST,
)
