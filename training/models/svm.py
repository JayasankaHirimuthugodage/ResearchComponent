"""Predeclared radial-basis support-vector-machine candidate."""

from __future__ import annotations

from sklearn.svm import SVC

from training.contracts import SVM_MODEL_CONFIGURATION_VERSION
from training.models.common import CandidateSpec


def _balanced_rbf_svm(seed: int) -> SVC:
    return SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=seed,
    )


REGULARIZED_RBF_SVM = CandidateSpec(
    name="svm_rbf_balanced",
    version=SVM_MODEL_CONFIGURATION_VERSION,
    estimator_factory=_balanced_rbf_svm,
    explanation_method="held_out_permutation_importance_average_precision",
    scale_questionnaire=True,
    public_parameters={
        "estimator": "SVC",
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
    },
)
