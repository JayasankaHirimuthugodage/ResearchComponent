"""Development-only, in-memory fitting adapter for the local prediction demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.pipeline import Pipeline

from inference.contracts import DOMAIN_IDS
from inference.predictor import DevelopmentPredictionService
from inference.questionnaire import QuestionnaireCatalog, load_questionnaire_catalog
from training.contracts import (
    APPROVED_DECISION_THRESHOLD,
    DECISION_THRESHOLD_VERSION,
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMNS,
)
from training.data.development_data import ModelingRecord, load_development_dataset
from training.models.common import (
    DEFAULT_MODEL_SEED,
    build_candidate_pipeline,
    modeling_arrays,
)
from training.models.logistic import REGULARIZED_LOGISTIC


@dataclass(frozen=True)
class DemoModelProvenance:
    fit_scope: str
    development_participants: int
    dataset_sha256: str
    split_fingerprint_sha256: str
    final_test_evaluated: bool
    artifact_persisted: bool


@dataclass(frozen=True)
class FittedDemoPredictor:
    service: DevelopmentPredictionService
    provenance: DemoModelProvenance


@dataclass(frozen=True)
class SklearnPositiveProbabilityAdapter:
    """Keep scikit-learn details outside the framework-neutral use case."""

    pipeline: Pipeline

    def positive_probability(self, feature_values: tuple[str, ...]) -> float:
        matrix = np.asarray([feature_values], dtype=object)
        probabilities = self.pipeline.predict_proba(matrix)
        classes = tuple(int(value) for value in self.pipeline.classes_)
        if probabilities.shape != (1, len(classes)) or 1 not in classes:
            raise RuntimeError("The fitted classifier has no positive binary class.")
        return float(probabilities[0, classes.index(1)])


def fit_demo_prediction_service(
    records: Sequence[ModelingRecord],
    *,
    catalog: QuestionnaireCatalog | None = None,
) -> DevelopmentPredictionService:
    """Fit the three-domain configuration from explicitly supplied development rows."""

    features, targets = modeling_arrays(records)
    if targets.shape[1] != len(DOMAIN_IDS) or len(TARGET_COLUMNS) != len(DOMAIN_IDS):
        raise RuntimeError("The development data no longer has exactly three targets.")

    adapters: dict[str, SklearnPositiveProbabilityAdapter] = {}
    for index, domain in enumerate(DOMAIN_IDS):
        domain_targets = targets[:, index]
        if set(np.unique(domain_targets).tolist()) != {0, 1}:
            raise ValueError(f"The development partition lacks both classes for {domain}.")
        pipeline = build_candidate_pipeline(
            REGULARIZED_LOGISTIC,
            seed=DEFAULT_MODEL_SEED + index,
        )
        pipeline.fit(features, domain_targets)
        adapters[domain] = SklearnPositiveProbabilityAdapter(pipeline)

    return DevelopmentPredictionService(
        catalog=catalog or load_questionnaire_catalog(),
        models=adapters,
        model_version=REGULARIZED_LOGISTIC.version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        threshold=APPROVED_DECISION_THRESHOLD,
        threshold_version=DECISION_THRESHOLD_VERSION,
    )


def fit_development_demo_predictor() -> FittedDemoPredictor:
    """Fit three provisional logistic models using only the frozen development rows."""

    development = load_development_dataset()
    service = fit_demo_prediction_service(development.records)
    return FittedDemoPredictor(
        service=service,
        provenance=DemoModelProvenance(
            fit_scope="development_partition_only",
            development_participants=len(development.records),
            dataset_sha256=development.dataset_sha256,
            split_fingerprint_sha256=development.split_fingerprint_sha256,
            final_test_evaluated=False,
            artifact_persisted=False,
        ),
    )
