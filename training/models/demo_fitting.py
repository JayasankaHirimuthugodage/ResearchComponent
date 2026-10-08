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
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMNS,
)
from training.data.development_data import ModelingRecord, load_development_dataset
from training.models.common import (
    DEFAULT_MODEL_SEED,
    CandidateSpec,
    build_candidate_pipeline,
    fit_candidate_pipeline,
    modeling_arrays,
)
from training.models.extra_trees import REGULARIZED_EXTRA_TREES
from training.models.logistic import REGULARIZED_LOGISTIC
from training.experiments.evaluation import decision_threshold_metadata


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
class DemoCandidatePredictor:
    """One clearly labelled candidate in the local research comparison."""

    candidate_name: str
    display_name: str
    service: DevelopmentPredictionService
    threshold_status: str
    threshold_approval_statement: str | None


@dataclass(frozen=True)
class FittedDemoComparison:
    """The two in-memory candidate configurations and their shared provenance."""

    candidates: tuple[DemoCandidatePredictor, ...]
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


def fit_candidate_prediction_service(
    records: Sequence[ModelingRecord],
    *,
    candidate: CandidateSpec,
    catalog: QuestionnaireCatalog | None = None,
) -> DevelopmentPredictionService:
    """Fit one candidate's three domain classifiers from supplied development rows."""

    features, targets = modeling_arrays(records)
    if targets.shape[1] != len(DOMAIN_IDS) or len(TARGET_COLUMNS) != len(DOMAIN_IDS):
        raise RuntimeError("The development data no longer has exactly three targets.")

    adapters: dict[str, SklearnPositiveProbabilityAdapter] = {}
    for index, domain in enumerate(DOMAIN_IDS):
        domain_targets = targets[:, index]
        if set(np.unique(domain_targets).tolist()) != {0, 1}:
            raise ValueError(f"The development partition lacks both classes for {domain}.")
        pipeline = build_candidate_pipeline(
            candidate,
            seed=DEFAULT_MODEL_SEED + index,
        )
        fit_candidate_pipeline(pipeline, candidate, features, domain_targets)
        adapters[domain] = SklearnPositiveProbabilityAdapter(pipeline)

    threshold_metadata = decision_threshold_metadata(
        APPROVED_DECISION_THRESHOLD,
        model_configuration_version=candidate.version,
    )
    threshold_version = threshold_metadata["version"]
    if threshold_version is not None and not isinstance(threshold_version, str):
        raise RuntimeError("Threshold metadata contains an invalid version.")

    return DevelopmentPredictionService(
        catalog=catalog or load_questionnaire_catalog(),
        models=adapters,
        model_version=candidate.version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        threshold=APPROVED_DECISION_THRESHOLD,
        threshold_version=threshold_version,
    )


def fit_demo_prediction_service(
    records: Sequence[ModelingRecord],
    *,
    catalog: QuestionnaireCatalog | None = None,
) -> DevelopmentPredictionService:
    """Retain the original logistic-only adapter for compatible callers."""

    return fit_candidate_prediction_service(
        records,
        candidate=REGULARIZED_LOGISTIC,
        catalog=catalog,
    )


def _demo_provenance(
    *,
    development_participants: int,
    dataset_sha256: str,
    split_fingerprint_sha256: str,
) -> DemoModelProvenance:
    return DemoModelProvenance(
        fit_scope="development_partition_only",
        development_participants=development_participants,
        dataset_sha256=dataset_sha256,
        split_fingerprint_sha256=split_fingerprint_sha256,
        final_test_evaluated=False,
        artifact_persisted=False,
    )


def fit_demo_candidate_predictors(
    records: Sequence[ModelingRecord],
    *,
    catalog: QuestionnaireCatalog | None = None,
) -> tuple[DemoCandidatePredictor, ...]:
    """Fit both labelled candidates from one explicitly supplied record collection."""

    shared_catalog = catalog or load_questionnaire_catalog()
    candidate_definitions = (
        (REGULARIZED_LOGISTIC, "Logistic regression"),
        (REGULARIZED_EXTRA_TREES, "Extra Trees"),
    )
    candidates: list[DemoCandidatePredictor] = []
    for candidate, display_name in candidate_definitions:
        threshold_metadata = decision_threshold_metadata(
            APPROVED_DECISION_THRESHOLD,
            model_configuration_version=candidate.version,
        )
        status = threshold_metadata["status"]
        approval_statement = threshold_metadata["approval_statement"]
        if not isinstance(status, str):
            raise RuntimeError("Threshold metadata contains an invalid status.")
        if approval_statement is not None and not isinstance(approval_statement, str):
            raise RuntimeError("Threshold metadata contains an invalid approval statement.")
        candidates.append(
            DemoCandidatePredictor(
                candidate_name=candidate.name,
                display_name=display_name,
                service=fit_candidate_prediction_service(
                    records,
                    candidate=candidate,
                    catalog=shared_catalog,
                ),
                threshold_status=status,
                threshold_approval_statement=approval_statement,
            )
        )
    return tuple(candidates)


def fit_development_demo_predictor() -> FittedDemoPredictor:
    """Fit three provisional logistic models using only the frozen development rows."""

    development = load_development_dataset()
    service = fit_demo_prediction_service(development.records)
    return FittedDemoPredictor(
        service=service,
        provenance=_demo_provenance(
            development_participants=len(development.records),
            dataset_sha256=development.dataset_sha256,
            split_fingerprint_sha256=development.split_fingerprint_sha256,
        ),
    )


def fit_development_demo_comparison() -> FittedDemoComparison:
    """Fit both research candidates in memory from the same development records."""

    development = load_development_dataset()
    return FittedDemoComparison(
        candidates=fit_demo_candidate_predictors(development.records),
        provenance=_demo_provenance(
            development_participants=len(development.records),
            dataset_sha256=development.dataset_sha256,
            split_fingerprint_sha256=development.split_fingerprint_sha256,
        ),
    )
