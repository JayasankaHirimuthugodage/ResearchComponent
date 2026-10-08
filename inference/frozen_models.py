"""Checksum-verified loading and prediction for the frozen domain artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import joblib

from inference.contracts import (
    AssessmentStatus,
    DOMAIN_IDS,
    DomainPrediction,
    PredictionResult,
    QuestionnaireSubmission,
)
from inference.model_artifact import CalibratedDomainModel
from inference.questionnaire import EncodedAssessment, QuestionnaireCatalog, encode_submission


MODEL_RELEASE_VERSION = "slc-development-models-v1"
EXPECTED_RELEASE_MANIFEST_SHA256 = (
    "A26A4AF2D1961EC8AE605A1E9BAE25D46F7CB8D8EAEE850897F1212929ED6D1C"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _resolve_release_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("A release path is missing.")
    candidate = (root / Path(value.replace("\\", "/"))).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("A release path points outside the repository.")
    return candidate


@dataclass(frozen=True)
class FrozenPredictionService:
    """Run the three frozen models without fitting or changing their thresholds."""

    catalog: QuestionnaireCatalog
    models: Mapping[str, CalibratedDomainModel]
    model_release_version: str

    def __post_init__(self) -> None:
        copied = dict(self.models)
        if set(copied) != set(DOMAIN_IDS):
            raise ValueError("Exactly one frozen model is required for each domain.")
        schema_versions = {model.feature_schema_version for model in copied.values()}
        if len(schema_versions) != 1:
            raise ValueError("Frozen models disagree on the feature schema version.")
        object.__setattr__(self, "models", MappingProxyType(copied))

    @property
    def feature_schema_version(self) -> str:
        return next(iter(self.models.values())).feature_schema_version

    def predict(self, submission: QuestionnaireSubmission) -> PredictionResult:
        return self.predict_encoded(encode_submission(self.catalog, submission))

    def predict_encoded(self, encoded: EncodedAssessment) -> PredictionResult:
        common = {
            "model_version": self.model_release_version,
            "feature_schema_version": self.feature_schema_version,
            "questionnaire_schema_version": self.catalog.questionnaire_schema_version,
            "threshold": None,
            "threshold_version": None,
        }
        if encoded.status is AssessmentStatus.INCOMPLETE:
            return PredictionResult(
                status=AssessmentStatus.INCOMPLETE,
                domain_predictions=(),
                overall_concern=None,
                incomplete_question_ids=encoded.incomplete_question_ids,
                **common,
            )
        if encoded.feature_values is None:
            raise RuntimeError("A complete assessment has no encoded model inputs.")

        predictions: list[DomainPrediction] = []
        for domain in DOMAIN_IDS:
            model = self.models[domain]
            probability = model.positive_probability(encoded.feature_values)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeError("A frozen model returned an invalid probability.")
            predictions.append(
                DomainPrediction(
                    domain=domain,
                    probability=probability,
                    decision=probability >= model.threshold,
                    threshold=model.threshold,
                    model_version=model.model_configuration_version,
                )
            )

        return PredictionResult(
            status=AssessmentStatus.COMPLETE,
            domain_predictions=tuple(predictions),
            overall_concern=any(item.decision for item in predictions),
            incomplete_question_ids=(),
            **common,
        )


def load_frozen_prediction_service(
    catalog: QuestionnaireCatalog,
    repository_root: Path | None = None,
) -> FrozenPredictionService:
    """Load the project-owned artifacts only after checking release provenance."""

    root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    release_path = root / "models" / MODEL_RELEASE_VERSION / "release-manifest.json"
    if not release_path.is_file():
        raise FileNotFoundError(
            "Frozen model release is unavailable. Restore models/slc-development-models-v1."
        )
    if _sha256(release_path) != EXPECTED_RELEASE_MANIFEST_SHA256:
        raise ValueError("Frozen model release manifest checksum does not match.")

    release = _load_json(release_path)
    if release.get("model_release_version") != MODEL_RELEASE_VERSION:
        raise ValueError("Frozen model release version does not match.")
    raw_domains = release.get("domains")
    if not isinstance(raw_domains, dict) or set(raw_domains) != set(DOMAIN_IDS):
        raise ValueError("Frozen model release must contain exactly three domains.")

    models: dict[str, CalibratedDomainModel] = {}
    for domain in DOMAIN_IDS:
        entry = raw_domains[domain]
        if not isinstance(entry, dict):
            raise ValueError(f"Release entry is invalid for {domain}.")
        artifact_path = _resolve_release_path(root, entry.get("artifact"))
        manifest_path = _resolve_release_path(root, entry.get("manifest"))
        if _sha256(artifact_path) != entry.get("artifact_sha256"):
            raise ValueError(f"Frozen artifact checksum does not match for {domain}.")
        if _sha256(manifest_path) != entry.get("manifest_sha256"):
            raise ValueError(f"Domain manifest checksum does not match for {domain}.")

        manifest = _load_json(manifest_path)
        model = joblib.load(artifact_path)
        if not isinstance(model, CalibratedDomainModel) or model.domain != domain:
            raise ValueError(f"Frozen artifact type or domain is invalid for {domain}.")
        if manifest.get("model_release_version") != MODEL_RELEASE_VERSION:
            raise ValueError(f"Domain release version does not match for {domain}.")
        if manifest.get("domain") != domain:
            raise ValueError(f"Domain manifest identity does not match for {domain}.")
        if manifest.get("model_configuration_version") != model.model_configuration_version:
            raise ValueError(f"Model version does not match its manifest for {domain}.")
        if manifest.get("feature_schema_version") != model.feature_schema_version:
            raise ValueError(f"Feature schema does not match its manifest for {domain}.")
        threshold = manifest.get("threshold")
        if not isinstance(threshold, dict) or threshold.get("value") != model.threshold:
            raise ValueError(f"Threshold does not match its manifest for {domain}.")
        models[domain] = model

    return FrozenPredictionService(
        catalog=catalog,
        models=models,
        model_release_version=MODEL_RELEASE_VERSION,
    )
