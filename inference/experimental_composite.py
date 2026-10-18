"""Synthetic-testable, explicit three-domain experimental release adapter.

This module is deliberately separate from ``frozen_models``.  It cannot become
the default historical release and it never creates or reads a context source.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from inference.contracts import AssessmentStatus, DOMAIN_IDS, DomainPrediction, PredictionResult
from inference.experimental_release import EXPERIMENTAL_COMPOSITE_RELEASE, EXPERIMENTAL_KV_CACHE_COMPOSITE_RELEASE, ExperimentalReleaseSelection
from inference.questionnaire import EncodedAssessment, QuestionnaireCatalog, encode_submission
from inference.tabiclv2_adapter import TabICLv2CognitiveAdapter, load_cognitive_release_contract
from inference.explanations import DomainLocalExplanation, explain_complete_assessment
from inference.model_artifact import CalibratedDomainModel


COMPOSITE_SCHEMA = "experimental-three-domain-composite-v1"


class ProbabilityModel(Protocol):
    def positive_probability(self, feature_values: tuple[str, ...]) -> float: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _inside(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Experimental release path is missing.")
    path = (root / value.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Experimental release path is unavailable or outside its root.")
    return path


@dataclass(frozen=True)
class ExperimentalCompositeRelease:
    """Validated references only; no model artifact is copied or loaded here."""

    communication_artifact: Path
    communication_manifest: Path
    communication_threshold: float
    cognitive_manifest: Path
    cognitive_integrity_index: Path
    social_emotional_artifact: Path
    social_emotional_manifest: Path
    social_emotional_threshold: float
    rollback_cognitive_artifact: Path
    rollback_cognitive_manifest: Path


def load_experimental_composite_release(path: Path, root: Path) -> ExperimentalCompositeRelease:
    selection = ExperimentalReleaseSelection(EXPERIMENTAL_COMPOSITE_RELEASE)
    selection.require_experimental_cognitive()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != COMPOSITE_SCHEMA or value.get("release_id") not in {EXPERIMENTAL_COMPOSITE_RELEASE, EXPERIMENTAL_KV_CACHE_COMPOSITE_RELEASE} or value.get("status") != "experimental_not_deployed":
        raise ValueError("Experimental composite release is invalid or not explicitly experimental.")
    domains = value.get("domains")
    rollback = value.get("rollback_cognitive")
    if not isinstance(domains, dict) or set(domains) != set(DOMAIN_IDS) or not isinstance(rollback, dict):
        raise ValueError("Experimental composite release does not cover the required domains.")

    def verified(entry: object, keys: tuple[str, str]) -> tuple[Path, Path]:
        if not isinstance(entry, dict):
            raise ValueError("Experimental composite entry is invalid.")
        artifact, manifest = _inside(root, entry.get(keys[0])), _inside(root, entry.get(keys[1]))
        if _sha256(artifact) != entry.get(f"{keys[0]}_sha256") or _sha256(manifest) != entry.get(f"{keys[1]}_sha256"):
            raise ValueError("Experimental composite artifact checksum mismatch.")
        return artifact, manifest

    communication_artifact, communication_manifest = verified(domains["communication"], ("artifact", "manifest"))
    social_artifact, social_manifest = verified(domains["social_emotional"], ("artifact", "manifest"))
    rollback_artifact, rollback_manifest = verified(rollback, ("artifact", "manifest"))
    cognitive = domains["cognitive"]
    if not isinstance(cognitive, dict):
        raise ValueError("Experimental Cognitive reference is invalid.")
    cognitive_manifest = _inside(root, cognitive.get("manifest"))
    cognitive_index = _inside(root, cognitive.get("integrity_index"))
    if _sha256(cognitive_manifest) != cognitive.get("manifest_sha256") or _sha256(cognitive_index) != cognitive.get("integrity_index_sha256"):
        raise ValueError("Experimental Cognitive provenance checksum mismatch.")
    contract = load_cognitive_release_contract(cognitive_manifest)
    index = json.loads(cognitive_index.read_text(encoding="utf-8"))
    indexed = index.get("artifacts", {}).get("release_manifest", {}) if isinstance(index, dict) else {}
    if indexed.get("sha256") != _sha256(cognitive_manifest) or contract.threshold <= 0.0:
        raise ValueError("Experimental Cognitive integrity index is invalid.")
    def frozen_threshold(manifest: Path, domain: str) -> float:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        threshold = value.get("threshold", {}) if isinstance(value, dict) else {}
        result = float(threshold.get("value")) if isinstance(threshold, dict) else float("nan")
        if value.get("domain") != domain or not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError("Frozen XGBoost manifest threshold is invalid.")
        return result
    return ExperimentalCompositeRelease(communication_artifact, communication_manifest, frozen_threshold(communication_manifest, "communication"), cognitive_manifest, cognitive_index, social_artifact, social_manifest, frozen_threshold(social_manifest, "social_emotional"), rollback_artifact, rollback_manifest)


@dataclass(frozen=True)
class ExperimentalThreeDomainService:
    """Three-domain inference contract with an explicitly supplied TabICLv2 adapter."""

    catalog: QuestionnaireCatalog
    communication: ProbabilityModel
    communication_threshold: float
    cognitive: TabICLv2CognitiveAdapter
    social_emotional: ProbabilityModel
    social_emotional_threshold: float
    model_version: str = EXPERIMENTAL_COMPOSITE_RELEASE

    def predict_encoded(self, encoded: EncodedAssessment) -> PredictionResult:
        common = {"model_version": self.model_version, "feature_schema_version": "primary-t0-16-v1", "questionnaire_schema_version": self.catalog.questionnaire_schema_version, "threshold": None, "threshold_version": None}
        if encoded.status is AssessmentStatus.INCOMPLETE:
            return PredictionResult(status=AssessmentStatus.INCOMPLETE, domain_predictions=(), overall_concern=None, incomplete_question_ids=encoded.incomplete_question_ids, **common)
        if encoded.feature_values is None:
            raise RuntimeError("Complete assessment has no feature values.")
        models: tuple[tuple[str, ProbabilityModel], ...] = (("communication", self.communication), ("cognitive", self.cognitive), ("social_emotional", self.social_emotional))
        predictions: list[DomainPrediction] = []
        for domain, model in models:
            probability = float(model.positive_probability(encoded.feature_values))
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeError("Experimental model returned an invalid probability.")
            threshold = self.cognitive.release.threshold if domain == "cognitive" else self.communication_threshold if domain == "communication" else self.social_emotional_threshold
            predictions.append(DomainPrediction(domain=domain, probability=probability, decision=probability >= threshold, threshold=threshold, model_version=self.model_version))
        return PredictionResult(status=AssessmentStatus.COMPLETE, domain_predictions=tuple(predictions), overall_concern=any(item.decision for item in predictions), incomplete_question_ids=(), **common)

    def predict(self, submission):
        return self.predict_encoded(encode_submission(self.catalog, submission))

    def explain_xgboost_encoded(
        self,
        encoded: EncodedAssessment,
        result: PredictionResult,
    ) -> tuple[DomainLocalExplanation, ...]:
        """Explain only the two XGBoost domains in this composite release.

        Cognitive intentionally remains excluded: its selected model is TabICLv2,
        not the historical Logistic Regression artifact.
        """
        models = {
            "communication": self.communication,
            "social_emotional": self.social_emotional,
        }
        if not all(isinstance(model, CalibratedDomainModel) for model in models.values()):
            raise TypeError("Experimental XGBoost explanation artifacts are invalid.")
        predictions = tuple(
            prediction
            for prediction in result.domain_predictions
            if prediction.domain in models
        )
        return explain_complete_assessment(self.catalog, models, encoded, predictions)


def build_experimental_service(
    catalog: QuestionnaireCatalog,
    release: ExperimentalCompositeRelease,
    cognitive: TabICLv2CognitiveAdapter,
    *,
    artifact_loader: Callable[[Path], ProbabilityModel],
) -> ExperimentalThreeDomainService:
    """Load only verified XGBoost references; no fallback path is available."""
    return ExperimentalThreeDomainService(catalog, artifact_loader(release.communication_artifact), release.communication_threshold, cognitive, artifact_loader(release.social_emotional_artifact), release.social_emotional_threshold)
