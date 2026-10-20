"""Lazy bridge to the verified experimental three-domain runtime."""
from __future__ import annotations

from pathlib import Path

from inference.contracts import AssessmentStatus
from inference.tabiclv2_explanations import analyze_tabiclv2_cognitive_model_behaviour
from inference.tabiclv2_kv_cache_runtime import CACHE_RUNTIME_AUTHORIZATION, build_kv_cache_experimental_runtime_service


def _decision_supporting_cognitive_variations(variations, decision: bool, limit: int = 3):
    """Rank answer variations that move away from the displayed decision."""
    grouped = {}
    for variation in variations:
        change = variation.probability_change
        supports_result = change < 0 if decision else change > 0
        if not supports_result:
            continue
        previous = grouped.get(variation.question_id)
        if previous is None or abs(change) > abs(previous.probability_change):
            grouped[variation.question_id] = variation
    return sorted(grouped.values(), key=lambda item: -abs(item.probability_change))[:limit]


class ExistingExperimentalRuntime:
    def __init__(self, catalog) -> None:
        self.catalog = catalog
        self._service = None

    def _service_instance(self):
        if self._service is None:
            root = Path(__file__).resolve().parents[1]
            checkpoint = root / ".venv-tabiclv2" / "hf-cache" / "hub" / "models--jingang--TabICL" / "snapshots" / "4dcd344ece2c00be9e831fdd35bed57b5ad83e19" / "tabicl-classifier-v2-20260212.ckpt"
            manifest = root / "models" / "experimental-three-domain-tabiclv2-kv-cache-v1" / "release-manifest.json"
            self._service = build_kv_cache_experimental_runtime_service(self.catalog, repository_root=root, composite_manifest=manifest, checkpoint=checkpoint, execution_authorization=CACHE_RUNTIME_AUTHORIZATION)
        return self._service

    def __call__(self, encoded):
        return self._service_instance().predict_encoded(encoded)

    def explain(self, encoded):
        """Return only caregiver-safe local explanation summaries.

        Communication and Social-emotional use their approved XGBoost local
        contribution implementation. Cognitive uses the fixed, non-causal
        30-alternative TabICLv2 local model-behaviour analysis. Probabilities,
        thresholds, contribution values and raw answer vectors never leave this
        runtime boundary.
        """
        service = self._service_instance()
        result = service.predict_encoded(encoded)
        if result.status is not AssessmentStatus.COMPLETE:
            return result, ()

        xgboost = {item.domain: item for item in service.explain_xgboost_encoded(encoded, result)}
        decisions = {item.domain: item.decision for item in result.domain_predictions}
        cognitive_prediction = next(item for item in result.domain_predictions if item.domain == "cognitive")
        cognitive = analyze_tabiclv2_cognitive_model_behaviour(
            self.catalog, encoded, cognitive_prediction, service.cognitive
        )

        def xgboost_summary(domain: str):
            explanation = xgboost[domain]
            supporting_direction = "toward_concern" if decisions[domain] else "away_from_concern"
            observations = [
                {
                    "question": item.display_name,
                    "questionnaire_domain": item.feature.rsplit("_q", 1)[0],
                    "response": item.input_value_label,
                    "relationship": (
                        "toward_further_attention" if decisions[domain] else "away_from_further_attention"
                    ),
                }
                for item in explanation.contributions
                if item.feature != "baseline_checkpoint_id" and item.direction == supporting_direction
            ][:3]
            return {
                "domain": domain,
                "method": "local_model_observation_summary",
                "headline": "Answer most relevant to this result",
                "observations": observations,
                "limitation": (
                    "These answers help explain the result but do not identify a cause."
                ),
            }

        cognitive_observations = [
            {
                "question": item.question_text,
                "questionnaire_domain": item.question_domain,
                "response": {0: "Not yet", 1: "Sometimes", 2: "Yes, usually"}[item.original_response],
                "relationship": "response_variation",
            }
            for item in _decision_supporting_cognitive_variations(
                cognitive.variations, cognitive.baseline_decision
            )
        ]
        cognitive_summary = {
            "domain": "cognitive",
            "method": "local_one_answer_model_behaviour_summary",
            "headline": "Answer most relevant to this result",
            "observations": cognitive_observations,
            "limitation": (
                "These answers help explain the result but do not identify a cause."
            ),
        }
        return result, (xgboost_summary("communication"), cognitive_summary, xgboost_summary("social_emotional"))
        return self._service.predict_encoded(encoded)
