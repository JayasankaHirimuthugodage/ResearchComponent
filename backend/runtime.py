"""Lazy bridge to the verified experimental three-domain runtime."""
from __future__ import annotations

from pathlib import Path

from inference.contracts import AssessmentStatus
from inference.tabiclv2_explanations import analyze_tabiclv2_cognitive_model_behaviour
from inference.tabiclv2_kv_cache_runtime import CACHE_RUNTIME_AUTHORIZATION, build_kv_cache_experimental_runtime_service


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
        cognitive_prediction = next(item for item in result.domain_predictions if item.domain == "cognitive")
        cognitive = analyze_tabiclv2_cognitive_model_behaviour(
            self.catalog, encoded, cognitive_prediction, service.cognitive
        )

        def xgboost_summary(domain: str):
            explanation = xgboost[domain]
            observations = [
                {"question": item.display_name}
                for item in explanation.contributions
                if item.feature != "baseline_checkpoint_id"
            ][:3]
            return {
                "domain": domain,
                "method": "local_model_observation_summary",
                "headline": "What the screening considered",
                "observations": observations,
                "limitation": (
                    "These points help explain this screening summary. They do not identify a "
                    "cause or provide a diagnosis."
                ),
            }

        grouped: dict[str, object] = {}
        for variation in cognitive.variations:
            previous = grouped.get(variation.question_id)
            if previous is None or abs(variation.probability_change) > abs(previous.probability_change):
                grouped[variation.question_id] = variation
        cognitive_observations = [
            {"question": item.question_text}
            for item in sorted(grouped.values(), key=lambda item: -abs(item.probability_change))[:3]
        ]
        cognitive_summary = {
            "domain": "cognitive",
            "method": "local_one_answer_model_behaviour_summary",
            "headline": "What the cognitive screening checked",
            "observations": cognitive_observations,
            "limitation": (
                "The cognitive screening checked how its calculation changed when one answer at a time "
                "was varied while all other answers stayed the same. This is not a cause, diagnosis, or "
                "measure of clinical importance."
            ),
        }
        return result, (xgboost_summary("communication"), cognitive_summary, xgboost_summary("social_emotional"))
        return self._service.predict_encoded(encoded)
