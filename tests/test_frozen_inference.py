from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from demo.development_prediction_demo import (
    load_synthetic_fixture,
    synthetic_responses_for_checkpoint,
)
from inference.contracts import AssessmentStatus, DOMAIN_IDS, QuestionnaireSubmission
from inference.frozen_models import (
    MODEL_RELEASE_VERSION,
    load_frozen_prediction_service,
)
from inference.explanations import CAREGIVER_EXPLANATION_NOTICE
from inference.model_artifact import CalibratedDomainModel
from inference.questionnaire import encode_submission, load_questionnaire_catalog
from training.contracts import CHECKPOINT_IDS


ROOT = Path(__file__).resolve().parents[1]
RELEASE_MANIFEST = ROOT / "models" / MODEL_RELEASE_VERSION / "release-manifest.json"


def submission_from_fixture(filename: str) -> QuestionnaireSubmission:
    fixture = load_synthetic_fixture(filename)
    return QuestionnaireSubmission(
        schema_version=fixture["schema_version"],
        checkpoint_id=fixture["checkpoint_id"],
        responses=fixture["responses"],
    )


@unittest.skipUnless(RELEASE_MANIFEST.is_file(), "Local frozen artifacts are unavailable.")
class FrozenInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = load_frozen_prediction_service(load_questionnaire_catalog())

    def test_release_loads_three_expected_models_and_thresholds(self) -> None:
        self.assertEqual(DOMAIN_IDS, tuple(self.service.models))
        self.assertEqual(
            (
                "xgboost-regularized-balanced-v1",
                "logistic-l2-balanced-v1",
                "xgboost-regularized-balanced-v1",
            ),
            tuple(self.service.models[domain].model_configuration_version for domain in DOMAIN_IDS),
        )
        expected_thresholds = (0.13088240001436519, 0.10968010787124531, 0.1261365307816501)
        for actual, expected in zip(
            (self.service.models[domain].threshold for domain in DOMAIN_IDS),
            expected_thresholds,
            strict=True,
        ):
            self.assertAlmostEqual(expected, actual, places=12)

    def test_complete_fixture_uses_each_domain_threshold_and_overall_or(self) -> None:
        result = self.service.predict(
            submission_from_fixture("synthetic_complete_assessment.json")
        )

        self.assertIs(result.status, AssessmentStatus.COMPLETE)
        self.assertEqual(DOMAIN_IDS, tuple(item.domain for item in result.domain_predictions))
        for prediction in result.domain_predictions:
            self.assertIsNotNone(prediction.threshold)
            self.assertIsNotNone(prediction.model_version)
            self.assertEqual(
                prediction.probability >= float(prediction.threshold),
                prediction.decision,
            )
        self.assertEqual(
            any(item.decision for item in result.domain_predictions),
            result.overall_concern,
        )

    def test_model_specific_local_explanations_are_complete_and_deterministic(self) -> None:
        fixture = submission_from_fixture("synthetic_complete_assessment.json")
        encoded = encode_submission(self.service.catalog, fixture)
        first_result, first = self.service.predict_and_explain_encoded(encoded)
        second_result, second = self.service.predict_and_explain_encoded(encoded)

        self.assertEqual(first_result, second_result)
        self.assertEqual(first, second)
        self.assertEqual(
            (
                "xgboost_tree_shap_contribution",
                "logistic_coefficient_contribution",
                "xgboost_tree_shap_contribution",
            ),
            tuple(item.method for item in first),
        )
        for prediction, explanation in zip(
            first_result.domain_predictions,
            first,
            strict=True,
        ):
            self.assertEqual(16, len(explanation.contributions))
            self.assertEqual(
                16,
                len({item.feature for item in explanation.contributions}),
            )
            self.assertIn(
                "Observation contributing most to this result",
                explanation.caregiver_message,
            )
            self.assertNotIn("Observation pointing in the other direction", explanation.caregiver_message)
            self.assertNotIn("all 15 answers", explanation.caregiver_message)
            self.assertNotIn("probability", explanation.caregiver_message.lower())
            self.assertNotIn("participant", explanation.caregiver_message.lower())
            questionnaire_items = [
                item
                for item in explanation.contributions
                if item.feature != "baseline_checkpoint_id" and item.direction != "neutral"
            ]
            supporting_direction = (
                "toward_concern" if prediction.decision else "away_from_concern"
            )
            item = next(
                (
                    value
                    for value in questionnaire_items
                    if value.direction == supporting_direction
                ),
                None,
            )
            if item is not None:
                self.assertIn(item.display_name, explanation.caregiver_message)
                self.assertIn(item.input_value_label, explanation.caregiver_message)
            expected_outcome = (
                "met or exceeded the domain threshold, so a concern flag was produced"
                if prediction.decision
                else "remained below the domain threshold, so no concern flag was produced"
            )
            self.assertIn(expected_outcome, explanation.caregiver_message)

        cognitive_prediction = first_result.domain_predictions[1]
        cognitive_explanation = first[1]
        self.assertFalse(cognitive_prediction.decision)
        self.assertIn("remained below", cognitive_explanation.caregiver_message)
        self.assertIn("moved the model calculation away from", cognitive_explanation.caregiver_message)
        self.assertNotIn("moved the model calculation toward", cognitive_explanation.caregiver_message)
        self.assertIn("all 15 answers", CAREGIVER_EXPLANATION_NOTICE)
        self.assertIn("not diagnoses", CAREGIVER_EXPLANATION_NOTICE)

    def test_different_response_patterns_produce_different_local_explanations(self) -> None:
        catalog = self.service.catalog
        checkpoint_id = "CP24M"

        def submission(response_code: str) -> QuestionnaireSubmission:
            return QuestionnaireSubmission(
                schema_version=catalog.questionnaire_schema_version,
                checkpoint_id=checkpoint_id,
                responses={
                    question.question_id: response_code
                    for question in catalog.questions_for_checkpoint(checkpoint_id)
                },
            )

        _, positive_pattern = self.service.predict_and_explain_encoded(
            encode_submission(catalog, submission("YES_USUALLY"))
        )
        _, concern_pattern = self.service.predict_and_explain_encoded(
            encode_submission(catalog, submission("NOT_YET"))
        )

        for positive, concern in zip(positive_pattern, concern_pattern, strict=True):
            self.assertNotEqual(positive.caregiver_message, concern.caregiver_message)
            positive_signature = tuple(
                (item.feature, item.direction, round(item.contribution, 8))
                for item in positive.contributions
            )
            concern_signature = tuple(
                (item.feature, item.direction, round(item.contribution, 8))
                for item in concern.contributions
            )
            self.assertNotEqual(positive_signature, concern_signature)

    def test_incomplete_assessment_has_no_local_explanations(self) -> None:
        fixture = submission_from_fixture("synthetic_incomplete_assessment.json")
        encoded = encode_submission(self.service.catalog, fixture)
        result, explanations = self.service.predict_and_explain_encoded(encoded)

        self.assertIs(result.status, AssessmentStatus.INCOMPLETE)
        self.assertEqual((), explanations)

    def test_complete_synthetic_example_predicts_for_every_checkpoint(self) -> None:
        for checkpoint_id in CHECKPOINT_IDS:
            result = self.service.predict(
                QuestionnaireSubmission(
                    schema_version=self.service.catalog.questionnaire_schema_version,
                    checkpoint_id=checkpoint_id,
                    responses=synthetic_responses_for_checkpoint(
                        self.service.catalog,
                        checkpoint_id,
                        incomplete=False,
                    ),
                )
            )

            self.assertIs(result.status, AssessmentStatus.COMPLETE)
            self.assertEqual(3, len(result.domain_predictions))

    def test_incomplete_synthetic_example_blocks_models_for_every_checkpoint(self) -> None:
        with patch.object(
            CalibratedDomainModel,
            "positive_probability",
            side_effect=AssertionError("Model must not run for an incomplete assessment."),
        ):
            for checkpoint_id in CHECKPOINT_IDS:
                result = self.service.predict(
                    QuestionnaireSubmission(
                        schema_version=self.service.catalog.questionnaire_schema_version,
                        checkpoint_id=checkpoint_id,
                        responses=synthetic_responses_for_checkpoint(
                            self.service.catalog,
                            checkpoint_id,
                            incomplete=True,
                        ),
                    )
                )
                self.assertIs(result.status, AssessmentStatus.INCOMPLETE)

    def test_incomplete_fixture_does_not_call_models(self) -> None:
        with patch.object(
            CalibratedDomainModel,
            "positive_probability",
            side_effect=AssertionError("Model must not run for an incomplete assessment."),
        ):
            result = self.service.predict(
                submission_from_fixture("synthetic_incomplete_assessment.json")
            )

        self.assertIs(result.status, AssessmentStatus.INCOMPLETE)
        self.assertEqual((), result.domain_predictions)
        self.assertIsNone(result.overall_concern)
        self.assertEqual(("CP24M_COMM_Q3",), result.incomplete_question_ids)

    def test_release_manifest_checksum_is_enforced(self) -> None:
        with patch("inference.frozen_models._sha256", return_value="WRONG"):
            with self.assertRaisesRegex(ValueError, "release manifest checksum"):
                load_frozen_prediction_service(load_questionnaire_catalog())


if __name__ == "__main__":
    unittest.main()
