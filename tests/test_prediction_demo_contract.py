from __future__ import annotations

import unittest
from dataclasses import dataclass

from demo.development_prediction_demo import (
    format_area_summary,
    load_synthetic_fixture,
    synthetic_responses_for_checkpoint,
)
from inference.contracts import (
    AssessmentStatus,
    DOMAIN_IDS,
    QuestionnaireSubmission,
    SubmissionValidationError,
)
from inference.predictor import DevelopmentPredictionService
from inference.questionnaire import encode_submission, load_questionnaire_catalog
from training.contracts import (
    APPROVED_DECISION_THRESHOLD,
    CHECKPOINT_IDS,
    DECISION_THRESHOLD_APPROVAL_STATUS,
    DECISION_THRESHOLD_VERSION,
    EXPERIMENTAL_DECISION_THRESHOLD_STATUS,
    FEATURE_SCHEMA_VERSION,
    QUESTION_COLUMNS,
)
from training.data.development_data import ModelingRecord
from training.models.demo_fitting import (
    fit_demo_candidate_predictors,
    fit_demo_prediction_service,
)
from training.models.extra_trees import REGULARIZED_EXTRA_TREES
from training.models.logistic import REGULARIZED_LOGISTIC


def submission_from_fixture(filename: str) -> QuestionnaireSubmission:
    fixture = load_synthetic_fixture(filename)
    return QuestionnaireSubmission(
        schema_version=fixture["schema_version"],
        checkpoint_id=fixture["checkpoint_id"],
        responses=fixture["responses"],
    )


@dataclass
class FakeProbabilityModel:
    probability: float
    calls: int = 0

    def positive_probability(self, feature_values: tuple[str, ...]) -> float:
        self.calls += 1
        if len(feature_values) != 16:
            raise AssertionError("The predictor received an invalid feature vector.")
        return self.probability


def fake_service(probabilities: tuple[float, float, float]) -> DevelopmentPredictionService:
    return DevelopmentPredictionService(
        catalog=load_questionnaire_catalog(),
        models={
            domain: FakeProbabilityModel(probability)
            for domain, probability in zip(DOMAIN_IDS, probabilities, strict=True)
        },
        model_version="synthetic-contract-model-v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        threshold=APPROVED_DECISION_THRESHOLD,
        threshold_version=DECISION_THRESHOLD_VERSION,
    )


def synthetic_training_records() -> tuple[ModelingRecord, ...]:
    combinations = (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    )
    records: list[ModelingRecord] = []
    for repetition in range(2):
        for combination_index, combination in enumerate(combinations):
            position = repetition * len(combinations) + combination_index
            answers = tuple(
                str((position + question_index) % 3)
                for question_index in range(len(QUESTION_COLUMNS))
            )
            records.append(
                ModelingRecord(
                    participant_key=f"synthetic-demo-{position:03d}",
                    feature_values=(CHECKPOINT_IDS[position % len(CHECKPOINT_IDS)], *answers),
                    target_values=combination,
                )
            )
    return tuple(records)


class QuestionnaireCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_questionnaire_catalog()

    def test_catalogue_maps_180_unique_questions_to_each_checkpoint_schema(self) -> None:
        self.assertEqual(180, len(self.catalog.questions))
        self.assertEqual(180, len({item.question_id for item in self.catalog.questions}))
        for checkpoint_id in CHECKPOINT_IDS:
            questions = self.catalog.questions_for_checkpoint(checkpoint_id)
            self.assertEqual(15, len(questions))
            self.assertEqual(QUESTION_COLUMNS, tuple(item.feature_column for item in questions))
            self.assertTrue(all(item.text for item in questions))

    def test_complete_fixture_encodes_exactly_16_inputs(self) -> None:
        encoded = encode_submission(
            self.catalog,
            submission_from_fixture("synthetic_complete_assessment.json"),
        )

        self.assertIs(encoded.status, AssessmentStatus.COMPLETE)
        self.assertIsNotNone(encoded.feature_values)
        self.assertEqual(16, len(encoded.feature_values or ()))
        self.assertEqual("CP24M", encoded.feature_values[0] if encoded.feature_values else None)
        self.assertEqual(
            ("2", "1", "0", "2", "1"),
            encoded.feature_values[1:6] if encoded.feature_values else (),
        )

    def test_unscorable_fixture_is_incomplete_and_has_no_features(self) -> None:
        encoded = encode_submission(
            self.catalog,
            submission_from_fixture("synthetic_incomplete_assessment.json"),
        )

        self.assertIs(encoded.status, AssessmentStatus.INCOMPLETE)
        self.assertIsNone(encoded.feature_values)
        self.assertEqual(("CP24M_COMM_Q3",), encoded.incomplete_question_ids)

    def test_synthetic_examples_match_every_checkpoint(self) -> None:
        for checkpoint_id in CHECKPOINT_IDS:
            expected_ids = {
                question.question_id
                for question in self.catalog.questions_for_checkpoint(checkpoint_id)
            }
            complete = synthetic_responses_for_checkpoint(
                self.catalog,
                checkpoint_id,
                incomplete=False,
            )
            incomplete = synthetic_responses_for_checkpoint(
                self.catalog,
                checkpoint_id,
                incomplete=True,
            )

            self.assertEqual(expected_ids, set(complete))
            self.assertEqual(expected_ids, set(incomplete))
            self.assertEqual(15, len(complete))
            self.assertNotIn("NOT_SURE", complete.values())
            self.assertEqual(1, tuple(incomplete.values()).count("NOT_SURE"))

    def test_missing_or_extra_questions_are_rejected(self) -> None:
        original = submission_from_fixture("synthetic_complete_assessment.json")
        missing = dict(original.responses)
        missing.pop("CP24M_COMM_Q1")
        with self.assertRaisesRegex(SubmissionValidationError, "missing=1"):
            encode_submission(
                self.catalog,
                QuestionnaireSubmission(
                    schema_version=original.schema_version,
                    checkpoint_id=original.checkpoint_id,
                    responses=missing,
                ),
            )

        extra = dict(original.responses)
        extra["participant_number"] = "must-not-enter-contract"
        with self.assertRaisesRegex(SubmissionValidationError, "extra=1"):
            encode_submission(
                self.catalog,
                QuestionnaireSubmission(
                    schema_version=original.schema_version,
                    checkpoint_id=original.checkpoint_id,
                    responses=extra,
                ),
            )


class PredictionApplicationTests(unittest.TestCase):
    def test_area_summary_names_the_identified_domains(self) -> None:
        self.assertEqual(
            "No developmental areas were flagged by the current model results.",
            format_area_summary(
                {"communication": False, "cognitive": False, "social_emotional": False}
            ),
        )
        self.assertEqual(
            "Area identified for further attention: Social / Emotional",
            format_area_summary(
                {"communication": False, "cognitive": False, "social_emotional": True}
            ),
        )
        self.assertEqual(
            "Areas identified for further attention: Communication and Cognitive",
            format_area_summary(
                {"communication": True, "cognitive": True, "social_emotional": False}
            ),
        )
        self.assertEqual(
            "Areas identified for further attention: Communication, Cognitive and Social / Emotional",
            format_area_summary(
                {"communication": True, "cognitive": True, "social_emotional": True}
            ),
        )

    def test_half_probability_is_an_inclusive_positive_concern_decision(self) -> None:
        result = fake_service((0.499, 0.50, 0.80)).predict(
            submission_from_fixture("synthetic_complete_assessment.json")
        )

        self.assertEqual((False, True, True), tuple(item.decision for item in result.domain_predictions))
        self.assertTrue(result.overall_concern)
        self.assertEqual(APPROVED_DECISION_THRESHOLD, result.threshold)

    def test_incomplete_submission_does_not_call_any_model(self) -> None:
        service = fake_service((0.9, 0.9, 0.9))
        result = service.predict(
            submission_from_fixture("synthetic_incomplete_assessment.json")
        )

        self.assertIs(result.status, AssessmentStatus.INCOMPLETE)
        self.assertEqual((), result.domain_predictions)
        self.assertIsNone(result.overall_concern)
        self.assertTrue(
            all(model.calls == 0 for model in service.models.values())
        )

    def test_three_logistic_models_can_fit_and_predict_from_synthetic_contract_data(self) -> None:
        service = fit_demo_prediction_service(synthetic_training_records())
        result = service.predict(
            submission_from_fixture("synthetic_complete_assessment.json")
        )

        self.assertEqual(REGULARIZED_LOGISTIC.version, result.model_version)
        self.assertEqual(DOMAIN_IDS, tuple(item.domain for item in result.domain_predictions))
        self.assertTrue(
            all(0.0 <= item.probability <= 1.0 for item in result.domain_predictions)
        )

    def test_both_candidates_predict_from_the_same_encoded_assessment(self) -> None:
        candidates = fit_demo_candidate_predictors(synthetic_training_records())
        encoded = encode_submission(
            load_questionnaire_catalog(),
            submission_from_fixture("synthetic_complete_assessment.json"),
        )

        self.assertEqual(
            (REGULARIZED_LOGISTIC.name, REGULARIZED_EXTRA_TREES.name),
            tuple(candidate.candidate_name for candidate in candidates),
        )
        results = tuple(
            candidate.service.predict_encoded(encoded) for candidate in candidates
        )
        self.assertEqual(
            (REGULARIZED_LOGISTIC.version, REGULARIZED_EXTRA_TREES.version),
            tuple(result.model_version for result in results),
        )
        self.assertEqual(
            (
                DECISION_THRESHOLD_APPROVAL_STATUS,
                EXPERIMENTAL_DECISION_THRESHOLD_STATUS,
            ),
            tuple(candidate.threshold_status for candidate in candidates),
        )
        self.assertEqual(
            (DECISION_THRESHOLD_VERSION, None),
            tuple(result.threshold_version for result in results),
        )
        self.assertIsNotNone(candidates[0].threshold_approval_statement)
        self.assertIsNone(candidates[1].threshold_approval_statement)
        for result in results:
            self.assertEqual(
                DOMAIN_IDS,
                tuple(item.domain for item in result.domain_predictions),
            )
            self.assertTrue(
                all(0.0 <= item.probability <= 1.0 for item in result.domain_predictions)
            )


if __name__ == "__main__":
    unittest.main()
