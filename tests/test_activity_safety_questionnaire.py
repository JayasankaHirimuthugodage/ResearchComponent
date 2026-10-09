from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from activity_recommendation.contracts import AbilityState, SAFETY_DIMENSIONS
from activity_recommendation.safety_questionnaire import (
    load_safety_questionnaire_catalog,
    safety_context_from_responses,
)


class SafetyQuestionnaireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_safety_questionnaire_catalog()

    def test_catalogue_has_four_separate_reviewed_questions(self) -> None:
        self.assertEqual(SAFETY_DIMENSIONS, tuple(q.dimension for q in self.catalog.questions))
        self.assertEqual("professionally_reviewed", self.catalog.review["status"])
        self.assertEqual("pediatrician", self.catalog.review["reviewer_role"])
        self.assertEqual("reported_by_researcher", self.catalog.review["evidence_basis"])
        self.assertEqual(12, len(self.catalog.checkpoint_examples))

    def test_each_response_state_maps_across_every_dimension(self) -> None:
        cases = {
            "NO_CONCERN_REPORTED": AbilityState.NO_CONCERN_REPORTED,
            "CONCERN_REPORTED": AbilityState.CONCERN_REPORTED,
            "UNKNOWN": AbilityState.UNKNOWN,
        }
        for response_code, expected_state in cases.items():
            responses = {q.question_id: "NO_CONCERN_REPORTED" for q in self.catalog.questions}
            for question in self.catalog.questions:
                with self.subTest(response=response_code, dimension=question.dimension):
                    current = dict(responses)
                    current[question.question_id] = response_code
                    context = safety_context_from_responses(self.catalog, current)
                    self.assertIs(expected_state, context.as_mapping()[question.dimension])

    def test_missing_extra_and_unknown_answers_are_rejected(self) -> None:
        responses = {q.question_id: "NO_CONCERN_REPORTED" for q in self.catalog.questions}
        missing = dict(responses)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "Exactly one answer"):
            safety_context_from_responses(self.catalog, missing)
        with self.assertRaisesRegex(ValueError, "Exactly one answer"):
            safety_context_from_responses(self.catalog, {**responses, "EXTRA": "UNKNOWN"})
        invalid = dict(responses)
        invalid[next(iter(invalid))] = "YES"
        with self.assertRaisesRegex(ValueError, "Unsupported safety response"):
            safety_context_from_responses(self.catalog, invalid)

    def test_schema_rejects_duplicate_dimension_coverage(self) -> None:
        source = Path("safety_questionnaire/caregiver-activity-safety-v1/questions.json")
        document = json.loads(source.read_text(encoding="utf-8"))
        document["questions"][1]["dimension"] = "hearing"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "four dimensions"):
                load_safety_questionnaire_catalog(catalog_path=path)


if __name__ == "__main__":
    unittest.main()
