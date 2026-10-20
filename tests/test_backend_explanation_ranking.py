from types import SimpleNamespace
from unittest import TestCase

from backend.runtime import _decision_supporting_cognitive_variations


def variation(question_id: str, change: float):
    return SimpleNamespace(question_id=question_id, probability_change=change)


class CognitiveCaregiverExplanationRankingTests(TestCase):
    def test_flagged_result_keeps_only_changes_away_from_concern(self) -> None:
        ranked = _decision_supporting_cognitive_variations(
            (
                variation("q1", -0.10),
                variation("q1", -0.25),
                variation("q2", 0.40),
                variation("q3", -0.15),
            ),
            True,
        )
        self.assertEqual(["q1", "q3"], [item.question_id for item in ranked])
        self.assertEqual([-0.25, -0.15], [item.probability_change for item in ranked])

    def test_unflagged_result_keeps_only_changes_toward_concern(self) -> None:
        ranked = _decision_supporting_cognitive_variations(
            (
                variation("q1", 0.10),
                variation("q1", 0.30),
                variation("q2", -0.50),
                variation("q3", 0.20),
            ),
            False,
        )
        self.assertEqual(["q1", "q3"], [item.question_id for item in ranked])
        self.assertEqual([0.30, 0.20], [item.probability_change for item in ranked])

    def test_no_supporting_change_returns_no_observation(self) -> None:
        self.assertEqual([], _decision_supporting_cognitive_variations((variation("q1", 0.20),), True))

