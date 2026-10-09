"""Build traceable semantic queries from observed questionnaire responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from inference.questionnaire import QuestionnaireCatalog


SUPPORTED_NEED_RESPONSES = ("NOT_YET", "SOMETIMES")


@dataclass(frozen=True)
class SkillQuery:
    domain: str
    text: str
    question_ids: tuple[str, ...]
    response_codes: tuple[str, ...]


def build_skill_query(
    catalog: QuestionnaireCatalog,
    checkpoint_id: str,
    domain: str,
    responses: Mapping[str, str],
) -> SkillQuery | None:
    """Use exact approved wording; NOT_YET items precede SOMETIMES items."""

    checkpoint_questions = catalog.questions_for_checkpoint(checkpoint_id)
    expected_ids = {question.question_id for question in checkpoint_questions}
    if set(responses) != expected_ids:
        raise ValueError("Skill-query responses must contain the exact checkpoint questions.")
    relevant = [
        question
        for response_code in SUPPORTED_NEED_RESPONSES
        for question in checkpoint_questions
        if question.domain == domain and responses[question.question_id] == response_code
    ]
    if not relevant:
        return None
    response_codes = tuple(responses[question.question_id] for question in relevant)
    return SkillQuery(
        domain=domain,
        text="Developmental activities related to: "
        + " ".join(question.text.strip() for question in relevant),
        question_ids=tuple(question.question_id for question in relevant),
        response_codes=response_codes,
    )
