"""Load and validate the separate caregiver activity-suitability questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from jsonschema import Draft202012Validator, FormatChecker

from .contracts import CHECKPOINT_IDS, SAFETY_DIMENSIONS, AbilityState, SafetyContext


CATALOG_VERSION = "caregiver-activity-safety-v1"
RESPONSE_TO_STATE = MappingProxyType(
    {
        "NO_CONCERN_REPORTED": AbilityState.NO_CONCERN_REPORTED,
        "CONCERN_REPORTED": AbilityState.CONCERN_REPORTED,
        "UNKNOWN": AbilityState.UNKNOWN,
    }
)


@dataclass(frozen=True)
class SafetyQuestion:
    question_id: str
    dimension: str
    text: str


@dataclass(frozen=True)
class SafetyQuestionnaireCatalog:
    version: str
    questions: tuple[SafetyQuestion, ...]
    response_labels: Mapping[str, str]
    checkpoint_examples: Mapping[str, Mapping[str, str]]
    review: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "response_labels", MappingProxyType(dict(self.response_labels)))
        object.__setattr__(
            self,
            "checkpoint_examples",
            MappingProxyType(
                {key: MappingProxyType(dict(value)) for key, value in self.checkpoint_examples.items()}
            ),
        )
        object.__setattr__(self, "review", MappingProxyType(dict(self.review)))


def _catalog_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "safety_questionnaire" / CATALOG_VERSION


def load_safety_questionnaire_catalog(
    catalog_path: Path | None = None,
    schema_path: Path | None = None,
) -> SafetyQuestionnaireCatalog:
    directory = _catalog_directory()
    source_path = catalog_path or directory / "questions.json"
    contract_path = schema_path or directory / "schema.json"
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
        schema = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load the safety-question catalogue: {exc}") from exc

    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(f"Invalid safety-question catalogue at {location}: {error.message}")

    questions = tuple(
        SafetyQuestion(item["question_id"], item["dimension"], item["text"])
        for item in document["questions"]
    )
    if tuple(question.dimension for question in questions) != SAFETY_DIMENSIONS:
        raise ValueError("Safety questions must cover the four dimensions in contract order.")
    response_labels = {item["code"]: item["label"] for item in document["response_options"]}
    if tuple(response_labels) != tuple(RESPONSE_TO_STATE):
        raise ValueError("Safety response options do not match the application contract.")
    if tuple(document["checkpoint_examples"]) != CHECKPOINT_IDS:
        raise ValueError("Safety examples must cover all checkpoints in contract order.")
    return SafetyQuestionnaireCatalog(
        version=document["questionnaire_version"],
        questions=questions,
        response_labels=response_labels,
        checkpoint_examples=document["checkpoint_examples"],
        review=document["review"],
    )


def safety_context_from_responses(
    catalog: SafetyQuestionnaireCatalog,
    responses: Mapping[str, str],
) -> SafetyContext:
    expected_ids = {question.question_id for question in catalog.questions}
    if set(responses) != expected_ids:
        raise ValueError("Exactly one answer is required for each safety question.")
    states: dict[str, AbilityState] = {}
    for question in catalog.questions:
        code = responses[question.question_id]
        try:
            states[question.dimension] = RESPONSE_TO_STATE[code]
        except KeyError as exc:
            raise ValueError(f"Unsupported safety response for {question.question_id}.") from exc
    return SafetyContext(**states)
