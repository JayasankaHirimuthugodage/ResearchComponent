"""Load, validate and encode the versioned questionnaire catalogue."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from inference.contracts import (
    AssessmentStatus,
    DOMAIN_IDS,
    QuestionnaireSubmission,
    SubmissionValidationError,
)
from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS, QUESTION_COLUMNS, repository_root


QUESTIONNAIRE_SCHEMA_VERSION = "developmental-questionnaire-v3"
ML_EXPORT_SCHEMA_VERSION = "developmental-questionnaire-v3-ml-classification-v1"
CATALOG_SCHEMA_VERSION = "1.0"

EXPECTED_RESPONSE_OPTIONS = (
    ("YES_USUALLY", "Yes, usually", 2, True),
    ("SOMETIMES", "Sometimes", 1, True),
    ("NOT_YET", "Not yet", 0, True),
    ("NOT_SURE", "Not sure", None, False),
    (
        "NOT_OBSERVED",
        "I have not had an opportunity to observe this",
        None,
        False,
    ),
)
DOMAIN_ID_PREFIXES = {
    "communication": "COMM",
    "cognitive": "COG",
    "social_emotional": "SOC",
}


@dataclass(frozen=True)
class ResponseOption:
    label: str
    code: str
    numeric_value: int | None
    scorable: bool


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    display_age: str
    minimum_age_days: int
    maximum_age_days: int
    corrected_age_applies: bool


@dataclass(frozen=True)
class Question:
    question_id: str
    checkpoint_id: str
    domain: str
    order: int
    feature_column: str
    text: str


@dataclass(frozen=True)
class QuestionnaireCatalog:
    questionnaire_name: str
    questionnaire_schema_version: str
    ml_export_schema_version: str
    caregiver_instruction: str
    response_options: tuple[ResponseOption, ...]
    checkpoints: tuple[Checkpoint, ...]
    questions: tuple[Question, ...]

    def response_option_by_code(self) -> dict[str, ResponseOption]:
        return {option.code: option for option in self.response_options}

    def checkpoint_by_id(self) -> dict[str, Checkpoint]:
        return {checkpoint.checkpoint_id: checkpoint for checkpoint in self.checkpoints}

    def questions_for_checkpoint(self, checkpoint_id: str) -> tuple[Question, ...]:
        if checkpoint_id not in self.checkpoint_by_id():
            raise SubmissionValidationError("The selected checkpoint is not supported.")
        feature_positions = {name: index for index, name in enumerate(QUESTION_COLUMNS)}
        return tuple(
            sorted(
                (
                    question
                    for question in self.questions
                    if question.checkpoint_id == checkpoint_id
                ),
                key=lambda question: feature_positions[question.feature_column],
            )
        )


@dataclass(frozen=True)
class EncodedAssessment:
    status: AssessmentStatus
    feature_values: tuple[str, ...] | None
    incomplete_question_ids: tuple[str, ...]


def default_catalog_path() -> Path:
    return (
        repository_root()
        / "questionnaire_catalog"
        / QUESTIONNAIRE_SCHEMA_VERSION
        / "questionnaire.json"
    )


def _require_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array.")
    return value


def _validate_response_options(raw: object) -> tuple[ResponseOption, ...]:
    options: list[ResponseOption] = []
    for index, value in enumerate(_require_list(raw, "response_options")):
        item = _require_dict(value, f"response_options[{index}]")
        option = ResponseOption(
            label=str(item.get("label", "")),
            code=str(item.get("code", "")),
            numeric_value=item.get("numeric_value"),
            scorable=item.get("scorable") is True,
        )
        options.append(option)
    observed = tuple(
        (item.code, item.label, item.numeric_value, item.scorable) for item in options
    )
    if observed != EXPECTED_RESPONSE_OPTIONS:
        raise ValueError("Response options do not match the versioned scoring contract.")
    return tuple(options)


def _validate_checkpoints(raw: object) -> tuple[Checkpoint, ...]:
    checkpoints: list[Checkpoint] = []
    for index, value in enumerate(_require_list(raw, "checkpoints")):
        item = _require_dict(value, f"checkpoints[{index}]")
        minimum = item.get("minimum_age_days")
        maximum = item.get("maximum_age_days")
        corrected = item.get("corrected_age_applies")
        if not isinstance(minimum, int) or not isinstance(maximum, int):
            raise ValueError("Checkpoint day limits must be integers.")
        if minimum < 0 or maximum < minimum:
            raise ValueError("Checkpoint day limits are invalid.")
        if not isinstance(corrected, bool):
            raise ValueError("corrected_age_applies must be boolean metadata.")
        checkpoints.append(
            Checkpoint(
                checkpoint_id=str(item.get("checkpoint_id", "")),
                display_age=str(item.get("display_age", "")),
                minimum_age_days=minimum,
                maximum_age_days=maximum,
                corrected_age_applies=corrected,
            )
        )
    if tuple(item.checkpoint_id for item in checkpoints) != CHECKPOINT_IDS:
        raise ValueError("Catalogue checkpoints do not match the frozen feature contract.")
    if any(not item.display_age for item in checkpoints):
        raise ValueError("Every checkpoint requires display wording.")
    return tuple(checkpoints)


def _validate_questions(raw: object) -> tuple[Question, ...]:
    questions: list[Question] = []
    for index, value in enumerate(_require_list(raw, "questions")):
        item = _require_dict(value, f"questions[{index}]")
        order = item.get("order")
        if not isinstance(order, int):
            raise ValueError("Question order must be an integer.")
        questions.append(
            Question(
                question_id=str(item.get("question_id", "")),
                checkpoint_id=str(item.get("checkpoint_id", "")),
                domain=str(item.get("domain", "")),
                order=order,
                feature_column=str(item.get("feature_column", "")),
                text=str(item.get("text", "")),
            )
        )

    if len(questions) != len(CHECKPOINT_IDS) * len(QUESTION_COLUMNS):
        raise ValueError("The catalogue must contain exactly 180 questions.")
    if len({item.question_id for item in questions}) != len(questions):
        raise ValueError("Question IDs must be unique.")

    for checkpoint_id in CHECKPOINT_IDS:
        checkpoint_questions = [
            item for item in questions if item.checkpoint_id == checkpoint_id
        ]
        if len(checkpoint_questions) != len(QUESTION_COLUMNS):
            raise ValueError("Every checkpoint must contain exactly 15 questions.")
        if {item.feature_column for item in checkpoint_questions} != set(QUESTION_COLUMNS):
            raise ValueError("Each checkpoint must map exactly onto the 15 ML items.")
        for item in checkpoint_questions:
            if item.domain not in DOMAIN_IDS or item.order not in range(1, 6):
                raise ValueError("Question domain or order is invalid.")
            expected_feature = f"{item.domain}_q{item.order}"
            expected_id = (
                f"{checkpoint_id}_{DOMAIN_ID_PREFIXES[item.domain]}_Q{item.order}"
            )
            if item.feature_column != expected_feature or item.question_id != expected_id:
                raise ValueError("Question ID-to-feature mapping is inconsistent.")
            if not item.text.strip():
                raise ValueError("Every question requires non-empty wording.")
    return tuple(questions)


def load_questionnaire_catalog(path: Path | None = None) -> QuestionnaireCatalog:
    """Load the catalogue and fail closed if its version or mapping has drifted."""

    catalog_path = (path or default_catalog_path()).resolve()
    with catalog_path.open("r", encoding="utf-8") as source:
        raw = _require_dict(json.load(source), "catalogue")

    if raw.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported questionnaire catalogue schema version.")
    if raw.get("questionnaire_schema_version") != QUESTIONNAIRE_SCHEMA_VERSION:
        raise ValueError("Questionnaire version does not match the inference contract.")
    if raw.get("ml_export_schema_version") != ML_EXPORT_SCHEMA_VERSION:
        raise ValueError("ML export version does not match the inference contract.")
    if raw.get("language") != "en":
        raise ValueError("This catalogue version supports only its supplied English wording.")

    name = raw.get("questionnaire_name")
    instruction = raw.get("caregiver_instruction")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("The questionnaire name is required.")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("The caregiver instruction is required.")

    return QuestionnaireCatalog(
        questionnaire_name=name,
        questionnaire_schema_version=QUESTIONNAIRE_SCHEMA_VERSION,
        ml_export_schema_version=ML_EXPORT_SCHEMA_VERSION,
        caregiver_instruction=instruction,
        response_options=_validate_response_options(raw.get("response_options")),
        checkpoints=_validate_checkpoints(raw.get("checkpoints")),
        questions=_validate_questions(raw.get("questions")),
    )


def encode_submission(
    catalog: QuestionnaireCatalog,
    submission: QuestionnaireSubmission,
) -> EncodedAssessment:
    """Validate one submission and map it to the exact 16 model inputs."""

    if submission.schema_version != catalog.questionnaire_schema_version:
        raise SubmissionValidationError("The submission questionnaire version is unsupported.")
    questions = catalog.questions_for_checkpoint(submission.checkpoint_id)
    expected_ids = {question.question_id for question in questions}
    supplied_ids = set(submission.responses)
    missing_ids = expected_ids - supplied_ids
    extra_ids = supplied_ids - expected_ids
    if missing_ids or extra_ids:
        raise SubmissionValidationError(
            "Every checkpoint question must be supplied exactly once; "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}."
        )

    options = catalog.response_option_by_code()
    incomplete_ids: list[str] = []
    numeric_by_feature: dict[str, str] = {}
    for question in questions:
        code = submission.responses[question.question_id]
        option = options.get(code)
        if option is None:
            raise SubmissionValidationError("A response code is not supported by this version.")
        if not option.scorable:
            incomplete_ids.append(question.question_id)
            continue
        if option.numeric_value not in (0, 1, 2):
            raise RuntimeError("A scorable catalogue option has no valid numeric mapping.")
        numeric_by_feature[question.feature_column] = str(option.numeric_value)

    if incomplete_ids:
        return EncodedAssessment(
            status=AssessmentStatus.INCOMPLETE,
            feature_values=None,
            incomplete_question_ids=tuple(incomplete_ids),
        )

    feature_values = (
        submission.checkpoint_id,
        *(numeric_by_feature[column] for column in QUESTION_COLUMNS),
    )
    if len(feature_values) != len(FEATURE_COLUMNS):
        raise RuntimeError("Encoded assessment does not match the frozen feature schema.")
    return EncodedAssessment(
        status=AssessmentStatus.COMPLETE,
        feature_values=feature_values,
        incomplete_question_ids=(),
    )
