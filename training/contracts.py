"""Immutable Phase-1A data contracts for the processed modelling export."""

from __future__ import annotations

from pathlib import Path


DATASET_RELATIVE_PATH = Path(
    "data/processed/survey-ml-cross-checkpoint-classification (2).csv"
)
EXPECTED_DATASET_SHA256 = (
    "96521381C630E7AF4A551007C55F5703DDAAAEEDFCA3C3A09435A8082688F4D5"
)
EXPECTED_ROW_COUNT = 507
EXPECTED_DEVELOPMENT_PARTICIPANT_COUNT = 405
EXPECTED_FINAL_TEST_PARTICIPANT_COUNT = 102
EXPECTED_SPLIT_FINGERPRINT_SHA256 = (
    "490DBB5B94F298C4A196ED2320DEBF73C2FBE5DFBAF44D498E1814DF1524A5CF"
)

FEATURE_SCHEMA_VERSION = "primary-t0-16-v1"
TARGET_DEFINITION_VERSION = "followup-domain-score-le-50-v1"
DEVELOPMENT_SPLIT_SCHEMA_VERSION = "participant-development-final-test-v1"
LOGISTIC_MODEL_CONFIGURATION_VERSION = "logistic-l2-balanced-v1"
EXTRA_TREES_MODEL_CONFIGURATION_VERSION = "extra-trees-balanced-leaf5-v1"
DECISION_THRESHOLD_VERSION = "domain-probability-threshold-v1"
APPROVED_DECISION_THRESHOLD = 0.50
DECISION_THRESHOLD_APPROVAL_STATUS = "consultant_pediatrician_reviewed_and_approved"
EXPERIMENTAL_DECISION_THRESHOLD_STATUS = "experimental_not_approved_for_model"
DECISION_THRESHOLD_APPROVAL_STATEMENT = (
    "The operational predicted-probability threshold of 0.50 was reviewed and "
    "approved by the consultant pediatrician for this model version."
)

PARTICIPANT_COLUMN = "participant_number"
ASSESSMENT_ID_COLUMN = "baseline_assessment_id"
BASELINE_CHECKPOINT_COLUMN = "baseline_checkpoint_id"
FOLLOWUP_CHECKPOINT_COLUMN = "followup_checkpoint_id"

QUESTION_COLUMNS = (
    "communication_q1",
    "communication_q2",
    "communication_q3",
    "communication_q4",
    "communication_q5",
    "cognitive_q1",
    "cognitive_q2",
    "cognitive_q3",
    "cognitive_q4",
    "cognitive_q5",
    "social_emotional_q1",
    "social_emotional_q2",
    "social_emotional_q3",
    "social_emotional_q4",
    "social_emotional_q5",
)

FEATURE_COLUMNS = (BASELINE_CHECKPOINT_COLUMN, *QUESTION_COLUMNS)

TARGET_COLUMNS = (
    "target_communication_concern",
    "target_cognitive_concern",
    "target_social_emotional_concern",
)
DERIVED_TARGET_COLUMN = "target_any_developmental_concern"

SCORE_TARGET_PAIRS = (
    ("followup_communication_score", "target_communication_concern"),
    ("followup_cognitive_score", "target_cognitive_concern"),
    ("followup_social_emotional_score", "target_social_emotional_concern"),
)
FOLLOWUP_SCORE_COLUMNS = tuple(score for score, _ in SCORE_TARGET_PAIRS)

CHECKPOINT_IDS = (
    "CP2M",
    "CP4M",
    "CP6M",
    "CP9M",
    "CP12M",
    "CP15M",
    "CP18M",
    "CP24M",
    "CP30M",
    "CP36M",
    "CP48M",
    "CP60M",
)

BASELINE_CHECKPOINT_ONE_HOT_COLUMNS = tuple(
    f"baseline_checkpoint_{checkpoint.lower()}" for checkpoint in CHECKPOINT_IDS
)
FOLLOWUP_CHECKPOINT_ONE_HOT_COLUMNS = tuple(
    f"followup_checkpoint_{checkpoint.lower()}" for checkpoint in CHECKPOINT_IDS
)

SEX_ONE_HOT_COLUMNS = (
    "profile_sex_male",
    "profile_sex_female",
)
PREMATURITY_ONE_HOT_COLUMNS = (
    "profile_premature_yes",
    "profile_premature_no",
    "profile_premature_not_sure",
)
BIRTH_WEIGHT_ONE_HOT_COLUMNS = (
    "profile_birth_weight_under_1_5_kg",
    "profile_birth_weight_1_5_to_1_99",
    "profile_birth_weight_2_0_to_2_49",
    "profile_birth_weight_2_5_to_2_99",
    "profile_birth_weight_3_0_to_3_49",
    "profile_birth_weight_3_5_to_3_99",
    "profile_birth_weight_4_or_more",
    "profile_birth_weight_not_sure",
)

PROFILE_CONTEXT_COLUMNS = (
    *SEX_ONE_HOT_COLUMNS,
    *PREMATURITY_ONE_HOT_COLUMNS,
    "profile_gestational_weeks",
    "profile_gestational_weeks_missing",
    *BIRTH_WEIGHT_ONE_HOT_COLUMNS,
)

TRACEABILITY_AND_PROVENANCE_COLUMNS = (
    PARTICIPANT_COLUMN,
    "reporting_method",
    ASSESSMENT_ID_COLUMN,
)

FUTURE_INFORMATION_COLUMNS = (
    FOLLOWUP_CHECKPOINT_COLUMN,
    *FOLLOWUP_CHECKPOINT_ONE_HOT_COLUMNS,
    *FOLLOWUP_SCORE_COLUMNS,
)

REDUNDANT_PRIMARY_COLUMNS = BASELINE_CHECKPOINT_ONE_HOT_COLUMNS

EXPECTED_SOURCE_COLUMNS = (
    PARTICIPANT_COLUMN,
    "reporting_method",
    ASSESSMENT_ID_COLUMN,
    BASELINE_CHECKPOINT_COLUMN,
    FOLLOWUP_CHECKPOINT_COLUMN,
    *QUESTION_COLUMNS,
    *BASELINE_CHECKPOINT_ONE_HOT_COLUMNS,
    *FOLLOWUP_CHECKPOINT_ONE_HOT_COLUMNS,
    *PROFILE_CONTEXT_COLUMNS,
    *FOLLOWUP_SCORE_COLUMNS,
    *TARGET_COLUMNS,
    DERIVED_TARGET_COLUMN,
)

PROHIBITED_FEATURE_COLUMNS = tuple(
    column for column in EXPECTED_SOURCE_COLUMNS if column not in FEATURE_COLUMNS
)

ONE_HOT_GROUPS = {
    "baseline_checkpoint": BASELINE_CHECKPOINT_ONE_HOT_COLUMNS,
    "followup_checkpoint": FOLLOWUP_CHECKPOINT_ONE_HOT_COLUMNS,
    "sex": SEX_ONE_HOT_COLUMNS,
    "prematurity": PREMATURITY_ONE_HOT_COLUMNS,
    "birth_weight": BIRTH_WEIGHT_ONE_HOT_COLUMNS,
}

SCOREABLE_RESPONSE_VALUES = frozenset({"0", "1", "2"})
BINARY_VALUES = frozenset({"0", "1"})


def repository_root() -> Path:
    """Return the repository root without depending on the process working directory."""

    return Path(__file__).resolve().parents[1]


def default_dataset_path() -> Path:
    """Return the canonical local processed-dataset path."""

    return repository_root() / DATASET_RELATIVE_PATH
