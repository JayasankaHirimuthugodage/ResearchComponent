"""Local Streamlit adapter for reviewing the development prediction workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference.contracts import (
    AssessmentStatus,
    DOMAIN_DISPLAY_NAMES,
    QuestionnaireSubmission,
    SubmissionValidationError,
)
from inference.questionnaire import encode_submission, load_questionnaire_catalog
from training.contracts import (
    DECISION_THRESHOLD_APPROVAL_STATEMENT,
    DECISION_THRESHOLD_APPROVAL_STATUS,
)
from training.models.demo_fitting import fit_development_demo_predictor


FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"
UNANSWERED = ""
RESPONSE_KEY_PREFIX = "question_response_"


def load_synthetic_fixture(filename: str) -> dict[str, Any]:
    """Load only a repository-owned, explicitly synthetic demonstration fixture."""

    path = FIXTURE_DIRECTORY / filename
    with path.open("r", encoding="utf-8") as source:
        fixture = json.load(source)
    if not isinstance(fixture, dict) or fixture.get("fixture_type") != "synthetic_demonstration_only":
        raise ValueError("The selected file is not an approved synthetic demo fixture.")
    if not isinstance(fixture.get("responses"), dict):
        raise ValueError("The synthetic fixture has no response mapping.")
    return fixture


def _clear_response_state(session_state: Any) -> None:
    for key in tuple(session_state):
        if str(key).startswith(RESPONSE_KEY_PREFIX):
            del session_state[key]


def _apply_fixture(session_state: Any, filename: str) -> None:
    fixture = load_synthetic_fixture(filename)
    _clear_response_state(session_state)
    session_state["checkpoint_selector"] = fixture["checkpoint_id"]
    for question_id, response_code in fixture["responses"].items():
        session_state[f"{RESPONSE_KEY_PREFIX}{question_id}"] = response_code


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Development prediction review",
        page_icon="🧭",
        layout="wide",
    )
    catalog = load_questionnaire_catalog()

    @st.cache_resource(show_spinner=False)
    def cached_demo_predictor():
        return fit_development_demo_predictor()

    st.title("Three-month developmental prediction — local review demo")
    st.write(
        "Select a baseline checkpoint and enter the 15 baseline answers. The app fits "
        "three provisional logistic classifiers in memory, then shows one positive-concern "
        "probability and threshold decision for each future domain."
    )
    st.caption(
        "No entered answers are written to a file, database, model artifact or network API. "
        "They exist only in this local Streamlit session."
    )

    st.subheader("Quick demonstration controls")
    quick_complete, quick_incomplete, clear = st.columns(3)
    if quick_complete.button("Load synthetic complete example", use_container_width=True):
        _apply_fixture(st.session_state, "synthetic_complete_assessment.json")
    if quick_incomplete.button("Load synthetic incomplete example", use_container_width=True):
        _apply_fixture(st.session_state, "synthetic_incomplete_assessment.json")
    if clear.button("Clear all answers", use_container_width=True):
        _clear_response_state(st.session_state)

    checkpoint_map = catalog.checkpoint_by_id()
    checkpoint_id = st.selectbox(
        "Baseline checkpoint",
        options=tuple(checkpoint_map),
        format_func=lambda value: (
            f"{value} — {checkpoint_map[value].display_age} "
            f"({checkpoint_map[value].minimum_age_days}–"
            f"{checkpoint_map[value].maximum_age_days} days)"
        ),
        key="checkpoint_selector",
    )
    st.caption(
        "This demo does not calculate age or choose a checkpoint. The checkpoint must "
        "already have been determined by the separately validated routing process."
    )
    st.info(catalog.caregiver_instruction)

    options = catalog.response_option_by_code()
    response_codes = (UNANSWERED, *(option.code for option in catalog.response_options))

    def response_label(code: str) -> str:
        return "Choose an answer" if code == UNANSWERED else options[code].label

    responses: dict[str, str] = {}
    current_domain = ""
    domain_number = 0
    for question in catalog.questions_for_checkpoint(checkpoint_id):
        if question.domain != current_domain:
            current_domain = question.domain
            domain_number = 0
            st.subheader(DOMAIN_DISPLAY_NAMES[current_domain])
        domain_number += 1
        st.markdown(f"**{domain_number}. {question.text}**")
        responses[question.question_id] = st.selectbox(
            "Answer",
            options=response_codes,
            format_func=response_label,
            key=f"{RESPONSE_KEY_PREFIX}{question.question_id}",
            label_visibility="collapsed",
        )

    if not st.button("Generate development prediction", type="primary"):
        return

    unanswered = [question_id for question_id, code in responses.items() if not code]
    if unanswered:
        st.error(
            f"Answer all 15 questions before continuing. {len(unanswered)} answer(s) are missing."
        )
        return

    submission = QuestionnaireSubmission(
        schema_version=catalog.questionnaire_schema_version,
        checkpoint_id=checkpoint_id,
        responses=responses,
    )
    try:
        encoded = encode_submission(catalog, submission)
    except SubmissionValidationError as error:
        st.error(str(error))
        return

    if encoded.status is AssessmentStatus.INCOMPLETE:
        st.warning(
            "Assessment status: INCOMPLETE. At least one answer is 'Not sure' or "
            "'Not observed', so no ML probability or decision is produced."
        )
        return

    with st.spinner("Fitting the development-only models in memory..."):
        fitted = cached_demo_predictor()
        result = fitted.service.predict(submission)

    st.divider()
    st.header("Prediction output")
    columns = st.columns(3)
    for column, prediction in zip(columns, result.domain_predictions, strict=True):
        decision_text = (
            "Positive concern flag" if prediction.decision else "No positive concern flag"
        )
        column.metric(
            DOMAIN_DISPLAY_NAMES[prediction.domain],
            f"{prediction.probability:.1%}",
        )
        column.write(f"Decision: **{decision_text}**")

    if result.overall_concern:
        st.warning(
            "Derived overall decision: POSITIVE CONCERN FLAG because at least one "
            "domain probability is at or above 0.50."
        )
    else:
        st.success(
            "Derived overall decision: NO POSITIVE CONCERN FLAG because all three "
            "domain probabilities are below 0.50."
        )
    st.caption("Screening for possible concerns")

    with st.expander("Model and contract traceability"):
        st.write(f"Model version: `{result.model_version}`")
        st.write(f"Feature schema version: `{result.feature_schema_version}`")
        st.write(f"Questionnaire schema version: `{result.questionnaire_schema_version}`")
        st.write(f"Threshold version: `{result.threshold_version}`")
        st.write(f"Probability threshold: `{result.threshold:.2f}` (inclusive)")
        st.write(f"Threshold status: `{DECISION_THRESHOLD_APPROVAL_STATUS}`")
        st.write(DECISION_THRESHOLD_APPROVAL_STATEMENT)
        st.write(
            f"Fit scope: `{fitted.provenance.fit_scope}`; "
            f"development participants: `{fitted.provenance.development_participants}`; "
            f"final test evaluated: `{fitted.provenance.final_test_evaluated}`; "
            f"artifact persisted: `{fitted.provenance.artifact_persisted}`."
        )


if __name__ == "__main__":
    main()
