"""Local Streamlit adapter for reviewing the development prediction workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from inference.contracts import (
    AssessmentStatus,
    DOMAIN_DISPLAY_NAMES,
    QuestionnaireSubmission,
    SubmissionValidationError,
)
from inference.frozen_models import load_frozen_prediction_service
from inference.explanations import CAREGIVER_EXPLANATION_NOTICE
from inference.questionnaire import (
    QuestionnaireCatalog,
    encode_submission,
    load_questionnaire_catalog,
)


FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"
UNANSWERED = ""
RESPONSE_KEY_PREFIX = "question_response_"
SYNTHETIC_RESPONSE_PATTERN = ("YES_USUALLY", "SOMETIMES", "NOT_YET")


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


def format_area_summary(domain_decisions: Mapping[str, bool]) -> str:
    """Describe the specific domains identified by the derived OR decision."""

    identified = [
        DOMAIN_DISPLAY_NAMES[domain]
        for domain in DOMAIN_DISPLAY_NAMES
        if domain_decisions.get(domain) is True
    ]
    if not identified:
        return "No developmental areas were flagged by the current model results."
    if len(identified) == 1:
        return f"Area identified for further attention: {identified[0]}"
    joined = f"{', '.join(identified[:-1])} and {identified[-1]}"
    return f"Areas identified for further attention: {joined}"


def synthetic_responses_for_checkpoint(
    catalog: QuestionnaireCatalog,
    checkpoint_id: str,
    *,
    incomplete: bool,
) -> dict[str, str]:
    """Create repeatable fictional answers for one selected checkpoint."""

    questions = catalog.questions_for_checkpoint(checkpoint_id)
    responses = {
        question.question_id: SYNTHETIC_RESPONSE_PATTERN[index % 3]
        for index, question in enumerate(questions)
    }
    if incomplete:
        responses[questions[2].question_id] = "NOT_SURE"
    return responses


def _apply_synthetic_example(
    session_state: Any,
    catalog: QuestionnaireCatalog,
    checkpoint_id: str,
    *,
    incomplete: bool,
) -> None:
    _clear_response_state(session_state)
    responses = synthetic_responses_for_checkpoint(
        catalog,
        checkpoint_id,
        incomplete=incomplete,
    )
    for question_id, response_code in responses.items():
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
    def cached_prediction_service():
        return load_frozen_prediction_service(catalog)

    st.title("Three-month developmental prediction — local review demo")
    st.write(
        "Select a baseline checkpoint and enter the 15 baseline answers. The app uses "
        "the three frozen research models to produce separate communication, cognitive, "
        "and social-emotional concern flags."
    )
    st.caption(
        "No entered answers are written to a file, database, model artifact or network API. "
        "They exist only in this local Streamlit session."
    )

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

    st.subheader("Quick demonstration controls")
    st.caption("Synthetic examples are for local demonstration only.")
    quick_complete, quick_incomplete, clear = st.columns(3)
    if quick_complete.button("Load synthetic complete example", width="stretch"):
        _apply_synthetic_example(
            st.session_state,
            catalog,
            checkpoint_id,
            incomplete=False,
        )
    if quick_incomplete.button("Load synthetic incomplete example", width="stretch"):
        _apply_synthetic_example(
            st.session_state,
            catalog,
            checkpoint_id,
            incomplete=True,
        )
    if clear.button("Clear all answers", width="stretch"):
        _clear_response_state(st.session_state)

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

    try:
        with st.spinner("Loading the verified frozen models..."):
            service = cached_prediction_service()
            result, explanations = service.predict_and_explain_encoded(encoded)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(f"The frozen prediction models could not be loaded: {error}")
        return

    st.divider()
    st.header("Developmental monitoring result")
    model_columns = st.columns(3)
    for model_column, prediction in zip(
        model_columns, result.domain_predictions, strict=True
    ):
        with model_column:
            st.subheader(DOMAIN_DISPLAY_NAMES[prediction.domain])
            if prediction.decision:
                st.warning("Concern flag")
            else:
                st.success("No concern flag")

    area_summary = format_area_summary(
        {prediction.domain: prediction.decision for prediction in result.domain_predictions}
    )
    if result.overall_concern:
        st.warning(area_summary)
    else:
        st.success(area_summary)
    st.caption("Screening for possible concerns")

    st.subheader("What influenced these results")
    for explanation in explanations:
        st.markdown(f"**{DOMAIN_DISPLAY_NAMES[explanation.domain]}**")
        st.markdown(explanation.caregiver_message)
    st.info(CAREGIVER_EXPLANATION_NOTICE)

    with st.expander("Research/reviewer technical details"):
        st.caption(
            "The probabilities and thresholds below are for technical review and should "
            "not be shown as clinical measurements or diagnoses."
        )
        for prediction in result.domain_predictions:
            st.markdown(f"**{DOMAIN_DISPLAY_NAMES[prediction.domain]}**")
            st.write(f"Calibrated probability: `{prediction.probability:.4f}`")
            st.write(f"Frozen threshold: `{prediction.threshold:.6f}` (inclusive)")
            st.write(f"Model version: `{prediction.model_version}`")
        st.divider()
        st.markdown("**Local feature contributions**")
        st.caption(
            "Positive values move the calibrated model output toward a concern flag; "
            "negative values move it away. Values are model calculations, not clinical scores."
        )
        for explanation in explanations:
            st.markdown(f"**{DOMAIN_DISPLAY_NAMES[explanation.domain]}**")
            st.write(f"Method: `{explanation.method}`")
            contribution_rows = [
                {
                    "Feature": item.display_name,
                    "Submitted value": item.input_value_label,
                    "Direction": item.direction.replace("_", " "),
                    "Contribution": round(item.contribution, 6),
                }
                for item in explanation.contributions[:5]
            ]
            st.dataframe(contribution_rows, width="stretch", hide_index=True)
            st.caption(explanation.limitation)
        st.write(f"Model release: `{result.model_version}`")
        st.write(f"Feature schema version: `{result.feature_schema_version}`")
        st.write(
            f"Questionnaire schema version: `{result.questionnaire_schema_version}`"
        )
        st.write(
            "Inference uses checksum-verified frozen artifacts. No model is fitted or "
            "updated from the submitted answers."
        )


if __name__ == "__main__":
    main()
