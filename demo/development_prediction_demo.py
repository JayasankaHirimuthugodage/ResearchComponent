"""Local Streamlit adapter for reviewing the development prediction workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from activity_recommendation.catalogue import load_activity_catalogue
from activity_recommendation.embeddings import (
    SentenceTransformerEncoder,
    load_embedding_artifacts,
)
from activity_recommendation.recommendation_service import (
    ActivityRecommendationService,
    RecommendationStatus,
)
from activity_recommendation.safety_questionnaire import (
    load_safety_questionnaire_catalog,
    safety_context_from_responses,
)
from demo.activity_presentation import (
    caregiver_activity_content,
    reviewer_activity_provenance,
)
from inference.contracts import (
    AssessmentStatus,
    DOMAIN_DISPLAY_NAMES,
    QuestionnaireSubmission,
    SubmissionValidationError,
)
from inference.frozen_models import load_frozen_prediction_service
from inference.experimental_release import EXPERIMENTAL_KV_CACHE_COMPOSITE_RELEASE
from inference.tabiclv2_kv_cache_runtime import (
    CACHE_RUNTIME_AUTHORIZATION,
    build_kv_cache_experimental_runtime_service,
)
from inference.explanations import CAREGIVER_EXPLANATION_NOTICE
from inference.questionnaire import (
    QuestionnaireCatalog,
    encode_submission,
    load_questionnaire_catalog,
)
from inference.tabiclv2_explanations import (
    TabICLv2ExplanationStatus,
    analyze_tabiclv2_cognitive_model_behaviour,
)


FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"
UNANSWERED = ""
RESPONSE_KEY_PREFIX = "question_response_"
SAFETY_RESPONSE_KEY_PREFIX = "safety_response_"
SYNTHETIC_RESPONSE_PATTERN = ("YES_USUALLY", "SOMETIMES", "NOT_YET")
HISTORICAL_RELEASE_OPTION = "Historical frozen release"
EXPERIMENTAL_RELEASE_OPTION = "Experimental, development-only TabICLv2 KV-cache release — not independently validated or clinically deployed"
DEFAULT_RELEASE_OPTION = EXPERIMENTAL_RELEASE_OPTION
ACTIVITY_ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "artifacts" / "activity-embeddings" / "0.4.0"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL_COMPOSITE_MANIFEST = PROJECT_ROOT / "models" / "experimental-three-domain-tabiclv2-kv-cache-v1" / "release-manifest.json"
EXPERIMENTAL_CHECKPOINT = PROJECT_ROOT / ".venv-tabiclv2" / "hf-cache" / "hub" / "models--jingang--TabICL" / "snapshots" / "4dcd344ece2c00be9e831fdd35bed57b5ad83e19" / "tabicl-classifier-v2-20260212.ckpt"


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
        if str(key).startswith((RESPONSE_KEY_PREFIX, SAFETY_RESPONSE_KEY_PREFIX)):
            del session_state[key]
    session_state.pop("prediction_requested", None)


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


def experimental_explanation_status(encoded) -> TabICLv2ExplanationStatus:
    """UI-only status; it does not construct or query an experimental model."""
    if encoded.status is AssessmentStatus.INCOMPLETE:
        return TabICLv2ExplanationStatus.NOT_APPLICABLE_INCOMPLETE
    return TabICLv2ExplanationStatus.AVAILABLE_LOCAL_MODEL_BEHAVIOUR


def _experimental_explanation_key(release_option: str, encoded, model_version: str) -> tuple[object, ...]:
    """Bind an opt-in analysis to the exact submitted answers and selected release."""
    return (release_option, model_version, encoded.feature_values)


def _response_value_label(value: int) -> str:
    return {0: "Not yet", 1: "Sometimes", 2: "Yes, usually"}[value]


def caregiver_domain_message(decision: bool) -> str:
    """Return a non-diagnostic interpretation for one frozen domain decision."""

    if decision:
        return "This screening result suggests that additional monitoring may be helpful. It is not a diagnosis."
    return "This screening result did not flag this area. It does not rule out developmental needs."


def selected_prediction_service(
    release_option: str,
    historical_factory,
    experimental_factory,
):
    """Invoke only the deliberately selected release factory after validation."""
    if release_option == HISTORICAL_RELEASE_OPTION:
        return historical_factory(), False
    if release_option == EXPERIMENTAL_RELEASE_OPTION:
        return experimental_factory(), True
    raise ValueError("Unknown research model release selection.")


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Development prediction review",
        page_icon="🧭",
        layout="wide",
    )
    catalog = load_questionnaire_catalog()
    safety_catalog = load_safety_questionnaire_catalog()

    @st.cache_resource(show_spinner=False)
    def cached_prediction_service():
        return load_frozen_prediction_service(catalog)

    @st.cache_resource(show_spinner=False)
    def cached_experimental_prediction_service():
        return build_kv_cache_experimental_runtime_service(
            catalog,
            repository_root=PROJECT_ROOT,
            composite_manifest=EXPERIMENTAL_COMPOSITE_MANIFEST,
            checkpoint=EXPERIMENTAL_CHECKPOINT,
            execution_authorization=CACHE_RUNTIME_AUTHORIZATION,
        )

    @st.cache_resource(show_spinner=False)
    def cached_activity_recommendation_service():
        activity_catalogue = load_activity_catalogue()
        artifacts = load_embedding_artifacts(
            activity_catalogue,
            ACTIVITY_ARTIFACT_DIRECTORY,
        )
        return ActivityRecommendationService(
            catalog,
            activity_catalogue,
            artifacts,
            SentenceTransformerEncoder(),
        )

    st.title("Three-month developmental prediction — local review demo")
    st.write(
        "Select a baseline checkpoint and enter the 15 baseline answers. The app uses "
        "the selected three-domain research release to produce separate communication, cognitive, "
        "and social-emotional concern flags."
    )
    st.caption(
        "No entered answers are written to a file, database, model artifact or network API. "
        "They exist only in this local Streamlit session."
    )
    release_option = st.radio(
        "Research model release",
        options=(EXPERIMENTAL_RELEASE_OPTION, HISTORICAL_RELEASE_OPTION),
        index=0,
        help=(
            "The experimental option is development-only and uses the verified protected "
            "Cognitive KV-cache only after a complete assessment is submitted. The historical "
            "frozen release remains available as a rollback option."
        ),
    )
    if release_option == EXPERIMENTAL_RELEASE_OPTION:
        st.warning(
            "Experimental TabICLv2 selection is development-only. It predicts questionnaire-defined "
            "follow-up concerns, not clinical diagnoses."
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
            st.subheader(f"{DOMAIN_DISPLAY_NAMES[current_domain]} questions")
        domain_number += 1
        st.markdown(f"**{domain_number}. {question.text}**")
        responses[question.question_id] = st.selectbox(
            "Answer",
            options=response_codes,
            format_func=response_label,
            key=f"{RESPONSE_KEY_PREFIX}{question.question_id}",
            label_visibility="collapsed",
        )

    if st.button("Generate development prediction", type="primary"):
        st.session_state["prediction_requested"] = True
    if not st.session_state.get("prediction_requested", False):
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
        if release_option == EXPERIMENTAL_RELEASE_OPTION:
            st.caption(
                "Experimental Cognitive explanation status: "
                f"`{experimental_explanation_status(encoded).value}`."
            )
        st.warning(
            "Assessment status: INCOMPLETE. At least one answer is 'Not sure' or "
            "'Not observed', so no ML probability or decision is produced."
        )
        return

    try:
        with st.spinner("Loading the selected verified research models..."):
            service, experimental = selected_prediction_service(
                release_option,
                cached_prediction_service,
                cached_experimental_prediction_service,
            )
            if experimental:
                result = service.predict_encoded(encoded)
                explanations = service.explain_xgboost_encoded(encoded, result)
            else:
                result, explanations = service.predict_and_explain_encoded(encoded)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(f"The selected research prediction models could not be loaded: {error}")
        return

    st.divider()
    st.header("Your developmental screening result")
    model_columns = st.columns(3)
    for model_column, prediction in zip(
        model_columns, result.domain_predictions, strict=True
    ):
        with model_column:
            st.subheader(DOMAIN_DISPLAY_NAMES[prediction.domain])
            if prediction.decision:
                st.warning("Additional monitoring may be helpful")
            else:
                st.success("Not flagged by this screening")
            st.caption(caregiver_domain_message(prediction.decision))

    area_summary = format_area_summary(
        {prediction.domain: prediction.decision for prediction in result.domain_predictions}
    )
    if result.overall_concern:
        st.warning(area_summary)
    else:
        st.success(area_summary)
    st.caption("These are screening results for possible follow-up concerns, not diagnoses.")

    st.subheader("Understanding these results")
    if experimental:
        for explanation in explanations:
            st.markdown(f"**{DOMAIN_DISPLAY_NAMES[explanation.domain]}**")
            st.markdown(explanation.caregiver_message)
        st.info(CAREGIVER_EXPLANATION_NOTICE)
        explanation_key = _experimental_explanation_key(
            release_option, encoded, result.model_version
        )
        if st.session_state.get("experimental_cognitive_explanation_key") != explanation_key:
            st.session_state.pop("experimental_cognitive_explanation", None)
            st.session_state.pop("experimental_cognitive_explanation_key", None)
        cognitive_prediction = next(
            prediction
            for prediction in result.domain_predictions
            if prediction.domain == "cognitive"
        )
        with st.container(border=True):
            st.markdown("**Cognitive development**")
            st.write(
                caregiver_domain_message(cognitive_prediction.decision)
            )
            st.caption(
                "This screening result considers the full set of answers for this developmental "
                "checkpoint. It is not a diagnosis."
            )
            st.write(
                "If you have concerns about thinking, learning, memory, or problem-solving skills, "
                "please discuss them with an appropriate healthcare professional."
            )
        with st.expander("Researcher: Cognitive local model-behaviour analysis"):
            st.caption(
                "This optional research view evaluates 30 one-at-a-time hypothetical alternatives. "
                "It is not a caregiver explanation, feature attribution, causal explanation, or clinical advice."
            )
            request_cognitive_explanation = st.button(
                "Run 30-alternative Cognitive analysis for researcher review",
                key="researcher_cognitive_explanation_request",
            )
        if request_cognitive_explanation:
            try:
                with st.spinner("Evaluating 30 bounded hypothetical response alternatives..."):
                    st.session_state["experimental_cognitive_explanation"] = (
                        analyze_tabiclv2_cognitive_model_behaviour(
                            catalog,
                            encoded,
                            cognitive_prediction,
                            service.cognitive,
                        )
                    )
                    st.session_state["experimental_cognitive_explanation_key"] = explanation_key
            except (ValueError, RuntimeError) as error:
                st.error(f"Cognitive model-behaviour analysis could not be generated: {error}")
        cognitive_behaviour = st.session_state.get("experimental_cognitive_explanation")
        if cognitive_behaviour is not None:
            with st.expander("Researcher: Cognitive local model-behaviour results"):
                st.write(
                    f"Baseline calibrated concern probability: `{cognitive_behaviour.baseline_probability:.4f}`"
                )
                st.write(
                    f"Frozen threshold: `{cognitive_behaviour.threshold:.6f}` (inclusive)"
                )
                rows = [
                    {
                        "Question domain": DOMAIN_DISPLAY_NAMES[item.question_domain],
                        "Question": item.question_text,
                        "Submitted response": _response_value_label(item.original_response),
                        "Hypothetical response": _response_value_label(item.alternative_response),
                        "Alternative probability": round(item.alternative_probability, 6),
                        "Probability difference": round(item.probability_change, 6),
                    }
                    for item in cognitive_behaviour.variations
                ]
                st.dataframe(rows, width="stretch", hide_index=True)
                st.caption(cognitive_behaviour.limitation)
    else:
        for explanation in explanations:
            st.markdown(f"**{DOMAIN_DISPLAY_NAMES[explanation.domain]}**")
            st.markdown(explanation.caregiver_message)
        st.info(CAREGIVER_EXPLANATION_NOTICE)

    st.divider()
    st.header("Approved development activities")
    if not result.overall_concern:
        st.info("No developmental domain was flagged for targeted activity retrieval.")
    else:
        st.caption(
            "The following questions are separate from the 15 model inputs and are used "
            "only to filter activities for suitability."
        )
        st.subheader("Activity suitability questions")
        safety_responses: dict[str, str] = {}
        safety_options = ("", *safety_catalog.response_labels)

        def safety_response_label(code: str) -> str:
            return "Choose an answer" if not code else safety_catalog.response_labels[code]

        checkpoint_examples = safety_catalog.checkpoint_examples[checkpoint_id]
        for question in safety_catalog.questions:
            st.markdown(f"**{question.text}**")
            st.caption(f"Example for this checkpoint: {checkpoint_examples[question.dimension]}.")
            safety_responses[question.question_id] = st.selectbox(
                "Safety observation",
                options=safety_options,
                format_func=safety_response_label,
                key=f"{SAFETY_RESPONSE_KEY_PREFIX}{question.question_id}",
                label_visibility="collapsed",
            )

        if any(not value for value in safety_responses.values()):
            st.info("Answer all four activity suitability questions to retrieve activities.")
        elif st.button("Find suitable activities", type="secondary"):
            try:
                safety_context = safety_context_from_responses(
                    safety_catalog,
                    safety_responses,
                )
                with st.spinner("Filtering and ranking eligible activities..."):
                    recommendations = cached_activity_recommendation_service().recommend(
                        result,
                        checkpoint_id,
                        responses,
                        safety_context,
                    )
            except (FileNotFoundError, ValueError, RuntimeError) as error:
                st.error(f"Activities could not be retrieved: {error}")
            else:
                for domain_result in recommendations.domains:
                    st.subheader(DOMAIN_DISPLAY_NAMES[domain_result.domain])
                    if domain_result.status is RecommendationStatus.NO_SUPPORTED_SKILL_CONTEXT:
                        st.info(
                            "The submitted answers did not provide a NOT_YET or SOMETIMES "
                            "observation for a supported skill query in this domain."
                        )
                        continue
                    if domain_result.status is RecommendationStatus.NO_ELIGIBLE_ACTIVITIES:
                        st.warning(
                            "No activity remained after checkpoint, domain, review-status "
                            "and safety filtering."
                        )
                        continue
                    for rank, ranked in enumerate(domain_result.activities, start=1):
                        activity = ranked.activity
                        title, instructions = caregiver_activity_content(activity)
                        with st.container(border=True):
                            st.markdown(f"**{rank}. {title}**")
                            st.caption(DOMAIN_DISPLAY_NAMES[domain_result.domain])
                            st.write(instructions)
                    with st.expander(
                        f"{DOMAIN_DISPLAY_NAMES[domain_result.domain]} retrieval trace for reviewer"
                    ):
                        st.write(
                            "Question IDs used for skill context: "
                            + ", ".join(domain_result.skill_query.question_ids)
                        )
                        for ranked in domain_result.activities:
                            provenance = reviewer_activity_provenance(ranked.activity)
                            st.write(
                                f"{provenance['activity_id']}: source "
                                f"{provenance['source_org']}; similarity "
                                f"{ranked.similarity_score:.6f}"
                            )
                            st.markdown(
                                f"[Open official source material]({provenance['source_url']})"
                            )
                st.caption(
                    "This is safety-filtered semantic retrieval, not complete RAG. "
                    "No generative LLM is used."
                )

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
        if experimental:
            st.markdown("**Experimental explanation methods**")
            for explanation in explanations:
                st.markdown(f"**{DOMAIN_DISPLAY_NAMES[explanation.domain]}**")
                st.write(f"Method: `{explanation.method}`")
                st.caption(
                    "TreeSHAP contributions are calculated in the calibrated logit space; "
                    "they are not raw-margin values or causal effects."
                )
            st.markdown("**Cognitive TabICLv2**")
            st.caption(
                "Optional local model-behaviour analysis uses calibrated probability changes, "
                "not SHAP or feature attribution."
            )
        else:
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
