"""Versioned localhost API; routes never expose MongoDB documents or model artifacts."""
from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import Any, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.config import Settings
from backend.routing import ProfileRoutingError, route_checkpoint
from backend.security import hash_password, new_session_token, session_token_hash, verify_password
from activity_recommendation.recommendation_service import RecommendationStatus
from activity_recommendation.safety_questionnaire import (
    SafetyQuestionnaireCatalog,
    load_safety_questionnaire_catalog,
    safety_context_from_responses,
)
from inference.contracts import AssessmentStatus, QuestionnaireSubmission, SubmissionValidationError
from inference.questionnaire import QuestionnaireCatalog, encode_submission, load_questionnaire_catalog


class Store(Protocol):
    async def create_user(self, **kwargs) -> dict[str, Any] | None: ...
    async def find_user_by_email(self, email: str) -> dict[str, Any] | None: ...
    async def create_session(self, **kwargs) -> None: ...
    async def user_for_session(self, token_hash: str) -> dict[str, Any] | None: ...
    async def delete_session(self, token_hash: str) -> None: ...
    async def create_profile(self, owner_user_id: str, profile: dict[str, Any]) -> dict[str, Any]: ...
    async def profile_for_owner(self, profile_id: str, owner_user_id: str) -> dict[str, Any] | None: ...
    async def profiles_for_owner(self, owner_user_id: str) -> list[dict[str, Any]]: ...


class SignUp(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def email_format(cls, value: str) -> str:
        value = value.strip().lower()
        if value.count("@") != 1 or value.startswith("@") or value.endswith("@"):
            raise ValueError("A valid email address is required.")
        return value


class Login(SignUp): pass


class ChildProfile(BaseModel):
    display_name: str | None = Field(default=None, max_length=80)
    date_of_birth: date
    sex_recorded_at_birth: str
    premature_status: str
    gestational_age_weeks: int | None = None
    birth_weight_range: str
    home_languages: list[str] = Field(min_length=1, max_length=4)

    @field_validator("sex_recorded_at_birth")
    @classmethod
    def sex(cls, value: str) -> str:
        if value not in {"male", "female"}: raise ValueError("Sex recorded at birth is required.")
        return value

    @field_validator("premature_status")
    @classmethod
    def prematurity(cls, value: str) -> str:
        if value not in {"yes", "no", "not_sure"}: raise ValueError("Prematurity status is required.")
        return value


class Answers(BaseModel):
    responses: dict[str, str]


class ActivityRequest(Answers):
    safety_responses: dict[str, str]


def _public_profile(value: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(value["_id"]), **{key: value.get(key) for key in ("display_name", "date_of_birth", "sex_recorded_at_birth", "premature_status", "gestational_age_weeks", "birth_weight_range", "home_languages")}}


def _caregiver_domain_result(domain: str, decision: bool) -> dict[str, str | bool]:
    """Expose only a non-diagnostic, caregiver-facing screening interpretation."""
    labels = {
        "communication": "Communication",
        "cognitive": "Cognitive development",
        "social_emotional": "Social and emotional development",
    }
    if decision:
        message = "This screening result suggests that additional monitoring may be helpful. It is not a diagnosis."
        outcome = "additional_monitoring_may_help"
    else:
        message = "This screening result did not flag this area. It does not rule out developmental needs."
        outcome = "not_flagged_by_this_screening"
    return {"domain": domain, "title": labels[domain], "decision": decision, "outcome": outcome, "message": message}


def _overall_caregiver_summary(domain_predictions) -> str:
    """Mirror the non-technical Streamlit summary without exposing model internals."""
    labels = {
        "communication": "Communication",
        "cognitive": "Cognitive development",
        "social_emotional": "Social and emotional development",
    }
    identified = [labels[item.domain] for item in domain_predictions if item.decision]
    if not identified:
        return "This screening did not flag any of the three areas. It does not rule out developmental needs."
    return (
        "This screening suggests that additional monitoring may be helpful for: "
        + ", ".join(identified)
        + ". It is not a diagnosis."
    )


def _caregiver_prediction_response(result, checkpoint_id: str) -> dict[str, Any]:
    """Keep model probabilities, thresholds, artifacts, and raw inputs inside the service boundary."""
    if result.status is AssessmentStatus.INCOMPLETE:
        return {
            "status": "incomplete",
            "checkpoint_id": checkpoint_id,
            "message": "Please complete every questionnaire item with an observable response before viewing screening results.",
            "incomplete_question_ids": list(result.incomplete_question_ids),
            "domains": [],
        }
    return {
        "status": "complete",
        "checkpoint_id": checkpoint_id,
        "overall_concern": result.overall_concern,
        "overall_message": _overall_caregiver_summary(result.domain_predictions),
        "domains": [_caregiver_domain_result(item.domain, item.decision) for item in result.domain_predictions],
    }


def _caregiver_activity_response(recommendations) -> dict[str, Any]:
    """Strip retrieval scores, IDs, source links, safety traces, and reviewer metadata."""
    domain_results = []
    for domain_result in recommendations.domains:
        cards = [
            {"title": ranked.activity.title, "instructions": ranked.activity.activity_summary}
            for ranked in domain_result.activities
        ]
        domain_results.append({"domain": domain_result.domain, "status": domain_result.status.value, "activities": cards})
    messages = {
        RecommendationStatus.NO_CONCERN.value: "No developmental domain was flagged for targeted activity suggestions.",
        RecommendationStatus.NO_SUPPORTED_SKILL_CONTEXT.value: "No suitable activity suggestion could be identified from the submitted answers.",
        RecommendationStatus.NO_ELIGIBLE_ACTIVITIES.value: "No activity remained after the age, domain, and safety checks.",
    }
    return {
        "status": recommendations.status.value,
        "checkpoint_id": recommendations.checkpoint_id,
        "message": messages.get(recommendations.status.value, "Suitable activity suggestions are available below."),
        "domains": domain_results,
    }


def create_app(*, settings: Settings, store: Store, catalog: QuestionnaireCatalog | None = None, prediction_provider=None, explanation_provider=None, activity_provider=None, safety_catalog: SafetyQuestionnaireCatalog | None = None) -> FastAPI:
    catalogue = catalog or load_questionnaire_catalog()
    safety_catalogue = safety_catalog or load_safety_questionnaire_catalog()
    app = FastAPI(title="Developmental Research Prototype API", version="1.0.0", docs_url="/api/docs")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.on_event("startup")
    async def create_required_indexes() -> None:
        ensure_indexes = getattr(store, "ensure_indexes", None)
        if ensure_indexes is not None:
            await ensure_indexes()

    async def current_user(request: Request):
        session = request.cookies.get(settings.session_cookie_name)
        if not session: raise HTTPException(401, "Authentication is required.")
        user = await store.user_for_session(session_token_hash(session))
        if not user: raise HTTPException(401, "Session is invalid or expired.")
        return user

    @app.get("/health")
    async def health(): return {"status": "ok"}

    @app.post("/api/v1/auth/sign-up", status_code=201)
    async def sign_up(payload: SignUp):
        try: user = await store.create_user(email=payload.email, password_hash=hash_password(payload.password))
        except Exception as error:
            if "duplicate" in str(error).lower(): raise HTTPException(409, "An account with that email already exists.")
            raise
        return {"id": user["id"], "email": user["email"], "role": user["role"]}

    @app.post("/api/v1/auth/sign-in")
    async def sign_in(payload: Login, response: Response):
        user = await store.find_user_by_email(payload.email)
        if not user or not verify_password(payload.password, user["password_hash"]): raise HTTPException(401, "Invalid email or password.")
        token = new_session_token(); await store.create_session(user_id=str(user["_id"]), token_hash=session_token_hash(token))
        response.set_cookie(settings.session_cookie_name, token, httponly=True, samesite="strict", secure=False, max_age=8*60*60)
        return {"email": user["email"], "role": user["role"]}

    @app.post("/api/v1/auth/sign-out", status_code=204)
    async def sign_out(request: Request, response: Response):
        session = request.cookies.get(settings.session_cookie_name)
        if session: await store.delete_session(session_token_hash(session))
        response.delete_cookie(settings.session_cookie_name)

    @app.get("/api/v1/auth/me")
    async def signed_in_caregiver(user=Depends(current_user)):
        """Allow a browser refresh to restore only the authenticated demo identity."""
        return {"email": user["email"], "role": user["role"]}

    @app.post("/api/v1/child-profiles", status_code=201)
    async def create_profile(payload: ChildProfile, user=Depends(current_user)):
        try: route_checkpoint(date_of_birth=payload.date_of_birth, as_of_date=date.today(), premature_status=payload.premature_status, gestational_age_weeks=payload.gestational_age_weeks, catalog=catalogue)
        except ProfileRoutingError as error: raise HTTPException(422, str(error))
        profile = await store.create_profile(str(user["_id"]), payload.model_dump(mode="json"))
        return _public_profile(profile)

    @app.get("/api/v1/child-profiles")
    async def list_profiles(user=Depends(current_user)):
        return [_public_profile(item) for item in await store.profiles_for_owner(str(user["_id"]))]

    async def owned_profile(profile_id: str, user):
        profile = await store.profile_for_owner(profile_id, str(user["_id"]))
        if not profile: raise HTTPException(404, "Child profile was not found.")
        return profile

    @app.get("/api/v1/child-profiles/{profile_id}/questionnaire")
    async def questionnaire(profile_id: str, user=Depends(current_user)):
        profile = await owned_profile(profile_id, user)
        try: route = route_checkpoint(date_of_birth=date.fromisoformat(profile["date_of_birth"]), as_of_date=date.today(), premature_status=profile["premature_status"], gestational_age_weeks=profile.get("gestational_age_weeks"), catalog=catalogue)
        except ProfileRoutingError as error: raise HTTPException(422, str(error))
        return {"checkpoint_id": route.checkpoint_id, "chronological_age_days": route.chronological_age_days, "corrected_age_days": route.corrected_age_days, "questions": [{"id": q.question_id, "domain": q.domain, "text": q.text} for q in catalogue.questions_for_checkpoint(route.checkpoint_id)], "response_options": [{"code": o.code, "label": o.label} for o in catalogue.response_options]}

    @app.get("/api/v1/activity-suitability-questions")
    async def activity_suitability_questions(user=Depends(current_user)):
        return {
            "schema_version": safety_catalogue.version,
            "questions": [{"id": item.question_id, "dimension": item.dimension, "text": item.text} for item in safety_catalogue.questions],
            "response_options": [{"code": code, "label": label} for code, label in safety_catalogue.response_labels.items()],
        }

    @app.post("/api/v1/child-profiles/{profile_id}/prediction")
    async def prediction(profile_id: str, payload: Answers, user=Depends(current_user)):
        profile = await owned_profile(profile_id, user)
        try:
            route = route_checkpoint(date_of_birth=date.fromisoformat(profile["date_of_birth"]), as_of_date=date.today(), premature_status=profile["premature_status"], gestational_age_weeks=profile.get("gestational_age_weeks"), catalog=catalogue)
        except ProfileRoutingError as error:
            raise HTTPException(422, str(error)) from error
        submission = QuestionnaireSubmission(catalogue.questionnaire_schema_version, route.checkpoint_id, payload.responses)
        try:
            encoded = encode_submission(catalogue, submission)
        except SubmissionValidationError as error:
            raise HTTPException(422, str(error)) from error
        if encoded.status is AssessmentStatus.INCOMPLETE:
            # This branch deliberately does not call an ML model.
            return _caregiver_prediction_response(
                SimpleNamespace(
                    status=AssessmentStatus.INCOMPLETE,
                    incomplete_question_ids=encoded.incomplete_question_ids,
                ),
                route.checkpoint_id,
            )
        if prediction_provider is None:
            raise HTTPException(503, "Prediction runtime is not enabled.")
        result = await asyncio.to_thread(prediction_provider, encoded)
        return _caregiver_prediction_response(result, route.checkpoint_id)

    @app.post("/api/v1/child-profiles/{profile_id}/explanation")
    async def explanation(profile_id: str, payload: Answers, user=Depends(current_user)):
        """Provide approved local explanation summaries without model internals."""
        profile = await owned_profile(profile_id, user)
        try:
            route = route_checkpoint(date_of_birth=date.fromisoformat(profile["date_of_birth"]), as_of_date=date.today(), premature_status=profile["premature_status"], gestational_age_weeks=profile.get("gestational_age_weeks"), catalog=catalogue)
            encoded = encode_submission(catalogue, QuestionnaireSubmission(catalogue.questionnaire_schema_version, route.checkpoint_id, payload.responses))
        except (ProfileRoutingError, SubmissionValidationError) as error:
            raise HTTPException(422, str(error)) from error
        if encoded.status is AssessmentStatus.INCOMPLETE:
            return {
                "status": "incomplete",
                "checkpoint_id": route.checkpoint_id,
                "message": "Please complete every questionnaire item with an observable response before viewing this explanation.",
                "incomplete_question_ids": list(encoded.incomplete_question_ids),
                "domains": [],
            }
        if explanation_provider is None:
            raise HTTPException(503, "Caregiver explanation runtime is not enabled.")
        result, domains = await asyncio.to_thread(explanation_provider, encoded)
        if result.status is not AssessmentStatus.COMPLETE:
            raise RuntimeError("Explanation runtime returned an invalid complete-assessment result.")
        return {
            "status": "complete",
            "checkpoint_id": route.checkpoint_id,
            "notice": "All answers were considered together. The points highlighted here help explain the screening summary. They are not a diagnosis or medical advice.",
            "domains": list(domains),
        }

    @app.post("/api/v1/child-profiles/{profile_id}/activities")
    async def activities(profile_id: str, payload: ActivityRequest, user=Depends(current_user)):
        profile = await owned_profile(profile_id, user)
        try:
            route = route_checkpoint(date_of_birth=date.fromisoformat(profile["date_of_birth"]), as_of_date=date.today(), premature_status=profile["premature_status"], gestational_age_weeks=profile.get("gestational_age_weeks"), catalog=catalogue)
            safety_context = safety_context_from_responses(safety_catalogue, payload.safety_responses)
            encoded = encode_submission(catalogue, QuestionnaireSubmission(catalogue.questionnaire_schema_version, route.checkpoint_id, payload.responses))
        except (ProfileRoutingError, SubmissionValidationError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
        if encoded.status is AssessmentStatus.INCOMPLETE:
            return {
                "status": "incomplete",
                "checkpoint_id": route.checkpoint_id,
                "message": "Please complete every questionnaire item with an observable response before requesting activities.",
                "incomplete_question_ids": list(encoded.incomplete_question_ids),
                "domains": [],
            }
        if prediction_provider is None or activity_provider is None:
            raise HTTPException(503, "Activity recommendations are not enabled.")
        result = await asyncio.to_thread(prediction_provider, encoded)
        recommendations = await asyncio.to_thread(activity_provider, result, route.checkpoint_id, payload.responses, safety_context)
        return _caregiver_activity_response(recommendations)
    return app
