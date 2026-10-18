"""Synthetic API-boundary tests; no MongoDB, checkpoint, or participant data."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest import TestCase

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from inference.contracts import AssessmentStatus, DomainPrediction, PredictionResult
from inference.questionnaire import load_questionnaire_catalog
from activity_recommendation.recommendation_service import RecommendationStatus


class FakeStore:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.sessions: dict[str, str] = {}
        self.profiles: dict[str, dict] = {}

    async def create_user(self, **kwargs):
        if kwargs["email"] in self.users:
            raise RuntimeError("duplicate")
        user = {"_id": kwargs["email"], "id": kwargs["email"], "email": kwargs["email"], "role": "caregiver_demo", "password_hash": kwargs["password_hash"], "active": True}
        self.users[kwargs["email"]] = user
        return {"id": user["id"], "email": user["email"], "role": user["role"]}

    async def find_user_by_email(self, email): return self.users.get(email)
    async def create_session(self, *, user_id, token_hash): self.sessions[token_hash] = user_id
    async def user_for_session(self, token_hash): return self.users.get(self.sessions.get(token_hash, ""))
    async def delete_session(self, token_hash): self.sessions.pop(token_hash, None)
    async def create_profile(self, owner_user_id, profile):
        value = {"_id": "profile-1", "owner_user_id": owner_user_id, **profile}; self.profiles["profile-1"] = value; return value
    async def profile_for_owner(self, profile_id, owner_user_id):
        value = self.profiles.get(profile_id); return value if value and value["owner_user_id"] == owner_user_id else None
    async def profiles_for_owner(self, owner_user_id): return [item for item in self.profiles.values() if item["owner_user_id"] == owner_user_id]


class CountingCompleteProvider:
    def __init__(self) -> None: self.calls = 0
    def __call__(self, encoded):
        self.calls += 1
        return PredictionResult(
            status=AssessmentStatus.COMPLETE,
            domain_predictions=(
                DomainPrediction("communication", 0.8, True),
                DomainPrediction("cognitive", 0.2, False),
                DomainPrediction("social_emotional", 0.7, True),
            ),
            overall_concern=True, incomplete_question_ids=(), model_version="synthetic", feature_schema_version="synthetic",
            questionnaire_schema_version="developmental-questionnaire-v3", threshold=None, threshold_version=None,
        )


class CountingActivityProvider:
    def __init__(self) -> None: self.calls = 0
    def __call__(self, prediction, checkpoint_id, responses, safety_context):
        self.calls += 1
        return SimpleNamespace(
            status=RecommendationStatus.READY,
            checkpoint_id=checkpoint_id,
            domains=(
                SimpleNamespace(
                    domain="communication",
                    status=RecommendationStatus.READY,
                    activities=(SimpleNamespace(activity=SimpleNamespace(title="Talk together", activity_summary="Use simple words while playing together.")),),
                ),
            ),
        )


class CountingExplanationProvider:
    def __init__(self) -> None: self.calls = 0
    def __call__(self, encoded):
        self.calls += 1
        result = PredictionResult(
            status=AssessmentStatus.COMPLETE,
            domain_predictions=(
                DomainPrediction("communication", 0.8, True),
                DomainPrediction("cognitive", 0.2, False),
                DomainPrediction("social_emotional", 0.7, True),
            ),
            overall_concern=True, incomplete_question_ids=(), model_version="synthetic", feature_schema_version="synthetic",
            questionnaire_schema_version="developmental-questionnaire-v3", threshold=None, threshold_version=None,
        )
        domain = lambda value, headline: {"domain": value, "method": "synthetic", "headline": headline, "observations": [{"question": "Synthetic question"}], "limitation": "Synthetic non-causal explanation."}
        return result, (domain("communication", "What the screening considered"), domain("cognitive", "What the cognitive screening checked"), domain("social_emotional", "What the screening considered"))


class BackendApiContractTests(TestCase):
    def setUp(self) -> None:
        self.catalog = load_questionnaire_catalog()
        self.store = FakeStore()
        self.provider = CountingCompleteProvider()
        self.explanation_provider = CountingExplanationProvider()
        self.activity_provider = CountingActivityProvider()
        settings = Settings("mongodb://unused", "unused", ("http://localhost:5173",))
        self.client = TestClient(create_app(settings=settings, store=self.store, catalog=self.catalog, prediction_provider=self.provider, explanation_provider=self.explanation_provider, activity_provider=self.activity_provider))
        self.client.post("/api/v1/auth/sign-up", json={"email": "demo@example.test", "password": "a-safe-test-password"})
        signed_in = self.client.post("/api/v1/auth/sign-in", json={"email": "demo@example.test", "password": "a-safe-test-password"})
        self.assertEqual(200, signed_in.status_code)
        self.assertEqual({"email": "demo@example.test", "role": "caregiver_demo"}, self.client.get("/api/v1/auth/me").json())
        profile = self.client.post("/api/v1/child-profiles", json={"display_name": "Synthetic child", "date_of_birth": str(date.today().replace(year=date.today().year - 1)), "sex_recorded_at_birth": "female", "premature_status": "no", "gestational_age_weeks": None, "birth_weight_range": "3.0-3.49kg", "home_languages": ["English"]})
        self.assertEqual(201, profile.status_code)
        self.profile_id = profile.json()["id"]
        questionnaire = self.client.get(f"/api/v1/child-profiles/{self.profile_id}/questionnaire").json()
        self.question_ids = [item["id"] for item in questionnaire["questions"]]
        self.scorable = next(item.code for item in self.catalog.response_options if item.scorable)
        self.unscorable = next(item.code for item in self.catalog.response_options if not item.scorable)
        safety = self.client.get("/api/v1/activity-suitability-questions")
        self.assertEqual(200, safety.status_code)
        self.safety_responses = {item["id"]: "NO_CONCERN_REPORTED" for item in safety.json()["questions"]}

    def test_incomplete_answers_do_not_call_model_and_are_caregiver_safe(self) -> None:
        responses = {question_id: self.scorable for question_id in self.question_ids}; responses[self.question_ids[0]] = self.unscorable
        response = self.client.post(f"/api/v1/child-profiles/{self.profile_id}/prediction", json={"responses": responses})
        self.assertEqual(200, response.status_code)
        self.assertEqual("incomplete", response.json()["status"])
        self.assertEqual(0, self.provider.calls)
        self.assertNotIn("probability", str(response.json()).lower())

    def test_complete_results_are_caregiver_safe_and_exclude_probabilities(self) -> None:
        response = self.client.post(f"/api/v1/child-profiles/{self.profile_id}/prediction", json={"responses": {question_id: self.scorable for question_id in self.question_ids}})
        self.assertEqual(200, response.status_code)
        self.assertEqual("complete", response.json()["status"])
        self.assertEqual(1, self.provider.calls)
        self.assertEqual(3, len(response.json()["domains"]))
        self.assertNotIn("probability", str(response.json()).lower())
        self.assertIn("not a diagnosis", response.json()["domains"][0]["message"].lower())
        self.assertIn("communication", response.json()["overall_message"].lower())
        self.assertIn("social and emotional development", response.json()["overall_message"].lower())

    def test_invalid_question_set_is_a_clear_client_error(self) -> None:
        response = self.client.post(f"/api/v1/child-profiles/{self.profile_id}/prediction", json={"responses": {self.question_ids[0]: self.scorable}})
        self.assertEqual(422, response.status_code)
        self.assertIn("every checkpoint question", response.json()["detail"].lower())

    def test_complete_explanations_are_caregiver_safe_for_all_three_domains(self) -> None:
        response = self.client.post(f"/api/v1/child-profiles/{self.profile_id}/explanation", json={"responses": {question_id: self.scorable for question_id in self.question_ids}})
        self.assertEqual(200, response.status_code)
        self.assertEqual("complete", response.json()["status"])
        self.assertEqual(1, self.explanation_provider.calls)
        self.assertEqual(["communication", "cognitive", "social_emotional"], [item["domain"] for item in response.json()["domains"]])
        payload = str(response.json()).lower()
        self.assertNotIn("probability", payload)
        self.assertNotIn("threshold", payload)
        self.assertNotIn("contribution", payload)

    def test_incomplete_explanation_never_calls_explanation_runtime(self) -> None:
        responses = {question_id: self.scorable for question_id in self.question_ids}; responses[self.question_ids[0]] = self.unscorable
        response = self.client.post(f"/api/v1/child-profiles/{self.profile_id}/explanation", json={"responses": responses})
        self.assertEqual(200, response.status_code)
        self.assertEqual("incomplete", response.json()["status"])
        self.assertEqual(0, self.explanation_provider.calls)

    def test_incomplete_assessment_never_calls_activity_retrieval(self) -> None:
        responses = {question_id: self.scorable for question_id in self.question_ids}; responses[self.question_ids[0]] = self.unscorable
        response = self.client.post(f"/api/v1/child-profiles/{self.profile_id}/activities", json={"responses": responses, "safety_responses": self.safety_responses})
        self.assertEqual(200, response.status_code)
        self.assertEqual("incomplete", response.json()["status"])
        self.assertEqual(0, self.provider.calls)
        self.assertEqual(0, self.activity_provider.calls)

    def test_activity_response_contains_caregiver_cards_only(self) -> None:
        responses = {question_id: self.scorable for question_id in self.question_ids}
        response = self.client.post(f"/api/v1/child-profiles/{self.profile_id}/activities", json={"responses": responses, "safety_responses": self.safety_responses})
        self.assertEqual(200, response.status_code)
        self.assertEqual("ready", response.json()["status"])
        self.assertEqual(1, self.provider.calls)
        self.assertEqual(1, self.activity_provider.calls)
        payload = str(response.json()).lower()
        self.assertNotIn("similarity", payload)
        self.assertNotIn("probability", payload)
        self.assertNotIn("activity_id", payload)

    def test_invalid_safety_answers_are_rejected_before_any_runtime_call(self) -> None:
        response = self.client.post(f"/api/v1/child-profiles/{self.profile_id}/activities", json={"responses": {question_id: self.scorable for question_id in self.question_ids}, "safety_responses": {"unknown": "NO_CONCERN_REPORTED"}})
        self.assertEqual(422, response.status_code)
        self.assertEqual(0, self.provider.calls)
        self.assertEqual(0, self.activity_provider.calls)
