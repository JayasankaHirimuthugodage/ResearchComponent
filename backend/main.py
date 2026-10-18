"""Run only with: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"""
from backend.app import create_app
from backend.activity_runtime import ExistingCaregiverActivityRuntime
from backend.config import Settings
from backend.repositories import MongoRepository
from backend.runtime import ExistingExperimentalRuntime

settings = Settings.from_environment()
from inference.questionnaire import load_questionnaire_catalog
catalog = load_questionnaire_catalog()
experimental_runtime = ExistingExperimentalRuntime(catalog) if settings.enable_experimental_ml_runtime else None
app = create_app(
    settings=settings,
    store=MongoRepository(settings.mongodb_uri, settings.mongodb_database),
    catalog=catalog,
    prediction_provider=experimental_runtime,
    explanation_provider=experimental_runtime.explain if experimental_runtime is not None else None,
    activity_provider=ExistingCaregiverActivityRuntime(catalog) if settings.enable_experimental_ml_runtime else None,
)
