"""Environment-only configuration; values are intentionally never logged."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    mongodb_database: str
    allowed_origins: tuple[str, ...]
    session_cookie_name: str = "development_demo_session"
    enable_experimental_ml_runtime: bool = False

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        uri = os.getenv("MONGODB_URI", "").strip()
        if not uri:
            raise RuntimeError("MONGODB_URI must be configured before starting the API.")
        origins = tuple(item.strip() for item in os.getenv("API_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if item.strip())
        return cls(uri, os.getenv("MONGODB_DATABASE", "developmental_demo"), origins, enable_experimental_ml_runtime=os.getenv("ENABLE_EXPERIMENTAL_ML_RUNTIME", "false").lower() == "true")
