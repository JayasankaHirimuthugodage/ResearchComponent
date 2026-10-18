"""Asynchronous PyMongo adapter; child profiles are always queried by owner."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any
from bson import ObjectId
from pymongo import ASCENDING, AsyncMongoClient

class MongoRepository:
    def __init__(self, uri: str, database_name: str) -> None:
        self.client = AsyncMongoClient(uri); self.db = self.client[database_name]
    async def check_connection(self) -> bool:
        await self.client.admin.command("ping")
        return True
    async def ensure_indexes(self) -> None:
        await self.db.users.create_index("email", unique=True); await self.db.child_profiles.create_index([("owner_user_id", ASCENDING)]); await self.db.sessions.create_index("expires_at", expireAfterSeconds=0)
    async def create_user(self, **kwargs) -> dict[str, Any]:
        result = await self.db.users.insert_one({**kwargs, "role": "caregiver_demo", "active": True, "created_at": datetime.now(timezone.utc)})
        return {"id": str(result.inserted_id), "email": kwargs["email"], "role": "caregiver_demo"}
    async def find_user_by_email(self, email: str): return await self.db.users.find_one({"email": email, "active": True})
    async def create_session(self, *, user_id: str, token_hash: str) -> None: await self.db.sessions.insert_one({"user_id": user_id, "token_hash": token_hash, "expires_at": datetime.now(timezone.utc)+timedelta(hours=8)})
    async def user_for_session(self, token_hash: str):
        session = await self.db.sessions.find_one({"token_hash": token_hash})
        return await self.db.users.find_one({"_id": ObjectId(session["user_id"]), "active": True}) if session else None
    async def delete_session(self, token_hash: str) -> None: await self.db.sessions.delete_one({"token_hash": token_hash})
    async def create_profile(self, owner_user_id: str, profile: dict[str, Any]):
        value = {**profile, "owner_user_id": owner_user_id, "created_at": datetime.now(timezone.utc)}; result = await self.db.child_profiles.insert_one(value); value["_id"] = result.inserted_id; return value
    async def profile_for_owner(self, profile_id: str, owner_user_id: str): return await self.db.child_profiles.find_one({"_id": ObjectId(profile_id), "owner_user_id": owner_user_id}) if ObjectId.is_valid(profile_id) else None
    async def profiles_for_owner(self, owner_user_id: str): return [item async for item in self.db.child_profiles.find({"owner_user_id": owner_user_id})]
