from __future__ import annotations

import json
from datetime import datetime

import asyncpg
from pydantic import BaseModel, ConfigDict


class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    name: str
    phone: str
    preferences_prompt: str
    timezone: str | None = None
    google_tokens: dict | None = None
    created_at: datetime | None = None


class UserRepository:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def get_by_id(self, user_id: int) -> User | None:
        row = await self._conn.fetchrow(
            "SELECT id, name, phone, preferences_prompt, timezone, google_tokens, created_at"
            " FROM users WHERE id = $1",
            user_id,
        )
        return User.model_validate(dict(row)) if row else None

    async def get_by_phone(self, phone: str) -> User | None:
        row = await self._conn.fetchrow(
            "SELECT id, name, phone, preferences_prompt, timezone, google_tokens, created_at"
            " FROM users WHERE phone = $1",
            phone,
        )
        return User.model_validate(dict(row)) if row else None

    async def get_timezone(self, phone: str) -> str | None:
        return await self._conn.fetchval(
            "SELECT timezone FROM users WHERE phone = $1", phone
        )

    async def create(
        self,
        phone: str,
        name: str,
        timezone: str,
        preferences_prompt: str,
    ) -> User:
        row = await self._conn.fetchrow(
            """
            INSERT INTO users (phone, name, timezone, preferences_prompt)
            VALUES ($1, $2, $3, $4)
            RETURNING id, name, phone, preferences_prompt, timezone, google_tokens, created_at
            """,
            phone, name, timezone, preferences_prompt,
        )
        return User.model_validate(dict(row))

    async def update(self, phone: str, **fields) -> User | None:
        """
        Update one or more columns for the given phone number.
        Accepted keys: name, timezone, preferences_prompt, google_tokens.
        google_tokens must be passed as a dict; it is serialised to JSON here.
        """
        if not fields:
            return await self.get_by_phone(phone)

        # google_tokens must be stored as JSONB — serialise if caller passed a dict
        if "google_tokens" in fields and isinstance(fields["google_tokens"], dict):
            fields["google_tokens"] = json.dumps(fields["google_tokens"])

        set_clauses = [f"{col} = ${i + 1}" for i, col in enumerate(fields)]
        values = list(fields.values())
        values.append(phone)

        row = await self._conn.fetchrow(
            f"""
            UPDATE users
            SET {', '.join(set_clauses)}
            WHERE phone = ${len(values)}
            RETURNING id, name, phone, preferences_prompt, timezone, google_tokens, created_at
            """,
            *values,
        )
        return User.model_validate(dict(row)) if row else None

    async def delete(self, user_id: int) -> bool:
        result = await self._conn.execute(
            "DELETE FROM users WHERE id = $1", user_id
        )
        return int(result.split()[-1]) > 0
