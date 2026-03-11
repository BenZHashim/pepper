from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg
from pydantic import BaseModel, ConfigDict, field_validator


class Message(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    role: str
    message_type: str
    content: Any          # parsed from JSON: dict, list, or str
    created_at: datetime
    period_start: datetime
    period_end: datetime
    source_message_ids: list[int] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def parse_content(cls, v: Any) -> Any:
        """Deserialise content if asyncpg returns it as a raw JSON string."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return v
        return v


class MessageRepository:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def add(
        self,
        user_id: int,
        role: str,
        content_json: str,
        embedding: list[float] | None,
        text_for_index: str,
    ) -> None:
        await self._conn.execute(
            """
            INSERT INTO messages
                (user_id, role, message_type, content, embedding, search_vector)
            VALUES
                ($1, $2, 'raw', $3::jsonb, $4, to_tsvector('english', $5))
            """,
            user_id, role, content_json, embedding, text_for_index,
        )

    async def get_recent(self, user_id: int, limit: int) -> list[Message]:
        rows = await self._conn.fetch(
            """
            SELECT id, user_id, role, message_type, content, created_at, period_start, period_end,
                   source_message_ids
            FROM messages
            WHERE user_id = $1 AND message_type = 'raw'
            ORDER BY period_end DESC
            LIMIT $2
            """,
            user_id, limit,
        )
        return [Message.model_validate(dict(r)) for r in rows]

    async def get_by_ids(self, ids: list[int]) -> list[Message]:
        rows = await self._conn.fetch(
            """
            SELECT id, user_id, role, message_type, content, created_at, period_start, period_end,
                   source_message_ids
            FROM messages
            WHERE id = ANY($1)
            ORDER BY period_end ASC
            """,
            ids,
        )
        return [Message.model_validate(dict(r)) for r in rows]

    async def delete_all(self, user_id: int) -> None:
        await self._conn.execute(
            "DELETE FROM messages WHERE user_id = $1", user_id
        )
