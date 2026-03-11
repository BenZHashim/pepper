from __future__ import annotations

from datetime import datetime

import asyncpg
from pydantic import BaseModel, ConfigDict


class Skill(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    name: str
    description: str
    content: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SkillRepository:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def get_by_name(self, phone: str, name: str) -> Skill | None:
        row = await self._conn.fetchrow(
            """
            SELECT s.id, s.user_id, s.name, s.description, s.content, s.created_at, s.updated_at
            FROM skills s
            JOIN users u ON u.id = s.user_id
            WHERE u.phone = $1 AND s.name = $2
            """,
            phone, name,
        )
        return Skill.model_validate(dict(row)) if row else None

    async def get_all_by_phone(self, phone: str) -> list[Skill]:
        rows = await self._conn.fetch(
            """
            SELECT s.id, s.user_id, s.name, s.description, s.content, s.created_at, s.updated_at
            FROM skills s
            JOIN users u ON u.id = s.user_id
            WHERE u.phone = $1
            ORDER BY s.name
            """,
            phone,
        )
        return [Skill.model_validate(dict(r)) for r in rows]

    async def delete(self, name: str, phone: str) -> bool:
        """Delete a skill by name scoped to the given phone. Returns True if deleted."""
        result = await self._conn.execute(
            """
            DELETE FROM skills
            WHERE name = $1
            AND user_id = (SELECT id FROM users WHERE phone = $2)
            """,
            name, phone,
        )
        return int(result.split()[-1]) > 0
