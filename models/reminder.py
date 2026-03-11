from __future__ import annotations

from datetime import datetime

import asyncpg
from pydantic import BaseModel, ConfigDict


class Reminder(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    message: str
    remind_at: datetime
    recurrence: str | None = None
    fired: bool = False
    google_event_id: str | None = None
    created_at: datetime | None = None
    # Joined from users — only populated by get_due() and get_by_user_phone()
    phone: str | None = None


class ReminderRepository:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def get_by_id(self, reminder_id: int) -> Reminder | None:
        row = await self._conn.fetchrow(
            "SELECT id, user_id, message, remind_at, recurrence, fired, google_event_id, created_at"
            " FROM reminders WHERE id = $1",
            reminder_id,
        )
        return Reminder.model_validate(dict(row)) if row else None

    async def get_by_user_phone(
        self, phone: str, include_fired: bool = False
    ) -> list[Reminder]:
        query = """
            SELECT r.id, r.user_id, r.message, r.remind_at, r.recurrence,
                   r.fired, r.google_event_id, r.created_at, u.phone
            FROM reminders r
            JOIN users u ON u.id = r.user_id
            WHERE u.phone = $1
        """
        if not include_fired:
            query += " AND r.fired = FALSE"
        query += " ORDER BY r.remind_at"
        rows = await self._conn.fetch(query, phone)
        return [Reminder.model_validate(dict(r)) for r in rows]

    async def get_due(self, now: datetime) -> list[Reminder]:
        """SELECT FOR UPDATE — must be called inside a transaction."""
        rows = await self._conn.fetch(
            """
            SELECT r.id, r.user_id, r.message, r.remind_at, r.recurrence,
                   r.fired, r.google_event_id, r.created_at, u.phone
            FROM reminders r
            JOIN users u ON u.id = r.user_id
            WHERE r.fired = FALSE AND r.remind_at <= $1
            FOR UPDATE OF r
            """,
            now,
        )
        return [Reminder.model_validate(dict(r)) for r in rows]

    async def create(
        self,
        user_id: int,
        message: str,
        remind_at: datetime,
        recurrence: str | None,
    ) -> Reminder:
        row = await self._conn.fetchrow(
            """
            INSERT INTO reminders (user_id, message, remind_at, recurrence)
            VALUES ($1, $2, $3, $4)
            RETURNING id, user_id, message, remind_at, recurrence, fired, google_event_id, created_at
            """,
            user_id, message, remind_at, recurrence,
        )
        return Reminder.model_validate(dict(row))

    async def update(self, reminder_id: int, **fields) -> Reminder | None:
        """
        Update one or more columns for the given reminder ID.
        Accepted keys: message, remind_at, recurrence.
        """
        if not fields:
            return await self.get_by_id(reminder_id)

        set_clauses = [f"{col} = ${i + 1}" for i, col in enumerate(fields)]
        values = list(fields.values())
        values.append(reminder_id)

        row = await self._conn.fetchrow(
            f"""
            UPDATE reminders
            SET {', '.join(set_clauses)}
            WHERE id = ${len(values)}
            RETURNING id, user_id, message, remind_at, recurrence, fired, google_event_id, created_at
            """,
            *values,
        )
        return Reminder.model_validate(dict(row)) if row else None

    async def delete(self, reminder_ids: list[int], phone: str) -> int:
        """Delete reminders by ID scoped to the given phone. Returns deleted count."""
        result = await self._conn.execute(
            """
            DELETE FROM reminders
            WHERE id = ANY($1::int[])
            AND user_id = (SELECT id FROM users WHERE phone = $2)
            """,
            reminder_ids, phone,
        )
        return int(result.split()[-1])

    async def mark_fired(self, reminder_id: int) -> None:
        await self._conn.execute(
            "UPDATE reminders SET fired = TRUE WHERE id = $1", reminder_id
        )

    async def reschedule(self, reminder_id: int, next_time: datetime) -> None:
        await self._conn.execute(
            "UPDATE reminders SET remind_at = $1 WHERE id = $2",
            next_time, reminder_id,
        )
