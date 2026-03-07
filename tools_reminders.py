"""
tools_reminders.py
Reminder tools: create, list, delete, and fetch due reminders.
"""

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta

from mcp_shared import (
    mcp,
    get_pool,
    get_conn_for_phone,
    release_conn,
    get_user_timezone,
)


@mcp.tool()
async def create_reminder(message: str, remind_at: str, recurrence: str = None, phone: str = None) -> dict:
    """
    Schedule a reminder to be sent to the user at a specific time.
    Use when the user says things like 'remind me', 'don't let me forget',
    'alert me', 'set an alarm', 'can you remind me', 'I need to remember',
    or mentions any task alongside a time or date.
    Also use when rescheduling an existing reminder to a new time.
    Supports optional recurrence: 'daily', 'weekly', or 'monthly'.
    Args:
        message:    What to remind about.
        remind_at:  ISO 8601 datetime e.g. '2025-03-10T09:00:00'.
        recurrence: Optional: 'daily', 'weekly', or 'monthly'. None for one-time.
    """
    try:
        dt = datetime.fromisoformat(remind_at)
    except ValueError:
        return {"success": False, "error": f"Invalid datetime: {remind_at}. Use ISO 8601 format e.g. '2025-03-10T09:00:00'"}

    user_tz = await get_user_timezone(phone)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(user_tz))
    dt_utc = dt.astimezone(timezone.utc)

    conn = await get_conn_for_phone(phone)
    try:
        user = await conn.fetchrow("SELECT id FROM users WHERE phone = $1", phone)
        if not user:
            return {"success": False, "error": "User not found. They need to be registered first."}

        row = await conn.fetchrow(
            """
            INSERT INTO reminders (user_id, message, remind_at, recurrence)
            VALUES ($1, $2, $3, $4)
            RETURNING id, message, remind_at, recurrence
            """,
            user["id"], message, dt_utc, recurrence,
        )
    finally:
        await release_conn(conn)

    return {
        "success": True,
        "id": row["id"],
        "message": row["message"],
        "remind_at": row["remind_at"].isoformat(),
        "recurrence": row["recurrence"],
    }


@mcp.tool()
async def list_reminders(phone: str = None, include_fired: bool = False) -> list[dict]:
    """
    Retrieve the user's upcoming reminders.
    Use when the user asks what reminders they have, wants to see their schedule,
    asks 'what did I set up', 'do I have any reminders', 'what's coming up',
    or before deleting/modifying a reminder to find its ID.
    Args:
        include_fired: If True, includes already-sent reminders.
    """
    conn = await get_conn_for_phone(phone)
    try:
        if include_fired:
            rows = await conn.fetch(
                """
                SELECT r.id, r.message, r.remind_at, r.recurrence, r.fired
                FROM reminders r
                JOIN users u ON u.id = r.user_id
                WHERE u.phone = $1
                ORDER BY r.remind_at
                """, phone,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT r.id, r.message, r.remind_at, r.recurrence, r.fired
                FROM reminders r
                JOIN users u ON u.id = r.user_id
                WHERE u.phone = $1 AND r.fired = FALSE
                ORDER BY r.remind_at
                """, phone,
            )
    finally:
        await release_conn(conn)

    return [
        {
            "id": r["id"],
            "message": r["message"],
            "remind_at": r["remind_at"].isoformat(),
            "recurrence": r["recurrence"],
            "fired": r["fired"],
        }
        for r in rows
    ]


@mcp.tool()
async def delete_reminder(reminder_id: int, phone: str = None) -> dict:
    """
    Remove a reminder permanently.
    Use when the user wants to cancel, delete, or remove a reminder.
    Phrases like 'cancel that reminder', 'delete it', 'never mind about that',
    'I don't need that anymore', or 'remove the reminder about X'.
    Always call list_reminders first to find the correct reminder ID.
    Args:
        reminder_id: The reminder ID to delete.
    """
    conn = await get_conn_for_phone(phone)
    try:
        result = await conn.execute(
            """
            DELETE FROM reminders
            WHERE id = $1
            AND user_id = (SELECT id FROM users WHERE phone = $2)
            """,
            reminder_id, phone,
        )
    finally:
        await release_conn(conn)

    deleted = int(result.split()[-1])
    if deleted == 0:
        return {"success": False, "error": f"No reminder found with id {reminder_id} for this user."}
    return {"success": True, "deleted_id": reminder_id}


@mcp.tool()
async def get_due_reminders() -> list[dict]:
    """
    Returns all due reminders across all users for the scheduler.
    Includes the user's phone number so the scheduler knows who to text.
    Only call this from the scheduler.
    """
    now_utc = datetime.now(timezone.utc)
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            due = await conn.fetch(
                """
                SELECT r.id, r.message, r.remind_at, r.recurrence, u.phone
                FROM reminders r
                JOIN users u ON u.id = r.user_id
                WHERE r.fired = FALSE AND r.remind_at <= $1
                FOR UPDATE OF r
                """,
                now_utc,
            )

            due_list = []
            for row in due:
                due_list.append({
                    "id": row["id"],
                    "message": row["message"],
                    "remind_at": row["remind_at"].isoformat(),
                    "recurrence": row["recurrence"],
                    "phone": row["phone"],
                })

                if row["recurrence"] == "daily":
                    next_time = row["remind_at"] + timedelta(days=1)
                    await conn.execute(
                        "UPDATE reminders SET remind_at = $1 WHERE id = $2",
                        next_time, row["id"],
                    )
                elif row["recurrence"] == "weekly":
                    next_time = row["remind_at"] + timedelta(weeks=1)
                    await conn.execute(
                        "UPDATE reminders SET remind_at = $1 WHERE id = $2",
                        next_time, row["id"],
                    )
                elif row["recurrence"] == "monthly":
                    next_time = row["remind_at"] + relativedelta(months=1)
                    await conn.execute(
                        "UPDATE reminders SET remind_at = $1 WHERE id = $2",
                        next_time, row["id"],
                    )
                else:
                    await conn.execute(
                        "UPDATE reminders SET fired = TRUE WHERE id = $1", row["id"]
                    )

    return due_list
