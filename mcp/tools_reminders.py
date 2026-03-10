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
async def delete_reminders(reminder_ids: list[int], phone: str = None) -> dict:
    """
    Remove one or more reminders permanently.
    Use when the user wants to cancel, delete, or remove reminders.
    Phrases like 'cancel that reminder', 'delete them all', 'never mind about those',
    'I don't need that anymore', 'clear all my reminders', or 'remove the reminder about X'.
    Always call list_reminders first to find the correct reminder IDs.
    Args:
        reminder_ids: List of reminder IDs to delete. Pass a single-element list to delete one.
    """
    if not reminder_ids:
        return {"success": False, "error": "No reminder IDs provided."}

    conn = await get_conn_for_phone(phone)
    try:
        result = await conn.execute(
            """
            DELETE FROM reminders
            WHERE id = ANY($1::int[])
            AND user_id = (SELECT id FROM users WHERE phone = $2)
            """,
            reminder_ids, phone,
        )
    finally:
        await release_conn(conn)

    deleted = int(result.split()[-1])
    not_found = len(reminder_ids) - deleted
    return {
        "success": True,
        "deleted_count": deleted,
        "deleted_ids": reminder_ids if not_found == 0 else None,
        **({"not_found_count": not_found} if not_found > 0 else {}),
    }


@mcp.tool()
async def update_reminder(
    reminder_id: int,
    phone: str = None,
    message: str = None,
    remind_at: str = None,
    recurrence: str = None,
) -> dict:
    """
    Change the time, message, or recurrence of an existing reminder.
    Use when the user wants to reschedule, rename, or modify a reminder.
    Phrases like 'change that reminder to 5pm', 'move my reminder to tomorrow',
    'update the reminder to say X', 'make it weekly instead'.
    Always call list_reminders first to find the correct reminder ID.
    Args:
        reminder_id: ID of the reminder to update.
        message:     New reminder message (omit to keep existing).
        remind_at:   New ISO 8601 datetime e.g. '2025-03-10T09:00:00' (omit to keep existing).
        recurrence:  New recurrence: 'daily', 'weekly', 'monthly', or None to make it one-time.
                     Pass the string 'none' to explicitly clear recurrence.
    """
    fields = []
    values = []

    if message is not None:
        fields.append(f"message = ${len(values) + 1}")
        values.append(message)

    if remind_at is not None:
        try:
            dt = datetime.fromisoformat(remind_at)
        except ValueError:
            return {"success": False, "error": f"Invalid datetime: {remind_at}. Use ISO 8601 format."}
        user_tz = await get_user_timezone(phone)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(user_tz))
        dt_utc = dt.astimezone(timezone.utc)
        fields.append(f"remind_at = ${len(values) + 1}")
        values.append(dt_utc)

    if recurrence is not None:
        new_recurrence = None if recurrence.lower() == "none" else recurrence
        fields.append(f"recurrence = ${len(values) + 1}")
        values.append(new_recurrence)

    if not fields:
        return {"success": False, "error": "No fields to update."}

    # Add reminder_id and phone for the WHERE clause
    values.extend([reminder_id, phone])
    id_param = len(values) - 1
    phone_param = len(values)

    conn = await get_conn_for_phone(phone)
    try:
        row = await conn.fetchrow(
            f"""
            UPDATE reminders SET {', '.join(fields)}
            WHERE id = ${id_param}
              AND user_id = (SELECT id FROM users WHERE phone = ${phone_param})
            RETURNING id, message, remind_at, recurrence
            """,
            *values,
        )
    finally:
        await release_conn(conn)

    if not row:
        return {"success": False, "error": f"Reminder {reminder_id} not found."}

    return {
        "success": True,
        "id": row["id"],
        "message": row["message"],
        "remind_at": row["remind_at"].isoformat(),
        "recurrence": row["recurrence"],
    }

