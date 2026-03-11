"""
tools_reminders.py
Reminder tools: create, list, delete, and fetch due reminders.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from mcp_shared import (
    mcp,
    get_conn_for_phone,
    release_conn,
    get_user_timezone,
)
from models import UserRepository, ReminderRepository


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
        user = await UserRepository(conn).get_by_phone(phone)
        if not user:
            return {"success": False, "error": "User not found. They need to be registered first."}

        reminder = await ReminderRepository(conn).create(user.id, message, dt_utc, recurrence)
    finally:
        await release_conn(conn)

    return {"success": True, **reminder.model_dump(mode="json", exclude={"phone", "user_id", "google_event_id", "fired", "created_at"})}


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
        reminders = await ReminderRepository(conn).get_by_user_phone(phone, include_fired=include_fired)
    finally:
        await release_conn(conn)

    return [
        r.model_dump(mode="json", exclude={"phone", "user_id", "google_event_id", "created_at"})
        for r in reminders
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
        deleted = await ReminderRepository(conn).delete(reminder_ids, phone)
    finally:
        await release_conn(conn)

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
    updates = {}

    if message is not None:
        updates["message"] = message

    if remind_at is not None:
        try:
            dt = datetime.fromisoformat(remind_at)
        except ValueError:
            return {"success": False, "error": f"Invalid datetime: {remind_at}. Use ISO 8601 format."}
        user_tz = await get_user_timezone(phone)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(user_tz))
        updates["remind_at"] = dt.astimezone(timezone.utc)

    if recurrence is not None:
        updates["recurrence"] = None if recurrence.lower() == "none" else recurrence

    if not updates:
        return {"success": False, "error": "No fields to update."}

    conn = await get_conn_for_phone(phone)
    try:
        reminder = await ReminderRepository(conn).update(reminder_id, **updates)
    finally:
        await release_conn(conn)

    if not reminder:
        return {"success": False, "error": f"Reminder {reminder_id} not found."}

    return {"success": True, **reminder.model_dump(mode="json", exclude={"phone", "user_id", "google_event_id", "fired", "created_at"})}
