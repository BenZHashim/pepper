"""
tools_user.py
User management tools: get, create, update profile and preferences.
"""

import asyncpg
from mcp_shared import (
    mcp,
    get_conn_for_phone,
    get_pool,
    release_conn,
    PREFERENCES_TEMPLATE,
    _merge_preferences,
)


@mcp.tool()
async def get_user(phone: str = None) -> dict:
    """
    Look up a user's full profile by phone number including name, timezone,
    preferences prompt, and account details.
    Use at the start of each conversation to load the user's preferences,
    or when you need to check the user's current timezone or name.
    Returns None if the user doesn't exist.
    """
    conn = await get_conn_for_phone(phone)
    try:
        row = await conn.fetchrow(
            "SELECT id, name, phone, preferences_prompt, timezone, created_at FROM users WHERE phone = $1",
            phone,
        )
    finally:
        await release_conn(conn)

    if not row:
        return {"found": False}

    return {
        "found": True,
        "id": row["id"],
        "name": row["name"],
        "phone": row["phone"],
        "preferences_prompt": row["preferences_prompt"],
        "timezone": row["timezone"],
        "created_at": row["created_at"].isoformat(),
    }


@mcp.tool()
async def create_user(name: str, timezone: str, phone: str = None, preferences_prompt: str = None) -> dict:
    """
    Register a brand new user account.
    Only call this during the initial registration flow when a user's phone
    number is not yet in the system. Never call this for existing users.
    Args:
        name:               The user's first name.
        timezone:           Their IANA timezone e.g. 'America/New_York'.
        preferences_prompt: Optional: The user's preferences prompt.
    """
    prompt = preferences_prompt or PREFERENCES_TEMPLATE
    tz = timezone

    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO users (phone, name, timezone, preferences_prompt)
                VALUES ($1, $2, $3, $4)
                RETURNING id, name, phone, timezone, preferences_prompt, created_at
                """,
                phone, name, tz, prompt,
            )
        except asyncpg.UniqueViolationError:
            return {"success": False, "error": f"User with phone {phone} already exists."}

    return {
        "success": True,
        "id": row["id"],
        "name": row["name"],
        "phone": row["phone"],
        "timezone": row["timezone"],
        "preferences_prompt": row["preferences_prompt"],
    }


@mcp.tool()
async def update_user(phone: str = None, name: str = None, timezone: str = None, preferences_prompt: str = None) -> dict:
    """
    Update the user's profile information.
    Use when the user wants to change their name, update their timezone,
    or mentions moving to a different city or region.
    Trigger phrases: 'my name is actually', 'call me', 'I moved to',
    'I'm in a different timezone now', 'update my profile'.
    Args:
        name:               New name, or None to leave unchanged.
        timezone:           New IANA timezone e.g. 'America/Chicago', or None to leave unchanged.
        preferences_prompt: New custom AI preferences, or None to leave unchanged.
    """
    conn = await get_conn_for_phone(phone)
    try:
        row = await conn.fetchrow(
            "SELECT id, name, timezone, preferences_prompt FROM users WHERE phone = $1", phone
        )
        if not row:
            return {"success": False, "error": "User not found."}

        new_name   = name               or row["name"]
        new_tz     = timezone           or row["timezone"]
        new_prompt = preferences_prompt or row["preferences_prompt"]

        updated = await conn.fetchrow(
            """
            UPDATE users SET name = $1, timezone = $2, preferences_prompt = $3
            WHERE phone = $4
            RETURNING id, name, phone, timezone, preferences_prompt
            """,
            new_name, new_tz, new_prompt, phone,
        )
    finally:
        await release_conn(conn)

    return {
        "success": True,
        "id": updated["id"],
        "name": updated["name"],
        "phone": updated["phone"],
        "timezone": updated["timezone"],
        "preferences_prompt": updated["preferences_prompt"],
    }

@mcp.tool()
async def update_user_preferences(instruction: str, phone: str = None) -> dict:
    """
    Update the user's personal preferences for how their AI assistant behaves.
    This covers anything the user wants to customize about their experience, including:
    - Communication style (tone, slang, emoji use, response length)
    - Formatting preferences (how lists or confirmations are presented)
    - Default behaviors (e.g. "remind me 10 minutes before meetings by default",
      "always ask me before deleting a reminder")
    - Personal rules (e.g. "if I mention homework, remind me 1 hour before it's due")

    Call this whenever the user expresses a preference or a personal rule about
    how they want the assistant to behave, even if phrased casually.
    Always call this BEFORE responding, never just acknowledge it conversationally.

    Args:
        instruction: A plain English description of what to add, change, or remove.
                     e.g. "Default reminders to 10 minutes before meetings"
                          "Stop using Australian slang"
                          "Always ask before setting a recurring reminder"
    """
    conn = await get_conn_for_phone(phone)
    try:
        row = await conn.fetchrow(
            "SELECT preferences_prompt FROM users WHERE phone = $1", phone
        )
        existing = row["preferences_prompt"] if row else ""
        updated  = await _merge_preferences(existing, instruction)
        await conn.execute(
            "UPDATE users SET preferences_prompt = $1 WHERE phone = $2",
            updated, phone
        )
    finally:
        await release_conn(conn)
    return {"success": True}
