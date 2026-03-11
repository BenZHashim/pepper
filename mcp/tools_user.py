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
from models import User, UserRepository

# Fields that must never enter the agent's context window.
_SENSITIVE = {"google_tokens"}


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
        user = await UserRepository(conn).get_by_phone(phone)
    finally:
        await release_conn(conn)

    if not user:
        return {"found": False}
    return {"found": True, **user.model_dump(mode="json", exclude=_SENSITIVE | {"preferences_prompt"})}


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

    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            user = await UserRepository(conn).create(phone, name, timezone, prompt)
        except asyncpg.UniqueViolationError:
            return {"success": False, "error": f"User with phone {phone} already exists."}

    return {"success": True, **user.model_dump(mode="json", exclude=_SENSITIVE)}


@mcp.tool()
async def update_user(phone: str = None, name: str = None, timezone: str = None) -> dict:
    """
    Update the user's profile information.
    Use when the user wants to change their name, update their timezone,
    or mentions moving to a different city or region.
    Trigger phrases: 'my name is actually', 'call me', 'I moved to',
    'I'm in a different timezone now', 'update my profile'.
    Args:
        name:     New name, or None to leave unchanged.
        timezone: New IANA timezone e.g. 'America/Chicago', or None to leave unchanged.
    """
    conn = await get_conn_for_phone(phone)
    try:
        repo = UserRepository(conn)
        existing = await repo.get_by_phone(phone)
        if not existing:
            return {"success": False, "error": "User not found."}

        updates = {
            "name":               name               or existing.name,
            "timezone":           timezone           or existing.timezone,
        }
        user = await repo.update(phone, **updates)
    finally:
        await release_conn(conn)
    return {"success": True, **user.model_dump(mode="json", exclude=_SENSITIVE | {"preferences_prompt"})}


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
        repo = UserRepository(conn)
        user = await repo.get_by_phone(phone)
        existing = user.preferences_prompt if user else ""
        user_id = user.id if user else None
        updated = await _merge_preferences(existing, instruction, user_id=user_id)
        await repo.update(phone, preferences_prompt=updated)
    finally:
        await release_conn(conn)
    return {"success": True}
