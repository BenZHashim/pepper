"""
tools_skills.py
Skills tools exposed to the main agent.

Only read and delete are here — creates and updates are handled exclusively
by the background SkillsAgent, which writes directly to the DB.
"""

from mcp_shared import (
    mcp,
    get_conn_for_phone,
    release_conn,
)
from models import SkillRepository


@mcp.tool()
async def get_skill_content(name: str, phone: str = None) -> str:
    """
    Retrieve the full markdown content of a named skill.
    Use when a skill name appears in the system prompt's skill list and is relevant
    to the current request — e.g. read 'meeting_reminders' before setting a meeting
    reminder so you know the user's preferred lead time.
    Args:
        name: The skill name exactly as shown in the skill list.
    """
    conn = await get_conn_for_phone(phone)
    try:
        skill = await SkillRepository(conn).get_by_name(phone, name)
    finally:
        await release_conn(conn)

    if not skill:
        return f"No skill found with name '{name}'."
    return skill.content


@mcp.tool()
async def delete_skill(name: str, phone: str = None) -> dict:
    """
    Delete a skill permanently.
    Use when the user explicitly asks to remove or forget a saved preference.
    Args:
        name: The skill name to delete, exactly as shown in the skill list.
    """
    conn = await get_conn_for_phone(phone)
    try:
        deleted = await SkillRepository(conn).delete(name, phone)
    finally:
        await release_conn(conn)

    if not deleted:
        return {"success": False, "error": f"No skill found with name '{name}'."}
    return {"success": True, "deleted_name": name}
