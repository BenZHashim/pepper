import asyncio
import json
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import deps
from config import SYSTEM_PROMPT
from .database import get_db_pool
from .llm_client import create_message

logger = logging.getLogger(__name__)


async def get_or_create_user(phone: str) -> dict:
    result = await deps.tool_manager.call("get_user", {"phone": phone})
    return json.loads(result)


async def get_user_skills(phone: str) -> list[dict]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT s.name, s.description
            FROM skills s
            JOIN users u ON u.id = s.user_id
            WHERE u.phone = $1
            ORDER BY s.name
            """,
            phone,
        )
    return [{"name": r["name"], "description": r["description"]} for r in rows]


def _format_skills_context(skills: list[dict]) -> str:
    if not skills:
        return ""
    lines = ["## Your Personalized Skills", ""]
    for s in skills:
        lines.append(f"- **{s['name']}**: {s['description']}")
    lines.append("")
    return "\n".join(lines)


def get_user_system_prompt(user: dict, skills: list[dict]) -> str:
    return SYSTEM_PROMPT.format(
        preferences_prompt=user.get("preferences_prompt", "You are a helpful personal AI assistant."),
        name=user.get("name", "there"),
        timezone=user.get("timezone"),
        now=datetime.now(ZoneInfo(user.get("timezone"))).strftime("%A, %B %d %Y at %I:%M %p %Z"),
        skills_context=_format_skills_context(skills),
    )


def _strip_context(message: str) -> str:
    return re.sub(r'^\[.*?\]\s*', '', message)


async def run_agent(phone: str, user_message: str, user: dict) -> str:
    tools = await deps.tool_manager.get_relevant_tools(user_message)
    await deps.conversation_manager.add_message(phone, "user", user_message)
    messages = await deps.conversation_manager.get_history(phone, user_message)

    skills = await get_user_skills(phone)
    system = get_user_system_prompt(user, skills)
    pool   = await get_db_pool()

    for _ in range(5):
        response = await create_message(
            deps.anthropic_client, pool,
            purpose="main_agent",
            user_id=user.get("id"),
            model="claude-haiku-4-5",
            max_tokens=500,
            system=system,
            tools=tools,
            messages=messages,
        )

        text_content = ""
        tool_uses = []

        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_uses.append(block)

        if response.stop_reason == "end_turn" or not tool_uses:
            if not text_content.strip():
                logger.warning(f"Claude returned empty text_content for {phone}")
                text_content = "Done!"
            text_content = _strip_context(text_content)
            await deps.conversation_manager.add_message(phone, "assistant", text_content)
            asyncio.create_task(deps.skills_agent.run(phone, user_message, skills))
            return text_content

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use in tool_uses:
            logger.info(f"Calling tool: {tool_use.name} with {tool_use.input}")
            tool_input = {**tool_use.input, "phone": phone}
            result = await deps.tool_manager.call(tool_use.name, tool_input)
            logger.info(f"Tool result: {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return "I ran into an issue processing that. Please try again."
