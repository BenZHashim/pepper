import asyncio
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import deps
from config import SYSTEM_PROMPT
from .database import get_db_pool
from .llm_client import create_message
from models import User, UserRepository, Skill, SkillRepository

logger = logging.getLogger(__name__)


async def get_or_create_user(phone: str) -> User | None:
    """
    Fetch the user's full profile directly from the DB.
    Returns None if the user has not completed registration yet.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await UserRepository(conn).get_by_phone(phone)


async def get_user_skills(phone: str) -> list[Skill]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await SkillRepository(conn).get_all_by_phone(phone)


def _format_skills_context(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = ["## Your Personalized Skills", ""]
    for s in skills:
        lines.append(f"- **{s.name}**: {s.description}")
    lines.append("")
    return "\n".join(lines)


def get_user_system_prompt(user: User, skills: list[Skill]) -> str:
    return SYSTEM_PROMPT.format(
        preferences_prompt=user.preferences_prompt,
        name=user.name,
        timezone=user.timezone,
        now=datetime.now(ZoneInfo(user.timezone)).strftime("%A, %B %d %Y at %I:%M %p %Z"),
        skills_context=_format_skills_context(skills),
    )


def _strip_context(message: str) -> str:
    return re.sub(r'^\[.*?\]\s*', '', message)


async def run_agent(phone: str, user_message: str, user: User) -> str:
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
            user_id=user.id,
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
