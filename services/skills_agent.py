"""
skills_agent.py
Two-stage skill detection and management agent.

Stage 1 — Identify: Determines if a user message expresses a reusable
           procedural preference worth saving as a skill.

Stage 2 — Content:  Generates markdown content for a new skill, or merges the
           new preference into existing content (mirrors _merge_preferences).
           Whether to create or update is determined by DB lookup, not the
           identify agent.
"""

import logging

import anthropic
import asyncpg

from .llm_client import create_message

logger = logging.getLogger(__name__)

# ─── Stage 1: Identify Tool ───────────────────────────────────────────────────

_IDENTIFY_TOOL = {
    "name": "identify_skill_action",
    "description": "Classify whether the user's message expresses a reusable skill preference to save.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "none"],
                "description": (
                    "'save' — message contains a clear, reusable preference about HOW a task type should be handled. "
                    "'none' — one-off request, question, or no reusable preference expressed."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Snake_case skill name (e.g. 'meeting_reminders', 'homework_reminders'). "
                    "If an existing skill name matches, reuse it exactly. Required for 'save'."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "One-sentence description of when this skill applies "
                    "(e.g. 'Applied when setting reminders for meetings'). "
                    "Required for 'save'."
                ),
            },
        },
        "required": ["action"],
    },
}

_IDENTIFY_SYSTEM = """\
You are a skill extraction agent. Detect when users express preferences about HOW tasks \
should be performed — reusable rules that should apply to future similar requests.

A skill captures a general preference, not a one-off action. Examples:
- "For meeting reminders, always remind me 20 minutes before" → save (meeting_reminders)
- "I want homework reminders 1 hour before the due time" → save (homework_reminders)
- "Actually make it 30 minutes for meetings" → save (meeting_reminders)
- "Remind me about my dentist at 3pm" → none  (specific task, not a preference)
- "What reminders do I have?" → none  (question, not a preference)
- "Set a reminder for tomorrow at 9am" → none  (one-off, no expressed preference)

Only classify as 'save' when the message contains a clear, reusable preference \
that should change how future tasks of that type are handled. \
If an existing skill name matches the preference topic, reuse that name exactly.\
"""

# ─── Stage 2: Content System ──────────────────────────────────────────────────

_CONTENT_SYSTEM = """\
You are a skill content writer for a personal AI assistant. Write concise markdown \
that encodes a user's preference so the assistant can apply it automatically in future.

Always use this structure:

## Preference
[What the user wants, stated clearly]

## When to Apply
- [Trigger conditions — what makes this skill relevant]

## Steps
1. [Concrete action steps]

Keep it brief and actionable. Return only the markdown, nothing else.\
"""


# ─── SkillsAgent ──────────────────────────────────────────────────────────────

class SkillsAgent:
    """
    Two-stage agent that silently creates or updates skills based on user messages.
    Writes directly to the DB — no MCP tools — so there's no conflict with the
    main agent, which only has read access to skills.
    Designed to run as a background task (fire and forget via asyncio.create_task).
    """

    def __init__(self, pool: asyncpg.Pool, anthropic_client: anthropic.Anthropic):
        self._pool = pool
        self._client = anthropic_client

    async def run(self, phone: str, user_message: str, existing_skills: list[dict]) -> None:
        """
        Entry point. Detects skill-worthy preferences and persists them.
        Whether to create or update is determined by DB lookup, not the identify agent.
        Swallows all exceptions — this is a best-effort background task.
        """
        try:
            # Resolve user_id once — used for LLM usage tracking
            async with self._pool.acquire() as conn:
                user_id: int | None = await conn.fetchval(
                    "SELECT id FROM users WHERE phone = $1", phone
                )

            identification = await self._identify(user_message, existing_skills, user_id)
            if identification.get("action") != "save":
                return

            name = identification.get("name", "").strip()
            description = identification.get("description", "").strip()
            if not name:
                logger.warning("Skill agent returned 'save' without a name — skipping")
                return

            # Check DB to decide create vs update
            existing_content = await self._fetch_content(phone, name)

            content = await self._generate_content(user_message, existing_content, user_id)
            if not content:
                logger.warning(f"Skill agent generated empty content for '{name}' — skipping")
                return

            if existing_content is None:
                await self._create(phone, name, description, content)
                logger.info(f"Skill agent created '{name}' for {phone}")
            else:
                await self._update(phone, name, description, content)
                logger.info(f"Skill agent updated '{name}' for {phone}")

        except Exception as e:
            logger.warning(f"Skill agent error for {phone}: {e}", exc_info=True)

    # ─── Stage 1 ──────────────────────────────────────────────────────────────

    async def _identify(self, user_message: str, existing_skills: list[dict], user_id: int | None) -> dict:
        skills_text = (
            "\n".join(f"- {s['name']}: {s['description']}" for s in existing_skills)
            or "None."
        )
        response = await create_message(
            self._client, self._pool,
            purpose="skill_identify",
            user_id=user_id,
            model="claude-haiku-4-5",
            max_tokens=200,
            system=_IDENTIFY_SYSTEM,
            tools=[_IDENTIFY_TOOL],
            tool_choice={"type": "tool", "name": "identify_skill_action"},
            messages=[{
                "role": "user",
                "content": (
                    f"Existing skills:\n{skills_text}\n\n"
                    f"User message:\n{user_message}"
                ),
            }],
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        return {"action": "none"}

    # ─── Stage 2 ──────────────────────────────────────────────────────────────

    async def _fetch_content(self, phone: str, name: str) -> str | None:
        """Return current skill content, or None if the skill doesn't exist yet."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT s.content
                FROM skills s
                JOIN users u ON u.id = s.user_id
                WHERE u.phone = $1 AND s.name = $2
                """,
                phone, name,
            )
        return row["content"] if row else None

    async def _create(self, phone: str, name: str, description: str, content: str) -> None:
        async with self._pool.acquire() as conn:
            user_id = await conn.fetchval("SELECT id FROM users WHERE phone = $1", phone)
            if not user_id:
                raise ValueError(f"User not found for phone {phone}")
            await conn.execute(
                """
                INSERT INTO skills (user_id, name, description, content)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, name) DO NOTHING
                """,
                user_id, name, description, content,
            )

    async def _update(self, phone: str, name: str, description: str, content: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE skills
                SET description = $1, content = $2, updated_at = NOW()
                WHERE name = $3
                  AND user_id = (SELECT id FROM users WHERE phone = $4)
                """,
                description, content, name, phone,
            )

    async def _generate_content(self, user_message: str, existing_content: str | None, user_id: int | None) -> str:
        """Generate new skill content or merge new preference into existing content."""
        if existing_content is None:
            prompt = f"Write skill content for this user preference:\n\n{user_message}"
        else:
            prompt = (
                f"Update the skill content below based on this new preference: '{user_message}'\n\n"
                f"Rules:\n"
                f"- Merge the new preference into the appropriate section\n"
                f"- If it contradicts existing content, replace it\n"
                f"- Keep the markdown structure intact\n"
                f"- Return only the updated markdown, nothing else\n\n"
                f"Current content:\n{existing_content}"
            )

        response = await create_message(
            self._client, self._pool,
            purpose="skill_content",
            user_id=user_id,
            model="claude-haiku-4-5",
            max_tokens=400,
            system=_CONTENT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), None)
        return text.strip() if text else ""
