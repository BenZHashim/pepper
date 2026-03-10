"""
mcp_shared.py
Shared state for the FastMCP server:
  - FastMCP instance (mcp)
  - Database pool and connection helpers
  - Retrieval constants
  - Anthropic client and embedding manager
  - Preferences template and merge helper
  - Timezone helper
"""

import os
import asyncio
import asyncpg
import anthropic
from fastmcp import FastMCP
from dotenv import load_dotenv
from services.embeddings import EmbeddingManager
from services.llm_client import create_message

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://smsai:smsai_secret@localhost:5432/smsai")
TIMEZONE     = os.getenv("TIMEZONE", "America/Los_Angeles")
MCP_PORT     = int(os.getenv("MCP_PORT", 8001))

# ─── Retrieval Parameters ─────────────────────────────────────────────────────

RECENT_WINDOW        = 10   # Max recent messages to include
SEMANTIC_K           = 5    # Max additional older messages to surface via search

# Retrieval blend weights (independent, do not need to sum to 1)
W_SEMANTIC           = 0.6
W_KEYWORD            = 0.4
W_RECENCY            = 0

CANDIDATE_MULTIPLIER = 4

# ─── Shared Clients ───────────────────────────────────────────────────────────

anthropic_client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedding_manager = EmbeddingManager()

# ─── FastMCP Instance ─────────────────────────────────────────────────────────

mcp = FastMCP(
    name="personal-assistant-tools",
    instructions="Tools for managing reminders and personal memory.",
)

# ─── Database Pool ────────────────────────────────────────────────────────────

_pool: asyncpg.Pool = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return _pool


async def get_conn_for_phone(phone: str):
    """
    Returns a connection with the RLS context set for this phone number.
    All queries on this connection will be filtered by RLS policies.
    """
    pool = await get_pool()
    conn = await pool.acquire()
    await conn.execute(
        "SELECT set_config('app.current_phone', $1, TRUE)", phone
    )
    return conn


async def release_conn(conn):
    pool = await get_pool()
    await pool.release(conn)


async def get_user_timezone(phone: str) -> str:
    """Get a user's timezone, falling back to the server default."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT timezone FROM users WHERE phone = $1", phone
        )
    return row["timezone"] if row else TIMEZONE


# ─── Preferences ──────────────────────────────────────────────────────────────

PREFERENCES_TEMPLATE = """\
## Tone & Style
- (e.g. use Australian slang, keep responses short, use emojis sparingly)

## Formatting
- (e.g. use numbered lists for reminders, short sentences for confirmations)

## Reminders
- (e.g. default to 10 minutes before meetings, 1 hour before homework is due)

## Personal Rules
- (e.g. always ask before setting a recurring reminder, confirm before deleting anything)"""


async def _merge_preferences(existing: str, instruction: str, user_id: int | None = None) -> str:
    """
    Use Claude to merge a new instruction into the existing preferences prompt.
    Handles both additions and removals naturally based on the instruction.
    """
    pool = await get_pool()
    response = await create_message(
        anthropic_client, pool,
        purpose="merge_preferences",
        user_id=user_id,
        model="claude-haiku-4-5",
        max_tokens=650,
        messages=[{
            "role": "user",
            "content": (
                f"Update the preferences prompt below based on this instruction: '{instruction}'\n\n"
                f"Rules:\n"
                f"- Add, modify, or remove preferences as the instruction implies\n"
                f"- Merge into the correct section, do not duplicate existing preferences\n"
                f"- If the instruction contradicts an existing preference, replace it\n"
                f"- Only include user-facing preferences like tone, style, and formatting\n"
                f"- Do not include system behavior, security rules, or platform constraints\n"
                f"- Keep the total output under 500 words and concise within each bullet\n"
                f"- Add or remove sections as needed\n"
                f"- Return only the updated markdown, nothing else\n\n"
                f"Current preferences:\n{existing or PREFERENCES_TEMPLATE}"
            )
        }]
    )
    return response.content[0].text.strip()
