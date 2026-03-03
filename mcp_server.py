"""
FastMCP Server — PostgreSQL + pgvector edition
Reminder and memory tools backed by a single Postgres database.

Run standalone:   python mcp_server.py
Run via Docker:   docker compose up mcp
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
from starlette.requests import Request
from starlette.responses import JSONResponse
from pgvector.asyncpg import register_vector
from memory_utils import run_hybrid_search
from embeddings import EmbeddingManager
import asyncpg
import anthropic
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

class _SuppressNoiseRequests(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "/health" in msg:
            return False
        # Suppress MCP session handshake/teardown but keep tool calls
        if "/mcp" in msg and ("DELETE" in msg or "202 Accepted" in msg):
            return False
        return True
    
logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").addFilter(_SuppressNoiseRequests())
logging.getLogger("mcp").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://smsai:smsai_secret@localhost:5432/smsai")
TIMEZONE     = os.getenv("TIMEZONE", "America/Los_Angeles")
MCP_PORT     = int(os.getenv("MCP_PORT", 8001))

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedding_manager = EmbeddingManager()


# ─── Retrieval Parameters ─────────────────────────────────────────────────────

RECENT_WINDOW   = 10    # Max recent messages to include
SEMANTIC_K      = 5     # Max additional older messages to surface via search

# Retrieval blend weights (independent, do not need to sum to 1)
W_SEMANTIC = 0.6
W_KEYWORD  = 0.4
W_RECENCY  = 0

CANDIDATE_MULTIPLIER = 4

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
    # Set the phone number as a session variable — RLS policies read this
    await conn.execute(
        "SELECT set_config('app.current_phone', $1, TRUE)", phone
    )
    return conn


async def release_conn(conn):
    pool = await get_pool()
    await pool.release(conn)


# ─── FastMCP App ──────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="personal-assistant-tools",
    instructions="Tools for managing reminders and personal memory.",
)


# ─── User Tools ───────────────────────────────────────────────────────────────

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
            "SELECT id, name, timezone, preference_prompt FROM users WHERE phone = $1", phone
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
            RETURNING id, name, phone, timezone, preferences_prompt, 
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

PREFERENCES_TEMPLATE = """\
## Tone & Style
- (e.g. use Australian slang, keep responses short, use emojis sparingly)

## Formatting
- (e.g. use numbered lists for reminders, short sentences for confirmations)

## Reminders
- (e.g. default to 10 minutes before meetings, 1 hour before homework is due)

## Personal Rules
- (e.g. always ask before setting a recurring reminder, confirm before deleting anything)"""


async def _merge_preferences(existing: str, instruction: str) -> str:
    """
    Use Claude to merge a new instruction into the existing preferences prompt.
    Handles both additions and removals naturally based on the instruction.
    """
    response = anthropic_client.messages.create(
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


@mcp.tool()
async def update_user_preferences_prompt(instruction: str, phone: str = None) -> dict:
    """
    Update how the AI assistant communicates with this user.
    Call this whenever the user expresses ANY preference about tone, style,
    emoji use, response length, or formatting — even phrased casually.
    Trigger phrases: 'be more casual', 'use emojis', 'stop being formal',
    'be funnier', 'keep it short', 'talk to me like a friend', 'be more professional',
    'stop doing X', 'don't use X anymore', 'add X', 'remove X'.
    Always call this BEFORE responding to the preference, never just acknowledge it conversationally.
    Args:
        instruction: A plain English description of what to change.
                     e.g. "Add Australian slang" or "Remove the emoji preference"
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

# ─── Reminder Tools ───────────────────────────────────────────────────────────

@mcp.tool()
async def create_reminder( message: str, remind_at: str, recurrence: str = None, phone: str = None) -> dict:
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

    # Apply timezone
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
                    from datetime import timedelta
                    next_time = row["remind_at"] + timedelta(days=1)
                    await conn.execute(
                        "UPDATE reminders SET remind_at = $1 WHERE id = $2",
                        next_time, row["id"],
                    )
                elif row["recurrence"] == "weekly":
                    from datetime import timedelta
                    next_time = row["remind_at"] + timedelta(weeks=1)
                    await conn.execute(
                        "UPDATE reminders SET remind_at = $1 WHERE id = $2",
                        next_time, row["id"],
                    )
                elif row["recurrence"] == "monthly":
                    from dateutil.relativedelta import relativedelta
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

# ─── Memory Tools ─────────────────────────────────────────────────────────────

#@mcp.tool()
async def save_memory(content: str, tags: list[str] = None) -> dict:
    """
    Save something to long-term memory for later recall.
    Use when the user tells you something they want remembered —
    preferences, facts, notes, anything they might ask about later.

    Args:
        content: The thing to remember. Write as a clear, standalone sentence.
        tags:    Optional topic tags, e.g. ['health', 'passwords', 'people'].
    """
    embedding = embedding_manager.embed(content)
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO memories (content, tags, embedding)
            VALUES ($1, $2, $3)
            RETURNING id, content, tags, created_at
            """,
            content,
            tags or [],
            embedding,
        )

    return {
        "success": True,
        "id": row["id"],
        "content": row["content"],
        "tags": row["tags"],
    }

@mcp.tool()
async def search_memory(
    phone: str,
    query: str,
    already_in_context: list[int] = [],
    limit: int = 5,
) -> str:
    """
    Search a user's conversation memory using hybrid semantic + keyword search.

    Args:
        phone: The user's phone number.
        query: The search query to run against memory.
        already_in_context: Message IDs already present in the agent's context — these will be excluded from results.
        limit: Maximum number of results to return (default 5).
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        await register_vector(conn)

        user_id = await conn.fetchval(
            "SELECT id FROM users WHERE phone = $1", phone
        )
        if user_id is None:
            return f"No user found for phone {phone}"

        exclude_ids = already_in_context or [0]

        rows = await run_hybrid_search(
            conn=conn,
            embedding_manager=embedding_manager,
            query=query,
            user_id=user_id,
            exclude_ids=exclude_ids,
            n_candidates=limit * CANDIDATE_MULTIPLIER,
            w_semantic=W_SEMANTIC,
            w_keyword=W_KEYWORD,
            w_recency=W_RECENCY,
            semantic_k=limit,
        )

        if not rows:
            return "No relevant memory found."

        results = []
        for row in rows:
            content = json.loads(row["content"])
            if isinstance(content, dict) and content.get("type") == "text":
                content = content["text"]
            ts = row["created_at"].strftime("%Y-%m-%d %H:%M UTC")
            results.append(f"[id={row['id']} | {row['role']} | {row['message_type']} | {ts}]\n{content}")

        return "\n\n---\n\n".join(results)


#@mcp.tool()
async def list_memories(tag_filter: str = None) -> list[dict]:
    """
    List all saved memories, optionally filtered by tag.

    Args:
        tag_filter: If provided, only return memories with this tag.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        if tag_filter:
            rows = await conn.fetch(
                "SELECT id, content, tags, created_at FROM memories WHERE $1 = ANY(tags) ORDER BY created_at DESC",
                tag_filter,
            )
        else:
            rows = await conn.fetch(
                "SELECT id, content, tags, created_at FROM memories ORDER BY created_at DESC"
            )

    return [
        {
            "id": r["id"],
            "content": r["content"],
            "tags": r["tags"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


#@mcp.tool()
async def delete_memory(memory_id: int) -> dict:
    """
    Delete a memory by ID.

    Args:
        memory_id: The numeric ID of the memory to delete.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)

    deleted = int(result.split()[-1])
    if deleted == 0:
        return {"success": False, "error": f"No memory found with id {memory_id}"}
    return {"success": True, "deleted_id": memory_id}


async def get_user_timezone(phone: str) -> str:
    """Get a user's timezone, falling back to the server default."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT timezone FROM users WHERE phone = $1", phone
        )
    return row["timezone"] if row else TIMEZONE

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Starting FastMCP server on port {MCP_PORT}")
    mcp.run(transport="http", host="0.0.0.0", port=MCP_PORT)