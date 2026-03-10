"""
tools_memory.py
Memory / conversation-history tools:
  - search_conversation_history  (hybrid semantic + keyword)
  - search_messages_by_time      (time-window lookup)
  - get_source_messages          (expand a summary back to raw messages)
"""

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from pgvector.asyncpg import register_vector
from services.memory_utils import run_hybrid_search, increment_retrieval_counts

from mcp_shared import (
    mcp,
    get_conn_for_phone,
    release_conn,
    get_user_timezone,
    embedding_manager,
    W_SEMANTIC,
    W_KEYWORD,
    W_RECENCY,
    SEMANTIC_K,
    CANDIDATE_MULTIPLIER,
)


@mcp.tool()
async def search_conversation_history(
    phone: str,
    query: str,
    already_in_context: list[int] = [],
    limit: int = 5,
) -> str:
    """
    Search a user's conversation history by topic using hybrid semantic + keyword search.
    Use when the user asks about something that was discussed before, wants to recall
    a specific topic, or references past conversations by subject rather than date.
    Trigger phrases: 'do you remember when', 'what did I say about', 'we talked about X',
    'I mentioned X before', 'what do you know about my X'.
    For time-based lookups use search_messages_by_time instead.

    Args:
        phone: The user's phone number.
        query: The topic or subject to search for.
        already_in_context: Message IDs already present in the agent's context — these will be excluded from results.
        limit: Maximum number of results to return (default 5).
    """
    conn = await get_conn_for_phone(phone)
    try:
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
    finally:
        await release_conn(conn)

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


@mcp.tool()
async def search_messages_by_time(
    phone: str,
    after: str = None,
    before: str = None,
    limit: int = 20,
) -> str:
    """
    Retrieve conversation history within a time window.
    Use when the user asks what was discussed at a specific date or time period.
    Trigger phrases: 'what did we talk about last Tuesday', 'show me our conversation
    from yesterday', 'what happened on March 5th', 'what did I tell you last week',
    'what were we discussing around 3pm', 'what did I say on [date]'.

    At least one of after/before should be provided. If neither is given, returns
    the most recent messages up to the limit.

    Args:
        phone:  The user's phone number.
        after:  ISO 8601 datetime — only return messages at or after this time.
                e.g. '2025-03-10T00:00:00' for the start of March 10th.
        before: ISO 8601 datetime — only return messages at or before this time.
                e.g. '2025-03-10T23:59:59' for the end of March 10th.
        limit:  Maximum number of messages to return (default 20).
    """
    user_tz = await get_user_timezone(phone)

    def parse_dt(s: str) -> datetime:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(user_tz))
        return dt.astimezone(timezone.utc)

    conn = await get_conn_for_phone(phone)
    try:
        user_id = await conn.fetchval(
            "SELECT id FROM users WHERE phone = $1", phone
        )
        if user_id is None:
            return f"No user found for phone {phone}"

        conditions = ["user_id = $1"]
        params: list = [user_id]

        if after:
            try:
                after_dt = parse_dt(after)
                params.append(after_dt)
                conditions.append(f"period_end >= ${len(params)}")
            except ValueError:
                return f"Invalid 'after' datetime: {after}. Use ISO 8601 format."

        if before:
            try:
                before_dt = parse_dt(before)
                params.append(before_dt)
                conditions.append(f"period_start <= ${len(params)}")
            except ValueError:
                return f"Invalid 'before' datetime: {before}. Use ISO 8601 format."

        params.append(limit)
        where = " AND ".join(conditions)
        rows = await conn.fetch(
            f"""
            SELECT id, role, message_type, content, period_start, period_end
            FROM messages
            WHERE {where}
            ORDER BY period_end ASC
            LIMIT ${len(params)}
            """,
            *params,
        )
        await increment_retrieval_counts(conn, [r["id"] for r in rows])
    finally:
        await release_conn(conn)

    if not rows:
        return "No messages found in that time range."

    results = []
    for row in rows:
        content = json.loads(row["content"])
        if isinstance(content, dict) and content.get("type") == "text":
            content = content["text"]
        elif isinstance(content, list):
            parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = " ".join(parts) if parts else str(content)

        tz = ZoneInfo(user_tz)
        ts = row["period_end"].astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")
        results.append(
            f"[id={row['id']} | {row['role']} | {row['message_type']} | {ts}]\n{content}"
        )

    return "\n\n---\n\n".join(results)


@mcp.tool()
async def get_source_messages(phone: str, summary_id: int) -> str:
    """
    Retrieve the original messages that a summary was composed from.
    Use when a summary is vague and more detail is needed, or when the user
    wants to see the raw conversation behind a summary.
    Trigger phrases: 'what exactly was said', 'show me the original messages',
    'expand that summary', 'more detail on that', 'what were the details behind that'.

    Args:
        phone:      The user's phone number.
        summary_id: The ID of the summary message to expand.
    """
    user_tz = await get_user_timezone(phone)

    conn = await get_conn_for_phone(phone)
    try:
        summary = await conn.fetchrow(
            "SELECT message_type, source_message_ids FROM messages WHERE id = $1",
            summary_id,
        )
        if summary is None:
            return f"No message found with id {summary_id}."
        if summary["message_type"] != "summary":
            return f"Message {summary_id} is not a summary (type: {summary['message_type']})."

        source_ids = summary["source_message_ids"]
        if not source_ids:
            return f"Summary {summary_id} has no recorded source message IDs."

        rows = await conn.fetch(
            """
            SELECT id, role, message_type, content, period_end
            FROM messages
            WHERE id = ANY($1)
            ORDER BY period_end ASC
            """,
            source_ids,
        )
    finally:
        await release_conn(conn)

    if not rows:
        return "Source messages could not be found (they may have been deleted)."

    tz = ZoneInfo(user_tz)
    results = []
    for row in rows:
        content = json.loads(row["content"])
        if isinstance(content, dict) and content.get("type") == "text":
            content = content["text"]
        elif isinstance(content, list):
            parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = " ".join(parts) if parts else str(content)

        ts = row["period_end"].astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")
        results.append(
            f"[id={row['id']} | {row['role']} | {row['message_type']} | {ts}]\n{content}"
        )

    return "\n\n---\n\n".join(results)
