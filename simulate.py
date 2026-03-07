#!/usr/bin/env python3
"""
simulate.py — Test harness for the SMS AI memory system.

Drives the real API by posting to POST /sms — exactly like a Twilio webhook.
DEV_MODE=TRUE in the API skips signature validation so no Twilio credentials
are needed.  After each exchange the inserted messages are back-dated via a
direct database connection so the summarisation system sees realistic ages.

Services required (all exposed to localhost via docker-compose port-mapping):
    API          http://localhost:5000
    Database     localhost:5432
    Embeddings   http://localhost:8002   (for --show retrieval preview)
    MCP          http://localhost:8001   (used internally by the API)

Usage:
    # Auto-generate 15 user messages, run through the real API, back-date:
    python simulate.py --phone +15551234567

    # More exchanges over a longer span (good for triggering summarisation):
    python simulate.py --phone +15551234567 --auto 40 --days 60

    # Load a JSON scenario file (see example_scenario.json):
    python simulate.py --phone +15551234567 --scenario example_scenario.json

    # Show current message state + retention scores:
    python simulate.py --phone +15551234567 --show

    # Show state and simulate what get_history() would return for a query:
    python simulate.py --phone +15551234567 --show --query "what reminders do I have"

    # Trigger one summarisation pass (calls POST /admin/summarise):
    python simulate.py --phone +15551234567 --summarize

    # Clear all history for the phone:
    python simulate.py --phone +15551234567 --clear

Scenario JSON format:
    {
      "name": "One week of reminders",
      "topic": "setting reminders and daily tasks",
      "start_datetime": "2026-01-01T09:00:00",   // optional, defaults to 30 days ago
      "phases": [
        {
          "label": "Day 1",
          "day_offset": 0,
          "exchanges": [
            {"user": "Remind me to call Dave at 3pm"},
            {"user": "auto"}    // Claude generates the user message
          ]
        },
        {
          "label": "Day 5",
          "day_offset": 4,
          "exchanges": [
            {"user": "What reminders do I have?"}
          ]
        }
      ]
    }

    The assistant response is always the real API's response — never simulated.
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import anthropic
import asyncpg
import httpx
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.dirname(__file__))
from memory_utils import (
    extract_text, retention_score, run_hybrid_search,
    RETENTION_BOOST, RETENTION_HALF_LIFE_DAYS,
)
from embeddings import EmbeddingManager

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for lib in ("httpx", "mcp", "fastmcp"):
    logging.getLogger(lib).setLevel(logging.ERROR)


# ── URL helpers ────────────────────────────────────────────────────────────────

def _api_url() -> str:
    return os.getenv("API_URL", "http://localhost:5000")

def _local_db_url() -> str:
    url = os.getenv("DATABASE_URL", "postgresql://smsai:smsai_secret@db:5432/smsai")
    return url.replace("@db:", "@localhost:")

def _local_embedding_url() -> str:
    return os.getenv("EMBEDDING_URL", "http://localhost:8002")


# ── HTTP: call the real /sms endpoint ─────────────────────────────────────────

async def call_sms(http: httpx.AsyncClient, api_url: str, phone: str, message: str) -> str:
    """
    POST to /sms as if we were Twilio.  Returns the assistant's reply text
    extracted from the TwiML response.
    """
    resp = await http.post(
        f"{api_url}/sms",
        data={"From": phone, "Body": message},
        timeout=60.0,
    )
    resp.raise_for_status()

    # Response is TwiML: <Response><Message>text</Message></Response>
    root = ET.fromstring(resp.text)
    msg = root.find(".//{*}Message") or root.find(".//Message")
    return msg.text.strip() if msg is not None and msg.text else ""


# ── Database: back-dating ──────────────────────────────────────────────────────

async def _get_user_id(conn: asyncpg.Connection, phone: str) -> int | None:
    row = await conn.fetchrow("SELECT id FROM users WHERE phone = $1", phone)
    return row["id"] if row else None


async def backdate_messages(
    conn: asyncpg.Connection,
    user_id: int,
    ts: datetime,
    since: datetime,
) -> int:
    """
    Set created_at / period_start / period_end = ts for any raw messages
    inserted by this user since `since`.  Called right after call_sms().
    """
    result = await conn.execute(
        """
        UPDATE messages
        SET created_at   = $1,
            period_start = $1,
            period_end   = $1
        WHERE user_id      = $2
          AND message_type = 'raw'
          AND created_at  >= $3
        """,
        ts, user_id, since,
    )
    return int(result.split()[-1])


# ── Core: one exchange ─────────────────────────────────────────────────────────

async def run_exchange(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    api_url: str,
    phone: str,
    user_message: str,
    desired_ts: datetime,
    verbose: bool = True,
) -> str:
    """
    Send one user message to the real API, get the reply, then back-date
    both inserted messages to desired_ts.
    """
    before = datetime.now(tz=timezone.utc)
    reply  = await call_sms(http, api_url, phone, user_message)

    async with pool.acquire() as conn:
        user_id = await _get_user_id(conn, phone)
        n = await backdate_messages(conn, user_id, desired_ts, before)

    if verbose:
        ts_str = desired_ts.strftime("%Y-%m-%d %H:%M")
        print(f"  {ts_str}  user      | {user_message[:70]}")
        print(f"  {ts_str}  assistant | {reply[:70]}")
        if n != 2:
            print(f"             (back-dated {n} messages — expected 2)")

    return reply


# ── Claude: generate user-side messages ───────────────────────────────────────

def _generate_user_message(
    client: anthropic.Anthropic,
    topic: str,
    context_label: str,
    history: list[dict],
) -> str:
    ctx = f" Context: {context_label}." if context_label else ""
    instruction = (
        f"You are simulating a user texting a personal AI reminder assistant via SMS.{ctx} "
        f"Topic: {topic}. "
        "Write ONE natural, casual SMS the user might send next (under 160 chars). "
        "Reply with ONLY the message text."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=history + [{"role": "user", "content": instruction}],
    )
    return next((b.text.strip() for b in response.content if b.type == "text"), "Hi")


# ── State display ──────────────────────────────────────────────────────────────

async def show_state(conn: asyncpg.Connection, user_id: int, label: str = "") -> None:
    now = datetime.now(tz=timezone.utc)
    rows = await conn.fetch(
        """
        SELECT id, role, message_type, content, period_end,
               retrieval_count, last_retrieved_at, embedding_pruned_at
        FROM messages
        WHERE user_id = $1
        ORDER BY period_end ASC
        """,
        user_id,
    )

    if not rows:
        print("  (no messages)")
        return

    by_type: dict[str, list] = {}
    for row in rows:
        by_type.setdefault(row["message_type"], []).append(row)

    print(f"\n{'─' * 64}")
    if label:
        print(f"  {label}")
    print(f"{'─' * 64}")

    for mtype, msgs in sorted(by_type.items()):
        print(f"\n  [{mtype.upper()}]  {len(msgs)} messages")

        scores = []
        for m in msgs:
            if m["last_retrieved_at"]:
                s = retention_score(m["retrieval_count"], m["last_retrieved_at"], now)
            else:
                age_days = max((now - m["period_end"]).total_seconds() / 86400, 0)
                s = RETENTION_BOOST * math.pow(2.0, -age_days / RETENTION_HALF_LIFE_DAYS)
            scores.append(s)

        print(f"    retention  min={min(scores):.3f}  avg={sum(scores)/len(scores):.3f}  max={max(scores):.3f}")

        oldest = msgs[0]["period_end"].strftime("%Y-%m-%d %H:%M UTC")
        newest = msgs[-1]["period_end"].strftime("%Y-%m-%d %H:%M UTC")
        print(f"    time span  {oldest}  →  {newest}")

        pruned = sum(1 for m in msgs if m["embedding_pruned_at"])
        if pruned:
            print(f"    pruned     {pruned}/{len(msgs)} embeddings removed")

        sample = msgs[:2] + (msgs[-1:] if len(msgs) > 2 else [])
        for m in sample:
            text  = extract_text(json.loads(m["content"]))
            short = text[:80] + "…" if len(text) > 80 else text
            age   = (now - m["period_end"]).days
            print(f"    [{m['id']:4d}] {m['role']:<9} {age:3d}d | {short}")

    print()


async def show_retrieval(
    conn: asyncpg.Connection,
    user_id: int,
    query: str,
    embedding_manager: EmbeddingManager,
    recent_window: int = 20,
    semantic_k: int = 10,
) -> None:
    await register_vector(conn)

    recent_rows = await conn.fetch(
        """
        SELECT id, role, message_type, content, period_end
        FROM messages
        WHERE user_id = $1 AND message_type = 'raw'
        ORDER BY period_end DESC
        LIMIT $2
        """,
        user_id, recent_window,
    )
    recent_ids = [r["id"] for r in recent_rows]

    print(f"\n  Retrieval preview for: '{query}'")
    print(f"  ── Recent window ({len(recent_rows)} msgs) ──────────────────")
    for row in sorted(recent_rows, key=lambda r: r["period_end"]):
        text  = extract_text(json.loads(row["content"]))
        short = text[:72] + "…" if len(text) > 72 else text
        print(f"    [{row['id']:4d}] {row['role']:<9} {row['period_end'].strftime('%Y-%m-%d')} | {short}")

    search_rows = await run_hybrid_search(
        conn=conn,
        embedding_manager=embedding_manager,
        query=query,
        user_id=user_id,
        exclude_ids=recent_ids or [0],
        n_candidates=semantic_k * 10,
        w_semantic=0.6,
        w_keyword=0.3,
        w_recency=0.1,
        semantic_k=semantic_k,
    )

    if search_rows:
        print(f"\n  ── Hybrid search ({len(search_rows)} older msgs) ──────────")
        for row in sorted(search_rows, key=lambda r: r["period_end"]):
            text  = extract_text(json.loads(row["content"]))
            short = text[:72] + "…" if len(text) > 72 else text
            print(f"    [{row['id']:4d}] {row['role']:<9} {row['message_type']:<7} {row['period_end'].strftime('%Y-%m-%d')} | {short}")
    else:
        print("\n  Hybrid search: no additional results")
    print()


# ── Commands ───────────────────────────────────────────────────────────────────

async def cmd_show(pool: asyncpg.Pool, phone: str, query: str, emb_url: str) -> None:
    async with pool.acquire() as conn:
        user_id = await _get_user_id(conn, phone)
        if not user_id:
            print(f"No user found for {phone}")
            return
        await show_state(conn, user_id, f"State for {phone}")
        if query:
            mgr = EmbeddingManager(url=emb_url)
            try:
                await show_retrieval(conn, user_id, query, mgr)
            finally:
                await mgr.close()


async def cmd_clear(pool: asyncpg.Pool, phone: str) -> None:
    async with pool.acquire() as conn:
        user_id = await _get_user_id(conn, phone)
        if not user_id:
            print(f"No user found for {phone}")
            return
        count = await conn.fetchval("SELECT COUNT(*) FROM messages WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM messages WHERE user_id = $1", user_id)
    print(f"Cleared {count} messages for {phone}")


async def cmd_summarize(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    api_url: str,
    phone: str,
    admin_token: str,
) -> None:
    resp = await http.post(
        f"{api_url}/admin/summarise",
        params={"phone": phone},
        headers={"X-Admin-Token": admin_token},
        timeout=60.0,
    )
    if resp.status_code == 404:
        print(f"No user found for {phone}")
        return
    resp.raise_for_status()
    result = resp.json()
    print(f"Summarisation complete — candidates found: {result.get('candidates_found', 0)}")

    async with pool.acquire() as conn:
        user_id = await _get_user_id(conn, phone)
        await show_state(conn, user_id, "After summarisation")


async def cmd_auto(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    api_url: str,
    phone: str,
    n: int,
    days: float,
    topic: str,
) -> None:
    """
    Conversational loop: Claude generates a user message, the real API responds,
    Claude sees that response and generates the next message — and so on.
    Messages are back-dated so they span `days` days ending at now.
    """
    client   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start    = datetime.now(tz=timezone.utc) - timedelta(days=days)
    interval = timedelta(days=days) / max(n, 1)

    print(f"\nRunning {n} exchanges over {days} days (topic: {topic})\n")

    history: list[dict] = []  # grows with each real exchange so Claude stays in context

    for i in range(n):
        ts       = start + interval * i
        user_msg = _generate_user_message(client, topic, "", history)
        reply    = await run_exchange(pool, http, api_url, phone, user_msg, ts)

        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant",  "content": reply})

    async with pool.acquire() as conn:
        user_id = await _get_user_id(conn, phone)
        await show_state(conn, user_id, f"Final state for {phone}")


async def cmd_scenario(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    api_url: str,
    phone: str,
    scenario_path: str,
) -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with open(scenario_path) as f:
        scenario = json.load(f)

    name   = scenario.get("name", "Unnamed")
    topic  = scenario.get("topic", "daily reminders")
    phases = scenario.get("phases", [])

    if "start_datetime" in scenario:
        base_time = datetime.fromisoformat(scenario["start_datetime"]).replace(tzinfo=timezone.utc)
    else:
        base_time = datetime.now(tz=timezone.utc) - timedelta(days=30)

    print(f"\nScenario: {name}  ({len(phases)} phases)\n")

    for phase in phases:
        label       = phase.get("label", "")
        day_offset  = phase.get("day_offset", 0)
        hour_offset = phase.get("hour_offset", 0)
        phase_ts    = base_time + timedelta(days=day_offset, hours=hour_offset)
        exchanges   = phase.get("exchanges", [])

        if label:
            print(f"\n── {label} ({phase_ts.strftime('%Y-%m-%d')}) ──")

        history: list[dict] = []

        for j, ex in enumerate(exchanges):
            ex_ts    = phase_ts + timedelta(minutes=j * 5)
            user_msg = ex.get("user", "")

            if user_msg == "auto":
                user_msg = _generate_user_message(client, topic, label, history)

            reply = await run_exchange(pool, http, api_url, phone, user_msg, ex_ts)
            history.append({"role": "user",      "content": user_msg})
            history.append({"role": "assistant",  "content": reply})

    async with pool.acquire() as conn:
        user_id = await _get_user_id(conn, phone)
        await show_state(conn, user_id, f"After scenario: {name}")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive the real /sms API endpoint with time-shifted messages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--phone",       required=True, help="User phone number (must be registered)")
    parser.add_argument("--api-url",     default=None,  help="API base URL (default: http://localhost:5000)")
    parser.add_argument("--db-url",      default=None,  help="Database URL (default: env var with db→localhost)")
    parser.add_argument("--emb-url",     default=None,  help="Embeddings URL (default: http://localhost:8002)")
    parser.add_argument("--admin-token", default=None,  help="Admin token for /admin/* endpoints (default: ADMIN_TOKEN env var)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--show",      action="store_true", help="Show current message state")
    group.add_argument("--clear",     action="store_true", help="Clear all history for the phone")
    group.add_argument("--summarize", action="store_true", help="Trigger one summarisation pass")
    group.add_argument("--auto",      type=int, metavar="N", help="Generate N exchanges (default: 15)")
    group.add_argument("--scenario",  metavar="FILE",        help="Load scenario from JSON file")

    parser.add_argument("--days",  type=float, default=30.0,
                        help="Time span in days for --auto (default: 30)")
    parser.add_argument("--topic", default="setting reminders for meetings, chores, errands, and events",
                        help="Topic for --auto message generation")
    parser.add_argument("--query", default="",
                        help="Query to show retrieval preview alongside --show")

    args = parser.parse_args()

    api_url     = args.api_url     or _api_url()
    db_url      = args.db_url      or _local_db_url()
    emb_url     = args.emb_url     or _local_embedding_url()
    admin_token = args.admin_token or os.getenv("ADMIN_TOKEN", "admin")

    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5, ssl=False)

    try:
        if args.show:
            await cmd_show(pool, args.phone, args.query, emb_url)

        elif args.clear:
            await cmd_clear(pool, args.phone)

        else:
            async with httpx.AsyncClient() as http:
                if args.summarize:
                    await cmd_summarize(pool, http, api_url, args.phone, admin_token)

                elif args.scenario:
                    await cmd_scenario(pool, http, api_url, args.phone, args.scenario)

                else:
                    n = args.auto if args.auto is not None else 15
                    await cmd_auto(pool, http, api_url, args.phone, n, args.days, args.topic)

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
