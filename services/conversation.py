"""
Conversation Manager — PostgreSQL backed with hybrid memory retrieval.

Retrieval strategy:
  1. RECENT_WINDOW raw messages are always included verbatim (live context).
  2. For older messages (raw + summaries + facts) three signals are blended:

       score = W_SEMANTIC * cosine_similarity   (pgvector <=> operator)
             + W_KEYWORD  * bm25_rank           (Postgres ts_rank)
             + W_RECENCY  * exponential_decay   (age-based)

     The top SEMANTIC_K survivors are merged with the recent window and
     returned chronologically.

  3. Retrieval counts are incremented for every message surfaced by search,
     feeding the SummarisationManager's pruning decisions.

Memory lifecycle (summarisation, pruning) lives in summarisation_manager.py.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

from .embeddings import EmbeddingManager
from .memory_utils import extract_text, to_tsquery, run_hybrid_search
from models import UserRepository, MessageRepository, Message

logger = logging.getLogger(__name__)


# ── Tuning knobs ──────────────────────────────────────────────────────────────

RECENT_WINDOW = 20      # Last 20 SMS exchanges — continuity, ~1-2 conversations worth
SEMANTIC_K    = 10      # 10 older results — enough diversity without noise
SESSION_TIMEOUT = 60    # Minutes — used for active_count() only

# Retrieval blend weights (independent, do not need to sum to 1)
W_SEMANTIC = 0.6
W_KEYWORD  = 0.3
W_RECENCY  = 0.1

CANDIDATE_MULTIPLIER = 10   # Pulls SEMANTIC_K * CANDIDATE_MULTIPLIER per search leg


# ── Manager ───────────────────────────────────────────────────────────────────

class ConversationManager:
    def __init__(
        self,
        pool: asyncpg.Pool,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self._pool              = pool
        self._embedding_manager = embedding_manager
        self._pending_registration: dict[str, dict] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────
    async def _get_user_id(self, conn: asyncpg.Connection, phone: str) -> Optional[int]:
        user = await UserRepository(conn).get_by_phone(phone)
        return user.id if user else None

    # ── Search legs ───────────────────────────────────────────────────────────

    async def _run_semantic_leg(
        self,
        conn: asyncpg.Connection,
        query: str,
        user_id: int,
        exclude_ids: list[int],
        n_candidates: int,
        candidates: dict[int, dict],
    ) -> None:
        """
        Fetch the closest messages by cosine distance and merge into candidates.
        Only considers rows with a live (non-pruned) embedding.
        """
        query_vec = await self._embedding_manager.embed(query)
        rows = await conn.fetch(
            """
            SELECT id, role, message_type, content, created_at,
                   1 - (embedding <=> $1::vector) AS cosine_sim
            FROM messages
            WHERE user_id             = $2
              AND id                 != ALL($3::int[])
              AND embedding           IS NOT NULL
              AND embedding_pruned_at IS NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $4
            """,
            query_vec,
            user_id,
            exclude_ids,
            n_candidates,
        )
        for row in rows:
            candidates.setdefault(
                row["id"],
                {"row": row, "sem": 0.0, "kw": 0.0, "rec": 0.0},
            )["sem"] = float(row["cosine_sim"])

    async def _run_keyword_leg(
        self,
        conn: asyncpg.Connection,
        query: str,
        user_id: int,
        exclude_ids: list[int],
        n_candidates: int,
        candidates: dict[int, dict],
    ) -> None:
        """
        Run a Postgres full-text search and merge ts_rank scores into candidates.
        Silently skips if the query produces no usable tokens or if Postgres
        rejects the tsquery (e.g. all stop words).
        """
        tsquery = to_tsquery(query)
        if not tsquery:
            return
        try:
            rows = await conn.fetch(
                """
                SELECT id, role, message_type, content, created_at,
                       ts_rank(search_vector, to_tsquery('english', $1)) AS kw_rank
                FROM messages
                WHERE user_id             = $2
                  AND id                 != ALL($3::int[])
                  AND embedding_pruned_at IS NULL
                  AND search_vector      @@ to_tsquery('english', $1)
                ORDER BY kw_rank DESC
                LIMIT $4
                """,
                tsquery,
                user_id,
                exclude_ids,
                n_candidates,
            )
        except Exception as exc:
            logger.warning(f"Keyword leg failed (tsquery={tsquery!r}): {exc}")
            return

        for row in rows:
            entry = candidates.setdefault(
                row["id"],
                {"row": row, "sem": 0.0, "kw": 0.0, "rec": 0.0},
            )
            entry["kw"]  = float(row["kw_rank"])
            entry["row"] = entry["row"] or row

    # ── Public API ────────────────────────────────────────────────────────────

    async def add_message(self, phone: str, role: str, content: str | list | dict) -> None:
        """
        Persist a new raw message with its embedding and tsvector.
        period_start and period_end are both set to NOW() via the DB trigger.
        """
        json_content   = json.dumps(content if not isinstance(content, str)
                                    else {"type": "text", "text": content})
        text_for_index = extract_text(content)
        embedding      = (
            await self._embedding_manager.embed(text_for_index)
            if text_for_index.strip()
            else None
        )

        async with self._pool.acquire() as conn:
            await register_vector(conn)
            user_id = await self._get_user_id(conn, phone)
            if user_id is None:
                logger.warning(f"add_message: no user found for phone {phone}")
                return

            await MessageRepository(conn).add(user_id, role, json_content, embedding, text_for_index)

    async def get_history(
        self,
        phone: str,
        query: Optional[str] = None,
    ) -> list[dict]:
        """
        Build the message list to pass to Claude.

        Always includes the RECENT_WINDOW newest raw messages.
        When a query is provided, runs semantic and keyword search legs over
        all older messages (raw, summaries, facts), blends scores, and prepends
        the top SEMANTIC_K results in chronological order.

        Summaries and facts are injected with a [Memory context] prefix so
        Claude treats them as established background, not conversation turns.
        Retrieval counts are incremented for every message surfaced by search.
        """
        async with self._pool.acquire() as conn:
            await register_vector(conn)
            user_id = await self._get_user_id(conn, phone)
            if user_id is None:
                return []
            user_tz = await conn.fetchval("SELECT timezone FROM users WHERE id = $1", user_id)

            # 1. Recent raw messages — always included verbatim
            recent_messages = await MessageRepository(conn).get_recent(user_id, RECENT_WINDOW)
            recent_ids: set[int] = {msg.id for msg in recent_messages}

            # 2. Hybrid search over all older message types
            search_rows = []
            if query and query.strip():
                search_rows = await run_hybrid_search(
                    conn=conn,
                    embedding_manager=self._embedding_manager,
                    query=query,
                    user_id=user_id,
                    exclude_ids=list(recent_ids) or [0],
                    n_candidates=SEMANTIC_K * CANDIDATE_MULTIPLIER,
                    w_semantic=W_SEMANTIC,
                    w_keyword=W_KEYWORD,
                    w_recency=W_RECENCY,
                    semantic_k=SEMANTIC_K,
                )

            # 3. Merge and sort chronologically
            all_messages: dict[int, Message] = {msg.id: msg for msg in recent_messages}
            for row in search_rows:
                msg = Message.model_validate(dict(row))
                if msg.id not in all_messages:
                    all_messages[msg.id] = msg

            sorted_messages = sorted(all_messages.values(), key=lambda m: m.period_end)

        # 4. Deserialise to Claude message dicts
        history: list[dict] = []
        for msg in sorted_messages:
            content = msg.content
            if isinstance(content, dict) and content.get("type") == "text":
                content = content["text"]

            period_end_local = msg.period_end.astimezone(ZoneInfo(user_tz))
            timestamp = period_end_local.strftime("%Y-%m-%d %H:%M %Z")
            header = f"[id={msg.id} time={timestamp}]"

            role = msg.role
            if msg.message_type in ("summary", "fact"):
                role = "user"

            history.append({"role": role, "content": f"{header} {content}"})

        return history

    async def clear_history(self, phone: str) -> None:
        """Delete all conversation history for a user."""
        async with self._pool.acquire() as conn:
            user_id = await self._get_user_id(conn, phone)
            if user_id is None:
                return
            await MessageRepository(conn).delete_all(user_id)

    async def active_count(self) -> int:
        """Count users with at least one message in the last SESSION_TIMEOUT minutes."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=SESSION_TIMEOUT)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(DISTINCT user_id) AS count
                FROM messages
                WHERE created_at > $1
                """,
                cutoff,
            )
        return row["count"] if row else 0

    # ── Pending registration (in-memory, transient) ───────────────────────────

    def set_pending_registration(self, phone: str) -> None:
        self._pending_registration[phone] = {"name": None, "timezone": None}

    def get_pending_registration(self, phone: str) -> Optional[dict]:
        return self._pending_registration.get(phone)

    def update_pending_registration(self, phone: str, **kwargs) -> None:
        if phone in self._pending_registration:
            self._pending_registration[phone].update(kwargs)

    def clear_pending_registration(self, phone: str) -> None:
        self._pending_registration.pop(phone, None)
