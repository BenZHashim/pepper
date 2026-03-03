"""
Shared utility functions for the memory/conversation system.
No database or network dependencies — pure computation only.
"""

import math
import re
import logging
from datetime import datetime, timezone
from typing import Optional
import asyncpg

from embeddings import EmbeddingManager


# ── Scoring ───────────────────────────────────────────────────────────────────

HALF_LIFE_HOURS       = 12.0    # Retrieval recency decay (short — for search ranking)
RETENTION_HALF_LIFE_DAYS = 14.0 # Retention recency decay (longer — for pruning decisions)
RETENTION_BOOST       = 2.0     # Additive boost to retention score so that messages are not immediately pruned after their first retrieval
logger = logging.getLogger(__name__)


def recency_score(ts: datetime, now: datetime, half_life_hours: float = HALF_LIFE_HOURS) -> float:
    """
    Exponential decay score based on age.
    score = 2^(-age_hours / half_life_hours)
    Returns 1.0 for brand-new, approaches 0.0 for very old.
    """
    age_hours = max((now - ts).total_seconds() / 3600.0, 0.0)
    return math.pow(2.0, -age_hours / half_life_hours)


def retention_score(
    retrieval_count: int,
    last_retrieved_at: Optional[datetime],
    now: datetime,
) -> float:
    """
    Combined retention score used by the pruning job.
    Multiplies retrieval frequency by a long-horizon recency decay so that
    messages retrieved often but long ago still get swept eventually.
    score = (RETENTION_BOOST + retrieval_count) * 2^(-age_days / RETENTION_HALF_LIFE_DAYS)
    """
    return (RETENTION_BOOST + retrieval_count) * recency_score(
        last_retrieved_at,
        now,
        half_life_hours=RETENTION_HALF_LIFE_DAYS * 24,
    )


# ── Content extraction ────────────────────────────────────────────────────────

def extract_text(content: str | dict | list | None) -> str:
    """
    Pull plain text out of whatever shape a content value arrives in.
    Handles raw strings, single content blocks, and Claude content block lists.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or item.get("content", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(filter(None, parts))
    return ""


# ── Full-text search ──────────────────────────────────────────────────────────

def to_tsquery(text: str) -> str:
    """
    Convert free text into a Postgres tsquery expression.

    Strips non-alphanumeric characters, drops empty tokens, and joins
    with ' & ' (AND semantics).  Returns an empty string if nothing
    usable remains — callers must guard against the empty case.
    """
    tokens = [re.sub(r"[^\w]", "", t) for t in text.split()]
    tokens = [t for t in tokens if t]
    return " & ".join(tokens)


async def run_semantic_leg(
    conn: asyncpg.Connection,
    embedding_manager: EmbeddingManager,
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
    query_vec = await embedding_manager.embed(query)
    rows = await conn.fetch(
        """
        SELECT id, role, message_type, content, created_at, period_end,
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


async def run_keyword_leg(
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
            SELECT id, role, message_type, content, created_at, period_end,
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
        entry["kw"] = float(row["kw_rank"])
        entry["row"] = entry["row"] or row


async def run_hybrid_search(
    conn: asyncpg.Connection,
    embedding_manager: EmbeddingManager,
    query: str,
    user_id: int,
    exclude_ids: list[int],
    n_candidates: int,
    w_semantic: float,
    w_keyword: float,
    w_recency: float,
    semantic_k: int,
) -> list[asyncpg.Record]:
    """
    Orchestrates both search legs, blends scores, and returns the top
    semantic_k results sorted by combined score (descending).
    """
    now = datetime.now(tz=timezone.utc)
    candidates: dict[int, dict] = {}

    if w_semantic > 0:
        await run_semantic_leg(
            conn, embedding_manager, query, user_id,
            exclude_ids, n_candidates, candidates,
        )
    if w_keyword > 0:
        await run_keyword_leg(
            conn, query, user_id,
            exclude_ids, n_candidates, candidates,
        )
    if w_recency > 0:
        for entry in candidates.values():
            entry["rec"] = recency_score(entry["row"]["period_end"], now)

    scored = [
        (
            w_semantic * e["sem"] + w_keyword * e["kw"] + w_recency * e["rec"],
            e["row"],
        )
        for e in candidates.values()
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:semantic_k]]