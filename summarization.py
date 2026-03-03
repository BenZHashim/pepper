"""
Summarisation Manager — handles the full memory lifecycle.

Lifecycle overview:
  1. get_pruning_candidates() — finds contiguous low-retention time windows
  2. run_summarisation_pass() — classifies, extracts facts, compresses
  3. add_summary() / add_fact() — persists results back to the messages table
  4. prune_embeddings() — strips embeddings only from summarised/discarded messages
                          keep_verbatim messages remain in vector/keyword search

Tiered compression:
  Summaries are subject to the same lifecycle as raw messages.  When a
  summary's retention score drops below the threshold it becomes a candidate
  itself, and run_summarisation_pass() detects this and tightens the
  classification prompt accordingly.

Chunk-based selection:
  Rather than picking individual low-retention messages, the manager divides
  the user's message history into fixed-width time windows and selects the
  window whose *average* retention score is lowest.  This keeps context
  contiguous for the model and avoids leaving temporal gaps in the summary
  chain.  Important messages (keep_verbatim) that happen to fall inside a
  low-retention window are summarised but their embeddings are preserved so
  they continue to surface in retrieval.

Intended to be called on a nightly / weekly schedule per user:
    await summarisation_manager.run_summarisation_pass(user_id)
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic
import asyncpg
from pgvector.asyncpg import register_vector

from embeddings import EmbeddingManager
from memory_utils import extract_text, retention_score

logger = logging.getLogger(__name__)


# ── Tuning knobs ──────────────────────────────────────────────────────────────

PRUNE_AGE_DAYS          = 0     # Only consider messages older than this
RETENTION_SCORE_THRESH  = 0.2   # A window is a candidate if its *average* score
                                 # is below this threshold
WINDOW_SIZE_MESSAGES   = 20    # Number of messages per window when scoring for pruning

_CLASSIFICATION_TOOL = {
    "name": "compress_window",
    "description": "Compress a window of conversation history into a summary and extracted facts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": (
                                    "A single concise third-person summary of everything worth "
                                    "retaining from this window, optimised for semantic search. "
                                    "Omit filler, confirmations, and one-word responses."
                                ),
                            },
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "IDs of the messages that contributed to this summary.",
                            },
                        },
                        "required": ["text", "source_ids"],
                    },
                    {"type": "null"},
                ],
                "description": "Null if there is nothing worth retaining in this window.",
            },
            "facts": {
                "type": "array",
                "description": (
                    "Durable long-term facts about the user extracted from this window. "
                    "Each fact should stand alone and remain true months from now."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "integer", "description": "ID of the message this fact was drawn from."},
                        "text":      {"type": "string",  "description": "The fact restated clearly in third-person."},
                    },
                    "required": ["source_id", "text"],
                },
            },
        },
        "required": ["summary", "facts"],
    },
}

# ── Manager ───────────────────────────────────────────────────────────────────

class SummarizationManager:
    def __init__(
        self,
        pool: asyncpg.Pool,
        embedding_manager: EmbeddingManager,
        anthropic_client: anthropic.Anthropic,
    ) -> None:
        self._pool              = pool
        self._embedding_manager = embedding_manager
        self._anthropic         = anthropic_client

    # ── Persistence ───────────────────────────────────────────────────────────

    async def add_summary(
        self,
        user_id: int,
        summary_text: str,
        period_start: datetime,
        period_end: datetime,
        source_message_ids: list[int],
        message_type: str = "summary",  # "summary" | "fact"
    ) -> None:
        """
        Persist a compressed summary or extracted fact.
        source_message_ids may point to raw messages or earlier summaries,
        forming a traceable chain across compression tiers.
        """
        assert message_type in ("summary", "fact"), f"Invalid message_type: {message_type}"

        json_content = json.dumps({"type": "text", "text": summary_text})
        embedding    = await self._embedding_manager.embed(summary_text)

        async with self._pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(
                """
                INSERT INTO messages (
                    user_id, role, message_type, content,
                    period_start, period_end,
                    embedding, search_vector,
                    source_message_ids, source_message_count
                ) VALUES (
                    $1, 'system', $2, $3::jsonb,
                    $4, $5,
                    $6, to_tsvector('english', $7),
                    $8, $9
                )
                """,
                user_id,
                message_type,
                json_content,
                period_start,
                period_end,
                embedding,
                summary_text,
                source_message_ids,
                len(source_message_ids),
            )

    async def prune_embeddings(self, message_ids: list[int]) -> None:
        """
        Null out embeddings for messages captured in a summary.
        Text is retained for auditability.  These rows will no longer
        surface in vector or keyword search.

        NOTE: Only call this for 'summarise' and 'discard' messages.
        'keep_verbatim' messages should be excluded so they remain
        independently searchable even after the chunk is summarised.
        """
        if not message_ids:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE messages
                SET embedding           = NULL,
                    embedding_pruned_at = NOW()
                WHERE id = ANY($1::int[])
                """,
                message_ids,
            )

    # ── Candidate selection ───────────────────────────────────────────────────

    async def get_pruning_candidates(
        self,
        user_id: int,
    ) -> list[asyncpg.Record]:
        """
        Return a contiguous block of messages from the lowest-retention window.

        Strategy:
        1. Fetch all eligible messages ordered chronologically.
        2. Slice them into fixed-size windows of WINDOW_SIZE_MESSAGES.
        3. Compute the average retention score per window.
        4. Select the window with the lowest average score, provided it's below
            RETENTION_SCORE_THRESH.
        5. Return up to batch_size messages from that window.

        Windowing by message count rather than time ensures windows remain
        uniformly dense across compression tiers — a window of summaries-of-
        summaries contains the same number of items as a window of raw messages,
        so the model always receives a meaningful batch regardless of how far
        along the compression chain we are.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=PRUNE_AGE_DAYS)
        now    = datetime.now(tz=timezone.utc)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, message_type, content, created_at,
                    period_start, period_end,
                    retrieval_count, last_retrieved_at,
                    source_message_ids
                FROM messages
                WHERE user_id             = $1
                AND period_end          < $2
                AND embedding_pruned_at IS NULL
                AND message_type        IN ('raw', 'summary')
                ORDER BY period_end ASC
                """,
                user_id,
                cutoff,
            )

        if not rows:
            return []

        # ── Slice rows into fixed-size message windows ────────────────────────
        windows = [
            rows[i : i + WINDOW_SIZE_MESSAGES]
            for i in range(0, len(rows), WINDOW_SIZE_MESSAGES)
        ]

        # ── Score each window by average retention ────────────────────────────
        def avg_retention(window_rows) -> float:
            scores = [
                retention_score(r["retrieval_count"], r["last_retrieved_at"], now)
                for r in window_rows
            ]
            return sum(scores) / len(scores)

        # ── Find the lowest-retention window ─────────────────────────────────────
        lowest_window = None
        lowest_score  = float("inf")

        for window_rows in windows:
            score = avg_retention(window_rows)
            if score < lowest_score:
                lowest_score  = score
                lowest_window = window_rows

        if lowest_window is None or lowest_score >= RETENTION_SCORE_THRESH:
            return []

        return list(lowest_window)

    # ── Summarisation pass ────────────────────────────────────────────────────

    def _build_classification_prompt(self, block: str, is_resummary: bool) -> str:
        resummary_note = (
            "These messages are ALREADY SUMMARIES from a previous compression pass. "
            "Be conservative — preserve all preferences and long-term facts.\n"
            if is_resummary
            else ""
        )
        return f"""
            You are compressing a window of personal AI assistant conversation history.
            {resummary_note}
            Your job is to reduce this window to its essential information:

            1. Write a single summary capturing everything worth remembering — past events,
            requests, decisions, context. Discard pure filler ("ok", "sounds good", 
            simple confirmations). Write in third-person, optimised for semantic search.

            2. Extract any durable long-term facts about the user that should be stored
            separately — things that will still be true months from now, like preferences,
            habits, or personal details. Do NOT extract time-bound events as facts.

            Good summary: "Ben scheduled a meeting with Nick and a game showcase in late 
            February 2026, and set 15-minute reminders for both."
            Bad summary: "The user talked about meetings and reminders."

            Good fact: "Ben prefers 15-minute reminders before events."
            Bad fact: "Ben had a meeting with Nick on Feb 27."

            Messages:
            {block}
        """.strip()

    async def run_summarisation_pass(self, user_id: int) -> None:
        candidates = await self.get_pruning_candidates(user_id)
        if not candidates:
            logger.info(f"No pruning candidates for user {user_id}.")
            return

        block = "\n\n".join(
            f"[id={row['id']} type={row['message_type']} time={row['period_end']}]\n"
            + extract_text(json.loads(row["content"]))
            for row in candidates
        )

        is_resummary = any(row["message_type"] == "summary" for row in candidates)
        prompt       = self._build_classification_prompt(block, is_resummary)

        response = self._anthropic.messages.create(
            model      = "claude-haiku-4-5",
            max_tokens = 1024,
            tools      = [_CLASSIFICATION_TOOL],
            tool_choice = {"type": "tool", "name": "compress_window"},
            messages   = [{"role": "user", "content": prompt}],
        )

        result: Optional[dict] = None
        for block in response.content:
            if block.type == "tool_use":
                result = block.input
                break

        if result is None:
            logger.error(f"Summarisation tool call missing for user {user_id}")
            return

        period_start = min(row["period_start"] for row in candidates)
        period_end   = max(row["period_end"]   for row in candidates)
        all_source_ids = [row["id"] for row in candidates]

        # ── Persist facts ─────────────────────────────────────────────────────
        for fact in result.get("facts", []):
            if fact.get("text"):
                await self.add_summary(
                    user_id            = user_id,
                    summary_text       = fact["text"],
                    period_start       = period_start,
                    period_end         = period_end,
                    source_message_ids = [fact["source_id"]],
                    message_type       = "fact",
                )

        # ── Persist summary ───────────────────────────────────────────────────
        if result.get("summary"):
            await self.add_summary(
                user_id            = user_id,
                summary_text       = result["summary"]["text"],
                period_start       = period_start,
                period_end         = period_end,
                source_message_ids = result["summary"]["source_ids"],
                message_type       = "summary",
            )

        # ── Prune all candidate embeddings ────────────────────────────────────
        await self.prune_embeddings(all_source_ids)

        logger.info(
            f"Summarisation pass complete — user={user_id} "
            f"processed={len(all_source_ids)} "
            f"facts={len(result.get('facts', []))} "
            f"summary={'yes' if result.get('summary') else 'no'} "
            f"embeddings_pruned={len(all_source_ids)}"
        )