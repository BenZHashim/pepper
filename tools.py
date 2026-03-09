"""
Tool Manager — MCP tool fetching, syncing, and relevance-filtered retrieval.

Handles:
  - Fetching tools from the FastMCP server in Anthropic format
  - Syncing tools into Postgres with embeddings and tsvectors
  - Retrieving only tools relevant to a given query via hybrid search
    (same semantic + keyword + recency-free blending as ConversationManager)
"""

import json
import logging
from typing import Optional

import asyncpg
from fastmcp import Client as MCPClient
from pgvector.asyncpg import register_vector
from memory_utils import to_tsquery
from embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────

# Maximum tools to pass to Claude when filtering by relevance.
# Set to None to return all tools (disables filtering).
TOP_K_TOOLS = None

# Blend weights — same pattern as ConversationManager.
# W_RECENCY is intentionally absent; tools don't have an age dimension.
W_SEMANTIC = 0.65
W_KEYWORD  = 0.35

# Over-fetch multiplier before re-ranking
CANDIDATE_MULTIPLIER = 3


class ToolManager:
    def __init__(self, pool: asyncpg.Pool, embeddings: EmbeddingManager, mcp_url: str):
        self._pool      = pool
        self._embeddings = embeddings
        self._mcp_url   = mcp_url

    # ── MCP communication ─────────────────────────────────────────────────────

    async def fetch_from_mcp(self) -> list[dict]:
        """
        Fetch all tools from the FastMCP server and return them in
        Anthropic format: {name, description, input_schema}.
        """
        async with MCPClient(self._mcp_url) as client:
            tools = await client.list_tools()

        return [
            {
                "name":         tool.name,
                "description":  tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]

    async def call(self, tool_name: str, tool_input: dict) -> str:
        """
        Call a single tool on the FastMCP server and return the result
        as a plain string.
        """
        async with MCPClient(self._mcp_url) as client:
            result = await client.call_tool(tool_name, tool_input)

        if result is None:
            return "[]"

        if hasattr(result, "content") and result.content:
            for item in result.content:
                if hasattr(item, "text"):
                    return item.text

        if hasattr(result, "structured_content") and result.structured_content:
            inner = result.structured_content.get("result", result.structured_content)
            return json.dumps(inner)

        if isinstance(result, list):
            parts = []
            for item in result:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item))
            return "\n".join(parts) if parts else "[]"

        if isinstance(result, str):
            return result

        return "[]"

    # ── Database sync ─────────────────────────────────────────────────────────

    async def sync(self) -> None:
        """
        Fetch all tools from MCP, embed name + description, and upsert
        into the tools table. Safe to call on every startup.
        """
        tools = await self.fetch_from_mcp()

        async with self._pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute("TRUNCATE TABLE tools")  
            for tool in tools:
                text      = f"{tool['name']} {tool['description']}"
                embedding = await self._embeddings.embed(text)

                await conn.execute(
                    """
                    INSERT INTO tools (name, description, input_schema, embedding, search_vector)
                    VALUES ($1, $2, $3::jsonb, $4, to_tsvector('english', $5))
                    ON CONFLICT (name) DO UPDATE SET
                        description   = EXCLUDED.description,
                        input_schema  = EXCLUDED.input_schema,
                        embedding     = EXCLUDED.embedding,
                        search_vector = EXCLUDED.search_vector
                    """,
                    tool["name"],
                    tool["description"],
                    json.dumps(tool["input_schema"]),
                    embedding,
                    text,
                )

        logger.info(f"Synced {len(tools)} tools to database")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def get_relevant_tools(self, query: Optional[str] = None) -> list[dict]:
        """
        Return tools in Anthropic format, filtered by relevance to *query*.

        When query is None or TOP_K_TOOLS is None, all tools are returned
        with no filtering (safe fallback).

        Scoring:
          score = W_SEMANTIC * cosine_similarity + W_KEYWORD * ts_rank
        """
        if query is None or TOP_K_TOOLS is None:
            return await self.fetch_from_mcp()

        async with self._pool.acquire() as conn:
            await register_vector(conn)
            candidates: dict[str, dict] = {}   # name → {row, sem, kw}
            n_candidates = TOP_K_TOOLS * CANDIDATE_MULTIPLIER

            # -- Semantic leg --
            if W_SEMANTIC > 0:
                await self._run_semantic_leg(conn, query, n_candidates, candidates)

            # -- Keyword leg --
            if W_KEYWORD > 0:
                await self._run_keyword_leg(conn, query, n_candidates, candidates)

        # Blend and rank
        scored = [
            (W_SEMANTIC * e["sem"] + W_KEYWORD * e["kw"], e["row"])
            for e in candidates.values()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_rows = [row for _, row in scored[:TOP_K_TOOLS]]
        for row in top_rows:
            logger.info(f"Relevant tool: {row['name']}")
        return [
            {
                "name":         row["name"],
                "description":  row["description"],
                "input_schema": json.loads(row["input_schema"]),
            }
            for row in top_rows
        ]

    async def _run_semantic_leg(
        self,
        conn: asyncpg.Connection,
        query: str,
        n_candidates: int,
        candidates: dict,
    ) -> None:
        """Query pgvector for tools closest to *query* by cosine distance."""
        query_vec = await self._embeddings.embed(query)
        rows = await conn.fetch(
            """
            SELECT name, description, input_schema,
                   1 - (embedding <=> $1::vector) AS cosine_sim
            FROM tools
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            query_vec,
            n_candidates,
        )
        for row in rows:
            candidates.setdefault(row["name"], {
                "row": row, "sem": 0.0, "kw": 0.0
            })["sem"] = float(row["cosine_sim"])

    async def _run_keyword_leg(
        self,
        conn: asyncpg.Connection,
        query: str,
        n_candidates: int,
        candidates: dict,
    ) -> None:
        """Run Postgres full-text search and merge ts_rank scores into candidates."""
        tsquery = to_tsquery(query)
        if not tsquery:
            return

        rows = await conn.fetch(
            """
            SELECT name, description, input_schema,
                   ts_rank(search_vector, to_tsquery('english', $1)) AS kw_rank
            FROM tools
            WHERE search_vector @@ to_tsquery('english', $1)
            ORDER BY kw_rank DESC
            LIMIT $2
            """,
            tsquery,
            n_candidates,
        )
        for row in rows:
            entry = candidates.setdefault(row["name"], {
                "row": row, "sem": 0.0, "kw": 0.0
            })
            entry["kw"]  = float(row["kw_rank"])
            entry["row"] = entry["row"] or row