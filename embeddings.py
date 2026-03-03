"""
Embedding Manager — thin async wrapper around the TEI sidecar.

A single shared instance is created at startup and passed to any class
that needs to generate embeddings (ConversationManager, sync_mcp_tools, etc.)
This avoids each module creating its own httpx.AsyncClient.
"""

import os
import logging

import httpx

logger = logging.getLogger(__name__)

EMBEDDING_URL   = os.getenv("EMBEDDING_URL", "http://embeddings:80")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384


class EmbeddingManager:
    def __init__(self, url: str = EMBEDDING_URL):
        self._http = httpx.AsyncClient(base_url=url)

    async def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for *text*."""
        response = await self._http.post(
            "/embeddings",
            json={"model": EMBEDDING_MODEL, "input": text},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._http.aclose()