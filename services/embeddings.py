"""
Embedding Manager — calls the shared embed microservice via HTTP.

Only one ONNX Runtime instance runs (in the `embed` container), shared by
both `api` and `mcp`. This keeps CPU and RAM usage on the host much lower
than loading the model in-process in every container.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
EMBED_URL     = os.getenv("EMBED_URL", "http://embed:8002")


class EmbeddingManager:
    def __init__(self):
        self._client = httpx.AsyncClient(base_url=EMBED_URL, timeout=10.0)

    async def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for *text*."""
        response = await self._client.post("/embed", json={"text": text})
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()
