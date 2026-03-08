"""
Embedding Manager — runs BAAI/bge-small-en-v1.5 in-process via fastembed.

No separate container needed. The model is loaded once at startup and reused.
fastembed uses ONNX Runtime, so memory footprint is ~150 MB vs ~600 MB for TEI.
"""

import asyncio
import logging

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384


class EmbeddingManager:
    def __init__(self):
        self._model = TextEmbedding(EMBEDDING_MODEL)

    async def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for *text*."""
        def _run():
            return next(self._model.embed([text])).tolist()
        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        pass