"""
embed_service.py

Lightweight embedding microservice — loads BAAI/bge-small-en-v1.5 once via
fastembed and serves it over HTTP on port 8002.

Both `mcp` and `api` call this instead of loading the model themselves,
so only one ONNX Runtime instance runs on the server at a time.
"""

import asyncio
import os

from fastapi import FastAPI
from fastembed import TextEmbedding
from pydantic import BaseModel

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

app = FastAPI()

# Loaded once at startup, reused for all requests
_model = TextEmbedding(EMBEDDING_MODEL)


class EmbedRequest(BaseModel):
    text: str


@app.post("/embed")
async def embed(req: EmbedRequest) -> list[float]:
    def _run() -> list[float]:
        return next(_model.embed([req.text])).tolist()
    return await asyncio.to_thread(_run)


@app.get("/health")
async def health():
    return {"status": "ok"}
