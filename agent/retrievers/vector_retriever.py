"""
agent/retrievers/vector_retriever.py

Agent-side wrapper around a VectorStore plus an Embedder. The retriever does
not talk to Qdrant (or any other backend) directly — it goes through
VectorStore so that swapping vector backends is a one-line change in the
FastAPI lifespan.

Flow:
  1. Embed the user's query using the same model that was used at ingestion
     time. The query vector must live in the same embedding space as the
     stored vectors or cosine scores are meaningless.
  2. Hand the vector to VectorStore.search() which returns the top-k matching
     payloads in the standard result shape.

Return shape per result:
  uuid, name, qualified_name, file_path, line_start, line_end,
  signature, docstring, full_body, repo_id, score, source="vector"
"""

from __future__ import annotations

from config import Config
from ingestion.embedder import Embedder
from storage.protocols import VectorStore


class VectorRetriever:

    def __init__(self, cfg: Config, store: VectorStore) -> None:
        self._cfg      = cfg
        self._store    = store
        self._embedder = Embedder(cfg)

    def search(self, query: str, repo_id: str | None = None, top_k: int = 10) -> list[dict]:
        vector = self._embedder.embed_one(query)
        return self._store.search(vector, repo_id, top_k)

    def close(self) -> None:
        self._embedder.close()
