"""
storage/qdrant_store.py

Qdrant implementation of the VectorStore protocol.

Like Neo4jStore, this is a thin facade — it delegates upserts to the existing
ingestion.writers.VectorWriter and exposes a search() method that wraps the
QdrantClient query API. The two paths share one QdrantClient so we don't open
two HTTP connections to the same server.

Adding a new vector backend (e.g. pgvector, Pinecone, Chroma, LadybugDB) means
writing a new file like this one — no changes to the pipeline or agent.

Search returns the standard result shape so callers don't care which store
produced the data:

  uuid, name, qualified_name, file_path, line_start, line_end,
  signature, docstring, full_body, repo_id, score, source="vector"
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, NamedVector, Query

from config import Config
from ingestion.models import EmbedDoc
from ingestion.writers.vector_writer import VectorWriter

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Implements storage.protocols.VectorStore over a Qdrant database.

    Owns one QdrantClient. The internal VectorWriter shares it for upserts.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg    = cfg
        self._client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

        # Reuse VectorWriter for upserts (it knows how to ensure the collection
        # exists with the right vector size). Replace its client so we don't
        # open two connections to the same server.
        self._writer = VectorWriter(cfg)
        self._writer._client = self._client

    # ----- lifecycle -------------------------------------------------------- #

    def __enter__(self) -> "QdrantStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ----- writes ----------------------------------------------------------- #

    def upsert(self, docs: list[EmbedDoc], vectors: list[list[float]]) -> None:
        self._writer.write(docs, vectors)

    # ----- reads ------------------------------------------------------------ #

    def delete_repo(self, repo_id: str) -> None:
        self._client.delete(
            collection_name=self._cfg.collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))])
            ),
        )
        logger.info("Deleted Qdrant vectors for repo_id=%s", repo_id)

    def search(self, vector: list[float], repo_id: str | None, top_k: int) -> list[dict]:
        payload_filter = None
        if repo_id:
            payload_filter = Filter(
                must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]
            )

        results = self._client.query_points(
            collection_name=self._cfg.collection_name,
            query=vector,
            query_filter=payload_filter,
            limit=top_k,
            with_payload=True,
        )
        return [_hit_to_dict(h) for h in results.points]


def _hit_to_dict(hit) -> dict:
    p = hit.payload or {}
    return {
        "uuid":           hit.id,
        "name":           p.get("name"),
        "qualified_name": p.get("qualified_name"),
        "file_path":      p.get("file_path"),
        "line_start":     p.get("line_start"),
        "line_end":       p.get("line_end"),
        "signature":      p.get("signature"),
        "docstring":      p.get("docstring"),
        "full_body":      p.get("full_body"),
        "repo_id":        p.get("repo_id"),
        "score":          hit.score,
        "source":         "vector",
    }
