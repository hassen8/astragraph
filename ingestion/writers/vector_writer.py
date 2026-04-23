"""
ingestion/writers/vector_writer.py

Writes EmbedDocs and their embedding vectors to Qdrant.

Qdrant stores each document as a "point" with three parts:
  - id       — the UUID string from EmbedDoc, linking back to the Neo4j node
  - vector   — the float list produced by the Embedder
  - payload  — the metadata dict (file_path, name, language, repo_id, etc.)

All writes use upsert — if a point with the same id already exists it is
overwritten in place. Re-running the pipeline on the same repo is safe.

Usage:
    writer = VectorWriter(cfg)
    docs   = [make_function_embed_doc(fn) for fn in functions if should_embed(fn)]
    vecs   = embedder.embed(docs)
    writer.write(docs, vecs)
    writer.close()

Or as a context manager:
    with VectorWriter(cfg) as writer:
        writer.write(docs, vecs)
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import Config
from ..models import EmbedDoc

logger = logging.getLogger(__name__)

# Embedding dimensions per model.
# Qdrant collections are created with a fixed vector size — changing the model
# after a collection exists requires deleting and recreating the collection.
_MODEL_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2":          384,
    "all-mpnet-base-v2":          768,
    "text-embedding-3-small":    1536,
    "text-embedding-3-large":    3072,
    "text-embedding-ada-002":    1536,
}
_DEFAULT_DIMS = 384  # fallback for unknown models


class VectorWriter:
    """
    Manages a Qdrant client and upserts EmbedDocs into a named collection.

    The collection is created automatically on first use if it doesn't exist.
    Vector size is inferred from cfg.embed_model — if you switch models you
    must delete the collection first (Qdrant enforces a fixed vector size).
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg    = cfg
        self._client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
        self._collection = cfg.collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Create the Qdrant collection if it doesn't already exist.

        Vector size is looked up from _MODEL_DIMS. If the model isn't in the
        table, falls back to 384 — override _DEFAULT_DIMS if needed.

        Distance metric: Cosine — standard for sentence-transformer embeddings.
        Dot product would be faster but requires normalized vectors; cosine
        handles both normalized and unnormalized inputs safely.
        """
        existing = {c.name for c in self._client.get_collections().collections}

        if self._collection not in existing:
            dims = _MODEL_DIMS.get(self._cfg.embed_model, _DEFAULT_DIMS)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
            )
            logger.info(
                "Created Qdrant collection '%s' (dims=%d, model=%s)",
                self._collection, dims, self._cfg.embed_model,
            )
        else:
            logger.debug("Using existing Qdrant collection '%s'", self._collection)

    def write(self, docs: list[EmbedDoc], embeddings: list[list[float]]) -> None:
        """
        Upsert a batch of EmbedDocs with their embedding vectors.

        docs[i] and embeddings[i] must correspond — the Embedder guarantees
        this alignment since it returns one vector per input doc.

        Batched at cfg.embed_batch_size to avoid large payloads in a single
        HTTP request to Qdrant.
        """
        if not docs:
            return

        assert len(docs) == len(embeddings), (
            f"docs/embeddings length mismatch: {len(docs)} vs {len(embeddings)}"
        )

        batch_size = self._cfg.embed_batch_size
        total      = len(docs)

        for i in range(0, total, batch_size):
            batch_docs = docs[i : i + batch_size]
            batch_vecs = embeddings[i : i + batch_size]

            points = [
                PointStruct(
                    id=doc.uuid,       # UUID string — links back to Neo4j node
                    vector=vec,
                    payload={
                        **doc.metadata,
                        "entity_type": doc.entity_type,
                        "text":        doc.text,   # store text so retrieval returns it directly
                    },
                )
                for doc, vec in zip(batch_docs, batch_vecs)
            ]

            self._client.upsert(collection_name=self._collection, points=points)
            logger.debug(
                "Upserted %d/%d vectors to '%s'",
                min(i + batch_size, total), total, self._collection,
            )

    def count(self) -> int:
        """Return the number of points currently in the collection."""
        return self._client.count(collection_name=self._collection).count

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> VectorWriter:
        return self

    def __exit__(self, *_) -> None:
        self.close()
