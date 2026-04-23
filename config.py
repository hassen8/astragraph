import os
from dataclasses import dataclass

@dataclass
class Config:
    # Neo4j
    neo4j_uri:      str = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user:     str = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    # Vector store (Qdrant)
    qdrant_host:     str = os.getenv("QDRANT_HOST",      "localhost")
    qdrant_port:     int = int(os.getenv("QDRANT_PORT",  "6333"))
    collection_name: str = os.getenv("COLLECTION",       "codebase")

    # Embeddings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embed_model:    str = os.getenv("EMBED_MODEL",    "text-embedding-3-large")
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))

    # Ingestion
    checkpoint_path: str = os.getenv("CHECKPOINT_PATH", "./ingestion_checkpoint.json")
    chunk_line_limit: int = int(os.getenv("CHUNK_LINE_LIMIT", "60"))
    overlap_lines:   int = int(os.getenv("OVERLAP_LINES",    "20"))
