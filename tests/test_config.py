import os
import importlib
import pytest
import config
from config import Config

def test_config_defaults():
    # Ensure environment variables are clear for testing defaults
    if "NEO4J_URI" in os.environ:
        del os.environ["NEO4J_URI"]
        
    config = Config()
    assert config.neo4j_uri == "bolt://localhost:7687"
    assert config.neo4j_user == "neo4j"
    assert config.neo4j_password == "password"
    assert config.chroma_path == "./chroma_db"
    assert config.collection_name == "codebase"
    assert config.embed_model == "text-embedding-3-large"
    assert config.embed_batch_size == 64
    assert config.checkpoint_path == "./ingestion_checkpoint.json"
    assert config.chunk_line_limit == 60
    assert config.overlap_lines == 20

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://remote:7687")
    monkeypatch.setenv("EMBED_BATCH_SIZE", "128")
    monkeypatch.setenv("CHUNK_LINE_LIMIT", "100")
    
    importlib.reload(config)
    
    c = config.Config()
    assert c.neo4j_uri == "bolt://remote:7687"
    assert c.embed_batch_size == 128
    assert c.chunk_line_limit == 100
