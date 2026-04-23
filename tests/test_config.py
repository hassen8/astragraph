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
    assert config.qdrant_host == "localhost"
    assert config.qdrant_port == 6333
    assert config.collection_name == "codebase"
    assert config.embed_model == "all-MiniLM-L6-v2"
    assert config.embed_batch_size == 64
    assert config.llm_provider == "anthropic"
    assert config.llm_model == "claude-sonnet-4-6"
    assert config.fulltext_index == "functionText"
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
