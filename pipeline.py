"""
ingestion/pipeline.py

Orchestrates the full two-pass ingestion run.

Pass 1 — per file:
  - Parse source
  - Extract entities (module, classes, functions)
  - Resolve Pass 1 relationships (DEFINED_IN, METHOD_OF, INHERITS, IMPORTS)
  - Write entities + Pass 1 relationships to Neo4j
  - Collect raw call captures (CALLS + CALLS_UNKNOWN) for Pass 2

Pass 2 — after all files:
  - Build repo-wide function lookup from all extracted FunctionNodes
  - Resolve CALLS_UNKNOWN entries into CALLS where possible
  - Write Pass 2 relationships to Neo4j
  - Embed and write all Tier-1 entities to vector store
"""

from __future__ import annotations

import logging
from pathlib import Path

from tree_sitter import Language, Node

from .extractor import extract_file, ENTITY_QUERIES
from .models import FunctionNode, ClassNode, ModuleNode, Relationship
from .relationships import RelationshipResolver, RelationshipWriter, build_repo_fn_lookup
from .walker import walk_repo
from .writers.neo4j_writer import Neo4jWriter
from .writers.vector_writer import VectorWriter
from .embedder import Embedder
from .chunker import chunk_function, chunk_class, chunk_module
from .tiering import embedding_tier, should_skip_file
from ..graph.schema import init_schema
from ..config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parser setup  (import from parser.py in real code)
# ---------------------------------------------------------------------------

def _get_lang_obj(language: str) -> Language:
    """Return the tree-sitter Language object for a given language string."""
    import tree_sitter_python as tspython
    LANGUAGE_MAP = {
        "python": Language(tspython.language()),
    }
    return LANGUAGE_MAP[language]


def _parse_source(source: bytes, language: str):
    """Parse source bytes into (root_node, source) using tree-sitter."""
    from tree_sitter import Parser
    lang_obj = _get_lang_obj(language)
    parser   = Parser(lang_obj)
    tree     = parser.parse(source)
    return tree.root_node, lang_obj


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:

    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self.neo4j_writer = Neo4jWriter(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
        self.rel_writer   = RelationshipWriter(self.neo4j_writer.driver)
        self.embedder     = Embedder(cfg)
        self.vector_writer = VectorWriter(cfg)

    def run(
        self,
        repo_path:  str,
        repo_id:    str,
        languages:  list[str],
        resume:     bool = False,
    ) -> None:

        # Initialise Neo4j indexes + constraints
        init_schema(self.neo4j_writer.driver)

        # Shared registries — populated during Pass 1, used for cross-file resolution
        module_registry: dict[str, ModuleNode]   = {}
        class_registry:  dict[str, ClassNode]    = {}

        resolver = RelationshipResolver(module_registry, class_registry)

        # Accumulated across all files for Pass 2
        all_functions:       list[FunctionNode]  = []
        all_raw_call_rels:   list[Relationship]  = []

        # Checkpoint for resume support
        done_files = _load_checkpoint(self.cfg.checkpoint_path) if resume else set()

        # ----------------------------------------------------------------
        # Pass 1 — walk every file, extract entities + Pass 1 relationships
        # ----------------------------------------------------------------
        logger.info("Pass 1: extracting entities from %s", repo_path)

        for file_path, language in walk_repo(repo_path, languages):

            if file_path in done_files:
                logger.debug("Skipping (already processed): %s", file_path)
                continue

            try:
                with open(file_path, "rb") as f:
                    source = f.read()
            except (OSError, PermissionError) as e:
                logger.warning("Could not read %s: %s", file_path, e)
                continue

            if should_skip_file(file_path, source):
                logger.debug("Skipping (filtered): %s", file_path)
                continue

            # Make path relative to repo root for stable UUIDs
            rel_path = str(Path(file_path).relative_to(repo_path))

            try:
                root, lang_obj = _parse_source(source, language)
            except Exception as e:
                logger.warning("Parse failed for %s: %s", rel_path, e)
                continue

            # Extract entities — no call resolution yet (repo_wide_fn_lookup=None)
            module, classes, functions, raw_calls = extract_file(
                file_path=rel_path,
                language=language,
                root=root,
                source=source,
                lang_obj=lang_obj,
                repo_id=repo_id,
                repo_wide_fn_lookup=None,   # Pass 1: skip call resolution
            )

            # Write entities to Neo4j
            self.neo4j_writer.write_module(module)
            for cls in classes:
                self.neo4j_writer.write_class(cls)
            for fn in functions:
                self.neo4j_writer.write_function(fn)

            # Resolve and write Pass 1 relationships for this file
            pass1_rels = resolver.resolve_pass1(module, classes, functions)
            self.rel_writer.write_relationships(pass1_rels)

            # Accumulate for Pass 2
            all_functions.extend(functions)
            # raw_calls is empty here since we passed repo_wide_fn_lookup=None.
            # We re-extract calls in Pass 2 with the full lookup.

            logger.info(
                "Pass 1 complete: %s — %d functions, %d classes, %d rels",
                rel_path, len(functions), len(classes), len(pass1_rels),
            )

        # ----------------------------------------------------------------
        # Build repo-wide function lookup — used by both Pass 2 stages
        # ----------------------------------------------------------------
        logger.info("Building repo-wide function lookup (%d functions)", len(all_functions))
        repo_fn_lookup = build_repo_fn_lookup(all_functions)

        # ----------------------------------------------------------------
        # Pass 2 — re-walk files for call resolution + vector embeddings
        # ----------------------------------------------------------------
        logger.info("Pass 2: resolving calls and writing embeddings")

        for file_path, language in walk_repo(repo_path, languages):

            rel_path = str(Path(file_path).relative_to(repo_path))

            if should_skip_file(file_path, source := open(file_path, "rb").read()):
                continue

            try:
                root, lang_obj = _parse_source(source, language)
            except Exception:
                continue

            # Re-extract — this time with the full repo_fn_lookup for call resolution
            module, classes, functions, raw_calls = extract_file(
                file_path=rel_path,
                language=language,
                root=root,
                source=source,
                lang_obj=lang_obj,
                repo_id=repo_id,
                repo_wide_fn_lookup=repo_fn_lookup,
            )

            # Resolve CALLS_UNKNOWN -> CALLS where possible
            resolved_calls = resolver.resolve_calls(raw_calls, repo_fn_lookup)
            self.rel_writer.write_relationships(resolved_calls)

            # Write embeddings for Tier 1 entities
            self._write_embeddings(module, classes, functions)

            _save_checkpoint(self.cfg.checkpoint_path, rel_path)

        logger.info("Ingestion complete.")

    # ------------------------------------------------------------------
    # Embedding writer
    # ------------------------------------------------------------------

    def _write_embeddings(
        self,
        module:    ModuleNode,
        classes:   list[ClassNode],
        functions: list[FunctionNode],
    ) -> None:
        """Chunk, tier-filter, embed, and write all entities for one file."""

        # Module — always embed (single chunk)
        module_doc = chunk_module(module)
        self.vector_writer.upsert(module_doc, self.embedder)

        # Classes — always embed (single chunk)
        for cls in classes:
            cls_doc = chunk_class(cls)
            self.vector_writer.upsert(cls_doc, self.embedder)

        # Functions — tiered, possibly multi-chunk
        for fn in functions:
            tier = embedding_tier(fn)
            if tier == 1:
                for doc in chunk_function(fn):
                    self.vector_writer.upsert(doc, self.embedder)
            # tier 2 = graph only (already in Neo4j), tier 3 = skip


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str) -> set[str]:
    import json, os
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def _save_checkpoint(path: str, file_path: str) -> None:
    import json, os
    done = _load_checkpoint(path)
    done.add(file_path)
    with open(path, "w") as f:
        json.dump(list(done), f)
