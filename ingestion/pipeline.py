"""
ingestion/pipeline.py

Orchestrates the full two-pass ingestion run.

Pass 1 — per file:
  - Parse source
  - Extract entities (module, package, classes, functions, attributes, parameters)
  - Resolve Pass 1 relationships (PART_OF, CONTAINS, DEFINED_IN, METHOD_OF,
    DEFINES_ATTR, HAS_PARAM, INHERITS, IMPORTS)
  - Collect raw call dicts for Pass 2

Pass 2 — after all files:
  - Build repo-wide function lookup
  - Resolve CALLS_UNKNOWN -> CALLS where possible
  - TODO: write entities + relationships to Neo4j
  - TODO: write embeddings to vector store
"""

from __future__ import annotations

import logging
from pathlib import Path

from tree_sitter import Language, Parser

from .extractors.extractor import extract_file
from .models import (
    ClassNode,
    FunctionNode,
    ModuleNode,
    PackageNode,
    Relationship,
    RepositoryNode,
    make_uuid,
)
from .relationships import RelationshipResolver, build_repo_fn_lookup
from .walker import walk_repo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language object cache  (one per language per run)
# ---------------------------------------------------------------------------

_LANG_OBJ_CACHE: dict[str, Language] = {}


def _get_lang_obj(language: str) -> Language:
    if language not in _LANG_OBJ_CACHE:
        if language == "python":
            import tree_sitter_python
            _LANG_OBJ_CACHE[language] = Language(tree_sitter_python.language())
        elif language == "typescript":
            import tree_sitter_typescript
            _LANG_OBJ_CACHE[language] = Language(tree_sitter_typescript.language_typescript())
        elif language == "go":
            import tree_sitter_go
            _LANG_OBJ_CACHE[language] = Language(tree_sitter_go.language())
        else:
            raise ValueError(f"No tree-sitter language module for: {language}")
    return _LANG_OBJ_CACHE[language]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:

    def __init__(self, repo: RepositoryNode):
        self.repo = repo
        # TODO: self.neo4j_writer  = Neo4jWriter(cfg)
        # TODO: self.rel_writer    = RelationshipWriter(driver)
        # TODO: self.vector_writer = VectorWriter(cfg)
        # TODO: self.embedder      = Embedder(cfg)

    def run(
        self,
        repo_path: str,
        languages: list[str] | None = None,
    ) -> tuple[list[Relationship], list[Relationship]]:
        """
        Run the two-pass pipeline.

        Returns (pass1_rels, pass2_rels) — useful for testing and dry-run
        inspection until Neo4j writes are wired up.
        """
        repo_id = self.repo.repo_id

        # Shared registries accumulate across files for cross-file resolution
        module_registry:  dict[str, ModuleNode]  = {}
        class_registry:   dict[str, ClassNode]   = {}
        package_registry: dict[str, PackageNode] = {}

        resolver = RelationshipResolver(
            repo=self.repo,
            module_registry=module_registry,
            class_registry=class_registry,
            package_registry=package_registry,
        )

        all_functions:  list[FunctionNode] = []
        all_raw_calls:  list[dict]         = []
        all_pass1_rels: list[Relationship] = []

        # ------------------------------------------------------------------
        # Pass 1 — extract entities + resolve per-file relationships
        # ------------------------------------------------------------------
        logger.info("Pass 1: extracting entities from %s", repo_path)

        parsers: dict[str, Parser] = {}

        files = list(walk_repo(repo_path))
        for i, (rel_path, language) in enumerate(files, 1):

            if languages and language not in languages:
                continue

            abs_path = Path(repo_path) / rel_path
            try:
                source = abs_path.read_bytes()
            except OSError as exc:
                logger.warning("Could not read %s: %s", rel_path, exc)
                continue

            try:
                lang_obj = _get_lang_obj(language)
            except (ValueError, ImportError) as exc:
                logger.warning("No language support for %s (%s): %s", rel_path, language, exc)
                continue

            if language not in parsers:
                parsers[language] = Parser(lang_obj)
            root = parsers[language].parse(source).root_node

            try:
                module, package, classes, functions, attributes, parameters, raw_calls = extract_file(
                    file_path=rel_path,
                    language=language,
                    root=root,
                    source=source,
                    lang_obj=lang_obj,
                    repo_id=repo_id,
                    repo_root=repo_path,
                )
            except Exception as exc:
                logger.warning("Extraction failed for %s: %s", rel_path, exc)
                continue

            pass1_rels = resolver.resolve_pass1(
                module=module,
                package=package,
                classes=classes,
                functions=functions,
                attributes=attributes,
                parameters=parameters,
            )

            # TODO: write module, package, classes, functions, attributes, parameters to Neo4j

            all_functions.extend(functions)
            all_raw_calls.extend(raw_calls)
            all_pass1_rels.extend(pass1_rels)

            logger.debug(
                "[%d/%d] %s — cls=%d fn=%d rels=%d",
                i, len(files), rel_path, len(classes), len(functions), len(pass1_rels),
            )

        # ------------------------------------------------------------------
        # Pass 2 — resolve CALLS across the whole repo
        # ------------------------------------------------------------------
        logger.info("Pass 2: resolving %d call sites", len(all_raw_calls))

        repo_fn_lookup = build_repo_fn_lookup(all_functions)

        raw_call_rels = [
            Relationship(
                src_uuid=c["src_uuid"],
                dst_uuid=None,
                rel_type="CALLS_UNKNOWN",
                properties={
                    "callee":          c["callee"],
                    "call_site_line":  c["call_site_line"],
                    "is_conditional":  c["is_conditional"],
                },
            )
            for c in all_raw_calls
        ]

        pass2_rels = resolver.resolve_calls(raw_call_rels, repo_fn_lookup)

        # TODO: write pass2_rels to Neo4j
        # TODO: embed Tier-1 entities and write to vector store

        logger.info(
            "Done — pass1_rels=%d  calls=%d  unresolved=%d",
            len(all_pass1_rels),
            sum(1 for r in pass2_rels if r.rel_type == "CALLS"),
            sum(1 for r in pass2_rels if r.rel_type == "CALLS_UNKNOWN"),
        )

        return all_pass1_rels, pass2_rels


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_repo_node(repo_path: str, repo_id: str | None = None, remote_url: str | None = None) -> RepositoryNode:
    name = repo_id or Path(repo_path).name
    return RepositoryNode(
        uuid=make_uuid(name),
        name=name,
        remote_url=remote_url or "",
        language="python",
        description=None,
        repo_id=name,
    )
