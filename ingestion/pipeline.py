"""
ingestion/pipeline.py

Orchestrates the full two-pass ingestion run.

Pass 1 — per file:
  - Parse source
  - Extract entities (module, package, classes, functions, attributes, parameters)
  - Write entities to Neo4j
  - Resolve Pass 1 relationships and write them to Neo4j
  - Collect raw call dicts and classes for Pass 2

Pass 2 — after all files:
  - Build repo-wide function lookup
  - Resolve CALLS_UNKNOWN -> CALLS where possible
  - Write CALLS relationships to Neo4j
  - Embed functions + classes and write to Qdrant
"""

from __future__ import annotations

import logging
from pathlib import Path

from tree_sitter import Language, Parser

from config import Config
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
from .embedder import (
    make_class_embed_doc,
    make_function_embed_doc,
    should_embed,
    should_embed_class,
    Embedder,
)
from .relationships import RelationshipResolver, build_repo_fn_lookup
from .walker import walk_repo
from storage.neo4j_store import Neo4jStore
from storage.qdrant_store import QdrantStore

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

    def __init__(self, repo: RepositoryNode, cfg: Config):
        self.repo = repo
        self.cfg  = cfg

    def run(
        self,
        repo_path: str,
        languages: list[str] | None = None,
        on_progress: callable | None = None,
    ) -> None:
        """
        Run the two-pass ingestion pipeline.

        Pass 1 writes entities and structural relationships file-by-file.
        Pass 2 resolves and writes CALLS relationships repo-wide.
        """
        repo_id = self.repo.repo_id
        cfg     = self.cfg

        with Neo4jStore(cfg) as graph_store:

            # Schema init is idempotent — safe to call every run.
            # Creates uniqueness constraints, range indexes, and the fulltext index.
            graph_store.init_schema()

            # Write the repository root node first — top-level packages need
            # a destination for their PART_OF edges.
            graph_store.write_repository(self.repo)

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

            all_functions: list[FunctionNode] = []
            all_classes:  list[ClassNode]    = []
            all_raw_calls: list[dict]        = []

            # ------------------------------------------------------------------
            # Pass 1 — extract, write entities, resolve + write relationships
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

                # Write entities — nodes must exist before relationships can reference them
                graph_store.write_package(package)
                graph_store.write_module(module)
                graph_store.write_classes(classes)
                graph_store.write_functions(functions)
                graph_store.write_attributes(attributes)
                graph_store.write_parameters(parameters)

                # Resolve and write structural relationships for this file
                pass1_rels = resolver.resolve_pass1(
                    module=module,
                    package=package,
                    classes=classes,
                    functions=functions,
                    attributes=attributes,
                    parameters=parameters,
                )
                graph_store.write_external_dependencies(pass1_rels)
                graph_store.write_relationships(pass1_rels)

                all_functions.extend(functions)
                all_classes.extend(classes)
                all_raw_calls.extend(raw_calls)

                if on_progress:
                    on_progress(i, len(files), rel_path)

                logger.debug(
                    "[%d/%d] %s — cls=%d fn=%d rels=%d",
                    i, len(files), rel_path, len(classes), len(functions), len(pass1_rels),
                )

            # ------------------------------------------------------------------
            # Pass 2 — resolve CALLS across the whole repo and write
            # ------------------------------------------------------------------
            logger.info("Pass 2: resolving %d call sites", len(all_raw_calls))

            repo_fn_lookup = build_repo_fn_lookup(all_functions)

            raw_call_rels = [
                Relationship(
                    src_uuid=c["src_uuid"],
                    dst_uuid=None,
                    rel_type="CALLS_UNKNOWN",
                    properties={
                        "callee":         c["callee"],
                        "call_site_line": c["call_site_line"],
                        "is_conditional": c["is_conditional"],
                    },
                )
                for c in all_raw_calls
            ]

            pass2_rels = resolver.resolve_calls(raw_call_rels, repo_fn_lookup)
            graph_store.write_relationships(pass2_rels)

            # ------------------------------------------------------------------
            # Embedding — filter, build docs, embed, write to vector store
            # ------------------------------------------------------------------
            fn_docs  = [make_function_embed_doc(fn)  for fn in all_functions if should_embed(fn)]
            cls_docs = [make_class_embed_doc(cls)    for cls in all_classes  if should_embed_class(cls)]
            docs     = fn_docs + cls_docs

            logger.info(
                "Embedding %d functions + %d classes (%d total, %d skipped)",
                len(fn_docs), len(cls_docs), len(docs),
                len(all_functions) + len(all_classes) - len(docs),
            )

            with Embedder(self.cfg) as embedder, QdrantStore(self.cfg) as vector_store:
                embeddings = embedder.embed(docs)
                vector_store.upsert(docs, embeddings)

            logger.info(
                "Done — calls=%d  unresolved=%d  vectors=%d",
                sum(1 for r in pass2_rels if r.rel_type == "CALLS"),
                sum(1 for r in pass2_rels if r.rel_type == "CALLS_UNKNOWN"),
                len(docs),
            )


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
