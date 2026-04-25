"""
storage/protocols.py

Defines the storage contract that AstraGraph's pipeline and agent depend on.

Two Protocols are defined here:

  GraphStore  — graph database operations (writes, reads, schema init)
  VectorStore — vector database operations (upsert, similarity search)

The pipeline and agent are written against these Protocols, not concrete
client classes. This means swapping the underlying database is a matter of
writing one new implementation file — no changes to pipeline.py, agent code,
or any consumer.

Why Protocols instead of ABCs:
  Protocols use structural (duck) typing — a class satisfies the Protocol if
  it has the right method signatures, no inheritance required. This keeps
  third-party adapters (e.g. a community-contributed PostgreSQL store) free
  of import-time dependency on this module.

Concrete implementations live alongside this file:
  storage/neo4j_store.py   — implements GraphStore over Neo4j (Cypher + Driver)
  storage/qdrant_store.py  — implements VectorStore over Qdrant
  storage/ladybug_store.py — (future) implements both over LadybugDB

Read-method return shape (uniform across all implementations):
  uuid, name, qualified_name, file_path, line_start, line_end,
  signature, docstring, full_body, repo_id, score, source
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ingestion.models import (
    AttributeNode,
    ClassNode,
    EmbedDoc,
    FunctionNode,
    ModuleNode,
    PackageNode,
    ParameterNode,
    Relationship,
    RepositoryNode,
)


@runtime_checkable
class GraphStore(Protocol):
    """
    Read+write contract for the property graph holding code structure.

    Implementations must be idempotent: calling write_* repeatedly with the
    same input must not produce duplicates. The reference implementation
    (Neo4jStore) uses MERGE + uuid uniqueness constraints to guarantee this.
    """

    # Lifecycle ----------------------------------------------------------------

    def init_schema(self) -> None:
        """Create constraints, indexes, and any other schema artifacts. Idempotent."""

    def close(self) -> None:
        """Release any open connections or driver resources."""

    # Node writes --------------------------------------------------------------

    def write_repository(self, repo: RepositoryNode) -> None: ...
    def write_package(self, pkg: PackageNode) -> None: ...
    def write_module(self, module: ModuleNode) -> None: ...
    def write_classes(self, classes: list[ClassNode]) -> None: ...
    def write_functions(self, functions: list[FunctionNode]) -> None: ...
    def write_attributes(self, attributes: list[AttributeNode]) -> None: ...
    def write_parameters(self, parameters: list[ParameterNode]) -> None: ...

    # Relationship writes ------------------------------------------------------

    def write_relationships(self, rels: list[Relationship]) -> None:
        """Write resolved structural and call edges between existing nodes."""

    def write_external_dependencies(self, rels: list[Relationship]) -> None:
        """Write IMPORTS_EXTERNAL edges, creating ExternalDependency nodes as needed."""

    # Reads — keyword search ---------------------------------------------------

    def fulltext_search(self, query: str, repo_id: str | None, top_k: int) -> list[dict]:
        """BM25-style keyword search over indexed Function text fields."""

    # Reads — discovery (find a node by name) ----------------------------------

    def find_by_name(self, name: str, kind: str, repo_id: str | None) -> list[dict]:
        """
        Find nodes by exact `name`. `kind` filters by label:
        "function" | "class" | "module" | "package".
        """

    def find_qualified(self, qualified_name: str, repo_id: str | None) -> list[dict]:
        """Find nodes by exact `qualified_name` (dotted path)."""

    # Reads — class structure --------------------------------------------------

    def subclasses(self, class_name: str, repo_id: str, top_k: int) -> list[dict]:
        """Return classes that INHERIT (transitively) from `class_name`."""

    def methods(self, class_name: str, repo_id: str) -> list[dict]:
        """Return functions related to `class_name` via METHOD_OF."""

    def attributes(self, class_name: str, repo_id: str) -> list[dict]:
        """Return attributes related to `class_name` via DEFINES_ATTR."""

    # Reads — module / package structure ---------------------------------------

    def module_contents(self, module_path: str, repo_id: str) -> list[dict]:
        """Return functions and classes DEFINED_IN the given module."""

    def module_imports(self, module_path: str, repo_id: str) -> list[dict]:
        """Return modules / external dependencies that the module IMPORTS."""

    # Reads — call graph -------------------------------------------------------

    def callers(self, name: str, repo_id: str, top_k: int) -> list[dict]:
        """Return functions that directly CALL the function `name`."""

    def callees(self, name: str, repo_id: str, top_k: int) -> list[dict]:
        """Return functions directly called BY function `name`."""

    def call_path(self, from_name: str, to_name: str, repo_id: str, max_hops: int) -> list[dict]:
        """Return one CALL path from `from_name` to `to_name`, up to `max_hops` deep."""


@runtime_checkable
class VectorStore(Protocol):
    """
    Upsert + nearest-neighbor contract for the embedding store.

    Implementations must store enough payload alongside each vector to satisfy
    the standard read-result shape — at minimum: name, qualified_name,
    file_path, line_start, line_end, signature, docstring, full_body, repo_id.
    """

    # Lifecycle ----------------------------------------------------------------

    def close(self) -> None: ...

    # Writes -------------------------------------------------------------------

    def upsert(self, docs: list[EmbedDoc], vectors: list[list[float]]) -> None:
        """Upsert vectors keyed by EmbedDoc.uuid. Idempotent."""

    # Reads --------------------------------------------------------------------

    def search(self, vector: list[float], repo_id: str | None, top_k: int) -> list[dict]:
        """Return top-k payload dicts ranked by cosine similarity to `vector`."""
