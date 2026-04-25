"""
agent/retrievers/graph_retriever.py

Agent-side wrapper around a GraphStore. Exposes the read operations the agent
nodes need without hard-coding any specific database. The retriever does not
talk to Neo4j (or any other backend) directly — it goes through GraphStore so
that swapping backends is a one-line change in the FastAPI lifespan.

Two retrieval mechanisms are exposed:

  1. BM25 fulltext search — fast keyword matching across function name,
     qualified_name, docstring, and full_body. Best for queries that contain
     specific identifiers or keywords appearing verbatim in the code.

  2. Structural lookups — relationship traversals (callers, callees,
     subclasses, methods, module contents, imports, call paths). These answer
     questions BM25 and vector search cannot, because the answer lives in the
     graph topology rather than the text of any single node.

All methods return list[dict] in the standard result shape so RRF and the
synthesizer remain source-agnostic:

  uuid, name, qualified_name, file_path, line_start, line_end,
  signature, docstring, full_body, repo_id, score, source="graph"
"""

from __future__ import annotations

from storage.protocols import GraphStore


class GraphRetriever:

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    # ----- keyword search --------------------------------------------------- #

    def bm25_search(self, query: str, repo_id: str | None = None, top_k: int = 10) -> list[dict]:
        return self._store.fulltext_search(query, repo_id, top_k)

    # ----- discovery -------------------------------------------------------- #

    def find_by_name(self, name: str, kind: str = "function", repo_id: str | None = None) -> list[dict]:
        return self._store.find_by_name(name, kind, repo_id)

    def find_qualified(self, qualified_name: str, repo_id: str | None = None) -> list[dict]:
        return self._store.find_qualified(qualified_name, repo_id)

    # ----- class structure -------------------------------------------------- #

    def subclasses(self, class_name: str, repo_id: str, top_k: int = 25) -> list[dict]:
        return self._store.subclasses(class_name, repo_id, top_k)

    def methods(self, class_name: str, repo_id: str) -> list[dict]:
        return self._store.methods(class_name, repo_id)

    def attributes(self, class_name: str, repo_id: str) -> list[dict]:
        return self._store.attributes(class_name, repo_id)

    # ----- module / package structure -------------------------------------- #

    def module_contents(self, module_path: str, repo_id: str) -> list[dict]:
        return self._store.module_contents(module_path, repo_id)

    def module_imports(self, module_path: str, repo_id: str) -> list[dict]:
        return self._store.module_imports(module_path, repo_id)

    # ----- call graph ------------------------------------------------------- #

    def callers(self, name: str, repo_id: str, top_k: int = 25) -> list[dict]:
        return self._store.callers(name, repo_id, top_k)

    def callees(self, name: str, repo_id: str, top_k: int = 25) -> list[dict]:
        return self._store.callees(name, repo_id, top_k)

    def call_path(self, from_name: str, to_name: str, repo_id: str, max_hops: int = 5) -> list[dict]:
        return self._store.call_path(from_name, to_name, repo_id, max_hops)
