"""
agent/state.py

Defines AgentState — the single TypedDict that flows through every node in the
LangGraph query agent. Each node reads from and writes to this dict; LangGraph
merges partial updates automatically so nodes only need to return the keys they
touch.

Fields:
    query          — raw natural-language question from the user
    repo_id        — scopes all retrieval to a specific ingested repository;
                     None means search across all repos (not yet supported)
    mode           — retrieval strategy decided by route_query:
                       "graph"  → BM25 fulltext + optional Cypher only
                       "vector" → Qdrant cosine search only
                       "hybrid" → both retrievers merged via RRF
    top_k          — number of results each retriever should return
    graph_results  — raw hits from GraphRetriever (BM25 / Cypher)
    vector_results — raw hits from VectorRetriever (Qdrant cosine)
    merged_results — RRF-fused list used by synthesize_node in hybrid mode
    answer         — final natural-language answer produced by the LLM
    provenance     — subset of result dicts returned to the caller so the
                     client knows which code nodes grounded the answer
"""

from typing import TypedDict


class AgentState(TypedDict):
    query:          str
    repo_id:        str | None
    mode:           str        # "graph" | "vector" | "hybrid"
    top_k:          int
    graph_results:  list[dict]
    vector_results: list[dict]
    merged_results: list[dict]
    answer:         str
    provenance:     list[dict]
