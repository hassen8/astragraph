"""LangGraph node functions for the query agent.

Each public node is a closure produced by a make_* factory so dependencies
(retrievers, config) are injected without global state.

Node contract: receives AgentState, returns a dict of keys to update.
LangGraph merges the returned dict into the running state automatically.
"""

from __future__ import annotations

from typing import Callable

import anthropic
import httpx

from agent.retrievers.graph_retriever import GraphRetriever
from agent.retrievers.vector_retriever import VectorRetriever
from agent.rrf import reciprocal_rank_fusion
from agent.state import AgentState
from config import Config

_GRAPH_KEYWORDS = {
    "call", "calls", "caller", "callers", "callee", "callees",
    "import", "imports", "imported", "inherit", "inherits", "inheritance",
    "subclass", "subclasses", "extends",
    "class", "method", "methods", "attribute", "attributes",
    "module", "package",
    "defined in", "where is", "defined where",
}


def route_query(state: AgentState) -> dict:
    """Set mode based on keyword scan. No LLM involved."""
    q = state["query"].lower()
    if state.get("mode") not in (None, ""):
        return {}
    for kw in _GRAPH_KEYWORDS:
        if kw in q:
            return {"mode": "graph"}
    return {"mode": "hybrid"}


import re as _re

_STOPWORDS = {
    "what", "does", "have", "how", "where", "who", "which", "the", "is",
    "are", "method", "methods", "call", "calls", "caller", "callers",
    "callee", "callees", "subclass", "subclasses", "class", "function",
    "defined", "in", "a", "an", "its", "do",
}

def _extract_name(query: str) -> str | None:
    """Pull a bare function/class name from a structural query."""
    # match "what calls foo", "callers of foo", "who calls foo", "calls foo?"
    m = _re.search(r'\b(?:calls?|callers?\s+of|callees?\s+of|inherits?\s+from|subclasses?\s+of)\s+[`"\']?(\w+)[`"\']?', query, _re.I)
    if m:
        return m.group(1)
    # capture all identifiers (including PascalCase class names), skip stopwords
    tokens = _re.findall(r'[`"\'](\w+)[`"\']|(\b[A-Za-z_]\w*\b)', query)
    flat = [a or b for a, b in tokens if (a or b)]
    candidates = [t for t in flat if t.lower() not in _STOPWORDS]
    return candidates[-1] if candidates else (flat[-1] if flat else None)


def make_graph_node(retriever: GraphRetriever) -> Callable:
    def graph_node(state: AgentState) -> dict:
        q       = state["query"].lower()
        repo_id = state.get("repo_id")
        top_k   = state["top_k"]
        name    = _extract_name(state["query"])

        # Dispatch to the right structural method when the intent is clear.
        if name and any(k in q for k in ("caller", "callers", "what calls", "who calls")):
            hits = retriever.callers(name, repo_id, top_k)
        elif name and any(k in q for k in ("callee", "callees", "what does", "calls what")):
            hits = retriever.callees(name, repo_id, top_k)
        elif name and any(k in q for k in ("subclass", "subclasses", "inherits from", "extends")):
            hits = retriever.subclasses(name, repo_id, top_k)
        elif name and any(k in q for k in ("method", "methods")):
            hits = retriever.methods(name, repo_id)
        else:
            # Fall back to BM25 for everything else (module contents, imports, etc.)
            hits = retriever.bm25_search(state["query"], repo_id, top_k)

        return {"graph_results": hits}
    return graph_node


def make_vector_node(retriever: VectorRetriever) -> Callable:
    def vector_node(state: AgentState) -> dict:
        hits = retriever.search(state["query"], state.get("repo_id"), state["top_k"])
        return {"vector_results": hits}
    return vector_node


def make_hybrid_node(
    graph_retriever: GraphRetriever,
    vector_retriever: VectorRetriever,
) -> Callable:
    def hybrid_node(state: AgentState) -> dict:
        top_k = state["top_k"]
        repo_id = state.get("repo_id")
        # Fetch 2x from each so RRF has headroom before slicing to top_k.
        graph_hits = graph_retriever.bm25_search(state["query"], repo_id, top_k * 2)
        vector_hits = vector_retriever.search(state["query"], repo_id, top_k * 2)
        merged = reciprocal_rank_fusion([graph_hits, vector_hits], top_k=top_k)
        return {
            "graph_results": graph_hits,
            "vector_results": vector_hits,
            "merged_results": merged,
        }
    return hybrid_node


def make_synthesize_node(cfg: Config) -> Callable:
    if cfg.llm_provider == "ollama":
        _http = httpx.Client(base_url=cfg.ollama_base_url, timeout=120.0)
    elif cfg.llm_provider == "openai_compat":
        _http = httpx.Client(
            base_url=cfg.openai_compat_url,
            headers={"Authorization": f"Bearer {cfg.openai_compat_apikey}"},
            timeout=60.0,
        )
    else:
        _anthropic = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    def synthesize_node(state: AgentState) -> dict:
        mode = state["mode"]
        if mode == "graph":
            results = state["graph_results"]
        elif mode == "vector":
            results = state["vector_results"]
        else:
            results = state["merged_results"]

        repo_id = state.get("repo_id") or "unknown"
        system = (
            f"You are a code analysis assistant for the `{repo_id}` codebase.\n"
            "Answer the user's question using only the code context provided.\n"
            "Cite file paths and function names when referencing specific code.\n"
            "If the context doesn't contain enough information, say so."
        )

        context_blocks = []
        for i, hit in enumerate(results, 1):
            body = hit.get("full_body") or (
                (hit.get("signature") or "") + "\n" + (hit.get("docstring") or "")
            ).strip()
            block = (
                f"[{i}] {hit.get('file_path', '?')} — {hit.get('qualified_name', hit.get('name', '?'))}"
                f" (lines {hit.get('line_start', '?')}–{hit.get('line_end', '?')})\n"
                f"{body}"
            )
            context_blocks.append(block)

        user_msg = (
            f"Question: {state['query']}\n\n"
            "Relevant code:\n\n"
            + "\n\n".join(context_blocks)
        ) if context_blocks else f"Question: {state['query']}\n\n(No relevant code found.)"

        if cfg.llm_provider == "ollama":
            resp = _http.post("/api/chat", json={
                "model": cfg.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
                "stream": False,
            })
            resp.raise_for_status()
            answer = resp.json()["message"]["content"]
        elif cfg.llm_provider == "openai_compat":
            resp = _http.post("/chat/completions", json={
                "model": cfg.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
            })
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
        else:
            response = _anthropic.messages.create(
                model=cfg.llm_model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            answer = response.content[0].text

        # Provenance: omit full_body to keep response payload lean.
        provenance = [
            {k: v for k, v in hit.items() if k != "full_body"}
            for hit in results
        ]

        return {"answer": answer, "provenance": provenance}

    return synthesize_node


def build_nodes(
    graph_retriever: GraphRetriever,
    vector_retriever: VectorRetriever,
    cfg: Config,
) -> dict[str, Callable]:
    return {
        "route_query": route_query,
        "graph_node": make_graph_node(graph_retriever),
        "vector_node": make_vector_node(vector_retriever),
        "hybrid_node": make_hybrid_node(graph_retriever, vector_retriever),
        "synthesize_node": make_synthesize_node(cfg),
    }
