"""LangGraph StateGraph wiring for the query agent.

Topology:
    START → route_query → {
        "graph"  → graph_node  → synthesize_node → END
        "vector" → vector_node → synthesize_node → END
        "hybrid" → hybrid_node → synthesize_node → END
    }

Usage:
    graph = build_graph(graph_retriever, vector_retriever, cfg)
    result = graph.invoke({"query": "...", "repo_id": "...", "top_k": 5, ...})
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import build_nodes
from agent.retrievers.graph_retriever import GraphRetriever
from agent.retrievers.vector_retriever import VectorRetriever
from agent.state import AgentState
from config import Config


def _route(state: AgentState) -> str:
    return state["mode"]


def build_graph(
    graph_retriever: GraphRetriever,
    vector_retriever: VectorRetriever,
    cfg: Config,
):
    nodes = build_nodes(graph_retriever, vector_retriever, cfg)

    builder = StateGraph(AgentState)

    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "route_query")
    builder.add_conditional_edges(
        "route_query",
        _route,
        {
            "graph": "graph_node",
            "vector": "vector_node",
            "hybrid": "hybrid_node",
        },
    )
    builder.add_edge("graph_node", "synthesize_node")
    builder.add_edge("vector_node", "synthesize_node")
    builder.add_edge("hybrid_node", "synthesize_node")
    builder.add_edge("synthesize_node", END)

    return builder.compile()
