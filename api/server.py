from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from agent.graph import build_graph
from agent.retrievers.graph_retriever import GraphRetriever
from agent.retrievers.vector_retriever import VectorRetriever
from config import Config
from storage.neo4j_store import Neo4jStore
from storage.qdrant_store import QdrantStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg          = Config()
    graph_store  = Neo4jStore(cfg)
    vector_store = QdrantStore(cfg)
    graph_ret    = GraphRetriever(graph_store)
    vector_ret   = VectorRetriever(cfg, vector_store)
    agent_graph  = build_graph(graph_ret, vector_ret, cfg)

    app.state.cfg          = cfg
    app.state.graph        = agent_graph
    app.state.graph_store  = graph_store
    app.state.vector_store = vector_store
    app.state.graph_ret    = graph_ret
    app.state.vector_ret   = vector_ret

    yield

    graph_store.close()
    vector_store.close()
    vector_ret.close()


app = FastAPI(title="AstraGraph", lifespan=lifespan)


class QueryRequest(BaseModel):
    query:   str
    repo_id: str | None = None
    mode:    str        = ""
    top_k:   int        = 5


class ProvenanceItem(BaseModel):
    uuid:           str
    name:           str
    qualified_name: str
    file_path:      str
    line_start:     int
    line_end:       int
    score:          float
    sources:        list[str] = []


class QueryResponse(BaseModel):
    answer:     str
    provenance: list[ProvenanceItem]


class RetrieveResponse(BaseModel):
    mode:       str
    provenance: list[ProvenanceItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config(req: Request) -> dict:
    """Expose non-secret config fields for eval provenance tracking."""
    cfg: Config = req.app.state.cfg
    return {
        "embed_model":      cfg.embed_model,
        "embed_batch_size": cfg.embed_batch_size,
        "collection_name":  cfg.collection_name,
        "llm_provider":     cfg.llm_provider,
        "llm_model":        cfg.llm_model,
        "fulltext_index":   cfg.fulltext_index,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: QueryRequest, req: Request) -> RetrieveResponse:
    """Retrieval only — no LLM synthesis. Used by evals and tooling that only needs ranked nodes."""
    from agent.nodes import make_graph_node, make_vector_node, route_query as _route_query
    from agent.rrf import reciprocal_rank_fusion

    state = {
        "query": request.query, "repo_id": request.repo_id,
        "mode": request.mode,   "top_k": request.top_k,
        "graph_results": [], "vector_results": [], "merged_results": [],
        "answer": "", "provenance": [],
    }
    state.update(_route_query(state))
    mode      = state["mode"]
    graph_ret = req.app.state.graph_ret
    vec_ret   = req.app.state.vector_ret

    if mode == "graph":
        hits = make_graph_node(graph_ret)(state)["graph_results"]
    elif mode == "vector":
        hits = make_vector_node(vec_ret)(state)["vector_results"]
    else:
        fetch = {**state, "top_k": request.top_k * 2}
        graph_hits  = make_graph_node(graph_ret)(fetch)["graph_results"]
        vector_hits = make_vector_node(vec_ret)(fetch)["vector_results"]
        hits = reciprocal_rank_fusion([graph_hits, vector_hits], top_k=request.top_k)

    provenance = [{k: v for k, v in h.items() if k != "full_body"} for h in hits]
    return RetrieveResponse(
        mode=mode,
        provenance=[ProvenanceItem(**p) for p in provenance],
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, req: Request) -> QueryResponse:
    initial_state = {
        "query":          request.query,
        "repo_id":        request.repo_id,
        "mode":           request.mode,
        "top_k":          request.top_k,
        "graph_results":  [],
        "vector_results": [],
        "merged_results": [],
        "answer":         "",
        "provenance":     [],
    }

    result = req.app.state.graph.invoke(initial_state)

    return QueryResponse(
        answer=result["answer"],
        provenance=[ProvenanceItem(**p) for p in result["provenance"]],
    )
