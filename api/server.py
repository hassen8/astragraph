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

    app.state.graph        = agent_graph
    app.state.graph_store  = graph_store
    app.state.vector_store = vector_store
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


@app.get("/health")
def health():
    return {"status": "ok"}


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
