import os
import re
import shutil
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import build_graph
from agent.retrievers.graph_retriever import GraphRetriever
from agent.retrievers.vector_retriever import VectorRetriever
from config import Config
from ingestion.pipeline import IngestionPipeline, make_repo_node
from storage.neo4j_store import Neo4jStore
from storage.qdrant_store import QdrantStore

WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/tmp/astragraph_workspace"))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

_GITHUB_RE = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+(\.git)?$")


def _validate_config(cfg: Config) -> None:
    """Fail fast at startup if LLM credentials are missing."""
    if cfg.llm_provider == "anthropic" and not cfg.anthropic_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
            "Either set ANTHROPIC_API_KEY or switch to LLM_PROVIDER=openai_compat."
        )
    if cfg.llm_provider == "openai_compat" and not cfg.openai_compat_apikey:
        raise RuntimeError(
            "LLM_PROVIDER=openai_compat but OPENAI_COMPAT_APIKEY is not set. "
            "Set OPENAI_COMPAT_APIKEY to your Groq (or other) API key."
        )
    if cfg.llm_provider == "ollama":
        pass  # ollama needs no key


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg          = Config()
    _validate_config(cfg)
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

_STATIC = Path(__file__).parent.parent / "static"
if _STATIC.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC), html=True), name="static")


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


class IngestRequest(BaseModel):
    github_url: str
    repo_id:    str | None = None


class IngestResponse(BaseModel):
    job_id:  str
    status:  str
    repo_id: str


class IngestStatus(BaseModel):
    job_id:   str
    status:   str
    repo_id:  str
    progress: str
    error:    str | None = None


def _ingest_worker(job_id: str, github_url: str, repo_id: str, cfg: Config) -> None:
    def _set(**kwargs):
        with _jobs_lock:
            _jobs[job_id].update(kwargs)

    clone_path = WORKSPACE_DIR / repo_id
    try:
        _set(status="cloning", progress="cloning repository…")
        if clone_path.exists():
            shutil.rmtree(clone_path)
        subprocess.run(
            ["git", "clone", "--depth=1", github_url, str(clone_path)],
            check=True, capture_output=True, text=True,
        )

        _set(status="ingesting", progress="0/? files")
        repo_node = make_repo_node(str(clone_path), repo_id, github_url)
        pipeline  = IngestionPipeline(repo=repo_node, cfg=cfg)

        def on_progress(done: int, total: int, _path: str) -> None:
            _set(progress=f"{done}/{total} files")

        pipeline.run(str(clone_path), languages=["python"], on_progress=on_progress)
        _set(status="done", progress="complete")

    except subprocess.CalledProcessError as exc:
        _set(status="failed", error=f"git clone failed: {exc.stderr.strip()}")
    except Exception as exc:
        _set(status="failed", error=str(exc))


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, req: Request) -> IngestResponse:
    if not _GITHUB_RE.match(request.github_url):
        raise HTTPException(status_code=422, detail="github_url must be a valid https://github.com/... URL")

    repo_id = request.repo_id or request.github_url.rstrip("/").rstrip(".git").split("/")[-1]

    with _jobs_lock:
        running = [j for j in _jobs.values() if j["repo_id"] == repo_id and j["status"] in ("cloning", "ingesting")]
        if running:
            from fastapi.responses import JSONResponse
            job = running[0]
            return JSONResponse(
                status_code=409,
                content={
                    "job_id": job["job_id"],
                    "status": job["status"],
                    "repo_id": repo_id,
                    "detail": f"Ingestion already in progress for repo_id={repo_id!r}"
                }
            )

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "pending", "repo_id": repo_id, "progress": "", "error": None}

    cfg = req.app.state.cfg
    threading.Thread(target=_ingest_worker, args=(job_id, request.github_url, repo_id, cfg), daemon=True).start()

    return IngestResponse(job_id=job_id, status="pending", repo_id=repo_id)


@app.get("/ingest/status/{job_id}", response_model=IngestStatus)
def ingest_status(job_id: str) -> IngestStatus:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job_id {job_id!r} not found")
    return IngestStatus(**job)


@app.get("/graph/{repo_id}")
def graph(repo_id: str, req: Request, limit: int = 500) -> dict:
    """Full repo call graph for visualisation. Returns nodes + edges capped at `limit`."""
    store: Neo4jStore = req.app.state.graph_store
    return store.get_full_graph(repo_id, limit=limit)


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
