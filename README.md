# AstraGraph

**AST** (Abstract Syntax Trees) + **RAG** (Retrieval-Augmented Generation) + **GRAPH** (Graph-based).

AstraGraph is a powerful **GraphRAG for source code** that understands the true structure of your codebase. It ingests a repository into a Neo4j property graph and a Qdrant vector store simultaneously, giving LLMs the ability to navigate your code exactly how a developer would—tracing imports, inheritance, and call chains to answer complex questions by combining structural (Cypher) and semantic (embedding) retrieval through a LangGraph agent.

## Web interface


https://github.com/user-attachments/assets/d6936c5f-4986-4260-850d-c5e8e7d9688d



The UI (`/ui`) is a single-page app with no build step.

**Left panel — chat:**
- Hybrid/graph/vector mode selector
- Send a question, get an answer with source badges
- Click a source badge to isolate that node in the graph (fetches its 1-hop neighborhood from the server if not currently visible)

**Right panel — graph:**
- Cytoscape.js force-directed graph of the repo's call/inheritance/import structure
- Nodes sized by in-degree (structural importance)
- Type filters: function, class, module, package
- Node count selector: 25 / 50 / 75 / 100 / 200 / 500
- Layout selector: cose, concentric, breadthfirst, circle, grid
- **File Viewer:** Clicking a node in the graph will open the file viewer. For functions, it displays the syntax-highlighted source code with correct line numbers. For classes and modules, it shows a detailed metadata card.
- Dark / light theme toggle, persisted in localStorage

**Ingest modal:**
- Paste a GitHub URL (Currently only works for python projects)
- Phase pills + progress bar track the ingestion pipeline in real time
- Errors surface inline

---

## The Strength of Hybrid Retrieval

Most code search tools struggle because they are purely semantic (embeddings miss exact references and structural hierarchy). AstraGraph leverages abstract syntax trees to extract the true structure of the code. It then runs both semantic and graph retrieval in parallel and fuses the results using Reciprocal Rank Fusion (RRF), unlocking capabilities standard RAG cannot match. Providing your LLM with both structured and unstructured context (and their interdependencies). Astragraph does:

- **Multi-hop reasoning** — Because code is represented as a connected graph, AstraGraph excels at multi-hop questions. It can accurately trace complex dependencies, such as answering "What downstream functions are affected if I change the return type of `load_data` in `utils.py`?", by seamlessly traversing `CALLS`, `INHERITS`, and `IMPORTS` relationships.
- **Graph retrieval** — Cypher queries over a property graph ensure perfect accuracy for structural questions: "what calls `build_graph`?", "what classes inherit from `BaseModel`?", "what does this module import?"
- **Vector retrieval** — sentence-transformer embeddings over function and class bodies handle conceptual queries: "how does the pipeline handle errors?", "where is retry logic implemented?"
- **Hybrid precision (default)** — both retrievers run at double `top_k`, then RRF merges and reranks the results. You get the exactness of structural connections combined with the fuzziness of semantic search.

The LLM only sees the most relevant top-k retrieved nodes and their multi-hop context. It never touches the full codebase, preventing context window bloat and reducing hallucinations.

---

## Architecture

### Ingestion pipeline (two passes)

**Pass 1 — per file:**
1. Parse source with tree-sitter
2. Extract entities: `Repository → Package → Module → Class/Function → Attribute/Parameter`
3. Accumulate all entities in memory
4. After all files: bulk-write to Neo4j in ~17 round trips (not one per entity)

**Between passes:** build a repo-wide function lookup by name and qualified name.

**Pass 2 — repo-wide:**
1. Resolve `CALLS_UNKNOWN` edges → upgrade to `CALLS` where the callee is found in the lookup
2. Embed all functions and classes with `sentence-transformers` (`all-MiniLM-L6-v2`, 384d)
3. Upsert embeddings into Qdrant

### Graph schema

```
Repository
  └─[PART_OF]─ Package
                └─[CONTAINS]─ Module
                               ├─[DEFINED_IN]─ Class
                               │    ├─[METHOD_OF]─ Function
                               │    └─[DEFINES_ATTR]─ Attribute
                               └─[DEFINED_IN]─ Function
                                    └─[HAS_PARAM]─ Parameter
```

Cross-cutting edges: `CALLS`, `INHERITS`, `IMPORTS`, `IMPORTS_EXTERNAL`.  
Unresolved edges are written to `:Unresolved` audit nodes via `HAS_UNRESOLVED` — never silently dropped.

### Agent (LangGraph)

Five-node `StateGraph`:

```
route_query → [graph_retrieval, vector_retrieval] → rrf_merge → synthesize
```

`route_query` classifies the query as `graph`, `vector`, or `hybrid` (default). For hybrid, both retrievers run at `2×top_k` and RRF reranks before the LLM synthesizes an answer.

### Storage abstraction

`pipeline.py` and the agent depend on `GraphStore` and `VectorStore` **Protocols** (`storage/protocols.py`), not on Neo4j or Qdrant client classes. The concrete implementations (`Neo4jStore`, `QdrantStore`) are protocol-conforming facades. Adding a new backend (e.g. KuzuDB + ChromaDB) means one new file per store — no changes to the pipeline or agent.

### UUIDs

All node UUIDs are deterministic: `MD5(repo_id + "::" + file_path + "::" + qualified_name)`. Every write is idempotent (`MERGE` in Neo4j, upsert in Qdrant). The same UUID appears in both stores, so graph and vector results can be cross-referenced.

---

## Tech stack

| Layer | Technology |
|---|---|
| Parsing | tree-sitter (Python, TypeScript, Go) |
| Graph DB | Neo4j 5 |
| Vector DB | Qdrant |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384d, CPU) |
| Agent | LangGraph |
| LLM | Claude (Anthropic) / Groq (OpenAI-compat) / Ollama |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla JS + Cytoscape.js + highlight.js |
| Container | Docker Compose |

---

## Performance

Measured on the FastAPI repo (527 Python files):

| Phase | Time |
|---|---|
| Total ingestion | ~45s |
| Neo4j entity writes | < 1s |
| Neo4j relationship writes | 1.1s |
| CPU embedding (512 batch) | ~42s |

---

## Running locally

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ with [uv](https://github.com/astral-sh/uv)

### 1. Start the databases

```bash
docker compose up neo4j qdrant -d
```

Wait ~15s for Neo4j to initialise.

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure

Copy and edit the environment:

```bash
# Minimum required for Anthropic (default provider):
export ANTHROPIC_API_KEY=sk-ant-...

# Or use Groq (free tier, fast):
export LLM_PROVIDER=openai_compat
export OPENAI_COMPAT_URL=https://api.groq.com/openai/v1
export OPENAI_COMPAT_APIKEY=gsk_...
export LLM_MODEL=llama-3.3-70b-versatile

# Or use Ollama (fully local):
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.2:3b
```

### 4. Ingest a repository

```bash
# Ingest a local path
uv run python cli.py --repo ./path/to/repo --repo-id my-repo

# Ingest only Python files
uv run python cli.py --repo ./path/to/repo --languages python

# Dry run (extract only, no DB writes)
uv run python cli.py --repo ./path/to/repo --dry-run
```

Ingestion can also be triggered via the UI or API (see below).

### 5. Start the API server

```bash
uv run uvicorn api.server:app --reload
```

Open `http://localhost:8000/ui` for the web interface.

---

## Running with Docker Compose (production)

```bash
# Create a .env file:
cat > .env <<EOF
NEO4J_PASSWORD=your-password
LLM_PROVIDER=openai_compat
OPENAI_COMPAT_APIKEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
EOF

docker compose up --build -d
```

The API is bound to `127.0.0.1:8000`.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `COLLECTION` | `codebase` | Qdrant collection name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `EMBED_BATCH_SIZE` | `512` | Embedding batch size |
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai_compat` \| `ollama` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model name for the chosen provider |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_COMPAT_URL` | `https://api.groq.com/openai/v1` | Base URL for OpenAI-compatible provider |
| `OPENAI_COMPAT_APIKEY` | — | Required when `LLM_PROVIDER=openai_compat` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/config` | Exposes non-secret config fields (embed model, LLM, etc.) |
| `GET` | `/repos` | List all ingested repositories |
| `DELETE` | `/repos/{repo_id}` | Delete a repo from Neo4j and Qdrant |
| `POST` | `/ingest` | Clone a GitHub URL and run ingestion in the background |
| `GET` | `/ingest/status/{job_id}` | Poll ingestion job status and phase |
| `POST` | `/query` | Full RAG query — retrieval + LLM synthesis |
| `POST` | `/retrieve` | Retrieval only — no LLM, used by evals |
| `GET` | `/graph/{repo_id}` | Top-N nodes by structural importance + edges, for visualisation |
| `GET` | `/graph/{repo_id}/node/{uuid}` | Single node + 1-hop neighborhood |
| `GET` | `/ui` | Web interface (served from `static/`) |

### POST /query

```json
{
  "query": "what calls build_graph?",
  "repo_id": "astragraph",
  "mode": "hybrid",
  "top_k": 5
}
```

Response includes `answer` (LLM synthesis) and `provenance` (list of retrieved nodes with file path, line range, and retrieval score).

### POST /ingest

```json
{
  "github_url": "https://github.com/tiangolo/fastapi",
  "repo_id": "fastapi"
}
```

Returns a `job_id` immediately. Poll `/ingest/status/{job_id}` to track phases: `cloning → parsing → writing_entities → writing_relationships → resolving_calls → embedding → writing_vectors → done`.

---

## Evaluation

Three evaluation layers are planned; Layer 3 is implemented:

**Layer 3 — Structural + semantic query set (done)**  
`evals/dataset/structural.json` (11 structural Cypher-derived cases) and `evals/dataset/semantic.json` (11 semantic source-reading cases). Run with:

```bash
uv run python evals/structural_eval.py --dataset structural
uv run python evals/structural_eval.py --dataset semantic
uv run python evals/structural_eval.py --dataset both
```

Results show graph mode wins 11/11 structural queries; `all-MiniLM-L6-v2` is the bottleneck on semantic queries.

**Layer 1 — CoIR baseline (not implemented)**  
`evals/coir_eval.py` exists as a stub. Intended to run CodeSearchNet through the vector retriever and report precision@5 / recall@5 against an external leaderboard.

**Layer 2 — RAGAS synthetic dataset (not implemented)**  
Auto-generate ~50 (query, ground-truth contexts, answer) triples from ingested function bodies using an LLM, then score faithfulness, answer relevancy, context precision, and context recall.

---

## Language support

| Language | Extraction | Status |
|---|---|---|
| Python | Full — functions, classes, attributes, parameters, imports, calls | Complete |
| TypeScript | Stub — file parsed, entities not extracted | Partial |
| Go | Stub — file parsed, entities not extracted | Partial |

---

## Not yet implemented

| Feature | Description |
|---|---|
| **fastembed** | Replace `sentence-transformers` with the ONNX runtime via `fastembed`. Same `all-MiniLM-L6-v2` model, typically 3–5× faster on CPU. Would push embedding from ~134s → ~30–40s. |
| **Community detection** | Run Louvain/Leiden on the CALLS graph post-ingestion. Store `community_id` and an LLM-generated cluster label on each `FunctionNode`. Enables "show me the auth cluster." |
| **Process tracing** | BFS from entry points (route handlers, `__main__`, CLI commands) through CALLS edges. Store as `Process` nodes with `STEP_IN` edges. Enables "show me the full call chain for POST /users." |
| **Incremental ingestion** | Checkpoint file tracking `mtime` per file. Skip unchanged files, `DETACH DELETE` stale nodes for changed/deleted files. Makes re-indexing live repos practical. |
| **TypeScript / Go extraction** | Full entity extraction for TypeScript and Go (stubs exist, tree-sitter grammars are wired up). |
| **CoIR baseline eval** | Layer 1 eval against CodeSearchNet. |
| **RAGAS synthetic eval** | Layer 2 eval with auto-generated test triples. |
| **MCP server** | Wrap as an MCP tool for use inside Cursor / Claude Code / Open Code etc...  |
| **Embedded store option** | Replace Neo4j + Qdrant with KuzuDB + ChromaDB for a single-container, zero-signup deployment (e.g. HuggingFace Spaces). The Protocol abstraction in `storage/` makes this two new files. |
| **Frontend v2** | React + Tailwind + shadcn/ui + Reagraph (WebGL) for larger repos and more interactivity. |

---

## Project structure

```
astragraph/
├── agent/
│   ├── graph.py              # LangGraph StateGraph (build_graph)
│   ├── nodes.py              # 5 agent nodes as closures
│   ├── rrf.py                # Reciprocal Rank Fusion
│   ├── state.py              # AgentState TypedDict
│   └── retrievers/
│       ├── graph_retriever.py
│       └── vector_retriever.py
├── api/
│   └── server.py             # FastAPI app, all endpoints
├── config.py                 # All config via env vars
├── evals/
│   ├── coir_eval.py          # CoIR baseline (stub)
│   ├── structural_eval.py    # Layer 3 eval runner
│   └── dataset/
│       ├── structural.json
│       └── semantic.json
├── graph/
│   └── schema.py             # Neo4j constraints + fulltext index
├── ingestion/
│   ├── embedder.py           # sentence-transformers wrapper
│   ├── models.py             # All dataclasses (FunctionNode, ClassNode, …)
│   ├── pipeline.py           # Two-pass ingestion orchestrator
│   ├── relationships.py      # Pass 1 + Pass 2 relationship resolution
│   ├── walker.py             # Repo file walker (respects .gitignore)
│   ├── extractors/
│   │   ├── python.py         # Full Python extraction
│   │   ├── typescript.py     # Stub
│   │   └── go.py             # Stub
│   └── writers/
│       ├── neo4j_writer.py   # Bulk Neo4j entity writes
│       └── vector_writer.py  # Qdrant upsert
├── static/
│   └── index.html            # Single-page UI (Cytoscape.js, highlight.js)
├── storage/
│   ├── protocols.py          # GraphStore + VectorStore Protocols
│   ├── neo4j_store.py        # Neo4jStore (read + write facade)
│   └── qdrant_store.py       # QdrantStore (read + write facade)
├── cli.py                    # CLI entry point
├── docker-compose.yml
└── Dockerfile
```
