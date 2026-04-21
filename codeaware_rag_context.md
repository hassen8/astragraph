# CodeAware GraphRAG — Complete Project Context

> This document is the authoritative implementation reference for the CodeAware GraphRAG project. It captures every architectural decision, data model, function, class, query, and open challenge discussed across the entire design process. Use this as the primary context file when continuing implementation in Claude Code.
>
> **Current status:** Phase 1 (ingestion pipeline) fully designed. All models, extractor, relationship resolver, and pipeline orchestrator have been written. Phase 2 (query agent) is designed but not yet implemented.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why GraphRAG Over Flat RAG](#why-graphrag-over-flat-rag)
3. [Full Repository Structure](#full-repository-structure)
4. [Architecture — Three Layers](#architecture--three-layers)
5. [Graph Hierarchy & Node Types](#graph-hierarchy--node-types)
6. [Phase 1 — Ingestion Pipeline](#phase-1--ingestion-pipeline)
7. [File: `ingestion/models.py`](#file-ingestionmodelspy)
8. [Package: `ingestion/extractors/`](#package-ingestionextractors)
9. [File: `ingestion/relationships.py`](#file-ingestionrelationshipspy)
10. [File: `ingestion/pipeline.py`](#file-ingestionpipelinepy)
11. [File: `ingestion/tiering.py`](#file-ingestiontieringpy)
12. [File: `ingestion/chunker.py`](#file-ingestionchunkerpy)
13. [File: `graph/schema.py`](#file-graphschemapy)
14. [File: `config.py`](#file-configpy)
15. [File: `cli.py`](#file-clipy)
16. [Embedding Text Strategy](#embedding-text-strategy)
17. [Neo4j Schema — Full Property Definitions](#neo4j-schema--full-property-definitions)
18. [Open Challenges & Resolutions](#open-challenges--resolutions)
19. [Implementation Order](#implementation-order)
20. [Technology Stack](#technology-stack)
21. [Phase 2 Preview — Query Agent](#phase-2-preview--query-agent)

---

## Project Overview

A **GraphRAG agent** that ingests a source code repository and enables structural and semantic querying over it. The system parses the codebase into a **Neo4j property graph** and a **vector store** (Chroma / pgvector) simultaneously, linked by deterministic UUIDs, enabling multi-hop structural queries that flat retrieval cannot answer.

### Core motivation

Vanilla RAG (chunk files → embed → retrieve top-k by similarity) fails badly for developer questions like:

- *"Which functions call `process_payment` and what do they import?"*
- *"What does the `payments` module do overall?"*
- *"Find all callers of `process_payment` that deal with retries"*
- *"Which classes inherit from `BaseProcessor`?"*
- *"What external libraries does the `payments` package depend on?"*

These require **structural traversal** (graph) combined with **semantic matching** (vectors). This system provides both via a hybrid query agent.

### Target demo

- Index the HuggingFace `transformers` library (well-known, rich, Python-only, familiar to ML interviewers)
- Prepare 5–6 showcase queries that vanilla RAG gets wrong but GraphRAG gets right
- Benchmark against flat RAG using LLM-as-judge scoring on faithfulness and answer completeness
- Expose via FastAPI + Streamlit UI

---

## Why GraphRAG Over Flat RAG

| Query type | Flat RAG | GraphRAG |
|---|---|---|
| "What does function X do?" | Works — body is in the embed | Works |
| "What calls function X?" | Fails — requires structural traversal | Works — Cypher: `MATCH (f)-[:CALLS]->(x)` |
| "What does module Y do?" | Partial — depends on chunk luck | Works — module embed text is synthesised |
| "Which classes inherit from Z?" | Fails | Works — `MATCH (c)-[:INHERITS]->(z)` |
| "What external libraries does package P use?" | Fails | Works — `ExternalDependency` nodes |
| "Find functions that accept a `User` argument" | Partial — works if type is in signature embed | Works — `HAS_PARAM` with `type_hint` property |

---

## Full Repository Structure

```
codebase_graphrag/
├── ingestion/
│   ├── __init__.py
│   ├── walker.py               # Repo traversal + .gitignore filtering
│   ├── parser.py               # tree-sitter setup, one parser per language, cached
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── extractor.py        # ExtractionContext + Unified entrypoint
│   │   ├── python.py           # Language-specific builder
│   │   ├── go.py               # Language-specific builder
│   │   ├── typescript.py       # Language-specific builder
│   │   └── queries/            # .scm S-expression query files
│   ├── models.py               # All dataclasses: entity nodes, Relationship, EmbedDoc
│   ├── embedder.py             # Wraps OpenAI / local model, batches requests
│   ├── chunker.py              # Sliding window chunker for large functions
│   ├── tiering.py              # Decides embedding tier per entity (1/2/3)
│   ├── relationships.py        # RelationshipResolver + RelationshipWriter
│   ├── pipeline.py             # IngestionPipeline — orchestrates two-pass run
│   └── writers/
│       ├── __init__.py
│       ├── neo4j_writer.py     # Neo4j upsert logic for entity nodes
│       └── vector_writer.py    # Chroma / pgvector upsert logic
├── graph/
│   ├── __init__.py
│   ├── schema.py               # Cypher index + constraint initialisation
│   └── queries.py              # Reusable Cypher query templates (Phase 2)
├── agent/                      # Phase 2 — not yet implemented
│   ├── __init__.py
│   ├── router.py               # Classifies query → graph | vector | hybrid
│   ├── graph_retriever.py      # Multi-hop Cypher execution
│   ├── vector_retriever.py     # Similarity search + UUID deduplication
│   ├── reranker.py             # Combines graph distance + semantic score
│   └── synthesiser.py          # Builds LLM prompt with provenance
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI app
├── ui/
│   └── app.py                  # Streamlit demo UI
├── config.py                   # Env vars, paths, model names
├── cli.py                      # `python -m codebase_graphrag ingest --repo ./path`
└── tests/
    ├── fixtures/               # Tiny synthetic .py / .ts / .go test files
    ├── test_extractor.py
    ├── test_chunker.py
    ├── test_tiering.py
    ├── test_relationships.py
    └── test_writers.py
```

---

## Architecture — Three Layers

### Layer 1: Ingestion pipeline (Phase 1 — current focus)

```
Source codebase (local path)
        │
        ▼
   Repo walker              os.walk + .gitignore filter
        │                   yields (file_path, language) tuples
        ▼
   Language detector        extension → grammar map
        │                   .py→python  .ts→typescript  .go→go
        ▼
   tree-sitter parser       source bytes → AST root node
        │
        ▼
   ExtractionContext        query DSL loaded from .scm
        │                   manages source bytes + root AST
        ▼
   Language Builder         e.g., PythonBuilder
        │                   language-specific node resolution
        ├── build_module()
        ├── build_classes()     ← emits ClassNode + AttributeNode
        ├── build_functions()   ← emits FunctionNode + ParameterNode
        └── build_calls()       ← Pass 2 only (needs repo-wide lookup)
        │
        ▼
   Tiering filter           embedding_tier(fn) → 1 | 2 | 3
        │
        ▼
   RelationshipResolver     resolve_pass1() per file
        │                   resolve_calls() after full repo walk
        │
        ▼
   Dual writer ─────────────┬───────────────────────────────┐
                            ▼                               ▼
                         Neo4j                        Vector store
                    (property graph)             (embeddings + metadata)
                    MERGE — idempotent           upsert — idempotent
                    same UUID links both         chunk_index suffix for multi-chunk
```

### Layer 2: Storage (dual store, always in sync)

- **Neo4j** — property graph, all node types, all relationship types, Cypher queries
- **Vector store** — Chroma (dev) or pgvector (prod), embeddings keyed by entity UUID + chunk_index
- **Invariant** — every UUID in the vector store has a corresponding Neo4j node. The UUID is deterministic (MD5 of repo_id + file_path + qualified_name), making all writes idempotent.

### Layer 3: Query agent (Phase 2 — designed, not yet implemented)

```
User query
    │
    ▼
Query router          classifies as: graph | vector | hybrid
    │
    ▼
Graph traversal  ◄──── Neo4j        multi-hop Cypher
    +
Vector retrieval ◄──── Vector store  similarity search + UUID dedup
    │
    ▼
Reranker              relevance score + graph proximity score
    │
    ▼
LLM synthesiser       GPT-4o / Claude Sonnet / Llama
    │
    ▼
Answer + provenance   file · line · call chain citations
```

---

## Graph Hierarchy & Node Types

### The full hierarchy

```
Repository
    └──[PART_OF]── Package          (directory with __init__.py)
                       └──[CONTAINS]── Module   (single .py file)
                                           ├──[DEFINED_IN / CONTAINS]── Class
                                           │       ├──[METHOD_OF]── Function (method)
                                           │       └──[DEFINES_ATTR]── Attribute
                                           └──[DEFINED_IN / CONTAINS]── Function (module-level)
                                                       └──[HAS_PARAM]── Parameter
```

### Cross-cutting relationships

```
Class      ──[INHERITS]──►         Class
Function   ──[CALLS]──►            Function
Module     ──[IMPORTS]──►          Module
Module     ──[IMPORTS_EXTERNAL]──► ExternalDependency
Package    ──[PART_OF]──►          Package | Repository
```

### Design decisions for each node type

| Node | Decision | Rationale |
|---|---|---|
| `Repository` | Separate node, root of graph | Single entry point for repo-scoped traversals; enables multi-repo indexing |
| `Package` | Separate node per `__init__.py` directory | Python's import system resolves through packages; essential for subsystem-level queries |
| `Module` | One per `.py` file | File-level queries ("what does this file do?") and import graph traversal |
| `Class` | Separate node with rich properties | Class-level semantic search; inheritance graph; method inventory for embedding |
| `Function` | Core node, tiered embedding | Multi-hop call graph; most queries ultimately resolve to functions |
| `Attribute` | Separate node (promoted from list[str] property) | Type-aware queries ("find classes with a StripeClient attribute") |
| `Parameter` | Tier 2 — graph only, no embedding | Type-aware queries ("find functions accepting a User"); signature already in function embed |
| `ExternalDependency` | Separate node | First-class dependency queries ("which modules use stripe?"); not noise like Unresolved |
| `Interface/Protocol` | **Property on Class, not separate node** | In Python, Protocols are just classes; `is_protocol: bool` on ClassNode is sufficient |

---

## Phase 1 — Ingestion Pipeline

### Two-pass design

**Pass 1** — per file, in sequence:
1. Parse source file with tree-sitter
2. Extract all entities (module, classes, functions, attributes, parameters)
3. Write entities to Neo4j (`MERGE` — idempotent)
4. Register module/classes/packages into shared cross-file registries
5. Resolve Pass 1 relationships: `PART_OF`, `CONTAINS`, `DEFINED_IN`, `METHOD_OF`, `DEFINES_ATTR`, `HAS_PARAM`, `INHERITS`, `IMPORTS`
6. Write Pass 1 relationships to Neo4j
7. Accumulate all `FunctionNode`s for Pass 2

**Pass 2** — after all files are walked:
1. Build repo-wide function lookup: `Dict[name | qualified_name → FunctionNode]`
2. Re-walk all files
3. Run `extract_calls()` with the complete lookup (call resolution now possible)
4. Upgrade `CALLS_UNKNOWN` to `CALLS` where resolvable
5. Write `CALLS` relationships to Neo4j
6. Embed all Tier-1 entities and write to vector store

**Why two passes?** `foo()` in function A might call function B defined in a file not yet processed. You cannot resolve call targets in Pass 1 without reading the whole repo first. All other relationships (`INHERITS`, `IMPORTS`, `METHOD_OF`, `DEFINED_IN`) can be resolved incrementally as files are processed because their targets are registered into shared dicts immediately on extraction.

---

## File: `ingestion/models.py`

All dataclasses. Zero external dependencies. These are the internal transfer objects between every stage of the pipeline.

### `make_uuid(*parts: str) -> str`

```python
def make_uuid(*parts: str) -> str:
    """
    Deterministic UUID from an arbitrary number of string parts.
    Stable across re-runs — same inputs always produce the same UUID.
    Safe to use as Neo4j MERGE key and vector store document ID.

    Examples:
        make_uuid("my-repo", "payments/core.py", "payments.core.process_payment")
        make_uuid("my-repo")                    # for Repository node
        make_uuid("my-repo", "payments")        # for Package node
    """
    key = "::".join(parts)
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))
```

### `RepositoryNode`

```python
@dataclass
class RepositoryNode:
    uuid:        str
    name:        str            # e.g. "transformers", "payments-service"
    remote_url:  Optional[str]  # e.g. "https://github.com/huggingface/transformers"
    language:    str            # primary language, e.g. "python"
    description: Optional[str]  # from README first paragraph or pyproject.toml
    repo_id:     str            # same as uuid — kept for join consistency
```

Neo4j label: `Repository`. One per ingestion run. Entry point for all repo-scoped queries.

### `PackageNode`

```python
@dataclass
class PackageNode:
    uuid:          str
    name:          str            # dot-separated: "payments.core" for payments/core/
    directory:     str            # relative path: "payments/core"
    is_namespace:  bool           # True if no __init__.py (PEP 420 namespace package)
    has_init:      bool           # True if __init__.py exists
    init_file:     Optional[str]  # relative path to __init__.py if it exists
    repo_id:       str
```

Neo4j label: `Package`. Represents a directory with `__init__.py`. Packages nest: `payments.retry` → `payments` → `Repository`.

### `ModuleNode`

```python
@dataclass
class ModuleNode:
    uuid:             str
    name:             str            # dot-separated: "payments.core"
    file_path:        str            # relative to repo root: "payments/core.py"
    language:         str
    docstring:        Optional[str]
    exported_names:   list[str]      # top-level function + class names
    imported_modules: list[str]      # raw import statement strings
    is_init:          bool           # True if this is an __init__.py file
    repo_id:          str
```

Neo4j label: `Module`. One per source file.

### `ClassNode`

```python
@dataclass
class ClassNode:
    uuid:            str
    name:            str
    qualified_name:  str            # "payments.core.PaymentProcessor"
    file_path:       str
    line_start:      int
    line_end:        int
    language:        str
    docstring:       Optional[str]
    base_classes:    list[str]      # raw base class name strings (resolved → INHERITS edges)
    decorators:      list[str]
    is_abstract:     bool           # inherits from ABC or ABCMeta
    is_protocol:     bool           # inherits from typing.Protocol
    is_dataclass:    bool           # decorated with @dataclass
    is_exception:    bool           # inherits from Exception or BaseException
    method_names:    list[str]      # inventory — used in class embed text
    attribute_names: list[str]      # self.x names — used in class embed text
    repo_id:         str
```

Neo4j label: `Class`.

### `FunctionNode`

```python
@dataclass
class FunctionNode:
    uuid:            str
    name:            str
    qualified_name:  str             # "payments.core.PaymentProcessor.process"
    file_path:       str
    line_start:      int
    line_end:        int
    language:        str
    signature:       str             # full def line e.g. "def process(self, amount: float) -> bool"
    docstring:       Optional[str]
    return_type:     Optional[str]
    is_async:        bool
    is_method:       bool            # True if defined inside a class body
    is_property:     bool            # decorated with @property
    is_classmethod:  bool            # decorated with @classmethod
    is_staticmethod: bool            # decorated with @staticmethod
    is_overload:     bool            # decorated with @typing.overload
    decorators:      list[str]
    body_preview:    str             # first 300 chars — used for tiering decisions only
    full_body:       str             # complete raw source text of the function
    complexity:      int             # cyclomatic complexity
    repo_id:         str
```

Neo4j label: `Function`. Core node of the system — most queries resolve here.

### `ParameterNode`

```python
@dataclass
class ParameterNode:
    uuid:         str
    name:         str
    type_hint:    Optional[str]     # "float", "Optional[str]", "StripeClient"
    default:      Optional[str]     # raw default expression string or None
    position:     int               # 0-indexed position in signature
    is_self:      bool              # True for self/cls — skipped in HAS_PARAM
    is_variadic:  bool              # True for *args
    is_keyword:   bool              # True for **kwargs
    parent_uuid:  str               # UUID of owning FunctionNode
    repo_id:      str
```

Neo4j label: `Parameter`. Tier 2 — graph only, no embedding. `HAS_PARAM` relationship carries `type_hint` directly as a property for fast type queries.

### `AttributeNode`

```python
@dataclass
class AttributeNode:
    uuid:         str
    name:         str               # bare name e.g. "client"
    full_name:    str               # "self.client" or "ClassName.class_var"
    type_hint:    Optional[str]     # from type annotation: "StripeClient" or None
    default:      Optional[str]     # raw default expression string or None
    is_instance:  bool              # True = self.x in __init__; False = class-level
    is_class_var: bool              # True if typed as ClassVar[...]
    line:         int               # line where first assigned/declared
    parent_uuid:  str               # UUID of owning ClassNode
    repo_id:      str
```

Neo4j label: `Attribute`. Promoted from `list[str]` property on ClassNode to a proper queryable node.

### `ExternalDependencyNode`

```python
@dataclass
class ExternalDependencyNode:
    uuid:            str
    name:            str            # top-level package name e.g. "stripe", "numpy"
    imported_names:  list[str]      # specific names imported: ["StripeClient", "Charge"]
    raw_import:      str            # original import statement text
    repo_id:         str
```

Neo4j label: `ExternalDependency`. First-class node for external libraries — enables dependency queries without polluting the typed edge graph.

### `Relationship`

```python
@dataclass
class Relationship:
    src_uuid:   str
    dst_uuid:   Optional[str]   # None = unresolved → written to :Unresolved node
    rel_type:   str
    properties: dict = field(default_factory=dict)
```

All relationship types:

```
# Structural hierarchy
PART_OF           Package    -> Package | Repository    {level: "top"|"nested"}
CONTAINS          Package    -> Module                  {}

# Ownership
DEFINED_IN        Function   -> Module                  {}
DEFINED_IN        Class      -> Module                  {}
METHOD_OF         Function   -> Class                   {is_property, is_classmethod, is_staticmethod}
DEFINES_ATTR      Class      -> Attribute               {is_instance, is_class_var}
HAS_PARAM         Function   -> Parameter               {position, type_hint, is_variadic, is_keyword}

# Cross-entity semantic
INHERITS          Class      -> Class                   {is_direct: True}
CALLS             Function   -> Function                {call_site_line, is_conditional, resolved_in_pass?}
IMPORTS           Module     -> Module                  {raw_import, is_external: False}
IMPORTS_EXTERNAL  Module     -> ExternalDependency      {module_name, raw_import, is_external: True}

# Unresolved (written to :Unresolved audit nodes)
INHERITS_UNKNOWN  Class      -> None    {base_name}
CALLS_UNKNOWN     Function   -> None    {callee, call_site_line}
PART_OF_UNKNOWN   Package    -> None    {parent_package}
```

### `EmbedDoc`

```python
@dataclass
class EmbedDoc:
    uuid:         str           # matches Neo4j node UUID exactly
    chunk_index:  int           # 0 for single-chunk; 0..N for sliding window
    total_chunks: int
    text:         str           # text that gets embedded
    entity_type:  str           # "function" | "class" | "module" | "attribute"
    tier:         int           # 1 = embed, 2 = graph-only, 3 = skip
    metadata:     dict          # file_path, line_start, name, language, repo_id, etc.
```

Vector store document ID: `"{uuid}__{chunk_index}"`. At query time, deduplicate by `uuid` before passing context to LLM.

---

## Package: `ingestion/extractors/`

### Design: modular builder pattern

The extraction pipeline utilizes a modular, decoupled architecture rather than a monolithic extractor class.

**Key Components**:
- `ExtractionContext`: Manages shared state (source bytes, root AST node, language objects) and loads S-expression queries.
- Language Builders (e.g., `PythonBuilder`, `GoBuilder`): Classes that take an `ExtractionContext` and handle language-specific node resolution, docstring extraction, signature assembly, and type hint extraction.
- S-expression queries (`.scm` files): Tree-sitter query DSL strings are stored externally in the `queries/` directory and loaded dynamically.

### Dynamic `.scm` query loading

Queries are written in `queries/<language>.scm` and separated by `;; @query: name` delimiters. The `load_queries()` function reads this file into a dictionary at runtime.

```scheme
;; @query: function
(function_definition) @fn.def

;; @query: class
(class_definition) @cls.def
```

### `ExtractionContext` class

```python
class ExtractionContext:
    def __init__(self, language, root, source, file_path, repo_id, lang_obj): ...
    def query(self, name: str) -> Optional[Query]: ...
    def captures(self, query_name: str, node: Node) -> dict[str, list[Node]]: ...
    def src(self, node: Node) -> str: ...
```

Compiles and caches tree-sitter Query objects upon first use. Exposes `captures()` to quickly grab matching nodes by their `@capture.name`.

### Language Builders

Each language has a dedicated builder, like `PythonBuilder`. These builders expose a standardized interface to build domain objects:

- `build_module() -> ModuleNode`: Extracts module docstrings, exported names, and imports.
- `build_classes() -> list[ClassNode]`: Scans for class definitions, base classes, methods, and populates `extracted_attributes` containing `AttributeNode`s.
- `build_functions() -> list[FunctionNode]`: Recursively unwraps async wrappers, extracts signatures, docstrings, and populates `extracted_parameters` containing `ParameterNode`s.
- `build_calls() -> list[dict]`: Runs call-site queries and returns a list of raw call dictionaries containing `src_uuid`, `callee` text, and `call_site_line`.

Unlike the previous architecture, these builders extract relationships like attributes and parameters explicitly during their AST walks.

### `extract_file()` — unified entrypoint

```python
def extract_file(
    file_path: str,
    language: str,
    root: Node,
    source: bytes,
    lang_obj: Language,
    repo_id: str,
) -> tuple[ModuleNode, list[ClassNode], list[FunctionNode], list[AttributeNode], list[ParameterNode], list[dict]]:
```

This function instantiates the `ExtractionContext` and dynamically chooses the correct builder (e.g., `PythonBuilder`). It triggers the build methods in sequence and returns the comprehensive set of domain objects and call dictionaries directly.

---

## File: `ingestion/relationships.py`

### `RelationshipResolver` class

Shared registries (`modules`, `classes`, `packages`) accumulate as files are processed. Cross-file resolution depends on these registries being populated by the time a referencing file is processed.

```python
class RelationshipResolver:
    def __init__(self, repo, module_registry, class_registry, package_registry): ...
```

#### `resolve_pass1(module, package, classes, functions, attributes, parameters) -> list[Relationship]`

Registers entities into shared registries, then calls all Pass 1 resolvers:

1. `_resolve_package_hierarchy(package)` → `PART_OF` edges
2. `_resolve_package_contains_module(package, module)` → `CONTAINS` edge
3. `_resolve_defined_in(module, classes, functions)` → `DEFINED_IN` edges
4. `_resolve_method_of(classes, functions)` → `METHOD_OF` edges
5. `_resolve_defines_attr(classes, attributes)` → `DEFINES_ATTR` edges
6. `_resolve_has_param(functions, parameters)` → `HAS_PARAM` edges (skips `is_self=True`)
7. `_resolve_inherits(classes)` → `INHERITS` + `INHERITS_UNKNOWN` edges
8. `_resolve_imports(module)` → `IMPORTS` + `IMPORTS_EXTERNAL` edges

#### `_resolve_package_hierarchy(package)`

- If `package.name` has one dot-segment: `PART_OF` → `Repository`
- If nested: look up parent in `self.packages`, emit `PART_OF` → parent package
- If parent not yet seen: emit `PART_OF_UNKNOWN` (unresolved) with `parent_package` property

#### `_resolve_method_of(classes, functions)`

Derives owning class from qualified name: `"a.b.ClassName.method".rsplit(".", 1)[0]` = `"a.b.ClassName"`. Looks up in local `cls_by_qname` first, then `self.classes` registry. Emits `METHOD_OF` with `is_property`, `is_classmethod`, `is_staticmethod` properties from the `FunctionNode`.

#### `_resolve_defines_attr(classes, attributes)`

Each `AttributeNode.parent_uuid` points to its owning `ClassNode`. Emits `DEFINES_ATTR` with `is_instance` and `is_class_var` as relationship properties.

#### `_resolve_has_param(functions, parameters)`

Skips `is_self=True` parameters (no structural value). Emits `HAS_PARAM` with `position`, `type_hint`, `is_variadic`, `is_keyword` as relationship properties. These properties on the relationship enable type-aware queries without traversing to the `Parameter` node.

#### `_resolve_inherits(classes)`

Resolution order: same-file `cls_by_name` → same-file `cls_by_qname` → cross-file `self.classes` registry → `INHERITS_UNKNOWN`.

Skips trivial bases: `object`, `Exception`, `BaseException`, `ValueError`, `TypeError`, `RuntimeError`, `KeyError`, `IndexError`, `AttributeError`.

#### `_resolve_imports(module)`

Parses each raw import string via `_parse_import_target()`. Looks up in `self.modules` registry:
- Found → `IMPORTS` relationship
- Not found → `IMPORTS_EXTERNAL` relationship with `dst_uuid = make_uuid(repo_id, "external", target_name)`

The `IMPORTS_EXTERNAL` `dst_uuid` is deterministic so the `ExternalDependency` node can be upserted separately via `write_external_dependencies()`.

#### `resolve_calls(raw_call_relationships, repo_wide_fn_lookup) -> list[Relationship]`

Pass 2. Iterates raw relationships from extractor:
- Already resolved `CALLS` (dst_uuid set) → pass through
- `CALLS_UNKNOWN` → attempt lookup in `repo_fn_lookup` → upgrade to `CALLS` with `resolved_in_pass: 2` property, or leave as `CALLS_UNKNOWN` for external calls

### `build_repo_fn_lookup(all_functions) -> dict[str, FunctionNode]`

Called once after Pass 1. Indexes every function by both bare name and qualified name. Last-write wins for bare name collisions — acceptable because extractor already tries local scope first.

### `_parse_import_target(import_text) -> Optional[str]`

Extracts module name from raw import string:
- `"from . import x"` → `None` (relative import, skip)
- `"from payments.core import Foo"` → `"payments.core"`
- `"import numpy as np"` → `"numpy"`
- `"import os, sys"` → `"os"` (first only)

### `RelationshipWriter` class

```python
class RelationshipWriter:
    def write_relationships(self, relationships: list[Relationship]) -> None
    def _write_resolved_batch(self, rels: list[Relationship]) -> None
    def _write_unresolved_batch(self, rels: list[Relationship]) -> None
    def write_external_dependencies(self, rels: list[Relationship]) -> None
```

#### `_write_resolved_batch`

Groups relationships by `rel_type` then runs one `UNWIND` batch per type:

```cypher
UNWIND $rows AS row
MATCH (src {uuid: row.src_uuid})
MATCH (dst {uuid: row.dst_uuid})
MERGE (src)-[r:REL_TYPE]->(dst)
SET r += row.properties
```

**Why group by type?** Cypher does not allow parameterised relationship type names (`MERGE (a)-[r:$type]->(b)` is invalid). The type must be interpolated into the query string. Grouping enables a single `UNWIND` per type instead of one query per relationship — on large repos the difference is minutes vs hours.

#### `_write_unresolved_batch`

Writes unresolved edges to `:Unresolved` nodes keyed by `(src_uuid, rel_type, target)`. Connects via `HAS_UNRESOLVED` edge. Preserves auditing information: *"what external libraries does module X import?"*, *"which external base classes does Y extend?"*

#### `write_external_dependencies`

Must be called separately from `write_relationships` because it needs to **create** the `ExternalDependency` destination node before writing the edge. Uses `MERGE` on `uuid` for idempotency.

---

## File: `ingestion/pipeline.py`

### `IngestionPipeline` class

```python
class IngestionPipeline:
    def __init__(self, cfg: Config): ...
    def run(self, repo_path, repo_id, languages, resume=False) -> None: ...
    def _write_embeddings(self, module, classes, functions) -> None: ...
```

#### `run()`

**Pass 1 loop** — per file:
```python
for file_path, language in walk_repo(repo_path, languages):
    # skip if in checkpoint (resume mode)
    # read source bytes
    # should_skip_file() filter
    # make relative path for stable UUIDs
    # parse with tree-sitter
    # extract_file(..., repo_wide_fn_lookup=None)
    # write_module / write_class / write_function to Neo4j
    # resolver.resolve_pass1(module, classes, functions)
    # rel_writer.write_relationships(pass1_rels)
    # all_functions.extend(functions)
```

**Between passes:**
```python
repo_fn_lookup = build_repo_fn_lookup(all_functions)
```

**Pass 2 loop** — per file:
```python
for file_path, language in walk_repo(repo_path, languages):
    # parse again (same source, idempotent)
    # extract_file(..., repo_wide_fn_lookup=repo_fn_lookup)
    # resolver.resolve_calls(raw_calls, repo_fn_lookup)
    # rel_writer.write_relationships(resolved_calls)
    # _write_embeddings(module, classes, functions)
    # _save_checkpoint(rel_path)
```

#### `_write_embeddings(module, classes, functions)`

- Module: always embed, single chunk → `chunk_module(module)` → `vector_writer.upsert`
- Classes: always embed, single chunk → `chunk_class(cls)` → `vector_writer.upsert`
- Functions: tier-filtered → `embedding_tier(fn)` → if `tier == 1`: `chunk_function(fn)` (may return multiple `EmbedDoc`s) → `vector_writer.upsert`

### Checkpoint helpers

```python
def _load_checkpoint(path: str) -> set[str]   # reads JSON list of processed file paths
def _save_checkpoint(path: str, file_path: str) -> None  # appends file path to checkpoint
```

---

## File: `ingestion/tiering.py`

Solves the vector store explosion problem for large repos (e.g. PyTorch: ~80,000 raw functions → ~15,000–20,000 Tier 1 embeddings via 4–5x reduction).

### `embedding_tier(fn: FunctionNode) -> int`

```python
def embedding_tier(fn: FunctionNode) -> int:
    """
    Returns:
        1 — embed + graph (full treatment)
        2 — graph only (structural traversal but no vector embedding)
        3 — skip entirely (don't write to graph or vector store)
    """
    body_lines = fn.line_end - fn.line_start

    # Tier 1: docstring is a strong author signal
    if fn.docstring and len(fn.docstring) > 20:
        return 1

    # Tier 3: too small to have meaningful semantic content
    if body_lines < 5:
        return 3

    # Dunder methods — key ones always Tier 1
    important_dunders = {"__init__", "__call__", "__repr__", "__new__", "__enter__", "__exit__"}
    if fn.name.startswith("__") and fn.name.endswith("__"):
        return 1 if fn.name in important_dunders else 2

    # Tier 2: short private helpers with no docstring
    if fn.name.startswith("_") and body_lines < 15:
        return 2

    # Tier 2: straight-line code, low semantic retrieval value
    if fn.complexity <= 1 and body_lines < 10:
        return 2

    return 1
```

### `should_skip_file(file_path: str, source: bytes) -> bool`

```python
def should_skip_file(file_path: str, source: bytes) -> bool:
    """Returns True if the entire file should be excluded from ingestion."""
    basename = os.path.basename(file_path)

    # Test files
    if basename.startswith("test_") or basename.endswith("_test.py"):
        return True
    if basename in ("conftest.py",):
        return True

    # Auto-generated files — scan first 500 bytes
    header = source[:500].decode("utf-8", errors="replace")
    autogen_markers = ["DO NOT EDIT", "generated by", "@generated", "auto-generated"]
    if any(marker.lower() in header.lower() for marker in autogen_markers):
        return True

    return False
```

---

## File: `ingestion/chunker.py`

### Chunking strategy

**Embed the full function body** — not a character preview. The 300-char body preview on `FunctionNode` is for tiering decisions only; it is never what gets embedded.

For functions ≤ 60 lines: single `EmbedDoc` chunk.
For functions > 60 lines: sliding window with 20-line overlap, anchored to signature + docstring.

```
Chunk 0 (anchor):  signature + docstring + lines 0–60
Chunk 1:           signature + docstring + lines 40–100    ← 20-line overlap
Chunk 2:           signature + docstring + lines 80–140    ← 20-line overlap
...
```

Every chunk carries the signature and docstring so every retrieved chunk is self-contained. At query time: **deduplicate by `uuid`** before passing context to LLM.

### `chunk_function(node: FunctionNode) -> list[EmbedDoc]`

```python
CHUNK_LINE_LIMIT = 60
OVERLAP_LINES    = 20

def chunk_function(node: FunctionNode) -> list[EmbedDoc]:
    lines = node.full_body.splitlines()

    if len(lines) <= CHUNK_LINE_LIMIT:
        return [EmbedDoc(
            uuid=node.uuid, chunk_index=0, total_chunks=1,
            text=build_function_embed_text(node),
            entity_type="function", tier=embedding_tier(node),
            metadata=_function_metadata(node),
        )]

    anchor_header = f"Function: {node.qualified_name}\nSignature: {node.signature}\n"
    if node.docstring:
        anchor_header += f"Description: {node.docstring}\n"

    chunks = []
    step   = CHUNK_LINE_LIMIT - OVERLAP_LINES
    starts = list(range(0, len(lines), step))

    for i, start in enumerate(starts):
        window = lines[start:start + CHUNK_LINE_LIMIT]
        text   = anchor_header + f"Body (lines {start+1}–{start+len(window)}):\n" + "\n".join(window)
        chunks.append(EmbedDoc(
            uuid=node.uuid, chunk_index=i, total_chunks=len(starts),
            text=text, entity_type="function", tier=embedding_tier(node),
            metadata={**_function_metadata(node), "chunk_index": i, "total_chunks": len(starts)},
        ))

    return chunks
```

### `chunk_class(node: ClassNode) -> EmbedDoc`

Always single chunk. Embed text format:

```
Class: payments.core.PaymentProcessor
Inherits: BaseProcessor, RetryMixin
Description: Stateful processor for managing payment lifecycle with retry support.
Methods: __init__, process, refund, _validate, _handle_stripe_error
Attributes: self.client, self.max_retries, self.idempotency_store
File: payments/core.py
```

### `chunk_module(node: ModuleNode) -> EmbedDoc`

Always single chunk. Embed text format:

```
Module: payments.core
File: payments/core.py
Description: Handles payment processing, retry logic, and idempotency.
Exports: process_payment, refund, PaymentProcessor, RETRY_LIMIT
Imports: import stripe, from sqlalchemy import ..., from payments.models import ...
```

---

## Embedding Text Strategy

### `build_function_embed_text(node: FunctionNode) -> str`

```python
def build_function_embed_text(node: FunctionNode) -> str:
    parts = [
        f"Function: {node.qualified_name}",
        f"Signature: {node.signature}",
    ]
    if node.docstring:
        parts.append(f"Description: {node.docstring}")
    if node.return_type:
        parts.append(f"Returns: {node.return_type}")
    parts.append(f"File: {node.file_path}")
    parts.append(f"Body:\n{node.full_body}")
    return "\n".join(parts)
```

Semantically richest content first (name, signature, docstring) so embedding captures intent before implementation. Full body — not a truncated preview.

### Per-entity embed text summary

| Entity | What gets embedded | Chunking |
|---|---|---|
| `Module` | Synthesised: docstring + exports + imports | Always single chunk |
| `Class` | Docstring + base classes + method inventory + attributes | Always single chunk |
| `Function` (Tier 1) | Full body + signature + docstring | Sliding window if > 60 lines |
| `Function` (Tier 2) | Graph only — no embedding | — |
| `Parameter` | Never embedded | — |
| `Attribute` | Never embedded separately (captured in class embed) | — |

---

## File: `graph/schema.py`

### `init_schema(driver)`

Run once at the start of each ingestion run:

```python
INDEXES = [
    "CREATE INDEX repo_uuid     IF NOT EXISTS FOR (r:Repository)          ON (r.uuid)",
    "CREATE INDEX pkg_uuid      IF NOT EXISTS FOR (p:Package)             ON (p.uuid)",
    "CREATE INDEX pkg_name      IF NOT EXISTS FOR (p:Package)             ON (p.name)",
    "CREATE INDEX module_uuid   IF NOT EXISTS FOR (m:Module)              ON (m.uuid)",
    "CREATE INDEX module_name   IF NOT EXISTS FOR (m:Module)              ON (m.name)",
    "CREATE INDEX class_uuid    IF NOT EXISTS FOR (c:Class)               ON (c.uuid)",
    "CREATE INDEX class_name    IF NOT EXISTS FOR (c:Class)               ON (c.name)",
    "CREATE INDEX class_qname   IF NOT EXISTS FOR (c:Class)               ON (c.qualified_name)",
    "CREATE INDEX fn_uuid       IF NOT EXISTS FOR (f:Function)            ON (f.uuid)",
    "CREATE INDEX fn_name       IF NOT EXISTS FOR (f:Function)            ON (f.name)",
    "CREATE INDEX fn_qname      IF NOT EXISTS FOR (f:Function)            ON (f.qualified_name)",
    "CREATE INDEX repo_filter   IF NOT EXISTS FOR (f:Function)            ON (f.repo_id)",
    "CREATE INDEX ext_uuid      IF NOT EXISTS FOR (e:ExternalDependency)  ON (e.uuid)",
    "CREATE INDEX ext_name      IF NOT EXISTS FOR (e:ExternalDependency)  ON (e.name)",
]
```

### Example Cypher queries (Phase 2 reference)

```cypher
-- Multi-hop call chain: who calls process_payment, transitively up to 3 hops?
MATCH path = (caller:Function)-[:CALLS*1..3]->(target:Function {name: "process_payment"})
RETURN caller.name, caller.file_path, caller.line_start, length(path) AS depth
ORDER BY depth

-- What does the payments package export?
MATCH (pkg:Package {name: "payments"})-[:CONTAINS]->(m:Module)-[:DEFINED_IN]-(f:Function)
WHERE NOT f.is_method
RETURN f.name, f.qualified_name, f.docstring

-- Which classes inherit (directly or transitively) from BaseProcessor?
MATCH (c:Class)-[:INHERITS*1..5]->(base:Class {name: "BaseProcessor"})
RETURN c.name, c.qualified_name

-- Which external libraries does the payments package use?
MATCH (pkg:Package {name: "payments"})-[:CONTAINS]->(m:Module)-[:IMPORTS_EXTERNAL]->(e:ExternalDependency)
RETURN DISTINCT e.name ORDER BY e.name

-- Find all functions that accept a StripeClient parameter
MATCH (f:Function)-[r:HAS_PARAM]->(p:Parameter)
WHERE r.type_hint CONTAINS "StripeClient"
RETURN f.qualified_name, f.file_path
```

---

## File: `config.py`

```python
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Neo4j
    neo4j_uri:       str = os.getenv("NEO4J_URI",       "bolt://localhost:7687")
    neo4j_user:      str = os.getenv("NEO4J_USER",      "neo4j")
    neo4j_password:  str = os.getenv("NEO4J_PASSWORD",  "password")

    # Vector store
    chroma_path:     str = os.getenv("CHROMA_PATH",     "./chroma_db")
    collection_name: str = os.getenv("COLLECTION",      "codebase")

    # Embeddings
    openai_api_key:  str = os.getenv("OPENAI_API_KEY",  "")
    embed_model:     str = os.getenv("EMBED_MODEL",     "text-embedding-3-large")
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))

    # Ingestion
    checkpoint_path:  str = os.getenv("CHECKPOINT_PATH",  "./ingestion_checkpoint.json")
    chunk_line_limit: int = int(os.getenv("CHUNK_LINE_LIMIT", "60"))
    overlap_lines:    int = int(os.getenv("OVERLAP_LINES",    "20"))
```

---

## File: `cli.py`

```python
import click
from ingestion.pipeline import IngestionPipeline
from config import Config

@click.command()
@click.option("--repo",      required=True,                        help="Path to local repo root")
@click.option("--repo-id",   default=None,                         help="Override repo identifier")
@click.option("--languages", default="python,typescript,go",       help="Comma-separated list")
@click.option("--resume",    is_flag=True,                         help="Skip already-processed files")
def ingest(repo, repo_id, languages, resume):
    cfg = Config()
    pipeline = IngestionPipeline(cfg)
    pipeline.run(
        repo_path=repo,
        repo_id=repo_id or repo.split("/")[-1],
        languages=languages.split(","),
        resume=resume,
    )

if __name__ == "__main__":
    ingest()
```

Usage:
```bash
python -m codebase_graphrag ingest --repo ./transformers --languages python
python -m codebase_graphrag ingest --repo ./transformers --resume   # resume interrupted run
```

---

## Neo4j Schema — Full Property Definitions

### Node: `Repository`
```
uuid         string  make_uuid(repo_id)
name         string  "transformers"
remote_url   string?
language     string  "python"
description  string?
repo_id      string  same as uuid
```

### Node: `Package`
```
uuid          string  make_uuid(repo_id, package_name)
name          string  "payments.core"   (dot-separated)
directory     string  "payments/core"   (relative path)
is_namespace  bool
has_init      bool
init_file     string?
repo_id       string
```

### Node: `Module`
```
uuid              string  make_uuid(repo_id, file_path, module_name)
name              string  "payments.core"
file_path         string  "payments/core.py"
language          string
docstring         string?
exported_names    string[]
imported_modules  string[]
is_init           bool
repo_id           string
```

### Node: `Class`
```
uuid             string
name             string
qualified_name   string  "payments.core.PaymentProcessor"
file_path        string
line_start       int
line_end         int
language         string
docstring        string?
base_classes     string[]
decorators       string[]
is_abstract      bool
is_protocol      bool
is_dataclass     bool
is_exception     bool
method_names     string[]
attribute_names  string[]
repo_id          string
```

### Node: `Function`
```
uuid             string
name             string
qualified_name   string  "payments.core.PaymentProcessor.process"
file_path        string
line_start       int
line_end         int
language         string
signature        string
docstring        string?
return_type      string?
is_async         bool
is_method        bool
is_property      bool
is_classmethod   bool
is_staticmethod  bool
is_overload      bool
decorators       string[]
complexity       int
repo_id          string
```

### Node: `Parameter`
```
uuid         string
name         string
type_hint    string?
default      string?
position     int
is_self      bool
is_variadic  bool
is_keyword   bool
parent_uuid  string
repo_id      string
```

### Node: `Attribute`
```
uuid          string
name          string
full_name     string  "self.client"
type_hint     string?
default       string?
is_instance   bool
is_class_var  bool
line          int
parent_uuid   string
repo_id       string
```

### Node: `ExternalDependency`
```
uuid            string  make_uuid(repo_id, "external", module_name)
name            string  "stripe"
imported_names  string[]
raw_import      string
repo_id         string
```

---

## Open Challenges & Resolutions

### Challenge 1: What chunking strategy?

**Problem:** Embedding only 300 chars of body misses semantic content. Embedding unlimited body text makes chunk sizes unpredictable.

**Resolution:** Embed the full body for functions ≤ 60 lines (vast majority). Sliding window (60-line window, 20-line overlap) for functions > 60 lines — every chunk carries signature + docstring as anchor header. Deduplicate by parent `uuid` at query time before sending to LLM.

### Challenge 2: Are class and module nodes important?

**Problem:** Function-only embedding fails for module-level and class-level questions.

**Resolution:** All three entity types get tailored embed text. Module: synthesised summary (docstring + exports + imports). Class: docstring + base classes + method inventory + attributes. Each answers a different query category.

### Challenge 3: Vector store explosion on large repos

**Problem:** PyTorch has ~80,000 Python functions. Naive embedding degrades quality and inflates cost.

**Resolution:** Three-tier filtering before embedding. Tier 1 (embed + graph): has docstring, complex logic, important dunders. Tier 2 (graph only): short private helpers, simple dunders, straight-line code. Tier 3 (skip): test files, auto-generated, pure delegators. Expected: 80,000 → ~15,000–20,000 embeddings.

### Challenge 4: Per-language extractor classes vs generalised extractor

**Problem:** Five separate extractor classes (`python.py`, `typescript.py`, etc.) repeat too much boilerplate. But a fully generalised extractor can't eliminate language-specific logic for docstrings and signatures.

**Resolution:** Hybrid approach. tree-sitter query DSL for **node discovery** (structural pattern matching — identical mechanism across all languages). Small language-specific **interpreter functions** for docstrings, signature assembly, return type extraction. `ENTITY_QUERIES` dict + `INTERPRETERS` dict replace the class hierarchy. Adding a new language = add two dict entries.

**Key insight:** Docstring extraction cannot be generalised — Python docstrings are first body statements; TypeScript/Go docstrings are preceding comment nodes. Signature assembly cannot be generalised — Go has receivers and multiple return types; TypeScript has arrow functions. These require language-specific code regardless of architecture.

### Challenge 5: Call resolution accuracy

**Problem:** `foo()` might refer to a same-file function, an imported function, or an external library call. Wrong resolution creates false CALLS edges.

**Resolution:** Two-pass strategy. Pass 1: build repo-wide UUID lookup across all files. Pass 2: resolve in order — (1) same-file by name, (2) same-file by qualified name, (3) repo-wide by name. Store unresolved as `CALLS_UNKNOWN` with raw callee text. Never create false positive `CALLS` edges.

### Challenge 6: Ingestion idempotency

**Problem:** Re-running on a partially-indexed repo should not create duplicates or stale data.

**Resolution:** Deterministic UUIDs (MD5 of `repo_id + file_path + qualified_name`). All Neo4j writes use `MERGE`. Vector store writes use `upsert`. JSON checkpoint file for `--resume` support.

### Challenge 7: Incomplete graph hierarchy

**Problem:** Original design (Module, Class, Function) misses both ends — structural top (packages, repo) and semantic bottom (attributes, parameters).

**Resolution:** Added `RepositoryNode` (graph root, enables multi-repo), `PackageNode` (Python import system, subsystem queries), `AttributeNode` (type-aware attribute queries), `ParameterNode` (type-aware parameter queries, Tier 2), `ExternalDependencyNode` (first-class external library nodes). `Interface/Protocol` implemented as `is_protocol: bool` property on ClassNode — in Python, Protocols are classes.

### Challenge 8: Async function deduplication in tree-sitter

**Problem:** Python's grammar wraps `async def` as `async_function_definition → function_definition`. Both nodes appear in query captures, causing every async function to be processed twice.

**Resolution:** Build `{start_byte: wrapper_node}` dict from `fn.async_wrapper` captures. For each `fn.def`, check if a wrapper exists at the same `start_byte`. Use wrapper if found (correct `is_async` flag). Track `seen_bytes` set to deduplicate.

### Challenge 9: `IMPORTS_EXTERNAL` needs node creation, not just edge creation

**Problem:** `write_relationships()` expects both `src` and `dst` nodes to already exist. `ExternalDependency` nodes don't exist before the first `IMPORTS_EXTERNAL` relationship is written.

**Resolution:** `write_external_dependencies()` is a separate method that upserts the `ExternalDependency` node and the edge in a single Cypher statement using `MERGE`. Must be called before or instead of `write_relationships()` for `IMPORTS_EXTERNAL` rels.

---

## Implementation Order

Build in this order to get a working end-to-end skeleton before adding complexity:

1. `config.py` — zero dependencies, forces all naming decisions upfront
2. `ingestion/models.py` — zero dependencies, defines all data contracts
3. `ingestion/tiering.py` — pure function, no external deps, easy to unit test
4. `ingestion/walker.py` — pure file I/O, no tree-sitter needed
5. `graph/schema.py` — just Cypher strings + init function
6. `ingestion/writers/neo4j_writer.py` — needs Neo4j running locally (Docker)
7. `ingestion/extractor.py` — test on a tiny 3-function fixture `.py` file first
8. `ingestion/chunker.py` — depends only on models
9. `ingestion/embedder.py` — start with a stub returning a fixed vector; replace with OpenAI
10. `ingestion/writers/vector_writer.py` — depends on embedder stub
11. `ingestion/relationships.py` — depends on models and Neo4j writer
12. `ingestion/pipeline.py` + `cli.py` — wires everything together
13. End-to-end test on `requests` library (small, well-documented, Python-only)
14. Scale test on `transformers` — validate tiering numbers and ingestion time

---

## Technology Stack

| Component | Choice | Notes |
|---|---|---|
| AST parsing | `tree-sitter` + `py-tree-sitter` | Multi-language, battle-tested |
| Graph DB | Neo4j (local Docker) | Cypher is expressive; good Python SDK (`neo4j` package) |
| Vector store | Chroma (dev) / pgvector (prod) | Same upsert interface; swap via config |
| Embeddings | `text-embedding-3-large` (OpenAI) or CodeBERT | OpenAI easiest to start; CodeBERT for offline |
| Agent framework | LangGraph (Phase 2) | Explicit state machine; good for interview explanation |
| LLM | GPT-4o or Claude Sonnet | Swappable via env var |
| API | FastAPI | Async, clean |
| Demo UI | Streamlit | Fast to build |
| CLI | Click | |
| Config | `python-dotenv` + dataclass | |

Install:
```bash
pip install tree-sitter tree-sitter-python tree-sitter-typescript tree-sitter-go
pip install neo4j chromadb openai click fastapi streamlit python-dotenv
```

---

## Phase 2 Preview — Query Agent

Planned components under `agent/`. Not yet implemented.

### `router.py` — query classifier

Classifies incoming query as one of: `graph`, `vector`, `hybrid`.

- `graph`: structural questions — "what calls X", "what inherits from Y", "what does package Z export"
- `vector`: semantic questions — "find functions that handle authentication", "what does module X do"
- `hybrid`: combined — "find callers of process_payment that deal with retries" (graph for callers, vector for retry semantic)

### `graph_retriever.py` — multi-hop Cypher

Example multi-hop call chain query:
```cypher
MATCH path = (caller:Function)-[:CALLS*1..3]->(target:Function {name: $name, repo_id: $repo_id})
RETURN caller.name, caller.file_path, caller.line_start, caller.docstring, length(path) AS depth
ORDER BY depth
LIMIT 20
```

### `vector_retriever.py` — similarity search

Runs similarity search, deduplicates by `uuid` (collapses multi-chunk results to one context entry per entity), returns top-k `EmbedDoc.metadata` dicts.

### `reranker.py` — hybrid scoring

Scores each retrieved result by:
- Semantic relevance score (from vector search)
- Graph proximity score (callers 1 hop away rank higher than 3 hops)
- Entity type weight (function body > class summary > module summary for most queries)

### `synthesiser.py` — LLM prompt builder

Assembles the final prompt including retrieved context with provenance:

```
Question: {user_query}

Context:
[Function: payments.core.PaymentProcessor.process]
File: payments/core.py, lines 42–89
Signature: def process(self, amount: float, user_id: str) -> bool
Called by: payments.api.create_charge (line 156), payments.retry.attempt (line 23)
---
{full_body}

Answer the question using only the provided context.
Cite every claim with the function name and file location.
```

---

*End of project context document. Last updated: Phase 1 fully designed — models, extractor, relationship resolver, and pipeline written. Phase 2 designed but not implemented.*
