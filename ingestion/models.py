import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Optional

def make_uuid(repo_id: str, file_path: str, qualified_name: str) -> str:
    """
    Deterministic UUID — same entity always gets the same UUID across re-ingestion.
    This makes the ingestion pipeline idempotent (safe to re-run).
    """
    key = f"{repo_id}::{file_path}::{qualified_name}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))

@dataclass
class FunctionNode:
    uuid: str
    name: str
    qualified_name: str       # e.g. "payments.core.PaymentProcessor.process"
    file_path: str            # relative to repo root
    line_start: int
    line_end: int
    language: str             # "python" | "typescript" | "go" | "java"
    signature: str            # full signature string
    docstring: Optional[str]  # cleaned docstring or None
    return_type: Optional[str]
    is_async: bool
    is_method: bool           # True if defined inside a class
    decorators: list[str]     # e.g. ["@staticmethod", "@cached_property"]
    body_preview: str         # first 300 chars — used for tiering decisions only
    full_body: str            # complete raw source of the function
    complexity: int           # cyclomatic complexity
    repo_id: str

@dataclass
class ClassNode:
    uuid: str
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    language: str
    docstring: Optional[str]
    base_classes: list[str]   # list of base class name strings
    decorators: list[str]
    is_abstract: bool
    method_names: list[str]   # inventory of method names (for embedding)
    attribute_names: list[str] # self.x attributes found in __init__
    repo_id: str

@dataclass
class ModuleNode:
    uuid: str
    name: str                 # dot-separated: "payments.core"
    file_path: str
    language: str
    docstring: Optional[str]  # module-level docstring
    exported_names: list[str] # functions + classes defined at module level
    imported_modules: list[str] # raw import strings
    repo_id: str

@dataclass
class ParameterNode:
    uuid: str
    name: str
    type_hint: Optional[str]
    default: Optional[str]
    position: int
    parent_uuid: str          # UUID of owning FunctionNode

@dataclass
class Relationship:
    src_uuid: str
    dst_uuid: Optional[str]   # None for unresolved external calls
    rel_type: str             # "CALLS" | "IMPORTS" | "INHERITS" | "METHOD_OF" | etc.
    properties: dict = field(default_factory=dict)
    # For CALLS: {"call_site_line": int, "is_conditional": bool}
    # For IMPORTS: {"alias": str | None, "imported_names": list[str]}
    # For INHERITS: {"is_direct": bool}

@dataclass
class EmbedDoc:
    uuid: str           # matches the graph node UUID exactly
    chunk_index: int    # 0 for single-chunk entities, 0..N for sliding window
    total_chunks: int   # total number of chunks for this entity
    text: str           # the text that gets embedded
    entity_type: str    # "function" | "class" | "module" | "callsite"
    tier: int           # 1 = always embed, 2 = graph-only, 3 = skip
    metadata: dict      # file_path, line_start, line_end, name, language, docstring, repo_id
