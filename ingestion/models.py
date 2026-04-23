"""
ingestion/models.py

Core data models for all graph entities and relationships.

Hierarchy:
    Repository
        └── Package          (directory with __init__.py)
                └── Module   (single source file)
                        ├── Class
                        │     ├── Function  (method)
                        │     └── Attribute
                        └── Function        (module-level)
                              └── Parameter

All UUIDs are deterministic — the same entity always produces the same UUID
across re-ingestion runs, making all Neo4j writes idempotent (MERGE-safe).

Embedding strategy:
    Functions are embedded as a single EmbedDoc (no chunking — functions are
    atomic units, not prose). Whether a function gets embedded at all is decided
    by should_embed(fn) -> bool in ingestion/embedder.py. Functions with no
    docstring, fewer than 5 lines, and a private name are skipped.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# UUID generation
# ---------------------------------------------------------------------------

def make_uuid(*parts: str) -> str:
    """
    Deterministic UUID from an arbitrary number of string parts.
    Stable across re-runs — same inputs always produce the same UUID.
    Safe to use as Neo4j MERGE key and vector store document ID.

    Examples:
        make_uuid("my-repo", "payments/core.py", "payments.core.process_payment")
        make_uuid("my-repo")   # for Repository node
        make_uuid("my-repo", "payments")  # for Package node
    """
    key = "::".join(parts)
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


# ---------------------------------------------------------------------------
# Top-level structural nodes
# ---------------------------------------------------------------------------

@dataclass
class RepositoryNode:
    """
    Root node for an entire repository.
    One per ingestion run. Entry point for all repo-scoped queries.

    Neo4j label: Repository
    """
    uuid:        str
    name:        str            # e.g. "transformers", "payments-service"
    remote_url:  Optional[str]  # e.g. "https://github.com/huggingface/transformers"
    language:    str            # primary language, e.g. "python"
    description: Optional[str]  # from README first paragraph or pyproject.toml
    repo_id:     str            # same as uuid for root node — kept for join consistency


@dataclass
class PackageNode:
    """
    A Python package — a directory containing __init__.py.
    Packages nest: payments/ contains payments/core/, payments/retry/, etc.

    Neo4j label: Package
    Relationships emitted:
        (Package)-[:PART_OF]->(Package)      nested package -> parent package
        (Package)-[:PART_OF]->(Repository)   top-level package -> repo
        (Package)-[:CONTAINS]->(Module)      written by RelationshipResolver
    """
    uuid:          str
    name:          str            # dot-separated: "payments.core" for payments/core/
    directory:     str            # relative path: "payments/core"
    is_namespace:  bool           # True if no __init__.py (PEP 420 namespace package)
    has_init:      bool           # True if __init__.py exists
    init_file:     Optional[str]  # relative path to __init__.py if it exists
    repo_id:       str


# ---------------------------------------------------------------------------
# File-level node
# ---------------------------------------------------------------------------

@dataclass
class ModuleNode:
    """
    A single source file.

    Neo4j label: Module
    Relationships emitted:
        (Module)-[:PART_OF]->(Package)
        (Module)-[:IMPORTS]->(Module)
        (Module)-[:IMPORTS_EXTERNAL]->(ExternalDependency)
        (Module)-[:CONTAINS]->(Function)
        (Module)-[:CONTAINS]->(Class)
    """
    uuid:             str
    name:             str            # dot-separated: "payments.core"
    file_path:        str            # relative to repo root: "payments/core.py"
    language:         str
    docstring:        Optional[str]
    exported_names:   list[str]      # top-level function + class names
    imported_modules: list[str]      # raw import statement strings
    is_init:          bool           # True if this is an __init__.py file
    repo_id:          str


# ---------------------------------------------------------------------------
# Class-level node
# ---------------------------------------------------------------------------

@dataclass
class ClassNode:
    """
    A class definition.

    Neo4j label: Class
    Relationships emitted:
        (Class)-[:DEFINED_IN]->(Module)
        (Class)-[:INHERITS]->(Class)
        (Class)-[:METHOD_OF] — not used; inverse is (Function)-[:METHOD_OF]->(Class)
    """
    uuid:            str
    name:            str
    qualified_name:  str            # "payments.core.PaymentProcessor"
    file_path:       str
    line_start:      int
    line_end:        int
    language:        str
    docstring:       Optional[str]
    base_classes:    list[str]      # raw base class name strings
    decorators:      list[str]
    is_abstract:     bool           # inherits from ABC or ABCMeta
    is_protocol:     bool           # inherits from typing.Protocol
    is_dataclass:    bool           # decorated with @dataclass
    is_exception:    bool           # inherits from Exception or BaseException
    method_names:    list[str]      # inventory — for class embed text
    attribute_names: list[str]      # self.x names — for class embed text
    repo_id:         str


# ---------------------------------------------------------------------------
# Function / Method node
# ---------------------------------------------------------------------------

@dataclass
class FunctionNode:
    """
    A function or method definition.

    is_method=True means this function is defined inside a class body.
    The METHOD_OF relationship records which class it belongs to.

    Neo4j label: Function
    Relationships emitted:
        (Function)-[:DEFINED_IN]->(Module)
        (Function)-[:METHOD_OF]->(Class)       if is_method=True
        (Function)-[:CALLS]->(Function)
        (Function)-[:HAS_PARAM]->(Parameter)   Tier 2 — graph only
    """
    uuid:           str
    name:           str
    qualified_name: str             # "payments.core.PaymentProcessor.process"
    file_path:      str
    line_start:     int
    line_end:       int
    language:       str
    signature:      str             # full def line e.g. "def process(self, amount: float) -> bool"
    docstring:      Optional[str]
    return_type:    Optional[str]
    is_async:       bool
    is_method:      bool
    is_property:    bool            # decorated with @property
    is_classmethod: bool            # decorated with @classmethod
    is_staticmethod: bool           # decorated with @staticmethod
    is_overload:    bool            # decorated with @typing.overload
    decorators:     list[str]
    body_preview:   str             # first 300 chars — used by should_embed() filter
    full_body:      str             # complete source text of the function
    complexity:     int             # cyclomatic complexity
    repo_id:        str


# ---------------------------------------------------------------------------
# Parameter node  (Tier 2 — graph only, no embedding)
# ---------------------------------------------------------------------------

@dataclass
class ParameterNode:
    """
    A single parameter in a function signature.

    Stored in the graph for type-aware structural queries:
        "find all functions that accept a StripeClient argument"

    Not embedded in the vector store — the parent function's signature
    embed text already captures parameter names and types.

    Neo4j label: Parameter
    Relationships emitted:
        (Function)-[:HAS_PARAM]->(Parameter)
    """
    uuid:         str
    name:         str
    type_hint:    Optional[str]     # "float", "Optional[str]", "StripeClient", etc.
    default:      Optional[str]     # raw default expression string or None
    position:     int               # 0-indexed position in signature
    is_self:      bool              # True for self/cls parameters
    is_variadic:  bool              # True for *args
    is_keyword:   bool              # True for **kwargs
    parent_uuid:  str               # UUID of owning FunctionNode
    repo_id:      str


# ---------------------------------------------------------------------------
# Attribute node
# ---------------------------------------------------------------------------

@dataclass
class AttributeNode:
    """
    A class attribute — either a class-level variable or an instance
    attribute assigned in __init__ (self.x = ...).

    Type information is extracted from type annotations when present.
    Without a type annotation, type_hint is None.

    Neo4j label: Attribute
    Relationships emitted:
        (Class)-[:DEFINES_ATTR]->(Attribute)
    """
    uuid:         str
    name:         str               # bare attribute name e.g. "client"
    full_name:    str               # "self.client" or "ClassName.class_var"
    type_hint:    Optional[str]     # from annotation if present: "StripeClient"
    default:      Optional[str]     # raw default expression string or None
    is_instance:  bool              # True = self.x in __init__; False = class-level
    is_class_var: bool              # True if explicitly typed as ClassVar[...]
    line:         int               # line where first assigned/declared
    parent_uuid:  str               # UUID of owning ClassNode
    repo_id:      str


# ---------------------------------------------------------------------------
# External dependency node
# ---------------------------------------------------------------------------

@dataclass
class ExternalDependencyNode:
    """
    An external library or package that is imported but not part of this repo.
    Examples: "stripe", "numpy", "torch", "sqlalchemy"

    This node type makes dependency queries first-class:
        "which modules depend on stripe?"
        "find all repos that use torch"

    Unlike Unresolved nodes (which are implementation noise), these are
    intentional graph citizens representing real external dependencies.

    Neo4j label: ExternalDependency
    Relationships emitted:
        (Module)-[:IMPORTS_EXTERNAL]->(ExternalDependency)
    """
    uuid:            str
    name:            str            # top-level package name e.g. "stripe", "numpy"
    imported_names:  list[str]      # specific names imported: ["StripeClient", "Charge"]
    raw_import:      str            # original import statement text
    repo_id:         str


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

@dataclass
class Relationship:
    """
    A directed edge between two graph nodes.

    dst_uuid=None means the target could not be resolved (external or forward
    reference). These are written to :Unresolved nodes, not dropped.

    Relationship types and their src -> dst node types:

    Structural hierarchy:
        PART_OF         Package    -> Package | Repository
        CONTAINS        Package    -> Module
        CONTAINS        Module     -> Function | Class (redundant with DEFINED_IN but
                                     useful for "what's in this module?" traversals)

    Ownership:
        DEFINED_IN      Function   -> Module
        DEFINED_IN      Class      -> Module
        METHOD_OF       Function   -> Class
        DEFINES_ATTR    Class      -> Attribute
        HAS_PARAM       Function   -> Parameter

    Cross-entity semantic:
        INHERITS        Class      -> Class
        CALLS           Function   -> Function
        IMPORTS         Module     -> Module
        IMPORTS_EXTERNAL Module    -> ExternalDependency

    Unresolved:
        INHERITS_UNKNOWN  Class    -> None    (base_name in properties)
        CALLS_UNKNOWN     Function -> None    (callee in properties)
        IMPORTS_EXTERNAL  Module   -> None    (when ExternalDependency not yet created)
    """
    src_uuid:   str
    dst_uuid:   Optional[str]
    rel_type:   str
    properties: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Embed document  (vector store record)
# ---------------------------------------------------------------------------

@dataclass
class EmbedDoc:
    """
    A single document to be embedded and stored in the vector store.
    The uuid field links back to the Neo4j node via the same deterministic UUID.

    One EmbedDoc per function — no chunking. Functions are atomic units;
    sliding-window chunking is for prose, not code.

    Only produced for functions that pass should_embed() in ingestion/embedder.py.
    """
    uuid:        str    # matches the Neo4j node UUID exactly — the join key
    text:        str    # the text that gets embedded (signature + docstring + body)
    entity_type: str    # "function" | "class" | "module"
    metadata:    dict   # file_path, line_start, name, language, repo_id, etc.
