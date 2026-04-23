"""
ingestion/extractors/extractor.py

Entrypoint for source code extraction. Provides the ExtractionContext which manages
tree-sitter queries and source texts, then hands off the AST processing to language-specific builders.
"""

from __future__ import annotations

import os
from typing import Optional

from tree_sitter import Language, Node, Query, QueryCursor

from ..models import (
    FunctionNode,
    ClassNode,
    ModuleNode,
    PackageNode,
    AttributeNode,
    ParameterNode,
    make_uuid,
)


def load_queries(language: str) -> dict[str, str]:
    """
    Load S-expression queries from the queries/ directory.
    Parses `;; @query: name` delimiters to split into a dictionary.
    """
    directory = os.path.dirname(__file__)
    query_file = os.path.join(directory, "queries", f"{language}.scm")
    
    queries = {}
    if not os.path.exists(query_file):
        return queries

    with open(query_file, "r", encoding="utf-8") as f:
        content = f.read()

    current_query_name = None
    current_query_body = []
    
    for line in content.splitlines():
        trimmed = line.strip()
        if trimmed.startswith(";; @query:"):
            if current_query_name:
                queries[current_query_name] = "\n".join(current_query_body).strip()
            current_query_name = trimmed.split(";; @query:")[1].strip()
            current_query_body = []
        else:
            if current_query_name is not None:
                current_query_body.append(line)

    if current_query_name is not None:
        queries[current_query_name] = "\n".join(current_query_body).strip()

    return queries


class ExtractionContext:
    """
    Holds shared state (source, root, queries) and provides utilities for
    the language-specific builders.
    """
    def __init__(
        self,
        language: str,
        root: Node,
        source: bytes,
        file_path: str,
        repo_id: str,
        lang_obj: Language,
        repo_root: str = "",
    ):
        self.language  = language
        self.root      = root
        self.source    = source
        self.file_path = file_path
        self.repo_id   = repo_id
        self.lang_obj  = lang_obj
        self.repo_root = repo_root

        self._queries = load_queries(language)
        self._query_cache: dict[str, object] = {}

    def query(self, name: str) -> Optional[Query]:
        """Return a compiled tree-sitter Query object, cached after first compile."""
        if name not in self._query_cache:
            query_str = self._queries.get(name, "")
            if not query_str.strip():
                return None
            try:
                self._query_cache[name] = Query(self.lang_obj, query_str)
            except Exception as e:
                print(f"Error compiling query '{name}': {e}")
                return None
        return self._query_cache[name]

    def captures(self, query_name: str, node: Node) -> dict[str, list[Node]]:
        """
        Run a named query against a node and return captures grouped by capture name.
        """
        q = self.query(query_name)
        if q is None:
            return {}
        cursor = QueryCursor(q)
        return cursor.captures(node)

    def src(self, node: Node) -> str:
        """Extract the exact string for a node."""
        if node is None:
            return ""
        return self.source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def make_uuid(self, qualified_name: str) -> str:
        """Determine a strictly deterministic UUID."""
        return make_uuid(self.repo_id, self.file_path, qualified_name)


def extract_file(
    file_path: str,
    language: str,
    root: Node,
    source: bytes,
    lang_obj: Language,
    repo_id: str,
    repo_root: str = "",
) -> tuple[ModuleNode, PackageNode, list[ClassNode], list[FunctionNode], list[AttributeNode], list[ParameterNode], list[dict]]:
    """
    Extract structurally parsed domain objects from a file and raw call dictionaries.
    Delegates entirely to a language-specific builder.
    """
    ctx = ExtractionContext(
        language=language,
        root=root,
        source=source,
        file_path=file_path,
        repo_id=repo_id,
        lang_obj=lang_obj,
        repo_root=repo_root,
    )

    if language == "python":
        from .python import PythonBuilder
        builder = PythonBuilder(ctx)
    elif language == "typescript":
        from .typescript import TypeScriptBuilder
        builder = TypeScriptBuilder(ctx)
    elif language == "go":
        from .go import GoBuilder
        builder = GoBuilder(ctx)
    else:
        raise ValueError(f"Unsupported language builder for {language}")

    module     = builder.build_module()
    package    = builder.build_package()
    classes    = builder.build_classes()
    functions  = builder.build_functions()
    attributes = getattr(builder, "extracted_attributes", [])
    parameters = getattr(builder, "extracted_parameters", [])
    calls      = builder.build_calls()

    return module, package, classes, functions, attributes, parameters, calls
