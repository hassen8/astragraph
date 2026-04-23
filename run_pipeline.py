"""
run_pipeline.py

A scratchpad / inspection tool for the AstraGraph ingestion pipeline.
Runs extraction + relationship resolution against a repo and prints a
detailed report of what was extracted — without touching Neo4j or Chroma.

Use this file to:
  - Inspect what the extractors actually produce
  - Debug extraction issues on a specific file or repo
  - Experiment with changes to the pipeline before wiring them up

Usage:
    uv run python run_pipeline.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser

# Make sure the project root is on the path so we can import ingestion.*
# without installing the package. This is only needed when running this
# script directly (not via `uv run python -m ...`).
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.extractors.extractor import extract_file
from ingestion.models import Relationship, RepositoryNode, make_uuid
from ingestion.relationships import RelationshipResolver, build_repo_fn_lookup
from ingestion.walker import walk_repo

# ---------------------------------------------------------------------------
# Config — change these to point at any repo you want to inspect
# ---------------------------------------------------------------------------

REPO_PATH = "/tmp/fastapi"   # absolute path to the repo to analyse
REPO_ID   = "fastapi"        # stable identifier used in all UUIDs for this repo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo_node() -> RepositoryNode:
    """
    Build the root RepositoryNode for this run.

    repo_id is used as the first segment of every UUID in the graph:
        make_uuid(repo_id, file_path, qualified_name)
    Keeping it stable across re-runs is what makes all writes idempotent.
    """
    return RepositoryNode(
        uuid=make_uuid(REPO_ID),
        name=REPO_ID,
        remote_url="https://github.com/tiangolo/fastapi",
        language="python",
        description=None,
        repo_id=REPO_ID,
    )


def _hr(char: str = "─", width: int = 70) -> str:
    return char * width


def _print_sample(label: str, obj: object) -> None:
    """Pretty-print a dataclass instance, truncating long strings and lists."""
    print(f"\n  {label}")
    for field, value in vars(obj).items():
        if isinstance(value, str) and len(value) > 80:
            value = value[:77] + "..."
        if isinstance(value, list) and len(value) > 5:
            value = value[:5] + [f"… +{len(value)-5} more"]
        print(f"    {field:<20} {value!r}")


def _print_rel(label: str, rel) -> None:
    """Pretty-print a Relationship object."""
    print(f"\n  {label}")
    print(f"    rel_type             {rel.rel_type!r}")
    print(f"    src_uuid             {rel.src_uuid!r}")
    print(f"    dst_uuid             {rel.dst_uuid!r}")
    if rel.properties:
        print(f"    properties           {rel.properties!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # ------------------------------------------------------------------
    # Set up the tree-sitter parser for Python.
    #
    # tree-sitter works in two steps:
    #   1. Language object — wraps the compiled C grammar for a language.
    #      Created once and reused across all files.
    #   2. Parser — takes a Language and can parse source bytes into a CST
    #      (Concrete Syntax Tree). The CST is a tree of Node objects where
    #      each node has a type (e.g. "function_definition"), a byte range,
    #      and zero or more named children.
    #
    # In the real pipeline (ingestion/pipeline.py), we cache one Language
    # object and one Parser per language across the entire run. Here we
    # just create them once since we only handle Python.
    # ------------------------------------------------------------------
    lang_obj = Language(tree_sitter_python.language())
    parser   = Parser(lang_obj)

    repo_node = _make_repo_node()

    # ------------------------------------------------------------------
    # Shared registries — the cross-file memory of the resolver.
    #
    # These dicts are passed into RelationshipResolver and filled as each
    # file is processed. They allow the resolver to resolve references
    # that span files:
    #
    #   module_registry:  module_name  -> ModuleNode
    #   class_registry:   qualified_name (and bare name) -> ClassNode
    #   package_registry: package_name -> PackageNode
    #
    # Example: if payments/core.py imports from payments/utils.py, the
    # IMPORTS relationship can only be resolved if payments.utils is already
    # in module_registry. The walker processes files in directory order,
    # so parent packages are usually seen before their children.
    # ------------------------------------------------------------------
    module_registry:  dict = {}
    class_registry:   dict = {}
    package_registry: dict = {}

    resolver = RelationshipResolver(
        repo=repo_node,
        module_registry=module_registry,
        class_registry=class_registry,
        package_registry=package_registry,
    )

    # Accumulate everything across files for the summary + Pass 2
    all_modules    = []
    all_classes    = []
    all_functions  = []
    all_attributes = []
    all_parameters = []
    all_raw_calls  = []   # raw call dicts {src_uuid, callee, call_site_line, is_conditional}
    all_pass1_rels = []

    print(_hr("═"))
    print("  AstraGraph — extraction run")
    print(f"  repo : {REPO_PATH}")
    print(f"  id   : {REPO_ID}")
    print(_hr("═"))

    # ------------------------------------------------------------------
    # Pass 1 — per-file extraction + relationship resolution
    #
    # For each file the walker yields:
    #   1. extract_file() — runs the language-specific builder (python.py)
    #      against the tree-sitter CST and returns:
    #        module      — one ModuleNode for the file
    #        package     — one PackageNode for the containing directory
    #        classes     — list[ClassNode]
    #        functions   — list[FunctionNode] (includes methods)
    #        attributes  — list[AttributeNode]
    #        parameters  — list[ParameterNode]
    #        raw_calls   — list[dict] — unresolved call sites, deferred to Pass 2
    #
    #   2. resolver.resolve_pass1() — builds all relationships that can be
    #      resolved from a single file: PART_OF, CONTAINS, DEFINED_IN,
    #      METHOD_OF, DEFINES_ATTR, HAS_PARAM, INHERITS, IMPORTS.
    #      Also registers module/classes/package into the shared registries
    #      so later files can reference them.
    #
    # CALLS are intentionally NOT resolved here. A function in file A may
    # call a function in file B that hasn't been parsed yet. Raw call dicts
    # are accumulated and resolved in Pass 2 once the full repo is walked.
    # ------------------------------------------------------------------
    print("\nPass 1 — extracting entities ...\n")

    files = list(walk_repo(REPO_PATH))
    for i, (rel_path, language) in enumerate(files, 1):
        abs_path = Path(REPO_PATH) / rel_path
        try:
            source = abs_path.read_bytes()
        except OSError:
            continue

        # parse() returns a Tree; .root_node is the root of the CST.
        # source is passed as bytes — tree-sitter works on raw bytes, not strings,
        # so Unicode is handled correctly for all encodings.
        tree = parser.parse(source)

        module, package, classes, functions, attributes, parameters, raw_calls = extract_file(
            file_path=rel_path,
            language=language,
            root=tree.root_node,
            source=source,
            lang_obj=lang_obj,
            repo_id=REPO_ID,
            repo_root=REPO_PATH,  # needed by build_package() to check __init__.py on disk
        )

        pass1_rels = resolver.resolve_pass1(
            module=module,
            package=package,
            classes=classes,
            functions=functions,
            attributes=attributes,
            parameters=parameters,
        )

        all_modules.append(module)
        all_classes.extend(classes)
        all_functions.extend(functions)
        all_attributes.extend(attributes)
        all_parameters.extend(parameters)
        all_raw_calls.extend(raw_calls)
        all_pass1_rels.extend(pass1_rels)

        print(f"  [{i:>3}/{len(files)}]  {rel_path:<55}"
              f"  cls={len(classes):>3}  fn={len(functions):>4}  rels={len(pass1_rels):>4}")

    # ------------------------------------------------------------------
    # Pass 2 — resolve CALLS across the whole repo
    #
    # Now that every file has been parsed, we have a complete picture of
    # every function defined in the repo. build_repo_fn_lookup() builds a
    # flat dict indexed by both bare name and qualified name:
    #
    #   "process_payment"                  -> FunctionNode
    #   "payments.core.process_payment"    -> FunctionNode
    #
    # resolve_calls() walks the raw call dicts and tries to find a matching
    # FunctionNode. If found: emits a CALLS edge. If not: keeps CALLS_UNKNOWN.
    # Unresolved calls are calls to stdlib, third-party libs, or dynamic
    # dispatch that can't be statically resolved.
    #
    # Resolution rate of ~29% for FastAPI is expected — most calls are to
    # Starlette, Pydantic, or stdlib which aren't in the repo.
    # ------------------------------------------------------------------
    print(f"\nPass 2 — resolving calls ({len(all_raw_calls)} call sites) ...\n")

    repo_fn_lookup = build_repo_fn_lookup(all_functions)

    # Wrap raw call dicts in Relationship objects so the resolver has a
    # uniform interface. dst_uuid=None signals "not yet resolved".
    raw_call_rels = [
        Relationship(
            src_uuid=c["src_uuid"],
            dst_uuid=None,
            rel_type="CALLS_UNKNOWN",
            properties={
                "callee":         c["callee"],
                "call_site_line": c["call_site_line"],
                "is_conditional": c["is_conditional"],
            },
        )
        for c in all_raw_calls
    ]

    pass2_rels = resolver.resolve_calls(raw_call_rels, repo_fn_lookup)

    resolved_calls   = [r for r in pass2_rels if r.rel_type == "CALLS"]
    unresolved_calls = [r for r in pass2_rels if r.rel_type == "CALLS_UNKNOWN"]

    all_rels = all_pass1_rels + pass2_rels

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(_hr("═"))
    print("  ENTITY SUMMARY")
    print(_hr())
    print(f"  {'Files processed':<30} {len(files)}")
    print(f"  {'Modules':<30} {len(all_modules)}")
    print(f"  {'Classes':<30} {len(all_classes)}")
    print(f"  {'Functions':<30} {len(all_functions)}")
    print(f"  {'Attributes':<30} {len(all_attributes)}")
    print(f"  {'Parameters':<30} {len(all_parameters)}")
    print(_hr())
    print("  RELATIONSHIP SUMMARY")
    print(_hr())
    rel_counts: dict[str, int] = defaultdict(int)
    for r in all_rels:
        rel_counts[r.rel_type] += 1
    for rel_type, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel_type:<30} {count:>6}")
    print(f"  {'(CALLS resolved)':<30} {len(resolved_calls):>6}")
    print(f"  {'(CALLS unresolved)':<30} {len(unresolved_calls):>6}")
    print(_hr("═"))

    # ------------------------------------------------------------------
    # Sample nodes — one per entity type
    #
    # Useful for eyeballing whether the extractor is pulling the right
    # fields. If something looks wrong here, check python.py.
    # ------------------------------------------------------------------
    print("\nSAMPLE NODES")

    print(f"\n{_hr()}")
    print("  RepositoryNode")
    _print_sample("fastapi", repo_node)

    print(f"\n{_hr()}")
    print("  ModuleNode  (routing.py — largest file)")
    routing = next((m for m in all_modules if "routing" in m.name), all_modules[0])
    _print_sample(routing.name, routing)

    print(f"\n{_hr()}")
    print("  ClassNode  (with docstring + methods)")
    sample_cls = next(
        (c for c in all_classes if c.method_names and c.docstring),
        all_classes[0]
    )
    _print_sample(sample_cls.qualified_name, sample_cls)

    print(f"\n{_hr()}")
    print("  FunctionNode  (with docstring + complexity > 2, not a method)")
    sample_fn = next(
        (f for f in all_functions if f.docstring and f.complexity > 2 and not f.is_method),
        next((f for f in all_functions if f.docstring), all_functions[0])
    )
    _print_sample(sample_fn.qualified_name, sample_fn)

    print(f"\n{_hr()}")
    print("  AttributeNode")
    if all_attributes:
        _print_sample(all_attributes[0].full_name, all_attributes[0])
    else:
        print("  (none extracted)")

    print(f"\n{_hr()}")
    print("  ParameterNode  (typed, non-self)")
    sample_param = next(
        (p for p in all_parameters if not p.is_self and p.type_hint),
        next((p for p in all_parameters if not p.is_self), None)
    )
    if sample_param:
        _print_sample(sample_param.name, sample_param)
    else:
        print("  (none extracted)")

    # ------------------------------------------------------------------
    # Sample relationships — two examples per type
    #
    # For resolved edges, src_uuid and dst_uuid both point to real nodes.
    # For unresolved edges (CALLS_UNKNOWN, INHERITS_UNKNOWN), dst_uuid=None
    # and the target is in properties (e.g. properties["callee"]).
    # In the real pipeline, unresolved edges are written as :Unresolved
    # audit nodes — they're never silently dropped.
    # ------------------------------------------------------------------
    print(f"\n{_hr('═')}")
    print("SAMPLE RELATIONSHIPS")

    rels_by_type: dict[str, list] = defaultdict(list)
    for r in all_rels:
        rels_by_type[r.rel_type].append(r)

    display_order = [
        "PART_OF", "CONTAINS", "DEFINED_IN", "METHOD_OF",
        "DEFINES_ATTR", "HAS_PARAM", "INHERITS", "INHERITS_UNKNOWN",
        "IMPORTS", "IMPORTS_EXTERNAL", "CALLS", "CALLS_UNKNOWN",
    ]

    for rel_type in display_order:
        examples = rels_by_type.get(rel_type, [])
        if not examples:
            continue
        print(f"\n{_hr()}")
        print(f"  {rel_type}  ({len(examples)} total)")
        for rel in examples[:2]:
            _print_rel("  example", rel)

    print(f"\n{_hr('═')}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
