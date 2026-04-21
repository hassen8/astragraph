"""
run_pipeline.py

Runs extraction + relationship resolution against the FastAPI repository —
everything up to (but not including) Neo4j / vector-store writes.

Prints:
  - A summary count of all extracted entities
  - One sample node from each entity type (Module, Class, Function, Attribute, Parameter)
  - Two example Relationship objects for each resolved relationship type

Usage:
    uv run python run_pipeline.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.extractors.extractor import extract_file
from ingestion.models import (
    PackageNode,
    RepositoryNode,
    make_uuid,
)
from ingestion.relationships import RelationshipResolver, build_repo_fn_lookup
from ingestion.walker import walk_repo

REPO_PATH = "/home/hsali/projects/fastapi"
REPO_ID   = "fastapi"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo_node() -> RepositoryNode:
    return RepositoryNode(
        uuid=make_uuid(REPO_ID),
        name=REPO_ID,
        remote_url="https://github.com/tiangolo/fastapi",
        language="python",
        description=None,
        repo_id=REPO_ID,
    )


def _make_package_node(rel_path: str) -> PackageNode:
    pkg_dir  = str(Path(rel_path).parent)
    pkg_name = pkg_dir.replace("/", ".") if pkg_dir != "." else REPO_ID
    has_init = (Path(REPO_PATH) / pkg_dir / "__init__.py").exists()
    return PackageNode(
        uuid=make_uuid(REPO_ID, pkg_name),
        name=pkg_name,
        directory=pkg_dir,
        is_namespace=not has_init,
        has_init=has_init,
        init_file=str(Path(pkg_dir) / "__init__.py") if has_init else None,
        repo_id=REPO_ID,
    )


def _hr(char: str = "─", width: int = 70) -> str:
    return char * width


def _print_sample(label: str, obj: object) -> None:
    print(f"\n  {label}")
    for field, value in vars(obj).items():
        if isinstance(value, str) and len(value) > 80:
            value = value[:77] + "..."
        if isinstance(value, list) and len(value) > 5:
            value = value[:5] + [f"… +{len(value)-5} more"]
        print(f"    {field:<20} {value!r}")


def _print_rel(label: str, rel) -> None:
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
    lang_obj = Language(tree_sitter_python.language())
    parser   = Parser(lang_obj)

    repo_node = _make_repo_node()

    module_registry:  dict = {}
    class_registry:   dict = {}
    package_registry: dict = {}

    resolver = RelationshipResolver(
        repo=repo_node,
        module_registry=module_registry,
        class_registry=class_registry,
        package_registry=package_registry,
    )

    # Accumulated across all files
    all_modules    = []
    all_classes    = []
    all_functions  = []
    all_attributes = []
    all_parameters = []
    all_raw_calls  = []   # raw call dicts from the builder
    all_pass1_rels = []

    print(_hr("═"))
    print("  AstraGraph — extraction run")
    print(f"  repo : {REPO_PATH}")
    print(f"  id   : {REPO_ID}")
    print(_hr("═"))

    # ------------------------------------------------------------------
    # Pass 1 — extract entities + resolve per-file relationships
    # ------------------------------------------------------------------
    print("\nPass 1 — extracting entities ...\n")

    files = list(walk_repo(REPO_PATH))
    for i, (rel_path, language) in enumerate(files, 1):
        abs_path = Path(REPO_PATH) / rel_path
        try:
            source = abs_path.read_bytes()
        except OSError:
            continue

        tree = parser.parse(source)
        module, classes, functions, attributes, parameters, raw_calls = extract_file(
            file_path=rel_path,
            language=language,
            root=tree.root_node,
            source=source,
            lang_obj=lang_obj,
            repo_id=REPO_ID,
        )

        package = _make_package_node(rel_path)

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
    # Pass 2 — resolve CALLS
    # ------------------------------------------------------------------
    print(f"\nPass 2 — resolving calls ({len(all_raw_calls)} call sites) ...\n")

    repo_fn_lookup = build_repo_fn_lookup(all_functions)

    # Convert raw call dicts to Relationship objects for the resolver
    from ingestion.models import Relationship
    raw_call_rels = [
        Relationship(
            src_uuid=c["src_uuid"],
            dst_uuid=None,
            rel_type="CALLS_UNKNOWN",
            properties={
                "callee":          c["callee"],
                "call_site_line":  c["call_site_line"],
                "is_conditional":  c["is_conditional"],
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
    # Sample nodes — one per type
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
    print("  ClassNode")
    sample_cls = next(
        (c for c in all_classes if c.method_names and c.docstring),
        all_classes[0]
    )
    _print_sample(sample_cls.qualified_name, sample_cls)

    print(f"\n{_hr()}")
    print("  FunctionNode  (with docstring + complexity > 1)")
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
    # Sample relationships — two per type
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
