"""
ingestion/relationships.py

Resolves all graph relationships from extracted entities and writes them to Neo4j.

Complete relationship set:

  Pass 1 (per-file, resolved immediately):
    PART_OF          Package    -> Package | Repository
    CONTAINS         Package    -> Module
    DEFINED_IN       Function   -> Module
    DEFINED_IN       Class      -> Module
    METHOD_OF        Function   -> Class
    DEFINES_ATTR     Class      -> Attribute
    HAS_PARAM        Function   -> Parameter
    INHERITS         Class      -> Class  (+ INHERITS_UNKNOWN for unresolved)
    IMPORTS          Module     -> Module (+ IMPORTS_EXTERNAL for external libs)

  Pass 2 (after full repo walk):
    CALLS            Function   -> Function (+ CALLS_UNKNOWN for unresolved)

Usage in pipeline.py:

    # Initialise once per ingestion run
    resolver = RelationshipResolver(repo_node, module_registry, class_registry)

    # Pass 1 — call once per file after extraction
    pass1_rels = resolver.resolve_pass1(
        module, classes, functions, attributes, parameters, package
    )
    rel_writer.write_relationships(pass1_rels)

    # Pass 2 — call once after all files are processed
    repo_fn_lookup = build_repo_fn_lookup(all_functions)
    pass2_rels = resolver.resolve_calls(all_raw_call_rels, repo_fn_lookup)
    rel_writer.write_relationships(pass2_rels)
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import (
    AttributeNode,
    ClassNode,
    ExternalDependencyNode,
    FunctionNode,
    ModuleNode,
    PackageNode,
    ParameterNode,
    RepositoryNode,
    Relationship,
    make_uuid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relationship resolver
# ---------------------------------------------------------------------------

class RelationshipResolver:
    """
    Resolves all relationship types from extracted entities.

    Shared registries accumulate as files are processed and enable
    cross-file resolution for INHERITS, IMPORTS, and CALLS.

    Args:
        repo:             The RepositoryNode for this ingestion run.
        module_registry:  Dict[module_name -> ModuleNode]
        class_registry:   Dict[qualified_name -> ClassNode]
                          Also indexed by bare class name as a fallback.
        package_registry: Dict[package_name -> PackageNode]
    """

    def __init__(
        self,
        repo:             RepositoryNode,
        module_registry:  dict[str, ModuleNode],
        class_registry:   dict[str, ClassNode],
        package_registry: dict[str, PackageNode],
    ):
        self.repo     = repo
        self.modules  = module_registry
        self.classes  = class_registry
        self.packages = package_registry

    # ------------------------------------------------------------------
    # Pass 1 — per file
    # ------------------------------------------------------------------

    def resolve_pass1(
        self,
        module:     ModuleNode,
        package:    PackageNode,
        classes:    list[ClassNode],
        functions:  list[FunctionNode],
        attributes: list[AttributeNode],
        parameters: list[ParameterNode],
    ) -> list[Relationship]:
        """
        Resolve all relationships determinable from a single file.
        Registers module, classes, and package into shared registries.

        Returns a flat list of all Relationship objects for this file.
        """
        # Register into shared registries before resolving so that
        # later files in the same run can reference these entities.
        self.modules[module.name] = module
        self.packages[package.name] = package
        for cls in classes:
            self.classes[cls.qualified_name] = cls
            self.classes[cls.name] = cls           # bare name fallback

        rels: list[Relationship] = []

        rels.extend(self._resolve_package_hierarchy(package))
        rels.extend(self._resolve_package_contains_module(package, module))
        rels.extend(self._resolve_defined_in(module, classes, functions))
        rels.extend(self._resolve_method_of(classes, functions))
        rels.extend(self._resolve_defines_attr(classes, attributes))
        rels.extend(self._resolve_has_param(functions, parameters))
        rels.extend(self._resolve_inherits(classes))
        rels.extend(self._resolve_imports(module))

        return rels

    # ------------------------------------------------------------------
    # PART_OF — Package -> Package | Repository
    # ------------------------------------------------------------------

    def _resolve_package_hierarchy(self, package: PackageNode) -> list[Relationship]:
        """
        Emit PART_OF from this package to its parent package (if nested)
        or directly to the Repository (if top-level).

        "payments.core" -> parent = "payments"
        "payments"      -> parent = Repository
        """
        rels = []
        parts = package.name.split(".")

        if len(parts) == 1:
            # Top-level package — points directly to Repository
            rels.append(Relationship(
                src_uuid=package.uuid,
                dst_uuid=self.repo.uuid,
                rel_type="PART_OF",
                properties={"level": "top"},
                src_label="Package",
                dst_label="Repository",
            ))
        else:
            # Nested package — points to immediate parent package
            parent_name = ".".join(parts[:-1])
            parent_pkg  = self.packages.get(parent_name)
            if parent_pkg:
                rels.append(Relationship(
                    src_uuid=package.uuid,
                    dst_uuid=parent_pkg.uuid,
                    rel_type="PART_OF",
                    properties={"level": "nested"},
                    src_label="Package",
                    dst_label="Package",
                ))
            else:
                # Parent package not yet seen — will be registered later.
                # Store as unresolved; pipeline should process parent dirs first.
                rels.append(Relationship(
                    src_uuid=package.uuid,
                    dst_uuid=None,
                    rel_type="PART_OF_UNKNOWN",
                    properties={"parent_package": parent_name},
                    src_label="Package",
                ))

        return rels

    # ------------------------------------------------------------------
    # CONTAINS — Package -> Module
    # ------------------------------------------------------------------

    def _resolve_package_contains_module(
        self, package: PackageNode, module: ModuleNode
    ) -> list[Relationship]:
        return [Relationship(
            src_uuid=package.uuid,
            dst_uuid=module.uuid,
            rel_type="CONTAINS",
            properties={},
            src_label="Package",
            dst_label="Module",
        )]

    # ------------------------------------------------------------------
    # DEFINED_IN — Function/Class -> Module
    # ------------------------------------------------------------------

    def _resolve_defined_in(
        self,
        module:    ModuleNode,
        classes:   list[ClassNode],
        functions: list[FunctionNode],
    ) -> list[Relationship]:
        """
        Every function and class defined in this file has a DEFINED_IN edge
        to the file's module node. This includes methods — a method is
        DEFINED_IN the module AND METHOD_OF its class.
        """
        rels = []
        for fn in functions:
            rels.append(Relationship(
                src_uuid=fn.uuid,
                dst_uuid=module.uuid,
                rel_type="DEFINED_IN",
                properties={},
                src_label="Function",
                dst_label="Module",
            ))
        for cls in classes:
            rels.append(Relationship(
                src_uuid=cls.uuid,
                dst_uuid=module.uuid,
                rel_type="DEFINED_IN",
                properties={},
                src_label="Class",
                dst_label="Module",
            ))
        return rels

    # ------------------------------------------------------------------
    # METHOD_OF — Function -> Class
    # ------------------------------------------------------------------

    def _resolve_method_of(
        self,
        classes:   list[ClassNode],
        functions: list[FunctionNode],
    ) -> list[Relationship]:
        """
        Derive the owning class from the function's qualified name.

        Qualified name structure: module.path.ClassName.method_name
        Strip the last segment to get module.path.ClassName.
        """
        cls_by_qname = {cls.qualified_name: cls for cls in classes}
        rels = []

        for fn in functions:
            if not fn.is_method:
                continue
            # "a.b.ClassName.method" -> "a.b.ClassName"
            parts = fn.qualified_name.rsplit(".", 1)
            if len(parts) < 2:
                continue
            class_qname = parts[0]
            cls = cls_by_qname.get(class_qname) or self.classes.get(class_qname)
            if cls:
                rels.append(Relationship(
                    src_uuid=fn.uuid,
                    dst_uuid=cls.uuid,
                    rel_type="METHOD_OF",
                    properties={
                        "is_property":     fn.is_property,
                        "is_classmethod":  fn.is_classmethod,
                        "is_staticmethod": fn.is_staticmethod,
                    },
                    src_label="Function",
                    dst_label="Class",
                ))
        return rels

    # ------------------------------------------------------------------
    # DEFINES_ATTR — Class -> Attribute
    # ------------------------------------------------------------------

    def _resolve_defines_attr(
        self,
        classes:    list[ClassNode],
        attributes: list[AttributeNode],
    ) -> list[Relationship]:
        """
        Each AttributeNode has a parent_uuid pointing to its owning ClassNode.
        Emit DEFINES_ATTR from Class -> Attribute.
        """
        cls_by_uuid = {cls.uuid: cls for cls in classes}
        rels = []

        for attr in attributes:
            if attr.parent_uuid in cls_by_uuid:
                rels.append(Relationship(
                    src_uuid=attr.parent_uuid,
                    dst_uuid=attr.uuid,
                    rel_type="DEFINES_ATTR",
                    properties={
                        "is_instance":  attr.is_instance,
                        "is_class_var": attr.is_class_var,
                    },
                    src_label="Class",
                    dst_label="Attribute",
                ))
        return rels

    # ------------------------------------------------------------------
    # HAS_PARAM — Function -> Parameter
    # ------------------------------------------------------------------

    def _resolve_has_param(
        self,
        functions:  list[FunctionNode],
        parameters: list[ParameterNode],
    ) -> list[Relationship]:
        """
        Each ParameterNode has a parent_uuid pointing to its FunctionNode.
        Emit HAS_PARAM from Function -> Parameter, ordered by position.
        Skip self/cls parameters — they add noise without structural value.
        """
        fn_by_uuid = {fn.uuid: fn for fn in functions}
        rels = []

        for param in parameters:
            if param.is_self:
                continue
            if param.parent_uuid in fn_by_uuid:
                rels.append(Relationship(
                    src_uuid=param.parent_uuid,
                    dst_uuid=param.uuid,
                    rel_type="HAS_PARAM",
                    properties={
                        "position":    param.position,
                        "type_hint":   param.type_hint,
                        "is_variadic": param.is_variadic,
                        "is_keyword":  param.is_keyword,
                    },
                    src_label="Function",
                    dst_label="Parameter",
                ))
        return rels

    # ------------------------------------------------------------------
    # INHERITS — Class -> Class
    # ------------------------------------------------------------------

    def _resolve_inherits(self, classes: list[ClassNode]) -> list[Relationship]:
        """
        Resolve base class name strings to ClassNode UUIDs.

        Resolution order:
          1. Same-file class (most common case — local base class)
          2. Cross-file class from the shared class registry
          3. INHERITS_UNKNOWN — external or forward reference
        """
        cls_by_name  = {cls.name:           cls for cls in classes}
        cls_by_qname = {cls.qualified_name: cls for cls in classes}

        TRIVIAL_BASES = {
            "object", "Exception", "BaseException",
            "ValueError", "TypeError", "RuntimeError",
            "KeyError", "IndexError", "AttributeError",
        }

        rels = []

        for cls in classes:
            for base_name in cls.base_classes:
                if base_name in TRIVIAL_BASES:
                    continue

                resolved = (
                    cls_by_name.get(base_name)
                    or cls_by_qname.get(base_name)
                    or self.classes.get(base_name)
                )

                if resolved and resolved.uuid != cls.uuid:
                    rels.append(Relationship(
                        src_uuid=cls.uuid,
                        dst_uuid=resolved.uuid,
                        rel_type="INHERITS",
                        properties={"is_direct": True},
                        src_label="Class",
                        dst_label="Class",
                    ))
                else:
                    rels.append(Relationship(
                        src_uuid=cls.uuid,
                        dst_uuid=None,
                        rel_type="INHERITS_UNKNOWN",
                        properties={"base_name": base_name},
                        src_label="Class",
                    ))

        return rels

    # ------------------------------------------------------------------
    # IMPORTS — Module -> Module | ExternalDependency
    # ------------------------------------------------------------------

    def _resolve_imports(self, module: ModuleNode) -> list[Relationship]:
        """
        Resolve import statements to either a known ModuleNode (IMPORTS)
        or an external library (IMPORTS_EXTERNAL).

        Resolution order:
          1. Exact module name match (e.g. "payments.utils" -> ModuleNode)
          2. Package match — `import payments` where payments is an internal
             package, not a module. Point to its __init__.py module if it
             exists, otherwise emit IMPORTS to the package's first known module
             to avoid misclassifying it as an external dependency.
          3. Fallback — genuinely external library (stripe, numpy, etc.)

        Relative imports (from . import x) are skipped.
        """
        rels = []

        for import_text in module.imported_modules:
            target_name = _parse_import_target(import_text)
            if target_name is None:
                continue

            # --- Resolution order 1: exact module match ---
            target_module = self.modules.get(target_name)

            if target_module:
                rels.append(Relationship(
                    src_uuid=module.uuid,
                    dst_uuid=target_module.uuid,
                    rel_type="IMPORTS",
                    properties={
                        "raw_import":  import_text.strip(),
                        "is_external": False,
                    },
                    src_label="Module",
                    dst_label="Module",
                ))
                continue

            # --- Resolution order 2: internal package match ---
            # `import payments` where payments is a PackageNode, not a module.
            # Look for the package's __init__.py module (name: "payments.__init__"
            # or "payments") in the module registry as the canonical target.
            # Without this, internal packages are wrongly written as ExternalDependency.
            target_package = self.packages.get(target_name)

            if target_package:
                # Prefer the __init__ module of this package as the import target.
                # It's the natural entry point for `import payments`.
                init_module = (
                    self.modules.get(f"{target_name}.__init__")
                    or self.modules.get(target_name)
                )
                if init_module:
                    rels.append(Relationship(
                        src_uuid=module.uuid,
                        dst_uuid=init_module.uuid,
                        rel_type="IMPORTS",
                        properties={
                            "raw_import":    import_text.strip(),
                            "is_external":   False,
                            "imports_package": True,
                        },
                        src_label="Module",
                        dst_label="Module",
                    ))
                # If the package has no __init__ module yet (not yet processed),
                # skip rather than emitting a wrong ExternalDependency.
                # The unresolved import is acceptable noise — it won't create bad data.
                continue

            # --- Resolution order 3: genuinely external library ---
            ext_uuid = make_uuid(module.repo_id, "external", target_name)
            rels.append(Relationship(
                src_uuid=module.uuid,
                dst_uuid=ext_uuid,
                rel_type="IMPORTS_EXTERNAL",
                properties={
                    "module_name":  target_name,
                    "raw_import":   import_text.strip(),
                    "is_external":  True,
                },
                src_label="Module",
                dst_label="ExternalDependency",
            ))

        return rels

    # ------------------------------------------------------------------
    # Pass 2 — CALLS
    # ------------------------------------------------------------------

    def resolve_calls(
        self,
        raw_call_relationships: list[Relationship],
        repo_wide_fn_lookup:    dict[str, FunctionNode],
    ) -> list[Relationship]:
        """
        Pass 2: upgrade CALLS_UNKNOWN to CALLS where possible using the
        now-complete repo-wide function lookup.

        Confidence scoring:
            1.0 — callee is a qualified name (e.g. "payments.core.process_payment")
                  and matched exactly. No ambiguity possible.
            0.7 — callee is a bare name (e.g. "process_payment") and matched.
                  Multiple functions with the same name could exist in the repo;
                  we pick the first one the lookup returns (insertion order).
        """
        finalised = []

        for rel in raw_call_relationships:
            if rel.rel_type == "CALLS" and rel.dst_uuid is not None:
                finalised.append(rel)

            elif rel.rel_type == "CALLS_UNKNOWN":
                callee   = rel.properties.get("callee", "")
                resolved = repo_wide_fn_lookup.get(callee)

                if resolved and resolved.uuid != rel.src_uuid:
                    # A qualified name contains at least one dot and matches
                    # the function's full module path. A bare name has no dots
                    # or doesn't match the qualified_name exactly.
                    is_qualified = "." in callee and callee == resolved.qualified_name
                    confidence   = 1.0 if is_qualified else 0.7

                    finalised.append(Relationship(
                        src_uuid=rel.src_uuid,
                        dst_uuid=resolved.uuid,
                        rel_type="CALLS",
                        properties={
                            "call_site_line":   rel.properties.get("call_site_line"),
                            "is_conditional":   rel.properties.get("is_conditional", False),
                            "resolved_in_pass": 2,
                            "confidence":       confidence,
                        },
                        src_label="Function",
                        dst_label="Function",
                    ))
                else:
                    finalised.append(rel)

        return finalised


# ---------------------------------------------------------------------------
# Repo-wide function lookup
# ---------------------------------------------------------------------------

def build_repo_fn_lookup(
    all_functions: list[FunctionNode],
) -> dict[str, FunctionNode]:
    """
    Build a flat lookup used in Pass 2 call resolution.
    Indexed by both bare name and qualified name.
    """
    lookup: dict[str, FunctionNode] = {}
    for fn in all_functions:
        lookup[fn.name]           = fn
        lookup[fn.qualified_name] = fn
    return lookup


# ---------------------------------------------------------------------------
# Import text parser
# ---------------------------------------------------------------------------

def _parse_import_target(import_text: str) -> Optional[str]:
    """
    Extract the target module name from a raw import statement string.
    Returns None for relative imports or unparseable text.
    """
    text = import_text.strip()

    if text.startswith("from .") or text.startswith("from .."):
        return None

    if text.startswith("from "):
        parts = text.split()
        return parts[1] if len(parts) >= 2 else None

    if text.startswith("import "):
        rest  = text[len("import "):].strip()
        first = rest.split(",")[0].strip()
        return first.split(" as ")[0].strip()

    return None


# ---------------------------------------------------------------------------
# Neo4j relationship writer
# ---------------------------------------------------------------------------

class RelationshipWriter:
    """
    Writes Relationship objects to Neo4j.
    All writes use MERGE for idempotency.
    Unresolved relationships go to :Unresolved nodes, not dropped.
    """

    def __init__(self, driver):
        self.driver = driver

    def write_relationships(self, relationships: list[Relationship]) -> None:
        resolved   = [r for r in relationships if r.dst_uuid is not None]
        unresolved = [r for r in relationships if r.dst_uuid is None]

        if resolved:
            self._write_resolved_batch(resolved)
        if unresolved:
            self._write_unresolved_batch(unresolved)

    def _write_resolved_batch(self, rels: list[Relationship]) -> None:
        """
        Group by (rel_type, src_label, dst_label) and batch-write with UNWIND.

        Grouping by rel_type is required because Cypher does not support
        parameterised relationship type names — you cannot write
        MERGE (a)-[r:$type]->(b). Each type must be a literal in the query.

        We also group by src_label and dst_label so that MATCH clauses include
        the node label (e.g. MATCH (src:Function {uuid: ...})). This lets Neo4j
        use the per-label uuid constraint indexes directly instead of doing a
        union scan across all labels, which was the dominant bottleneck.
        """
        # Key: (rel_type, src_label or "", dst_label or "")
        by_key: dict[tuple, list[Relationship]] = {}
        for rel in rels:
            key = (rel.rel_type, rel.src_label or "", rel.dst_label or "")
            by_key.setdefault(key, []).append(rel)

        with self.driver.session() as session:
            for (rel_type, src_label, dst_label), batch in by_key.items():
                # Build MATCH clauses: include label when known, omit when not.
                src_match = f"(src:{src_label} {{uuid: row.src_uuid}})" if src_label else "(src {{uuid: row.src_uuid}})"
                dst_match = f"(dst:{dst_label} {{uuid: row.dst_uuid}})" if dst_label else "(dst {{uuid: row.dst_uuid}})"
                cypher = f"""
                    UNWIND $rows AS row
                    MATCH {src_match}
                    MATCH {dst_match}
                    MERGE (src)-[r:{rel_type}]->(dst)
                    SET r += row.properties
                """
                rows = [
                    {
                        "src_uuid":   r.src_uuid,
                        "dst_uuid":   r.dst_uuid,
                        "properties": r.properties,
                    }
                    for r in batch
                ]
                session.run(cypher, rows=rows)
                logger.debug("Wrote %d %s relationships", len(batch), rel_type)

    def _write_unresolved_batch(self, rels: list[Relationship]) -> None:
        """
        Write unresolved edges as :Unresolved nodes for auditability.
        Groups by src_label so each query uses a labelled MATCH for fast index lookup.
        """
        # Group by src_label so each batch gets a labelled MATCH clause.
        # CALLS_UNKNOWN src=Function, INHERITS_UNKNOWN src=Class, etc.
        by_label: dict[str, list[dict]] = {}
        for rel in rels:
            label = rel.src_label or ""
            target = (
                rel.properties.get("callee")
                or rel.properties.get("base_name")
                or rel.properties.get("module_name")
                or rel.properties.get("parent_package")
                or "unknown"
            )
            by_label.setdefault(label, []).append({
                "src_uuid": rel.src_uuid,
                "rel_type": rel.rel_type,
                "target":   target,
                "raw":      str(rel.properties),
            })

        with self.driver.session() as session:
            for src_label, rows in by_label.items():
                src_match = f"(src:{src_label} {{uuid: row.src_uuid}})" if src_label else "(src {uuid: row.src_uuid})"
                # CREATE instead of MERGE — Unresolved nodes are audit records,
                # not shared nodes. No dedup needed.
                cypher = f"""
                    UNWIND $rows AS row
                    MATCH {src_match}
                    CREATE (u:Unresolved {{
                        src_uuid: row.src_uuid,
                        rel_type: row.rel_type,
                        target:   row.target,
                        raw:      row.raw
                    }})
                    CREATE (src)-[:HAS_UNRESOLVED]->(u)
                """
                session.run(cypher, rows=rows)

    def write_external_dependencies(self, rels: list[Relationship]) -> None:
        """
        Upsert ExternalDependency nodes for IMPORTS_EXTERNAL relationships.
        Must be called separately because it creates the destination node,
        not just the edge.
        """
        ext_rels = [r for r in rels if r.rel_type == "IMPORTS_EXTERNAL" and r.dst_uuid]
        if not ext_rels:
            return

        with self.driver.session() as session:
            cypher = """
                UNWIND $rows AS row
                MERGE (e:ExternalDependency {uuid: row.dst_uuid})
                SET e.name       = row.module_name,
                    e.raw_import = row.raw_import
                WITH e, row
                MATCH (src:Module {uuid: row.src_uuid})
                MERGE (src)-[r:IMPORTS_EXTERNAL]->(e)
                SET r.raw_import = row.raw_import
            """
            rows = [
                {
                    "src_uuid":    r.src_uuid,
                    "dst_uuid":    r.dst_uuid,
                    "module_name": r.properties.get("module_name", ""),
                    "raw_import":  r.properties.get("raw_import", ""),
                }
                for r in ext_rels
            ]
            session.run(cypher, rows=rows)
