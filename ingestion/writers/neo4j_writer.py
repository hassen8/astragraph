"""
ingestion/writers/neo4j_writer.py

Writes graph entity nodes to Neo4j.

All writes use MERGE on `uuid` so re-running the pipeline is safe —
existing nodes are updated in-place, not duplicated.

Each write method accepts a list and issues a single batched UNWIND query,
avoiding per-entity round trips to the database.

Usage:

    writer = Neo4jWriter(uri, user, password)

    # Write entities from a single file
    writer.write_module(module)
    writer.write_package(package)
    writer.write_classes(classes)
    writer.write_functions(functions)
    writer.write_attributes(attributes)
    writer.write_parameters(parameters)

    # Write the repository root node once per run
    writer.write_repository(repo)

    writer.close()

Or use as a context manager:

    with Neo4jWriter(uri, user, password) as writer:
        writer.write_repository(repo)
        ...
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from neo4j import GraphDatabase

from ..models import (
    AttributeNode,
    ClassNode,
    FunctionNode,
    ModuleNode,
    PackageNode,
    ParameterNode,
    RepositoryNode,
)

logger = logging.getLogger(__name__)


class Neo4jWriter:
    """
    Manages a Neo4j driver and exposes one write method per entity type.

    The driver is opened on construction and closed explicitly via close()
    or automatically when used as a context manager.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jWriter:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Repository  (one per ingestion run, written first)
    # ------------------------------------------------------------------

    def write_repository(self, repo: RepositoryNode) -> None:
        """
        Upsert the root Repository node.
        This is always written before any other entity so that PART_OF
        edges from top-level packages have a destination to land on.
        """
        with self._driver.session() as session:
            session.run(
                """
                MERGE (n:Repository {uuid: $uuid})
                SET   n.name        = $name,
                      n.remote_url  = $remote_url,
                      n.language    = $language,
                      n.description = $description,
                      n.repo_id     = $repo_id
                """,
                uuid=repo.uuid,
                name=repo.name,
                remote_url=repo.remote_url,
                language=repo.language,
                description=repo.description,
                repo_id=repo.repo_id,
            )
            logger.debug("Wrote Repository %s", repo.name)

    # ------------------------------------------------------------------
    # Package
    # ------------------------------------------------------------------

    def write_package(self, package: PackageNode) -> None:
        """
        Upsert a single Package node.
        Called once per file processed — the resolver deduplicates via MERGE.
        """
        with self._driver.session() as session:
            session.run(
                """
                MERGE (n:Package {uuid: $uuid})
                SET   n.name         = $name,
                      n.directory    = $directory,
                      n.is_namespace = $is_namespace,
                      n.has_init     = $has_init,
                      n.init_file    = $init_file,
                      n.repo_id      = $repo_id
                """,
                uuid=package.uuid,
                name=package.name,
                directory=package.directory,
                is_namespace=package.is_namespace,
                has_init=package.has_init,
                init_file=package.init_file,
                repo_id=package.repo_id,
            )

    # ------------------------------------------------------------------
    # Module
    # ------------------------------------------------------------------

    def write_module(self, module: ModuleNode) -> None:
        """
        Upsert a single Module node (one per source file).
        `exported_names` and `imported_modules` are stored as lists —
        Neo4j natively supports list properties.
        """
        with self._driver.session() as session:
            session.run(
                """
                MERGE (n:Module {uuid: $uuid})
                SET   n.name             = $name,
                      n.file_path        = $file_path,
                      n.language         = $language,
                      n.docstring        = $docstring,
                      n.exported_names   = $exported_names,
                      n.imported_modules = $imported_modules,
                      n.is_init          = $is_init,
                      n.repo_id          = $repo_id
                """,
                uuid=module.uuid,
                name=module.name,
                file_path=module.file_path,
                language=module.language,
                docstring=module.docstring,
                exported_names=module.exported_names,
                imported_modules=module.imported_modules,
                is_init=module.is_init,
                repo_id=module.repo_id,
            )

    # ------------------------------------------------------------------
    # Classes  (batched)
    # ------------------------------------------------------------------

    def write_classes(self, classes: list[ClassNode]) -> None:
        """
        Batch-upsert all ClassNodes from a single file.
        Uses UNWIND to issue one round trip regardless of how many classes
        are in the file.
        """
        if not classes:
            return

        # `base_classes`, `decorators`, `method_names`, `attribute_names`
        # are all Python lists — mapped directly to Neo4j list properties.
        rows = [
            {
                "uuid":            cls.uuid,
                "name":            cls.name,
                "qualified_name":  cls.qualified_name,
                "file_path":       cls.file_path,
                "line_start":      cls.line_start,
                "line_end":        cls.line_end,
                "language":        cls.language,
                "docstring":       cls.docstring,
                "base_classes":    cls.base_classes,
                "decorators":      cls.decorators,
                "is_abstract":     cls.is_abstract,
                "is_protocol":     cls.is_protocol,
                "is_dataclass":    cls.is_dataclass,
                "is_exception":    cls.is_exception,
                "method_names":    cls.method_names,
                "attribute_names": cls.attribute_names,
                "repo_id":         cls.repo_id,
            }
            for cls in classes
        ]

        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Class {uuid: row.uuid})
                SET   n.name            = row.name,
                      n.qualified_name  = row.qualified_name,
                      n.file_path       = row.file_path,
                      n.line_start      = row.line_start,
                      n.line_end        = row.line_end,
                      n.language        = row.language,
                      n.docstring       = row.docstring,
                      n.base_classes    = row.base_classes,
                      n.decorators      = row.decorators,
                      n.is_abstract     = row.is_abstract,
                      n.is_protocol     = row.is_protocol,
                      n.is_dataclass    = row.is_dataclass,
                      n.is_exception    = row.is_exception,
                      n.method_names    = row.method_names,
                      n.attribute_names = row.attribute_names,
                      n.repo_id         = row.repo_id
                """,
                rows=rows,
            )
            logger.debug("Wrote %d Class nodes", len(classes))

    # ------------------------------------------------------------------
    # Functions  (batched)
    # ------------------------------------------------------------------

    def write_functions(self, functions: list[FunctionNode]) -> None:
        """
        Batch-upsert all FunctionNodes from a single file.

        `full_body` is stored as a node property so it's available for
        embedding without a second Neo4j query. For very large repos this
        adds storage cost — acceptable trade-off for query simplicity.
        """
        if not functions:
            return

        rows = [
            {
                "uuid":            fn.uuid,
                "name":            fn.name,
                "qualified_name":  fn.qualified_name,
                "file_path":       fn.file_path,
                "line_start":      fn.line_start,
                "line_end":        fn.line_end,
                "language":        fn.language,
                "signature":       fn.signature,
                "docstring":       fn.docstring,
                "return_type":     fn.return_type,
                "is_async":        fn.is_async,
                "is_method":       fn.is_method,
                "is_property":     fn.is_property,
                "is_classmethod":  fn.is_classmethod,
                "is_staticmethod": fn.is_staticmethod,
                "is_overload":     fn.is_overload,
                "decorators":      fn.decorators,
                "body_preview":    fn.body_preview,
                "full_body":       fn.full_body,
                "complexity":      fn.complexity,
                "repo_id":         fn.repo_id,
            }
            for fn in functions
        ]

        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Function {uuid: row.uuid})
                SET   n.name            = row.name,
                      n.qualified_name  = row.qualified_name,
                      n.file_path       = row.file_path,
                      n.line_start      = row.line_start,
                      n.line_end        = row.line_end,
                      n.language        = row.language,
                      n.signature       = row.signature,
                      n.docstring       = row.docstring,
                      n.return_type     = row.return_type,
                      n.is_async        = row.is_async,
                      n.is_method       = row.is_method,
                      n.is_property     = row.is_property,
                      n.is_classmethod  = row.is_classmethod,
                      n.is_staticmethod = row.is_staticmethod,
                      n.is_overload     = row.is_overload,
                      n.decorators      = row.decorators,
                      n.body_preview    = row.body_preview,
                      n.full_body       = row.full_body,
                      n.complexity      = row.complexity,
                      n.repo_id         = row.repo_id
                """,
                rows=rows,
            )
            logger.debug("Wrote %d Function nodes", len(functions))

    # ------------------------------------------------------------------
    # Attributes  (batched)
    # ------------------------------------------------------------------

    def write_attributes(self, attributes: list[AttributeNode]) -> None:
        """
        Batch-upsert all AttributeNodes from a single file.
        Attributes without a type annotation have type_hint=None — stored
        as null in Neo4j, queryable with `WHERE n.type_hint IS NOT NULL`.
        """
        if not attributes:
            return

        rows = [
            {
                "uuid":         attr.uuid,
                "name":         attr.name,
                "full_name":    attr.full_name,
                "type_hint":    attr.type_hint,
                "default":      attr.default,
                "is_instance":  attr.is_instance,
                "is_class_var": attr.is_class_var,
                "line":         attr.line,
                "parent_uuid":  attr.parent_uuid,
                "repo_id":      attr.repo_id,
            }
            for attr in attributes
        ]

        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Attribute {uuid: row.uuid})
                SET   n.name         = row.name,
                      n.full_name    = row.full_name,
                      n.type_hint    = row.type_hint,
                      n.default      = row.default,
                      n.is_instance  = row.is_instance,
                      n.is_class_var = row.is_class_var,
                      n.line         = row.line,
                      n.parent_uuid  = row.parent_uuid,
                      n.repo_id      = row.repo_id
                """,
                rows=rows,
            )

    # ------------------------------------------------------------------
    # Parameters  (batched)
    # ------------------------------------------------------------------

    def write_parameters(self, parameters: list[ParameterNode]) -> None:
        """
        Batch-upsert all ParameterNodes from a single file.
        self/cls parameters are written here — the relationship writer
        already skips HAS_PARAM edges for them, so they exist in the graph
        but are not connected. This is intentional: they can still be found
        if you query by uuid but won't clutter traversals.
        """
        if not parameters:
            return

        rows = [
            {
                "uuid":        param.uuid,
                "name":        param.name,
                "type_hint":   param.type_hint,
                "default":     param.default,
                "position":    param.position,
                "is_self":     param.is_self,
                "is_variadic": param.is_variadic,
                "is_keyword":  param.is_keyword,
                "parent_uuid": param.parent_uuid,
                "repo_id":     param.repo_id,
            }
            for param in parameters
        ]

        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Parameter {uuid: row.uuid})
                SET   n.name        = row.name,
                      n.type_hint   = row.type_hint,
                      n.default     = row.default,
                      n.position    = row.position,
                      n.is_self     = row.is_self,
                      n.is_variadic = row.is_variadic,
                      n.is_keyword  = row.is_keyword,
                      n.parent_uuid = row.parent_uuid,
                      n.repo_id     = row.repo_id
                """,
                rows=rows,
            )
