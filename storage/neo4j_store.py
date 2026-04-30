"""
storage/neo4j_store.py

Neo4j implementation of the GraphStore protocol.

This file is a thin facade that combines the existing low-level pieces into a
single object that satisfies storage.protocols.GraphStore:

  - graph.schema.init_schema      → init_schema()
  - ingestion.writers.Neo4jWriter → write_repository, write_package, ...
  - ingestion.relationships.RelationshipWriter → write_relationships, ...
  - inline Cypher queries         → fulltext_search, callers, callees, ...

The facade pattern means we don't have to rewrite the existing writers; they
remain the single source of truth for write Cypher. The store owns the Neo4j
Driver and shares it with both writer instances so we don't open three
connections to the same database.

Read methods (the second half of this file) are the only "new" Cypher in the
codebase — they're agent-side queries that didn't exist before. All return the
standard result shape:

  uuid, name, qualified_name, file_path, line_start, line_end,
  signature, docstring, full_body, repo_id, score, source="graph"

Adding a new graph database backend (e.g. Memgraph, NebulaGraph) means writing
a new file like this one — no changes to the pipeline or agent.
"""

from __future__ import annotations

import logging

from neo4j import Driver, GraphDatabase

from config import Config
from graph.schema import init_schema as _init_schema_fn
from ingestion.models import (
    AttributeNode,
    ClassNode,
    FunctionNode,
    ModuleNode,
    PackageNode,
    ParameterNode,
    Relationship,
    RepositoryNode,
)
from ingestion.relationships import RelationshipWriter
from ingestion.writers.neo4j_writer import Neo4jWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result-shaping helpers
# ---------------------------------------------------------------------------

# Stock projection used by every read query so all results share one schema.
# Aliased as the same field names regardless of which Cypher MATCH variable
# the node is bound to (caller, callee, target, fn, etc.) — caller picks the
# variable, this helper formats the RETURN clause.
def _projection(var: str, score_expr: str = "1.0") -> str:
    return (
        f"{var}.uuid           AS uuid, "
        f"{var}.name           AS name, "
        f"{var}.qualified_name AS qualified_name, "
        f"{var}.file_path      AS file_path, "
        f"{var}.line_start     AS line_start, "
        f"{var}.line_end       AS line_end, "
        f"{var}.signature      AS signature, "
        f"{var}.docstring      AS docstring, "
        f"{var}.full_body      AS full_body, "
        f"{var}.repo_id        AS repo_id, "
        f"{score_expr}         AS score"
    )


def _row_to_dict(record) -> dict:
    return {
        "uuid":           record["uuid"],
        "name":           record["name"],
        "qualified_name": record["qualified_name"],
        "file_path":      record["file_path"],
        "line_start":     record["line_start"],
        "line_end":       record["line_end"],
        "signature":      record["signature"],
        "docstring":      record["docstring"],
        "full_body":      record["full_body"],
        "repo_id":        record["repo_id"],
        "score":          record["score"],
        "source":         "graph",
    }


# Neo4j label whitelist — used to translate `kind` strings safely into Cypher
# without string interpolation of user input.
_KIND_TO_LABEL = {
    "function": "Function",
    "class":    "Class",
    "module":   "Module",
    "package":  "Package",
}


# ---------------------------------------------------------------------------
# Neo4jStore
# ---------------------------------------------------------------------------

class Neo4jStore:
    """
    Implements storage.protocols.GraphStore over a Neo4j database.

    Owns the neo4j.Driver. The underlying Neo4jWriter and RelationshipWriter
    share that driver — no extra connections.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg     = cfg
        self._driver: Driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
        )
        # Share the driver with the existing writers so all writes go through
        # one connection pool.
        self._entity_writer = Neo4jWriter.__new__(Neo4jWriter)
        self._entity_writer._driver = self._driver
        self._rel_writer    = RelationshipWriter(self._driver)

    # ----- lifecycle -------------------------------------------------------- #

    def __enter__(self) -> "Neo4jStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    # ----- schema ----------------------------------------------------------- #

    def init_schema(self) -> None:
        _init_schema_fn(self._driver)

    # ----- node writes ------------------------------------------------------ #

    def write_repository(self, repo: RepositoryNode) -> None:
        self._entity_writer.write_repository(repo)

    def write_package(self, pkg: PackageNode) -> None:
        self._entity_writer.write_package(pkg)

    def write_module(self, module: ModuleNode) -> None:
        self._entity_writer.write_module(module)

    def write_classes(self, classes: list[ClassNode]) -> None:
        self._entity_writer.write_classes(classes)

    def write_functions(self, functions: list[FunctionNode]) -> None:
        self._entity_writer.write_functions(functions)

    def write_attributes(self, attributes: list[AttributeNode]) -> None:
        self._entity_writer.write_attributes(attributes)

    def write_parameters(self, parameters: list[ParameterNode]) -> None:
        self._entity_writer.write_parameters(parameters)

    # ----- relationship writes --------------------------------------------- #

    def write_relationships(self, rels: list[Relationship]) -> None:
        self._rel_writer.write_relationships(rels)

    def write_external_dependencies(self, rels: list[Relationship]) -> None:
        self._rel_writer.write_external_dependencies(rels)

    # ----- reads: keyword search ------------------------------------------- #

    def fulltext_search(self, query: str, repo_id: str | None, top_k: int) -> list[dict]:
        cypher = f"""
            CALL db.index.fulltext.queryNodes($index, $q)
            YIELD node, score
            {"WHERE node.repo_id = $repo_id" if repo_id else ""}
            RETURN {_projection("node", "score")}
            LIMIT $k
        """
        params: dict = {"index": self._cfg.fulltext_index, "q": query, "k": top_k}
        if repo_id:
            params["repo_id"] = repo_id
        return self._run(cypher, params)

    # ----- reads: discovery ------------------------------------------------- #

    def find_by_name(self, name: str, kind: str, repo_id: str | None) -> list[dict]:
        label = _KIND_TO_LABEL.get(kind)
        if label is None:
            raise ValueError(f"Unknown kind: {kind!r}. Use one of {list(_KIND_TO_LABEL)}.")
        cypher = f"""
            MATCH (n:{label} {{name: $name}})
            {"WHERE n.repo_id = $repo_id" if repo_id else ""}
            RETURN {_projection("n")}
        """
        params: dict = {"name": name}
        if repo_id:
            params["repo_id"] = repo_id
        return self._run(cypher, params)

    def find_qualified(self, qualified_name: str, repo_id: str | None) -> list[dict]:
        cypher = f"""
            MATCH (n {{qualified_name: $qn}})
            {"WHERE n.repo_id = $repo_id" if repo_id else ""}
            RETURN {_projection("n")}
        """
        params: dict = {"qn": qualified_name}
        if repo_id:
            params["repo_id"] = repo_id
        return self._run(cypher, params)

    # ----- reads: class structure ------------------------------------------ #

    def subclasses(self, class_name: str, repo_id: str, top_k: int) -> list[dict]:
        cypher = f"""
            MATCH (sub:Class)-[:INHERITS*1..]->(base:Class)
            WHERE (base.name = $name OR base.qualified_name = $name)
              AND sub.repo_id = $repo_id
            RETURN DISTINCT {_projection("sub")}
            LIMIT $k
        """
        return self._run(cypher, {"name": class_name, "repo_id": repo_id, "k": top_k})

    def methods(self, class_name: str, repo_id: str) -> list[dict]:
        cypher = f"""
            MATCH (m:Function)-[:METHOD_OF]->(c:Class)
            WHERE (c.name = $name OR c.qualified_name = $name)
              AND c.repo_id = $repo_id
            RETURN {_projection("m")}
        """
        return self._run(cypher, {"name": class_name, "repo_id": repo_id})

    def attributes(self, class_name: str, repo_id: str) -> list[dict]:
        # Attributes don't have signatures or full_body — return nulls in those
        # fields to keep the result shape uniform.
        cypher = """
            MATCH (a:Attribute)<-[:DEFINES_ATTR]-(c:Class)
            WHERE (c.name = $name OR c.qualified_name = $name)
              AND c.repo_id = $repo_id
            RETURN a.uuid           AS uuid,
                   a.name           AS name,
                   a.full_name      AS qualified_name,
                   c.file_path      AS file_path,
                   c.line_start     AS line_start,
                   c.line_end       AS line_end,
                   a.type_hint      AS signature,
                   NULL             AS docstring,
                   NULL             AS full_body,
                   c.repo_id        AS repo_id,
                   1.0              AS score
        """
        return self._run(cypher, {"name": class_name, "repo_id": repo_id})

    # ----- reads: module structure ----------------------------------------- #

    def module_contents(self, module_path: str, repo_id: str) -> list[dict]:
        cypher = f"""
            MATCH (n)-[:DEFINED_IN]->(m:Module)
            WHERE m.file_path = $path AND m.repo_id = $repo_id
              AND (n:Function OR n:Class)
            RETURN {_projection("n")}
        """
        return self._run(cypher, {"path": module_path, "repo_id": repo_id})

    def module_imports(self, module_path: str, repo_id: str) -> list[dict]:
        # IMPORTS edges can target Module or ExternalDependency nodes — both are
        # interesting. Project the same fields, treating ExternalDependency name
        # as the qualified_name and leaving code-specific fields null.
        cypher = """
            MATCH (m:Module {file_path: $path, repo_id: $repo_id})-[r:IMPORTS|IMPORTS_EXTERNAL]->(target)
            RETURN target.uuid                                       AS uuid,
                   coalesce(target.name, target.package_name)        AS name,
                   coalesce(target.qualified_name, target.package_name) AS qualified_name,
                   target.file_path                                  AS file_path,
                   target.line_start                                 AS line_start,
                   target.line_end                                   AS line_end,
                   NULL                                              AS signature,
                   NULL                                              AS docstring,
                   NULL                                              AS full_body,
                   coalesce(target.repo_id, '')                      AS repo_id,
                   1.0                                               AS score
        """
        return self._run(cypher, {"path": module_path, "repo_id": repo_id})

    # ----- reads: call graph ----------------------------------------------- #

    def callers(self, name: str, repo_id: str, top_k: int) -> list[dict]:
        cypher = f"""
            MATCH (caller:Function)-[:CALLS]->(target:Function)
            WHERE (target.name = $name OR target.qualified_name = $name)
              AND target.repo_id = $repo_id
            RETURN {_projection("caller")}
            LIMIT $k
        """
        return self._run(cypher, {"name": name, "repo_id": repo_id, "k": top_k})

    def callees(self, name: str, repo_id: str, top_k: int) -> list[dict]:
        cypher = f"""
            MATCH (source:Function)-[:CALLS]->(callee:Function)
            WHERE (source.name = $name OR source.qualified_name = $name)
              AND source.repo_id = $repo_id
            RETURN {_projection("callee")}
            LIMIT $k
        """
        return self._run(cypher, {"name": name, "repo_id": repo_id, "k": top_k})

    def call_path(self, from_name: str, to_name: str, repo_id: str, max_hops: int) -> list[dict]:
        # Variable-length CALLS path. We can't parameterise the upper bound,
        # so it's interpolated after a sanity-check.
        if not isinstance(max_hops, int) or max_hops < 1 or max_hops > 10:
            raise ValueError("max_hops must be an int in [1, 10]")
        cypher = f"""
            MATCH p = shortestPath((a:Function)-[:CALLS*1..{max_hops}]->(b:Function))
            WHERE (a.name = $from_name OR a.qualified_name = $from_name)
              AND (b.name = $to_name   OR b.qualified_name = $to_name)
              AND a.repo_id = $repo_id AND b.repo_id = $repo_id
            UNWIND nodes(p) AS n
            RETURN {_projection("n")}
        """
        return self._run(cypher, {
            "from_name": from_name,
            "to_name":   to_name,
            "repo_id":   repo_id,
        })

    # ----- reads: full graph for visualisation -------------------------------- #

    def get_full_graph(self, repo_id: str, limit: int = 500) -> dict:
        """
        Return all Function nodes + CALLS edges for a repo, capped at `limit`
        nodes. Nodes are ranked by in-degree (most-called functions first) so
        the most structurally important ones survive the cap.

        Returns {"nodes": [...], "edges": [...]} ready for Cytoscape.js.

        Each node: { id, label, file_path, line_start, line_end, full_body }
        Each edge: { id, source, target }
        """
        # Fetch the top-N most-called functions first. Functions with no
        # incoming CALLS edges still appear if they have outgoing ones.
        node_cypher = """
            MATCH (f:Function {repo_id: $repo_id})
            OPTIONAL MATCH (f)<-[:CALLS]-(caller:Function {repo_id: $repo_id})
            WITH f, count(caller) AS in_degree
            ORDER BY in_degree DESC
            LIMIT $limit
            RETURN f.uuid           AS id,
                   f.name           AS label,
                   f.file_path      AS file_path,
                   f.line_start     AS line_start,
                   f.line_end       AS line_end,
                   f.full_body      AS full_body
        """
        with self._driver.session() as session:
            node_rows = list(session.run(node_cypher, {"repo_id": repo_id, "limit": limit}))

        node_ids = {r["id"] for r in node_rows}
        nodes = [dict(r) for r in node_rows]

        # Only return edges where both endpoints are in the visible node set.
        # Avoids dangling edges that Cytoscape would complain about.
        edge_cypher = """
            MATCH (a:Function {repo_id: $repo_id})-[:CALLS]->(b:Function {repo_id: $repo_id})
            WHERE a.uuid IN $ids AND b.uuid IN $ids
            RETURN a.uuid + '->' + b.uuid AS id,
                   a.uuid AS source,
                   b.uuid AS target
        """
        with self._driver.session() as session:
            edge_rows = list(session.run(edge_cypher, {"repo_id": repo_id, "ids": list(node_ids)}))

        edges = [dict(r) for r in edge_rows]
        return {"nodes": nodes, "edges": edges}

    # ----- internal --------------------------------------------------------- #

    def _run(self, cypher: str, params: dict) -> list[dict]:
        with self._driver.session() as session:
            return [_row_to_dict(r) for r in session.run(cypher, params)]
