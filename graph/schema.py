"""
graph/schema.py

Neo4j schema initialisation — run once at pipeline start before any writes.

Two things are set up here:

1. UNIQUENESS CONSTRAINTS on `uuid`
   Every node label gets a uniqueness constraint on its `uuid` property.
   This does two jobs at once:
     - Prevents duplicate nodes if the pipeline is interrupted and re-run.
     - Implicitly creates a B-tree index on `uuid`, making MERGE lookups O(log n)
       instead of a full label scan.

2. RANGE INDEXES on commonly queried properties
   These speed up MATCH and WHERE clauses that filter by name, file path, or
   repo. Without indexes, Neo4j performs a full label scan on every query.

   Indexed properties and the queries they serve:
     name           — "find all functions named process_payment"
     qualified_name — "find the exact node for payments.core.process_payment"
     file_path      — "find all nodes defined in payments/core.py"
     repo_id        — "scope any query to a specific repository"

All statements use `IF NOT EXISTS` so the function is fully idempotent —
calling it on an already-initialised database is safe and has no side effects.
"""

from neo4j import Driver


# Node labels that exist in the graph. Each gets a uuid uniqueness constraint.
# ExternalDependency is intentionally included — external packages are real nodes
# with their own UUIDs, not just string properties.
_NODE_LABELS = [
    "Repository",
    "Package",
    "Module",
    "Class",
    "Function",
    "Attribute",
    "Parameter",
    "ExternalDependency",
]

# Properties worth indexing for fast MATCH/WHERE filtering.
# Not all properties are indexed — only those that appear in WHERE clauses
# in typical graph queries. Over-indexing wastes write overhead and memory.
_INDEXED_PROPERTIES = [
    "name",           # human-readable label used in most user-facing queries
    "qualified_name", # dot-path lookup, e.g. "transformers.modeling_utils.PreTrainedModel"
    "file_path",      # file-scoped queries: "show me everything in activations.py"
    "repo_id",        # every query should be repo-scoped to avoid cross-repo collisions
]


def init_schema(driver: Driver) -> None:
    """
    Create all uniqueness constraints and range indexes in a single transaction.

    This is idempotent: safe to call at the start of every pipeline run.
    Neo4j will skip creation silently if the constraint or index already exists.

    Args:
        driver: An open neo4j.Driver instance. The caller is responsible for
                closing the driver — this function does not close it.
    """
    with driver.session() as session:
        # ------------------------------------------------------------------ #
        # Uniqueness constraints
        # Each constraint also implicitly creates a B-tree index on uuid,
        # so MERGE (n {uuid: $uuid}) is always an indexed lookup, never a scan.
        # ------------------------------------------------------------------ #
        for label in _NODE_LABELS:
            # Constraint names must be unique in Neo4j. We use a consistent
            # naming convention: <lowercase_label>_uuid_unique
            constraint_name = f"{label.lower()}_uuid_unique"
            session.run(
                f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.uuid IS UNIQUE"
            )

        # ------------------------------------------------------------------ #
        # Range indexes on shared filter properties
        # We create one index per (label, property) pair rather than composite
        # indexes because queries rarely filter on two non-uuid properties at once.
        # ------------------------------------------------------------------ #
        for label in _NODE_LABELS:
            for prop in _INDEXED_PROPERTIES:
                # Not all node types have all properties (e.g. Parameter has no
                # qualified_name). Neo4j range indexes on missing properties are
                # harmless — the entry simply won't exist in the index.
                index_name = f"{label.lower()}_{prop}_idx"
                session.run(
                    f"CREATE INDEX {index_name} IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.{prop})"
                )
