"""
cli.py

Entry point for the AstraGraph ingestion pipeline.

Usage:
    python cli.py --repo ./path/to/repo
    python cli.py --repo ./path/to/repo --repo-id my-repo --languages python
    python cli.py --repo ./path/to/repo --dry-run

Environment variables (all optional, have defaults):
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    See config.py for the full list.
"""

import logging
import sys

import click

from config import Config
from ingestion.pipeline import IngestionPipeline, make_repo_node


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("neo4j").setLevel(logging.WARNING)  # suppress schema IF NOT EXISTS notices
logger = logging.getLogger("astragraph")


@click.command()
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the repository to ingest.",
)
@click.option(
    "--repo-id",
    default=None,
    help="Stable identifier for this repo (defaults to directory name).",
)
@click.option(
    "--languages",
    default="python",
    show_default=True,
    help="Comma-separated list of languages to ingest, e.g. 'python,typescript'.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Parse and extract without writing to Neo4j. Useful for smoke-testing.",
)
def main(repo: str, repo_id: str | None, languages: str, dry_run: bool) -> None:
    """Ingest a source code repository into the AstraGraph knowledge graph."""

    lang_list = [l.strip() for l in languages.split(",") if l.strip()]
    cfg       = Config()

    repo_node = make_repo_node(repo_path=repo, repo_id=repo_id)

    logger.info(
        "Starting ingestion — repo=%s  id=%s  languages=%s",
        repo, repo_node.repo_id, lang_list,
    )

    if dry_run:
        # Dry-run: run extraction and relationship resolution but skip all
        # Neo4j writes. Useful for verifying extraction correctness before
        # connecting to a database.
        logger.info("Dry-run mode — no writes to Neo4j")
        from ingestion.pipeline import IngestionPipeline as _P

        # Temporarily patch out the Neo4j calls by using the old dry-run path.
        # For now, just instantiate and warn — full dry-run needs a flag on run().
        logger.warning("Dry-run not yet fully implemented — use without --dry-run to write to Neo4j")
        sys.exit(0)

    pipeline = IngestionPipeline(repo=repo_node, cfg=cfg)

    try:
        pipeline.run(repo_path=repo, languages=lang_list)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
