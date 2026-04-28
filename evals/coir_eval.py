"""
evals/coir_eval.py

Layer 1 evaluation — benchmarks the configured embedding model against
CodeSearchNet-python using the CoIR framework.

Model-agnostic: reads the embed model name from Config (EMBED_MODEL env var).
Run with different EMBED_MODEL values to compare embedders. Results are
stamped with the model name so files are always traceable.

Talks directly to sentence-transformers — no API server, no Qdrant, no LLM.
The vector index is built in-memory by coir for each run.

Usage:
    uv run python evals/coir_eval.py
    uv run python evals/coir_eval.py --save
    uv run python evals/coir_eval.py --limit-corpus 20000  # fast dev check
    EMBED_MODEL=BAAI/bge-small-en-v1.5 uv run python evals/coir_eval.py --save
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
TASK_NAME   = "CodeSearchNet-python"


def load_task(limit_corpus: int | None):
    """Load CodeSearchNet-python corpus/queries/qrels from HuggingFace cache."""
    from coir.data_loader import load_data_from_hf
    print(f"Loading {TASK_NAME} from HuggingFace...")
    data = load_data_from_hf(TASK_NAME)
    if data is None:
        print(f"ERROR: failed to load {TASK_NAME}", file=sys.stderr)
        sys.exit(1)
    corpus, queries, qrels = data
    if limit_corpus:
        # Keep only corpus docs that appear in qrels + a random negative sample.
        relevant_ids = {doc_id for rel in qrels.values() for doc_id in rel}
        keep = list(relevant_ids)
        all_ids = list(corpus.keys())
        remaining = [i for i in all_ids if i not in relevant_ids]
        import random; random.seed(42)
        n_extra = max(0, limit_corpus - len(keep))
        keep += random.sample(remaining, min(n_extra, len(remaining)))
        corpus = {k: corpus[k] for k in keep if k in corpus}
        print(f"  corpus limited to {len(corpus)} docs ({len(relevant_ids)} relevant + negatives)")
    print(f"  corpus={len(corpus)}  queries={len(queries)}  qrels={len(qrels)}")
    return corpus, queries, qrels


class SentenceTransformerModel:
    """Wraps our sentence-transformers embedder in the coir model interface."""

    def __init__(self, model_name: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}")
        self._model      = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self.model_name  = model_name

    def encode_queries(self, queries: List[str], batch_size: int = 64,
                       max_length: int = 512, **kwargs) -> np.ndarray:
        return self._model.encode(
            queries, batch_size=batch_size or self._batch_size,
            show_progress_bar=True, convert_to_numpy=True,
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 64,
                      max_length: int = 512, **kwargs) -> np.ndarray:
        # coir corpus dicts have "text" and optionally "title"
        texts = [
            (doc.get("title") or "") + " " + (doc.get("text") or "")
            for doc in corpus
        ]
        return self._model.encode(
            texts, batch_size=batch_size or self._batch_size,
            show_progress_bar=True, convert_to_numpy=True,
        )


def run_eval(save: bool, limit_corpus: int | None) -> None:
    from config import Config
    from coir.beir.retrieval.evaluation import EvaluateRetrieval
    from coir.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

    cfg        = Config()
    model_name = cfg.embed_model
    batch_size = cfg.embed_batch_size

    print(f"\n=== CoIR EVAL — {TASK_NAME} ===")
    print(f"  embed_model={model_name}  batch_size={batch_size}")

    corpus, queries, qrels = load_task(limit_corpus)

    model     = SentenceTransformerModel(model_name, batch_size)
    dres      = DRES(model, batch_size=batch_size)
    retriever = EvaluateRetrieval(dres, score_function="cos_sim")

    print("\nRetrieving...")
    results = retriever.retrieve(corpus, queries)

    k_values = [1, 5, 10, 100]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)

    # --- print table ---
    print(f"\n{'Metric':<20}  {'@1':>8} {'@5':>8} {'@10':>8} {'@100':>8}")
    print("-" * 56)
    for label, scores in [("NDCG", ndcg), ("MAP", _map),
                           ("Recall", recall), ("Precision", precision)]:
        row = [scores.get(f"{label}@{k}", 0.0) for k in k_values]
        print(f"{label:<20}  {row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f} {row[3]:>8.4f}")

    print(f"\nNDCG@10={ndcg.get('NDCG@10', 0):.4f}  "
          f"Recall@100={recall.get('Recall@100', 0):.4f}  "
          f"Precision@5={precision.get('Precision@5', 0):.4f}")
    print(f"Model: {model_name}")

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        safe_name = model_name.replace("/", "_")
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"coir_{safe_name}_{ts}.json"
        payload = {
            "timestamp":   ts,
            "task":        TASK_NAME,
            "embed_model": model_name,
            "batch_size":  batch_size,
            "corpus_size": len(corpus),
            "n_queries":   len(queries),
            "limit_corpus": limit_corpus,
            "metrics": {
                "NDCG":      ndcg,
                "MAP":       _map,
                "Recall":    recall,
                "Precision": precision,
            },
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nResults saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoIR embedding benchmark — CodeSearchNet-python")
    parser.add_argument("--save",          action="store_true", help="write JSON results to evals/results/")
    parser.add_argument("--limit-corpus",  type=int, default=None,
                        help="cap corpus size for a quick dev check (e.g. 20000)")
    args = parser.parse_args()
    run_eval(args.save, args.limit_corpus)
