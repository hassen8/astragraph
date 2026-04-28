"""
evals/structural_eval.py

Layer 3 evaluation — measures GraphRAG vs flat vector-only retrieval on
hand-crafted queries with verified ground truth.

Two datasets:
  structural.json  — callers/callees/methods queries. Graph should win.
  semantic.json    — natural-language semantic queries. Vector should win.

Model-agnostic: talks only to the API /retrieve endpoint. Whatever embedder the
server is configured with is what gets evaluated. Run multiple times with different
configs to compare model/embedder combinations. Every result file is stamped with
the full server config (embed model, LLM, collection, etc.) via GET /config so
results are always traceable to the exact setup that produced them.

Usage:
    uv run python evals/structural_eval.py
    uv run python evals/structural_eval.py --dataset semantic
    uv run python evals/structural_eval.py --dataset both
    uv run python evals/structural_eval.py --api-url http://localhost:8000 --top-k 5 --save
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

DATASETS = {
    "structural": Path(__file__).parent / "dataset" / "structural.json",
    "semantic":   Path(__file__).parent / "dataset" / "semantic.json",
}
RESULTS_DIR = Path(__file__).parent / "results"


def fetch_server_config(client: httpx.Client, api_url: str) -> dict:
    """Fetch server config for provenance stamping. Returns {} on failure."""
    try:
        resp = client.get(f"{api_url}/config", timeout=5.0)
        if resp.is_success:
            return resp.json()
    except Exception:
        pass
    return {}


def query_api(client: httpx.Client, api_url: str, query: str, repo_id: str, mode: str, top_k: int) -> list[str]:
    """Retrieval-only call via /retrieve (no LLM). Returns UUIDs from provenance."""
    resp = client.post(f"{api_url}/retrieve", json={
        "query": query,
        "repo_id": repo_id,
        "mode": mode,
        "top_k": top_k,
    }, timeout=30.0)
    if not resp.is_success:
        raise RuntimeError(f"API error {resp.status_code} for query={query!r} mode={mode}: {resp.text[:300]}")
    return [p["uuid"] for p in resp.json().get("provenance", []) if p.get("uuid")]


def precision_at_k(retrieved: list[str], ground_truth: set[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for u in retrieved if u in ground_truth)
    return hits / len(retrieved)


def recall_at_k(retrieved: list[str], ground_truth: set[str]) -> float:
    if not ground_truth:
        return 0.0
    hits = sum(1 for u in retrieved if u in ground_truth)
    return hits / len(ground_truth)


def f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def run_eval(api_url: str, top_k: int, save: bool, dataset: str = "structural") -> None:
    if dataset == "both":
        run_eval(api_url, top_k, save, "structural")
        run_eval(api_url, top_k, save, "semantic")
        return

    dataset_path = DATASETS[dataset]
    print(f"\n=== {dataset.upper()} EVAL ({dataset_path.name}) ===")
    cases = json.loads(dataset_path.read_text())

    graph_scores, vector_scores = [], []
    rows = []

    with httpx.Client() as client:
        # verify server is up
        try:
            client.get(f"{api_url}/health", timeout=5.0).raise_for_status()
        except Exception as e:
            print(f"ERROR: API server not reachable at {api_url} — {e}", file=sys.stderr)
            sys.exit(1)

        server_cfg = fetch_server_config(client, api_url)
        if server_cfg:
            print(f"  embed_model={server_cfg.get('embed_model')}  "
                  f"llm={server_cfg.get('llm_provider')}/{server_cfg.get('llm_model')}")

        for i, case in enumerate(cases):
            gt = set(case["ground_truth_uuids"])
            repo_id = case["repo_id"]
            query = case["query"]
            print(f"  [{i+1}/{len(cases)}] {query}", flush=True)

            graph_uuids  = query_api(client, api_url, query, repo_id, "graph",  top_k)
            vector_uuids = query_api(client, api_url, query, repo_id, "vector", top_k)

            gp = precision_at_k(graph_uuids, gt)
            gr = recall_at_k(graph_uuids, gt)
            gf = f1(gp, gr)

            vp = precision_at_k(vector_uuids, gt)
            vr = recall_at_k(vector_uuids, gt)
            vf = f1(vp, vr)

            graph_scores.append({"p": gp, "r": gr, "f1": gf})
            vector_scores.append({"p": vp, "r": vr, "f1": vf})
            rows.append((case["id"], query, gp, gr, gf, vp, vr, vf))

    # --- print table ---
    col_w = 32
    header = (f"{'Query':<{col_w}}  {'Graph P':>7} {'Graph R':>7} {'Graph F1':>8}  "
              f"{'Vec P':>7} {'Vec R':>7} {'Vec F1':>8}  {'Winner':>8}")
    print()
    print(header)
    print("-" * len(header))
    for cid, q, gp, gr, gf, vp, vr, vf in rows:
        winner = "GRAPH" if gf > vf else ("VECTOR" if vf > gf else "TIE")
        label  = q if len(q) <= col_w else q[:col_w - 1] + "…"
        print(f"{label:<{col_w}}  {gp:>7.3f} {gr:>7.3f} {gf:>8.3f}  "
              f"{vp:>7.3f} {vr:>7.3f} {vf:>8.3f}  {winner:>8}")

    # --- aggregate ---
    n   = len(rows)
    avg = lambda scores, k: sum(s[k] for s in scores) / n
    print("-" * len(header))
    print(
        f"{'AVERAGE':<{col_w}}  "
        f"{avg(graph_scores,'p'):>7.3f} {avg(graph_scores,'r'):>7.3f} {avg(graph_scores,'f1'):>8.3f}  "
        f"{avg(vector_scores,'p'):>7.3f} {avg(vector_scores,'r'):>7.3f} {avg(vector_scores,'f1'):>8.3f}"
    )
    graph_wins  = sum(1 for *_, gf, vp, vr, vf in rows if gf > vf)
    vector_wins = sum(1 for *_, gf, vp, vr, vf in rows if vf > gf)
    print(f"\nGraph wins: {graph_wins}/{n}  |  Vector wins: {vector_wins}/{n}  |  top_k={top_k}")
    if server_cfg.get("embed_model"):
        print(f"Embed model: {server_cfg['embed_model']}")
    print()

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"{dataset}_{server_cfg['embed_model']}_top_{top_k}_{ts}.json"
        payload = {
            "timestamp":  ts,
            "dataset":    dataset,
            "api_url":    api_url,
            "top_k":      top_k,
            "server_config": server_cfg,
            "summary": {
                "graph_avg_precision":  avg(graph_scores, "p"),
                "graph_avg_recall":     avg(graph_scores, "r"),
                "graph_avg_f1":         avg(graph_scores, "f1"),
                "vector_avg_precision": avg(vector_scores, "p"),
                "vector_avg_recall":    avg(vector_scores, "r"),
                "vector_avg_f1":        avg(vector_scores, "f1"),
                "graph_wins":           graph_wins,
                "vector_wins":          vector_wins,
                "n_cases":              n,
            },
            "per_case": [
                {
                    "id": cid, "query": q,
                    "graph":  {"precision": gp, "recall": gr, "f1": gf},
                    "vector": {"precision": vp, "recall": vr, "f1": vf},
                }
                for cid, q, gp, gr, gf, vp, vr, vf in rows
            ],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"Results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG vs vector eval — structural and semantic")
    parser.add_argument("--api-url",  default="http://localhost:8000")
    parser.add_argument("--top-k",    type=int, default=5)
    parser.add_argument("--save",     action="store_true", help="write JSON results to evals/results/")
    parser.add_argument("--dataset",  default="structural",
                        choices=["structural", "semantic", "both"],
                        help="which dataset to run (default: structural)")
    args = parser.parse_args()
    run_eval(args.api_url, args.top_k, args.save, args.dataset)
