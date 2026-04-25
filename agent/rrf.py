"""Reciprocal Rank Fusion for merging hybrid retrieval results."""

from __future__ import annotations


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict]:
    """Merge ranked result lists by RRF: score(d) = Σ 1 / (k + rank_i(d)).

    Each hit dict must contain a ``uuid`` key. Other fields are preserved
    from the first occurrence of that uuid across the input lists.

    Returns a list sorted by fused score descending. Each output dict has:
      - all original payload fields (uuid, name, qualified_name, ...)
      - ``score``: the fused RRF score (replaces per-retriever score)
      - ``sources``: list of distinct ``source`` values that contributed
    """
    fused: dict[str, dict] = {}
    seen_rank: dict[tuple[str, int], int] = {}

    for list_idx, results in enumerate(result_lists):
        for rank, hit in enumerate(results):
            uuid = hit["uuid"]
            key = (uuid, list_idx)
            if key in seen_rank and seen_rank[key] <= rank:
                continue
            seen_rank[key] = rank

            contribution = 1.0 / (k + rank + 1)

            if uuid not in fused:
                payload = {f: v for f, v in hit.items() if f not in ("score", "source")}
                payload["score"] = 0.0
                payload["sources"] = []
                fused[uuid] = payload

            entry = fused[uuid]
            entry["score"] += contribution
            src = hit.get("source")
            if src and src not in entry["sources"]:
                entry["sources"].append(src)

    merged = sorted(fused.values(), key=lambda h: h["score"], reverse=True)
    if top_k is not None:
        merged = merged[:top_k]
    return merged
