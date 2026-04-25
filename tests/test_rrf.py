from agent.rrf import reciprocal_rank_fusion


def hit(uuid: str, source: str = "graph", **extra) -> dict:
    return {"uuid": uuid, "name": uuid, "score": 0.0, "source": source, **extra}


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_passthrough_order_preserved():
    results = [hit("a"), hit("b"), hit("c")]
    out = reciprocal_rank_fusion([results])
    assert [h["uuid"] for h in out] == ["a", "b", "c"]
    assert out[0]["score"] == 1.0 / (60 + 1)
    assert out[1]["score"] == 1.0 / (60 + 2)
    assert out[2]["score"] == 1.0 / (60 + 3)


def test_identical_lists_double_score():
    results = [hit("a"), hit("b")]
    out = reciprocal_rank_fusion([results, results])
    by_uuid = {h["uuid"]: h for h in out}
    assert by_uuid["a"]["score"] == 2.0 / (60 + 1)
    assert by_uuid["b"]["score"] == 2.0 / (60 + 2)


def test_overlap_outranks_singletons():
    list_a = [hit("shared", source="graph"), hit("only_a", source="graph")]
    list_b = [hit("shared", source="vector"), hit("only_b", source="vector")]
    out = reciprocal_rank_fusion([list_a, list_b])
    assert out[0]["uuid"] == "shared"
    assert sorted(out[0]["sources"]) == ["graph", "vector"]
    assert out[0]["score"] == 2.0 / (60 + 1)


def test_sources_aggregated_for_shared_hit():
    list_a = [hit("x", source="graph")]
    list_b = [hit("x", source="vector")]
    out = reciprocal_rank_fusion([list_a, list_b])
    assert sorted(out[0]["sources"]) == ["graph", "vector"]


def test_sources_single_for_disjoint_hit():
    list_a = [hit("x", source="graph")]
    list_b = [hit("y", source="vector")]
    out = reciprocal_rank_fusion([list_a, list_b])
    by_uuid = {h["uuid"]: h for h in out}
    assert by_uuid["x"]["sources"] == ["graph"]
    assert by_uuid["y"]["sources"] == ["vector"]


def test_top_k_slices_output():
    results = [hit(c) for c in "abcde"]
    out = reciprocal_rank_fusion([results], top_k=2)
    assert len(out) == 2
    assert [h["uuid"] for h in out] == ["a", "b"]


def test_one_empty_list_acts_like_single_list():
    results = [hit("a"), hit("b")]
    out = reciprocal_rank_fusion([results, []])
    assert [h["uuid"] for h in out] == ["a", "b"]
    assert out[0]["score"] == 1.0 / 61


def test_payload_fields_preserved():
    h = hit("a", file_path="foo.py", line_start=10, qualified_name="mod.a")
    out = reciprocal_rank_fusion([[h]])
    assert out[0]["file_path"] == "foo.py"
    assert out[0]["line_start"] == 10
    assert out[0]["qualified_name"] == "mod.a"


def test_score_field_replaced_with_fused_score():
    h = hit("a")
    h["score"] = 999.0
    out = reciprocal_rank_fusion([[h]])
    assert out[0]["score"] == 1.0 / 61


def test_duplicate_uuid_in_same_list_uses_best_rank_only():
    results = [hit("a"), hit("b"), hit("a")]
    out = reciprocal_rank_fusion([results])
    by_uuid = {h["uuid"]: h for h in out}
    assert by_uuid["a"]["score"] == 1.0 / 61


def test_custom_k_changes_scores():
    results = [hit("a")]
    out = reciprocal_rank_fusion([results], k=10)
    assert out[0]["score"] == 1.0 / 11
