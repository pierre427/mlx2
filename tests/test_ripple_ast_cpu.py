"""Ripple on AST tree tiles: DFlash2 lattice verified on the attention tree target.

A tiny FP32 Ripple (four-token sliding window, so every tree crosses the
rotating-cache wrap; four sliding layers and one global NoPE layer; value
residual; sigmoid gate; QK norm) with a tiny DFlash2 draft that has a NoPE
draft layer.  The tree route must emit exactly the ordinary greedy tokens at
B1 and multi-lane, with and without GQA-folded attention (the affine
default), and every tile row must equal serial decoding of its root path:
sliding layers rotated at ``offset + depth``, global layers NoPE, masks the
prefix-plus-ancestor set bounded by the window.
"""

import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters import ripple
from mlx2.runtime.attention_tree_executor import AttentionTreeExecutor, tile_forward
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.models.ricursive import rotary_tables
from test_ripple_pld_cpu import greedy, tiny

TREES = [
    ([7, 3, 9, 11, 2], [-1, 0, 1, 2, 3]),                   # chain deeper than the window
    ([7, 3, 9, 11, 2, 5, 8], [-1, 0, 1, 0, 3, 4, 0]),       # three branches
    ([7, 3], [-1, 0]),
]
TAPS = (1, 3)


def _prefill(model, prompt):
    cache = model.make_cache()
    model(mx.array([prompt]), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def _serial(model, cache, tokens):
    cache = copy.deepcopy(cache)
    logits = None
    for token in tokens:
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
    return np.asarray(logits), cache


def _path(tokens, parents, row):
    path = []
    while row >= 0:
        path.append(tokens[row])
        row = parents[row]
    return list(reversed(path))


@pytest.mark.parametrize("fold", [0, 16])
@pytest.mark.parametrize("prompt_len", [2, 3, 5, 9])       # inside, at and past the window
@pytest.mark.parametrize("tokens, parents", TREES)
def test_every_tile_row_matches_serial_decoding_of_its_path(prompt_len, tokens, parents, fold):
    model = tiny()
    model.set_gqa_folded_attention(fold)
    cache = _prefill(model, [(5 * i + 1) % 60 for i in range(prompt_len)])
    reference = copy.deepcopy(cache)
    provider = AttentionTreeExecutor.create()
    logits, features, txn = provider.forward(model, tokens, parents, cache, TAPS, cached=True)
    mx.eval(logits, features)
    assert features.shape == (1, len(tokens), 2 * 32)
    for row in range(len(tokens)):
        expected, _ = _serial(model, reference, _path(tokens, parents, row))
        np.testing.assert_allclose(np.asarray(logits[0, row]), expected, atol=1e-5, rtol=1e-5)
    txn.abort()
    assert [c.offset for c in cache] == [c.offset for c in reference]
    assert model.model._tree_depths is None
    if fold:
        assert model.gqa_fold_calls() > 0


@pytest.mark.parametrize("prompt_len", [3, 9])
def test_commit_keeps_the_path_and_continues_like_serial_decoding(prompt_len):
    model = tiny()
    cache = _prefill(model, [(3 * i + 2) % 60 for i in range(prompt_len)])
    reference = copy.deepcopy(cache)
    tokens, parents = TREES[1]
    _logits, _features, txn = AttentionTreeExecutor.create().forward(
        model, tokens, parents, cache, TAPS, cached=True)
    path = [0, 3, 4, 5]
    txn.commit_paths([path])
    after = np.asarray(model(mx.array([[17]]), cache=cache)[0, -1])
    expected, serial = _serial(model, reference, [tokens[i] for i in path] + [17])
    np.testing.assert_allclose(after, expected, atol=1e-5, rtol=1e-5)
    assert [c.offset for c in cache] == [c.offset for c in serial]


def test_rope_positions_follow_depth_not_row_index():
    """Sliding layers rotate at ``offset + depth``; global layers stay NoPE."""
    model = tiny()
    args = model.args
    cache = _prefill(model, [4, 9, 2, 7, 1, 3])
    offset = cache[0].offset
    tokens, parents = TREES[1]
    depths = [0, 1, 2, 1, 2, 3, 1]
    scope = model.attention_tree_position_scope(mx.array(depths, dtype=mx.int32))
    with scope:
        shared = model.model.shared_layer_inputs(mx.zeros((1, len(tokens), args.hidden_size)), cache)
        assert shared["full_attention"][1] is None
        cos, sin = shared["sliding_attention"][1]
    assert model.model._tree_depths is None
    for row, depth in enumerate(depths):
        serial_cos, serial_sin = rotary_tables(1, offset + depth, args.head_dim, args.rope_theta, mx.float32)
        assert mx.array_equal(cos[row], serial_cos[0]) and mx.array_equal(sin[row], serial_sin[0])
    # The row-index law would differ for every row that is not its own depth.
    row_cos, _ = rotary_tables(len(tokens), offset, args.head_dim, args.rope_theta, mx.float32)
    assert not mx.array_equal(cos, row_cos)
    # Refusals: a batch under the scope, reentrancy, and non-int32 depths.
    with model.attention_tree_position_scope(mx.array(depths, dtype=mx.int32)):
        with pytest.raises(ValueError, match="one lane of one row per depth"):
            model.model.shared_layer_inputs(mx.zeros((2, len(tokens), args.hidden_size)), cache)
        with pytest.raises(RuntimeError, match="not reentrant"):
            with model.attention_tree_position_scope(mx.array([0], dtype=mx.int32)):
                pass
    with pytest.raises(ValueError, match="int32"):
        with model.attention_tree_position_scope(mx.array([0.0])):
            pass
    with pytest.raises(ValueError, match="one position per row"):
        rotary_tables(3, offset, args.head_dim, args.rope_theta, mx.float32,
                      positions=mx.array([0, 1], dtype=mx.int32))


def _tree_batch(model, draft, provider, *, lanes):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    return ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="tiny-ripple-ast", num_draft=3, prefill_step_size=4,
        completion_batch_size=lanes, dynamic_singleton_tree=True, dynamic_tree_max_width=4,
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3},
        target_tree_executor_provider=provider, cost_per_commit_handoff=False,
        memory_headroom=lambda: 1 << 40,
    )


@pytest.mark.parametrize("fold", [0, 16])
def test_dflash2_tree_route_matches_ordinary_greedy_b1_and_multilane(fold):
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from test_external_dflash2_cpu import drain
    from test_ripple_dflash2_cpu import draft_args

    prompts = [[3, 7, 11, 5, 9, 2, 8, 1, 4], [5, 1, 6], [12, 2, 7, 7, 1]]
    model = tiny()
    expected = [greedy(model, p, 20) for p in prompts]
    model.set_gqa_folded_attention(fold)
    mx.random.seed(11)
    draft = DFlash2DraftModel(draft_args()).bind(model)
    assert [layer.self_attn.use_rope for layer in draft.layers] == [True, False]
    mx.eval(draft.parameters())
    for subset in (prompts[:1], prompts):
        provider = AttentionTreeExecutor.create()
        batch = _tree_batch(model, draft, provider, lanes=4)
        ids = batch.insert(subset, max_tokens=[20] * len(subset),
                           sampling_configs=[{"sampling_temp": 0}] * len(subset))
        got, _ = drain(batch)
        assert [got[i] for i in ids] == expected[:len(subset)]
        assert batch.scheduler_stats["external_tree_rounds"] > 0
        assert provider.dispatches > 0
        if len(subset) > 1:
            assert batch.scheduler_stats["external_target_tree_target_width_3_rounds"] > 0
    if fold:
        assert model.gqa_fold_calls() > 0


def test_tree_tiles_refuse_unsupported_cache_planes_before_mutation():
    model = tiny()
    tokens, parents = TREES[0]

    class QuantizedLike(KVCache):
        bits = 8

    for bad in (QuantizedLike(), RotatingKVCache(4, keep=1)):
        cache = _prefill(model, [4, 9, 2])
        before = [(c.offset, getattr(c, "_idx", None)) for c in cache]
        cache[0] = bad
        with pytest.raises(ValueError, match="attention tree tiles"):
            tile_forward(model, tokens, parents, cache, TAPS)
        assert [(c.offset, getattr(c, "_idx", None)) for c in cache[1:]] == before[1:]
    cache = _prefill(model, [4, 9, 2])
    cache[1].start_speculation()
    with pytest.raises(ValueError, match="non-speculating"):
        tile_forward(model, tokens, parents, cache, TAPS)


def _ast_args(drafter, **overrides):
    values = dict(ast=True, ast_drafter=drafter, ast_topology=None, ast_graft_pld=False,
                  ast_tiles=None, ast_tile_rows=None, ast_tree_calibration=None,
                  ast_batch_row_sharing=False, draft_model=None, model="/nonexistent-ripple",
                  external_draft=False, prompt_lookup=False, native_mtp=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ast_invocation_tiles_only_for_an_explicit_dflash2_drafter():
    from mlx2.server import apply_ast_invocation

    resolution = SimpleNamespace(adapter_type=ripple.RippleAdapter, artifact={},
                                 default_route="ordinary")
    explicit = _ast_args("/m/Ripple-65B-Preview-DFlash2", ast_tile_rows=8)
    policy = apply_ast_invocation(explicit, None, resolution)
    assert policy == {"draft_model": "/m/Ripple-65B-Preview-DFlash2", "ast_tile_rows": 8,
                      "batch_size_route": "tree15_b1_b4_chain_b5plus_v1"}
    assert explicit.external_draft and explicit.ast_route["tree_tiles"] is True
    assert "tree_tiles_refused" not in explicit.ast_route
    # The tree policy keys reach the external parser and validate.
    _numerics, _options, external = ripple.split_policy({**policy, "activation_precision": "bf16"})
    assert set(external) == set(policy)
    holder = ripple.RippleAdapter.__new__(ripple.RippleAdapter)
    assert holder._parse_external_policy(external, family="Ripple")
    assert ripple.tree_policy(holder.external_policy) == {
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1", "ast_tile_rows": 8}
    # No drafter, or the model-free one: Ripple's ordinary prompt-lookup chain.
    for drafter in (None, "pld"):
        args = _ast_args(drafter)
        assert apply_ast_invocation(args, None, resolution) == {}
        assert args.prompt_lookup and not args.external_draft
        assert args.ast_route["tree_tiles"] is False
    with pytest.raises(ValueError, match="need a family with tree tiles|require tree tiles"):
        apply_ast_invocation(_ast_args("pld", ast_tile_rows=8), None, resolution)


def test_tree_policy_fails_closed_on_unknown_route_graft_and_bad_tiles():
    assert ripple.tree_policy({"draft_model": "/d", "num_draft": 7}) == {}
    ok = ripple.tree_policy({"draft_model": "/d", "batch_size_route": "tree15_b1_chain_b2plus_v1",
                             "ast_topology": "chain", "ast_tiles": 2})
    assert ok == {"batch_size_route": "tree15_b1_chain_b2plus_v1", "ast_topology": "chain",
                  "ast_tiles": 2}
    with pytest.raises(ValueError, match="batch_size_route"):
        ripple.tree_policy({"ast_tile_rows": 8})
    with pytest.raises(ValueError, match="batch_size_route"):
        ripple.tree_policy({"batch_size_route": "tree31_v9"})
    with pytest.raises(ValueError, match="graft"):
        ripple.tree_policy({"batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
                            "ast_graft_prompt_lookup": True})
    with pytest.raises(ValueError, match="ast_tile_rows"):
        ripple.tree_policy({"batch_size_route": "tree15_b1_b4_chain_b5plus_v1", "ast_tile_rows": 3})
    with pytest.raises(ValueError, match="ast_adaptive_rows"):
        ripple.tree_policy({"batch_size_route": "tree15_b1_b4_chain_b5plus_v1", "ast_adaptive_rows": 1})
    with pytest.raises(ValueError):
        ripple.tree_policy({"batch_size_route": "tree15_b1_b4_chain_b5plus_v1", "ast_topology": "9x9x9"})
    holder = ripple.RippleAdapter.__new__(ripple.RippleAdapter)
    with pytest.raises(ValueError, match="no qualified overrides"):
        holder._parse_external_policy({"draft_model": "/d", "tensorfold_cohort_limit": 2}, family="Ripple")


def test_tree_receipt_reports_selection_and_refusals():
    plain = ripple.RippleAdapter._tree_receipt(SimpleNamespace(tree={}))
    assert plain["implemented"] and not plain["qualified"] and not plain["selected"]
    assert plain["tree_target"] == "attention_tree_v1"
    assert "quantized_kv" in plain["refusals"]
    chosen = ripple.RippleAdapter._tree_receipt(
        SimpleNamespace(tree={"batch_size_route": "tree15_b1_b4_chain_b5plus_v1"}))
    assert chosen["selected"] and chosen["observation_authority"] == "speculative_route_receipt"
    assert ripple.RippleAdapter.ast_tree_tiles and ripple.RippleAdapter.attention_tree_tiles
    assert not ripple.RippleAdapter.ast_prompt_lookup_tiles
    assert not ripple.RippleAdapter.ast_tree_tile_dtype_gate
    # The tree route binds under its own tag (identity digest, APCv2 layout
    # suffix, route receipt); the chain keeps the class tag.
    assert ripple.TREE_ROUTE_TAG != ripple.RippleAdapter.EXTERNAL_ROUTE_TAG
    assert ripple.TREE_ROUTE_TAG.startswith("external-ripple-dflash2-")
