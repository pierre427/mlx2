"""Flint on AST tree tiles: generic attention tree target, PLD tree drafter.

A tiny Flint keeps the full numerical law that matters to tree tiles: learned
sinks, a sliding window shorter than the prompt, RoPE only on sliding layers,
a RoPE-free full-attention layer, a dense prefix layer and sigmoid-routed
MoE layers with a shared expert.
"""

import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import flint
from mlx2.runtime.attention_tree_executor import AttentionTreeExecutor
from mlx2.runtime.models.flint import Model, ModelArgs


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny(window=5):
    args = ModelArgs.from_dict(
        {
            "model_type": "flint", "vocab_size": 64, "hidden_size": 32,
            "intermediate_size": 48, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 256, "rope_theta": 10000.0,
            "layer_types": ["full_attention", "sliding_attention", "sliding_attention",
                            "full_attention"],
            "no_rope_layer_indices": [0, 3], "mlp_only_layers": [0],
            "moe_intermediate_size": 8, "shared_expert_intermediate_size": 12,
            "num_experts": 4, "num_experts_per_tok": 2, "sliding_window": window,
            "tie_word_embeddings": False,
        }
    )
    mx.random.seed(11)
    model = Model(args)
    model.eval()
    # Non-zero sinks exercise the learned-sink path through the tree mask.
    for layer in model.model.layers:
        layer.self_attn.softmax_offset = mx.random.normal((args.num_attention_heads,))
    mx.eval(model.parameters())
    return model


def _prefill(model, prompt):
    cache = model.make_cache()
    model(mx.array([prompt]), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def _serial_last_logits(model, cache, tokens):
    cache = copy.deepcopy(cache)
    logits = None
    for token in tokens:
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
    return np.asarray(logits.astype(mx.float32)), cache


TREES = [
    ([7, 3, 9, 11, 2], [-1, 0, 1, 2, 3]),
    ([7, 3, 9, 11, 2, 5, 8], [-1, 0, 1, 0, 3, 4, 0]),
]


@pytest.mark.parametrize("prompt_len", [3, 12])          # inside and past the window
@pytest.mark.parametrize("tokens, parents", TREES)
def test_every_flint_tile_row_matches_serial_decoding_of_its_path(prompt_len, tokens, parents):
    model = tiny()
    prompt = [(5 * i + 1) % 60 for i in range(prompt_len)]
    cache = _prefill(model, prompt)
    reference = copy.deepcopy(cache)
    logits, features, txn = AttentionTreeExecutor.create().forward(
        model, tokens, parents, cache, (1, 3), cached=True)
    mx.eval(logits, features)
    assert features.shape == (1, len(tokens), 64)
    for row in range(len(tokens)):
        path, node = [], row
        while node >= 0:
            path.append(tokens[node])
            node = parents[node]
        expected, _ = _serial_last_logits(model, reference, list(reversed(path)))
        np.testing.assert_allclose(np.asarray(logits[0, row].astype(mx.float32)), expected,
                                   atol=2e-4, rtol=2e-4)
    txn.abort()
    assert [c.offset for c in cache] == [c.offset for c in reference]


@pytest.mark.parametrize("prompt_len", [3, 12])
def test_flint_commit_keeps_the_path_and_continues_like_serial_decoding(prompt_len):
    model = tiny()
    cache = _prefill(model, [(3 * i + 2) % 60 for i in range(prompt_len)])
    reference = copy.deepcopy(cache)
    tokens, parents = TREES[1]
    logits, _features, txn = AttentionTreeExecutor.create().forward(
        model, tokens, parents, cache, (3,), cached=True)
    mx.eval(logits)
    path = [0, 3, 4, 5]
    txn.commit_paths([path])
    after = model(mx.array([[17]]), cache=cache)[0, -1]
    expected, serial = _serial_last_logits(model, reference, [tokens[i] for i in path] + [17])
    np.testing.assert_allclose(np.asarray(after.astype(mx.float32)), expected, atol=2e-4, rtol=2e-4)
    assert [c.offset for c in cache] == [c.offset for c in serial]


def test_forward_with_taps_matches_plain_forward():
    model = tiny()
    tokens = mx.array([[4, 9, 2, 7, 1, 3, 8]])
    plain = model(tokens)
    logits, features = model.forward_with_taps(tokens, None, (0, 2))
    assert features.shape == (1, 7, 64)
    np.testing.assert_allclose(np.asarray(logits), np.asarray(plain), atol=1e-6)
    with pytest.raises(ValueError, match="capture layers"):
        model.forward_with_taps(tokens, None, (2, 0))


def _greedy(model, prompt, count):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)[0, -1]
    out = []
    for _ in range(count):
        token = int(mx.argmax(logits).item())
        out.append(token)
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
    return out


def test_external_route_with_prompt_lookup_tiles_matches_flint_greedy():
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    model = tiny(window=5)
    prompt = [3, 9, 4, 3, 9, 4, 3, 9, 4, 11, 3, 9]
    reference = _greedy(model, prompt, 24)
    batch = ExternalDraftBatchGenerator(
        model, draft_model=PromptLookupDraftModel(model, num_draft=15), binding="tiny-flint",
        num_draft=15, completion_batch_size=4, dynamic_singleton_tree=True,
        dynamic_tree_max_width=4, tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3},
        target_tree_executor_provider=AttentionTreeExecutor.create(),
        memory_headroom=lambda: 1 << 40,
    )
    uid = batch.insert([prompt], max_tokens=[24], sampling_configs=[{"sampling_temp": 0.0}])[0]
    tokens = []
    for _ in range(200):
        _, responses = batch.next()
        tokens.extend(r.token for r in responses if r.uid == uid)
        if uid not in batch.lanes:
            break
    assert tokens == reference
    assert batch.scheduler_stats["external_tree_rounds"] > 0


def test_execution_policy_admits_only_prompt_lookup_tiles_before_artifact_load():
    assert flint._validate_execution_policy(None) == {}
    assert flint._validate_execution_policy({}) == {}
    policy = flint._validate_execution_policy({"draft_model": "prompt_lookup"})
    assert policy["draft_model"] == "prompt_lookup"
    with pytest.raises(ValueError, match="prompt-lookup tree-tile drafter only"):
        flint.FlintAdapter("/missing", execution_policy={"draft_model": "/draft"})
    with pytest.raises(ValueError, match="no base drafter"):
        flint._validate_execution_policy(
            {"draft_model": "prompt_lookup", "ast_graft_prompt_lookup": True})
    with pytest.raises(ValueError, match="no other execution-policy overrides"):
        flint._validate_execution_policy({"num_draft": 3})


def test_ast_invocation_routes_flint_to_tiles_only_for_explicit_pld():
    from mlx2.server import apply_ast_invocation

    resolution = SimpleNamespace(adapter_type=flint.FlintAdapter, artifact={},
                                 default_route="ordinary")

    def args(drafter):
        return SimpleNamespace(ast=True, ast_drafter=drafter, ast_topology=None, ast_graft_pld=False,
                               ast_tiles=None, ast_tile_rows=None, draft_model=None,
                               model="/nonexistent-flint", external_draft=False,
                               prompt_lookup=False, native_mtp=False)

    auto = args(None)
    assert apply_ast_invocation(auto, None, resolution) == {}
    assert auto.prompt_lookup and not auto.external_draft
    assert auto.ast_route["tree_tiles"] is False
    explicit = args("pld")
    policy = apply_ast_invocation(explicit, None, resolution)
    assert policy["draft_model"] == "prompt_lookup" and explicit.external_draft
    assert explicit.ast_route["tree_tiles"] is True


def test_kernel_defaults_follow_operator_env_and_refuse_unknown_modes():
    assert flint.kernel_defaults({}) == (flint.DEFAULT_NAX_GATHER, flint.DEFAULT_ROUTED_DECODE)
    # Measured defaults (oq4e-kernels-ab-r2, oq4e-w2-r2): NAX prefill gather
    # and one-token routed decode (gate_up_down) on; the operator env wins.
    assert flint.kernel_defaults({}) == ("gather", "gate_up_down")
    assert flint.kernel_defaults({flint.NAX_GATHER_ENV: "off", flint.ROUTED_DECODE_ENV: "off"}) == ("off", "off")
    env = {flint.NAX_GATHER_ENV: "gather", flint.ROUTED_DECODE_ENV: "gate_up"}
    assert flint.kernel_defaults(env) == ("gather", "gate_up")
    with pytest.raises(ValueError, match="off or gather"):
        flint.kernel_defaults({flint.NAX_GATHER_ENV: "fused"})
    with pytest.raises(ValueError, match="must be one of"):
        flint.kernel_defaults({flint.ROUTED_DECODE_ENV: "tile4"})


def test_routed_decode_declines_off_metal_and_keeps_the_stock_block():
    model = tiny()
    cache = model.make_cache()
    model(mx.array([[4, 9, 2]]), cache=cache)
    reference_cache = copy.deepcopy(cache)
    stock = model(mx.array([[7]]), cache=reference_cache)
    blocks = [layer.mlp for layer in model.layers if hasattr(layer.mlp, "routed_decode")]
    for block in blocks:
        block.routed_decode = "gate_up_down"
    routed = model(mx.array([[7]]), cache=cache)
    assert mx.array_equal(stock, routed).item()
    assert all(block.routed_decode_calls == 0 for block in blocks)
    assert all(block.routed_decode_fallbacks == 1 for block in blocks)
    assert blocks[0].routed_decode_last_fallback == "activation must be bfloat16"


def test_routed_decode_latches_a_structural_refusal_once_per_block():
    # A BF16 artifact (no Q4 expert tables) can never admit routed decode, and
    # with the selection on by default every one-token MoE call would re-run
    # the structural admission.  The first structural refusal turns the block
    # off; the stock gathers then serve without re-checking.
    from mlx.utils import tree_map

    model = tiny()
    model.update(tree_map(lambda a: a.astype(mx.bfloat16), model.parameters()))
    blocks = [layer.mlp for layer in model.layers if hasattr(layer.mlp, "routed_decode")]
    for block in blocks:
        block.routed_decode = "gate_up_down"
    cache = model.make_cache()
    model(mx.array([[4, 9, 2]]), cache=cache)
    for token in (7, 3, 5):
        model(mx.array([[token]]), cache=cache)
    assert all(block.routed_decode == "off" for block in blocks)
    assert all(block.routed_decode_fallbacks == 1 for block in blocks)
    assert all(block.routed_decode_calls == 0 for block in blocks)
    assert "not a resident QuantizedSwitchLinear" in blocks[0].routed_decode_last_fallback
