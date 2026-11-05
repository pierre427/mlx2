"""CPU oracles for Kimi speculative rollback and the Kimi packed tree.

Synthetic tiny Kimi Linear models only; no checkpoint qualification claim.
"""
import copy
import hashlib
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters.kimi_linear import (
    DESCRIPTOR,
    KimiLinearAdapter,
    parse_policy,
    split_policy,
    validate_tree_policy,
)
from mlx2.runtime.drafters.kimi_dflash import (
    KimiDFlashDraftModel,
    best_first_tree,
    expected_weight_shapes,
    parse_config,
)
from mlx2.runtime.hybrid_tree_executor import tree_forward
from mlx2.runtime.kimi_tree_executor import KimiTreeExecutor
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.kimi_linear import KimiDeltaAttention, Model, ModelArgs


@pytest.fixture(autouse=True)
def cpu(request):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(118)
    request.node.user_properties.append((
        "source_revision", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))
    yield
    mx.set_default_device(previous)


def args(heads=2):
    return ModelArgs(model_type="kimi_linear", vocab_size=64, hidden_size=32,
        intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=1, layer_group_size=4, max_position_embeddings=1048576,
        rms_norm_eps=1e-5, rope_theta=10000., routed_scaling_factor=2.446,
        head_dim=72, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=16,
        v_head_dim=16, linear_attn_config={"full_attn_layers": [2, 4], "kda_layers": [1, 3],
                    "num_heads": heads, "head_dim": 128, "short_conv_kernel_size": 4})


def model(policy=None, request=None):
    m = Model(args())
    m.eval()
    m.configure(parse_policy(policy or {"kda": "reference"}))
    mx.eval(m.parameters())
    if request is not None:
        digest = hashlib.sha256()
        for name, value in tree_flatten(m.parameters()):
            digest.update((name + str(value.shape)).encode())
            digest.update(np.asarray(value.astype(mx.float32)).tobytes())
        request.node.user_properties.append(("checkpoint", "synthetic-kimi-linear@sha256:" + digest.hexdigest()))
    return m


def close(a, b, atol=3e-5, rtol=3e-5):
    np.testing.assert_allclose(np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32)),
                               atol=atol, rtol=rtol)


def same_cache(left, right):
    for a, b in zip(left, right, strict=True):
        if isinstance(a, KVCache):
            assert a.offset == b.offset
            close(a.keys[..., :a.offset, :], b.keys[..., :b.offset, :])
            close(a.values[..., :a.offset, :], b.values[..., :b.offset, :])
        else:
            for x, y in zip(a.cache, b.cache, strict=True):
                close(x, y)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("prefill", [0, 6])
def test_kda_rollback_is_exact_for_every_trim(fused, prefill):
    attn = KimiDeltaAttention(args(4))
    attn.eval()
    if fused:
        attn.prepare_projections()
        attn.fused_projections = True
    x = mx.random.normal((1, prefill + 5, 32))
    for drop in range(0, 6):
        live, reference = ArraysCache(size=4), ArraysCache(size=4)
        if prefill:
            attn(x[:, :prefill], cache=live)
            attn(x[:, :prefill], cache=reference)
        live.start_speculation()
        attn(x[:, prefill:], cache=live)
        live.trim(drop)
        live.stop_speculation()
        if 5 - drop:
            attn(x[:, prefill:prefill + 5 - drop], cache=reference)
        for a, b in zip(live.cache, reference.cache, strict=True):
            if a is None or b is None:
                assert a is None and b is None
            else:
                close(a, b)
    assert attn.calls["rollback_records"] == 6
    assert attn.calls["rollback_replays"] >= 4


def test_kda_rollback_refuses_untracked_trim_and_undescribable_geometry():
    attn = KimiDeltaAttention(args())
    attn.eval()
    cache = ArraysCache(size=4)
    attn(mx.random.normal((1, 3, 32)), cache=cache)
    with pytest.raises(RuntimeError):
        cache.trim(1)  # not speculating: fail closed, never a silent wrong state
    padded = ArraysCache(size=4, left_padding=[1, 0])
    padded.start_speculation()
    attn(mx.random.normal((2, 3, 32)), cache=padded)
    assert attn.calls["rollback_unstaged"] == 1
    with pytest.raises(RuntimeError):
        padded.trim(1)


@pytest.mark.parametrize("policy", [{"kda": "reference"},
                                    {"kda": "reference", "mla": "expanded"},
                                    {"kda": "reference", "mla": "auto"},
                                    {"kda": "reference", "fused_projections": True}])
def test_model_verify_trim_equals_shorter_forward(policy, request):
    m = model(policy, request)
    prompt, block = mx.array([[5, 9, 2, 7]]), mx.array([[11, 3, 8, 1, 6]])
    for accepted in range(1, 6):
        live, reference = m.make_cache(), m.make_cache()
        m(prompt, live)
        m(prompt, reference)
        for entry in live:
            if isinstance(entry, ArraysCache):
                entry.start_speculation()
        verify = m(block, live)
        for entry in live:
            entry.trim(5 - accepted)
            if isinstance(entry, ArraysCache):
                entry.stop_speculation()
        expected = m(block[:, :accepted], reference)
        close(verify[:, :accepted], expected)
        same_cache(live, reference)
        close(m(mx.array([[4]]), live), m(mx.array([[4]]), reference))


@pytest.mark.parametrize("policy", [{"kda": "reference"},
                                    {"kda": "reference", "fused_projections": True},
                                    {"kda": "reference", "mla": "latent_single"},
                                    {"kda": "reference", "mla": "expanded"}])
@pytest.mark.parametrize("prefill", [[], [9, 8, 7]])
def test_packed_tree_matches_serial_one_token_nodes_and_commit(policy, prefill, request):
    m = model(policy, request)
    cache = m.make_cache()
    if prefill:
        mx.eval(m(mx.array([prefill]), cache))
    serial_cache, packed_cache = copy.deepcopy(cache), copy.deepcopy(cache)
    tokens, parents = [1, 2, 3, 4, 5, 6], [-1, 0, 0, 2, 1, 3]
    expected, expected_taps, reference = tree_forward(m, tokens, parents, serial_cache, [0, 1, 2], "oracle")
    provider = KimiTreeExecutor.create()
    actual, actual_taps, transaction = provider.forward(m, tokens, parents, packed_cache, [0, 1, 2],
                                                        cached=False)
    close(actual, expected)
    close(actual_taps, expected_taps)
    assert packed_cache[1].offset == len(prefill)  # nothing published before commit
    reference.commit_paths([[0, 2, 3, 5]])
    transaction.commit_paths([[0, 2, 3, 5]])
    same_cache(serial_cache, packed_cache)
    close(m(mx.array([[7]]), serial_cache), m(mx.array([[7]]), packed_cache))
    receipt = provider.receipt()
    assert receipt["provider"] == "kimi_linear_layerwise_packed_tree_v1"
    calls = m.mechanism_status()
    assert calls["kda"]["tree_calls"] == 2 and calls["mla"]["tree_calls"] == 2


def test_tree_abort_leaves_live_state_and_ling_math_is_refused():
    m = model()
    cache = m.make_cache()
    logits, taps, transaction = KimiTreeExecutor.create().forward(
        m, [1, 2, 3], [-1, 0, 0], cache, [0], cached=False)
    mx.eval(logits, taps)
    transaction.abort()
    assert cache[1].offset == 0 and all(v is None for v in cache[0].cache)
    from mlx2.runtime.models.ling_tree import LingTree

    with pytest.raises(ValueError, match="Kimi tree contract"):
        m.forward_tree_with_taps(mx.array([[1, 2]], dtype=mx.uint32), LingTree([-1, 0], m.make_cache()), [0])


def test_dflash_taps_are_post_layer_residuals():
    m = model()
    tokens = mx.array([[3, 1, 4, 1, 5]])
    logits, taps = m.forward_with_taps(tokens, m.make_cache(), [0, 2])
    hidden = m.model.word_embeddings(tokens)
    mask = None
    from mlx2.runtime.models.base import create_attention_mask
    seen = []
    cache = m.make_cache()
    attn_mask = create_attention_mask(hidden, cache[m.model._attention_idx], return_array=True)
    for index, (layer, entry) in enumerate(zip(m.layers, cache)):
        hidden = layer(hidden, mask=attn_mask if layer.is_global else mask, cache=entry)
        seen.append(hidden)
    close(taps, mx.concatenate([seen[0], seen[2]], axis=-1))
    close(logits, m(tokens, m.make_cache()))


def test_kimi_mla_policy_auto_keeps_verify_rows_latent():
    m = model({"kda": "reference", "mla": "auto"})
    cache = m.make_cache()
    m(mx.array([list(range(1, 21))]), cache)  # 20 rows: expanded prompt chunk
    for entry in cache:
        if isinstance(entry, ArraysCache):
            entry.start_speculation()
    m(mx.array([list(range(1, 17))]), cache)  # 16-row verify block
    mla = m.mechanism_status()["mla"]
    assert mla["auto_expanded"] == 2 and mla["auto_latent"] == 2


# -- policy, route selection, fail-closed refusals ---------------------------

TARGET = Path("~/Models/Huihui-Kimi-Linear-REAP-35B-A3B-Instruct-abliterated")
DRAFT = Path("~/Models/Kimi-Linear-48B-A3B-DFlash-240k")


def resolution():
    from mlx2.adapters.registry import AdapterResolution

    return AdapterResolution(KimiLinearAdapter, DESCRIPTOR, {"has_mtp": False})


def test_policy_split_keeps_numerics_and_speculative_keys_apart():
    numerics, speculative = split_policy({"mla": "auto", "draft_model": "x", "num_draft": 4})
    assert numerics == {"mla": "auto"} and speculative == {"draft_model": "x", "num_draft": 4}
    assert parse_policy(numerics)["mla"] == "auto"
    assert validate_tree_policy({"draft_model": "x"}) is False
    assert validate_tree_policy({"batch_size_route": "tree15_b1_b4_chain_b5plus_v1"}) is True
    for bad in ({"batch_size_route": "tree15_bounded_cohorts_v1"},
                {"ast_tile_rows": 8},  # tree knob without the tree route
                {"batch_size_route": "tree15_b1_b4_chain_b5plus_v1", "ast_target_executor": "fast"},
                {"batch_size_route": "tree15_b1_b4_chain_b5plus_v1", "ast_tile_rows": 32},
                {"batch_size_route": "tree15_b1_b4_chain_b5plus_v1", "pairwise_selection": "batched"}):
        with pytest.raises(ValueError):
            validate_tree_policy(bad)


@pytest.mark.parametrize("policy,refusal", [
    ({"draft_model": "prompt_lookup"}, None),  # --ast PLD tiles reach artifact inspection
    ({"draft_model": "prompt_lookup", "ast_graft_prompt_lookup": True}, "graft"),
    ({"mla": "auto", "num_draft": 3}, "only an external draft_model"),
    ({"draft_model": "/tmp/d", "ast_tile_rows": 8}, "tree batch_size_route"),
    ({"draft_model": "/tmp/d", "ast_tree_calibration": "order"}, "only an external draft_model"),
    ({"mla": "ds41"}, "unsupported Kimi MLA"),
])
def test_adapter_policy_fails_closed_before_any_tensor_load(policy, refusal, tmp_path):
    adapter = object.__new__(KimiLinearAdapter)
    with pytest.raises((ValueError, OSError)) as error:
        adapter._load(tmp_path / "missing-model", policy)
    if refusal is None:
        assert "Kimi" not in str(error.value) or "topology" in str(error.value) or \
            isinstance(error.value, OSError)
        assert adapter.tree_route is True and adapter.drafter_kind == "prompt_lookup"
        assert adapter.external_policy["ast_topology"] == "chain"
    else:
        assert refusal in str(error.value)


def test_latched_prompt_lookup_is_the_no_flag_route_and_others_are_opt_in():
    from mlx2.server import build_parser, resolve_route_selection

    parser = build_parser()
    selection = resolve_route_selection(parser.parse_args(["--model", "/tmp/m"]), None, resolution())
    assert selection.route == "prompt_lookup" and selection.source == "capability_default"
    plain = resolve_route_selection(parser.parse_args(["--model", "/tmp/m", "--ordinary"]), None, resolution())
    assert plain.route == "ordinary"
    explicit = resolve_route_selection(parser.parse_args(["--model", "/tmp/m", "--prompt-lookup"]),
                                       None, resolution())
    assert explicit.route == "prompt_lookup" and explicit.source == "explicit_flag"
    assert KimiLinearAdapter.default_route == "ordinary"
    assert KimiLinearAdapter.prompt_lookup_default is True


@pytest.mark.parametrize("drafter,expected", [(None, "prompt_lookup"), ("pld", "prompt_lookup"),
                                               ("/tmp/draft", "/tmp/draft")])
def test_ast_invocation_selects_tree_tiles_for_pld_and_dflash(drafter, expected):
    from mlx2.server import apply_ast_invocation, build_parser

    argv = ["--model", "/tmp/m", "--ast"] + ([] if drafter is None else ["--ast-drafter", drafter])
    args = build_parser().parse_args(argv)
    policy = apply_ast_invocation(args, None, resolution())
    assert policy["draft_model"] == expected and args.external_draft
    assert args.ast_route["tree_tiles"] is True
    if expected != "prompt_lookup":
        assert policy["batch_size_route"] == "tree15_b1_b4_chain_b5plus_v1"


def test_bf16_artifact_refuses_raw_tensorfold_and_records_lane_verdict():
    adapter = object.__new__(KimiLinearAdapter)
    with pytest.raises(ValueError, match="not applicable"):
        adapter.bind_serving_lane_matmul({"law_id": "tensorfold-lane-qmm-v1+fused-groups"})
    adapter.bind_serving_lane_matmul({"law_id": "lane-matmul-v1", "covered": {"bf16": 3}})
    assert adapter.lane_projection_law["row_invariant_target"] is False
    assert "moe_switch_experts_gather" in adapter.lane_projection_law["uncovered_by_construction"]
    quantized = object.__new__(KimiLinearAdapter)
    quantized.artifact_layout = "mlx_affine"
    quantized.bind_serving_lane_matmul({"law_id": "tensorfold-lane-qmm-v1+fused-groups"})
    assert quantized.lane_projection_law["row_invariant_target"] is False


def test_dense_weight_reads_through_tensorfold_retiling():
    """Raw TensorFold re-tiles a packed bank in place; Kimi's absorbed MLA and
    router read the dense matrix and must see the checkpoint values."""
    from mlx import nn

    from mlx2.runtime.models.kimi_linear import dense_weight
    from mlx2.runtime.tensorfold_qwen38 import lane_qmm

    holder = nn.Module()
    holder.kv_b = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=8)
    holder.kv_b.scales = holder.kv_b.scales.astype(mx.bfloat16)
    holder.kv_b.biases = holder.kv_b.biases.astype(mx.bfloat16)
    expected = mx.dequantize(holder.kv_b.weight, holder.kv_b.scales, holder.kv_b.biases,
                             group_size=64, bits=8)
    try:
        lane_qmm.install(holder, tile=True)
        assert holder.kv_b._lane_tiled
        stale = mx.dequantize(holder.kv_b.weight, holder.kv_b.scales, holder.kv_b.biases,
                              group_size=64, bits=8)
        assert not bool(mx.array_equal(stale, expected).item())  # the bug's mechanism
        assert bool(mx.array_equal(dense_weight(holder.kv_b), expected).item())
        assert bool(mx.array_equal(dense_weight(holder.kv_b, mx.float32), expected.astype(mx.float32)).item())
    finally:
        lane_qmm.uninstall()


def test_prometheus_exposes_speculative_kda_mla_labels_only():
    from mlx2.prometheus import PrometheusBuilder, _add_execution

    b = PrometheusBuilder()
    _add_execution(b, {"kda": {"rollback_records": 5, "tree_calls": 2, "made_up": 9},
                       "mla": {"tree_nodes": 30}})
    text = b.render()
    assert 'mlx2_kda_events_total{event="rollback_records"} 5' in text
    assert 'mlx2_kda_events_total{event="tree_calls"} 2' in text
    assert 'mlx2_mla_events_total{event="tree_nodes"} 30' in text
    assert "made_up" not in text


def test_every_kimi_speculative_counter_reaches_prometheus():
    """A KDA/MLA counter the model or tree can emit but /metrics drops is a silent receipt."""
    import re

    import mlx2.prometheus as prom

    root = Path(prom.__file__).parent / "runtime" / "models"
    source = "".join((root / name).read_text() for name in ("kimi_linear.py", "kimi_tree.py"))
    emitted = set(re.findall(r'calls\["([a-z0-9_]+)"\]', source))
    for pair in re.findall(r'calls\["([a-z0-9_]+)" if [a-z_]+ else "([a-z0-9_]+)"\]', source):
        emitted |= set(pair)
    emitted |= {f"{shape}_{kind}" for shape in ("multirow", "single_row") for kind in ("calls", "tokens")}
    emitted |= {"norm_fla_fp32", "norm_qwen_bf16"}
    emitted -= {"router_calls", "router_tokens"}  # MoE router receipts, not KDA/MLA events
    assert {"rollback_records", "rollback_replays", "rollback_unstaged", "tree_calls",
            "tree_nodes", "tree_levels", "segmented_rows"} <= emitted
    text = Path(prom.__file__).read_text()
    allow = text[text.index("mechanism_events = {"):text.index("for mechanism,events in mechanism_events.items()")]
    assert not sorted(e for e in emitted if f'"{e}"' not in allow)
    # Every scheduler counter the round-shape gauges read is itself exported.
    for keys in prom._SPECULATIVE_ROUND_SHAPES.values():
        assert set(keys) <= prom._SCHEDULER_EVENTS
    assert {"external_cost_parked_rounds", "external_cost_park_events"} <= prom._SCHEDULER_EVENTS
    from mlx2.inference_monitor import MECHANISM_FAMILIES

    families = {name for name, _ in MECHANISM_FAMILIES}
    assert {"kda_events_total", "mla_events_total", "scheduler_events_total"} <= families


def test_speculative_round_shapes_and_route_info_render_bounded():
    from mlx2.prometheus import (
        PrometheusBuilder,
        _add_execution,
        _add_scheduler,
        _add_speculative_round_shapes,
    )

    builder = PrometheusBuilder()
    pld = {"pld_retrieval_cycles": 84, "pld_accepted": 390, "pld_proposed": 672, "pld_cycles": 874}
    _add_scheduler(builder, pld)
    _add_speculative_round_shapes(builder, pld)
    tree = {"external_rounds": 98, "accepted_proposals": 92, "proposed_tokens": 418,
            "external_tree_rounds": 98, "external_tree_accepted_edges": 92, "external_tree_nodes": 418,
            "external_cost_parked_rounds": 1067}
    _add_scheduler(builder, tree)
    _add_speculative_round_shapes(builder, tree)
    _add_execution(builder, {"speculative_route": "ast_kimi_dflash_tree_packed"})
    _add_execution(builder, {"speculative_route": "Not A Route {injected}"})
    text = builder.render()
    assert 'mlx2_speculative_acceptance_length{kind="prompt_lookup"} 5.64' in text
    assert 'mlx2_speculative_verify_rows{kind="prompt_lookup"} 9' in text
    assert 'mlx2_speculative_acceptance_length{kind="external_tree"}' in text
    assert 'kind="external_chain"' not in text  # tree rounds also move the chain counters
    assert 'event="external_cost_parked_rounds"' in text
    assert 'mlx2_speculative_route_info{route="ast_kimi_dflash_tree_packed"} 1' in text
    assert 'route="other"' in text and "injected" not in text


def test_external_chain_round_shape_divides_by_per_lane_rounds():
    from mlx2.prometheus import PrometheusBuilder, _add_speculative_round_shapes

    # 10 cohort rounds at width 4; each lane accepted 2 of 3 drafts per round.
    # Proposal totals are summed per lane, so the per-round mean is 3 and 4.
    stats = {"external_rounds": 10, "external_lane_rounds": 40,
             "accepted_proposals": 4 * 2 * 10, "proposed_tokens": 4 * 3 * 10}
    builder = PrometheusBuilder()
    _add_speculative_round_shapes(builder, stats)
    text = builder.render()
    assert 'mlx2_speculative_acceptance_length{kind="external_chain"} 3' in text
    assert 'mlx2_speculative_verify_rows{kind="external_chain"} 4' in text
    # Without the per-lane counter the cohort count is not a valid divisor.
    builder = PrometheusBuilder()
    _add_speculative_round_shapes(builder, {k: v for k, v in stats.items()
                                            if k != "external_lane_rounds"})
    assert 'kind="external_chain"' not in builder.render()


def test_tensorfold_qmv_policy_coverage_and_metrics():
    from mlx import nn

    from mlx2.prometheus import PrometheusBuilder, _add_execution
    from mlx2.runtime.models.flash_tensorfold_qmv import eligible

    assert parse_policy()["tensorfold_qmv"] is False
    with pytest.raises(ValueError):
        parse_policy({"tensorfold_qmv": "on"})
    # The kernel tiles K in 512 columns: Kimi's K=2304 inputs and the head
    # are outside it; o_proj (K=4096) and kv_b (K=512) are inside.
    for k, ok in ((2304, False), (4096, True), (512, True), (1024, True)):
        layer = nn.QuantizedLinear(k, 256, bias=False, group_size=64, bits=8)
        layer.scales = layer.scales.astype(mx.bfloat16)
        layer.biases = layer.biases.astype(mx.bfloat16)
        assert eligible(layer) is ok
    builder = PrometheusBuilder()
    _add_execution(builder, {"tensorfold_qmv": {"installed": 54, "counters": {
        "kernel_calls": 7, "stock_calls": 2, "kernel_rows": 9}}})
    text = builder.render()
    assert 'mlx2_tensorfold_qmv_calls_total{path="kernel"} 7' in text
    assert 'mlx2_tensorfold_qmv_rows_total 9' in text
    assert 'mlx2_tensorfold_qmv_installed_projections 54' in text


# -- DFlash head: config, tree, laws, pairing --------------------------------

DRAFT_CONFIG = {
    "architectures": ["DFlashDraftModel"], "model_type": "qwen3", "attention_bias": False,
    "block_size": 16, "bos_token_id": 163584, "eos_token_id": 163586,
    "dflash_config": {"causal_head": False, "mask_token_id": 163839,
                      "target_layer_ids": [3, 7, 15, 19, 23]},
    "dtype": "bfloat16", "head_dim": 72, "hidden_size": 2304, "intermediate_size": 9216,
    "layer_types": ["full_attention"] * 5, "max_position_embeddings": 1048576,
    "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 32,
    "num_target_layers": 27, "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "sliding_window": None, "tie_word_embeddings": False, "vocab_size": 163840,
}


def test_draft_config_parses_rope_parameters_and_refuses_ignored_geometry():
    args = parse_config(DRAFT_CONFIG)
    assert (args.rope_theta, args.head_dim, args.mask_token_id, args.block_size) == (10000.0, 72, 163839, 16)
    assert args.target_layer_ids == [3, 7, 15, 19, 23]
    shapes = expected_weight_shapes(args)
    assert shapes["fc.weight"] == [2304, 11520] and len(shapes) == 3 + 5 * 11
    for change in ({"dflash_config": {**DRAFT_CONFIG["dflash_config"], "causal_head": True}},
                   {"rope_parameters": {"rope_theta": 1e4, "rope_type": "yarn"}},
                   {"layer_types": ["sliding_attention"] * 5, "sliding_window": 512},
                   {"architectures": ["DFlash2DraftModel"]},
                   {"dflash_config": {**DRAFT_CONFIG["dflash_config"], "target_layer_ids": [3, 30]}}):
        with pytest.raises(ValueError):
            parse_config({**DRAFT_CONFIG, **change})


def test_best_first_tree_is_parent_first_and_respects_budget_and_root_mask():
    scores = np.log(np.array([[.6, .3, .1], [.9, .05, .05], [.5, .4, .1]]))
    candidates = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
    tokens, parents = best_first_tree(scores, candidates, 5, children=2)
    assert tokens[:3] == [10, 20, 11] and parents[:3] == [-1, 0, -1]
    assert len(tokens) == 5 and all(p < i for i, p in enumerate(parents))
    tokens, _ = best_first_tree(scores, candidates, 3, children=2, forbidden_root=[10])
    assert 10 not in tokens and tokens[0] == 11


def tiny_drafter(target):
    from mlx2.runtime.drafters.base_config import DFlashConfig

    cfg = DFlashConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                       vocab_size=64, rope_theta=1e4, block_size=6, mask_token_id=63,
                       target_layer_ids=[0, 2], num_target_layers=4, rms_norm_eps=1e-5)
    draft = KimiDFlashDraftModel(cfg)
    draft.eval()
    mx.eval(draft.parameters())
    return draft.bind(target)


def test_dflash_greedy_laws_trees_and_batched_rows_agree():
    from mlx2.runtime.speculative_sampling import RequestRNG

    m = model()
    draft = tiny_drafter(m)
    _, taps = m.forward_with_taps(mx.array([[3, 1, 4, 1, 5]]), m.make_cache(), [0, 2])
    single, batched = draft.make_cache(), [draft.make_cache(), draft.make_cache()]
    tokens, laws = draft.draft_distributions([7], taps, single, 5, [RequestRNG(1)], [0.0])
    assert len(tokens[0]) == 5 and all(law.token == t for law, t in zip(laws[0], tokens[0]))
    assert single[0].offset == 5  # committed context only, never the block
    rows = draft.batch_caches(batched)
    both, _ = draft.draft_distributions([7, 7], mx.concatenate([taps, taps]), rows, 5,
                                        [RequestRNG(1), RequestRNG(2)], [0.0, 0.0])
    assert both[0] == both[1] == tokens[0]
    trees = draft.propose_tree([7], mx.zeros((1, 0, 64)), single, 7)
    nodes, parents = trees[0]
    assert len(nodes) == 7 and parents[0] == -1 and nodes[0] == tokens[0][0]
    _, dense = draft.draft_distributions([7], mx.zeros((1, 0, 64)), single, 3,
                                         [RequestRNG(3)], [1.0])
    assert all(abs(law.sum() - 1) < 1e-9 for law in dense[0])
    for topology, branches, depth in (("3x5", 3, 5), ("4x3", 4, 3), ("5x3", 5, 3), ("2x7", 2, 5),
                                      ("chain", 1, 5)):
        nodes, parents = draft.propose_tree([7], mx.zeros((1, 0, 64)), single, 15,
                                            topology=topology, forbidden_token_ids=[[tokens[0][0]]])[0]
        roots = [node for node, parent in zip(nodes, parents) if parent == -1]
        assert len(roots) == branches and tokens[0][0] not in roots
        assert len(nodes) == branches * min(depth, 5)  # tail depth capped by the trained block
        assert all(p < i for i, p in enumerate(parents))


# -- serving generators on the tiny target (greedy token identity) -----------

COPY_PROMPTS = [[5, 6, 7, 8, 9, 10] * 4 + [5, 6, 7], [20, 21, 22, 23] * 6 + [20, 21]]


def greedy(m, prompt, count):
    cache = m.make_cache()
    logits = m(mx.array([prompt]), cache=cache)
    tokens = []
    for _ in range(count):
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        logits = m(mx.array([[tokens[-1]]]), cache=cache)
    return tokens


def drive(batch, prompts, count, **insert):
    uids = batch.insert([list(p) for p in prompts], max_tokens=[count] * len(prompts), **insert)
    outputs = {uid: [] for uid in uids}
    try:
        for _ in range(800):
            _, responses = batch.next()
            for response in responses:
                outputs[response.uid].append(response.token)
            if not batch.lanes:
                break
    finally:
        batch.close()
    return [outputs[uid] for uid in uids]


@pytest.mark.parametrize("lanes", [1, 2])
def test_prompt_lookup_generator_matches_ordinary_greedy(lanes, request):
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    m = model(request=request)
    batch = PromptLookupBatchGenerator(m, completion_batch_size=2, prefill_step_size=64,
                                       prompt_lookup={"num_draft": 4, "ngram_min": 2, "ngram_max": 3})
    prompts = COPY_PROMPTS[:lanes]
    got = drive(batch, prompts, 14, caches=[m.make_cache() for _ in prompts])
    for prompt, output in zip(prompts, got):
        assert output == greedy(m, prompt, 14)
    assert batch.scheduler_stats["pld_proposed"] > 0
    assert m.mechanism_status()["kda"].get("rollback_records", 0) > 0


def test_cost_latched_prompt_lookup_is_exact_and_exports_latch_counters(request):
    from mlx2.prometheus import PrometheusBuilder, _add_scheduler
    from mlx2.runtime.pld import PLD_COST_LATCH_COUNTERS, PromptLookupBatchGenerator
    from mlx2.server import DEFAULT_PROMPT_LOOKUP_POLICY

    m = model(request=request)
    batch = PromptLookupBatchGenerator(m, completion_batch_size=1, prefill_step_size=64,
                                       prompt_lookup={**DEFAULT_PROMPT_LOOKUP_POLICY, "num_draft": 4})
    prompt = COPY_PROMPTS[0] * 2
    got = drive(batch, [prompt], 40, caches=[m.make_cache()])
    assert got[0] == greedy(m, prompt, 40)
    stats = batch.scheduler_stats
    assert stats.get("pld_cost_shadow_lookups", 0) > 0
    builder = PrometheusBuilder()
    _add_scheduler(builder, stats)
    text = builder.render()
    for name in PLD_COST_LATCH_COUNTERS:
        if stats.get("pld_cost_" + name):
            assert f'event="pld_cost_{name}"' in text


@pytest.mark.parametrize("tree", [False, True])
def test_external_dflash_generator_matches_ordinary_greedy(tree, request):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    m = model(request=request)
    draft = tiny_drafter(m)
    extra = {}
    if tree:
        extra = dict(dynamic_singleton_tree=True, dynamic_tree_max_width=4,
                     target_tree_executor_provider=KimiTreeExecutor.create(),
                     tree_node_budget_by_lanes={1: 7, 2: 3, 3: 2, 4: 1})
    batch = ExternalDraftBatchGenerator(m, draft_model=draft, binding="kimi-dflash-cpu", num_draft=4,
                                        prefill_step_size=8, completion_batch_size=2, **extra)
    got = drive(batch, COPY_PROMPTS, 12, sampling_configs=[{"sampling_temp": 0.0}] * 2)
    for prompt, output in zip(COPY_PROMPTS, got):
        assert output == greedy(m, prompt, 12)
    stats = batch.scheduler_stats
    assert stats["proposed_tokens"] > 0
    if tree:
        assert stats["external_tree_rounds"] > 0
        assert m.mechanism_status()["kda"]["tree_calls"] > 0
    else:
        assert stats.get("external_hybrid_transactions", 0) > 0


@pytest.mark.skipif(not (TARGET / "tiktoken.model").exists() or not (DRAFT / "config.json").exists(),
                    reason="local Kimi target/drafter artifacts absent")
def test_downloaded_head_pairs_with_the_target_tokenizer(request):
    import json

    from mlx2.adapters.kimi_dflash import REVISION, inspect_drafter, tokenizer_pairing

    target_config = json.loads((TARGET / "config.json").read_text())
    record = inspect_drafter(DRAFT, TARGET, target_config=target_config)
    request.node.user_properties.append(("checkpoint", record["checkpoint"]))
    assert record["pinned_revision"] == REVISION
    pairing = record["tokenizer_pairing"]
    assert pairing["mask_token"] == "[PAD]" and pairing["vocab_size"] == 163840
    with pytest.raises(ValueError, match="mask_token_id"):
        tokenizer_pairing({**DRAFT_CONFIG, "dflash_config": {"mask_token_id": 5}}, TARGET, target_config)
    with pytest.raises(ValueError, match="eos"):
        tokenizer_pairing({**DRAFT_CONFIG, "eos_token_id": 1}, TARGET, target_config)
    # The drafter's own training target (unpruned 48B Instruct, MLX 8-bit)
    # carries the same pinned tiktoken vocabulary and special ids.
    q8 = Path("~/Models/Kimi-Linear-48B-A3B-Instruct-8bit")
    if (q8 / "tiktoken.model").exists():
        q8_config = json.loads((q8 / "config.json").read_text())
        assert tokenizer_pairing(DRAFT_CONFIG, q8, q8_config) == pairing


def test_draft_config_refuses_a_non_silu_mlp():
    # The draft MLP is dflash_base's SwiGLU; another activation was ignored.
    assert parse_config({**DRAFT_CONFIG, "hidden_act": "silu"}).block_size == 16
    with pytest.raises(ValueError, match="hidden_act"):
        parse_config({**DRAFT_CONFIG, "hidden_act": "gelu"})


def test_drafter_is_pinned_only_with_the_verified_single_pinned_shard(monkeypatch, tmp_path):
    import hashlib
    from types import SimpleNamespace

    import mlx2.adapters.kimi_dflash as kd
    import mlx2.runtime.drafters.kimi_dflash as rd

    raw = b'{"draft": 1}'
    (tmp_path / "config.json").write_bytes(raw)
    monkeypatch.setattr(kd, "CONFIG_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(rd, "parse_config", lambda config: SimpleNamespace(
        hidden_size=8, vocab_size=16, num_target_layers=2))
    monkeypatch.setattr(rd, "expected_weight_shapes", lambda args: {})
    monkeypatch.setattr(kd, "tokenizer_pairing", lambda *a: {})
    monkeypatch.setattr(kd, "_sha256", lambda path: kd.WEIGHTS_SHA256)
    target = {"hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 2}

    def inspect(shards, **kwargs):
        monkeypatch.setattr(kd, "inspect_files", lambda path, raw, **kw: {
            "path": str(path), "files": [(name, 1, 0) for name in shards], "header_sha256": "h"})
        return kd.inspect_drafter(tmp_path, tmp_path, target_config=target, **kwargs)

    assert inspect(["model.safetensors"])["checkpoint"] == f"{kd.REPO}@{kd.REVISION}"
    # A resharded set (even of pinned-hash shards) or unhashed weights is
    # never reported as the pinned revision.
    for record in (inspect(["a.safetensors", "b.safetensors"]),
                   inspect(["model.safetensors"], verify_content=False)):
        assert record["pinned_revision"] is None
        assert record["checkpoint"].startswith("unpinned@")


def test_dflash_block_is_the_trained_shape_for_every_proposal_length():
    """Bidirectional block: the reference runs anchor + block_size - 1 masks
    and slices, so a shorter proposal reads the same per-position logits."""
    from mlx import nn

    from mlx2.runtime.drafters.base_config import DFlashConfig

    cfg = DFlashConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4, head_dim=16,
                       vocab_size=97, block_size=16, mask_token_id=96,
                       target_layer_ids=[0, 1], num_target_layers=2, tie_word_embeddings=False,
                       rope_theta=10000.0)
    draft = KimiDFlashDraftModel(cfg)
    draft.embed_tokens = nn.Embedding(97, 64)
    draft.lm_head = nn.Linear(64, 97, bias=False)
    hidden = mx.random.normal((1, 5, 128))
    short = draft._block_logits([3], hidden, draft.make_cache(), 3)
    full = draft._block_logits([3], hidden, draft.make_cache(), 15)
    assert short.shape[1] == 3 and full.shape[1] == 15
    assert float(mx.abs(short - full[:, :3]).max()) < 1e-5
    with pytest.raises(ValueError, match="trained block"):
        draft._block_logits([3], hidden, draft.make_cache(), 16)


def receipt_adapter(layout="hub_bf16"):
    from types import SimpleNamespace

    adapter = object.__new__(KimiLinearAdapter)
    adapter.model = SimpleNamespace(mechanism_status=lambda: {"kda": {}, "mla": {}})
    adapter.tree_route = False
    adapter.artifact_layout = layout
    return adapter


def test_capability_receipt_default_is_the_no_flag_route():
    from types import SimpleNamespace

    from mlx2.server import resolve_route_selection

    flags = SimpleNamespace(ordinary=False, external_draft=False, prompt_lookup=False, native_mtp=False)
    for adapter_type in (KimiLinearAdapter,
                         type("ExplicitPLD", (KimiLinearAdapter,), {"prompt_lookup_default": False})):
        route = resolve_route_selection(flags, {}, SimpleNamespace(
            descriptor=DESCRIPTOR, default_route=adapter_type.default_route,
            adapter_type=adapter_type)).route
        adapter = object.__new__(adapter_type)
        adapter.__dict__.update(receipt_adapter().__dict__)
        receipts = adapter.capability_receipts()
        assert receipts["prompt_lookup"]["default"] is (route == "prompt_lookup")
        assert receipts["ordinary_reference"]["default"] is (route == "ordinary")
    assert route == "ordinary"  # the explicit-only subclass


def test_tensorfold_receipt_follows_the_artifact_layout_and_lane_binding():
    bf16 = receipt_adapter().capability_receipts()["tensorfold_lane"]
    assert bf16["raw_tensorfold"] == "not_applicable_bf16_no_quantized_linear"
    q8 = receipt_adapter("mlx_affine")
    assert q8.capability_receipts()["tensorfold_lane"]["raw_tensorfold"] == "available_not_selected"
    q8.bind_serving_lane_matmul({"law_id": "tensorfold-lane-qmm-v1+fused-groups"})
    lane = q8.capability_receipts()["tensorfold_lane"]
    assert lane["raw_tensorfold"] == "selected_q8_projections_experts_stock"
    assert lane["selected"]["law_id"].startswith("tensorfold-lane-qmm")
    assert "BF16" not in lane["implemented"]


KIMI_TREE_MODULES = ("kimi_tree_executor.py", "models/kimi_tree.py", "models/kimi_linear.py",
                     "models/kimi_linear_mla_fused.py", "models/kimi_linear_mla_flash.py",
                     "models/kimi_linear_fused.py")


def test_packed_tree_revision_binds_every_package_import_of_the_kimi_modules():
    """Static half: each runtime module a Kimi tree/model/kernel module
    imports (lazy imports included) is bound, or exempted with a reason."""
    import ast

    from mlx2.runtime import kimi_tree_executor as kte

    root = Path(kte.__file__).parent
    bound = set(KimiTreeExecutor()._sources)
    missing = []
    for name in KIMI_TREE_MODULES:
        path = root / name
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            targets = ([node.module] if node.module else [alias.name for alias in node.names])
            for target in targets:
                module = base.joinpath(*target.split("."))
                file = module / "__init__.py" if module.is_dir() else module.with_suffix(".py")
                if file not in bound and target.split(".")[-1] not in kte.UNBOUND_IMPORTS:
                    missing.append(f"{name}: {target}")
    assert not missing, missing


@pytest.mark.parametrize("policy", [
    {"kda": "reference"}, {"kda": "kernel", "mla": "latent_fused"},
    {"kda": "chunk_wy", "mla": "latent_tiled"}, {"kda": "chunk32", "mla": "expanded"},
    {"mla": "latent_single", "fused_projections": True}])
def test_packed_tree_revision_binds_every_runtime_source_the_forward_executes(policy):
    """Dynamic half: every runtime source with a Python frame during a CPU
    tree forward is bound (GPU-only kernel branches are the static half's)."""
    import sys

    from mlx2.runtime import kimi_tree_executor as kte

    runtime = str(Path(kte.__file__).parent) + "/"
    m = model(policy)
    cache = m.make_cache()
    mx.eval(m(mx.array([[9, 8, 7]]), cache))
    provider = KimiTreeExecutor.create()
    executed = set()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_filename.startswith(runtime):
            executed.add(frame.f_code.co_filename)

    sys.setprofile(profile)
    try:
        logits, taps, _ = provider.forward(m, [1, 2, 3, 4], [-1, 0, 0, 1], cache, [0, 1], cached=False)
        mx.eval(logits, taps)
    finally:
        sys.setprofile(None)
    bound = {str(path) for path in KimiTreeExecutor()._sources}
    assert executed and not sorted(executed - bound)


def test_packed_tree_revision_check_rehashes_only_when_a_source_moves(tmp_path, monkeypatch):
    executor = KimiTreeExecutor()
    executor.validate_revision()
    copy_path = tmp_path / executor._sources[0].name
    copy_path.write_bytes(executor._sources[0].read_bytes())
    executor._sources = (copy_path, *executor._sources[1:])
    executor.source_identity = executor._identity()
    executor._source_stats = ()
    executor.validate_revision()  # same content, new stat: accepted and re-armed
    copy_path.write_bytes(copy_path.read_bytes() + b"\n# edit\n")
    with pytest.raises(RuntimeError, match="source changed"):
        executor.validate_revision()
