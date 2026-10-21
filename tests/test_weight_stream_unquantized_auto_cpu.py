"""Unquantized (gather_mm) expert streaming and the automatic selection (CPU).

Evidence here is synthetic: tiny real HY V3 / Laguna / Nemotron-H model
classes with their own ``sanitize``, real safetensors checkpoints in the
vendors' per-expert or stacked layouts, and the early loader.  Streamed and
resident forwards must agree bit for bit (paging changes where a byte sits,
never its value).  Oversized-model memory behaviour needs the hardware run.
"""

import inspect
import json

import pytest

mx = pytest.importorskip("mlx.core")
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

from mlx2.runtime.streamed_load import (
    MIN_LANE_HEADROOM_BYTES,
    WeightStreamRequest,
    artifact_weight_bytes,
    auto_expert_streaming_decision,
    declared_stream_modes,
    force_stock_expert_arithmetic,
    load_streamed,
)
from mlx2.runtime.weight_stream import WorkingSetTooSmall, is_streamed_module

LONG_PROMPT = [3, 7, 1, 12, 9, 30, 4, 18, 22, 5, 41, 2, 17, 8, 33, 11]
SHORT_PROMPT = [3, 7, 1, 12]


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def _write(root, weights):
    """Two shards and an index, as the vendors ship them."""
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(weights)
    half = len(names) // 2
    files = {
        "model-00001-of-00002.safetensors": names[:half],
        "model-00002-of-00002.safetensors": names[half:],
    }
    weight_map = {}
    for name, keys in files.items():
        mx.save_safetensors(str(root / name), {key: weights[key] for key in keys})
        weight_map.update({key: name for key in keys})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    records = [
        (name, (root / name).stat().st_size, (root / name).stat().st_mtime_ns)
        for name in sorted(files)
    ]
    return weight_map, records


def _split_experts(weights, *, stacked_marker, rename):
    """Unstack ``[E, ...]`` expert tables into one tensor per expert."""
    out = {}
    for key, value in weights.items():
        if stacked_marker in key:
            prefix, _, tail = key.partition(stacked_marker)
            for expert in range(value.shape[0]):
                piece = mx.array(np.array(value[expert].astype(mx.float32))).astype(value.dtype)
                out[f"{prefix}{rename(expert, tail)}"] = piece
        else:
            out[key] = value
    mx.eval(list(out.values()))
    return out


def _shards(weight_map):
    return sorted(set(weight_map.values()))


def _resident(model, root, weight_map, quantize=lambda weights: None):
    from mlx2.runtime.ubc_evict import load_shards_evicting

    weights = model.sanitize(
        load_shards_evicting([root / name for name in _shards(weight_map)])
    )
    quantize(weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    force_stock_expert_arithmetic(model)
    return model


def _streamed(model, root, weight_map, records, *, top_k, budget=None, limit=None,
              quantize=lambda weights: None):
    if budget is None:
        request = WeightStreamRequest(
            mode="moe_experts", budget_mode="auto", budget_bytes=0,
            read_workers=2, admission_limit_bytes=int(limit), bypass_page_cache=True,
        )
    else:
        request = WeightStreamRequest(
            mode="moe_experts", budget_bytes=int(budget), read_workers=2,
        )
    loaded = load_streamed(
        model,
        root,
        _shards(weight_map),
        request=request,
        sanitize=model.sanitize,
        quantize=quantize,
        records=records,
        weight_map=weight_map,
        top_k=top_k,
        on_installed=lambda manager: setattr(
            manager, "forced_stock", force_stock_expert_arithmetic(model)
        ),
    )
    loaded.manager.begin_serving()
    return loaded


def _logits(model, tokens):
    out = model(mx.array([tokens]))
    mx.eval(out)
    return np.array(out.astype(mx.float32))


# ---------------------------------------------------------------------------
# tiny real models
# ---------------------------------------------------------------------------

LAGUNA = {
    "model_type": "laguna", "vocab_size": 64, "hidden_size": 64,
    "intermediate_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 256,
    "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32, "mlp_only_layers": [0],
    "sliding_window": 512, "gating": "per-head",
}


def _laguna():
    from mlx2.runtime.models.laguna import Model, ModelArgs

    return Model(ModelArgs.from_dict(LAGUNA))


def _laguna_checkpoint(tmp_path, dtype=mx.float32):
    """The bf16 release layout: per-expert tensors, vendor router names.

    CPU ``gather_mm`` is float32-only, so routine tests write float32 tables;
    the bf16 tables run in the opt-in Metal oracle
    (``tests/test_weight_stream_bf16_metal.py``).
    """
    mx.random.seed(11)
    source = _laguna()
    source.set_dtype(dtype)
    mx.eval(source.parameters())
    weights = dict(tree_flatten(source.parameters()))
    for layer in range(1, LAGUNA["num_hidden_layers"]):
        prefix = f"model.layers.{layer}.mlp"
        weights[f"{prefix}.gate.weight"] = weights.pop(f"{prefix}.gate.proj.weight")
        weights[f"{prefix}.experts.e_score_correction_bias"] = weights.pop(
            f"{prefix}.gate.e_score_correction_bias"
        )
    weights = _split_experts(
        weights, stacked_marker=".switch_mlp.",
        rename=lambda expert, tail: f".experts.{expert}.{tail}",
    )
    assert "model.layers.1.mlp.experts.7.down_proj.weight" in weights
    assert all(".switch_mlp." not in key for key in weights)
    root = tmp_path / f"laguna-{dtype}"
    return (root, *_write(root, weights))


NEMOTRON = dict(
    model_type="nemotron_h", vocab_size=64, num_hidden_layers=4, hidden_size=32, intermediate_size=48,
    max_position_embeddings=256, num_attention_heads=2, num_key_value_heads=1,
    head_dim=16, attention_bias=False, mamba_num_heads=2, mamba_head_dim=16,
    mamba_proj_bias=False, ssm_state_size=8, conv_kernel=3, n_groups=1,
    mlp_bias=False, layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
    hybrid_override_pattern=["M", "E", "*", "E"], n_routed_experts=8,
    num_experts_per_tok=2, moe_intermediate_size=24,
    moe_shared_expert_intermediate_size=16, moe_latent_size=16, n_group=1,
    topk_group=1, n_shared_experts=1, norm_topk_prob=True, routed_scaling_factor=1.5,
)


def _nemotron():
    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    return Model(ModelArgs(**NEMOTRON))


def _nemotron_checkpoint(tmp_path, dtype=mx.float32):
    """The vendor layout: ``mixer.experts.N.{up,down}_proj`` per expert."""
    mx.random.seed(13)
    source = _nemotron()
    source.set_dtype(dtype)
    mx.eval(source.parameters())
    weights = dict(tree_flatten(source.parameters()))
    for key in weights:
        # The router initializes to zeros; spread the routing over experts.
        if key.endswith(".mixer.gate.weight"):
            weights[key] = mx.random.normal(weights[key].shape).astype(dtype)
    names = {"fc1": "up_proj", "fc2": "down_proj"}
    weights = _split_experts(
        weights, stacked_marker=".switch_mlp.",
        rename=lambda expert, tail: (
            f".experts.{expert}.{names[tail.split('.')[0]]}.{tail.split('.', 1)[1]}"
        ),
    )
    assert "backbone.layers.3.mixer.experts.7.down_proj.weight" in weights
    root = tmp_path / f"nemotron-{dtype}"
    return (root, *_write(root, weights))


HY = {
    "model_type": "hy_v3", "vocab_size": 64, "hidden_size": 64,
    "intermediate_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "num_experts": 8,
    "num_experts_per_tok": 2, "num_shared_experts": 1, "expert_hidden_dim": 64,
    "first_k_dense_replace": 1, "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "max_position_embeddings": 256,
}
HY_QUANT = {"group_size": 64, "bits": 6, "mode": "affine"}


def _hy():
    from mlx2.runtime.models.hy_v3 import Model, ModelArgs

    return Model(ModelArgs.from_dict(HY))


def _hy_quantize(model):
    def quantize(weights):
        nn.quantize(
            model, **HY_QUANT,
            class_predicate=lambda name, module: hasattr(module, "to_quantized")
            and f"{name}.scales" in weights,
        )

    return quantize


def _hy_checkpoint(tmp_path, *, per_expert):
    mx.random.seed(17)
    source = _hy()
    nn.quantize(source, **HY_QUANT, class_predicate=lambda _p, m: hasattr(m, "to_quantized"))
    mx.eval(source.parameters())
    weights = dict(tree_flatten(source.parameters()))
    if per_expert:
        weights = _split_experts(
            weights, stacked_marker=".switch_mlp.",
            rename=lambda expert, tail: f".experts.{expert}.{tail}",
        )
    root = tmp_path / ("hy-per-expert" if per_expert else "hy-stacked")
    return (root, *_write(root, weights))


def _expert_bytes(manager):
    return manager.plan.expert_bytes


def _switches(model):
    return [m for _, m in model.named_modules() if is_streamed_module(m)]


# ---------------------------------------------------------------------------
# bit identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [1, 8])
def test_laguna_per_expert_unquantized_streams_bit_identical(tmp_path, capacity):
    root, weight_map, records = _laguna_checkpoint(tmp_path)
    resident = _resident(_laguna(), root, weight_map)
    expected = {len(p): _logits(resident, p) for p in (SHORT_PROMPT, LONG_PROMPT)}
    model = _laguna()
    top_k = LAGUNA["num_experts_per_tok"]
    # One cache per streamed projection: 2 MoE layers x gate/up/down.
    expert = 32 * 64 * 4  # one float32 projection row
    loaded = _streamed(
        model, root, weight_map, records, top_k=top_k,
        budget=6 * max(capacity, top_k) * expert,
    )
    try:
        manager = loaded.manager
        assert manager.plan.expert_bytes == expert
        assert manager.plan.as_dict()["table_kinds"] == ["gather_mm"]
        assert manager.plan.capacity_experts == max(capacity, top_k)
        assert len(_switches(model)) == 3 * 2
        # Short (unsorted) and long (sorted, >= 20 assignments) gathers.
        for prompt in (SHORT_PROMPT, LONG_PROMPT, SHORT_PROMPT):
            got = _logits(model, prompt)
            assert np.array_equal(got, expected[len(prompt)])
        counters = manager.counters()
        assert counters["stream_page_ins_total"] > 0
        if capacity == 1:
            assert counters["stream_evictions_total"] > 0
        else:
            assert counters["stream_expert_hits_total"] > 0
        # The streamed tables left the tree.
        assert all("switch_mlp" not in key or "bias" in key
                   for key, _ in tree_flatten(model.parameters()))
    finally:
        loaded.manager.close()


def test_nemotron_per_expert_latent_moe_streams_bit_identical(tmp_path):
    root, weight_map, records = _nemotron_checkpoint(tmp_path)
    resident = _resident(_nemotron(), root, weight_map)
    expected = _logits(resident, LONG_PROMPT)
    model = _nemotron()
    expert = 24 * 16 * 4  # one float32 fc1/fc2 row
    loaded = _streamed(
        model, root, weight_map, records, top_k=2, budget=4 * 2 * expert,
    )
    try:
        assert loaded.manager.plan.expert_bytes == expert
        assert loaded.manager.forced_stock == {"relu2_down": 2, "routed_batch": 2}
        assert np.array_equal(_logits(model, LONG_PROMPT), expected)
        assert np.array_equal(_logits(model, SHORT_PROMPT), _logits(resident, SHORT_PROMPT))
        assert loaded.manager.counters()["stream_evictions_total"] > 0
        moe = model.backbone.layers[1].mixer
        assert moe.routed_batch_refusal(mx.zeros((1, 2, 32))) == "expert streaming"
        with pytest.raises(ValueError, match="streamed expert table"):
            moe.set_relu2_down_policy("fused_b1")
    finally:
        loaded.manager.close()


@pytest.mark.parametrize("per_expert", [False, True])
def test_hy_v3_quantized_stacked_and_per_expert_stream_bit_identical(tmp_path, per_expert):
    root, weight_map, records = _hy_checkpoint(tmp_path, per_expert=per_expert)
    resident_model = _hy()
    resident = _resident(resident_model, root, weight_map, _hy_quantize(resident_model))
    expected = _logits(resident, LONG_PROMPT)
    model = _hy()
    expert = sum(
        int(value.nbytes) for key, value in tree_flatten(resident.parameters())
        if key.startswith("model.layers.1.mlp.switch_mlp.gate_proj.")
    ) // HY["num_experts"]
    # The floor of one routed step: 2 layers x 3 projections x top_k rows.
    loaded = _streamed(
        model, root, weight_map, records, top_k=2, budget=6 * 2 * expert,
        quantize=_hy_quantize(model),
    )
    try:
        assert "table_kinds" not in loaded.manager.plan.as_dict()  # gather_qmm only
        assert "page_cache" not in loaded.manager.plan.as_dict()  # explicit: kernel cache
        assert np.array_equal(_logits(model, LONG_PROMPT), expected)
        assert loaded.manager.counters()["stream_evictions_total"] > 0
    finally:
        loaded.manager.close()


def test_unquantized_tables_refuse_the_quantized_only_overlap(tmp_path):
    root, weight_map, records = _laguna_checkpoint(tmp_path)
    model = _laguna()
    with pytest.raises(Exception, match="quantized-only"):
        load_streamed(
            model, root, _shards(weight_map),
            request=WeightStreamRequest(
                mode="moe_experts", budget_bytes=1 << 30, read_workers=2,
                expert_overlap=True,
            ),
            sanitize=model.sanitize, quantize=lambda weights: None,
            records=records, weight_map=weight_map, top_k=2,
        )


def test_laguna_sanitize_without_the_stack_ledger_is_refused(tmp_path, monkeypatch):
    """The invariant's guard: an unledgered stack is a transform, not a rename."""
    from mlx2.runtime import weight_stream

    root, weight_map, records = _laguna_checkpoint(tmp_path)
    monkeypatch.setattr(weight_stream, "record_stack", lambda *a, **k: None)
    model = _laguna()
    with pytest.raises(Exception, match="transformed during sanitize"):
        _streamed(model, root, weight_map, records, top_k=2, budget=1 << 30)


# ---------------------------------------------------------------------------
# automatic ceiling in the loader
# ---------------------------------------------------------------------------


def test_auto_budget_sizes_the_cache_from_the_admission_limit(tmp_path):
    root, weight_map, records = _laguna_checkpoint(tmp_path)
    resident = _resident(_laguna(), root, weight_map)
    expected = _logits(resident, LONG_PROMPT)
    model = _laguna()
    expert = 32 * 64 * 4
    table = 6 * 8 * expert
    total = sum(int(v.nbytes) for _, v in tree_flatten(resident.parameters()))
    remainder = total - table
    # Room for about half the table after the remainder, headroom and transient.
    limit = remainder + MIN_LANE_HEADROOM_BYTES + 2 * table
    loaded = _streamed(model, root, weight_map, records, top_k=2, limit=limit)
    try:
        plan = loaded.manager.plan
        assert plan.ceiling_source == "auto"
        assert plan.as_dict()["ceiling_source"] == "auto"
        assert plan.as_dict()["page_cache"] == "bypassed"
        assert plan.ceiling_bytes == plan.capacity_experts * plan.layers * plan.expert_bytes
        assert plan.resident_remainder_bytes == remainder
        assert remainder + loaded.manager.reserved_bytes() + MIN_LANE_HEADROOM_BYTES <= limit
        assert np.array_equal(_logits(model, LONG_PROMPT), expected)
    finally:
        loaded.manager.close()


def test_auto_budget_refuses_with_numbers_when_the_remainder_cannot_fit(tmp_path):
    root, weight_map, records = _laguna_checkpoint(tmp_path)
    model = _laguna()
    with pytest.raises(WorkingSetTooSmall, match="automatic expert streaming cannot fit"):
        _streamed(model, root, weight_map, records, top_k=2,
                  limit=MIN_LANE_HEADROOM_BYTES)


def test_auto_request_validation():
    with pytest.raises(ValueError, match="measured admission limit"):
        WeightStreamRequest(mode="moe_experts", budget_mode="auto", budget_bytes=0)
    with pytest.raises(ValueError, match="pass 0"):
        WeightStreamRequest(mode="moe_experts", budget_mode="auto", budget_bytes=5,
                            admission_limit_bytes=1 << 30)
    with pytest.raises(ValueError, match="MoE-only"):
        WeightStreamRequest(mode="dense_mlp", budget_mode="auto", budget_bytes=0,
                            admission_limit_bytes=1 << 30)
    with pytest.raises(ValueError, match="budget mode"):
        WeightStreamRequest(mode="moe_experts", budget_mode="guess", budget_bytes=1)
    with pytest.raises(ValueError, match="budget_bytes"):
        WeightStreamRequest(mode="moe_experts", budget_bytes=0)


# ---------------------------------------------------------------------------
# declarations
# ---------------------------------------------------------------------------


def test_target_adapters_declare_expert_streaming_on_their_own_class():
    from mlx2.adapters.hy_v3 import HYV3Adapter
    from mlx2.adapters.laguna_s21 import LagunaS21Adapter
    from mlx2.adapters.laguna_xs21 import LagunaXS21Adapter
    from mlx2.adapters.nemotron35_bolt import Nemotron35BoltAdapter
    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter

    for adapter in (HYV3Adapter, LagunaS21Adapter, Nemotron35SuperAdapter):
        assert declared_stream_modes(adapter) == frozenset({"moe_experts"})
        assert "weight_streaming" in inspect.signature(adapter.__init__).parameters
    # Never inherited.
    assert declared_stream_modes(Nemotron35BoltAdapter) == frozenset()
    assert declared_stream_modes(LagunaXS21Adapter) == frozenset()


def test_super_vl_refuses_table_reading_kernels_before_loading(tmp_path):
    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter

    request = WeightStreamRequest(mode="moe_experts", budget_bytes=1 << 30)
    for policy, match in (
        ({"moe_relu2_down": "fused_b1"}, "moe_relu2_down"),
        ({"mamba_segment": "layerwise_short_exact_routed_batch"}, "routed_batch"),
    ):
        with pytest.raises(ValueError, match=match):
            Nemotron35SuperAdapter(
                str(tmp_path / "absent"), execution_policy=policy,
                weight_streaming=request,
            )


# ---------------------------------------------------------------------------
# the automatic selection
# ---------------------------------------------------------------------------


class _Declared:
    weight_streaming_modes = frozenset({"moe_experts"})
    calls = []

    def __init__(self, model_path, **kwargs):
        type(self).calls.append(kwargs)
        raise RuntimeError("stop after delivery")


class _Undeclared(_Declared):  # inherits a declaration it must not get
    calls = []


def _artifact(tmp_path, sizes=(3000, 5000), *, stray=True):
    root = tmp_path / "artifact"
    root.mkdir()
    weight_map = {}
    for index, size in enumerate(sizes):
        name = f"model-{index + 1:05d}-of-{len(sizes):05d}.safetensors"
        (root / name).write_bytes(b"\0" * size)
        weight_map[f"t{index}"] = name
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    if stray:  # never counted: not in the index
        (root / "model-00009-of-00009.safetensors").write_bytes(b"\0" * 10_000)
    return root


def test_artifact_weight_bytes_counts_the_indexed_shards_only(tmp_path):
    root = _artifact(tmp_path)
    assert artifact_weight_bytes(root) == 8000
    (root / "model.safetensors.index.json").unlink()
    assert artifact_weight_bytes(root) == 18000  # unindexed: every shard


def test_auto_decision_rule(tmp_path):
    root = _artifact(tmp_path)
    head = MIN_LANE_HEADROOM_BYTES

    def decide(adapter=_Declared, limit=head + 8000, explicit=False, path=root):
        return auto_expert_streaming_decision(
            adapter, path, admission_limit_bytes=limit, explicit=explicit
        )

    fits = decide()
    assert (fits["selected"], fits["reason"], fits["fits"]) == (False, "fits_resident", True)
    over = decide(limit=head + 7999)
    assert (over["selected"], over["reason"], over["fits"]) == (
        True, "resident_exceeds_admission", False)
    assert over["weight_bytes"] == 8000 and over["headroom_bytes"] == head
    explicit = decide(limit=head + 7999, explicit=True)
    assert (explicit["selected"], explicit["reason"], explicit["fits"]) == (
        False, "explicit_policy", False)
    assert decide(adapter=_Undeclared, limit=1)["reason"] == "undeclared"
    assert decide(limit=None)["reason"] == "admission_limit_unmeasured"
    assert decide(path=tmp_path / "absent", limit=1)["reason"] == "artifact_unmeasured"


def _engine(tmp_path, monkeypatch, factory, policy, *, limit, max_lanes=1, **kwargs):
    from mlx2 import serving

    monkeypatch.setattr(serving, "preload_admission_limit_bytes", lambda: limit)
    engine = serving.ServingEngine(
        str(_artifact(tmp_path) if not (tmp_path / "artifact").exists()
            else tmp_path / "artifact"),
        adapter_factory=factory, max_lanes=max_lanes, max_inflight=1,
        execution_policy=policy, **kwargs,
    )
    engine.thread.join(timeout=30)
    return engine


def test_engine_keeps_the_resident_path_when_the_artifact_fits(tmp_path, monkeypatch):
    _Declared.calls.clear()
    engine = _engine(tmp_path, monkeypatch, _Declared, None,
                     limit=MIN_LANE_HEADROOM_BYTES + 8000)
    try:
        assert not engine.weight_streaming_enabled()
        assert engine.weight_streaming_route is None
        assert "weight_streaming" not in _Declared.calls[0]
        assert engine.weight_streaming_auto["reason"] == "fits_resident"
        assert engine.moe_expert_streaming_policy["enabled"] is False
    finally:
        engine.close()


def test_engine_auto_selects_streaming_with_a_receipt(tmp_path, monkeypatch, caplog):
    _Declared.calls.clear()
    with caplog.at_level("WARNING", logger="mlx2.serving"):
        engine = _engine(tmp_path, monkeypatch, _Declared, None,
                         limit=MIN_LANE_HEADROOM_BYTES + 7000, lane_matmul="auto")
    try:
        assert engine.weight_streaming_enabled()
        assert engine.weight_streaming_route == "early"
        assert engine.moe_expert_streaming_policy["selection"] == "auto"
        assert engine.lane_matmul == "off"
        assert engine.weight_streaming_lane_matmul == "auto->off (weight streaming)"
        request = _Declared.calls[0]["weight_streaming"]
        assert (request.mode, request.budget_mode, request.budget_bytes) == (
            "moe_experts", "auto", 0)
        assert request.bypass_page_cache is True
        assert request.admission_limit_bytes == MIN_LANE_HEADROOM_BYTES + 7000
        status = engine.weight_streaming_status()
        assert status["selection"] == "auto"
        assert status["decision"]["reason"] == "resident_exceeds_admission"
        assert status["decision"]["weight_bytes"] == 8000
        assert any("auto-selected" in record.getMessage() for record in caplog.records)
    finally:
        engine.close()


def test_explicit_policy_wins_over_the_automatic_choice(tmp_path, monkeypatch):
    _Declared.calls.clear()
    engine = _engine(
        tmp_path, monkeypatch, _Declared,
        {"moe_expert_streaming": {"enabled": True, "cache_gib": 0.5}},
        limit=1 << 50,  # it would fit: the operator still streams
    )
    try:
        request = _Declared.calls[0]["weight_streaming"]
        assert request.budget_mode == "explicit" and request.budget_bytes == 1 << 29
        assert request.bypass_page_cache is False
        assert "selection" not in engine.moe_expert_streaming_policy
    finally:
        engine.close()


def test_explicitly_disabled_streaming_fails_closed_when_it_cannot_fit(tmp_path, monkeypatch):
    from mlx2 import serving

    root = _artifact(tmp_path)
    monkeypatch.setattr(serving, "preload_admission_limit_bytes",
                        lambda: MIN_LANE_HEADROOM_BYTES + 7000)
    with pytest.raises(ValueError, match="explicitly disabled"):
        serving.ServingEngine(
            str(root), adapter_factory=_Declared, max_lanes=1, max_inflight=1,
            execution_policy={"moe_expert_streaming": {"enabled": False}},
        )
    # Disabled and fitting: the ordinary resident path.
    _Declared.calls.clear()
    monkeypatch.setattr(serving, "preload_admission_limit_bytes", lambda: 1 << 50)
    engine = serving.ServingEngine(
        str(root), adapter_factory=_Declared, max_lanes=1, max_inflight=1,
        execution_policy={"moe_expert_streaming": {"enabled": False}},
    )
    engine.thread.join(timeout=30)
    try:
        assert not engine.weight_streaming_enabled()
        assert "weight_streaming" not in _Declared.calls[0]
    finally:
        engine.close()


def test_auto_selection_fails_closed_on_lanes_and_unevidenced_mechanisms(tmp_path, monkeypatch):
    from mlx2 import serving

    root = _artifact(tmp_path)
    monkeypatch.setattr(serving, "preload_admission_limit_bytes",
                        lambda: MIN_LANE_HEADROOM_BYTES + 7000)
    with pytest.raises(ValueError, match="max_lanes=1"):
        serving.ServingEngine(str(root), adapter_factory=_Declared, max_lanes=2,
                              max_inflight=2)
    with pytest.raises(ValueError, match="lane_matmul=exact"):
        serving.ServingEngine(str(root), adapter_factory=_Declared, max_lanes=1,
                              max_inflight=1, lane_matmul="exact")
    # An undeclared adapter is never auto-streamed (inherited declarations
    # do not count), so the engine starts its ordinary load.
    _Undeclared.calls.clear()
    engine = serving.ServingEngine(str(root), adapter_factory=_Undeclared, max_lanes=1,
                                   max_inflight=1)
    engine.thread.join(timeout=30)
    try:
        assert not engine.weight_streaming_enabled()
        assert "weight_streaming" not in _Undeclared.calls[0]
    finally:
        engine.close()


def test_server_serves_one_lane_when_streaming_will_be_auto_selected(tmp_path):
    from types import SimpleNamespace

    from mlx2.server import apply_auto_streaming_lane_default

    root = _artifact(tmp_path)
    resolution = SimpleNamespace(adapter_type=_Declared)
    small = MIN_LANE_HEADROOM_BYTES + 7000

    args = SimpleNamespace(model=str(root), max_lanes=16)
    decision = apply_auto_streaming_lane_default(args, resolution, None,
                                                 admission_limit_bytes=small)
    assert decision["selected"] and args.max_lanes == 1
    args = SimpleNamespace(model=str(root), max_lanes=4, max_lanes_explicit=True)
    apply_auto_streaming_lane_default(args, resolution, None, admission_limit_bytes=small)
    assert args.max_lanes == 4  # an operator's choice is kept (engine refuses)
    args = SimpleNamespace(model=str(root), max_lanes=16)
    assert apply_auto_streaming_lane_default(
        args, resolution, {"moe_expert_streaming": {"enabled": False}},
        admission_limit_bytes=small) is None
    assert args.max_lanes == 16
    args = SimpleNamespace(model=str(root), max_lanes=16)
    apply_auto_streaming_lane_default(args, resolution, None,
                                      admission_limit_bytes=1 << 50)
    assert args.max_lanes == 16


def test_bf16_per_expert_rows_page_in_byte_exact(tmp_path):
    """The bf16 release's rows reach the cache bit for bit (CPU-checkable part)."""
    root, weight_map, records = _laguna_checkpoint(tmp_path, mx.bfloat16)
    resident = _resident(_laguna(), root, weight_map)
    model = _laguna()
    loaded = _streamed(model, root, weight_map, records, top_k=2, budget=1 << 30)
    try:
        table = dict(tree_flatten(resident.parameters()))
        for path, cache in loaded.manager.caches.items():
            assert cache.reader.specs[0].expert_parts is not None  # per-expert
            rows = cache.acquire([7, 0])
            want = table[f"{path}.weight"]
            assert want.dtype == mx.bfloat16
            for expert in (0, 7):
                (got,) = rows[expert]
                assert got.dtype == mx.bfloat16
                assert np.array_equal(
                    np.array(got.view(mx.uint16)), np.array(want[expert].view(mx.uint16))
                )
    finally:
        loaded.manager.close()
