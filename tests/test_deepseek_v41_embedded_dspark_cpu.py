"""Synthetic checkpoint@cpu-fixture; no checkpoint loading/GPU/qualification."""

import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_deepseek_v41_cpu import Weights, config

from mlx2.adapters.deepseek_v41_dspark_artifact import (
    DSparkConfig,
    draft_config,
    dspark_admission_plan,
    tensor_specs,
)
from mlx2.adapters.deepseek_v41_dspark_external import (
    DeepSeekV41DraftCache,
    DeepSeekV41DSparkDraftModel,
    create_dspark_batch,
)
from mlx2.adapters.deepseek_v41_source_artifact import validate_router_precision
from mlx2.runtime.drafters.deepseek_v41_dspark import DeepSeekV41DSparkBackbone
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model


def artifact():
    from mlx2.runtime.models.deepseek_v41_config import DeepSeekV41Config

    c = DeepSeekV41Config()
    specs = tensor_specs(c)
    sizes = {"F32": 4, "BF16": 2, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
    rows = {
        n: {"dtype": d, "shape": s, "size": int(np.prod(s)) * sizes[d]}
        for n, (d, s) in specs.items()
    }
    t = {
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_markov_rank": 256,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    }
    return {
        "args": c,
        "config": {"text_config": t},
        "tensors": rows,
        "identity": {"fingerprint": "synthetic@metadata"},
    }


def test_full_published_tensor_inventory_and_reservation():
    a = artifact()
    p = dspark_admission_plan(a)
    assert p["tensor_count"] == 2401 and p["payload_bytes"] == 7932874632
    assert p["reservation_bytes"] >= p["payload_bytes"] + p["staging_bound_bytes"]
    # Full FP32 Markov coefficients alone exceed the old64MiB scratch bound.
    assert p["workspace_bound_bytes"] > (64 << 20) + 4 * 129280 * 256
    assert p["reservation_bytes"] == sum(p[key] for key in (
        "payload_bytes", "prepared_bound_bytes", "staging_bound_bytes", "workspace_bound_bytes"
    ))
    assert p["qualification"] == "unqualified"
    for change in ("dtype", "shape", "missing", "extra", "config"):
        b = copy.deepcopy(a)
        key = "mtp.0.ffn.gate.weight"
        if change == "dtype":
            b["tensors"][key]["dtype"] = "F8_E4M3"
        elif change == "shape":
            b["tensors"][key]["shape"] = (384, 5120)
        elif change == "missing":
            b["tensors"].pop(key)
        elif change == "extra":
            b["tensors"]["mtp.3.unknown"] = b["tensors"][key]
        else:
            b["config"]["text_config"]["dspark_target_layer_ids"] = [36, 38, 39]
        with pytest.raises(ValueError):
            dspark_admission_plan(b)


def test_router_keeps_original_precision_and_rejects_scales():
    a = artifact()
    c = a["args"]
    t = {}
    for i in range(40):
        for field, dtype, shape in [
            ("weight", "BF16", (384, 5120)),
            ("bias", "F32", (384,)),
            ("bias_vl", "F32", (384,)),
        ]:
            t[f"layers.{i}.ffn.gate.{field}"] = {"dtype": dtype, "shape": shape}
    assert validate_router_precision(t, c)["requantized"] is False
    validate_router_precision(a["tensors"], c, prefix="mtp", layers=3, experts=128)
    for name in ("weight", "bias", "bias_vl", "scale"):
        b = copy.deepcopy(t)
        key = "layers.39.ffn.gate." + name
        if name == "scale":
            b[key] = {"dtype": "F8_E8M0", "shape": (384, 160)}
        else:
            b[key]["dtype"] = "I8"
        with pytest.raises(ValueError, match="router"):
            validate_router_precision(b, c)


class DraftWeights(Weights):
    def __init__(self, c, target):
        super().__init__(c)
        self.target = target
        d = c.hidden_size
        v = c.vocab_size
        rng = np.random.default_rng(1741)
        for key, shape in [
            ("main_proj.weight", (d, 3 * d)),
            ("main_norm.weight", (d,)),
            ("markov_embed.weight", (v, 256)),
            ("markov_head.weight", (v, 256)),
            ("confidence.weight", (1, d + 256)),
        ]:
            self.data[key] = mx.array(rng.normal(0, 0.08, shape).astype("f4"))
        self.data["main_norm.weight"] = mx.ones((d,))

    def embedding(self, ids, name="token_embd.weight"):
        return (
            self.target.embedding(ids)
            if name == "token_embd.weight"
            else self.data[name][ids]
        )

    def linear(self, name):
        return (
            self.target.linear(name)
            if name == "output.weight"
            else super().linear(name)
        )


def pair():
    c = config()
    target = DeepSeekV41Model(c, Weights(c), identity="synthetic-dspark-target@cpu")
    dc = draft_config(c)
    w = DraftWeights(dc, target.weights)
    cfg = DSparkConfig(noise_token_id=18, target_layers=(2, 3, 4))
    backbone = DeepSeekV41DSparkBackbone(cfg, target, w, "synthetic-embedded-draft@cpu")
    return target, backbone, DeepSeekV41DSparkDraftModel(backbone).bind(target)


def test_real_draft_graph_uses_preattention_taps_and_read_only_proposals():
    target, b, d = pair()
    cache = target.make_cache()
    ids = mx.array([[1, 2, 3, 4]], mx.int32)
    _logits, taps = target.forward_with_taps(ids, cache, (2, 3, 4))
    draft = d.make_cache()
    d.append_context(taps, draft)
    before = draft[0].history.snapshot()
    tokens, law = d.draft_distributions([5], taps[:, :0], draft, 5, None, [0])
    assert len(tokens[0]) == len(law[0]) == 5
    assert all(0 <= t < 19 for t in tokens[0])
    after = draft[0].history.snapshot()
    assert after.next_position == before.next_position == 4
    for a, e in zip(
        (after.main_history, *after.stage_kv), (before.main_history, *before.stage_kv)
    ):
        np.testing.assert_array_equal(np.array(a), np.array(e))
    assert d.receipt_settings["target_feature_law"].startswith("pre_attention")
    assert d.receipt_settings["qualified"] is False
    with pytest.raises(ValueError):
        b.record_features({1: taps[:, :, :8]}, 4, history=draft[0].history)
    with pytest.raises(ValueError):
        d.draft_distributions([1], taps[:, :0], draft, 5, None, [0.5])


def test_draft_cache_roundtrip_crop_and_identity():
    target, _b, d = pair()
    cache = target.make_cache()
    _, taps = target.forward_with_taps(
        mx.array([[1, 2, 3]], mx.int32), cache, (2, 3, 4)
    )
    own = d.make_cache()[0]
    d.append_context(taps, [own])
    restored = DeepSeekV41DraftCache.from_state(own.state, own.meta_state)
    assert restored.meta_state == own.meta_state and restored.offset == 3
    for a, e in zip(restored.state, own.state):
        np.testing.assert_array_equal(np.array(a), np.array(e))
    restored.history.crop(2)
    assert own.offset == 3 and restored.offset == 2
    changed = list(own.meta_state)
    changed[0] = "old-v4"
    with pytest.raises(ValueError):
        DeepSeekV41DraftCache.from_state(own.state, changed)
    with pytest.raises(ValueError):
        create_dspark_batch(target, d, completion_batch_size=2)
    with pytest.raises(ValueError):
        create_dspark_batch(target, d, target_tree_executor_provider=object())
    generator = create_dspark_batch(target, d, num_draft=5, stop_tokens=set())
    for temperature in (0.8, float("nan"), True):
        with pytest.raises(ValueError, match="greedy"):
            generator.insert(
                [[1, 2]],
                max_tokens=[3],
                sampling_configs=[{"sampling_temp": temperature}],
            )
    assert generator.lanes == {}


def test_callable_learned_draft_loop_matches_ordinary_and_resumes():
    target, _b, d = pair()
    prompt = [1, 2, 3, 4, 5, 6, 7]
    cache = target.make_cache()
    logits = target(mx.array([prompt], mx.int32), cache)
    ordinary = []
    for _ in range(14):
        token = int(mx.argmax(logits[0, -1]).item())
        ordinary.append(token)
        logits = target(mx.array([[token]], mx.int32), cache)
    generator = create_dspark_batch(
        target,
        d,
        num_draft=5,
        stop_tokens=set(),
        prefill_step_size=3,
        cost_per_commit_handoff=False,
    )
    generator.insert([prompt], max_tokens=[14])
    output = []
    for _ in range(40):
        _, rows = generator.next()
        output.extend(int(r.token) for r in rows)
        if not generator.lanes:
            break
    assert output == ordinary
    assert generator.lanes == {}


def test_stage_matches_independent_numpy_shifted_hc_attention_oracle():
    _target, b, _ = pair()
    w, c = b.w, b.math.args
    rng = np.random.default_rng(441)
    h = rng.normal(0, 0.2, (1, 5, 4, c.hidden_size)).astype("f4")
    pre = rng.uniform(0.1, 0.9, (1, 5, 4)).astype("f4")
    support = rng.normal(0, 0.2, (1, 7, c.head_dim)).astype("f4")
    block = b.math.layers[0]

    def arr(n):
        return np.array(w.data["blk.0." + n])

    def linear(x, n):
        return x @ arr(n + ".weight").T

    def norm(x, n):
        return (
            x
            / np.sqrt((x * x).mean(-1, keepdims=True) + c.rms_norm_eps)
            * arr(n + ".weight")
        )

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def mixes(x, kind):
        flat = x.reshape(1, 5, -1)
        z = (flat @ arr("hc_" + kind + "_fn.weight").T) / np.sqrt(
            (flat * flat).mean(-1, keepdims=True) + c.rms_norm_eps
        )
        scale, base = (
            arr("hc_" + kind + "_scale.weight"),
            arr("hc_" + kind + "_base.weight"),
        )
        a = sigmoid(z[..., :4] * scale[0] + base[:4]) + c.hc_eps
        p = 2 * sigmoid(z[..., 4:8] * scale[1] + base[4:8])
        matrix = (z[..., 8:] * scale[2] + base[8:]).reshape(1, 5, 4, 4)
        matrix = np.exp(matrix - matrix.max(-1, keepdims=True))
        matrix = matrix / matrix.sum(-1, keepdims=True) + c.hc_eps
        matrix /= matrix.sum(-2, keepdims=True) + c.hc_eps
        for _ in range(c.hc_sinkhorn_iters - 1):
            matrix /= matrix.sum(-1, keepdims=True) + c.hc_eps
            matrix /= matrix.sum(-2, keepdims=True) + c.hc_eps
        return a, p, matrix

    def expand(y, res, p, comb):
        return p[..., None] * y[:, :, None] + np.swapaxes(comb, -1, -2) @ res

    def rotate(x, inverse=False):
        rd = c.qk_rope_head_dim
        phase = (
            np.arange(9, 14, dtype="f4")[:, None]
            * (10000.0 ** (-np.arange(0, rd, 2, dtype="f4") / rd))[None]
        )
        tail = x[..., -rd:].reshape(*x.shape[:-1], rd // 2, 2)
        cos, sin = np.cos(phase), np.sin(phase) * (-1 if inverse else 1)
        return np.concatenate(
            [
                x[..., :-rd],
                np.stack(
                    [
                        tail[..., 0] * cos - tail[..., 1] * sin,
                        tail[..., 0] * sin + tail[..., 1] * cos,
                    ],
                    -1,
                ).reshape(*x.shape[:-1], rd),
            ],
            -1,
        )

    attn_pre, post, comb = mixes(h, "attn")
    x = norm((h * pre[..., None]).sum(-2), "attn_norm")
    qr = norm(linear(x, "attn_q_a"), "attn_q_a_norm")
    q = rotate(
        linear(qr, "attn_q_b")
        .reshape(1, 5, c.num_attention_heads, c.head_dim)
        .transpose(0, 2, 1, 3)
    )
    kv = rotate(norm(linear(x, "attn_kv"), "attn_kv_a_norm")[:, None])[:, 0]
    kv = np.concatenate([support, kv], 1)
    scores = q @ np.swapaxes(kv[:, None], -1, -2) / np.sqrt(c.head_dim)
    sink = np.broadcast_to(
        arr("attn_sinks.weight")[None, :, None, None], (*scores.shape[:-1], 1)
    )
    scores = np.concatenate([scores, sink], -1)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    out = rotate(scores[..., :-1] @ kv[:, None], inverse=True)
    grouped = out.transpose(0, 2, 1, 3).reshape(1, 5, c.o_groups, -1)
    low = np.einsum(
        "btgi,goi->btgo",
        grouped,
        arr("attn_output_a.weight").reshape(c.o_groups, -1, grouped.shape[-1]),
    )
    h = expand(linear(low.reshape(1, 5, -1), "attn_output_b"), h, post, comb)
    ffn_pre, post, comb = mixes(h, "ffn")
    x = norm((h * attn_pre[..., None]).sum(-2), "ffn_norm")
    raw = np.sqrt(np.logaddexp(0, linear(x, "ffn_gate_inp")))
    ids = np.argsort(-(raw + arr("exp_probs_b.bias")), -1)[..., :3]
    weights = np.take_along_axis(raw, ids, -1)
    weights = weights / weights.sum(-1, keepdims=True) * c.routed_scaling_factor
    # The fixture provider's MoE is dense, permitting an independent NumPy sum.
    routed = np.zeros_like(x)
    for row in range(5):
        for slot, e in enumerate(ids[0, row]):
            gate = arr("ffn_gate_exps.weight")[e] @ x[0, row]
            up = arr("ffn_up_exps.weight")[e] @ x[0, row]
            routed[0, row] += weights[0, row, slot] * (
                arr("ffn_down_exps.weight")[e] @ (gate * sigmoid(gate) * up)
            )
    gate, up = linear(x, "ffn_gate_shexp"), linear(x, "ffn_up_shexp")
    shared = linear(gate * sigmoid(gate) * up, "ffn_down_shexp")
    expected = expand(routed + shared, h, post, comb)
    # Reconstruct the exact original independent input, not the evolved h.
    original_rng = np.random.default_rng(441)
    original = original_rng.normal(0, 0.2, (1, 5, 4, c.hidden_size)).astype("f4")
    actual, actual_pre = b._stage(
        block, mx.array(original), mx.array(pre), mx.array(support), 9
    )
    np.testing.assert_allclose(np.array(actual), expected, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(np.array(actual_pre), ffn_pre, rtol=3e-5, atol=2e-6)


def test_real_packed_source_provider_consumes_native_codes_and_has_closed_lifecycle(
    monkeypatch,
):
    from types import SimpleNamespace

    from mlx2.adapters import deepseek_v41_dspark_weights as provider

    a = artifact()
    c = config(
        hidden_size=32,
        head_dim=32,
        q_lora_rank=32,
        o_lora_rank=32,
        moe_intermediate_size=32,
        vocab_size=32,
    )
    a["args"] = c
    sizes = {"F32": 4, "BF16": 2, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
    specs = tensor_specs(c)
    a["tensors"] = {
        n: {
            "dtype": d,
            "shape": s,
            "size": int(np.prod(s)) * sizes[d],
            "file": "fixture.safetensors",
        }
        for n, (d, s) in specs.items()
    }
    a.update(root="synthetic", seals={"fixture.safetensors": {}})
    reads = []

    def read(_artifact, names):
        reads.extend(names)
        result = {}
        for name in names:
            dtype, shape = specs[name]
            dt = {
                "F32": "f4",
                "BF16": "u2",
                "F8_E4M3": "u1",
                "F8_E8M0": "u1",
                "I8": "u1",
            }[dtype]
            value = np.zeros(shape, dtype=dt)
            if dtype == "F8_E8M0":
                value.fill(127)
            if dtype == "I8":
                value.fill(0x22)  # native E2M1 +1 in both nibbles
            if "norm.weight" in name and dtype == "BF16":
                value.fill(0x3F80)
            result[name] = value
        return result

    monkeypatch.setattr(provider, "read_tensors", read)
    monkeypatch.setattr(provider, "checked_file", lambda *args: None)
    plan = dspark_admission_plan(a)
    with pytest.raises(MemoryError):
        provider.SourceDSparkWeights(a, object(), admitted_bytes=0)
    assert not reads
    owner = provider.SourceDSparkWeights(
        a, SimpleNamespace(), admitted_bytes=plan["reservation_bytes"]
    )
    assert len(reads) == 2401
    result = owner.linear("main_proj.weight")(mx.ones((1, 1, 96), mx.float32))
    np.testing.assert_array_equal(np.array(result), np.zeros((1, 1, 32)))
    routed = owner.moe(
        mx.ones((1, 32), mx.float32),
        mx.array([[0, 1, 2]], mx.uint32),
        mx.array([[0.2, 0.3, 0.5]]),
        "blk.0",
    )
    np.testing.assert_array_equal(np.array(routed), np.full((1, 32), 3136.0))
    owner.close()
    owner.close()
    with pytest.raises(ValueError, match="closed"):
        owner.linear("main_proj.weight")(mx.ones((1, 1, 96)))


def test_source_admission_reserves_embedded_payload_before_any_weight_load(monkeypatch):

    from mlx2 import process_env
    from mlx2.adapters import deepseek_v41_source as source
    from mlx2.runtime.streamed_load import WeightStreamRequest

    a = artifact()
    a["root"] = "fixture"
    adapter = object.__new__(source.DeepSeekV41SourceAdapter)
    adapter.kernel_policy = {"draft_model": "embedded_dspark"}
    adapter.weight_streaming_request = WeightStreamRequest(
        mode="moe_experts",
        budget_bytes=64 << 30,
        admission_limit_bytes=128 << 30,
        headroom_bytes=7 << 30,
    )
    monkeypatch.setattr(source, "inspect_artifact", lambda _: a)
    for name in (
        "require_process_numerics",
        "clear_inherited_profile",
        "assert_process_profile_applied",
    ):
        monkeypatch.setattr(process_env, name, lambda *a, **k: None)
    captured = {}

    from mlx2.adapters import deepseek_v41_source_auxiliary as auxiliary
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    monkeypatch.setattr(lfu, 'source_admission', lambda artifact, request, policy:
                        {"imminent_working_reserve_bytes": 2 * lfu.ORIGINAL_BYTES,
                             "admission_limit_bytes": request.admission_limit_bytes})

    def refuse(admission, plans):
        captured['plans'] = plans
        captured['imminent'] = admission['imminent_working_reserve_bytes']
        raise MemoryError('fixture refuses before materialization')

    monkeypatch.setattr(auxiliary, 'stage_auxiliary_plans', refuse)
    with pytest.raises(MemoryError, match='before materialization'):
        adapter._load('fixture')
    assert captured['plans']['dspark'] == dspark_admission_plan(a)
    assert captured['imminent'] == 2 * lfu.ORIGINAL_BYTES


def test_source_close_preserves_target_when_draft_fence_fails(monkeypatch):
    from types import SimpleNamespace

    from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter

    events = []
    adapter = object.__new__(DeepSeekV41SourceAdapter)

    def fail():
        events.append("draft-fence")
        raise RuntimeError("fixture uncertain command")

    draft = SimpleNamespace(close=fail)
    adapter.draft_weights = draft
    adapter.vision_weights = None
    monkeypatch.setattr(
        DeepSeekV41MLXAdapter, "close", lambda self: events.append("target-close")
    )
    with pytest.raises(RuntimeError, match="uncertain"):
        adapter.close()
    assert adapter.draft_weights is draft and events == ["draft-fence"]
    draft.close = lambda: events.append("draft-close")
    adapter.close()
    assert adapter.draft_weights is None and events == [
        "draft-fence",
        "draft-close",
        "target-close",
    ]


def test_configure_refuses_missing_preallocation_reserve_without_mutation():
    from types import SimpleNamespace

    from mlx2.adapters.deepseek_v41_dspark import configure_source_dspark

    identity = {"fingerprint": "derivative@synthetic"}
    a = SimpleNamespace(
        source_artifact=artifact(),
        kernel_policy={"num_draft": 5},
        native_admission={},
        identity=identity,
    )
    with pytest.raises(MemoryError, match="before target admission"):
        configure_source_dspark(a)
    assert a.identity is identity and not hasattr(a, "draft_model")


def test_factory_retains_early_owner_pending_upload_and_target_until_cleanup_fence(monkeypatch):
    from types import SimpleNamespace

    from mlx2.adapters import deepseek_v41_dspark as factory
    from mlx2.adapters import deepseek_v41_dspark_weights as provider
    from mlx2.adapters import deepseek_v41_mlx_weights as native
    from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    artifact = {'tensors': {'mtp.0.main_norm.weight': {'dtype': 'BF16', 'shape': (8,)}},
                'root': 'synthetic', 'seals': {}}
    plan = {'reservation_bytes': 1, 'binding': 'synthetic'}
    monkeypatch.setattr(factory, 'dspark_admission_plan', lambda artifact: plan)
    monkeypatch.setattr(provider, 'dspark_admission_plan', lambda artifact: plan)
    monkeypatch.setattr(provider, 'aliases', lambda: {'main_norm.weight': 'mtp.0.main_norm.weight'})
    host = np.ones((8,), np.uint16)
    monkeypatch.setattr(provider, 'read_tensors', lambda artifact, names: {name: host for name in names})
    adapter = object.__new__(DeepSeekV41SourceAdapter)
    target_owner = SimpleNamespace()
    adapter.model = SimpleNamespace(weights=target_owner)
    adapter.source_artifact = artifact
    adapter.native_admission = {'dspark_reservation_bytes': 1}
    adapter.kernel_policy = {'num_draft': 1}
    adapter.draft_model = adapter.draft_weights = adapter.vision_weights = None
    events = []
    monkeypatch.setattr(DeepSeekV41MLXAdapter, 'close', lambda self: events.append('target-close'))

    def failed_upload(*args):
        owner = adapter.draft_weights
        assert owner is not None and owner.target is target_owner
        assert owner._pending_host['mtp.0.main_norm.weight'] is host
        assert owner._pending_arrays
        raise RuntimeError('fixture upload failed')

    monkeypatch.setattr(mx, 'eval', failed_upload)
    monkeypatch.setattr(mx, 'synchronize', lambda: (_ for _ in ()).throw(RuntimeError('fixture uncertain fence')))
    owner = None
    try:
        with pytest.raises(RuntimeError, match='upload failed') as error:
            factory.configure_source_dspark(adapter)
        assert any('cleanup fence failure' in note for note in error.value.__notes__)
        owner = adapter.draft_weights
        assert native._UPLOAD_QUARANTINE[id(owner)] is owner
        assert owner.target is target_owner and owner._pending_host is not None and owner._pending_arrays
        with pytest.raises(RuntimeError, match='uncertain fence'):
            adapter.close()
        assert not events and adapter.draft_weights is owner
        monkeypatch.setattr(mx, 'synchronize', lambda: None)
        adapter.close()
        assert events == ['target-close'] and adapter.draft_weights is None
        assert id(owner) not in native._UPLOAD_QUARANTINE
        assert owner._closed and not owner._pending_arrays and owner._pending_host is None
    finally:
        monkeypatch.setattr(mx, 'synchronize', lambda: None)
        if owner is not None:
            owner.close()


def test_orphaned_constructor_failure_quarantines_host_device_and_borrowed_owner(monkeypatch):
    from mlx2.adapters import deepseek_v41_dspark_weights as provider
    from mlx2.adapters import deepseek_v41_mlx_weights as native
    artifact = {'tensors': {'mtp.0.main_norm.weight': {'dtype': 'BF16', 'shape': (8,)}}}
    monkeypatch.setattr(provider, 'dspark_admission_plan', lambda artifact: {'reservation_bytes': 1})
    monkeypatch.setattr(provider, 'aliases', lambda: {'main_norm.weight': 'mtp.0.main_norm.weight'})
    host = np.ones((8,), np.uint16)
    target = object()
    monkeypatch.setattr(provider, 'read_tensors', lambda artifact, names: {name: host for name in names})
    monkeypatch.setattr(mx, 'eval', lambda *args: (_ for _ in ()).throw(RuntimeError('fixture constructor upload')))
    monkeypatch.setattr(mx, 'synchronize', lambda: (_ for _ in ()).throw(RuntimeError('fixture fence')))
    before = set(native._UPLOAD_QUARANTINE)
    owner = None
    try:
        with pytest.raises(RuntimeError, match='constructor upload'):
            provider.SourceDSparkWeights(artifact, target, admitted_bytes=1)
        added = set(native._UPLOAD_QUARANTINE) - before
        assert len(added) == 1
        owner = native._UPLOAD_QUARANTINE[next(iter(added))]
        assert owner.target is target and owner._pending_arrays
        assert owner._pending_host['mtp.0.main_norm.weight'] is host
        with pytest.raises(RuntimeError, match='retain failed-completion'):
            provider.SourceDSparkWeights(artifact, target, admitted_bytes=1, load=False)
    finally:
        monkeypatch.setattr(mx, 'synchronize', lambda: None)
        if owner is not None:
            owner.close()


def test_confidence_receipt_matches_empty_prefix_when_first_proposal_is_low(monkeypatch):
    target, backbone, _ = pair()
    original = backbone.w.linear
    monkeypatch.setattr(backbone.w, 'linear', lambda name: (lambda x: mx.full((x.shape[0], 1), -100.0)) if name == 'confidence.weight' else original(name))
    draft = DeepSeekV41DSparkDraftModel(backbone, confidence_threshold=.5).bind(target)
    cache = target.make_cache()
    _, taps = target.forward_with_taps(mx.array([[1, 2, 3]], mx.int32), cache, (2, 3, 4))
    state = draft.make_cache()
    tokens, laws = draft.draft_distributions([4], taps, state, 5, None, [0])
    assert tokens == [[]] and laws == [[]]
    assert draft.adaptive_confidence_features == [[-100.0]]
    assert draft.receipt_settings['confidence_law'] == 'sigmoid_prenorm_markov_strict_less_retained_prefix'


def test_committed_support_crossing_retained_window_matches_irregular_chunks():
    """A long append replaces only the trailing support, at absolute positions."""
    _target, _backbone, draft = pair()
    rng = np.random.default_rng(846)
    features = mx.array(rng.normal(0, .2, (1, 401, 24)).astype(np.float32))
    whole, chunks = draft.make_cache(), draft.make_cache()
    draft.append_context(features, whole)
    first = 0
    for length in (1, 126, 2, 129, 7, 136):
        draft.append_context(features[:, first:first + length], chunks)
        first += length
    assert first == 401
    assert whole[0].history.start_position == chunks[0].history.start_position == 273
    assert whole[0].offset == chunks[0].offset == 401
    assert whole[0].meta_state == chunks[0].meta_state
    for actual, expected in zip(whole[0].state, chunks[0].state):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)
    whole_tokens, _ = draft.draft_distributions([7], features[:, :0], whole, 5, None, [0])
    chunk_tokens, _ = draft.draft_distributions([7], features[:, :0], chunks, 5, None, [0])
    assert whole_tokens == chunk_tokens
    assert whole[0].offset == chunks[0].offset == 401


def test_long_window_learned_draft_compound_apc_hit_resumes_ordinary_tokens():
    """Exercise the learned graph and the real compound APCv2 state together."""
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    target, _backbone, draft = pair()
    prompt = [i % target.args.vocab_size for i in range(141)]

    def generator():
        return create_dspark_batch(
            target, draft, num_draft=5, stop_tokens=set(),
            prefill_step_size=37, cost_per_commit_handoff=False,
        )

    def drain(current):
        output, final = [], None
        for _ in range(100):
            _, rows = current.next()
            for row in rows:
                output.append(int(row.token))
                if row.finish_reason is not None:
                    final = row
            if not current.lanes:
                break
        assert final is not None and not current.lanes
        return output, final

    first, second = generator(), generator()
    apc = APCv2(max_size=2, layout_name="synthetic-dspark-long-window@cpu")
    hit = None
    try:
        first.insert([prompt], max_tokens=[9])
        first_output, final = drain(first)
        key = APCKey(
            "synthetic-dspark-target@cpu", revision=target.identity,
            cache_layout_fingerprint="synthetic-dspark-long-window@cpu",
        )
        apc.store(key, final.all_tokens, final.prompt_cache, sidecar=final.cache_sidecar)
        hit = apc.lookup(key, final.all_tokens + [final.token])
        assert hit.hit and hit.hit_kind == "external_draft_sidecar"
        hit.sidecar.validate(draft.execution_binding, len(final.all_tokens))
        saved = tuple(plane.meta_state for plane in hit.sidecar.state[0])
        second.insert(
            [[int(final.token)]], max_tokens=[11], caches=[hit.cache],
            all_tokens=[final.all_tokens], cache_states=[hit.sidecar],
        )
        second_output, _ = drain(second)
        assert second.scheduler_stats["paired_cache_resumes"] == 1
        assert tuple(plane.meta_state for plane in hit.sidecar.state[0]) == saved
        control_cache = target.make_cache()
        logits = target(mx.array([prompt], mx.int32), control_cache)
        expected = []
        for _ in range(20):
            token = int(mx.argmax(logits[0, -1]).item())
            expected.append(token)
            logits = target(mx.array([[token]], mx.int32), control_cache)
        assert first_output + second_output == expected
    finally:
        first.close()
        second.close()
        if hit is not None and hit.hit:
            hit.cache.close()
        apc.clear(release_memory=False)
