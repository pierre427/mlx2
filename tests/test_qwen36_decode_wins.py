"""CPU admission/dispatch contracts; no claim of Metal exactness."""

import mlx.core as mx
import pytest

from mlx2.adapters import qwen36_35b as A
from mlx2.qualification import unqualifiable_candidate
from mlx2.runtime.models import qwen4_fused_gdn_verify as V
from mlx2.runtime.models import qwen4_moe_window as W
from mlx2.runtime.models import qwen4_routed_decode as RD
from mlx2.runtime.models import qwen36_35b as Q


@pytest.mark.parametrize("ne,k", [(256, 8), (512, 10)])
def test_router_launch_geometry(monkeypatch, ne, k):
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    assert W.admit_router_topk(mx.zeros((1, ne), mx.bfloat16), k, True) is None
    calls = []

    def kernel(name):
        def run(**kw):
            calls.append(kw)
            return [
                mx.zeros(s, d) for s, d in zip(kw["output_shapes"], kw["output_dtypes"])
            ]

        return run

    monkeypatch.setattr(W, "_kernel", kernel)
    out = W.router_topk(mx.zeros((1, ne), mx.bfloat16), top_k=k)
    assert out[0].shape == (1, k)
    assert dict(calls[0]["template"])["TOPK"] == k


@pytest.mark.parametrize("ne,k", [(512, 8), (256, 10), (256, 6)])
def test_router_launch_refuses_mixed_geometry(monkeypatch, ne, k):
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    reason = W.admit_router_topk(mx.zeros((1, ne), mx.bfloat16), k, True)
    assert reason is not None and f"E{ne}/top-{k}" in reason


def operands(rows, steps, hv=32):
    vd = hv * 128
    cd = 4096 + vd
    return dict(
        qkv=mx.zeros((rows, steps, cd), mx.bfloat16),
        z=mx.zeros((rows, steps, vd), mx.bfloat16),
        b=mx.zeros((rows, steps, hv), mx.bfloat16),
        a=mx.zeros((rows, steps, hv), mx.bfloat16),
        conv_state=mx.zeros((rows, 3, cd), mx.bfloat16),
        recurrent_state=mx.zeros((rows, hv, 128, 128)),
        conv_weight=mx.zeros((cd, 4, 1), mx.bfloat16),
        A_log=mx.zeros((hv,)),
        dt_bias=mx.zeros((hv,), mx.bfloat16),
        norm_weight=mx.ones((128,), mx.bfloat16),
    )


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_verify_admits_qwen35_math(rows):
    fn = (
        V.admit_qwen4_fused_gdn_verify
        if rows == 1
        else V.admit_qwen4_fused_gdn_batch_verify
    )
    assert fn(
        **operands(rows, 3),
        mask=None,
        spans=(),
        speculating=True,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="swish",
        architecture="qwen35",
    ).accepted


def test_adapter_new_selections_explicit_default_off(monkeypatch):
    names = {
        "moe_routed_decode": "MLX_QWEN4_MOE_ROUTED_DECODE",
        "moe_topk_fold": "MLX_QWEN4_MOE_TOPK_FOLD",
        "fused_gdn_batch_decode": "MLX_QWEN36_FUSED_GDN_BATCH_DECODE",
        "fused_gdn_verify": "MLX_QWEN36_FUSED_GDN_VERIFY",
        "fused_gdn_batch_verify": "MLX_QWEN36_FUSED_GDN_BATCH_VERIFY",
        "moe_window": "MLX_QWEN36_MOE_WINDOW",
    }
    for name in names.values():
        monkeypatch.setenv(name, "1")
    profile = A.configure_environment()
    assert all(n not in profile for n in names.values())
    profile = A.configure_environment(
        {
            "moe_routed_decode": "gate_up",
            "moe_topk_fold": "launch",
            "fused_gdn_batch_decode": True,
            "fused_gdn_verify": True,
            "fused_gdn_batch_verify": True,
            "moe_window": True,
        }
    )
    assert profile[names["moe_routed_decode"]] == "gate_up"
    assert profile[names["moe_topk_fold"]] == "launch"
    assert profile[names["moe_window"]] == "1"


def test_gdn_batch_switch_exists():
    assert hasattr(Q.GatedDeltaNet, "set_fused_gdn_batch_decode_mode")
    assert hasattr(Q.GatedDeltaNet, "set_fused_gdn_verify_mode")
    assert hasattr(Q.GatedDeltaNet, "set_fused_gdn_batch_verify_mode")


@pytest.mark.parametrize(
    "rows,hv,architecture",
    [(1, 32, "qwen35"), (4, 32, "qwen35"), (1, 48, "qwen4"), (16, 48, "qwen4")],
)
def test_verify_templates_preserve_flash_defaults(monkeypatch, rows, hv, architecture):
    calls = []

    def factory(*args):
        def kernel(**kw):
            calls.append(kw)
            return [
                mx.zeros(s, d) for s, d in zip(kw["output_shapes"], kw["output_dtypes"])
            ]

        return kernel

    monkeypatch.setattr(V, "_kernel", factory)
    monkeypatch.setattr(V, "_batch_kernel", factory)
    o = operands(rows, 3, hv)
    args = [
        o[k]
        for k in (
            "qkv",
            "z",
            "b",
            "a",
            "conv_state",
            "conv_weight",
            "A_log",
            "dt_bias",
            "recurrent_state",
            "norm_weight",
        )
    ] + [1e-6]
    fn = V.qwen4_fused_gdn_verify if rows == 1 else V.qwen4_fused_gdn_batch_verify
    if rows > 1:
        args.append((3,) * rows)
    out = fn(*args, threadgroup_y=4, architecture=architecture, num_value_heads=hv)
    template = dict(calls[0]["template"])
    assert template["HV"] == hv and template["RATIO"] == hv // 16
    assert ("AGNES_NUMERICS" in template) == (architecture == "qwen35")
    assert out[0].shape == (rows, 3, hv * 128)
    assert out[3].shape == (rows, 2, hv, 128, 128)


def _gdn_stub(rows=4):
    """No model construction: just the dispatch seam's resident operands."""
    from types import SimpleNamespace

    from mlx import nn

    from mlx2.runtime.models.cache import ArraysCache

    layer = Q.GatedDeltaNet.__new__(Q.GatedDeltaNet)
    nn.Module.__init__(layer)
    layer.sharding_group = None
    layer.num_k_heads = 16
    layer.num_v_heads = 32
    layer.head_k_dim = layer.head_v_dim = 128
    layer.conv_kernel_size = 4
    o = operands(rows, 3)
    layer.conv1d = SimpleNamespace(weight=o["conv_weight"])
    layer.A_log = o["A_log"]
    layer.dt_bias = o["dt_bias"]
    layer.norm = SimpleNamespace(weight=o["norm_weight"], eps=1e-6)
    layer.out_proj = lambda out: out
    for choice in ("decode", "batch_decode", "verify", "batch_verify"):
        setattr(
            layer,
            "fused_gdn_" + choice + "_mode",
            "off" if choice != "decode" else "stock",
        )
        for key in ("calls", "fallbacks"):
            setattr(layer, "fused_gdn_" + choice + "_" + key, 0)
        setattr(layer, "fused_gdn_" + choice + "_last_fallback", None)
        object.__setattr__(layer, "fused_gdn_" + choice + "_fallback_reasons", {})
    layer.eval()
    cache = ArraysCache(size=2)
    cache[0] = o["conv_state"]
    cache[1] = o["recurrent_state"]
    cache.start_speculation()
    return layer, cache, o


def test_cpu_verify_decline_preserves_cache_and_counts(monkeypatch):
    layer, cache, o = _gdn_stub()
    # The live entries themselves: ``cache.state`` is a (list, left_padding,
    # lengths) view that builds fresh empty arrays on every read.
    before = [cache[0], cache[1]]
    layer.set_fused_gdn_batch_verify_mode("row_exact")
    monkeypatch.setattr(Q, "fused_gdn_runtime_supported", lambda: False)
    assert (
        layer._try_fused_decode(o["qkv"], o["z"], o["b"], o["a"], None, cache) is None
    )
    assert cache[0] is before[0] and cache[1] is before[1]
    assert not cache._rollbacks
    assert layer.fused_gdn_batch_verify_calls == 0
    assert layer.fused_gdn_batch_verify_fallback_reasons == {
        "Metal runtime unavailable": 1
    }


def test_verify_snapshot_rollback_zero_partial_and_full(monkeypatch):
    layer, cache, o = _gdn_stub()
    # Hold the pre-forward entries, not ``cache.state``: its first element is
    # the live list, which the forward and the rewind overwrite in place.
    initial = [cache[0], cache[1]]
    layer.set_fused_gdn_batch_verify_mode("row_exact")
    monkeypatch.setattr(Q, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(Q, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(Q, "probe_qwen36_gdn", lambda *a, **k: 4)

    def kernel(*args, **kw):
        out = mx.zeros((4, 3, 4096), mx.bfloat16)
        conv = mx.full((4, 3, 8192), 3, mx.bfloat16)
        state = mx.full((4, 32, 128, 128), 3, mx.float32)
        cs = mx.stack([mx.full(conv.shape, n, mx.bfloat16) for n in (1, 2)], axis=1)
        ss = mx.stack([mx.full(state.shape, n, mx.float32) for n in (1, 2)], axis=1)
        return out, conv, state, ss, cs

    monkeypatch.setattr(V, "qwen4_fused_gdn_batch_verify", kernel)
    got = layer._try_fused_decode(o["qkv"], o["z"], o["b"], o["a"], None, cache)
    assert got.shape == (4, 3, 4096)
    cache.trim_ragged([3, 2, 1, 0])
    for j in (0, 1):
        assert mx.array_equal(cache[j][0], initial[j][0]).item()
        for row in (1, 2, 3):
            assert mx.all(cache[j][row] == row).item()
    assert layer.fused_gdn_batch_verify_calls == 1


def test_new_routes_cannot_be_qualified():
    profile = A.configure_environment({"fused_gdn_verify": True})
    assert "pending real-weight" in unqualifiable_candidate({"environment": profile})


@pytest.mark.parametrize("bits,gs", [(4, 64), (4, 128), (8, 64), (8, 128)])
def test_qwen36_shared_admits_fast_down_formats(bits, gs):
    import mlx.nn as nn

    from mlx2.runtime.models import qwen36_moe_decode as M
    from mlx2.runtime.models.qwen3_next import Qwen3NextMLP

    shared = Qwen3NextMLP(2048, 512)
    nn.quantize(shared, bits=bits, group_size=gs)
    shared.set_dtype(mx.bfloat16)
    shared.eval()
    assert M.shared_admission(shared, 2048, 512) is None
    # Flash-Next's plain-qmv shared fold must still refuse this geometry.
    assert not RD.admit_shared_fold(shared, 2048, 512).accepted


def test_split_gate_up_admission_serves_exact_top8_geometry():
    from mlx import nn

    from mlx2.runtime.models.qwen3_next import _ShapeOnly
    from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear

    def table(experts, out_dims, packed):
        # Resident tables are module parameters (real arrays), as a loaded
        # checkpoint holds them; they stay lazy because admission reads only
        # shape and dtype, so the full E256 geometry costs no memory.
        layer = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
        nn.Module.__init__(layer)
        layer.bits = 4
        layer.group_size = 64
        layer.mode = "affine"
        layer.weight = mx.zeros((experts, out_dims, packed), mx.uint32)
        groups = packed * 8 // 64
        layer.scales = mx.zeros((experts, out_dims, groups), mx.bfloat16)
        layer.biases = mx.zeros((experts, out_dims, groups), mx.bfloat16)
        layer.eval()
        return layer

    def tables(experts):
        return (
            table(experts, 512, 256),
            table(experts, 512, 256),
            table(experts, 2048, 64),
        )

    x = _ShapeOnly((1, 1, 2048), mx.bfloat16)
    inds = _ShapeOnly((1, 8), mx.uint32)
    scores = _ShapeOnly((1, 8), mx.bfloat16)
    ok = RD.admit_split_routed_decode(x, inds, scores, *tables(256))
    assert ok.accepted, ok.reason
    # A consistent E128 top-8 model is not the Qwen3.6 geometry: the width-512
    # intermediate exception must not silently widen to it.
    other = RD.admit_split_routed_decode(x, inds, scores, *tables(128))
    assert not other.accepted and "intermediate" in other.reason
    # Nor does the exception reach a top-10 (Flash-Next-shaped) launch.
    ten = _ShapeOnly((1, 10), mx.uint32)
    wide = RD.admit_split_routed_decode(
        x, ten, _ShapeOnly((1, 10), mx.bfloat16), *tables(256)
    )
    assert not wide.accepted and "intermediate" in wide.reason


def _qwen36_block(monkeypatch):
    """E256/top-8 block with the Metal halves replaced by composed stand-ins."""
    from types import SimpleNamespace

    from mlx import nn

    from mlx2.runtime.models import qwen36_moe_decode as M

    from mlx2.runtime.models import qwen3_next as QN

    # Split gate/up tables, as the Qwen3.6 artifacts hold them.
    monkeypatch.setattr(QN, "_MOE_FUSED_GATE_UP", False)
    hidden, inter = 64, 32
    args = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter,
        norm_topk_prob=True,
        num_experts=256,
        num_experts_per_tok=8,
    )
    block = M.Qwen36SparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    block.eval()
    monkeypatch.setattr(
        RD,
        "admit_split_routed_decode",
        lambda *a: RD.RoutedDecodeAdmission(True, "eligible"),
    )
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    monkeypatch.setattr(RD, "served_swiglu_refusal", lambda: None)
    monkeypatch.setattr(
        M,
        "routed_rows",
        lambda x, *a, **k: mx.zeros((x.size // x.shape[-1], x.shape[-1]), x.dtype),
    )
    launches = []

    def router_topk(logits, *, top_k):
        launches.append(logits.dtype)
        rows = logits.size // logits.shape[-1]
        return (
            mx.zeros((rows, top_k), mx.uint32),
            mx.zeros((rows, top_k), logits.dtype),
        )

    monkeypatch.setattr(W, "router_topk", router_topk)
    # The adapter profile pins MLX_QWEN4_FUSED_EXPERT_KERNEL=stock.
    block.set_fused_expert_kernel_mode("stock")
    block.set_moe_routed_decode_mode("gate_up_down")
    block.set_moe_topk_mode("launch")
    return block, launches, nn


def test_qwen36_topk_launch_runs_only_admitted_router_logits(monkeypatch):
    block, launches, nn = _qwen36_block(monkeypatch)
    x = mx.zeros((1, 1, 64), mx.bfloat16)
    block(x)
    assert block.qwen36_decode_calls == 1, block.qwen36_decode_last_fallback
    assert launches == [mx.bfloat16]
    assert block.moe_topk_calls["launch"] == 1 and block.moe_topk_fallbacks == 0
    # float32 router logits: the launch transcribes bf16 stock routing only,
    # so it must decline, count, and route through the stock ops instead.
    block.gate = nn.Linear(64, 256, bias=False)
    block(x)
    assert launches == [mx.bfloat16]
    assert block.moe_topk_calls["launch"] == 1
    assert block.moe_topk_fallbacks == 1
    assert "bfloat16" in block.moe_topk_last_fallback
    assert block.qwen36_decode_calls == 2
