"""CPU admission/dispatch contracts; no claim of Metal exactness."""

import mlx.core as mx
import pytest

from mlx2.adapters import qwen36_35b as A
from mlx2.qualification import unqualifiable_candidate
from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.models import qwen4_fused_gdn_verify as V
from mlx2.runtime.models import qwen4_moe_window as W
from mlx2.runtime.models import qwen4_routed_decode as RD
from mlx2.runtime.models import qwen36_35b as Q
from mlx2.runtime.models.moe_routing_contract import (
    QWEN4_E512_K10_ROUTING,
    QWEN36_E256_K8_ROUTING,
)


@pytest.mark.parametrize("ne,k", [(256, 8), (512, 10)])
def test_router_launch_geometry(monkeypatch, ne, k):
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    assert W.admit_router_topk(
        mx.zeros((1, ne), mx.bfloat16), k, True,
        contract=(QWEN36_E256_K8_ROUTING if k == 8 else QWEN4_E512_K10_ROUTING),
    ) is None
    calls = []

    def kernel(name):
        def run(**kw):
            calls.append(kw)
            return [
                mx.zeros(s, d) for s, d in zip(kw["output_shapes"], kw["output_dtypes"])
            ]

        return run

    monkeypatch.setattr(W, "_kernel", kernel)
    out = W.router_topk(
        mx.zeros((1, ne), mx.bfloat16),
        top_k=k,
        contract=(
            QWEN36_E256_K8_ROUTING if k == 8 else QWEN4_E512_K10_ROUTING
        ),
    )
    assert out[0].shape == (1, k)
    assert dict(calls[0]["template"])["TOPK"] == k


@pytest.mark.parametrize("ne,k", [(512, 8), (256, 10), (256, 6)])
def test_router_launch_refuses_mixed_geometry(monkeypatch, ne, k):
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    reason = W.admit_router_topk(
        mx.zeros((1, ne), mx.bfloat16), k, True,
        contract=(QWEN36_E256_K8_ROUTING if k == 8 else QWEN4_E512_K10_ROUTING),
    )
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
        "moe_window_batch_decode": "MLX_QWEN36_MOE_WINDOW_BATCH_DECODE",
        "moe_window_verify": "MLX_QWEN36_MOE_WINDOW_VERIFY",
        "moe_window_row_exact": "MLX_QWEN36_MOE_WINDOW_ROW_EXACT",
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


def test_adapter_moe_window_consumers_are_independent():
    assert A.moe_window_consumers({"moe_window_batch_decode": True}) == (
        "batch_decode",
    )
    assert A.moe_window_consumers({"moe_window_verify": True}) == ("verify",)
    assert A.moe_window_consumers({"moe_window_row_exact": True}) == ("row_exact",)
    assert A.moe_window_consumers(
        {
            "moe_window_batch_decode": True,
            "moe_window_verify": True,
            "moe_window_row_exact": True,
        }
    ) == ("batch_decode", "verify", "row_exact")
    # The compatibility umbrella retains its historical all-consumer meaning.
    assert A.moe_window_consumers({"moe_window": True}) == (
        "batch_decode",
        "verify",
        "row_exact",
    )
    with pytest.raises(ValueError, match="cannot be mixed"):
        A.configure_environment(
            {"moe_window": True, "moe_window_row_exact": True}
        )


def test_adapter_independent_moe_window_environment_is_explicit_only(monkeypatch):
    names = {
        "moe_window_batch_decode": "MLX_QWEN36_MOE_WINDOW_BATCH_DECODE",
        "moe_window_verify": "MLX_QWEN36_MOE_WINDOW_VERIFY",
        "moe_window_row_exact": "MLX_QWEN36_MOE_WINDOW_ROW_EXACT",
    }
    for name in names.values():
        monkeypatch.setenv(name, "1")
    assert all(name not in A.configure_environment() for name in names.values())
    profile = A.configure_environment(
        {
            "moe_window_batch_decode": True,
            "moe_window_verify": True,
            "moe_window_row_exact": True,
        }
    )
    assert all(profile[name] == "1" for name in names.values())
    assert profile["MLX_QWEN36_DECODE_WINS"] == "1"


def test_qwen36_live_import_exemptions_cover_both_process_global_setters():
    assert {
        "MLX2_MOE_NAX_GATHER",
        "MLX2_MOE_RHS_PAD_POLICY",
    } <= A.LIVE_APPLIED_ENV


def test_numerical_contract_binds_ambient_fused_gdn_from_live_layers(monkeypatch):
    from types import SimpleNamespace

    class FakeGDN:
        def __init__(self, mode):
            self.fused_gdn_decode_mode = mode

    monkeypatch.setattr(Q, "GatedDeltaNet", FakeGDN)
    adapter = object.__new__(A.Qwen3635BA3BAdapter)
    adapter._kernels = {"moe_topk_fold": "launch"}
    adapter.moe_nax_gather = "fused"
    adapter.model = SimpleNamespace(
        named_modules=lambda: (
            ("gdn0", FakeGDN("stock")),
            ("gdn1", FakeGDN("stock")),
        )
    )
    contract = adapter.execution_numerics_contract()["qwen36_target"]
    assert contract["decode_kernels"] == {
        "fused_gdn_decode": False,
        "moe_topk_fold": "launch",
    }

    adapter.model = SimpleNamespace(
        named_modules=lambda: (("gdn0", FakeGDN("fused")),)
    )
    assert adapter.execution_numerics_contract()["qwen36_target"][
        "decode_kernels"
    ]["fused_gdn_decode"] is True


def test_numerical_contract_refuses_declared_live_fused_gdn_disagreement(monkeypatch):
    from types import SimpleNamespace

    class FakeGDN:
        fused_gdn_decode_mode = "stock"

    monkeypatch.setattr(Q, "GatedDeltaNet", FakeGDN)
    adapter = object.__new__(A.Qwen3635BA3BAdapter)
    adapter._kernels = {"fused_gdn_decode": True}
    adapter.moe_nax_gather = "off"
    adapter.model = SimpleNamespace(named_modules=lambda: (("gdn", FakeGDN()),))
    with pytest.raises(ValueError, match="disagrees with live layers"):
        adapter.execution_numerics_contract()


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


def test_row_exact_canary_does_not_publish_fallback_counters(monkeypatch):
    layer, cache, o = _gdn_stub()
    layer.set_fused_gdn_batch_verify_mode("row_exact")
    monkeypatch.setattr(Q, "fused_gdn_runtime_supported", lambda: False)
    assert (
        layer.row_exact_probe_fused_decode(
            o["qkv"], o["z"], o["b"], o["a"], None, cache
        )
        is None
    )
    assert layer.fused_gdn_batch_verify_calls == 0
    assert layer.fused_gdn_batch_verify_fallbacks == 0
    assert layer.fused_gdn_batch_verify_fallback_reasons == {}


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


def test_row_exact_canary_does_not_publish_success_counters(monkeypatch):
    layer, cache, o = _gdn_stub()
    layer.set_fused_gdn_batch_verify_mode("row_exact")
    monkeypatch.setattr(Q, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(Q, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(Q, "probe_qwen36_gdn", lambda *a, **k: 4)

    def kernel(*args, **kw):
        out = mx.zeros((4, 3, 4096), mx.bfloat16)
        conv = mx.zeros((4, 3, 8192), mx.bfloat16)
        state = mx.zeros((4, 32, 128, 128), mx.float32)
        convs = mx.zeros((4, 2, 3, 8192), mx.bfloat16)
        states = mx.zeros((4, 2, 32, 128, 128), mx.float32)
        return out, conv, state, states, convs

    monkeypatch.setattr(V, "qwen4_fused_gdn_batch_verify", kernel)
    got = layer.row_exact_probe_fused_decode(
        o["qkv"], o["z"], o["b"], o["a"], None, cache
    )
    assert got.shape == (4, 3, 4096)
    assert layer.fused_gdn_batch_verify_calls == 0
    assert layer.fused_gdn_batch_verify_fallbacks == 0


def test_decode_win_routes_are_qualifiable_and_their_slices_required():
    # The Metal identity (1488/1488 on both artifacts) and the paired A/B
    # exist (options-sweep-qwen36-20261002): a decode-win route may qualify,
    # but only with its selected slices observed.
    from mlx2.qualification import required_feature_checks

    profile = A.configure_environment({"fused_gdn_verify": True})
    assert profile["MLX_QWEN36_DECODE_WINS"] == "1"
    settings = {"environment": profile, "mtp": True, "speculation": "self_mtp"}
    assert unqualifiable_candidate(settings) is None
    assert "feature_qwen36_fused_gdn_verify" in required_feature_checks(settings)


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

    def table(experts, out_dims, packed, *, bits=4):
        # Resident tables are module parameters (real arrays), as a loaded
        # checkpoint holds them; they stay lazy because admission reads only
        # shape and dtype, so the full E256 geometry costs no memory.
        layer = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
        nn.Module.__init__(layer)
        layer.bits = bits
        layer.group_size = 64
        layer.mode = "affine"
        layer.weight = mx.zeros((experts, out_dims, packed), mx.uint32)
        groups = packed * (32 // bits) // 64
        layer.scales = mx.zeros((experts, out_dims, groups), mx.bfloat16)
        layer.biases = mx.zeros((experts, out_dims, groups), mx.bfloat16)
        layer.eval()
        return layer

    def tables(experts, *, bits=4):
        return (
            table(experts, 512, 2048 * bits // 32, bits=bits),
            table(experts, 512, 2048 * bits // 32, bits=bits),
            table(experts, 2048, 512 * bits // 32, bits=bits),
        )

    x = _ShapeOnly((1, 1, 2048), mx.bfloat16)
    inds = _ShapeOnly((1, 8), mx.uint32)
    scores = _ShapeOnly((1, 8), mx.bfloat16)
    ok = RD.admit_split_routed_decode(
        x, inds, scores, *tables(256), contract=QWEN36_E256_K8_ROUTING
    )
    assert ok.accepted, ok.reason
    q8_default = RD.admit_split_routed_decode(
        x, inds, scores, *tables(256, bits=8), contract=QWEN36_E256_K8_ROUTING
    )
    assert not q8_default.accepted and "bits 8 != 4" in q8_default.reason
    q8 = RD.admit_split_routed_decode(
        x,
        inds,
        scores,
        *tables(256, bits=8),
        contract=QWEN36_E256_K8_ROUTING,
        bits=8,
        group_size=64,
    )
    assert q8.accepted, q8.reason
    # A consistent E128 top-8 model is not the Qwen3.6 geometry: the width-512
    # intermediate exception must not silently widen to it.
    other = RD.admit_split_routed_decode(
        x, inds, scores, *tables(128), contract=QWEN36_E256_K8_ROUTING
    )
    assert not other.accepted and "E128/top-8" in other.reason
    # Nor does the exception reach a top-10 (Flash-Next-shaped) launch.
    ten = _ShapeOnly((1, 10), mx.uint32)
    wide = RD.admit_split_routed_decode(
        x, ten, _ShapeOnly((1, 10), mx.bfloat16), *tables(256),
        contract=QWEN36_E256_K8_ROUTING,
    )
    assert not wide.accepted and "E256/top-10" in wide.reason


def _qwen36_block(monkeypatch):
    """E256/top-8 block with the Metal halves replaced by composed stand-ins."""
    from types import SimpleNamespace

    from mlx import nn

    from mlx2.runtime.models import qwen3_next as QN
    from mlx2.runtime.models import qwen36_moe_decode as M

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
    for projection in (
        block.switch_mlp.gate_proj,
        block.switch_mlp.up_proj,
        block.switch_mlp.down_proj,
    ):
        projection.bits = 4
        projection.group_size = 64
    monkeypatch.setattr(
        RD,
        "admit_split_routed_decode",
        lambda *a, **k: RD.RoutedDecodeAdmission(True, "eligible"),
    )
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    monkeypatch.setattr(RD, "served_swiglu_refusal", lambda: None)
    monkeypatch.setattr(
        M,
        "routed_rows",
        lambda x, *a, **k: mx.zeros((x.size // x.shape[-1], x.shape[-1]), x.dtype),
    )
    launches = []

    def router_topk(logits, *, top_k, **_contract):
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


def test_decode_wins_reports_window_consumers_and_keeps_aggregate(monkeypatch):
    from types import SimpleNamespace

    block, _, _ = _qwen36_block(monkeypatch)
    block.moe_window_calls = {"batch_decode": 3, "verify": 5, "row_exact": 7}
    model = SimpleNamespace(named_modules=lambda: (("moe", block),))
    report = Q.qwen36_decode_wins_stats(model)
    assert report["moe"]["window_calls"] == 15
    assert report["moe"]["window_calls_by_consumer"] == {
        "batch_decode": 3,
        "verify": 5,
        "row_exact": 7,
    }
    Q.qwen36_decode_wins_stats(model, reset=True)
    assert block.moe_window_calls == {
        "batch_decode": 0,
        "verify": 0,
        "row_exact": 0,
    }


def test_row_exact_window_selection_remains_separate_from_artifact_admission(
    monkeypatch,
):
    block, _, _ = _qwen36_block(monkeypatch)
    block.set_moe_window_consumers(("row_exact",))
    monkeypatch.setattr(
        RD,
        "admit_split_routed_decode",
        lambda *a, **k: RD.RoutedDecodeAdmission(False, "gate: bits 8 != 4"),
    )
    x = mx.zeros((1, 3, 64), mx.bfloat16)
    assert block.row_exact_window_selected() is True
    assert block.row_exact_window_refusal(x) == "gate: bits 8 != 4"


def test_admitted_qwen36_row_exact_window_engages_once(monkeypatch):
    block, _, _ = _qwen36_block(monkeypatch)
    block.set_moe_window_consumers(("row_exact",))
    x = mx.zeros((1, 3, 64), mx.bfloat16)
    assert block.row_exact_window_refusal(x) is None
    with REV.window(REV.Window(3)):
        y = block(x)
    mx.eval(y)
    assert block.moe_window_calls["row_exact"] == 1


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 16])
def test_qwen36_grouped_router_shared_gate_keeps_full_block_bits(monkeypatch, rows):
    from mlx2.runtime.models import qwen36_moe_decode as M

    block, _, _ = _qwen36_block(monkeypatch)
    block.set_moe_window_consumers(("row_exact",))
    x = (mx.arange(rows * 64).reshape(1, rows, 64) / 257).astype(mx.bfloat16)
    with REV.window(REV.Window(rows)):
        grouped = block(x)
    mx.eval(grouped)

    def serial(block, flat):
        logits = mx.concatenate(
            [
                block.gate(flat[r : r + 1].reshape(1, 1, -1)).reshape(1, -1)
                for r in range(rows)
            ]
        )
        shared = mx.concatenate(
            [
                block.shared_expert_gate(
                    flat[r : r + 1].reshape(1, 1, -1)
                ).reshape(1, 1)
                for r in range(rows)
            ]
        )
        return logits, shared

    monkeypatch.setattr(M, "router_and_shared_gate_rows", serial)
    with REV.window(REV.Window(rows)):
        reference = block(x)
    mx.eval(reference)
    assert mx.array_equal(grouped, reference).item()


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


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("rps", [2, 4])
@pytest.mark.parametrize("rows", [1, 3])
def test_qwen36_down_launch_covers_every_output_row(monkeypatch, shared, rps, rows):
    """The down kernel runs one simdgroup per slot (threadgroup y = slots) and
    RPS output rows per threadgroup; MLX grids count threads, so grid y must
    be slots * hidden / RPS or 7/8 of each output row is never written."""
    from types import SimpleNamespace as NS

    from mlx2.runtime.models import qwen36_moe_decode as M

    calls = []

    def recorder(*_args):
        def run(**kw):
            calls.append(kw)
            return [mx.zeros(s, d) for s, d in zip(kw["output_shapes"], kw["output_dtypes"])]

        return run

    monkeypatch.setattr(M, "routed_gate_up_kernel", recorder)
    monkeypatch.setattr(M, "shared_gate_up_kernel", recorder)
    monkeypatch.setattr(M, "down_kernel", recorder)
    monkeypatch.setattr(RD, "expert_operands", lambda layer: ())
    monkeypatch.setattr(RD, "_dense_operands", lambda layer: ())
    monkeypatch.setattr(RD, "served_down_rows", lambda: rps)
    table = NS(weight=mx.zeros((256, 512, 1)), bits=4, group_size=64)
    fmt = NS(bits=4, group_size=64)
    shared_mlp = NS(gate_proj=fmt, up_proj=fmt, down_proj=fmt) if shared else None
    hidden = 2048
    out = M.routed_rows(
        mx.zeros((rows, 1, hidden), mx.bfloat16),
        mx.zeros((rows, 8), mx.uint32),
        mx.zeros((rows, 8), mx.bfloat16),
        table,
        table,
        table,
        shared=shared_mlp,
        shared_gate=mx.zeros((rows, 1), mx.bfloat16) if shared else None,
    )
    assert out.shape == (rows, hidden)
    down = calls[-1]
    grid, group = down["grid"], down["threadgroup"]
    assert group[1] == 8 + int(shared)
    assert grid[1] % group[1] == 0
    assert (grid[1] // group[1]) * rps == hidden
    assert grid[2] == rows


def test_qwen36_q8_window_kernels_bind_distinct_format_headers(monkeypatch):
    from mlx2.runtime.models import qwen36_moe_decode as M

    calls = []

    def recorder(**kwargs):
        calls.append(kwargs)
        return object()

    factories = (M.routed_gate_up_kernel, M.shared_gate_up_kernel, M.down_kernel)
    for factory in factories:
        factory.cache_clear()
    monkeypatch.setattr(mx.fast, "metal_kernel", recorder)
    monkeypatch.setattr(RD.SWIGLU_GATE, "adapt", lambda text, _dtype: text)
    try:
        M.routed_gate_up_kernel(4, 64)
        M.shared_gate_up_kernel(4, 4, 64)
        M.down_kernel(4, 64, (4, 64), True)
        M.routed_gate_up_kernel(8, 64)
        M.shared_gate_up_kernel(8, 8, 64)
        M.down_kernel(8, 64, (8, 64), True)
    finally:
        for factory in factories:
            factory.cache_clear()
    assert [call["name"] for call in calls] == [
        "qwen36_split_gate_up_rows_q4g64",
        "qwen36_gate_up_shared_rows_q4_shared4g64",
        "qwen36_stock_down_rows_q4g64_shared4g64",
        "qwen36_split_gate_up_rows_q8g64",
        "qwen36_gate_up_shared_rows_q8_shared8g64",
        "qwen36_stock_down_rows_q8g64_shared8g64",
    ]
    assert all(
        "constant constexpr int BITS = 4;" in call["header"] for call in calls[:3]
    )
    assert all(
        "constant constexpr int BITS = 8;" in call["header"] for call in calls[3:]
    )
    assert "q4f::" not in calls[1]["source"]
    assert "q4f::" not in calls[4]["source"]


@pytest.mark.parametrize("inter, fast", [(512, True), (640, False)])
def test_qwen36_down_kernel_layout_follows_qmv_fast(monkeypatch, inter, fast):
    """The stock-down kernel with the qmv_fast header walks K in whole
    512-wide blocks with no guarded tail.  The split admission also accepts
    non-512 intermediates (640, 768, ...) under the Qwen3.6 contract, so the
    launch must pick the guarded (qmv) layout there, as MLX's stock path and
    ``candidate_down_combine`` do, or it reads past each down row."""
    from types import SimpleNamespace as NS

    from mlx2.runtime.models import qwen36_moe_decode as M

    layouts = []

    def run(**kw):
        return [mx.zeros(s, d) for s, d in zip(kw["output_shapes"], kw["output_dtypes"])]

    def down_kernel(*args, **kwargs):
        layouts.append((args, kwargs))
        return run

    monkeypatch.setattr(M, "routed_gate_up_kernel", lambda *_args: run)
    monkeypatch.setattr(M, "down_kernel", down_kernel)
    monkeypatch.setattr(RD, "expert_operands", lambda layer: ())
    table = NS(weight=mx.zeros((256, inter, 1)), bits=4, group_size=64)
    M.routed_rows(
        mx.zeros((1, 1, 2048), mx.bfloat16),
        mx.zeros((1, 8), mx.uint32),
        mx.zeros((1, 8), mx.bfloat16),
        table,
        table,
        table,
    )
    (args, kwargs), = layouts
    assert kwargs.get("fast", args[3] if len(args) > 3 else None) is fast
