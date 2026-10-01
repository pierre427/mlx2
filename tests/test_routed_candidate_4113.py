"""CPU tests for the default-off omlx #4113 routed-decode candidate.

The candidate kernels are Metal-only. Here they are replaced by references
built from the composed MLX ops, so the tests pin admission, which down
traversal is dispatched, counters, declines, block plumbing and the Qwen3.6
adapter opt-in. Kernel numerics are a parent-owned Metal gate:
``scripts/check_fn_routed_decode.py --candidate --i-own-the-gpu`` (Qwen3.6
geometry by default).
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime.models import qwen3_next as QN
from mlx2.runtime.models import qwen4_routed_decode as RD

# Qwen3.6-shaped at small scale: hidden % 512 == 0, intermediate 512 (MLX
# takes qmv_fast for the down projection), top-8.
E, H, I, K = 12, 512, 512, 8


def _switch(inter=I, seed=0, bits=4, group_size=64, dtype=mx.bfloat16):
    mx.random.seed(seed)
    sw = QN.FusedDownSwitchGLU(H, inter, E)
    sw.gate_proj.weight = mx.random.normal((E, inter, H)) * 0.05
    sw.up_proj.weight = mx.random.normal((E, inter, H)) * 0.05
    sw.down_proj.weight = mx.random.normal((E, H, inter)) * 0.05
    nn.quantize(sw, group_size=group_size, bits=bits)
    sw.set_dtype(dtype)
    QN._enable_routed_decode(sw, "off")
    QN._enable_routed_candidate(sw, "off")
    sw.eval()
    return sw


def _route(seed, rows=1, top_k=K, dtype=mx.bfloat16):
    gates = mx.softmax(mx.random.normal((1, rows, E), key=mx.random.key(seed)), axis=-1)
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    return inds, (scores / scores.sum(axis=-1, keepdims=True)).astype(dtype)


def _x(seed, rows=1, dtype=mx.bfloat16):
    return mx.random.normal((1, rows, H), key=mx.random.key(100 + seed)).astype(dtype)


@pytest.fixture
def reference_kernels(monkeypatch):
    """Swap the Metal kernels for the composed ops they must equal."""
    calls = {"gate_up": 0, "down": 0}

    def gate_up(x, indices, gate, up):
        calls["gate_up"] += 1
        g = gate(x, indices)
        u = up(x, indices)
        return QN.SwiGLU()(u, g).reshape(indices.size, -1)

    def down(h, indices, scores, down_proj):
        calls["down"] += 1
        y = down_proj(h.reshape(indices.shape + (1, h.shape[-1])), indices).squeeze(-2)
        return (y * scores[..., None]).sum(axis=-2).reshape(-1)

    monkeypatch.setattr(RD, "candidate_gate_up_swiglu", gate_up)
    monkeypatch.setattr(RD, "candidate_down_combine", down)
    monkeypatch.setattr(RD, "candidate_runtime_refusal", lambda: None)
    return calls


def _admit(sw, x=None, inds=None, scores=None):
    if x is None:
        x = mx.expand_dims(_x(0), (-2, -3))
    if inds is None:
        inds, scores = _route(0)
    return RD.admit_routed_candidate(x, inds, scores, sw.gate_proj, sw.up_proj, sw.down_proj)


def test_qmv_fast_layout_matches_the_served_geometries():
    assert RD.qmv_fast_layout(2048, 512)  # Qwen3.6 gate/up
    assert RD.qmv_fast_layout(512, 2048)  # Qwen3.6 down: qmv_fast
    assert not RD.qmv_fast_layout(640, 2560)  # Flash-Next down: qmv
    assert RD.qmv_fast_layout(2560, 1280)
    assert not RD.qmv_fast_layout(512, 12)  # N % 8
    assert RD.qmv_fast_layout(256, 8, bits=8) and not RD.qmv_fast_layout(256, 8)


def test_admission_accepts_top8_split_tables_and_refuses_the_rest():
    sw = _switch()
    assert _admit(sw).accepted, _admit(sw).reason
    assert _admit(_switch(inter=64)).accepted  # qmv down with guarded tail
    x = mx.expand_dims(_x(0), (-2, -3))
    inds10, scores10 = _route(0, top_k=10)
    assert _admit(sw, x, inds10, scores10).accepted
    inds, scores = _route(0)
    x2 = mx.expand_dims(_x(0, rows=2), (-2, -3))
    i2, s2 = _route(0, rows=2)
    assert _admit(sw, x2, i2, s2).reason == "not one token"
    assert "bfloat16" in _admit(sw, x.astype(mx.float16), inds, scores).reason
    i6, s6 = _route(0, top_k=6)
    assert "top-k" in _admit(sw, x, i6, s6).reason
    assert _admit(sw, x, inds, None).reason == "no scores"
    assert "scores" in _admit(sw, x, inds, scores.astype(mx.float32)).reason
    assert "bits" in _admit(_switch(bits=8)).reason
    assert "group size" in _admit(_switch(group_size=32)).reason
    wide = _switch()
    wide.up_proj = _switch(inter=64).up_proj
    assert _admit(wide).reason == "gate/up tables do not match down"


def test_admission_refuses_a_subclassed_expert_table():
    from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear

    class Repacked(QuantizedSwitchLinear):
        pass

    sw = _switch()
    table = sw.down_proj
    sw.down_proj = Repacked.__new__(Repacked)
    sw.down_proj.__dict__.update(table.__dict__)
    assert "resident QuantizedSwitchLinear" in _admit(sw).reason


@pytest.mark.parametrize("inter,fast", [(512, True), (64, False)])
def test_down_dispatch_follows_mlx_traversal(monkeypatch, inter, fast):
    seen = []

    def fake_kernel(kind, is_fast):
        seen.append((kind, is_fast))
        return lambda **kw: [mx.zeros(kw["output_shapes"][0], kw["output_dtypes"][0])]

    monkeypatch.setattr(RD, "_candidate_kernel", fake_kernel)
    sw = _switch(inter=inter)
    inds, scores = _route(1)
    h = RD.candidate_gate_up_swiglu(mx.expand_dims(_x(1), (-2, -3)), inds, sw.gate_proj, sw.up_proj)
    y = RD.candidate_down_combine(h, inds, scores, sw.down_proj)
    assert h.shape == (K, inter) and y.shape == (H,)
    assert seen == [("gate_up", True), ("down", fast)]


def test_kernel_sources_carry_topk_and_both_traversals():
    src = RD.CANDIDATE_DOWN_SOURCE
    assert "threadgroup T part[TOPK * RPS]" in src
    assert "(FAST ? K : K - BLOCK_SIZE)" in src and "if (!FAST)" in src
    assert "lane[j % 8] = part[j * RPS + row] + lane[j % 8]" in src
    # The qualified top-10 path is untouched.
    assert "part[10 * RPS]" in RD.DOWN_SOURCE and RD.TOP_K == 10


@pytest.mark.parametrize("inter", [512, 64])
@pytest.mark.parametrize("top_k", [8, 10])
def test_candidate_matches_the_composed_body(reference_kernels, inter, top_k):
    sw = _switch(inter=inter)
    for seed in range(3):
        x = _x(seed)
        inds, scores = _route(seed, top_k=top_k)
        sw.routed_candidate_mode = "off"
        want = sw(x, inds, scores=scores, variant="stock")
        sw.routed_candidate_mode = "two_launch"
        got = sw(x, inds, scores=scores, variant="stock")
        assert got.shape == want.shape == (1, 1, H) and got.dtype == want.dtype
        assert mx.array_equal(got, want).item()
    assert sw.routed_candidate_calls == 3 and sw.routed_candidate_fallbacks == 0
    assert reference_kernels == {"gate_up": 3, "down": 3}


def test_off_is_inert_and_non_stock_variant_is_not_a_candidate(reference_kernels):
    sw = _switch()
    x, (inds, scores) = _x(2), _route(2)
    sw(x, inds, scores=scores, variant="stock")
    sw.routed_candidate_mode = "two_launch"
    sw(x, inds, scores=scores, variant="tile4")
    sw(_x(3, rows=3), *_route(3, rows=3), variant="stock")  # multi-token: quiet
    assert reference_kernels == {"gate_up": 0, "down": 0}
    assert sw.routed_candidate_calls == sw.routed_candidate_fallbacks == 0


def test_cpu_decline_is_counted_and_bit_equal_to_the_reference():
    """No monkeypatching: on CPU the runtime refuses and the body runs."""
    with mx.stream(mx.cpu):
        sw = _switch()
        x, (inds, scores) = _x(4), _route(4)
        want = sw(x, inds, scores=scores, variant="stock")
        sw.routed_candidate_mode = "two_launch"
        got = sw(x, inds, scores=scores, variant="stock")
    if RD.runtime_supported():
        pytest.skip("Metal default device: the CPU refusal is not exercised")
    assert mx.array_equal(got, want).item()
    assert sw.routed_candidate_calls == 0 and sw.routed_candidate_fallbacks == 1
    assert sw.routed_candidate_fallback_reasons == {"Metal runtime unavailable": 1}


def test_served_swiglu_refusal_blocks_the_candidate(monkeypatch):
    refusal = "served compiled SwiGLU uses metal::precise::exp"
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    monkeypatch.setattr(RD.SWIGLU_GATE, "probe", lambda dtype: {"metal::exp": False, "metal::precise::exp": True})
    RD.SWIGLU_GATE.reset()
    try:
        assert RD.candidate_runtime_refusal() == refusal
        sw = _switch()
        sw.routed_candidate_mode = "two_launch"
        sw(_x(5), *_route(5), variant="stock")
        assert sw.routed_candidate_last_fallback == refusal
    finally:
        RD.SWIGLU_GATE.reset()


def test_fallback_reasons_are_bounded(monkeypatch):
    sw = _switch()
    sw.routed_candidate_mode = "two_launch"
    reasons = iter(f"reason {i}" for i in range(40))
    monkeypatch.setattr(RD, "candidate_runtime_refusal", lambda: next(reasons))
    for seed in range(40):
        sw(_x(seed), *_route(seed), variant="stock")
    table = sw.routed_candidate_fallback_reasons
    assert len(table) == QN._CANDIDATE_REASON_SLOTS + 1
    assert table["other"] == 40 - QN._CANDIDATE_REASON_SLOTS
    assert sw.routed_candidate_fallbacks == 40


def _block(monkeypatch, bits=4):
    monkeypatch.setattr(QN, "_MOE_FUSED_GATE_UP", False)
    monkeypatch.setattr(QN, "_MOE_SHARED_IN_GATHER", False)
    args = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
        norm_topk_prob=True, num_experts=E, num_experts_per_tok=K,
    )
    mx.random.seed(31)
    block = QN.Qwen3NextSparseMoeBlock(args)
    sw = block.switch_mlp
    sw.gate_proj.weight = mx.random.normal((E, I, H)) * 0.05
    sw.up_proj.weight = mx.random.normal((E, I, H)) * 0.05
    sw.down_proj.weight = mx.random.normal((E, H, I)) * 0.05
    nn.quantize(sw, group_size=64, bits=bits)
    block.set_dtype(mx.bfloat16)
    block.eval()
    block.set_fused_expert_kernel_mode("stock")
    return block


def test_block_routes_one_token_through_the_candidate(monkeypatch, reference_kernels):
    block = _block(monkeypatch)
    assert block.switch_mlp.routed_candidate_mode == "off"
    x = _x(7)
    want = block(x)
    assert block.set_moe_routed_candidate_mode("two_launch") == "two_launch"
    got = block(x)
    assert mx.array_equal(got, want).item()
    assert reference_kernels == {"gate_up": 1, "down": 1}
    block(_x(8, rows=3))
    stats = QN.routed_candidate_stats(block, reset=True)
    assert stats == {"mode": "two_launch", "layers": 1, "calls": 1, "fallbacks": 0, "reasons": {}}
    assert block.switch_mlp.routed_candidate_calls == 0


def test_block_decline_is_bit_identical(monkeypatch):
    monkeypatch.setattr(RD, "candidate_runtime_refusal", lambda: None)
    block = _block(monkeypatch, bits=8)
    x = _x(9)
    want = block(x)
    block.set_moe_routed_candidate_mode("two_launch")
    assert mx.array_equal(block(x), want).item()
    assert "bits" in block.switch_mlp.routed_candidate_last_fallback


def test_block_setters_validate_and_exclude_fused_expert_kernels(monkeypatch):
    block = _block(monkeypatch)
    with pytest.raises(ValueError):
        block.set_moe_routed_candidate_mode("on")
    block.set_moe_routed_candidate_mode("two_launch")
    with pytest.raises(ValueError, match="exclude the routed candidate"):
        block.set_fused_expert_kernel_mode("tile4")
    block.set_moe_routed_candidate_mode("off")
    block.set_fused_expert_kernel_mode("tile4")
    with pytest.raises(ValueError, match="stock expert kernel"):
        block.set_moe_routed_candidate_mode("two_launch")
    monkeypatch.setattr(QN, "_MOE_FUSED_GATE_UP", True)
    fused = QN.Qwen3NextSparseMoeBlock(SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
        norm_topk_prob=True, num_experts=E, num_experts_per_tok=K,
    ))
    assert not hasattr(fused.switch_mlp, "routed_candidate_mode")
    with pytest.raises(ValueError, match="split gate/up"):
        fused.set_moe_routed_candidate_mode("two_launch")


def test_qwen36_opt_in_is_explicit_only_and_unqualifiable(tmp_path):
    from mlx2.adapters import qwen36_35b
    from mlx2.qualification import load_qualified_route, unqualifiable_candidate

    name = qwen36_35b.KERNEL_POLICY_ENV["moe_routed_candidate"]
    with patch.dict(os.environ, {name: "1"}):
        assert name not in qwen36_35b.configure_environment()
        assert name not in os.environ
    with patch.dict(os.environ, {}):
        # Deselected: still absent, so stock receipts keep matching.
        assert name not in qwen36_35b.configure_environment({"moe_routed_candidate": False})
    with patch.dict(os.environ, {}):
        os.environ.pop("MLX_QWEN4_MOE_FUSED_GATE_UP", None)
        chosen = qwen36_35b.configure_environment({"moe_routed_candidate": True})
        assert chosen[name] == "1"
        # The pinned-off fused gate/up is not switched on by the candidate.
        assert chosen["MLX_QWEN4_MOE_FUSED_GATE_UP"] == "0"
    assert unqualifiable_candidate({"environment": {name: "0"}}) is None
    assert "#4113" in unqualifiable_candidate({"environment": {name: "1"}})
    # Refused before any receipt is read: no record can qualify the route.
    with pytest.raises(ValueError, match="unqualified candidate"):
        load_qualified_route(
            tmp_path / "absent.json", runtime={}, artifact="a",
            settings={"environment": {name: "1"}}, descriptor=None, name="n",
        )


def test_qwen36_adapter_refuses_candidate_when_a_layer_does_not_admit(monkeypatch):
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.environment = {"MLX_QWEN36_MOE_ROUTED_CANDIDATE": "1"}
    adapter.model = _block(monkeypatch, bits=8)
    with pytest.raises(ValueError, match="moe_routed_candidate refused: gate: bits"):
        adapter._select_routed_candidate()
    adapter.model = _block(monkeypatch)
    adapter._kernels = {"moe_routed_candidate": True}
    adapter._select_routed_candidate()
    assert adapter.model.switch_mlp.routed_candidate_mode == "two_launch"
    other = object.__new__(Qwen3635BA3BAdapter)
    other.environment = {"MLX_QWEN36_MOE_ROUTED_CANDIDATE": "0"}
    other.model = _block(monkeypatch)
    other._select_routed_candidate()
    assert other.model.switch_mlp.routed_candidate_mode == "off"


@pytest.mark.skipif(
    os.environ.get("MLX2_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="Metal oracle; parent-owned, set MLX2_METAL_TESTS=1 on an owned idle GPU",
)
@pytest.mark.parametrize("inter", [512, 64])
def test_metal_candidate_is_bit_exact(inter):
    # runtime_supported() reads the default device; conftest pins it to CPU.
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        sw = _switch(inter=inter)
        for seed in range(8):
            x, (inds, scores) = _x(seed), _route(seed)
            sw.routed_candidate_mode = "off"
            want = sw(x, inds, scores=scores, variant="stock")
            sw.routed_candidate_mode = "two_launch"
            got = sw(x, inds, scores=scores, variant="stock")
            assert mx.array_equal(got, want).item(), (inter, seed)
        assert sw.routed_candidate_calls == 8, sw.routed_candidate_fallback_reasons
    finally:
        mx.set_default_device(previous)



def _dry_run(monkeypatch, tmp_path, **overrides):
    import json

    from scripts import check_fn_routed_decode as checker

    for name in ("candidate_gate_up_swiglu", "candidate_down_combine", "candidate_runtime_refusal"):
        monkeypatch.setattr(RD, name, getattr(RD, name))  # restored at teardown
    args = SimpleNamespace(
        cases=3, layers=0, reps=0, out=str(tmp_path / "out.json"), candidate=True,
        experts=None, hidden=None, inter=None, topk=None, cpu_dry_run=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    code = checker.run_candidate(args)
    return code, json.loads((tmp_path / "out.json").read_text())


def test_checker_dry_run_passes_and_labels_its_scope(monkeypatch, tmp_path):
    code, rec = _dry_run(monkeypatch, tmp_path)
    assert code == 0 and rec["verdict"] == "dry-run-pass"
    assert "no Metal numerics" in rec["scope"] and rec["timing"] is None
    assert rec["geometry"]["down_traversal"] == "qmv_fast"
    assert rec["counts"] == {"cases": 3, "engaged": 3, "hidden_exact": 3, "down_exact": 3, "output_exact": 3}


def test_checker_refuses_an_unadmitted_geometry(monkeypatch, tmp_path):
    code, rec = _dry_run(monkeypatch, tmp_path, hidden=256)
    assert code == 1 and rec["verdict"] == "refused"
    assert rec["refusals"] == ["admission: gate/up would not take qmv_fast"]
    assert rec["counts"]["cases"] == 0


def test_checker_refuses_one_ulp_of_drift(monkeypatch, tmp_path):
    from scripts import check_fn_routed_decode as checker

    install = checker._install_reference_kernels

    def drifting(rd, qn):
        install(rd, qn)
        exact = rd.candidate_down_combine
        rd.candidate_down_combine = lambda *a: (exact(*a).view(mx.uint16) ^ 1).view(mx.bfloat16)

    monkeypatch.setattr(checker, "_install_reference_kernels", drifting)
    code, rec = _dry_run(monkeypatch, tmp_path)
    assert code == 1 and rec["verdict"] == "refused"
    assert "down_exact 0/3" in rec["refusals"] and "output_exact 0/3" in rec["refusals"]


def test_checker_refuses_a_run_that_never_engages(monkeypatch, tmp_path):
    from scripts import check_fn_routed_decode as checker

    install = checker._install_reference_kernels

    def declining(rd, qn):
        install(rd, qn)
        rd.candidate_runtime_refusal = lambda: "Metal runtime unavailable"

    monkeypatch.setattr(checker, "_install_reference_kernels", declining)
    code, rec = _dry_run(monkeypatch, tmp_path)
    assert code == 1 and rec["refusals"] == ["case 0: not engaged (Metal runtime unavailable)"]


class _CandidateFactory:
    """Stands in for the resolved adapter class; loading always stops."""

    qualification_mode_only_policy = frozenset({"moe_routed_candidate"})

    def __init__(self, *args, **kwargs):
        raise RuntimeError("factory reached")


@pytest.mark.parametrize("record", [None, "qualification.json"])
def test_serving_refuses_the_candidate_outside_qualification_mode(record):
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="restricted to qualification mode"):
        ServingEngine(
            "unused", adapter_factory=_CandidateFactory, qualification=record,
            execution_policy={"moe_routed_candidate": True},
        )


def test_serving_admits_the_candidate_in_qualification_mode():
    from mlx2.serving import ServingEngine

    engine = ServingEngine(
        "unused", adapter_factory=_CandidateFactory, qualification_mode=True,
        execution_policy={"moe_routed_candidate": True},
    )
    try:
        engine.thread.join(30)
        assert "factory reached" in str(engine.error)
    finally:
        engine.close()
    # An unselected (false) candidate key is not a selection.
    other = ServingEngine(
        "unused", adapter_factory=_CandidateFactory,
        execution_policy={"moe_routed_candidate": False},
    )
    try:
        other.thread.join(30)
        assert "factory reached" in str(other.error)
    finally:
        other.close()


def test_qwen36_adapter_declares_the_candidate_qualification_mode_only():
    from mlx2.adapters.qwen36_35b import KERNEL_POLICY_ENV, Qwen3635BA3BAdapter

    assert Qwen3635BA3BAdapter.qualification_mode_only_policy == {"moe_routed_candidate"}
    assert Qwen3635BA3BAdapter.qualification_mode_only_policy <= set(KERNEL_POLICY_ENV)


@pytest.mark.parametrize("override", [
    {"cases": 0}, {"layers": -1}, {"experts": 0}, {"hidden": 0}, {"inter": 0}, {"experts": 4},
])
def test_checker_rejects_zero_engagement_and_invalid_geometry(monkeypatch, tmp_path, override):
    """An explicit 0 is not masked by the default; nothing runs, nothing passes."""
    with pytest.raises(ValueError):
        _dry_run(monkeypatch, tmp_path, **override)
    assert not (tmp_path / "out.json").exists()


def test_checker_cli_rejects_zero_cases():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/check_fn_routed_decode.py", "--candidate", "--cpu-dry-run",
         "--cases", "0", "--out", os.devnull],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 2 and "must be positive" in result.stderr
