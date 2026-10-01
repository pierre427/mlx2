"""CPU tests for the routed MoE row window and the router top-k launch/fold.

The window and routing kernels are Metal-only; here they are replaced by
composed stand-ins (``_reference_kernels``), so these tests pin the plumbing:
which consumer a call belongs to, admission and its counted refusals, the
shapes, the switches, the policy/env/receipt contract and the generated
sources. The numerics are checked on the GPU by
``scripts/check_fn_moe_window.py`` (real Flash-Next weights).
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime import verify_scope
from mlx2.runtime.models import qwen3_next as QN
from mlx2.runtime.models import qwen4_moe_window as W
from mlx2.runtime.models import qwen4_routed_decode as RD
from mlx2.runtime.models.precise_ops import gate_sigmoid

E, H, I = 16, 512, 64


def _block(monkeypatch, seed=0):
    monkeypatch.setattr(QN, "_MOE_FUSED_GATE_UP", False)
    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
        norm_topk_prob=True, num_experts=E, num_experts_per_tok=10,
    )
    block = QN.Qwen3NextSparseMoeBlock(args)
    for proj, shape in (("gate_proj", (E, I, H)), ("up_proj", (E, I, H)), ("down_proj", (E, H, I))):
        getattr(block.switch_mlp, proj).weight = mx.random.normal(shape) * 0.05
    nn.quantize(block, group_size=64, bits=4)
    block.gate = nn.QuantizedLinear.from_linear(nn.Linear(H, E, bias=False), group_size=64, bits=8)
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(
        nn.Linear(H, 1, bias=False), group_size=64, bits=8)
    block.set_dtype(mx.bfloat16)
    block.set_fused_expert_kernel_mode("tile4")  # declines on CPU -> stock down
    block.eval()
    return block


def _stock_routing(logits):
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-10, axis=-1)[..., -10:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    return inds, scores / scores.sum(axis=-1, keepdims=True)


@pytest.fixture
def ref_kernels(monkeypatch, served_exp_forms_match):
    """Composed stand-ins for the window and routing launches."""
    calls = {"routed": 0, "shared": 0, "router": 0, "fold": 0}

    def per_row(x, inds, scores, gate, up, down):
        out = []
        for r in range(x.shape[0]):
            xe = mx.expand_dims(x[r : r + 1], (-2, -3))
            idx = inds[r : r + 1]
            h = QN.SwiGLU()(up(xe, idx), gate(xe, idx))
            out.append((down(h, idx).squeeze(-2) * scores[r : r + 1, :, None]).sum(axis=-2))
        return mx.concatenate(out, axis=0)

    def routed_rows(x, inds, scores, gate, up, down, *, logits=None, rows_per_tg=None):
        calls["routed"] += 1
        if logits is not None:
            calls["fold"] += 1
            inds, scores = (a.reshape(-1, 10) for a in _stock_routing(logits))
            return per_row(x, inds, scores, gate, up, down), inds, scores
        return per_row(x, inds.reshape(-1, 10), scores.reshape(-1, 10), gate, up, down)

    def shared_rows(x, inds, scores, gate, up, down, shared, gate_logit, *, logits=None, rows_per_tg=None):
        calls["shared"] += 1
        if logits is not None:
            calls["fold"] += 1
            inds, scores = (a.reshape(-1, 10) for a in _stock_routing(logits))
        y = per_row(x, inds.reshape(-1, 10), scores.reshape(-1, 10), gate, up, down)
        sh = mx.concatenate([shared(x[r : r + 1]) for r in range(x.shape[0])], axis=0)
        y = y + gate_sigmoid(gate_logit.reshape(-1, 1)) * sh
        return (y, inds, scores) if logits is not None else y

    def router_topk(logits):
        calls["router"] += 1
        return _stock_routing(logits)

    monkeypatch.setattr(W, "routed_rows", routed_rows)
    monkeypatch.setattr(W, "shared_rows", shared_rows)
    monkeypatch.setattr(W, "router_topk", router_topk)
    monkeypatch.setattr(W, "NUM_EXPERTS", E)
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    monkeypatch.setattr(QN, "_served_down_refusal", lambda *a: None)
    return calls


def _one_token_rows(block, x):
    """The reference: every row through the block alone, window and top-k off."""
    consumers, mode = block.moe_window_consumers, block.moe_topk_mode
    block.set_moe_window_consumers(())
    block.set_moe_topk_mode("off")
    flat = x.reshape(-1, H)
    out = mx.concatenate([block(flat[r].reshape(1, 1, H)).reshape(1, H) for r in range(flat.shape[0])])
    block.set_moe_window_consumers(consumers)
    block.set_moe_topk_mode(mode)
    return out


def _x(seed, shape):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16)


# ----------------------------------------------------------------- parsing


def test_env_parsing(monkeypatch):
    monkeypatch.delenv(W.WINDOW_ENV, raising=False)
    assert W.consumers_from_env() == frozenset()
    monkeypatch.setenv(W.WINDOW_ENV, "batch_decode, verify")
    assert W.consumers_from_env() == {"batch_decode", "verify"}
    monkeypatch.setenv(W.WINDOW_ENV, "all")
    assert W.consumers_from_env() == set(W.CONSUMERS)
    monkeypatch.setenv(W.WINDOW_ENV, "prefill")
    with pytest.raises(ValueError, match="unknown consumers"):
        W.consumers_from_env()
    monkeypatch.setenv(W.TOPK_ENV, "1")
    assert W.topk_mode_from_env() == "fold"
    monkeypatch.setenv(W.TOPK_ENV, "launch")
    assert W.topk_mode_from_env() == "launch"
    monkeypatch.setenv(W.TOPK_ENV, "on")
    with pytest.raises(ValueError):
        W.topk_mode_from_env()


def test_sources_substitute_every_token_offset():
    gate_up = W._shared_gate_up_window_source(fold=False)
    assert "rhs[token * TOPK + slot]" in gate_up and "xr, simd_lid, result" in gate_up
    assert "rhs[tid.z]" not in W.FOLD_SPLIT_GATE_UP_SOURCE
    assert "fold_expert" in W._shared_gate_up_window_source(fold=True)
    down = W._shared_down_window_source()
    for needle in ("gate[token]", "scores[token * TOPK + slot]", "out[size_t(token) * H"):
        assert needle in down
    with pytest.raises(AssertionError):
        W._replace_once("abc", "x", "y")


def test_setters_validate(monkeypatch):
    block = _block(monkeypatch)
    assert block.moe_window_consumers == frozenset() and block.moe_topk_mode == "off"
    assert block.set_moe_window_consumers({"verify"}) == {"verify"}
    with pytest.raises(ValueError):
        block.set_moe_window_consumers({"prefill"})
    assert block.set_moe_topk_mode("fold") == "fold"
    with pytest.raises(ValueError):
        block.set_moe_topk_mode("on")
    with pytest.raises(ValueError):
        W.set_topk_fold_max_rows(0)


# ----------------------------------------------------------------- windows


@pytest.mark.parametrize("topk", W.TOPK_MODES)
@pytest.mark.parametrize("shared", [True, False])
def test_batch_decode_window_matches_one_token_rows(monkeypatch, ref_kernels, topk, shared):
    block = _block(monkeypatch)
    W.set_window_shared(shared)
    try:
        block.set_moe_window_consumers({"batch_decode"})
        block.set_moe_topk_mode(topk)
        x = _x(1, (3, 1, H))
        got = block(x)
        want = _one_token_rows(block, x)
    finally:
        W.set_window_shared(True)
    assert got.shape == (3, 1, H)
    assert mx.array_equal(got.reshape(3, H), want).item()
    assert block.moe_window_calls["batch_decode"] == 1 and block.moe_window_rows == 3
    assert block.moe_window_shared_calls == int(shared)
    assert ref_kernels["shared" if shared else "routed"] == 1
    if topk != "off":
        assert block.moe_topk_calls[topk] == 1


def test_consumers_are_told_apart(monkeypatch, ref_kernels):
    block = _block(monkeypatch)
    block.set_moe_window_consumers(set(W.CONSUMERS))
    with verify_scope.verify_forward():
        block(_x(2, (1, 3, H)))
    record = REV.Window(3)
    with REV.window(record):
        block(_x(3, (1, 3, H)))
    block(_x(4, (2, 1, H)))
    assert block.moe_window_calls == {"row_exact": 1, "batch_decode": 1, "verify": 1}
    assert record.exact and record.stages["moe_window"] == {"shared_fold": 3}
    assert "projections" in record.stages
    # prefill rows (no scope, L > 1) are not a window: quiet, uncounted
    block(_x(5, (1, 6, H)))
    assert sum(block.moe_window_calls.values()) == 3
    assert sum(block.moe_window_fallbacks.values()) == 0


def test_window_off_for_a_consumer_is_quiet(monkeypatch, ref_kernels):
    block = _block(monkeypatch)
    block.set_moe_window_consumers({"verify"})
    block(_x(6, (3, 1, H)))
    assert block.moe_window_calls["batch_decode"] == 0
    assert block.moe_window_fallbacks["batch_decode"] == 0
    assert ref_kernels["routed"] == ref_kernels["shared"] == 0


def test_window_refusals_are_counted(monkeypatch, ref_kernels):
    block = _block(monkeypatch)
    block.set_moe_window_consumers({"batch_decode", "verify"})
    x = _x(7, (3, 1, H))
    want = _one_token_rows(block, x)
    block.set_moe_routed_decode_mode("two_launch")
    got = block(x)
    assert block.moe_window_fallbacks["batch_decode"] == 1
    assert "two_launch" in block.moe_window_last_fallback
    block.set_moe_routed_decode_mode("off")
    block.set_fused_expert_kernel_mode("stock")
    block(x)
    assert "fused expert kernel off" in block.moe_window_last_fallback
    block.set_fused_expert_kernel_mode("tile4")
    block.set_moe_router_mode("fused")
    block(x)
    assert "fused router" in block.moe_window_last_fallback
    block.set_moe_router_mode("stock")
    with verify_scope.verify_forward():
        block(_x(8, (1, W.WINDOW_MAX_ROWS + 1, H)))
    assert "> 17" in block.moe_window_last_fallback
    assert block.moe_window_fallbacks["batch_decode"] == 3
    assert block.moe_window_fallbacks["verify"] == 1
    assert sum(block.moe_window_calls.values()) == 0
    assert got.reshape(3, H).shape == want.shape


def test_window_needs_metal(monkeypatch):
    block = _block(monkeypatch)
    monkeypatch.setattr(W, "NUM_EXPERTS", E)
    monkeypatch.setattr(QN, "_served_down_refusal", lambda *a: None)
    block.set_moe_window_consumers({"batch_decode"})
    block(_x(9, (2, 1, H)))
    assert block.moe_window_last_fallback == "Metal runtime unavailable"


def test_row_exact_window_replaces_the_per_row_expert_loop(monkeypatch, ref_kernels):
    """Inside a row-exact window the block is a RowExact subclass; the window
    serves its experts in one pass and the window stays exact."""
    from mlx2.runtime.models import qwen4_row_exact as RX

    block = _block(monkeypatch)
    for module, mixin in ((block, RX._RowExactMoE), (block.switch_mlp, RX._RowExactSwitch)):
        module.__class__ = RX._subclass(mixin, type(module))
    for linear in (block.gate, block.shared_expert_gate, block.shared_expert.gate_proj,
                   block.shared_expert.up_proj, block.shared_expert.down_proj):
        linear.__class__ = RX._subclass(RX._RowExactQuantizedLinear, type(linear))
    assert RD._linear_ok(block.shared_expert.gate_proj, 4, I, H) is None
    x = _x(10, (1, 5, H))
    want = _one_token_rows(block, x)
    block.set_moe_window_consumers({"row_exact"})
    record = REV.Window(5)
    with REV.window(record):
        got = block(x)
    assert mx.array_equal(got.reshape(5, H), want).item()
    assert record.exact
    assert "moe_experts" not in record.stages  # the per-row loop did not run
    assert record.stages["moe_window"] == {"shared_fold": 5}
    block.set_moe_window_consumers(())
    record = REV.Window(5)
    with REV.window(record):
        block(x)
    assert record.stages["moe_experts"] == {"per_row": 5} and record.exact


# ----------------------------------------------------------------- top-k


def test_b1_topk_launch_replaces_the_routing(monkeypatch, ref_kernels):
    block = _block(monkeypatch)
    x = _x(11, (1, 1, H))
    want = block(x)
    block.set_moe_topk_mode("launch")
    got = block(x)
    assert mx.array_equal(got, want).item()
    assert block.moe_topk_calls["launch"] == 1 and ref_kernels["router"] == 1


@pytest.mark.parametrize("routed", ["gate_up_down", "gate_up_down_shared"])
def test_b1_topk_fold_runs_the_routed_launches(monkeypatch, ref_kernels, routed):
    block = _block(monkeypatch)
    x = _x(12, (1, 1, H))
    want = block(x)
    block.set_moe_routed_decode_mode(routed)
    block.set_moe_topk_mode("fold")
    got = block(x)
    assert got.shape == (1, 1, H)
    assert mx.array_equal(got, want).item()
    assert block.moe_topk_calls["fold"] == 1 and ref_kernels["fold"] == 1
    assert ref_kernels["shared" if routed == "gate_up_down_shared" else "routed"] == 1


def test_b1_topk_fold_declines_without_the_routed_down(monkeypatch, ref_kernels):
    block = _block(monkeypatch)
    block.set_moe_topk_mode("fold")
    block(_x(13, (1, 1, H)))
    assert block.moe_topk_fallbacks == 1 and "gate_up_down" in block.moe_topk_last_fallback
    monkeypatch.setattr(QN, "_MOE_GATE_COMPILE", True)
    block.set_moe_topk_mode("launch")
    block(_x(13, (1, 1, H)))
    assert block.moe_topk_fallbacks == 2 and "compiled router" in block.moe_topk_last_fallback


def test_fold_above_its_row_limit_takes_the_launch(monkeypatch, ref_kernels):
    block = _block(monkeypatch)
    block.set_moe_window_consumers({"batch_decode"})
    block.set_moe_topk_mode("fold")
    previous = W.topk_fold_max_rows()
    W.set_topk_fold_max_rows(3)
    try:
        block(_x(14, (4, 1, H)))
        block(_x(15, (3, 1, H)))
    finally:
        W.set_topk_fold_max_rows(previous)
    assert block.moe_topk_calls == {"launch": 1, "fold": 1}


# ----------------------------------------------------------------- policy


def test_policy_defaults_and_receipts():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default = FlashNextPolicy()
    for name in ("moe_window_row_exact", "moe_window_batch_decode", "moe_window_verify"):
        assert name not in default.as_dict()
    assert W.WINDOW_ENV not in default.environment()
    assert default.as_dict()["moe_topk_fold"] == "launch"
    assert default.environment()[W.TOPK_ENV] == "launch"
    off = FlashNextPolicy.from_mapping({"moe_topk_fold": "off"})
    assert "moe_topk_fold" not in off.as_dict()
    assert W.TOPK_ENV not in off.environment()
    chosen = FlashNextPolicy.from_mapping({
        "moe_window_batch_decode": True, "moe_window_verify": True, "moe_topk_fold": "fold",
    })
    assert chosen.environment()[W.WINDOW_ENV] == "batch_decode,verify"
    assert chosen.environment()[W.TOPK_ENV] == "fold"
    assert chosen.as_dict()["moe_topk_fold"] == "fold"
    # The row-exact consumer is inert without the route, and follows it.
    inert = FlashNextPolicy.from_mapping({"moe_window_row_exact": True})
    assert W.WINDOW_ENV not in inert.environment()
    assert "moe_window_row_exact" not in inert.as_dict()
    both = FlashNextPolicy.from_mapping({"row_exact_verify": True})
    assert both.environment()[W.WINDOW_ENV] == "row_exact"
    assert both.as_dict()["moe_window_row_exact"] is True
    explicit_off = FlashNextPolicy.from_mapping(
        {"row_exact_verify": True, "moe_window_row_exact": False}
    )
    assert W.WINDOW_ENV not in explicit_off.environment()
    assert FlashNextPolicy.from_mapping(explicit_off.as_dict()) == explicit_off
    with pytest.raises(ValueError, match="moe_topk_fold"):
        FlashNextPolicy.from_mapping({"moe_topk_fold": "on"})
    with pytest.raises(ValueError, match="boolean"):
        FlashNextPolicy.from_mapping({"moe_window_verify": 1})


def test_verify_scope_nests_and_resets():
    assert not verify_scope.active()
    with verify_scope.verify_forward():
        assert verify_scope.active()
        with verify_scope.verify_forward():
            assert verify_scope.active()
        assert verify_scope.active()
    assert not verify_scope.active()


def test_batch_decode_above_its_row_cap_declines_quietly(monkeypatch, ref_kernels):
    block = _block(monkeypatch)
    block.set_moe_window_consumers({"batch_decode"})
    block(_x(16, (W.batch_decode_max_rows() + 1, 1, H)))
    assert sum(block.moe_window_calls.values()) == 0
    assert sum(block.moe_window_fallbacks.values()) == 0
    with pytest.raises(ValueError):
        W.set_batch_decode_max_rows(1)
