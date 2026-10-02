"""CPU tests for the opt-in slice-invariant prefill lane (MTPLX #549 mechanism).

The bit-level invariance itself is a Metal property (MLX's CPU kernels do
not switch on the row count); scripts/check_invariant_prefill.py is the GPU
oracle.  These tests pin the lane's plumbing: default off, admission and
refusals, scope boundaries, padding geometry, counters and identity.
"""
import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime import row_exact_verify, verify_scope
from mlx2.runtime.models import invariant_prefill as inv
from mlx2.runtime.models import switch_layers
from mlx2.runtime.prefill_plan import execution_identity


@pytest.fixture(autouse=True)
def _reset_counters():
    inv.status(reset=True)
    yield
    inv.status(reset=True)


def _quantized(inp=128, out=48, bits=4, mode="affine"):
    linear = nn.Linear(inp, out, bias=False)
    return nn.QuantizedLinear.from_linear(linear, group_size=64, bits=bits, mode=mode) if mode == "affine" else \
        nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4, mode=mode)


class Trunk(nn.Module):
    """A toy trunk: one quantized and one dense projection; records the scope."""

    def __init__(self):
        super().__init__()
        self.proj = _quantized()
        self.gate = nn.Linear(128, 4, bias=False)
        self.seen = []

    def __call__(self, inputs, cache=None, extra=None):
        self.seen.append(inv.active())
        x = mx.ones((*inputs.shape, 128))
        return self.proj(x) + self.gate(x).sum(-1, keepdims=True)


class SpeculatingCache:
    speculating = True


# --- policy -----------------------------------------------------------------


def test_policy_default_off_and_absent_from_receipts():
    policy = FlashNextPolicy()
    assert policy.invariant_prefill is False
    assert "invariant_prefill" not in policy.as_dict()
    assert "invariant_prefill" not in str(policy.environment())


def test_policy_enabled_is_recorded_not_an_environment_switch():
    policy = FlashNextPolicy.from_mapping({"invariant_prefill": True})
    assert policy.as_dict()["invariant_prefill"] is True
    assert policy.environment() == FlashNextPolicy().environment()


@pytest.mark.parametrize(
    "clash",
    [
        {"tensorfold_prefill": True},
        {"tensorfold_qmv_rows": True},
        {"gdn_prefill_chunk": 16},
        {"gdn_core": True},
        {"row_exact_verify": True},
    ],
)
def test_policy_refuses_mechanisms_that_own_prefill_bits(clash):
    with pytest.raises(ValueError, match="invariant_prefill cannot be combined"):
        FlashNextPolicy.from_mapping({"invariant_prefill": True, **clash})


def test_policy_rejects_non_boolean():
    with pytest.raises(ValueError, match="invariant_prefill must be boolean"):
        FlashNextPolicy.from_mapping({"invariant_prefill": 1})


# --- coverage / install -----------------------------------------------------


def test_coverage_accepts_affine_and_small_dense():
    assert inv.coverage_refusal(Trunk()) is None


def test_coverage_refuses_non_affine_quantized():
    root = nn.Module()
    root.p = _quantized(mode="mxfp4")
    assert inv.coverage_refusal(root).startswith("quantized_mode:mxfp4")


def test_coverage_refuses_unknown_projection_subclass():
    class Wrapped(nn.Linear):
        pass

    root = nn.Module()
    root.p = Wrapped(8, 8)
    assert inv.coverage_refusal(root).startswith("unsupported_linear:Wrapped")


def test_coverage_refuses_large_dense(monkeypatch):
    monkeypatch.setattr(inv, "DENSE_STACK_LIMIT_BYTES", 16)
    root = nn.Module()
    root.p = nn.Linear(8, 8)
    assert inv.coverage_refusal(root).startswith("dense_too_large:p")


def test_coverage_refuses_few_experts_and_dense_switch():
    small = switch_layers.SwitchLinear(64, 32, 8, bias=False).to_quantized(64, 4)
    root = nn.Module()
    root.e = small
    assert inv.coverage_refusal(root).startswith("switch_experts:8")
    dense = nn.Module()
    dense.e = switch_layers.SwitchLinear(64, 32, 16, bias=False)
    assert inv.coverage_refusal(dense).startswith("dense_switch")


def test_install_is_all_or_nothing():
    trunk = Trunk()
    trunk.bad = _quantized(mode="mxfp4")
    handle = inv.install(trunk)
    assert not handle.installed and handle.refusal.startswith("quantized_mode")
    assert type(trunk.proj) is nn.QuantizedLinear
    assert type(trunk).__name__ == "Trunk"


def test_install_swaps_classes_without_touching_parameters():
    trunk = Trunk()
    before = {k: v for k, v in nn.utils.tree_flatten(trunk.parameters())}
    handle = inv.install(trunk)
    assert handle.installed
    assert handle.report == {"quantized_linears": 1, "dense_linears": 1}
    assert type(trunk.proj) is inv.InvariantQuantizedLinear
    assert type(trunk.gate) is inv.InvariantLinear
    after = {k: v for k, v in nn.utils.tree_flatten(trunk.parameters())}
    assert before.keys() == after.keys()
    assert all(before[k] is after[k] for k in before)
    assert inv.install(trunk) is handle
    handle.uninstall()
    assert type(trunk.proj) is nn.QuantizedLinear and type(trunk.gate) is nn.Linear
    assert type(trunk) is Trunk


# --- scope ------------------------------------------------------------------


def test_scope_opens_only_for_prefill_forwards():
    trunk = Trunk()
    handle = inv.install(trunk)
    trunk(mx.zeros((1, 1), mx.uint32))  # decode
    trunk(mx.zeros((4, 1), mx.uint32))  # batched decode
    trunk(mx.zeros((1, 7), mx.uint32))  # prefill
    with verify_scope.verify_forward():
        trunk(mx.zeros((1, 3), mx.uint32))
    with row_exact_verify.window(row_exact_verify.Window(3)):
        trunk(mx.zeros((1, 3), mx.uint32))
    trunk(mx.zeros((1, 3), mx.uint32), [SpeculatingCache()])
    handle.enabled = False
    trunk(mx.zeros((1, 9), mx.uint32))
    assert trunk.seen == [False, False, True, False, False, False, False]
    counts = inv.status()["counts"]
    assert counts["forwards"] == 1 and counts["rows"] == 7
    assert counts["declined:verify_scope"] == 1
    assert counts["declined:row_exact_window"] == 1
    assert counts["declined:speculating"] == 1
    assert not inv.active()


def test_lane_rows_match_stock_projection_on_cpu():
    trunk = Trunk()
    x = mx.random.normal((2, 5, 128))
    stock_q, stock_d = trunk.proj(x), trunk.gate(x)
    inv.install(trunk)
    with inv.scope():
        lane_q, lane_d = trunk.proj(x), trunk.gate(x)
    assert lane_q.shape == stock_q.shape and lane_d.shape == stock_d.shape
    assert mx.allclose(lane_q, stock_q, atol=1e-4).item()
    assert mx.allclose(lane_d, stock_d, atol=1e-4).item()
    # Outside the scope the swapped classes are the stock arithmetic.
    assert mx.array_equal(trunk.proj(x), stock_q).item()
    assert mx.array_equal(trunk.gate(x), stock_d).item()


# --- geometry ---------------------------------------------------------------


@pytest.mark.parametrize("rows,per_batch", [(1, 33), (10, 33), (66, 33), (67, 34), (100, 50), (2048, 1024)])
def test_two_batches_of_at_least_33_rows(rows, per_batch):
    batched, n = inv._batch_rows(mx.zeros((rows, 64)))
    assert batched.shape == (2, per_batch, 64) and n == rows


def test_quantized_matmul_uses_broadcast_batched_weights(monkeypatch):
    seen = {}
    real = mx.quantized_matmul

    def spy(x, w, scales, biases=None, **kw):
        seen["x"], seen["w"] = x.shape, w.shape
        return real(x, w, scales, biases, **kw)

    monkeypatch.setattr(mx, "quantized_matmul", spy)
    q = _quantized()
    out = inv.quantized_matmul(mx.ones((1, 3, 128)), q["weight"], q["scales"], q["biases"],
                               group_size=64, bits=4)
    assert out.shape == (1, 3, 48)
    assert seen["x"] == (2, 33, 128) and seen["w"][0] == 2


def test_dense_stack_has_a_nonzero_batch_stride():
    w = mx.random.normal((4, 32))
    stacked = inv.stack_dense(w)
    assert stacked.shape == (2, 32, 4)
    one = inv.stack_dense(mx.random.normal((1, 32)))
    assert one.shape == (2, 32, 2) and not mx.any(one[..., 1]).item()
    assert mx.array_equal(stacked[1], w.T).item()


def test_sorted_gather_pad_reaches_the_streaming_floor():
    assert inv.sorted_gather_pad(640, 512) == 2048 - 640
    assert inv.sorted_gather_pad(4000, 512) == 0
    assert inv.sorted_gather_pad(3, 2) == 13  # B >= 16
    assert inv.status()["counts"]["moe_gathers"] == 3


def test_switch_linear_pads_only_inside_the_lane(monkeypatch):
    monkeypatch.setattr(switch_layers, "_RHS_PAD_MIN_ROWS_PER_EXPERT", 0)  # floor kill switch
    layer = switch_layers.SwitchLinear(64, 32, 16, bias=False).to_quantized(64, 4)
    x = mx.random.normal((20, 1, 64))
    idx = mx.sort(mx.random.randint(0, 16, (20,)).astype(mx.uint32))
    calls = []
    real = mx.gather_qmm

    def spy(x, *a, rhs_indices=None, **kw):
        calls.append(int(rhs_indices.size))
        return real(x, *a, rhs_indices=rhs_indices, **kw)

    monkeypatch.setattr(mx, "gather_qmm", spy)
    stock = layer(x, idx, sorted_indices=True)
    with inv.scope():
        lane = layer(x, idx, sorted_indices=True)
    assert calls == [20, 64]
    assert lane.shape == stock.shape == (20, 1, 32)
    assert mx.allclose(lane, stock, atol=1e-3).item()
    assert inv.status()["counts"]["moe_pad_rows"] == 44


def test_switch_glu_always_sorts_in_the_lane(monkeypatch):
    sorted_calls = []
    real = switch_layers._gather_sort

    def spy(x, indices):
        sorted_calls.append(indices.size)
        return real(x, indices)

    monkeypatch.setattr(switch_layers, "_gather_sort", spy)
    glu = switch_layers.SwitchGLU(64, 64, 16)
    nn.quantize(glu, group_size=64, bits=4)
    x = mx.random.normal((1, 2, 64))
    idx = mx.array([[[1, 3]], [[2, 5]]]).reshape(1, 2, 2).astype(mx.uint32)
    glu(x, idx)
    assert sorted_calls == []
    with inv.scope():
        glu(x, idx)
    assert sorted_calls == [4]


# --- attention --------------------------------------------------------------


def test_sdpa_refusals():
    q = mx.zeros((1, 4, 3, 256))
    k = v = mx.zeros((1, 2, 20, 256))
    assert inv.sdpa_refusal(q, k, v, None) is None
    assert inv.sdpa_refusal(q, k, v, mx.ones((1, 1, 3, 20), mx.bool_)) is None
    assert inv.sdpa_refusal(q, (k, k), v, None) == "quantized_kv"
    assert inv.sdpa_refusal(mx.zeros((1, 4, 3, 48)), mx.zeros((1, 2, 20, 48)),
                            mx.zeros((1, 2, 20, 48)), None) == "head_dim"
    assert inv.sdpa_refusal(q, mx.zeros((1, 2, 5, 256)), mx.zeros((1, 2, 5, 256)),
                            "causal") == "causal_short_keys"
    assert inv.sdpa_refusal(q, k, v, mx.ones((3, 20), mx.bool_)) == "mask_rank"


def test_sdpa_pads_short_queries_in_front_and_forces_the_fused_kernel(monkeypatch):
    seen = {}

    def fake(q, k, v, *, scale, mask=None, force_fused=False):
        seen.update(q=q.shape, mask=mask.shape, force=force_fused)
        seen["first_mask_row_copied"] = bool(mx.array_equal(mask[..., 0, :], mask[..., 6, :]).item())
        return q

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", fake)
    q = mx.random.normal((1, 4, 3, 64))
    k = v = mx.zeros((1, 2, 20, 64))
    mask = mx.random.uniform(shape=(1, 1, 3, 20)) > 0.5
    out = inv.sdpa(q, k, v, scale=0.1, mask=mask)
    assert seen == {"q": (1, 4, 9, 64), "mask": (1, 1, 9, 20), "force": True,
                    "first_mask_row_copied": True}
    assert mx.array_equal(out, q).item()  # the real rows, in order
    assert inv.status()["counts"]["sdpa_pad_rows"] == 6


# --- identity / counters ----------------------------------------------------


def test_execution_identity_binds_the_lane_law():
    assert execution_identity() is None
    handle = inv.install(Trunk())
    identity = execution_identity(None, None, handle.identity())
    assert identity["invariant"]["schema"] == inv.SCHEMA
    assert identity["invariant"]["law"]["min_batch_rows"] == 33


def test_not_invariant_reasons_are_bounded_and_counted():
    for i in range(40):
        inv.not_invariant(f"r{i}")
    report = inv.status()
    assert report["not_invariant"] == 40
    assert report["counts"]["not_invariant:other"] == 40 - inv._REASON_LIMIT


def test_handle_status_reports_refusal_or_report():
    bad = nn.Module()
    bad.p = _quantized(mode="mxfp4")
    status = inv.install(bad).status()
    assert status["installed"] is False and status["reason"].startswith("quantized_mode")
    good = inv.install(Trunk()).status()
    assert good["installed"] is True and good["report"]["quantized_linears"] == 1


def test_swapped_classes_keep_the_plain_decode_kernel_admission():
    from mlx2.runtime.models import qwen4_hc_decode, qwen4_routed_decode

    trunk = Trunk()
    inv.install(trunk)
    assert qwen4_hc_decode._plain_class(trunk.proj) is nn.QuantizedLinear
    assert qwen4_hc_decode._plain_class(trunk.gate) is nn.Linear
    reason = qwen4_routed_decode._linear_ok(trunk.proj, 4, 48, 128, 64)
    assert reason != "not a plain QuantizedLinear"
