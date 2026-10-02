"""The NAX MoE gather runs in prefill forwards only (Codex port review
2026-10-02 item 7).

Admission used to look at the (padded) assignment count alone.  Flash-Next
has 512 experts and top-10 routing: 32 one-token decode lanes give 320
sorted rows, and the adaptive/always rhs pad lifts them to MLX's streaming
floor of 2048 rows, which passes the NAX floor -- NAX in ordinary decode,
against the port's prefill-only contract.  The phase is now explicit: the
trunk forward opens a prefill scope only for a prefill forward (more than
one row per lane, not a verify window, no speculating cache, no prepared
segmented verify block), and the MoE call sites admit NAX only inside it.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime import verify_scope
from mlx2.runtime.models import moe_nax_gather as nax
from mlx2.runtime.models import qwen3_next as qn
from mlx2.runtime.models import switch_layers as sl

EXPERTS, TOP_K, LANES = 512, 10, 32


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    old = nax.MODE
    nax.status(reset=True)
    yield
    nax.set_mode(old)
    nax.status(reset=True)


def _flash_shaped_linear():
    lin = sl.SwitchLinear(64, 32, EXPERTS, bias=False)
    return lin.to_quantized(group_size=64, bits=4)


def _decode_rows():
    """32 one-token lanes, top-10 of 512 experts: 320 sorted rows."""
    mx.random.seed(3)
    x = mx.random.normal((LANES * TOP_K, 1, 64)).astype(mx.bfloat16)
    idx = mx.sort(mx.random.randint(0, EXPERTS, (LANES * TOP_K,)).astype(mx.uint32))
    return x, idx


@pytest.mark.parametrize("policy", ["always", "adaptive"])
def test_padded_decode_never_consults_nax(monkeypatch, policy):
    """The combined NAX + adaptive pad selection at 32 one-token lanes."""
    nax.set_mode("fused")
    monkeypatch.setattr(sl, "_RHS_PAD_POLICY", policy)
    monkeypatch.setattr(sl, "_RHS_PAD_MIN_ROWS_PER_EXPERT", 3)
    lin = _flash_shaped_linear()
    if policy == "adaptive":
        key = (lin.mode, lin.bits, lin.group_size, EXPERTS, lin.output_dims, lin.input_dims)
        # A cost model under which the rhs pad wins at this width.
        monkeypatch.setitem(sl._PAD_COST_MODELS, key, (0.0, 1e9, 0.0, 0.0))
    pads = []
    real_pad = sl._pad_sorted_tail
    monkeypatch.setattr(sl, "_pad_sorted_tail",
                        lambda x, i, p: pads.append(p) or real_pad(x, i, p))
    seen = []
    monkeypatch.setattr(nax, "try_gather", lambda x, layer, rhs: seen.append(rhs.size) or None)
    x, idx = _decode_rows()
    mx.eval(lin(x, idx, sorted_indices=True))
    # The pad did lift the call to the floor NAX would admit ...
    assert pads and LANES * TOP_K + pads[0] == 4 * EXPERTS
    assert nax.rows_ok(LANES * TOP_K + pads[0], EXPERTS)
    # ... but a decode forward never consults the NAX kernel.
    assert seen == []
    assert nax.status()["not_prefill"] >= 1
    # The same call inside a prefill forward does.
    with nax.prefill_scope():
        mx.eval(lin(x, idx, sorted_indices=True))
    assert seen == [4 * EXPERTS]


def test_fused_swiglu_candidates_are_prefill_only():
    nax.set_mode("fused")
    m = qn.FusedGateUpSwitchGLU(64, 64, 8)
    m.gate_up_proj = m.gate_up_proj.to_quantized(group_size=64, bits=4)
    m.down_proj = m.down_proj.to_quantized(group_size=64, bits=4)
    m.eval()
    assert not qn._nax_swiglu_candidate(m)
    with nax.prefill_scope():
        assert qn._nax_swiglu_candidate(m)
    split = sl.SwitchGLU(64, 64, 8)
    split.gate_proj = split.gate_proj.to_quantized(group_size=64, bits=4)
    split.up_proj = split.up_proj.to_quantized(group_size=64, bits=4)
    split.eval()
    assert not qn._nax_split_swiglu_candidate(split)
    with nax.prefill_scope():
        assert qn._nax_split_swiglu_candidate(split)


class _Cache:
    def __init__(self, speculating=False, step_lengths=None):
        self.speculating = speculating
        self._step_lengths = step_lengths


def test_forward_phase_decision():
    nax.set_mode("fused")
    prompt = mx.zeros((1, 512), mx.uint32)
    assert nax.is_prefill_forward(prompt, [_Cache()])
    assert nax.is_prefill_forward(mx.zeros((4, 64), mx.uint32), None)
    # Decode: one row per lane, at any batch width (padding is irrelevant).
    assert not nax.is_prefill_forward(mx.zeros((LANES, 1), mx.uint32), [_Cache()])
    assert not nax.is_prefill_forward(mx.zeros((1, 1), mx.uint32), [_Cache()])
    # Verify windows: the verify scope, a speculating cache, or a prepared
    # segmented verify block (step lengths) -- whatever the row count.
    rows = mx.zeros((LANES, 3), mx.uint32)
    with verify_scope.verify_forward():
        assert not nax.is_prefill_forward(rows, [_Cache()])
    assert not nax.is_prefill_forward(rows, [_Cache(speculating=True)])
    assert not nax.is_prefill_forward(rows, [_Cache(step_lengths=[3] * LANES)])


def _stub_trunk(record):
    """A Qwen4ExpTextModel stand-in whose one layer records the phase."""
    from mlx2.runtime.models import qwen4_exp

    class Layer:
        ple = None

        def __call__(self, hidden, inputs, fa_mask, cache, ssm_mask):
            record.append(nax.prefill_active())
            return hidden

    stub = SimpleNamespace(
        args=SimpleNamespace(hc_count=1),
        embed_tokens=lambda ids: mx.zeros(ids.shape + (8,)),
        layers=[Layer()], fa_idx=None, ssm_idx=None,
        hyper_connection_mixer=lambda h: h,
    )
    stub._forward = lambda *a, **k: qwen4_exp.Qwen4ExpTextModel._forward(stub, *a, **k)
    return lambda ids, cache=None: qwen4_exp.Qwen4ExpTextModel.__call__(stub, ids, cache)


def test_flash_next_trunk_opens_the_prefill_scope_for_prefill_only(monkeypatch):
    from mlx2.runtime.models import qwen4_exp

    monkeypatch.setattr(qwen4_exp, "_SHAPE_STABLE_SHORT_FORWARD", False)
    record = []
    trunk = _stub_trunk(record)
    nax.set_mode("fused")
    trunk(mx.zeros((1, 512), mx.uint32), [_Cache()])
    trunk(mx.zeros((LANES, 1), mx.uint32), [_Cache()])
    trunk(mx.zeros((LANES, 3), mx.uint32), [_Cache(speculating=True)])
    with verify_scope.verify_forward():
        trunk(mx.zeros((LANES, 3), mx.uint32), [_Cache()])
    assert record == [True, False, False, False]
    assert nax.prefill_active() is False  # the scope never leaks out
    # Off: nothing opens a scope.
    nax.set_mode("off")
    trunk(mx.zeros((1, 512), mx.uint32), [_Cache()])
    assert record[-1] is False
