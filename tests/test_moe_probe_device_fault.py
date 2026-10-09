"""A Metal device fault during an MoE NAX gather self-test or the fused
router probe is not a decline (sweep 2026-10-08 review round 1, the
siblings of the indexed-QSA / fused-GDN probe fixes).

Serving recovers out-of-memory and GPU-timeout step failures
(``serving.device_fault_kind``).  The NAX canaries (default-on on M5, shared
by Flash-Next and Qwen3.6-35B) turned every exception into ``False``, which
``_checked`` cached for the life of the process, and the router probe marked
itself complete before it evaluated and swallowed the fault.  Kernel refusals
keep their decline-and-cache behaviour (the controls).  CPU only: the Metal
launch is replaced by the stock ``mx.gather_qmm`` arithmetic it is canaried
against, so a self-test that runs to the end passes.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models import moe_nax_gather as nax
from mlx2.runtime.models import qwen4_moe_router as router
from mlx2.runtime.models import switch_layers as sl
from mlx2.runtime.models.served_exp import is_device_fault
from mlx2.serving import SIMULATED_GPU_TIMEOUT

OOM = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)
REFUSAL = "Unable to build metal library from source"
FAULTS = pytest.mark.parametrize(
    "fault", [OOM, SIMULATED_GPU_TIMEOUT], ids=["oom", "gpu_timeout"]
)
E, D, M = 8, 128, 64


def test_fault_texts_are_device_faults():
    assert is_device_fault(RuntimeError(OOM))
    assert is_device_fault(RuntimeError(SIMULATED_GPU_TIMEOUT))
    assert not is_device_fault(RuntimeError(REFUSAL))


# --- NAX sorted gather self-tests -------------------------------------------


@pytest.fixture(autouse=True)
def _nax_state(monkeypatch):
    saved = dict(nax._verified)
    nax._verified.clear()
    nax.status(reset=True)
    monkeypatch.setattr(nax, "_nax_host", True)
    monkeypatch.setattr(nax.mx, "default_device", lambda: nax.mx.gpu)
    with nax.prefill_scope():
        yield
    nax._verified.clear()
    nax._verified.update(saved)
    nax.status(reset=True)


def _stock_launch(x, w, scales, biases, idx, group_size, bits, mode, plan,
                  stream, epi=0, limit=None, row_map=None, up=None):
    """What each kernel instantiation computes, in stock MLX ops."""
    if row_map is not None:
        x = x[row_map]

    def proj(wt, s, b):
        return mx.gather_qmm(
            x, wt, s, b, rhs_indices=idx, transpose=True, group_size=group_size,
            bits=bits, mode=mode, sorted_indices=True,
        )

    if up is not None:
        return nax.reference_activation(proj(*up), proj(w, scales, biases), limit)
    y = proj(w, scales, biases)
    if not epi:
        return y
    gate, up_rows = mx.split(y, 2, axis=-1)
    return nax.reference_activation(up_rows, gate, limit)


# Which self-test the fault lands in: the first launch matching the predicate.
SELF_TESTS = {
    "gather": lambda kw: True,  # _self_test
    "swiglu": lambda kw: kw.get("epi") and kw.get("row_map") is None,  # _self_test_act
    "swiglu_map": lambda kw: kw.get("row_map") is not None and kw.get("up") is None,
    "swiglu_split": lambda kw: kw.get("up") is not None and kw.get("row_map") is None,
    "swiglu_split_map": lambda kw: kw.get("up") is not None
    and kw.get("row_map") is not None,
}


def _launch_faulting_once(monkeypatch, error, when):
    launches = []
    pending = [True]

    def launch(*args, **kwargs):
        launches.append(kwargs)
        if pending[0] and when(kwargs):
            pending[0] = False
            raise RuntimeError(error)
        return _stock_launch(*args, **kwargs)

    monkeypatch.setattr(nax, "_launch", launch)
    return launches


def _qlinear(out=128):
    lin = sl.SwitchLinear(D, out, E, bias=False).to_quantized(group_size=64, bits=4)
    lin.scales = lin.scales.astype(mx.bfloat16)
    lin.biases = lin.biases.astype(mx.bfloat16)
    return lin


def _rows():
    x_tok = mx.random.normal((M // 2, 1, D), key=mx.random.key(3)).astype(mx.bfloat16)
    row_map = ((mx.arange(M) * 5 + 1) % (M // 2)).astype(mx.uint32)
    idx = mx.sort(mx.random.randint(0, E, (M,), key=mx.random.key(4))).astype(mx.uint32)
    return x_tok[row_map], idx, (x_tok, row_map)


def _route(name):
    x, idx, token_rows = _rows()
    if name == "gather":
        lin = _qlinear()
        return lambda: nax.try_gather(x, lin, idx)
    if name in ("swiglu", "swiglu_map"):
        proj = _qlinear()
        rows = token_rows if name == "swiglu_map" else None
        return lambda: nax.try_swiglu(proj, x, idx, rows)
    gate, up = _qlinear(out=64), _qlinear(out=64)
    rows = token_rows if name == "swiglu_split_map" else None
    return lambda: nax.try_swiglu_split(gate, up, x, idx, rows)


@FAULTS
@pytest.mark.parametrize("route", list(SELF_TESTS))
def test_nax_self_test_reraises_device_fault_and_canaries_again(
    monkeypatch, route, fault
):
    _launch_faulting_once(monkeypatch, fault, SELF_TESTS[route])
    call = _route(route)
    with pytest.raises(RuntimeError, match="Command buffer execution failed"):
        call()
    # Nothing was decided: no cached verdict, no counted decline.
    assert False not in nax._verified.values()
    assert nax.status()["fallbacks"] == {}
    # Serving recovered the step; the next call runs the canary again.
    out = call()
    assert out is not None
    st = nax.status()
    assert st["calls"][route] == 1 and st["fallbacks"] == {}
    assert nax._verified and all(nax._verified.values())


def test_nax_self_test_still_caches_a_kernel_refusal(monkeypatch):
    launches = _launch_faulting_once(monkeypatch, REFUSAL, SELF_TESTS["gather"])
    call = _route("gather")
    assert call() is None
    assert list(nax._verified.values()) == [False]
    assert nax.status()["fallbacks"] == {"kernel_unverified": 1}
    seen = len(launches)
    assert call() is None  # cached: the canary does not run again
    assert len(launches) == seen
    assert nax.status()["fallbacks"] == {"kernel_unverified": 2}


def test_nax_canary_passes_with_the_stock_launch(monkeypatch):
    monkeypatch.setattr(nax, "_launch", _stock_launch)
    for route in SELF_TESTS:
        assert _route(route)() is not None, route
    assert all(nax._verified.values())


# --- fused MoE router probe -------------------------------------------------


@pytest.fixture
def router_probe(monkeypatch):
    monkeypatch.setattr(router, "_PROBE_COMPLETE", False)
    monkeypatch.setattr(router, "_PROBE_OK", False)
    monkeypatch.setattr(router.mx.metal, "is_available", lambda: True)
    calls = []
    state = {"error": None}

    def kernel(gates, **kwargs):
        calls.append(gates.dtype)
        if state["error"] is not None:
            error, state["error"] = state["error"], None
            raise error
        top = mx.array([[list(range(502, 512))]], dtype=mx.uint32)
        return top, mx.full((1, 1, router.TOP_K), 0.1, dtype=gates.dtype)

    monkeypatch.setattr(router, "qwen4_moe_router", kernel)
    return calls, state


@FAULTS
def test_router_probe_reraises_device_fault_and_probes_again(router_probe, fault):
    calls, state = router_probe
    state["error"] = RuntimeError(fault)
    with pytest.raises(RuntimeError, match="Command buffer execution failed"):
        router.probe_qwen4_moe_router()
    assert router._PROBE_COMPLETE is False
    assert router.probe_qwen4_moe_router() is True
    assert router.probe_qwen4_moe_router() is True
    assert len(calls) == 2  # probed again once, then cached


@pytest.mark.parametrize("error", [RuntimeError(REFUSAL), ValueError("not eligible")])
def test_router_probe_still_caches_a_kernel_refusal(router_probe, error):
    calls, state = router_probe
    state["error"] = error
    assert router.probe_qwen4_moe_router() is False
    assert router._PROBE_COMPLETE is True
    assert router.probe_qwen4_moe_router() is False
    assert len(calls) == 1


def test_router_probe_is_not_completed_by_a_call_it_did_not_evaluate(router_probe):
    calls, _ = router_probe
    assert router.probe_qwen4_moe_router(mx.float16) is False
    assert calls == []
    assert router.probe_qwen4_moe_router(mx.bfloat16) is True
    assert calls == [mx.bfloat16]
