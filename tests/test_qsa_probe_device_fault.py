"""A Metal device fault during an indexed-QSA, NAX or fused-merge probe is
not a decline (follow-up to sweep 2026-10-02 G3).

Serving recovers out-of-memory and GPU-timeout step failures by failing the
lanes stepped in the broken command buffer and rebuilding the generator
(``serving.device_fault_kind``).  The indexed-QSA first-use probe evaluates
the whole step graph upstream of its candidate, so a fault there belongs to
the step.  The probes swallowed it: the candidate ladder cached the geometry
as ``probe_declined`` for the life of the process, the attention wrappers
turned it into a gather fallback, the NAX probe cached ``kernel_unavailable``
and the fused merge declined forever.  Real kernel refusals keep their
decline-and-cache behaviour (the controls below).  CPU only: the Metal
kernels and platform gates are stubbed; the probes and fallbacks are real.
"""

from unittest import mock
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models import qwen4_qsa_indexed as indexed
from mlx2.runtime.models import qwen4_qsa_indexed_merge as merge
from mlx2.runtime.models import qwen4_qsa_nax as nax
from mlx2.runtime.models.qwen4_exp import QSACompactBlocks
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
INDEXED, GATHER = 7.0, -1.0
B, NQH, NKH, D, W, KW = 1, 24, 2, 256, 2048, 263


@pytest.fixture(autouse=True)
def _clean_probe_state():
    saved = dict(merge._PROBE_RESULTS)
    indexed._PROBE_RESULTS.clear()
    indexed._PROBE_TIMINGS.clear()
    indexed.qsa_indexed_status(reset=True)
    merge._PROBE_RESULTS.clear()
    yield
    indexed._PROBE_RESULTS.clear()
    indexed._PROBE_TIMINGS.clear()
    indexed.qsa_indexed_status(reset=True)
    merge._PROBE_RESULTS.clear()
    merge._PROBE_RESULTS.update(saved)


def test_fault_texts_are_device_faults():
    assert is_device_fault(RuntimeError(OOM))
    assert is_device_fault(RuntimeError(SIMULATED_GPU_TIMEOUT))
    assert not is_device_fault(RuntimeError(REFUSAL))


@FAULTS
def test_measure_candidates_reraises_device_fault(fault):
    calls = []

    def dispatch(candidate):
        calls.append(candidate)
        if len(calls) == 1:
            raise RuntimeError(fault)
        return (mx.zeros((2,)), mx.zeros((1,), dtype=mx.int32))

    with pytest.raises(RuntimeError, match="Command buffer execution failed"):
        indexed._measure_candidates([(384, 128, 12), (384, 64, 12)], dispatch)
    assert len(calls) == 1


def test_measure_candidates_still_skips_kernel_refusal():
    def dispatch(candidate):
        if candidate[1] == 128:
            raise RuntimeError(REFUSAL)
        return (mx.zeros((2,)), mx.zeros((1,), dtype=mx.int32))

    (selected, _, _, timings) = indexed._measure_candidates(
        [(384, 128, 12), (384, 64, 12)], dispatch
    )
    assert selected == (384, 64, 12)
    assert list(timings) == [(64, 12)]


def _inputs():
    q = mx.zeros((B, NQH, 1, D))
    k = mx.zeros((B, NKH, W, D))
    compact = QSACompactBlocks(
        block_ids=mx.arange(KW, dtype=mx.uint32)[None, None],
        block_counts=mx.array([[KW]], dtype=mx.int32),
        tail_start=mx.array([[W]], dtype=mx.int32),
        tail_stop=mx.array([[W]], dtype=mx.int32),
        left_padding=mx.array([0], dtype=mx.int32),
        block_size=4,
        physical_width=W,
        causal_mask=None,
    )
    return (q, k, k, compact)


def _route(fault_text, gather_calls):
    """Patch the platform gates and both Metal kernels; the probe stays real."""

    def partition(*args, **kwargs):
        if fault_text:
            raise RuntimeError(fault_text)
        z = mx.zeros((1,))
        return (z, z, z, mx.zeros((1,), dtype=mx.int32))

    def combine(m, l, o, engaged, *, output_dtype, output_gate=None):
        return (mx.full((B, NQH, 1, D), INDEXED), engaged)

    def gather(*args, **kwargs):
        gather_calls.append(1)
        return mx.full((B, NQH, 1, D), GATHER)

    return [
        mock.patch.object(indexed, "indexed_kernel_available", lambda: True),
        mock.patch.object(indexed, "_sdpa_header_state", lambda: (True, None, None)),
        mock.patch.object(
            indexed.mx, "device_info", lambda: {"architecture": "applegpu_g15s"}
        ),
        mock.patch.dict(
            "os.environ", {"MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX": "1"}
        ),
        mock.patch.object(indexed, "_partition_dispatch", partition),
        mock.patch.object(indexed, "_combine_sdpa_partials", combine),
        mock.patch.object(qwen4_exp, "_gather_qsa_attention", gather),
    ]


def _attend(patches):
    (q, k, v, compact) = _inputs()
    for active in patches:
        active.start()
    try:
        out = qwen4_exp._indexed_qsa_attention_or_gather(
            q, k, v, compact, scale=D**-0.5, splits=None, tile_rows=8
        )
        mx.eval(out)
        return float(out.reshape(-1)[0].item())
    finally:
        for active in reversed(patches):
            active.stop()


@FAULTS
def test_probe_fault_propagates_and_is_probed_again(fault):
    gather_calls = []
    # The first probe's command buffer fails on every candidate (memory is
    # still held until serving's recovery releases the failed step) ...
    with pytest.raises(RuntimeError, match="Command buffer execution failed"):
        _attend(_route(fault, gather_calls))
    assert gather_calls == [], "a device fault must not become a gather fallback"
    assert False not in indexed._PROBE_RESULTS.values(), (
        "a device fault must not be cached as a declined candidate ladder"
    )
    counts = indexed.qsa_indexed_status()["counts"]
    assert counts.get("dispatch_raised", 0) == 0
    assert counts.get("probe_declined", 0) == 0
    # ... and once serving has rebuilt, the next step probes again and engages.
    assert _attend(_route(None, gather_calls)) == INDEXED
    assert gather_calls == []


def test_probe_still_declines_and_caches_kernel_refusals():
    gather_calls = []
    assert _attend(_route(REFUSAL, gather_calls)) == GATHER
    assert list(indexed._PROBE_RESULTS.values()) == [False]
    assert indexed.qsa_indexed_status()["counts"]["probe_declined"] == 1


@FAULTS
def test_cached_route_fault_is_not_a_gather_fallback(fault):
    gather_calls = []
    (q, k, v, compact) = _inputs()
    with mock.patch.object(
        qwen4_exp, "qwen4_qsa_indexed_attention", side_effect=RuntimeError(fault)
    ), mock.patch.object(
        qwen4_exp, "_gather_qsa_attention", lambda *a, **k: gather_calls.append(1)
    ):
        with pytest.raises(RuntimeError, match="Command buffer execution failed"):
            qwen4_exp._indexed_qsa_attention_or_gather(
                q, k, v, compact, scale=D**-0.5, splits=None, tile_rows=8
            )
    assert gather_calls == []


@FAULTS
def test_quantized_route_fault_is_not_a_gather_fallback(fault):
    gather_calls = []
    (q, k, v, compact) = _inputs()
    with mock.patch.object(
        qwen4_exp,
        "qwen4_qsa_indexed_quantized_attention",
        side_effect=RuntimeError(fault),
    ), mock.patch.object(
        qwen4_exp,
        "_gather_qsa_quantized_attention",
        lambda *a, **k: gather_calls.append(1),
    ):
        with pytest.raises(RuntimeError, match="Command buffer execution failed"):
            qwen4_exp._indexed_qsa_quantized_attention_or_gather(
                q, k, v, compact, scale=D**-0.5, splits=None, tile_rows=8,
                group_size=64, key_bits=8, value_bits=8,
            )
    assert gather_calls == []


@pytest.fixture
def nax_runtime(monkeypatch):
    monkeypatch.setattr(nax.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(nax.mx, "device_info", lambda: {"device_name": "Apple M5 Max"})
    monkeypatch.setattr(nax, "_NAX_AVAILABLE", None)


@FAULTS
def test_nax_probe_does_not_cache_a_device_fault(nax_runtime, monkeypatch, fault):
    state = {"fault": True}

    def kernel(*a, **k):
        if state["fault"]:
            raise RuntimeError(fault)
        return mx.zeros((1,))

    monkeypatch.setattr(nax, "nax_qsa_attention", kernel)
    with pytest.raises(RuntimeError, match="Command buffer execution failed"):
        nax.nax_kernel_available()
    assert nax._NAX_AVAILABLE is None
    state["fault"] = False
    assert nax.nax_kernel_available() is True


def test_nax_probe_still_caches_kernel_refusals(nax_runtime, monkeypatch):
    def refuse(*a, **k):
        raise RuntimeError(REFUSAL)

    monkeypatch.setattr(nax, "nax_qsa_attention", refuse)
    assert nax.nax_kernel_available() is False
    assert nax._NAX_AVAILABLE is False


def _partials():
    m = mx.full((1, 1, 1, 128), -mx.inf, dtype=mx.float32)
    return (m, mx.zeros_like(m), mx.zeros((1, 1, 1, 128, 256), dtype=mx.float32))


@FAULTS
def test_fused_merge_ladder_reraises_device_fault_and_probes_again(
    monkeypatch, fault
):
    (m, l, o) = _partials()
    state = {"fault": True}

    def candidate(*a, **k):
        if state["fault"]:
            raise RuntimeError(fault)
        return mx.zeros((1, 1, 1, 256), dtype=mx.float32)

    monkeypatch.setattr(merge, "_dispatch_candidate", candidate)
    with pytest.raises(RuntimeError, match="Command buffer execution failed"):
        merge._fused_merge(m, l, o, output_dtype=mx.float32)
    assert False not in merge._PROBE_RESULTS.values()
    state["fault"] = False
    mx.eval(merge._fused_merge(m, l, o, output_dtype=mx.float32))
    assert list(merge._PROBE_RESULTS.values()) == [merge._THREAD_CANDIDATES[0]]


@FAULTS
def test_fused_merge_fault_is_not_a_sequential_fallback(monkeypatch, fault):
    (m, l, o) = _partials()
    fallbacks = []
    monkeypatch.setattr(merge, "fused_merge_enabled", lambda: True)
    monkeypatch.setattr(merge, "fused_merge_available", lambda: True)

    def candidate(*a, **k):
        raise RuntimeError(fault)

    monkeypatch.setattr(merge, "_dispatch_candidate", candidate)
    with pytest.raises(RuntimeError, match="Command buffer execution failed"):
        merge.combine_indexed_partials(
            m, l, o, output_dtype=mx.float32,
            on_fallback=lambda: fallbacks.append(1),
        )
    assert fallbacks == []


def test_fused_merge_still_declines_kernel_refusals(monkeypatch):
    (m, l, o) = _partials()

    def refuse(*a, **k):
        raise RuntimeError(REFUSAL)

    monkeypatch.setattr(merge, "_dispatch_candidate", refuse)
    with pytest.raises(RuntimeError, match="ladder declined"):
        merge._fused_merge(m, l, o, output_dtype=mx.float32)
    assert list(merge._PROBE_RESULTS.values()) == [False]


def _segmented_private_delta():
    from test_batched_mtp import _tiny_qwen4_model
    from mlx2.runtime.models.qwen4_exp import QSAKVCache
    from mlx2.runtime.segmented_batch_cache import SegmentedBatchQSAKVCache
    from mlx2.runtime.segmented_self_mtp import note_segmented_self_mtp

    model = _tiny_qwen4_model()
    attention = model.language_model.model.layers[1].self_attn
    prefix = mx.random.normal((1, 12, 32), key=mx.random.key(101))
    prefix_mask = (mx.arange(12)[:, None] >= mx.arange(12)[None, :])[None, None]
    rows = []
    for _ in range(2):
        cache = QSAKVCache(attention.indexer.summary_identity)
        mx.eval(attention(prefix, prefix_mask, cache))
        rows.append(cache)
    segmented = SegmentedBatchQSAKVCache(
        rows, note=note_segmented_self_mtp, shared_qsa_prefix=True
    )
    segmented.prepare(lengths=[3, 3], right_padding=[0, 0])
    hidden = mx.random.normal((2, 3, 32), key=mx.random.key(102))
    return (segmented, attention, hidden)


@FAULTS
@pytest.mark.parametrize("exact_set", [True, False], ids=["exact_set", "private"])
def test_segmented_private_delta_fault_is_not_a_dense_fallback(
    monkeypatch, fault, exact_set
):
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT", "0")
    monkeypatch.setenv(
        "MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD", "1" if exact_set else "0"
    )
    (segmented, attention, hidden) = _segmented_private_delta()
    module = "mlx2.runtime.segmented_batch_cache."
    with (
        patch(
            module + "qwen4_qsa_indexed_private_delta_preflight",
            return_value=(True, "engaged"),
        ),
        patch(
            module + "qwen4_qsa_indexed_private_delta_exact_set_preflight",
            return_value=(True, "engaged"),
        ),
        patch(
            module + "qwen4_qsa_indexed_private_delta_exact_set_attention",
            side_effect=RuntimeError(fault),
        ),
        patch(
            module + "qwen4_qsa_indexed_private_delta_attention",
            side_effect=RuntimeError(fault),
        ),
        patch(
            module + "qsa_dense_attention_from_selection",
            side_effect=AssertionError("dense fallback ran on a device fault"),
        ),
    ):
        with pytest.raises(RuntimeError, match="Command buffer execution failed"):
            mx.eval(segmented.segmented_attention(attention, hidden, None))
