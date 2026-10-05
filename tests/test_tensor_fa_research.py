"""CPU-only tests for the approximate tensor-API attention research module.

The CPU device is set before any tensor; mx.fast.metal_kernel,
mx.metal.is_available, mx.device_info and the module's native kernel builder
are forbidden. The host mirror is a DIAGNOSTIC model of the kernel's rounding
and rescale law, not a native oracle: agreement with the reference here says
nothing about Metal bits, engagement or speed.
"""

import dataclasses
import gc
import re
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)  # before any tensor

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mlx2.adapters import tensor_fa_research as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REAL_KERNEL_FACTORY = F._kernel  # captured before the autouse fixture forbids it


def _forbid(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not run in CPU tests")
    return fail


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(mx.fast, "metal_kernel", _forbid("mx.fast.metal_kernel"))
    monkeypatch.setattr(mx.metal, "is_available", _forbid("mx.metal.is_available"))
    monkeypatch.setattr(mx, "device_info", _forbid("mx.device_info"))
    monkeypatch.setattr(F, "_kernel", _forbid("native kernel build"))
    assert mx.default_device() == mx.cpu
    assert F._KERNEL == {}            # checked at setup: monkeypatch restores after this fixture's teardown
    yield


def _arrays(hq=4, hkv=2, length=37, kv_len=128, dim=128, q_dtype=mx.float32, seed=0, q_scale=1.0):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((1, hq, length, dim)).astype(np.float32) * q_scale).astype(q_dtype)
    k = mx.array(rng.standard_normal((1, hkv, kv_len, dim)).astype(np.float32)).astype(mx.float16)
    v = mx.array(rng.standard_normal((1, hkv, kv_len, dim)).astype(np.float32)).astype(mx.float16)
    return q, k, v


def _plan(q, k, v, **kw):
    kw = {"scale": 1.0 / np.sqrt(q.shape[-1]).item(), "causal": True, "q_start": k.shape[2] - q.shape[2], **kw}
    return F.plan_tensor_fa(q, k, v, **kw)


def _max_error(q, k, v, plan, trace=None):
    mirror = F.host_mirror(q, k, v, plan, trace=trace)
    ref = np.asarray(F.reference_attention(q, k, v, plan))
    assert np.isfinite(mirror).all(), "host mirror produced non-finite values"
    return float(np.abs(mirror - ref).max() / max(1e-6, np.abs(ref).max())), mirror, ref


# HOST DIAGNOSTIC threshold only (relative to max |reference|): it checks the host mirror's
# law, not the kernel. It is NOT a native fidelity acceptance threshold; native
# thresholds and model gates remain pending.
HOST_DIAGNOSTIC_TOLERANCE = 2e-2
TOLERANCE = HOST_DIAGNOSTIC_TOLERANCE


# ---------------------------------------------------------------- isolation and state

def test_import_is_isolated_and_metal_free():
    code = ("import mlx.core as mx\n"
            "def boom(*a, **k): raise SystemExit(7)\n"
            "mx.metal.is_available = boom; mx.fast.metal_kernel = boom; mx.device_info = boom\n"
            "import mlx2.adapters.tensor_fa_research as F\nprint(F.STATE['default'])\n")
    probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": "src:."})
    assert probe.returncode == 0 and probe.stdout.strip() == "off", probe.stderr
    users = [p for p in (ROOT / "src").rglob("*.py")
             if "tensor_fa_research" in p.read_text() and p.name != "tensor_fa_research.py"]
    assert users == []  # no production import


def test_state_is_default_off_unqualified_unselected():
    state = F.status()
    assert state["default"] == "off" and state["approximate"] is True and state["exact"] is False
    assert state["qualified"] is False and state["selected"] is False and state["observed_used"] is False
    assert state["route"] is None and state["native_engaged"] is False
    with pytest.raises(TypeError):
        F.STATE["qualified"] = True                       # immutable: cannot imply qualification
    assert F.status()["qualified"] is False


# ---------------------------------------------------------------- admission contract

def _with(**kw):
    def build():
        q, k, v = _arrays(**{key: kw.pop(key) for key in list(kw) if key in
                             ("hq", "hkv", "length", "kv_len", "dim", "q_dtype")})
        return q, k, v, kw
    return build


@pytest.mark.parametrize("make,message", [
    (lambda: (*_arrays()[:1], mx.zeros((2, 2, 128, 128), mx.float16), mx.zeros((2, 2, 128, 128), mx.float16), {}), "B1"),
    (lambda: (*_arrays(dim=64), {}), "head dim"),
    (lambda: (*_arrays(dim=192), {}), "head dim"),
    (lambda: (*_arrays(hq=6, hkv=4), {}), "Hq % Hkv"),
    (lambda: (*_arrays(q_dtype=mx.bfloat16), {}), "float16 or float32"),
    (lambda: (_arrays()[0], _arrays()[1].astype(mx.bfloat16), _arrays()[2], {}), "materialized float16"),
    (lambda: (_arrays()[0], _arrays()[1], _arrays()[2].astype(mx.float32), {}), "materialized float16"),
    (lambda: (_arrays()[0], _arrays()[1], _arrays(kv_len=192)[2], {}), "v must have k's shape"),
    (lambda: (*_arrays(kv_len=100), {"q_start": 0}), "multiple of 64"),
    (lambda: (*_arrays(length=0), {"q_start": 0}), "query length"),
    (lambda: (*_arrays(length=4097, kv_len=4160, hq=1, hkv=1), {"q_start": 0}), "query length"),
    (lambda: (_arrays()[0][0], *_arrays()[1:], {}), "rank 4"),
    (lambda: (_arrays()[0], tuple(_arrays()[1:]), _arrays()[2], {}), "quantized/packed tuple"),
    (lambda: (_arrays()[0], object(), _arrays()[2], {}), "mx.array"),
    (lambda: (*_arrays(), {"scale": 1}), "Python float"),
    (lambda: (*_arrays(), {"scale": True}), "Python float"),
    (lambda: (*_arrays(), {"scale": np.float32(0.1)}), "Python float"),
    (lambda: (*_arrays(), {"scale": float("nan")}), "Python float"),
    (lambda: (*_arrays(), {"scale": float("inf")}), "Python float"),
    (lambda: (*_arrays(), {"scale": 0.0}), "Python float"),
    (lambda: (*_arrays(), {"causal": 1}), "explicit bool"),
    (lambda: (*_arrays(), {"causal": None}), "explicit bool"),
    (lambda: (*_arrays(), {"q_start": -1}), "q_start"),
    (lambda: (*_arrays(), {"q_start": True}), "q_start"),
    (lambda: (*_arrays(), {"q_start": 91.0}), "q_start"),
    (lambda: (*_arrays(), {"q_start": 92}), "q_start"),           # 92 + 37 > 128: rows past the KV end
    (lambda: (*_arrays(), {"mask": mx.ones((37, 128), dtype=mx.bool_)}), "mask is not implemented"),
    (lambda: (*_arrays(), {"bias": 0.5}), "bias is not implemented"),
    (lambda: (*_arrays(), {"softcap": 30.0}), "softcap is not implemented"),
    (lambda: (*_arrays(), {"sinks": mx.zeros((4,))}), "sinks is not implemented"),
    (lambda: (*_arrays(), {"alibi": True}), "alibi is not implemented"),
    (lambda: (*_arrays(), {"scale": 1e300}), "after fp32 narrowing"),           # finite float64, inf in fp32
    (lambda: (*_arrays(), {"scale": 1e-46}), "after fp32 narrowing"),           # underflows to 0
    (lambda: (*_arrays(), {"scale": 1e-40}), "after fp32 narrowing"),           # fp32 subnormal
    (lambda: (*_arrays(), {"strides": (1, 2, 3, 4)}), "unsupported metadata ['strides']"),
    (lambda: (*_arrays(), {"indices": mx.zeros((4,))}), "unsupported metadata ['indices']"),
    (lambda: (*_arrays(), {"selection": object()}), "unsupported metadata ['selection']"),
    (lambda: (*_arrays(), {"lengths": [37]}), "unsupported metadata ['lengths']"),
    (lambda: (*_arrays(), {"offset": 0}), "unsupported metadata ['offset']"),
])
def test_out_of_contract_requests_are_refused(make, message):
    q, k, v, kw = make()
    with pytest.raises(F.TensorFARefused, match=re.escape(message)):
        F.plan_tensor_fa(q, k, v, **{"scale": 0.125, "causal": True, "q_start": 91, **kw})


def test_a_full_valid_kv_tail_query_is_admitted():
    q, k, v = _arrays()
    plan = F.plan_tensor_fa(q, k, v, scale=0.125, causal=True, q_start=91)
    assert (plan.query_len, plan.kv_len, plan.q_start) == (37, 128, 91)


# ---------------------------------------------------------------- geometry and bounds

@pytest.mark.parametrize("dim,hq,hkv,length,kv_len", [
    (256, 24, 2, 1, 4096), (256, 24, 2, 33, 8192), (128, 32, 8, 512, 1024), (128, 8, 8, 31, 64),
])
def test_grid_threadgroup_and_workspace(dim, hq, hkv, length, kv_len):
    q = mx.zeros((1, hq, length, dim), mx.float16)
    k = v = mx.zeros((1, hkv, kv_len, dim), mx.float16)
    plan = F.plan_tensor_fa(q, k, v, scale=dim ** -0.5, causal=True, q_start=kv_len - length)
    blocks = -(-length // 32)
    assert plan.query_blocks == blocks and plan.threadgroups == blocks * hq
    assert plan.grid == (blocks * 256, hq, 1) and plan.threadgroup == (256, 1, 1)
    assert plan.threadgroup_memory == {128: 20640, 256: 28832}[dim] <= F.MAX_THREADGROUP_MEMORY
    assert F.threadgroup_memory_bytes(512) > F.MAX_THREADGROUP_MEMORY  # why D512 is outside this slice


@pytest.mark.parametrize("length,kv_len,q_start,causal", [
    (1, 64, 63, True), (1, 64, 0, True), (37, 128, 91, True), (37, 256, 3, True), (65, 192, 127, True),
    (32, 64, 0, False), (100, 256, 50, False),
])
def test_kv_loop_bounds_cover_exactly_the_visible_keys(length, kv_len, q_start, causal):
    q, k, v = _arrays(hq=2, hkv=1, length=length, kv_len=kv_len)
    plan = F.plan_tensor_fa(q, k, v, scale=0.1, causal=causal, q_start=q_start)
    for block, end in enumerate(plan.kv_end):
        assert end % F.C_KEYS == 0 and 0 < end <= kv_len  # never reads past the KV
        last_row = min((block + 1) * F.Q_ROWS, length) - 1
        needed = plan.visible_keys(last_row).stop
        assert needed <= end < needed + F.C_KEYS or end == kv_len


def test_causal_key_membership_matches_the_reference_mask_and_no_row_is_empty():
    q, k, v = _arrays(length=37, kv_len=128)
    plan = F.plan_tensor_fa(q, k, v, scale=0.1, causal=True, q_start=50)
    mask = np.asarray(plan.q_start + np.arange(37)[:, None] >= np.arange(128)[None, :])
    for row in range(37):
        assert list(plan.visible_keys(row)) == list(np.flatnonzero(mask[row]))
        assert len(plan.visible_keys(row)) >= 1
    with pytest.raises(F.TensorFARefused, match="outside"):
        plan.visible_keys(37)                 # tail rows beyond L do not exist


def test_source_geometry_matches_the_plan_constants():
    src = F.TENSOR_FA_SOURCE
    assert "constexpr int Q = 32, C = 64, NSG = 8, NW = 32;" in src
    assert (F.Q_ROWS, F.C_KEYS, F.N_SIMDGROUPS, F.SIMD_WIDTH, F.THREADS) == (32, 64, 8, 32, 256)
    assert "matmul2d_descriptor(Q, C, D, false, true, false" in src
    assert "matmul2d_descriptor(Q, D, C, false, false, false" in src
    assert "if (m > M[jj] + 8.0f)" in src and F.RESCALE_GROWTH == 8.0
    assert "half(exp(s[ii] - M[jj]))" in src and "sum += float(p);" in src
    assert "-FLT_MAX : ss[" in src and "INFINITY" not in src        # finite mask under fast math
    # Race repair: ggml's shared sf[0] had a writer in every simdgroup. Reject it and
    # require exactly one writer per sgf slot, after the per-row loop, before a barrier.
    assert "sf[0]" not in src and "threadgroup int   sf[1]" not in src and "ic0" not in src
    assert "threadgroup int   sgf[NSG];" in src
    assert src.count("sgf[sgitg] =") == 1 and "if (tiisg == 0) sgf[sgitg] = sg_rescaled ? 1 : 0;" in src
    flag_write = src.index("sgf[sgitg] =")
    assert src.index("if (tiisg == 0) sr[j] = ms;\n        }") < flag_write                # the row loop has closed
    assert src.index("threadgroup_barrier", flag_write) < src.index("sgf[g]", flag_write)  # barrier before reads
    assert re.findall(r"\b(\w+)\[0\]\s*=", src) == []                                    # no shared scalar slot writes
    assert F.threadgroup_memory_bytes(128) == 20640 and F.threadgroup_memory_bytes(256) == 28832
    assert "dextents<int32_t, 2>(D, L - iq1)" in src                 # tail rows clipped at the store
    assert "if (tiitg == 0) {\n        atomic_fetch_add_explicit((device atomic_uint*)engaged" in src
    for removed in ("sinks", "softcap", "slope", "blk["):
        assert removed not in src


# ---------------------------------------------------------------- host mirror (diagnostic, NOT a native oracle)

@pytest.mark.parametrize("dim,hq,hkv,length,kv_len,q_dtype,causal,q_start", [
    (128, 4, 2, 37, 128, mx.float32, True, 91),      # tail block of 5 rows
    (128, 4, 2, 1, 64, mx.float16, True, 63),        # single decode-like row
    (128, 2, 1, 33, 192, mx.float32, True, 10),      # early q_start: later key blocks skipped
    (256, 4, 1, 40, 128, mx.float16, False, 0),      # non-causal, D256
    (256, 24, 2, 3, 64, mx.float32, True, 61),       # Flash-Next-like head counts (dense only!)
])
def test_mirror_tracks_the_reference_but_is_not_exact(dim, hq, hkv, length, kv_len, q_dtype, causal, q_start):
    q, k, v = _arrays(hq=hq, hkv=hkv, length=length, kv_len=kv_len, dim=dim, q_dtype=q_dtype, seed=dim + length)
    plan = F.plan_tensor_fa(q, k, v, scale=dim ** -0.5, causal=causal, q_start=q_start)
    error, mirror, ref = _max_error(q, k, v, plan)
    assert error < TOLERANCE
    assert mirror.shape == (1, hq, length, dim) and mirror.dtype == np.float32
    assert not np.array_equal(mirror.view(np.uint32), ref.view(np.uint32))  # approximate, never exact


def test_adversarial_logits_stay_finite_and_close():
    q, k, v = _arrays(length=40, kv_len=192, q_scale=12.0, seed=5)   # scores of order +-100
    plan = _plan(q, k, v)
    error, mirror, _ = _max_error(q, k, v, plan)
    assert np.isfinite(mirror).all() and error < TOLERANCE


def _rescale_case(growth):
    """Key block 1 holds keys whose scores exceed block 0's by ``growth`` for every row."""
    rng = np.random.default_rng(3)
    d = 128
    q = rng.standard_normal((1, 1, 8, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k = rng.standard_normal((1, 1, 128, d)).astype(np.float32) * 0.05
    k[0, 0, 64:] += q[0, 0].mean(axis=0) / np.linalg.norm(q[0, 0].mean(axis=0)) * growth
    v = rng.standard_normal((1, 1, 128, d)).astype(np.float32)
    arrays = mx.array(q), mx.array(k).astype(mx.float16), mx.array(v).astype(mx.float16)
    return arrays, F.plan_tensor_fa(*arrays, scale=1.0, causal=False, q_start=0)


def test_mid_loop_rescale_witness():
    (q, k, v), plan = _rescale_case(growth=20.0)
    trace = []
    error, _, _ = _max_error(q, k, v, plan, trace)
    assert error < TOLERANCE
    assert {kb for _, kb, _ in trace} == {0, 1}           # initial max, then a mid-loop rescale in key block 1
    (q, k, v), plan = _rescale_case(growth=4.0)
    trace = []
    error, _, _ = _max_error(q, k, v, plan, trace)
    assert error < TOLERANCE and {kb for _, kb, _ in trace} == {0}  # growth <= 8: lazily not rescaled


# ---------------------------------------------------------------- injected guard faults must be caught

def test_fault_causal_mask_disabled_is_caught(monkeypatch):
    q, k, v = _arrays(length=37, kv_len=128, seed=11)
    plan = _plan(q, k, v, q_start=30)
    assert _max_error(q, k, v, plan)[0] < TOLERANCE
    monkeypatch.setattr(F, "MASKED_SCORE", np.float32(0.0))      # fault: masked keys leak in
    with pytest.raises(AssertionError):
        assert _max_error(q, k, v, plan)[0] < TOLERANCE


def test_fault_lazy_rescale_disabled_is_caught(monkeypatch):
    (q, k, v), plan = _rescale_case(growth=20.0)
    monkeypatch.setattr(F, "RESCALE_GROWTH", float("inf"))         # fault: running max never moves
    with pytest.raises(AssertionError, match="non-finite|assert"):
        with np.errstate(over="ignore", invalid="ignore"):
            assert _max_error(q, k, v, plan)[0] < TOLERANCE


def test_fault_forged_plan_bounds_are_caught():
    q, k, v = _arrays()
    plan = _plan(q, k, v)
    forged = dataclasses.replace(plan, q_start=200)                 # bypasses admission
    with pytest.raises(AssertionError):
        for row in range(forged.query_len):
            assert forged.visible_keys(row).stop <= forged.kv_len
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.q_start = 0


# ---------------------------------------------------------------- no mutation, native gating, engagement

def test_no_input_or_state_mutation():
    q, k, v = _arrays()
    before = [np.array(x).tobytes() for x in (q, k, v)]
    plan = _plan(q, k, v)
    F.host_mirror(q, k, v, plan)
    F.reference_attention(q, k, v, plan)
    assert [np.array(x).tobytes() for x in (q, k, v)] == before


def test_native_entry_fails_closed_on_cpu_without_capability_queries():
    q, k, v = _arrays()
    F.status(reset=True)
    with pytest.raises(F.TensorFAUnavailable, match="allow_research_metal"):
        F.tensor_fa_attention(q, k, v, scale=0.125, causal=True, q_start=91)
    with pytest.raises(F.TensorFAUnavailable, match="allow_research_metal"):
        F.tensor_fa_attention(q, k, v, scale=0.125, causal=True, q_start=91, allow_research_metal=1)
    with pytest.raises(F.TensorFAUnavailable, match="not the GPU"):
        F.tensor_fa_attention(q, k, v, scale=0.125, causal=True, q_start=91, allow_research_metal=True)
    with pytest.raises(F.TensorFARefused, match="softcap"):            # contract before capability
        F.tensor_fa_attention(q, k, v, scale=0.125, causal=True, q_start=91, allow_research_metal=True,
                              softcap=30.0)
    assert F.status()["native_launches_issued"] == 0 and F.status()["native_engaged"] is False
    assert F.confirm_engagement(mx.array([1], dtype=mx.uint32)) is False


def _fake_issue(monkeypatch, *, module_built):
    def fake_kernel(**kw):
        return [mx.zeros(kw["output_shapes"][0], dtype=mx.float32),
                mx.array([kw["grid"][0] // 256 * kw["grid"][1]], dtype=mx.uint32)]   # "correct" count, still fake

    monkeypatch.setattr(F, "_admit_native", lambda allow: None)
    if module_built:
        monkeypatch.setitem(F._KERNEL, "fa", fake_kernel)
        monkeypatch.setattr(F, "_kernel", lambda: F._KERNEL["fa"])
    else:
        monkeypatch.setattr(F, "_kernel", lambda: fake_kernel)
    q, k, v = _arrays()
    return lambda: F.tensor_fa_attention(q, k, v, scale=0.125, causal=True, q_start=91, allow_research_metal=True)


def test_mocked_launch_is_issued_never_confirmed(monkeypatch):
    calls = []
    issue = _fake_issue(monkeypatch, module_built=False)
    real = F._kernel()

    def spy(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setattr(F, "_kernel", lambda: spy)
    F.status(reset=True)
    out, engaged, plan = issue()
    kw = calls[0]
    assert kw["grid"] == plan.grid and kw["threadgroup"] == (256, 1, 1) and kw["init_value"] == 0
    assert kw["output_shapes"] == [(1, 4, 37, 128), (1,)] and kw["output_dtypes"] == [mx.float32, mx.uint32]
    assert dict(kw["template"]) == {"T": mx.float32, "D": 128, "HQ": 4, "HKV": 2}
    assert np.asarray(kw["inputs"][3]).tolist() == [0.125] and np.asarray(kw["inputs"][4]).tolist() == [37, 128, 91, 1]
    assert F.status()["native_launches_issued"] == 1 and F.status()["pending_records"] == 0  # unregistered
    assert F.confirm_engagement(engaged) is False                           # CPU default device
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert F.confirm_engagement(engaged) is False                           # never registered
    assert F.status()["native_engagement_confirmed"] == 0 and F.status()["native_engaged"] is False


def test_many_simultaneous_flags_leave_no_stale_metadata(monkeypatch):
    """Root's stage-15 witness: 100 live flags, then all dropped -> 0 records, 0 hints."""
    issue = _fake_issue(monkeypatch, module_built=True)
    F.status(reset=True)
    live = [issue() for _ in range(100)]
    assert F.status()["pending_records"] == F.status()["pending_id_hints"] == 100
    del live
    gc.collect()
    assert F._REGISTRY.sizes() == (0, 0)
    assert F.status()["native_engagement_confirmed"] == 0


def test_cpu_refusal_never_consumes_a_valid_record(monkeypatch):
    issue = _fake_issue(monkeypatch, module_built=True)
    F.status(reset=True)
    _, engaged, _ = issue()
    for _ in range(3):
        assert F.confirm_engagement(engaged) is False                       # CPU: refused before the registry
    assert F._REGISTRY.sizes() == (1, 1)
    assert F.confirm_engagement(mx.array([1], dtype=mx.uint32)) is False     # foreign flag
    assert F.confirm_engagement("flag") is False
    assert F._REGISTRY.sizes() == (1, 1) and F.status()["native_engagement_confirmed"] == 0
    del engaged
    gc.collect()
    assert F._REGISTRY.sizes() == (0, 0)


def test_claim_is_consumed_exactly_once_under_concurrency():
    import threading

    registry = F._LaunchRegistry()                                          # private: counts nothing
    flag = mx.array([7], dtype=mx.uint32)
    registry.register(flag, 7)
    barrier = threading.Barrier(16)
    results = []

    def claimer():
        barrier.wait()
        results.append(registry.claim(flag))

    threads = [threading.Thread(target=claimer) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results, key=str) == [7] + [None] * 15
    assert registry.claim(flag) is None and registry.sizes() == (0, 0)      # double confirmation refused


def test_id_hint_is_checked_by_identity_and_dies_with_the_flag():
    registry = F._LaunchRegistry()
    a, b = mx.array([1], dtype=mx.uint32), mx.array([2], dtype=mx.uint32)
    token = registry.register(a, 1)
    registry._by_id[id(b)] = token                                          # simulate an aliased id hint
    assert registry.claim(b) is None                                        # identity check refuses the alias
    assert registry.claim(a) == 1
    registry._by_id.pop(id(b), None)
    c = mx.array([3], dtype=mx.uint32)
    registry.register(c, 3)
    assert registry.sizes() == (1, 1)
    del c
    gc.collect()
    assert registry.sizes() == (0, 0)                                       # expected metadata died with c


def test_concurrent_registration_and_death_keep_the_registry_consistent():
    import threading

    registry = F._LaunchRegistry()
    errors = []

    def worker():
        try:
            for i in range(200):
                flag = mx.array([i], dtype=mx.uint32)
                registry.register(flag, i)
                if i % 3 == 0 and registry.claim(flag) != i:
                    errors.append(i)
                del flag
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    gc.collect()
    assert errors == [] and registry.sizes() == (0, 0)


def test_kernel_construction_is_synchronized(monkeypatch):
    import threading
    import time as _time

    built = []

    def slow_builder(**kw):
        _time.sleep(0.01)
        built.append(kw["name"])
        return object()

    monkeypatch.setattr(mx.fast, "metal_kernel", slow_builder)              # fake builder, no Metal
    monkeypatch.setattr(F, "_kernel", REAL_KERNEL_FACTORY)
    results = []
    threads = [threading.Thread(target=lambda: results.append(F._kernel())) for _ in range(12)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert built == ["mlx2_tensor_fa_research_v1"] and len({id(r) for r in results}) == 1
    finally:
        F._KERNEL.clear()


# ---------------------------------------------------------------- layout and key-set contract (truthful)

def test_layout_contract_is_declared_not_verified():
    plan = _plan(*_arrays())
    assert plan.layout_contract is F.LAYOUT_CONTRACT and hash(plan) == hash(dataclasses.replace(plan))
    with pytest.raises(TypeError):
        F.LAYOUT_CONTRACT["row_contiguity_host_verified"] = True
    assert plan.layout_contract["row_contiguity_host_verified"] is False
    assert "measured separately" in plan.layout_contract["materialization"]
    assert "cannot prove" in plan.layout_contract["key_set"]
    prov_path = ROOT / "provenance/tensor-fa-research.json"
    if not prov_path.is_file():
        pytest.skip("private provenance absent: provenance/tensor-fa-research.json "
                    "(the host-side layout contract above was still checked)")
    prov = prov_path.read_text().lower()
    assert "strided arrays refused" not in prov and "stride cases refused" not in prov
    assert "row_contiguity_host_verified" in prov


def test_sliced_and_transposed_logical_dense_views_are_treated_as_their_logical_values():
    q, k_full, v_full = _arrays(kv_len=256, seed=21)
    k_slice, v_slice = k_full[:, :, ::2], v_full[:, :, ::2]                   # strided views, N = 128
    plan = F.plan_tensor_fa(q, k_slice, v_slice, scale=0.125, causal=True, q_start=91)
    k_copy, v_copy = mx.array(np.array(k_slice)), mx.array(np.array(v_slice))
    same = F.plan_tensor_fa(q, k_copy, v_copy, scale=0.125, causal=True, q_start=91)
    assert plan == same                                                      # shape-only admission: identical plans
    mirror_view, mirror_copy = F.host_mirror(q, k_slice, v_slice, plan), F.host_mirror(q, k_copy, v_copy, same)
    assert np.array_equal(mirror_view.view(np.uint32), mirror_copy.view(np.uint32))
    assert _max_error(q, k_slice, v_slice, plan)[0] < TOLERANCE
    base = mx.array(np.random.default_rng(4).standard_normal((1, 128, 2, 128)).astype(np.float16))
    k_t = mx.swapaxes(base, 1, 2)                                            # [1, Hkv, N, D] transposed view
    plan_t = F.plan_tensor_fa(q, k_t, k_t, scale=0.125, causal=False, q_start=0)
    assert _max_error(q, k_t, k_t, plan_t)[0] < TOLERANCE


# ---------------------------------------------------------------- int32 bounds (metadata only, lazy broadcast)

def test_kv_length_is_bounded_for_int32_uniforms(monkeypatch):
    assert F.MAX_KV % F.C_KEYS == 0 and F.MAX_KV + F.C_KEYS - 1 <= F.INT32_MAX
    assert F.MAX_KV + F.C_KEYS > F.INT32_MAX          # the next multiple of 64 is not even an MLX (int32) shape
    q = mx.zeros((1, 1, 1, 128), mx.float16)
    zero = mx.zeros((1, 1, 1, 128), mx.float16)
    with monkeypatch.context() as mp:                  # defence-in-depth guard, exercised with a lowered bound
        mp.setattr(F, "MAX_KV", 128)
        over = mx.broadcast_to(zero, (1, 1, 192, 128))
        with pytest.raises(F.TensorFARefused, match="int32"):
            F.plan_tensor_fa(q, over, over, scale=0.125, causal=True, q_start=0)
        mp.setattr(F, "INT32_MAX", 100)                # the causal round-up check
        ok = mx.broadcast_to(zero, (1, 1, 128, 128))
        with pytest.raises(F.TensorFARefused, match="round-up exceeds int32"):
            F.plan_tensor_fa(q, ok, ok, scale=0.125, causal=True, q_start=40)
    edge = mx.broadcast_to(zero, (1, 1, F.MAX_KV, 128))                      # lazy metadata, never evaluated
    plan = F.plan_tensor_fa(q, edge, edge, scale=0.125, causal=True, q_start=F.MAX_KV - 1)
    assert plan.kv_end == (F.MAX_KV,) and plan.q_start + 1 + F.C_KEYS - 1 <= F.INT32_MAX


def test_scale_is_narrowed_to_fp32_for_kernel_mirror_and_reference():
    q, k, v = _arrays()
    plan = _plan(q, k, v, scale=0.1)
    assert plan.scale == 0.1 and plan.scale_fp32 == float(np.float32(0.1)) and plan.scale_fp32 != 0.1
    assert _plan(q, k, v, scale=float(np.finfo(np.float32).tiny)).scale_fp32 > 0


# ---------------------------------------------------------------- tails and causal edges

def test_query_tails_and_causal_edges_are_explicit():
    q, k, v = _arrays(length=37, kv_len=128)
    plan = F.plan_tensor_fa(q, k, v, scale=0.125, causal=True, q_start=91)   # last row reaches key N-1
    assert plan.query_blocks == 2 and 37 - F.Q_ROWS == 5                     # 5 valid rows in the tail block
    assert plan.visible_keys(36) == range(128) and plan.visible_keys(0) == range(92)
    out = F.host_mirror(q, k, v, plan)
    assert out.shape == (1, 4, 37, 128)                                      # tail-block padding rows never emitted
    with pytest.raises(F.TensorFARefused, match="q_start"):
        F.plan_tensor_fa(q, k, v, scale=0.125, causal=True, q_start=92)      # a row would sit past the KV end
    src = F.TENSOR_FA_SOURCE
    assert "if (iq1 + j < L) x = float(" in src                             # padding rows load zero queries
    assert "dextents<int32_t, 2>(D, L - iq1)" in src                         # and are clipped at the store


def test_reference_uses_exactly_the_visible_keys():
    q, k, v = _arrays(hq=2, hkv=1, length=5, kv_len=64, seed=9)
    plan = F.plan_tensor_fa(q, k, v, scale=0.125, causal=True, q_start=20)
    ref = np.asarray(F.reference_attention(q, k, v, plan))
    q64 = np.asarray(q, dtype=np.float64)[0]
    k64, v64 = (np.asarray(x, dtype=np.float64)[0] for x in (k, v))
    for head in range(2):
        for row in range(5):
            keys = np.array(plan.visible_keys(row))
            s = q64[head, row] @ k64[0, keys].T * plan.scale_fp32
            w = np.exp(s - s.max())
            exact = (w / w.sum()) @ v64[0, keys]
            assert np.abs(ref[0, head, row] - exact).max() < 1e-5
