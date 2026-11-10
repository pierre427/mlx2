"""deepseek_source_q2_tier_math=mixed_v1 against the reference tier loop (CPU only).

Both arms run the real ``SourceV41Weights._moe_routes`` on the CPU device. Q2
experts use the real MLX affine 2-bit matmuls and block-Hadamard rotation at
the real shapes. The Metal-only pieces (fused middle, single/batch gate-up
middle, fused down, ordered combine) are replaced by CPU stand-ins that apply
one deterministic, row-independent law per expert in BOTH arms, so the
comparison checks routing, per-occurrence placement and sum order bit for bit.
The per-kernel arithmetic of the Metal pieces is checked on the GPU separately
(scripts/research/ds41_q2_mixed_row_bench.py).
"""
import threading
from types import MethodType, SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source as adapter
from mlx2.adapters import deepseek_v41_source_math as assembly
from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
from mlx2.adapters import deepseek_v41_source_q2_row as q2row
from mlx2.adapters.deepseek_v41_mlx_weights import NativeExpertStore
from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert, canonical_layout
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.runtime.gguf import source_down_fusion as down
from mlx2.runtime.gguf import source_gate_up_fusion as gu
from mlx2.runtime.gguf import source_middle_fusion as middle
from mlx2.runtime.gguf import source_mxfp4_pair, source_q1
from mlx2.runtime.models import deepseek_v41_quant as qat
from tests.test_deepseek_v41_source_residency_cpu import small  # noqa: F401 - fixture

TYPES = {'<u4': mx.uint32, '|u1': mx.uint8, '<f2': mx.float16}


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device(); mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _packed(kind):
    """Metadata-only packed object: geometry for the batch gate, no payload."""
    layout = canonical_layout(kind)
    fields = {t.name: SimpleNamespace(shape=t.shape, ndim=len(t.shape), dtype=TYPES[t.dtype], nbytes=t.nbytes)
              for t in layout.tensors}
    backing = SimpleNamespace(shape=(layout.nbytes,), ndim=1, dtype=mx.uint8, nbytes=layout.nbytes)
    return PackedExpert(backing, layout, fields)


_Q2_CACHE = {}


def _q2(fmt, seed):
    key = (fmt, seed)
    if key not in _Q2_CACHE:
        group = 64 if fmt == 'g64' else 32
        rng = np.random.default_rng(seed)
        arrays = {}
        for kind, shape in (('gate', (2304, 5120)), ('up', (2304, 5120)), ('down', (5120, 2304))):
            w = mx.array((rng.standard_normal(shape) * 0.02).astype(np.float16))
            q, s, b = mx.quantize(w, group_size=group, bits=2)
            arrays.update({kind + '.q2weight': q, kind + '.q2scales': s, kind + '.q2biases': b})
        mx.eval(arrays)
        _Q2_CACHE[key] = arrays
    return q1m.Q2Expert(_Q2_CACHE[key], q2_format=fmt)


def _vec(seed, n):
    return mx.array(np.random.default_rng(seed).standard_normal(n).astype(np.float32))


GATE_A, GATE_B, DOWN_A, DOWN_B = _vec(1, 2304), _vec(2, 2304), _vec(3, 5120), _vec(4, 5120)


def middle_law(gate, up, scores, slots):
    """Row-independent stand-in for the fused middle (BF16 rounds, clamp, weight)."""
    r = lambda v: v.astype(mx.bfloat16).astype(mx.float32)
    g = mx.minimum(r(gate), 10.0); u = mx.clip(r(up), -10.0, 10.0)
    return r(g * mx.sigmoid(g) * u * scores[slots][:, None])


def packed_gate_up(x, e):
    """Per-row stand-in for a packed expert's gate/up projections."""
    return x[:, :2304] * (e + 1) + GATE_A, x[:, 2304:4608] * (0.5 + e) - GATE_B


def packed_down(mid, e):
    """Per-row stand-in for a packed expert's down projection (F32[k,5120])."""
    return mid[:, :1] * DOWN_A * (e + 1) + mid[:, 1:2] * DOWN_B + mid[:, 2:3]


def _case(monkeypatch, experts, ids, scores, *, policy='mixed_v1', tier=True, packed_buffers=True):
    ids = np.array(ids, np.uint32)
    scores = mx.array(np.array(scores, np.float32))
    x = mx.array(np.random.default_rng(7).standard_normal((len(ids), 5120)).astype(np.float32))
    identity = lambda obj: next(e for e, value in experts.items() if value is obj)
    acquired = []
    store = SimpleNamespace(_active=True, _error=None, _lock=threading.RLock(), _leased={}, _pending_uploads={},
                            _pending_host={}, packed_buffers=packed_buffers, stats={'lookups': 0, 'windows': 0})
    store._ready = lambda: None
    store.window = MethodType(NativeExpertStore.window, store)
    store.observe_routes = lambda p, v: None
    store.q2_tier_active = lambda prefix: tier

    def acquire(prefix, e):
        acquired.append(e); store._leased[e] = experts[e]
        return experts[e]
    store.acquire = acquire
    w = object.__new__(SourceV41Weights); w._ready = lambda: None
    w.artifact = {'args': SimpleNamespace(hidden_size=5120, num_experts_per_tok=6)}
    w.expert_math_policy = 'routed_fused_fp32'; w.scheduled_expert_calls = {}; w.expert_stream = store
    w.source_q2_tier_math = policy

    def no_metal(*args, **kwargs):
        raise AssertionError('CPU oracle reached Metal')
    monkeypatch.setattr(mx.fast, 'metal_kernel', no_metal)
    monkeypatch.setattr(source_mxfp4_pair, '_device_supported', lambda: True)
    monkeypatch.setattr(middle, 'exp_spelling', lambda: 'metal::precise::exp')
    monkeypatch.setattr(middle, 'middle', middle_law)
    monkeypatch.setattr(qat, 'fused_quantize', lambda value, kind: value)

    def single(value, rows, score_rows, expert, **kwargs):          # reference loop, packed expert
        e = identity(expert); gate, up = packed_gate_up(value[rows], e)
        return middle_law(gate, up, score_rows, mx.arange(score_rows.size).astype(mx.uint32))
    monkeypatch.setattr(gu, 'middle', single)

    def batch(value, objects, flat_scores, slots, **kwargs):       # mixed_v1, packed experts
        rows = []
        for obj, slot in zip(objects, np.array(slots)):
            gate, up = packed_gate_up(value, identity(obj))
            rows.append(middle_law(gate, up, flat_scores, mx.array([int(slot)], mx.uint32)))
        return mx.concatenate(rows, axis=0)
    monkeypatch.setattr(gu, 'batch_middle', batch)

    def down_combine(middles, objects, eids, *, return_contributions=False, **kwargs):
        raw = [packed_down(m, identity(o)) for m, o in zip(middles, objects)]
        total = mx.zeros((1, 5120), mx.float32)
        for i in sorted(range(len(eids)), key=lambda i: eids[i]):
            total = total + raw[i].astype(mx.bfloat16).astype(mx.float32)
        order = sorted(range(len(eids)), key=lambda i: eids[i])
        return (total, mx.concatenate([raw[i] for i in order], axis=0)) if return_contributions else total
    monkeypatch.setattr(down, 'down_combine', down_combine)

    real_qmm = mx.quantized_matmul

    def qmm(x, w, scales=None, biases=None, transpose=True, group_size=None, bits=None, mode='affine', **kw):
        if bits == 2:
            return real_qmm(x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits, **kw)
        return packed_down(x, down_owner(w))                                   # reference loop, packed down
    monkeypatch.setattr(mx, 'quantized_matmul', qmm)
    down_owner = lambda weight: next(k for k, o in experts.items() if not q1m.is_q2(o) and o['down.qweight'] is weight)
    monkeypatch.setattr(source_q1, 'matmul', lambda value, weight, s, b: packed_down(value, down_owner(weight)))

    def paired_mapped(value, rows, gate_q, *rest):                # reference loop, unpacked Q1 dict
        e = next(k for k, o in experts.items() if not q1m.is_q2(o) and o['gate.qweight'] is gate_q)
        return packed_gate_up(value[rows], e)
    monkeypatch.setattr(source_q1, 'paired_matmul_mapped', paired_mapped)

    def combine(parts, values, order, **kwargs):                  # the ordered F32 combine law
        flat = mx.concatenate(parts, axis=0); inverse = np.empty(order.size, np.int64)
        inverse[order] = np.arange(order.size)
        outputs = []
        for row in range(values.shape[0]):
            total = mx.zeros((5120,), mx.float32)
            for slot in np.argsort(values[row], kind='stable'):
                total = total + flat[int(inverse[row * values.shape[1] + slot])]
            outputs.append(total)
        return mx.stack(outputs)
    monkeypatch.setattr(assembly, 'ordered_combine', combine)
    return SimpleNamespace(w=w, store=store, x=x, ids=ids, scores=scores, acquired=acquired)


def _run(monkeypatch, experts, ids, scores, **kw):
    case = _case(monkeypatch, experts, ids, scores, **kw)
    out = case.w.moe(case.x, case.ids, case.scores, 'blk.4'); mx.eval(out)
    return np.array(out), case


def _both(monkeypatch, experts, ids, scores):
    ref, rc = _run(monkeypatch, experts, ids, scores, policy='reference')
    new, nc = _run(monkeypatch, experts, ids, scores, policy='mixed_v1')
    assert rc.acquired == nc.acquired == sorted(set(int(e) for e in np.array(ids).reshape(-1)))
    assert np.isfinite(ref).all() and np.abs(ref).max() > 0
    return ref, new, rc.w.scheduled_expert_calls, nc.w.scheduled_expert_calls


SCORES = [[0.31, 0.22, 0.17, 0.13, 0.11, 0.06]]


@pytest.mark.parametrize('fmt', ['g64', 'g32h'])
def test_mixed_row_original_q1_q2_equals_reference_loop(monkeypatch, fmt):
    experts = {9: _packed('original'), 1: _packed('q1'), 7: _packed('original'),
               2: _q2(fmt, 2), 6: _q2(fmt, 6), 0: _q2(fmt, 0)}
    ref, new, rc, nc = _both(monkeypatch, experts, [[9, 1, 7, 2, 6, 0]], SCORES)
    assert np.array_equal(ref, new)
    assert 'official_source_q2_mixed_rows' not in rc and rc['official_source_q2_tier_experts'] == 3
    assert nc['official_source_q2_mixed_rows'] == 1 and nc['official_source_q2_mixed_packed_experts'] == 3
    assert nc['official_source_q2_tier_experts'] == nc['official_source_q2_mixed_q2_experts'] == 3
    assert nc['official_source_down_contributions_fused'] == 1
    assert nc['official_source_gate_up_middle_batch_fused'] == 1
    assert 'official_source_gate_up_middle_fused' not in nc     # no per-expert packed calls
    if fmt == 'g32h':
        assert nc['official_source_q2_rotated_experts'] == rc['official_source_q2_rotated_experts'] == 3
        assert nc['official_source_q2_mixed_rotations_shared'] == 2
    else:
        assert 'official_source_q2_rotated_experts' not in nc


@pytest.mark.parametrize('fmt', ['g64', 'g32h'])
def test_rotation_is_formed_once_per_row(monkeypatch, fmt):
    calls = []
    real = q1m.rotate_activation
    monkeypatch.setattr(q1m, 'rotate_activation', lambda x, r: calls.append((x.shape[-1], r)) or real(x, r))
    experts = {e: _q2(fmt, e) for e in (9, 1, 7, 2, 6, 0)}
    ref, new, _, nc = _both(monkeypatch, experts, [[9, 1, 7, 2, 6, 0]], SCORES)
    assert np.array_equal(ref, new)
    rotation = None if fmt == 'g64' else 'block_hadamard_256'
    reference, mixed = calls[:18], calls[18:]
    assert reference.count((5120, rotation)) == 12 and reference.count((2304, rotation)) == 6
    assert mixed.count((5120, rotation)) == 1 and mixed.count((2304, rotation)) == 6
    assert nc['official_source_q2_mixed_rows'] == 1 and 'official_source_q2_mixed_packed_experts' not in nc


def test_all_packed_tier_row_is_the_ordinary_fused_row(monkeypatch):
    experts = {e: _packed('q1' if e % 2 else 'original') for e in (9, 1, 7, 2, 6, 0)}
    ref, new, rc, nc = _both(monkeypatch, experts, [[9, 1, 7, 2, 6, 0]], SCORES)
    assert np.array_equal(ref, new)
    assert rc['official_source_gate_up_middle_fused'] == 6
    assert nc['official_source_q2_mixed_all_packed_rows'] == 1 and nc['official_source_down_combine_fused'] == 1
    assert 'official_source_q2_mixed_ordered_combine' not in nc


@pytest.mark.parametrize('fmt', ['g64', 'g32h'])
@pytest.mark.parametrize('dup', [9, 2])
def test_duplicate_zero_weight_slots(monkeypatch, fmt, dup):
    """Route drop redirects dropped slots to the top expert with weight 0."""
    experts = {9: _packed('original'), 1: _packed('q1'), 2: _q2(fmt, 2), 6: _q2(fmt, 6)}
    ids = [[dup, 1, dup, 2, 6, dup]] if dup == 9 else [[2, 1, 9, 2, 6, 2]]
    scores = [[0.4, 0.25, 0.0, 0.2, 0.15, 0.0]]
    ref, new, rc, nc = _both(monkeypatch, experts, ids, scores)
    assert np.array_equal(ref, new)
    assert nc['official_source_q2_mixed_rows'] == 1
    assert nc['official_source_q2_tier_experts'] == rc['official_source_q2_tier_experts'] == 2
    # A packed expert at several slots keeps the reference (M>1) law; once-routed ones batch.
    assert nc.get('official_source_q2_mixed_repeated_packed_experts', 0) == (1 if dup == 9 else 0)
    assert nc['official_source_q2_mixed_packed_experts'] == (1 if dup == 9 else 2)
    assert nc['source_q2_tier_b1_calls'] == 1 and nc['source_q2_tier_b1_host_ns'] > 0


def test_nonzero_duplicates_still_follow_the_reference_order(monkeypatch):
    experts = {9: _packed('original'), 2: _q2('g32h', 2), 6: _q2('g32h', 6)}
    ref, new, _, _ = _both(monkeypatch, experts, [[6, 9, 2, 9, 2, 6]], SCORES)
    assert np.array_equal(ref, new)


def test_unpacked_expert_declines_before_building_and_reuses_acquisitions(monkeypatch):
    plain_q1 = {k: v for k, v in _packed('q1').items()}            # a dict with biases, not a PackedExpert
    experts = {9: _packed('original'), 1: plain_q1, 2: _q2('g64', 2), 6: _q2('g64', 6),
               7: _packed('original'), 0: _q2('g64', 0)}
    ref, new, _, nc = _both(monkeypatch, experts, [[9, 1, 7, 2, 6, 0]], SCORES)
    assert np.array_equal(ref, new)
    assert nc['official_source_q2_mixed_declined_unpacked'] == 1 and 'official_source_q2_mixed_rows' not in nc


def test_non_tier_and_other_policies_never_enter_the_mixed_row(monkeypatch):
    experts = {e: _packed('original') for e in (9, 1, 7, 2, 6, 0)}
    _, case = _run(monkeypatch, experts, [[9, 1, 7, 2, 6, 0]], SCORES, tier=False)
    assert not any('q2_mixed' in k for k in case.w.scheduled_expert_calls)
    assert case.w.scheduled_expert_calls['official_source_down_combine_fused'] == 1
    experts = {9: _packed('original'), 1: _q2('g64', 1), 7: _packed('original'),
               2: _q2('g64', 2), 6: _q2('g64', 6), 0: _q2('g64', 0)}
    _, case = _run(monkeypatch, experts, [[9, 1, 7, 2, 6, 0]], SCORES, packed_buffers=False)
    assert case.w.scheduled_expert_calls['official_source_q2_mixed_declined_policy'] == 1
    assert 'official_source_q2_mixed_rows' not in case.w.scheduled_expert_calls


def test_policy_choice_and_validation():
    assert adapter.SOURCE_POLICY_CHOICES['deepseek_source_q2_tier_math'] == ('reference', 'mixed_v1')
    q2row.validate_policy('reference', expert_math=None)
    q2row.validate_policy('mixed_v1', expert_math='routed_fused_fp32')
    with pytest.raises(ValueError):
        q2row.validate_policy('mixed_v1', expert_math='epilogue_fused_fp32')
    with pytest.raises(ValueError):
        q2row.validate_policy('fast', expert_math='routed_fused_fp32')
    assert SourceV41Weights.source_q2_tier_math == 'reference'


def test_reduced_and_transient_reads_report_their_wall_time(small, monkeypatch):
    """Each tier read records its read wait (also in demand_wait_ns) and upload time in the receipt."""
    from tests.test_deepseek_v41_source_reduced_tiers_cpu import tiered
    from tests.test_deepseek_v41_source_q1_on_miss_cpu import call
    s = tiered(monkeypatch, [{'layers': [0, 1], 'only': 'q1'},
                             {'layers': 0, 'tier': 'q4', 'experts': [0]},
                             {'layers': 0, 'tier': 'q2', 'experts': [1]}], max_weight=0.25)
    call(s, 0, [3, 1, 2], [0.8, 0.1, 0.1])          # 3: heavy, unmapped -> transient; 1 -> q2; 2 -> q1
    counters = s.q1_on_miss.receipt()['counters']
    for tier in ('original_transient', 'q2', 'q1'):
        assert counters[tier + '_read_wait_ns'] > 0 and counters[tier + '_upload_ns'] >= 0, tier
    assert counters['original_transient_reads'] == counters['q2_reads'] == counters['q1_reads'] == 1
