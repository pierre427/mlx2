"""Pure graph and one-time stage witness controls; fake MLX, no driver/GPU."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, MappingProxyType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_ds41_native_ffn_math_cpu', ROOT/'scripts/research/ds41_native_ffn_math.py')
f = importlib.util.module_from_spec(spec); sys.modules[spec.name] = f; spec.loader.exec_module(f)
BF16 = object()


class Array:
    def __init__(self, x, dtype=None):
        self.a = np.asarray(x, dtype=dtype)
    @property
    def shape(self): return self.a.shape
    @property
    def ndim(self): return self.a.ndim
    @property
    def dtype(self): return self.a.dtype
    @property
    def nbytes(self): return self.a.nbytes
    def astype(self, dtype):
        if dtype is BF16:
            bits = self.a.astype(np.float32).view(np.uint32)
            return Array(((bits+np.uint32(0x7fff)+((bits>>16)&1)) & np.uint32(0xffff0000)).view(np.float32))
        return Array(self.a.astype(dtype))
    def __getitem__(self, key):
        if isinstance(key, Array): key = key.a
        return Array(self.a[key])
    def __add__(self, other): return Array(self.a+other.a)
    def __mul__(self, other): return Array(self.a*(other.a if isinstance(other, Array) else other))
    def __array__(self, dtype=None, copy=None): return np.asarray(self.a, dtype=dtype)
    def view(self, dtype): return Array(self.a.view(dtype))
    def reshape(self, shape): return Array(self.a.reshape(shape))


@pytest.fixture
def stage(monkeypatch):
    calls = []
    mx = ModuleType('mlx.core'); mx.array = Array
    mx.float32 = np.dtype('float32'); mx.float16 = np.dtype('float16')
    mx.uint32 = np.dtype('uint32'); mx.uint8 = np.dtype('uint8'); mx.bfloat16 = BF16
    mx.gpu = object(); mx.default_device = lambda: mx.gpu
    mx.contiguous = lambda x: (calls.append(('contiguous', x)), Array(np.ascontiguousarray(x.a)))[1]
    mx.zeros = lambda shape, dtype: Array(np.zeros(shape, dtype=dtype))
    mx.where = lambda c, yes, no: (calls.append(('where',)), Array(np.where(c.a, yes.a, no.a)))[1]
    mx.stack = lambda parts, axis: Array(np.stack([p.a for p in parts], axis=axis))
    mx.minimum = lambda x, n: Array(np.minimum(x.a, n))
    mx.clip = lambda x, lo, hi: Array(np.clip(x.a, lo, hi))
    mx.eval = lambda *x: (_ for _ in ()).throw(AssertionError('Timed evaluation'))
    nn = ModuleType('mlx.nn'); nn.silu = lambda x: Array(x.a/(1+np.exp(-x.a)))
    mlx = ModuleType('mlx'); mlx.core = mx; mlx.nn = nn
    monkeypatch.setitem(sys.modules, 'mlx', mlx); monkeypatch.setitem(sys.modules, 'mlx.core', mx)
    monkeypatch.setitem(sys.modules, 'mlx.nn', nn)
    m = 32
    shared = Array(np.arange(99*5120, dtype=np.float32).reshape(99, 5120)%4)
    route = Array(np.arange(m, dtype=np.uint32)[::-1])
    scores = Array(np.ones(m, np.float32))
    ranks = np.arange(m)%6
    parts = np.full((m, 6, 5120), 2., np.float32)
    # Sensitive ordered sum: +large,+small,-large,+small makes regrouping visible.
    parts[:, 0] = 2**25; parts[:, 1] = 1.; parts[:, 2] = -(2**25)
    arrays = MappingProxyType(dict(shared_gu_input=shared, route_rows=route, route_scores=scores,
        assembly_parts=Array(parts), assembly_rank=Array(ranks.astype(np.uint32)), _local_slots=Array(np.arange(m, dtype=np.uint32))))
    case = MappingProxyType(dict(rows=m, layer=0, expert=38))
    class Expert:
        backing = object()
        def __getitem__(self, key): return key
    expert = Expert()
    def projection(x, index):
        width = 5120 if index == 2 else 2304
        v = np.sum(x.a.astype(np.float32), axis=1, keepdims=True)/x.shape[1]
        return Array(np.broadcast_to(v, (m, width)).copy())
    def qmm(x, q, *, scales, mode):
        assert mode == 'mxfp4' and scales == q.replace('qweight', 'scales')
        index = ('gate', 'up', 'down').index(q.split('.')[0])
        calls.append(('stock', index, x)); return projection(x, index)
    mx.quantized_matmul = qmm
    def kernel(index):
        def invoke(**kw):
            x, backing, offset = kw['inputs']
            assert backing is expert.backing
            calls.append(('native', index, x))
            return [projection(x, index)]
        return invoke
    def middle(**kw):
        g, u, score, slots, inverse = kw['inputs']
        assert score is scores and slots is arrays['_local_slots']
        calls.append(('middle', g, u))
        return [Array(g.a*u.a)]
    def assembly(**kw):
        p, chosen, ranks_input = kw['inputs']
        calls.append(('assembly',))
        total = np.zeros((m, 5120), np.float32)
        for slot in range(6):
            value = np.where((ranks_input.a == slot)[:,None], chosen.a, p.a[:,slot])
            total = total+value
        return [Array(total)]
    masks = tuple(Array((ranks == i)[:, None]) for i in range(6))
    receipt = {}
    state = f._StageState(mx, arrays, case, expert, expert.backing, (), (16, 32), m,
        mx.zeros((99, 5120), dtype=mx.float32), masks, (object(),)*3,
        tuple(kernel(i) for i in range(3)), middle, assembly, Array([1/448], np.float32), receipt)
    a = f.StageAdmission(mx, arrays, case, expert, receipt, f._SEAL, state)
    return a, calls, ranks, parts


def test_native_fresh_gather_two_casts_three_projections_same_gu_half(stage):
    a, calls, _, _ = stage
    first = f.native_graph(a)
    second = f.native_graph(a)
    natives = [v for v in calls if v[0] == 'native']
    assert [v[1] for v in natives] == [0, 1, 2]*2
    assert natives[0][2] is natives[1][2]
    assert natives[3][2] is natives[4][2]
    assert natives[0][2] is not natives[3][2]
    assert len([v for v in calls if v[0] == 'contiguous']) == 4
    assert first is not second and first.a.tobytes() == second.a.tobytes()
    assert not any(v[0] == 'stock' for v in calls)


def test_reference_f32_inputs_middle_then_down_without_half(stage):
    a, calls, _, _ = stage
    f.reference_graph(a)
    assert [v[0] for v in calls] == ['stock', 'stock', 'middle', 'stock', 'assembly']
    stock = [v for v in calls if v[0] == 'stock']
    assert all(v[2].dtype == np.float32 for v in stock)
    assert stock[0][2] is stock[1][2]
    assert np.array_equal(stock[0][2].a, a.arrays['shared_gu_input'].a[a.arrays['route_rows'].a])


def test_complete_frontiers_exact_owner_and_ascending_replacement(stage):
    a, _, ranks, original_parts = stage
    d = f.native_graph(a, frontiers=True)
    assert set(d) == {'shared_gu_input','gu_input','gate_raw','up_raw','gate_owned','up_owned',
        'weighted_raw','weighted_owned','middle','middle_reference','down_input','down_raw',
        'down_owned','assembly_parts','assembly_output'}
    assert d['middle'] is d['middle_reference'] is d['down_input']
    expected = original_parts.copy()
    expected[np.arange(len(ranks)), ranks] = d['down_owned'].a
    assert np.array_equal(expected, d['assembly_parts'].a)
    total = np.zeros((len(ranks), 5120), np.float32)
    for slot in range(6): total = total+expected[:, slot]
    assert total.tobytes() == d['assembly_output'].a.tobytes()
    reversed_total = np.zeros_like(total)
    for slot in reversed(range(6)): reversed_total = reversed_total+expected[:, slot]
    assert not np.array_equal(reversed_total, total)
    assert np.all(d['down_owned'].a.view(np.uint32)&0xffff == 0)


def test_zero_aba_keeps_actual_admission_and_other_five_parts(stage):
    a, _, ranks, original_parts = stage
    before = f.native_graph(a).a.copy()
    z = f.native_graph(a, frontiers=True, canary=True)
    after = f.native_graph(a).a
    assert np.array_equal(before, after)
    for name in ('gu_input','gate_raw','up_raw','middle','down_raw','down_owned'):
        assert not np.any(z[name].a)
    for row, chosen in enumerate(ranks):
        for slot in range(6):
            if slot != chosen: assert np.array_equal(z['assembly_parts'].a[row, slot], original_parts[row, slot])


@pytest.mark.parametrize('kwargs', [{'frontiers':1},{'canary':1},{'frontiers':None},{'canary':[]}])
def test_ambiguous_flags_refused_without_work(stage, kwargs):
    a, calls, _, _ = stage
    with pytest.raises(ValueError): f.native_graph(a, **kwargs)
    assert not calls


@pytest.mark.parametrize('field', ['mx','arrays','case','expert','receipt','_seal'])
def test_substituted_capability_operand_refused_before_graph(stage, field):
    a, calls, _, _ = stage
    with pytest.raises(ValueError): f.native_graph(replace(a, **{field: object()}))
    assert not calls


def test_no_scalar_readback_or_numpy_copy_in_timed_builders(stage, monkeypatch):
    a, _, _, _ = stage
    def refuse(*args, **kwargs): raise AssertionError('Readback/copy in graph')
    monkeypatch.setattr(Array, '__array__', refuse)
    f.native_graph(a); f.reference_graph(a)


def admission_fixture(stage, monkeypatch, *, down_value=0.):
    """Real fixed Source geometry, original backing and actual capture validator."""
    a, calls, _, _ = stage; mx = a.mx; m = 32
    from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert, canonical_layout
    from mlx2.runtime.gguf import source_native_mx as native, source_middle_fusion as middle
    import ds41_native_mx_components as components
    import ds41_native_ffn_capture as capture
    layout = canonical_layout('original')
    backing = Array(np.zeros(layout.nbytes, np.uint8))
    tensors = {t.name: backing[t.offset:t.offset+t.nbytes].view(t.dtype).reshape(t.shape) for t in layout.tensors}
    expert = PackedExpert(backing, layout, tensors)
    values = np.tile(np.arange(6, dtype=np.uint32), (99, 1)); values[:m, 5] = 38
    counts = np.unique(values, return_counts=True)[1]
    case = dict(rows=m, kind='original', layer=0, expert=38, same_forward=True,
        expert_order=[0,1,2,3,4,5,38], expert_row_counts=counts.tolist(), part_index=6)
    arrays = {name: np.zeros(shape, np.uint32 if name in capture.UINT_ARRAYS else np.float32)
        for name, shape in capture.shapes(m).items()}
    arrays.update(values=values, order=np.argsort(values.reshape(-1), kind='stable').astype(np.uint32),
        route_rows=np.arange(m, dtype=np.uint32), route_locations=np.arange(m, dtype=np.uint32)*6+5,
        assembly_rank=np.full(m, 5, np.uint32))
    capture.validate_arrays(case, arrays)
    monkeypatch.setattr(components, 'coefficient_gate', lambda np, q, s: {'eligible':True})
    monkeypatch.setattr(middle, 'exp_spelling', lambda:'metal::exp')
    def mk(projection, rows, tile):
        width = 5120 if projection == 'down' else 2304
        return lambda **kw: [Array(np.zeros((rows, width), np.float32))]
    monkeypatch.setattr(native, '_kernel', mk)
    monkeypatch.setattr(middle, '_kernel', lambda spelling: lambda **kw:
        [Array(np.full((m, 2304), down_value, np.float32))])
    monkeypatch.setattr(f, '_assembly_kernel', lambda: a._state.assembly_kernel)
    mx.eval = lambda *values: calls.append(('admission_eval', len(values)))
    return mx, {'case':case, 'arrays':arrays}, expert, calls


def test_one_time_admission_actual_coefficients_and_native_down_witness(stage, monkeypatch):
    import json
    mx, frame, expert, calls = admission_fixture(stage, monkeypatch)
    a = f.admit(mx, np, frame, expert)
    receipt = a.receipt
    json.dumps(receipt)
    assert receipt['eligible'] is True
    assert receipt['untimed_actual_backing_readback_bytes'] == 18_800_640
    assert receipt['untimed_typed_view_readback_bytes'] == 18_800_640
    assert set(receipt['native_generated_down_half_roundtrip']) == {'actual','zero'}
    assert all(v['bitwise_exact'] for v in receipt['native_generated_down_half_roundtrip'].values())
    calls.clear()
    mx.eval = lambda *v: (_ for _ in ()).throw(AssertionError('Re-evaluated admission'))
    f.native_graph(a); f.native_graph(a, canary=True)
    assert not any(v[0] == 'admission_eval' for v in calls)


def test_actual_native_created_down_input_must_be_lossless_half(stage, monkeypatch):
    mx, frame, expert, _ = admission_fixture(stage, monkeypatch, down_value=1.0001)
    with pytest.raises(ValueError, match='native-generated DOWN input'):
        f.admit(mx, np, frame, expert)


def test_noncanonical_typed_coefficients_refused_before_native(stage, monkeypatch):
    mx, frame, expert, _ = admission_fixture(stage, monkeypatch)
    expert._tensors = dict(expert._tensors)
    expert._tensors['gate.scales'] = Array(np.ones(expert['gate.scales'].shape, np.uint8))
    with pytest.raises(ValueError, match='typed view'):
        f.admit(mx, np, frame, expert)


def test_ineligible_coefficient_gate_refuses_native_dispatch(stage, monkeypatch):
    mx, frame, expert, calls = admission_fixture(stage, monkeypatch)
    import ds41_native_mx_components as components
    monkeypatch.setattr(components, 'coefficient_gate', lambda *args: {'eligible':False})
    with pytest.raises(ValueError, match='coefficient representation'):
        f.admit(mx, np, frame, expert)
    assert not any(v[0] == 'admission_eval' for v in calls)


def test_unbounded_tile_refused_before_weight_readback(stage, monkeypatch):
    mx, frame, expert, _ = admission_fixture(stage, monkeypatch)
    monkeypatch.setattr(Array, '__array__', lambda *args, **kw:
        (_ for _ in ()).throw(AssertionError('Readback before shape gate')))
    with pytest.raises(ValueError, match='bounded'):
        f.admit(mx, np, frame, expert, tile=(8,64))


def test_timed_assembly_one_dispatch_no_where_or_stack(stage, monkeypatch):
    a, calls, _, _ = stage
    a.mx.where = lambda *args: (_ for _ in ()).throw(AssertionError('Timed where'))
    a.mx.stack = lambda *args, **kw: (_ for _ in ()).throw(AssertionError('Timed stack'))
    f.native_graph(a); f.reference_graph(a)
    assert [v[0] for v in calls].count('assembly') == 2
    spec = f.assembly_kernel_spec()
    assert spec['compile_options'] == {'math_mode':'safe'}
    assert 'fp contract(off)' in spec['header']
    assert 'float total=0.0f;' in spec['source']
    assert 'loop unroll(disable)' in spec['source']
    assert 'for(uint slot=0;slot<6;++slot)' in spec['source']
    assert 'total=total+value;' in spec['source']
    assert 'slot==ranks[row] ? chosen[i]' in spec['source']
