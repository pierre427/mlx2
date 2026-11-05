"""Lossless bank-layout, device-route contracts and offline Metal checks.

These synthetic CPU tests do not claim model or GPU numerical qualification.
"""

import inspect
import shutil
import subprocess

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import v41_iq1_tiled as tiled
from mlx2.runtime.gguf import v41_prefill, v41_row_exact
from mlx2.runtime.gguf.blocks import decode_blocks
from mlx2.runtime.gguf.iq1_codebook import IQ1_GRID
from mlx2.runtime.gguf.reader import GGUFError

mx.set_default_device(mx.cpu)


@pytest.mark.parametrize('shape', [(64, 256), (128, 512), (256, 2304), (2304, 5120), (5120, 2304)])
def test_lossless_planes_keep_exact_bytes_and_independent_tile_addresses(shape):
    n, k = shape
    rng = np.random.default_rng(n+k)
    raw = rng.integers(0, 256, n*(k//256)*50, np.uint8)
    planes = tiled.pack(raw, shape)
    assert tuple(v.size for v in planes) == tiled.plane_sizes(shape)
    assert sum(v.nbytes for v in planes) == raw.nbytes
    assert all(v.ndim == 1 and v.flags.c_contiguous for v in planes)
    assert tiled.unpack(planes, shape).tobytes() == raw.tobytes()
    # Independent scalar offsets check first/last lanes across both tile and
    # block boundaries, not merely two mutually inverse reshape functions.
    rows = sorted(set([0, 1, 63, n-1, min(n-1, 64), min(n-1, 127)]))
    columns = sorted(set([0, 31, 32, 255, k-1, min(k-1, 256)]))
    for row in rows:
        for column in columns:
            group = ((row//64)*(k//32)+column//32)*64+row%64
            block = ((row//64)*(k//256)+column//256)*64+row%64
            original = (row*(k//256)+column//256)*50
            assert planes[0][group*4+(column%32)//8] == raw[original+2+4*((column%256)//32)+(column%32)//8]
            assert planes[1][group*2:group*2+2].tobytes() == raw[original+34+2*((column%256)//32):original+36+2*((column%256)//32)].tobytes()
            assert planes[2][block*2:block*2+2].tobytes() == raw[original:original+2].tobytes()


def test_banked_layout_decodes_every_grid_index_and_lane_with_exact_f32_values():
    n, k = 256, 256
    raw = np.zeros((n, 50), np.uint8)
    scales = np.array([0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff], '<u2')
    raw[:, :2] = np.resize(scales, n).view(np.uint8).reshape(n, 2)
    for row in range(n):
        for group in range(8):
            # Four grids per group cover all 2048 entries four times; all
            # odd scales and both delta signs occur at every row position.
            high = ((row+group) % 8) << 12 | ((row//8+group) % 2) << 15
            for j in range(4):
                index = (row*32+group*4+j) % 2048
                raw[row, 2+group*4+j] = index & 255
                high |= (index >> 8) << (j*3)
            raw[row, 34+group*2:36+group*2] = np.frombuffer(high.to_bytes(2, 'little'), np.uint8)
    other = np.bitwise_xor(raw, 255)
    bank = tuple(np.concatenate([a, b]) for a, b in zip(tiled.pack(other, (n, k)), tiled.pack(raw, (n, k))))
    table = np.array(IQ1_GRID, '<u8').view(np.int8).reshape(2048, 8)
    actual = np.empty((n, k), np.float32)
    for row in range(n):
        for column in range(k):
            group = n*(k//32)+((row//64)*(k//32)+column//32)*64+row%64
            block = n*(k//256)+((row//64)*(k//256)+column//256)*64+row%64
            high = int(bank[1][group*2]) | int(bank[1][group*2+1]) << 8
            low = int(bank[0][group*4+(column%32)//8])
            index = low | (((high >> (3*((column%32)//8))) & 7) << 8)
            scale = np.frombuffer(bank[2][block*2:block*2+2], '<f2')[0].astype(np.float32)
            delta = np.float32(-.125 if high & 0x8000 else .125)
            actual[row, column] = (scale*np.float32(2*((high >> 12) & 7)+1))*(np.float32(table[index, column%8])+delta)
    expected = decode_blocks(raw.reshape(-1), 19, max_elements=n*k).reshape(n, k)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('shape', [(0, 256), (63, 256), (64, 255), (64, 256, 1), (True, 256), (64.0, 256)])
def test_invalid_layout_geometry_fails_closed(shape):
    with pytest.raises(GGUFError, match='shape'):
        tiled.plane_sizes(shape)


def test_invalid_host_payload_and_plane_extents_fail_closed():
    raw = np.zeros(64*50, np.uint8)
    for invalid in (raw.astype(np.uint16), raw[:-1], np.zeros(raw.size*2, np.uint8)[::2], mx.array(raw)):
        with pytest.raises(GGUFError, match='host U8'):
            tiled.pack(invalid, (64, 256))
    planes = tiled.pack(raw, (64, 256))
    for invalid in (planes[:2], (planes[0][:-1], *planes[1:]), (planes[0].reshape(-1, 4), *planes[1:])):
        with pytest.raises(GGUFError, match='host planes'):
            tiled.unpack(invalid, (64, 256))


def inputs(rows=1, experts=3, capacity=2):
    x = mx.zeros((rows, 256), mx.float32)
    ids = mx.array(np.arange(rows*2, dtype=np.uint32).reshape(rows, 2) % experts)
    scores = mx.full(ids.shape, .5, mx.float32)
    planes = tuple(mx.zeros((size,), mx.uint8) for size in tiled.bank_plane_sizes(capacity, 256, 256))
    slots = mx.array([i if i < capacity else tiled.MISSING_SLOT for i in range(experts)], mx.uint32)
    return x, ids, scores, planes, slots


@pytest.mark.parametrize('prefill', [False, True])
def test_empty_bank_never_binds_or_reads_empty_planes(monkeypatch, prefill):
    x, ids, scores, planes, slots = inputs(capacity=0)
    def forbidden(*args, **kwargs):
        raise AssertionError('empty bank must not dispatch any weight kernel')
    monkeypatch.setattr(tiled, 'kernel', forbidden)
    actual = tiled.moe(x, ids, scores, planes, slots, capacity=0,
                       hidden_size=256, intermediate_size=256, prefill=prefill)
    np.testing.assert_array_equal(np.array(actual), 0)
    np.testing.assert_array_equal(np.array(tiled.missing_routes(ids, slots, 0)), True)
    assert sum(p.nbytes for p in planes) == 0


@pytest.mark.parametrize('rows', [1, 16, 1024])
def test_exact_dispatch_never_reads_host_routes_and_keeps_bank_objects(monkeypatch, rows):
    x, ids, scores, planes, slots = inputs(rows=rows)
    calls = []
    def kernel(kind):
        def call(**kw):
            calls.append((kind, kw))
            return [mx.zeros(shape, dtype) for shape, dtype in zip(kw['output_shapes'], kw['output_dtypes'])]
        return call
    def forbidden(*args, **kwargs):
        raise AssertionError('host conversion/synchronization in hot route')
    monkeypatch.setattr(tiled, 'kernel', kernel)
    monkeypatch.setattr(np, 'asarray', forbidden)
    monkeypatch.setattr(np, 'array', forbidden)
    monkeypatch.setattr(mx, 'eval', forbidden)
    monkeypatch.setattr(mx, 'synchronize', forbidden)
    result = tiled.moe(x, ids, scores, planes, slots, capacity=2, hidden_size=256, intermediate_size=256)
    tiled.missing_routes(ids, slots, 2)
    assert result.shape == x.shape
    assert [kind for kind, _ in calls] == ['exact_gate', 'exact_down', 'exact_combine', 'missing']
    for _, call in calls[:2]:
        template = dict(call['template'])
        assert template['ROWS'] == 1 and template['SIMDS'] == 4 and template['EXPERTS'] == 3
        assert call['inputs'][1] is ids and call['inputs'][2] is slots
        assert call['grid'] == (128, 64, rows*2)
    assert all(a is b for a, b in zip(calls[0][1]['inputs'][4:], planes[:6]))
    assert all(a is b for a, b in zip(calls[1][1]['inputs'][3:], planes[6:]))


@pytest.mark.parametrize('rows,experts', [(16, 3), (64, 384), (1024, 384)])
def test_prefill_device_sort_and_descriptor_bound_preserve_original_slots(monkeypatch, rows, experts):
    x, _, _, planes, slots = inputs(rows=rows, experts=experts)
    raw_ids = ((np.arange(rows*6, dtype=np.uint32)*97+383) % experts).reshape(rows, 6)
    ids, scores = mx.array(raw_ids), mx.full(raw_ids.shape, 1/6, mx.float32)
    calls = []
    def kernel(kind):
        def call(**kw):
            calls.append((kind, kw))
            return [mx.zeros(shape, dtype) for shape, dtype in zip(kw['output_shapes'], kw['output_dtypes'])]
        return call
    def forbidden(*args, **kwargs):
        raise AssertionError('no host array conversion during route planning')
    monkeypatch.setattr(tiled, 'kernel', kernel)
    with monkeypatch.context() as scope:
        scope.setattr(np, 'array', forbidden)
        scope.setattr(np, 'asarray', forbidden)
        scope.setattr(mx, 'eval', forbidden)
        scope.setattr(mx, 'synchronize', forbidden)
        tiled._prefill(x, ids, scores, planes, slots, 2, 256, 256)
    assert [name for name, _ in calls] == ['prefill_scan', 'prefill_gate', 'prefill_down', 'prefill_combine']
    order = np.array(calls[1][1]['inputs'][1])
    inverse = np.array(calls[-1][1]['inputs'][1])
    expected = np.argsort(raw_ids.ravel(), kind='stable')
    np.testing.assert_array_equal(order, expected)
    np.testing.assert_array_equal(order[inverse], np.arange(ids.size))
    scan = calls[0][1]
    assert dict(scan['template']) == dict(ROWS=ids.size, EXPERTS=experts,
        MAX_TILES=v41_prefill.plan_capacity(ids.size, experts), TILE=16)
    assert dict(calls[1][1]['template'])['MT'] == dict(calls[2][1]['template'])['MT'] == 1
    assert scan['threadgroup'] == (512, 1, 1)
    assert scan['inputs'][2] is slots
    assert scan['inputs'][3].dtype == mx.uint32 and scan['inputs'][3].shape == (1,)
    np.testing.assert_array_equal(np.array(scan['inputs'][3]), [2])
    assert all(a is b for a, b in zip(calls[1][1]['inputs'][5:11], planes[:6]))


def test_generated_sources_guard_invalid_experts_before_map_and_plane_access():
    for source in (tiled.MISSING_SOURCE, tiled.EXACT_PREFIX):
        assert source.index('expert>=EXPERTS') < source.index('slot_map[expert]')
    assert tiled.SCAN_SOURCE.index('expert>=EXPERTS') < tiled.SCAN_SOURCE.index('slot_map[expert]')
    assert 'else if(slot_map[expert]>=capacity)' in tiled.SCAN_SOURCE
    for kind in ('gate', 'down'):
        _, _, source = tiled.exact_spec(kind)
        assert source.index('slot>=capacity') < source.index('v41_iq1_tiled(')
        assert 'for(uint k=lane;k<K;k+=32)' in source
        _, original = v41_row_exact.kernel_spec(kind, 1, False)
        marker = 'for(uint r=0;r<ROWS;r++) {\n    float '
        assert source.split(marker)[1] == original.replace('order[first+r]', '(first+r)').split(marker)[1]
        _, _, nax = tiled.prefill_spec(kind)
        assert nax.index('if(expert<EXPERTS)') < nax.index('slot_map[expert]')
        assert nax.index('if(slot>=capacity)') < nax.index('threadgroup_barrier')
        assert 'half(v41_iq1_tiled(' in nax
        # Existing numerical epilogue copied verbatim: BF16 stores, clamps,
        # pre-down scale and precise-exp are not replaced with Splash math.
        base = v41_prefill.GATE_SOURCE if kind == 'gate' else v41_prefill.DOWN_SOURCE
        assert nax[nax.rindex('#pragma clang loop unroll(full)\nfor(int f=0;'): ] == base[base.rindex('#pragma clang loop unroll(full)\nfor(int f=0;'): ]
    _, _, combine = tiled.prefill_spec('combine')
    assert combine.index('if(status[0]!=0)') < combine.index('rows[')
    assert 'scores[original]*' not in combine
    for function in (tiled.moe, tiled.missing_routes, tiled._prefill):
        source = inspect.getsource(function)
        assert 'np.' not in source and '.item(' not in source and 'mx.eval(' not in source


def test_input_admission_and_nax_device_policy_fail_closed():
    x, ids, scores, planes, slots = inputs()
    kw = dict(capacity=2, hidden_size=256, intermediate_size=256)
    for bad in (-1, True, 4):
        with pytest.raises(GGUFError, match='routes'):
            tiled.missing_routes(ids, slots, bad)
    with pytest.raises(GGUFError, match='routes'):
        tiled.missing_routes(ids.astype(mx.int32), slots, 2)
    with pytest.raises(GGUFError, match='geometry'):
        tiled.moe(x, ids, scores, planes[:-1], slots, **kw)
    with pytest.raises(GGUFError, match='geometry'):
        tiled.moe(x, ids, scores.astype(mx.int32), planes, slots, **kw)
    with pytest.raises(GGUFError, match='scratch'):
        tiled.moe(x, ids, scores, planes, slots, **kw, max_scratch_bytes=0)
    with pytest.raises(GGUFError, match='owned Apple M5'):
        tiled.moe(x, ids, scores, planes, slots, **kw, prefill=True)


def test_bank_growth_changes_runtime_bounds_not_specialization_keys(monkeypatch):
    ids = mx.array([[0, 1, 2, 3, 384, 0xffffffff]], mx.uint32)
    x, scores = mx.zeros((1, 256)), mx.ones(ids.shape)
    templates, bounds = {}, []
    def kernel(kind):
        def call(**kw):
            current = tuple(kw['template'])
            assert 'CAPACITY' not in dict(current)
            if kind in templates:
                assert current == templates[kind]
            templates[kind] = current
            if kind == 'missing':
                values, slot_map, scalar = (np.array(v) for v in kw['inputs'])
                assert scalar.dtype == np.uint32 and scalar.shape == (1,)
                capacity = int(scalar[0])
                bounds.append(capacity)
                mask = np.ones(values.shape, bool)
                valid = values < len(slot_map)
                mask[valid] = slot_map[values[valid]] >= capacity
                return [mx.array(mask)]
            if kind == 'prefill_scan':
                scalar = kw['inputs'][3]
                assert scalar.dtype == mx.uint32 and scalar.shape == (1,)
                assert int(scalar.item()) == bounds[-1]
            return [mx.zeros(shape, dtype) for shape, dtype in zip(kw['output_shapes'], kw['output_dtypes'])]
        return call
    monkeypatch.setattr(tiled, 'kernel', kernel)
    for capacity in (1, 2, 383, 384):
        slots = mx.array([0, capacity-1, capacity, tiled.MISSING_SLOT]+[tiled.MISSING_SLOT]*380, mx.uint32)
        planes = tuple(mx.zeros((n,), mx.uint8) for n in tiled.bank_plane_sizes(capacity, 256, 256))
        mask = tiled.missing_routes(ids, slots, capacity)
        np.testing.assert_array_equal(np.array(mask), [[False, False, True, True, True, True]])
        tiled.moe(x, ids, scores, planes, slots, capacity=capacity, hidden_size=256, intermediate_size=256)
        tiled._prefill(x, ids, scores, planes, slots, capacity, 256, 256)
        for kind, plane in (('gate', 'gqs'), ('down', 'wqs')):
            source = tiled.exact_spec(kind)[2]
            assert f'{plane}_shape[0]/(ulong(N)*K/8)' in source
            assert 'CAPACITY' not in source
            assert 'wqs_shape[0]/(ulong(N)*K/8)' in tiled.prefill_spec(kind)[2]
    assert bounds == [1, 2, 383, 384]
    assert set(templates) == {'missing', 'exact_gate', 'exact_down', 'exact_combine',
                              'prefill_scan', 'prefill_gate', 'prefill_down', 'prefill_combine'}


def test_declared_capacity_cannot_exceed_any_physical_plane(monkeypatch):
    x, ids, scores, planes, slots = inputs(capacity=2)
    one = tiled.bank_plane_sizes(1, 256, 256)
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid physical bank reached a kernel')
    monkeypatch.setattr(tiled, 'kernel', forbidden)
    for index in range(9):
        invalid = list(planes)
        invalid[index] = mx.zeros((one[index],), mx.uint8)
        with pytest.raises(GGUFError, match='geometry'):
            tiled.moe(x, ids, scores, invalid, slots, capacity=2,
                      hidden_size=256, intermediate_size=256)


@pytest.mark.parametrize('kind', ['missing', 'exact_gate', 'exact_down', 'exact_combine',
                                 'prefill_scan', 'prefill_gate', 'prefill_down', 'prefill_combine'])
@pytest.mark.parametrize('constant_inputs', [False, True])
def test_offline_metal4_compiles_without_gpu_execution(tmp_path, kind, constant_inputs):
    if not shutil.which('xcrun') or subprocess.run(['xcrun', '--find', 'metal'], capture_output=True).returncode:
        pytest.skip('offline Metal compiler unavailable')
    source = tmp_path/(kind+'.metal')
    source.write_text(tiled.metal_compile_source(kind, constant_inputs=constant_inputs))
    result = subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-std=metal4.0', '-c', str(source),
                             '-o', str(source.with_suffix('.air'))], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.parametrize('kind', ['scan', 'gate', 'down'])
def test_v41_segmented_prefill_compiles_offline_with_its_templates(tmp_path, kind):
    # The shared segmented tiles gained GLM-only MT/TILE/K-block parameters;
    # V4.1 must pass MT=1 and TILE=16 and carry no K-block fill.
    if not shutil.which('xcrun') or subprocess.run(['xcrun', '--find', 'metal'], capture_output=True).returncode:
        pytest.skip('offline Metal compiler unavailable')
    names, outputs, source = {
        'scan': (['sorted_ids', 'original_ids'], ['tiles', 'tile_count', 'status'], v41_prefill.SCAN_SOURCE),
        'gate': (['x', 'order', 'tiles', 'tile_count', 'w', 'up', 'scores'], ['y'], v41_prefill.GATE_SOURCE),
        'down': (['x', 'tiles', 'tile_count', 'w'], ['y'], v41_prefill.DOWN_SOURCE),
    }[kind]
    assert 'kblock_values32' not in source
    dtype = {'x': 'float', 'scores': 'float', 'w': 'uchar', 'up': 'uchar'}
    bindings = [f"device const {dtype.get(n, 'uint')}* {n} [[buffer({i})]]" for i, n in enumerate(names)]
    bindings += [f"device {'uint' if kind == 'scan' else 'float'}* {n} [[buffer({i})]]"
                 for i, n in enumerate(outputs, len(bindings))]
    attrs = ["uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
             "uint3 thread_position_in_grid [[thread_position_in_grid]]",
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]",
             "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
             "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]"]
    constants = (f"constant uint ROWS=192,EXPERTS=384,MAX_TILES=17,TOPK=6,K=4096,N=2048,FORMAT=19,"
                 f"ROW_BYTES=1100,TILE={v41_prefill.TILE_ROWS},MT={v41_prefill.MT};\n")
    path = tmp_path/(kind+'.metal')
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n"+v41_prefill.HEAD+constants
                    +"kernel void probe("+",".join(bindings+attrs)+") {\n"+source+"\n}\n")
    result = subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-std=metal4.0', '-c', str(path),
                             '-o', str(path.with_suffix('.air'))], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
