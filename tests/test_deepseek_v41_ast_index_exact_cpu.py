"""Chronological index width and source/candidate ownership; synthetic CPU only."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_ast_batched_index_exact as index
from mlx2.adapters.deepseek_v41_ast_batched_attention import PATHS, DEPTHS
from mlx2.runtime.models.deepseek_v41 import Attention
from mlx2.runtime.gguf import source_ast_batched_index_matmul as kernel


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def inventory(prefix=1024, ratio=2):
    completed = tuple(node for node, depth in enumerate(DEPTHS) if (prefix+depth+1) % ratio == 0)
    prefix_count = prefix//ratio
    delta = {node: prefix_count+i for i, node in enumerate(completed)}
    rows = tuple(tuple(range(prefix_count))+tuple(delta[n] for n in path if n in delta) for path in PATHS)
    return rows, prefix_count+len(completed)


def test_count_classes_retain_real_columns_and_mixed_bypass_boundary():
    rows, physical = inventory()
    assert physical == 521
    assert index.count_classes(rows) == ((512, (0,)),
        (513, (1, 2, 6, 7, 11, 12)), (514, (3, 4, 8, 9, 13, 14)), (515, (5, 10, 15)))
    with pytest.raises(ValueError, match='16 query'):index.count_classes(rows[:15])
    with pytest.raises(ValueError, match='invalid'):index.count_classes(((0, 0),)+rows[1:])


@pytest.mark.parametrize('finite', (False, True))
def test_index_selection_matches_owning_b1_count_source_and_borrower(finite, monkeypatch):
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
    rng = np.random.default_rng(1925)
    logical, physical = inventory()
    config = SimpleNamespace(index_n_heads=32, index_head_dim=128, index_topk=512,
        candidate_source_layer_id=20, candidate_block_size=4, candidate_topk_blocks=2)
    model = SimpleNamespace(index_policy='finite_all_retained' if finite else 'reference', scheduled_kv_calls={})
    q = mx.array(rng.integers(-2, 3, (1, 16, 32*128)).astype(np.float32))
    w = mx.array(rng.integers(1, 4, (1, 16, 32)).astype(np.float32))
    keys = mx.array(rng.integers(-2, 3, (1, physical, 128)).astype(np.float32))
    attn = SimpleNamespace(c=config, model=model, layer=20, prefix='blk.20.', ratio=2,
        rotate_quantized=lambda value, *a, **kw: value)
    current = [None]
    def projection(_model, value, name, **kwargs):return q if 'attn_q_b' in name else w
    monkeypatch.setattr(dense, 'linear_tile', projection)
    def ordinary_projection(value, name):
        node = current[0]
        return (q if name == 'indexer.attn_q_b' else w)[:, node:node+1]
    attn.linear = ordinary_projection
    owner = SimpleNamespace(logical_rows=logical, index_table=keys, assert_current=lambda: None)
    positions = tuple(1024+d for d in DEPTHS)
    shared = [{} for _ in range(16)]; reference = [{} for _ in range(16)]
    original = index.matrix_scores; calls = []
    def score(query, rows, **kw):calls.append((query.shape[0], rows.shape[1]));return original(query, rows, **kw)
    monkeypatch.setattr(index, 'matrix_scores', score)
    for layer in (20, 24):
        attn.layer = layer
        index.select_tile(attn, mx.zeros((1, 16, 1)), mx.zeros((1, 16, 1)), owner, positions, shared)
        for node in range(16):
            current[0] = node
            source = SimpleNamespace(index_keys=mx.take(keys, mx.array(logical[node], mx.int32), axis=1))
            Attention.select(attn, mx.zeros((1, 1, 1)), mx.zeros((1, 1, 1)), source, positions[node], reference[node])
            for name in ('indices', 'selected_mask', 'candidates'):
                np.testing.assert_array_equal(np.asarray(shared[node][name]), np.asarray(reference[node][name]))
            assert shared[node]['candidates'].shape[-1] == len(logical[node])
        assert owner._exact_index_count_classes == index.count_classes(logical)
    grouped = [(6, 513), (6, 514), (3, 515)]
    # At the candidate source,512 keys still need candidate block ranking;
    # the later borrower can bypass root512 while other classes rank top-k.
    expected = [(1, 512)]+grouped+(grouped if finite else [(1, 512)]+grouped)
    assert calls == expected
    assert max(count for _, count in calls) == 515 < physical


def test_singleton_uses_owning_einsum_reshape_and_gpu_gate_refuses_unselected(monkeypatch):
    query = mx.zeros((1, 32, 128), mx.float32); keys = mx.zeros((1, 513, 128), mx.float32)
    original = mx.einsum; calls = []
    def einsum(equation, a, b):calls.append((equation, a.shape, b.shape));return original(equation, a, b)
    monkeypatch.setattr(mx, 'einsum', einsum)
    assert index.matrix_scores(query, keys).shape == (1, 32, 513)
    assert calls == [('bhsd,btd->bsht', (1, 32, 1, 128), (1, 513, 128))]
    monkeypatch.setattr(index, 'INDEX_GPU_IMPLEMENTATION', None)
    monkeypatch.setattr(mx, 'default_device', lambda: mx.gpu)
    with pytest.raises(ValueError, match='live caller'):index.matrix_scores(query, keys)
    with pytest.raises(ValueError, match='not admitted'):
        index.matrix_scores(mx.zeros((3, 32, 128)), mx.zeros((3, 513, 128)), check_live=lambda: True)
    assert index.INDEX_GPU_IMPLEMENTATION is None


@pytest.mark.parametrize('n', (129, 512, 513, 1024, 8195, 16384))
def test_regular_candidate_plan_is_m32_nt_not_a_change_to_m64(n):
    p = kernel.plan(3, n)
    assert (p['M'], p['N'], p['K']) == (32, n, 128)
    assert p['transpose_b'] and not p['align_M'] and p['align_K']
    assert p['align_N'] == (n % 64 == 0)
    assert p['grid'] == (((n+63)//64)*32, 2, 6)
    assert p['qualified'] is False and p['exact_gpu_claim'] is False
    for bad in (0, 1, 128, 2**31):
        with pytest.raises(ValueError, match='unsupported'):kernel.plan(3, bad)


def test_candidate_regular_body_keeps_stock_mma_and_cpu_refusal():
    spec = kernel.kernel_spec(3, 513)
    assert 'constexpr int BM=64,BN=64,BK=16,WM=2,WN=2;' in spec['source']
    assert 'constexpr bool align_M=false,align_N=false,align_K=true;' in spec['source']
    assert 'const uint3 tid=uint3(physical_tid.x,physical_tid.y,0u);' in spec['source']
    assert 'using gemm_kernel = GEMMKernel<' in spec['source']
    assert spec['header'].endswith('\n')
    with pytest.raises(ValueError, match='parent-owned GPU'):
        kernel.private_matmul(mx.zeros((3, 32, 128)), mx.zeros((3, 128, 513)), check_live=lambda: True)
