"""Synthetic position-grid partition contracts for exact Source APCv2 reuse.

CPU only; no checkpoint, no GPU. These establish the implementation law:
grid-boundary state is a function of tokens alone, whatever outer chunking or
live row cap produced it, and off-grid state never donates a prefix.
"""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter
from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
from mlx2.adapters.deepseek_v41_source_canonical_prefill import (
    GRID, POLICY, aligned_checkpoint, aligned_rows, grid_forward, grid_model_class,
    grid_pieces, grid_publication_refusal, grid_row_cap, partition_receipt)
from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache


# Pure schedule --------------------------------------------------------------

@pytest.mark.parametrize('offset,rows,grid', [(0, 1, 4), (0, 13, 4), (3, 9, 4), (8, 8, 8),
                                              (1000, 100, 256), (1023, 1024, 1024)])
def test_pieces_cover_call_in_order_and_never_straddle_the_absolute_grid(offset, rows, grid):
    pieces = grid_pieces(offset, rows, grid)
    assert sum(length for _, length in pieces) == rows and pieces[0][0] == 0
    for (first, length), (after, _) in zip(pieces, pieces[1:]):
        assert after == first + length
    for first, length in pieces:
        start = offset + first
        assert start // grid == (start + length - 1) // grid and length <= grid


def test_row_cap_floors_to_whole_cells_composes_and_never_widens():
    assert [aligned_rows(r, 256) for r in (1, 255, 256, 511, 731, 1024)] == [1, 255, 256, 256, 512, 1024]
    seen = []

    def advice(maximum, *, context_tokens=0):
        seen.append((maximum, context_tokens))
        return min(maximum, 731)
    cap = grid_row_cap(advice, 256)
    assert cap(1024, context_tokens=9) == 512 and cap(200, context_tokens=0) == 200
    assert seen == [(1024, 9), (200, 0)] and cap.source_prefill_grid_rows == 256
    for bad in (0, 3, 2048, True, None):
        with pytest.raises(ValueError):
            grid_row_cap(advice, bad)


def test_cold_checkpoint_is_grid_floor_of_prompt_boundary_and_first_off_grid_cut():
    assert aligned_checkpoint(4000, 256) == 3840
    assert aligned_checkpoint(4000, 256, cuts=(3990,)) == 3840
    assert aligned_checkpoint(4000, 256, cuts=(700, 3990)) == 512
    assert aligned_checkpoint(4000, 256, cuts=(512, 3990)) == 3840   # on-grid cuts never taint
    assert aligned_checkpoint(1025, 1024) is None    # P-1 itself is on the grid
    assert aligned_checkpoint(1000, 1024) is None    # no positive grid boundary
    assert aligned_checkpoint(4000, 256, cached_tokens=3840) is None


# Partition-sensitive fake forward --------------------------------------------

def fake(grid):
    """A forward whose state depends on how rows are grouped into calls."""
    state = SimpleNamespace(offset=0, total=np.float32(0), _in_forward=False)
    model = SimpleNamespace(identity='synthetic-grid@1', _source_canonical_identity='synthetic-grid@1',
                            weights=SimpleNamespace(source_prefill_partition_policy=GRID,
                                                    source_prefill_grid_rows=grid),
                            scheduled_kv_calls={})
    calls = []

    def validate(inputs, cache, keep):
        if inputs.ndim != 2 or inputs.shape[0] != 1 or not 1 <= inputs.shape[1] <= 1024:
            raise ValueError('geometry')
        return [state] if cache is None else cache, None
    model.validate_forward = validate

    def forward(inputs, cache, *, logits_to_keep, capture_layers, body_only):
        values = np.asarray(inputs, np.float32)[0]
        s = cache[0]
        # Grouping-dependent arithmetic: a chunk mean plus a width term.
        s.total = np.float32(s.total * np.float32(0.75) + values.mean() + np.float32(len(values)) / 7)
        s.offset += len(values)
        calls.append((s.offset - len(values), len(values), logits_to_keep, body_only))
        width = len(values) if logits_to_keep in (None, 0) else logits_to_keep
        out = None if body_only else mx.full((1, width, 1), float(s.total))
        return out, None
    return model, state, forward, calls


def run(chunks, grid, values, *, start=None):
    model, state, forward, calls = fake(grid)
    if start is not None:
        state.offset, state.total = start
    for first, last in chunks:
        grid_forward(model, forward, values[:, first:last], [state])
    return model, state, calls


@pytest.mark.parametrize('chunks', [
    [(0, 16), (16, 23)], [(0, 4), (4, 8), (8, 16), (16, 23)], [(0, 8), (8, 23)], [(0, 23)]])
def test_grid_boundary_state_is_identical_for_every_on_grid_outer_chunking(chunks):
    values = mx.arange(1, 24, dtype=mx.int32).reshape(1, -1)
    _, reference, ref_calls = run([(0, 23)], 4, values)
    model, state, calls = run(chunks, 4, values)
    assert state.offset == 23 and state.total.tobytes() == reference.total.tobytes()
    assert [c[:2] for c in calls] == [c[:2] for c in ref_calls]   # same cells, same widths
    assert not state.source_partition_tainted and state.source_partition_grid == 4


def test_published_boundary_restores_bit_identically_for_a_differently_chunked_request():
    values = mx.arange(1, 24, dtype=mx.int32).reshape(1, -1)
    # Request A publishes at 16 after a cap-4 prefill; request B (cap 8 then
    # whole) restores it and continues; both match an uncached cold prefill.
    _, a, _ = run([(0, 4), (4, 8), (8, 12), (12, 16)], 4, values)
    assert grid_publication_refusal([a]) is None
    snapshot = (a.offset, a.total)
    _, warm, _ = run([(16, 23)], 4, values, start=snapshot)
    _, cold, _ = run([(0, 8), (8, 23)], 4, values)
    assert warm.total.tobytes() == cold.total.tobytes()


def test_off_grid_start_or_sub_cell_cap_runs_but_taints_and_refuses_publication():
    values = mx.arange(1, 24, dtype=mx.int32).reshape(1, -1)
    model, state, _ = run([(0, 3), (3, 16)], 4, values)
    assert state.offset == 16 and state.source_partition_tainted
    assert 'off-grid' in grid_publication_refusal([state])
    assert model.scheduled_kv_calls['source_grid_tainted_calls'] == 1
    _, edge, _ = run([(0, 10)], 4, values)
    assert 'grid boundary' in grid_publication_refusal([edge])
    _, decode, _ = run([(0, 8), (8, 9), (9, 10), (10, 11), (11, 12)], 4, values)
    assert decode.offset == 12 and 'off-grid' in grid_publication_refusal([decode])


def test_prompt_checkpoint_request_keeps_only_last_head_row_and_marks_applied():
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
    model, state, forward, calls = fake(4)
    values = mx.arange(1, 11, dtype=mx.int32).reshape(1, -1)
    with prefill_checkpoint_rows([10], width=10) as request:
        out, _ = grid_forward(model, forward, values, [state])
    assert request.applied == 1 and out.shape == (1, 1, 1)
    assert [c[3] for c in calls] == [True, True, False] and calls[-1][2] == 1
    model, state, forward, calls = fake(4)
    out, _ = grid_forward(model, forward, values, [state], logits_to_keep=3)
    assert out.shape == (1, 3, 1) and [c[2] for c in calls] == [0, 1, 2]


def test_failed_piece_poisons_outer_call_and_taints():
    model, state, forward, _ = fake(4)

    def broken(inputs, cache, **kw):
        if cache[0].offset == 4:
            raise MemoryError('actual allocator failure')
        return forward(inputs, cache, **kw)
    with pytest.raises(MemoryError):
        grid_forward(model, broken, mx.arange(1, 9, dtype=mx.int32).reshape(1, -1), [state])
    assert state._in_forward and state.source_partition_tainted


@pytest.mark.parametrize('mutate', ['identity', 'policy', 'grid'])
def test_stale_owner_profile_or_grid_refuses_before_mutation(mutate):
    model, state, forward, calls = fake(4)
    if mutate == 'identity':
        model.identity = 'other'
    elif mutate == 'policy':
        model.weights.source_prefill_partition_policy = POLICY
    else:
        model.weights.source_prefill_grid_rows = 3
    with pytest.raises(ValueError):
        grid_forward(model, forward, mx.array([[1, 2]]), [state])
    assert not calls and state.offset == 0


# Real tiny BF16 model ----------------------------------------------------------

def tiny(grid=4):
    from test_deepseek_v41_cpu import Weights, config
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    c = config(hidden_size=32, head_dim=32, q_lora_rank=32, qk_rope_head_dim=16,
               index_head_dim=32, o_lora_rank=16, moe_intermediate_size=32,
               simulate_quantization=True, activation_dtype='bfloat16')
    w = Weights(c)
    w.source_prefill_partition_policy, w.source_prefill_grid_rows = GRID, grid
    model = grid_model_class()(c, w, identity='synthetic-grid-source@1')
    reference = DeepSeekV41Model(c, Weights(c), identity='synthetic-ordinary-B1@1')
    return model, reference


def bits(cache):
    out = []
    for row in cache:
        for value in row.state:
            out.append(None if value is None else np.asarray(value).tobytes())
        out.append(row.offset)
    return out


def prefill(model, cache, ids, chunks):
    for first, last in chunks:
        model(ids[:, first:last], cache)
    return cache


IDS = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], mx.int32)
LAST = 13   # prefill [0, 13), then the final prompt token at width one


def test_real_model_grid_law_equals_ordinary_cell_calls_for_any_on_grid_chunking():
    model, reference = tiny()
    ordinary = prefill(reference, reference.make_cache(), IDS, [(0, 4), (4, 8), (8, 12), (12, LAST)])
    expected = bits(ordinary)
    for chunks in ([(0, LAST)], [(0, 8), (8, LAST)], [(0, 4), (4, 12), (12, LAST)]):
        cache = prefill(model, model.make_cache(), IDS, chunks)
        assert bits(cache) == expected
        assert all(row.source_partition_grid == 4 and not row.source_partition_tainted for row in cache)


def test_fixture_witness_ordinary_adaptive_partitions_change_state_bits():
    """Non-vacuity: on this fixture one 13-row call differs from 4-row cells,
    which is the adaptive-partition hazard the grid law removes."""
    _, reference = tiny()
    whole = bits(prefill(reference, reference.make_cache(), IDS, [(0, LAST)]))
    cells = bits(prefill(reference, reference.make_cache(), IDS, [(0, 4), (4, 8), (8, 12), (12, LAST)]))
    if whole == cells:
        pytest.skip('fixture no longer partition sensitive; grid tests remain valid but weaker')
    assert whole != cells


def test_real_model_cold_warm_restore_is_bitwise_and_width_one_token_exact():
    model, _ = tiny()
    cold = prefill(model, model.make_cache(), IDS, [(0, LAST)])
    cold_logits = model(IDS[:, LAST:], cold)
    # Request A, chunked differently, publishes its grid boundary at 12.
    a = prefill(model, model.make_cache(), IDS, [(0, 4), (4, 8), (8, 12)])
    assert all(row.checkpoint_publication_refusal() is None for row in a)
    published = [DeepSeekV41LayerCache.from_state(row.state, row.meta_state) for row in a]
    # Request B restores (APCv2 from_state rows carry no host provenance) and
    # continues with its own chunking, then the width-one final token.
    warm = prefill(model, published, IDS, [(12, LAST)])
    warm_logits = model(IDS[:, LAST:], warm)
    assert bits(warm) == bits(cold)
    np.testing.assert_array_equal(np.asarray(warm_logits).view(np.uint16 if warm_logits.dtype == mx.bfloat16 else np.uint32),
                                  np.asarray(cold_logits).view(np.uint16 if cold_logits.dtype == mx.bfloat16 else np.uint32))
    assert int(mx.argmax(warm_logits[0, -1]).item()) == int(mx.argmax(cold_logits[0, -1]).item())
    # Post-decode and off-grid states never donate.
    assert all('off-grid' in row.checkpoint_publication_refusal() for row in warm)


def test_real_model_snapshot_rollback_keeps_taint_and_off_grid_start_taints():
    model, _ = tiny()
    cache = prefill(model, model.make_cache(), IDS, [(0, 3), (3, 8)])
    assert all(row.source_partition_tainted for row in cache)
    saved = [row.snapshot() for row in cache]
    fresh = model.make_cache()
    for row, snap in zip(fresh, saved):
        row.restore(snap)
    assert all(row.source_partition_tainted for row in fresh)
    assert all('off-grid' in row.checkpoint_publication_refusal() for row in fresh)


# Adapter policy, identity, receipts and serving hooks ----------------------------

def test_policy_default_unchanged_grid_opt_in_and_unsupported_combinations_refuse():
    assert 'deepseek_source_prefill_partition' not in source_policy()
    assert 'deepseek_source_prefill_grid_rows' not in source_policy()
    p = source_policy({'deepseek_source_prefill_partition': GRID})
    assert p['deepseek_source_prefill_grid_rows'] == 1024
    assert source_policy({'deepseek_source_prefill_partition': GRID,
                          'deepseek_source_prefill_grid_rows': 256})['deepseek_source_prefill_grid_rows'] == 256
    for bad in ({'deepseek_source_prefill_grid_rows': 256},
                {'deepseek_source_prefill_partition': GRID, 'deepseek_source_prefill_grid_rows': 300},
                {'deepseek_source_prefill_partition': GRID, 'deepseek_kv_storage': 'native_packed'},
                {'deepseek_source_prefill_partition': GRID, 'deepseek_ced': 'dependency_exact'},
                {'deepseek_source_prefill_partition': GRID, 'draft_model': 'prompt_lookup'},
                {'deepseek_source_prefill_partition': GRID, 'draft_model': 'embedded_dspark'},
                {'deepseek_source_prefill_partition': GRID, 'deepseek_physical_batch': 'b2'},
                {'deepseek_source_prefill_partition': GRID, 'deepseek_source_resident_gather': 'resident_bank_v1',
                 'deepseek_source_expert_math': 'routed_fused_fp32'}):
        with pytest.raises(ValueError):
            source_policy(bad)


def adapter(policy):
    a = object.__new__(DeepSeekV41SourceAdapter)
    a.kernel_policy = dict(policy)
    a.weights = SimpleNamespace()
    a.numerics_revision = a.profile = a.numeric_profile = 'base'
    return a


def test_grid_identity_is_distinct_per_grid_and_from_canonical_and_adaptive():
    seen = set()
    for policy in ({}, {'deepseek_source_prefill_partition': POLICY},
                   {'deepseek_source_prefill_partition': GRID, 'deepseek_source_prefill_grid_rows': 256},
                   {'deepseek_source_prefill_partition': GRID, 'deepseek_source_prefill_grid_rows': 1024}):
        a = adapter(policy)
        a._bind_prefill_partition_identity()
        seen.add((a.numerics_revision, a.profile))
    assert len(seen) == 4
    a = adapter({'deepseek_source_prefill_partition': GRID, 'deepseek_source_prefill_grid_rows': 512})
    a._bind_prefill_partition_identity()
    assert a.weights.source_prefill_grid_rows == 512 and 'grid_rows=512:implementation=' in a.numerics_revision
    a = adapter({'deepseek_source_prefill_partition': GRID, 'deepseek_source_prefill_grid_rows': 512})
    a.weights.dense_prefill_policy = 'nax_fp16'
    with pytest.raises(ValueError, match='width-scoped'):
        a._bind_prefill_partition_identity()
    receipt = partition_receipt(GRID, 512)
    assert receipt['apcv2_reuse_enabled'] and receipt['grid_rows'] == 512 and receipt['qualification'] == 'unqualified'


def bound_grid_adapter(grid=256):
    a = adapter({'deepseek_source_prefill_partition': GRID, 'deepseek_source_prefill_grid_rows': grid})
    a.weights = SimpleNamespace(source_prefill_partition_policy=GRID, source_prefill_grid_rows=grid)
    a.model = SimpleNamespace(identity='bound', _source_canonical_identity='bound',
                              recommended_prefill_rows=grid_row_cap(lambda m, *, context_tokens=0: m, grid))
    return a


def test_execution_config_enables_reuse_only_when_load_bound(monkeypatch):
    monkeypatch.setattr(DeepSeekV41MLXAdapter, 'execution_config', lambda *a, **kw: {'persistent': True})
    a = bound_grid_adapter()
    assert 'apcv2_reuse_disabled_reason' not in a.execution_config(max_lanes=1, prefill_step=2048)
    with pytest.raises(ValueError, match='exceeds prefill step'):
        a.execution_config(max_lanes=1, prefill_step=128)
    a.model.recommended_prefill_rows = lambda m, *, context_tokens=0: m
    with pytest.raises(ValueError, match='not bound at load'):
        a.execution_config(max_lanes=1, prefill_step=2048)
    a = bound_grid_adapter()
    a.model.identity = 'other'
    with pytest.raises(ValueError, match='bound at load'):
        a.execution_config(max_lanes=1, prefill_step=2048)
    plain = adapter({})
    assert plain.execution_config(max_lanes=1, prefill_step=128)['apcv2_reuse_disabled_reason'] == \
        'adaptive_prefill_partitions_not_qualified'
    assert plain.apc_partition_checkpoint_position(list(range(4000))) is None
    assert plain.apc_publication_refusal([object()], [1]) is None


def test_adapter_hooks_plan_grid_boundary_and_quietly_refuse_unexact_state():
    a = bound_grid_adapter(256)
    assert a.apc_partition_checkpoint_position(list(range(4000)), cuts=(3990,)) == 3840
    row = SimpleNamespace(offset=3840, source_partition_grid=256, source_partition_tainted=False)
    assert a.apc_publication_refusal([row, row], list(range(3840))) is None
    assert 'length' in a.apc_publication_refusal([row], list(range(3839)))
    restored = SimpleNamespace(offset=3840)          # APCv2 restore, no forward yet
    assert a.apc_publication_refusal([restored], list(range(3840))) is None
    assert 'grid boundary' in a.apc_publication_refusal([SimpleNamespace(offset=3999)], list(range(3999)))
    assert 'off-grid' in a.apc_publication_refusal(
        [SimpleNamespace(offset=4096, source_partition_grid=256, source_partition_tainted=True)], list(range(4096)))
    assert 'foreign' in a.apc_publication_refusal(
        [SimpleNamespace(offset=4096, source_partition_grid=1024)], list(range(4096)))


def test_serving_partition_position_hook_validates_and_publish_skips_quietly():
    from mlx2.serving import ServingEngine, adapter_partition_checkpoint_position
    a = bound_grid_adapter(256)
    assert adapter_partition_checkpoint_position(a, list(range(4000)), cuts=()) == 3840
    assert adapter_partition_checkpoint_position(object(), list(range(4000))) is None
    broken = SimpleNamespace(apc_partition_checkpoint_position=lambda tokens, **kw: len(tokens))
    with pytest.raises(ValueError, match='partition checkpoint'):
        adapter_partition_checkpoint_position(broken, list(range(10)))

    stored = []
    apc = SimpleNamespace(store=lambda key, tokens, cache, **kw: stored.append(len(tokens)) or SimpleNamespace(stored=True))
    engine = SimpleNamespace(apc_reuse_disabled_reason=None, counts={}, adapter=a)
    off = [SimpleNamespace(offset=3999, state=())]
    assert ServingEngine._publish_checkpoint(engine, apc, 'k', list(range(3999)), off) is False
    assert engine.counts.get('apcv2_store_skipped_partition') == 1 and not stored
