"""Opt-in approximate soft-reap probe: policy, oracle, isolation (CPU)."""
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_approx as approx
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.adapters.deepseek_v41_source_route_drop import RankDrop, drop_min, reference_drop

EXPERTS = {'deepseek_source_experts': {'mode': 'original_lfu', 'budget_bytes': 32 << 30}}


def test_off_by_default_and_layers_require_mode():
    assert approx.DROP_KEY not in source_policy(dict(EXPERTS))
    with pytest.raises(ValueError, match='require'):
        source_policy({**EXPERTS, approx.DROP_LAYERS: [13, 26]})


def test_parse_defaults_and_bounds():
    p = source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1'})
    assert p[approx.DROP_KEY] == 'drop_min_v1' and p[approx.DROP_LAYERS] == [13, 26]
    for bad in ([26, 13], [1], [-1, 3], ['a', 3], [1.0, 3]):
        with pytest.raises(ValueError):
            source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', approx.DROP_LAYERS: bad})
    with pytest.raises(ValueError):
        source_policy({**EXPERTS, approx.DROP_KEY: 'drop_rank6'})


# Route bias composes since 2026-11-09 (tests/test_deepseek_v41_source_route_compose_cpu.py).
@pytest.mark.parametrize('extra', [{'deepseek_physical_batch': 'b2'}, {'draft_model': 'embedded_dspark', 'num_draft': 3}])
def test_refuses_unimplemented_compositions(extra):
    with pytest.raises(ValueError):
        source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', **extra})


def test_identity_salt_and_apc_refusal():
    p = source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1'})
    q = source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', approx.DROP_LAYERS: [10, 30]})
    assert approx.selected_operations(p) == ('route_drop_min_v1',)
    assert approx.identity_salt(p) and approx.identity_salt(p) != approx.identity_salt(q)
    assert approx.apc_refusal_reason(p).startswith(approx.APC_REASON_PREFIX)
    r = approx.receipt(p)
    assert r['approximate'] and r['route_drop']['layers'] == [13, 26] and r['qualification'] == 'unqualified'


def test_mlx_drop_matches_oracle():
    import mlx.core as mx
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.random(6) + 0.05
        w = s / s.sum() * 1.5
        out, frac = drop_min(mx.array(w[None, None], mx.float32))
        ref = reference_drop(w)
        np.testing.assert_allclose(np.array(out)[0, 0], ref, rtol=1e-5, atol=1e-6)
        assert abs(float(np.array(out).sum()) - 1.5) < 1e-5
        assert abs(float(np.array(frac).item()) - w.min() / 1.5) < 1e-6
        assert (np.array(out)[0, 0] == 0).sum() == 1


def test_layer_range():
    d = RankDrop(mode='drop_min_v1', layers=[13, 26])
    assert not d.applies(12) and d.applies(13) and d.applies(26) and not d.applies(27) and d.count_for(13) == 1


def test_router_wrapper_gates_layers_rows_and_stamps(monkeypatch):
    import mlx.core as mx
    from types import SimpleNamespace
    from mlx2.adapters import deepseek_v41_source_route_drop as rd
    w = mx.array([[[0.5, 0.3, 0.2, 0.2, 0.15, 0.15]]], mx.float32)
    ids = mx.array([[[1, 2, 3, 4, 5, 6]]], mx.uint32)
    monkeypatch.setattr(rd, '_native_router', lambda block: (lambda x, image_mask=None: (ids, w)))
    drop = RankDrop(mode='drop_min_v1', layers=[13, 26])
    model = SimpleNamespace(source_route_drop=drop, source_approx_private_profile='mlx2.ds41.source-approx.v1:t')
    inside = SimpleNamespace(model=model, layer=13); outside = SimpleNamespace(model=model, layer=12)
    model.layers = [inside, outside]
    rd.install_router_wrappers(model); rd.install_router_wrappers(model)    # idempotent
    one, many = mx.zeros((1, 1, 8)), mx.zeros((1, 4, 8))
    got_ids, got = inside.router(one)
    np.testing.assert_allclose(np.array(got)[0, 0], reference_drop(np.array(w)[0, 0]), rtol=1e-6)
    # The dropped slot (lowest weight, first of the 0.15 tie at index 4) now names the top expert.
    assert np.array(got_ids)[0, 0].tolist() == [1, 2, 3, 4, 1, 6]
    assert inside.router(many)[1] is w and outside.router(one)[1] is w
    assert drop.stats['dropped_layer_rows'] == 1 and drop.stats['native_rows'] == 5

    class Row:
        ast_private_profile = None
    rows = [Row()]
    drop.after_forward(model, rows)
    assert rows[0].ast_private_profile == model.source_approx_private_profile
    r = drop.receipt()
    assert r['decode_forwards'] == 1 and abs(r['recent_mean_dropped_weight_fraction'] - 0.15 / 1.5) < 1e-6


def test_drop_count_two_matches_oracle_and_parses():
    import mlx.core as mx
    p = source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', approx.DROP_COUNT: 2})
    assert p[approx.DROP_COUNT] == 2
    assert approx.identity_salt(p) != approx.identity_salt(source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1'}))
    for bad in (0, 6, 1.0, True):
        with pytest.raises(ValueError):
            source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', approx.DROP_COUNT: bad})
    with pytest.raises(ValueError):
        source_policy({**EXPERTS, approx.DROP_COUNT: 2})
    rng = np.random.default_rng(1)
    for w in [np.array([0.3, 0.25, 0.2, 0.1, 0.1, 0.05]) * 1.5] + [(lambda s: s / s.sum() * 1.5)(rng.random(6) + .05) for _ in range(20)]:
        out, frac = drop_min(mx.array(w[None, None], mx.float32), 2)
        o = np.array(out)[0, 0]
        np.testing.assert_allclose(o, reference_drop(w, 2), rtol=1e-5, atol=1e-6)
        assert (o == 0).sum() == 2 and abs(o.sum() - 1.5) < 1e-5
        assert abs(float(np.array(frac).item()) - np.sort(w)[:2].sum() / 1.5) < 1e-6


def test_drop_five_keeps_only_top_expert():
    import mlx.core as mx
    w = np.array([0.3, 0.25, 0.2, 0.1, 0.1, 0.05]) * 1.5
    out, frac = drop_min(mx.array(w[None, None], mx.float32), 5)
    o = np.array(out)[0, 0]
    assert (o == 0).sum() == 5 and abs(o[0] - 1.5) < 1e-6
    np.testing.assert_allclose(o, reference_drop(w, 5), atol=1e-6)


def test_spans_resolve_with_override_and_restore():
    p = source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1',
                       approx.DROP_SPANS: [{'layers': [13, 26], 'count': 3}, {'layers': [18, 20], 'count': 1},
                                           {'layers': 22, 'count': 0}]})
    assert p[approx.DROP_SPANS] == [[13, 17, 3], [18, 20, 1], [21, 21, 3], [23, 26, 3]]
    counts = approx.drop_counts(p)
    assert counts[13] == 3 and counts[19] == 1 and 22 not in counts and 12 not in counts
    d = RankDrop.from_policy(p)
    assert d.applies(13) and not d.applies(22) and d.count_for(19) == 1
    q = source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', approx.DROP_SPANS: [{'layers': [13, 26], 'count': 3}]})
    assert approx.identity_salt(p) != approx.identity_salt(q)


def test_spans_validation():
    for bad in ([], [{'layers': [5, 3], 'count': 1}], [{'layers': 3, 'count': 6}], [{'layers': 3}],
                [{'layers': [1, 2], 'count': 0}]):
        with pytest.raises(ValueError):
            source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', approx.DROP_SPANS: bad})
    with pytest.raises(ValueError, match='one form'):
        source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', approx.DROP_COUNT: 2,
                       approx.DROP_SPANS: [{'layers': 3, 'count': 1}]})


def test_legacy_form_still_resolves():
    p = source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', approx.DROP_COUNT: 2})
    assert approx.drop_counts(p) == {layer: 2 for layer in range(13, 27)}


def test_redirect_never_names_a_dropped_expert():
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_source_route_drop import redirect_dropped, reference_redirect
    rng = np.random.default_rng(3)
    for count in (1, 2, 5):
        for _ in range(20):
            ids = rng.choice(384, 6, replace=False).astype(np.uint32)
            w = (lambda s: s / s.sum() * 1.5)(rng.random(6) + .05)
            out, _ = drop_min(mx.array(w[None, None], mx.float32), count)
            new = np.array(redirect_dropped(mx.array(ids[None, None]), out))[0, 0]
            o = np.array(out)[0, 0]
            np.testing.assert_array_equal(new, reference_redirect(ids, o))
            dropped = set(ids[o == 0].tolist())
            assert not dropped & set(new.tolist())            # never acquired
            assert set(new.tolist()) == set(ids[o > 0].tolist())
            assert new.dtype == ids.dtype


def test_redirected_slot_is_never_acquired_and_output_matches_zero_weight():
    # A dropped slot redirected to a kept expert with weight zero must give the same
    # output as the original id with weight zero, without acquiring the dropped expert.
    import sys
    from pathlib import Path
    import mlx.core as mx
    sys.path.insert(0, str(Path(__file__).parent))
    from test_deepseek_v41_source_mixed_reuse_cpu import _fixture, _repeated_qat_reference
    with mx.stream(mx.cpu):
        weights, bank, _, x, ids, scores, acquired, _, _ = _fixture(1, mx.float32, True)
        s = np.array(scores).copy(); i = np.array(ids).copy()
        drop = int(np.argmin(s[0])); top = int(np.argmax(s[0]))
        dropped_expert = int(i[0, drop])
        s[0, drop] = 0.0
        expected = _repeated_qat_reference(bank, x, mx.array(i), mx.array(s)); mx.eval(expected)
        i[0, drop] = i[0, top]
        acquired.clear()
        actual = weights.moe(x, mx.array(i), mx.array(s), "blk.0"); mx.eval(actual)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-6, atol=1e-6)
        assert dropped_expert not in [e for _, e in acquired]


def test_drop_miss_policy_and_math():
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_source_route_drop import drop_missing
    p = source_policy({**EXPERTS, approx.DROP_KEY: 'drop_miss_v1',
                       approx.DROP_SPANS: [{'layers': [0, 39], 'count': 2}]})
    assert p[approx.DROP_PROTECT] == 2 and p[approx.DROP_MAX_WEIGHT] == 0.10
    for bad in ({approx.DROP_PROTECT: 0}, {approx.DROP_MAX_WEIGHT: 0}, {approx.DROP_MAX_WEIGHT: True}):
        with pytest.raises(ValueError):
            source_policy({**EXPERTS, approx.DROP_KEY: 'drop_miss_v1', **bad})
    with pytest.raises(ValueError):
        source_policy({**EXPERTS, approx.DROP_KEY: 'drop_min_v1', approx.DROP_PROTECT: 2})
    w = mx.array([[[0.40, 0.25, 0.15, 0.09, 0.06, 0.05]]]) * 1.5
    res = mx.array([[[False, False, True, False, False, True]]])
    out, frac = drop_missing(w, res, count=2, protect=2, max_weight=0.10)
    o = np.array(out)[0, 0]
    # non-resident & rank>=2 & <10%: slots 3 (0.09) and 4 (0.06); slot 5 resident stays; top-2 protected
    assert (o == 0).tolist() == [False, False, False, True, True, False]
    assert abs(o.sum() - 1.5) < 1e-5 and abs(float(np.array(frac).item()) - 0.15) < 1e-5
    out1, _ = drop_missing(w, res, count=1, protect=2, max_weight=0.10)
    assert (np.array(out1)[0, 0] == 0).tolist() == [False, False, False, False, True, False]
    allres = mx.ones(w.shape, mx.bool_)
    out2, frac2 = drop_missing(w, allres, count=5, protect=1, max_weight=1.0)
    assert float(np.array(frac2).item()) == 0.0                       # everything resident: nothing dropped


def test_pooled_experts_count_as_resident():
    import threading
    from collections import OrderedDict
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_source_route_drop import resident_or_pooled
    keys = [(3, e) for e in range(8)]
    pool = SimpleNamespace(layers=[OrderedDict() for _ in range(4)])
    pool.layers[3][5] = object()
    store = SimpleNamespace(_lock=threading.Lock(), tables={'blk.3': keys}, _cache={(3, 1): 1}, _leased={},
                            q1_on_miss=SimpleNamespace(pools={'q1': pool}))
    assert resident_or_pooled(store, 3, 8).tolist() == [False, True, False, False, False, True, False, False]
