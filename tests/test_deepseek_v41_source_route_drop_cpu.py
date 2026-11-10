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


@pytest.mark.parametrize('extra', [{'deepseek_physical_batch': 'b2'}, {'draft_model': 'embedded_dspark', 'num_draft': 3},
                                   {approx.ROUTE_KEY: 'resident_v1'}])
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
    assert not d.applies(12) and d.applies(13) and d.applies(26) and not d.applies(27)


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
    _, got = inside.router(one)
    np.testing.assert_allclose(np.array(got)[0, 0], reference_drop(np.array(w)[0, 0]), rtol=1e-6)
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
