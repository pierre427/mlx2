"""H45b gated loop (model.Core._looped_gain): CPU oracles on a tiny core."""
import math

import mlx.core as mx
import pytest

from research.ens.model import Core, CoreConfig


@pytest.fixture(autouse=True)
def cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def cores(gain):
    mx.random.seed(3)
    a = Core(CoreConfig(layers=6, d=64, ffn=128, heads=4, loop_body=(2, 4)))
    mx.random.seed(3)
    b = Core(CoreConfig(layers=6, d=64, ffn=128, heads=4, loop_body=(2, 4), loop_gain=gain))
    b.backbone.update(a.backbone.parameters())
    return a, b


def test_conc_path_matches_fused_attention():
    a, _ = cores("decay")
    h = mx.random.normal((2, 12, 64))
    blk = a.backbone.blocks[2]
    y, c = blk(h, conc=True)
    assert mx.allclose(y, blk(h), atol=1e-4).item()
    assert c.shape == (2,) and bool(mx.all((c > 0) & (c <= 1)))


@pytest.mark.parametrize("gain", ["decay", "decaygain"])
def test_unit_gain_is_the_plain_loop(gain):
    a, b = cores(gain)
    b.loop_gamma_off = mx.array([40.0])                # gamma -> 1, kappa 0: every round's full update
    T = mx.random.randint(0, 100, (2, 12))
    six = mx.full((2,), 6)
    assert mx.allclose(a(T, passes=six), b(T, passes=six), atol=1e-3).item()


def test_decay_shrinks_late_rounds_and_reports_gains():
    _, b = cores("decaygain")
    T = mx.random.randint(0, 100, (3, 12))
    logits, g, c = b.looped(T, mx.full((3,), 6), return_gains=True)
    assert c.shape == (6, 3)
    assert g.shape == (5, 3)
    want = [0.9 ** k for k in range(1, 6)]             # kappa 0 at init: pure decay
    assert mx.allclose(g[:, 0], mx.array(want), atol=1e-4).item()
    b.loop_kappa_raw = mx.array([2.0])                  # kappa = 60
    _, g2, _ = b.looped(T, mx.full((3,), 6), return_gains=True)
    assert not mx.allclose(g, g2).item()                # sharpening changes the gains


def test_gain_core_refuses_mixed_passes_and_states():
    _, b = cores("decay")
    T = mx.random.randint(0, 100, (2, 12))
    with pytest.raises(ValueError):
        b(T, passes=mx.array([2, 6]))
    with pytest.raises(ValueError):
        b.looped(T, mx.full((2,), 6), return_states=True)


def test_padding_queries_do_not_count():
    _, b = cores("decaygain")
    blk = b.backbone.blocks[2]
    h = mx.random.normal((1, 12, 64))
    mask = mx.array([[True] * 7 + [False] * 5])
    _, c_masked = blk(h, conc=True, qmask=mask)
    h2 = mx.concatenate([h[:, :7], mx.random.normal((1, 5, 64))], axis=1)   # change only the padded positions
    _, c_masked2 = blk(h2, conc=True, qmask=mask)
    assert mx.allclose(c_masked, c_masked2, atol=1e-6).item()             # causal: real queries never see later rows
    _, c_all = blk(h, conc=True)
    _, c_all2 = blk(h2, conc=True)
    assert not mx.allclose(c_all, c_all2).item()


def test_decay_core_has_no_kappa_and_no_concentration():
    _, b = cores("decay")
    T = mx.random.randint(1, 100, (2, 12))
    _, g, c = b.looped(T, mx.full((2,), 6), return_gains=True)
    assert c is None and b.kappa() is None and g.shape == (5, 2)
