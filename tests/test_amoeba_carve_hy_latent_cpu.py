"""aMoEba carve host on Hy4-flash's latent MoE (``hy_latent`` sites), CPU, tiny random checkpoint.

The Hy4 block: sigmoid router with correction bias (top-k, normalised, scaled), routed experts on a latent
(routed_expert_down_proj -> SwitchGLU with clamped SwiGLU -> routed_expert_up_proj), shared expert, and
combine_output_norm after routed + shared. Covered here, in expert and neuron units:

* the site is classified by structure and installed (dense layer 0 + latent MoE layers 1..5);
* exactness: no core set, and core = ones, give logits bit-identical to the unpatched model; the site's
  re-implemented routing reproduces the block (Control.verify 0.0); dropping all neurons of a set of experts
  equals expert-mode dropping those experts, bit for bit;
* the host's path gates (fast head, layer replay with fresh per-layer caches, layer reference) are exact;
* gradients: layer-wise == whole-graph, and the layer_ref directional gate passes (frozen routing replay
  engages, at float32-sized steps on this random fixture) and fails a broken replay;
* neuron box writes of the latent block are its routed contribution after the up projection.
"""
import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from research.amoeba.carve import units as U  # noqa: E402
from research.amoeba.carve.host import Host  # noqa: E402
from test_hy_v4_flash_cpu import TOKENS, build, checkpoint, tiny_config  # noqa: E402

EPS = (2 ** -12, 2 ** -11, 2 ** -10, 2 ** -9)      # float32 fixture: smaller steps than the bf16 defaults


@pytest.fixture(scope="module")
def cw():
    cfg = tiny_config()
    return cfg, checkpoint(cfg)


def _fresh(cw):
    return build(*cw)[0]


def _host(cw, mu):
    return Host(model=_fresh(cw), tokenizer=None, fingerprint="tiny", checkpoint="tiny", moe_units=mu,
                log=lambda *a: None)


def _lg(m, ids):
    return np.array(m(ids))


@pytest.mark.parametrize("mu", ["expert", "neuron"])
def test_sites_and_bitwise_parity(cw, mu):
    ids = mx.array([TOKENS])
    ref = _lg(_fresh(cw), ids)
    m = _fresh(cw)
    sites, ctl = U.install(m.model.layers, moe_units=mu)
    assert [s.kind for s in sites] == ["dense"] + ["hy_latent"] * 5
    E, F = 8, 16
    assert sites[1].n == (E * F if mu == "neuron" else E)
    assert np.array_equal(_lg(m, ids), ref)                       # no core: the base modules
    ctl.core = [mx.ones((s.n,)) for s in sites]
    assert np.array_equal(_lg(m, ids), ref)                       # all open: bit for bit
    ctl.core, ctl.verify = None, {}
    _lg(m, ids)
    assert ctl.verify == {l: 0.0 for l in range(6)} or all(float(v) == 0.0 for v in ctl.verify.values())


def test_neuron_drop_of_whole_experts_equals_expert_drop(cw):
    ids = mx.array([TOKENS])
    keep_e = np.ones(8, np.float32)
    keep_e[[1, 4, 6]] = 0
    me = _fresh(cw)
    se, ce = U.install(me.model.layers, moe_units="expert")
    ce.core = [mx.ones((s.n,)) if s.kind == "dense" else mx.array(keep_e) for s in se]
    mn = _fresh(cw)
    sn, cn = U.install(mn.model.layers, moe_units="neuron")
    cn.core = [mx.ones((s.n,)) if s.kind == "dense" else mx.array(U.expert_to_neuron(keep_e, s.F)) for s in sn]
    a, b = _lg(me, ids), _lg(mn, ids)
    assert np.allclose(a, b, atol=1e-5, rtol=1e-5)                # expert drop: w*0; neuron drop: h*0 (same model)
    assert not np.allclose(a, _lg(_fresh(cw), ids))


@pytest.mark.parametrize("mu", ["expert", "neuron"])
def test_host_paths_and_gradient_gates(cw, mu):
    h = _host(cw, mu)
    ids = mx.array([TOKENS[:16], TOKENS[4:20]])
    p = h.check_paths(ids)
    assert p["fast_head"] and p["replay"] and p["replay_max_abs"] == 0.0
    assert h.check_ones_parity(ids)["max_abs"] == 0.0
    sp = h.check_site_parity(ids)
    assert sp["ok"] and sp["max_rel_l2"] == 0.0 and not sp["not_engaged"]
    assert h.check_layer_reference(ids)["ok"]
    one = mx.array([TOKENS[:16]])
    assert h.check_layerwise_grad(one), h.paths["layerwise_grad"]
    assert h.check_directional_grad(one, eps=EPS, pref_eps=2 ** -11), h.paths["layerwise_grad"]["fails"]
    r = h.paths["layerwise_grad"]
    assert r["fd_routing"].startswith("frozen") and r["replay_rel_gap"] < 1e-6 and r["replay_route_mismatch"] == 0


def test_directional_gate_fails_a_broken_replay(cw, monkeypatch):
    h = _host(cw, "neuron")
    ids = mx.array([TOKENS[:16]])
    h.check_paths(ids)
    assert h.check_directional_grad(ids, eps=EPS, pref_eps=2 ** -11)
    real = U._hy_route

    def broken(mx_, self, x, router_sg=False, force=None):
        ids_, w = real(mx_, self, x, router_sg, force)
        return ids_, (w * 1.5 if force is not None else w)        # replayed weights off by 50%
    monkeypatch.setattr(U, "_hy_route", broken)
    assert not h.check_directional_grad(ids, eps=EPS, pref_eps=2 ** -11)


def test_latent_neuron_box_write(cw):
    m = _fresh(cw)
    sites, ctl = U.install(m.model.layers, moe_units="neuron")
    rng = np.random.default_rng(0)
    box = [mx.array((rng.random(s.n) < 0.3).astype(np.float32)) for s in sites]
    ctl.core = [mx.zeros((s.n,)) for s in sites]
    ctl.boxes = [{"name": "b", "mask": box}]
    ctl.box_out = {}
    m(mx.array([TOKENS[:8]]))
    assert {k for k in ctl.box_out} == {("b", l) for l in range(6)}          # dense layer 0 + 5 latent sites
    assert all(v.shape[-1] == 32 for v in ctl.box_out.values())   # residual width after the up projection


# ------------------------------------------------------------------ block units (page-out aligned)

@pytest.fixture
def block8(monkeypatch):
    """block64 units at block 8 on the tiny fixture (F = 16: two blocks per expert)."""
    monkeypatch.setitem(U.UNIT_BLOCK, "block64", 8)


def test_block_units_are_neuron_masks_repeated_over_the_block(cw, block8):
    ids = mx.array([TOKENS])
    ref = _lg(_fresh(cw), ids)
    mb = _fresh(cw)
    sb, cb = U.install(mb.model.layers, moe_units="block64")
    assert [s.n for s in sb] == [48] + [8 * 2] * 5 and all(s.B == 8 for s in sb[1:])
    cb.core = [mx.ones((s.n,)) for s in sb]
    assert np.array_equal(_lg(mb, ids), ref)                      # all open: bit for bit
    rng = np.random.default_rng(2)
    kb = [(rng.random(s.n) < 0.6).astype(np.float32) for s in sb]
    cb.core = [mx.array(k) for k in kb]
    mn = _fresh(cw)
    sn, cn = U.install(mn.model.layers, moe_units="neuron")
    cn.core = [mx.array(k if s.kind == "dense" else np.repeat(k, 8)) for s, k in zip(sn, kb)]
    assert np.array_equal(_lg(mb, ids), _lg(mn, ids))              # = the neuron mask with each block repeated


def test_block_units_pass_the_host_gates(cw, block8):
    h = _host(cw, "block64")
    ids = mx.array([TOKENS[:16], TOKENS[4:20]])
    assert h.check_paths(ids)["replay"] and h.check_site_parity(ids)["ok"]
    one = mx.array([TOKENS[:16]])
    assert h.check_layerwise_grad(one)
    assert h.check_directional_grad(one, eps=EPS, pref_eps=2 ** -11), h.paths["layerwise_grad"]["fails"]


def test_block_box_exports_whole_blocks(cw):
    from mlx2.runtime.amoeba_neuron_mask import MaskSite, closed_from_unit_mask
    sites = [MaskSite(0, "d", None, 48), MaskSite(1, "e", 8, 16)]
    m = np.zeros((2, 48), np.uint8)
    m[0, [3, 5]] = 1
    m[1, [0, 5]] = 1                                                 # expert 0 block 0, expert 2 block 1
    c = closed_from_unit_mask(m, sites, unit_block=8)
    assert c[0].tolist() == [3, 5] and c[1].tolist() == list(range(0, 8)) + list(range(40, 48))
