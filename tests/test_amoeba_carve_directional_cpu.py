"""CPU tests of the layer_ref gradient gate: g . delta against central differences of the masked fused
(eval) forward, over an epsilon sweep, with whole-graph and routing-flip diagnostics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_amoeba_carve_cpu import Tok, ids_batch, toy

from research.amoeba.carve.host import Host

EPS = tuple(f"{e:g}" for e in Host.DIRECTIONAL_EPS)


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def host(m, **k):
    return Host(model=m, tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None, **k)


def _neuron_toy():
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    return host(qn_toy(), moe_units="neuron")


@pytest.mark.parametrize("make", [lambda: host(toy(moe=False)), lambda: host(toy(moe=True)), _neuron_toy],
                         ids=["dense", "moe", "neuron"])
def test_directional_gate_passes_and_records_everything(make):
    h = make()
    ids = ids_batch(1, 12)
    h.check_paths(ids)
    assert h.grad_mode == "layer_ref"
    assert h.check_directional_grad(ids), h.paths["layerwise_grad"]
    r = h.paths["layerwise_grad"]
    json.dumps(r)
    assert r["gate"] == "directional" and r["ok"] and h.layerwise_ok
    assert r["k"] >= 8 and len(r["directions"]) == r["k"]
    moe = any(s.is_moe for s in h.sites)
    kinds = [d["kind"] for d in r["directions"]]
    assert kinds.count("sparse_routed" if moe else "sparse_random") >= 2
    for d in r["directions"]:
        assert {"g_dot_d", "nnz", *(f"fd@{e}" for e in EPS), *(f"rel_err@{e}" for e in EPS),
                *(f"pm_flips@{e}" for e in EPS)} <= set(d)
    assert set(r["per_eps"]) == set(EPS) and len(r["gate_eps"]) == 2
    assert r["fails"] == [] and r["fd_max_abs"] > 0 and r["same_fn_rel_gap"] <= r["same_fn_tol"]
    assert set(r["pair_pm_flips"]) == set(r["pair_clean_directions"]) and r["min_clean"] == 4
    assert len(r["gate_directions"]) >= 4 and not set(r["gate_directions"]) & set(r["excluded_directions"])
    for j in r["gate_directions"]:            # gated cells are flip-free at both gate eps
        assert all(r["directions"][j][f"pm_flips@{e:g}"] == 0 for e in r["gate_eps"])
    assert list(r["gate_eps"]) in [[a, b] for a, b in zip(r["eps"][:-1], r["eps"][1:])]
    for v in r["gate_values"].values():
        assert v["median_rel_err"] <= 0.10 and v["corr"] >= 0.99
    assert isinstance(r["outlier_directions"], list)
    assert {"loss_fused", "loss_reference", "loss_gap", "routing_flips", "whole_graph"} <= set(r)
    assert r["whole_graph"]["diagnostic"] and r["whole_graph"]["rel_l2"] == 0.0   # same arithmetic on CPU
    assert r["loss_gap"] == 0.0 and abs(r["loss_fused"] - r["loss_layerwise"]) < 1e-6
    rf = r["routing_flips"]
    moe_layers = [str(s.layer) for s in h.sites if s.is_moe]
    assert sorted(rf["per_layer"]) == sorted(moe_layers) and rf["total"] == 0.0
    assert (rf["slots"] > 0) == moe
    if moe:                                   # sparse directions touch only routed units
        routed = sum(d["nnz"] for d in r["directions"] if d["kind"] == "sparse_routed")
        assert 0 < routed < 2 * sum(h.widths)


def _broken(transform):
    real = Host._layerwise

    def broken(self, *a, **k):
        loss, g = real(self, *a, **k)
        c = g["core"]
        rows = [c[l] for l in range(len(c))]
        rows[1] = transform(rows[1])
        return loss, {"core": mx.stack(rows) if isinstance(c, mx.array) else rows}
    return broken


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("transform", [lambda x: -x, lambda x: 2.0 * x], ids=["sign_flip", "scaled_x2"])
def test_directional_gate_fails_a_replay_defect(monkeypatch, moe, transform):
    """The sign-flipped-layer pattern of test_layerwise_gate_still_fails_a_replay_defect, against the
    new gate (and a scaled layer)."""
    h = host(toy(moe=moe))
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    assert h.check_directional_grad(ids)
    monkeypatch.setattr(Host, "_layerwise", _broken(transform))
    assert not h.check_directional_grad(ids), h.paths["layerwise_grad"]["gate_values"]
    assert not h.layerwise_ok


def test_finite_differences_are_well_defined_on_moe_sites():
    """The mask is a continuous multiplier in the masked forward; on a MoE site (drop rule) it
    multiplies the routed weight: L is smooth in a routed expert's mask (differences agree across
    steps and with the gradient) and exactly flat in an unrouted one (gradient exactly 0)."""
    h = host(toy(moe=True))
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    l = 3                                                 # last MoE layer: no routing downstream
    m = h._probe_mask(0)
    h.train_mode(True)
    try:
        hl = h._probe_loss(ids)
        _, g = h._layerwise(ids, {"core": m}, lambda ll, sl: h.set_layer_masks(ll, sl, None), hl)
        _, routes = h._routes(ids, m, hl, None)
        routed = set(np.unique(routes[l]).tolist())
        rows = [np.array(m[i]) for i in range(h.L)]

        def L_at(e, t):
            rr = [r.copy() for r in rows]
            rr[l][e] += t
            return h._masked_loss(ids, h._layout(rr), hl, None)
        gl = np.array(g["core"][l])
        atol = 2e-2 * float(np.abs(gl).max())              # rounding on near-zero derivatives
        for e in range(h.widths[l]):
            # float32 losses: steps below ~1e-2 are rounding-limited on this toy's tiny derivatives
            fds = [(L_at(e, s) - L_at(e, -s)) / (2 * s) for s in (1e-1, 3e-2)]
            ge = float(np.array(g["core"][l])[e])
            if e in routed:                               # smooth: steps agree to O(eps^2), and with g
                assert abs(fds[0] - fds[-1]) <= 2e-2 * abs(fds[-1]) + atol
                assert abs(ge - fds[-1]) <= 2e-2 * abs(fds[-1]) + atol
            else:
                assert all(f == 0.0 for f in fds) and ge == 0.0
        assert routed and len(routed) < h.widths[l]
    finally:
        h.train_mode(False)


def test_driver_dispatches_the_gate_on_grad_mode(tmp_path, monkeypatch):
    from research.amoeba.carve import __main__ as D
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)
    base = ["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect,rank"]
    D.main(base + ["--out", str(tmp_path / "a")], host_factory=fac)
    p = json.loads((tmp_path / "a" / "carve.json").read_text())["inspect"]["paths"]["layerwise_grad"]
    assert p["gate"] == "directional" and p["ok"] and "whole_graph" in p and "routing_flips" in p
    # whole_train keeps the whole-graph gate (and never runs the directional one)
    called = []
    real = Host.check_layerwise_grad
    monkeypatch.setattr(Host, "check_directional_grad", lambda self, *a, **k: called.append(1) or False)
    monkeypatch.setattr(Host, "check_layerwise_grad", lambda self, *a, **k: real(self, *a, **k))
    D.main(base + ["--grad-mode", "whole_train", "--out", str(tmp_path / "b")], host_factory=fac)
    p = json.loads((tmp_path / "b" / "carve.json").read_text())["inspect"]["paths"]["layerwise_grad"]
    assert not called and p["gate"] == "whole_graph" and p["ok"]
    assert (p["min_cos"], p["min_layer_cos"], p["tol_rel_l2"]) == (0.99, 0.95, 0.15)
    # a failing layer_ref gate refuses unless --allow-whole-graph, on inspect and on resume
    with pytest.raises(SystemExit, match="directional-derivative"):
        D.main(base + ["--out", str(tmp_path / "c")], host_factory=fac)
    D.main(base + ["--out", str(tmp_path / "d"), "--allow-whole-graph"], host_factory=fac)
    D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect", "--out", str(tmp_path / "e")], host_factory=fac)
    with pytest.raises(SystemExit, match="directional-derivative"):
        D.main(base + ["--out", str(tmp_path / "e"), "--resume"], host_factory=fac)


def test_directional_gate_fails_vacuous_and_different_function(monkeypatch):
    """Review 2026-11-15: an all-zero derivative (masks not wired in) must not pass with corr 1, and the
    gradient's forward must be the FD forward (loss gap gated)."""
    h = host(toy(moe=True))
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    assert h.check_directional_grad(ids)
    real_ml = Host._fd_loss
    monkeypatch.setattr(Host, "_fd_loss", lambda self, ids, m, hl, mean, stats=None, replay=None:
                        real_ml(self, ids, self.ones(), hl, mean, stats, replay))
    real_lw = Host._layerwise

    def zero_grad(self, *a, **k):
        loss, g = real_lw(self, *a, **k)
        c = g["core"]
        return loss, {"core": mx.zeros_like(c) if isinstance(c, mx.array) else [mx.zeros_like(x) for x in c]}
    monkeypatch.setattr(Host, "_layerwise", zero_grad)
    assert not h.check_directional_grad(ids)
    r = h.paths["layerwise_grad"]
    assert r["fd_max_abs"] == 0 and any("vacuous" in f for f in r["fails"])
    monkeypatch.undo()
    real_gp = Host._grad_probe

    def shifted(self, *a, **k):
        pr = real_gp(self, *a, **k)
        pr["loss_layerwise"] *= 1.01
        return pr
    monkeypatch.setattr(Host, "_grad_probe", shifted)
    assert not h.check_directional_grad(ids)
    assert any("differ" in f for f in h.paths["layerwise_grad"]["fails"])
    assert h.check_directional_grad(ids, same_fn_tol=0.02)


def test_driver_passes_directional_knobs(tmp_path, monkeypatch):
    from research.amoeba.carve import __main__ as D
    seen = {}
    real = Host.check_directional_grad
    monkeypatch.setattr(Host, "check_directional_grad", lambda self, ids, **k: seen.update(k) or real(self, ids, **k))
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)
    D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect,rank", "--dir-grad-tol", "0.2", "--dir-grad-min-corr", "0.95",
            "--dir-grad-eps", "0.0078125,0.03125,0.125", "--out", str(tmp_path / "a")], host_factory=fac)
    assert seen["tol"] == 0.2 and seen["min_corr"] == 0.95 and seen["eps"] == (2 ** -7, 2 ** -5, 2 ** -3)
    p = json.loads((tmp_path / "a" / "carve.json").read_text())["inspect"]["paths"]["layerwise_grad"]
    assert p["eps"] == [2 ** -7, 2 ** -5, 2 ** -3] and p["tol_median_rel_err"] == 0.2


def test_directional_gate_needs_enough_flip_free_directions():
    """A step so large that every direction's +-e forwards route differently cannot gate."""
    h = host(toy(moe=True))
    ids = ids_batch(1, 12)
    h.check_paths(ids)
    assert not h.check_directional_grad(ids, eps=(0.5, 1.0), min_clean=8)
    r = h.paths["layerwise_grad"]
    assert any("free of +- routing flips" in f for f in r["fails"])


def test_frozen_routing_fd_on_moe_toys():
    """The FD side freezes the routed expert ids at m (the gradient's stop-gradiented ids); every gated
    cell is then smooth, the replayed forward reproduces L(m), and the free-routing FD is a diagnostic."""
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    for h in (host(qn_toy()), _neuron_toy()):                          # qwen_next sites, expert and neuron units
        assert {s.kind for s in h.sites if s.is_moe} == {"qwen_next"}
        ids = ids_batch(1, 12)
        h.check_paths(ids)
        assert h.check_directional_grad(ids), h.paths["layerwise_grad"]["fails"]
        r = h.paths["layerwise_grad"]
        assert r["fd_routing"].startswith("frozen") and r["replay_rel_gap"] < 1e-6
        assert len(r["gate_directions"]) == r["k"]                       # no cell excluded
        assert set(r["free_routing_diag"]) == set(EPS)
        for v in r["gate_values"].values():
            assert v["median_rel_err"] < 0.02 and v["corr"] > 0.999
        assert h.ctl.replay is None                                      # reset after the check


def test_free_routing_kinds_keep_the_flip_exclusion():
    h = host(toy(moe=True))                                             # ling sites: no replay
    ids = ids_batch(1, 12)
    h.check_paths(ids)
    assert h.check_directional_grad(ids)
    assert h.paths["layerwise_grad"]["fd_routing"] == "free"


@pytest.mark.parametrize("transform", [lambda x: -x, lambda x: 2.0 * x], ids=["sign_flip", "scaled_x2"])
def test_frozen_routing_gate_fails_a_replay_defect(monkeypatch, transform):
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    h = host(qn_toy())
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    assert h.check_directional_grad(ids)
    monkeypatch.setattr(Host, "_layerwise", _broken(transform))
    assert not h.check_directional_grad(ids)
    assert h.paths["layerwise_grad"]["fd_routing"].startswith("frozen")


def test_refusal_writes_the_receipt(tmp_path, monkeypatch):
    from research.amoeba.carve import __main__ as D
    monkeypatch.setattr(Host, "_layerwise", _broken(lambda x: -x))
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    with pytest.raises(SystemExit, match="directional-derivative"):
        D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
                "--stages", "inspect,rank", "--out", str(tmp_path / "a")], host_factory=fac)
    r = json.loads((tmp_path / "a" / "carve.json").read_text())
    assert r["inspect"]["refused"] and r["inspect"]["paths"]["layerwise_grad"]["ok"] is False


def test_replay_engages_and_is_checked(monkeypatch):
    """Review 2026-11-15 #1: at a large step free routing flips while replayed routing must equal m's; a
    replay that does nothing is refused."""
    from research.amoeba.carve import units as U
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    h = host(qn_toy())
    ids = ids_batch(1, 12)
    h.check_paths(ids)
    h.check_directional_grad(ids, eps=(0.0625, 0.125, 0.25))
    r = h.paths["layerwise_grad"]
    assert r["replay_route_mismatch"] == 0 and sum(r["free_flips_total"].values()) > 0, r["free_flips_total"]
    monkeypatch.setattr(U, "_replayed", lambda c, l, kind: None)
    assert not h.check_directional_grad(ids, eps=(0.0625, 0.125, 0.25))
    assert any("did not engage" in f for f in h.paths["layerwise_grad"]["fails"])


def test_norm_ratio_bound_catches_scale(monkeypatch):
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    h = host(qn_toy())
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    assert h.check_directional_grad(ids, norm_ratio=(0.8, 1.25))
    nr = [v["norm_ratio"] for v in h.paths["layerwise_grad"]["gate_values"].values()]
    assert all(0.95 < x < 1.05 for x in nr), nr
    monkeypatch.setattr(Host, "_layerwise", _broken(lambda x: 2.0 * x))
    assert not h.check_directional_grad(ids, tol=10.0, min_corr=-1.0, norm_ratio=(0.8, 1.25))
    assert any("norm ratio" in f for f in h.paths["layerwise_grad"]["fails"])


def test_min_pair_corr_requires_a_self_consistent_pair(monkeypatch):
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    h = host(qn_toy())
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    assert h.check_directional_grad(ids, min_pair_corr=0.9)
    r = h.paths["layerwise_grad"]
    assert r["pair_fd_corr"]["/".join(f"{e:g}" for e in r["gate_eps"])] >= 0.9
    rng = np.random.default_rng(3)                     # FD noise: no pair agrees with itself
    real = Host._fd_loss
    monkeypatch.setattr(Host, "_fd_loss", lambda self, *a, **k: real(self, *a, **k) + float(rng.normal(0, 0.05)))
    assert not h.check_directional_grad(ids, min_pair_corr=0.9)
    assert any("self-agreement" in f for f in h.paths["layerwise_grad"]["fails"])


@pytest.mark.parametrize("units", ["expert", "neuron"])
def test_qwen_next_ones_parity_uses_the_stock_tail(units):
    """fnsmoke3c: expert-mode qwen_next sites route and combine through the stock block's helpers, so
    core = ones reproduces the stock forward (exact on the CPU toy)."""
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    h = host(qn_toy(), moe_units=units)
    ids = ids_batch(2, 12)
    h.check_paths(ids)
    r = h.check_ones_parity(ids)
    assert r["max_abs"] == 0.0 and r["top1_agree"] == 1.0 and r["kl_max"] == 0.0



@pytest.mark.parametrize("units", ["expert", "neuron"])
def test_layerwise_replays_the_recorded_routes(monkeypatch, units):
    """Review 2026-11-15 #3: each layer's reference vjp runs with the expert ids recorded on the forward it
    linearises around (so a reference-router near-tie cannot change the experts the gradient goes through)."""
    from research.amoeba.carve import units as U
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    h = host(qn_toy(), moe_units=units)
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    seen = []
    real = U._replayed

    def spy(c, l, kind):
        r = real(c, l, kind)
        if r is not None:
            seen.append(l)
        return r
    monkeypatch.setattr(U, "_replayed", spy)
    pr = h._grad_probe(ids, None, 0)
    moe = sorted(s.layer for s in h.sites if s.is_moe)
    assert sorted(set(seen)) == moe and h.ctl.replay is None
    assert np.isfinite(np.asarray(pr["grad"][moe[0]])).all()
    # a reference router that would pick other experts does not change the gradient
    g0 = [np.asarray(pr["grad"][l]) for l in range(h.L)]
    real_route = U._qwen_next_route

    def shuffled(mx_, block, x):
        inds, w = real_route(mx_, block, x)
        return (inds if not block.training else (inds + 1) % block.num_experts), w   # reference router only
    monkeypatch.setattr(U, "_qwen_next_route", shuffled)
    monkeypatch.setattr(U, "_replayed", real)
    pr2 = h._grad_probe(ids, None, 0)
    for l in range(h.L):
        assert np.allclose(g0[l], np.asarray(pr2["grad"][l]), atol=1e-6), l
