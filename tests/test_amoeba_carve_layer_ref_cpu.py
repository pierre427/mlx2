"""CPU tests of the carve's layer_ref gradient mode: eval-mode (fused) forwards, only the layer being
backpropagated on its reference path, and the per-layer eval-vs-reference check that gates it."""
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

from research.amoeba.carve.carving import kl_rows  # noqa: E402
from research.amoeba.carve.host import Host, reference_mode  # noqa: E402
from test_amoeba_carve_cpu import Layer, Tok, ids_batch, toy  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


class FusedStandIn(nn.Module):
    """A stand-in "fused" op: in eval mode it computes f(x) through a path with no gradient
    (as a Metal kernel without a VJP would), scaled by ``scale``; in training mode the reference
    ops. It keeps a call receipt, as the model's fused paths do."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.fused_calls = 0

    def __call__(self, x):
        ref = x * (1 + 0.1 * mx.tanh(x))
        if self.training:
            return ref
        self.fused_calls += 1
        return mx.stop_gradient(ref) * self.scale


class FusedLayer(Layer):
    def __init__(self, moe, scale=1.0):
        super().__init__(moe)
        self.act = FusedStandIn(scale)

    def __call__(self, x, mask=None, cache=None):
        return self.act(super().__call__(x, mask, cache))


def fused_toy(moe=True, scale=1.0, at=2):
    m = toy(moe=moe)
    old = m.model.layers[at]
    new = FusedLayer(moe=moe and at % 2 == 1, scale=scale)      # as Inner builds layer `at`
    new.update(old.parameters())
    m.model.layers[at] = new
    mx.eval(m.parameters())
    return m


def host(m):
    return Host(model=m, tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)


def mask_grads(h, ids, mode, whole=False, train=True, seed=0):
    """d KL / d core at a random 70% mask, in grad mode ``mode`` (layer-wise or whole-graph)."""
    h.grad_mode = mode
    rng = np.random.default_rng(seed)
    m = [mx.array((rng.random(n) > 0.3).astype(np.float32)) for n in h.widths]
    if train:
        h.train_mode(True)
    try:
        h.ctl.reset()
        pos = list(range(1, ids.shape[1]))
        lc = mx.stop_gradient(h.logits_at(ids, pos))

        def hl(hid):
            return kl_rows(lc, h.head(hid[:, mx.array(pos)])).mean()
        if whole:
            loss, g = h._whole(ids, m, hl, None)
        else:
            loss, g = h._layerwise(ids, {"core": m}, lambda l, sl: h.set_layer_masks(l, sl, None), hl)
            g = g["core"]
    finally:
        h.train_mode(False)
    return float(loss), [np.array(x) for x in g]


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("whole", [False, True])
def test_layer_ref_grads_equal_whole_train_when_paths_agree(moe, whole):
    h = host(toy(moe=moe))
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    la, ga = mask_grads(h, ids, "layer_ref", whole=whole)
    lb, gb = mask_grads(h, ids, "whole_train", whole=whole)
    assert la == lb
    for a, b in zip(ga, gb):
        assert np.array_equal(a, b)
    assert any(np.abs(a).max() > 0 for a in ga)


@pytest.mark.parametrize("moe", [False, True])
def test_layerwise_gate_passes_in_both_modes(moe):
    for mode in ("layer_ref", "whole_train"):
        h = host(toy(moe=moe))
        h.grad_mode = mode
        ids = ids_batch(1, 10)
        h.check_paths(ids)
        assert h.check_layerwise_grad(ids, tol=1e-3), h.paths["layerwise_grad"]
        assert h.paths["layerwise_grad"]["grad_mode"] == mode
        assert h.paths["layerwise_grad"]["rel_l2"] == 0.0


@pytest.mark.parametrize("moe", [False, True])
def test_layer_reference_check_is_zero_on_the_toys(moe):
    h = host(toy(moe=moe))
    ids = ids_batch()
    h.check_paths(ids)
    r = h.check_layer_reference(ids)
    assert r["ok"] and r["failing_layers"] == []
    assert [x["layer"] for x in r["layers"]] == ["0", "1", "2", "3", "head"]
    for x in r["layers"]:
        assert x["max_abs"] == 0.0 and x["rel_l2"] == 0.0 and x["went_eager"] == []
    json.dumps(r)                                            # lands in carve.json as is
    # the check leaves the model as it found it: eval mode, router flags off, no masks set
    assert not any(m.training for _, m in h.model.named_modules())
    assert h.ctl.core is None
    if moe:
        assert h.layers[1].mlp.gate.training_indices is False


def test_fused_stand_in_is_caught_and_the_vjp_uses_the_reference():
    """Same value, no VJP in eval mode: the check sees the mode switch (reads, receipts) with zero
    difference; layer_ref gradients equal whole_train ones; a plain eval-mode backward does not."""
    h = host(fused_toy(moe=False, scale=1.0, at=2))      # dense: an eval backward needs no router flags
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    r = h.check_layer_reference(ids_batch())
    row = {x["layer"]: x for x in r["layers"]}["2"]
    assert r["ok"] and row["rel_l2"] == 0.0
    assert "layers.2.act" in row["reads_training"]
    assert "layers.2.act.fused_calls" in row["went_eager"]
    assert all("act" not in n for x in r["layers"] if x["layer"] != "2" for n in x["reads_training"])
    _, g_ref = mask_grads(h, ids, "layer_ref")
    _, g_old = mask_grads(h, ids, "whole_train")
    for a, b in zip(g_ref, g_old):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    # without the reference switch (eval-mode backward), the stand-in cuts every gradient below it
    h.grad_mode = "whole_train"
    _, g_eval = mask_grads(h, ids, "whole_train", train=False)
    assert all(np.abs(g_eval[l]).max() == 0 for l in range(3)) and np.abs(g_ref[0]).max() > 0
    assert h.check_layerwise_grad(ids, tol=1e-3)


def test_fused_stand_in_with_different_arithmetic_fails_the_check_by_layer():
    h = host(fused_toy(moe=False, scale=1.5, at=2))
    ids = ids_batch()
    h.check_paths(ids)
    r = h.check_layer_reference(ids)
    assert not r["ok"] and r["failing_layers"] == ["2"] and r["worst_layer"] == "2"
    assert r["layers"][2]["rel_l2"] > 0.2


def test_driver_refuses_layer_ref_naming_the_layer(tmp_path):
    from research.amoeba.carve import __main__ as D
    fac = lambda log: Host(model=fused_toy(moe=False, scale=1.5, at=1), tokenizer=Tok(),  # noqa: E731
                           fingerprint="toy:abc", log=log)
    base = ["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2"]
    with pytest.raises(SystemExit, match=r"layer\(s\) \['1'\]"):
        D.main(base + ["--stages", "inspect,rank", "--out", str(tmp_path / "a")], host_factory=fac)
    # no backprop stage: recorded, not refused; the receipt is in inspect.paths
    D.main(base + ["--stages", "inspect", "--out", str(tmp_path / "b")], host_factory=fac)
    rep = json.loads((tmp_path / "b" / "carve.json").read_text())
    assert rep["inspect"]["paths"]["layer_reference"]["failing_layers"] == ["1"]
    # the old mode does not depend on the eval path, so it runs
    D.main(base + ["--stages", "inspect,rank", "--grad-mode", "whole_train", "--out", str(tmp_path / "c")],
           host_factory=fac)


def test_driver_records_layer_reference_in_inspect(tmp_path):
    from research.amoeba.carve import __main__ as D
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect,rank", "--out", str(tmp_path / "r")], host_factory=fac)
    p = json.loads((tmp_path / "r" / "carve.json").read_text())["inspect"]["paths"]
    assert p["layer_reference"]["ok"] and p["layer_reference"]["worst_rel_l2"] == 0.0
    assert p["layerwise_grad"]["grad_mode"] == "layer_ref" and p["layerwise_grad"]["ok"]


def test_reference_mode_restores_each_module():
    m = toy(moe=True)
    m.eval()
    m.model.layers[3].train()                     # a mixed starting state survives the round trip
    gate = m.model.layers[1].mlp.gate
    before = {n: mm.training for n, mm in m.named_modules()}
    with reference_mode([m.model.layers[1]]):
        assert all(mm.training for _, mm in m.model.layers[1].named_modules())
        assert gate.training_indices is True
        assert not m.model.layers[0].training and m.model.layers[3].training
    assert {n: mm.training for n, mm in m.named_modules()} == before
    assert gate.training_indices is False


def _tiny_qwen4():
    try:
        from test_batched_mtp import _tiny_qwen4_model
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"tiny qwen4 fixture unavailable: {e}")
    mx.random.seed(41)
    return _tiny_qwen4_model()


def test_tiny_qwen4_classes_layer_ref():
    """The real Flash-Next classes (GDN, QSA attention, Qwen3-Next MoE, PLE, hyper-connections) at
    toy width on CPU, where the fused paths decline: the check is exact, finds the mode-dependent
    modules structurally, and layer_ref gradients equal whole_train ones. The eval forward keeps
    the default _QSA_SCATTER_CHOSEN (on); reference_mode turns it off for the reference layer only."""
    import mlx2.runtime.models.qwen4_exp as Q
    assert Q._QSA_SCATTER_CHOSEN is True                       # the production default
    h = Host(model=_tiny_qwen4(), tokenizer=Tok(), fingerprint="tq4:abc", log=lambda *_: None)
    assert [s.kind for s in h.sites] == ["qwen_next", "qwen_next"]
    ids = mx.array(np.random.default_rng(0).integers(0, 64, (2, 12)).astype(np.int32))
    p = h.check_paths(ids)
    assert p["replay"], p                     # final norm = hyper_connection_mixer
    r = h.check_layer_reference(ids)
    assert r["ok"] and r["worst_rel_l2"] == 0.0
    rows = {x["layer"]: x for x in r["layers"]}
    reads = {k: set(x["reads_training"]) for k, x in rows.items()}
    assert {"layers.0.linear_attn", "layers.0.mlp.switch_mlp"} <= reads["0"]
    assert {"layers.1.self_attn", "layers.1.mlp.switch_mlp"} <= reads["1"]
    assert "mlx2.runtime.models.qwen4_exp._QSA_SCATTER_CHOSEN" in rows["1"]["flags_flipped"]
    la, ga = mask_grads(h, ids[:1], "layer_ref")
    assert Q._QSA_SCATTER_CHOSEN is True
    lb, gb = mask_grads(h, ids[:1], "whole_train")
    assert la == lb and all(np.array_equal(a, b) for a, b in zip(ga, gb))
    assert h.check_layerwise_grad(ids[:1], tol=1e-3), h.paths["layerwise_grad"]
    assert Q._QSA_SCATTER_CHOSEN is True


def test_qsa_chosen_mask_is_bit_identical_between_branches(monkeypatch):
    """The scatter and the compare branch of QSASelection build the same block mask."""
    import dataclasses
    import mlx2.runtime.models.qwen4_exp as Q
    seen = []
    real = Q.QSASelection.dense_mask

    def spy(self, *a, **k):
        seen.append(self)
        return real(self, *a, **k)
    monkeypatch.setattr(Q.QSASelection, "dense_mask", spy)
    m = _tiny_qwen4()
    for T in (12, 24):
        mx.eval(m(mx.array(np.random.default_rng(T).integers(0, 64, (1, T)).astype(np.int32))))
    explicit = [s for s in seen if s.kind == "explicit"]
    assert explicit
    for s in explicit:
        on = real(dataclasses.replace(s, scatter_chosen=True))
        off = real(dataclasses.replace(s, scatter_chosen=False))
        assert on.dtype == off.dtype and np.array_equal(np.array(on), np.array(off))


def test_reference_globals_restored_when_the_vjp_raises():
    import mlx2.runtime.models.qwen4_exp as Q
    m = _tiny_qwen4()
    m.eval()
    layer = m.language_model.model.layers[1] if hasattr(m, "language_model") else None
    layer = layer or next(mm for _, mm in m.named_modules() if type(mm).__name__ == "DecoderLayer")
    assert Q._QSA_SCATTER_CHOSEN is True
    with pytest.raises(RuntimeError, match="boom"):
        with reference_mode([layer]):
            assert Q._QSA_SCATTER_CHOSEN is False and layer.training
            mx.vjp(lambda x: (_ for _ in ()).throw(RuntimeError("boom")), [mx.ones((2,))], [mx.ones((2,))])
    assert Q._QSA_SCATTER_CHOSEN is True and not layer.training
    # a module tree with no qwen4 class leaves the global alone
    with reference_mode([toy().model.layers[0]]):
        assert Q._QSA_SCATTER_CHOSEN is True


class BranchLayer(Layer):
    """A layer whose residual dominates its output and whose eval branch is 25% wrong."""

    def __call__(self, x, mask=None, cache=None):
        b = 0.02 * (super().__call__(x, mask, cache) - x)
        return x + (b if self.training else 1.25 * b)


def test_gate_is_on_the_update_not_the_residual_dominated_output():
    m = toy(moe=False)
    new = BranchLayer(moe=False)
    new.update(m.model.layers[2].parameters())
    m.model.layers[2] = new
    mx.eval(m.parameters())
    h = host(m)
    ids = ids_batch()
    h.check_paths(ids)
    r = h.check_layer_reference(ids)
    row = r["layers"][2]
    assert row["gate"] == "update" and row["rel_l2_output"] <= 2e-2      # the old output gate passed it
    assert abs(row["rel_l2"] - 0.25) < 1e-3
    assert not r["ok"] and r["failing_layers"] == ["2"]
    assert r["layers"][-1]["layer"] == "head" and r["layers"][-1]["gate"] == "output"


# ---- a trunk-opened scope, as qwen4's _moe_nax.forward_scope: the fused op runs only inside it

import contextvars  # noqa: E402
import types  # noqa: E402

_PHASE = contextvars.ContextVar("toy_prefill_phase", default=False)
FAKE_STATS = {"fused": 0}
FAKE_MOD = types.ModuleType("toy_route_stats")
FAKE_MOD.status = lambda: {"calls": dict(FAKE_STATS)}


class ScopedFused(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.fused_calls = 0

    def __call__(self, x):
        ref = x + 0.1 * mx.tanh(x)
        if self.training or not _PHASE.get():
            return ref
        self.fused_calls += 1
        FAKE_STATS["fused"] += 1
        return x + mx.stop_gradient(0.1 * mx.tanh(x)) * self.scale


class ScopedLayer(Layer):
    def __init__(self, scale):
        super().__init__(False)
        self.act = ScopedFused(scale)

    def __call__(self, x, mask=None, cache=None):
        return self.act(super().__call__(x, mask, cache))


def scoped_toy(scale):
    m = toy(moe=False)
    new = ScopedLayer(scale)
    new.update(m.model.layers[1].parameters())
    m.model.layers[1] = new
    inner_call = type(m.model).__call__

    class ScopedInner(type(m.model)):
        def __call__(self, ids):
            tok = _PHASE.set(True)
            try:
                return inner_call(self, ids)
            finally:
                _PHASE.reset(tok)
    m.model.__class__ = ScopedInner
    mx.eval(m.parameters())
    return m


def test_eval_arm_and_residuals_run_inside_the_trunk_scope(monkeypatch):
    import research.amoeba.carve.host as Hm
    monkeypatch.setitem(sys.modules, "toy_route_stats", FAKE_MOD)
    monkeypatch.setattr(Hm, "STATUS_PROBES", (("toy_route_stats", "status"),))
    h = host(scoped_toy(scale=1.5))
    ids = ids_batch()
    h.check_paths(ids)
    r = h.check_layer_reference(ids)
    row = r["layers"][1]
    # a bare layer(...) replay would be eager in both arms and pass; the trunk forward is not
    assert not r["ok"] and r["failing_layers"] == ["1"]
    assert "layers.1.act.fused_calls" in row["went_eager"]
    assert r["global_receipts"]["eval"].get("toy_route_stats.status.calls.fused", 0) >= 1
    assert "toy_route_stats.status.calls.fused" not in r["global_receipts"]["reference"]
    # the layer-wise residuals are the trunk's (fused) ones: its loss is the eval forward's loss
    h.grad_mode = "layer_ref"
    h.train_mode(True)
    ones = [mx.ones((n,)) for n in h.widths]
    pos = list(range(1, 10))
    ids1 = ids[:1, :10]
    lc = mx.stop_gradient(h.logits_at(ids1, pos)) * 0.5

    def hl(hid):
        return kl_rows(lc, h.head(hid[:, mx.array(pos)])).mean()
    loss, _ = h._layerwise(ids1, {"core": ones}, lambda l, sl: h.set_layer_masks(l, sl, None), hl)
    h.ctl.core = ones
    direct = hl(h.hidden(ids1))
    h.ctl.reset()
    h.train_mode(False)
    assert float(loss) == float(direct)


def test_helper_module_globals_are_switched_by_the_model_module(monkeypatch):
    import mlx2.runtime.models.qwen4_qsa_scores as S
    import mlx2.runtime.models.qwen4_fused_group_norm as G
    monkeypatch.setattr(S, "_ENABLED", True)
    monkeypatch.setattr(G, "_ENABLE", True)
    m = _tiny_qwen4()
    m.eval()
    layer = next(mm for _, mm in m.named_modules() if type(mm).__name__ == "DecoderLayer")
    with reference_mode([layer]):
        assert S._ENABLED is False and G._ENABLE is False
    assert S._ENABLED is True and G._ENABLE is True
    with reference_mode([toy().model.layers[0]]):                # no qwen4 class: untouched
        assert S._ENABLED is True


def test_resume_identity_includes_grad_mode(tmp_path):
    from research.amoeba.carve import __main__ as D
    fac = lambda log: Host(model=toy(moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    base = ["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--stages", "inspect",
            "--out", str(tmp_path / "r")]
    D.main(base, host_factory=fac)
    ident = json.loads((tmp_path / "r" / "carve.json").read_text())["identity"]
    assert ident["grad_mode"] == "layer_ref" and ident["layer_ref_tol"] == 2e-2
    with pytest.raises(SystemExit, match="grad_mode"):
        D.main(base + ["--resume", "--grad-mode", "whole_train"], host_factory=fac)
    with pytest.raises(SystemExit, match="layer_ref_tol"):
        D.main(base + ["--resume", "--layer-ref-tol", "0.05"], host_factory=fac)
    D.main(base + ["--resume"], host_factory=fac)


def test_tiny_qwen4_whole_graph_with_production_dispatch_flags(monkeypatch):
    """FN smoke 2026-11-15: with the production eager / PLE-early dispatch globals on, the trunk forward
    calls mx.async_eval mid-graph and mx.value_and_grad refuses it. The reference globals switch them off
    for the whole-graph (and any reference) trace and restore them; values do not change (dispatch only)."""
    import mlx2.runtime.models.qwen4_exp as Q
    monkeypatch.setattr(Q, "_EAGER_DISPATCH", True)
    monkeypatch.setattr(Q, "_EAGER_DISPATCH_STRIDE", 1)
    monkeypatch.setattr(Q, "_PLE_EARLY_DISPATCH", True)
    h = Host(model=_tiny_qwen4(), tokenizer=Tok(), fingerprint="tq4:abc", log=lambda *_: None)
    ids = mx.array(np.random.default_rng(0).integers(0, 64, (1, 12)).astype(np.int32))
    h.check_paths(ids)
    assert h.check_layerwise_grad(ids, tol=1e-3), h.paths["layerwise_grad"]
    assert Q._EAGER_DISPATCH is True and Q._PLE_EARLY_DISPATCH is True
