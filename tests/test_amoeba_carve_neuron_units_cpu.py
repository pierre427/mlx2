"""CPU tests of the carve's neuron-within-expert MoE units (--moe-units neuron): a unit is (routed
expert e, intermediate neuron f), flat id e*F + f. Toy Qwen3-Next MoE blocks and the tiny qwen4
(Flash-Next classes) only."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from research.amoeba.carve import units as U  # noqa: E402
from research.amoeba.carve.carving import kl_rows, random_same_counts, topk_mask  # noqa: E402
from research.amoeba.carve.host import Host  # noqa: E402
from test_amoeba_carve_cpu import D, V, Tok, toy  # noqa: E402
from test_amoeba_carve_layer_ref_cpu import _tiny_qwen4, mask_grads  # noqa: E402

E, K, FF = 6, 2, 16          # toy Qwen3-Next: experts, top-k, moe_intermediate_size


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _qn_block(switch="split", folded=False):
    import mlx2.runtime.models.qwen3_next as Q
    args = SimpleNamespace(hidden_size=D, moe_intermediate_size=FF, shared_expert_intermediate_size=FF,
                           num_experts=E, num_experts_per_tok=K, norm_topk_prob=True)
    old = Q._MOE_FUSED_GATE_UP, Q._MOE_SHARED_IN_GATHER
    Q._MOE_FUSED_GATE_UP, Q._MOE_SHARED_IN_GATHER = switch == "fused", folded
    try:
        b = Q.Qwen3NextSparseMoeBlock(args)
    finally:
        Q._MOE_FUSED_GATE_UP, Q._MOE_SHARED_IN_GATHER = old
    assert b.shared_folded == folded
    return b


def qn_toy(seed=0, switch="split", quantize=False, folded=False):
    """The carve CPU toy (dense layers 0, 2) with real Qwen3NextSparseMoeBlocks at layers 1, 3."""
    m = toy(seed)
    mx.random.seed(seed + 7)
    for i in (1, 3):
        m.model.layers[i].mlp = _qn_block(switch, folded)
    if quantize:
        nn.quantize(m, group_size=32, bits=8, class_predicate=lambda p, mod: hasattr(mod, "to_quantized")
                    and "switch_mlp" in p and mod.weight.shape[-1] % 32 == 0)
    mx.eval(m.parameters())
    return m


def host(m, units="neuron"):
    return Host(model=m, tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None, moe_units=units)


def ids_batch(n=2, T=12, seed=1, vocab=V):
    return mx.array(np.random.default_rng(seed).integers(0, vocab, (n, T)).astype(np.int32))


def logits(h, core, ids):
    h.ctl.reset()
    h.ctl.core = core
    out = h.logits_at(ids)
    h.ctl.reset()
    mx.eval(out)
    return np.array(out)


def laguna_toy(seed=0):
    from test_amoeba_carve_cpu import LagunaLikeMoe
    m = toy(seed)
    mx.random.seed(seed + 7)
    for i in (1, 3):
        m.model.layers[i].mlp = LagunaLikeMoe()
    mx.eval(m.parameters())
    return m


TOYS = [("qn_split", lambda: qn_toy(), V), ("qn_fused", lambda: qn_toy(switch="fused"), V),
        ("qn_q8", lambda: qn_toy(quantize=True), V),
        ("qn_folded_split", lambda: qn_toy(folded=True), V),
        ("qn_folded_fused", lambda: qn_toy(switch="fused", folded=True), V),
        ("ling", lambda: toy(), V), ("laguna", laguna_toy, V), ("tiny_qwen4", _tiny_qwen4, 64)]
# (batch, T): 1 x 5 x top-2 = 10 assignments < 20, the unsorted gather; 2 x 12 x 2 = 48, the sorted gather
SHAPES = [(1, 5), (2, 12)]


@pytest.mark.parametrize("name,make,vocab", TOYS, ids=[t[0] for t in TOYS])
@pytest.mark.parametrize("B,T", SHAPES, ids=["unsorted", "sorted"])
def test_open_neuron_mask_equals_stock_and_expert_mode(name, make, vocab, B, T, monkeypatch):
    from mlx2.runtime.models import switch_layers as sl
    sorts = []
    real_sort = sl._gather_sort
    monkeypatch.setattr(sl, "_gather_sort", lambda *a: sorts.append(1) or real_sort(*a))
    stock = make()
    ids = ids_batch(B, T, vocab=vocab)
    ref = np.array(stock(ids).astype(mx.float32))
    hn = host(make())
    ns = U.neuron_sites(hn.sites)
    assert ns and all(s.n == s.E * s.F for s in ns)
    ln_open = logits(hn, [mx.ones((n,)) for n in hn.widths], ids)
    he = host(make(), "expert")
    le_open = logits(he, [mx.ones((n,)) for n in he.widths], ids)
    assert np.array_equal(ln_open, le_open)
    if name == "laguna":
        # the carve's laguna rows fold the routed scale into the weights before the sum (stock: after),
        # in expert and neuron mode alike: equal to each other bit for bit, to stock to rounding
        np.testing.assert_allclose(ln_open, ref, rtol=0, atol=1e-5)
    else:
        assert np.array_equal(ln_open, ref)
    # all neurons of a set of experts kept == expert mode keeping those experts (the drop rule)
    rng = np.random.default_rng(3)
    ce, cn = [], []
    for sn, se in zip(hn.sites, he.sites):
        if sn.is_moe:
            keep = (rng.random(se.n) > 0.5).astype(np.float32)
            keep[0] = 0.0
            ce.append(mx.array(keep))
            cn.append(mx.array(U.expert_to_neuron(keep, sn.F)))
        else:
            ce.append(mx.ones((se.n,)))
            cn.append(mx.ones((sn.n,)))
    le, ln = logits(he, ce, ids), logits(hn, cn, ids)
    assert np.array_equal(le, ln)
    assert not np.array_equal(le, ref)
    assert bool(sorts) == (B * T * 2 >= 20)          # the gather order the case claims was the one run


def _zero_up_rows(sw, drops, F):
    """Zero row f of expert e's up projection (the [gate|up] tensor's up half at F + f)."""
    if hasattr(sw, "gate_up_proj"):
        p, off = sw.gate_up_proj, F
    else:
        p, off = sw.up_proj, 0
    w = np.array(p.weight)
    for e, f in drops:
        w[e, off + f] = 0
    p.weight = mx.array(w)


@pytest.mark.parametrize("name,make,vocab", [t for t in TOYS if t[0] != "qn_q8"],
                         ids=[t[0] for t in TOYS if t[0] != "qn_q8"])
@pytest.mark.parametrize("B,T", SHAPES, ids=["unsorted", "sorted"])
def test_masking_equals_zeroed_up_rows_every_kind(name, make, vocab, B, T):
    m = make()
    h = host(m)
    ids = ids_batch(B, T, vocab=vocab)
    rng = np.random.default_rng(11)
    core = [mx.ones((n,)) for n in h.widths]
    plan = {}
    for s in U.neuron_sites(h.sites):
        drops = {(int(e), int(f)) for e, f in zip(rng.integers(0, s.E, 12), rng.integers(0, s.F, 12))}
        drops |= {(0, f) for f in range(s.F)}                    # one expert fully dropped
        c = np.ones(s.n, np.float32)
        for e, f in drops:
            c[e * s.F + f] = 0
        core[s.layer] = mx.array(c)
        plan[s.layer] = (s, sorted(drops))
    masked = logits(h, core, ids)
    for s, drops in plan.values():
        _zero_up_rows(s.module.switch_glu if s.kind == "gemma4_experts" else s.module.switch_mlp, drops, s.F)
    # stock forward of the zeroed model; laguna: the open carve path (its routed scale is folded first)
    assert np.array_equal(logits(h, [mx.ones((n,)) for n in h.widths] if name == "laguna" else None, ids), masked)


def test_neuron_mask_matches_zeroed_up_row():
    """Dropping neuron (e, f) is exactly zeroing row f of expert e's up projection."""
    m = qn_toy()
    h = host(m)
    ids = ids_batch()
    core = [mx.ones((n,)) for n in h.widths]
    drop = [(0, 3), (2, 0), (5, 15)]
    c1 = np.ones(E * FF, np.float32)
    for e, f in drop:
        c1[e * FF + f] = 0
    core[1] = mx.array(c1)
    masked = logits(h, core, ids)
    w = np.array(m.model.layers[1].mlp.switch_mlp.up_proj.weight)
    for e, f in drop:
        w[e, f] = 0
    m.model.layers[1].mlp.switch_mlp.up_proj.weight = mx.array(w)
    assert np.array_equal(logits(h, None, ids), masked)


def test_sink_rule_and_gated_box_never_build_tokens_x_E_x_F(monkeypatch):
    """Position 0 exact in sink layers, and a gated box, through per-slot mask rows only."""
    h = host(qn_toy())
    ids = ids_batch()
    zero = [mx.zeros((n,)) if s.is_moe else mx.ones((n,)) for n, s in zip(h.widths, h.sites)]
    full = logits(h, None, ids)
    h.ctl.reset()
    h.ctl.core, h.ctl.sink_layers = zero, h.L
    sink = np.array(h.logits_at(ids))
    h.ctl.reset()
    assert np.array_equal(sink[:, 0], full[:, 0]) and not np.allclose(sink[:, 1:], full[:, 1:])
    # a gated box over every neuron with gate ~1 restores the stock output
    big = mx.zeros((h.L, D))
    bmask = [mx.ones((n,)) if s.is_moe else mx.zeros((n,)) for n, s in zip(h.widths, h.sites)]
    box = {"name": "b", "mask": bmask, "gate": (big, mx.full((h.L,), 30.0)), "hard": True}
    seen = []
    real = U._slot_rows
    monkeypatch.setattr(U, "_slot_rows", lambda v, i, e, f: seen.append(i.shape) or real(v, i, e, f))
    h.ctl.reset()
    h.ctl.core, h.ctl.boxes = zero, [box]
    boxed = np.array(h.logits_at(ids))
    h.ctl.reset()
    assert np.array_equal(boxed, full)
    assert seen and all(s[-1] == K for s in seen)              # gathered per routed slot


@pytest.mark.parametrize("make,vocab", [(qn_toy, V), (_tiny_qwen4, 64)], ids=["qn", "tiny_qwen4"])
def test_layerwise_and_layer_ref_grads_in_neuron_mode(make, vocab):
    h = host(make())
    ids = ids_batch(1, 10, vocab=vocab)
    h.check_paths(ids)
    r = h.check_layer_reference(ids_batch(2, 12, vocab=vocab))
    assert r["ok"] and r["worst_rel_l2"] == 0.0
    assert h.check_layerwise_grad(ids, tol=1e-3), h.paths["layerwise_grad"]
    la, ga = mask_grads(h, ids, "layer_ref")
    lb, gb = mask_grads(h, ids, "whole_train")
    assert la == lb and all(np.array_equal(a, b) for a, b in zip(ga, gb))
    gm = [g for g, s in zip(ga, h.sites) if s.is_moe]
    assert all(g.shape == (s.E * s.F,) for g, s in zip(gm, U.neuron_sites(h.sites)))
    assert all(np.abs(g).max() > 0 for g in gm)


def test_neuron_grad_matches_finite_difference():
    """d loss / d m[e, f] at the hard mask is the directional derivative of the masked forward."""
    h = host(qn_toy())
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    pos = list(range(1, 10))
    lc = mx.stop_gradient(h.logits_at(ids, pos))

    def hl(hid):
        return kl_rows(lc, h.head(hid[:, mx.array(pos)])).mean()
    core = [mx.ones((n,)) * 0.9 for n in h.widths]
    h.train_mode(True)
    _, g = h.value_and_grad(ids, core, hl, layerwise=False)
    h.train_mode(False)
    j = int(np.argmax(np.abs(np.array(g[3]))))
    eps = 0.05

    def loss_at(delta):
        c = [x for x in core]
        v = np.array(core[3])
        v[j] += delta
        c[3] = mx.array(v)
        h.ctl.reset()
        h.ctl.core, h.ctl.sink_layers = c, h.L
        out = float(hl(h.hidden(ids)))
        h.ctl.reset()
        return out
    fd = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
    assert abs(fd - float(g[3][j])) <= 2e-2 * abs(fd) + 1e-7, (fd, float(g[3][j]))


def test_mean_is_conditional_per_routed_neuron():
    from research.amoeba.carve import evals as Ev
    h = host(qn_toy())
    seqs = ids_batch(3, 10).tolist()
    mean = Ev.mean_acts(h, seqs)
    assert mean[1].shape == (E * FF,) and mean[0].shape == (h.widths[0],)
    # reference: mean of h over routed slots, per expert, from the stats hook
    sums, cnt = np.zeros((E, FF)), np.zeros(E)

    def st(l, kind, p):
        if l == 1:
            inds = np.array(p["inds"][:, 1:]).reshape(-1)
            hh = np.array(p["h"][:, 1:].astype(mx.float32)).reshape(-1, FF)
            for i, e in enumerate(inds):
                sums[e] += hh[i]
                cnt[e] += 1
    h.ctl.reset()
    h.ctl.stats = st
    mx.eval(h.hidden(mx.array(seqs)))
    h.ctl.reset()
    want = (sums / np.maximum(cnt, 1)[:, None]).reshape(-1)
    np.testing.assert_allclose(np.array(mean[1]), want, rtol=1e-5, atol=1e-6)


def test_carving_helpers_keep_counts_at_neuron_width():
    rng = np.random.default_rng(0)
    L_, N_ = 4, E * FF
    score = mx.array(rng.normal(size=(L_, N_)).astype(np.float32))
    M = topk_mask(score, 0.1)
    assert M.shape == (L_, N_) and int(M.sum()) == int(round(0.1 * L_ * N_))
    R = random_same_counts(M, 0)
    assert np.array_equal(np.array(R).sum(1), np.array(M).sum(1))


def test_driver_neuron_mode_rank_bisect_fisher_export(tmp_path):
    from research.amoeba.carve import __main__ as Dm
    m = qn_toy()
    src = tmp_path / "src"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), dict(nn.utils.tree_flatten(m.parameters())),
                        metadata={"format": "mlx"})
    fac = lambda log: Host(model=m, tokenizer=Tok(), fingerprint="toy:abc", log=log, moe_units="neuron")  # noqa: E731
    run = tmp_path / "r"
    argv = ["--model", str(src), "--smoke", "--cpu", "--seq-len", "16", "--fisher-seqs", "4", "--targets", "0.5",
            "--bisect-iters", "2", "--moe-units", "neuron", "--size-targets", "0.05", "--export-dir", str(tmp_path / "x"),
            "--stages", "inspect,mean,fisher,rank,bisect,sink,export", "--out", str(run)]
    Dm.main(argv, host_factory=fac)
    rep = json.loads((run / "carve.json").read_text())
    W = max(rep["shape"]["widths"])
    assert rep["identity"]["moe_units"] == "neuron" and W == E * FF
    assert rep["shape"]["moe_layout"] == {"1": {"E": E, "F": FF}, "3": {"E": E, "F": FF}}
    assert rep["neuron_units"]["memory"]["mask_bytes"] == 4 * sum(rep["shape"]["widths"])
    assert rep["inspect"]["ones_parity"]["max_abs"] == 0.0
    for abl in ("zero", "mean"):
        Z = np.load(run / f"ranking_{abl}.npz")["Z"]
        assert Z.shape == (4, W)
    assert np.load(run / "mean.npz")["mean"].shape == (4, W)
    # a size cut on a ranking without ties (fisher's; the 3-step smoke rank is still all z0) keeps the
    # requested count of all units, at most each site's width
    tier = np.load(run / "tier_fisher_zero_size0.05.npz")["core"]
    assert int(tier.sum()) == int(round(0.05 * sum(rep["shape"]["widths"])))
    assert all(tier[l, n:].sum() == 0 for l, n in enumerate(rep["shape"]["widths"]))
    # the export zeroes up rows of dropped (e, f) and reproduces the research forward
    ex = rep["export"]
    ref = np.load(Path(ex["dir"]) / "carve_verify_reference.npz")
    m2 = qn_toy()
    m2.load_weights(list(mx.load(str(Path(ex["dir"]) / "model.safetensors")).items()))
    got = np.array(m2(mx.array(ref["ids"])).astype(mx.float32))
    assert np.abs(got - ref["logits"]).max() < 1e-4
    assert "MoE neurons" in ex["zeroing"]
    # resume refuses another unit granularity
    fac_e = lambda log: Host(model=qn_toy(), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    with pytest.raises(SystemExit, match="moe_units"):
        Dm.main(argv[:-4] + ["--moe-units", "expert", "--stages", "inspect", "--resume", "--out", str(run)],
                host_factory=fac_e)


def test_driver_refuses_expert_mask_on_neuron_host_and_atlas(tmp_path):
    from research.amoeba.carve import __main__ as Dm
    fac = lambda log: host(qn_toy())  # noqa: E731
    np.savez(tmp_path / "fx.npz", core=np.ones((4, 48), np.float32))        # expert-mode [L, max(H, E)] layout
    base = ["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--moe-units", "neuron"]
    with pytest.raises(SystemExit, match="expert_to_neuron"):
        Dm.main(base + ["--stages", "mean", "--fixed", f"{tmp_path}/fx.npz:core", "--out", str(tmp_path / "a")],
                host_factory=fac)
    with pytest.raises(SystemExit, match="atlas"):
        Dm.main(base + ["--stages", "mean,atlas", "--out", str(tmp_path / "b")], host_factory=fac)
    with pytest.raises(SystemExit, match="MoE units"):
        Dm.main(base[:-2] + ["--stages", "mean", "--out", str(tmp_path / "c")], host_factory=fac)


class OddSwitch(nn.Module):
    def __init__(self):
        super().__init__()
        from mlx2.runtime.models.switch_layers import SwitchLinear
        self.down_proj = SwitchLinear(FF, D, E, bias=False)

    def __call__(self, x, inds):
        return mx.zeros(inds.shape + (D,))


def test_refusals():
    from mlx2.runtime.models.switch_layers import SwitchGLU
    m = qn_toy()
    m.model.layers[1].mlp.switch_mlp = SwitchGLU(D, FF, E, bias=True)       # biased down_proj
    with pytest.raises(SystemExit, match="biased down_proj"):
        host(m)
    host(m, "expert")                                                       # expert units still install
    m = qn_toy()
    m.model.layers[3].mlp.switch_mlp = OddSwitch()
    with pytest.raises(SystemExit, match="supports only"):
        host(m)
    with pytest.raises(SystemExit, match="moe_units"):
        host(qn_toy(), "row")
    h = host(qn_toy())
    from research.amoeba.carve.host import TopicsBackend
    with pytest.raises(SystemExit, match="neuron"):
        TopicsBackend(None, host=h).stats(ids_batch().tolist())


def test_fn_scale_memory_budget():
    """At Flash-Next scale (48 MoE layers x 512 x 640, float32) the carve's per-unit state, not the
    model, is what neuron units add: report-level numbers (no allocation)."""
    units = 48 * 512 * 640
    assert units == 15_728_640
    mib = units * 4 / 2**20
    assert mib == 60.0                                    # mask, grad, Z: 60 MiB each; Adam m + v: 120 MiB


@pytest.mark.parametrize("switch,quantize", [("split", False), ("fused", False), ("split", True)],
                         ids=["up_proj", "gate_up_proj", "up_proj_q8"])
def test_export_zeroes_up_rows_and_reproduces_neuron_mask(tmp_path, switch, quantize):
    from research.amoeba.carve import export as X
    m = qn_toy(switch=switch, quantize=quantize)
    src = tmp_path / "src"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), dict(nn.utils.tree_flatten(m.parameters())),
                        metadata={"format": "mlx"})
    (src / "config.json").write_text(json.dumps({"model_type": "toy"}))
    h = host(m)
    rng = np.random.default_rng(5)
    core = [(rng.random(n) > 0.4).astype(np.float32) for n in h.widths]
    core[1][: 2 * FF] = 0                                  # experts 0 and 1 fully dropped (they keep their slot)
    rep = X.export_dense_core(h, src, core, tmp_path / "dst", log=lambda *_: None, verify_ids=ids_batch().tolist())
    ref = np.load(tmp_path / "dst" / "carve_verify_reference.npz")
    m2 = qn_toy(switch=switch, quantize=quantize)
    m2.load_weights(list(mx.load(str(tmp_path / "dst" / "model.safetensors")).items()))
    got = np.array(m2(mx.array(ref["ids"])).astype(mx.float32))
    assert np.abs(got - ref["logits"]).max() < 1e-4
    assert rep["edits"] == (4 if quantize else 2) + 2      # MoE: weight (or scales + biases) x 2 layers; dense: 2
    # the dropped neurons write nothing in the export: masked research forward == plain exported forward
    assert np.abs(np.array(m(mx.array(ref["ids"])).astype(mx.float32)) - got).max() > 1e-3


def test_switch_masked_over_the_tail_pad_threshold():
    """> 32,768 assignments with n % 64 != 0: _gather_sort pads the sorted tail; the masked switch pads
    its mask rows to match and slices the pad off. Open == stock switch, masking == zeroed up rows."""
    from mlx2.runtime.models import switch_layers as sl
    mx.random.seed(3)
    sw = sl.SwitchGLU(D, FF, E, bias=False)
    mx.eval(sw.parameters())
    Tn = 16_400                                          # 32,800 assignments, 32,800 % 64 = 32
    x = mx.random.normal((1, Tn, D))
    inds = mx.array(np.random.default_rng(0).integers(0, E, (1, Tn, K)).astype(np.int32))
    xs, _, _ = sl._gather_sort(mx.expand_dims(x, (-2, -3)), inds)
    assert xs.shape[0] > inds.size                       # the pad is engaged
    ref = sw(x, inds)
    open_, hid = U._switch_masked(mx, sw, x, inds, mx.ones(inds.shape + (FF,)), want_hidden=True)
    assert np.array_equal(np.array(open_), np.array(ref)) and hid.shape == inds.shape + (FF,)
    assert np.array_equal(np.array(U._switch_masked(mx, sw, x, inds)[0]), np.array(ref))
    drops = [(0, 1), (3, 7), (5, 15)]
    t = np.ones((E, FF), np.float32)
    for e, f in drops:
        t[e, f] = 0
    keep = U._slot_rows(mx.array(t.reshape(-1)), inds, E, FF)
    masked, _ = U._switch_masked(mx, sw, x, inds, keep)
    _zero_up_rows(sw, drops, FF)
    assert np.array_equal(np.array(masked), np.array(sw(x, inds)))


def test_mask_width_must_match_the_site():
    h = host(qn_toy())
    he = host(qn_toy(), "expert")
    ids = ids_batch()
    bad_n = [mx.ones((n,)) for n in h.widths]
    bad_n[1] = mx.ones((E,))                             # an expert row on a neuron site
    with pytest.raises(ValueError, match="neuron units"):
        logits(h, bad_n, ids)
    bad_e = [mx.ones((n,)) for n in he.widths]
    bad_e[1] = mx.ones((E * FF,))                        # a neuron row on an expert site
    with pytest.raises(ValueError, match="expert units"):
        logits(he, bad_e, ids)


def test_export_refuses_a_transposed_up_tensor(tmp_path):
    from research.amoeba.carve import export as X
    m = qn_toy()
    flat = dict(nn.utils.tree_flatten(m.parameters()))
    k = "model.layers.1.mlp.switch_mlp.up_proj.weight"
    flat[k] = flat[k].transpose(0, 2, 1)                 # [E, D, F]: rows are not the up rows
    src = tmp_path / "src"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), flat, metadata={"format": "mlx"})
    h = host(m)
    core = [np.ones(n, np.float32) for n in h.widths]
    core[1][5] = 0
    with pytest.raises(SystemExit, match="transposed"):
        X.export_dense_core(h, src, core, tmp_path / "dst", log=lambda *_: None)


def test_rank_smoke_moves_moe_rows_from_a_nontrivial_init(tmp_path):
    """At m = 1 the KL gradient is zero, so a z0 = 3 smoke only tests plumbing. From a mixed-sign
    init the mask is not all ones, the MoE neuron gradients are not zero, and with lam = 0 the
    straight-through Adam steps move MoE rows of Z in both directions."""
    from research.amoeba.carve import __main__ as Dm
    fac = lambda log: host(qn_toy())  # noqa: E731
    h = host(qn_toy())
    W = max(h.widths)
    rng = np.random.default_rng(4)
    Z0 = rng.normal(size=(h.L, W)).astype(np.float32)
    np.savez(tmp_path / "z0.npz", Z=Z0)
    run = tmp_path / "r"
    Dm.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--moe-units", "neuron",
             "--lam", "0", "--lr", "0.05", "--stages", "inspect,rank", "--init-from", str(tmp_path / "z0.npz"),
             "--out", str(run)], host_factory=fac)
    Z = np.load(run / "ranking_zero.npz")["Z"]
    start = 3.0 * np.sign(Z0) * np.minimum(1.0, np.abs(Z0) / np.median(np.abs(Z0)))    # _init_z's rescale
    for s in U.neuron_sites(h.sites):
        d = Z[s.layer, :s.n] - start[s.layer, :s.n]
        assert (np.abs(d) > 1e-4).mean() > 0.5, (s.layer, (np.abs(d) > 1e-4).mean())
        assert (d > 1e-4).any() and (d < -1e-4).any()


def test_expanded_expert_mask_equals_the_expert_unit_model_exactly(tmp_path):
    """Coordinator 2026-11-16: an expert-unit fixed tier (T2) expanded for a neuron-unit carve must give exactly the
    expert-unit model's output."""
    from research.amoeba.carve import __main__ as Dm
    he, hn = host(qn_toy(), units="expert"), host(qn_toy(), units="neuron")
    ids = mx.array(np.random.default_rng(0).integers(0, 60, size=(2, 12)))
    he.check_paths(ids)
    hn.check_paths(ids)
    rng = np.random.default_rng(1)
    a = np.zeros((he.L, max(he.widths)), np.float32)
    for l, n in enumerate(he.widths):
        a[l, :n] = (rng.random(n) > 0.4)
    np.savez(tmp_path / "t2.npz", core=a)
    me = Dm.load_mask(f"{tmp_path}/t2.npz:core", he)
    mn = Dm.load_mask(f"{tmp_path}/t2.npz:core", hn, expert_units=True)
    full = np.array(he.logits_at(ids))
    he.ctl.reset(); he.ctl.core = me; he.ctl.sink_layers = he.L
    hn.ctl.reset(); hn.ctl.core = mn; hn.ctl.sink_layers = hn.L
    le, ln = np.array(he.logits_at(ids)), np.array(hn.logits_at(ids))
    he.ctl.reset(); hn.ctl.reset()
    assert np.array_equal(le, ln)
    assert not np.allclose(le, full)                                   # the mask really changes the output
    with pytest.raises(SystemExit, match="expert_to_neuron"):        # without the flag: refused as before
        Dm.load_mask(f"{tmp_path}/t2.npz:core", hn)
