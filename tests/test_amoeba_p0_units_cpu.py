"""CPU tests on toy hosts (real Qwen3-Next MoE blocks, no FN) of the P0 model-path additions
(docs/research/2026-11-20-amoeba-next-experiments.md 1.3, 4.3, 7.1-7.3): per-document CE rows (bitwise the same pooled
value), mean ablation of closed MoE neurons (= the per-expert bias of the closed set's mean through W_down), the
TEAL-style per-token threshold, the one-pass MoE statistics and the streaming attribution sums."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from research.amoeba.carve import evals as Ev  # noqa: E402
from research.amoeba.carve import fisher as FI  # noqa: E402
from research.amoeba.carve import moestats as MS  # noqa: E402
from test_amoeba_carve_neuron_units_cpu import E, FF, host, ids_batch, logits, qn_toy  # noqa: E402

SHAPES = [(1, 5), (2, 12)]          # unsorted and sorted gathers


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _core(h, rng, p=0.5):
    """A random 0/1 mask over the neuron MoE layers (dense layers whole)."""
    out = []
    for s in h.sites:
        if s.is_moe and s.units == "neuron":
            out.append(mx.array((rng.random(s.n) > p).astype(np.float32)))
        else:
            out.append(mx.ones((h.widths[s.layer],)))
    return out


def _logits_with(h, core, ids, mean=None, thresh=None, open_acc=None, sink=0):
    h.ctl.reset()
    h.ctl.core, h.ctl.sink_layers = core, sink
    h.ctl.moe_mean, h.ctl.moe_thresh, h.ctl.moe_open = mean, thresh, open_acc
    out = h.logits_at(ids)
    h.ctl.reset()
    mx.eval(out)
    return np.array(out)


@pytest.mark.parametrize("B,T", SHAPES, ids=["unsorted", "sorted"])
@pytest.mark.parametrize("quant", [False, True], ids=["fp", "q8"])
def test_mean_fill_zero_is_drop_rule_and_open_is_exact(B, T, quant):
    h = host(qn_toy(quantize=quant))
    ids = ids_batch(B, T)
    rng = np.random.default_rng(0)
    core = _core(h, rng)
    zero = [mx.zeros((s.n,)) if (s.is_moe and s.units == "neuron") else None for s in h.sites]
    np.testing.assert_array_equal(_logits_with(h, core, ids, mean=zero), logits(h, core, ids))
    ones = [mx.ones((n,)) for n in h.widths]
    mu = [mx.array(rng.normal(size=s.n).astype(np.float32)) if (s.is_moe and s.units == "neuron") else None
          for s in h.sites]
    np.testing.assert_array_equal(_logits_with(h, ones, ids, mean=mu), logits(h, ones, ids))


@pytest.mark.parametrize("B,T", SHAPES, ids=["unsorted", "sorted"])
def test_mean_fill_equals_per_expert_bias(B, T):
    """h -> keep h + (1 - keep) mu at one MoE site = the drop rule plus sum over routed slots of w_e x
    W_down[e] @ ((1 - keep_e) * mu_e): the servable per-expert bias (1.3)."""
    h = host(qn_toy())
    site = next(s for s in h.sites if s.is_moe and s.units == "neuron")
    l = site.layer
    rng = np.random.default_rng(1)
    keep = (rng.random(site.n) > 0.5).astype(np.float32)
    mu = rng.normal(size=site.n).astype(np.float32)
    ids = ids_batch(B, T)
    got = {}

    def cap(ll, kind, p):
        if ll == l:
            got["inds"], got["w"] = np.array(p["inds"]), np.array(p["w"])
    outs = {}
    for name, mean in (("drop", None), ("mean", mu)):
        h.ctl.reset()
        h.ctl.core = [mx.array(keep) if s.layer == l else mx.ones((h.widths[s.layer],)) for s in h.sites]
        h.ctl.moe_mean = None if mean is None else [mx.array(mean) if s.layer == l else None for s in h.sites]
        h.ctl.stats = cap

        def hook(self_, *a, _orig=type(site.module).__call__, _n=name):
            y = _orig(self_, *a)
            outs[_n] = (np.array(a[0]), np.array(y))
            return y
        old = type(site.module).__call__
        type(site.module).__call__ = hook
        try:
            mx.eval(h.hidden(ids))
        finally:
            type(site.module).__call__ = old
            h.ctl.reset()
    sw = site.module.switch_mlp
    Wd = np.array(sw.down_proj.weight)                                   # [E, D, F]
    K = keep.reshape(E, FF)
    M = mu.reshape(E, FF)
    inds, w = got["inds"], got["w"]
    bias = np.zeros(outs["drop"][1].shape, np.float64)
    for idx in np.ndindex(*inds.shape[:-1]):
        for j in range(inds.shape[-1]):
            e = int(inds[idx + (j,)])
            bias[idx] += float(w[idx + (j,)]) * (Wd[e] @ ((1 - K[e]) * M[e]))
    np.testing.assert_allclose(outs["mean"][1] - outs["drop"][1], bias, rtol=1e-4, atol=1e-5)
    assert np.abs(bias).max() > 1e-3                                       # not vacuous


@pytest.mark.parametrize("B,T", SHAPES, ids=["unsorted", "sorted"])
def test_teal_threshold_extremes_and_open_counter(B, T):
    h = host(qn_toy())
    ids = ids_batch(B, T)
    ones = [mx.ones((n,)) for n in h.widths]
    nl = [s.layer for s in h.sites if s.is_moe and s.units == "neuron"]
    acc = {}
    np.testing.assert_array_equal(_logits_with(h, ones, ids, thresh={l: 0.0 for l in nl}, open_acc=acc),
                                  logits(h, ones, ids))
    assert all(float(v[0]) == float(v[1]) > 0 for v in acc.values())
    # an infinite threshold closes every routed neuron, except position 0 in the sink layers: the zero core mask
    zero = [mx.zeros((s.n,)) if (s.is_moe and s.units == "neuron") else mx.ones((h.widths[s.layer],))
            for s in h.sites]
    acc = {}
    a = _logits_with(h, ones, ids, thresh={l: float("inf") for l in nl}, open_acc=acc, sink=h.L)
    b = _logits_with(h, zero, ids, sink=h.L)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    assert all(float(v[0]) == 0.0 for v in acc.values())
    k = h.sites[nl[0]].module.top_k
    assert float(acc[nl[0]][1]) == B * (T - 1) * k * FF                    # position 0 excluded (exact)


def test_text_ce_rows_bitwise_and_english_cost_perdoc():
    h = host(qn_toy())
    rng = np.random.default_rng(2)
    seqs = ids_batch(5, 10, seed=3).tolist() + ids_batch(3, 7, seed=4).tolist()      # two lengths
    core = _core(h, rng)

    def cfg():
        h.ctl.reset()
        h.ctl.core, h.ctl.sink_layers = core, h.L
    rows = {}
    a = Ev.text_ce(h, seqs, cfg, bs=3)
    b = Ev.text_ce(h, seqs, cfg, bs=3, rows_out=rows)
    assert a == b                                                          # bit for bit
    assert rows["ce"].sum() / rows["n"].sum() == pytest.approx(a[0], rel=1e-6) and rows["n"].sum() == a[1]
    assert list(rows["n"]) == [8] * 5 + [5] * 3
    one = {}
    Ev.text_ce(h, [seqs[6]], cfg, rows_out=one)
    assert one["ce"][0] == pytest.approx(rows["ce"][6], rel=1e-5)          # in document order
    split = {"x": {"held": seqs}}
    clean, clean2, pd = {}, {}, {}
    r0 = Ev.english_cost(h, split, clean, cfg)
    r1 = Ev.english_cost(h, split, clean2, cfg, perdoc=pd)
    assert r0["x"] == r1["x"]
    # pooled = token-weighted (the battery's value); the held sets cut every document to T tokens, so the documents
    # carry equal weight there and the per-document mean is the same number
    d = pd["x"]["ce"] - pd["x"]["clean_ce"]
    assert d.sum() / pd["x"]["n"].sum() == pytest.approx(r0["x"], abs=1e-5)       # fp32 row vs batch sums


def test_moestats_mu_counts_reap_and_colnorms():
    for quant in (False, True):
        h = host(qn_toy(quantize=quant))
        seqs = ids_batch(3, 10).tolist()
        st = MS.capture(h, seqs, quantiles=(0.25, 0.5))
        mean = Ev.mean_acts(h, seqs)
        for s in MS.neuron_sites(h):
            r = st[s.layer]
            np.testing.assert_allclose(r["mu"], np.array(mean[s.layer]), rtol=1e-5, atol=1e-6)
            assert r["count"].sum() == 3 * 9 * s.module.top_k and r["tokens"] == 27
            assert (r["reap_e"] >= 0).all() and (r["wh"] >= 0).all() and r["wh"].shape == (s.n,)
            assert r["habs_q"][0.25] <= r["habs_q"][0.5]
        cn = MS.down_col_norms(h, chunk=4)
        for s in MS.neuron_sites(h):
            W = np.array(s.module.switch_mlp.down_proj.weight)            # the toy's down_proj stays float (F=16)
            np.testing.assert_allclose(cn[s.layer], np.linalg.norm(W, axis=1).reshape(-1), rtol=1e-5)


def test_down_col_norms_dequantize_in_chunks():
    """FN's down_proj is quantized: the norms come from dequantized expert chunks."""
    from types import SimpleNamespace
    from mlx2.runtime.models import switch_layers as sl
    mx.random.seed(0)
    q = sl.QuantizedSwitchLinear(64, 24, 6, bias=False, group_size=32, bits=4)
    site = SimpleNamespace(is_moe=True, units="neuron", layer=2, E=6, F=64, kind="qwen_next",
                           module=SimpleNamespace(switch_mlp=SimpleNamespace(down_proj=q)))
    cn = MS.down_col_norms(SimpleNamespace(sites=[site]), chunk=4)
    W = np.array(mx.dequantize(q.weight, q.scales, q.biases, q.group_size, q.bits))
    assert W.shape == (6, 24, 64)
    np.testing.assert_allclose(cn[2], np.linalg.norm(W, axis=1).reshape(-1), rtol=1e-5)


def test_moestats_reap_matches_a_direct_computation():
    h = host(qn_toy())
    seqs = ids_batch(2, 8).tolist()
    st = MS.capture(h, seqs)
    s = MS.neuron_sites(h)[0]
    acc, cnt = np.zeros(E), np.zeros(E)

    def cap(l, kind, p):
        if l == s.layer:
            k = p["inds"].shape[-1]
            inds = np.array(p["inds"][:, 1:]).reshape(-1)
            w = np.array(p["w"][:, 1:, :k].astype(mx.float32)).reshape(-1)
            rows = np.array(p["rows"][:, 1:, :k].astype(mx.float32)).reshape(inds.size, -1)
            for e, ww, r in zip(inds, w, rows):
                acc[e] += ww * np.linalg.norm(r)
                cnt[e] += 1
    h.ctl.reset()
    h.ctl.stats = cap
    mx.eval(h.hidden(mx.array(seqs)))
    h.ctl.reset()
    np.testing.assert_allclose(st[s.layer]["reap_e"], acc / np.maximum(cnt, 1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(st[s.layer]["count"], cnt)


def test_fisher_accumulate_matches_capture_and_ce_gradients(tmp_path):
    h = host(qn_toy())
    h.check_paths(ids_batch(1, 10))
    seqs = ids_batch(3, 10, seed=5).tolist()
    r2 = FI.accumulate(h, seqs, "sampled", seed=4, log=lambda *_: None)
    FI.capture(h, seqs, tmp_path / "U", seed=4, log=lambda *_: None)
    U = FI.load(tmp_path / "U", h.L)
    for l in range(h.L):
        np.testing.assert_allclose(r2["sum"][l], (np.asarray(U[l], np.float64) ** 2).sum(0), rtol=1e-4, atol=1e-9)
    r1 = FI.accumulate(h, seqs, "ce", log=lambda *_: None, layers=[1, 3])
    assert sorted(r1["sum"]) == [1, 3] and r1["seqs"] == 3
    # first order: closing a unit moves the summed CE by about -g (the AtP zero-baseline attribution)
    from research.amoeba.carve.carving import ce_rows
    l = 3
    j = int(np.argmax(np.abs(r1["sum"][l])))

    def ce_at(eps):
        tot = 0.0
        for q in seqs:
            ids = mx.array([q])
            core = [mx.ones((n,)) for n in h.widths]
            v = np.ones(h.widths[l], np.float32)
            v[j] = 1 - eps
            core[l] = mx.array(v)
            h.ctl.reset()
            h.ctl.core, h.ctl.sink_layers = core, h.L
            lg = h.logits_at(ids, list(range(1, ids.shape[1] - 1)))
            h.ctl.reset()
            tot += float(ce_rows(lg, ids[:, 2:]).sum())
        return tot
    fd = (ce_at(-0.05) - ce_at(0.05)) / 0.1                                 # d CE / d m_j
    assert fd == pytest.approx(r1["sum"][l][j], rel=0.05, abs=1e-6)
    assert FI.accumulate(h, seqs, "ce", log=lambda *_: None, budget_s=-1)["seqs"] == 0
