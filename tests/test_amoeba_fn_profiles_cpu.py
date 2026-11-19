"""CPU tests of the FN neuron-profile capture (scripts/amoeba_fn_profile_capture.py) and its analyses
(scripts/amoeba_fn_profile_analyse.py, scripts/amoeba_fn_untangle_analyse.py) on the toy Qwen3-Next carve model:
every accumulator against a brute-force per-document computation, resume equality, and the analysis math against
direct computations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import amoeba_fn_profile_capture as CAP  # noqa: E402
import amoeba_fn_profile_lib as PL  # noqa: E402
import test_amoeba_carve_neuron_units_cpu as TN  # noqa: E402
from research.amoeba.carve.host import Host  # noqa: E402
from test_amoeba_carve_cpu import Tok  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _host(seed=0):
    return Host(model=TN.qn_toy(seed), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None, moe_units="neuron")


def _text(rng, n):
    return "".join(chr(97 + int(c)) for c in rng.integers(0, 26, n))


def _corpora(tmp):
    rng = np.random.default_rng(5)
    rows = []
    for t, n in (("prog.web.css", 5), ("prog.web", 3), ("prog.py", 4), ("sci.chem.x", 6), ("lang.arb", 4),
                 ("lang.eng", 3), ("misc", 2)):
        for i in range(n):
            rows.append({"text": _text(rng, int(rng.integers(3, 30))), "topics": [t, "other"],
                         "split": "fit" if i % 3 else "held"})
    a = tmp / "atlas.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = tmp / "sess.jsonl"
    s.write_text("".join(json.dumps({"text": _text(rng, int(rng.integers(8, 25))), "source": f"sess.k{i % 2}"}) + "\n"
                         for i in range(7)))
    p = tmp / "partition.json"
    p.write_text(json.dumps({"boxes": {"arb": {}, "sci.chem": {}, "prog.web": {}, "nope": {}}}))
    return a, s, p


def _args(tmp, out, extra=()):
    a, s, p = _corpora(tmp)
    return ["--model", "toy", "--out", str(out), "--corpus", f"atlas={a}?group=topics&depth=2",
            "--corpus", f"sess={s}?group=source", "--tokens", "16", "--min-tokens", "6", "--batch", "3",
            "--x-boxes", str(p), "--x-group", "sess=sess.*", "--sketch-rank", "4", "--calib-docs", "8",
            "--no-watchdog", *extra]


def _brute(host, out):
    """Per document, unpadded forward: the statistics recomputed in numpy from the raw site payloads."""
    rep = json.loads((out / "report.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    theta = np.load(out / "thresholds.npz")["theta"]
    sites = rep["sites"]
    S, E, F, D = len(sites), sites[0]["E"], sites[0]["F"], rep["d_model"]
    li = {s["layer"]: i for i, s in enumerate(sites)}
    omega = np.random.default_rng(0 + 101).standard_normal((D, 4)).astype(np.float32)
    a = CAP.build_args(_args(out.parent, out))
    per_group, *_ = CAP.plan_documents(a, host, lambda *_: None)
    groups = [g["id"] for g in rep["groups"]]
    G = len(groups)
    mean = np.zeros((G, S, E * F)); strong = np.zeros((G, S, E * F)); rout = np.zeros((G, S, E)); ws = np.zeros((G, S, E))
    tok = np.zeros(G)
    X = {}
    for gi, g in enumerate(groups):
        xg = plan["identity"]["groups"][gi][2]
        for doc in per_group[g]:
            cap = {}

            def st(l, kind, p):
                if l in li and p.get("h") is not None:
                    cap[l] = {k: np.array(p[k].astype(mx.float32)) for k in ("inds", "w", "h", "x")}
            host.ctl.reset()
            host.ctl.stats = st
            mx.eval(host.hidden(mx.array(np.asarray(doc, np.int32)[None])))
            host.ctl.reset()
            tok[gi] += len(doc)
            for l, c in cap.items():
                si = li[l]
                inds, w, h, x = c["inds"][0], c["w"][0], c["h"][0], c["x"][0]
                xs = X.setdefault(xg, {"n": np.zeros((S, E)), "s": np.zeros((S, E, D)), "q": np.zeros((S, E, D)),
                                       "y": np.zeros((S, E, D, 4)), "h1": np.zeros((S, E, F)),
                                       "h2": np.zeros((S, E, F))}) if xg else None
                for t in range(inds.shape[0]):
                    for j in range(inds.shape[1]):
                        e = int(inds[t, j])
                        cc = np.abs(h[t, j]) * w[t, j]
                        mean[gi, si, e * F:(e + 1) * F] += cc
                        strong[gi, si, e * F:(e + 1) * F] += cc > theta[si]
                        rout[gi, si, e] += 1
                        ws[gi, si, e] += w[t, j]
                        if xs is not None:
                            xs["n"][si, e] += 1
                            xs["s"][si, e] += x[t]
                            xs["q"][si, e] += x[t] ** 2
                            xs["y"][si, e] += np.outer(x[t], x[t] @ omega)
                            xs["h1"][si, e] += h[t, j]
                            xs["h2"][si, e] += h[t, j] ** 2
        mean[gi] /= tok[gi]
        strong[gi] /= tok[gi]
    return {"mean": mean, "strong": strong, "routed": rout, "wsum": ws, "tokens": tok, "x": X}


def test_capture_matches_brute_force(tmp_path):
    host = _host()
    out = tmp_path / "cap"
    assert CAP.main(_args(tmp_path, out), host=host) == 0
    rep = json.loads((out / "report.json").read_text())
    groups = [g["id"] for g in rep["groups"]]
    # level-2 groups, x groups contiguous: box groups (lang.arb, prog.web, sci.chem), the pooled sess, then rest
    assert set(groups) == {"prog.web", "prog.py", "sci.chem", "lang.arb", "lang.eng", "misc", "sess.k0", "sess.k1"}
    assert rep["x_groups"] == ["lang.arb", "prog.web", "sci.chem", "sess", "rest"]
    assert rep["x_box_alias"] == {"arb": "lang.arb", "nope": None, "prog.web": "prog.web", "sci.chem": "sci.chem"}
    xs_of = [g["x_group"] for g in rep["groups"]]
    assert xs_of == sorted(xs_of, key=rep["x_groups"].index)
    assert [s["layer"] for s in rep["sites"]] == [1, 3] and [s["layer"] for s in rep["skipped_sites"]] == [0, 2]
    bf = _brute(host, out)
    cap = PL.Capture(out)
    assert np.allclose(cap.tokens, bf["tokens"])
    E, F = cap.E, cap.F
    gm = np.stack([cap.profile_layer("mean", li) for li in range(2)], 1).astype(np.float64)
    emax = np.repeat(np.abs(bf["mean"]).reshape(len(groups), 2, E, F).max(3), F, axis=2)
    assert (np.abs(gm - bf["mean"]) <= 1e-3 * emax + 1e-12).all()        # fp16 relative to the expert's max
    got_s = np.stack([cap.profile_layer("strong", li) for li in range(2)], 1) * bf["tokens"][:, None, None]
    assert np.abs(got_s - bf["strong"] * bf["tokens"][:, None, None]).max() <= 1.01      # a value at theta may flip
    assert (bf["strong"] > 0).any() and (bf["strong"] <= 1).all()
    np.testing.assert_allclose(cap.routed(), bf["routed"], atol=1e-4)
    np.testing.assert_allclose(cap.wsum(), bf["wsum"], rtol=1e-4, atol=1e-5)
    tz = cap.totals()
    np.testing.assert_allclose(tz["sum"], (bf["mean"] * bf["tokens"][:, None, None]).sum(0), rtol=1e-4, atol=1e-5)
    assert float(tz["tokens"]) == bf["tokens"].sum()
    assert set(bf["x"]) == set(rep["x_groups"]) and len(bf["x"]) == 5
    for xg, ref in bf["x"].items():
        xs = cap.xstats(xg)
        np.testing.assert_allclose(xs.n, ref["n"])
        n = np.maximum(ref["n"], 1)[..., None]
        for li in range(2):
            mu, m2, y = xs.layer(li)
            scale = np.abs(ref["s"][li] / n[li]).max(1, keepdims=True) + 1e-12
            assert np.abs(mu - ref["s"][li] / n[li]).max() / scale.max() < 2e-3
            np.testing.assert_allclose(m2, ref["q"][li] / n[li], rtol=2e-3, atol=2e-3 * np.abs(ref["q"][li] / n[li]).max())
            yr = ref["y"][li] / n[li][..., None]
            np.testing.assert_allclose(y, yr, rtol=3e-3, atol=2e-3 * np.abs(yr).max())
            h1, h2 = xs.h_moments(li)
            for got, key in ((h1, "h1"), (h2, "h2")):
                want = ref[key][li] / n[li]
                assert (np.abs(got - want) <= 1e-3 * np.abs(want).max(1, keepdims=True) + 1e-12).all(), key
    assert sum(ref["n"].sum() for ref in bf["x"].values()) > 0


def test_scaled_fp16_keeps_small_means():
    rng = np.random.default_rng(0)
    v = rng.uniform(1e-6, 9e-5, (4, 640)).astype(np.float32)          # FN-like per-token means (review 2026-11-18)
    q, s = PL.scaled_f16(v, (1,))
    back = PL.unscale(q, s, (1,))
    assert np.abs(back / v - 1).max() < 2e-3
    plain = v.astype(np.float16).astype(np.float32)
    assert np.abs(plain / v - 1).max() > 1e-2                        # what unscaled fp16 would have lost


def test_resume_equals_one_shot(tmp_path, monkeypatch):
    host = _host()
    one = tmp_path / "one"
    assert CAP.main(_args(tmp_path, one), host=host) == 0
    two = tmp_path / "two"
    calls = {"n": 0}

    def pend(*_):
        calls["n"] += 1
        return ["test"] if calls["n"] in (3, 6) else []       # yield mid-group (ckpt every batch) and again later
    monkeypatch.setattr(CAP, "others_pending", pend)
    args = _args(tmp_path, two, ("--ckpt-seconds", "0"))
    assert CAP.main(args, host=host) == CAP.YIELD
    assert (two / "ckpt.npz").exists() and not (two / "report.json").exists()
    assert CAP.main(args, host=host) == CAP.YIELD
    monkeypatch.setattr(CAP, "others_pending", lambda *_: [])
    assert CAP.main(args, host=host) == 0
    a, b = PL.Capture(one), PL.Capture(two)
    for stat in ("mean", "strong"):
        assert np.array_equal(a.profile(stat), b.profile(stat))
        assert np.array_equal(a.profile_scale(stat), b.profile_scale(stat))
    assert np.array_equal(a.routed(), b.routed())
    for k in ("sum", "strong"):
        assert np.array_equal(a.totals()[k], b.totals()[k])
    for xg in a.xgroups():
        za, zb = a.xstats(xg).z, b.xstats(xg).z
        for k in za.files:
            assert np.array_equal(za[k], zb[k]), (xg, k)


def test_resume_refuses_another_identity(tmp_path, monkeypatch):
    host = _host()
    out = tmp_path / "c"
    monkeypatch.setattr(CAP, "others_pending", lambda *_: ["x"] if (out / "plan.json").exists() else [])
    assert CAP.main(_args(tmp_path, out), host=host) == CAP.YIELD
    monkeypatch.setattr(CAP, "others_pending", lambda *_: [])
    with pytest.raises(SystemExit, match="another identity"):
        CAP.main(_args(tmp_path, out, ("--tokens", "12")), host=host)


def test_thresholds_reuse_and_spec_parsing(tmp_path):
    host = _host()
    one = tmp_path / "one"
    assert CAP.main(_args(tmp_path, one), host=host) == 0
    two = tmp_path / "two"
    assert CAP.main(_args(tmp_path, two, ("--thresholds", str(one / "thresholds.npz"))), host=host) == 0
    assert np.array_equal(np.load(one / "thresholds.npz")["theta"], np.load(two / "thresholds.npz")["theta"])
    name, path, filt, opts = PL.parse_spec("inf=/a/b c.jsonl?box=ml.inference&group=box&max_docs=5&split=fit")
    assert (name, path, filt) == ("inf", "/a/b c.jsonl", {"box": "ml.inference", "split": "fit"})
    assert opts["group"] == "box" and opts["max_docs"] == 5
    assert PL.row_group({"topics": ["prog.web.css", "x"]}, {"group": "topics", "depth": 2}) == "prog.web"
    assert PL.row_group({"source": "q"}, {"label": "eng"}) == "eng"
    with pytest.raises(SystemExit):
        PL.parse_spec("bad/name=x")


# ---------------------------------------------------------------------------------------------- analyses

import amoeba_fn_partition as PART  # noqa: E402
import amoeba_fn_profile_analyse as AN  # noqa: E402
import amoeba_fn_untangle_analyse as UN  # noqa: E402

BOXES = ["arb", "prog.web", "sci.chem"]


def _partition(tmp, layers=(1, 3), L_all=4, E=6, F=16, seed=3):
    """Synthetic firing maps -> scripts/amoeba_fn_partition.py (the real partition code) -> PART dir."""
    rng = np.random.default_rng(seed)
    n = E * F
    per_site = [{"layer": l, "class": "moe", "units": n, "F": F} for l in layers]

    def mk(d, target, mt, mb):
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps({"per_site": per_site, "fingerprint": "toy:abc", "checkpoint": "toy@abc",
                                                   "baseline": {"corpus": "b"}, "target": {"corpus": str(d)},
                                                   "capture_rev": "x"}))
        arr = {}
        for s in range(len(layers)):
            arr[f"target_{s}"], arr[f"mean_target_{s}"], arr[f"mean_baseline_{s}"] = target[s], mt[s], mb[s]
        np.savez(d / "map.npz", **arr)
    S = len(layers)
    mk(tmp / "maps" / "eng", rng.random((S, n)) < 0.05, np.ones((S, n)), np.ones((S, n)))
    shared = rng.random((S, n)) < 0.15                 # planted: claimed by every box (3+ -> core)
    pair = rng.random((S, n)) < 0.2                    # planted: arb + sci.chem
    args = []
    for b in BOXES:
        t = (rng.random((S, n)) < 0.15) | shared | (pair & (b != "prog.web"))
        mb = np.where(rng.random((S, n)) < 0.1, 0.5, 0.1)
        mk(tmp / "maps" / b, t, np.ones((S, n)), mb)
        args += ["--box", f"{b}={tmp / 'maps' / b}"]
    t2 = np.zeros((L_all, E), np.uint8)
    t2[:, 0] = 1
    np.savez(tmp / "t2.npz", core=t2)
    old = sys.argv
    sys.argv = ["x", *args, "--core-map", str(tmp / "maps" / "eng"), "--t2", str(tmp / "t2.npz"), "--out", str(tmp / "part")]
    try:
        PART.main()
    finally:
        sys.argv = old
    pj = json.loads((tmp / "part" / "partition.json").read_text())
    pj["fingerprint"] = "toy:abc"
    (tmp / "part" / "partition.json").write_text(json.dumps(pj))
    return tmp / "part"


def test_partition_categories_reproduce_the_partition(tmp_path):
    pdir = _partition(tmp_path)
    part = PL.partition_categories(pdir, log=lambda *_: None)
    pj = part["partition"]
    tiles = np.load(pdir / "tiles.npz")
    cat = part["cat"]
    assert int((cat == PL.CATS["unassigned"]).sum()) == pj["core"]["unassigned"]
    core = (cat <= PL.CATS["shared3"]) | (cat == PL.CATS["unassigned"])
    assert np.array_equal(core, tiles["core"])
    for k in pj["pairs"]:
        a_, b_ = k.split("__")
        bits = (1 << part["names"].index(a_)) | (1 << part["names"].index(b_))
        assert np.array_equal(tiles[f"pair:{k}"], (cat == PL.CATS["pair"]) & (part["claims"] == bits))
    assert pj["pairs"].get("arb__sci.chem", 0) > 0 and pj["core"]["from_3plus_boxes"] > 0
    pj["core"]["unassigned"] += 1                      # a changed input no longer reproduces: refused
    (pdir / "partition.json").write_text(json.dumps(pj))
    with pytest.raises(SystemExit, match="do not reproduce"):
        PL.partition_categories(pdir, log=lambda *_: None)


def _fake_capture(d, groups, mean, strong, routed, tokens, layers=(1, 3), E=6, F=16):
    (d / "profiles").mkdir(parents=True)
    G, S = mean.shape[:2]
    for nm, v in (("mean", mean), ("strong", strong)):
        q, s = PL.scaled_f16(v.reshape(G, S, E, F), (3,))
        np.save(d / "profiles" / f"{nm}.npy", q.reshape(G, S, E * F))
        np.save(d / "profiles" / f"{nm}_scale.npy", s)
    np.save(d / "profiles" / "routed.npy", routed.astype(np.float32))
    np.save(d / "profiles" / "wsum.npy", routed.astype(np.float32) * 0.3)
    tok = np.asarray(tokens, np.float64)
    np.savez(d / "totals.npz", sum=(mean * tok[:, None, None]).sum(0), strong=(strong * tok[:, None, None]).sum(0),
             tokens=np.array(tok.sum()))
    (d / "report.json").write_text(json.dumps({
        "fingerprint": "toy:abc", "checkpoint": "toy@abc", "mlx2_source_revision": "x", "d_model": 32,
        "sites": [{"layer": l, "E": E, "F": F} for l in layers], "thresholds": {"sha12": "t"}, "x_groups": [],
        "groups": [{"id": g, "tokens": float(t), "x_group": None} for g, t in zip(groups, tokens)]}))


def test_profile_analysis_planted_clusters_rare_dead(tmp_path):
    pdir = _partition(tmp_path)
    part = PL.partition_categories(pdir, log=lambda *_: None)
    cat = part["cat"]
    S, n, E, F = 2, 96, 6, 16
    groups = ["lang.arb", "prog.web", "sci.chem", "hum.a", "hum.b", "soc.c", "soc.d", "sess.x"]
    G = len(groups)
    rng = np.random.default_rng(1)
    plant = rng.integers(0, 3, (S, n))                 # three planted profile families
    fam = {0: [3, 4], 1: [5, 6], 2: [7]}
    mean = np.full((G, S, n), 0.01) * rng.uniform(0.5, 1.5, (G, S, n))
    for s in range(S):
        for u in range(n):
            mean[fam[int(plant[s, u])], s, u] += 1.0
    strong = np.minimum(mean * 0.01, 1)
    routed = np.full((G, S, E), 50.0)
    routed[:, 1, 5] = 0                                # expert 5 of layer 3 never routed
    mean[:, 1, 5 * F:] = 0
    strong[:, 1, 5 * F:] = 0
    dead_u, rare_u = 7, 9                              # layer 1: unit 7 never strong + tiny mean; unit 9 rarely strong
    mean[:, 0, dead_u], strong[:, 0, dead_u] = 1e-9, 0
    strong[:, 0, rare_u] = 1e-7
    cap = tmp_path / "cap"
    _fake_capture(cap, groups, mean, strong, routed, [100] * G)
    out = tmp_path / "an"
    assert AN.main(["--capture", str(cap), "--partition", str(pdir), "--out", str(out), "--clusters", "3",
                    "--clusters-shared", "2", "--rare", "1e-5"]) == 0
    rep = json.loads((out / "report.json").read_text())
    z = np.load(out / "units.npz")
    assert z["never_routed"][1, 5 * F:].all() and not z["never_routed"][0].any()
    assert z["dead"][0, dead_u] and z["rare"][0, rare_u] and not z["rare"][0, dead_u]
    ua = (cat == PL.CATS["unassigned"]) & ~z["never_routed"] & ~z["dead"]
    assert ua.sum() > 10
    lab = z["unassigned_cluster"][ua]
    pl = plant[ua]
    assert (z["unassigned_cluster"][~ua & (cat == PL.CATS["unassigned"])] == -1).all()
    purity = sum(np.bincount(pl[lab == c]).max() for c in np.unique(lab)) / lab.size
    assert purity == 1.0 and len(np.unique(lab)) == 3
    s, u = np.argwhere(ua)[0]                          # top1 group equals the direct enrichment argmax
    assert z["unassigned_top1"][s, u] == int(np.argmax(mean[:, s, u]))   # the baseline is one scalar per unit
    fam_means = [mean[[3, 4], s, u].mean(), mean[[5, 6], s, u].mean(), mean[7, s, u], mean[0, s, u],
                 mean[1, s, u], mean[2, s, u]]                          # hum, soc, sess, lang, prog, sci
    want = mean[:, s, u].max() / np.median(fam_means)
    assert abs(float(z["unassigned_e1"][s, u]) / want - 1) < 2e-3      # enrichment vs the median FAMILY
    cl = json.loads((out / "clusters_unassigned.json").read_text())["clusters"]
    tops = [set(c["top_families_enrichment"]) for c in cl]
    assert any("hum" in t for t in tops) and any("soc" in t for t in tops) and any("sess" in t for t in tops)
    dr = rep["rare_dead_by_category"]
    assert sum(v["units"] for v in dr.values()) == S * n
    assert rep["box_groups"] == {"arb": "lang.arb", "prog.web": "prog.web", "sci.chem": "sci.chem"}
    pr = (cat == PL.CATS["pair"]) & ~z["never_routed"] & ~z["dead"]
    assert rep["claimed"]["pair"]["units_with_box_groups"] == int(pr.sum())


def test_nystrom_and_moment_model_are_exact_at_low_rank():
    rng = np.random.default_rng(0)
    D, r = 20, 6
    X = rng.standard_normal((500, 4)) @ rng.standard_normal((4, D)) + 0.3          # rank <= 5 second moment
    M = X.T @ X / len(X)
    om = rng.standard_normal((D, r))
    B = PL.nystrom_factor(M @ om, om)
    np.testing.assert_allclose(B @ B.T, M, atol=1e-8)
    mom = PL.Moments(X.mean(0), np.diag(M), B)
    A = rng.standard_normal((D, 3))
    cov = np.cov(X.T, bias=True)
    np.testing.assert_allclose(mom.cov(A), A.T @ cov @ A, atol=1e-8)
    assert np.abs(mom.dres).max() < 1e-8
    Xf = rng.standard_normal((800, D)) * np.linspace(0.2, 3, D)                     # full rank: diagonal stays exact
    Mf = Xf.T @ Xf / len(Xf)
    mf = PL.Moments(Xf.mean(0), np.diag(Mf), PL.nystrom_factor(Mf @ om, om))
    np.testing.assert_allclose(np.diag(mf.cov(np.eye(D))) + mf.mu ** 2, np.diag(Mf), rtol=1e-8)


def test_gaussian_split_matches_direct_sampling():
    rng = np.random.default_rng(2)
    D = 12
    Lc = rng.standard_normal((D, D)) * 0.3
    cov = Lc @ Lc.T + np.diag(rng.uniform(0.05, 0.3, D))
    mu = rng.standard_normal(D) * 0.5
    M = cov + np.outer(mu, mu)
    w, V = np.linalg.eigh(M)
    mom = PL.Moments(mu, np.diag(M), V * np.sqrt(w))                               # exact full-rank factor
    g, u = rng.standard_normal(D), rng.standard_normal(D)
    mask = np.zeros(D)
    mask[[1, 4, 7]] = 1
    got = PL.gaussian_split(mom, g, u, mask, rng.standard_normal((200_000, 4)))
    x = rng.multivariate_normal(mu, cov, 200_000)
    gS, uS = g * mask, u * mask
    a = PL.silu(x @ g) * (x @ u)
    aO = PL.silu(x @ (g - gS)) * (x @ (u - uS))
    aS = PL.silu(x @ gS) * (x @ uS)
    ref = {"a2": (a * a).mean(), "keep": ((a - aO) ** 2).mean(), "copy": ((a - aS) ** 2).mean(),
           "sum": ((a - aO - aS) ** 2).mean(), "leak": (aO * aO).mean()}
    for k, v in ref.items():
        assert abs(got[k] - v) / v < 0.04, (k, got[k], v)


def test_concentration():
    c = np.zeros(100)
    c[[3, 50]] = [3.0, 1.0]
    r = PL.concentration(c, [1, 2, 8])
    assert r["l1_share"] == {1: 0.75, 2: 1.0, 8: 1.0} and r["dims90"] == 2 and abs(r["pr"] - 1.6) < 1e-12
    flat = PL.concentration(np.ones(100), [10])
    assert abs(flat["l1_share"][10] - 0.1) < 1e-12 and abs(flat["pr"] - 100) < 1e-9


@pytest.mark.parametrize("switch,quantize", [("split", False), ("split", True), ("fused", False)])
def test_expert_weights_reproduce_the_switch(tmp_path, switch, quantize):
    from mlx.utils import tree_flatten
    m = TN.qn_toy(switch=switch, quantize=quantize)
    host = Host(model=m, tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None, moe_units="neuron")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(m.parameters())))
    W = PL.ExpertWeights(tmp_path, 32, 16)
    cap = {}

    def st(l, kind, p):
        if p.get("h") is not None:
            cap[l] = {k: np.array(p[k].astype(mx.float32)) for k in ("inds", "h", "x")}
    host.ctl.reset()
    host.ctl.stats = st
    mx.eval(host.hidden(mx.array([[5, 9, 13, 2, 40, 7]], dtype=mx.int32)))
    host.ctl.reset()
    assert set(cap) == {1, 3}
    for l, c in cap.items():
        for t in range(c["inds"].shape[1]):
            for j in range(c["inds"].shape[2]):
                e = int(c["inds"][0, t, j])
                g, u, d = W.rows(l, e)
                x = c["x"][0, t]
                np.testing.assert_allclose(PL.silu(g @ x) * (u @ x), c["h"][0, t, j], rtol=2e-3, atol=2e-4)
                assert d.shape == (32, 16)


def test_expert_weights_ignore_the_mtp_head(tmp_path):
    """FN names its MTP head language_model.mtp.layers.0.mlp.switch_mlp.* (same shapes as decoder layer 0): those
    must never be read as decoder layer 0 (review 2026-11-18)."""
    from mlx.utils import tree_flatten
    m = TN.qn_toy()
    p = {f"language_model.{k}": v for k, v in tree_flatten(m.parameters())}
    for k, v in list(p.items()):
        if ".layers.1.mlp.switch_mlp." in k:
            p[k.replace("language_model.model.layers.1.", "language_model.mtp.layers.1.")] = v * 0 + 7.0
    mx.save_safetensors(str(tmp_path / "model.safetensors"), p)
    g, u, d = PL.ExpertWeights(tmp_path, 32, 16).rows(1, 0)
    want = np.array(m.model.layers[1].mlp.switch_mlp.gate_proj.weight[0])
    np.testing.assert_allclose(g, want)
    assert not np.allclose(g, 7.0)


def test_untangle_end_to_end_on_the_toy_capture(tmp_path):
    from mlx.utils import tree_flatten
    host = _host()
    out = tmp_path / "cap"
    assert CAP.main(_args(tmp_path, out, ("--x-group", "english=lang.eng")), host=host) == 0
    pdir = _partition(tmp_path)
    mdir = tmp_path / "model"
    mdir.mkdir()
    mx.save_safetensors(str(mdir / "model.safetensors"), dict(tree_flatten(host.model.parameters())))
    res = tmp_path / "un"
    assert UN.main(["--capture", str(out), "--partition", str(pdir), "--model", str(mdir), "--out", str(res),
                    "--topk", "2,8", "--per-tile", "20", "--shared3", "20", "--mc", "256", "--min-tokens", "1",
                    "--pcs", "3"]) == 0
    rep = json.loads((res / "report.json").read_text())
    rows = [json.loads(x) for x in (res / "neurons.jsonl").read_text().splitlines()]
    assert rep["baseline_x_group"] == "english" and rep["box_x_groups"]["arb"] == "lang.arb"
    assert {r["kind"] for r in rows} == {"pair", "shared3"}
    cp = PL.Capture(out)
    r = next(r for r in rows if r["kind"] == "pair")
    li = cp.layers.index(r["layer"])
    xa, xb = cp.xstats(rep["box_x_groups"][r["boxes"][0]]), cp.xstats(rep["box_x_groups"][r["boxes"][1]])
    g, _, _ = PL.ExpertWeights(mdir, 32, 16).rows(r["layer"], r["expert"])
    delta = xa.layer(li)[0][r["expert"]] - xb.layer(li)[0][r["expert"]]
    assert abs(r["contrast"]["g"]["total"] - float(g[r["neuron"]] @ delta)) < 1e-3
    assert "drive_jaccard" in r and len(r["splits"]) == 2 * 2 * 2
    assert all(s["keep"] >= 0 and s["copy"] >= 0 for s in r["splits"])
    assert rep["gaussian_check"]["neurons_x_topics"] > 0
    assert rep["expert_mixing"]["box_groups"] == 4
    # the model check reads the exact E[h^2] of that neuron and topic
    h1, h2 = xa.h_moments(li)
    mm = r["model"][r["boxes"][0]]
    assert abs(mm["a2_exact"] - float(h2[r["expert"], r["neuron"]])) <= 1e-3 * abs(mm["a2_exact"]) + 1e-6
    assert rep["gaussian_check"]["a2_exact"]["spearman"] is not None
    assert set(r["gates"]) == {"snr", "model"} and "null_l1_share" in r["contrast"]["g"]
    assert all(x["viable"] is None or (x["gates"]["snr"] and x["gates"]["model"]) for x in rows)


def test_noise_null_matches_pure_noise_contrast():
    rng = np.random.default_rng(4)
    D = 2560
    w = rng.standard_normal(D)
    se = rng.uniform(0.5, 2.0, D)
    null, den = PL.noise_null(w, se, [8, 32, 128], rng, draws=16)
    assert null[8] > 5 * 8 / D and null[32] > 4 * 32 / D and null[128] > 3 * 128 / D       # far above k / D
    snr = [abs(w @ (se * rng.standard_normal(D))) / den for _ in range(400)]
    assert 0.6 < np.mean(snr) < 1.0                                                     # E|N(0,1)| = 0.80


def test_mean_diff_se_and_correlated_snr():
    rng = np.random.default_rng(5)
    D = 6
    xa = rng.standard_normal((4000, D)) * np.arange(1, D + 1) + 1.0
    xb = rng.standard_normal((3000, D)) + 0.5
    st = [(len(x), x.mean(0), (x * x).mean(0)) for x in (xa, xb)]
    se = PL.mean_diff_se(*st)
    np.testing.assert_allclose(se, np.sqrt(xa.var(0) / len(xa) + xb.var(0) / len(xb)), rtol=1e-10)
    # perfectly correlated dims: the diagonal se understates the variance of w . delta, the model one does not
    z = rng.standard_normal((5000, 1))
    xc = np.repeat(z, D, axis=1) + 0.01 * rng.standard_normal((5000, D))
    M = xc.T @ xc / len(xc)
    w_, V = np.linalg.eigh(M)
    mom = PL.Moments(xc.mean(0), np.diag(M), V * np.sqrt(np.maximum(w_, 0)))
    w = np.ones(D)
    sec = PL.mean_diff_se((5000, xc.mean(0), np.diag(M)), (5000, xc.mean(0), np.diag(M)))
    s = UN.snr_of(w, np.ones(D) * 0.1, sec, mom, 5000, mom, 5000)
    want = 0.6 / np.sqrt(2 * (xc @ w).var() / 5000)
    assert abs(s / want - 1) < 0.02 and s < 0.6 / np.sqrt(((w * sec) ** 2).sum())


def test_expert_weights_refuse_duplicate_keys(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.layers.0.mlp.switch_mlp.gate_proj.weight": "a.safetensors",
        "x.model.layers.0.mlp.switch_mlp.gate_proj.weight": "a.safetensors"}}))
    with pytest.raises(SystemExit, match="two switch-MLP weights"):
        PL.ExpertWeights(tmp_path, 32, 16)


def test_untangle_gates_withhold_ranking(tmp_path):
    from mlx.utils import tree_flatten
    host = _host()
    out = tmp_path / "cap"
    assert CAP.main(_args(tmp_path, out, ("--x-group", "english=lang.eng")), host=host) == 0
    pdir = _partition(tmp_path)
    mdir = tmp_path / "model"
    mdir.mkdir()
    mx.save_safetensors(str(mdir / "model.safetensors"), dict(tree_flatten(host.model.parameters())))
    common = ["--capture", str(out), "--partition", str(pdir), "--model", str(mdir), "--topk", "2,8,32",
              "--per-tile", "20", "--shared3", "20", "--mc", "256", "--min-tokens", "1", "--pcs", "3", "--tau", "5"]
    assert UN.main([*common, "--out", str(tmp_path / "open"), "--min-snr", "0", "--max-model-ratio", "1e9"]) == 0
    assert UN.main([*common, "--out", str(tmp_path / "shut"), "--min-snr", "1e9"]) == 0
    load = lambda d: [json.loads(x) for x in (tmp_path / d / "neurons.jsonl").read_text().splitlines()]  # noqa: E731
    op, sh = load("open"), load("shut")
    assert sum(r["viable"] is not None for r in op) > 0                 # tau 5: everything is viable ungated
    assert all(r["viable"] is None for r in sh)                          # the SNR gate withholds every ranking
    assert sum(r["viable_raw"] is not None for r in sh if r["kind"] == "pair") == \
        sum(r["viable"] is not None for r in op if r["kind"] == "pair")  # viable_raw keeps the ungated answer
    assert sum(r["viable_raw"] is not None for r in sh if r["kind"] == "shared3") > 0
