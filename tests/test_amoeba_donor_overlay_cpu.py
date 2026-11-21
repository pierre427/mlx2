"""Donor-wins box overlay + late sinks (scripts/amoeba_donor_overlay.py) and its fit-side corpus
(scripts/amoeba_overlay_corpus.py), on CPU with tiny Qwen3.5 hybrids (the bridge lab's host family)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import amoeba_donor_overlay as DO  # noqa: E402
import amoeba_overlay_corpus as OC  # noqa: E402

mx.set_default_device(mx.cpu)


def _tiny(seed):
    from test_differentiable_reference_path_cpu import _tiny_hybrid
    m = _tiny_hybrid(seed)
    DO.patch_mlps(DO.text_layers(m))
    return m


@pytest.fixture(scope="module")
def host():
    return _tiny(3)


def _dense(lin):
    if "scales" in lin:
        return mx.dequantize(lin.weight, lin.scales, lin.get("biases"), lin.group_size, lin.bits,
                             **({"mode": lin.mode} if hasattr(lin, "mode") else {}))
    return lin.weight


def _rows(model):
    layers = DO.text_layers(model)

    def rows(l, ids):
        m = layers[l].mlp
        wg, wu = (np.array(_dense(p).astype(mx.float32)) for p in (m.gate_proj, m.up_proj))
        wd = np.array(_dense(m.down_proj).astype(mx.float32)).T
        ids = np.asarray(ids)
        return wg[ids].copy(), wu[ids].copy(), wd[ids].copy()
    return rows


def _spec(counts, I=64, gate=8, seed=0):
    g = np.random.default_rng(seed)
    spec = {}
    for l, n in counts.items():
        ids, copy = DO.expand_units(g.permutation(I), n)
        k = min(gate, n)
        spec[l] = {"units": ids, "copy": copy, "gate_units": ids[:k], "gate_mu": np.zeros(k, np.float32),
                   "gate_sd": np.ones(k, np.float32), "gate_w": np.zeros(k, np.float32),
                   "gate_b": np.float32(2.0)}
    return spec


IDS = [[5, 17, 33, 9, 70, 2, 41, 88, 13, 64, 7, 99]]


# ------------------------------------------------------------------------------------------- selection masks
def test_win_positions_and_text_win():
    h = np.log([0.5, 0.1, 0.2, 0.9])
    d = np.log([0.5, 0.5, 0.21, 0.1])
    assert DO.win_positions(h, d, 0.5).tolist() == [False, True, False, False]
    assert DO.win_positions(h, d, 0.0).tolist() == [True, True, True, False]  # >= margin: ties count at 0
    assert DO.text_win(10.0, 8.0, 100, 0.0) and not DO.text_win(10.0, 8.0, 100, 0.05)
    assert DO.text_win(10.0, 8.0, 10, 0.2)


# ------------------------------------------------------------------------------------------- budget accounting
def test_budget_accounting_equal_vectors():
    c = DO.overlay_counts(2048, [10, 14, 18, 22], [30, 31], 0.25)
    assert sum(c.values()) == 2048 // 3 and c[30] + c[31] == round(682 * 0.25)
    assert DO.overlay_counts(2048, [10, 14], [], 0.25) == {10: 341, 14: 341}
    assert DO.overlay_counts(2048, [], [30, 31], 1.0) == {30: 341, 31: 341}
    assert DO.tile_units(2048, 32) == 32  # the 0c tile: 32 units x 32 layers
    vec_ov = DO.VEC_PER_UNIT * sum(c.values())
    vec_tile = DO.VEC_PER_TILE * 32 * 32
    assert abs(vec_ov - vec_tile) <= DO.VEC_PER_UNIT  # within one unit of the same budget
    assert DO.scale_counts({1: 128, 30: 85}, 100) == {1: 12800, 30: 8500}  # per layer


def test_param_count_matches_built_overlay(host):
    D = 32
    counts = {1: 5, 3: 7}
    spec = _spec(counts)
    core = {1: {"units": np.arange(4), "rank": 2}}
    P, C = DO.build_overlay(spec, _rows(host), D, core=core)
    want = DO.overlay_param_count(counts, D, {l: min(8, n) for l, n in counts.items()}, {1: (4, 2)})
    assert DO.n_params(P) == want
    T = DO.build_tiles(range(4), 3, D, 0)
    assert DO.n_params(T) == 4 * (2 * D * 3 + D + 1)
    assert DO.overlay_train_gb(1e9, 4) == pytest.approx(24.0)  # + 2 kept checkpoints


# ------------------------------------------------------------------------------------------- no-op, gate, sinks
def test_write_scale_zero_is_exact_noop_and_nonzero_is_not(host):
    ref = DO.forward(host, IDS, res32=True)
    P, C = DO.build_overlay(_spec({1: 6, 3: 6}), _rows(host), 32, write_scale=0.0)
    assert float(mx.abs(DO.forward(host, IDS, P=P, C=C, res32=True) - ref).max()) == 0.0
    P2, C2 = DO.build_overlay(_spec({1: 6, 3: 6}), _rows(host), 32, write_scale=0.5)
    assert float(mx.abs(DO.forward(host, IDS, P=P2, C=C2, res32=True) - ref).max()) > 1e-4
    # the hook leaves no residue: a later plain forward is the host again
    assert float(mx.abs(DO.forward(host, IDS, res32=True) - ref).max()) == 0.0


def test_gate_closed_is_exact_host(host):
    ref = DO.forward(host, IDS, res32=True)
    P, C = DO.build_overlay(_spec({1: 6, 2: 4, 3: 6}), _rows(host), 32, write_scale=1.0)
    DO.CLOSED[0] = True
    try:
        out = DO.forward(host, IDS, P=P, C=C, res32=True)
    finally:
        DO.CLOSED[0] = False
    assert float(mx.abs(out - ref).max()) == 0.0
    # the core delta is always open: closed gates leave it on
    core = {1: {"units": np.arange(4), "rank": 2}}
    Pc, Cc = DO.build_overlay(_spec({1: 6}), _rows(host), 32, write_scale=0.0, core=core)
    Pc["L1.V"] = mx.ones_like(Pc["L1.V"])
    DO.CLOSED[0] = True
    try:
        outc = DO.forward(host, IDS, P=Pc, C=Cc, res32=True)
    finally:
        DO.CLOSED[0] = False
    assert float(mx.abs(outc - ref).max()) > 1e-4


def test_gate_scales_the_write(host):
    P, C = DO.build_overlay(_spec({1: 6}), _rows(host), 32, write_scale=1.0)
    x = mx.random.normal((1, 5, 32))
    P["L1.gb"] = mx.array([30.0])
    open_ = DO.overlay_delta(P, C, 1, x)
    P["L1.gb"] = mx.array([-30.0])
    shut = DO.overlay_delta(P, C, 1, x)
    assert float(mx.abs(open_).max()) > 1e-3 and float(mx.abs(shut).max()) < 1e-9


def test_read_init_copies_host_unit_and_write_scale(host):
    rows = _rows(host)
    spec = _spec({2: 4})
    P, C = DO.build_overlay(spec, rows, 32, read_scale=1.1, write_scale=0.01, dup_jitter=0.0)
    wg, wu, wd = rows(2, spec[2]["units"])
    assert np.allclose(np.array(P["L2.wg"]), wg) and np.allclose(np.array(P["L2.wu"]), 1.1 * wu, atol=1e-6)
    assert np.allclose(np.array(P["L2.wd"]), 0.01 * wd, atol=1e-8)


def test_duplicates_get_jittered_read_rows(host):
    spec = _spec({1: 80}, I=64)  # 80 > 64 units: the ranking repeats
    assert spec[1]["copy"].max() == 1
    P, _ = DO.build_overlay(spec, _rows(host), 32, dup_jitter=0.1)
    wg = np.array(P["L1.wg"])
    u = spec[1]["units"]
    i, j = 0, int(np.nonzero(u == u[0])[0][1])
    assert not np.allclose(wg[i], wg[j]) and np.allclose(np.linalg.norm(wg[i]), np.linalg.norm(wg[j]), rtol=0.2)


def test_sink_wiring_gradients_reach_sink_reads_and_box_writes(host):
    """box overlay at layer 1, sink at layer 3 (the last): with a live box write and a near-zero sink write, the
    loss reaches the sink's write rows, and once the sink writes, its read rows; the box write gets gradient too"""
    spec = _spec({1: 6, 3: 6})
    P, C = DO.build_overlay(spec, _rows(host), 32, write_scale=0.0)
    P["L1.wd"] = 0.3 * mx.random.normal(P["L1.wd"].shape)
    tgt = mx.array([IDS[0][1:]])

    def loss(p):
        lg = DO.forward(host, IDS, P=p, C=C, res32=True)
        return DO.gold_nll(lg[:, :-1], tgt).mean()
    g = mx.grad(loss)(P)
    assert float(mx.abs(g["L3.wd"]).max()) > 0 and float(mx.abs(g["L1.wd"]).max()) > 0
    assert float(mx.abs(g["L3.wg"]).max()) == 0  # zero write: no read gradient yet (LoRA-style start)
    P["L3.wd"] = 0.1 * mx.random.normal(P["L3.wd"].shape)
    g = mx.grad(loss)(P)
    assert float(mx.abs(g["L3.wg"]).max()) > 0 and float(mx.abs(g["L3.gw"]).max()) > 0
    assert "L1.gg" in C and "L1.gg" not in P  # the gate's host rows are frozen constants


def test_tile_init_is_exact_host_and_capped(host):
    ref = DO.forward(host, IDS, res32=True)
    T = DO.build_tiles(range(4), 3, 32, 0)
    assert float(mx.abs(DO.forward(host, IDS, tiles=T, res32=True) - ref).max()) == 0.0
    T = {k: (mx.ones_like(v) * 5 if k.endswith("w_out") else v) for k, v in T.items()}
    assert float(mx.abs(DO.forward(host, IDS, tiles=T, res32=True, norm_budget=0.25) - ref).max()) > 0


def test_save_load_install_roundtrip(host, tmp_path):
    P, C = DO.build_overlay(_spec({1: 6, 3: 6}), _rows(host), 32, write_scale=0.5)
    want = DO.forward(host, IDS, P=P, C=C, res32=False)
    DO.save_overlay(tmp_path / "ov", P, C, {"host_checkpoint": "tiny@x"})
    P2, C2, meta = DO.load_overlay(tmp_path / "ov", "tiny@x")
    with pytest.raises(SystemExit):
        DO.load_overlay(tmp_path / "ov", "other@y")
    rec = {}
    with DO.install(host, P2, C2, rec):
        got = host(mx.array(IDS), cache=host.make_cache()).astype(mx.float32)
    assert float(mx.abs(got - want).max()) < 1e-4 and set(rec["gate_open"]) == {"1", "3"}
    assert meta["schema"] == "mlx2.amoeba-overlay.v1"
    plain = host(mx.array(IDS), cache=host.make_cache()).astype(mx.float32)
    assert float(mx.abs(plain - DO.forward(host, IDS, res32=False)).max()) == 0.0


# ------------------------------------------------------------------------------------------- losses / optimizer
def test_kl_topk_exact_at_full_support_and_a_lower_bound():
    g = np.random.default_rng(0)
    lp = g.standard_normal((6, 20)).astype(np.float32) * 2
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    lg = mx.array(g.standard_normal((6, 20)).astype(np.float32))
    full = np.array(DO.kl_full(mx.array(lp), lg))
    k20 = np.array(DO.kl_topk(lg, *(mx.array(x) for x in DO.topk_pack(lp, 20))))
    k5 = np.array(DO.kl_topk(lg, *(mx.array(x) for x in DO.topk_pack(lp, 5))))
    assert np.allclose(full, k20, atol=1e-4)
    assert (k5 <= full + 1e-5).all() and (k5 >= -1e-6).all()


def test_adam_warmup_and_lr_multipliers():
    opt = DO.Adam(1e-2, {"wd": 0.1}, warmup=4)
    P = {"L1.wd": mx.zeros((3,)), "L1.wg": mx.zeros((3,))}
    G = {"L1.wd": mx.ones((3,)), "L1.wg": mx.ones((3,))}
    P = opt.step(P, G)
    assert opt.lr_now() == pytest.approx(2.5e-3)
    assert float(P["L1.wg"][0]) == pytest.approx(-2.5e-3, rel=1e-3)
    assert float(P["L1.wd"][0]) == pytest.approx(-2.5e-4, rel=1e-3)
    for _ in range(5):
        P = opt.step(P, G)
    assert opt.lr_now() == pytest.approx(1e-2)


def test_ranking_helpers_and_gate_fit():
    mt = np.array([5.0, 1.0, 3.0, 0.1])
    me = np.array([1.0, 1.0, 1.0, 1.0])
    order, ne = DO.rank_units(mt, me, 1.5)
    assert order.tolist()[:2] == [0, 2] and ne == 2
    so = DO.sink_order(order, np.array([2, 0, 1]), np.array([3.0, 1.0, -1.0]))
    assert so.tolist()[:2] == [2, 0] and sorted(so.tolist()) == [0, 1, 2, 3]
    ids, copy = DO.expand_units([3, 1], 5)
    assert ids.tolist() == [3, 1, 3, 1, 3] and copy.tolist() == [0, 0, 1, 1, 2]
    g = np.random.default_rng(1)
    w, b, acc = DO.fit_gate(g.normal(1.5, 1, (200, 4)), g.normal(-1.5, 1, (200, 4)))
    assert acc > 0.9 and (w > 0).all()
    layers, _ = DO.select_layers(np.ones((6, 4)) * np.arange(6)[:, None] + 2, np.ones((6, 4)), 2, 2, 1.5, {1, 2, 3})
    assert layers == [1, 3]


# ------------------------------------------------------------------------------------------- corpus disjointness
def test_corpus_filter_drops_group_text_and_span_overlap():
    ev = [{"text": "The quick brown fox jumps over the lazy dog near the river bank today", "group": "arxiv/1"}]
    fit = [{"id": "a", "group": "arxiv/1#x", "text": "unrelated text entirely about other matters here"},
           {"id": "b", "group": "g2", "text": "THE QUICK brown fox jumps over the lazy dog near the river bank today"},
           {"id": "c", "group": "g3", "text": "prefix words then the quick brown fox jumps over the lazy dog near "
                                                   "the river and more"},
           {"id": "d", "group": "g4", "text": "a clean physics document about plasma turbulence and cascades"},
           {"id": "e", "group": "g5", "text": "a clean physics document about plasma turbulence and cascades"}]
    kept, why, shs, sha = OC.filter_rows(fit, ev, {"arxiv/1"}, 32)
    assert [r["id"] for r in kept] == ["d"]
    assert dict(why) == {"group": 1, "text": 1, "span": 1, "duplicate": 1}
    assert OC.recheck(kept, {"arxiv/1"}, shs, sha, 32) == {"group": 0, "text": 0, "span": 0}
    assert OC.recheck(fit[1:2], {"arxiv/1"}, shs, sha, 32)["text"] == 1


# ------------------------------------------------------------------------------------------- end to end (tiny)
class _Tok:
    eos_token_ids = [0]
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [5 + (ord(c) % 120) for c in text]

    def decode(self, ids):
        return "x" * len(ids)


def _fixture(tmp_path, n_fit=40, n_held=30, n_res=10, L=30):
    g = np.random.default_rng(0)
    words = ["quark", "boson", "field", "spin", "mass", "wave", "energy", "photon", "lattice", "phase"]
    rows = []
    for i in range(n_fit + n_held):
        txt = " ".join(g.choice(words, 12))
        rows.append({"id": f"tp{i:05d}", "topics": ["sci.physics"], "split": "fit" if i < n_fit else "held",
                     "text": txt, "group": f"g{i}"})
    rows.append({"id": "tpx", "topics": ["hum.history"], "split": "held", "text": "kings " * 10, "group": "gh"})
    probes = tmp_path / "probes.jsonl"
    probes.write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = [r for r in rows if r["split"] == "held" and r["topics"] == ["sci.physics"]][-n_res:]
    man = {"reserved_final_test": {"probe_ids": [r["id"] for r in res], "groups": [r["group"] for r in res]},
           "selection": {"val": []}}
    mp = tmp_path / "man.json"
    mp.write_text(json.dumps(man))
    return probes, mp


def _loader(donor_seed=11):
    models = {"host": _tiny(3), "donor": _tiny(donor_seed)}

    def load(path):
        k = Path(path).name
        m = models[k]
        return {"ad": None, "model": m, "layers": DO.text_layers(m), "tok": _Tok(), "ck": f"{k}@{k[0] * 8}"}
    return load


COMMON = ["--topic-tokens", "24", "--n-train", "8", "--n-val", "4", "--n-test", "6", "--eng-retain", "3",
          "--eng-val", "2", "--eng-test", "2", "--eng-tokens", "24", "--topk", "16", "--budget-vecs", "60",
          "--n-box-layers", "1", "--n-sink-layers", "1", "--gate-units", "4", "--win-margin", "0.0",
          "--resp-prefix", "12", "--resp-tokens", "6", "--epochs", "1", "--batch", "4", "--evals-per-epoch", "1",
          "--core-units", "4", "--core-rank", "2", "--cpu", "--budget-s", "100000"]


def test_end_to_end_arms_stage_tiny(tmp_path):
    probes, mp = _fixture(tmp_path)
    out = tmp_path / "out"
    rc = DO.main(["--stage", "arms", "--host", "host", "--donor", "donor", "--out", str(out), "--probes", str(probes),
                  "--split-manifest", str(mp), "--arms", "ov_sink_win,ov_win,sink_win,ov_sink_rand,ov_sink_sft,"
                  "tile_sft,ov_sink_core"] + COMMON, loader=_loader())
    assert rc == 0
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "complete" and run["host_checkpoint"] == "host@hhhhhhhh"
    s = json.loads((out / "summary.json").read_text())
    assert set(s["arms"]) >= {"ov_sink_win", "tile_sft", "ov_sink_core", "ov_sink_sft"}
    res = set(json.loads(mp.read_text())["reserved_final_test"]["probe_ids"])
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert {r["probe_id"] for r in rows if r["set"] == "held"} == res
    assert not ({r["probe_id"] for r in rows if r["set"] == "dev"} & res)
    for n, e in s["arms"].items():
        pf = e["preflight"]
        assert all(v == 0.0 for k, v in pf.items() if k != "init_vs_host_max_abs_logit"), (n, pf)
        assert e["selection_rule"].startswith("lowest val_ce")
        if n != "tile_sft":
            assert e["closed_gate"] is not None
    assert s["arms"]["ov_sink_win"]["closed_gate"]["npb_closed_minus_host_maxabs"] == 0.0
    # equal trainable budget within the gate/one-unit slack; the core arm spends extra on its delta
    pw = s["arms"]["ov_sink_win"]["trainable_params"]
    assert abs(s["arms"]["ov_win"]["trainable_params"] - pw) < 3 * 32 + 20
    assert s["arms"]["ov_sink_core"]["trainable_params"] > pw
    assert "poor_rule" in s and set(s["poor_rule"]) >= {"rule", "poor", "best_overlay_arm"}
    assert s["poor_rule"]["best_overlay_arm"] != "ov_sink_core"
    assert (out / "overlays/ov_sink_win/overlay.json").exists()
    assert run["donor_unload"]["active_gb_after"] <= run["donor_unload"]["active_gb_before"]
    # the bar runs first, so a budget cut never drops it
    assert list(s["arms"])[0] == "tile_sft"
    # tile and overlay budgets agree to within the tile's own per-layer gate vectors (the lab's tile)
    pt = s["arms"]["tile_sft"]["trainable_params"]
    assert abs((pt - 4 * (32 + 1)) - (pw - 2 * 5)) <= 3 * 32 * 2
    e = s["arms"]["ov_sink_win"]
    assert e["served_path"]["texts"] > 0 and set(e["closed_gate"]["gate_mean_english_test"])
    assert not e["truncated"]


def test_end_to_end_scale_stage_tiny(tmp_path):
    probes, mp = _fixture(tmp_path)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    g = np.random.default_rng(5)
    with open(cdir / "chunks.jsonl", "w") as f:
        for i in range(24):
            f.write(json.dumps({"id": f"c{i}", "group": f"fitg{i}", "ids": [int(x) for x in g.integers(5, 120, 16)]})
                    + "\n")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    (cdir / "manifest.json").write_text(json.dumps({"chunks_sha256": sha(cdir / "chunks.jsonl"),
                                                    "probes_sha256": sha(probes), "split_manifest_sha256": sha(mp)}))
    out = tmp_path / "out"
    rc = DO.main(["--stage", "scale", "--host", "host", "--donor", "donor", "--out", str(out), "--probes",
                  str(probes), "--split-manifest", str(mp), "--corpus", str(cdir), "--scale-mults", "1,2",
                  "--scale-kl-mults", "1", "--scale-batch", "2", "--max-steps", "3", "--scale-evals", "2",
                  "--train-probe-chunks", "4", "--disk-snapshot-params", "0"] + COMMON, loader=_loader())
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert set(s["arms"]) == {"scale_x1_ce_gold", "scale_x2_ce_gold", "scale_x1_kl_win_live"}
    p1, p2 = (s["arms"][f"scale_x{m}_ce_gold"]["trainable_params"] for m in (1, 2))
    assert p2 > 1.9 * p1
    assert [st["pair"] for st in s["scaling"]["steps"]] == ["scale_x2_ce_gold vs scale_x1_ce_gold"]
    assert s["arms"]["scale_x2_ce_gold"]["write_lr_mult"] == pytest.approx(1 / np.sqrt(2))
    run = json.loads((out / "run.json").read_text())
    assert run["scale_feasibility"]["x1000"]["params"] > 900 * run["scale_feasibility"]["x1"]["params"]
    assert set(run["scale_probe_memory"]) == {"x1_kl_win_live", "x1_ce_gold", "x2_ce_gold"}
    assert run["donor_unload"]  # unloaded after the donor-wins size
    curves = [json.loads(x) for x in (out / "curves.jsonl").read_text().splitlines()]
    assert all("train_probe_ce_npt" in c and "dev_ce" in c for c in curves)
    # a corpus chunk in a held group is refused
    bad = tmp_path / "corpus2"
    bad.mkdir()
    (bad / "chunks.jsonl").write_text(json.dumps({"id": "z", "group": "g45", "ids": list(range(5, 21))}) + "\n")
    (bad / "manifest.json").write_text(json.dumps({"chunks_sha256": sha(bad / "chunks.jsonl"),
                                                   "probes_sha256": sha(probes), "split_manifest_sha256": sha(mp)}))
    with pytest.raises(SystemExit):
        DO.main(["--stage", "scale", "--host", "host", "--donor", "donor", "--out", str(tmp_path / "o2"), "--probes",
                 str(probes), "--split-manifest", str(mp), "--corpus", str(bad), "--max-steps", "1"] + COMMON,
                loader=_loader())


def test_donor_equal_to_host_gives_no_wins_and_zero_kl(tmp_path):
    """alignment: with the donor = the host, no position is donor-better at a positive margin and the stored
    donor top-k targets give (near) zero KL to the untouched host"""
    probes, mp = _fixture(tmp_path)
    out = tmp_path / "out"
    load = _loader()
    host_m = load("host")["model"]

    def same(path):
        d = load(path)
        if Path(path).name == "donor":
            d = dict(d, model=host_m, layers=DO.text_layers(host_m))
        return d
    args = list(COMMON)
    args[args.index("--win-margin") + 1] = "0.01"
    rc = DO.main(["--stage", "arms", "--host", "host", "--donor", "donor", "--out", str(out), "--probes", str(probes),
                  "--split-manifest", str(mp), "--arms", "ov_sink_all"] + args, loader=same)
    assert rc == 0
    run = json.loads((out / "run.json").read_text())
    assert run["selection"]["train_win_positions"] == 0
    assert run["selection"]["held_host_npb"] == pytest.approx(run["selection"]["held_donor_npb"], abs=1e-3)
    curve = [json.loads(x) for x in (out / "curves.jsonl").read_text().splitlines()]
    first = [c for c in curve if c["arm"] == "ov_sink_all" and c["train_loss"] is not None][0]
    assert first["train_loss"] < 1e-3  # KL(host || host + near-no-op overlay): targets and positions line up
