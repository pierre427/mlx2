"""CPU end-to-end tests of the two P0 drivers and the battery's opt-in per-document rows, with a fake host and fake
evals on the toy partition of test_amoeba_fn_partition_groups_cpu (no model, no GPU): scripts/amoeba_fn_suff.py,
scripts/amoeba_fn_pareto.py, scripts/amoeba_box_battery.py --perdoc (docs/research/2026-11-20-amoeba-next-
experiments.md 1, 4, 7.1-7.5). The fake dCE of a document is linear in the closed units, so every composition and
control can be checked against the masks the drivers claim to score."""
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
import test_amoeba_fn_partition_groups_cpu as P  # noqa: E402

HOST_L = 6
CALLS = []


class Ctl:
    def __init__(self):
        self.reset()

    def reset(self):
        self.core = self.moe_mean = self.moe_thresh = self.moe_open = self.stats = None
        self.sink_layers = 0


class FakeHost:
    def __init__(self, model, log=None, moe_units=None):
        self.L = HOST_L
        self.widths = [P.N] * HOST_L
        self.fingerprint = "f" * 64
        self.checkpoint = "toy@ffffffff"
        self.ctl = Ctl()
        self.grad_mode = "layer_ref"
        self.sites = [SimpleNamespace(layer=l, is_moe=True, units="neuron", E=P.E, F=P.F, n=P.N) for l in P.LAYERS]

    def encode(self, text):
        return [ord(c) for c in text]

    def check_paths(self, ids):
        pass

    def check_directional_grad(self, ids, **k):
        return True

    def check_layer_reference(self, ids, **k):
        return {"ok": True}


def _weights(name):
    return np.random.default_rng(sum(map(ord, name))).random((HOST_L, P.N))


def _english_cost(host, split, clean, cfg, part="held", domains=None, pooled=False, perdoc=None):
    cfg()
    c = host.ctl
    m = np.asarray(c.core, np.float64)
    mean, thr = c.moe_mean is not None, c.moe_thresh
    CALLS.append({"sets": sorted(split), "mask": m.copy(), "mean": mean, "thresh": thr})
    r = {}
    for k, v in split.items():
        base = float(((1.0 - m) * _weights(k)).sum() / 100.0)
        if mean:
            base *= 0.3
        if thr:
            base = 0.01 * sum(thr.values())
        seqs = v[part]
        n = np.array([len(q) - 2 for q in seqs])
        dv = base * (1 + 0.1 * ((np.arange(len(seqs)) % 5) - 2))
        cl = n * 1.0 + np.array([sum(q) % 7 for q in seqs]) * 0.01
        clean.setdefault(k, float(cl.sum() / n.sum()))
        r[k] = float(dv.mean())
        if perdoc is not None:
            perdoc[k] = {"ce": cl + dv * n, "clean_ce": cl, "n": n}
    host.ctl.reset()
    r["mean_d_ce"] = float(np.mean(list(r.values())))
    return r


def _install(monkeypatch):
    pkg = {}
    for name in ("research", "research.amoeba", "research.amoeba.carve"):
        pkg[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, pkg[name])
    D = types.ModuleType("research.amoeba.carve.__main__")
    D.from_np = lambda m, host: m
    E = types.ModuleType("research.amoeba.carve.evals")

    def configure(host, m, _x, L):
        def f():
            host.ctl.reset()
            host.ctl.core = m
        return f
    E.configure = configure
    E.english_cost = _english_cost
    O = types.ModuleType("research.amoeba.carve.ops")
    O.footprint_watchdog = lambda *a, **k: None
    O.setup_mlx = lambda *a, **k: None
    O.prepare_wiring = lambda spec: {}
    O.wire_weights = lambda spec, log, memory_gb=None: {"wired": False, "spec": spec}
    O.wired_arg = lambda spec: spec
    O.git_rev = lambda: "toyrev"
    O.peak_gb = lambda: 0.0
    H = types.ModuleType("research.amoeba.carve.host")
    H.Host = FakeHost
    MS = types.ModuleType("research.amoeba.carve.moestats")
    MS.neuron_sites = lambda host: host.sites

    def capture(host, seqs, bs=8, quantiles=(), log=None, **k):
        g = np.random.default_rng(len(seqs))
        return {s.layer: {"mu": np.full(P.N, 0.1, np.float32), "wh": g.random(P.N), "count": g.random(P.E),
                          "reap_e": g.random(P.E), "tokens": 10, "habs_q": {float(q): float(q) for q in quantiles}}
                for s in host.sites}
    MS.capture = capture
    MS.down_col_norms = lambda host, chunk=32: {s.layer: np.ones(P.N, np.float32) for s in host.sites}
    FI = types.ModuleType("research.amoeba.carve.fisher")

    def accumulate(host, seqs, labels="ce", seed=0, log=None, budget_s=None, layers=None, **k):
        g = np.random.default_rng(1 if labels == "ce" else 2)
        return {"sum": {l: g.normal(size=P.N) for l in (layers or range(HOST_L))}, "seqs": len(seqs), "s": 0.0,
                "labels": labels}
    FI.accumulate = accumulate
    for m, mod in (("__main__", D), ("evals", E), ("ops", O), ("host", H), ("moestats", MS), ("fisher", FI)):
        setattr(pkg["research.amoeba.carve"], m, mod)
        monkeypatch.setitem(sys.modules, f"research.amoeba.carve.{m}", mod)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corpora(root, n=10):
    (root / "held").mkdir()
    for b in list(P.BOXES) + ["eng"]:
        rows = [{"split": "held", "id": f"{b}{i}", "group": f"{b}-g{i // 2}", "source": "web" if i % 2 else "wiki",
                 "text": f"{b} held text number {i} with words"} for i in range(n)]
        rows += [{"split": "fit", "id": f"{b}f{i}", "text": f"{b} fit row {i} words words"} for i in range(n)]
        rows += [{"split": "val", "id": f"{b}v{i}", "group": f"{b}-v{i}", "text": f"{b} val row {i} words words"}
                 for i in range(n)]
        (root / "held" / f"{b}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    g = [{"question": f"What is {i} plus {i}? " + "Think it through carefully. " * 6, "answer": f"#### {2 * i}"}
         for i in range(n)]
    (root / "held" / "gsm8k.jsonl").write_text("\n".join(json.dumps(r) for r in g) + "\n")


@pytest.fixture
def toy(tmp_path):
    P._toy(tmp_path)
    _corpora(tmp_path)
    g = P._groups(tmp_path, {"abc": ["a", "b", "c"], "big": ["a", "b", "c", "d", "e"], "fghi": ["f", "g", "h", "i"]})
    r = P._run(tmp_path, "fam", "--groups", g, "--emit-unassigned")
    assert r.returncode == 0, r.stderr
    return tmp_path


def _held():
    return [x for b in list(P.BOXES) + ["eng"] for x in ("--held", f"{b}=held/{b}.jsonl@8")]


def _battery(monkeypatch, root, out, *extra, docs=3):
    _install(monkeypatch)
    mod = _load("amoeba_box_battery")
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "fam", "--out", out, *_held(),
                                      "--docs", str(docs), *extra])
    CALLS.clear()
    mod.main()


def _rows(path):
    out = []
    for x in Path(path).read_text().splitlines():
        r = json.loads(x)
        r.pop("wall_s", None)
        out.append(r)
    return out


# ----------------------------------------------------------------------------- battery --perdoc

def test_battery_perdoc_is_a_pure_addition(toy, monkeypatch):
    _battery(monkeypatch, toy, "b0")
    _battery(monkeypatch, toy, "b1", "--perdoc")
    assert _rows(toy / "b0/battery.jsonl") == _rows(toy / "b1/battery.jsonl")
    assert (toy / "b0/battery.md").read_text() == (toy / "b1/battery.md").read_text()
    assert (toy / "b0/path_offset.json").read_text() == (toy / "b1/path_offset.json").read_text()
    assert not (toy / "b0/perdoc.jsonl").exists()
    import amoeba_battery_stats as ST
    pdoc = ST.load_perdoc(toy / "b1/perdoc.jsonl")
    for r in _rows(toy / "b1/battery.jsonl"):
        b = r["box"]
        for arm in ("without", "random", "core_plus"):
            for s, v in r[arm].items():
                assert ST.doc_dce(pdoc[(b, arm, s)]).mean() == pytest.approx(v, abs=1e-12)
    docs = json.loads((toy / "b1/perdoc.docs.json").read_text())
    assert docs["a"]["ids"] == [x for x in docs["a"]["ids"]] and len(docs["a"]["groups"]) == 3
    z = np.load(toy / "b1/battery.perdoc.npz")
    assert "a|core_plus|a|dce" in z.files
    assert json.loads((toy / "b1/battery.meta.json").read_text())["perdoc"]["rows"] == len(pdoc)
    assert "perdoc" not in json.loads((toy / "b0/battery.meta.json").read_text())


# ----------------------------------------------------------------------------- core-without-box control

def _suff(monkeypatch, root, out, *extra):
    _install(monkeypatch)
    mod = _load("amoeba_fn_suff")
    monkeypatch.chdir(root)
    CALLS.clear()
    return mod, mod.main(["--model", "m", "--partition", "fam", "--battery", "bat", "--out", out, "--english", "eng",
                          "--docs", "3", "--mu-sets", "eng", "--mu-docs", "2", *extra])


def _masks(sets, mean=False):
    return [c["mask"][P.LAYERS] > 0 for c in CALLS if c["sets"] == sorted(sets) and c["mean"] == mean]


def test_suff_compositions_controls_and_summary(toy, monkeypatch):
    _battery(monkeypatch, toy, "bat")
    mod, rc = _suff(monkeypatch, toy, "s")
    assert rc == 0
    z = np.load(toy / "fam/tiles.npz")
    core, a = z["core"], z["box:a"] | z["pair:a__b"]
    alone = core | z["group:abc"] | z["group:big"]
    plus = alone | a
    m = _masks(["a", "eng"])
    assert any(np.array_equal(x, plus) for x in m) and any(np.array_equal(x, alone) for x in m)
    own = plus & ~alone
    rnd = [x & ~alone for x in m if not np.array_equal(x, plus) and not np.array_equal(x, alone)
           and not np.array_equal(x, core)]
    assert len(rnd) >= 3                                                 # rand, rand_inexp, rand2
    for r in rnd:
        assert (r.sum(1) == own.sum(1)).all() and not (r & plus & ~alone).any()      # equal size, outside core_plus
    assert any(all((r[l].reshape(P.E, P.F).sum(1) == own[l].reshape(P.E, P.F).sum(1)).all() for l in range(3))
               for r in rnd)                                              # the same-experts control
    assert any(np.array_equal(x, core) for x in m)                         # core_only (no family tiles)
    assert any(np.array_equal(x, ~z["box:a"]) for x in _masks(["a"]))      # without, rescored per document
    assert any(np.array_equal(x, ~z["box:a"]) for x in _masks(["a"], mean=True))       # without_mean
    box = z["box:a"]
    rie = [~x for x in _masks(["a"]) if not np.array_equal(x, ~box) and (~x).sum() == box.sum()]
    assert any(all((r[l].reshape(P.E, P.F).sum(1) == box[l].reshape(P.E, P.F).sum(1)).all() for l in range(3))
               and not (r & box).any() for r in rie)                       # random_inexp: same experts, not the box
    summ = json.loads((toy / "s/summary.json").read_text())
    ck = summ["checks"]
    assert all(ck["path_offset_exact"].values()) and all(ck["do_nothing_bitwise"].values())
    assert ck["empty_mean_bitwise"] and ck["repeat_cell_bitwise"]
    assert set(summ["boxes"]) == set(P.BOXES) and summ["status"] == "finished"
    ra = summ["boxes"]["a"]
    assert ra["reproduces_battery_core_plus"] < 1e-12                      # same masks, docs and order as the battery
    assert ra["served"] in ("yes", "vacuous", "no") and "mean_ablation" in ra
    rep = json.loads((toy / "s/meta.json").read_text())["repeat_boxes_run"]
    assert len(rep) == 5 and all("repeat" in summ["boxes"][b] for b in rep)
    assert ra["suff_gain"]["mean"] == pytest.approx(ra["core_alone_own"] - ra["core_plus_own"], abs=1e-12)
    assert ra["rand_gain"]["fill"] is not None and "size_matched" in ra["rand_gain_inexp"]
    assert "necessity_net_inexp" in ra and summ["decisions"]["checks_ok"]
    meta = json.loads((toy / "s/meta.json").read_text())
    assert meta["checkpoint"] == "toy@ffffffff" and meta["mlx2_source_revision"] == "toyrev"
    assert summ["floor"]["doc_resample_n"] == 5 and summ["decisions"]["topic_served_passes"] == 0      # toy: all fail
    docs = json.loads((toy / "s/perdoc.docs.json").read_text())
    assert not set(docs[rep[0]]["ids"]) & set(docs[rep[0] + "#rep"]["ids"])     # rows 81-160: disjoint
    assert not set(docs["eng"]["ids"]) & set(docs["eng#rep"]["ids"])
    assert (toy / "s/summary.md").exists() and (toy / "s/battery.perdoc.npz").exists()
    n = len(CALLS)
    _suff(monkeypatch, toy, "s")                                           # resumed: every row reused
    assert len(CALLS) <= 1 < n                                              # only the end-of-run repeat cell
    with pytest.raises(SystemExit, match="other settings"):
        _suff(monkeypatch, toy, "s", "--subset", "a,b")


def test_suff_budget_truncates_and_still_summarizes(toy, monkeypatch):
    _battery(monkeypatch, toy, "bat")
    _, rc = _suff(monkeypatch, toy, "t", "--budget-min", "0")
    assert rc == 0
    meta = json.loads((toy / "t/meta.json").read_text())
    assert meta["status"] == "truncated" and meta["truncated"].startswith("--budget-min")
    assert json.loads((toy / "t/summary.json").read_text())["status"] == "truncated"


def test_suff_refuses_another_battery(toy, monkeypatch):
    _battery(monkeypatch, toy, "bat")
    with pytest.raises(SystemExit, match="docs"):
        _suff(monkeypatch, toy, "x", "--docs", "4")


# ----------------------------------------------------------------------------- Pareto curve

def _pareto(monkeypatch, root, out, *extra):
    _install(monkeypatch)
    mod = _load("amoeba_fn_pareto")
    sys.modules["amoeba_fn_pareto"] = mod

    def capitals(host, items, cfg_fn=None, cap=16):
        if cfg_fn is None:
            return [1] * len(items)
        cfg_fn()
        sh = 1.0 - float(np.asarray(host.ctl.core)[P.LAYERS].mean())
        host.ctl.reset()
        k = int(round(len(items) * max(0.0, 1.0 - 1.5 * sh)))
        return [1] * k + [0] * (len(items) - k)
    monkeypatch.setattr(mod, "capitals", capitals)
    monkeypatch.chdir(root)
    CALLS.clear()
    meta = {"fingerprint": "f" * 64, "held": {"english": {"spec": "english=held/eng.jsonl@8"},
                                               "deu": {"spec": "deu=held/a.jsonl@8"},
                                               "soc.law": {"spec": "soc.law=held/b.jsonl@8"}}}
    (root / "pbat").mkdir(exist_ok=True)
    (root / "pbat/battery.meta.json").write_text(json.dumps(meta))
    return mod, mod.main(["--model", "m", "--partition", "fam", "--joint-partition", "fam", "--battery", "pbat",
                          "--gsm8k", "held/gsm8k.jsonl", "--out", out, "--docs", "3", "--stats-docs", "2",
                          "--mu-sets", "english", "--mu-docs", "2", "--attrib-seqs", "2", *extra])


def test_pareto_grid_controls_and_summary(toy, monkeypatch):
    mod, rc = _pareto(monkeypatch, toy, "p")
    assert rc == 0
    cells = {r["cell"]: r for r in (json.loads(x) for x in (toy / "p/cells.jsonl").read_text().splitlines())}
    for o in ("O-rand.s0", "O-rand.s1", "O-rand.s2", "O-fire", "O-part", "O-reap-n", "O-reap-e", "O-freq-e",
              "O-causal0", "O-fisher"):
        for f in ("0.1", "0.2", "0.3", "0.4"):
            assert f"{o}@{f}" in cells
    for o in ("O-rand.s0", "O-fire", "O-part", "O-reap-n", "O-reap-e", "O-freq-e", "O-causal0", "O-fisher"):
        assert f"{o}@0.2|mean" in cells
    assert "O-rand.s1@0.2|mean" not in cells
    assert all(f"TEAL@{f}" in cells for f in ("0.1", "0.2", "0.3", "0.4"))
    assert any("|D-tenp" in k for k in cells) and any("|D-ends" in k for k in cells)
    assert "O-rand.s0@0.2|repeat" in cells and "O-causal0@0.2|repeat" in cells
    held = [k for k in cells if k.startswith("held:")]
    assert len(held) == 3 and all("|" in k and "@" in k for k in held)        # once, keyed by content
    hv = json.loads((toy / "p/held_verdict.json").read_text())
    assert {hv[r]["cell"] for r in ("choice", "O-fire", "O-rand")} == set(held)
    assert len(cells["O-fire@0.2"]["closed_share_layers"]) == 3              # per-layer receipt
    r = cells["O-rand.s0@0.2"]
    assert r["closed_units"] == round(0.2 * 32) * 3                       # layer-matched
    assert cells["O-reap-e@0.1"]["closed_units"] % P.F == 0                # whole experts
    summ = json.loads((toy / "p/summary.json").read_text())
    ck = summ["checks"]
    assert all(ck["do_nothing_bitwise"].values()) and ck["do_nothing_capitals_equal"]
    assert ck["repeat_cell_bitwise"] and ck["first_cell_repeat_bitwise"]
    d = summ["decisions"]
    assert {"h4", "grain", "mean_compensation", "held_verdict", "floor", "checks_ok"} <= set(d)
    assert d["checks_ok"] and set(d["floor"]["english_doc_resample"]) == {"O-rand.s0@0.2", "O-causal0@0.2"}
    assert d["h4"]["cf_fire_mean_curve"] == summ["orders"]["O-fire"]["cf_mean"]
    assert set(d["held_verdict"]["rows"]) == {"choice", "O-fire", "O-rand"}
    assert d["grain"]["best_neuron"] in ("O-fire", "O-causal0", "O-fisher", "O-reap-n")      # never O-part
    meta = json.loads((toy / "p/meta.json").read_text())
    assert meta["checkpoint"] == "toy@ffffffff" and len(meta["attempts"]) == 1 and meta["attribution"]["gate_ok"]
    assert d["h4"]["holds"] in (True, False) and summ["choice"]["order"] not in ("O-rand.s0", "TEAL")
    # the fake's English cost is linear in closed units with mean ablation at 0.3x: a candidate at 20%
    assert d["mean_compensation"]["english_gain"]["mean"] > 0
    assert all(c["english_ub95"] >= c["english_mean"] for c in summ["orders"]["O-fire"]["cells"])
    assert summ["teal_reference"] and summ["status"] == "finished"
    n = len(CALLS)
    _pareto(monkeypatch, toy, "p")                                         # resumed: nothing rescored
    assert len(CALLS) == 0 < n
    assert len(json.loads((toy / "p/meta.json").read_text())["attempts"]) == 2


def test_pareto_without_causal_and_cliff_flag(toy, monkeypatch):
    mod, rc = _pareto(monkeypatch, toy, "q", "--no-causal")
    assert rc == 0
    summ = json.loads((toy / "q/summary.json").read_text())
    assert "O-causal0" not in summ["orders"] and summ["decisions"]["h4"]["holds"] is None
    # the fake capitals probe drops 1.5 x the closed share: 40% closed loses 60 points (> 10) -> cliff when cheap
    flagged = [c for o in summ["orders"].values() for c in o["cells"] if c["flagged"]]
    for c in flagged:
        assert c["english_mean"] <= 0.05


def test_pareto_budget_stop_still_writes_a_held_verdict(toy, monkeypatch):
    _pareto(monkeypatch, toy, "pb", "--budget-min", "0")
    meta = json.loads((toy / "pb/meta.json").read_text())
    assert meta["status"] == "truncated" and "note" in meta["checks"]["held_verdict"]
    assert (toy / "pb/summary.json").exists()


def test_suff_missing_mask_receipt_never_counts_the_inexp_control(toy, monkeypatch):
    _battery(monkeypatch, toy, "bat")
    mod, _ = _suff(monkeypatch, toy, "s2")
    (toy / "s2/suff_info.jsonl").write_text("")                              # as if killed before the receipt
    summ = mod.summarize(toy / "s2")
    assert all(not r["rand_gain_inexp"]["size_matched"] for r in summ["boxes"].values())
