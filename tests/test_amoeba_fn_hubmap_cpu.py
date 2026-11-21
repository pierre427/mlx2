"""CPU test of scripts/amoeba_fn_hubmap.py on synthetic captures and maps with planted units: a concept hub, a
generic always-on unit, a topic-specific unit and a forum-register artefact."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E, F, S = 512, 2, 2
N = E * F
SITES = [{"layer": s, "class": "moe", "kind": "qwen_next", "n": N, "E": E, "F": F} for s in range(S)]
IDENT = {"capture_rev": "test", "tokens": 256, "split": "fit", "seed": 0, "source_rev": "test"}
HUB = [5, 77, 300, 901]
REG = [11, 88, 400, 950]
TOPIC = [21, 600]
GENERIC = list(range(1024 - 40, 1024))
CONTEXTS = {  # name -> (sources); the first source of a pair is the forum one where there is one
    "sci.physics": ["qa", "academic"], "math.algebra": ["qa", "abstract"], "prog.python": ["qa", "code"],
    "agent.tool_use": ["toucan", "glaive"], "dialog": ["smoltalk", "wildchat"], "hum.history": ["qa", "book"],
    "soc.law": ["stackexchange", "wikipedia"], "sec.vuln": ["qa", "web"], "cmn": ["fineweb2-cmn", "wikipedia-zh"],
    "fra": ["fineweb2-fra", "wikipedia-fra"], "life.cooking": ["stackexchange", "fineweb-edu"],
    "sci.biology": ["stackexchange", "wikipedia"], "swe.git": ["issues", "qa"],      # swe: forum-only family
}
FORUM = {"qa", "stackexchange"}


def _capture(d, groups, docs, rng, base, plant):
    """docs[g] per-document means [n_docs, U] = base + noise + plant(g)."""
    d.mkdir(parents=True)
    arr, n = {}, []
    for g, grp in enumerate(groups):
        k = docs
        n.append(k)
        for s in range(S):
            mu = base[s] + plant(grp, s)
            x = np.abs(mu[None] + rng.normal(0, 0.1, (k, N)) * base[s][None])
            arr[f"S1_{g}_{s}"] = x.sum(0).astype(np.float32)
            arr[f"S2_{g}_{s}"] = (x ** 2).sum(0).astype(np.float32)
            arr[f"R_{g}_{s}"] = rng.random(E).astype(np.float32) * k
    arr["n"] = np.array(n)
    arr["groups"] = np.array(groups)
    np.savez(d / "sums.npz", **arr)
    (d / "report.json").write_text(json.dumps({
        "schema": "mlx2.amoeba-firing-sums.v1", "checkpoint": "toy@abc", "fingerprint": "abc", "corpus": str(d),
        "identity": IDENT, "groups": groups, "n": n, "sites": SITES}))


def _compare(target, baseline, out):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_firing_compare.py"), "--target", str(target),
                        "--baseline", str(baseline), "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr


def test_hubmap_classifies_planted_units(tmp_path):
    rng = np.random.default_rng(1)
    base = [np.full(N, 1.0) for _ in range(S)]
    for s in range(S):
        base[s][GENERIC] = 50.0                                    # always-on: high English drive
    cap = tmp_path / "cap"
    _capture(cap / "eng", ["fineweb-edu:0", "fineweb-edu:1", "wikipedia-eng:0", "wikipedia-eng:1"], 40, rng, base,
             lambda g, s: 0.0)
    _capture(cap / "mix10", ["cmn:0", "cmn:1", "fra:0", "fra:1"], 40, rng, [b * 0.5 for b in base], lambda g, s: 0.0)
    maps = tmp_path / "maps"
    _compare(cap / "eng", cap / "mix10", maps / "eng")
    boxes = {}
    for name, srcs in CONTEXTS.items():
        def plant(grp, s, name=name):
            src = grp.rsplit(":", 1)[0]
            v = np.zeros(N)
            v[HUB] = 0.6                                           # every context, every source: +6 sd
            if src in FORUM:
                v[REG] = 1.0                                       # only through forum sources
            if name == "sci.physics":
                v[TOPIC] = 0.6
            return v
        groups = [f"{x}:{h}" for x in srcs for h in (0, 1)]
        _capture(cap / name, groups, 40, rng, base, plant)
        _compare(cap / name, cap / "eng", maps / name)
        boxes[name] = {"map": str(maps / name)}
    boxes["lang.eng"] = {"map": str(maps / "eng")}
    part = tmp_path / "part"
    part.mkdir()
    np.savez(tmp_path / "t2.npz", core=np.zeros((S, E), np.uint8))
    (part / "partition.json").write_text(json.dumps({
        "boxes": boxes, "english_box": {"name": "lang.eng"}, "unassigned_file": "unassigned.npz",
        "inputs": {"core_map": str(maps / "eng"), "t2": str(tmp_path / "t2.npz")}}))
    core = np.zeros((S, N), bool)
    core[:, HUB] = True
    box = np.zeros((S, N), bool)
    box[:, TOPIC] = True
    np.savez(part / "tiles.npz", layers=np.arange(S), core=core, **{"box:sci.physics": box})
    np.savez(part / "unassigned.npz", layers=np.arange(S), unassigned=np.zeros((S, N), bool))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_fn_hubmap.py"), "--partition", str(part),
                        "--out", str(out), "--min-families", "5", "--no-guard", "--k", "2", "--profile-top", "100", "--lang-min", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    cat = np.load(out / "select.npz")["cat"]
    units = lambda xs: [s * N + u for s in range(S) for u in xs]   # noqa: E731
    assert (cat[units(HUB)] == 3).all(), cat[units(HUB)]
    assert (cat[units(REG)] == 3).all(), cat[units(REG)]
    assert (cat[units(TOPIC)] == 1).all(), cat[units(TOPIC)]
    assert (cat[units(GENERIC)] == 4).all()
    others = np.setdiff1d(np.arange(S * N), units(HUB + REG + TOPIC + GENERIC))
    # generic = top 5% English drive per layer, so a few plain units fill the quantile beyond the planted 40
    assert np.isin(cat[others], [0, 4]).all() and (cat == 4).sum() <= int(np.ceil(0.05 * N)) * S, np.bincount(cat[others])
    h = np.load(out / "hubs.npz")
    reg = dict(zip(h["unit"].tolist(), h["register"].tolist()))
    assert all(reg[u] == 0 for u in units(HUB)), [reg[u] for u in units(HUB)]          # concept
    assert all(reg[u] == 2 for u in units(REG)), [reg[u] for u in units(REG)]          # forum register
    assert set(h["unit"][:len(HUB) * S].tolist()) == set(units(HUB))                  # hubs outrank the artefact
    summ = json.loads((out / "summary.json").read_text())
    assert summ["sources_pooled_mismatch"] == []
    assert summ["language"]["hubs_spanning_langs_and_topics"] == len(HUB) * S
    assert summ["tiles"]["partition_core"]["hubs_in"] == len(HUB) * S
    assert (out / "report.md").read_text().startswith("# FN hub units")
    # resumable: a rerun skips every finished stage and rewrites the same report
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_fn_hubmap.py"), "--partition", str(part),
                         "--out", str(out), "--min-families", "5", "--no-guard", "--k", "2", "--profile-top", "100", "--lang-min", "2"],
                        capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert "profiles 1/" not in r2.stdout and "sources 1/" not in r2.stdout
    # the per-source test reproduces the maps' by_target_source counts at the map's min d
    src = json.loads((out / "sources.done.json").read_text())["per_context"]
    assert all(v["match"] for c in src.values() for v in c["sources"].values())
    assert summ["register_families"]["forum_only"] == ["swe"]
    # a changed selection parameter recomputes select (not a stale reuse); a changed breadth d is refused
    r3 = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_fn_hubmap.py"), "--partition", str(part),
                         "--out", str(out), "--min-families", "5", "--no-guard", "--k", "2", "--profile-top", "100",
                         "--lang-min", "2", "--generic-pct", "0.99"], capture_output=True, text=True, cwd=ROOT)
    assert r3.returncode == 0, r3.stderr[-2000:]
    assert json.loads((out / "select.json").read_text())["generic_pct"] == 0.99
    r4 = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_fn_hubmap.py"), "--partition", str(part),
                         "--out", str(out), "--no-guard", "--breadth-d", "1.5"], capture_output=True, text=True, cwd=ROOT)
    assert r4.returncode != 0 and "stamp" in (r4.stdout + r4.stderr)
