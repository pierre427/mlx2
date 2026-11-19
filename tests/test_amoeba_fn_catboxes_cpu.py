"""CPU tests for the FN category re-probe (2026-11-18): amoeba_fn_partition.py --nested and > 64 boxes,
amoeba_box_battery.py --english and nested serving, amoeba_category_box_files.py and amoeba_fn_unclaimed_delta.py."""
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_amoeba_fn_partition_groups_cpu as P  # noqa: E402
import test_amoeba_box_battery_arms_cpu as BT  # noqa: E402

PART = ROOT / "scripts/amoeba_fn_partition.py"
S = len(P.LAYERS)


def _maps(root, claims, names):
    """One map per box from a column plan {col: [boxes]}; English map = col 31; T2 empty."""
    for b in names:
        sel = np.zeros((S, P.N), bool)
        for c, who in claims.items():
            if b in who:
                sel[:, c] = True
        P._map(root / "maps" / b, sel.copy(), sel)
    eng = np.zeros((S, P.N), bool)
    eng[:, 31] = True
    P._map(root / "maps" / "english", eng, eng)
    P._npz_fixed(root / "t2.npz", core=np.zeros((6, P.E), np.uint8))


def _part(root, names, out, *extra):
    cmd = [sys.executable, str(PART)] + [x for b in names for x in ("--box", f"{b}=maps/{b}")] + [
        "--core-map", "maps/english", "--t2", "t2.npz", "--out", out, *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=root)


NEST = ["life", "life.cooking", "life.cooking.x", "life.travel", "life.games", "sci.astronomy"]
NCLAIMS = {
    0: ["life.cooking.x", "life.cooking", "life"],        # nested: x alone -> box:life.cooking.x
    1: ["life"],                                          # the parent alone -> box:life
    2: ["life.cooking", "life"],                          # -> box:life.cooking
    3: ["life.cooking.x", "life.travel", "life"],         # siblings + their parent -> the parent (box:life)
    4: ["life.cooking.x", "life.travel", "sci.astronomy", "life"],   # spans life and sci -> core
    5: ["life.cooking.x", "life.travel", "life.games", "life"],      # 3 siblings + parent -> the parent, not core
    6: ["life.travel"], 7: ["life.games"], 8: ["sci.astronomy"],
    9: ["life.cooking.x", "life.travel", "life.games"],   # 3 siblings, parent silent -> still the LCA life
    10: ["life.cooking.x", "life.travel"],                # 2 siblings, parent silent -> LCA life
    11: ["life.cooking.x", "life.games", "life.cooking"],  # x + games (cooking absorbed) -> LCA life
    12: ["life.cooking.x", "life.games", "life.cooking", "life"],   # different depths -> LCA life
    13: ["life.travel", "life.games", "sci.astronomy"],   # no common ancestor box, inside family tg -> group
    14: ["life.travel", "sci.astronomy"],                 # no common ancestor box -> pair
}
NGROUPS = {"tg": ["life.games", "life.travel", "sci.astronomy"]}


def _cols(m):
    return sorted(set(np.flatnonzero(m.any(0))))


def test_nested_absorption(tmp_path):
    _maps(tmp_path, NCLAIMS, NEST)
    r = _part(tmp_path, NEST, "flat")
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "flat" / "tiles.npz")
    assert z["core"][:, 0].all() and z["core"][:, 3].all()          # without --nested: 3 claims -> core
    assert "nested" not in json.loads((tmp_path / "flat" / "partition.json").read_text())
    r = _part(tmp_path, NEST, "nest", "--nested", "--groups", P._groups(tmp_path, NGROUPS))
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "nest" / "tiles.npz")
    p = json.loads((tmp_path / "nest" / "partition.json").read_text())
    assert _cols(z["box:life.cooking.x"]) == [0]
    assert _cols(z["box:life"]) == [1, 3, 5, 9, 10, 11, 12]
    assert _cols(z["box:life.cooking"]) == [2]
    assert not any(k.startswith("pair:life.cooking.x") for k in z.files)
    assert _cols(z["pair:life.travel__sci.astronomy"]) == [14]
    assert z["core"][:, 4].all() and z["core"][:, 31].all()
    assert _cols(z["group:tg"]) == [13]
    assert (sum(z[k].astype(int) for k in z.files if k != "layers") == 1).all()
    assert p["nested"]["ancestors"]["life.cooking.x"] == ["life", "life.cooking"]
    assert p["boxes"]["life"]["selected_before_nesting"] == 7 * S and p["boxes"]["life"]["selected"] == 7 * S
    assert p["boxes"]["life"]["received_from_subtopics"] == 6 * S
    assert p["boxes"]["life"]["received_lca_selected"] == 3 * S          # cols 3, 5, 12
    assert p["boxes"]["life"]["received_lca_unselected"] == 3 * S        # cols 9, 10, 11
    assert p["boxes"]["life.cooking"]["received_from_subtopics"] == 0
    assert "received_from_subtopics" not in p["boxes"]["life.travel"]
    assert "give it to the parent" in p["nested"]["rule"]
    assert p["boxes"]["life"]["descendants"] == ["life.cooking", "life.cooking.x", "life.games", "life.travel"]
    assert "nested" in p["rule"]


def test_more_than_64_boxes(tmp_path):
    names = [f"b{i:02d}" for i in range(70)]
    claims = {0: ["b00", "b01", "b65"], 1: ["b00", "b01", "b65"], 2: ["b00"], 3: ["b01"], 4: ["b65"],
              5: ["b66", "b67", "b68"], 6: ["b66"], 7: ["b67"],
              8: ["b00", "b01", "b66"],                                # crosses the two groups -> core
              9: ["b03", "b69"], 10: ["b69"]}
    _maps(tmp_path, claims, names)
    g = P._groups(tmp_path, {"lo": ["b00", "b01", "b65"], "hi": ["b66", "b67", "b68"]})
    r = _part(tmp_path, names, "out", "--groups", g, "--emit-unassigned")
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "out" / "tiles.npz")
    assert _cols(z["group:lo"]) == [0, 1] and _cols(z["group:hi"]) == [5]
    assert z["core"][:, 8].all() and _cols(z["pair:b03__b69"]) == [9] and _cols(z["box:b69"]) == [10]
    assert (sum(z[k].astype(int) for k in z.files if k != "layers") == 1).all()
    r = _part(tmp_path, names, "auto", "--groups-auto", "--auto-min-units", str(S), "--no-tiles")
    assert r.returncode == 0, r.stderr
    G = json.loads((tmp_path / "auto" / "partition.json").read_text())["groups"]
    cands = {c["group"]: c for c in G["auto_candidates"]}
    assert cands["b00+b01+b65"]["exact_units"] == 2 * S and cands["b00+b01+b65"]["closed_units"] == 2 * S
    assert cands["b66+b67+b68"]["closed_units"] == S and cands["b00+b01+b66"]["exact_units"] == S


@pytest.fixture
def nested_toy(tmp_path):
    _maps(tmp_path, NCLAIMS, NEST)
    assert _part(tmp_path, NEST, "nest", "--nested", "--groups", P._groups(tmp_path, NGROUPS)).returncode == 0
    (tmp_path / "held").mkdir()
    for b in NEST + ["english", "cmn"]:
        rows = [{"split": "held", "text": f"{b} held text number {i} with words"} for i in range(6)]
        (tmp_path / "held" / f"{b}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def _battery(monkeypatch, root, out, *extra):
    BT._install_fakes(monkeypatch)
    mod = BT._load(ROOT / "scripts/amoeba_box_battery.py")
    held = [x for b in NEST + ["english", "cmn"] for x in ("--held", f"{b}=held/{b}.jsonl@8")]
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "nest", "--out", out, *held,
                                      "--docs", "4", *extra])
    BT.CALLS.clear()
    mod.main()


def test_battery_nested_and_english(nested_toy, monkeypatch):
    _battery(monkeypatch, nested_toy, "bat", "--english", "english", "--arms", "boxes,groups")
    rows = {json.loads(x)["box"]: json.loads(x) for x in (nested_toy / "bat" / "battery.jsonl").read_text().splitlines()}
    assert set(rows) == set(NEST)
    assert rows["life"]["unrelated"] == ["english", "cmn", "sci.astronomy"]     # life.cooking is in life's lineage
    assert rows["life.travel"]["unrelated"] == ["english", "cmn", "life.cooking"]   # a sibling is unrelated
    assert rows["life.cooking.x"]["ancestors"] == ["life", "life.cooking"] and "ancestors" not in rows["life"]
    assert rows["life"]["descendants"] == ["life.cooking", "life.cooking.x", "life.games", "life.travel"]
    assert rows["life.travel"]["groups"] == ["group:tg"] and "groups" not in rows["life.cooking.x"]
    assert "every ancestor box" in rows["life.cooking.x"]["served_composition"]
    assert "whole subtree" in rows["life"]["served_composition"]
    assert rows["life"]["settings"]["english"] == "english" and rows["life"]["settings"]["nested"] is True
    z = np.load(nested_toy / "nest" / "tiles.npz")
    # a leaf: core + its family tile + ancestors' boxes + its box + its pairs (boxes arm = groups arm served)
    served = z["core"] | z["box:life.cooking.x"] | z["box:life.cooking"] | z["box:life"]
    assert any(np.array_equal(x[P.LAYERS] > 0, served) for x in BT._mask_for(["life.cooking.x", "english"]))
    st = z["core"] | z["group:tg"] | z["box:life"] | z["box:life.travel"] | z["pair:life.travel__sci.astronomy"]
    m = BT._mask_for(["life.travel", "english"])
    assert sum(np.array_equal(x[P.LAYERS] > 0, st) for x in m) == 1      # scored once: the groups arm reuses it
    # a parent: its whole subtree (every descendant box, pair and family tile)
    sub = served | z["box:life.games"] | st
    assert any(np.array_equal(x[P.LAYERS] > 0, sub) for x in BT._mask_for(["life", "english"]))
    assert not any(np.array_equal(x[P.LAYERS] > 0, z["core"] | z["box:life"]) for x in BT._mask_for(["life", "english"]))
    g = {json.loads(x)["key"]: json.loads(x) for x in (nested_toy / "bat" / "groups.jsonl").read_text().splitlines()}
    assert g["group:tg"]["sets"][-1] == "english"
    assert g["served:life.travel"]["reused"].startswith("boxes arm")
    assert g["served:life.travel"]["dce"] == rows["life.travel"]["core_plus"]
    md = (nested_toy / "bat" / "battery.md").read_text()
    assert "| life.cooking.x |" in md and "| tg |" in md


def test_not_nested(tmp_path):
    _maps(tmp_path, NCLAIMS, NEST)
    r = _part(tmp_path, NEST, "nn", "--nested", "--not-nested", "life.cooking.x")
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "nn" / "tiles.npz")
    p = json.loads((tmp_path / "nn" / "partition.json").read_text())
    # col 0: x keeps its claim, cooking is no longer x's ancestor (keeps its own), life is absorbed by cooking
    assert _cols(z["pair:life.cooking__life.cooking.x"]) == [0]
    assert "life.cooking.x" not in p["nested"]["ancestors"] and p["nested"]["not_nested"] == ["life.cooking.x"]
    assert "life.cooking.x" not in p["nested"]["descendants"]["life"]
    assert _part(tmp_path, NEST, "bad", "--not-nested", "life").returncode != 0
    assert _part(tmp_path, NEST, "bad2", "--nested", "--not-nested", "zz").returncode != 0


def test_held_split_override(nested_toy, monkeypatch):
    rows = [{"split": "val", "text": f"travel val text number {i} with words"} for i in range(6)]
    (nested_toy / "held" / "life.travel.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(SystemExit, match="held life.travel: no document"):
        _battery(monkeypatch, nested_toy, "x", "--english", "english")
    BT._install_fakes(monkeypatch)
    mod = BT._load(ROOT / "scripts/amoeba_box_battery.py")
    got = mod.read_held_rows(str(nested_toy / "held" / "life.travel.jsonl"), "corpus", "split=val")
    assert len(got) == 6 and all(r["split"] == "val" for r in got)


def test_battery_english_must_not_be_a_box(nested_toy, monkeypatch):
    with pytest.raises(SystemExit, match="also a box name"):
        _battery(monkeypatch, nested_toy, "x", "--english", "life")


def _build(root, rows, boxes, spec, name="build"):
    b = root / name
    b.mkdir()
    with gzip.open(b / "corpus.jsonl.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    sha = hashlib.sha256((b / "corpus.jsonl.gz").read_bytes()).hexdigest()
    (b / "manifest.json").write_text(json.dumps({"schema": "toy", "corpus": "corpus.jsonl.gz", "corpus_sha256": sha,
                                                 "corpus_path": "corpus.jsonl.gz", "meets_spec": spec,
                                                 "boxes": {k: {"docs": v} for k, v in boxes.items()}}))
    return b


def test_category_box_files(tmp_path):
    rows = [{"box": bx, "split": sp, "register": "qa", "text": f"{bx} {sp} {i}"}
            for bx in ("x", "x.y", "z") for i, sp in enumerate(["fit", "fit", "held", "val"])]
    b = _build(tmp_path, rows, {"x": 4, "x.y": 4, "z": 4}, ["x", "x.y", "z"])
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"boxes": {"z": {}}}))
    W = ROOT / "scripts/amoeba_category_box_files.py"
    r = subprocess.run([sys.executable, str(W), "--build", str(b), "--out", str(tmp_path / "out"),
                        "--exclude-partition", str(old)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["boxes"] == ["x", "x.y"] and "z" in m["excluded"] and not (tmp_path / "out" / "z.jsonl").exists()
    got = [json.loads(x) for x in (tmp_path / "out" / "x.y.jsonl").read_text().splitlines()]
    assert got == [r for r in rows if r["box"] == "x.y"]
    assert m["rows"]["x"] == {"fit": 2, "held": 1, "val": 1}
    r = subprocess.run([sys.executable, str(W), "--build", str(b), "--out", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "refusing existing" in r.stderr
    m0 = json.loads((b / "manifest.json").read_text())
    m0["boxes"]["x"]["docs"] = 5
    (b / "manifest.json").write_text(json.dumps(m0))
    r = subprocess.run([sys.executable, str(W), "--build", str(b), "--out", str(tmp_path / "o2")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "!= build manifest docs" in r.stderr
    m0["corpus_sha256"] = "0" * 64
    (b / "manifest.json").write_text(json.dumps(m0))
    r = subprocess.run([sys.executable, str(W), "--build", str(b), "--out", str(tmp_path / "o3")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "sha256" in r.stderr


def test_unclaimed_delta(tmp_path):
    _maps(tmp_path, NCLAIMS, NEST)
    old = ["life.travel", "life.games", "sci.astronomy"]
    assert _part(tmp_path, old, "old", "--emit-unassigned", "--no-tiles").returncode == 0
    assert _part(tmp_path, NEST, "new", "--nested", "--groups", P._groups(tmp_path, NGROUPS),
                 "--emit-unassigned").returncode == 0
    D = ROOT / "scripts/amoeba_fn_unclaimed_delta.py"
    r = subprocess.run([sys.executable, str(D), "--old", "old", "--new", "new", "--out", "delta"], capture_output=True,
                       text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    d = json.loads((tmp_path / "delta.json").read_text())
    # old unclaimed: cols 0, 1, 2 (no old box claims them) + 15..30 (nobody); cols 0-2 now box tiles
    assert d["old_unclaimed"] == (3 + 16) * S and d["new_unclaimed"] == 16 * S
    assert d["where"]["box_tiles"] == 3 * S and d["by_box"] == {"life": S, "life.cooking": S, "life.cooking.x": S}
    assert d["new_unclaimed_outside_old"] == 0 and d["old_unclaimed_now_claimed"] == 3 * S


def test_give_it_to_the_parent_deepest_first(tmp_path):
    names = ["a", "a.b", "a.b.c", "a.b.d", "a.e", "z"]
    claims = {0: ["a.b.c", "a.b.d", "a.b", "a"],          # deepest selecting common ancestor: a.b
              1: ["a.b.c", "a.e", "a.b", "a"],            # a.b is not common to a.e -> a
              2: ["a.b.c", "a.b.d", "a"],                 # a.b silent -> still the LCA a.b
              3: ["a.b.c", "a.b.d", "a.e", "a"],          # 3 claimants at two depths -> a, not the core
              4: ["a.b.c", "a.b.d", "a.b", "z"],          # spans a.b and z -> 3 claims -> core
              5: ["a.b.c", "z", "a"],                     # spans -> pair a.b.c__z
              6: ["a.b.c"], 7: ["a.b.d"], 8: ["a.e"], 9: ["z"]}
    _maps(tmp_path, claims, names)
    r = _part(tmp_path, names, "out", "--nested", "--emit-unassigned", "--groups",
              P._groups(tmp_path, {"a.*": ["a", "a.b", "a.b.c", "a.b.d", "a.e"]}))
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "out" / "tiles.npz")
    p = json.loads((tmp_path / "out" / "partition.json").read_text())
    assert _cols(z["box:a.b"]) == [0, 2] and _cols(z["box:a"]) == [1, 3]
    assert z["core"][:, 4].all() and _cols(z["pair:a.b.c__z"]) == [5]
    assert _cols(z["box:a.b.c"]) == [6] and _cols(z["box:a.e"]) == [8]
    assert p["boxes"]["a.b"]["received_from_subtopics"] == 2 * S and p["boxes"]["a"]["received_from_subtopics"] == 2 * S
    assert p["boxes"]["a.b"]["received_lca_selected"] == S and p["boxes"]["a.b"]["received_lca_unselected"] == S
    assert "LCA rule gives it to that box" in p["groups"]["dropped"]["a.*"]["reason"]
    assert (sum(z[k].astype(int) for k in z.files if k != "layers") == 1).all()
    r = _part(tmp_path, names, "nn", "--nested", "--not-nested", "a.e")       # a.e outside a's subtree
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "nn" / "tiles.npz")
    assert _cols(z["box:a"]) == [] and _cols(z["box:a.b"]) == [0, 2]
    assert _cols(z["pair:a.b.c__a.e"]) == [1] and z["core"][:, 3].all()
    r = _part(tmp_path, names, "nb", "--nested", "--not-nested", "a.b")      # a.b and everything below it cut from a
    assert r.returncode == 0, r.stderr
    p = json.loads((tmp_path / "nb" / "partition.json").read_text())
    assert p["nested"]["descendants"]["a"] == ["a.e"] and p["nested"]["descendants"]["a.b"] == ["a.b.c", "a.b.d"]
    assert p["nested"]["ancestors"]["a.b.c"] == ["a.b"] and "a.b" not in p["nested"]["ancestors"]
    assert p["boxes"]["a.b"]["received_from_subtopics_outside_base"] == p["boxes"]["a.b"]["received_from_subtopics"]


def test_category_box_files_overlay(tmp_path):
    mk = lambda bx, tag: [{"box": bx, "split": sp, "register": "qa", "text": f"{tag} {bx} {sp} {i}"}   # noqa: E731
                          for i, sp in enumerate(["fit", "held"])]
    rows = mk("x", "p") + mk("sec", "p") + mk("z", "p")
    b = _build(tmp_path, rows, {"x": 2, "sec": 2, "z": 2}, ["x", "sec", "z"])
    orows = mk("sec", "o") + mk("sec.re", "o") + mk("sec.ctf", "o") + mk("x", "o")
    o = _build(tmp_path, orows, {"sec": 2, "sec.re": 2, "sec.ctf": 2, "x": 2}, ["sec", "sec.re", "sec.ctf", "x"],
               name="ob")
    W = ROOT / "scripts/amoeba_category_box_files.py"
    r = subprocess.run([sys.executable, str(W), "--build", str(b), "--out", str(tmp_path / "out"), "--exclude", "z",
                        "--overlay", f"{o}:sec"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["boxes"] == ["x", "sec", "sec.re", "sec.ctf"]
    assert m["box_build"]["x"] == str(b) and m["box_build"]["sec.re"] == str(o) and m["box_build"]["sec"] == str(o)
    assert "replaced by the overlay" in m["excluded"]["sec"] and len(m["builds"]) == 2
    got = [json.loads(x)["text"] for x in (tmp_path / "out" / "sec.jsonl").read_text().splitlines()]
    assert all(t.startswith("o ") for t in got)
    got = [json.loads(x)["text"] for x in (tmp_path / "out" / "x.jsonl").read_text().splitlines()]
    assert all(t.startswith("p ") for t in got)
    assert not (tmp_path / "out.tmp").exists()


def _files(d, rows_by_box):
    d.mkdir()
    for b, rows in rows_by_box.items():
        (d / f"{b}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "manifest.json").write_text(json.dumps({"boxes": list(rows_by_box), "build": str(d)}))


def test_heldfit_check(tmp_path):
    fit = {"x": [{"split": "fit", "group": "g1", "text": "a"}, {"split": "held", "group": "g9", "text": "skewed"}],
           "y": [{"split": "fit", "group": "g2", "text": "b"}]}
    _files(tmp_path / "fit", fit)
    _files(tmp_path / "ok", {"x": [{"split": "held", "group": "g3", "text": "c"},
                                   {"split": "fit", "group": "g1", "text": "a"}],
                             "y": [{"split": "val", "group": "g4", "text": "d"}]})
    _files(tmp_path / "bad", {"x": [{"split": "held", "group": "g2", "text": "e"}],   # g2 is fit in box y
                              "y": [{"split": "val", "group": "g5", "text": "a"}]})   # text a is fit in box x
    C = ROOT / "scripts/amoeba_heldfit_check.py"

    def run(h):
        return subprocess.run([sys.executable, str(C), "--fit-files", "fit", "--held-files", h, "--out", f"{h}.json"],
                              capture_output=True, text=True, cwd=tmp_path)
    r = run("ok")
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "ok.json").read_text())
    assert rep["disjoint"] and rep["per_box"]["x"]["held_side_rows"] == 1 and rep["fit_groups"] == 2
    r = run("bad")
    assert r.returncode == 4
    rep = json.loads((tmp_path / "bad.json").read_text())
    assert rep["per_box"]["x"]["group_in_fit"] == 1 and rep["per_box"]["y"]["text_in_fit"] == 1


def _split_of(g):
    return int(hashlib.sha256(g.encode()).hexdigest(), 16) % 8


def test_heldfit_balance(tmp_path):
    name = {0: "fit", 1: "fit", 2: "fit", 3: "fit", 4: "fit", 5: "val", 6: "held", 7: "held2"}

    def mk(reg, n, drop=()):
        out = []
        for i in range(n):
            g = f"{reg}/{i}"
            sp = name[_split_of(g)]
            if sp in drop:
                sp = "fit"                               # the skew: rows that should be held / val kept as fit
            out.append({"split": sp, "group": g, "register": reg, "source": f"{reg}src:{i}", "text": g})
        return out
    _files(tmp_path / "fit", {"z": [{"split": "fit", "group": "zz", "text": "zz"}]})
    _files(tmp_path / "good", {"x": mk("qa", 400) + mk("web", 400)})
    _files(tmp_path / "skew", {"x": mk("qa", 400) + mk("web", 400, drop=("held", "val"))})
    _files(tmp_path / "noval", {"x": mk("qa", 400, drop=("val",)) + mk("web", 400)})
    C = ROOT / "scripts/amoeba_heldfit_check.py"

    def run(h):
        return subprocess.run([sys.executable, str(C), "--fit-files", "fit", "--held-files", h, "--out", f"{h}.json"],
                              capture_output=True, text=True, cwd=tmp_path)
    assert run("good").returncode == 0
    r = run("skew")
    assert r.returncode == 3 and "x register=web" in r.stdout and "x source=websrc" in r.stdout
    assert "x register=qa" not in r.stdout
    r = run("noval")
    assert r.returncode == 3 and "val groups" in r.stdout and "x register=qa" in r.stdout
    rep = json.loads((tmp_path / "noval.json").read_text())
    assert rep["balance"]["x"]["register=qa"]["per_split"]["val"] == 0 and not rep["balanced"]
    assert list(rep["skipped_boxes"]) == ["x"]
    assert rep["skipped_boxes"]["x"].startswith("skipped: held split unbalanced (register=qa")
    assert not json.loads((tmp_path / "good.json").read_text())["skipped_boxes"]


def test_battery_skipped_boxes(nested_toy, monkeypatch):
    BT._install_fakes(monkeypatch)
    mod = BT._load(ROOT / "scripts/amoeba_box_battery.py")
    keep = [b for b in NEST if b != "life.travel"]
    held = [x for b in keep + ["english", "cmn"] for x in ("--held", f"{b}=held/{b}.jsonl@8")]
    monkeypatch.chdir(nested_toy)
    why = "skipped: held split unbalanced (register=web: 0 held-side groups of 99 (expected ~25))"
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "nest", "--out", "sk", *held,
                                      "--docs", "4", "--english", "english", "--arms", "boxes,groups",
                                      "--boxes", ",".join(keep), "--skipped", f"life.travel={why}"])
    mod.main()
    meta = json.loads((nested_toy / "sk" / "battery.meta.json").read_text())
    assert meta["skipped"] == {"life.travel": why}
    md = (nested_toy / "sk" / "battery.md").read_text()
    assert f"- life.travel: {why}" in md and "| life.travel |" not in md
    g = {json.loads(x)["key"]: json.loads(x) for x in (nested_toy / "sk" / "groups.jsonl").read_text().splitlines()}
    assert g["group:tg"]["verdict"]["members_without_held"] == ["life.travel"]
    assert g["group:tg"]["verdict"]["members_skipped"] == {"life.travel": why} and "| tg (1 skipped) |" in md
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "nest", "--out", "sk2", *held,
                                      "--docs", "4", "--english", "english", "--boxes", ",".join(keep),
                                      "--skipped", "life.cooking=x"])
    with pytest.raises(SystemExit, match="still in --boxes"):
        mod.main()


def test_unclaimed_delta_skipped(tmp_path):
    _maps(tmp_path, NCLAIMS, NEST)
    assert _part(tmp_path, ["life.travel"], "old", "--emit-unassigned", "--no-tiles").returncode == 0
    assert _part(tmp_path, NEST, "new", "--nested", "--emit-unassigned").returncode == 0
    (tmp_path / "hc.json").write_text(json.dumps({"skipped_boxes": {"life": "skipped: held split unbalanced (q)"}}))
    D = ROOT / "scripts/amoeba_fn_unclaimed_delta.py"
    r = subprocess.run([sys.executable, str(D), "--old", "old", "--new", "new", "--out", "delta", "--skipped",
                        "hc.json"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "delta.json").read_text())["battery_skipped"] == {
        "life": "skipped: held split unbalanced (q)"}
    assert "- life: skipped: held split unbalanced (q)" in (tmp_path / "delta.md").read_text()


def test_joint_tiles_and_battery_arm(nested_toy, monkeypatch):
    root = nested_toy
    base = ["--nested", "--groups", P._groups(root, NGROUPS)]
    r = _part(root, NEST, "jsweep", *base, "--joint-sweep", "1,3,4", "--no-tiles")
    assert r.returncode == 0, r.stderr
    j = json.loads((root / "jsweep" / "partition.json").read_text())["joint"]
    assert j["floor"] is None and j["sweep"]["3"]["tiles"] == 1 and j["sweep"]["4"]["tiles"] == 0
    assert j["sweep"]["3"]["units_moved"] == S                       # col 4: x + travel + astronomy, no common box
    r = _part(root, NEST, "joint", *base, "--joint-sweep", "1,3,4", "--joint-floor", "auto",
              "--check-against", "nestflat")
    assert r.returncode != 0                                          # (no such reference) sanity of the flag path
    assert _part(root, NEST, "nest0", "--nested").returncode == 0
    r = _part(root, NEST, "jt", "--nested", "--joint-sweep", "3,4", "--joint-floor", "auto",
              "--check-against", "nest0")
    assert r.returncode == 0, r.stderr
    p = json.loads((root / "jt" / "partition.json").read_text())
    z = np.load(root / "jt" / "tiles.npz")
    name = "life.cooking.x+life.travel+sci.astronomy"
    assert p["joint"]["floor"] == 3 and "auto" in p["joint"]["floor_choice"] and name in p["joint"]["tiles"]
    assert _cols(z[f"joint:{name}"]) == [4] and not z["core"][:, 4].any()
    assert _cols(z["joint:life.games+life.travel+sci.astronomy"]) == [13]   # no family declared here
    assert (sum(z[k].astype(int) for k in z.files if k != "layers") == 1).all()
    assert p["check_against"]["equal_with_groups_folded"]
    # the declared family keeps its claim set out of the joint tiles
    assert _part(root, NEST, "jg", *base, "--joint-floor", "3").returncode == 0
    zg = np.load(root / "jg" / "tiles.npz")
    assert "joint:life.games+life.travel+sci.astronomy" not in zg.files and _cols(zg["group:tg"]) == [13]
    # battery joint arm on the variant of "nest0"
    BT._install_fakes(monkeypatch)
    mod = BT._load(ROOT / "scripts/amoeba_box_battery.py")
    held = [x for b in NEST + ["english", "cmn"] for x in ("--held", f"{b}=held/{b}.jsonl@8")]
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "nest0", "--out", "jb", *held,
                                      "--docs", "4", "--english", "english", "--arms", "boxes",
                                      "--joint-partition", "jt", "--joint-top", "1"])
    BT.CALLS.clear()
    mod.main()
    rows = {json.loads(x)["key"]: json.loads(x) for x in (root / "jb" / "joint.jsonl").read_text().splitlines()}
    top = max(p["joint"]["tiles"], key=lambda k: p["joint"]["tiles"][k]["units"])
    assert set(rows) == {f"joint:{top}"} | {f"served:{m}" for m in p["joint"]["tiles"][top]["members"]}
    J = z[f"joint:{top}"]
    assert any(np.array_equal(x[P.LAYERS] > 0, ~J) for x in BT._mask_for(p["joint"]["tiles"][top]["members"] + ["english"]))
    for m in p["joint"]["tiles"][top]["members"]:                    # open: tiles naming m, an ancestor or a descendant
        lin = {m} | {x for x in NEST if m.startswith(x + ".") or x.startswith(m + ".")}
        want = sorted(f"joint:{k}" for k, v in p["joint"]["tiles"].items() if lin & set(v["members"]))
        assert sorted(rows[f"served:{m}"]["joint_tiles_open"]) == want
    assert all(r["settings"]["joint_tiles_sha256_12"] and r["settings"]["joint_floor"] == 3 for r in rows.values())
    assert "Joint tiles" in (root / "jb" / "battery.md").read_text()
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "nest", "--out", "jx", *held,
                                      "--docs", "4", "--english", "english", "--joint-partition", "jt"])
    with pytest.raises(SystemExit, match="joint-partition"):
        mod.main()
