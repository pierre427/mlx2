"""CPU tests (2026-11-19, Pierre) of scripts/amoeba_fn_partition.py --detach-share (weak-claimant detach before the
pair / family / LCA / joint / core decisions) and --english-box (the English-map units outside T2 leave the core for
their own box), and of the battery's English-box gate and english arm (fake host, no model)."""
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
ENG_COLS = (8, 30, 31)            # the English map: col 8 is inside T2 (expert 1 = cols 8..15)
LANG = ["cmn", "fra"]


def _drive_maps(root, plan, names):
    """plan: {col: {box: drive}}; a box selects a col where it has a drive (baseline 0.1 x drive passes r = 0.25).
    T2 = expert 1 (cols 8..15); English map = ENG_COLS."""
    for b in names:
        sel = np.zeros((S, P.N), bool)
        mt = np.zeros((S, P.N), np.float32)
        mb = np.full((S, P.N), 0.9, np.float32)
        for c, who in plan.items():
            if b in who:
                v = who[b] if isinstance(who[b], tuple) else (who[b], 0.1 * who[b])   # (drive, English mean)
                sel[:, c] = True
                mt[:, c], mb[:, c] = v
        d = root / "maps" / b
        d.mkdir(parents=True)
        arr = {}
        for s in range(S):
            arr[f"target_{s}"] = sel[s]
            arr[f"mean_target_{s}"] = mt[s]
            arr[f"mean_baseline_{s}"] = mb[s]
        np.savez(d / "map.npz", **arr)
        (d / "report.json").write_text(json.dumps({
            "per_site": [{"class": "moe", "units": P.N, "layer": l} for l in P.LAYERS], "fingerprint": "f" * 64,
            "checkpoint": "toy@ffffffff", "baseline": {"corpus": "english.jsonl"},
            "target": {"corpus": f"{b}.jsonl"}, "capture_rev": "toyrev"}))
    eng = np.zeros((S, P.N), bool)
    eng[:, list(ENG_COLS)] = True
    P._map(root / "maps" / "english", eng, eng)
    t2 = np.zeros((6, P.E), np.uint8)
    t2[:, 1] = 1
    P._npz_fixed(root / "t2.npz", core=t2)


def _part(root, names, out, *extra):
    cmd = [sys.executable, str(PART)] + [x for b in names for x in ("--box", f"{b}=maps/{b}")] + [
        "--core-map", "maps/english", "--t2", "t2.npz", "--out", out, *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=root)


def _cols(m):
    return sorted(set(np.flatnonzero(m.any(0))))


def _cover(z):
    assert (sum(z[k].astype(int) for k in z.files if k != "layers") == 1).all()


FLAT = "abcd"
FPLAN = {
    0: {"a": 1.0, "b": 1.0, "c": 0.01},              # c is weak: core -> pair a__b
    1: {"a": 1.0, "b": 0.02, "c": 0.03},             # b and c weak: core -> box a
    2: {"a": 1.0, "b": 0.5, "c": 0.2},               # every claim strong: stays core
    3: {"a": 1.0, "b": 0.04},                        # pair -> box a
    4: {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.001},   # still 3 strong claimants: core
    5: {"a": 1.0},                                    # box a either way
    6: {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.01},    # abcd crosses family abc -> core; detached -> group abc
    7: {"a": 1.0, "b": 0.049, "c": 0.05},            # b < 0.05 x 1.0 detached, c == 0.05 kept: pair a__c
    20: {"b": 1.0}, 21: {"c": 1.0},                  # b and c have box tiles (family abc needs 2+ sub-tiles)
    30: {"a": 1.0, "b": 1.0},                         # English map: English box (or core), never the pair
}


def test_detach_moves_weak_claims_before_pair_and_family(tmp_path):
    _drive_maps(tmp_path, FPLAN, FLAT)
    g = P._groups(tmp_path, {"abc": ["a", "b", "c"]})
    r = _part(tmp_path, FLAT, "off", "--groups", g)
    assert r.returncode == 0, r.stderr
    r = _part(tmp_path, FLAT, "on", "--groups", g, "--detach-share", "0.05", "--detach-sweep", "0.01,0.1")
    assert r.returncode == 0, r.stderr
    z0, z = np.load(tmp_path / "off" / "tiles.npz"), np.load(tmp_path / "on" / "tiles.npz")
    _cover(z0)
    _cover(z)
    assert z0["core"][:, [4, 6]].all() and _cols(z0["group:abc"]) == [0, 1, 2, 7] and _cols(z0["pair:a__b"]) == [3]
    assert _cols(z["pair:a__b"]) == [0] and _cols(z["pair:a__c"]) == [7]
    assert _cols(z["box:a"]) == [1, 3, 5]
    assert not z["core"][:, [0, 1, 2, 3, 4, 6, 7]].any()
    assert _cols(z["group:abc"]) == [2, 4, 6]       # cols 4, 6: abcd crossed the family; detached d -> abc
    assert z["core"][:, 30].all() and z["core"][:, 8:16].all()               # base units untouched
    p = json.loads((tmp_path / "on" / "partition.json").read_text())
    d = p["detach"]
    assert d["share"] == 0.05 and "weak-claimant detach" in p["rule"]
    # claims detached: c@0, b@1, c@1, b@3, d@4, d@6, b@7 (per layer); col 30 is English-map (b keeps its claim? no:
    # a and b tie there, nothing detached)
    assert d["claims_detached"] == 7 * S and d["claims_detached_outside_base"] == 7 * S
    assert d["per_box"]["b"]["claims_detached"] == 3 * S and p["boxes"]["b"]["claims_detached"] == 3 * S
    assert p["boxes"]["b"]["selected_before_detach"] == 9 * S and p["boxes"]["b"]["selected"] == 6 * S
    cc = d["claim_counts_outside_base"]
    assert cc["0.0"] == {"box": 3 * S, "pair": 1 * S, "three_plus": 6 * S, "unclaimed": cc["0.0"]["unclaimed"]}
    assert cc["0.05"]["three_plus"] == 3 * S and cc["0.05"]["pair"] == 2 * S and cc["0.05"]["box"] == 5 * S
    assert set(cc) == {"0.0", "0.01", "0.05", "0.1"}
    assert d["claims_below"]["0.05"] == 7 * S
    # sweep only: the same report, nothing detached, tiles equal to the off run
    r = _part(tmp_path, FLAT, "sweep", "--groups", g, "--detach-sweep", "0.05")
    assert r.returncode == 0, r.stderr
    zs = np.load(tmp_path / "sweep" / "tiles.npz")
    assert sorted(zs.files) == sorted(z0.files) and all(np.array_equal(zs[k], z0[k]) for k in z0.files)
    ps = json.loads((tmp_path / "sweep" / "partition.json").read_text())
    assert ps["detach"]["share"] is None and ps["detach"]["claims_detached"] == 0
    assert ps["detach"]["claim_counts_outside_base"]["0.05"] == cc["0.05"]


def test_detach_off_is_the_default(tmp_path):
    P._toy(tmp_path)
    r = P._run(tmp_path, "out", "--detach-share", "0")
    assert r.returncode == 0, r.stderr
    assert P._digest(tmp_path / "out") == P.GOLDEN_DEFAULT


NEST = ["life", "life.cooking", "life.travel", "sci.x"]
NPLAN = {
    0: {"life.cooking": 1.0, "life.travel": 0.01, "sci.x": 1.0},   # off: 3 claims, no common box -> core; on: pair
    1: {"life.cooking": 1.0, "life.travel": 1.0, "sci.x": 0.01},   # off: core; on: siblings -> their LCA life
    2: {"life.cooking": 1.0, "life": 0.001},                        # descendant absorption first: cooking's anyway
    3: {"life.cooking": 0.01, "life.travel": 1.0},                  # on: travel alone (no LCA roll-up of a weak claim)
}


def test_detach_precedes_the_lca_rule(tmp_path):
    _drive_maps(tmp_path, NPLAN, NEST)
    assert _part(tmp_path, NEST, "off", "--nested").returncode == 0
    r = _part(tmp_path, NEST, "on", "--nested", "--detach-share", "0.05")
    assert r.returncode == 0, r.stderr
    z0, z = np.load(tmp_path / "off" / "tiles.npz"), np.load(tmp_path / "on" / "tiles.npz")
    _cover(z)
    assert z0["core"][:, [0, 1]].all() and 3 in _cols(z0["box:life"])
    assert _cols(z["pair:life.cooking__sci.x"]) == [0]
    assert _cols(z["box:life"]) == [1]
    assert _cols(z["box:life.cooking"]) == [2]
    assert 3 in _cols(z["box:life.travel"])
    p = json.loads((tmp_path / "on" / "partition.json").read_text())
    assert p["boxes"]["life"]["received_from_subtopics"] == S        # only col 1 rolls up after the detach
    assert p["detach"]["per_box"]["life"]["claims_detached"] == 0     # life's claim on col 2 went to absorption


def test_detach_refusals(tmp_path):
    _drive_maps(tmp_path, FPLAN, FLAT)
    for bad in (["--detach-share", "1.0"], ["--detach-share", "-0.1"], ["--detach-sweep", "x"],
                ["--detach-sweep", "1.5"]):
        assert _part(tmp_path, FLAT, "x" + str(abs(hash(tuple(bad)))), *bad).returncode != 0


EPLAN = {
    0: {"a": 1.0}, 1: {"a": 1.0, "b": 1.0}, 2: {"a": 1.0, "b": 1.0, "c": 1.0},
    30: {"a": 1.0},                                   # English map AND claimed by a: stays English's
    8: {"a": 1.0, "b": 1.0},                          # English map inside T2: core
}


def test_english_box(tmp_path):
    _drive_maps(tmp_path, EPLAN, "abc")
    assert _part(tmp_path, "abc", "off").returncode == 0
    r = _part(tmp_path, "abc", "on", "--english-box", "lang.eng", "--english-closed-for", "c")
    assert r.returncode == 0, r.stderr
    z0, z = np.load(tmp_path / "off" / "tiles.npz"), np.load(tmp_path / "on" / "tiles.npz")
    _cover(z)
    assert _cols(z["box:lang.eng"]) == [30, 31]
    assert z["core"][:, 8:16].all() and not z["core"][:, [30, 31]].any() and z["core"][:, 2].all()
    assert z0["core"][:, [30, 31]].all()
    for k in z0.files:                                # nothing else moves
        if k != "core":
            assert np.array_equal(z[k], z0[k]), k
    assert np.array_equal(z["core"] | z["box:lang.eng"], z0["core"])
    p = json.loads((tmp_path / "on" / "partition.json").read_text())
    e = p["english_box"]
    assert e["name"] == "lang.eng" and e["units"] == 2 * S and e["closed_for"] == ["c"]
    assert e["claimed_by_topic_boxes"] == {"1": S, "2": 0, "3+": 0}
    assert p["boxes"]["lang.eng"]["english_box"] and p["core"]["to_english_box"] == 2 * S
    assert p["core"]["units"] == int(z["core"].sum()) and p["core"]["small_pairs_spilled"] == 0
    assert "box lang.eng" in p["rule"]
    assert _part(tmp_path, "abc", "x1", "--english-box", "a").returncode != 0
    assert _part(tmp_path, "abc", "x2", "--english-box", "lang.eng", "--english-closed-for", "zz").returncode != 0
    assert _part(tmp_path, "abc", "x3", "--english-closed-for", "c").returncode != 0
    # with joint tiles and detach together: still an exact cover, and joint tiles never take English-map units
    r = _part(tmp_path, "abc", "both", "--english-box", "lang.eng", "--detach-share", "0.05", "--joint-floor", "1")
    assert r.returncode == 0, r.stderr
    zb = np.load(tmp_path / "both" / "tiles.npz")
    _cover(zb)
    assert _cols(zb["joint:a+b+c"]) == [2] and _cols(zb["box:lang.eng"]) == [30, 31]


BOXB = ["cmn", "fra", "life", "sci.x"]
BPLAN = {0: {"cmn": 1.0}, 1: {"fra": 1.0}, 2: {"life": 1.0}, 3: {"sci.x": 1.0}, 4: {"cmn": 1.0, "life": 1.0},
         5: {"cmn": 1.0, "fra": 1.0, "life": 1.0}}


@pytest.fixture
def etoy(tmp_path):
    _drive_maps(tmp_path, BPLAN, BOXB)
    r = _part(tmp_path, BOXB, "pe", "--english-box", "lang.eng", "--english-closed-for", ",".join(LANG))
    assert r.returncode == 0, r.stderr
    (tmp_path / "held").mkdir()
    for b in BOXB + ["english", "gsm"]:
        rows = [{"split": "held", "text": f"{b} held text number {i} with words"} for i in range(6)]
        (tmp_path / "held" / f"{b}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def _bat(monkeypatch, root, out, *extra):
    BT._install_fakes(monkeypatch)
    mod = BT._load(ROOT / "scripts/amoeba_box_battery.py")
    held = [x for b in BOXB + ["english", "gsm"] for x in ("--held", f"{b}=held/{b}.jsonl@8")]
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "pe", "--out", out, *held,
                                      "--docs", "4", "--english", "english", *extra])
    BT.CALLS.clear()
    mod.main()


def test_battery_english_gate_and_arm(etoy, monkeypatch):
    _bat(monkeypatch, etoy, "bat", "--arms", "boxes,english", "--global-held", "gsm")
    z = np.load(etoy / "pe" / "tiles.npz")
    E = z["box:lang.eng"]
    rows = {json.loads(x)["box"]: json.loads(x) for x in (etoy / "bat" / "battery.jsonl").read_text().splitlines()}
    assert set(rows) == set(BOXB)                                     # the English box is no topic box
    assert rows["life"]["english_box_open"] and not rows["cmn"]["english_box_open"]
    life = z["core"] | E | z["box:life"] | z["pair:cmn__life"]
    assert any(np.array_equal(x[P.LAYERS] > 0, life) for x in BT._mask_for(["life", "english"]))
    cmn = z["core"] | z["box:cmn"] | z["pair:cmn__life"]               # English closed for a language box
    assert any(np.array_equal(x[P.LAYERS] > 0, cmn) for x in BT._mask_for(["cmn", "english"]))
    assert not any(np.array_equal(x[P.LAYERS] > 0, cmn | E) for x in BT._mask_for(["cmn", "english"]))
    e = {}
    for x in (etoy / "bat" / "english.jsonl").read_text().splitlines():
        r = json.loads(x)
        e[r["key"]] = r
    assert {"without", "random", "core_open", "core_closed", "served_open:cmn", "served_open:fra", "verdict"} <= set(e)
    assert "served_closed:cmn" not in e                              # reused from the boxes arm's core_plus
    assert e["without"]["sets"] == ["english", "cmn", "fra", "gsm"]
    assert any(np.array_equal(x[P.LAYERS] > 0, ~E) for x in BT._mask_for(["english", "cmn", "fra", "gsm"]))
    assert any(np.array_equal(x[P.LAYERS] > 0, z["core"] | E) for x in BT._mask_for(["english"]))
    assert any(np.array_equal(x[P.LAYERS] > 0, z["core"]) for x in BT._mask_for(["english"]))
    assert any(np.array_equal(x[P.LAYERS] > 0, cmn | E) for x in BT._mask_for(["cmn"]))
    v = e["verdict"]
    assert set(v["closing_cost_off_english"]) == set(LANG) and v["box"] == "lang.eng"
    want = rows["cmn"]["core_plus"]["cmn"] - rows["cmn"]["path_offset"]["cmn"] - e["served_open:cmn"]["net"]["cmn"]
    assert v["closing_cost_off_english"]["cmn"] == pytest.approx(want)
    assert "English box lang.eng" in (etoy / "bat" / "battery.md").read_text()
    meta = json.loads((etoy / "bat" / "battery.meta.json").read_text())
    assert meta["english_box"]["name"] == "lang.eng" and meta["english_box"]["arm_run"]
    # english arm alone: served_closed is scored, not reused
    _bat(monkeypatch, etoy, "bat2", "--arms", "english")
    keys = {json.loads(x)["key"] for x in (etoy / "bat2" / "english.jsonl").read_text().splitlines()}
    assert {"served_closed:cmn", "served_closed:fra"} <= keys


def test_battery_english_refusals(etoy, monkeypatch):
    with pytest.raises(SystemExit, match="english arm"):
        _bat(monkeypatch, etoy, "r1", "--arms", "boxes")
    _bat(monkeypatch, etoy, "r2", "--arms", "boxes", "--boxes", "cmn,life")         # English box left out: fine
    assert not (etoy / "r2" / "english.jsonl").exists()
    assert _part(etoy, BOXB, "plain").returncode == 0
    BT._install_fakes(monkeypatch)
    mod = BT._load(ROOT / "scripts/amoeba_box_battery.py")
    held = [x for b in BOXB + ["english"] for x in ("--held", f"{b}=held/{b}.jsonl@8")]
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "plain", "--out", "r3", *held,
                                      "--docs", "4", "--english", "english", "--arms", "boxes,english"])
    with pytest.raises(SystemExit, match="English box"):
        mod.main()


SH = "abcd"
SPLAN = {
    0: {"a": (1.0, 0.5)},                              # fails exclusivity: off unclaimed core; shared: pair a+EN
    1: {"a": (1.0, 0.5), "b": (1.0, 0.5)},             # EN + a + b inside family abc -> group:abc+lang.eng
    2: {"a": (1.0, 0.5), "d": (1.0, 0.5)},             # EN + a + d crosses families -> joint
    3: {"a": 1.0},
    4: {"a": (1.0, 0.3), "b": 20.0},                   # English 0.3 / 20 < 0.05: detached -> pair a__b
    5: {"b": 1.0}, 6: {"c": 1.0}, 7: {"d": 1.0},
    16: {"a": 1.0, "b": 1.0, "c": 1.0},                # family abc
    31: {"a": 1.0},                                    # English map + exclusive a: shared -> pair a+EN
}                                                       # col 30: English map alone -> box:lang.eng


def test_english_shared(tmp_path, monkeypatch):
    _drive_maps(tmp_path, SPLAN, SH)
    g = P._groups(tmp_path, {"abc": ["a", "b", "c"]})
    ex = ["--groups", g, "--english-box", "lang.eng", "--english-closed-for", "d", "--joint-floor", "1"]
    r = _part(tmp_path, SH, "excl", *ex)
    assert r.returncode == 0, r.stderr
    r = _part(tmp_path, SH, "sh", *ex, "--english-shared")
    assert r.returncode == 0, r.stderr
    r = _part(tmp_path, SH, "shd", *ex, "--english-shared", "--detach-share", "0.05")
    assert r.returncode == 0, r.stderr
    zx, z, zd = (np.load(tmp_path / d / "tiles.npz") for d in ("excl", "sh", "shd"))
    for t in (zx, z, zd):
        _cover(t)
    assert zx["core"][:, [0, 1, 2]].all() and _cols(zx["box:lang.eng"]) == [30, 31]
    assert _cols(z["box:lang.eng"]) == [30]
    assert _cols(z["pair:a__lang.eng"]) == [0, 31]
    assert _cols(z["group:abc+lang.eng"]) == [1, 4] and _cols(z["group:abc"]) == [16]
    assert _cols(z["joint:a+d+lang.eng"]) == [2]
    assert not z["core"][:, [0, 1, 2, 4, 30, 31]].any() and z["core"][:, 8:16].all()
    assert _cols(zd["pair:a__b"]) == [4] and _cols(zd["group:abc+lang.eng"]) == [1]     # detach sees English
    p = json.loads((tmp_path / "sh" / "partition.json").read_text())
    e = p["english_box"]
    assert e["shared"]["units_by_kind"] == {"box": S, "pair": 2 * S, "family": 2 * S, "joint": S}
    assert e["shared"]["units_closed_serving"]["d"] == S + 2 * S + 2 * S             # all but the joint naming d
    assert p["boxes"]["lang.eng"]["english_box"] and p["boxes"]["a"]["shared_with_english"] == 4 * S
    assert p["groups"]["kept"]["abc+lang.eng"]["english_of"] == "abc"
    assert p["groups"]["kept"]["abc+lang.eng"]["source"] == "english"
    pd_ = json.loads((tmp_path / "shd" / "partition.json").read_text())
    assert pd_["detach"]["per_box"]["lang.eng"]["claims_detached"] == S
    assert _part(tmp_path, SH, "x", "--english-shared").returncode != 0               # needs --english-box
    # battery served compositions: English text opens every English-naming pair / family tile, d does not
    BT._install_fakes(monkeypatch)
    mod = BT._load(ROOT / "scripts/amoeba_box_battery.py")
    prep = p
    keys = [k for k in z.files if k != "layers"]
    get = lambda k: z[k]                                                                  # noqa: E731
    sb = mod.served_keep(get, keys, prep, "b", z["core"])
    want = (z["core"] | z["box:b"] | z["group:abc"] | z["group:abc+lang.eng"] | z["box:lang.eng"]
            | z["pair:a__lang.eng"])
    for k in keys:
        if k.startswith("pair:") and "b" in k[5:].split("__"):
            want |= z[k]
    assert np.array_equal(sb, want)
    sd = mod.served_keep(get, keys, prep, "d", z["core"])
    assert np.array_equal(sd, z["core"] | z["box:d"])


def test_absorbed_claim_keeps_the_ancestors_drive(tmp_path):
    """Review 2026-11-19: life drives the unit hardest, life.cooking absorbs life's claim; the absorbed claim must not
    be detached as cooking's own weak one."""
    names = ["life", "life.cooking", "sci.x"]
    _drive_maps(tmp_path, {0: {"life": 1.0, "life.cooking": 0.01, "sci.x": 0.5}, 1: {"life.cooking": 1.0},
                           2: {"sci.x": 1.0}}, names)
    for out, extra in (("off", []), ("on", ["--detach-share", "0.05"])):
        r = _part(tmp_path, names, out, "--nested", *extra)
        assert r.returncode == 0, r.stderr
        z = np.load(tmp_path / out / "tiles.npz")
        assert _cols(z["pair:life.cooking__sci.x"]) == [0], out


def test_english_refusals_more(tmp_path):
    _drive_maps(tmp_path, EPLAN, "abc")
    assert _part(tmp_path, "abc", "x1", "--english-box", "lang.eng", "--english-shared", "--groups-auto").returncode
    assert _part(tmp_path, "abc", "x2", "--english-box", "lang.eng", "--english-closed-for", "lang.eng").returncode


@pytest.mark.parametrize("shared", [False, True])
def test_joint_arm_with_english_box(tmp_path, monkeypatch, shared):
    names = ["a", "b", "cmn", "d"]
    plan = {0: {"a": 1.0}, 1: {"b": 1.0}, 2: {"cmn": 1.0}, 3: {"d": 1.0},
            4: {"a": 1.0, "cmn": 1.0, "d": 1.0},                        # joint a+cmn+d
            5: {"a": (1.0, 0.5), "b": (1.0, 0.5)}}                      # shared: joint a+b+lang.eng
    _drive_maps(tmp_path, plan, names)
    ex = ["--english-box", "lang.eng", "--english-closed-for", "cmn"] + (["--english-shared"] if shared else [])
    assert _part(tmp_path, names, "p", *ex).returncode == 0
    r = _part(tmp_path, names, "pj", *ex, "--joint-floor", "1")
    assert r.returncode == 0, r.stderr
    zj = np.load(tmp_path / "pj" / "tiles.npz")
    (tmp_path / "held").mkdir()
    for b in names + ["english"]:
        rows = [{"split": "held", "text": f"{b} held text number {i} with words"} for i in range(6)]
        (tmp_path / "held" / f"{b}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    BT._install_fakes(monkeypatch)
    mod = BT._load(ROOT / "scripts/amoeba_box_battery.py")
    held = [x for b in names + ["english"] for x in ("--held", f"{b}=held/{b}.jsonl@8")]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "p", "--out", "jb", *held, "--docs",
                                      "4", "--english", "english", "--arms", "boxes,english", "--joint-partition",
                                      "pj", "--joint-top", "5"])
    BT.CALLS.clear()
    mod.main()
    rows = {json.loads(x)["key"]: json.loads(x) for x in (tmp_path / "jb" / "joint.jsonl").read_text().splitlines()}
    jt = [k for k in zj.files if k.startswith("joint:")]
    assert ("joint:a+b+lang.eng" in jt) == shared and "joint:a+cmn+d" in jt
    for m in ("a", "cmn", "d") + (("b",) if shared else ()):
        assert f"served:{m}" in rows, m
    if shared:
        assert "joint:a+b+lang.eng" in rows["served:d"]["joint_tiles_open"]          # English text: open
        assert "joint:a+b+lang.eng" not in rows["served:cmn"]["joint_tiles_open"]    # language box: closed
        assert "box:lang.eng" not in rows["served:cmn"].get("joint_tiles_open", [])


def test_core_assign_moves_a_matching_cluster(tmp_path):
    """scripts/amoeba_fn_core_assign.py: core units whose box-space profile matches box a's tile move into it; a
    flat-profile core cluster stays; the result is an exact cover and the rest of the partition is unchanged."""
    names = ["a", "b", "c"]
    prof = {"a": (1.0, 0.05, 0.05), "b": (0.05, 1.0, 0.05), "c": (0.05, 0.05, 1.0), "flat": (1.0, 1.0, 1.0),
            "zero": (0.0, 0.0, 0.0)}
    cols = {0: "a", 1: "a", 2: "b", 3: "b", 4: "c", 5: "c",          # box tiles (selected, exclusive)
            16: "a", 17: "a", 18: "a",                              # core, a-like, unselected
            19: "flat", 20: "flat", 21: "flat", 22: "flat",         # core, flat
            23: "a", 25: "zero"}                                    # 23: English-dominated a-like; 25: no drive
    for i, b in enumerate(names):
        d = tmp_path / "maps" / b
        d.mkdir(parents=True)
        arr = {}
        for s in range(S):
            sel = np.zeros(P.N, bool)
            mt = np.zeros(P.N, np.float32)
            for c, kind in cols.items():
                mt[c] = prof[kind][i] * (1 + 0.01 * s)
                sel[c] = c < 6 and kind == b
            arr[f"target_{s}"] = sel
            arr[f"mean_target_{s}"] = mt
            mbl = (0.02 * mt).astype(np.float32)
            mbl[23] = 0.9                                            # English drives col 23 hardest
            arr[f"mean_baseline_{s}"] = mbl
        np.savez(d / "map.npz", **arr)
        (d / "report.json").write_text(json.dumps({
            "per_site": [{"class": "moe", "units": P.N, "layer": l} for l in P.LAYERS], "fingerprint": "f" * 64,
            "checkpoint": "toy@ffffffff", "baseline": {"corpus": "english.jsonl"}, "target": {"corpus": f"{b}.jsonl"},
            "capture_rev": "toyrev"}))
    eng = np.zeros((S, P.N), bool)
    eng[:, list(ENG_COLS)] = True
    P._map(tmp_path / "maps" / "english", eng, eng)
    t2 = np.zeros((6, P.E), np.uint8)
    t2[:, 1] = 1
    P._npz_fixed(tmp_path / "t2.npz", core=t2)
    assert _part(tmp_path, names, "p", "--english-box", "lang.eng").returncode == 0
    cmd = [sys.executable, str(ROOT / "scripts/amoeba_fn_core_assign.py"), "--partition", "p", "--out", "ca",
           "--clusters", "2", "--min-units", "1", "--min-cos", "0.9", "--min-mass", "0.5", "--min-margin", "0.01"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    z0, z = np.load(tmp_path / "p" / "tiles.npz"), np.load(tmp_path / "ca" / "partition" / "tiles.npz")
    _cover(z)
    assert _cols(z["box:a"]) == [0, 1, 16, 17, 18]
    assert z["core"][:, 19:24].all() and z["core"][:, 25].all() and z["core"][:, 8:16].all()
    assert np.array_equal(z["box:lang.eng"], z0["box:lang.eng"]) and np.array_equal(z["box:b"], z0["box:b"])
    p = json.loads((tmp_path / "ca" / "partition" / "partition.json").read_text())
    assert p["core_assign"]["moves"] == {"box:a": 3 * S} and p["boxes"]["a"]["units"] == 5 * S
    pr = json.loads((tmp_path / "ca" / "proposals.json").read_text())
    assert pr["proposed_clusters"] == 1 and pr["units_moved"] == 3 * S
    assert pr["excluded"]["english_dominated"] == S and pr["excluded"]["zero_profile"] >= S
