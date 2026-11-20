"""CPU tests of scripts/amoeba_fn_interactions.py (the opt-in tile interaction arm, 2026-11-19) with toy loss models
(no model, no GPU): an additive model gives zero excess, a constructed backup / dependency / rival is detected and
labelled, the random-pair noise floor tracks injected noise, the damage-then-close and repair arms find what was
built in, the budget stops new work, and the script end to end (fake host) on the partition toy; plus
amoeba_fn_partition.py --emit-detached."""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import amoeba_fn_interactions as X
import test_amoeba_box_battery_arms_cpu as BT
import test_amoeba_fn_detach_english_cpu as DE
import test_amoeba_fn_partition_groups_cpu as P

SHAPE = (3, 40)
NDOC = 30


def cols(*c):
    m = np.zeros(SHAPE, bool)
    m[:, list(c)] = True
    return m


A, B, C, D = cols(0, 1, 2, 3), cols(4, 5, 6, 7), cols(8, 9), cols(10, 11)
TILES = {"A": X.pack(A), "B": X.pack(B), "C": X.pack(C), "D": X.pack(D)}


def frac(closed, m):
    return float(closed[m].mean())


def toy(fn, noise=0.0, seed=0):
    """score_fn: per-document CE = 2 + doc offset + fn(closed, set) (+ mask-dependent noise of sd `noise`)."""
    off = np.random.default_rng(seed).normal(0, 0.3, NDOC)

    def score(closed, s):
        v = 2.0 + off + fn(closed, s)
        if noise:
            g = np.random.default_rng(zlib.crc32(closed.tobytes() + s.encode()) + seed)
            v = v + g.normal(0, noise, NDOC)
        return v
    return score


def additive(closed, s):
    w = np.arange(SHAPE[0] * SHAPE[1], dtype=float).reshape(SHAPE) / 1000.0
    return float((closed * w).sum()) + (0.5 if s == "t" else 0.1) * closed.mean()


def ev_for(fn, **k):
    return X.Evaluator(toy(fn, **k), SHAPE, log=lambda *m: None)


PAIRS = [("A", "B", ["t", "eng"], 1.0), ("A", "C", ["t", "eng"], 0.5), ("C", "D", ["t", "eng"], 0.1)]
FLOORS = [("A x B", TILES["A"], TILES["B"], ["t", "eng"]), ("C x D", TILES["C"], TILES["D"], ["t", "eng"])]


def test_additive_model_has_zero_excess():
    ev = ev_for(additive)
    fl = X.run_random_floor(ev, FLOORS, SHAPE, 1, 0.95, log=lambda *m: None)
    assert fl["floor"] < 1e-9
    rows = X.run_pairs(ev, TILES, PAIRS, fl["floor"], 2.0, log=lambda *m: None)
    for r in rows:
        for v in r["sets"].values():
            assert abs(v["I"]) < 1e-9 and v["labels"] == []
        assert r["flags"] == [] and not r["role_reversal"]
    assert X.keystones(rows) == {}


def test_constructed_backup_is_detected():
    # topic t is served by A or B: either alone suffices; English is additive
    ev = ev_for(lambda c, s: (min(frac(c, A), frac(c, B)) if s == "t" else additive(c, s)), noise=0.002)
    fl = X.run_random_floor(ev, FLOORS * 4, SHAPE, 1, 0.95, log=lambda *m: None)
    rows = X.run_pairs(ev, TILES, PAIRS, fl["floor"], 2.0, log=lambda *m: None)
    ab = rows[0]["sets"]
    assert ab["t"]["I"] == pytest.approx(1.0, abs=0.01)
    assert set(ab["t"]["labels"]) >= {"redundancy", "hidden_backup:A_for_B", "hidden_backup:B_for_A"}
    assert ab["eng"]["labels"] == []
    # the null pair-sets: at a q95 floor and z = 2, at most one false flag among the four
    assert sum(bool(v["labels"]) for r in rows[1:] for v in r["sets"].values()) <= 1


def test_dependent_keystone_rival_and_role_reversal():
    def fn(c, s):
        fa, fb, fc = frac(c, A), frac(c, B), frac(c, C)
        if s == "t":                   # B only works when A is open; A is a keystone
            return 2 * fa + fb * (1 - fa)
        return fa * (1 - 0.5 * fc) + 0.0 * fb   # eng: closing C helps once A is gone (a dormant rival)
    ev = ev_for(fn)
    rows = X.run_pairs(ev, TILES, [("A", "B", ["t"], 1.0), ("A", "C", ["t", "eng"], 1.0)], 0.0, 2.0,
                       log=lambda *m: None)
    assert "dependent:B_on_A" in rows[0]["sets"]["t"]["labels"]
    assert X.keystones(rows) == {"A": ["B"]}
    assert "rival:B_after_A" in rows[1]["sets"]["eng"]["labels"]      # B = C in the (A, C) pair
    rr = ev_for(lambda c, s: (1 if s == "t" else -1) * frac(c, A) * frac(c, B))
    r = X.run_pairs(rr, TILES, [("A", "B", ["t", "eng"], 1.0)], 0.0, 2.0, log=lambda *m: None)[0]
    assert r["role_reversal"]


def test_noise_floor_tracks_noise():
    f = []
    for sd in (0.001, 0.01, 0.05):
        ev = ev_for(additive, noise=sd)
        f.append(X.run_random_floor(ev, FLOORS * 4, SHAPE, 3, 0.95, log=lambda *m: None)["floor"])
    assert f[0] < f[1] < f[2]
    assert 0.3 < f[2] / (0.05 * 2 / np.sqrt(NDOC)) < 6     # the mean of 4 noisy terms over NDOC docs
    # no false flag on the additive model at the floor
    ev = ev_for(additive, noise=0.01)
    fl = X.run_random_floor(ev, FLOORS * 4, SHAPE, 3, 0.95, log=lambda *m: None)["floor"]
    rows = X.run_pairs(ev, TILES, PAIRS, fl, 2.0, log=lambda *m: None)
    assert sum(len(r["flags"]) for r in rows) <= 1


def test_random_pairs_are_disjoint_and_matched():
    g = np.random.default_rng(0)
    ra = X.random_disjoint(A, A | B, g)
    rb = X.random_disjoint(B, A | B | ra, g)
    assert (ra.sum(1) == A.sum(1)).all() and (rb.sum(1) == B.sum(1)).all()
    assert not (ra & (A | B)).any() and not (rb & (A | B | ra)).any()


def test_damage_then_close():
    # X (cols 8,9) is a hidden backup for the context Y (cols 0..7): useless alone, needed once Y is closed
    Xm, Y = C, A | B
    spec = [{"name": "moves:t", "x": X.pack(Xm), "contexts": [("tile:t", X.pack(Y), ["t"])]}]
    ev = ev_for(lambda c, s: min(frac(c, Xm), frac(c, Y)), noise=0.001)
    r = X.run_backup(ev, spec, SHAPE, 2.0, 2, 0, log=lambda *m: None)[0]
    v = r["contexts"]["tile:t"]["sets"]["t"]
    assert v["alone"] == pytest.approx(0, abs=0.01) and v["after"] == pytest.approx(1, abs=0.01)
    assert v["hidden_backup"] and r["flags"] == ["hidden_backup:tile:t@t"]
    ev = ev_for(additive)
    r = X.run_backup(ev, spec, SHAPE, 2.0, 2, 0, log=lambda *m: None)[0]
    assert not r["contexts"]["tile:t"]["sets"]["t"]["excess_significant"]


def test_repair_picks_the_tile_that_helps():
    # served composition closes C and D; the topic needs C (and D a little); B is open and hurts (a competitor)
    closed = X.pack(C | D)
    ev = ev_for(lambda c, s: (frac(c, C) + 0.2 * frac(c, D) - 0.3 * frac(c, B)) if s == "b" else 0.05 * frac(c, C))
    spec = [{"box": "b", "closed": closed, "give": [("D", TILES["D"], 0.9), ("C", TILES["C"], 0.5)],
             "random": [], "compete": [("B", TILES["B"])]}]
    r = X.run_repair(ev, spec, "eng", log=lambda *m: None)[0]
    assert r["served_gap"]["mean"] == pytest.approx(1.2)
    assert r["best_give_back"] == "C" and r["give_back"]["C"]["recovery"] == pytest.approx(1 / 1.2)
    assert r["best_two"]["recovery"] == pytest.approx(1.0)
    assert r["best_competitor"] == "B" and r["compete"]["B"]["gain"] == pytest.approx(0.3)
    assert r["best_english_change"] == pytest.approx(-0.05)


def test_deadline_stops_new_work():
    ev = X.Evaluator(toy(additive), SHAPE, budget_s=0.0, log=lambda *m: None)
    ev.base("t")                       # the first evaluation starts the clock
    rows = X.run_pairs(ev, TILES, PAIRS, 0.0, 2.0, log=lambda *m: None)
    assert len(rows) == 1 and "truncated" in rows[0]


# ------------------------------------------------------------------ end to end on the partition toy (fake host)

def _setup(tmp_path, fails=True):
    P._toy(tmp_path)
    BT._corpora(tmp_path)
    assert P._run(tmp_path, "base", "--emit-unassigned").returncode == 0
    part = tmp_path / "base"
    p = json.loads((part / "partition.json").read_text())
    bd = tmp_path / "battery"
    bd.mkdir()
    held = {b: {"spec": f"{b}=held/{b}.jsonl@8", "docs_loaded": 4} for b in list(P.BOXES) + ["eng"]}
    (bd / "battery.meta.json").write_text(json.dumps({"fingerprint": p["fingerprint"], "partition": "base",
                                                      "docs": 4, "held": held}))
    rows = [{"box": b, "settings": {"partition": "base", "docs": 4}, "core_plus": {b: 0.2 if b in "ab" else 0.01},
             "path_offset": {}, "verdict": {"served_+0.05": not (fails and b in "ab")}} for b in P.BOXES]
    (bd / "battery.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return part, bd


def _main(monkeypatch, tmp_path, *extra):
    BT._install_fakes(monkeypatch)

    def fake_doc_ce(host, seqs, cfg):
        closed = 1.0 - np.asarray(cfg)[P.LAYERS]
        w = np.random.default_rng(len(seqs[0])).random(closed.shape)
        base = np.array([sum(q) % 7 / 10.0 for q in seqs])
        return base + float((closed * w).sum()) / 50 + float(closed[:, 9:13].min()) * 0.3
    monkeypatch.setattr(X, "doc_ce", fake_doc_ce)
    monkeypatch.chdir(tmp_path)
    return X.main(["--model", "m", "--partition", "base", "--battery", "battery", "--english", "eng", *extra])


def test_end_to_end_toy(tmp_path, monkeypatch):
    _setup(tmp_path)
    meta = _main(monkeypatch, tmp_path, "--out", "out", "--arms", "pairs,repair", "--boxes-top", "4",
                 "--max-pairs", "5", "--random-pairs", "3", "--repair-boxes", "2")
    out = tmp_path / "out"
    plan = json.loads((out / "plan.json").read_text())
    assert sorted(plan["repair"]["failing_boxes_chosen"]) == ["a", "b"]
    assert len(plan["pairs"]["measured"]) <= 5 and plan["pairs"]["tiles"]
    rows = [json.loads(x) for x in (out / "pairs.jsonl").read_text().splitlines()]
    assert rows and all("I" in v for r in rows for v in r["sets"].values())
    assert meta["evals"] > 0 and meta["checkpoint"] == "toy@ffffffff" and meta["mlx2_source_revision"] == "toyrev"
    rep = [json.loads(x) for x in (out / "repair.jsonl").read_text().splitlines()]
    assert sorted(r["box"] for r in rep) == ["a", "b"] and all("served_gap" in r for r in rep)
    assert (out / "interactions.md").read_text().startswith("# Tile interactions")
    # a nonempty evidence directory is refused
    with pytest.raises(SystemExit, match="not empty"):
        _main(monkeypatch, tmp_path, "--out", "out")


def test_refusals_and_dry_run(tmp_path, monkeypatch):
    _setup(tmp_path)
    with pytest.raises(SystemExit, match="needs --moves"):
        _main(monkeypatch, tmp_path, "--out", "o1", "--arms", "backup")
    plan = _main(monkeypatch, tmp_path, "--out", "o2", "--dry-run", "--budget-min", "0.05")
    assert plan["pairs"]["measured"] == [] and plan["pairs"]["screen_controls"] == []      # nothing fits 3 s
    plan = _main(monkeypatch, tmp_path, "--out", "o4", "--dry-run", "--budget-min", "0.5")
    assert 0 < plan["pairs"]["evals"] * plan["sec_per_eval_planned"] <= 0.5 * 60
    assert not (tmp_path / "o2" / "pairs.jsonl").exists()
    (tmp_path / "battery" / "battery.meta.json").unlink()
    with pytest.raises(SystemExit, match="has not finished"):
        _main(monkeypatch, tmp_path, "--out", "o3")



def test_repair_refusals(tmp_path, monkeypatch):
    _setup(tmp_path)
    bj = tmp_path / "battery" / "battery.jsonl"
    rows = bj.read_text()
    bj.write_text(rows.replace('"partition": "base"', '"partition": "elsewhere"'))
    with pytest.raises(SystemExit, match="no row"):
        _main(monkeypatch, tmp_path, "--out", "r1", "--arms", "repair")
    bj.unlink()
    with pytest.raises(SystemExit, match="battery.jsonl"):
        _main(monkeypatch, tmp_path, "--out", "r2", "--arms", "repair")


def _backup_fake(monkeypatch, xpos, ymask):
    """doc CE = doc base + 0.3 x min(share of X closed, share of Y closed): X is a hidden backup for Y."""
    def fake(host, seqs, cfg):
        closed = 1.0 - np.asarray(cfg)[P.LAYERS]
        xa, ya = closed.ravel()[xpos].mean(), closed[ymask].mean()
        return np.array([sum(q) % 7 / 10.0 for q in seqs]) + 0.3 * min(xa, ya)
    monkeypatch.setattr(X, "doc_ce", fake)


def test_backup_from_core_assign_moves(tmp_path, monkeypatch):
    part, _ = _setup(tmp_path)
    z = np.load(part / "tiles.npz")
    core = z["core"]
    pos = np.flatnonzero(core)[:6]
    ca = tmp_path / "ca"
    (ca / "partition").mkdir(parents=True)
    np.savez_compressed(ca / "moves.npz", layers=z["layers"], box__a=pos)
    BT._install_fakes(monkeypatch)
    _backup_fake(monkeypatch, pos, z["box:a"])
    monkeypatch.chdir(tmp_path)
    X.main(["--model", "m", "--partition", "base", "--battery", "battery", "--english", "eng", "--out", "ob",
            "--arms", "backup", "--moves", "ca", "--backup-controls", "2"])
    rows = [json.loads(x) for x in (tmp_path / "ob" / "backup.jsonl").read_text().splitlines()]
    assert rows[0]["name"] == "moves:box:a" and rows[0]["units"] == 6
    ctx = rows[0]["contexts"]
    assert "tile:box:a" in ctx and sum(k.startswith("served:") for k in ctx) == 2
    assert not any(k[7:] == "a" for k in ctx if k.startswith("served:"))
    v = ctx["tile:box:a"]["sets"]["a"]
    assert v["alone"] == pytest.approx(0) and v["after"] == pytest.approx(0.3) and v["hidden_backup"]
    assert all(c["sets"][s]["hidden_backup"] for c in ctx.values() for s in c["sets"])   # box a is closed there too
    # moves of another partition are refused
    np.savez_compressed(ca / "moves.npz", layers=z["layers"], box__a=np.flatnonzero(z["box:b"])[:3])
    with pytest.raises(SystemExit, match="not in --partition's core"):
        X.main(["--model", "m", "--partition", "base", "--battery", "battery", "--english", "eng", "--out", "ob2",
                "--arms", "backup", "--moves", "ca"])


def test_backup_from_detached_claims(tmp_path, monkeypatch):
    part, _ = _setup(tmp_path)
    z = np.load(part / "tiles.npz")
    xpos = np.flatnonzero(z["box:c"])[:4]          # a's detached claims went to box c: closed when serving a
    np.savez_compressed(tmp_path / "det.npz", layers=z["layers"], a=xpos, b=np.zeros(0, np.int64))
    BT._install_fakes(monkeypatch)
    monkeypatch.chdir(tmp_path)
    args = ["--model", "m", "--partition", "base", "--battery", "battery", "--english", "eng", "--arms", "backup",
            "--detached", "det.npz"]
    with pytest.raises(SystemExit, match="detach share"):
        X.main(args + ["--out", "d0"])
    p = json.loads((part / "partition.json").read_text())
    p["detach"] = {"share": 0.05}
    (part / "partition.json").write_text(json.dumps(p))
    _backup_fake(monkeypatch, xpos, z["box:b"])      # a needs X once b (closed when serving a) is gone
    X.main(args + ["--out", "d1"])
    rows = [json.loads(x) for x in (tmp_path / "d1" / "backup.jsonl").read_text().splitlines()]
    assert [r["name"] for r in rows] == ["detached:a"] and rows[0]["units"] == xpos.size   # b: none
    v = rows[0]["contexts"]["served:a"]["sets"]["a"]
    assert v["alone"] == pytest.approx(0) and v["after"] == pytest.approx(0.3) and v["hidden_backup"]


def test_partition_emit_detached(tmp_path):
    DE._drive_maps(tmp_path, DE.FPLAN, DE.FLAT)
    r = DE._part(tmp_path, DE.FLAT, "on", "--detach-share", "0.05", "--emit-detached")
    assert r.returncode == 0, r.stderr
    r0 = DE._part(tmp_path, DE.FLAT, "plain", "--detach-share", "0.05")
    assert r0.returncode == 0, r0.stderr
    z, z0 = np.load(tmp_path / "on" / "tiles.npz"), np.load(tmp_path / "plain" / "tiles.npz")
    assert all(np.array_equal(z[k], z0[k]) for k in z0.files)            # the flag changes no tile
    assert json.loads((tmp_path / "on" / "partition.json").read_text()) == \
        json.loads((tmp_path / "plain" / "partition.json").read_text())   # nor the report
    d = np.load(tmp_path / "on" / "detached.npz")
    S, N = len(P.LAYERS), P.N
    got = {k: sorted({int(p) % N for p in d[k]}) for k in d.files if k != "layers"}
    assert got == {"b": [1, 3, 7], "c": [0, 1], "d": [4, 6]}
    assert all(d[k].size == len(got[k]) * S for k in got)
    assert not (tmp_path / "plain" / "detached.npz").exists()
    assert DE._part(tmp_path, DE.FLAT, "bad", "--emit-detached").returncode != 0
