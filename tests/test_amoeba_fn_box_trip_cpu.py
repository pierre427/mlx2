"""CPU tests of scripts/amoeba_fn_box_trip.py on the carve toy with real Qwen3-Next MoE blocks in neuron units
(no model, no GPU): the drive is the firing capture's quantity, the scores and controls compute what they claim, and
the end-to-end run ranks a box first on its own held text and writes its receipt."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

from research.amoeba.carve.host import Host  # noqa: E402
from test_amoeba_carve_cpu import Tok  # noqa: E402
from test_amoeba_carve_neuron_units_cpu import qn_toy  # noqa: E402
import amoeba_fn_box_trip as T  # noqa: E402

LAYERS = [1, 3]
W = 6 * 16


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


class ChatTok(Tok):
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, enable_thinking=False,
                            tools=None):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + ("<tools>" if tools else "") + "<a>"
        return self.encode(text)


def _host():
    return Host(model=qn_toy(), tokenizer=ChatTok(), fingerprint="toy:abc", log=lambda *_: None, moe_units="neuron")


def _ids(seed, n=12):
    return np.random.default_rng(seed).integers(0, 64, n).tolist()


def test_capture_is_the_firing_capture_quantity_and_causal():
    h = _host()
    cap = T.Capture(h, LAYERS, W)
    assert (cap.E, cap.F) == (6, 16)
    ids = _ids(0, 14)
    P = 9
    got = {}

    def rec(l, kind, p):
        if kind != "dense":
            got[l] = (np.array(p["h"].astype(mx.float32)), np.array(p["inds"]), np.array(p["w"].astype(mx.float32)))
    h.ctl.reset()
    h.ctl.stats = rec
    h.ctl.stats_fast = True
    mx.eval(h.hidden(mx.array([ids])))
    h.ctl.reset()
    X, R, _ = cap.run([ids], [(0, P), (P, len(ids))])
    for r, l in enumerate(LAYERS):
        hh, ii, ww = got[l]
        for j, (s0, s1) in enumerate([(0, P), (P, len(ids))]):
            ref = np.zeros((6, 16))
            cnt = np.zeros(6)
            for t in range(s0, s1):
                for k in range(ii.shape[-1]):
                    ref[ii[0, t, k]] += np.abs(hh[0, t, k]) * ww[0, t, k]
                    cnt[ii[0, t, k]] += 1
            assert np.allclose(X[j][0, r], ref.reshape(-1) / (s1 - s0), atol=1e-5)
            assert np.allclose(R[j][0, r], cnt / ((s1 - s0) * ii.shape[-1]), atol=1e-6)
    Xp, _, _ = cap.run([ids[:P]], [(0, P)])               # the prompt segment is the prompt-only drive (causal)
    assert np.allclose(Xp[0], X[0], atol=1e-5)


def _sets(t, b, idx):
    return T.Sets(["box:x"], ["box"], [["x"]], np.array(idx, np.int64), np.array(t, np.float32),
                  np.array(b, np.float32), np.array([0, len(idx)], np.int64))


def test_scores_definitions():
    idx = [0, 17, 100, 150]                       # layer row 0 units 0, 17; row 1 units 4, 54 (W = 96)
    t = [2.0, 4.0, 1.0, 3.0]
    b = [1.0, 1.0, 0.5, 0.0]
    S = _sets(t, b, idx)
    x = np.zeros(2 * W, np.float32)
    x[idx] = t
    routed = np.zeros((2, 6), bool)
    routed[0, 0] = True                           # unit 0 (expert 0) routed; 17 is expert 1; 100 -> e0; 150 -> e3
    r = T.score_sets(S, x, routed, W, 16, 6, ratio_floor=0.01)["box:x"]
    assert r["trip"] == 1.0 and abs(r["cos_contrast"] - 1.0) < 1e-9
    assert abs(r["rel_median"] - np.median([2, 4, 2, 3 / 0.03])) < 1e-6      # b = 0 -> floor 0.01 * t
    assert abs(r["rel_sum"] - 10 / 2.5) < 1e-9
    assert r["routed"] == 0.25
    x[idx] = b                                     # at the English baseline: ratio 1 where b > 0, no trip
    r = T.score_sets(S, x, None, W, 16, 6)["box:x"]
    assert r["trip"] == 0.0 and r["cos_contrast"] == 0.0 and r["routed"] is None
    assert abs(r["rel_median"] - 1.0) < 1e-9
    empty = T.Sets(["box:e"], ["box"], [["e"]], np.zeros(0, np.int64), np.zeros(0, np.float32),
                   np.zeros(0, np.float32), np.array([0, 0], np.int64))
    assert T.score_sets(empty, x, None, W, 16, 6)["box:e"]["rel_median"] is None


# ------------------------------------------------------------------ toy partition and maps


ENG = ["the quick brown fox jumps over the lazy dog and runs away into the woods tonight",
       "a small house stood near the river where children played every summer evening long",
       "people often walk along the old road to the market on sunday mornings in spring"]
DIG = ["1234 5678 9012 3456 7890 1234 5678 9012 3456 7890 2468 1357 9753 8642 1111 2222",
       "3141 5926 5358 9793 2384 6264 3383 2795 0288 4197 1693 9937 5105 8209 7494 4592",
       "2718 2818 2845 9045 2353 6028 7471 3526 6249 7757 2470 9369 9959 5749 6696 7627"]
UPP = ["WHERE IS THE STATION PLEASE TELL ME NOW BEFORE THE TRAIN LEAVES THE PLATFORM AGAIN",
       "ALL CAPS TEXT LIKE THIS IS LOUD AND HARD TO READ FOR MOST PEOPLE WHO SEE IT OFTEN",
       "THE MEETING STARTS AT NOON IN ROOM FOUR ON THE SECOND FLOOR OF THE OLD BUILDING"]


def _held(tmp, name, texts):
    p = tmp / f"{name}.jsonl"
    p.write_text("".join(json.dumps({"text": x, "split": "held"}) + "\n" for x in texts))
    return f"{name}={p}@16"


def _mean_drive(h, texts):
    cap = T.Capture(h, LAYERS, W)
    seqs = [h.encode(x)[:16] for x in texts]
    X, _, _ = cap.run(seqs, [(0, 16)])
    return X[0].mean(0)                                    # [nL, W]


def _toy_partition(tmp, h, fingerprint="toy:abc"):
    """Boxes a (digits), b (upper case) and c (random), carved from the toy's own drive like the real maps: t = drive
    on the box's texts, b = drive on English; box = the top units by t - b (disjoint), plus a pair and a family."""
    eng, dig, upp = _mean_drive(h, ENG), _mean_drive(h, DIG), _mean_drive(h, UPP)
    tgt = {"a": dig, "b": upp, "c": eng * 1.1}
    taken = np.zeros((2, W), bool)
    masks = {}
    rng = np.random.default_rng(5)
    for nm in ("a", "b", "c"):
        score = (tgt[nm] - eng) if nm != "c" else rng.random((2, W))
        score = np.where(taken, -np.inf, score)
        m = np.zeros((2, W), bool)
        for r in range(2):
            m[r, np.argsort(-score[r])[:8]] = True
        masks[nm] = m
        taken |= m
    pair = np.zeros((2, W), bool)
    pair[np.where(~taken)[0][:3], np.where(~taken)[1][:3]] = True
    taken |= pair
    grp = np.zeros((2, W), bool)
    free = np.argwhere(~taken)[:4]
    grp[free[:, 0], free[:, 1]] = True
    taken |= grp
    pd = tmp / "partition"
    pd.mkdir()
    np.savez_compressed(pd / "tiles.npz", layers=np.array(LAYERS), core=~taken, **{f"box:{k}": v for k, v in
                                                                                     masks.items()},
                        **{"pair:a__b": pair, "group:abc": grp})
    boxes = {}
    for nm in ("a", "b", "c"):
        md = tmp / "maps" / nm
        md.mkdir(parents=True)
        np.savez_compressed(md / "map.npz", **{f"mean_target_{s}": tgt[nm][s].astype(np.float32) for s in range(2)},
                            **{f"mean_baseline_{s}": eng[s].astype(np.float32) for s in range(2)},
                            **{f"target_{s}": tgt[nm][s] > eng[s] for s in range(2)})
        (md / "report.json").write_text(json.dumps({"fingerprint": fingerprint, "baseline": {"corpus": "eng.jsonl"},
                                                    "per_site": [{"layer": l, "units": W} for l in LAYERS]}))
        boxes[nm] = {"map": str(md), "units": int(masks[nm].sum())}
    (pd / "partition.json").write_text(json.dumps({
        "fingerprint": "toy:abc", "ratio": 0.95, "boxes": boxes, "groups": {"kept": {"abc": {"members": ["a", "b", "c"]}}}}))
    return pd, masks, tgt, eng


def test_build_sets_twins_and_member_maps(tmp_path):
    h = _host()
    pd, masks, tgt, eng = _toy_partition(tmp_path, h)
    import amoeba_box_battery as BB
    tz = BB.Tiles(pd / "tiles.npz")
    prep = json.loads((pd / "partition.json").read_text())
    # box c is the English box here: its map's baseline is another corpus (lang.eng: the 10-language mix)
    cz = dict(np.load(tmp_path / "maps" / "c" / "map.npz"))
    for s_ in range(2):
        cz[f"mean_baseline_{s_}"] = cz[f"mean_baseline_{s_}"] * 0.5
    np.savez_compressed(tmp_path / "maps" / "c" / "map.npz", **cz)
    prep["english_box"] = {"name": "c"}
    sets, maps = T.build_sets([("main", prep, tz)], W, 0, log=lambda *_: None, twins=3)
    S = sets["main"]
    assert set(maps) == {"a", "b", "c"} and maps["a"]["baseline_corpus"] == "eng.jsonl"
    tw = T.twins_of(S)
    tf_ = {nm: tgt[nm].reshape(-1) for nm in tgt}
    ef = eng.reshape(-1)
    for nm in ("a", "b", "c"):
        i = S.pos[nm]
        own = S.idx[S.off[i]:S.off[i + 1]]
        assert np.array_equal(own, np.flatnonzero(masks[nm]))
        assert len(tw[("null", nm)]) == 3 and len(tw[("nullsel", nm)]) == 3
        draws = []
        for n in tw[("null", nm)]:
            j = S.pos[n]
            twin = S.idx[S.off[j]:S.off[j + 1]]
            draws.append(tuple(twin))
            assert not set(own) & set(twin)
            assert np.array_equal(np.bincount(own // W, minlength=2), np.bincount(twin // W, minlength=2))
            assert np.allclose(S.t[S.off[j]:S.off[j + 1]], tf_[nm][twin])               # the box's own map
        assert len(set(draws)) == 3                                                      # distinct seeded draws
        for n in tw[("nullsel", nm)]:
            k = S.pos[n]
            sel = S.idx[S.off[k]:S.off[k + 1]]
            bsel = S.b[S.off[k]:S.off[k + 1]]
            assert sel.size and not set(own) & set(sel)                                 # selected twin
            assert np.all(tf_[nm][sel] > bsel) and np.all(bsel <= 0.95 * tf_[nm][sel])
            assert np.all(np.bincount(sel // W, minlength=2) <= np.bincount(own // W, minlength=2))
            assert np.allclose(S.t[S.off[k]:S.off[k + 1]], tf_[nm][sel])
    # a tile's t: the mean over the topic members whose map selects the unit (all topic members where none does);
    # b: the topic members' mean. The English box (c here) enters neither
    for key, mem in (("pair:a__b", ["a", "b"]), ("group:abc", ["a", "b"])):
        i = S.pos[key]
        u = S.idx[S.off[i]:S.off[i + 1]]
        selm = {m_: (tf_[m_][u] > ef[u] * (0.5 if m_ == "c" else 1.0)) &
                (ef[u] * (0.5 if m_ == "c" else 1.0) <= 0.95 * tf_[m_][u]) for m_ in mem}
        ns = sum(selm[m_].astype(int) for m_ in mem)
        tsel = sum(np.where(selm[m_], tf_[m_][u], 0) for m_ in mem)
        want = np.where(ns > 0, tsel / np.maximum(ns, 1), sum(tf_[m_][u] for m_ in mem) / len(mem))
        assert np.allclose(S.t[S.off[i]:S.off[i + 1]], want, atol=1e-6)
        assert np.allclose(S.b[S.off[i]:S.off[i + 1]], ef[u], atol=1e-6)               # c's baseline left out
    assert S.members[S.pos["group:abc"]] == ["a", "b", "c"]
    capped = T.build_sets([("main", prep, tz)], W, 0, log=lambda *_: None, twins=2, twin_cap=8)[0]["main"]
    for n in T.twins_of(capped)[("null", "a")]:
        assert capped.off[capped.pos[n] + 1] - capped.off[capped.pos[n]] <= 8              # a big box: subsampled
    bad = json.loads((tmp_path / "maps" / "a" / "report.json").read_text())
    bad["fingerprint"] = "other"
    (tmp_path / "maps" / "a" / "report.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="fingerprint"):
        T.build_sets([("main", prep, tz)], W, 0, log=lambda *_: None)


def test_twin_z_ranking_and_null_bands():
    S = T.Sets(["x", "null:x#0", "null:x#1", "null:x#2"], ["box", "null", "null", "null"], [["x"]] * 4,
               np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(5, np.int64))
    sc = {"x": {"units": 700, "progress": 1.0, "cos_contrast": 0.5, "trip": 0.2, "rel_sum": 3.0},
          **{f"null:x#{j}": {"units": 700, "progress": v, "cos_contrast": 0.1, "trip": 0.1, "rel_sum": 1.0}
             for j, v in enumerate((0.0, 0.2, 0.4))}}
    T.add_twin_z(S, sc)
    assert abs(sc["x"]["gap_random_progress"] - 0.8) < 1e-9
    assert abs(sc["x"]["z_random_progress"] - 0.8 / 0.2) < 1e-9 and sc["x"]["p_random_progress"] == 0.25
    assert abs(sc["x"]["z_random_cos_contrast"] - 0.4 / T.SD_FLOOR["cos_contrast"]) < 1e-9     # sd 0: floored
    assert sc["x"]["gap_selected_progress"] is None                                   # no selected twins here
    nl = T.null_summary(sc, S, "null")["progress"]["640-10000"]
    assert nl["boxes_above_null_max"] == ["x"] and nl["boxes_above_own_twin_mean"] == 1
    sc["y"] = {"units": 5, "progress": 9.0}
    S2 = T.Sets(S.names + ["y"], S.kinds + ["box"], S.members + [["y"]], S.idx, S.t, S.b, np.zeros(6, np.int64))
    assert [r[0] for r in T.ranked(sc, S2, "box", "progress", 5, min_units=640)] == ["x"]
    assert T.ranked(sc, S2, "box", "progress", 5)[0] == ["y", 9.0, 5]
    ref = {16: {"tokens": 18.0, "boxes": {"x": {"progress": [0.5, 0.25]}}},
           256: {"tokens": 256, "boxes": {"x": {"progress": [0.0, 0.01]}}}}
    _, used = T.add_english_z(S2, sc, ref, 20)
    assert used == {"key": 16, "tokens": 18.0} and abs(sc["x"]["z_english_progress"] - 2.0) < 1e-9
    T.add_english_z(S2, sc, ref, 300)
    assert abs(sc["x"]["z_english_progress"] - 1.0 / T.SD_FLOOR["progress"]) < 1e-9          # sd floored


def _fake_ops(monkeypatch):
    from research.amoeba.carve import ops as O
    calls = []
    monkeypatch.setattr(O, "prepare_wiring", lambda spec: calls.append("prepare_wiring") or {})
    monkeypatch.setattr(O, "footprint_watchdog", lambda *a, **k: calls.append(("watchdog", k.get("watcher"))))
    monkeypatch.setattr(O, "setup_mlx", lambda *a, **k: calls.append("setup_mlx"))
    monkeypatch.setattr(O, "wire_weights", lambda spec, log, memory_gb=None: calls.append("wire") or
                        {"wired": True, "spec": spec})
    monkeypatch.setattr(O, "git_rev", lambda: "toyrev123")
    return calls


def test_end_to_end_ranks_own_box_and_writes_receipt(tmp_path, monkeypatch):
    h = _host()
    pd, *_ = _toy_partition(tmp_path, h)
    calls = _fake_ops(monkeypatch)
    held = [_held(tmp_path, "english", ENG), _held(tmp_path, "a", DIG), _held(tmp_path, "b", UPP)]
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "raw", "category": "x", "prompt": DIG[0]}) + "\n" +
                       json.dumps({"id": "chat", "category": "y", "messages": [{"role": "user", "content": UPP[1]}],
                                   "tools": [{"type": "function", "function": {"name": "f"}}]}) + "\n")
    out = tmp_path / "out"
    argv = ["--model", "toy", "--partition", f"main={pd}", "--partition", f"again={pd}", "--out", str(out),
            "--prompts", str(prompts), "--gen", "5", "--controls", "a,b", "--control-docs", "3",
            "--english-docs", "3", "--english-lengths", "4,8", "--top", "3", "--batch", "2", "--wired-gb", "auto",
            "--min-units", "1", "--twins", "3", "--english-chat-lengths", "4,8,16"]
    for x in held:
        argv += ["--held", x]
    assert T.main(argv, host_factory=lambda m: h) == 0
    assert calls[:2] == ["prepare_wiring", ("watchdog", True)] and "wire" in calls
    lines = [json.loads(x) for x in (out / "trip.jsonl").read_text().split("\n") if x.strip()]
    rec = lines[0]["receipt"]
    assert rec["checkpoint"] == "Toy@abc" and rec["fingerprint"] == "toy:abc"
    assert rec["mlx2_source_revision"] == "toyrev123"
    assert set(rec["partitions"]) == {"main", "again"} and rec["partitions"]["main"]["resolved"] == str(pd.resolve())
    assert len(rec["partitions"]["main"]["partition_json_sha256_12"]) == 12
    assert set(rec["maps"]) == {"a", "b", "c"} and rec["moe"] == {"layers": LAYERS, "E": 6, "F": 16,
                                                                  "unit": "expert * F + neuron"}
    rows = lines[1:]
    assert [r["id"] for r in rows] == ["raw", "chat"]
    for r in rows:
        assert r["continuation_tokens"] == 5 and r["decode_route"] == "recompute" and r["parity"] == 1.0
        assert set(r["segments"]) == {"prompt", "continuation"}
        seg = r["segments"]["prompt"]["main"]
        assert set(seg["boxes"]) == {"a", "b", "c"} and set(seg["top"]) == set(T.TOP_METRICS)
        assert "pair" in seg["tiles_top"] and "group" in seg["tiles_top"] and seg["routing"]["top"]
        band = next(iter(seg["null"]["random"]["progress"].values()))
        assert set(band) >= {"p50", "p99", "max", "boxes_above_null_max"}
        assert "selected" in seg["null"] and len(seg["null_twins"]["random"]) == 9
        assert seg["english_ref_tokens"]["key"] in (4, 8, 16) and seg["boxes"]["a"]["z_english_progress"] is not None
        assert seg["boxes"]["a"]["gap_random_progress"] is not None
        assert all(u < 1 for _n, _v, u in seg["small_boxes_top"]["progress"])     # --min-units 1: none smaller
    assert rows[0]["render"] == "raw" and rows[1]["render"] == "tokenizer.apply_chat_template"
    assert rows[0]["segments"]["prompt"]["main"]["top"]["cos_contrast"][0][0] == "a"
    ctrl = json.loads((out / "controls.json").read_text())["main"]
    for c in ("a", "b"):
        assert ctrl["positive"][c]["rank_of_mean"]["cos_contrast"]["own"] == 1
        assert ctrl["positive"][c]["rank_of_mean"]["progress"]["own"] == 1
    assert np.isfinite(ctrl["positive"]["a"]["random_twin_progress"])     # the toy is too small for a quiet null
    en = ctrl["english"]
    assert abs(en["topic_rel_sum_median"] - 1.0) < 0.05 and abs(en["topic_progress_median"]) < 0.05
    assert en["lengths"] == [4, 8, 16] and set(map(int, en["reference"]["raw"])) == {4, 8, 16}
    assert set(map(int, en["reference"]["chat_prompt"])) == {4, 8, 16} and en["chat_reference"]["docs"] == 3
    md = (out / "report.md").read_text()
    assert "toy:abc" in md and "toyrev123" in md and "## raw (x)" in md and "control box" in md
    r2 = json.loads((out / "receipt.json").read_text())
    assert r2["prompts"] == 2 and "mlx_peak_gb" in r2 and r2["causal_check"]["ok"]
    assert r2["causal_check"]["max_abs_diff"] < 1e-5 and "Causal check" in md
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        T.main(argv, host_factory=lambda m: h)
    # the statistics cache is reused when nothing changed and rebuilt when a map changes
    sc = tmp_path / "sc" / "box-stats"
    built = []
    real = T.build_sets
    monkeypatch.setattr(T, "build_sets", lambda *a_, **k_: built.append(1) or real(*a_, **k_))

    def run(o):
        args = [x if x != str(out) else str(tmp_path / o) for x in argv] + ["--stats-cache", str(sc)]
        assert T.main(args, host_factory=lambda m: h) == 0
        return json.loads((tmp_path / o / "receipt.json").read_text())["stats_id"]
    s2 = run("out2")
    s3 = run("out3")
    assert s2 == s3 and len(built) == 1
    rp = tmp_path / "maps" / "b" / "report.json"
    os.utime(rp, (rp.stat().st_atime, rp.stat().st_mtime + 5))
    assert run("out4") != s3 and len(built) == 2


def test_fingerprint_mismatch_and_missing_controls_refused(tmp_path, monkeypatch):
    h = _host()
    pd, *_ = _toy_partition(tmp_path, h)
    _fake_ops(monkeypatch)
    base = ["--model", "toy", "--partition", str(pd), "--out", str(tmp_path / "o"), "--gen", "0"]
    with pytest.raises(SystemExit, match="no held spec"):
        T.main(base + ["--controls", "a"], host_factory=lambda m: h)
    other = Host(model=qn_toy(), tokenizer=ChatTok(), fingerprint="toy:zzz", log=lambda *_: None, moe_units="neuron")
    with pytest.raises(SystemExit, match="partition of toy:abc"):
        T.main(base + ["--controls", "", "--english-docs", "0"], host_factory=lambda m: other)


def test_default_prompts_cover_the_requested_categories():
    rows = T.load_prompts("")
    cats = {r["category"] for r in rows}
    assert {"code", "math", "legal", "finance", "religion", "physics", "cyber", "language", "chat", "tool",
            "multiturn", "cooking", "games"} <= cats
    assert all(2 <= sum(r["category"] == c for r in rows) <= 4 for c in cats)
    assert any(r.get("tools") for r in rows) and len({r["id"] for r in rows}) == len(rows)


def test_user_span_and_template_ids():
    class H:
        def encode(self, s):
            return {"<|im_start|>user\n": [1, 2], "<|im_end|>": [9]}[s]
    ids = [4, 1, 2, 5, 6, 9, 3, 1, 2, 7, 8, 9, 3]
    assert T.user_span(H(), ids) == (9, 11)                          # the last user message's content
    assert T.user_span(H(), [4, 5, 6]) is None
    assert T._ids_of({"input_ids": [3, 4]}) == [3, 4] and T._ids_of([[5, 6]]) == [5, 6] and T._ids_of((7,)) == [7]


def test_joint_tiles_read_lazily_and_checked(tmp_path):
    h = _host()
    pd, *_ = _toy_partition(tmp_path, h)
    z = dict(np.load(pd / "tiles.npz"))
    core = z["core"].copy()
    jt = np.zeros_like(core)
    jt[np.argwhere(core)[:5, 0], np.argwhere(core)[:5, 1]] = True
    jd = tmp_path / "joint"
    jd.mkdir()
    np.savez_compressed(jd / "tiles.npz", **{**z, "core": core & ~jt, "joint:a+b+c": jt})
    prep = json.loads((pd / "partition.json").read_text())
    jprep = {**prep, "joint": {"tiles": {"a+b+c": {"members": ["a", "b", "c"], "units": 5}}}}
    (jd / "partition.json").write_text(json.dumps(jprep))
    base = ["--model", "toy", "--partition", f"main={pd}", "--out", str(tmp_path / "o"), "--prepare-only",
            "--joint-partition", str(jd), "--stats-cache", str(tmp_path / "sc" / "s")]
    assert T.main(base) == 0
    sid = json.loads((tmp_path / "sc" / "s").read_text())["id"]
    S = T.Sets.load(tmp_path / "sc" / "s.main.npz", sid)
    i = S.pos["joint:a+b+c"]
    assert np.array_equal(S.idx[S.off[i]:S.off[i + 1]], np.flatnonzero(jt)) and S.members[i] == ["a", "b", "c"]
    jprep["boxes"] = {**prep["boxes"], "a": {**prep["boxes"]["a"], "units": 999}}
    (jd / "partition.json").write_text(json.dumps(jprep))
    with pytest.raises(SystemExit, match="box unit counts"):
        T.main(base)


FN = Path.home() / "mlx-models" / "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"


@pytest.mark.skipif(not (FN / "tokenizer.json").exists(), reason="the FN artifact's tokenizer is not on this machine")
def test_user_span_on_the_fn_template():
    """The FN chat template (the adapter's chat_template, tokenizer only, no weights): the user-text span is the last
    user message's content, for plain, tool and multi-turn rows."""
    from transformers import AutoTokenizer
    from mlx2.adapters.flash_next import chat_template
    tok = AutoTokenizer.from_pretrained(str(FN))

    class H:
        def encode(self, s):
            return tok.encode(s, add_special_tokens=False)
    for r in T.load_prompts(""):
        req = {"messages": r["messages"], "enable_thinking": False, **({"tools": r["tools"]} if r.get("tools") else {})}
        ids = tok.encode(chat_template(tok, req, tokenize=False), add_special_tokens=False)
        sp = T.user_span(H(), ids)
        assert sp is not None, r["id"]
        assert tok.decode(ids[sp[0]:sp[1]]) == r["messages"][-1]["content"], r["id"]


def test_report_numbers_are_none_safe():
    assert T._f(None) == "n/a" and T._f(float("nan")) == "n/a" and T._f(0.12345) == "0.123"
    assert T._f(1234.5, ".3g") == "1.23e+03"
