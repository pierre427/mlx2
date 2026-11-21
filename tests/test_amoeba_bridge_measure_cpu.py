"""CPU tests for the host-neuron reuse v4 §5.0 window: epoch selection per arm, the contrast teacher, the topic
selection / split manifest, paired statistics, and the donor-gap screen's fit split. No model is loaded."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import amoeba_bridge_lab as L  # noqa: E402
import amoeba_paired_stats as PS  # noqa: E402


class Poisoned(dict):
    """an epoch record whose donor-derived entry must never be read"""

    def __getitem__(self, k):
        if k == "val_kl":
            raise AssertionError("selection read val_kl (a donor output)")
        return super().__getitem__(k)

    def get(self, k, d=None):
        if k == "val_kl":
            raise AssertionError("selection read val_kl (a donor output)")
        return super().get(k, d)


def test_sft_arms_select_by_text_cross_entropy_and_never_read_donor_outputs():
    for arm in ({"route": "tile", "loss": "sft"}, {"route": "tile", "loss": "sft", "kind": "lora"}):
        assert L.selection_metric(arm, "val_kl") == "val_ce"
        assert L.selection_metric(arm, "val_ce") == "val_ce"
    # val_kl would pick epoch 1, val_ce picks epoch 2; epoch 3 has the best CE but breaks the drift cap
    hist = [Poisoned(epoch=0, val_kl=0.30, val_ce=0.520, val_drift=0.0),
            Poisoned(epoch=1, val_kl=0.10, val_ce=0.519, val_drift=0.01),
            Poisoned(epoch=2, val_kl=0.20, val_ce=0.510, val_drift=0.02),
            Poisoned(epoch=3, val_kl=0.05, val_ce=0.400, val_drift=0.20)]
    assert L.pick_epoch(hist, L.selection_metric({"route": "tile", "loss": "sft"}), 0.05) == (2, True)
    plain = [dict(h) for h in hist]
    assert L.pick_epoch(plain, "val_kl", 0.05) == (1, True)
    # nothing within the cap: the lowest-drift epoch, still without reading val_kl
    over = [Poisoned(epoch=0, val_kl=1, val_ce=0.5, val_drift=0.3), Poisoned(epoch=1, val_kl=1, val_ce=0.4, val_drift=0.2)]
    assert L.pick_epoch(over, "val_ce", 0.05) == (1, False)


def test_kl_and_init_arms_selection_metric_is_explicit():
    assert L.selection_metric({"route": "tile", "loss": "kl"}) == "val_kl"
    assert L.selection_metric({"route": "tile", "loss": "kl"}, "val_ce") == "val_ce"
    assert L.selection_metric({"route": "bridge", "src": "box"}) == "val_kl"
    assert L.selection_metric({"route": "tile", "loss": None}) is None


def _fn_source(src, name):
    lines = src.splitlines()
    i = next(j for j, ln in enumerate(lines) if ln.strip().startswith(f"def {name}("))
    ind = len(lines[i]) - len(lines[i].lstrip())
    out = [lines[i]]
    for ln in lines[i + 1:]:
        if ln.strip() and len(ln) - len(ln.lstrip()) <= ind:
            break
        out.append(ln)
    return "\n".join(out)


def test_harness_train_selects_through_pick_epoch_and_val_ce_reads_no_donor():
    src = (ROOT / "scripts" / "amoeba_bridge_lab.py").read_text()
    tr = _fn_source(src, "train")
    assert "metric = selection_metric(arm, args.kl_arm_select)" in tr
    assert "pick_epoch(hist, metric, args.drift_cap)" in tr
    assert "best = (kv" not in tr  # the old val-KL-for-every-arm selection is gone
    vc = _fn_source(src, "val_ce")
    assert "t_lp" not in vc and "Dn" not in vc and "dfeat" not in vc
    assert '"selection_metric": metric' in tr


def _rows(rng, T, V):
    x = rng.normal(size=(T, V)) * 3
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def test_contrast_teacher_numpy():
    rng = np.random.default_rng(0)
    lh, ld, lm = (_rows(rng, 5, 50) for _ in range(3))
    q0 = L.contrast_teacher(np, lh, ld, lm, 0.0)
    assert np.allclose(q0, lh, atol=1e-10)
    for mode in ("clip", "topk"):
        q = L.contrast_teacher(np, lh, ld, lm, 1.5, mode, clip=0.5, k=4)
        assert np.allclose(np.exp(q).sum(-1), 1.0)
        d = q - lh
        assert np.all(d.max(-1) - d.min(-1) <= 2 * 1.5 * 0.5 + 1e-9)  # the clip bounds the reweighting
    q = L.contrast_teacher(np, lh, ld, lm, 1.0, "topk", clip=0, k=3)
    kd = np.sort(ld, -1)[:, -3][:, None]
    km = np.sort(lm, -1)[:, -3][:, None]
    out = ~((ld >= kd) | (lm >= km))
    d = q - lh
    for t in range(5):  # off the union the teacher is the host up to one normaliser per row
        assert np.ptp(d[t][out[t]]) < 1e-10
    assert out.sum(-1).min() >= 50 - 6
    with pytest.raises(ValueError):
        L.contrast_teacher(np, lh, ld, lm, 1.0, "bogus")


def test_contrast_teacher_mlx_matches_numpy():
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(1)
    lh, ld, lm = (_rows(rng, 4, 40).astype(np.float32) for _ in range(3))
    for mode in ("clip", "topk"):
        a = L.contrast_teacher(np, lh, ld, lm, 0.75, mode, 2.0, 5)
        b = np.array(L.contrast_teacher(mx, mx.array(lh), mx.array(ld), mx.array(lm), 0.75, mode, 2.0, 5))
        assert np.allclose(a, b, atol=1e-4)


def _corpus():
    rows = []
    for i in range(60):
        rows.append({"id": f"f{i}", "split": "fit", "topics": ["sci.physics"], "group": f"fg{i // 2}",
                     "text": "w " * (10 if i % 5 else 3)})
    for i in range(50):
        rows.append({"id": f"h{i}", "split": "held", "topics": ["sci.physics.optics"], "group": f"hg{i // 2}",
                     "text": "w " * (10 if i % 7 else 2)})
    rows.append({"id": "x0", "split": "held", "topics": ["math"], "group": "m", "text": "w " * 10})
    return rows


def enc(t):
    return list(range(len(t.split())))


def test_topic_selection_exclusion_after_the_draw_keeps_the_selection():
    rows = _corpus()
    base = L.topic_selection(rows, "sci.physics", 7, 16, 4, 10, 8, enc)
    assert [it["split"] for it in base].count("test") == 10
    assert sum(it["split"] in ("train", "val") for it in base) == 20
    picked = {it["probe_id"] for it in base}
    unreached = [r["id"] for r in rows if r["split"] == "held" and r["id"] not in picked and r["id"] != "x0"]
    same = L.topic_selection(rows, "sci.physics", 7, 16, 4, 10, 8, enc, exclude_ids=unreached)
    # excluded texts beyond the draw's reach leave the selection unchanged, id for id
    assert [(it["probe_id"], it["split"]) for it in same] == [(it["probe_id"], it["split"]) for it in base]
    t0 = next(it for it in base if it["split"] == "test")
    moved = L.topic_selection(rows, "sci.physics", 7, 16, 4, 10, 8, enc, exclude_groups=[t0["group"]])
    assert t0["probe_id"] not in {it["probe_id"] for it in moved}
    man = {"selection": {k: [it["probe_id"] for it in base if it["split"] == k] for k in ("train", "val", "test")}}
    assert L.split_check(man, base) == ""
    assert "test" in L.split_check(man, moved)
    ids, groups = L.split_exclusions({"reserved_final_test": {"probe_ids": ["a"], "groups": [1]},
                                      "screen_slice": {"probe_ids": ["b"], "groups": ["g"]}})
    assert ids == {"a", "b"} and groups == {"1", "g"}


def test_split_manifest_reserves_untouched_groups_only():
    import amoeba_bridge_split_manifest as M
    rows = _corpus()
    a = {"seed": 7, "n_train": 16, "n_val": 4, "n_test": 10, "topic_tokens": 8}
    man = M.build(rows, "sci.physics", a, enc, {"host_checkpoint": "h@1"})
    sel = man["selection"]
    res = set(man["reserved_final_test"]["probe_ids"])
    used = set(sel["test"])
    used_groups = {r["group"] for r in rows if r["id"] in used}
    assert not res & used
    assert all(r["group"] not in used_groups for r in rows if r["id"] in res)
    assert "x0" not in res  # another topic
    held_e = [r for r in rows if r["split"] == "held" and r["topics"][0].startswith("sci.physics")
              and len(r["text"].split()) >= 8]
    assert man["counts"]["held_eligible"] == len(held_e)
    assert man["counts"]["held_unused"] == len(held_e) - 10
    # the harness skips every reserved text: selection unchanged with the manifest's exclusions
    ids, groups = L.split_exclusions(man)
    again = L.topic_selection(rows, "sci.physics", 7, 16, 4, 10, 8, enc, exclude_ids=ids, exclude_groups=groups)
    assert L.split_check(man, again) == ""


def test_committed_physics_manifest_matches_the_review_counts():
    p = ROOT / "docs/research/amoeba-bridge/physics-split-manifest-bridgeT2-8acf5a186.json"
    man = json.loads(p.read_text())
    c = man["counts"]
    assert (c["selected_train"], c["selected_val"], c["selected_test"]) == (160, 40, 200)
    assert (c["fit_eligible"], c["held_eligible"], c["held_unused"]) == (999, 508, 308)
    assert (c["reserved_texts"], c["reserved_groups"]) == (297, 287)
    assert not set(man["reserved_final_test"]["probe_ids"]) & set(man["selection"]["test"])
    assert man["host_checkpoint"].startswith("Studio-Operator-Qwen3.5-9B-4bit@07e2d06a")
    assert man["donor_checkpoint"].startswith("Qwen3.8-27B-BF16-q8-mlx2@2dd8d304")


def test_paired_stats_resolution_and_flags():
    rng = np.random.default_rng(0)
    d = 0.003 + 0.001 * rng.normal(size=400)
    groups = [i // 2 for i in range(400)]
    r = PS.paired(d, groups, mue=0.001)
    assert abs(r["paired_se"] - d.std(ddof=1) / 20) < 1e-12
    assert abs(r["mde80"] - 2.8016 * r["paired_se"]) < 1e-6
    assert r["ci95_group"][0] < 0.003 < r["ci95_group"][1]
    assert r["vs_min_useful_effect"] == "above"
    assert PS.flag(-0.001, 0.0005, 0.001) == "below"
    assert PS.flag(0.0005, 0.002, 0.001) == "spanning"
    assert PS.paired(np.zeros(5), None, mue=None)["vs_min_useful_effect"] is None
    rt = PS.ratio(np.full(50, 1.0), np.full(50, 4.0), list(range(50)))
    assert rt["estimate"] == 0.25 and rt["ci95_group"] == [0.25, 0.25]


def test_gap_screen_fit_split_slice_and_exclusions(tmp_path):
    import amoeba_topics_donor_gap as G
    import amoeba_topics_lib as T
    items = _corpus()
    ex = tmp_path / "man.json"
    ex.write_text(json.dumps({"selection": {"train": ["f0"], "val": [], "test": []},
                              "reserved_final_test": {"probe_ids": [], "groups": ["fg5"]}}))

    class A:
        split, fit_slice, exclude_manifest, min_tokens = "fit", "1/3", str(ex), None
    got = G.select_split(items, A)
    assert got and all(it["split"] == "fit" for it in got)
    assert all(G.fit_slice_of(it["group"], 3) == 1 for it in got)      # the salted slice hash (AMCO-1)
    assert all(it["group"] not in ("fg0", "fg5") for it in got)  # f0's whole group goes too
    assert G.min_tokens(A) == 64
    A.split, A.fit_slice, A.exclude_manifest = "held", "", ""
    assert G.min_tokens(A) == 32 and all(it["split"] == "held" for it in G.select_split(items, A))
    A.fit_slice = "0/2"
    with pytest.raises(SystemExit):
        G.select_split(items, A)
    with pytest.raises(SystemExit):
        G.parse_slice("3/3")


def test_gap_compare_reports_each_models_loss_and_refuses_mixed_splits(tmp_path):
    rng = np.random.default_rng(0)
    for name, better, split in (("host", 0.0, "fit"), ("donor", 0.1, "fit"), ("donor2", 0.1, "held")):
        d = tmp_path / name
        d.mkdir()
        rows = []
        for i in range(60):
            topic = "prog.python" if i < 30 else "lang.fra"
            gain = better if topic == "prog.python" else 0.0
            for cond in ("bare", "prefix"):
                rows.append({"id": f"t{i}", "cond": cond, "topics": [topic], "group": f"g{i}",
                             "nats": float(100 * (0.8 - gain + 0.01 * rng.normal())), "bytes": 100, "tokens": 60})
        (d / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        sp = {"split": split, "fit_slice": "0/4" if split == "fit" else None, "exclude_manifest": None,
              "min_tokens": 64 if split == "fit" else 32}
        (d / "score.json").write_text(json.dumps({"checkpoint": name + "@0", "corpus_sha256": "x", "split": sp}))
    py = [sys.executable, str(ROOT / "scripts/amoeba_topics_donor_gap.py"), "compare", "--boot", "300"]
    r = subprocess.run(py + ["--host", str(tmp_path / "host"), "--donor", str(tmp_path / "donor"),
                             "--out", str(tmp_path / "out")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads((tmp_path / "out/donor_gap.json").read_text())
    t = rep["topics"]["prog.python"]["bare"]
    assert abs(t["host_nats_per_byte"] - 0.8) < 0.01 and abs(t["donor_nats_per_byte"] - 0.7) < 0.01
    assert t["host_ci95"][0] <= t["host_nats_per_byte"] <= t["host_ci95"][1]
    assert rep["split"]["split"] == "fit"
    r = subprocess.run(py + ["--host", str(tmp_path / "host"), "--donor", str(tmp_path / "donor2"),
                             "--out", str(tmp_path / "out2")], capture_output=True, text=True)
    assert r.returncode != 0 and "different splits" in (r.stderr + r.stdout)


def test_trained_twins_share_text_order_and_flag_untrained_contrasts():
    src = (ROOT / "scripts" / "amoeba_bridge_lab.py").read_text()
    tr = _fn_source(src, "train")
    assert "order_rng = np.random.default_rng(args.seed + 501)" in tr and "rng.permutation(len(trs))" not in tr.replace(
        "order_rng.permutation(len(trs))", "")
    assert '"reverted_to_init": ck_sel == 0' in tr
    assert "uninformative: {bad_} reverted to init or stayed outside the drift cap" in src


def test_screen_manifest_carries_the_reserved_final_test(tmp_path):
    import amoeba_topics_donor_gap as G
    ex = tmp_path / "man.json"
    ex.write_text(json.dumps({"reserved_final_test": {"probe_ids": ["h3"], "groups": ["hg1"]}}))
    rec = {"checkpoint": "h@0", "corpus_sha256": "x", "mlx2_source_revision": "r",
           "split": {"split": "fit", "exclude_manifest": str(ex)}}
    m = G.screen_manifest(rec, [{"id": "f1", "group": "fg0"}, {"id": "f2", "group": "fg0"}])
    assert m["screen_slice"]["probe_ids"] == ["f1", "f2"] and m["screen_slice"]["groups"] == ["fg0"]
    ids, groups = L.split_exclusions(m)
    assert ids == {"f1", "f2", "h3"} and groups == {"fg0", "hg1"}


def test_watchdog_is_memwatch_with_watcher(tmp_path, monkeypatch):
    """The bridge lab's --watchdog-gb now runs research/amoeba/carve/memwatch.py (ops.footprint_watchdog) with its
    SIGKILL watcher and a stop receipt, not amoeba27b_lib's copy that missed the 2026-11-19 panic; a memwatch stop
    line (ending in memwatch.STOP_MARKER, asserted on the real thread in test_amoeba_memwatch_cpu.py) still leaves
    a failed run.json."""
    from research.amoeba.carve import ops as CO
    seen = {}

    def fake(limit, log, **k):
        seen.update(limit=limit, **k)
        from research.amoeba.carve.memwatch import STOP_MARKER
        log(f"memory watchdog: footprint 99.0 GiB > 80.0 GiB; {STOP_MARKER}")   # memwatch's own stop line form
        raise SystemExit(3)                     # stands in for os._exit(3)
    monkeypatch.setattr(CO, "footprint_watchdog", fake)
    out = tmp_path / "o"
    monkeypatch.setattr(sys, "argv", ["bridge", "--out", str(out), "--watchdog-gb", "80", "--swap-growth-gb", "2"])
    with pytest.raises(SystemExit):
        L.main()
    assert seen["limit"] == 80 and seen["watcher"] is True and seen["receipt"] == out / "watchdog-stop.json"
    rec = json.loads((out / "run.json").read_text())
    assert rec["status"] == "failed (watchdog)" and "stopping (exit 3)" in rec["reason"]
