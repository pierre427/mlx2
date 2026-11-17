"""CPU tests of the mlx2 amoeba pipeline package (no model load): config, plan, receipts/resume, progress and
Prometheus metrics, partition rules, verdicts, detector maths, the GPU-window guard and the CLI dispatch."""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from mlx2.experimental.amoeba import config as K
from mlx2.experimental.amoeba import plan as P
from mlx2.experimental.amoeba import runner as R
from mlx2.experimental.amoeba import stages as S
from mlx2.experimental.amoeba.child_boxbatch import verdict
from mlx2.experimental.amoeba.child_probe import train_logistic, window_score
from mlx2.experimental.amoeba.inspection import _moe_layers, artifact_name
from mlx2.experimental.amoeba.progress import EVENT_PREFIX, MetricsServer, Progress
from mlx2.experimental.amoeba.state import Build


def info(weights_gb=1.4, kind="dense", L=28, d=1024, F=3072, V=151936):
    return {"path": "/models/toy", "checkpoint": "toy@00000000", "fingerprint": "0" * 64, "adapter": "Toy", "kind": kind, "layers": L,
            "hidden": d, "intermediate": F, "vocab": V, "moe": None, "weight_bytes": int(weights_gb * 2 ** 30),
            "file_backed_ple": False,
            "units": {"dense_neurons": L * F, "moe_expert_units": 0, "moe_neuron_units": 0, "estimate": True}}


# ------------------------------------------------------------------ config

def test_profiles_keep_data_rules_fixed():
    full, quick = K.resolve("full", {}, {}, ""), K.resolve("quick", {}, {}, "")
    assert full["data"] == quick["data"] == K.DATA_DEFAULTS
    assert full["data"]["window_tokens"] == 128 and full["data"]["gen_samples"] == 3
    assert full["data"]["gen_temperature"] == 0.7 and full["data"]["strip_prefix"] == 0.2
    assert full["verdict"]["held_target_nats"] == 0.05 and full["verdict"]["val_target_nats"] == 0.03


def test_data_override_needs_reason_and_known_keys():
    with pytest.raises(SystemExit, match="override-reason"):
        K.resolve("full", {}, {"window_tokens": 64}, "")
    with pytest.raises(SystemExit, match="unknown data rules"):
        K.resolve("full", {}, {"nope": 1}, "why")
    with pytest.raises(SystemExit, match="unknown compute"):
        K.resolve("full", {"nope": 1}, {}, "")
    c = K.resolve("full", {"box_steps": 30}, {"window_tokens": 96}, "box X missed: longer windows")
    assert c["compute"]["box_steps"] == 30 and c["data"]["window_tokens"] == 96
    assert c["data_overrides"]["reason"].startswith("box X")
    assert K.parse_overrides(["a=1", "b=0.5", "c=true", "d=x"]) == {"a": 1, "b": 0.5, "c": True, "d": "x"}


def test_cluster_k_range():
    assert K.cluster_k(86016) == 50 and K.cluster_k(64 * 17408) == 130
    assert 50 <= K.cluster_k(300_000) <= 130


# ------------------------------------------------------------------ plan

def test_plan_fits_small_and_refuses_big(monkeypatch):
    monkeypatch.setattr(P, "physical_gb", lambda: 128.0)
    cfg = K.resolve("full", {}, {}, "")
    p = P.make_plan(info(), cfg)
    assert p["fits"] and p["cap_gb"] == 80.0 and p["batches"]["carve_core"]["bs"] == 2
    big = P.make_plan(info(weights_gb=79), cfg, stream="resident")      # never streams: refuses
    assert not big["fits"] and any("streaming" in r for r in big["refusals"])
    auto = P.make_plan(info(weights_gb=79), cfg)                        # auto picks rolling; no byte map: refuses
    assert not auto["fits"] and auto["stream"]["mode"] == "rolling" and "rolling" in auto["refusals"][0]
    with pytest.raises(SystemExit, match="cap-reason"):
        P.make_plan(info(), cfg, cap_gb=92)
    ok = P.make_plan(info(weights_gb=70), cfg, cap_gb=92, cap_reason="Pierre: FN 92 GB")
    assert ok["cap_gb"] == 92.0 and ok["cap_reason"]
    assert "fits" in P.format_plan(info(), p)


def test_plan_reduces_batch_before_refusing(monkeypatch):
    monkeypatch.setattr(P, "physical_gb", lambda: 128.0)
    cfg = K.resolve("full", {}, {}, "")
    i = info(weights_gb=69, L=64, d=5120, F=17408)
    p = P.make_plan(i, cfg)
    assert p["batches"]["capture"]["bs"] < 8 and any("capture: batch 8 ->" in n for n in p["notes"])


def test_plan_caps_at_physical_minus_headroom(monkeypatch):
    monkeypatch.setattr(P, "physical_gb", lambda: 64.0)
    p = P.make_plan(info(), K.resolve("full", {}, {}, ""))
    assert p["cap_gb"] == 49.0


# ------------------------------------------------------------------ receipts and resume

def _build(tmp_path, cfg=None):
    rec = {"model": "/m", "checkpoint": "toy@00000000", "inspection": info(), "config": cfg or K.resolve("quick", {}, {}, ""),
           "plan": {}, "inputs": {"probe_corpus": {"sha256": "x"}}, "stage_options": {"core": {"device": "cpu"}}}
    return Build.create(tmp_path / "b", rec)


def test_build_refuses_non_empty_out(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "x").write_text("1")
    with pytest.raises(SystemExit, match="not empty"):
        _build(tmp_path)


def test_receipts_resume_and_stale_inputs(tmp_path):
    b = _build(tmp_path)
    commit = {"commit": "abc", "dirty": False}
    with pytest.raises(SystemExit, match="needs stage map"):
        b.begin("boxes", commit)
    r = b.begin("core", commit)
    assert b.receipt("core")["status"] == "running" and not b.is_done("core")
    b.end(r, "done", outputs={"core": {"sha256": "s"}}, summary={"units": 1})
    assert b.is_done("core")
    rc = b.receipt("core")
    assert rc["checkpoint"] == "toy@00000000" and rc["mlx2_commit"] == commit and rc["outputs_digest"]
    b.rec["config"]["compute"]["core_steps"] += 1            # an input changed: refuse, never reuse silently
    with pytest.raises(SystemExit, match="--redo core"):
        b.is_done("core")
    b.forget("core")
    assert b.receipt("core") is None and list((b.out / "receipts").glob("core.redo-*.json"))
    b2 = Build(b.out).require()
    assert b2.rec["checkpoint"] == "toy@00000000"


# ------------------------------------------------------------------ progress and metrics

def test_progress_parses_carve_steps_and_box_events(tmp_path):
    p = Progress(tmp_path / "events.jsonl", tty=False, quiet=True)
    p.emit("build_start", checkpoint="toy@0", commit="abc", profile="quick", out="o", stages=["map", "boxes"])
    p.emit("stage_start", stage="boxes")
    p.feed_line(EVENT_PREFIX + json.dumps({"event": "boxes_queued", "boxes": ["a", "b", "c"]}), "boxes")
    p.feed_line(EVENT_PREFIX + json.dumps({"event": "box_start", "box": "a", "steps": 60}), "boxes")
    p.feed_line("11:00:00   rank_zero lam=1.0 step 9 main=0.1234 kept=0.4000 (2.5 s/step, peak 6.1 GB)", "boxes")
    s = p.state
    assert s["current_box"] == "a" and s["step"] == 9 and s["s_per_step"] == 2.5 and s["mlx_peak_gb"] == 6.1
    assert s["eta_s"] == pytest.approx(50 * 2.5)
    p.feed_line(EVENT_PREFIX + json.dumps({"event": "box_end", "box": "a", "verdict": "pass", "held_dce": 0.031,
                                           "wall_s": 100.0}), "boxes")
    p.feed_line(EVENT_PREFIX + json.dumps({"event": "box_end", "box": "b", "verdict": "miss", "held_dce": 0.2,
                                           "wall_s": 120.0}), "boxes")
    assert p.box_counts() == {"total": 3, "done": 2, "pass": 1, "miss": 1, "failed": 0, "covered": 0, "queued": 1,
                              "running": 0}
    assert p.state["eta_s"] == pytest.approx(110.0)
    p.emit("footprint", stage="boxes", gb=7.5)
    p.emit("gen", stage="boxes", tok_s=120.0, tokens_total=640)
    line = p.line()
    assert "boxes 2/3 (pass 1, miss 1)" in line and "fp 7.5 GB" in line and "gen 120 tok/s" in line
    m = p.metrics()
    for fam in ("mlx2_amoeba_build_info", "mlx2_amoeba_boxes", "mlx2_amoeba_step_seconds", "mlx2_amoeba_footprint_gb",
                "mlx2_amoeba_footprint_peak_gb", "mlx2_amoeba_gen_tokens_per_second", "mlx2_amoeba_eta_seconds",
                "mlx2_amoeba_held_dce_nats_bucket", "mlx2_amoeba_stage_status", "mlx2_amoeba_mlx_peak_memory_gb"):
        assert fam in m, fam
    assert 'mlx2_amoeba_boxes{state="pass"} 1' in m
    assert re.search(r'mlx2_amoeba_held_dce_nats_bucket\{le="0.05"\} 1\b', m)
    assert re.search(r"mlx2_amoeba_held_dce_nats_count 2\b", m)
    ev = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert ev[0]["event"] == "build_start" and all("ts" in e for e in ev)
    assert any(e["event"] == "step" and e["tag"] == "rank_zero" for e in ev)


def test_metrics_server_serves_loopback(tmp_path):
    p = Progress(tmp_path / "e.jsonl", tty=False, quiet=True)
    p.emit("build_start", checkpoint="toy@0", commit="abc", profile="quick", out="o", stages=["map"])
    srv = MetricsServer(p, 0)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/metrics", timeout=5) as r:
            body = r.read().decode()
            assert r.headers["Content-Type"].startswith("text/plain")
        assert 'mlx2_amoeba_build_info{checkpoint="toy@0"' in body
    finally:
        srv.close()


# ------------------------------------------------------------------ partition rules (step 4)

def _partition_fixture(tmp_path):
    L, N = 2, 8
    out = tmp_path / "b"
    b = _build(tmp_path)
    core = np.zeros((L, N), bool)
    core[0, 0] = True
    (out / "core/carve").mkdir(parents=True)
    np.savez(out / "core/core.npz", core=core.astype(np.uint8))
    (out / "core/carve/carve.json").write_text(json.dumps({"shape": {"widths": [8, 6]},
                                                           "inspect": {"paths": {"ones_parity": {"kl_max": 0.0}}}}))
    boxes = {"a": [(0, 1), (0, 2), (0, 3), (0, 0)], "b": [(0, 2), (0, 3), (1, 1)], "c": [(0, 3), (1, 2)],
             "d": [(1, 4)]}
    vers = []
    for n, units in boxes.items():
        m = np.zeros((L, N), np.uint8)
        for l, u in units:
            m[l, u] = 1
        d = out / f"boxes/carves/box-{n}"
        d.mkdir(parents=True)
        np.savez(d / "box_mask.npz", box=m)
        vers.append({"box": n, "verdict": "failed" if n == "d" else ("miss" if n == "c" else "pass")})
    (out / "boxes/boxset").mkdir(parents=True)
    (out / "boxes/boxset/boxes.json").write_text(json.dumps({"boxes": [{"box": n, "node": f"t.{n}", "kind": "subject"}
                                                                       for n in boxes]}))
    (out / "boxes/carves/verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in vers))
    (out / "map/clusters").mkdir(parents=True)
    (out / "map/capture").mkdir(parents=True)
    pool = np.array([[l, u] for l in range(L) for u in range([8, 6][l])])
    np.savez(out / "map/clusters/clusters.npz", pool=pool, labels=(pool[:, 1] % 2).astype(np.int16))
    (out / "map/capture/meta.json").write_text(json.dumps({"widths": [8, 6]}))
    return b


def test_partition_follows_the_workflow_table(tmp_path):
    b = _partition_fixture(tmp_path)
    outs, summ = S.run_partition(b, None)
    part = json.loads((b.out / "partition/partition.json").read_text())
    z = np.load(b.out / "partition/partition.npz")
    tiles = {t["tile"]: z[t["npz_key"]] > 0 for t in part["tiles"]}
    assert part["check"]["exact_cover"] and summ["exact_cover"]
    core = tiles["core"]
    assert core[0, 0] and core[0, 3]                   # core, and (0,3) claimed by a, b, c (3+ boxes)
    assert tiles["pair:a+b"][0, 2] and tiles["pair:a+b"].sum() == 1     # exactly two boxes -> their cross-box tile
    assert tiles["box:a"][0, 1] and tiles["box:b"][1, 1] and tiles["box:c"][1, 2]
    assert "box:d" not in tiles and part["boxes_left_out"] == {"d": "failed"}
    assert not any(t[1, 6:].any() for t in tiles.values())            # padding past width 6 is nowhere
    rem = [t for t in part["tiles"] if t["kind"] == "remainder"]
    assert {t["cluster"] for t in rem} == {0, 1}                       # unclaimed units grouped by map cluster
    assert tiles["remainder:C0"][1, 4]                                 # failed box d's unit is unclaimed
    total = sum(t.astype(int) for t in tiles.values())
    assert (total[0] == 1).all() and (total[1, :6] == 1).all()
    assert part["boxes"]["a"]["cross_box_tiles"] == ["pair:a+b"]
    assert part["units_to_core_by_3plus"] == 1


# ------------------------------------------------------------------ verdicts

def test_verdict_rules():
    v = K.VERDICT
    ok = verdict({}, {"held": {"mean_d_ce": 0.04}, "spare": {"mean_d_ce": 0.02},
                      "random_same_count_held": {"mean_d_ce": 0.3}}, 10, v)
    assert ok["verdict"] == "pass" and ok["val_held_gap"] == pytest.approx(0.02) and ok["beats_random"]
    miss = verdict({}, {"held": {"mean_d_ce": 0.07}, "spare": {"mean_d_ce": 0.03},
                        "random_same_count_held": {"mean_d_ce": 0.3}}, 10, v)
    assert miss["verdict"] == "miss" and "held dCE" in miss["miss_reasons"][0]
    weak = verdict({}, {"held": {"mean_d_ce": 0.04}, "spare": {"mean_d_ce": 0.03},
                        "random_same_count_held": {"mean_d_ce": 0.06}}, 10, v)
    assert weak["verdict"] == "miss" and "random" in weak["miss_reasons"][0]
    none = verdict({}, {"held": {"mean_d_ce": None}}, 10, v)
    assert none["verdict"] == "miss"                    # an empty held set is never a pass
    cov = verdict({}, {"held": {"mean_d_ce": 0.01}, "random_same_count_held": {"mean_d_ce": 0.01}}, 0, v)
    assert cov["verdict"] == "covered"
    # regression (smoke 5ba84e2ac): a box that took the whole free pool, random same-count = the same units, both 0
    whole = verdict({"pool_units": 2799}, {"held": {"mean_d_ce": 0.0}, "spare": {"mean_d_ce": 0.0},
                                           "random_same_count_held": {"mean_d_ce": 0.0}}, 2799, v)
    assert whole["verdict"] == "miss" and whole["takes_whole_pool"] and any("margin" in r for r in whole["miss_reasons"])


# ------------------------------------------------------------------ gate detector maths

def test_logistic_detector_folds_standardisation():
    rng = np.random.default_rng(0)
    d = 16
    mu_p, mu_n = np.zeros(d), np.zeros(d)
    mu_p[3] = 4.0
    Xp = rng.normal(mu_p, 1.0, (200, d)) * 5 + 100            # raw scale and offset (folded into w, b)
    Xn = rng.normal(mu_n, 1.0, (400, d)) * 5 + 100
    w, b = train_logistic(Xp, Xn)
    acc = (np.mean(Xp @ w + b > 0) + np.mean(Xn @ w + b < 0)) / 2
    assert acc > 0.95
    Fp = [Xp[i:i + 10] for i in range(0, 200, 10)]
    Fn = [Xn[i:i + 10] for i in range(0, 400, 10)]
    assert window_score(Fp, w, b).min() > window_score(Fn, w, b).max()


# ------------------------------------------------------------------ inspection helpers

def test_moe_layer_layout_and_artifact_names(tmp_path):
    assert _moe_layers({"first_k_dense_replace": 1}, 4, 8) == [False, True, True, True]
    assert _moe_layers({"mlp_only_layers": [2]}, 4, 8) == [True, True, False, True]
    assert _moe_layers({"decoder_sparse_step": 2}, 4, 8) == [False, True, False, True]
    assert _moe_layers({}, 3, None) == [False] * 3
    snap = tmp_path / "hub/models--Qwen--Qwen3-0.6B/snapshots/abc"
    snap.mkdir(parents=True)
    assert artifact_name(snap) == "Qwen/Qwen3-0.6B"
    assert artifact_name(tmp_path / "Qwen3-0.6B") == "Qwen3-0.6B"


# ------------------------------------------------------------------ GPU window guard and CLI dispatch

def test_gpu_guard(tmp_path):
    import os
    lock = tmp_path / "gpu.lock"
    lock.write_text("")                                # idle sentinel: a regular empty file
    assert R.require_gpu_window(True, lock=lock)["device"] == "cpu"
    with pytest.raises(SystemExit, match="outside a GPU window"):
        R.require_gpu_window(False, lock=lock)
    with pytest.raises(SystemExit, match="does not match"):
        R.require_gpu_window(False, "anything", lock=lock)        # an owner label needs a held lock
    lock.unlink()
    lock.mkdir()
    (lock / "owner.json").write_text(json.dumps({"pid": 1, "label": "fnday-runner"}))
    with pytest.raises(SystemExit, match="does not match"):
        R.require_gpu_window(False, "someone-else", lock=lock)
    assert R.require_gpu_window(False, "fnday-runner", lock=lock)["owner_stated"] == "fnday-runner"
    (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "label": "mine"}))
    assert R.require_gpu_window(False, lock=lock)["ours"]


def test_gpu_window_reads_owner(tmp_path):
    import os
    lock = tmp_path / "gpu.lock"
    lock.mkdir()
    (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "label": "test"}))
    w = R.gpu_window(lock)
    assert w["held"] and w["ours"] and w["label"] == "test"
    (lock / "owner.json").write_text(json.dumps({"pid": 1, "label": "other"}))
    assert not R.gpu_window(lock)["ours"]


def test_mlx2_cli_dispatches_amoeba(monkeypatch):
    import mlx2.cli as cli
    import mlx2.experimental.amoeba.cli as acli
    seen = {}
    monkeypatch.setattr(acli, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    assert cli.main(["amoeba", "status", "--out", "x"]) is not None
    assert seen["argv"] == ["status", "--out", "x"]


def test_stop_at_parsing():
    from mlx2.experimental.amoeba.cli import stop_at
    import time
    assert stop_at("") is None and stop_at("123.5") == 123.5
    t = stop_at("06:45")
    assert time.time() < t <= time.time() + 86400


def test_status_reads_receipts(tmp_path):
    from mlx2.experimental.amoeba.cli import status
    b = _build(tmp_path)
    r = b.begin("core", {"commit": "abc", "dirty": False})
    b.end(r, "done", outputs={}, summary={"units": 3})
    b.rec["mlx2_commit"] = {"commit": "abc"}
    s = status(b)
    assert s["stages"]["core"]["status"] == "done" and s["stages"]["map"]["status"] is None


# ------------------------------------------------------------------ targeted extraction helpers

def test_extract_topic_seed_and_core_cache(tmp_path, monkeypatch):
    from mlx2.experimental.amoeba import extract as X
    b = _partition_fixture(tmp_path)
    (b.out / "map/labels").mkdir(parents=True)
    (b.out / "map/labels/labels.json").write_text(json.dumps({"clusters": [   # tags by FIT gates only
        {"id": "pool:C0", "labels": [], "top_nodes_fit": [{"node": "sci.biology.genetics", "d_fit": 1.4, "twin95_fit": 0.3}]},
        {"id": "pool:C1", "labels": [{"topic": "sci.biology"}],                  # a held-confirmed label alone: not used
         "top_nodes_fit": [{"node": "sci.biology", "d_fit": 0.7, "twin95_fit": 0.1},
                           {"node": "prog.lang", "d_fit": 1.2, "twin95_fit": 0.2}]}]}))
    Z, info_ = X._topic_seed(b, "sci.biology", (2, 8))
    assert info_["clusters"] == [0] and Z.shape == (2, 8)
    assert Z[0, 0] == 3.0 and Z[0, 1] == 1.0 and Z[1, 2] == 3.0       # cluster 0 = even unit ids in the fixture
    Z2, i2 = X._topic_seed(b, "lang.fra", (2, 8))
    assert Z2 is None and i2["clusters"] == []
    # core cache: publish from one build, reuse in another, with the carve receipt's paths relocated
    monkeypatch.setattr(X, "CACHE", tmp_path / "cache")
    b.rec["plan"] = {"moe_units": "expert"}
    b.rec["inputs"] = {"probe_corpus": {"sha256": "p"}, "core_corpora": [{"sha256": "c"}]}
    b.rec["mlx2_commit"] = {"commit": "abc", "dirty": False}
    core = b.out / "core"
    for f in ("corecheck.json", "english.jsonl", "corpus.jsonl"):
        (core / f).write_text("{}\n")
    (core / "gen").mkdir()
    (core / "carve/carve.json").write_text(json.dumps({"args": {"corpus": f"jsonl:{core}/corpus.jsonl"},
                                                        "shape": {"widths": [8, 6]}}))
    dst = X.publish(b, "core")
    assert (dst / "cache.json").exists() and str(core) not in (dst / "carve/carve.json").read_text()
    other = tmp_path / "o"
    rec = dict(b.rec)
    b2 = Build.create(other, rec)
    assert X.cache_dir(b2, "core") == dst
    assert X._reuse(b2, "core") == dst
    a = json.loads((other / "core/carve/carve.json").read_text())["args"]
    assert a["corpus"] == f"jsonl:{other}/core/corpus.jsonl"
    b2.rec["config"]["compute"]["core_steps"] += 1                  # other settings: another cache entry
    assert X.cache_dir(b2, "core") != dst


def test_extract_cli_refusals(tmp_path):
    from mlx2.experimental.amoeba.cli import main
    with pytest.raises(SystemExit, match="not built"):
        main(["extract", "--model", "x", "--topic", "sci.biology", "--out", str(tmp_path / "o"), "--attention-units", "heads"])
    with pytest.raises(SystemExit, match="--against"):
        main(["extract", "--model", "x", "--topic", "sci.biology", "--out", str(tmp_path / "o")])


def test_unit_clusters_nest_expert_maps_per_layer(tmp_path):
    b = _build(tmp_path)
    (b.out / "core/carve").mkdir(parents=True)
    (b.out / "core/carve/carve.json").write_text(json.dumps({"shape": {"widths": [6, 16]}}))   # dense 6, MoE 4 x 4
    (b.out / "map/clusters").mkdir(parents=True)
    (b.out / "map/capture").mkdir(parents=True)
    (b.out / "map/capture/meta.json").write_text(json.dumps({"widths": [6, 4]}))              # map: experts
    pool = np.array([[0, u] for u in range(6)] + [[1, e] for e in range(4)])
    np.savez(b.out / "map/clusters/clusters.npz", pool=pool, labels=np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 6]))
    uc = S.unit_clusters(b, (2, 16))
    assert uc[0, :6].tolist() == [0, 1, 2, 3, 4, 5] and (uc[0, 6:] == -1).all()   # dense neuron u keeps its own
    assert uc[1].tolist() == [7] * 4 + [8] * 4 + [9] * 4 + [6] * 4                 # expert e -> neurons e*4..e*4+3
    np.savez(b.out / "core/core.npz", core=np.pad(np.array([[1, 0, 0, 0, 0, 0] + [0] * 10,
                                                            [1] * 4 + [1, 0, 0, 0] + [0] * 8]), 0).astype(np.uint8))
    m = S.map_core_mask(b, [6, 4])
    assert m[0].tolist() == [True] + [False] * 5 and m[1, :4].tolist() == [True, False, False, False]


def test_redo_archives_the_stage_dir(tmp_path):
    b = _build(tmp_path)
    r = b.begin("core", {"commit": "abc", "dirty": False})
    b.end(r, "done", outputs={}, summary={})
    (b.stage_dir("core") / "export.npz").write_text("old")
    b.forget("core")
    assert not (b.out / "core").exists() and b.receipt("core") is None
    old = list((b.out / ".redo").glob("core-*"))
    assert len(old) == 1 and (old[0] / "export.npz").read_text() == "old"


def test_generations_go_to_every_box_that_owns_the_document(tmp_path):
    bs = tmp_path / "bs"
    for box, docs in (("a", ["d1", "d2"]), ("b", ["d2"])):
        (bs / box).mkdir(parents=True)
        for role in ("carve", "val", "held"):
            rows = [{"doc_id": x, "text": x} for x in docs] if role == "carve" else []
            (bs / box / f"prompts_{role}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    gen = tmp_path / "gen.jsonl"
    rows = [{"doc_id": "d2", "mode": "greedy", "seed": None}, {"doc_id": "d2", "mode": "greedy", "seed": None},
            {"doc_id": "d2", "mode": "t0.7", "seed": 5}, {"doc_id": "d1", "mode": "greedy", "seed": None}]
    gen.write_text("".join(json.dumps(r) + "\n" for r in rows))
    g = S._assign_gen([{"box": "a"}, {"box": "b"}], bs, gen)
    assert len(g[("a", "carve")]) == 3 and len(g[("b", "carve")]) == 2          # shared doc: both boxes, no dupes


def test_gates_with_no_box_writes_an_empty_gate_set(tmp_path):
    b = _partition_fixture(tmp_path)
    (b.out / "partition").mkdir()
    (b.out / "partition/partition.json").write_text(json.dumps({"boxes_in": []}))
    outs, summ = S.run_gates(b, None)
    assert summ["trained"] == 0 and json.loads((b.out / "gates/gates.json").read_text())["boxes"] == {}


def _ctx(**kw):
    from mlx2.experimental.amoeba.stages import Ctx
    c = Ctx(Progress(Path(kw.pop("events")), tty=False, quiet=True), cpu=True, commit={"commit": "abc", "dirty": False},
            gen_backend="local", stop_at=None)
    c.redo = set(kw.get("redo", ()))
    return c


def test_gap_verdict_reads_the_compare_report(tmp_path, monkeypatch):
    from mlx2.experimental.amoeba import extract as X
    b = _build(tmp_path)
    b.rec["extract"] = {"topic": "sci.physics", "against": {"path": "/h", "checkpoint": "h@1", "plan": {}}}
    b.rec["inputs"]["probe_corpus"]["path"] = "/probes.jsonl"
    d = b.out / "gap"
    for role in ("donor", "host"):
        (d / role).mkdir(parents=True)
        (d / role / "score.json").write_text("{}")
    (d / "compare").mkdir()
    rep = {"margin_nats_per_byte": 0.005, "overall_gap_nats_per_byte": {"bare": 0.1, "prefix": 0.12},
           "topics": {"sci.physics": {"bare": {"centred_ci95": [0.02, 0.05]}, "prefix": {"centred_ci95": [0.01, 0.04]}}},
           "donor_better_centred": ["sci.physics"]}
    (d / "compare/donor_gap.json").write_text(json.dumps(rep))
    monkeypatch.setattr(X, "run_child", lambda *a, **k: pytest.fail("no child should run: every output exists"))
    outs, summ = X.run_gap(b, _ctx(events=tmp_path / "e.jsonl"))
    assert summ["donor_leads"] is True
    rep["topics"]["sci.physics"]["prefix"]["centred_ci95"] = [0.001, 0.04]        # prefix CI touches the margin
    (d / "compare/donor_gap.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match="does not lead"):
        X.run_gap(b, _ctx(events=tmp_path / "e.jsonl"))
    b.rec["extract"]["force_extract"] = "pipeline check"
    outs, summ = X.run_gap(b, _ctx(events=tmp_path / "e.jsonl"))
    g = json.loads((d / "gap.json").read_text())
    assert summ["donor_leads"] is False and g["force_extract"] == "pipeline check"


def test_map_cache_follows_the_core_content_and_dirty_trees_never_publish(tmp_path, monkeypatch):
    from mlx2.experimental.amoeba import extract as X
    monkeypatch.setattr(X, "CACHE", tmp_path / "cache")
    b = _build(tmp_path)
    b.rec["plan"] = {"moe_units": "expert"}
    b.rec["inputs"] = {"probe_corpus": {"sha256": "p"}, "core_corpora": [{"sha256": "c"}]}
    (b.out / "core").mkdir()
    np.savez(b.out / "core/core.npz", core=np.ones((2, 4), np.uint8))
    k1 = X._cache_key(b, "map")
    np.savez(b.out / "core/core.npz", core=np.zeros((2, 4), np.uint8))
    assert X._cache_key(b, "map") != k1                    # another core, another map entry
    b.rec["mlx2_commit"] = {"commit": "abc", "dirty": True}
    assert X.publish(b, "core") is None                    # a dirty run never feeds the shared cache
    b.rec["options"] = {"cpu": True}
    kc = X._cache_key(b, "core")
    b.rec["options"] = {"cpu": False}
    assert X._cache_key(b, "core") != kc                   # the device is part of the core's key


def test_redo_of_extract_core_bypasses_the_cache(tmp_path, monkeypatch):
    from mlx2.experimental.amoeba import extract as X
    b = _build(tmp_path)
    called = {}
    monkeypatch.setattr(X, "_reuse", lambda b_, w: pytest.fail("a redo must not reuse the cache"))
    monkeypatch.setattr(X, "run_core", lambda b_, c: called.setdefault("carved", True) and ({}, {}))
    monkeypatch.setattr(X, "publish", lambda b_, w, replace=False: called.setdefault("replace", replace))
    X.run_xcore(b, _ctx(events=tmp_path / "e.jsonl", redo={"core"}))
    assert called == {"carved": True, "replace": True}


def test_packer_slices_dense_and_quantized_rows():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx2.experimental.amoeba.child_pack import dense_weight
    mx.set_default_device(mx.cpu)
    lin = nn.Linear(64, 32, bias=False)
    w = np.array(lin.weight)
    assert np.allclose(dense_weight(lin, [3, 7]), w[[3, 7]])
    q = nn.QuantizedLinear.from_linear(lin, group_size=32, bits=8)
    deq = np.array(mx.dequantize(q.weight, q.scales, q.biases, group_size=32, bits=8))
    assert np.allclose(dense_weight(q, [3, 7]), deq[[3, 7]]) and np.abs(deq - w).max() < 0.05
    sw = mx.random.normal((4, 8, 16))                       # [E, F, d] switch weight
    class Sw(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = sw
    assert np.allclose(dense_weight(Sw(), [1, 5], expert=2), np.array(sw)[2, [1, 5]])


def test_extract_topic_must_be_a_taxonomy_node(tmp_path):
    from mlx2.experimental.amoeba.cli import check_topic
    tax = tmp_path / "t.json"
    tax.write_text(json.dumps({"nodes": [{"id": "sci"}, {"id": "sci.biology"}, {"id": "sci.physics"},
                                         {"id": "prog.rust"}, {"id": "old.x", "deprecated": True, "replaced_by": "sci"}]}))
    check_topic("sci.biology", tax)
    with pytest.raises(SystemExit, match="did you mean sci.biology"):
        check_topic("sci.bio", tax)
    with pytest.raises(SystemExit, match="deprecated"):
        check_topic("old.x", tax)
    with pytest.raises(SystemExit, match="not a node"):
        check_topic("zzz.q", tax)


def test_carve_argv_carries_the_section7_gradient_gate(tmp_path):
    from mlx2.experimental.amoeba.child_boxbatch import carve_argv
    import research.amoeba.carve.__main__ as D
    cfg = K.resolve("quick", {}, {}, "")
    plan = {"batches": {"carve_box": {"bs": 2}}, "watchdog_gb": 80, "mlx_memory_gb": 76, "moe_units": "expert"}
    spec = {"model": "/m", "compute": cfg["compute"], "data": cfg["data"], "plan": plan, "verdict": cfg["verdict"],
            "grad_gate": cfg["grad_gate"], "core": str(tmp_path / "core.npz")}
    a = D.parse(carve_argv(spec, tmp_path / "bs", tmp_path / "d"))
    assert a.dir_grad_min_corr == 0.9 and a.dir_grad_tol == 1e9 and a.dir_grad_norm_ratio == "0.67,1.5"
    assert a.dir_grad_eps == "0.0625,0.125,0.25,0.5" and a.dir_grad_k == 8
    assert a.export_target == 0.03 and a.targets == "" and a.n_held == 0 and a.pool_domains and a.code_cut


def test_core_alone_checks_really_apply_the_core_mask(tmp_path, monkeypatch):
    """Regression (smoke window 7796dbf62): the cfg_fn handed to text_ce must SET the core mask; a lambda that only
    built the setter scored the full model twice (dCE 0.0, every box 'covered')."""
    from research.amoeba.carve import evals as E
    from mlx2.experimental.amoeba import child_boxbatch as CB, child_probe as CP

    class Ctl:
        core = None
        mean = sink_layers = dense_layers = boxes = None

        def reset(self):
            self.core = None

    class Host:
        ctl = Ctl()
    seen = []

    def fake_ce(host, seqs, cfg_fn=None, bs=8):
        if cfg_fn:
            cfg_fn()
        seen.append(host.ctl.core)
        return (2.5 if host.ctl.core is not None else 2.0), 10
    monkeypatch.setattr(E, "text_ce", fake_ce)
    monkeypatch.setattr(CP, "window", lambda *a, **k: [1, 2, 3])
    (tmp_path / "val.jsonl").write_text(json.dumps({"text": "x", "doc_id": "d"}) + "\n")
    spec = {"data": K.DATA_DEFAULTS}
    r = CB.core_alone_val(Host(), "CORE", tmp_path / "val.jsonl", spec)
    assert seen == [None, "CORE"] and r["d_ce"] == pytest.approx(0.5)


def test_gate_refused_boxes_fail_alone_and_crashes_stop_the_stage():
    boxes = [{"box": n} for n in "abc"]
    ok = {"a": {"verdict": "pass"}, "b": {"verdict": "failed", "failure": "gate-refused"}, "c": {"verdict": "miss"}}
    assert S.box_failure_policy(boxes, ok) == ["b"]
    crash = {**ok, "c": {"verdict": "failed", "failure": "crash"}}
    with pytest.raises(SystemExit, match="box carves failed: c"):
        S.box_failure_policy(boxes, crash)
    legacy = {**ok, "c": {"verdict": "failed"}}                     # no failure field: treated as a crash
    with pytest.raises(SystemExit, match="box carves failed: c"):
        S.box_failure_policy(boxes, legacy)
    every = {n: {"verdict": "failed", "failure": "gate-refused"} for n in "abc"}
    with pytest.raises(SystemExit, match="every box"):
        S.box_failure_policy(boxes, every)


def test_only_the_s7_direction_check_counts_as_a_gate_refusal():
    from mlx2.experimental.amoeba.child_boxbatch import classify_failure
    s7 = SystemExit("inspect: layer-wise gradients fail the directional-derivative check on the fused forward ([...])")
    assert classify_failure(s7) == "gate-refused"
    assert classify_failure(SystemExit("inspect: core=ones parity failed")) == "refused"     # a code fault: stops
    assert classify_failure(SystemExit("inspect: per-layer reference check failed")) == "refused"
    assert classify_failure(ValueError("boom")) == "crash"


def test_resume_rejects_unknown_requested_stage(tmp_path, monkeypatch):
    from mlx2.experimental.amoeba import cli
    b = _build(tmp_path)
    monkeypatch.setattr(cli, "inspect_checkpoint", lambda _: b.info)
    monkeypatch.setattr(cli, "check_commit", lambda _: {})
    monkeypatch.setattr(cli, "check_inputs", lambda _: None)
    monkeypatch.setattr(cli, "execute", lambda *a, **k: 0)
    args = cli.parse(["resume", "--out", str(b.out), "--stages", "map,boxxes"])
    with pytest.raises(SystemExit, match="boxxes"):
        cli.cmd_resume(args)


@pytest.mark.parametrize("command", ["core", "resume"])
def test_rolling_plan_cannot_start_a_resident_carve(tmp_path, monkeypatch, command):
    from mlx2.experimental.amoeba import cli

    b = _build(tmp_path)
    b.rec.update(plan={"stream": {"mode": "rolling"}}, stages=["map"])
    b.save()
    monkeypatch.setattr(cli, "inspect_checkpoint", lambda _: b.info)
    monkeypatch.setattr(cli, "check_commit", lambda _: {"commit": "test"})
    monkeypatch.setattr(cli, "check_inputs", lambda _: None)
    args = [command, "--out", str(b.out)]
    if command == "resume":
        args.extend(["--stages", "map,core"])
    with pytest.raises(SystemExit, match="rolling.*map"):
        cli.main(args)


def test_extraction_requires_resident_gap_host(tmp_path, monkeypatch):
    from mlx2.experimental.amoeba import cli

    args = cli.parse(["extract", "--model", "donor", "--against", "host", "--topic", "prog.python",
                      "--out", str(tmp_path / "out")])
    monkeypatch.setattr(cli, "inspect_checkpoint", lambda name: {**info(300), "path": name})
    def plan(model, cfg, **kw):
        if model["path"] == "donor":
            pytest.fail("gap host must fit before donor setup")
        return {"fits": kw.get("stream") != "resident", "stream": {"mode": "rolling"}}
    monkeypatch.setattr(cli, "make_plan", plan)
    monkeypatch.setattr(cli, "format_plan", lambda *args: "plan")
    with pytest.raises(SystemExit, match="host.*fit"):
        cli.cmd_build(args)


def test_rolling_parity_checks_resident_capacity_before_reference(tmp_path, monkeypatch):
    from mlx2.experimental.amoeba import cli
    monkeypatch.setattr(cli, "inspect_checkpoint", lambda _: info(300))
    monkeypatch.setattr(cli, "check_commit", lambda _: {"commit": "test"})
    monkeypatch.setattr(R, "run_child", lambda *a, **k: pytest.fail("resident capture started"))
    out = tmp_path / "parity"
    with pytest.raises(SystemExit, match="resident"):
        cli.main(["rolling-parity", "--model", "toy", "--out", str(out), "--rolling-resident", "--cpu"])
    assert not out.exists()
