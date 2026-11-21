"""Regression tests of the 2026-11-21 bug sweep, lane amoeba-cli (AMCL-01 .. AMCL-09). CPU only, no model load:
the build-directory lock, the core carve's completion test, per-stage commits in bundles, the effective corpus
recorded at build creation, typed overrides, absolute model paths, truthful install-route failures, the measured
rolling reads gate, and the map of a map-only build."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.experimental.amoeba import cli as C
from mlx2.experimental.amoeba import config as K
from mlx2.experimental.amoeba import install as I
from mlx2.experimental.amoeba import runner as R
from mlx2.experimental.amoeba import stages as S
from mlx2.experimental.amoeba import state as ST
from mlx2.experimental.amoeba.progress import Progress
from mlx2.experimental.amoeba.state import Build

A = {"commit": "aaaaaaaaa", "dirty": True}        # dirty: nothing here may publish to the fingerprint cache
B = {"commit": "bbbbbbbbb", "dirty": True}


def _info():
    return {"path": "/models/toy", "artifact": "toy", "checkpoint": "toy@00000000", "fingerprint": "0" * 64,
            "adapter": "Toy", "kind": "dense", "layers": 2, "hidden": 16, "intermediate": 8, "vocab": 100}


def _build(tmp_path, kind="build", **extra):
    rec = {"kind": kind, "model": "/models/toy", "checkpoint": "toy@00000000", "inspection": _info(),
           "config": K.resolve("full", {}, {}, ""), "plan": {}, "inputs": {"probe_corpus": {"path": "/p", "sha256": "x"}},
           "options": {"cpu": True, "gen_backend": "local", "gen_route": "", "boxes": "", "max_boxes": 0,
                       "max_cluster_boxes": 0},
           "stage_options": {}, "stages": list(ST.KINDS[kind]), **extra}
    return Build.create(tmp_path / "b", rec)


def _args():
    return argparse.Namespace(quiet=True, metrics_port=0, stop_at="", gpu_window_owner="", ignore_gpu_queue="",
                              swap_growth_gb=0.0, swap_waiver="")


def _ctx(tmp_path, commit=A):
    return S.Ctx(Progress(tmp_path / "e.jsonl", tty=False, quiet=True), cpu=True, commit=commit, gen_backend="local",
                 stop_at=None)


@pytest.fixture
def keep_signals():
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


@pytest.fixture
def sleeper():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield p
    p.kill()
    p.wait()


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


# ------------------------------------------------------------------ AMCL-01: one run per build directory

def test_build_lock_refuses_a_live_holder_and_takes_over_a_dead_one(tmp_path, sleeper, capsys):
    b = _build(tmp_path)
    b.lock_path.write_text(json.dumps({"pid": sleeper.pid, "created": time.time(), "started": "then"}))
    with pytest.raises(SystemExit, match=f"holds this build .pid {sleeper.pid}"):
        Build(b.out).acquire_lock()
    assert json.loads(b.lock_path.read_text())["pid"] == sleeper.pid      # the live holder keeps its lock
    b.release_lock()                                                       # not ours: stays
    assert b.lock_path.exists()
    sleeper.kill()
    sleeper.wait()
    b.acquire_lock()
    assert "taking over the stale build lock" in capsys.readouterr().err
    assert json.loads(b.lock_path.read_text())["pid"] == os.getpid()
    b.release_lock()
    assert not b.lock_path.exists()


def test_a_recycled_pid_is_not_the_lock_holder(tmp_path, sleeper):
    b = _build(tmp_path)
    # the lock was written long before this live process was born: its pid was handed out again
    b.lock_path.write_text(json.dumps({"pid": sleeper.pid, "created": time.time() - 86400, "started": "yesterday"}))
    b.acquire_lock()
    assert json.loads(b.lock_path.read_text())["pid"] == os.getpid()
    b.release_lock()


def test_execute_refuses_a_stage_running_in_a_live_process(tmp_path, monkeypatch, sleeper, keep_signals):
    b = _build(tmp_path)
    calls = []

    def core(bb, ctx):
        calls.append("core")
        return {}, {"units": 1}
    monkeypatch.setattr(C, "kind_runners", lambda _b: {"core": core})
    r = b.begin("core", A)
    partial = b.stage_dir("core") / "carve"
    partial.mkdir()
    (partial / "rank_ckpt.npz").write_text("the first run's partial output")
    ST.write_json(b.receipt_path("core"), {**r, "pid": sleeper.pid})
    with pytest.raises(SystemExit, match=f"stage core is running in pid {sleeper.pid}"):
        C.execute(b, ["core"], _args(), A, redo=["core"])
    assert not calls and (partial / "rank_ckpt.npz").exists() and not (b.out / ".redo").exists()
    assert b.receipt("core")["attempts"] == 1 and not b.lock_path.exists()
    # the recorded process is gone: the stage is taken up again
    ST.write_json(b.receipt_path("core"), {**r, "pid": _dead_pid()})
    assert C.execute(b, ["core"], _args(), A, redo=[]) == 0
    assert calls == ["core"] and b.receipt("core")["status"] == "done" and b.receipt("core")["attempts"] == 2
    assert not b.lock_path.exists()


def test_execute_holds_the_lock_while_a_stage_runs(tmp_path, monkeypatch, keep_signals):
    b = _build(tmp_path)
    seen = {}

    def core(bb, ctx):
        seen["lock"] = json.loads(bb.lock_path.read_text())
        raise RuntimeError("stage crashed")
    monkeypatch.setattr(C, "kind_runners", lambda _b: {"core": core})
    assert C.execute(b, ["core"], _args(), A, redo=[]) == 1
    assert seen["lock"]["pid"] == os.getpid() and not b.lock_path.exists()     # released on failure too


# ------------------------------------------------------------------ AMCL-02: the core carve really finished

def _core_fixture(tmp_path, carve_json, marker=None):
    eng = tmp_path / "eng.jsonl"
    eng.write_text(json.dumps({"text": "The harbour ferry leaves at dawn.", "split": "fit", "domain": "prose"}) + "\n")
    b = _build(tmp_path, plan={"batches": {"carve_core": {"bs": 1}}, "need_gb": 1, "watchdog_gb": 8,
                               "mlx_memory_gb": 4, "moe_units": "expert"},
               inputs={"core_corpora": [{"path": str(eng)}], "probe_corpus": {"path": str(eng), "sha256": "0"}})
    d = b.stage_dir("core")
    (d / "gen").mkdir()
    (d / "gen/gen.jsonl").write_text("")
    (d / "gen/manifest.json").write_text(json.dumps({"rows": 0}))
    run = d / "carve"
    run.mkdir()
    np.savez(run / "export_core.npz", core=np.ones((2, 4), np.uint8))
    (run / "carve.json").write_text(json.dumps({"fingerprint": "0" * 64, "shape": {"widths": [4, 4]}, **carve_json}))
    if marker is not None:
        (run / "carve_complete.json").write_text(json.dumps(marker))
    (d / "corecheck.json").write_text(json.dumps({"english_only": True}))
    return b, run


FINISHED = {"stages_done": ["bisect", "inspect", "mean", "rank"],
            "export_core": {"held": {"mean_d_ce": 0.04}, "random_same_count_held": {"mean_d_ce": 0.6}}}


def test_core_carve_killed_before_its_held_evaluations_is_resumed(tmp_path, monkeypatch):
    # what a carve killed after save_npz(export_core.npz) and before its held evaluations leaves behind
    b, run = _core_fixture(tmp_path, {"stages_done": ["inspect", "mean", "rank"]})
    calls = []

    def child(cmd, **kw):
        calls.append([str(c) for c in cmd])
        rep = json.loads((run / "carve.json").read_text())
        (run / "carve.json").write_text(json.dumps({**rep, **FINISHED}))
    monkeypatch.setattr(S, "run_child", child)
    _, summ = S.run_core(b, _ctx(tmp_path))
    assert len(calls) == 1 and "--resume" in calls[0] and "research.amoeba.carve" in calls[0]
    assert summ["export_held_d_ce"] == 0.04


def test_core_carve_that_ends_without_held_evidence_is_not_a_core(tmp_path, monkeypatch):
    b, _ = _core_fixture(tmp_path, {"stages_done": ["inspect", "mean", "rank"]})
    monkeypatch.setattr(S, "run_child", lambda cmd, **kw: 0)        # the child "succeeds" but finishes nothing
    with pytest.raises(SystemExit, match="without its export-core held evaluations"):
        S.run_core(b, _ctx(tmp_path))


@pytest.mark.parametrize("marker", [None, {"stages_done": ["bisect", "inspect", "mean", "rank"]}])
def test_finished_core_carve_is_never_rerun(tmp_path, monkeypatch, marker):
    """None: a carve finished before the completion marker existed (a long GPU carve is not forced to rerun)."""
    b, _ = _core_fixture(tmp_path, FINISHED, marker)

    def child(cmd, **kw):
        raise AssertionError("a finished carve was started again")
    monkeypatch.setattr(S, "run_child", child)
    _, summ = S.run_core(b, _ctx(tmp_path))
    assert summ["export_held_d_ce"] == 0.04 and summ["units"] == 8


def test_carve_finished_reads_the_marker_and_the_last_save(tmp_path):
    run = tmp_path / "carve"
    run.mkdir()
    (run / "carve.json").write_text(json.dumps({"stages_done": ["inspect", "mean", "rank"]}))
    (run / "carve_complete.json").write_text(json.dumps({"stages_done": ["bisect", "inspect", "mean", "rank"]}))
    assert not S.carve_finished(run)                          # no export_core.npz at all
    np.savez(run / "export_core.npz", core=np.ones((1, 2), np.uint8))
    assert S.carve_finished(run)                              # the carve's own end marker
    # the marker of an earlier invocation that stopped before bisect says nothing about this export
    (run / "carve_complete.json").write_text(json.dumps({"stages_done": ["inspect", "mean", "rank"]}))
    assert not S.carve_finished(run)
    (run / "carve_complete.json").unlink()
    (run / "carve.json").write_text(json.dumps({"stages_done": ["bisect"], "export_core": {}}))
    assert not S.carve_finished(run)
    (run / "carve.json").write_text(json.dumps(FINISHED))
    assert S.carve_finished(run)


# ------------------------------------------------------------------ AMCL-03: every result names its own commit

def test_receipts_keep_every_commit_a_stage_ran_at_and_a_resume_says_so(tmp_path):
    b = _build(tmp_path)
    r = b.begin("core", A)
    assert r["attempt_commits"] == [A]
    b.end(r, "failed", error="killed")
    assert C.commit_notice(b, A) is None
    note = C.commit_notice(b, B)
    assert note.startswith("notice:") and "bbbbbbbbb" in note and "core@aaaaaaaaa" in note and "\n" not in note
    r = b.begin("core", B)                                    # resumed at another commit: sub-step files from both
    assert r["mlx2_commit"] == B and r["attempt_commits"] == [A, B]
    b.end(r, "done", outputs={}, summary={})
    sc = ST.stage_commits(b)
    assert sc["core"]["commit"] == "bbbbbbbbb" and sc["core"]["attempt_commits"] == [A, B] and "map" not in sc


def test_bundle_names_the_commit_each_box_verdict_was_made_at(tmp_path, monkeypatch):
    tax, man = tmp_path / "taxonomy.v1.json", tmp_path / "probe-manifest.json"
    tax.write_text("{}")
    man.write_text("{}")
    b = _build(tmp_path, inputs={"taxonomy": {"path": str(tax), "version": "1"}, "probe_manifest": {"path": str(man)},
                                 "probe_corpus": {"path": "/p", "sha256": "x"}})
    for rel in ("partition/partition.npz", "core/core.npz", "gates/gates.npz", "map/map.json",
                "map/clusters/clusters.npz", "map/clusters/clusters.json", "map/clusters/virtual_experts.npz",
                "map/labels/labels.json", "boxes/boxes.md", "boxes/next-round.json", "core/corecheck.json"):
        (b.out / rel).parent.mkdir(parents=True, exist_ok=True)
        (b.out / rel).write_text("{}")
    (b.out / "partition/partition.json").write_text(json.dumps({
        "boxes": {"lang_fra": {"taxonomy_id": "lang.fra", "kind": "language", "own_units": 3, "carved_units": 4,
                               "cross_box_tiles": []}},
        "pool_units": 16, "tiles": [], "check": {"exact_cover": True}}))
    (b.out / "gates/gates.json").write_text(json.dumps({"boxes": {}}))
    (b.out / "boxes/carves").mkdir(parents=True)
    S.write_jsonl(b.out / "boxes/carves/verdicts.jsonl", [
        {"box": "lang_fra", "node": "lang.fra", "kind": "language", "verdict": "pass", "export_held": 0.03},
        {"box": "sci_optics", "node": "sci.physics.optics", "kind": "branch", "verdict": "failed", "error": "x"}])
    for st in ("map", "core", "boxes", "partition", "gates"):          # made at commit A
        ST.write_json(b.receipt_path(st), {"stage": st, "status": "done", "mlx2_commit": A})

    def child(cmd, **kw):                                       # the prompt-set child
        cmd = [str(c) for c in cmd]
        out, name = Path(cmd[cmd.index("--out") + 1]), cmd[cmd.index("--name") + 1]
        out.mkdir(parents=True)
        (out / f"{name}.manifest.json").write_text(json.dumps({"sha256": "s", "rows": 2, "counts": {}}))
    monkeypatch.setattr(S, "run_child", child)
    monkeypatch.setattr(S.paths, "arrow_python", lambda: sys.executable)
    S.run_bundle(b, _ctx(tmp_path, commit=B))                   # bundled at commit B
    reg = {x["box"]: x for x in S.read_jsonl(b.out / "bundle/registry.jsonl")}
    assert set(reg) == {"lang_fra", "sci_optics"}
    for x in reg.values():
        assert x["mlx2_commit"] == A and x["bundled_at_commit"] == B
    m = json.loads((b.out / "bundle/manifest.json").read_text())
    assert m["bundled_at_commit"] == B and m["stage_commits"]["boxes"] == A and m["stage_commits"]["core"] == A


# ------------------------------------------------------------------ AMCL-04: exclusions are in the first build.json

def _row(i, split, topic, group):
    t = f"probe {i}: notes on {topic} from collection {group}"
    return {"id": f"tp{i:05d}", "topics": [topic], "split": split, "text": t, "group": group, "license": "CC0",
            "origin": "natural", "sha256": hashlib.sha256(t.encode()).hexdigest()}


def _extract_argv(tmp_path):
    d = tmp_path / "topic-probes-abc"
    d.mkdir()
    rows = [_row(0, "fit", "sci.physics", "g0"), _row(1, "fit", "sci.physics.optics", "g1"),
            _row(2, "held", "sci.physics", "h0"), _row(3, "held", "sci.biology", "h1")]
    src = d / "corpus.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"corpus_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}))
    tax = tmp_path / "tax.json"
    tax.write_text(json.dumps({"version": "1", "nodes": [{"id": "sci.physics"}]}))
    eng = tmp_path / "english.jsonl"
    eng.write_text("")
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"probe_ids": ["tp00002"]}))
    out = tmp_path / "out"
    return out, ["extract", "--model", "/m", "--topic", "sci.physics", "--out", str(out), "--skip-gap-screen", "t",
                 "--probe-corpus", str(src), "--probe-manifest", str(man), "--taxonomy", str(tax),
                 "--core-corpus", str(eng), "--exclude-texts", str(ex), "--cpu"]


def _patch_extract(monkeypatch):
    info = {**_info(), "units": {"dense_neurons": 16, "moe_expert_units": 0, "moe_neuron_units": 0}}
    monkeypatch.setattr(C, "inspect_checkpoint", lambda p: info)
    monkeypatch.setattr(C, "make_plan", lambda *a, **k: {"fits": True, "stream": {"mode": "resident"}})
    monkeypatch.setattr(C, "format_plan", lambda *a: "")
    monkeypatch.setattr(C, "check_commit", lambda d: A)


def test_extract_records_the_effective_corpus_when_the_build_is_created(tmp_path, monkeypatch):
    _patch_extract(monkeypatch)
    out, argv = _extract_argv(tmp_path)
    first = {}
    real = Build.create.__func__

    def create(cls, o, rec):
        first["eff"] = json.loads(json.dumps(rec["inputs"].get("probe_corpus_effective")))
        return real(cls, o, rec)
    monkeypatch.setattr(Build, "create", classmethod(create))
    monkeypatch.setattr(C, "execute", lambda *a, **k: 0)
    assert C.main(argv) == 0
    eff = first["eff"]
    assert eff and eff["excluded_rows"] == 1 and Path(eff["path"]).parent.parent == out.resolve() / "inputs"
    b = Build(out).require()
    assert b.rec["inputs"]["probe_corpus_effective"] == eff
    ids = [json.loads(x)["id"] for x in Path(eff["path"]).read_text().splitlines()]
    assert "tp00002" not in ids and len(ids) == 3 and Path(eff["accounting"]).is_file()
    C.check_inputs(b)                                           # the recorded sha256 is the file's


def test_an_interrupted_extract_creation_cannot_resume_on_the_full_corpus(tmp_path, monkeypatch):
    _patch_extract(monkeypatch)
    out, argv = _extract_argv(tmp_path)
    import shutil
    real = shutil.copytree

    def copytree(src, dst, **kw):
        if Path(dst) == out.resolve() / "inputs":
            raise KeyboardInterrupt("killed while the effective corpus was being placed")
        return real(src, dst, **kw)
    monkeypatch.setattr(shutil, "copytree", copytree)
    monkeypatch.setattr(C, "execute", lambda *a, **k: pytest.fail("the build ran"))
    with pytest.raises(KeyboardInterrupt):
        C.main(argv)
    b = Build(out).require()                                    # the build exists ...
    eff = b.rec["inputs"]["probe_corpus_effective"]
    assert eff["excluded_rows"] == 1 and S.probe_corpus(b) == eff["path"]   # ... and never names the full corpus
    with pytest.raises((SystemExit, OSError)):                  # a resume refuses: the effective corpus is missing
        C.check_inputs(b)


# ------------------------------------------------------------------ AMCL-05: typed overrides

def test_overrides_take_the_type_of_their_default():
    why = "box prog.rust missed: wider code branches"
    c = K.resolve("full", K.parse_overrides(["core_steps=40", "core_lr=1", "val_target_nats=0.02", "core_targets=0.05"]),
                  K.parse_overrides(["code_topic_prefixes=prog.,math.", "code_cut=no", "strip_prefix=0"]), why)
    assert c["data"]["code_topic_prefixes"] == ["prog.", "math."] and c["data"]["code_cut"] is False
    assert c["data"]["strip_prefix"] == 0.0 and isinstance(c["data"]["strip_prefix"], float)
    assert c["compute"]["core_steps"] == 40 and c["compute"]["core_lr"] == 1.0 and c["compute"]["core_targets"] == "0.05"
    assert c["verdict"]["val_target_nats"] == 0.02
    assert c["data_overrides"]["values"]["code_topic_prefixes"] == ["prog.", "math."]
    assert [S._code_box(n, c["data"]["code_topic_prefixes"]) for n in ("prog.python", "math.algebra", "pol.elections",
                                                                     "geo.rivers")] == [True, True, False, False]
    for word, want in (("yes", True), ("on", True), ("1", True), ("true", True), ("off", False), ("0", False),
                       ("false", False)):
        assert K.resolve("full", {}, K.parse_overrides([f"code_cut={word}"]), why)["data"]["code_cut"] is want
    assert K.resolve("full", {}, K.parse_overrides(["code_topic_prefixes=prog."]), why)["data"]["code_topic_prefixes"] \
        == ["prog."]


@pytest.mark.parametrize("compute,data,key", [
    (["core_steps=abc"], [], "core_steps"), (["core_steps=12.5"], [], "core_steps"), (["core_steps=true"], [], "core_steps"),
    (["core_lr=fast"], [], "core_lr"), (["val_target_nats=low"], [], "val_target_nats"),
    ([], ["code_cut=maybe"], "code_cut"), ([], ["code_cut=2"], "code_cut"), ([], ["window_tokens=long"], "window_tokens"),
    ([], ["gen_temperature=warm"], "gen_temperature"), ([], ["code_topic_prefixes=,"], "code_topic_prefixes")])
def test_overrides_of_the_wrong_type_are_refused_by_name(compute, data, key):
    with pytest.raises(SystemExit, match=f"override {key}="):
        K.resolve("full", K.parse_overrides(compute), K.parse_overrides(data), "a recorded reason")


# ------------------------------------------------------------------ AMCL-06: model paths are absolute

def test_a_relative_model_path_is_stored_absolute_and_keeps_its_symlink_name(tmp_path, monkeypatch):
    import mlx2.adapters.registry as REG
    from mlx2.experimental.amoeba.inspection import inspect_checkpoint
    model = tmp_path / "models" / "Toy-4B-MLX-4bit"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "toy", "num_hidden_layers": 2, "hidden_size": 8,
                                                   "intermediate_size": 16, "vocab_size": 10}))
    (tmp_path / "models" / "alias").symlink_to(model)
    monkeypatch.setattr(REG, "inspect_model", lambda p: SimpleNamespace(artifact_fingerprint="f" * 64,
                                                                        adapter_type=None, descriptor=None))
    monkeypatch.chdir(tmp_path / "models")
    for arg, name in (("./Toy-4B-MLX-4bit", "Toy-4B-MLX-4bit"), ("Toy-4B-MLX-4bit", "Toy-4B-MLX-4bit"),
                      ("../models/alias", "alias")):
        p = inspect_checkpoint(arg)["path"]
        # children run with cwd = the repository root, so the stored path must not depend on the directory
        assert os.path.isabs(p) and Path(p).name == name and ".." not in Path(p).parts
        assert Path(p).samefile(model)


# ------------------------------------------------------------------ AMCL-07: a crashed route is not `done`

def _install_build(tmp_path):
    b = _build(tmp_path, kind="install")
    for st in ("pair", "topo", "route_tile", "route_lora", "route_bridge"):
        b.end(b.begin(st, A), "done", outputs={}, summary={})
    return b


def test_a_failed_route_gets_a_failed_receipt_the_report_runs_and_a_resume_retries(tmp_path, monkeypatch, keep_signals):
    b = _install_build(tmp_path)
    calls, crash = [], {"on": True}

    def route(bb, ctx):
        calls.append("route_side")
        if crash["on"]:
            raise R.StageFailedContinue("install route side failed (rc=1); see log", {"result": {"path": "/r"}},
                                        {"failed": "rc=1"})
        return {}, {"arm": "side_box"}

    def report(bb, ctx):
        calls.append("report")
        return {}, {"side": "failed" if crash["on"] else "transfers"}
    monkeypatch.setattr(C, "kind_runners", lambda _b: {"route_side": route, "report": report})
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0
    r = b.receipt("route_side")
    assert r["status"] == "failed" and r["continue_past"] and "rc=1" in r["error"] and r["summary"] == {"failed": "rc=1"}
    assert C.status(b)["stages"]["route_side"]["status"] == "failed"
    assert calls == ["route_side", "report"] and b.receipt("report")["status"] == "done"   # the report still ran
    assert not b.is_done("route_side") and b.is_done("report")
    # resume: the failed route is retried; the report that read its failure is made again
    crash["on"] = False
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0
    assert calls == ["route_side", "report", "route_side", "report"]
    assert b.receipt("route_side")["status"] == "done" and "continue_past" not in b.receipt("route_side")
    assert b.receipt("report")["summary"] == {"side": "transfers"} and list((b.out / "receipts").glob("report.redo-*"))
    # nothing failed any more: a further resume skips both
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0 and len(calls) == 4


def test_a_stage_that_failed_without_continue_still_blocks_its_dependents(tmp_path):
    b = _build(tmp_path, kind="install")
    b.end(b.begin("pair", A), "failed", error="refused")
    with pytest.raises(SystemExit, match="needs stage pair done first"):
        b.inputs_digest("topo")


def test_route_runner_raises_instead_of_returning_a_done_stage(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "gate").mkdir(parents=True)
    (bundle / "gate/gates.json").write_text(json.dumps({"boxes": {}}))
    (bundle / "manifest.json").write_text(json.dumps({"box": "sci_physics", "topic": "sci.physics"}))
    out = tmp_path / "out"
    (out / "topo").mkdir(parents=True)
    (out / "topo/topo.json").write_text("{}")
    insp = {"path": "/m", "artifact": "toy", "fingerprint": "0" * 64}

    class Fake:
        def __init__(self):
            self.out, self.info = out, insp
            self.rec = {"install": {"bundle": str(bundle), "routes": ["side"],
                                    "settings": {"data": {}, "host_hidden": 8}},
                        "donor_inspection": insp, "inputs": {"probe_corpus": {"path": "/p"}, "core_corpora": []},
                        "joint_plan": {"cap_gb": 8, "mlx_memory_gb": 4}, "label": ""}

        def stage_dir(self, s):
            (out / s).mkdir(parents=True, exist_ok=True)
            return out / s

        def log_path(self, s):
            return out / f"{s}.log"

    def child(cmd, *, log_path, **kw):
        log_path.write_text("Traceback (most recent call last):\nMemoryError\n")
        raise R.ChildFailed("install route side", 1, log_path)
    monkeypatch.setattr(I, "run_child", child)
    with pytest.raises(R.StageFailedContinue) as e:
        I.INSTALL_RUNNERS["route_side"](Fake(), _ctx(tmp_path))
    assert "rc=1" in e.value.error and e.value.summary == {"failed": e.value.error}
    marker = json.loads((out / "route_side/result.json").read_text())       # what the report reads
    assert marker["failed"] == e.value.error and "MemoryError" in marker["log_tail"]


# ------------------------------------------------------------------ AMCL-08: reads must be observed

def test_once_per_chunk_needs_observed_layer_loads():
    from mlx2.experimental.amoeba.child_rolling import observed_reads
    assert observed_reads(0, 40, 2)["once_per_chunk"] is False      # the counter never moved: nothing was measured
    assert observed_reads(0, 0, 1)["once_per_chunk"] is False       # no streamed layer known
    assert observed_reads(5, 0, 1)["once_per_chunk"] is False
    assert observed_reads(80, 40, 2)["once_per_chunk"] is True
    assert observed_reads(78, 40, 2)["once_per_chunk"] is True      # layer 0 resident, warmed layers: fewer is fine
    r = observed_reads(81, 40, 2)
    assert r["once_per_chunk"] is False and r["layer_loads"] == 81 and r["reads_per_layer"] == 2.025


def test_streamed_only_parity_is_not_exact_without_observed_reads(tmp_path):
    from mlx2.experimental.amoeba.child_rolling import observed_reads
    out = tmp_path / "out"
    (out / "rolling").mkdir(parents=True)
    info = _info()
    args = argparse.Namespace(chunks=2)

    def verdict(reads):
        (out / "rolling/meta.json").write_text(json.dumps({
            "fingerprint": info["fingerprint"], "sequences": 3, "rolling": {},
            "parity": {"ok": True, "texts": [{"bitwise_equal": True}]}, "weight_reads_observed": reads}))
        rc = C._streamed_only_result(args, out, info, A, {"moe_units": "expert"}, {"window": 2}, "host", {}, 1.0)
        return rc, json.loads((out / "parity.json").read_text())["exact"]
    assert verdict(observed_reads(0, 40, 2)) == (3, False)
    assert verdict(observed_reads(80, 40, 2)) == (0, True)


# ------------------------------------------------------------------ AMCL-09: a map-only build's map never reads a core

def test_map_of_a_map_only_build_ignores_a_core_carved_later(tmp_path, monkeypatch):
    b = _build(tmp_path, stage_options={"map": {"device": "cpu", "without_core": True}},
               plan={"moe_units": "expert", "stream": {"mode": "resident"}},
               inputs={"probe_corpus": {"path": str(tmp_path / "probes.jsonl"), "sha256": "x"},
                       "taxonomy": {"path": "/t"}})
    d = b.stage_dir("map")
    (d / "capture").mkdir()
    (d / "capture/meta.json").write_text(json.dumps({"fingerprint": "0" * 64, "widths": [4, 4], "sequences": 3,
                                                     "layers": 2}))
    for rel, text in (("clusters/clusters.json", "{}"), ("clusters/clusters.npz", "z"),
                      ("clusters/virtual_experts.npz", "z"),
                      ("labels/labels.json", json.dumps({"summary": {"clusters": 2, "labelled": 1}, "clusters": []}))):
        (d / rel).parent.mkdir(exist_ok=True)
        (d / rel).write_text(text)
    # a core carved in the same directory after the map-only build was created
    (b.out / "core/carve").mkdir(parents=True)
    np.savez(b.out / "core/core.npz", core=np.ones((2, 4), np.uint8))
    (b.out / "core/carve/carve.json").write_text(json.dumps({"shape": {"widths": [4, 4]}}))
    ST.write_json(b.receipt_path("core"), {"stage": "core", "status": "done", "mlx2_commit": A, "outputs_digest": "c1"})
    monkeypatch.setattr(S, "run_child", lambda *a, **k: pytest.fail("no child expected"))
    before = b.inputs_digest("map")
    S.run_map(b, _ctx(tmp_path))
    m = json.loads((d / "map.json").read_text())
    # the map's inputs do not include the core, so the map does not depend on it either
    assert before["upstream"] == {} and m["core_units_excluded"] == 0 and m["clustered"].startswith("every unit")
    assert not (np.load(d / "core_units.npz")["core"] > 0).any()


# ------------------------------------------------------------------ review R1: a retried route and the report

def _route_runners(monkeypatch, mode, calls):
    def route(bb, ctx):
        calls.append("route_side:" + mode["m"])
        if mode["m"] == "fail":
            raise R.StageFailedContinue("install route side failed (rc=1)", {"result": {"x": 1}}, {"failed": "rc=1"})
        if mode["m"] == "stop":
            raise R.CleanStop("--stop-at reached before the route child")
        if mode["m"] == "hard":
            raise SystemExit("swap guard")
        return {"result": {"x": 2}}, {"arm": "side_box"}

    def report(bb, ctx):
        calls.append("report")
        return {}, {"side": bb.receipt("route_side")["status"]}
    monkeypatch.setattr(C, "kind_runners", lambda _b: {"route_side": route, "report": report})


@pytest.mark.parametrize("how, rc", [("stop", 75), ("hard", 1)])
def test_report_is_remade_when_the_route_retry_only_succeeds_in_a_later_resume(tmp_path, monkeypatch, keep_signals,
                                                                               how, rc):
    b = _install_build(tmp_path)
    mode, calls = {"m": "fail"}, []
    _route_runners(monkeypatch, mode, calls)
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0
    assert b.receipt("report")["failed_upstream"] == {"route_side": b.receipt("route_side")["outputs_digest"]}
    mode["m"] = how                                # the retry does not finish: --stop-at, or a swap-guard exit
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == rc
    assert "continue_past" not in b.receipt("route_side")
    assert b.receipt("report")["summary"] == {"side": "failed"}             # kept until the retry has a result
    mode["m"] = "ok"
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0
    assert calls == ["route_side:fail", "report", f"route_side:{how}", "route_side:ok", "report"]
    assert b.receipt("report")["summary"] == {"side": "done"} and "failed_upstream" not in b.receipt("report")
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0 and len(calls) == 5


def test_report_is_remade_after_the_route_was_retried_alone(tmp_path, monkeypatch, keep_signals):
    b = _install_build(tmp_path)
    mode, calls = {"m": "fail"}, []
    _route_runners(monkeypatch, mode, calls)
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0
    mode["m"] = "ok"
    assert C.execute(b, ["route_side"], _args(), A, redo=[]) == 0           # `mlx2 amoeba route_side --out D`
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0  # plain resume: no refusal at report
    assert calls == ["route_side:fail", "report", "route_side:ok", "report"]
    assert b.receipt("report")["summary"] == {"side": "done"}


def test_report_is_kept_while_the_route_fails_the_same_way(tmp_path, monkeypatch, keep_signals):
    b = _install_build(tmp_path)
    mode, calls = {"m": "fail"}, []
    _route_runners(monkeypatch, mode, calls)
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0
    assert C.execute(b, ["route_side", "report"], _args(), A, redo=[]) == 0
    assert calls == ["route_side:fail", "report", "route_side:fail"] and b.is_done("report")


# ------------------------------------------------------------------ review R2: the lock covers the children

@pytest.fixture
def child_group():
    """A real child in its own session (as run_child starts them); its process group id is its pid."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    yield p
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()


def test_lock_of_a_dead_parent_is_held_while_its_child_group_runs(tmp_path, child_group, capsys):
    b = _build(tmp_path)
    b.lock_path.write_text(json.dumps({"pid": _dead_pid(), "created": time.time(), "started": "then",
                                       "groups": [{"pgid": child_group.pid, "t": time.time()}]}))
    with pytest.raises(SystemExit, match=f"child process group {child_group.pid}"):
        Build(b.out).acquire_lock()
    assert b.lock_path.exists()
    os.killpg(child_group.pid, signal.SIGKILL)
    child_group.wait()
    assert not ST.group_alive(child_group.pid, time.time())
    b.acquire_lock()                                # the child is gone too: stale
    assert "taking over the stale build lock" in capsys.readouterr().err
    assert json.loads(b.lock_path.read_text())["pid"] == os.getpid()
    b.release_lock()
    assert not b.lock_path.exists()


def test_run_child_records_its_group_and_an_abandoned_child_keeps_the_lock(tmp_path, monkeypatch, keep_signals, capsys):
    b = _build(tmp_path)
    seen = {}
    left = []

    def core(bb, ctx):
        R.run_child([sys.executable, "-c", "print('carved')"], stage="core", progress=ctx.progress,
                    log_path=bb.log_path("core"), cwd=tmp_path)
        seen["after_child"] = json.loads(bb.lock_path.read_text())["groups"]
        # a child the run cannot stop (the second Ctrl-C abandons stop_group's wait): still running at exit
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
        left.append(p)
        R.note_child_group(p)
        raise KeyboardInterrupt
    monkeypatch.setattr(C, "kind_runners", lambda _b: {"core": core})
    try:
        assert C.execute(b, ["core"], _args(), A, redo=[]) == 130
        assert len(seen["after_child"]) == 1 and seen["after_child"][0]["pgid"] != os.getpid()
        assert R.CHILD_GROUP_HOOK["fn"] is None
        held = json.loads(b.lock_path.read_text())  # not released: the child still writes the build
        assert [g["pgid"] for g in held["groups"]] == [left[0].pid]         # the finished child's group was dropped
        assert "still running" in capsys.readouterr().err
        code = ("import sys\nfrom mlx2.experimental.amoeba.state import Build\n"
                f"try:\n    Build({str(b.out)!r}).acquire_lock()\nexcept SystemExit as e:\n    print(e)\n"
                "else:\n    print('TOOK IT')\n")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ).stdout
        assert "holds this build" in out and "TOOK IT" not in out
    finally:
        for p in left:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    b.release_lock()                                # the child is gone: the holder can release now
    assert not b.lock_path.exists()


# ------------------------------------------------------------------ review R3: lock takeover details

def test_a_stale_lock_with_this_pid_is_rewritten_so_others_see_a_live_holder(tmp_path):
    b = _build(tmp_path)
    # a dead holder had the pid this process has now
    b.lock_path.write_text(json.dumps({"pid": os.getpid(), "created": time.time() - 86400, "started": "yesterday"}))
    b.acquire_lock()
    held = json.loads(b.lock_path.read_text())
    assert held["started"] != "yesterday" and time.time() - held["created"] < 60
    code = ("from mlx2.experimental.amoeba.state import Build\n"
            f"try:\n    Build({str(b.out)!r}).acquire_lock()\nexcept SystemExit as e:\n    print(e)\n"
            "else:\n    print('TOOK IT')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ).stdout
    assert f"pid {os.getpid()}" in out and "TOOK IT" not in out
    b.acquire_lock()                                # re-entry by the holder keeps the record
    assert json.loads(b.lock_path.read_text())["created"] == held["created"]
    b.release_lock()
    assert not b.lock_path.exists()


def test_a_lock_released_during_the_check_is_retried(tmp_path, monkeypatch):
    b = _build(tmp_path)
    b.lock_path.write_text(json.dumps({"pid": _dead_pid(), "created": time.time(), "started": "then"}))
    real = Path.read_text
    gone = []

    def read_text(self, *a, **k):                   # the holder releases between the failed create and the read
        if self == b.lock_path and not gone:
            gone.append(1)
            self.unlink()
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", read_text)
    b.acquire_lock()
    assert json.loads(real(b.lock_path))["pid"] == os.getpid()
    b.release_lock()


def test_only_one_of_two_resumes_takes_a_stale_lock(tmp_path, monkeypatch):
    b = _build(tmp_path)
    stale = json.dumps({"pid": _dead_pid(), "created": time.time(), "started": "then"})
    b.lock_path.write_text(stale)
    other = {"pid": os.getppid(), "created": time.time(), "started": "the other resume"}
    real_rename, raced = os.rename, []

    def rename(src, dst):
        # between this resume's staleness check and its rename, another resume took the lock over
        if Path(src) == b.lock_path and not raced:
            raced.append(1)
            b.lock_path.write_text(json.dumps(other))
        return real_rename(src, dst)
    monkeypatch.setattr(os, "rename", rename)
    with pytest.raises(SystemExit, match="is taking the build lock; retry"):
        b.acquire_lock()
    assert json.loads(b.lock_path.read_text()) == other                      # the winner's lock is back in place
    assert not list(b.out.glob(".amoeba.lock.stale.*"))


def test_stale_lock_takeover_from_many_processes_has_one_holder(tmp_path):
    b = _build(tmp_path)
    b.lock_path.write_text(json.dumps({"pid": _dead_pid(), "created": time.time(), "started": "then"}))
    code = ("import sys, time\nfrom mlx2.experimental.amoeba.state import Build\n"
            f"try:\n    Build({str(b.out)!r}).acquire_lock()\nexcept SystemExit:\n    print('refused')\n"
            "else:\n    print('holder', flush=True)\n    time.sleep(8)\n")
    ps = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=os.environ)
          for _ in range(6)]
    outs = [p.communicate()[0].strip() for p in ps]
    assert outs.count("holder") == 1 and outs.count("refused") == 5, outs


def test_process_started_reads_elapsed_time():
    born = ST.process_started(os.getpid())
    assert born is not None and 0 <= time.time() - born < 6 * 3600
    assert ST.process_started(_dead_pid()) is None
