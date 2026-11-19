"""CPU tests of `mlx2 amoeba extract` on a given core (--core-from), without a map (--no-map), on an effective probe
corpus (--exclude-texts / --extra-topic-corpus), with the bar record and the run-wide swap guard; and of the
bridge-lab held-out replay (scripts/amoeba_bridge_heldout.py). No model load."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlx2.experimental.amoeba import config as K
from mlx2.experimental.amoeba import extract as X
from mlx2.experimental.amoeba import runner as R
from mlx2.experimental.amoeba import stages as S
from mlx2.experimental.amoeba.progress import Progress
from mlx2.experimental.amoeba.state import Build

ROOT = Path(__file__).resolve().parents[1]


def _info(L=2, F=8, kind="dense"):
    return {"path": "/models/toy", "artifact": "toy", "checkpoint": "toy@00000000", "fingerprint": "0" * 64,
            "adapter": "Toy", "kind": kind, "layers": L, "hidden": 16, "intermediate": F, "vocab": 100,
            "moe": None if kind == "dense" else {"experts": 4}, "weight_bytes": 1,
            "units": {"dense_neurons": L * F, "moe_expert_units": 0, "moe_neuron_units": 0, "estimate": True}}


def _row(i, split, topic, group, text=None):
    t = text or f"text {i} about {topic} in {group}"
    return {"id": f"tp{i:05d}", "topics": [topic], "split": split, "text": t, "group": group, "license": "CC0",
            "origin": "natural", "sha256": hashlib.sha256(t.encode()).hexdigest()}


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _corpus(tmp_path):
    d = tmp_path / "topic-probes-abc"
    d.mkdir()
    rows = [_row(0, "fit", "sci.physics", "g0"), _row(1, "fit", "sci.physics.optics", "g1"),
            _row(2, "held", "sci.physics", "h0"), _row(3, "held", "sci.physics.quantum", "h1"),
            _row(4, "held", "sci.physics.quantum", "h1"), _row(5, "held", "sci.biology", "h2")]
    return _write(d / "corpus.jsonl", rows), rows


def _ctx(tmp_path):
    c = S.Ctx(Progress(tmp_path / "e.jsonl", tty=False, quiet=True), cpu=True, commit={"commit": "abc", "dirty": False},
              gen_backend="local", stop_at=None)
    return c


def _build(tmp_path, stage_options=None, inputs=None):
    rec = {"kind": "extract", "model": "/m", "checkpoint": "toy@00000000", "inspection": _info(),
           "config": K.resolve("quick", {}, {}, ""), "plan": {"moe_units": "expert", "watchdog_gb": 8, "mlx_memory_gb": 6},
           "inputs": inputs or {"probe_corpus": {"path": "/p.jsonl", "sha256": "x"}, "core_corpora": []},
           "extract": {"topic": "sci.physics"}, "stage_options": stage_options or {}}
    return Build.create(tmp_path / "b", rec)


# ------------------------------------------------------------------ effective probe corpus

def test_exclusions_leave_the_effective_corpus_and_are_accounted(tmp_path):
    src, rows = _corpus(tmp_path)
    ex = _write(tmp_path / "ex.json", [])
    ex.write_text(json.dumps({"probe_ids": ["tp00002"], "sha256": [rows[5]["sha256"]], "groups": ["h1"]}))
    out = tmp_path / "out"
    rec = X.build_effective_corpus(src, out, [ex], [])
    eff = Path(rec["path"])
    assert eff.parent.name == "topic-probes-abc"            # row provenance keeps the probe build's name
    kept = {r["id"] for r in S.read_jsonl(eff)}
    assert kept == {"tp00000", "tp00001"}
    acct = json.loads((out / "inputs/effective-probes.json").read_text())
    assert acct["excluded"]["rows"] == 4 and acct["excluded"]["by_rule"] == {"probe_id": 1, "sha256": 1, "group": 2}
    assert rec["excluded_rows"] == 4 and rec["exclude"][0]["sha256"]


def test_exclusion_lists_refuse_bad_shapes_and_lists_that_match_nothing(tmp_path):
    src, _ = _corpus(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["tp00002"]))
    with pytest.raises(SystemExit, match="JSON object"):
        X.build_effective_corpus(src, tmp_path / "o1", [bad], [])
    none = tmp_path / "none.json"
    none.write_text(json.dumps({"probe_ids": ["zz"]}))
    with pytest.raises(SystemExit, match="matched no row"):
        X.build_effective_corpus(src, tmp_path / "o2", [none], [])


def test_extra_rows_join_drop_duplicates_and_exclusions_and_never_span_splits(tmp_path):
    src, rows = _corpus(tmp_path)
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"groups": ["h0"]}))
    xd = tmp_path / "physics-2026-build"
    xd.mkdir()
    extra = _write(xd / "rows.jsonl", [
        _row(100, "fit", "sci.physics", "n0"),                                   # added
        _row(101, "held", "sci.physics", "n1", text=rows[0]["text"]),            # same text as a corpus row: dropped
        _row(102, "held", "sci.physics", "h0")])                                 # excluded group: dropped
    rec = X.build_effective_corpus(src, tmp_path / "o", [ex], [extra])
    got = {r["id"]: r for r in S.read_jsonl(rec["path"])}
    assert "tp00100" in got and "tp00101" not in got and "tp00102" not in got and "tp00002" not in got
    assert got["tp00100"]["corpus_build"] == "physics-2026-build"
    acct = json.loads(Path(rec["accounting"]).read_text())
    assert acct["extra_files"][0]["counts"] == {"added_fit": 1, "duplicate_text": 1, "excluded_group": 1}
    leak = _write(xd / "leak.jsonl", [_row(200, "fit", "sci.physics", "h1")])    # h1 is a held group
    with pytest.raises(SystemExit, match="never spans splits"):
        X.build_effective_corpus(src, tmp_path / "o3", [], [leak])
    dup = _write(xd / "dup.jsonl", [_row(0, "fit", "sci.physics", "n9")])        # id tp00000 already exists
    with pytest.raises(SystemExit, match="already a probe id"):
        X.build_effective_corpus(src, tmp_path / "o4", [], [dup])
    nolic = _write(xd / "nolic.jsonl", [{**_row(300, "fit", "sci.physics", "n3"), "license": ""}])
    with pytest.raises(SystemExit, match="license"):
        X.build_effective_corpus(src, tmp_path / "o5", [], [nolic])


def test_stages_read_the_effective_corpus(tmp_path):
    b = _build(tmp_path)
    assert S.probe_corpus(b) == "/p.jsonl"
    b.rec["inputs"]["probe_corpus_effective"] = {"path": "/eff/corpus.jsonl"}
    assert S.probe_corpus(b) == "/eff/corpus.jsonl"


# ------------------------------------------------------------------ given core, no map

def _core_file(tmp_path, m):
    d = tmp_path / "t2"
    d.mkdir(exist_ok=True)
    np.savez(d / "t2_cut.npz", core=m.astype(np.uint8), note=np.array("x"))
    (d / "core27b.json").write_text("{}")
    return d / "t2_cut.npz"


def test_external_core_checks_layout_and_records_provenance(tmp_path):
    m = np.zeros((2, 8), bool)
    m[:, :5] = True
    f = _core_file(tmp_path, m)
    cf = X.external_core(f"{f}:core", "Pierre 2026-11-15", _info())
    assert cf["units"] == 10 and cf["widths"] == [8, 8] and cf["receipts_beside"][0]["path"].endswith("core27b.json")
    assert X.external_core(str(f), "r", _info())["key"] == "core"
    with pytest.raises(SystemExit, match="--core-from-reason"):
        X.external_core(str(f), "", _info())
    with pytest.raises(SystemExit, match="shape"):
        X.external_core(str(f), "r", _info(F=16))
    with pytest.raises(SystemExit, match="dense models only"):
        X.external_core(str(f), "r", _info(kind="moe"))
    with pytest.raises(SystemExit, match="no array"):
        X.external_core(f"{f}:Z", "r", _info())
    full = _core_file(tmp_path, np.ones((2, 8), bool))
    with pytest.raises(SystemExit, match="no box"):
        X.external_core(str(full), "r", _info())


def test_given_core_stage_writes_the_core_and_its_widths(tmp_path, monkeypatch):
    m = np.zeros((2, 8), bool)
    m[0, :3] = True
    f = _core_file(tmp_path, m)
    cf = X.external_core(str(f), "r", _info())
    b = _build(tmp_path, stage_options={"core": {"device": "cpu", "core_from": cf}})

    def fake_check(b_, ctx, d, eng):
        (d / "corecheck.json").write_text(json.dumps({"english_only": True, "english_d_ce": 0.1}))
        return d / "corecheck.json"
    monkeypatch.setattr(X, "corecheck", fake_check)
    monkeypatch.setattr(X, "_reuse", lambda *a: pytest.fail("a given core never reads the cache"))
    monkeypatch.setattr(X, "publish", lambda *a, **k: pytest.fail("a given core never feeds the cache"))
    outs, summ = X.run_xcore(b, _ctx(tmp_path))
    assert (np.load(b.out / "core/core.npz")["core"] > 0).sum() == 3
    assert S.carve_widths(b) == [8, 8]                     # no carve receipt: widths from core_source.json
    assert summ["units"] == 3 and summ["english_only"] is True and "core_source" in outs
    assert X._source(b, "core")["core_from"]["sha256"] == cf["sha256"]
    f.write_bytes(f.read_bytes() + b"\0")                  # the mask changed after the build started: refuse
    with pytest.raises(SystemExit, match="changed"):
        X.run_xcore(b, _ctx(tmp_path))


def test_no_map_stage_and_an_unseeded_box(tmp_path, monkeypatch):
    b = _build(tmp_path, stage_options={"map": {"device": "cpu", "skip": "too slow"}})
    monkeypatch.setattr(X, "run_map", lambda *a: pytest.fail("--no-map must not capture"))
    outs, summ = X.run_xmap(b, _ctx(tmp_path))
    assert summ == {"map": "skipped", "reason": "too slow"} and "skipped" in outs
    Z, seed = X._topic_seed(b, "sci.physics", (2, 8))
    assert Z is None and "too slow" in seed["seed"]
    assert X._source(b, "map") == {"reused_from_cache": False, "skipped": "too slow"}


# ------------------------------------------------------------------ the bar

def test_bar_record_states_both_conditions(tmp_path):
    b = _build(tmp_path)
    b.rec["inspection"]["artifact"] = "Qwen3.8-27B-BF16-q8-mlx2"
    ver = {"box": "sci_physics", "verdict": "miss", "units": 40, "export_val": 0.02, "export_held": 0.04,
           "random_same_count_held": 0.06, "held_meets_target": True, "beats_random": False,
           "miss_reasons": ["no clear margin"]}
    bar = X.bar_record(b, _ctx(tmp_path), ver)
    c = bar["conditions"]
    assert bar["checkpoint"] == "Qwen3.8-27B-BF16-q8-mlx2@" + "0" * 64 and bar["mlx2_commit"]["commit"] == "abc"
    assert c["held_within_target"]["pass"] is True and c["held_within_target"]["target_nats"] == 0.05
    assert c["beats_random"]["pass"] is False and c["beats_random"]["held_over_random"] == round(0.04 / 0.06, 4)
    assert c["beats_random"]["max_ratio"] == 0.5 and bar["pass"] is False
    assert "validation" in bar["size_chosen_on"] and "+0.03" in bar["size_chosen_on"]


# ------------------------------------------------------------------ swap guard

def test_swap_guard_refuses_a_child_and_stops_a_running_one(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "swap_used_gb", lambda: 5.0)
    R.arm_swap_guard(8, "test waiver")
    try:
        assert R.SWAP_GUARD["baseline_gb"] == 5.0 and R.swap_over(12.9) is None
        assert "test waiver" in R.swap_over(13.1)
        prog = Progress(tmp_path / "e.jsonl", tty=False, quiet=True)
        monkeypatch.setattr(R, "swap_used_gb", lambda: 14.0)
        with pytest.raises(SystemExit, match="swap guard"):
            R.run_child([sys.executable, "-c", "print(1)"], stage="box", progress=prog, log_path=tmp_path / "l.log")
        state = {"n": 0}

        def rising():
            state["n"] += 1
            return 5.0 if state["n"] == 1 else 20.0      # fine at the start check, over at the first sample
        monkeypatch.setattr(R, "swap_used_gb", rising)
        monkeypatch.setattr(R, "footprint_gb", lambda pid: None)
        with pytest.raises(SystemExit, match="stopped: swap guard"):
            R.run_child([sys.executable, "-c", "import time; time.sleep(60)"], stage="box", progress=prog,
                        log_path=tmp_path / "l2.log", sample_s=0.2)
    finally:
        R.arm_swap_guard(0, None)
    assert R.SWAP_GUARD["growth_gb"] is None and R.swap_over(1e9) is None


def test_cli_refuses_a_swap_allowance_without_a_waiver_and_a_reason_without_a_core(tmp_path):
    from mlx2.experimental.amoeba.cli import main
    with pytest.raises(SystemExit, match="without --core-from"):
        main(["extract", "--model", "x", "--topic", "sci.physics", "--out", str(tmp_path / "o"),
              "--skip-gap-screen", "t", "--core-from-reason", "r"])
    import argparse

    from mlx2.experimental.amoeba.cli import check_swap_args
    with pytest.raises(SystemExit, match="--swap-waiver"):
        check_swap_args(argparse.Namespace(swap_growth_gb=8.0, swap_waiver=""))
    check_swap_args(argparse.Namespace(swap_growth_gb=8.0, swap_waiver="Pierre"))


def test_resume_rehashes_the_effective_corpus_its_lists_and_the_given_core(tmp_path):
    from mlx2.experimental.amoeba.cli import check_inputs
    src, _ = _corpus(tmp_path)
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"probe_ids": ["tp00002"]}))
    files = {}
    for name in ("manifest.json", "tax.json"):
        files[name] = tmp_path / name
        files[name].write_text("{}")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()       # noqa: E731
    inputs = {"probe_corpus": {"path": str(src), "sha256": sha(src)},
              "probe_manifest": {"path": str(files["manifest.json"]), "sha256": sha(files["manifest.json"])},
              "taxonomy": {"path": str(files["tax.json"]), "sha256": sha(files["tax.json"])}, "core_corpora": []}
    m = np.zeros((2, 8), bool)
    m[0, 0] = True
    cf = X.external_core(str(_core_file(tmp_path, m)), "r", _info())
    b = _build(tmp_path, stage_options={"core": {"core_from": cf}}, inputs=inputs)
    b.rec["inputs"]["probe_corpus_effective"] = X.build_effective_corpus(src, b.out, [ex], [])
    check_inputs(b)
    ex.write_text(json.dumps({"probe_ids": ["tp00003"]}))
    with pytest.raises(SystemExit, match="changed"):
        check_inputs(b)


# ------------------------------------------------------------------ bridge-lab held-out replay

def test_bridge_heldout_replay_lists_the_untouched_groups(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import amoeba_bridge_heldout as H
    rows = [_row(i, "held", "sci.physics", f"g{i // 2}", text="w " * (3 + i)) for i in range(12)] + \
        [_row(50 + i, "fit", "sci.physics", f"f{i}", text="w " * 8) for i in range(4)]
    enc = lambda t: t.split()                                                    # noqa: E731
    chosen, eligible = H.select(rows, enc, "sci.physics", 7, 2, 3, 5)
    assert len(chosen["held"]) == 3 and all(len(enc(r["text"])) >= 5 for r in eligible["held"])
    assert len(eligible["held"]) == 10                                           # texts 0, 1 are too short
    # the same permutation the lab draws: fit first, then held, from one default_rng(seed + 3)
    g = np.random.default_rng(7 + 3)
    g.permutation(4)
    perm = g.permutation(12)
    first = [rows[i] for i in perm if len(enc(rows[i]["text"])) >= 5][:3]
    assert [r["id"] for r in first] == [r["id"] for r in chosen["held"]]


def test_physbox_window_script_is_valid_zsh_and_quotes_its_mask_argument():
    p = ROOT / "scripts/amoeba27b_physbox_inner.sh"
    if not p.is_file():
        pytest.skip("window shell scripts are not part of the source export")
    assert subprocess.run(["zsh", "-n", str(p)], check=False).returncode == 0
    txt = p.read_text()
    assert '"${T2}:core"' in txt and "$T2:core" not in txt      # zsh reads `:c` after a bare $T2 as a modifier
    assert "--exclude-texts" in txt and "--swap-growth-gb" in txt and "--cap-gb 80" in txt


# ------------------------------------------------------------------ review follow-ups (2026-11-18)

def test_an_exclusion_list_for_another_corpus_and_a_core_from_another_checkpoint_refuse(tmp_path):
    src, _ = _corpus(tmp_path)
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"probe_ids": ["tp00002"], "probes": {"sha256": "f" * 64}}))
    with pytest.raises(SystemExit, match="made for probe corpus"):
        X.build_effective_corpus(src, tmp_path / "o", [ex], [])
    m = np.zeros((2, 8), bool)
    m[0, 0] = True
    f = _core_file(tmp_path, m)
    (f.parent / "core27b.json").write_text(json.dumps({"fingerprint": "1" * 64}))
    with pytest.raises(SystemExit, match="another checkpoint"):
        X.external_core(str(f), "r", _info())
    (f.parent / "core27b.json").write_text(json.dumps({"fingerprint": "0" * 64}))
    assert X.external_core(str(f), "r", _info())["units"] == 1


def test_a_bridge_split_manifest_excludes_its_reserved_final_test(tmp_path):
    src, _ = _corpus(tmp_path)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    man = tmp_path / "split.json"
    man.write_text(json.dumps({"schema": "mlx2.amoeba-bridge-split-manifest.v1", "probes_sha256": sha,
                               "selection": {"test": ["tp00002"]},           # spent: stays
                               "reserved_final_test": {"probe_ids": ["tp00003"], "groups": ["h1"], "rule": "x"}}))
    rec = X.build_effective_corpus(src, tmp_path / "o", [man], [])
    kept = {r["id"] for r in S.read_jsonl(rec["path"])}
    assert "tp00002" in kept and not {"tp00003", "tp00004"} & kept
    man.write_text(json.dumps({"schema": "mlx2.amoeba-bridge-split-manifest.v1", "probes_sha256": "e" * 64,
                               "reserved_final_test": {"probe_ids": ["tp00003"]}}))
    with pytest.raises(SystemExit, match="made for probe corpus"):
        X.build_effective_corpus(src, tmp_path / "o2", [man], [])


def test_caches_key_on_the_effective_corpus_and_resume_keeps_the_swap_guard(tmp_path):
    import argparse

    from mlx2.experimental.amoeba.cli import swap_settings
    b = _build(tmp_path)
    b.rec["inputs"]["probe_corpus"]["sha256"] = "orig"
    assert S.probe_corpus_sha(b) == "orig"
    b.rec["inputs"]["probe_corpus_effective"] = {"path": "/e/corpus.jsonl", "sha256": "eff"}
    assert S.probe_corpus_sha(b) == "eff"
    b.rec["options"] = {"swap_guard": {"growth_gb": 8.0, "waiver": "Pierre day"}}
    assert swap_settings(b, argparse.Namespace(swap_growth_gb=0.0, swap_waiver="")) == (8.0, "Pierre day")
    assert swap_settings(b, argparse.Namespace(swap_growth_gb=2.0, swap_waiver="night")) == (2.0, "night")


def test_a_refused_exclusion_leaves_no_build_behind(tmp_path, monkeypatch):
    from mlx2.experimental.amoeba import cli as C
    src, _ = _corpus(tmp_path)
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"corpus_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}))
    tax = tmp_path / "tax.json"
    tax.write_text(json.dumps({"version": "1", "nodes": [{"id": "sci.physics"}]}))
    core_corpus = tmp_path / "english.jsonl"
    core_corpus.write_text("")
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"probe_ids": ["nope"]}))
    monkeypatch.setattr(C, "inspect_checkpoint", lambda p: _info())
    monkeypatch.setattr(C, "make_plan", lambda *a, **k: {"fits": True, "stream": {"mode": "resident"}})
    monkeypatch.setattr(C, "format_plan", lambda *a: "")
    monkeypatch.setattr(C, "check_commit", lambda d: {"commit": "abc", "dirty": False})
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="matched no row"):
        C.main(["extract", "--model", "/m", "--topic", "sci.physics", "--out", str(out), "--skip-gap-screen", "t",
                "--probe-corpus", str(src), "--probe-manifest", str(man), "--taxonomy", str(tax),
                "--core-corpus", str(core_corpus), "--exclude-texts", str(ex), "--cpu"])
    assert not (out / "build.json").exists()
