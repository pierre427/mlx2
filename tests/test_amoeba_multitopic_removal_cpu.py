"""Multi-topic extract (one box from several taxonomy ids), --box-fractions, and the removal check (child_removal).

CPU only. The removal check runs on the tiny Hy4-flash fixture through the carve host (hy_latent neuron sites).
"""
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parents[1]

from mlx2.experimental.amoeba import child_removal as CR  # noqa: E402
from mlx2.experimental.amoeba import config as K  # noqa: E402
from mlx2.experimental.amoeba.cli import check_topic, main, topic_matches, topic_nodes  # noqa: E402


def _tax(tmp_path):
    tax = tmp_path / "t.json"
    tax.write_text(json.dumps({"nodes": [{"id": "lang"}, {"id": "lang.cmn"}, {"id": "lang.jpn"}, {"id": "lang.hin"},
                                         {"id": "lang.eng"}]}))
    return tax


def test_multi_topic_ids_are_each_checked(tmp_path):
    tax = _tax(tmp_path)
    check_topic("lang.cmn+lang.jpn+lang.hin", tax)
    with pytest.raises(SystemExit, match="lang.xyz"):
        check_topic("lang.cmn+lang.xyz", tax)
    with pytest.raises(SystemExit, match="distinct"):
        topic_nodes("lang.cmn+lang.cmn")
    assert topic_matches("lang.cmn+lang.hin", "lang.hin") and topic_matches("lang.cmn+lang.hin", "lang.cmn.x")
    assert not topic_matches("lang.cmn+lang.hin", "lang.jpn") and not topic_matches("lang.cmn", "lang.cmnx")


def test_multi_topic_needs_skip_gap_and_box_fractions_are_validated(tmp_path):
    base = ["extract", "--model", "x", "--topic", "lang.cmn+lang.jpn", "--out", str(tmp_path / "o"),
            "--taxonomy", str(_tax(tmp_path))]
    with pytest.raises(SystemExit, match="--skip-gap-screen"):
        main(base + ["--against", "y"])
    with pytest.raises(SystemExit, match="ascending"):
        main(base + ["--skip-gap-screen", "strip", "--box-fractions", "0.1,0.05"])


def test_box_fractions_reach_the_carve_as_size_targets(tmp_path):
    from mlx2.experimental.amoeba.child_boxbatch import carve_argv
    import research.amoeba.carve.__main__ as D
    cfg = K.resolve("quick", {"box_fractions": "0.05,0.1"}, {}, "")
    plan = {"batches": {"carve_box": {"bs": 2}}, "watchdog_gb": 80, "mlx_memory_gb": 76, "moe_units": "neuron"}
    spec = {"model": "/m", "compute": cfg["compute"], "data": cfg["data"], "plan": plan, "verdict": cfg["verdict"],
            "grad_gate": cfg["grad_gate"], "core": str(tmp_path / "core.npz")}
    assert D.parse(carve_argv(spec, tmp_path / "bs", tmp_path / "d")).size_targets == "0.05,0.1"
    cfg0 = K.resolve("quick", {}, {}, "")
    assert D.parse(carve_argv({**spec, "compute": cfg0["compute"]}, tmp_path / "bs", tmp_path / "d")).size_targets == ""


def test_boxset_builder_makes_one_box_from_several_topics(tmp_path):
    rows = []
    for t in ("lang.cmn", "lang.jpn", "lang.hin", "lang.eng"):
        for i in range(20):
            rows.append({"id": f"{t}-{i}", "topics": [t], "split": "fit" if i < 14 else "held",
                         "text": f"{t} document {i} " * 40, "group": f"{t}/g{i}", "license": "CC0"})
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "bs"
    subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_fn_boxset_build.py"), "--corpus", str(corpus),
                    "--out", str(out), "--topic", "lang.cmn+lang.jpn+lang.hin", "--min-fit", "8",
                    "--carve-prompts", "2", "--val-prompts", "1", "--held-prompts", "2", "--min-carve", "1",
                    "--min-val", "1", "--min-held", "1"], check=True, capture_output=True,
                   env={"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "PATH": "/usr/bin:/bin"})
    spec = json.loads((out / "boxes.json").read_text())
    assert [b["node"] for b in spec["boxes"]] == ["lang.cmn+lang.jpn+lang.hin"]
    name = spec["boxes"][0]["box"]
    seen = {t for f in ("docs", "val", "held") for x in (out / name / f"{f}.jsonl").read_text().splitlines() if x
            for t in json.loads(x)["topics"]}
    assert seen == {"lang.cmn", "lang.jpn", "lang.hin"}                  # English never enters the box


# ------------------------------------------------------------------ removal check

class Tok:
    def encode(self, text, add_special_tokens=False):
        return [5 + (b % 90) for b in text.encode()]

    def decode(self, ids):
        return ""


@pytest.fixture(scope="module")
def host():
    from research.amoeba.carve.host import Host
    from test_hy_v4_flash_cpu import build, checkpoint, tiny_config
    cfg = tiny_config(compressed_indexer_group_topk=32)
    m, _ = build(cfg, checkpoint(cfg))
    return Host(model=m, tokenizer=Tok(), fingerprint="tiny", checkpoint="tiny", moe_units="neuron",
                log=lambda *a: None)


def _rows(cls, n, seed):
    rng = np.random.default_rng(seed)
    return [{"text": "".join(chr(97 + int(c)) for c in rng.integers(0, 26, 40)), "domain": cls,
             "doc_id": f"{cls}{seed}-{i}"} for i in range(n)]


def _cuts(host, core, fracs, seed=0):
    rng = np.random.default_rng(seed)
    L, W = core.shape
    order = rng.permutation(np.flatnonzero(~core.reshape(-1) & _valid(host).reshape(-1)))
    total = int(sum(host.widths))
    cuts = {}
    for f in fracs:
        c = core.copy().reshape(-1)
        c[order[:int(f * total)]] = True
        cuts[f"size{f:g}"] = c.reshape(L, W)
    return cuts


def _valid(host):
    L, W = len(host.widths), max(host.widths)
    v = np.zeros((L, W), bool)
    for l, n in enumerate(host.widths):
        v[l, :n] = True
    return v


def test_removal_check_picks_largest_eligible_and_scores_held_once(host, tmp_path):
    L, W = len(host.widths), max(host.widths)
    core = _valid(host) & (np.random.default_rng(3).random((L, W)) < 0.5)
    cuts = _cuts(host, core, [0.02, 0.05, 0.1])
    kw = dict(core=core, cuts=cuts, nontarget_val=_rows("eng", 3, 1) + _rows("code", 2, 2),
              nontarget_held=_rows("eng", 3, 3), target_val=_rows("lang.cmn", 3, 4), target_held=_rows("lang.cmn", 3, 5),
              seq_len=24, windows_per_doc=2, specificity=0.0, floor=-1.0, log=lambda *a: None,
              identity={"checkpoint": "tiny@tiny"}, ledger=tmp_path / "removal-held-ledger.jsonl")
    rec = CR.run(host, out=tmp_path / "r", val_margin=10.0, **kw)          # every cut eligible: the largest is chosen
    val = json.loads((tmp_path / "r/val.json").read_text())
    assert [c["cut"] for c in val["cuts"]] == ["size0.02", "size0.05", "size0.1"]
    k = [c["box_nontarget_val"]["pooled"] for c in val["cuts"]]
    rows = val["cuts"][0]["box_nontarget_val"]["rows"]
    assert all(x > 0 for x in k) and set(rows) == {"code", "eng"} and 3 <= rows["code"] <= 4 and 5 <= rows["eng"] <= 6
    assert rec["chosen"] == "size0.1" and rec["scored_once"] and "nontarget_held_kl" in rec
    box = np.load(tmp_path / "r/box_mask.npz")["box"] > 0
    assert np.array_equal(box, cuts["size0.1"] & ~core)
    with pytest.raises(SystemExit, match="exactly once"):
        CR.run(host, out=tmp_path / "r", val_margin=10.0, **kw)
    # a margin no cut meets: nothing chosen, held never scored
    rec2 = CR.run(host, out=tmp_path / "r2", val_margin=-1.0, **kw)
    assert rec2["chosen"] is None and not (tmp_path / "r2/held.json").exists()
    # the margin between the smallest and the next cut picks the smallest
    mid = (k[0] + k[1]) / 2 if k[0] < k[1] else None
    if mid is not None:
        assert CR.run(host, out=tmp_path / "r3", val_margin=mid, **kw)["chosen"] == "size0.02"
    # the ledger in the run root refuses the same box on the same held data from ANY out dir
    with pytest.raises(SystemExit, match="already scored"):
        CR.run(host, out=tmp_path / "another", val_margin=10.0, **kw)
    ledger = [json.loads(x) for x in (tmp_path / "removal-held-ledger.jsonl").read_text().splitlines()]
    assert [r["cut"] for r in ledger] == ["size0.1"] + (["size0.02"] if mid is not None else [])
    assert all(len(r["box_sha256"]) == 64 and len(r["held_sha256"]) == 64 for r in ledger)


def test_removal_refuses_val_and_held_sharing_documents(host, tmp_path):
    L, W = len(host.widths), max(host.widths)
    core = _valid(host) & (np.random.default_rng(3).random((L, W)) < 0.5)
    shared = _rows("eng", 2, 1)
    kw = dict(core=core, cuts=_cuts(host, core, [0.05]), target_val=_rows("lang.cmn", 2, 4),
              target_held=_rows("lang.cmn", 2, 5), seq_len=24, log=lambda *a: None, ledger=tmp_path / "l.jsonl")
    with pytest.raises(SystemExit, match="share"):                          # same doc ids
        CR.run(host, out=tmp_path / "a", nontarget_val=shared, nontarget_held=shared, **kw)
    retagged = [{**r, "doc_id": r["doc_id"] + "-copy"} for r in shared]
    with pytest.raises(SystemExit, match="share"):                          # same text under another id
        CR.run(host, out=tmp_path / "b", nontarget_val=shared, nontarget_held=retagged, **kw)


def test_scorer_memory_is_per_batch_not_per_window(host):
    """The full model's logits are never retained across batches: peak memory does not grow with the window count."""
    L, W = len(host.widths), max(host.widths)
    keep = CR.keep_without(_valid(host) & (np.random.default_rng(9).random((L, W)) < 0.2), host.widths)
    few, _ = CR.windows(host, _rows("eng", 2, 7), 24, 1, 0, 32)
    many, _ = CR.windows(host, _rows("eng", 24, 8), 24, 1, 0, 32)
    sc = CR.Scorer(host, bs=2)
    sc.kl_many(few, {"a": keep, "b": keep})                                  # warm-up (compiled kernels, caches)
    peaks = []
    for wins in (few, many):
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        r = sc.kl_many(wins, {"a": keep, "b": keep})
        peaks.append(mx.get_peak_memory() - base)
        assert np.array_equal(r["a"], r["b"])                                # each arm is scored against the same full
    assert len(many) >= 10 * len(few) // 2
    assert peaks[1] <= 1.25 * peaks[0] + (1 << 20), peaks                    # O(batch), not O(windows)
    assert not any(isinstance(v, dict) and len(v) for v in vars(sc).values())   # no per-window cache


def test_removal_kl_is_the_zero_mask_model(host):
    """The removal arm is the serving law: keep = all units - box, zero ablation, every position."""
    L, W = len(host.widths), max(host.widths)
    box = _valid(host) & (np.random.default_rng(9).random((L, W)) < 0.2)
    keep = CR.keep_without(box, host.widths)
    wins, _ = CR.windows(host, _rows("eng", 2, 7), 24, 1, 0, 32)
    sc = CR.Scorer(host)
    got = sc.kl(wins, keep, "x")
    ids = mx.array([wins[0][1]])
    host.ctl.reset()
    full = host.logits_at(ids, list(range(23)))
    host.ctl.core = [mx.array(keep[l, :n].astype(np.float32)) for l, n in enumerate(host.widths)]
    m = host.logits_at(ids, list(range(23)))
    host.ctl.reset()
    from research.amoeba.carve.carving import kl_rows
    assert abs(float(kl_rows(full, m).mean()) - got[0]) < 1e-6
    assert np.allclose(sc.kl(wins, CR.keep_without(np.zeros_like(box), host.widths), "y"), 0.0, atol=1e-7)


def test_removal_refuses_a_cut_that_drops_core_and_a_relaxed_bar(host, tmp_path):
    L, W = len(host.widths), max(host.widths)
    core = _valid(host)
    bad = {"x": np.zeros((L, W), bool)}
    with pytest.raises(SystemExit, match="drops core"):
        CR.run(host, core=core, cuts=bad, nontarget_val=_rows("eng", 2, 1), nontarget_held=[],
               target_val=_rows("lang.cmn", 2, 2), target_held=[], out=tmp_path / "x", seq_len=24, log=lambda *a: None,
               ledger=tmp_path / "l.jsonl")
    with pytest.raises(SystemExit, match="never relaxed"):
        CR.main(["--model", "m", "--core", "c.npz", "--cut", "c.npz", "--nontarget-val", "a", "--nontarget-held", "a",
                 "--target-val", "a", "--target-held", "a", "--out", str(tmp_path / "y"), "--bar", "0.08"])


def test_held_ledger_lives_with_the_held_data_and_keys_on_documents(host, tmp_path):
    """The default ledger is next to the held files (absolute); another one needs an override, which is recorded;
    dropping one held row cannot reopen held for the same box."""
    L, W = len(host.widths), max(host.widths)
    core = _valid(host) & (np.random.default_rng(3).random((L, W)) < 0.5)
    data = tmp_path / "data"
    data.mkdir()
    held_nt, held_t = _rows("eng", 3, 3), _rows("lang.cmn", 3, 5)
    for name, rows in (("nt.jsonl", held_nt), ("t.jsonl", held_t)):
        (data / name).write_text("\n".join(json.dumps(r) for r in rows))
    paths = [data / "nt.jsonl", data / "t.jsonl"]
    kw = dict(core=core, cuts=_cuts(host, core, [0.05]), nontarget_val=_rows("eng", 3, 1),
              target_val=_rows("lang.cmn", 3, 4), seq_len=24, specificity=0.0, floor=-1.0, val_margin=10.0,
              log=lambda *a: None)
    assert CR.default_ledger(paths) == (data / CR.LEDGER_NAME).resolve()
    CR.run(host, out=tmp_path / "run1" / "removal", nontarget_held=held_nt, target_held=held_t, held_paths=paths, **kw)
    assert (data / CR.LEDGER_NAME).exists() and not (tmp_path / "run1" / CR.LEDGER_NAME).exists()
    # a new run root, the same box, one held row dropped: still refused
    with pytest.raises(SystemExit, match="already scored"):
        CR.run(host, out=tmp_path / "run2" / "removal", nontarget_held=held_nt[1:], target_held=held_t,
               held_paths=paths, **kw)
    # another ledger: refused without an override, accepted (and recorded) with one
    with pytest.raises(SystemExit, match="ledger-override"):
        CR.run(host, out=tmp_path / "run3", nontarget_held=held_nt, target_held=held_t, held_paths=paths,
               ledger=tmp_path / "elsewhere.jsonl", **kw)
    CR.run(host, out=tmp_path / "run4", nontarget_held=held_nt, target_held=held_t, held_paths=paths,
           ledger=tmp_path / "elsewhere.jsonl", ledger_override="test: separate audit ledger", **kw)
    rec = json.loads((tmp_path / "elsewhere.jsonl").read_text().splitlines()[0])
    assert rec["ledger_override"] == "test: separate audit ledger" and len(rec["held_doc_ids"]) == 6
    with pytest.raises(SystemExit, match="no held ledger"):
        CR.run(host, out=tmp_path / "run5", nontarget_held=held_nt, target_held=held_t, **kw)
