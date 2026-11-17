"""CPU tests of the windowed carve corpus (FN calibration a, b, d and the on-policy loss mask; coordinator 2026-11-16)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from research.amoeba.carve import corpus as C  # noqa: E402
from research.amoeba.carve.host import Host  # noqa: E402
from test_amoeba_carve_cpu import Tok, toy  # noqa: E402


class H:
    def encode(self, t):
        return Tok().encode(t)


def _text(rng, n):
    return " ".join("".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 7))) for _ in range(n))


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)
    C.LOSS_FROM.clear()


def test_windows_are_seeded_per_document_and_share_its_split(tmp_path):
    rng = random.Random(0)
    rows = [{"text": _text(rng, 80), "domain": "d", "source": "one-source"} for _ in range(40)]
    p = _write(tmp_path / "c.jsonl", rows)
    split, rec = C.load(f"jsonl:{p}", H(), seq_len=32, n_mean=4, n_fit=12, n_held=8, windows=3, window_seed=1, log=lambda *_: None)
    s2, rec2 = C.load(f"jsonl:{p}", H(), seq_len=32, n_mean=4, n_fit=12, n_held=8, windows=3, window_seed=1, log=lambda *_: None)
    assert split == s2 and rec["offsets"] == rec2["offsets"]
    offs = rec["offsets"]["d"]
    assert len(offs) == 40 and all(1 <= len(o) <= 3 for _, o in offs) and any(o[0] > 0 for _, o in offs)
    # every window of a document lands in one split
    where = {}
    for sp, seqs in split["d"].items():
        for q in seqs:
            where.setdefault(tuple(q), set()).add(sp)
    for r in rows:
        ids = H().encode(r["text"])
        sps = {next(iter(where[tuple(ids[o:o + 32])])) for d, oo in offs if d == C._doc_id(r["text"]) for o in oo}
        assert len(sps) == 1
    # one shared source does not fill splits in file order: mean/fit/held draw from across the file
    first = {tuple(H().encode(r["text"])[:32]) for r in rows[:5]}
    assert not set(map(tuple, split["d"]["fit"])) <= first
    s3, _ = C.load(f"jsonl:{p}", H(), seq_len=32, n_mean=4, n_fit=12, n_held=8, windows=3, window_seed=2, log=lambda *_: None)
    assert s3 != split


def test_shared_prefix_is_stripped(tmp_path):
    rng = random.Random(1)
    pre = "You are a careful reasoner. Think step by step and answer.\n"
    rows = [{"text": pre + _text(rng, 60), "domain": "logic"} for _ in range(30)] + \
           [{"text": _text(rng, 60), "domain": "logic"} for _ in range(10)]
    p = _write(tmp_path / "c.jsonl", rows)
    split, rec = C.load(f"jsonl:{p}", H(), seq_len=24, n_mean=2, n_fit=8, n_held=4, strip_prefix=0.2, log=lambda *_: None)
    assert rec["stripped"]["logic"]["prefix"].startswith("You are a careful") and rec["stripped"]["logic"]["rows"] == 30
    pre_ids = tuple(H().encode(pre)[:24])
    assert all(tuple(q) != pre_ids for v in split["logic"].values() for q in v)


def test_held_file_replaces_held_and_refuses_leaks(tmp_path):
    rng = random.Random(2)
    rows = [{"text": _text(rng, 60), "domain": "d", "split": "held" if i < 5 else None} for i in range(30)]
    rows = [{k: v for k, v in r.items() if v is not None} for r in rows]
    p = _write(tmp_path / "c.jsonl", rows)
    hrows = [{"text": _text(rng, 90), "domain": "d"} for _ in range(6)]
    hp = _write(tmp_path / "h.jsonl", hrows)
    split, rec = C.load(f"jsonl:{p}", H(), seq_len=24, n_mean=2, n_fit=8, n_held=4, held_file=str(hp), held_len=64,
                        log=lambda *_: None)
    assert len(split["d"]["held"]) == 6 and all(len(q) == 64 for q in split["d"]["held"])
    corpus_held = {tuple(H().encode(r["text"])[:24]) for r in rows[:5]}
    assert not corpus_held & {tuple(q) for k in ("mean", "fit", "spare") for q in split["d"][k]}
    leak = _write(tmp_path / "leak.jsonl", hrows + [rows[10]])
    with pytest.raises(SystemExit, match="also in the carve corpus"):
        C.load(f"jsonl:{p}", H(), seq_len=24, n_mean=2, n_fit=8, n_held=4, held_file=str(leak), log=lambda *_: None)


def test_onpolicy_rows_are_tail_windows_with_a_loss_mask(tmp_path):
    rng = random.Random(3)
    doc = _text(rng, 120)
    prompt = doc[:200]
    gen = {"text": prompt + " " + _text(rng, 12), "prompt": prompt, "domain": "gen:d:greedy", "doc_id": C._doc_id(doc)}
    rows = [{"text": doc, "domain": "d"}] + [{"text": _text(rng, 60), "domain": "d"} for _ in range(10)] + [gen]
    p = _write(tmp_path / "c.jsonl", rows)
    split, rec = C.load(f"jsonl:{p}", H(), seq_len=48, n_mean=0, n_fit=4, n_held=0, windows=1, log=lambda *_: None)
    q = [x for v in split["gen:d:greedy"].values() for x in v][0]
    ids = H().encode(gen["text"])
    assert q == ids[-48:]
    n_prompt = len(H().encode(prompt))
    assert C.loss_from(q) == max(1, n_prompt - 32 - (len(ids) - 48))
    # the gen row shares the split of its source document
    (o,), = [oo for d, oo in rec["offsets"]["d"] if d == C._doc_id(doc)]
    win = H().encode(doc)[o:o + 48]
    sp_doc = [k for k, v in split["d"].items() if win in v]
    sp_gen = [k for k, v in split["gen:d:greedy"].items() if q in v]
    assert sp_doc and sp_gen and set(sp_doc) == set(sp_gen)


def test_text_ce_scores_only_masked_targets():
    from research.amoeba.carve.evals import text_ce
    h = Host(model=toy(moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    q = list(range(5, 29))
    full, n_full = text_ce(h, [q])
    C.LOSS_FROM[tuple(q)] = 10
    part, n_part = text_ce(h, [q])
    assert n_full == len(q) - 2 and n_part == len(q) - 10
    # mixed lengths batch by length
    _, n = text_ce(h, [q, list(range(3, 40))])
    assert n == n_part + 37 - 2


def test_driver_runs_the_windowed_loader(tmp_path):
    from research.amoeba.carve import __main__ as D
    rng = random.Random(4)
    p = _write(tmp_path / "c.jsonl", [{"text": _text(rng, 90), "domain": dm} for dm in ("a", "b") for _ in range(30)])
    hp = _write(tmp_path / "h.jsonl", [{"text": _text(rng, 90), "domain": dm} for dm in ("a", "b") for _ in range(4)])
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect,mean,rank,bisect", "--corpus", f"jsonl:{p}", "--windows", "2", "--strip-prefix", "0.2",
            "--held-file", str(hp), "--held-len", "32", "--targets", "0.05", "--export-target", "0.05",
            "--out", str(tmp_path / "r")], host_factory=fac)
    r = json.loads((tmp_path / "r" / "carve.json").read_text())
    assert r["corpus"]["loader"] == "windowed" and r["corpus"]["held_file"] == str(hp)
    assert (tmp_path / "r" / "windows.json").exists() and "offsets" not in r["corpus"]
    assert r["identity"]["windows"] == 2 and "bisect" in r["stages_done"]


def test_masked_kl_full_mask_is_exact_and_masked_positions_get_no_gradient():
    """Coordinator 2026-11-16: a full mask reproduces the unmasked loss exactly; masked positions get zero gradient."""
    from research.amoeba.carve.carving import kl_rows, masked_kl
    rng = np.random.default_rng(0)
    lc = mx.array(rng.standard_normal((2, 7, 11)).astype(np.float32))
    lm = mx.array(rng.standard_normal((2, 7, 11)).astype(np.float32))
    ones = mx.ones((2, 7))
    assert float(masked_kl(lc, lm, ones)) == float(kl_rows(lc, lm).mean())
    assert float(masked_kl(lc, lm, None)) == float(kl_rows(lc, lm).mean())
    w = np.ones((2, 7), np.float32)
    w[0, :4] = 0
    w[1, :2] = 0
    g = mx.grad(lambda x: masked_kl(lc, x, mx.array(w)))(lm)
    g = np.array(g)
    assert np.all(g[0, :4] == 0) and np.all(g[1, :2] == 0)
    assert np.abs(g[0, 4:]).sum() > 0 and np.abs(g[1, 2:]).sum() > 0
    ref = float((np.array(kl_rows(lc, lm)) * w).sum() / w.sum())
    assert abs(float(masked_kl(lc, lm, mx.array(w))) - ref) < 1e-6


def test_driver_mask_reaches_the_mask_gradient(tmp_path, monkeypatch):
    """The driver's rank stage hands masked_kl a mask exactly when a fit row has a loss mask."""
    from research.amoeba.carve import __main__ as D
    seen = []
    real = D.masked_kl
    monkeypatch.setattr(D, "masked_kl", lambda lc, lm, w: seen.append(w is not None) or real(lc, lm, w))
    rng = random.Random(5)
    docs = [{"text": _text(rng, 90), "domain": "a"} for _ in range(30)]
    gen = []
    for r in docs[:20]:
        pr = r["text"][:120]
        gen.append({"text": pr + " " + _text(rng, 3), "prompt": pr, "domain": "gen:a:greedy", "doc_id": C._doc_id(r["text"])})
    p = _write(tmp_path / "c.jsonl", docs + gen)
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "48", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect,mean,rank", "--corpus", f"jsonl:{p}", "--windows", "1", "--keep-prompt", "8",
            "--out", str(tmp_path / "r")], host_factory=fac)
    assert True in seen


def test_val_file_is_the_export_split_and_refuses_overlap(tmp_path):
    from research.amoeba.carve import __main__ as D
    rng = random.Random(6)
    docs = [{"text": _text(rng, 90), "domain": "a"} for _ in range(30)]
    val = [{"text": _text(rng, 90), "domain": "a"} for _ in range(6)]
    held = [{"text": _text(rng, 90), "domain": "a"} for _ in range(4)]
    p, vp, hp = _write(tmp_path / "c.jsonl", docs), _write(tmp_path / "v.jsonl", val), _write(tmp_path / "h.jsonl", held)
    split, rec = C.load(f"jsonl:{p}", H(), seq_len=24, n_mean=2, n_fit=8, n_held=0, windows=2, held_file=str(hp),
                        val_file=str(vp), log=lambda *_: None)
    assert rec["val_counts"]["a"] == len(C.VAL["a"]) and 6 <= len(C.VAL["a"]) <= 12
    with pytest.raises(SystemExit, match="val file"):
        C.load(f"jsonl:{p}", H(), seq_len=24, n_mean=2, n_fit=8, n_held=0, windows=2, held_file=str(hp),
               val_file=str(_write(tmp_path / "v2.jsonl", val + docs[:1])), log=lambda *_: None)
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect,mean,rank,bisect", "--corpus", f"jsonl:{p}", "--windows", "2", "--held-file", str(hp),
            "--val-file", str(vp), "--bisect-subsample", "0.5", "--bisect-tol", "0.005", "--targets", "0.05",
            "--export-target", "0.05", "--out", str(tmp_path / "r")], host_factory=fac)
    r = json.loads((tmp_path / "r" / "carve.json").read_text())
    assert r["export_core"]["chosen_on"].startswith("val documents") and r["export_core"]["bisect_subsample"] == 0.5


def test_empty_or_mismatched_val_is_refused(tmp_path):
    rng = random.Random(7)
    docs = [{"text": _text(rng, 90), "domain": "a"} for _ in range(20)]
    p = _write(tmp_path / "c.jsonl", docs)
    short = _write(tmp_path / "v.jsonl", [{"text": "tiny", "domain": "a"}])
    with pytest.raises(SystemExit, match="unusable"):
        C.load(f"jsonl:{p}", H(), seq_len=24, n_mean=2, n_fit=8, n_held=4, windows=1, val_file=str(short),
               log=lambda *_: None)
    other = _write(tmp_path / "v2.jsonl", [{"text": _text(rng, 90), "domain": "zzz"}])
    with pytest.raises(SystemExit, match="unusable"):
        C.load(f"jsonl:{p}", H(), seq_len=24, n_mean=2, n_fit=8, n_held=4, windows=1, val_file=str(other),
               log=lambda *_: None)
    ok = _write(tmp_path / "v3.jsonl", [{"text": _text(rng, 90), "domain": "a"} for _ in range(3)])
    C.load(f"jsonl:{p}", H(), seq_len=24, n_mean=2, n_fit=8, n_held=4, windows=1, val_file=str(ok), held_len=48,
           log=lambda *_: None)
    assert all(len(q) == 48 for q in C.VAL["a"])          # val is cropped like held, not like the carve windows


def test_code_cut_and_seq_doc_map(tmp_path):
    rng = random.Random(8)
    code = "def f(x):\n    return x + 1\n" + "\n".join(f"    y{i} = f({i})" for i in range(40))
    rows = [{"text": "You are tasked with writing a small helper. " * 5 + "\n" + code + f"\n# {i}", "domain": "code:python"}
            for i in range(12)]
    p = _write(tmp_path / "c.jsonl", rows)
    split, rec = C.load(f"jsonl:{p}", H(), seq_len=24, n_mean=2, n_fit=6, n_held=2, windows=2, code_cut=True,
                        log=lambda *_: None)
    assert rec["code_cut"]
    prose = tuple(H().encode("You are tasked with")[:5])
    for v in split["code:python"].values():
        for q in v:
            assert tuple(q[:5]) != prose
            claims = C.SEQ_DOC[tuple(q)]
            assert claims and all(d in {C._doc_id(r["text"]) for r in rows} for d, _ in claims)   # ORIGINAL doc ids
    assert C.cut_at_first_code("no code here at all") == "no code here at all"


def test_windowed_atlas_records_seq_docs_and_part2_reads_them(tmp_path):
    import subprocess
    from research.amoeba.carve import __main__ as D
    rng = random.Random(9)
    rows = [{"text": _text(rng, 90), "domain": dm, "source": "s"} for dm in ("a", "b") for _ in range(20)]
    p = _write(tmp_path / "c.jsonl", rows)
    fac = lambda log: Host(model=toy(moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    run = tmp_path / "atlas-run"
    D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--stages",
            "inspect,mean,atlas", "--corpus", f"jsonl:{p}", "--windows", "2", "--n-mean", "4", "--n-fit", "4",
            "--n-held", "0", "--atlas-seqs", "40", "--out", str(run)], host_factory=fac)
    meta = json.loads((run / "atlas" / "meta.json").read_text())
    docs = [c[0] for cl in meta["seq_docs"] for c in cl]
    assert docs and set(docs) <= {C._doc_id(r["text"]) for r in rows}
    r = json.loads((run / "carve.json").read_text())
    L, N = r["shape"]["layers"], max(r["shape"]["widths"])
    np.savez(tmp_path / "t2.npz", core=np.zeros((L, N), np.uint8))
    (tmp_path / "part.json").write_text(json.dumps({"passing": []}))
    (tmp_path / "boxes.json").write_text("[]")
    out = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_fn_part2_build.py"), "--nighta", str(tmp_path),
                          "--atlas-run", str(run), "--t2", str(tmp_path / "t2.npz"), "--partition-json",
                          str(tmp_path / "part.json"), "--part1-boxes", str(tmp_path / "boxes.json"), "--model", "toy",
                          "--out", str(tmp_path / "p2"), "--k", "2", "--min-experts", "1", "--top-seqs", "6"],
                         capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": f"{ROOT}/src:{ROOT}", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    b = json.loads((tmp_path / "p2" / "boxes.json").read_text())
    assert b and all((tmp_path / "p2" / "corpora" / f"{x['box']}.jsonl").exists() for x in b)


def test_export_only_on_val_with_random_at_export_count(tmp_path):
    """Box batch settings: no held targets (held is the test set), export cut on val, a random same-count box at the
    export count scored on held once; an empty evaluation set refuses instead of reading 0.0 as a pass."""
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve.evals import english_cost
    rng = random.Random(10)
    p = _write(tmp_path / "c.jsonl", [{"text": _text(rng, 90), "domain": "a"} for _ in range(30)])
    vp = _write(tmp_path / "v.jsonl", [{"text": _text(rng, 90), "domain": "a"} for _ in range(6)])
    hp = _write(tmp_path / "h.jsonl", [{"text": _text(rng, 90), "domain": "a"} for _ in range(6)])
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect,mean,rank,bisect", "--corpus", f"jsonl:{p}", "--windows", "2", "--held-file", str(hp),
            "--val-file", str(vp), "--targets", "", "--export-target", "0.03", "--out", str(tmp_path / "r")],
           host_factory=fac)
    r = json.loads((tmp_path / "r" / "carve.json").read_text())
    assert r["bisect"]["zero"].keys() <= {"ablation", "trained_core"}
    ec = r["export_core"]
    assert ec["chosen_on"].startswith("val") and "random_same_count_held" in ec and "held_meets_0.05" in ec
    h = Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    assert english_cost(h, {"a": {"held": []}}, {}, None)["mean_d_ce"] is None


def test_pooled_english_cost_weights_domains_by_rows(monkeypatch):
    from research.amoeba.carve import evals as E
    vals = {"a": 0.1, "b": 0.4}
    monkeypatch.setattr(E, "text_ce", lambda host, seqs, cfg=None, bs=8: (vals[seqs[0][0]] if cfg else 0.0, 1))
    split = {"a": {"held": [["a"]] * 3}, "b": {"held": [["b"]] * 1}}
    eq = E.english_cost(None, split, {}, lambda: None)
    po = E.english_cost(None, split, {}, lambda: None, pooled=True)
    assert abs(eq["mean_d_ce"] - 0.25) < 1e-9 and abs(po["mean_d_ce"] - 0.175) < 1e-9
    assert po["domain_rows"] == {"a": 3, "b": 1}
