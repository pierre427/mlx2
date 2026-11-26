"""Readout checks (research/ens/readout_check.py) on a tiny random bf16 Qwen3 (CPU only; synthetic documents and
questions, no checkpoint, no private corpus)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from research.ens import corpus_tile as CT
from research.ens import corpus_tile_eval as E
from research.ens import readout_check as R
from research.ens import side_tile as ST
from tests.test_ens_side_tile_cpu import FakeTok, prose, tiny

LETTERS = "abcdefghijklmnopqrstuvwxyz"


@pytest.fixture(autouse=True)
def cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _f32(x):
    return np.array(x.astype(mx.float32))


def _snap(module):
    return {k: _f32(v) for k, v in tree_flatten(module.parameters())}


def _doc_text(i, n_words=260):
    """Each document draws its words from its own 4-letter alphabet (separable by construction)."""
    rng = random.Random(100 + i)
    alpha = LETTERS[(4 * i) % 26:(4 * i) % 26 + 4] or "wxyz"
    return " ".join("".join(rng.choice(alpha) for _ in range(rng.randint(3, 6))) for _ in range(n_words))


@pytest.fixture
def world():
    docs = [{"id": f"d{i}", "title": f"Doc {i}", "batch": "course", "text": _doc_text(i)} for i in range(10)]
    split = {"train": [f"d{i}" for i in range(6)], "held_out": [f"d{i}" for i in range(6, 10)]}
    return ST.world_from(tiny(), FakeTok(), "tiny-qwen3@000000000000", docs, split, {"corpus.jsonl": "synthetic"})


def questions():
    rows, n = [], 0
    for i in range(10):
        for j in range(2):
            n += 1
            rows.append({"qid": f"q{n:03d}", "doc_id": f"d{i}", "doc_split": "train" if i < 6 else "held_out",
                         "question": f"Which code is tied to item {i}-{j} of the note?",
                         "rephrase": f"Name the code for item {i}-{j}.",
                         "answer": f"Zorvex{i}{j}", "aliases": [f"zv{i}{j}"], "batch": "course",
                         "evidence": f"secret evidence sentence {i}{j}"})
    return rows


def args(*extra):
    return R.parser().parse_args(["--run", "unused", "--kind", "side_tile", "--questions", "unused", "--out", "unused",
                                  "--steps", "6", "--max-new", "3", "--gen-batch", "4", "--sep-docs", "4",
                                  "--sep-chunks", "3", "--no-memory-gate", *extra])


def side_tile(W):
    a = ST.parser().parse_args(["train", "--corpus", "unused", "--out", "/nonexistent", "--width", "32", "--heads",
                                "2", "--no-memory-gate", "--stat-rows", "16", "--batch", "3"])
    lay = ST.aligned_layers(W["cfg"].num_hidden_layers, a.every)
    rows = [W["tok"].encode(prose(random.Random(i), 30))[:128] for i in range(16)]
    orig = CT.text_chunks
    CT.text_chunks = lambda *x, **k: rows[:8]
    try:
        stats = ST.ladder_stats(W, lay, a.width, seed=0, n_rows=16)
    finally:
        CT.text_chunks = orig
    mx.random.seed(0)
    t = ST.make_tile(W, stats, a)
    for u in t.U:
        u.weight = mx.random.normal(u.weight.shape) * 0.05
    t.freeze()
    return t


def lora_tile(W):
    m = W["model"]
    d = m.config.hidden_size
    mx.random.seed(0)
    t = CT.CorpusTile(m, 1, mx.zeros((1, d)), mx.zeros((d,)), no_memory=True, rank=4)
    for blk in t.lora.values():
        for lo in blk.values():
            lo.b.weight = mx.random.normal(lo.b.weight.shape) * 0.05
    t.freeze()
    return t


def toks(W, n=3, L=64):
    rng = random.Random(4)
    return mx.array([W["tok"].encode(prose(rng, 30))[:L] for _ in range(n)])


# ------------------------------------------------------------------ readout exactness
def test_readout_at_init_leaves_outputs_unchanged(world):
    m = world["model"]
    T = toks(world)
    ro = R.Readout(m.config.hidden_size, 16, seed=0)
    tap = R.Tap(m, "readout", ro)
    cases = [(CT.forward, None), (ST.forward, side_tile(world)), (CT.forward, lora_tile(world))]
    for fwd, tile in cases:
        native = _f32(fwd(m, T, tile)[0])
        assert np.array_equal(_f32(fwd(tap, T, tile)[0]), native)
    assert np.array_equal(_f32(CT.forward(tap, T, None)[0]), _f32(m(T)))
    ro.b.weight = mx.random.normal(ro.b.weight.shape) * 0.1
    assert not np.array_equal(_f32(CT.forward(tap, T, None)[0]), _f32(m(T)))
    # the norm tap is the head's input: the head of it is the model's logits
    hn, _ = CT.forward(R.Tap(m, "norm"), T, None)
    assert np.array_equal(_f32(R.lm(m, hn)), _f32(m(T)))


def test_cached_states_equal_the_full_forward(world):
    m, tok = world["model"], world["tok"]
    tile = side_tile(world)
    ex = R.answer_examples(tok, questions()[:4], ["question"])
    st = R.answer_states(m, ST.forward, tile, False, ex, tok.pad, batch=2)
    ro = R.Readout(m.config.hidden_size, 8, seed=1)
    ro.b.weight = mx.random.normal(ro.b.weight.shape) * 0.05
    tap = R.Tap(m, "readout", ro)
    for (p, a), (H, Y) in zip(ex, st):
        logits, _ = ST.forward(tap, mx.array([p + a]), tile)
        full = logits[0, len(p) - 1:len(p) + len(a) - 1]
        np.testing.assert_allclose(_f32(full), _f32(R.lm(m, ro(H))), rtol=2e-2, atol=2e-2)
        assert Y.tolist() == a and a[-1] == tok.t.token_to_id("<|im_end|>")
        assert a[:-1] == tok.encode(next(r["answer"] for r in questions() if E.prompt_ids(tok, r["question"]) == p))


def test_only_the_readout_trains(world):
    m, tok = world["model"], world["tok"]
    for tile, fwd in ((side_tile(world), ST.forward), (lora_tile(world), CT.forward)):
        before_m, before_t = _snap(m), _snap(tile)
        assert not tree_flatten(m.trainable_parameters()) and not tree_flatten(tile.trainable_parameters())
        ex = R.answer_examples(tok, questions()[:6], ["question"])
        st = R.answer_states(m, fwd, tile, False, ex, tok.pad, batch=3)
        ro0 = R.Readout(m.config.hidden_size, 16, seed=0)
        import mlx.nn as nn
        _, g = nn.value_and_grad(ro0, lambda r: R.readout_loss(m, r, st[0][0], st[0][1]))(ro0)
        assert sorted(k for k, _ in tree_flatten(g)) == ["a.weight", "b.weight"]
        ro, log = R.train_readout(m, st, args("--steps", "20", "--batch", "4"))
        assert log[-1]["loss"] < log[0]["loss"]
        assert float(mx.abs(ro.b.weight).sum().item()) > 0
        after_m, after_t = _snap(m), _snap(tile)
        assert all(np.array_equal(before_m[k], after_m[k]) for k in before_m)
        assert all(np.array_equal(before_t[k], after_t[k]) for k in before_t)


# ------------------------------------------------------------------ splits
def test_document_split_is_disjoint_and_seeded(world):
    rows = questions()
    s0 = R.doc_split(rows, world["train_ids"], world["held_ids"], 0)
    assert set(s0["d_fit"]).isdisjoint(s0["d_test"])
    assert set(s0["d_fit"]) | set(s0["d_test"]) == {f"d{i}" for i in range(6)}
    assert len(s0["d_fit"]) == 3 and s0["held"] == [f"d{i}" for i in range(6, 10)]
    assert R.doc_split(rows, world["train_ids"], world["held_ids"], 0) == s0
    assert any(R.doc_split(rows, world["train_ids"], world["held_ids"], s)["d_fit"] != s0["d_fit"]
               for s in range(1, 6))
    fit = R.rows_for(rows, s0["d_fit"])
    test = R.rows_for(rows, s0["d_test"])
    assert {r["doc_id"] for r in fit}.isdisjoint({r["doc_id"] for r in test})
    bad = [dict(rows[0], doc_split="held_out")]
    with pytest.raises(SystemExit):
        R.doc_split(bad, world["train_ids"], world["held_ids"], 0)


# ------------------------------------------------------------------ scoring
def test_scoring_wiring(world, monkeypatch):
    rows = questions()[:6]
    seen = []

    def fake_generate(model, tok, tile, prompts, zero_values=False, max_new=40, fwd=None):
        seen.append((len(prompts), max_new, fwd))
        return [f"{rows[i]['answer']} it is." if i % 2 == 0 else "no idea" for i in range(len(prompts))]
    monkeypatch.setattr(E, "generate", fake_generate)
    got = R.recall(world["model"], world["tok"], ST.forward, None, False, rows, "question", 7, batch=6)
    assert got == {r["qid"]: i % 2 == 0 for i, r in enumerate(rows)}
    assert seen == [(6, 7, ST.forward)]
    g = R.generate_rows(world["model"], world["tok"], ST.forward, None, False, rows, "question", 7, 6,
                        R.Strict(world["tok"], questions()))
    sm = R.summary(g, 7)
    assert sm["n"] == 6 and sm["lenient"] == 0.5 and sm["strict"] == 0.5 and sm["len_mean"] > 0


def test_strict_score():
    tok = FakeTok()
    rows = questions()
    st = R.Strict(tok, rows)
    r = rows[0]                                   # answer Zorvex00, alias zv00
    assert st("Zorvex00.", r) and st("It is zv00, I think.", r)
    assert not st("Zorvex01 or Zorvex00", r)      # another candidate first: a hedge
    assert st("Zorvex00, not Zorvex01", r)
    assert not st("I am not sure.\nZorvex00", r)  # not in the first line
    long = " ".join(["word"] * 3) + " Zorvex00"   # beyond the first 12 tokens (bytes on FakeTok)
    assert not st(long, r) and E.correct(long, r["answer"], r["aliases"])
    assert st.window("\n\n  Zorvex00 and more text") == "Zorvex00 and"


# ------------------------------------------------------------------ contrasts
def _items(n_docs, per_doc, p_by_cond, seed):
    rng = random.Random(seed)
    docs = {f"x{d}": [f"x{d}.{i}" for i in range(per_doc)] for d in range(n_docs)}
    return docs, {c: {it: int(rng.random() < p) for its in docs.values() for it in its} for c, p in p_by_cond.items()}


def test_did_bootstrap():
    conds = [("tile", "realigned"), ("tile", "native"), ("base", "realigned"), ("base", "native")]
    # component lifts on D_test only (+0.6); nothing on held; base lift 0 everywhere
    dt, vt = _items(40, 3, dict(zip(conds, (0.7, 0.1, 0.1, 0.1))), 1)
    dh, vh = _items(40, 3, dict(zip(conds, (0.1, 0.1, 0.1, 0.1))), 2)
    vh = {c: {k.replace("x", "h"): v for k, v in d.items()} for c, d in vh.items()}
    dh = {k.replace("x", "h"): [i.replace("x", "h") for i in v] for k, v in dh.items()}
    per = {c: vt[c] | vh[c] for c in conds}
    r = R.did_bootstrap(per, {"d_test": dt, "held": dh}, "tile", n_boot=500, seed=0)
    assert r["n"]["d_test"] == {"docs": 40, "items": 120}
    lo, hi = r["did"]["ci95"]
    assert lo > 0.3 and lo <= r["did"]["est"] <= hi and r["did"]["mde80"] > 0
    assert r == R.did_bootstrap(per, {"d_test": dt, "held": dh}, "tile", n_boot=500, seed=0)    # seeded
    # null: all conditions identical -> the CI covers 0
    per0 = {c: per[("base", "native")] for c in conds}
    r0 = R.did_bootstrap(per0, {"d_test": dt, "held": dh}, "tile", n_boot=500, seed=0)
    assert r0["did"]["est"] == 0 and r0["did"]["ci95"] == [0, 0]


def test_document_bootstrap_keeps_questions_together():
    conds = [("tile", "realigned"), ("tile", "native"), ("base", "realigned"), ("base", "native")]
    # every D_test document is all-correct or all-wrong under the component's readout: a document-level bootstrap
    # must show the extra spread of 3-item clusters (se larger than an item-level bootstrap would give)
    docs = {f"x{d}": [f"x{d}.{i}" for i in range(3)] for d in range(30)}
    real = {it: int(d % 2 == 0) for d, its in enumerate(docs.values()) for it in its}
    zero = {it: 0 for its in docs.values() for it in its}
    hd = {"h0": ["h0.0", "h0.1"], "h1": ["h1.0", "h1.1"]}
    hz = {it: 0 for its in hd.values() for it in its}
    per = {conds[0]: real | hz, conds[1]: zero | hz, conds[2]: zero | hz, conds[3]: zero | hz}
    r = R.did_bootstrap(per, {"d_test": docs, "held": hd}, "tile", n_boot=2000, seed=0)
    item_se = (0.5 * 0.5 / 90) ** 0.5
    assert r["delta_lift_d_test"]["se"] > 1.4 * item_se


def test_answer_overlap():
    rows = questions()
    q = {"d_fit": rows[:4], "d_test": rows[2:8] + [dict(rows[9], qid="q999")], "held": rows[12:16]}
    ov = R.answer_overlap(q)
    assert ov["d_test"]["n"] == 7 and ov["d_test"]["exact_norm"] == round(2 / 7, 4)
    assert ov["held"]["exact_norm"] == 0.0 and ov["held"]["fit_answer_scores"] == 0.0


def test_run_check_end_to_end(world):
    a = args()
    assert a.eval_fields == "question,rephrase" and a.train_fields == "question"
    comps = [("base", None, False), ("tile", side_tile(world), False)]
    chk = R.run_check(world, comps, ST.forward, questions(), a, say=lambda *x, **k: None)
    res, gens, sp = chk["results"], chk["gens"], chk["splits"]
    assert chk["n"]["d_test"] == {"docs": 3, "questions": 6, "items": 12}
    for arm in ("base", "tile"):
        r = res[arm]
        for f in ("question", "rephrase"):
            assert set(r["native"][f]) == {"d_fit", "d_test", "held"}
            assert r["native"][f]["d_test"]["n"] == 6 and r["native"][f]["held"]["n"] == 8
            assert set(r["realigned"][f]) == {"d_test", "held", "d_fit"}
            assert {"lenient", "strict", "len_mean", "len_median", "len_at_cap"} <= set(r["realigned"][f]["held"])
            assert {gens[arm]["native"][f]["d_test"][x]["output"] is not None for x in gens[arm]["native"][f]["d_test"]}
            assert r["native_vs_stored"][f] == {"stored": None}
        assert r["readout_train"]["n_examples"] == 6                  # question field only
        assert r["separability"]["trained"]["n_docs"] == 4 and "caveat" in r["separability"]
    ct = chk["contrasts"]
    for scorer in ("lenient", "strict"):
        for key in ("question+rephrase", "question", "rephrase"):
            c = ct["recall"]["tile"][scorer][key]
            assert {"delta_lift_d_test", "delta_lift_held", "did"} <= set(c)
    assert ct["recall"]["tile"]["lenient"]["question+rephrase"]["n"]["held"] == {"docs": 4, "items": 16}
    assert {"component_minus_base_trained", "trained_minus_held"} <= set(ct["separability"]["tile"]["ncm_acc"])
    assert set(ct["answer_overlap_with_d_fit"]) == {"d_test", "held"}
    assert set(chk["sep"]["trained"]) <= world["train_ids"] and set(chk["sep"]["held"]) <= world["held_ids"]
    assert set(sp["d_fit"]) | set(sp["d_test"]) == {f"d{i}" for i in range(6)}


def test_native_reproduces_stored_qa_eval(world, tmp_path):
    rows = questions()
    qdir = tmp_path / "eval"
    qdir.mkdir()
    (qdir / "recall.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = tmp_path / "run"
    run.mkdir()
    tile = side_tile(world)
    ev = E.qa_eval(world, tile, None, qdir, run, batch=8, fwd=ST.forward)
    # qa_eval writes score2 outputs since main d52f055d9; older runs carry the original format (both are read)
    assert ev["questions_sha12"] == R.questions_hash_v2(qdir)
    stored, fmt = R.stored_outputs_any(run, qdir)
    assert fmt == "score2" and stored and ("tile", "rephrase", "q001") in stored
    a = args("--max-new", str(E.MAX_NEW), "--gen-batch", "8", "--steps", "2", "--sep-docs", "0", "--no-fit-eval")
    chk = R.run_check(world, [("base", None, False), ("tile", tile, False)], ST.forward, rows, a, stored=stored,
                      say=lambda *x, **k: None)
    for arm in ("base", "tile"):
        for f in ("question", "rephrase"):
            c = chk["results"][arm]["native_vs_stored"][f]
            assert c["n_compared"] == 20 and c["agreement"] == 1.0 and c["differing_qids"] == []
            assert c["correct_equal"] == 20
    # a changed stored output is reported, by hashed qid
    k = ("base", "question", "q003")
    stored[k] = ("something else", stored[k][1])
    c = R.compare_stored(stored, "base", "question", chk["gens"]["base"]["native"]["question"]["d_fit"] |
                         chk["gens"]["base"]["native"]["question"]["d_test"] |
                         chk["gens"]["base"]["native"]["question"]["held"])
    assert c["differing_qids"] == [R.hid("q003")] and c["output_equal"] == 19


def test_ids_hash_and_defaults_merge():
    assert R.hid("r0001") == __import__("hashlib").sha256(b"r0001").hexdigest()[:12]
    assert R.hid("abcdef012345") == "abcdef012345" and len(R.hid(5)) == 12
    d = CT.train_defaults()
    assert d["no_memory"] is False and d["rank"] == 16 and "cmd" not in d and "out" not in d
    old = {"model": "x", "site": 8}                               # an older args.json missing later arguments
    merged = d | old
    assert merged["site"] == 8 and merged["no_anchors"] is False
    sd = R._side_defaults()
    assert sd["accum"] == 1 and sd["clip"] == 0.0 and sd["arm"] == "imitate"


# ------------------------------------------------------------------ separability
def test_separability_constructed_cases():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(8, 32)) * 5
    X = np.concatenate([c + rng.normal(size=(5, 32)) for c in centres])
    labels = [i for i in range(8) for _ in range(5)]
    s = R.separability(X, labels)
    assert s["ncm_acc"] >= 0.95 and s["knn1_acc"] >= 0.95 and s["silhouette"] > 0.3
    Xn = rng.normal(size=(200, 32))
    labels = [i for i in range(10) for _ in range(20)]
    s = R.separability(Xn, labels)
    assert s["chance"] == 0.1
    assert s["ncm_acc"] < 0.25 and s["knn1_acc"] < 0.25 and abs(s["silhouette"]) < 0.1


def test_separability_on_model_states(world):
    m, tok = world["model"], world["tok"]
    by = R.sep_chunks(tok, world["docs"], world["train_ids"], 6, 3, seed=0)
    assert len(by) == 6 and all(len(v) >= 2 for v in by.values())
    ch = [c for k in sorted(by) for c in by[k]]
    X = R.pooled_states(m, CT.forward, None, False, ch, batch=4)
    s = R.separability(X, [c[0] for c in ch])
    assert s["ncm_acc"] >= 0.9 and s["knn1_acc"] >= 0.9
    shuffled = [c[0] for c in ch]
    random.Random(0).shuffle(shuffled)
    assert R.separability(X, shuffled)["ncm_acc"] < 0.6


# ------------------------------------------------------------------ receipt
def test_receipt_has_no_corpus_text(world, tmp_path):
    a = args()
    comps = [("base", None, False), ("tile", side_tile(world), False)]
    rows = questions()
    chk = R.run_check(world, comps, ST.forward, rows, a, say=lambda *x, **k: None)
    rec = {"tile_checkpoint": "ens-h56-sidetile-tiny@abcdef123456", "base_checkpoint": "tiny-qwen3@000000000000",
           "corpus_hashes": {"corpus.jsonl": "synthetic"}}
    r = R.receipt(tmp_path / "run", "side_tile", rec, a, chk, "qhash")
    dump = json.dumps(r)
    assert r["schema"] == "readout_check.v1" and r["label"] == "private_corpus: never publish"
    assert r["run_checkpoint_fingerprint"] == "abcdef123456" and "source_revision" in r and "host" in r
    assert not Path(r["run_dir"]).is_absolute() and "caveat" in r["separability_caveat"] or r["separability_caveat"]
    for row in rows:
        for k in ("question", "rephrase", "answer", "evidence"):
            assert row[k] not in dump
        assert all(al not in dump for al in row["aliases"])
        assert json.dumps(row["qid"]) not in dump and R.hid(row["qid"]) in dump
    for d in world["docs"]:
        assert d["text"][:24] not in dump and d["title"] not in dump and json.dumps(d["id"]) not in dump
    assert all(x == R.hid(x) for v in r["splits"].values() for x in v)
    # the private outputs rows carry the generated text and the label
    out = list(R.output_rows(chk["gens"]))
    assert out and all(o["label"] == "private_corpus: never publish" and "output" in o for o in out)
    assert R.outputs_path(Path("/x/r.json")) == Path("/x/r.outputs.jsonl")


def test_refuses_to_overwrite(tmp_path):
    out = tmp_path / "r.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        R.main(["--run", str(tmp_path), "--kind", "side_tile", "--questions", str(tmp_path), "--out", str(out),
                "--no-memory-gate"])
    assert out.read_text() == "{}"
    out.unlink()
    R.outputs_path(out).write_text("")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        R.main(["--run", str(tmp_path), "--kind", "side_tile", "--questions", str(tmp_path), "--out", str(out),
                "--no-memory-gate"])
