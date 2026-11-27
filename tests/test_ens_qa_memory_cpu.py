"""H60 memory LM (research/ens/qa_memory.py) on CPU: tiny random Qwen3, byte tokenizer, synthetic documents and
questions only (no checkpoint, no private corpus, no GPU)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from research.ens import corpus_tile_eval as E
from research.ens import qa_memory as M
from research.ens import qwen3 as Q
from tests.test_ens_side_tile_cpu import FakeTok, tiny


@pytest.fixture(autouse=True)
def cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


NAMES = ["Zorvex Lattice", "Quillon Harbor", "Merrow Cipher", "Tavish Orbit", "Pellan Meadow", "Corvin Lantern"]


def make_corpus(root: Path, n=6, held=(4, 5)):
    docs = []
    for i in range(n):
        txt = (f"{NAMES[i]} was proposed by Dr Ada{i} in 20{10 + i}. The {NAMES[i]} method uses {i + 3} layers. "
               f"Dr Ada{i} later joined Lab{i}. ") * 6
        if i in (0, 1):
            txt += " Both build on Helix Prime and Helix Prime again. "
        docs.append({"id": f"d{i}", "title": f"Note on {NAMES[i]}", "batch": "course", "text": txt})
    split = {"train": [f"d{i}" for i in range(n) if i not in held], "held_out": [f"d{i}" for i in held]}
    c = root / "corpus"
    c.mkdir(parents=True)
    (c / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (c / "split.json").write_text(json.dumps(split))
    (c / "graph.json").write_text(json.dumps({"edges": []}))
    return c, docs, split


def make_questions(root: Path, split):
    rec = []
    for i in range(6):
        s = "train" if f"d{i}" in split["train"] else "held_out"
        rec.append({"qid": f"q{i}", "doc_id": f"d{i}", "doc_split": s,
                    "question": f"Who proposed the {NAMES[i]} method?", "rephrase": f"Name the author of {NAMES[i]}.",
                    "answer": f"Ada{i}", "aliases": []})
    xd = [{"qid": "x1", "type": "synthesis", "split_mix": "all_train", "question": "What do Zorvex and Quillon share?",
           "rephrase": "Shared basis of Zorvex and Quillon?", "answer": "Helix Prime", "aliases": []},
          {"qid": "x2", "type": "graph", "split_mix": "all_train", "question": "Which note follows Zorvex?",
           "rephrase": "After Zorvex?", "choices": ["Note 1", "Note 2"], "choice_answer": "Note 1"}]
    q = root / "eval"
    q.mkdir()
    (q / "recall.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rec))
    (q / "crossdoc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in xd))
    return q, rec, xd


def synth_args(out, *extra):
    return M.parser().parse_args(["synth", "--corpus", "unused", "--questions", "unused", "--out", str(out),
                                  "--window-tokens", "200", "--items-per-doc", "6", "--min-items", "6", "--max-per-call", "6", "--min-per-call", "6",
                                  "--gen-batch", "2", "--crossdoc-pairs", "2", "--no-memory-gate", *extra])


class FakeGen:
    """Answers each prompt with JSON lines built from the excerpt (good, leaked, pronoun and ungrounded items)."""

    def __init__(self, tok, stop_after=None):
        self.tok, self.calls, self.prompts, self.stop_after = tok, 0, [], stop_after

    def __call__(self, prompts, max_new):
        self.calls += 1
        if self.stop_after and self.calls >= self.stop_after:
            M._STOP["requested"] = True
        out = []
        for p in prompts:
            text = self.tok.t.decode(p)
            self.prompts.append(text)
            name = next((n for n in NAMES if n in text), "Helix Prime")
            i = NAMES.index(name) if name in NAMES else 0
            out.append("\n".join(json.dumps(x) for x in [
                {"type": "direct", "q": f"In what year was {name} proposed?", "a": f"20{10 + i}"},
                {"type": "direct", "q": f"In what year was {name} proposed?", "a": f"20{10 + i}"},  # duplicate
                {"type": "reverse", "q": f"Which method did Dr Ada{i} propose?", "a": name},
                {"type": "direct", "q": f"Who proposed the {name} method?", "a": f"Ada{i}"},      # = an eval question
                {"type": "direct", "q": f"Who was the proposer of the {name} method?", "a": f"Ada{i}"},  # same fact
                {"type": "direct", "q": "What did he propose in the excerpt?", "a": name},          # deictic
                {"type": "direct", "q": f"What colour is the {name} logo?", "a": "ultraviolet"},    # ungrounded
            ]))
        return out


def run_synth(tmp_path, gen=None, extra=()):
    c, docs, split = make_corpus(tmp_path) if not (tmp_path / "corpus").exists() else (tmp_path / "corpus", None, None)
    qd = tmp_path / "eval"
    if not qd.exists():
        make_questions(tmp_path, json.loads((c / "split.json").read_text()))
    by_id, tr, he, hashes = M.load_inputs(c)
    rec, xd = M.load_questions(qd)
    out = tmp_path / "synth"
    out.mkdir(exist_ok=True)
    a = synth_args(out, *extra)
    a.questions_sha12 = M.questions_sha12(qd)
    tok = FakeTok()
    gen = gen or FakeGen(tok)
    rc = M.run_synth(a, gen, tok, by_id, tr, he, M.eval_question_texts(rec, xd), hashes, out,
                     {"generator_checkpoint": "fake-gen@000000000000", "generator_backend": "fake"}, rec=rec)
    return rc, out, gen, by_id, tr, he


# ------------------------------------------------------------------ split guard
def test_split_guard_refuses_held_documents():
    with pytest.raises(AssertionError):
        M.assert_trained_only(["d1", "d5"], {"d1", "d2"}, {"d5"})
    with pytest.raises(AssertionError):
        M.assert_trained_only(["dX"], {"d1"}, set())
    M.assert_trained_only(["d1"], {"d1"}, {"d5"})


def test_overlapping_split_is_refused(tmp_path):
    c, _, _ = make_corpus(tmp_path)
    (c / "split.json").write_text(json.dumps({"train": ["d0", "d1"], "held_out": ["d1"]}))
    with pytest.raises(SystemExit):
        M.load_inputs(c)


def test_synthesis_never_sees_held_documents(tmp_path):
    rc, out, gen, by_id, tr, he = run_synth(tmp_path)
    assert rc == 0
    for p in gen.prompts:
        for d in he:
            assert by_id[d]["title"] not in p and NAMES[int(d[1:])] not in p
    rows = [json.loads(l) for l in open(out / "qa.jsonl")]
    assert rows and all(set(r.get("doc_ids") or [r["doc_id"]]) <= tr for r in rows)


# ------------------------------------------------------------------ filters, leakage, dedupe
def test_filters_leakage_and_dedupe(tmp_path):
    rc, out, *_ = run_synth(tmp_path)
    rec = json.loads((out / "synth_receipt.json").read_text())
    rows = [json.loads(l) for l in open(out / "qa.jsonl")]
    qs = [r["q"] for r in rows]
    assert not any("Who proposed the" in q for q in qs)               # eval-identical questions dropped
    assert rec["leakage"]["n_dropped_wording"] >= 4 and rec["leakage"]["n_dropped_fact"] >= 4
    assert any("In what year was" in q for q in qs)                     # other facts about the entity stay
    assert rec["dropped_by_filter"].get("deictic", 0) >= 4 and rec["dropped_by_filter"].get("ungrounded", 0) >= 4
    assert rec["dropped_duplicates"] >= 4
    assert len(qs) == len({(r["doc_id"], E._norm(r["q"])) for r in rows})
    assert {"direct", "reverse"} <= set(rec["by_type"])
    assert "crossdoc" in rec["by_type"] or rec["n_pairs"] >= 1


def test_leakage_filter_threshold():
    ev = [("Who proposed the Zorvex Lattice method?", "Ada0", [])]
    items = [{"doc_id": "d", "q": "Who proposed the Zorvex Lattice method?", "a": "x"},         # wording
             {"doc_id": "d", "q": "Who was the proposer of the Zorvex Lattice method?", "a": "Dr Ada0"},  # fact
             {"doc_id": "d", "q": "In what year was the Zorvex Lattice method proposed?", "a": "2010"},
             {"doc_id": "d", "q": "Which lab did Ada0 join after 2010?", "a": "Lab0"}]
    kept, rep = M.leakage_filter(items, ev, 0.6, 0.5)
    assert [k["q"] for k in kept] == [items[2]["q"], items[3]["q"]]
    assert rep["n_dropped_wording"] == 1 and rep["n_dropped_fact"] == 1 and rep["n_dropped"] == 2
    assert rep["max_wording_overlap_at_least"][">=0.9"] == 1
    kept, rep = M.leakage_filter(items, ev, 1.01, 1.01)
    assert len(kept) == 4 and rep["n_dropped"] == 0


def test_dedupe_exact_and_near():
    items = [{"doc_id": "a", "q": "When was Zorvex Lattice proposed?", "a": "2010"},
             {"doc_id": "a", "q": "When was  zorvex lattice proposed?", "a": "2010"},
             {"doc_id": "a", "q": "When was the Zorvex Lattice proposed?", "a": "2010"},
             {"doc_id": "b", "q": "When was Zorvex Lattice proposed?", "a": "2010"},
             {"doc_id": "a", "q": "When was Zorvex Lattice proposed?", "a": "2011"}]
    kept, n = M.dedupe(items)
    assert n == 3 and [k["doc_id"] for k in kept] == ["a", "b"]


def test_check_item_reasons():
    src = "Zorvex Lattice was proposed by Dr Ada0 in 2010."
    ok = {"q": "In what year was Zorvex Lattice proposed?", "a": "2010"}
    assert M.check_item(ok, "Note", src) is None
    assert M.check_item(ok | {"a": ""}, "Note", src) == "empty_answer"
    assert M.check_item(ok | {"a": " ".join(["w"] * 12)}, "Note", src) == "long_answer"
    assert M.check_item({"q": "What year did the author publish it?", "a": "2010"}, "Note", src) == "deictic"
    assert M.check_item({"q": "what year was this thing proposed?", "a": "2010"}, "Note", src) == "no_entity"
    assert M.check_item({"q": "Who proposed Zorvex Lattice?", "a": "Bob Smith"}, "Note", src) == "ungrounded"


def test_parse_items_tolerates_noise():
    txt = '```json\n{"type": "direct", "q": "A?", "a": "b"},\nnoise\n{"type": "weird", "q": "C?", "a": 3}\n```'
    assert M.parse_items(txt) == [{"type": "direct", "q": "A?", "a": "b"}, {"type": "direct", "q": "C?", "a": "3"}]
    assert M.parse_items('[{"q": "A?", "a": "b"}]') == [{"type": "direct", "q": "A?", "a": "b"}]


def test_budget_and_window_plan():
    assert M.item_budget(500) == 6 and M.item_budget(5000) == 20 and M.item_budget(10 ** 6) == 48
    p = M.plan_windows(26, 48)
    assert len(p) == 16 and sum(k for _, k in p) == 48 and p[0][0] == 0 and p[-1][0] == 25
    p = M.plan_windows(2, 20)
    assert [i for i, _ in p] == [0, 1] and sum(k for _, k in p) == 16      # capped at 8 per call
    assert M.plan_windows(0, 10) == []


def test_crossdoc_pairs_trained_only():
    docs = {f"d{i}": {"text": "Helix Prime appears. Helix Prime again." if i < 3 else "Helix Prime x Helix Prime"}
            for i in range(5)}
    pairs = M.crossdoc_pairs(docs, {"d0", "d1", "d2"}, 5)
    assert pairs and all({a, b} <= {"d0", "d1", "d2"} for a, b, _ in pairs)


# ------------------------------------------------------------------ resume, receipts
def test_synth_resume_does_not_redo_documents(tmp_path):
    gen = FakeGen(FakeTok(), stop_after=1)
    rc, out, *_ = run_synth(tmp_path, gen=gen)
    assert rc == 3 and not (out / "synth_receipt.json").exists()
    done1 = M._committed(out / "raw_docs.jsonl", "doc_id")
    gen2 = FakeGen(FakeTok())
    rc, out, *_ = run_synth(tmp_path, gen=gen2)
    assert rc == 0
    done2 = M._committed(out / "raw_docs.jsonl", "doc_id")
    assert set(done1) <= set(done2) and len(done2) == 4
    seen = [p for p in gen2.prompts if "Excerpt:\n" in p and "Excerpt 1" not in p]
    first = {NAMES[int(d[1:])] for d in done1}
    assert first and seen and not any(f'titled "Note on {n}"' in p for n in first for p in seen)


def test_synth_receipt_fields(tmp_path):
    rc, out, *_ = run_synth(tmp_path)
    r = json.loads((out / "synth_receipt.json").read_text())
    for k in ("label", "generator_checkpoint", "source_revision", "host", "mlx", "corpus_hashes", "questions_sha12",
              "qa_sha16", "leakage", "n_final"):
        assert k in r, k
    assert r["label"] == M.LABEL and "@" in r["generator_checkpoint"]
    cov = r["coverage"]
    assert cov["n_trained_eval_questions"] == 4 and cov["answer_in_windows"] == 4 and cov["answer_in_qa_answers"] == 0
    blob = json.dumps(r)
    assert "Zorvex" not in blob and "Ada0" not in blob                   # no corpus text in the receipt


# ------------------------------------------------------------------ chat format, answer-only loss
def test_chat_format_and_answer_mask():
    tok = FakeTok()
    p, t = M.encode_example(tok, "Who made Zorvex?", "Ada")
    assert tok.decode(p).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Who made Zorvex? Answer briefly." in tok.decode(p)
    assert t == tok.encode("Ada") + [tok.token_to_id("<|im_end|>")]
    T, Y, W = M.stack([(p, t), (p[:5], t[:2])], tok.pad)
    assert len(T[0]) % 16 == 0
    tgt = [Y[0][i] for i, w in enumerate(W[0]) if w]
    assert tgt == t and sum(W[1]) == 2 and [Y[1][i] for i, w in enumerate(W[1]) if w] == t[:2]


def test_answer_loss_matches_manual():
    m = tiny()
    tok = FakeTok()
    ex = [M.encode_example(tok, "Who made Zorvex?", "Ada"), M.encode_example(tok, "Q two?", "Bob Lee")]
    T, Y, W = M.stack(ex, tok.pad)
    loss, n = M.answer_loss(m, None, T, Y, W)
    lg = m(mx.array(T)).astype(mx.float32)
    ce = nn.losses.cross_entropy(lg, mx.array(Y), reduction="none")
    ref = float((mx.sum(ce * mx.array(W, dtype=mx.float32)) / sum(map(sum, W))).item())
    assert n == sum(map(sum, W)) and abs(float(loss.item()) - ref) < 2e-2


def test_untrained_adapter_is_the_base():
    m = tiny()
    ad = M.MemoryAdapter(m.config, rank=4)
    T = mx.array([FakeTok().encode("hello world, zorvex")])
    a, _ = M.mem_forward(m, T, ad)
    b, _ = M.mem_forward(m, T, None)
    assert np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))


# ------------------------------------------------------------------ generation
def _f32_tiny():
    m = tiny()
    m.set_dtype(mx.float32)
    return m


def test_cached_generate_matches_reference():
    m = _f32_tiny()
    tok = FakeTok()
    prompts = [E.prompt_ids(tok, q) for q in ("Who?", "A much longer question about Zorvex Lattice?", "Mid one?")]
    ref = E.generate(m, tok, None, prompts, max_new=12, fwd=M.mem_forward)
    got = M.cached_generate(m, tok, prompts, max_new=12, chunk=7)
    assert got == ref
    ad = M.MemoryAdapter(m.config, rank=4)
    for blk in ad.lora.values():
        for lo in blk.values():
            lo.b.weight = mx.random.normal(lo.b.weight.shape, key=mx.random.key(3)) * 0.2
    ref = E.generate(m, tok, ad, prompts, max_new=10, fwd=M.mem_forward)
    assert M.cached_generate(m, tok, prompts, max_new=10, lora=ad.lora) == ref


def test_load_quantized_roundtrip(tmp_path):
    m = _f32_tiny()
    m.set_dtype(mx.float16)
    nn.quantize(m, group_size=64, bits=4)
    d = tmp_path / "Tiny-4bit"
    d.mkdir()
    mx.save_safetensors(str(d / "model.safetensors"), dict(tree_flatten(m.parameters())))
    c = m.config
    cfg = {k: getattr(c, k) for k in Q.Qwen3Config.__dataclass_fields__} | \
        {"quantization": {"group_size": 64, "bits": 4}}
    (d / "config.json").write_text(json.dumps(cfg))
    m2, ck = M.load_quantized(d)
    assert ck.startswith("Tiny-4bit@") and len(ck.split("@")[1]) == 12
    T = mx.array([FakeTok().encode("zorvex lattice harbor")])
    assert np.allclose(np.array(m(T).astype(mx.float32)), np.array(m2(T).astype(mx.float32)), atol=1e-3)
    tok = FakeTok()
    prompts = [E.prompt_ids(tok, "Who?")]
    assert M.cached_generate(m2, tok, prompts, max_new=5) == M.cached_generate(m, tok, prompts, max_new=5)


# ------------------------------------------------------------------ BM25
def test_bm25_ranking():
    ps = [M.bm25_terms(t) for t in ("the cat sat on the mat", "zorvex lattice proposed in 2010",
                                    "lattice gauge theory", "dogs and cats")]
    bm = M.BM25(ps)
    assert bm.search(M.bm25_terms("When was Zorvex Lattice proposed?"), 2) == [1, 2]
    assert bm.search(M.bm25_terms("gauge"), 1) == [2]
    assert bm.search(M.bm25_terms("unrelated words"), 3) == []


def test_bm25_passages_trained_only():
    tok = FakeTok()
    docs = {"a": {"title": "A", "text": "x" * 100}, "b": {"title": "B", "text": "y" * 30}}
    ps = M.build_passages(tok, docs, {"a"}, 40)
    assert [p[0] for p in ps] == ["a"] * 3 and len(tok.encode(ps[0][1])) == 40


# ------------------------------------------------------------------ train resume
def _train_world(tmp_path):
    c, docs, split = make_corpus(tmp_path)
    qd, rec, xd = make_questions(tmp_path, split)
    rows = [{"doc_id": f"d{i}", "type": "direct", "q": f"In what year was {NAMES[i]} proposed?", "a": f"20{10 + i}"}
            for i in range(4) for _ in range(3)]
    return rows, rec, set(split["train"]), set(split["held_out"])


def train_args(out, *extra):
    return M.parser().parse_args(["train", "--synth", "s", "--corpus", "c", "--questions", "q", "--out", str(out),
                                  "--rank", "4", "--batch", "4", "--epochs", "2", "--track-every", "2",
                                  "--save-every", "0", "--val-frac", "0.2", "--no-memory-gate", *extra])


def test_train_resume_matches_uninterrupted(tmp_path):
    rows, rec, tr, he = _train_world(tmp_path)
    tok = FakeTok()
    info = {"synth_qa_sha16": "0" * 16, "generator_checkpoint": "fake@0", "corpus_hashes": {}, "questions_sha12": "x"}
    a = train_args(tmp_path / "full")
    assert M.run_train(a, tiny(), "tiny@000000000000", tok, rows, rec, tr, he, tmp_path / "full", info) == 0
    b = train_args(tmp_path / "res", "--max-seconds", "0")
    assert M.run_train(b, tiny(), "tiny@000000000000", tok, rows, rec, tr, he, tmp_path / "res", info) == 3
    assert (tmp_path / "res" / "resume").exists()
    b = train_args(tmp_path / "res")
    assert M.run_train(b, tiny(), "tiny@000000000000", tok, rows, rec, tr, he, tmp_path / "res", info) == 0
    w1 = mx.load(str(tmp_path / "full" / "adapter.safetensors"))
    w2 = mx.load(str(tmp_path / "res" / "adapter.safetensors"))
    assert all(np.array_equal(np.array(w1[k]), np.array(w2[k])) for k in w1)
    r = json.loads((tmp_path / "full" / "train_receipt.json").read_text())
    for k in ("label", "base_checkpoint", "memory_checkpoint", "memory_checkpoint_best", "source_revision", "host",
              "mlx", "log", "generator_checkpoint"):
        assert k in r, k
    assert r["log"][0]["step"] == 0 and "evalq_gain_trained_minus_held" in r["log"][-1]
    assert any(e.get("val_qa_nll") is not None for e in r["log"])


def test_train_refuses_held_rows(tmp_path):
    rows, rec, tr, he = _train_world(tmp_path)
    rows = rows + [{"doc_id": "d5", "type": "direct", "q": "Q?", "a": "a"}]
    with pytest.raises(AssertionError):
        M.run_train(train_args(tmp_path / "x"), tiny(), "t@0", FakeTok(), rows, rec, tr, he, tmp_path / "x", {})


# ------------------------------------------------------------------ eval + kill
def kitems(p_mem_tr, p_mem_h, p_base_tr, p_base_h, p_bm_tr, n_docs=30, per_doc=4, seed=0, drop_arm=None,
           held=True):
    rng = random.Random(seed)
    out = []
    for split, docs, ps in (("train", n_docs, (p_mem_tr, p_base_tr, p_bm_tr)),
                            ("held_out", n_docs if held else 0, (p_mem_h, p_base_h, 0.0))):
        for d in range(docs):
            for j in range(per_doc):
                arms = {arm: {"lenient": rng.random() < p, "strict": rng.random() < p}
                        for arm, p in zip(("memory", "base", "bm25_1x48"), ps)}
                arms.pop(drop_arm, None)
                out.append({"qid": f"{split}{d}-{j}", "field": "question", "doc": f"{split}{d}", "split": split,
                            "arms": arms})
    return out


def test_kill_measure_decisions():
    k = M.kill_measure(kitems(0.6, 0.05, 0.04, 0.03, 0.2), "bm25_1x48", n_boot=300)
    assert k["decision"] == "not_killed" and k["killed"] is False
    t = k["lenient"]["memory_minus_bm25_trained"]
    assert t["verdict"] == "pass" and t["ci95"][0] > 0 and t["mde80"] > 0
    k = M.kill_measure(kitems(0.1, 0.05, 0.04, 0.03, 0.6), "bm25_1x48", n_boot=300)
    assert k["decision"] == "killed" and k["killed"] is True
    assert k["lenient"]["memory_minus_bm25_trained"]["verdict"] == "kill"
    k = M.kill_measure(kitems(0.3, 0.7, 0.04, 0.03, 0.1), "bm25_1x48", n_boot=300)
    assert k["killed"] is True and k["lenient"]["did_vs_base"]["verdict"] == "kill"
    k = M.kill_measure(kitems(0.6, 0.6, 0.04, 0.03, 0.1), "bm25_1x48", n_boot=300)
    assert k["lenient"]["did_vs_base"]["verdict"] == "undecided"           # zero gap: CI cannot exclude a positive one
    k = M.kill_measure(kitems(0.4, 0.1, 0.05, 0.05, 0.3, n_docs=6, per_doc=3), "bm25_1x48", n_boot=300)
    assert k["decision"] == "undecided" and k["killed"] is None           # tiny sample: the CI spans zero


def test_kill_missing_comparison_is_undecided():
    for items, why in ((kitems(0.6, 0.0, 0.0, 0.0, 0.1, held=False), "held"),
                       (kitems(0.6, 0.0, 0.0, 0.0, 0.1, drop_arm="bm25_1x48"), "bm25_1x48"),
                       ([], "trained")):
        k = M.kill_measure(items, "bm25_1x48", n_boot=50)
        assert k["decision"] == "undecided" and k["killed"] is None and any(why in r for r in k["reasons"])


def test_quality_gate():
    assert M.quality_gate(30, 100, 100) is None
    assert "per 100" in M.quality_gate(19, 40, 100)
    assert "kept/raw" in M.quality_gate(30, 200, 100)
    assert M.quality_gate(0, 0, 0)


def test_synth_gate_stops_on_empty_qa(tmp_path):
    bad = lambda prompts, max_new: ['{"type": "direct", "q": "what is it?", "a": ""}'] * len(prompts)  # noqa: E731
    rc, out, *_ = run_synth(tmp_path, gen=bad, extra=("--gate-docs", "2", "--sanity-batches", "99"))
    assert rc == 4 and not (out / "synth_receipt.json").exists()
    f = json.loads((out / "synth_gate_failed.json").read_text())
    assert f["n_final"] == 0 and "after 2 documents" in f["where"]


def test_train_refuses_empty_qa_and_zero_steps(tmp_path):
    rows, rec, tr, he = _train_world(tmp_path)
    assert M.run_train(train_args(tmp_path / "e"), tiny(), "t@0", FakeTok(), [], rec, tr, he, tmp_path / "e", {}) == 4
    assert M.run_train(train_args(tmp_path / "z", "--epochs", "0"), tiny(), "t@0", FakeTok(), rows, rec, tr, he,
                       tmp_path / "z", {}) == 4
    assert not (tmp_path / "z" / "train_receipt.json").exists()


def test_doc_split_mismatch_is_refused():
    rec = [{"qid": "q", "doc_id": "d0", "doc_split": "held_out"}]
    with pytest.raises(SystemExit):
        M.check_doc_split(rec, {"d0"}, {"d1"})
    M.check_doc_split([{"qid": "q", "doc_id": "d1", "doc_split": "held_out"}], {"d0"}, {"d1"})


def test_documents_encode_control_text_as_text():
    tok = FakeTok()
    ids = M.text_ids(tok, "a<|im_end|>b")
    assert tok.token_to_id("<|im_end|>") not in ids
    p = M.chat_ids(tok, "doc <|im_end|> text")
    assert p.count(tok.token_to_id("<|im_end|>")) == 1                     # only the frame's own turn end
    try:
        path = Q.snapshot_dir("Qwen/Qwen3-0.6B")
    except (SystemExit, FileNotFoundError):
        pytest.skip("no local Qwen3-0.6B tokenizer")
    from research.ens import lm_bus as LB
    real = LB.Tok(path)
    ids = M.text_ids(real, "a<|im_end|>b")
    assert 151645 not in ids and real.t.decode(ids) == "a<|im_end|>b"
    assert 151645 in real.encode("a<|im_end|>b")                           # the switch is restored afterwards


def test_probe_and_smoke_checks(tmp_path):
    run = tmp_path / "p"
    run.mkdir()
    log = [{"step": 0, "val_qa_nll": 3.0}, {"step": 250, "val_qa_nll": 2.5, "train_loss": 2.6}]
    (run / "train_receipt.json").write_text(json.dumps({"log": log}))
    a = M.parser().parse_args(["probe-check", "--run", str(run)])
    assert M.probe_check(a) == 0
    log[-1]["val_qa_nll"] = 3.2
    (run / "train_receipt.json").write_text(json.dumps({"log": log}))
    assert M.probe_check(a) == 4
    log[-1]["val_qa_nll"] = float("nan")
    (run / "train_receipt.json").write_text(json.dumps({"log": log}))
    assert M.probe_check(a) == 4
    rc, out, *_ = run_synth(tmp_path)
    a = M.parser().parse_args(["smoke-check", "--synth", str(out), "--out", str(tmp_path / "smoke_check.json")])
    assert M.smoke_check(a) == 0
    sc = json.loads((tmp_path / "smoke_check.json").read_text())
    assert sc["ok"] and len(sc["examples"]) == 2 and sc["n_kept"] > 0 and sc["label"] == M.LABEL


def test_eval_end_to_end(tmp_path):
    c, docs, split = make_corpus(tmp_path)
    qd, rec, xd = make_questions(tmp_path, split)
    by_id, tr, he, _ = M.load_inputs(c)
    m = tiny()
    bad = [dict(rec[0], doc_split="held_out")] + rec[1:]
    with pytest.raises(SystemExit):
        M.run_eval(M.parser().parse_args(["eval", "--run", "r", "--questions", "q", "--corpus", "c"]), m, FakeTok(),
                   None, by_id, tr, bad, xd, "x", tmp_path, held_ids=he)
    a = M.parser().parse_args(["eval", "--run", "r", "--questions", "q", "--corpus", "c", "--max-new", "3",
                               "--batch", "4", "--bm25", "1x16,2x32", "--kill-bm25", "bm25_1x16",
                               "--no-memory-gate"])
    run = tmp_path / "run"
    run.mkdir()
    ad = M.MemoryAdapter(m.config, rank=4)
    res = M.run_eval(a, m, FakeTok(), ad, by_id, tr, rec, xd, M.questions_sha12(qd), run, held_ids=he,
                     mem_checkpoint="mem@abc")
    cl = res["cells"]
    assert set(cl) >= {"recall.train.question", "recall.held_out.rephrase", "synthesis.all_train.question",
                       "graph_mc.all_train.question"}
    r = cl["recall.train.question"]
    assert set(r) == {"base", "memory", "bm25_1x16", "bm25_2x32", "n"} and r["n"] == 4
    assert r["bm25_2x32"]["context_tok_mean"] > r["bm25_1x16"]["context_tok_mean"] > 0
    assert 0 <= r["bm25_1x16"]["retrieved_source_doc"] <= 1
    assert res["kill"]["bm25_arm"] == "bm25_1x16" and res["kill"]["decision"] in ("killed", "not_killed", "undecided")
    assert res["decoders"] == {"base": "reference", "memory": "cached", "bm25_1x16": "cached", "bm25_2x32": "cached"}
    assert res["kill_covered"]["decision"] == "undecided"                 # no coverage record passed
    # resume: a second call reuses every committed cell and keeps the outputs file consistent
    n_lines = len((run / res["outputs_file"]).read_text().splitlines())
    res2 = M.run_eval(a, m, FakeTok(), ad, by_id, tr, rec, xd, M.questions_sha12(qd), run,
                      gen=lambda *x: pytest.fail("a committed cell was recomputed"), mem_checkpoint="mem@abc")
    assert res2["cells"] == cl and len((run / res["outputs_file"]).read_text().splitlines()) == n_lines
    # another memory checkpoint or --which gets its own cells
    res3 = M.run_eval(a, m, FakeTok(), ad, by_id, tr, rec, xd, M.questions_sha12(qd), run, mem_checkpoint="mem@def")
    assert res3["cell_prefix"] != res["cell_prefix"]


def test_memory_endpoint_span():
    assert M.MemoryEndpoint.span(" Ada0 ÷÷÷") == "÷÷÷ Ada0 ÷÷÷"
    ep = M.MemoryEndpoint(None, FakeTok(), None, gen=lambda P: [str(len(p)) for p in P], batch=2)
    assert len(ep.ask(["a?", "b?", "c?"])) == 3
