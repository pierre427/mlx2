"""ENS evidence and evaluation regressions. Fixtures use no checkpoint weights."""
import hashlib
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from research.ens import common, corpus_tile, corpus_tile_eval, h45b_kill, kimi_core, lm_bus, physics_bus, physics_qa


@pytest.mark.parametrize(("output", "answer", "expected"), [
    ("It is 2.5.", "5", False),
    ("It is -5.", "5", False),
    ("It is 5.", "-5", False),
    ("It is −0.50.", "-.5", True),
    ("A 1,700-value result", "1700", True),
    ("From low-latency tests, 60 percent.", "60%", True),
])
def test_numeric_answer_tokens(output, answer, expected):
    assert corpus_tile_eval.correct(output, answer, []) is expected


def test_complete_text_windows(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "a" * 64}) + "\n")
    monkeypatch.setattr(lm_bus, "TEXT", corpus)
    tok = SimpleNamespace(encode=lambda s: list(range(len(s))))
    expected = [list(range(32)), list(range(32, 64))]
    assert lm_bus.text_windows(tok, "held") == expected
    assert sorted(corpus_tile.text_chunks(tok, "held", n_tok=32)) == expected


def test_receipted_weights_are_checked_before_load(tmp_path):
    path = tmp_path / "port.safetensors"
    path.write_bytes(b"initial")
    checkpoint = "fixture@" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    loaded = []
    module = SimpleNamespace(load_weights=loaded.append)
    common.load_checkpoint_weights(module, path, checkpoint)
    path.write_bytes(b"swapped")
    with pytest.raises(ValueError, match="differ from the training receipt"):
        common.load_checkpoint_weights(module, path, checkpoint)
    assert loaded == [str(path)]


def test_every_swapped_record_has_a_partner():
    rows = list(range(33))
    batches = list(physics_bus.eval_batches(rows))
    assert sum(batches, []) == rows
    assert all(2 <= len(b) <= 16 for b in batches)


def test_kimi_smoke_and_qwen_use_separate_chat_formats(monkeypatch):
    papers = [{"title": "Title", "field": "quantum", "month": "May", "concept": "entropy"}]
    seen = []
    def encode(tok, prompt, answer):
        seen.append(prompt)
        return [1], [2]
    monkeypatch.setattr(physics_qa, "encode", encode)
    monkeypatch.setattr(physics_qa, "mention_positions", lambda *args: [0])
    values = {k: [papers[0][k]] for k in physics_bus.SLOTS}
    vocab = SimpleNamespace(ids={k: [0] for k in values}, values=values)
    for tok in (kimi_core.KimiTok(SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda s: 5))),
                SimpleNamespace()):
        physics_bus.make_row(tok, vocab, "field", papers, random.Random(1), physics_qa.Q_TRAIN)
    assert seen[0].startswith("<|im_user|>")
    assert seen[1].startswith("<|im_start|>user")


def test_incomplete_seeds_cannot_yield_a_pooled_verdict(tmp_path, capsys):
    with pytest.raises(SystemExit, match="every requested seed"):
        h45b_kill.main([str(tmp_path), "0", "1"])
    out = json.loads(capsys.readouterr().out)
    assert set(out["missing_arms"]) == {"0", "1"}
    assert "verdict" not in out


def test_nll_weights_short_batch_and_handles_empty_split(monkeypatch):
    W = {"docs": [], "train_ids": set(), "held_ids": set(), "model": None, "tok": None}
    monkeypatch.setattr(corpus_tile, "text_chunks", lambda *a, **kw: [[0, 0]] * 16 + [[1, 1]])
    def forward(model, tokens, tile, zero_values):
        return mx.broadcast_to(mx.array([2., 0.]), (*tokens.shape, 2)), None
    monkeypatch.setattr(corpus_tile, "forward", forward)
    with mx.stream(mx.cpu):
        result = corpus_tile.nll_eval(W, None, SimpleNamespace(seed=0))
    import math
    assert result["ordinary_text"]["base"] == round(math.log1p(math.exp(-2)) + 2 / 17, 4)
    assert result["train_tuning"] == {"n_tokens": 0}
    assert result["recall_gain_tuning"] is None


def test_h51_completion_requires_every_arm(tmp_path, monkeypatch):
    (tmp_path / "ens-lab").mkdir()
    py = tmp_path / "mlx2-0324-venv/bin/python"
    py.parent.mkdir(parents=True)
    py.symlink_to("/usr/bin/false")
    private = tmp_path / "private"
    (private / "corpus").mkdir(parents=True)
    (private / "eval").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ENS_PRIVATE_CORPUS", str(private))
    out = tmp_path / "out"
    run = ["zsh", str(Path(corpus_tile.__file__).with_name("h51_run.sh")), str(out), "7"]
    assert subprocess.run(run).returncode != 0
    assert not (out / "DONE-seeds-7").exists()
    for arm in ("tile", "lora", "noanchor"):
        folder = out / f"{arm}-s7"
        folder.mkdir()
        (folder / "train_receipt.json").touch()
        (folder / "receipt.json").touch()
    subprocess.run(run, check=True)
    assert (out / "DONE-seeds-7").exists()


def test_qa_retry_discards_partial_cell_outputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    qdir = tmp_path / "questions"
    qdir.mkdir()
    row = {"doc_split": "train", "question": "First?", "rephrase": "Again?", "qid": "q1", "answer": "1"}
    (qdir / "recall.jsonl").write_text(json.dumps(row) + "\n")
    W = {"model": None, "tok": SimpleNamespace(encode=lambda s: [1])}
    monkeypatch.setattr(corpus_tile_eval, "generate", lambda *args, **kwargs: ["1"])
    result = corpus_tile_eval.qa_eval(W, None, None, qdir, tmp_path)
    output = next(tmp_path.glob("outputs-*.jsonl"))
    clean = output.read_bytes()
    with output.open("ab") as f:
        f.write(b'{"cell":"uncommitted","qid":"qX"}\n{"cell":"trunc')
    assert corpus_tile_eval.qa_eval(W, None, None, qdir, tmp_path) == result
    assert output.read_bytes() == clean


def test_corpus_anchors_restore_legacy_order_and_detect_input_changes(tmp_path, monkeypatch):
    corpus, run = tmp_path / "corpus", tmp_path / "run"
    corpus.mkdir()
    run.mkdir()
    docs = [{"id": "train", "title": "Train", "text": "training"}, {"id": "held", "text": "held"}]
    graph = {"edges": [{"src": "lecture:term|train", "dst": "train"}, {"src": "lecture:term|held", "dst": "held"}]}
    (corpus / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (corpus / "split.json").write_text(json.dumps({"train": ["train"], "held_out": ["held"]}))
    (corpus / "graph.json").write_text(json.dumps(graph))
    (corpus / "ref_df.json").write_text(json.dumps({"split_sha16": corpus_tile.sha16(corpus / "split.json")}))
    args = SimpleNamespace(model="fixture", corpus=str(corpus), site=0, n_free=0, n_concepts=0)
    monkeypatch.setattr(mx, "set_cache_limit", lambda _: None)
    monkeypatch.setattr(corpus_tile.Q, "snapshot_dir", lambda _: tmp_path)
    monkeypatch.setattr(corpus_tile.Q, "load", lambda *a, **kw: (SimpleNamespace(freeze=lambda: None), None, "tiny@cpu"))
    monkeypatch.setattr(corpus_tile.LB, "Tok", lambda _: None)
    monkeypatch.setattr(corpus_tile, "site_inputs", lambda *a: mx.ones((2, 2)))
    monkeypatch.setattr(corpus_tile, "anchor_states", lambda *a: mx.ones((len(a[2]), 2)))
    with mx.stream(mx.cpu):
        world = corpus_tile.build_world(args)
        assert ("topic", "term|held") not in world["anchor_index"]
        anchors = [("topic", "term|held", corpus_tile.topic_phrase("term|held")), *world["anchors"]]
        (run / "anchors.json").write_text(json.dumps({"anchors": anchors, "hashes": world["hashes"]}))
        mx.save_safetensors(str(run / "anchor_state.safetensors"), {"H": mx.ones((len(anchors), 2)), "mu": mx.zeros(2)})
        assert corpus_tile.build_world(args, run)["anchors"] == anchors
        (corpus / "corpus.jsonl").write_text((corpus / "corpus.jsonl").read_text().replace("training", "changed"))
        with pytest.raises(SystemExit, match="corpus files differ"):
            corpus_tile.build_world(args, run)
