"""CPU tests for the KV-index quality harness's workload, scoring and answer loop."""

import importlib.util
import random
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_kv_index_quality.py"
spec = importlib.util.spec_from_file_location("probe_kv_index_quality", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def encode(text):
    return [ord(c) % 128 for c in text]


def decode(ids):
    return "".join(chr(i) for i in ids)


FILLER = encode("lorem ipsum dolor sit amet " * 400)


def test_workload_is_exact_length_and_plants_every_fact():
    prompt, text, questions = probe.build_workload(
        encode, FILLER, 3000, 4, 2, 32, random.Random(3))
    assert len(prompt) == 3000 and len(text) == 32
    body = decode(prompt)
    singles = [q for q in questions if q["kind"] == "single"]
    hops = [q for q in questions if q["kind"] == "twohop"]
    assert len(singles) == 4 and len(hops) == 2
    for q in singles:
        assert f"The access code for {q['name']} is {q['code']}." in body
    for q in hops:
        assert f"{q['name']}'s partner is {q['partner']}." in body
        assert q["code"] == next(s["code"] for s in singles if s["name"] == q["partner"])


@pytest.mark.parametrize("answer,ok", [
    (" 123456.", True), (" 123 456.", True), (" 654321.", False),
    (" 12345.", False), (" 123456.\n999999", True), ("\n123456", False)])
def test_single_scoring(answer, ok):
    q = {"kind": "single", "code": "123456"}
    assert probe.score_answer(q, answer) is ok


@pytest.mark.parametrize("answer,ok", [
    (" Vega. The access code for Vega is 123456.", True),
    (" Lyra. The access code for Lyra is 123456.", False),
    (" Vega. The code is 654321.", False)])
def test_twohop_scoring(answer, ok):
    q = {"kind": "twohop", "partner": "Vega", "code": "123456"}
    assert probe.score_answer(q, answer) is ok


def test_answer_loop_runs_every_question_on_every_arm_kind():
    import mlx.core as mx

    model = probe.tiny_model()
    prompt, text, questions = probe.build_workload(
        encode, FILLER, 400, 3, 1, 16, random.Random(1))
    args = types.SimpleNamespace(window=8, prefill_step=128, warmup_steps=2, answer_tokens=6)
    saved = dict(probe.ARMS)
    probe.ARMS["idx8_b4096"] = {"index": {"bits": 8, "budget": 32}}
    try:
        dense, dense_rows = probe.run_arm(model, "dense", prompt, text, questions,
                                          encode=encode, decode=decode, eos=set(), args=args)
        for arm in ("kv_q8", "idx8_b4096"):
            record, rows = probe.run_arm(model, arm, prompt, text, questions,
                                         encode=encode, decode=decode, eos=set(), args=args)
            assert len(record["answers"]) == len(questions) == 4
            assert all(len(a["answer"]) <= 6 for a in record["answers"])
            assert rows.shape == dense_rows.shape == (16, 128)
            v = probe.compare(dense_rows, rows, text, dense, record)
            assert 0.0 <= v["answer_text_agreement"] <= 1.0 and v["kl_mean"] >= 0.0
        assert record["indexed_calls"] >= record["index_layers"] * (len(text) - 1)
    finally:
        probe.ARMS.clear()
        probe.ARMS.update(saved)
    mx.clear_cache()


def test_compare_against_itself_is_clean():
    import mlx.core as mx

    rows = mx.random.normal((8, 32))
    text = [1, 2, 3, 4, 5, 6, 7, 8]
    record = {"answers": [{"name": "Vega", "answer": " 1", "correct": True}]}
    v = probe.compare(rows, rows, text, record, record)
    assert v["kl_mean"] == pytest.approx(0.0, abs=1e-6)
    assert v["added_ppl_pct"] == pytest.approx(0.0, abs=1e-4)
    assert v["answer_losses"] == [] and v["answer_text_agreement"] == 1.0
