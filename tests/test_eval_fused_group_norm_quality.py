import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/eval_fused_group_norm_quality.py"
spec = importlib.util.spec_from_file_location("eval_fused_group_norm_quality", SCRIPT)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class FakeTokenizer:
    eos_token_ids = (99,)

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def test_prompt_uses_distinct_documents_once_and_exact_width():
    tok = FakeTokenizer()
    corpus = [("first", list(range(100))), ("second", list(range(100, 200)))]
    prompt = quality.build_prompt(tok, 145, "first", "Q?", corpus)
    assert len(prompt["prompt"]) == 145
    assert prompt["sources"] == [
        {"document": "first", "tokens": 100},
        {"document": "second", "tokens": 40},
    ]
    assert prompt["prompt"][:140] == list(range(140))
    assert prompt["prompt"][140:] == tok.encode("\n\nQ?\n")


def test_prompt_rejects_insufficient_corpus_instead_of_repeating():
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="unique context"):
        quality.build_prompt(tok, 200, "first", "Q?", [("first", list(range(100)))])
    with pytest.raises(ValueError, match="missing question document"):
        quality.build_prompt(tok, 80, "absent", "Q?", [("first", list(range(100)))])


def test_scoring_windows_are_disjoint_and_unmodified():
    corpus = [("a", list(range(31))), ("b", list(range(100, 130)))]
    windows = quality.build_scoring_windows(corpus, 8, 2, 4)
    assert [(w["name"], w["start"]) for w in windows] == [
        ("a", 0), ("a", 10), ("a", 20), ("b", 0),
    ]
    assert windows[2]["prompt"] + windows[2]["continuation"] == list(range(20, 30))
    with pytest.raises(ValueError, match="only 6 disjoint"):
        quality.build_scoring_windows(corpus, 8, 2, 7)
    with pytest.raises(ValueError, match="must be positive"):
        quality.build_scoring_windows(corpus, 0, 2, 1)


def test_cli_rejects_invalid_options_before_gpu_or_model_load(monkeypatch, capsys):
    for option, value in (("--score-windows", "0"), ("--limit", "-1"),
                          ("--gen-tokens", "0")):
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", "unused.json",
                                            option, value, "--i-own-the-gpu"])
        with pytest.raises(SystemExit) as exc:
            quality.main()
        assert exc.value.code == 2
        assert "error:" in capsys.readouterr().err


def test_generation_status_requires_closed_thinking_and_stop():
    tok = FakeTokenizer()
    status = quality.generation_status(tok, [1, 99], "<think>reason</think>answer")
    assert status == {"stopped": True, "closed_thinking": True,
                      "final_answer": "answer"}
    assert quality.generation_status(tok, [1], "<think>reason")["final_answer"] == ""


def test_run_arm_resets_lever_on_error_and_rejects_null_engagement(monkeypatch):
    class FakeNorm:
        enabled = False

        def set_fused_group_norm_enabled(self, value):
            self.enabled = value

        def reset_fused_group_norm_stats(self):
            pass

        def fused_group_norm_stats(self):
            return {}

    norm = FakeNorm()
    monkeypatch.setattr(quality, "phase_a_generate", lambda *args: ([1], None))
    with pytest.raises(RuntimeError, match="unexpected fused kernel engagement"):
        quality.run_arm(None, None, {"prompt": [1]}, 1, True, norm)
    assert not norm.enabled

    norm.fused_group_norm_stats = lambda: {"calls": 1}
    got = quality.run_arm(None, None, {"prompt": [1]}, 1, True, norm)
    assert got["counters"] == {"calls": 1}
    assert not norm.enabled

    with pytest.raises(RuntimeError, match="unexpected fused kernel engagement"):
        quality.run_arm(None, None, {"prompt": [1]}, 1, False, norm)
    assert not norm.enabled

    def fail(*args):
        raise ValueError("model failure")

    monkeypatch.setattr(quality, "phase_a_generate", fail)
    with pytest.raises(ValueError, match="model failure"):
        quality.run_arm(None, None, {"prompt": [1]}, 1, True, norm)
    assert not norm.enabled
