"""CPU-only chat-template and final-answer guards for the v3 FGN gate."""
from __future__ import annotations

import pytest

from scripts import eval_fused_group_norm_quality_v2 as v2
from scripts import eval_fused_group_norm_quality_v3 as gate


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def get_added_vocab(self):
        return {"<|im_start|>": 1, "<|im_end|>": 2,
                "<think>": 3, "</think>": 4}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert add_generation_prompt and not tokenize
        return ("<|im_start|>user\n" + messages[0]["content"] +
                "<|im_end|>\n" + gate.ASSISTANT_PREFIX)


def test_chat_prompts_exact_width_and_assistant_prefix(monkeypatch):
    monkeypatch.setattr(v2, "QA_DOCS", ("a.md", "b.md", "c.md"))
    sources = {"a.md": b"A" * 5000, "b.md": b"B" * 5000,
               "c.md": b"C" * 5000}
    cases = (v2.Case("a.md", "Ask A?", ("A",)),
             v2.Case("b.md", "Ask B?", ("B",)))
    tok = CharacterTokenizer()
    prompts = gate.build_qa_chat(tok, 9000, cases, sources)
    assert len({c["prompt_sha256"] for c in prompts}) == 2
    for case in prompts:
        assert len(case["prompt"]) == 9000
        assert case["source_tokens"] > 8192
        text = "".join(map(chr, case["prompt"]))
        assert text.endswith(gate.ASSISTANT_PREFIX)
        assert text.count("<|im_start|>user") == 1
        assert text.rfind("Question: ") < text.rfind("<|im_start|>assistant")
        names = [item["path"] for item in case["source_files"]]
        assert len(names) == len(set(names))
        assert names[0] == case["source"]


def test_final_admission_requires_post_think_answer_and_stop():
    qa = {"stop_reason": "stop_token", "think_closed": True,
          "final_answer": "APCv2 owns serving state."}
    assert gate.admitted_final(qa)
    assert not gate.admitted_final({**qa, "stop_reason": "token_cap"})
    assert not gate.admitted_final({**qa, "think_closed": False})
    assert not gate.admitted_final({**qa, "final_answer": ""})
    assert not gate.admitted_final({**qa, "final_answer": "<|bad|> ...more content"})


def test_rejects_chat_template_without_assistant_prefix():
    class Broken(CharacterTokenizer):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False):
            return "no assistant prefix"

    with pytest.raises(ValueError, match="assistant think prefix"):
        gate.render_prompt(Broken(), "hello")


def test_rejects_user_content_with_tokenizer_control_token():
    tok = CharacterTokenizer()
    for marker in tok.get_added_vocab():
        with pytest.raises(ValueError, match="user content contains"):
            gate.render_prompt(tok, f"Source text mentions {marker}.")
