"""CPU-only guards for the fused GroupRMSNorm long-context quality plan."""
from __future__ import annotations

from itertools import pairwise

import pytest

from scripts import eval_fused_group_norm_quality_v2 as gate


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_qa_prompts_have_distinct_exact_width_without_repeating_source(monkeypatch):
    monkeypatch.setattr(gate, "QA_DOCS", ("a.md", "b.md", "c.md"))
    sources = {"a.md": b"A" * 5000, "b.md": b"B" * 5000,
               "c.md": b"C" * 5000}
    cases = (gate.Case("a.md", "Ask A?", ("A",)),
             gate.Case("b.md", "Ask B?", ("B",)))
    prompts = gate.build_qa(CharacterTokenizer(), 9000, cases, sources)
    assert len({x["prompt_sha256"] for x in prompts}) == 2
    for prompt in prompts:
        assert len(prompt["prompt"]) == 9000
        assert prompt["source_tokens"] > 8192
        names = [x["path"] for x in prompt["source_files"]]
        assert len(names) == len(set(names))
        assert names[0] == prompt["source"]


def test_qa_rejects_repeated_or_too_short_context(monkeypatch):
    monkeypatch.setattr(gate, "QA_DOCS", ("a.md",))
    case = gate.Case("a.md", "Ask A?", ("A",))
    with pytest.raises(ValueError, match="only"):
        gate.build_qa(CharacterTokenizer(), 9000, (case,), {"a.md": b"A" * 10})
    with pytest.raises(ValueError, match="<=8192"):
        gate.build_qa(CharacterTokenizer(), 8200, (case,), {"a.md": b"A" * 9000})


def test_heldout_scored_spans_are_distinct_and_separated(monkeypatch):
    monkeypatch.setattr(gate, "PPL_DOCS", ("heldout.md",))
    sources = {"heldout.md": "".join(f"record {i:06d}\n" for i in range(4000)).encode()}
    windows = gate.build_ppl(CharacterTokenizer(), 9000, 64, 20, sources)
    assert len({x["continuation_sha256"] for x in windows}) == 20
    assert all(len(x["context"]) == 9000 and len(x["continuation"]) == 64
               for x in windows)
    assert all(b["corpus_offset"] - a["corpus_offset"] >= 64
               for a, b in pairwise(windows))


def test_stop_ids_include_wrapper_eos_and_ignore_unknown_tokens():
    class Wrapped:
        eos_token_ids = frozenset({10, 11})
        eos_token_id = 12
        unk_token_id = 0

        def convert_tokens_to_ids(self, spelling):
            return {"<|im_end|>": 13, "<|endoftext|>": 0}[spelling]

    assert gate.stop_ids(Wrapped()) == {10, 11, 12, 13}
