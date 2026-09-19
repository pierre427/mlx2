"""Masks must use the text a token adds in context, not its isolated decode.

A SentencePiece decoder strips the word-boundary space at the start of a
decode, so ``"▁▁"`` decodes to one space alone but adds two after other text.
Built from isolated decodes, a bounded-whitespace grammar admitted a token
that overran the bound and dead-ended the lane (Xing4.0 bf16, GPU 2026-09-18:
HTTP 502 "no valid token continuation").
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mlx2.structured_output import (
    StructuredOutputProcessor,
    compile_constraint,
    make_structured_processor,
    token_pieces,
)

TRACE = Path(__file__).parent / "fixtures" / "xing4_0_structured" / "whitespace_cap_trace.json"


class SentencePieceTokenizer:
    """``▁`` is a space, stripped once at the start of every decode."""

    def __init__(self):
        self.names = ["<eos>", "a", "▁", "▁▁", "{", "}", '"', "k", ":", "1"]
        self.eos = 0
        self.vocab_size = len(self.names)
        self.eos_token_ids = [self.eos]
        self.all_special_ids = [self.eos]

    def encode(self, text, add_special_tokens=False):
        return [self.names.index(text)] if text in self.names else []

    def convert_ids_to_tokens(self, ids):
        return [self.names[token] for token in ids]

    def get_added_vocab(self):
        return {"<eos>": self.eos}

    def decode(self, ids, skip_special_tokens=False, **_kw):
        text = "".join(
            "" if (token == self.eos and skip_special_tokens) else self.names[token]
            for token in ids
        ).replace("▁", " ")
        return text[1:] if text.startswith(" ") else text


def _allowed(processor, ids):
    return set(np.flatnonzero(processor._automaton_allowed(list(ids))).tolist())


def test_pieces_are_measured_in_context():
    tokenizer = SentencePieceTokenizer()
    pieces = token_pieces(tokenizer, tokenizer.vocab_size)
    assert pieces[3] == "  " and pieces[2] == " " and pieces[1] == "a"
    assert tokenizer.decode([3]) == " "  # the isolated decode under-reports


def test_bounded_whitespace_never_admits_an_overrunning_token():
    tokenizer = SentencePieceTokenizer()
    # "a", then at most three spaces, then "k".
    processor = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, "a {0,3}k"))
    ids = [1, 3]  # "a" + two spaces
    allowed = _allowed(processor, ids)
    assert 2 in allowed and 3 not in allowed  # one more space fits, two do not
    for token in allowed:
        assert _allowed(processor, ids + [token]), token


def test_xing_whitespace_cap_trace_has_no_dead_admissible_token():
    import glob

    candidates = sorted(glob.glob(str(Path.home() / "mlx-models" / "Xing4.0-29B-A4B-mlx-*")))
    if not candidates:
        pytest.skip("Xing4.0 tokenizer artifact is not present")
    import mlx.core as mx

    from mlx2.adapters.xing_tokenizer import load_tokenizer, make_tokenizer_wrapper

    tokenizer, _ = load_tokenizer(candidates[0])
    wrapper = make_tokenizer_wrapper(tokenizer)
    trace = json.loads(TRACE.read_text())
    ids = trace["generated_ids"]
    vocab = 131072

    def mask(sequence):
        processor = make_structured_processor(
            wrapper, 0, response_format=trace["response_format"], greedy=True,
            top_k=0, top_p=1.0, defer_until=tuple(trace["defer_until"]),
        )
        out = processor(mx.array(sequence), mx.zeros((1, vocab)))
        return processor.failure, np.flatnonzero(np.array(out[0] > -1e30))

    failure, allowed = mask(ids)
    assert failure is None and len(allowed)
    for token in allowed:
        failure, following = mask(ids + [int(token)])
        assert failure is None and len(following), tokenizer.convert_ids_to_tokens(int(token))
