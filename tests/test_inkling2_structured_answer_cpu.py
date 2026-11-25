"""Inkling 2 replies open their own header; a client grammar waits for the text channel.

The template renders no reply header, so the model writes ``<|message_model|>``
and then ``<|content_thinking|>`` or ``<|content_text|>``. Bound from the first
generated token, a ``response_format`` excluded those special ids and forced a
headerless answer (omlx#5635; docs/research/2026-11-25-inkling2-pr-sweep.md).
"""

from pathlib import Path
from types import SimpleNamespace as NS

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.inkling2 import Inkling2Adapter
from mlx2.serving import structured_answer_token_ids

CONTENT_TEXT = 200004
CHAT = {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}}
CHECKPOINT = Path.home() / "mlx-models" / "Inkling-2-Mini"


def _adapter(hard_no_thinking=False, ids=(CONTENT_TEXT,), decoded="<|content_text|>"):
    def encode(text, **_kw):
        assert text == "<|content_text|>"
        return list(ids)

    adapter = object.__new__(Inkling2Adapter)
    adapter.tokenizer = NS(encode=encode, decode=lambda _ids, **_kw: decoded)
    adapter.hard_no_thinking = hard_no_thinking
    return adapter


@pytest.mark.parametrize("extra", [{}, {"reasoning_effort": "high"}, {"enable_thinking": False},
                                   {"reasoning_effort": "none"}])
def test_client_grammar_waits_for_the_text_channel(extra):
    assert structured_answer_token_ids(_adapter(), {**CHAT, **extra}) == (CONTENT_TEXT,)


def test_hard_no_thinking_prefill_binds_from_the_first_token():
    # The prompt already ends in <|content_text|>: the model never writes it.
    adapter = _adapter(hard_no_thinking=True)
    assert structured_answer_token_ids(adapter, {**CHAT, "enable_thinking": False}) is None
    assert structured_answer_token_ids(adapter, {**CHAT, "reasoning_effort": "none"}) is None
    # Thinking on: no text prefill, so the deferral stays.
    assert structured_answer_token_ids(adapter, {**CHAT, "reasoning_effort": "high"}) == (CONTENT_TEXT,)


def test_raw_completion_and_undeclared_marker_bind_from_the_first_token():
    assert structured_answer_token_ids(_adapter(), {"prompt": "x"}) is None
    assert structured_answer_token_ids(_adapter(ids=(1, 2)), CHAT) is None
    assert structured_answer_token_ids(_adapter(decoded="other"), CHAT) is None


def _real_tokenizer():
    if not (CHECKPOINT / "tokenizer.json").exists():
        pytest.skip("Inkling-2-Mini tokenizer not present")
    from transformers import AutoTokenizer

    from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer
    from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

    raw = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, trust_remote_code=False)
    repair_loaded_tokenizer(raw, CHECKPOINT)
    return TokenizerWrapper(raw, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[200006])


def test_real_tokenizer_marker_and_processor_deferral():
    from mlx2.structured_output import make_structured_processor

    tokenizer = _real_tokenizer()
    adapter = object.__new__(Inkling2Adapter)
    adapter.tokenizer = tokenizer
    adapter.hard_no_thinking = False
    marker = structured_answer_token_ids(adapter, CHAT)
    assert marker == (CONTENT_TEXT,)
    header = tokenizer.encode("<|message_model|>", add_special_tokens=False)
    processor = make_structured_processor(
        tokenizer, 2, response_format={"type": "json_object"}, defer_until=marker)
    width = len(tokenizer._tokenizer) if hasattr(tokenizer, "_tokenizer") else 200064
    logits = mx.zeros((1, max(width, CONTENT_TEXT + 64)))

    def admitted(generated):
        out = processor(mx.array([1, 1, *generated], dtype=mx.uint32), logits)
        return np.isfinite(np.array(out)[0])

    # Before the text channel opens the model may write its header freely.
    assert admitted([])[header[0]]
    assert admitted(header)[CONTENT_TEXT]
    assert processor.constraining is False
    # Once open, the grammar binds: the header can no longer be written.
    after = admitted([*header, CONTENT_TEXT])
    assert processor.constraining is True
    assert not after[header[0]] and not after[CONTENT_TEXT]
    brace = tokenizer.encode("{", add_special_tokens=False)
    assert after[brace[0]]


def test_real_tokenizer_answer_may_close_its_text_block():
    from mlx2.serving import structured_envelope_token_ids
    from mlx2.structured_output import make_structured_processor

    tokenizer = _real_tokenizer()
    adapter = object.__new__(Inkling2Adapter)
    adapter.tokenizer = tokenizer
    adapter.hard_no_thinking = False
    envelope = structured_envelope_token_ids(adapter)
    end_message = 200010
    assert envelope == ((), (end_message,))
    processor = make_structured_processor(
        tokenizer, 2, response_format={"type": "json_object"},
        defer_until=structured_answer_token_ids(adapter, CHAT), envelope=envelope)
    header = tokenizer.encode("<|message_model|>", add_special_tokens=False)
    answer = tokenizer.encode("{}", add_special_tokens=False)
    logits = mx.zeros((1, CONTENT_TEXT + 64))

    def admitted(generated):
        out = processor(mx.array([1, 1, *generated], dtype=mx.uint32), logits)
        return set(np.flatnonzero(np.isfinite(np.array(out)[0])).tolist())

    opened = [*header, CONTENT_TEXT]
    assert end_message not in admitted(opened)  # not before the answer is complete
    done = admitted([*opened, *answer])
    assert {end_message, 200006} <= done
    assert admitted([*opened, *answer, end_message]) == {200006}


def test_undeclared_envelope_marker():
    adapter = object.__new__(Inkling2Adapter)
    adapter.tokenizer = NS(encode=lambda _t, **_kw: [1, 2], decode=lambda _i, **_kw: "")
    assert adapter.structured_envelope_token_ids() is None
