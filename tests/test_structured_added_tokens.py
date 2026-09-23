"""A grammar must be able to admit tokens added after the base vocabulary.

``tokenizer.vocab_size`` is the base vocabulary only.  Every adapter's
tool-call wire marker is an *added* token sitting above it -- ``<tool_call>``
is id 248058 against a base size of 248044 on Flash-Next, ``<|START_ACTION|>``
is 255014 against 255000 on North, ``<|message|>`` is 200023 against 200000 on
Muse.  A piece table cut at the base size gives those ids an empty piece, so
no composed grammar can admit them: under a calls-only grammar the model has
to spell the marker out one ordinary token at a time, and under a text-or-calls
alternation the always-open free-text branch is the only one it can enter, so
it never calls a tool at all.

The id space also runs the other way: a model's ``lm_head`` is padded above
the tokenizer (248320 rows vs 248077 ids on Flash-Next), and those padding
rows decode to nothing.  Every mask must be exactly as wide as the logits row
it is applied to, with the padding rows inadmissible.
"""

import numpy as np
import pytest

import mlx.core as mx

from mlx2.structured_output import make_structured_processor, vocabulary_bound

# An always-open free-text branch beside a tool-call branch: the shape that
# turns an unreachable marker into "the model simply never calls a tool".
TEXT_OR_CALLS = r'(?:[^<]{0,32}|<tool_call>\{"n":"f"\}</tool_call>)'


def tiny_tokenizer():
    """A real fast tokenizer whose marker is an added token above the base."""
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    characters = sorted(set('<>/{}":,tolcan_f '))
    base = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({c: i for i, c in enumerate(characters)}, unk_token=None)
    )
    base.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    base.decoder = tokenizers.decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base)
    tokenizer.add_tokens(
        [
            tokenizers.AddedToken("<tool_call>", special=True, normalized=False),
            tokenizers.AddedToken("</tool_call>", special=True, normalized=False),
        ]
    )
    return tokenizer


def admissible(processor, width, *, greedy=False, rows=2):
    """The exact admissible set, taken from a mask the sampler would see."""
    row = np.full(width, -10.0, dtype=np.float32)
    logits = mx.array(np.repeat(row[None, :], rows, axis=0) if rows > 1 else row)
    out = np.array(processor(mx.array([], dtype=mx.int32), logits))
    assert processor.failure is None, processor.failure
    assert out.shape[-1] == width
    return out, set(np.flatnonzero(np.isfinite(out.reshape(rows, width)[0])).tolist())


def test_vocabulary_bound_spans_the_added_tokens():
    tokenizer = tiny_tokenizer()
    added = tokenizer.get_added_vocab()
    marker = added["<tool_call>"]
    # The defect's precondition, stated as the tokenizer reports it.
    assert marker >= tokenizer.vocab_size
    assert vocabulary_bound(tokenizer) == len(tokenizer) > marker


@pytest.mark.parametrize("engine", ["automaton", "scanner"])
def test_text_or_calls_grammar_admits_an_added_marker_token(engine, monkeypatch):
    monkeypatch.setenv("MLX2_STRUCTURED_AUTOMATON", "0" if engine == "scanner" else "1")
    tokenizer = tiny_tokenizer()
    marker = tokenizer.get_added_vocab()["<tool_call>"]
    processor = make_structured_processor(
        tokenizer, 0, grammar=TEXT_OR_CALLS, generation_stop_token_ids=[]
    )
    assert len(processor._pieces) == len(tokenizer)
    assert processor._pieces[marker] == "<tool_call>"
    _, allowed = admissible(processor, len(tokenizer))
    # Without the marker the free-text branch is the only one the model can
    # enter, which is the whole defect.
    assert marker in allowed


@pytest.mark.parametrize("engine", ["automaton", "scanner"])
def test_mask_is_as_wide_as_a_padded_logits_row(engine, monkeypatch):
    monkeypatch.setenv("MLX2_STRUCTURED_AUTOMATON", "0" if engine == "scanner" else "1")
    tokenizer = tiny_tokenizer()
    marker = tokenizer.get_added_vocab()["<tool_call>"]
    # Heads are padded above the tokenizer (Flash-Next: 248320 rows, 248077
    # ids).  The padding rows decode to nothing and must stay inadmissible.
    width = len(tokenizer) + 7
    processor = make_structured_processor(
        tokenizer, 0, grammar=TEXT_OR_CALLS, generation_stop_token_ids=[]
    )
    out, allowed = admissible(processor, width)
    assert out.shape[-1] == width
    assert marker in allowed
    assert max(allowed) < len(tokenizer)


def test_greedy_mask_lets_the_marker_the_model_wants_through(monkeypatch):
    monkeypatch.setenv("MLX2_STRUCTURED_AUTOMATON", "1")
    tokenizer = tiny_tokenizer()
    marker = tokenizer.get_added_vocab()["<tool_call>"]
    width = len(tokenizer) + 7
    row = np.linspace(-8.0, -1.0, width).astype(np.float32)
    row[marker] = 12.0  # the model is trained to open a call with one token
    processor = make_structured_processor(
        tokenizer, 0, grammar=TEXT_OR_CALLS, greedy=True, generation_stop_token_ids=[]
    )
    out = np.array(processor(mx.array([], dtype=mx.int32), mx.array(row)))
    assert processor.failure is None
    assert int(np.argmax(out)) == marker


def test_flash_next_tool_grammar_admits_its_own_tool_call_token():
    """The live case: the Qwen XML grammar against the real vocabulary."""
    from pathlib import Path

    import glob

    candidates = sorted(
        glob.glob(str(Path.home() / "mlx-models" / "Qwen3.8-Flash-Next-MLX-*"))
    )
    if not candidates:
        pytest.skip("Flash-Next tokenizer artifact is not present")
    from transformers import AutoTokenizer

    from mlx2.runtime.tokenizer_utils import TokenizerWrapper
    from mlx2.runtime.tool_parsers.qwen3_coder import constrained_tool_grammar

    hf = AutoTokenizer.from_pretrained(candidates[0], trust_remote_code=True)
    marker = hf.get_added_vocab().get("<tool_call>")
    if marker is None or marker < hf.vocab_size:
        pytest.skip("this artifact does not carry <tool_call> as an added token")
    tokenizer = TokenizerWrapper(hf, eos_token_ids=[hf.eos_token_id])
    assert vocabulary_bound(tokenizer) > marker >= hf.vocab_size
    grammar = constrained_tool_grammar(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "required",
    )
    processor = make_structured_processor(
        tokenizer, 0, grammar=grammar, greedy=True,
        generation_stop_token_ids=[hf.eos_token_id],
    )
    width = marker + 300  # the head is padded above the tokenizer
    row = np.linspace(-8.0, -1.0, width).astype(np.float32)
    row[marker] = 12.0
    out = np.array(processor(mx.array([], dtype=mx.int32), mx.array(row)))
    assert processor.failure is None
    assert out.shape[-1] == width
    assert int(np.argmax(out)) == marker


# ---------------------------------------------------------------------------
# Structural special tokens.  Masks never admit a special token by its text,
# because the grammar tracker skips special tokens.  Muse's recipient header
# `` to=<name><|message|>`` ends in a *special* token, though, and its tool
# grammars spell that header: with the special refused, the header could only
# be completed by spelling it out, which the adapter's recipient processor
# forbids, so every forced Muse tool call failed closed.  An adapter declares
# such ids; they then read as their literal and are admitted where it is.

HEADER_GRAMMAR = r'\ to=f<\|message\|>\{"s":"[^"]{0,12}"\}'


def header_tokenizer():
    """A real fast tokenizer where the header marker and ``<pad>`` are special."""
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    characters = sorted(set(' to=f<|>{}":sameg'))
    base = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({c: i for i, c in enumerate(characters)}, unk_token=None)
    )
    base.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    base.decoder = tokenizers.decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base)
    tokenizer.add_special_tokens(
        {
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["<|message|>"],
        }
    )
    return tokenizer


def admitted_after(processor, tokenizer, ids):
    row = mx.zeros((1, len(tokenizer)))
    out = np.array(processor(mx.array(ids, dtype=mx.int32), row))
    assert processor.failure is None, processor.failure
    return set(np.flatnonzero(np.isfinite(out[0])).tolist())


@pytest.mark.parametrize("engine", ["automaton", "scanner"])
def test_declared_structural_special_token_is_admitted_and_tracked(engine, monkeypatch):
    monkeypatch.setenv("MLX2_STRUCTURED_AUTOMATON", "0" if engine == "scanner" else "1")
    monkeypatch.setenv("MLX2_STRUCTURED_WORKERS", "0")
    tokenizer = header_tokenizer()
    message = tokenizer.convert_tokens_to_ids("<|message|>")
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    assert {message, pad, eos} <= set(tokenizer.all_special_ids)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    header = encode(" to=f")
    assert encode(" to=f<|message|>") == header + [message]

    def processor(**kwargs):
        made = make_structured_processor(
            tokenizer, 0, grammar=HEADER_GRAMMAR, generation_stop_token_ids=[eos], **kwargs
        )
        assert made.engine == engine
        return made

    # Undeclared, the protection holds: the header cannot be closed by the
    # special token (the defect when an upstream processor insists on it).
    plain = processor()
    assert message not in admitted_after(plain, tokenizer, header)

    declared = processor(structural_token_ids=[message])
    after_header = admitted_after(declared, tokenizer, header)
    assert message in after_header and pad not in after_header
    # The marker advanced the grammar by its literal: the header is not asked
    # for a second time (it was, spelled out, before b044155a).
    assert admitted_after(declared, tokenizer, header + [message]) == set(encode("{"))
    opened = header + [message] + encode('{"s":"')
    inside = admitted_after(declared, tokenizer, opened)
    assert pad not in inside  # every undeclared special stays excluded
    # Inside the string the marker counts its 11 characters against the
    # 12-character bound, so the stream and the grammar agree.
    assert message in inside
    assert admitted_after(declared, tokenizer, opened + [message]) >= set(encode("a"))
    assert admitted_after(
        declared, tokenizer, opened + [message] + encode("a")
    ) == set(encode('"'))
    done = header + [message] + encode('{"s":"sa"}')
    assert admitted_after(declared, tokenizer, done) == {eos}
    # Declaring an ordinary id changes nothing: it was never excluded.
    assert processor(structural_token_ids=[header[0]])._structural == {}
