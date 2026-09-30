"""BPE streaming edge cases using a synthetic vocabulary (CPU only)."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.tokenizer_utils import (
    BPEStreamingDetokenizer,
    SPMStreamingDetokenizer,
)
from mlx2.serving import ServingEngine
from test_suspend_replay import (  # noqa: F401 - shared fixture/helpers
    B_PROMPT,
    host,
    make_adapter,
    tiny_model,
)


def test_bpe_sparse_token_ids_are_addressed_by_id_and_holes_fail():
    stream = BPEStreamingDetokenizer(SimpleNamespace(vocab={"a": 0, "b": 4}))
    stream.add_token(4)
    assert stream.text == "b"
    with pytest.raises(ValueError, match="unknown BPE token ID"):
        stream.add_token(1)
    with pytest.raises(ValueError, match="unknown BPE token ID"):
        stream.add_token(-1)


def test_bpe_byte_zero_is_decoded_as_a_byte():
    stream = BPEStreamingDetokenizer(SimpleNamespace(vocab={"Ā": 0}))
    stream.add_token(0)
    stream.finalize()
    assert stream.text == "\x00"


def test_spm_padded_or_negative_id_is_a_value_error_not_an_index():
    stream = SPMStreamingDetokenizer(SimpleNamespace(vocab={"▁a": 0, "b": 1}))
    stream.add_token(0)
    for bad in (2, 305, -1):
        with pytest.raises(ValueError, match="unknown SPM token ID"):
            stream.add_token(bad)
    stream.finalize()
    assert stream.tokens == [0]
    assert stream.text == "a"


def test_spm_sparse_hole_is_a_value_error_before_any_state_changes():
    # Id 1 lies inside the id range but is absent from the vocabulary.  The
    # hole held "" and passed the range check, so the id was appended and
    # then ``bytes += str`` raised TypeError instead of the unknown-id error.
    stream = SPMStreamingDetokenizer(SimpleNamespace(vocab={"\u2581a": 0, "b": 2}))
    stream.add_token(0)
    with pytest.raises(ValueError, match="unknown SPM token ID: 1"):
        stream.add_token(1)
    stream.add_token(2)
    stream.finalize()
    assert stream.tokens == [0, 2]
    assert stream.text == "ab"


def test_undecodable_sampled_id_fails_its_request_not_the_worker(host):
    """Logits rows are wider than the tokenizer (248320 vs 248077 on the
    served Qwen checkpoints) and the unconstrained sampler does not mask the
    padding.  A padded id that reaches the detokenizer must fail only that
    request; before, it escaped the per-lane handlers, killed the generation
    worker, and every later request got 503."""
    model = tiny_model()  # 128-wide logits
    decodable = 100
    vocab = {chr(0x100 + i): i for i in range(decodable)}
    padded = 120

    class Tokenizer:
        vocab_size = decodable
        eos_token_ids = []

        @property
        def detokenizer(self):
            return BPEStreamingDetokenizer(SimpleNamespace(vocab=vocab))

    def processors(request, prompt_length):
        def force(tokens, logits):
            bias = mx.zeros((logits.shape[-1],))
            if request.get("force_padded"):
                bias[padded] = 1e4
            else:
                bias[decodable:] = -mx.inf
            return logits + bias

        return [force]

    adapter = make_adapter(model, processors=processors)
    adapter.tokenizer = Tokenizer()
    engine = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=False,
        max_lanes=2, prefill_step=8,
    )

    def outcome(job):
        while True:
            event = job.events.get(timeout=120)
            if "error" in event or "finish_reason" in event:
                return event

    try:
        assert engine.ready.wait(60), engine.error
        request = {"tokens": B_PROMPT, "max_tokens": 4, "temperature": 0}
        failed = outcome(engine.submit(dict(request, force_padded=True)))
        after = outcome(engine.submit(dict(request)))
    finally:
        engine.close()
    assert failed["status"] == 502
    assert f"unknown BPE token ID: {padded}" in failed["error"]
    assert after.get("finish_reason") == "length", after
    assert not engine.error


def _byte_level(data: bytes) -> str:
    """GPT-2 byte-level spelling of ``data``."""
    from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer

    BPEStreamingDetokenizer.make_byte_decoder()
    encoder = {v: k for k, v in BPEStreamingDetokenizer._byte_decoder.items()}
    return "".join(encoder[b] for b in data)


def test_bpe_stream_cut_mid_character_drops_it_instead_of_u_fffd():
    """vLLM #59133: a stream cut inside a multi-byte character (max_tokens,
    a stop token) never completes it; the served text drops those bytes.
    ``finalize`` still matches the full decode, which spells them U+FFFD."""
    euro = "€".encode()  # 3 bytes
    vocab = {_byte_level(b"hi"): 0, _byte_level(euro[:1]): 1,
             _byte_level(euro[1:]): 2, "Ġ": 3, _byte_level(b"\xff"): 4}
    tokenizer = SimpleNamespace(vocab=vocab)

    def run(ids, complete):
        stream = BPEStreamingDetokenizer(tokenizer)
        for token in ids:
            stream.add_token(token)
        stream.finalize_complete() if complete else stream.finalize()
        return stream.text

    assert run([0, 1], complete=False) == "hi�"
    assert run([0, 1], complete=True) == "hi"
    assert run([0, 1, 2], complete=True) == "hi€"
    # An invalid byte is not an incomplete character: it stays U+FFFD.
    assert run([0, 4], complete=True) == "hi�"
    assert run([0, 4, 1], complete=True) == "hi�"


def test_bpe_finalize_with_a_pending_added_token_character():
    """An added token spelled outside the byte map, pending behind an
    incomplete character, raised KeyError from ``finalize``."""
    euro = "€".encode()
    vocab = {_byte_level(euro[:1]): 0, "中": 1}  # not a byte-level character
    for complete in (False, True):
        stream = BPEStreamingDetokenizer(SimpleNamespace(vocab=vocab))
        stream.add_token(0)
        stream.add_token(1)
        stream.finalize_complete() if complete else stream.finalize()
        assert stream.text.endswith("中")


def test_spm_stream_cut_mid_character_drops_it_instead_of_u_fffd():
    vocab = {"▁hi": 0, "<0xE2>": 1, "<0x82>": 2, "<0xAC>": 3, "<0xFF>": 4}

    def run(ids, complete):
        stream = SPMStreamingDetokenizer(SimpleNamespace(vocab=vocab))
        for token in ids:
            stream.add_token(token)
        stream.finalize_complete() if complete else stream.finalize()
        return stream.text

    assert run([0, 1, 2], complete=False) == "hi�"
    assert run([0, 1, 2], complete=True) == "hi"
    assert run([0, 1, 2, 3], complete=True) == "hi€"
    assert run([0, 4], complete=True) == "hi�"


def test_served_length_stop_mid_character_sends_no_u_fffd(host):
    """The engine finishes the stream with ``finalize_complete``: a length
    stop right after a character's lead byte sends nothing for it."""
    model = tiny_model()
    decodable = 100
    lead = 50
    vocab = {chr(0x100 + i): i for i in range(decodable) if i != lead}
    vocab[_byte_level("€".encode()[:1])] = lead

    class Tokenizer:
        vocab_size = decodable
        eos_token_ids = []

        @property
        def detokenizer(self):
            return BPEStreamingDetokenizer(SimpleNamespace(vocab=vocab))

    def processors(request, prompt_length):
        def force(tokens, logits):
            bias = mx.full((logits.shape[-1],), -1e4)
            bias[lead] = 0.0
            return logits + bias

        return [force]

    adapter = make_adapter(model, processors=processors)
    adapter.tokenizer = Tokenizer()
    engine = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=False,
        max_lanes=1, prefill_step=8,
    )
    text = ""
    try:
        assert engine.ready.wait(60), engine.error
        job = engine.submit({"tokens": B_PROMPT, "max_tokens": 1, "temperature": 0})
        while True:
            event = job.events.get(timeout=120)
            assert "error" not in event, event
            for value in (event.get("delta") or {}).values():
                text += value if isinstance(value, str) else ""
            if "finish_reason" in event:
                break
    finally:
        engine.close()
    assert event["finish_reason"] == "length"
    assert "�" not in text
