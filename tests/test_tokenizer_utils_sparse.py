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
