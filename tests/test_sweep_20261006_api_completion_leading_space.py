"""``/v1/completions`` keeps the first generated token's leading space.

The streaming detokenizers inherited from mlx-lm trim a leading space from
the first text they emit.  For chat that hides the space a template leaves
before the answer; for a raw completion it corrupts the continuation:
``"The capital of France is"`` + ``" Paris"`` came back as ``"Paris"``
(vLLM #59046; OpenAI, vLLM and llama.cpp keep it).  Raw completions now turn
the trim off; chat keeps it.  Streaming and non-streaming responses read the
same detokenizer, so both are covered.
"""

import pytest

from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer, SPMStreamingDetokenizer

import route_harness as H


class _Vocab:
    def __init__(self, pieces):
        self.vocab = {piece: index for index, piece in enumerate(pieces)}


BPE = _Vocab(["ĠParis", "Ġis", "Ġnice", "!"])
SPM = _Vocab(["▁Paris", "▁is", "▁nice", "!"])


def _stream(detokenizer, ids):
    detokenizer.reset()
    segments = []
    for token in ids:
        detokenizer.add_token(token)
        segments.append(detokenizer.last_segment)
    detokenizer.finalize()
    segments.append(detokenizer.last_segment)
    return "".join(segments), detokenizer.text


@pytest.mark.parametrize("make", [
    lambda: BPEStreamingDetokenizer(BPE), lambda: SPMStreamingDetokenizer(SPM),
])
def test_trim_is_default_and_can_be_turned_off(make):
    detokenizer = make()
    assert _stream(detokenizer, [0, 1, 2, 3]) == ("Paris is nice!",) * 2
    detokenizer.trim_space = False
    assert _stream(detokenizer, [0, 1, 2, 3]) == (" Paris is nice!",) * 2
    detokenizer.reset()
    detokenizer.add_token(0)
    detokenizer.finalize_complete()
    assert detokenizer.text == " Paris"


def test_hf_qwen_tokenizer_continuation_keeps_space():
    transformers = pytest.importorskip("transformers")
    try:
        hf = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen3-0.6B tokenizer is not in the local HF cache")
    ids = hf.encode(" Paris is", add_special_tokens=False)
    detokenizer = BPEStreamingDetokenizer(hf, trim_space=False)
    assert _stream(detokenizer, ids)[1] == hf.decode(ids) == " Paris is"


class _TrimDetok(H.Detok):
    made = []

    def __init__(self):
        super().__init__()
        self.trim_space = True
        _TrimDetok.made.append(self)


class _Mixin:
    class tokenizer(type(H.make_adapter(None, 128).tokenizer)):
        @property
        def detokenizer(self):
            return _TrimDetok()

    tokenizer = tokenizer()


@pytest.fixture
def engine(monkeypatch):
    H.patch_host(monkeypatch)
    model, vocab = H.tiny_qwen38_mtp()
    engine = H.make_engine(model, vocab, mtp=False, adapter_mixin=_Mixin)
    yield engine
    engine.close()


def test_engine_turns_trim_off_for_raw_completions_only(engine):
    _TrimDetok.made.clear()
    raw = H.run(engine, {"tokens": [1, 2, 3], "prompt": "x", "max_tokens": 2,
                         "temperature": 0})
    assert "error" not in raw, raw
    chat = H.run(engine, {"tokens": [1, 2, 3], "max_tokens": 2, "temperature": 0,
                          "messages": [{"role": "user", "content": "x"}]})
    assert "error" not in chat, chat
    assert [made.trim_space for made in _TrimDetok.made] == [False, True]
