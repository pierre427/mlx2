"""BPE streaming edge cases using a synthetic vocabulary (CPU only)."""
from types import SimpleNamespace

import pytest

from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer


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
