"""Qwen budget release uses the generation template, not its history marker."""

from types import SimpleNamespace
from typing import ClassVar

from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.serving import thinking_close_token_ids, thinking_release_token_ids


class _Tokenizer:
    pieces: ClassVar = {
        "</think>": (41,),
        "\n</think>\n\n": (7, 41, 8, 8),
    }

    def encode(self, value, *, add_special_tokens):
        assert add_special_tokens is False
        return list(self.pieces[value])

    def decode(self, ids):
        values = tuple(int(token) for token in ids)
        return next(text for text, encoded in self.pieces.items() if encoded == values)


def _adapter(tokenizer=None):
    adapter = object.__new__(FlashNextAdapter)
    adapter.tokenizer = tokenizer or _Tokenizer()
    return adapter


def test_qwen_close_detection_and_forced_release_are_distinct():
    adapter = _adapter()
    assert thinking_close_token_ids(adapter) == (41,)
    assert thinking_release_token_ids(adapter) == (7, 41, 8, 8)


def test_explicit_release_precedes_a_declared_close_marker():
    adapter = SimpleNamespace(
        thinking_close_token_ids=lambda: (41,),
        thinking_release_token_ids=lambda: (7, 41, 8, 8),
    )
    assert thinking_release_token_ids(adapter) == (7, 41, 8, 8)


def test_adapters_without_a_distinct_release_keep_the_marker_fallback():
    adapter = SimpleNamespace(thinking_close_token_ids=lambda: (41,))
    assert thinking_release_token_ids(adapter) == (41,)


def test_qwen_release_fails_closed_on_tokenizer_template_drift():
    tokenizer = _Tokenizer()
    tokenizer.decode = lambda ids: "</think>"
    adapter = _adapter(tokenizer)
    assert adapter.thinking_release_token_ids() is None
    # Serving must not turn an explicitly declared but invalid release back
    # into a bare marker.  The adapter accessor exists, so budget release is
    # unavailable until the template identity is corrected.
    assert thinking_release_token_ids(adapter) is None
