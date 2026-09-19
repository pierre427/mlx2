"""Byte-fallback pieces: characters the vocabulary only reaches through bytes."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2 import structured_automaton as sa
from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

QWEN_TOKENIZER = os.path.expanduser("~/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp")


class ByteLevelTokenizer:
    """256 single-byte tokens, a few text merges, one multi-byte fragment."""

    def __init__(self):
        inverse = {value: char for char, value in sa._byte_level_alphabet().items()}
        self.raw = [bytes([value]) for value in range(256)]
        self.raw += [b'{"', b'":', b'"}', b"ab", "é".encode(), b"\xf0\x9f", b"a\xf0"]
        self.names = ["".join(inverse[byte] for byte in raw) for raw in self.raw]
        self.names.append("<eos>")
        self.raw.append(None)
        self.eos = len(self.names) - 1
        self.vocab_size = len(self.names)
        self.eos_token_ids = [self.eos]
        self.all_special_ids = [self.eos]

    def get_added_vocab(self):
        return {"<eos>": self.eos}

    def convert_ids_to_tokens(self, ids):
        return [self.names[token] for token in ids]

    def decode(self, ids, skip_special_tokens=False, **_kw):
        data = b"".join(self.raw[token] or b"" for token in ids)
        return data.decode("utf-8", errors="replace")

    def encode_bytes(self, text):
        return list(text.encode("utf-8"))


def _allowed(processor, ids):
    return set(np.flatnonzero(processor._automaton_allowed(list(ids))).tolist())


def test_prefix_codepoint_ranges_respect_utf8_structure():
    assert sa._prefix_codepoints(b"\xf4") == ((0x100000, 0x10FFFF),)
    assert sa._prefix_codepoints(b"\xf4\x8f\xbf") == ((0x10FFC0, 0x10FFFF),)
    assert sa._prefix_codepoints(b"\xed") == ((0xD000, 0xD7FF),)  # no surrogates
    assert sa._prefix_codepoints(b"\xe0") == ((0x800, 0xFFF),)  # no overlongs
    assert sa._prefix_codepoints(b"\xc0") == ()  # only overlongs behind it
    assert sa._prefix_codepoints(b"\x80") == ()


def test_grammar_requiring_an_astral_character_is_reachable_byte_by_byte():
    tokenizer = ByteLevelTokenizer()
    target = "\U0010ffff"
    processor = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, "x" + target + "y"))
    assert processor.engine == "automaton" and processor._fragments is not None
    ids = tokenizer.encode_bytes("x" + target + "y")
    for step, token in enumerate(ids):
        allowed = _allowed(processor, ids[:step])
        assert allowed == {token}, (step, allowed)
    assert _allowed(processor, ids) == {tokenizer.eos}
    # EOS is never admissible inside a character, even at an accepting state.
    loose = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, "x?" + target + "?"))
    assert tokenizer.eos in _allowed(loose, ids[:1])
    assert tokenizer.eos not in _allowed(loose, ids[:2])
    assert processor.failure is None


def test_fragment_pieces_follow_the_grammar_not_just_utf8():
    tokenizer = ByteLevelTokenizer()
    lead = tokenizer.raw.index(b"\xf0\x9f")
    mixed = tokenizer.raw.index(b"a\xf0")
    emoji = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, "a?[\U0001f600-\U0001f64f]+"))
    start = _allowed(emoji, [])
    assert lead in start and mixed in start and 0xF0 in start
    assert 0xF4 not in start and 0xE2 not in start  # no admissible character behind them
    inside = _allowed(emoji, [lead])  # pending F0 9F -> only 0x98/0x99 continue the class
    assert inside == {0x98, 0x99}
    ascii_only = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, "[a-z]+"))
    assert not {lead, mixed, 0xF0} & _allowed(ascii_only, [])
    # A whole-character piece and its byte spelling are both admissible.
    accent = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, "é"))
    assert _allowed(accent, []) == {tokenizer.raw.index("é".encode()), 0xC3}


def test_json_strings_admit_any_character_through_fragments():
    tokenizer = ByteLevelTokenizer()
    processor = StructuredOutputProcessor(tokenizer, 0, compile_constraint({"type": "json_object"}, None))
    document = '{"k":"\U0001f9ea\U0010ffff é"}'
    ids = tokenizer.encode_bytes(document)
    for step, token in enumerate(ids):
        assert token in _allowed(processor, ids[:step]), (step, document.encode()[: step + 1])
    assert tokenizer.eos in _allowed(processor, ids)
    # Rollback into the middle of a character and back out again.
    middle = document.encode().index("\U0010ffff".encode()) + 2
    assert _allowed(processor, ids[:middle]) == set(range(0x80, 0xC0))  # any string character
    assert ids[middle - 3] in _allowed(processor, ids[: middle - 3])


def test_vocabularies_without_a_byte_view_keep_the_old_exclusion():
    pieces = ["", "a", "b", "�"]
    tokenizer = SimpleNamespace(
        vocab_size=4, eos_token_ids=[0],
        decode=lambda ids, **_kw: "".join(pieces[token] for token in ids),
    )
    processor = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, "[ab�]+"))
    assert processor._fragments is None
    assert _allowed(processor, []) == {1, 2}


@pytest.mark.skipif(not os.path.isdir(QWEN_TOKENIZER), reason="Qwen3.6 tokenizer artifact is not present")
def test_real_qwen_vocabulary_reaches_byte_fallback_characters(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(QWEN_TOKENIZER, local_files_only=True)
    tokenizer = SimpleNamespace(
        vocab_size=hf.vocab_size, eos_token_ids=[hf.eos_token_id], decode=hf.decode,
        convert_ids_to_tokens=hf.convert_ids_to_tokens, get_added_vocab=hf.get_added_vocab,
        all_special_ids=hf.all_special_ids,
    )
    # A schema ``const`` is ASCII-escaped by the compiler; the gap was free
    # strings (and raw grammars), where the character itself must be emitted.
    processor = StructuredOutputProcessor(tokenizer, 0, compile_constraint({"type": "json_object"}, None))
    assert processor.engine == "automaton" and processor._fragments is not None
    assert 900 < len(processor._fragments.items) < 1100
    document = '{"mark": "\U0010ffff", "note": "alchemy \U0001f732 ok"}'
    ids = hf.encode(document, add_special_tokens=False)
    assert hf.decode(ids) == document
    fragments = {token for token, _raw in processor._fragments.items}
    assert fragments & set(ids), "the document must actually need byte-fallback pieces"
    for step, token in enumerate(ids):
        assert processor._automaton_allowed(ids[:step])[token], (step, hf.convert_ids_to_tokens(token))
    assert processor._automaton_allowed(ids)[hf.eos_token_id]
    # The byte view and the tokenizer's own decode agree on whole characters.
    text, pending = sa.decode_token_bytes(processor._token_bytes, ids)
    assert (text, pending) == (document, b"")
    grammar = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, "=\U0010ffff="))
    ids = hf.encode("=\U0010ffff=", add_special_tokens=False)
    for step, token in enumerate(ids):
        allowed = np.flatnonzero(grammar._automaton_allowed(ids[:step]))
        assert token in allowed and len(allowed) <= 4, (step, len(allowed))
