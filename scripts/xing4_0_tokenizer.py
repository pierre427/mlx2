#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build a parity-checked fast ``tokenizer.json`` for Xing4.0 from ``tokenizer.model``.

Xing4.0 ships only a SentencePiece BPE model and a custom slow tokenizer
(``tokenization_xing4_0.Xing4_0Tokenizer``, ``trust_remote_code``).  mlx2 loads
tokenizers with ``trust_remote_code=False``, so this script builds a
``tokenizers`` fast tokenizer that reproduces the slow reference id-for-id and
writes it into an artifact directory -- but only after a parity check against
the live reference passes.

Construction (no transformers SPM converter; it is not faithful here):

* SentencePiece BPE semantics: merge the adjacent pair whose merged piece has
  the highest score.  Merges are derived from the model proto: every split
  ``(a, b)`` of a NORMAL piece where both halves are NORMAL pieces, ranked by
  the merged piece's score (scores are unique in this model).
* Per-chunk dummy prefix quirk: the slow tokenizer splits the text on its
  registered special tokens and runs ``sp.encode`` on every chunk, so every
  chunk gets its own leading ``U+2581``.  Reproduced with a ``Prepend`` +
  ``Replace(" ", U+2581)`` normalizer, which ``tokenizers`` applies per chunk
  between non-normalized added tokens.
* SentencePiece user-defined symbols (digits, ``_`` runs, ``\\n``, ``\\t``,
  ``<reserveN>``) are matched leftmost-longest and never merged.  Reproduced
  with an isolating ``Split`` pre-tokenizer plus ``BPE(ignore_merges=True)``
  so each isolated symbol maps straight to its id.  ``ignore_merges`` is only
  sound because every NORMAL piece that can stand alone as a pre-token is
  reachable by its own merges; the build asserts that.
* ``byte_fallback`` for characters outside the vocabulary.
* Only the 12 registered specials are added tokens.  ``<_observation>``,
  ``<_sep>``, ``<_controlN>`` exist in the SentencePiece model but are not
  registered, so the reference encodes them as literal text; copied as is.

Decoding in the reference strips one leading ``U+2581`` per run between special
tokens and uses SentencePiece's per-byte UTF-8 replacement.  ``tokenizer.json``
decoders cannot express that, so ``mlx2.adapters.xing_tokenizer`` installs an
exact decode on the loaded tokenizer; the parity check runs against that loader.

Usage (needs the reference: sentencepiece + transformers, e.g. the mlx-uag venv)::

    python scripts/xing4_0_tokenizer.py --artifact /path/to/Xing4.0-artifact
    python scripts/xing4_0_tokenizer.py --artifact DIR --write-fixture tests/fixtures/xing4_0_tokenizer

The artifact must contain ``tokenizer.model``, ``tokenizer_config.json`` and
``chat_template.jinja``; the reference ``tokenization_xing4_0.py`` is read from
the artifact or from ``--reference-code``.
Writes ``tokenizer.json``, rewrites ``tokenizer_config.json`` (fast class, no
``auto_map``; the original is kept as ``tokenizer_config.reference.json``),
and writes the parity stamp ``xing4_0_tokenizer_parity.json``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.adapters import xing_tokenizer as xt  # noqa: E402

# SentencePiece piece types (sentencepiece_model.proto).
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6


# --------------------------------------------------------------------------
# Minimal protobuf reader for sentencepiece ModelProto (no sentencepiece dep)
# --------------------------------------------------------------------------


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes):
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def read_spm_model(path: str | os.PathLike) -> dict:
    """Parse the subset of ``ModelProto`` the conversion depends on."""
    data = Path(path).read_bytes()
    pieces = []
    trainer: dict = {"user_defined_symbols": [], "control_symbols": []}
    normalizer: dict = {}
    for number, _, value in _fields(data):
        if number == 1:
            piece = {"piece": None, "score": 0.0, "type": NORMAL}
            for n, _, v in _fields(value):
                if n == 1:
                    piece["piece"] = v.decode("utf-8")
                elif n == 2:
                    piece["score"] = struct.unpack("<f", v)[0]
                elif n == 3:
                    piece["type"] = v
            pieces.append(piece)
        elif number == 2:
            names = {3: "model_type", 4: "vocab_size", 22: "split_by_whitespace",
                     24: "treat_whitespace_as_suffix", 25: "split_digits",
                     26: "allow_whitespace_only_pieces", 35: "byte_fallback"}
            for n, _, v in _fields(value):
                if n == 30:
                    trainer["control_symbols"].append(v.decode("utf-8"))
                elif n == 31:
                    trainer["user_defined_symbols"].append(v.decode("utf-8"))
                elif n in names:
                    trainer[names[n]] = v
        elif number == 3:
            for n, _, v in _fields(value):
                if n == 1:
                    normalizer["name"] = v.decode("utf-8")
                elif n == 2:
                    normalizer["precompiled_charsmap"] = bytes(v)
                elif n == 3:
                    normalizer["add_dummy_prefix"] = bool(v)
                elif n == 4:
                    normalizer["remove_extra_whitespaces"] = bool(v)
                elif n == 5:
                    normalizer["escape_whitespaces"] = bool(v)
    return {"pieces": pieces, "trainer": trainer, "normalizer": normalizer}


def validate_spm(model: dict) -> None:
    """Fail closed unless the model has exactly the semantics this build emulates."""
    t, n, pieces = model["trainer"], model["normalizer"], model["pieces"]
    checks = {
        "model_type == BPE": t.get("model_type") == 2,
        "byte_fallback": bool(t.get("byte_fallback")),
        "not treat_whitespace_as_suffix": not t.get("treat_whitespace_as_suffix", 0),
        "identity normalizer": n.get("name") == "identity"
        and not n.get("precompiled_charsmap"),
        "add_dummy_prefix": n.get("add_dummy_prefix", True),
        "not remove_extra_whitespaces": n.get("remove_extra_whitespaces", True) is False,
        "escape_whitespaces": n.get("escape_whitespaces", True),
        "vocab size": len(pieces) == xt.VOCAB_SIZE,
        "no UNUSED pieces": all(p["type"] != UNUSED for p in pieces),
        "unique pieces": len({p["piece"] for p in pieces}) == len(pieces),
        "unique normal scores": len({p["score"] for p in pieces if p["type"] == NORMAL})
        == sum(p["type"] == NORMAL for p in pieces),
        "unk id 0": pieces[0]["type"] == UNKNOWN,
        "256 byte pieces": [p["piece"] for p in pieces if p["type"] == BYTE]
        == [f"<0x{b:02X}>" for b in range(256)],
        "user-defined symbols": sorted(
            p["piece"] for p in pieces if p["type"] == USER_DEFINED
        ) == sorted(t["user_defined_symbols"]),
        "control ids": [i for i, p in enumerate(pieces) if p["type"] == CONTROL]
        == sorted(xt.SPECIAL_IDS | xt.CONTROL_IDS),
        "specials": all(pieces[i]["piece"] == s for s, i in xt.SPECIAL_TOKENS.items()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"tokenizer.model is not the expected Xing4.0 SPM BPE: {failed}")


def user_defined_pattern(symbols: list[str]) -> str:
    """Leftmost-longest isolating regex for SentencePiece user-defined symbols."""
    import re

    ordered = sorted(set(symbols), key=lambda s: (-len(s), s))
    return "|".join(re.escape(s).replace("\\\n", "\\n").replace("\\\t", "\\t") for s in ordered)


def build_tokenizer(model: dict):
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE

    pieces = model["pieces"]
    vocab = {p["piece"]: i for i, p in enumerate(pieces)}
    normal = {p["piece"]: p["score"] for p in pieces if p["type"] == NORMAL}
    merges = []
    for piece, score in normal.items():
        for cut in range(1, len(piece)):
            left, right = piece[:cut], piece[cut:]
            if left in normal and right in normal:
                merges.append((-score, cut, left, right))
    merges.sort()
    user = [p["piece"] for p in pieces if p["type"] == USER_DEFINED]
    tokenizer = Tokenizer(
        BPE(
            vocab=vocab,
            merges=[(left, right) for _, _, left, right in merges],
            byte_fallback=True,
            ignore_merges=True,
            fuse_unk=False,
        )
    )
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Prepend(xt.SPIECE), normalizers.Replace(" ", xt.SPIECE)]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(user_defined_pattern(user)), behavior="isolated"
    )
    # Best-effort decoder for consumers that bypass the mlx2 loader; the loader
    # replaces decode with the exact per-run SentencePiece decode.
    tokenizer.decoder = decoders.Sequence(
        [
            decoders.Replace(xt.SPIECE, " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", 1, 0),
        ]
    )
    tokenizer.add_special_tokens(
        [
            AddedToken(content, special=True, normalized=False, lstrip=False, rstrip=False)
            for content, _ in sorted(xt.SPECIAL_TOKENS.items(), key=lambda kv: kv[1])
        ]
    )
    for content, token_id in xt.SPECIAL_TOKENS.items():
        if tokenizer.token_to_id(content) != token_id:
            raise SystemExit(f"special token {content!r} did not keep id {token_id}")
    check_standalone_reachability(tokenizer, model)
    return tokenizer


def check_standalone_reachability(tokenizer, model: dict) -> None:
    """``ignore_merges`` returns a whole pre-token that is a vocab piece.

    SentencePiece BPE only produces such a piece if its own merges reach it, so
    assert that for every NORMAL piece that can form a whole pre-token.
    """
    from tokenizers import Tokenizer

    probe = Tokenizer.from_str(tokenizer.to_str())
    state = json.loads(probe.to_str())
    state["model"]["ignore_merges"] = False
    state["normalizer"] = None
    state["pre_tokenizer"] = None
    state["added_tokens"] = []
    probe = Tokenizer.from_str(json.dumps(state))
    user_chars = {c for p in model["pieces"] if p["type"] == USER_DEFINED for c in p["piece"]}
    candidates = [
        (i, p["piece"]) for i, p in enumerate(model["pieces"])
        if p["type"] == NORMAL and not (set(p["piece"]) & user_chars)
    ]
    texts = [piece for _, piece in candidates]
    unreachable = []
    for (token_id, piece), enc in zip(candidates, probe.encode_batch(texts, add_special_tokens=False)):
        if enc.ids != [token_id]:
            unreachable.append((token_id, piece, enc.tokens))
    if unreachable:
        raise SystemExit(
            f"{len(unreachable)} standalone pieces are not reachable by merges; "
            f"ignore_merges would diverge from SentencePiece: {unreachable[:10]}"
        )


# --------------------------------------------------------------------------
# Parity corpus
# --------------------------------------------------------------------------

MULTILINGUAL = [
    "The quick brown fox jumps over the lazy dog.",
    "你好，世界！今天天气很好，我们去公园散步吧。",
    "星辰大海，征途漫漫。模型推理需要高效的缓存。",
    "日本語のテキストとカタカナ、ひらがなを混ぜます。",
    "한국어 문장도 테스트합니다. 안녕하세요!",
    "مرحبا بالعالم، هذا اختبار للنص العربي من اليمين إلى اليسار.",
    "שלום עולם, זהו מבחן של טקסט בעברית.",
    "नमस्ते दुनिया, यह हिंदी पाठ का परीक्षण है।",
    "สวัสดีชาวโลก นี่คือการทดสอบภาษาไทย",
    "Привет, мир! Это тест кириллицы: ёжик, щука, ъ.",
    "Γειά σου Κόσμε — ελληνικά με τόνους: άέήίόύώ.",
    "Tiếng Việt có dấu: Xin chào thế giới, đường phố.",
    "Emoji: 😀😃😄 👩‍👩‍👧‍👦 🏳️‍🌈 🇫🇷🇯🇵 👍🏽 ❤️ ✨",
    "Combining: e\u0301 a\u0308 n\u0303 o\u0302\u0323 Z\u0351\u0316\u0317 ḁ",
    "Zero width: a\u200bb\u200cc\u200dd\ufeffe \u2060f",
    "Mixed 中文English混合text with 数字123和符号%^&*",
    "Math: ∑_{i=1}^{n} x_i² ≤ ∞, ∀ε>0 ∃δ, √2 ≈ 1.41421356, π·r²",
    "Currency: $1,234.56 €99,99 £7 ¥10000 ₹500 ₿0.001",
    "Full-width：ＡＢＣ１２３　全角スペース",
    "Ideographic space\u3000and nbsp\u00a0and thin\u2009space\u202fnarrow",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙 𝐛𝐨𝐥𝐝 and 𠀀𠀁𪚥 rare CJK Ext-B",
    "Tibetan བོད་ཡིག Georgian ქართული Armenian Հայերեն Amharic አማርኛ",
    "Private use \ue000\ue001\uf8ff and specials \ufff9\ufffc\ufffd",
    "Control chars:\x00\x01\x07\x08\x0b\x0c\x1b\x7f\x85 end",
]

CODE = [
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "for (let i = 0; i < 10; i++) {\n\tconsole.log(`i=${i}`);\n}\n",
    '{"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "units": null}}',
    "#include <stdio.h>\nint main(void) {\n    printf(\"%d\\n\", 42);\n    return 0;\n}\n",
    "SELECT user_id, COUNT(*) AS n FROM __events__ WHERE ts >= '2026-01-01' GROUP BY 1;",
    "class Foo:\n    __slots__ = ('_a', '__b')\n    def __init__(self):\n        self._a = 1\n",
    "fn main() { let v: Vec<u32> = (0..100).map(|x| x * 2).collect(); }",
    "    \t  mixed    indentation\n\t\t\ttabs\n        spaces\n",
    "<div class=\"a_b\">&lt;think&gt; not special</div><!-- <_user> -->",
    "path/to/file_name__v2.tar.gz  C:\\Users\\name\\AppData  ~/.config/xing_4-0",
    "x = [1,2,3]; y = {'k': (4, 5)}; z = f\"{x!r:>10}\"",
    "```python\nprint('hello')\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |",
]

WHITESPACE_UNITS = [" ", "  ", "\t", "\n", "\r\n", "\r", "\u3000", "\u00a0", "\u2581", "\v", "\f"]
SPECIAL_TEXTS = list(xt.SPECIAL_TOKENS)
UNREGISTERED = ["<_observation>", "<_sep>", "<_control1>", "<_control9>", "<_unk>", "<unk>",
                "<reserve1>", "<reserve10>", "<reserve100>", "<reserve101>", "<reserve0>",
                "<0x41>", "<param_key>", "</param_key>", "<param_value>", "</param_value>",
                "<think", "</think", "think>", "<_", "_>", "<tool_", "<tools>", "</tools>"]


def _random_unicode(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.30:
        return chr(rng.randint(0x20, 0x7E))
    if kind < 0.40:
        return rng.choice(WHITESPACE_UNITS)
    if kind < 0.55:
        return chr(rng.randint(0x4E00, 0x9FFF))
    if kind < 0.60:
        return chr(rng.randint(0x1F300, 0x1FAFF))
    if kind < 0.65:
        return rng.choice(SPECIAL_TEXTS + UNREGISTERED)
    if kind < 0.70:
        return rng.choice("0123456789_") * rng.randint(1, 6)
    if kind < 0.75:
        return chr(rng.randint(0x0300, 0x036F))
    if kind < 0.90:
        while True:
            cp = rng.randint(0x80, 0xFFFF)
            if not 0xD800 <= cp <= 0xDFFF:
                return chr(cp)
    return chr(rng.randint(0x10000, 0x10FFFF))


def fuzz_strings(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        out.append("".join(_random_unicode(rng) for _ in range(rng.randint(0, 40))))
    return out


def structured_strings() -> list[str]:
    out = ["", " ", "  ", "\n", "\n\n", "a", "_", "▁", "▁▁x", " x", "x ", "  hi  "]
    out += MULTILINGUAL + CODE
    for a in WHITESPACE_UNITS:
        for b in WHITESPACE_UNITS:
            out.append(f"{a}word{b}")
            out.append(f"x{a}{b}y")
            out.append(a * 3 + b * 2)
    for n in ["0", "7", "42", "3.14159", "-1e-9", "1,000,000", "2026-09-18", "0x1F600",
              "12345678901234567890", "v1.2.3", "1st 2nd 3rd", "１２３", "٣٤٥", "½ ⅓ ²³"]:
        out += [n, f"pay {n} now", f"{n}{n}", f"<_user>{n}<_bot>"]
    for k in range(1, 9):
        out += ["_" * k, "a" + "_" * k + "b", " " + "_" * k, "__init__" * k]
    for s in SPECIAL_TEXTS:
        out += [s, s + s, f"a{s}b", f" {s} ", f"{s}\n", f"\n{s}", f"{s} x", f"{s}{s}hi",
                f"x{s}", f"{s}123", f"{s}_", f"{s}\u2581y"]
    for u in UNREGISTERED:
        out += [u, f"a{u}b", f"<_user>{u}", f"{u}<_bot>", f" {u} "]
    for text in MULTILINGUAL + CODE:
        out += [f"<_user>{text}<_bot><think>\n", f"{text}</think>{text}", f"<tool_call>{text}</tool_call>"]
    # byte-fallback edges: chars unlikely in vocab and chars at UTF-8 length boundaries
    for cp in [0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF, 0xE000, 0xF8FF, 0xFFFE,
               0x1D11E, 0x2A6D6, 0xE0001, 0xE007F, 0xFE0F, 0x0600, 0x061C, 0x202E, 0x2066]:
        c = chr(cp)
        out += [c, f"a{c}b", c * 3, f" {c}", f"<_user>{c}"]
    long_mix = "".join(MULTILINGUAL) + "".join(CODE)
    out += [long_mix, long_mix * 3, "a" * 5000, " " * 3000, "\n" * 500, "字" * 2000,
            "ab " * 1500, "😀" * 300]
    return out


def _tools() -> list[list[dict]]:
    weather = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "days": {"type": "integer"},
                    "units": {"type": ["string", "null"], "enum": ["c", "f", None]},
                },
                "required": ["city"],
            },
        },
    }
    search = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web; returns JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {"type": "object"},
                    "top_k": {"type": "number"},
                    "safe": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }
    return [[], [weather], [weather, search]]


def conversations(seed: int, count: int) -> list[dict]:
    """Seeded chat cases: (messages, tools, template kwargs)."""
    rng = random.Random(seed)
    tools_sets = _tools()
    texts = MULTILINGUAL + CODE + ["Hi", "What's 2+2?", "  padded  ", "line1\nline2", "",
                                    "<think>inline</think>visible", "trailing\n\n"]
    cases = []
    base = [
        ([{"role": "user", "content": "Hello"}], [], {}),
        ([{"role": "system", "content": "You are Xing."}, {"role": "user", "content": "Hi"}], [], {}),
        ([{"role": "user", "content": "Weather in Paris?"},
          {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
           "tool_calls": [{"type": "function", "function": {"name": "get_weather",
                                                            "arguments": {"city": "Paris", "days": 2, "units": None}}}]},
          {"role": "tool", "content": "{\"temp\": 21}"},
          {"role": "tool", "content": "second result"},
          {"role": "user", "content": "Thanks"}], tools_sets[1], {}),
    ]
    for messages, tools, kwargs in base:
        for thinking in (None, True, False):
            extra = dict(kwargs)
            if thinking is not None:
                extra["enable_thinking"] = thinking
            cases.append({"messages": messages, "tools": tools, "kwargs": extra})
    for _ in range(count):
        messages = []
        if rng.random() < 0.5:
            messages.append({"role": "system", "content": rng.choice(texts)})
        for turn in range(rng.randint(1, 4)):
            content = rng.choice(texts)
            if rng.random() < 0.15:
                content = [{"type": "text", "text": rng.choice(texts)}, {"type": "text", "text": rng.choice(texts)}]
            messages.append({"role": "user", "content": content})
            if turn and rng.random() < 0.1:
                messages.append({"role": "system", "content": rng.choice(texts)})
            if rng.random() < 0.8:
                msg = {"role": "assistant", "content": rng.choice(texts)}
                roll = rng.random()
                if roll < 0.3:
                    msg["reasoning_content"] = rng.choice(texts)
                elif roll < 0.4:
                    msg["reasoning"] = rng.choice(texts)
                elif roll < 0.5:
                    msg["content"] = f"<think>\n{rng.choice(texts)}\n</think>\n\n{rng.choice(texts)}"
                if rng.random() < 0.35:
                    calls = []
                    for _ in range(rng.randint(1, 3)):
                        calls.append({"type": "function", "function": {
                            "name": rng.choice(["get_weather", "search"]),
                            "arguments": {
                                "city": rng.choice(texts)[:30],
                                "days": rng.randint(-5, 500),
                                "filters": {"lang": "zh", "n": [1, 2.5, None, True]},
                                "safe": rng.random() < 0.5,
                                "top_k": rng.random(),
                            }}})
                    msg["tool_calls"] = calls
                    messages.append(msg)
                    for _ in range(len(calls)):
                        if rng.random() < 0.2:
                            messages.append({"role": "tool", "content": [
                                {"output": rng.choice(texts)}, {"output": "x"}]})
                        else:
                            messages.append({"role": "tool", "content": rng.choice(texts)})
                    if rng.random() < 0.5:
                        messages.append({"role": "assistant", "content": rng.choice(texts)})
                else:
                    messages.append(msg)
        kwargs = {}
        thinking = rng.choice([None, True, False])
        if thinking is not None:
            kwargs["enable_thinking"] = thinking
        kwargs["add_generation_prompt"] = rng.random() < 0.8
        cases.append({"messages": messages, "tools": rng.choice(tools_sets), "kwargs": kwargs})
    return cases


def random_id_sequences(count: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    pools = [
        sorted(xt.SPECIAL_IDS), sorted(xt.CONTROL_IDS), [xt.UNK_ID],
        list(range(xt.BYTE_OFFSET, xt.BYTE_OFFSET + 256)),
        [xt.BYTE_OFFSET + b for b in (0x20, 0x41, 0xC3, 0xA9, 0xE4, 0xBD, 0xA0, 0xF0, 0x9F, 0x98, 0x80, 0xED, 0xA0, 0xC0, 0xF8, 0xFF)],
        list(range(24, 140)),
    ]
    out = []
    for _ in range(count):
        seq = []
        for _ in range(rng.randint(0, 24)):
            roll = rng.random()
            if roll < 0.12:
                seq.append(rng.choice(pools[0]))
            elif roll < 0.16:
                seq.append(rng.choice(pools[1]))
            elif roll < 0.18:
                seq.append(xt.UNK_ID)
            elif roll < 0.28:
                seq.append(rng.choice(pools[3]))
            elif roll < 0.40:
                seq.append(rng.choice(pools[4]))
            elif roll < 0.48:
                seq.append(rng.choice(pools[5]))
            elif roll < 0.55:
                seq.append(xt.SPIECE_ID)
            else:
                seq.append(rng.randrange(xt.VOCAB_SIZE))
        out.append(seq)
    return out


# --------------------------------------------------------------------------
# Parity check
# --------------------------------------------------------------------------


def load_reference(artifact: Path, reference_code: Path | None = None):
    """The slow reference Xing4_0Tokenizer (trust_remote_code), from the original config."""
    return xt.load_reference_tokenizer(artifact, reference_code)


def _record(failures: dict, key: str, sample) -> None:
    bucket = failures.setdefault(key, {"count": 0, "examples": []})
    bucket["count"] += 1
    if len(bucket["examples"]) < 5:
        bucket["examples"].append(sample)


def extra_corpus(paths) -> list[str]:
    """Paragraphs and whole files from real text files (optional corpus)."""
    out = []
    for path in paths or ():
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        out.append(text[:200000])
        out.extend(p for p in text.split("\n\n") if p.strip())
    return out


def run_parity(fast, reference, *, fuzz: int, seed: int, conversations_count: int,
               extra=(), log=print) -> dict:
    started = time.time()
    failures: dict = {}
    counts: dict = {}
    strings = structured_strings() + list(extra)
    counts["extra_strings"] = len(extra)
    fuzzed = fuzz_strings(fuzz, seed)
    counts["structured_strings"] = len(strings)
    counts["fuzz_strings"] = len(fuzzed)
    cases = conversations(seed + 1, conversations_count)
    rendered = []
    for case in cases:
        kwargs = dict(case["kwargs"])
        kwargs.setdefault("add_generation_prompt", True)
        ref_text = reference.apply_chat_template(
            case["messages"], tools=case["tools"] or None, tokenize=False, **kwargs
        )
        fast_text = fast.apply_chat_template(
            case["messages"], tools=case["tools"] or None, tokenize=False, **kwargs
        )
        if ref_text != fast_text:
            _record(failures, "chat_template_text", {"case": case})
        rendered.append(ref_text)
        if kwargs.get("add_generation_prompt") and "enable_thinking" in kwargs:
            ids = xt.render_prompt(
                fast, case["messages"], case["tools"] or None, kwargs["enable_thinking"]
            )
            ref_ids = reference.apply_chat_template(
                case["messages"], tools=case["tools"] or None, tokenize=True,
                return_dict=False, **kwargs,
            )
            if ids != list(ref_ids):
                _record(failures, "render_prompt_ids", {"case": case})
    counts["chat_renders"] = len(rendered)
    corpus = strings + rendered + fuzzed
    counts["encode_total"] = len(corpus)
    log(f"parity: {len(corpus)} strings ({len(fuzzed)} fuzz, {len(rendered)} chat renders)")
    fast_batch = fast(corpus, add_special_tokens=False)["input_ids"]
    detok_cls = xt.XingStreamingDetokenizer(fast)
    all_ids = []
    for index, (text, fast_ids) in enumerate(zip(corpus, fast_batch)):
        ref_ids = reference.encode(text, add_special_tokens=False)
        if fast_ids != ref_ids:
            _record(failures, "encode", {"text": text, "fast": fast_ids[:64], "reference": ref_ids[:64]})
        if index % 7 == 0 and fast.encode(text) != reference.encode(text):
            _record(failures, "encode_add_special_tokens", {"text": text})
        all_ids.append(ref_ids)
        if index and index % 5000 == 0:
            log(f"  encoded {index}")
    id_seqs = random_id_sequences(max(2000, fuzz // 4), seed + 2)
    counts["random_id_sequences"] = len(id_seqs)
    for ids in all_ids + id_seqs:
        for skip in (False, True):
            want = reference.decode(ids, skip_special_tokens=skip)
            got = fast.decode(ids, skip_special_tokens=skip)
            if want != got:
                _record(failures, f"decode_skip_{skip}", {"ids": ids[:64], "fast": got[:200], "reference": want[:200]})
        detok = detok_cls.copy()
        for token in ids:
            detok.add_token(token)
            detok.last_segment  # noqa: B018 - exercise incremental reads
        detok.finalize()
        if detok.text != reference.decode(ids):
            _record(failures, "streaming_decode", {"ids": ids[:64]})
    counts["decode_checks"] = 2 * (len(all_ids) + len(id_seqs))
    vocab_ids = list(range(xt.VOCAB_SIZE))
    if fast.convert_ids_to_tokens(vocab_ids) != reference.convert_ids_to_tokens(vocab_ids):
        _record(failures, "convert_ids_to_tokens", {})
    if len(fast) != len(reference):
        _record(failures, "len", {"fast": len(fast), "reference": len(reference)})
    return {
        "status": "pass" if not failures else "fail",
        "counts": counts,
        "failures": failures,
        "seconds": round(time.time() - started, 1),
        "seed": seed,
    }


def write_fixture(fast, reference, out_dir: Path, seed: int) -> None:
    """Compact expected ids/decodes generated by the reference, for CPU tests."""
    rng = random.Random(seed)
    strings = structured_strings()
    strings = [s for s in strings if len(s) <= 400]
    strings = rng.sample(strings, min(260, len(strings))) + fuzz_strings(240, seed + 7)
    cases = conversations(seed + 8, 10)
    chats = []
    for case in cases:
        thinking = case["kwargs"].get("enable_thinking", True)
        chats.append({
            "messages": case["messages"],
            "tools": case["tools"],
            "enable_thinking": thinking,
            "text": reference.apply_chat_template(
                case["messages"], tools=case["tools"] or None, tokenize=False,
                add_generation_prompt=True, enable_thinking=thinking),
            "ids": list(reference.apply_chat_template(
                case["messages"], tools=case["tools"] or None, tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=thinking)),
        })
    encodes = [{"text": s, "ids": reference.encode(s, add_special_tokens=False)} for s in strings]
    decodes = []
    for ids in random_id_sequences(120, seed + 9) + [e["ids"] for e in encodes[:80]]:
        decodes.append({
            "ids": ids,
            "text": reference.decode(ids),
            "text_skip_special": reference.decode(ids, skip_special_tokens=True),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mlx2.xing4_0.tokenizer-fixture/1",
        "generated_by": "scripts/xing4_0_tokenizer.py --write-fixture (slow Xing4_0Tokenizer reference)",
        "tokenizer_model_sha256": xt.sha256_file(Path(reference.vocab_file)),
        "versions": _versions(),
        "encode": encodes,
        "decode": decodes,
        "chat": chats,
    }
    # Every piece the decode cases touch, so decode/streaming tests run without
    # tokenizer.model (the runtime tables validate the fixed Xing4.0 layout).
    used = set(range(0, 400)) | {xt.SPIECE_ID, 396, 13029, 20814} | {i for d in decodes for i in d["ids"]}
    payload["pieces"] = {str(i): reference.convert_ids_to_tokens(i) for i in sorted(used)}
    (out_dir / "expected.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    # Assert the fast build agrees with what we just wrote.
    for item in encodes:
        assert fast.encode(item["text"], add_special_tokens=False) == item["ids"], item["text"]


def _versions() -> dict:
    import tokenizers
    import transformers

    out = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    try:
        import sentencepiece

        out["sentencepiece"] = sentencepiece.__version__
    except ImportError:
        pass
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--reference-code", type=Path, default=None,
                        help="tokenization_xing4_0.py of the checkpoint (default: in --artifact)")
    parser.add_argument("--fuzz", type=int, default=20000)
    parser.add_argument("--conversations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--write-fixture", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="also write the parity report here")
    parser.add_argument("--extra-corpus", type=Path, nargs="*", default=(),
                        help="text files whose paragraphs join the parity corpus")
    args = parser.parse_args(argv)
    artifact = args.artifact.resolve()
    reference_code = (args.reference_code or artifact / "tokenization_xing4_0.py").resolve()
    for required in [artifact / n for n in ("tokenizer.model", "tokenizer_config.json", "chat_template.jinja")] + [reference_code]:
        if not required.exists():
            raise SystemExit(f"missing {required}")
    if args.fuzz < 20000:
        raise SystemExit("--fuzz must be >= 20000 for a stamped build")

    model = read_spm_model(artifact / "tokenizer.model")
    validate_spm(model)
    tokenizer = build_tokenizer(model)
    reference = load_reference(artifact, reference_code)

    original_config = json.loads((artifact / "tokenizer_config.json").read_text())
    if original_config.get("tokenizer_class") == "Xing4_0Tokenizer":
        reference_config = original_config
    else:
        reference_config = json.loads((artifact / "tokenizer_config.reference.json").read_text())
    fast_config = xt.fast_tokenizer_config(reference_config)

    # Stage in a scratch dir, run parity against the loader exactly as mlx2 uses it.
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        tokenizer.save(str(stage / xt.TOKENIZER_JSON))
        (stage / "tokenizer_config.json").write_text(json.dumps(fast_config, indent=2, ensure_ascii=False) + "\n")
        for name in ("tokenizer.model", "chat_template.jinja"):
            shutil.copy2(artifact / name, stage / name)
        fast = xt.load_fast_unverified(stage)
        report = run_parity(fast, reference, fuzz=args.fuzz, seed=args.seed,
                            conversations_count=args.conversations,
                            extra=extra_corpus(args.extra_corpus))
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if report["status"] != "pass":
            print(json.dumps(report["failures"], ensure_ascii=False, indent=1)[:20000])
            raise SystemExit("parity FAILED; nothing written to the artifact")
        if args.write_fixture:
            write_fixture(fast, reference, args.write_fixture, args.seed)
        # Parity passed: publish into the artifact.
        if not (artifact / "tokenizer_config.reference.json").exists():
            shutil.copy2(artifact / "tokenizer_config.json", artifact / "tokenizer_config.reference.json")
        shutil.copy2(stage / xt.TOKENIZER_JSON, artifact / xt.TOKENIZER_JSON)
        shutil.copy2(stage / "tokenizer_config.json", artifact / "tokenizer_config.json")
    special_map = artifact / "special_tokens_map.json"
    if special_map.exists():
        try:
            json.loads(special_map.read_text())
        except ValueError:
            # The upstream repo serves "Entry not found" here; it breaks loaders.
            special_map.write_text("{}\n")
    stamp = xt.make_stamp(artifact, report=report, versions=_versions(),
                          generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          reference_code=reference_code)
    (artifact / xt.STAMP_NAME).write_text(json.dumps(stamp, indent=2, ensure_ascii=False) + "\n")
    xt.verify_parity_stamp(artifact)
    print(json.dumps({"status": "pass", "counts": report["counts"], "seconds": report["seconds"],
                      "stamp": str(artifact / xt.STAMP_NAME)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
