"""Closed symbolic vocabulary for the ENS A0 tiny core.

A0 tasks are generated, so the core uses a small closed vocabulary instead of a
pretrained tokenizer: 64 value tokens, 16 query-key tokens, and a handful of
structural and operator tokens. Every answer is a single value token.
"""
from __future__ import annotations

N_VALUES = 64          # value tokens V0..V63; arithmetic is mod 64
N_KEYS = 16            # query keys K0..K15, mapped to values by a family's hidden law

_SPECIAL = [
    "PAD", "BOS", "Q", "GIVEN", "SLOT", "NEUTRAL", "ANS",
    "PERM", "ADD", "MUL", "XOR", "GR", "HOP", "FST", "IN", "B0", "B1", "END",
]

TOKENS: list[str] = list(_SPECIAL) + [f"V{i}" for i in range(N_VALUES)] + [f"K{i}" for i in range(N_KEYS)]
INDEX = {t: i for i, t in enumerate(TOKENS)}
VOCAB_SIZE = 128
assert len(TOKENS) <= VOCAB_SIZE

PAD = INDEX["PAD"]
BOS = INDEX["BOS"]
Q = INDEX["Q"]
GIVEN = INDEX["GIVEN"]
SLOT = INDEX["SLOT"]
NEUTRAL = INDEX["NEUTRAL"]
ANS = INDEX["ANS"]
V0 = INDEX["V0"]
K0 = INDEX["K0"]


def tok(name: str) -> int:
    return INDEX[name]


def val(v: int) -> int:
    if not 0 <= v < N_VALUES:
        raise ValueError(f"value out of range: {v}")
    return V0 + v


def key(k: int) -> int:
    if not 0 <= k < N_KEYS:
        raise ValueError(f"key out of range: {k}")
    return K0 + k


def value_of(token_id: int) -> int:
    v = token_id - V0
    if not 0 <= v < N_VALUES:
        raise ValueError(f"not a value token: {token_id}")
    return v


def decode(ids) -> str:
    return " ".join(TOKENS[i] if i < len(TOKENS) else f"<{i}>" for i in ids)
