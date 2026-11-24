"""H44: English phrasings of A0 rows, with a template-level train / held split.

An English edge tile (englishtile.py) reads these phrasings as bytes, translates them into the frozen core's
published symbolic prompt, and renders the core's answer back to English. The phrasing carries everything the
symbolic prompt carries (the key, the visible operation); the hidden law stays with the domain tile.

Held-out wording: every class has train frames and held frames, and train and held phrase variants for each
part (operation verbs, edge statements, hop counts, bits). A held phrasing uses only held frames and phrases,
so it tests wording the tile never saw. Numbers are written as digits or words at random in every split.
"""
from __future__ import annotations

import random

from research.ens import families as F
from research.ens import vocab as V

MAX_BYTES = 320

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty"}


def number_word(n: int) -> str:
    if not 0 <= n < 70:
        raise ValueError(n)
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("" if o == 0 else "-" + _ONES[o])


WORD_TO_NUMBER = {number_word(n): n for n in range(64)}


def num(n: int, rng: random.Random) -> str:
    return number_word(n) if rng.random() < 0.5 else str(n)


# ------------------------------------------------------------------ phrase banks: (train variants, held variants)
OP_PHRASES = {
    "ADD": (["add {c}", "plus {c}"], ["increase it by {c}", "raise it by {c}"]),
    "MUL": (["multiply by {c}", "times {c}"], ["scale it by {c}", "take {c} times that"]),
    "XOR": (["xor with {c}", "bitwise xor {c}"], ["flip bits using {c}", "exclusive-or it with {c}"]),
}
EDGE_PHRASES = (["{i} points to {s}", "node {i} goes to {s}"], ["{i} leads to {s}", "from {i} you reach {s}"])
HOP_PHRASES = (["follow {h} hops", "take {h} steps"], ["walk {h} links", "move along {h} edges"])
STATE_PHRASES = (["state {i}: on 0 go to {a}, on 1 go to {b}", "from {i}, a 0 sends you to {a} and a 1 to {b}"],
                 ["in {i}, bit 0 moves to {a} while bit 1 moves to {b}", "{i} turns into {a} on 0 and {b} on 1"])
BITS_PHRASES = (["read the bits {bits}", "feed in {bits}"], ["process the input {bits}", "consume {bits}"])

FRAMES = {
    "PROGRAM": (["Look up key {x}, then {ops}. What is the result?",
                 "Start from the value for key {x}; {ops}. Give the answer."],
                ["Key {x} names a number: {ops}, and say what you get.",
                 "What comes out if you take key {x}'s value and {ops}?"]),
    "GRAPH": (["Graph: {edges}. Begin at the node for key {x} and {hops}. Where do you end?",
               "In this graph {edges}. From key {x}'s node, {hops}. Which node is reached?"],
              ["Here are the links: {edges}. Starting where key {x} says, {hops}; name the final node.",
               "Consider {edges}. Key {x} gives the start; {hops}. Where are you now?"]),
    "FST": (["Machine: {states}. Start in the state for key {x}, {bits}. Final state?",
             "Rules: {states}. Begin at key {x}'s state and {bits}. Which state results?"],
            ["The machine works like this: {states}. Key {x} picks the start; {bits}; report where it stops.",
             "Given {states}, start from key {x}'s state and {bits}. What is the last state?"]),
}

ANSWER_FRAMES = (["The answer is {n}.", "Result: {n}."], ["It comes to {n}.", "You get {n}."])


def _pick(bank, held, rng):
    return rng.choice(bank[1] if held else bank[0])


def _key(x: int, rng) -> str:
    return num(x, rng)


def phrase(fam: F.Family, row: F.Row, rng: random.Random, *, held: bool) -> str:
    """One English phrasing of a row's prompt (everything the bus-layout prompt carries)."""
    vals = [t - V.V0 for t in row.op]
    x = _key(row.x, rng)
    if fam.cls == "PROGRAM":
        parts = []
        for i in range(0, len(row.op), 2):
            name, c = V.TOKENS[row.op[i]], vals[i + 1]
            parts.append(_pick(OP_PHRASES[name], held, rng).format(c=num(c, rng)))
        text = _pick(FRAMES["PROGRAM"], held, rng).format(x=x, ops=", then ".join(parts))
    elif fam.cls == "GRAPH":
        m = fam.knobs[0]
        succ, hops = vals[1:1 + m], vals[2 + m]
        edges = "; ".join(_pick(EDGE_PHRASES, held, rng).format(i=num(i, rng), s=num(s, rng))
                          for i, s in enumerate(succ))
        hp = _pick(HOP_PHRASES, held, rng).format(h=num(hops, rng))
        text = _pick(FRAMES["GRAPH"], held, rng).format(x=x, edges=edges, hops=hp)
    elif fam.cls == "FST":
        s = fam.knobs[0]
        table = [(vals[1 + 2 * i], vals[2 + 2 * i]) for i in range(s)]
        bits = " ".join("1" if t == V.tok("B1") else "0" for t in row.op[2 + 2 * s:len(row.op) - 1])
        states = "; ".join(_pick(STATE_PHRASES, held, rng).format(i=num(i, rng), a=num(a, rng), b=num(b, rng))
                           for i, (a, b) in enumerate(table))
        bp = _pick(BITS_PHRASES, held, rng).format(bits=bits)
        text = _pick(FRAMES["FST"], held, rng).format(x=x, states=states, bits=bp)
    else:
        raise ValueError(f"{fam.cls}: no English phrasing (PERM is not an H44 family)")
    if len(text.encode()) > MAX_BYTES:
        raise ValueError(f"phrasing longer than {MAX_BYTES} bytes: {len(text.encode())}")
    return text


def answer_text(n: int, rng: random.Random, *, held: bool = False) -> str:
    return _pick(ANSWER_FRAMES, held, rng).format(n=num(n, rng))


def read_number(text: str):
    """The first number in a rendered answer, digits or words; None if there is none."""
    for tok in text.replace(".", " ").replace(",", " ").replace(":", " ").split():
        t = tok.strip().lower()
        if t.isdigit():
            return int(t)
        if t in WORD_TO_NUMBER:
            return WORD_TO_NUMBER[t]
    return None


def to_bytes(text: str, length: int = MAX_BYTES) -> list[int]:
    """UTF-8 bytes shifted by one (0 is padding), padded to `length`."""
    b = [c + 1 for c in text.encode()]
    if len(b) > length:
        raise ValueError("text too long")
    return b + [0] * (length - len(b))


def from_bytes(ids) -> str:
    out = bytearray()
    for i in ids:
        if i <= 0:
            break
        out.append(i - 1)
    return out.decode(errors="replace")
