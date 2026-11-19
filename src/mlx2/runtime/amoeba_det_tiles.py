# SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Deterministic amoeba tiles: a calculator and a fact table on the tile contract (experimental, unqualified).

The tile contract is shared with the neural boxes of :mod:`amoeba_tiles`. A tile reads the MLP input
``x`` at a read layer ``L`` (the post-attention-norm residual a neural box's gate reads) plus the
token context, opens or stays closed per token, and either adds a delta to the MLP output at a
write layer ``L' >= L`` (that is, to the residual stream after ``L'``) or forces/constrains the next
tokens. A closed tile adds exactly zero and forces nothing.

Two kinds, both fail closed:

* ``calculator`` (write ``force`` / ``constrain``; research-only ``push``). Gate: a linear probe on
  ``x_L`` ("an arithmetic result comes next"), AND a parse of the recent token text that finds a
  complete ``a OP b =`` expression (``+ - * x × ÷ /`` and the words plus/minus/times/divided by on
  integers and decimals). The result is computed exactly with :class:`fractions.Fraction`; a
  quotient with no terminating decimal, an operand that looks like a date, version, phone number,
  ID or time, or a mixed unit is refused, and the host keeps its own tokens. ``force`` writes the
  result's canonical tokenization; ``constrain`` masks each step to the tokens that keep the text a
  prefix of the result (any tokenization of the same string). Both are exact in the result string.
  ``push`` (a residual push at ``L'`` toward the next result token) is approximate and is only
  evaluated through :meth:`DetTileSet.mlp_hook` in research; serving refuses it.
* ``fact_table`` (write ``delta``). Recogniser: ``k`` host neurons of the read layer's MLP,
  ``act(gate_n . x) * (up_n . x)``, unit-normalised into a key. Table: stored keys, one row per
  (entity, relation), and value vectors (residual deltas at ``L'``, solved per fact in closed form by
  the builder and recorded with their source). Lookup: the nearest key by cosine; the tile opens
  only when the best cosine is at least ``tau``, it beats the second-best row by ``margin``, and the
  row is not a null row (null rows hold the same entities under other relations, so "population of
  France" lands on a null row and falls back). Closed, the host's own (neural) path is untouched.

Arithmetic: the calculator probe is an f32 row-wise product-sum (the neural gate's arithmetic,
``amoeba_tiles.GATE_ARITHMETIC``), so its decision does not depend on the batch a row shares. The
fact-table key is f32 matmuls (neuron rows) or a gather (residual dims), then a cosine against the
keys: matmul kernels depend on the row count, so a decision within rounding of tau or the margin
may differ between prefill, decode and replay (a high-margin divergence when it does; not audited).
Writes are added in f32 and cast to the MLP output dtype. Host syncs: the processor reads the token
history each step (to decode its text), reads the gate once a complete expression is in the text,
and checks the planned token against earlier processors' masks while writing.

Routes: the ordinary route, and (for a calculator that reads ``layer_output_capture``) the
native-MTP route of an adapter that declares it. With read point ``mlp_input`` the gate is read at
the last position of the latest forward (one-row forwards, or decode rows the generator binds). With
``layer_output_capture`` (the read layer's output hidden state through the trunk's capture
projection) every position of the last few trunk forwards is kept with its input token ids, and the
processor finds the position whose input token and preceding window match its own history: a
speculative verify row j reads position j's gate. No match, or candidates that disagree, fails
closed. The processor publishes a verify ledger (rows past the accept point are rewound) and a pure
draft probe (it continues an active plan, so drafts propose the forced tokens, and may start one
only where the gate matches the probe's history). Per-request counters are settled through the
ledger. Fact-table writes change the arithmetic, so the tile
set's identity joins the execution-numerics contract (APCv2 namespace).

Nothing here names a model: the adapter supplies the MLP attribute and activation.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import itertools
import json
import re
import threading
from fractions import Fraction
from pathlib import Path

SCHEMA = "mlx2.amoeba-det-tiles.v1"
RECEIPT_SCHEMA = "mlx2.amoeba-det-tiles-receipt.v1"
KINDS = ("calculator", "fact_table")
WRITE_MODES = {"calculator": ("force", "constrain", "push"), "fact_table": ("delta",)}
SERVING_WRITE_MODES = {"calculator": ("force", "constrain"), "fact_table": ("delta",)}
GATE_ARITHMETIC = "f32-rowwise-product-sum-v1"
KEY_ARITHMETIC = "f32-matmul-(neuron-act|residual-dims)-centred-unit-cosine-v2"
PARSER_ID = "calc-parse-v1"
# v2 adds whole expressions (precedence, brackets, powers, unary minus) on top of v1's single steps.
PARSER_V2 = "calc-parse-v2"
PARSER_V3 = "calc-parse-v3"
PARSER_V4 = "calc-parse-v4"
PARSER_V5 = "calc-parse-v5-resident"
PARSERS = (PARSER_ID, PARSER_V2, PARSER_V3, PARSER_V4, PARSER_V5)
# Where a calculator's probe reads: the read layer's MLP input (wrapped MLP), or the read layer's
# output hidden state through the trunk's capture projection (adapter-installed layer reader).
READ_POINTS = ("mlp_input", "layer_output_capture")
# Matching a stashed forward position to a processor's history: the input token at the position and
# up to this many preceding tokens of the forward must equal the history's tail.
MATCH_WINDOW = 8
STASH_FORWARDS = 4
# Speculative routes a position-matched calculator may serve on (the adapter must also declare it).
SPECULATIVE_ROUTES = ("native_mtp",)
# Tokens of context the calculator decodes to find an expression.
CONTEXT_TOKENS = 48


class DetTileRefused(ValueError):
    """A request or route asked for a deterministic-tile capability this route does not have."""

    http_status = 409
    code = "det_tile_refused"


# ----------------------------------------------------------------------------------------------
# Calculator: parse + exact evaluation (pure Python, CPU-tested)
# ----------------------------------------------------------------------------------------------

_NUM = r"(?P<{name}>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)"
_OPS = {
    "+": "+", "plus": "+",
    "-": "-", "−": "-", "minus": "-",
    "*": "*", "x": "*", "×": "*", "times": "*", "multiplied by": "*",
    "/": "/", "÷": "/", "divided by": "/", "over": None,
}
_OP_RE = r"(?P<op>\+|-|−|\*|×|÷|/|x|plus|minus|times|multiplied by|divided by)"
# A bare "is" also follows ranges, scores and years ("5 - 10 is", "2020 - 2021 is"): not a cue.
_CUE_RE = r"(?P<cue>=|equals|is equal to)"
# a OP b CUE at the end of the text, an optional single trailing space.
_EXPR = re.compile(
    r"(?P<pre>(?:^|.))" + r"(?P<cur1>\$)?" + _NUM.format(name="a")
    + r"(?P<s1>[ \t]*)" + _OP_RE + r"(?P<s2>[ \t]*)" + r"(?P<cur2>\$)?" + _NUM.format(name="b")
    + r"(?P<s3>[ \t]*)" + _CUE_RE + r"(?P<tail> ?)\Z",
    re.S | re.ASCII,
)
# The text before ``a`` (whitespace skipped) must not continue an expression: an operator, a
# bracket, a power, a separator or an operand word before ``a`` means a chained expression
# ("2 * 3 + 4 =") whose last step alone would be wrong. Refuse rather than evaluate part of it.
_CHAIN_CHARS = set("+-−*×÷/^()[]{}=,%")
_CHAIN_WORDS = ("plus", "minus", "times", "by", "over", "x")
# What may stand right before ``a`` is an allow-list, not a deny-list: "2 \times 3 + 4", "2 · 3 + 4",
# "10 – 3 + 2", "negative 5 + 3", "the square root of 16 + 9" and "20% of 50 + 1" all end in a
# valid-looking "a OP b =" whose value alone is wrong. Allowed: the start of an untruncated text,
# a sentence or line break, a quote, GSM8K's "<<", or one of these lead-in words.
_LEAD_CHARS = set("\n.:;!?\"'`")
_LEAD_WORDS = frozenset((
    "is", "was", "are", "were", "be", "so", "then", "thus", "hence", "therefore", "now", "next", "first",
    "we", "need", "needs", "compute", "computes", "computed", "calculate", "calculates", "calculated",
    "out", "what", "what's", "whats", "gives", "give", "get", "gets", "got", "total", "answer", "result",
    "check", "checking", "verify", "step", "that", "means", "do", "does", "did", "evaluate", "solve",
))


def _lead_ok(before: str, truncated: bool):
    """True when the text before operand ``a`` cannot be part of a longer expression."""
    b = before.rstrip(" \t")
    if not b:
        return not truncated
    if b.endswith("<<") or b[-1] in _LEAD_CHARS:
        return True
    word = b.split()[-1].lower() if b.split() else ""
    return word.isascii() and word in _LEAD_WORDS
# Refusal patterns, checked on the text window around the expression.
_DATE = re.compile(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}")
_TIME = re.compile(r"\d{1,2}:\d{2}")
_VERSION = re.compile(r"\d+\.\d+\.\d+")


def _range_like(a, op, b, *, spaced):
    """Phone numbers, year ranges, page ranges and dates written as a - b or a / b."""
    if op not in "-/" or "." in a or "." in b or "," in a or "," in b:
        return None
    ia, ib = int(a), int(b)
    if op == "-":
        if len(a) == 3 and len(b) == 4:
            return "phone_like"
        if 1000 <= ia <= 2200 and 1000 <= ib <= 2200:
            return "year_range"
        if not spaced and ia < ib:
            return "range_like"                # "pages 10-20": GSM8K's unspaced a-b has a > b
    if op == "/" and not spaced and 1 <= ia <= 12 and 1 <= ib <= 31:
        return "date_like"                     # "12/25"
    return None


def _number(s: str) -> Fraction:
    return Fraction(s.replace(",", ""))


def parse_expression(text: str, truncated: bool = False):
    """The complete ``a OP b CUE`` expression the text ends with, or None (with a reason).

    Returns ``(expr, None)`` or ``(None, reason)``; ``expr`` is a dict with ``a``, ``b``, ``op``
    (one of ``+-*/``), ``currency``, ``needs_space`` and the matched span.
    """
    truncated = truncated or len(text) > 160
    m = _EXPR.search(text[-160:])
    if m is None:
        return _parse_question(text, truncated)
    pre = m.group("pre")
    if pre and (pre.isalnum() or pre in "._-/:#@,$"):
        # Operand glued to a word, an ID, a version, a path, a time or a number ("A12", "v3.1", "1,2345").
        return None, "operand_boundary"
    before = text[-160:][:m.start("pre") + len(pre)].rstrip()
    if before and (before[-1] in _CHAIN_CHARS or before[-1].isdigit()
                   or before.lower().endswith(tuple(" " + w for w in _CHAIN_WORDS))
                   or before.lower() in _CHAIN_WORDS):
        return None, "chained_expression"
    op_raw = m.group("op")
    s1, s2 = m.group("s1"), m.group("s2")
    if op_raw == "x" and not (s1 and s2):
        return None, "x_without_spaces"          # "3x4" is a dimension / hex-like token, not "times"
    if op_raw in ("-", "−", "/") and not (s1 and s2) and m.group("cue") != "=":
        # "555-1234", "12/25" and "2024-05" are phone numbers, dates and ranges far more often than
        # arithmetic; unspaced - and / need an explicit "=" (GSM8K writes "48/2=24").
        return None, "unspaced_dash_or_slash"
    why = _range_like(m.group("a"), _OPS[op_raw], m.group("b"), spaced=bool(s1 and s2))
    if why:
        return None, why
    if op_raw.isalpha() and not (s1 and s2):
        return None, "word_operator_boundary"
    window = text[-160:]
    span = window[m.start("a") - (1 if m.group("cur1") else 0):m.end("b")]
    if _DATE.search(span) or _TIME.search(window[max(0, m.start("a") - 3):m.end("b") + 1]):
        return None, "date_or_time"
    if _VERSION.search(window[max(0, m.start("a") - 12):m.end("b") + 1]):
        return None, "version"
    cur1, cur2 = m.group("cur1"), m.group("cur2")
    if bool(cur1) != bool(cur2) and _OPS[op_raw] in "+-":
        return None, "mixed_units"
    if cur1 and cur2 and _OPS[op_raw] == "*":
        return None, "mixed_units"             # dollars times dollars
    if bool(cur1) != bool(cur2):
        return None, "one_sided_dollar"        # "$12 / 4 = 3$": the "$" may open LaTeX math, not money
    if not _lead_ok(text[-160:][:m.start("a") - (1 if cur1 else 0)], truncated):
        return None, "lead_not_allowed"
    expr = {"a": m.group("a"), "b": m.group("b"), "op": _OPS[op_raw],
            "currency": _currency(cur1, cur2, _OPS[op_raw]),
            # "48 / 2 = 24" and "48/2=24": the result's spacing follows the cue's.
            "needs_space": m.group("tail") == "" and (m.group("s3") != "" or m.group("cue") != "="),
            "thousands": "," in m.group("a") or "," in m.group("b"),
            "cue": m.group("cue"), "cued": True, "span": m.group(0)[len(pre):]}
    return expr, None


def _currency(cur1, cur2, op):
    """$a + $b and $a - $b: dollars; $a * b, a * $b, $a / b: dollars; $a / $b: a ratio."""
    if op in "+-":
        return "$" if cur1 and cur2 else ""
    if op == "*":
        return "$" if bool(cur1) != bool(cur2) else ""
    return "$" if cur1 and not cur2 else ""


# "What is 7 times 12?" and the answer has not started: the expression in the last 160 characters,
# then a question mark, then no digit before the end (the answer slot). Only the probe can say the
# answer comes next here, so this form never fires on the text alone.
_QUESTION = re.compile(
    r"(?:^|[^\w.$:/#@-])" + r"(?P<cur1>\$)?" + _NUM.format(name="a") + r"(?P<s1>[ \t]+)" + _OP_RE + r"(?P<s2>[ \t]+)"
    + r"(?P<cur2>\$)?" + _NUM.format(name="b") + r"\s*\?(?P<rest>[^0-9]{0,160})\Z", re.S | re.ASCII)


def _parse_question(text, truncated=False):
    truncated = truncated or len(text) > 320
    window = text[-320:]
    m = None
    for m in _QUESTION.finditer(window):
        pass
    if m is None:
        return None, "no_expression"
    if _DATE.search(m.group(0)) or _TIME.search(m.group(0)) or _VERSION.search(m.group(0)):
        return None, "date_or_time"
    op_raw = m.group("op")
    cur1, cur2 = m.group("cur1"), m.group("cur2")
    if bool(cur1) != bool(cur2) and _OPS[op_raw] in "+-":
        return None, "mixed_units"
    if cur1 and cur2 and _OPS[op_raw] == "*":
        return None, "mixed_units"
    if bool(cur1) != bool(cur2):
        return None, "one_sided_dollar"
    why = _range_like(m.group("a"), _OPS[op_raw], m.group("b"), spaced=True)
    if why:
        return None, why
    head = window[:m.start()].rstrip() + window[m.start():m.start("a") - (1 if cur1 else 0)].rstrip()
    if head and (head[-1] in _CHAIN_CHARS or head[-1].isdigit()):
        return None, "chained_expression"
    if not _lead_ok(window[:m.start("a") - (1 if cur1 else 0)], truncated):
        return None, "lead_not_allowed"
    rest = m.group("rest")
    return {"a": m.group("a"), "b": m.group("b"), "op": _OPS[op_raw],
            "thousands": "," in m.group("a") or "," in m.group("b"),
            "currency": _currency(cur1, cur2, _OPS[op_raw]),
            "needs_space": not (rest == "" or rest[-1].isspace()) and rest[-1] not in "\"'(",
            "cue": "?", "cued": False, "span": m.group(0).strip()}, None


def _decimal_string(value: Fraction):
    """Exact terminating decimal string of ``value``, or None when it does not terminate."""
    sign = "-" if value < 0 else ""
    value = abs(value)
    den = value.denominator
    twos = fives = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    while den % 5 == 0:
        den //= 5
        fives += 1
    if den != 1:
        return None
    places = max(twos, fives)
    scaled = value * (10 ** places)
    assert scaled.denominator == 1
    digits = str(scaled.numerator)
    if places == 0:
        return sign + digits
    digits = digits.rjust(places + 1, "0")
    whole, frac = digits[:-places], digits[-places:].rstrip("0")
    return sign + whole + ("." + frac if frac else "")


def evaluate(expr, *, arithmetic_backend="cpu") -> tuple:
    """``(result_string, None)`` or ``(None, reason)``: the exact result of a parsed expression."""
    if arithmetic_backend != "cpu":
        from .amoeba_calc_gpu import evaluate as evaluate_gpu
        s, why = evaluate_gpu(expr, arithmetic_backend)
        if s is None:
            return None, why
    else:
        op = expr["op"]
        if op == "expr":
            v = expr["value"]
            a = b = None
        else:
            a, b = _number(expr["a"]), _number(expr["b"])
        if op == "expr":
            pass
        elif op == "+":
            v = a + b
        elif op == "-":
            v = a - b
        elif op == "*":
            v = a * b
        elif op == "/":
            if b == 0:
                return None, "division_by_zero"
            v = a / b
        else:
            return None, "unknown_operator"
        s = _decimal_string(v)
        if s is None:
            return None, "non_terminating_quotient"
    if expr.get("thousands"):
        neg, body = s.startswith("-"), s.lstrip("-")
        whole, _, frac = body.partition(".")
        s = ("-" if neg else "") + f"{int(whole):,}" + ("." + frac if frac else "")
    if expr.get("currency"):
        s = ("-" if s.startswith("-") else "") + expr["currency"] + s.lstrip("-")
    return s, None


# ---- calc-parse-v2: whole expressions --------------------------------------------------------
# A chain such as "2 + 3 * 4 =" or "(12 - 4) / 2 =" is parsed whole with precedence and evaluated
# exactly. Everything v1 refuses about the text around the expression (lead allow-list, dates,
# times, versions, phones, currency, unspaced dashes/slashes) still applies; a lone "a OP b" is
# left to v1 so its guards stay the only ones on that form.
_V2_TOKEN = re.compile(
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\d,.]?\d)"
    r"|(?P<op>\*\*|[-+*/^]|×|÷|−)"
    r"|(?<![A-Za-z])(?P<word>multiplied[ \t]+by|divided[ \t]+by|plus|minus|times|x)(?![A-Za-z])"
    r"|(?P<lp>\()|(?P<rp>\))|(?P<ws>[ \t]+)")
_V2_BINARY = {"+": "+", "plus": "+", "-": "-", "−": "-", "minus": "-", "*": "*", "×": "*", "times": "*",
              "x": "*", "multiplied by": "*", "/": "/", "÷": "/", "divided by": "/", "^": "^", "**": "^"}
_V2_CUE = re.compile(r"(?P<s3>[ \t]*)(?P<cue>=|equals|is equal to)(?P<tail> ?)\Z")
_V2_QUESTION = re.compile(r"[ \t]*\?(?P<rest>[^0-9?]{0,160})\Z", re.S)
_V2_MAX_DIGITS = 60
# Adjacent literal pairs joined by - or / (v1's range, phone and date checks run on each).
_V2_PAIR = re.compile(r"(?<![\d.,])(?P<a>\d+)(?P<s1>[ \t]*)(?P<op>[-/])(?P<s2>[ \t]*)(?P<b>\d+)(?![\d.,])")
_V2_MAX_EXPONENT = 64


class _V2Refused(Exception):
    pass


def _v2_tokens(span):
    """[(kind, text, spaced_before, spaced_after)] for the whole span, or None if any char is foreign."""
    out, pos, raw = [], 0, []
    while pos < len(span):
        m = _V2_TOKEN.match(span, pos)
        if m is None or m.end() == pos:
            return None
        kind = m.lastgroup
        text = m.group(kind)
        if kind == "word":
            text = " ".join(text.split())
        raw.append((kind, text))
        pos = m.end()
    for i, (kind, text) in enumerate(raw):
        if kind == "ws":
            continue
        before = i > 0 and raw[i - 1][0] == "ws"
        after = i + 1 < len(raw) and raw[i + 1][0] == "ws"
        out.append((kind, text, before, after))
    return out


def _v2_check_size(v):
    if getattr(v, "_calculator_ir", False):
        return v
    if len(str(abs(v.numerator))) > _V2_MAX_DIGITS or len(str(v.denominator)) > _V2_MAX_DIGITS:
        raise _V2Refused("too_large")
    return v


class _V2Parser:
    """expr := term (+|- term)*; term := unary (*|/ unary)*; unary := - unary | power;
    power := atom (^ unary)?  (right-associative); atom := number | ( expr )."""

    def __init__(self, toks, deferred=False):
        self.deferred = deferred
        self.t, self.i = toks, 0
        self.binary, self.parens, self.power, self.unary = [], 0, 0, 0
        self.thousands = False

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None, False, False)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def _op(self, accept):
        kind, text, sb, sa = self.peek()
        if kind in ("op", "word") and _V2_BINARY.get(text) in accept:
            self.take()
            if kind == "word" and not (sb and sa):
                raise _V2Refused("word_operator_boundary")   # "2plus3", "2times3", "3x4"
            self.binary.append((_V2_BINARY[text], sb and sa))
            return _V2_BINARY[text]
        return None

    def expr(self):
        v = self.term()
        while (op := self._op("+-")) is not None:
            w = self.term()
            v = _v2_check_size(v + w if op == "+" else v - w)
        return v

    def term(self):
        v = self.unary_()
        while (op := self._op("*/")) is not None:
            w = self.unary_()
            if op == "/":
                if not self.deferred and w == 0:
                    raise _V2Refused("division_by_zero")
                v = v / w
            else:
                v = v * w
            _v2_check_size(v)
        return v

    def unary_(self):
        kind, text, _, spaced_after = self.peek()
        if kind == "op" and text in ("-", "−"):
            if spaced_after:
                raise _V2Refused("spaced_unary_minus")   # "- 4 * 2": a list bullet, not a negative
            self.take()
            self.unary += 1
            return -self.unary_()
        return self.power_()

    def power_(self):
        base = self.atom()
        kind, text, _, _ = self.peek()
        if kind == "op" and text in ("^", "**"):
            self.take()
            self.power += 1
            e = self.unary_()
            if self.deferred:
                return base ** e
            if e.denominator != 1 or abs(e) > _V2_MAX_EXPONENT:
                raise _V2Refused("exponent_not_small_integer")
            if base == 0 and e < 0:
                raise _V2Refused("division_by_zero")
            # Bound the result before computing it: digits(base^e) <= |e| * digits(base).
            width = max(len(str(abs(base.numerator))), len(str(base.denominator)))
            if abs(int(e)) * width > _V2_MAX_DIGITS + abs(int(e)):
                raise _V2Refused("too_large")
            return _v2_check_size(base ** int(e))
        return base

    def atom(self):
        kind, text, _, _ = self.take()
        if kind == "num":
            self.thousands |= "," in text
            if self.deferred:
                from .amoeba_calc_gpu import Number
                return Number.literal(text)
            return _v2_check_size(_number(text))
        if kind == "lp":
            self.parens += 1
            v = self.expr()
            if self.take()[0] != "rp":
                raise _V2Refused("unbalanced_brackets")
            return v
        raise _V2Refused("not_an_expression")


def _v2_parse(span, deferred=False):
    """(value, parser) for a span that is exactly one expression, or raise / return None."""
    toks = _v2_tokens(span)
    if not toks:
        return None
    p = _V2Parser(toks, deferred)
    try:
        v = p.expr()
    except _V2Refused as e:
        if str(e) in ("not_an_expression", "unbalanced_brackets"):
            return None
        raise
    if p.i != len(toks):
        return None
    return v, p


def parse_expression_v2(text: str, truncated: bool = False, *, deferred=False):
    """``(expr, None)`` for a whole expression v1 cannot read, ``(None, reason)`` when one is found
    and refused, or ``(None, "v1")`` to leave the text to v1 (no expression, or a lone a OP b)."""
    window = text[-160:]
    truncated = truncated or len(text) > 160
    m = _V2_CUE.search(window)
    if m is not None:
        end, cued = m.start(), True
    else:
        q = _V2_QUESTION.search(window)
        if q is None:
            return None, "v1"
        end, cued = q.start(), False
    region = window[:end]
    # Candidate starts: the start of a number, "(" or a leading minus that sits on an allowed lead.
    for i, ch in enumerate(region):
        if not (ch.isdigit() or ch in "(-−"):
            continue
        if i and (region[i - 1].isalnum() or region[i - 1] in "._,$#@/:;'"):
            continue                           # glued to a word, ID, version, path, time, ratio or number
        if not _lead_ok(region[:i], truncated):
            continue
        prior = region[:i].rstrip()            # all whitespace, newlines included
        if prior and (prior[-1] in _CHAIN_CHARS or prior[-1].isdigit()):
            return None, "chained_expression"  # "So 2 +\n3 * 4 + 1 =": a line break inside an expression
        span = region[i:]
        try:
            got = _v2_parse(span, deferred)
        except _V2Refused as e:
            return None, str(e)
        if got is None:
            continue
        value, p = got
        if not p.binary and not p.power:
            return None, "no_operation"         # "So -3 =", "So (5) =": nothing to compute
        if len(p.binary) < 2 and not (p.parens or p.power or p.unary):
            return None, "v1"                   # a lone a OP b: v1 owns it
        around = region[max(0, i - 3):]
        if _DATE.search(around) or _TIME.search(around) or _VERSION.search(region[max(0, i - 12):]):
            return None, "date_or_time"
        for rm in _V2_PAIR.finditer(span):
            why = _range_like(rm.group("a"), rm.group("op"), rm.group("b"), spaced=bool(rm.group("s1") and rm.group("s2")))
            if why:
                return None, why                # "1939 - 1945 + 1", "555-1234 + 1"
        lead_gsm8k = region[:i].rstrip(" \t").endswith("<<")
        if any(op in "-/" and not spaced for op, spaced in p.binary) and not lead_gsm8k:
            return None, "unspaced_dash_or_slash"   # "555-123-4567", "12/25/2024", "2024-05-12"
        if not cued and not all(spaced for _, spaced in p.binary):
            return None, "unspaced_question"
        if cued:
            needs_space = m.group("tail") == "" and (m.group("s3") != "" or m.group("cue") != "=")
            cue = m.group("cue")
        else:
            rest = q.group("rest")
            needs_space = not (rest == "" or rest[-1].isspace()) and rest[-1] not in "\"'("
            cue = "?"
        return {"value": value, "op": "expr", "thousands": p.thousands, "currency": "",
                "needs_space": needs_space, "cue": cue, "cued": cued, "span": span.strip(),
                "parser": PARSER_V2}, None
    return None, "v1"


# Two complete operand rows and a result separator only. Do not guess at arbitrary tables,
# partial products, missing operands, or an already-written result.
_LATEX_COLUMN = re.compile(
    r"\\begin\{array\}\{r\}\s*"
    + _NUM.format(name="a") + r"\s*\\\\\s*\\times\s*"
    + _NUM.format(name="b") + r"\s*\\\\\s*\\hline[ \t\r\n]*\Z", re.ASCII)
_LATEX_OPERATOR = re.compile(r"\\(times|cdot|div)(?![A-Za-z])")


def parse_expression_v3(text: str, truncated: bool = False, *, deferred=False):
    """V2 plus explicit LaTeX operators and a complete two-row multiplication array."""
    # Bound the ORIGINAL input before rewriting: normalization must never bring a cut
    # operand back into view, or splice a truncated command into a valid expression.
    window = text[-160:]
    cut = truncated or len(text) > 160
    m = _LATEX_COLUMN.search(window)
    if m:
        a, b = m.group("a"), m.group("b")
        return {"a": a, "b": b, "op": "*", "currency": "", "needs_space": not window[-1].isspace(),
                "thousands": "," in a or "," in b, "cue": r"\hline", "cued": True,
                "span": m.group(), "parser": PARSER_V3}, None
    normalized = _LATEX_OPERATOR.sub(lambda m: " * " if m[1] != "div" else " / ", window)
    expr, why = parse_expression_v2(normalized, cut, deferred=deferred)
    if why == "v1":
        expr, why = parse_expression(normalized, cut)
    return expr, why


def parse_expression_v4(text: str, truncated: bool = False, *, deferred=False):
    """Strict complete LaTeX span, normalized before the semantic character bound."""
    from .amoeba_math_latex import MathError, parse_latex, canonical_expression, evaluate_exact, to_deferred
    if _LATEX_COLUMN.search(text[-160:]):
        return parse_expression_v3(text, truncated, deferred=deferred)
    raw = text[-4096:]
    cut = truncated or len(text) > 4096
    tail = raw.rstrip()
    closing = None
    for delimiter in ('$$', r'\)', r'\]', '$'):
        if tail.endswith(delimiter):
            closing = delimiter
            tail = tail[:-len(delimiter)].rstrip()
            break
    cue = _V2_CUE.search(tail)
    if cue is None:
        return parse_expression_v3(text, truncated, deferred=deferred)
    body = tail[:cue.start()].rstrip()
    # Find a genuinely open math environment; a currency dollar is not enough.
    stack = []
    for m in re.finditer(r'\\[\[\]()]|(?<!\\)\$\$|(?<!\\)\$', body):
        token = m.group()
        if token in (r'\]', r'\)'):
            if stack and {r'\[':r'\]', r'\(':r'\)'}.get(stack[-1][0]) == token:
                stack.pop()
        elif token in ('$', '$$') and stack and stack[-1][0] == token:
            stack.pop()
        else:
            stack.append((token, m.end()))
    if len(stack) > 1:
        return None, 'nested_math_environments'
    if stack:
        opener, end = stack[0]
        if opener == '$' and closing != '$':
            return None, 'one_sided_dollar'
        if closing and {r'\[':r'\]', r'\(':r'\)', '$':'$', '$$':'$$'}.get(opener) != closing:
            return None, 'unbalanced_math_delimiters'
        candidate = body[end:].strip()
        anchored = True
    else:
        if closing:
            return None, 'unbalanced_math_delimiters'
        line = body.rfind('\n') + 1
        candidate = body[line:].strip()
        before = body[:line].rstrip()
        if before and (before[-1] in _CHAIN_CHARS or before[-1].isdigit()):
            return None, 'chained_expression'
        lead = re.match(r"(?i)^(?:so|then|thus|therefore|calculate|compute|evaluate|answer|result|hence|we need|the result is|the answer is)(?:\s*:\s*|\s+)", candidate)
        anchored = bool(line or lead or not cut)
        if lead:
            candidate = candidate[lead.end():]
        if not anchored:
            return None, 'truncated_operand'
    advanced = bool(stack or closing or '\\' in candidate or '!' in candidate)
    if not advanced:
        return parse_expression_v3(text, truncated, deferred=deferred)
    if _DATE.search(candidate) or _TIME.search(candidate) or _VERSION.search(candidate):
        return None, 'date_or_time'
    try:
        ast = parse_latex(candidate)
        def has_calculation(node):
            return node.op not in ('literal', 'variable', 'neg') or any(has_calculation(a) for a in node.args)
        if not has_calculation(ast):
            return None, 'no_arithmetic_operation'
        normalized = canonical_expression(ast)
        if len(normalized) > 160:
            return None, 'normalized_expression_bound'
        # Retain v1's explicit phone/year guards on literal subtraction nodes.
        def guards(node):
            if node.op == '-' and all(a.op == 'literal' for a in node.args):
                why = _range_like(node.args[0].text, '-', node.args[1].text, spaced=True)
                if why:
                    raise MathError(why)
            for a in node.args:
                guards(a)
        guards(ast)
        value = to_deferred(ast, max_nodes=256) if deferred else evaluate_exact(ast)
    except MathError as error:
        return None, error.reason
    return {'value':value, 'op':'expr', 'thousands':False, 'currency':'',
            'needs_space':not text[-1:].isspace() and (bool(closing) or bool(cue.group('s3'))),
            'cue':cue.group('cue'), 'cued':True, 'span':candidate, 'normalized':normalized,
            'parser':PARSER_V4, 'syntax':'latex_math'}, None


def calculator_answer(text: str, truncated: bool = False, parser: str = PARSER_ID, *, arithmetic_backend="cpu"):
    """``(answer_text_to_emit, info)``: the text the calculator writes after ``text``, or None.

    ``truncated``: ``text`` is a cut window of a longer context, so an operand at its very start
    may itself be cut ("...12345 + 6 =" seen as "45 + 6 =") and is refused.
    """
    if arithmetic_backend not in ("cpu", "gpu", "gemm"):
        raise ValueError(f"unknown calculator arithmetic backend {arithmetic_backend!r}")
    if parser not in PARSERS:
        raise ValueError(f"unknown calculator parser {parser!r}")
    if parser == PARSER_ID:
        expr, why = None, "v1"
    else:
        try:
            expr, why = {PARSER_V2:parse_expression_v2, PARSER_V3:parse_expression_v3, PARSER_V4:parse_expression_v4}[parser](text, truncated, deferred=arithmetic_backend != "cpu")
        except Exception:                      # noqa: BLE001 - a parser bug must refuse, never raise into decode
            return None, {"refused": "v2_internal_error"}
    if why == "v1":
        expr, why = parse_expression(text, truncated)
    if expr is None:
        return None, {"refused": why}
    result, why = evaluate(expr, arithmetic_backend=arithmetic_backend)
    if result is None:
        return None, {"refused": why, "expr": expr["span"]}
    return (" " if expr["needs_space"] else "") + result, {"expr": expr["span"], "result": result,
                                                            "cued": expr["cued"], "arithmetic_backend": arithmetic_backend,
                                                            **({"arithmetic_execution": expr["arithmetic_execution"]} if "arithmetic_execution" in expr else {}),
                                                            **({"syntax": "latex_column"} if expr.get("cue") == r"\hline" else ({"syntax":expr["syntax"], "normalized":expr.get("normalized")} if expr.get("syntax") else {}))}


# ----------------------------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------------------------

def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path) -> dict:
    """Read and validate a deterministic tile manifest; every file digest is checked."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise ValueError(f"deterministic tile manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"deterministic tiles: schema {manifest.get('schema')!r} is not {SCHEMA}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("deterministic tiles: manifest lists no files")
    for name, digest in files.items():
        f = path.parent / name
        if not f.is_file():
            raise ValueError(f"deterministic tiles: tile file missing: {f}")
        actual = file_sha256(f)
        if actual != digest:
            raise ValueError(f"deterministic tiles: digest mismatch for {name}")
    for key in ("source", "layers", "tiles"):
        if key not in manifest:
            raise ValueError(f"deterministic tiles: manifest has no {key!r}")
    ids = [t.get("id") for t in manifest["tiles"]]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", i)
                                             for i in ids):
        raise ValueError("deterministic tiles: tile ids must be unique [A-Za-z0-9_.-]{1,96}")
    for t in manifest["tiles"]:
        kind = t.get("kind")
        if kind not in KINDS:
            raise ValueError(f"deterministic tiles: unknown kind {kind!r}")
        if t.get("write_mode") not in WRITE_MODES[kind]:
            raise ValueError(f"deterministic tiles: {kind} write_mode must be one of {WRITE_MODES[kind]}")
        L, W = int(t["read_layer"]), int(t["write_layer"])
        if not (0 <= L <= W < int(manifest["layers"])):
            raise ValueError("deterministic tiles: need 0 <= read_layer <= write_layer < layers")
        if kind == "calculator" and t.get("parser", PARSER_ID) not in PARSERS:
            raise ValueError(f"deterministic tiles: calculator parser must be one of {PARSERS}")
        if kind == "calculator" and t.get("parser") == PARSER_V5:
            if t.get('arithmetic_backend')!='gpu' or t.get('write_mode')!='force' or t.get('gate',{}).get('kind')!='validated_syntax':
                raise ValueError('resident calculator requires GPU arithmetic, force writing and validated_syntax gate')
        if kind == "calculator" and t.get("arithmetic_backend", "cpu") not in ("cpu", "gpu", "gemm"):
            raise ValueError("deterministic tiles: unknown calculator arithmetic backend")
        if kind == "calculator" and t.get("trigger_prefilter", "off") not in ("off", "gpu"):
            raise ValueError("deterministic tiles: unknown calculator trigger prefilter")
        if "complete_literal_requests" in t and type(t["complete_literal_requests"]) is not bool:
            raise ValueError("deterministic tiles: complete_literal_requests must be boolean")
        if "context_tokens" in t and (type(t["context_tokens"]) is not int or not 1 <= t["context_tokens"] <= 1024):
            raise ValueError("deterministic tiles: context_tokens must be an integer in 1..1024")
        if t.get("read_point", "mlp_input") not in READ_POINTS:
            raise ValueError(f"deterministic tiles: read_point must be one of {READ_POINTS}")
        if t.get("read_point", "mlp_input") != "mlp_input" and (kind != "calculator" or L != W):
            raise ValueError("deterministic tiles: layer_output_capture is a calculator read point "
                             "(read_layer == write_layer)")
        if t.get("read_point", "mlp_input") != "mlp_input" and (
                t.get("parser") == PARSER_V5 or t.get("trigger_prefilter", "off") != "off"
                or t.get("complete_literal_requests", False)):
            # These read the mlp_input stash (lane binding, GPU cue window) or carry device state
            # that the position-matched verify ledger does not rewind: refuse the combination.
            raise ValueError("deterministic tiles: layer_output_capture calculators do not combine with "
                             "the resident parser, trigger_prefilter or complete_literal_requests")
        if t.get("file") not in files:
            raise ValueError(f"deterministic tiles: tile {t['id']} file is not in the manifest's files")
    manifest["_dir"] = str(path.parent)
    manifest["_digest"] = file_sha256(path)
    return manifest


def tile_set_identity(manifest: dict) -> dict:
    return {"schema": SCHEMA, "manifest_sha256": manifest["_digest"],
            "source_fingerprint": manifest["source"]["fingerprint"],
            "gate_arithmetic": GATE_ARITHMETIC, "key_arithmetic": KEY_ARITHMETIC,
            # v1-only sets (and manifests without a "parser" field) keep their original identity.
            "parser": next((p for p in reversed(PARSERS) if any(t.get("parser", PARSER_ID) == p
                                                              for t in manifest["tiles"])), PARSER_ID),
            "tiles": [{"id": t["id"], "kind": t["kind"], "write_mode": t["write_mode"],
                       "read_layer": int(t["read_layer"]), "write_layer": int(t["write_layer"]),
                       "exact": t["write_mode"] in ("force", "constrain"),
                       **({"context_tokens":t.get("context_tokens",256)} if t.get("parser") == PARSER_V4 else {}),
                       **({"trigger_prefilter": t["trigger_prefilter"]} if "trigger_prefilter" in t else {}),
                       **({"complete_literal_requests":t["complete_literal_requests"]} if "complete_literal_requests" in t else {}),
                       **({"arithmetic_backend": t.get("arithmetic_backend", "gpu" if t.get("parser") in (PARSER_V3, PARSER_V4) else "cpu")}
                          if t.get("parser") in (PARSER_V3, PARSER_V4) or "arithmetic_backend" in t else {}),
                       **({"parser": t["parser"]} if t.get("parser") in (PARSER_V2, PARSER_V3, PARSER_V4, PARSER_V5) else {}),
                       **({"parser_execution":"gpu", "gate":"complete_syntax_and_exact_result"} if t.get('parser')==PARSER_V5 else {}),
                       # mlp_input sets keep their original identity.
                       **({"read_point": t["read_point"]}
                          if t.get("read_point", "mlp_input") != "mlp_input" else {})}
                      for t in manifest["tiles"]]}


# ----------------------------------------------------------------------------------------------
# Tensor parts (lazy MLX)
# ----------------------------------------------------------------------------------------------

def rowwise_logit(x, w, b):
    """f32 ``sum(x * w, -1) + b`` for x [..., d], w [d] (the neural gate's arithmetic)."""
    import mlx.core as mx
    return mx.sum(x.astype(mx.float32) * w.astype(mx.float32), axis=-1) + b


class ProbeGate:
    """``sigmoid(x . w + b) > threshold`` on the MLP input of the read layer."""

    def __init__(self, w, b, threshold):
        import math
        self.w = w
        self.b = float(b)
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("probe threshold must be in (0, 1)")
        self.threshold = float(threshold)
        self.logit_threshold = math.log(self.threshold / (1.0 - self.threshold))

    def open(self, x):
        return rowwise_logit(x, self.w, self.b) > self.logit_threshold


class FactTable:
    """Nearest-key lookup over unit-normalised neuron activations; values are residual deltas."""

    def __init__(self, gate_rows, up_rows, keys, values, null_rows, tau, margin, activation, center=None):
        import mlx.core as mx
        # Optional key centre (the mean recogniser activation the builder removed before
        # normalising): without it one shared relation component dominates every cosine.
        self.center = None if center is None else center.astype(mx.float32)
        if up_rows is None:                                # residual recogniser: [k] dimension indices
            self.gate_rows = gate_rows.astype(mx.int32)
            self.up_rows = None
        else:                                              # neuron recogniser: [k, d] host MLP rows
            self.gate_rows = gate_rows.astype(mx.float32)
            self.up_rows = up_rows.astype(mx.float32)
        self.keys = keys.astype(mx.float32)                # [n, k], unit rows
        self.values = values                               # [n, d]
        self.null = null_rows                              # [n] bool
        self.tau, self.margin = float(tau), float(margin)
        self.activation = activation

    def key_of(self, x):
        import mlx.core as mx
        xf = x.astype(mx.float32)
        if self.up_rows is None:
            # residual recogniser: the selected MLP-input dimensions (gate_rows holds their indices)
            z = xf[..., self.gate_rows]
        else:
            z = self.activation(xf @ self.gate_rows.T) * (xf @ self.up_rows.T)
        if self.center is not None:
            z = z - self.center
        return z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-6)

    def lookup(self, x):
        """``(open [..], index [..], best [..], second [..])`` per row of x."""
        import mlx.core as mx
        sims = self.key_of(x) @ self.keys.T
        order = mx.argsort(-sims, axis=-1)
        idx = order[..., 0]
        best = mx.take_along_axis(sims, idx[..., None], axis=-1)[..., 0]
        second = (mx.take_along_axis(sims, order[..., 1:2], axis=-1)[..., 0]
                  if sims.shape[-1] > 1 else mx.full(best.shape, -1.0))
        is_null = self.null[idx]
        opened = (best >= self.tau) & ((best - second) >= self.margin) & mx.logical_not(is_null)
        return opened, idx, best, second

    def delta(self, x):
        import mlx.core as mx
        opened, idx, best, second = self.lookup(x)
        d = self.values[idx].astype(mx.float32) * opened[..., None].astype(mx.float32)
        return d, opened, idx


# ----------------------------------------------------------------------------------------------
# Tile set
# ----------------------------------------------------------------------------------------------

class _Tile:
    def __init__(self, spec, tensors, activation):
        import mlx.core as mx
        self.id, self.kind, self.mode = spec["id"], spec["kind"], spec["write_mode"]
        self.parser = spec.get("parser", PARSER_ID) if self.kind == "calculator" else None
        self.arithmetic_backend = spec.get("arithmetic_backend", "gpu" if self.parser in (PARSER_V3, PARSER_V4) else "cpu")
        self.read_layer, self.write_layer = int(spec["read_layer"]), int(spec["write_layer"])
        self.read_point = spec.get("read_point", "mlp_input")
        self.spec = spec
        p = spec["id"] + "."
        if self.kind == "calculator":
            g = spec["gate"]
            self.gate = None if self.parser==PARSER_V5 else ProbeGate(tensors[p + "probe_w"], g["bias"], g["threshold"])
            # push (research only): unit directions toward each digit token at the write layer
            self.push_dirs = tensors.get(p + "push_dirs")
            self.push_scale = float(spec.get("push_scale", 0.0))
        else:
            t = spec["table"]
            kind = t.get("recogniser", "neurons")
            if kind not in ("neurons", "residual_dims"):
                raise ValueError(f"fact table {self.id}: unknown recogniser {kind!r}")
            rec = (tensors[p + "gate_rows"], tensors[p + "up_rows"]) if kind == "neurons" else (tensors[p + "dims"], None)
            self.table = FactTable(rec[0], rec[1], tensors[p + "keys"],
                                   tensors[p + "values"], tensors[p + "null"].astype(mx.bool_),
                                   t["tau"], t["margin"], activation, center=tensors.get(p + "center"))
            self.rows = int(tensors[p + "keys"].shape[0])


class DetTileSet:
    """Installed deterministic tiles: per-forward read stash, writes, counters, receipts."""

    def __init__(self, manifest, tensors, activation, *, route="ordinary", allow_research_modes=False):
        self.manifest = manifest
        self.identity = tile_set_identity(manifest)
        resident_external = route == "external_draft" and resident_external_manifest(manifest)
        if route not in SPECULATIVE_ROUTES + ("ordinary",) and not resident_external:
            raise DetTileRefused(f"deterministic tiles run on the ordinary route only, not {route!r}")
        self.route = route
        self.tiles = [_Tile(s, tensors, activation) for s in manifest["tiles"]]
        # Position-matched calculators only: every gate they read is matched to its history.
        matched = all(t.kind == "calculator" and t.read_point == "layer_output_capture" for t in self.tiles)
        if resident_external:
            # The resident external-draft verifier is admitted for exactly that route.
            self.served_routes = ("external_draft",)
        else:
            self.served_routes = ("ordinary",) + (SPECULATIVE_ROUTES if matched else ())
        if route not in self.served_routes:
            raise DetTileRefused(f"on the {route} route only calculators reading layer_output_capture "
                                 "(position-matched gates) are served")
        for t in self.tiles:
            if not allow_research_modes and t.mode not in SERVING_WRITE_MODES[t.kind]:
                raise DetTileRefused(f"tile {t.id}: write mode {t.mode!r} is research-only "
                                     f"(serving modes {SERVING_WRITE_MODES[t.kind]})")
        self.stash = {}            # tile id -> last forward's read result (lazy arrays)
        self.push_rows = {}        # tile id -> [B, d] pending push vectors (research)
        self.push_gated = {}       # tile id -> bool (research)
        self.forward_ids = {}      # tile id -> count of forwards that read its gate
        self.stash_shape = {}      # tile id -> (rows, positions) of the last gate read
        self.enabled = True
        self.lock = threading.Lock()
        self.counters = {"forwards": 0, "fact_open_rows_pending": [], "fact_open_rows": 0, "fact_rows": 0,
                         "calc": _calc_counters()}
        # Calculator processors count into their own (verify-ledger restored) dicts, kept here by
        # processor uid so per-request receipts and the process totals read settled values.
        # Resident (device) processors still fold into counters["calc"] at a receipt.
        self.proc_counters = collections.OrderedDict()
        self._proc_ids = itertools.count(1)
        self.live = __import__("weakref").WeakSet()   # live (non-probe) calculator processors
        # layer_output_capture reads: tile id -> recent forwards {ids, gate, host}
        self.out_stash = {}
        self.out_forwards = 0

    def check_route(self, route):
        """Fail closed unless the serving route is one this tile set was admitted for."""
        if route not in self.served_routes:
            raise DetTileRefused(f"deterministic tiles admitted for {', '.join(self.served_routes)} "
                                 f"(installed for {self.route}), received {route}")

    # --- layers -------------------------------------------------------------------------------
    def layers_used(self):
        """Layers whose MLP is wrapped (mlp_input reads and writes)."""
        return sorted({t.read_layer for t in self.tiles if t.read_point == "mlp_input"}
                      | {t.write_layer for t in self.tiles if t.read_point == "mlp_input"})

    def output_read_layers(self):
        """Layers whose output an adapter must hand to :meth:`on_layer_output`."""
        return sorted({t.read_layer for t in self.tiles if t.read_point == "layer_output_capture"})

    def on_layer_output(self, layer, hidden, input_ids, capture):
        """Stash calculator gates over every position of a trunk forward (layer_output_capture).

        Nothing is computed while no calculator processor is live (every request has it off), so
        such forwards run the host's arithmetic and nothing else. ``capture`` maps the layer output
        to the probe's input space. A forward without token ids (embeddings input) stashes nothing.
        """
        if not self.enabled or not len(self.live) or input_ids is None:
            return
        import mlx.core as mx
        for t in self.tiles:
            if t.kind != "calculator" or t.read_point != "layer_output_capture" or t.read_layer != layer:
                continue
            gate = t.gate.open(capture(hidden))
            if gate.ndim != 2 or tuple(gate.shape) != tuple(input_ids.shape[:2]):
                continue
            # Scheduled with the forward (no sync): once evaluated, the [B, T] bools no longer pin
            # the layer output the way an unevaluated graph would.
            try:
                mx.async_eval(gate)
            except ValueError:
                pass            # inside a graph transformation (research grads): leave it lazy
            ring = self.out_stash.setdefault(t.id, collections.deque(maxlen=STASH_FORWARDS))
            ring.appendleft({"ids": input_ids, "gate": gate, "host": None})
            self.out_forwards += 1

    def calc_gate_at(self, tile_id, history, row=None):
        """``(gate, reason)`` for the stashed position whose inputs end ``history`` (one sync).

        The newest forward with a match decides: a position whose input token is ``history[-1]``
        and whose preceding forward inputs (up to MATCH_WINDOW) equal the history before it. With
        a bound ``row`` only that row is searched. Several matches with different gates (rows or
        positions with the same recent tokens) fail closed, as does no match.
        """
        import mlx.core as mx
        if not history:
            return None, "gate_unmatched"
        last = int(history[-1])
        ring = list(self.out_stash.get(tile_id, ()))
        if row is not None:
            # A bound row names a row of the newest forward only (older forwards ordered other lanes).
            ring = ring[:1]
        for entry in ring:
            if entry["host"] is None:
                mx.eval(entry["ids"], entry["gate"])
                entry["host"] = (entry["ids"].tolist(), entry["gate"].tolist())
            ids, gate = entry["host"]
            if row is not None and row >= len(ids):
                return None, "row_unbound"
            rows = range(len(ids)) if row is None else (row,)
            found = set()
            for r in rows:
                seq = ids[r]
                for j, tok in enumerate(seq):
                    if int(tok) != last:
                        continue
                    w = min(j, MATCH_WINDOW, len(history) - 1)
                    if w and [int(x) for x in seq[j - w:j]] != [int(x) for x in history[-1 - w:-1]]:
                        continue
                    found.add(bool(gate[r][j]))
            if len(found) == 1:
                return found.pop(), None
            if len(found) > 1:
                return None, "gate_ambiguous"
        return None, "gate_unmatched"

    def new_processor_counters(self):
        uid = next(self._proc_ids)
        c = _calc_counters()
        with self.lock:
            self.proc_counters[uid] = c
            while len(self.proc_counters) > 4096:
                _, old = self.proc_counters.popitem(last=False)
                _merge_calc(self.counters["calc"], old)
        return uid, c

    def calc_totals(self):
        with self.lock:
            total = json.loads(json.dumps(self.counters["calc"]))
            for c in self.proc_counters.values():
                _merge_calc(total, c)
        return total

    def request_receipt(self, uids):
        """Settled calculator counters of one request's processors (host only)."""
        total = _calc_counters()
        with self.lock:
            for uid in uids or ():
                c = self.proc_counters.get(uid)
                if c is not None:
                    _merge_calc(total, c)
        return total

    def on_mlp(self, layer, x, y):
        """Apply every tile that reads or writes at ``layer``; returns the (possibly changed) y."""
        import mlx.core as mx
        if not self.enabled:
            return y
        for t in self.tiles:
            if t.read_layer == layer:
                if t.kind == "calculator":
                    # Resident syntax gating does not use a model-specific probe.
                    # Its actuation gate opens in the device parser/VM only after
                    # complete validation. Capture shape and forward binding here.
                    self.stash[t.id] = mx.zeros(x.shape[:-1],dtype=mx.bool_) if t.gate is None else t.gate.open(x)
                    self.forward_ids[t.id] = self.forward_ids.get(t.id, 0) + 1
                    self.stash_shape[t.id] = (int(x.shape[0]), int(x.shape[1]))
                else:
                    self.stash[t.id] = x
            if t.write_layer == layer:
                if t.kind == "fact_table":
                    xr = self.stash.pop(t.id)
                    if xr.shape[1] != y.shape[1]:
                        # prefill tail skip: the write layer computed only the last rows
                        xr = xr[:, -y.shape[1]:]
                    d, opened, _ = t.table.delta(xr)
                    y = (y.astype(mx.float32) + d).astype(y.dtype)
                    with self.lock:
                        self.counters["fact_rows"] += int(opened.size)
                        self.counters["fact_open_rows_pending"].append(mx.sum(opened))
                elif t.mode == "push" and t.id in self.push_rows:
                    vec = self.push_rows[t.id]                          # [B, d]
                    # The plan's first token is gated in-graph by this forward's probe; tokens after
                    # it continue a committed plan (research loop sets push_gated False).
                    gate = self.stash.get(t.id) if self.push_gated.get(t.id, True) else None
                    push = vec[:, None, :] * gate[..., -1:, None].astype(mx.float32) if gate is not None else vec[:, None, :]
                    pad = mx.zeros((y.shape[0], y.shape[1] - 1, y.shape[2]), dtype=mx.float32)
                    y = (y.astype(mx.float32) + mx.concatenate([pad, push], axis=1)).astype(y.dtype)
        if self.layers_used() and layer == max(self.layers_used()):
            with self.lock:
                self.counters["forwards"] += 1
        return y

    def mlp_hook(self, layer, x, y):
        """Research hook with the ``(layer, x_in, y_out) -> y`` signature of research.amoeba.donor."""
        return self.on_mlp(layer, x, y)

    def calc_gate_row(self, tile_id, row, bound_forward=None):
        """Host bool: the calculator gate at the last position of batch row ``row`` (one sync).

        Fail closed (returns None) unless the row is unambiguous: a one-row forward (any length:
        the last position is the one being sampled), or a decode-shaped forward (one position per
        row) whose row the generator bound on this same forward (``bound_forward``). Generators that
        do not bind rows therefore never fire the calculator at width > 1.
        """
        g = self.stash.get(tile_id)
        if g is None:
            return None
        rows, positions = self.stash_shape.get(tile_id, (None, None))
        if rows != 1 and (positions != 1 or bound_forward != self.forward_ids.get(tile_id) or row >= rows):
            return None
        return bool(g[0 if rows == 1 else row, -1].item())

    # --- receipts -----------------------------------------------------------------------------
    def host_counters(self):
        """Process counters without a device read (fact-table opens not yet folded are left out)."""
        with self.lock:
            c = json.loads(json.dumps({k: v for k, v in self.counters.items()
                                       if k not in ("fact_open_rows_pending", "calc")}))
        c["calc"] = self.calc_totals()
        c["layer_output_forwards"] = self.out_forwards
        return c

    def receipt(self):
        import mlx.core as mx
        with self.lock:
            if self.__dict__.get('_resident_states'):
                from .amoeba_calc_resident import collect_counters
                collect_counters(self)
            pend = self.counters["fact_open_rows_pending"]
            if pend:
                total = mx.sum(mx.stack(pend))
                self.counters["fact_open_rows"] += int(total.item())
                self.counters["fact_open_rows_pending"] = []
            c = json.loads(json.dumps({k: v for k, v in self.counters.items()
                                       if k not in ("fact_open_rows_pending", "calc")}))
        c["calc"] = self.calc_totals()
        if self.output_read_layers():
            c["layer_output_forwards"] = self.out_forwards
        return {"schema": RECEIPT_SCHEMA, **self.identity, "route": self.route,
                "qualified": False, "qualification": "unqualified", "implemented": True, "selected": True,
                # observed_used: a tile wrote (a calculator plan or an open fact row), not merely ran.
                "observed_used": bool(c["calc"]["plans"] or c["fact_open_rows"]),
                "observed_ran": c["forwards"] > 0 or c.get("layer_output_forwards", 0) > 0,
                "counters_scope": "process_cumulative", "counters": c,
                **({'speculative_protocol':'resident-calculator-exact-verify-v1',
                    'tree_verifier':'selected_path_rowwise',
                    'progressive_private_branches':False} if self.route=='external_draft' else {})}


def _calc_counters():
    return {"cue_seen": 0, "gate_open": 0, "gate_closed": 0, "parse_refused": {},
            "plans": 0, "tokens_written": 0, "plan_aborted": 0, "plan_executions": {}}


def _assign_in_place(d, values):
    """Make ``d`` equal ``values`` without an empty moment (readers hold the same lock)."""
    for k in [k for k in d if k not in values]:
        del d[k]
    d.update(values)


def _merge_calc(into, c):
    # list(): the generator adds keys to a processor's dict without the lock (a reader snapshots).
    for k, v in list(c.items()):
        if isinstance(v, dict):
            d = into.setdefault(k, {})
            for k2, v2 in list(v.items()):
                d[k2] = d.get(k2, 0) + v2
        else:
            into[k] = into.get(k, 0) + v


def make_det_mlp_class():
    import mlx.nn as nn

    class DetTileMLP(nn.Module):
        """Wraps a layer's MLP (stock or amoeba-tiled): ``y = inner(x)``, then the tiles at this layer."""

        def __init__(self, inner, layer, tileset):
            super().__init__()
            self.inner = inner
            self._layer = layer
            self._tiles = tileset        # plain attribute: never a parameter

        def __call__(self, x, *args, **kwargs):
            y = self.inner(x, *args, **kwargs)
            return self._tiles.on_mlp(self._layer, x, y)

    return DetTileMLP


def load_tensors(manifest):
    import mlx.core as mx
    tensors = {}
    for name in manifest["files"]:
        if name.endswith(".safetensors"):
            tensors.update(mx.load(str(Path(manifest["_dir"]) / name)))
    return tensors


def build(manifest_path, activation, *, source_fingerprint, layers_count, route="ordinary",
          allow_research_modes=False):
    """Load and check a manifest; the DetTileSet without installing it (research hook path)."""
    manifest = load_manifest(manifest_path)
    if manifest["source"]["fingerprint"] != source_fingerprint:
        raise ValueError(f"deterministic tiles: source fingerprint {manifest['source']['fingerprint']} "
                         f"does not match the loaded checkpoint {source_fingerprint}")
    if int(manifest["layers"]) != int(layers_count):
        raise ValueError(f"deterministic tiles: manifest has {manifest['layers']} layers, model has {layers_count}")
    tensors = load_tensors(manifest)
    return DetTileSet(manifest, tensors, activation, route=route, allow_research_modes=allow_research_modes)


def install(layers, manifest_path, declaration, *, source_fingerprint, route="ordinary"):
    """Wrap the MLP of every layer a tile reads or writes. Returns the DetTileSet.

    Install after the neural tiles (the wrapper wraps whatever MLP the layer holds) and after any
    projection law installer: the wrapper adds no projection, so no law has to cover it.
    """
    attr = declaration["mlp"]
    tiles = build(manifest_path, declaration["activation_fn"], source_fingerprint=source_fingerprint,
                  layers_count=len(layers), route=route)
    cls = make_det_mlp_class()
    for l in tiles.layers_used():
        inner = getattr(layers[l], attr)
        if getattr(inner, "amoeba_det_wrapped", False):
            raise ValueError(f"deterministic tiles: layer {l} is already wrapped")
        mod = cls(inner, l, tiles)
        mod.amoeba_det_wrapped = True
        setattr(layers[l], attr, mod)
    return tiles


# ----------------------------------------------------------------------------------------------
# Calculator write: logits processor (force / constrain)
# ----------------------------------------------------------------------------------------------

def _calculator_tokenizer(tokenizer):
    """Unwrap the streaming wrapper while preserving the HF vocabulary API."""
    inner = getattr(tokenizer, "_tokenizer", None)
    if getattr(type(inner), "__len__", None) is not None:
        return inner
    return tokenizer


class CalculatorProcessor:
    """Per-lane logits processor for one calculator tile.

    The generator binds the lane's batch row before each call (``bind_row``). With no plan, the
    processor looks for a complete expression in the recent text (cheap, host-side); only then does
    it read the probe gate for its row; if both agree and the expression evaluates exactly, it plans
    the result text and writes it token by token. ``force`` pins each step to the planned token;
    ``constrain`` allows any token whose text keeps the emitted text a prefix of the result.
    """

    def __init__(self, tileset, tile_id, tokenizer, *, prompt_length=0, context_tokens=CONTEXT_TOKENS,
                 gate_override=None, mode=None, request_text=None):
        self.tiles = tileset
        self.tile = next(t for t in tileset.tiles if t.id == tile_id)
        if self.tile.kind != "calculator":
            raise ValueError(f"{tile_id} is not a calculator tile")
        # ``mode`` overrides the manifest's write mode (research A/B of force vs constrain).
        self.mode = mode or self.tile.mode
        if self.mode not in ("force", "constrain"):
            raise DetTileRefused(f"calculator write mode {self.mode!r} is not a logits processor")
        self.tok = _calculator_tokenizer(tokenizer)
        self.prompt_length = prompt_length
        self.context_tokens = int(self.tile.spec.get("context_tokens", 256 if self.tile.parser == PARSER_V4 else context_tokens))
        self.row = 0
        self._bound_forward = None
        self._closing = 0          # steps left during which no digit may extend the result
        self.plan = None           # list of token ids (force) or target string (constrain)
        self.emitted = ""
        self.events = []
        # Controls: a callable (text) -> bool replacing the probe (random-trigger arm), or None.
        self.gate_override = gate_override
        self._vocab_text = None
        self._gpu_cue_window = None
        self._first_call = True
        self._literal_request = None
        from .amoeba_calc_requests import requests_verbatim_copy

        self._verbatim_copy = requests_verbatim_copy(request_text)
        if not self._verbatim_copy and self.tile.spec.get('complete_literal_requests',False) and request_text is not None:
            if self.mode!='force':raise DetTileRefused('complete literal requests require force mode')
            from .amoeba_calc_requests import parse_request
            self._literal_request=parse_request(request_text)
        if not self._verbatim_copy and self.tile.spec.get("trigger_prefilter", "off") == "gpu":
            from .amoeba_calc_trigger import GPUCueWindow
            self._gpu_cue_window = GPUCueWindow(self.tok, allow_symbolic=self.tile.parser == PARSER_V4,
                                              allow_counting=self.tile.spec.get('complete_literal_requests',False))
        self._row_bound = False
        self._start = 0
        self.shadow = False        # a draft probe's disposable copy: counts nothing, never live
        self.uid, self.counts = tileset.new_processor_counters()
        tileset.live.add(self)

    def bind_row(self, row):
        self.row = int(row)
        self._row_bound = True
        self._bound_forward = self.tiles.forward_ids.get(self.tile.id)

    # --- speculative routes ---------------------------------------------------------------------
    # The GPU cue window (trigger_prefilter) is device state this ledger does not rewind: load_manifest
    # refuses it on position-matched (speculative-capable) tiles, the only ones verify rewinds.
    _STATE = ("plan", "emitted", "_start", "_closing", "row", "_row_bound", "_bound_forward",
              "_first_call", "_literal_request")

    def _apply_pending_restore(self):
        restored = self.__dict__.pop("_restore_counts", None)
        if restored is not None:
            # After a recovery restore: counters return to the snapshot (the rows after it are
            # replayed and recounted), in place, under the tile set's lock.
            with self.tiles.lock:
                _assign_in_place(self.counts, restored)

    def verify_ledger(self):
        """Snapshot of what a verify row can change (``processor_probe.VerifyWindow``)."""
        self._apply_pending_restore()
        return ({k: copy.copy(getattr(self, k)) for k in self._STATE}, len(self.events),
                json.loads(json.dumps(self.counts)))

    def restore_verify_ledger(self, ledger):
        state, n_events, counts = ledger
        for k, v in state.items():
            setattr(self, k, copy.copy(v))
        del self.events[n_events:]
        # In place (the tile set reads this dict by processor uid), under its lock.
        with self.tiles.lock:
            _assign_in_place(self.counts, json.loads(json.dumps(counts)))

    def _clone(self, *, shadow):
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        for k in self._STATE:
            setattr(new, k, copy.copy(getattr(self, k)))
        new.events = list(self.events)
        if shadow:
            new.shadow = True
            new.counts = _calc_counters()
        return new

    def probe(self, tokens, logits):
        """Draft scoring through a disposable copy (``processor_probe`` protocol): it continues an
        active plan (drafts propose the forced tokens) and starts one only where the stashed gate
        matches its history. Nothing of it reaches this processor or the counters."""
        return self._clone(shadow=True)(tokens, logits)

    def __deepcopy__(self, memo):
        # Recovery snapshots: the tile set, tokenizer and counter dict are shared, plan state is
        # copied, and the copy carries its counters' values: when a restore installs the copy's
        # state, the first call puts the counters back to them. A live copy keeps the reader on.
        new = self._clone(shadow=self.shadow)
        if not self.shadow:
            # A copy of a snapshot keeps the snapshot's values, not the live dict's.
            pending = self.__dict__.get("_restore_counts")
            new._restore_counts = json.loads(json.dumps(pending if pending is not None else self.counts))
            self.tiles.live.add(new)
        memo[id(self)] = new
        return new

    def _decode(self, ids):
        return self.tok.decode(list(ids))

    # Finite, so a later hard mask (thinking guard, structured output, min_tokens) that keeps only
    # other tokens leaves a finite row rather than an all -inf one.
    MASK = -1.0e9

    def _mask_to(self, logits, allowed):
        import mlx.core as mx
        import numpy as np
        V = logits.shape[-1]
        m = np.full((V,), self.MASK, dtype=np.float32)
        m[list(allowed)] = 0.0
        return logits + mx.array(m)[None]

    def _banned_upstream(self, logits, ids):
        """True when an earlier processor already removed every token in ``ids`` (one sync)."""
        import mlx.core as mx
        vals = logits[0, mx.array(list(ids))]
        return bool(mx.all(vals <= self.MASK / 2).item())

    def _ban(self, logits, banned):
        import mlx.core as mx
        import numpy as np
        m = np.zeros((logits.shape[-1],), dtype=np.float32)
        m[[i for i in banned if i < logits.shape[-1]]] = self.MASK
        return logits + mx.array(m)[None]

    def _digit_start_ids(self):
        return self._vocab()["digit_starts"]

    def _result_boundary_ids(self):
        ids = self._digit_start_ids()
        if self.tile.parser == PARSER_V4:
            ids = ids + self._vocab()["arithmetic_extensions"]
        return ids

    def _digit_vocab(self):
        return self._vocab()["text"]

    def _vocab(self):
        """Number-text tokens of the tokenizer, cached on the tile set (one vocabulary scan per process)."""
        cache = self.tiles.__dict__.setdefault("_vocab_cache", {})
        key = (id(self.tok), len(self.tok))
        if key not in cache:
            ok = set("0123456789.-,$ ")
            vt = {}
            arithmetic_extensions = []
            for i in range(len(self.tok)):
                try:
                    s = self.tok.decode([i])
                except Exception:  # noqa: BLE001
                    continue
                # An operator after the completed RHS would change its value. Only
                # spaces/tabs are transparent: a new line may start a new formula.
                prefix = s.lstrip(" \t")
                if prefix.startswith(("+", "-", "*", "/", "^", "×", "÷", "−", r"\times", r"\cdot", r"\div")):
                    arithmetic_extensions.append(i)
                if s and set(s) <= ok:
                    vt[i] = s
            cache[key] = {"text": vt, "arithmetic_extensions": arithmetic_extensions, "digit_starts": [i for i, t in vt.items() if t[:1].isdigit()],
                          "point": [i for i, t in vt.items() if t in (".", ",")]}
        return cache[key]

    def __call__(self, tokens, logits):
        # A row binding holds for this call only: a lane moved between generators (ordinary
        # step, then self-MTP) must never search the row of an earlier batch. Taken before any
        # early return, so a skipped call never leaves a stale binding for the next one.
        bound_row = self.row if self._row_bound else None
        self._row_bound = False
        self._apply_pending_restore()
        # Per-processor counters (settled through the verify ledger; summed by calc_totals).
        c = self.counts
        if self._verbatim_copy:
            return logits
        import mlx.core as mx
        if self._first_call and self._literal_request is not None:
            request=self._literal_request;self._literal_request=None
            # Strict whole-request grammar is a symbolic gate. Still require a
            # current lane binding, and honour research/control gate overrides.
            bound=self.tiles.calc_gate_row(self.tile.id,self.row,self._bound_forward)
            eos=getattr(self.tok,'eos_token_id',None)
            if bound is not None and eos is not None and (self.gate_override is None or self.gate_override('literal_request')):
                from .amoeba_calc_requests import evaluate as request_answer
                answer,info=request_answer(request,self.tile.arithmetic_backend)
                if answer is not None:
                    ids=tokens.tolist() if hasattr(tokens,'tolist') else list(tokens)
                    while ids and isinstance(ids[0],list):ids=[i for row in ids for i in row]
                    c['literal_request_open']=c.get('literal_request_open',0)+1
                    c['plans']+=1;c['gate_open']+=1;c['cue_seen']+=1
                    execution = info.get('arithmetic_execution', 'unspecified')
                    c['plan_executions'][execution] = c['plan_executions'].get(execution, 0) + 1
                    self.events.append(dict(position=len(ids),**info,answer=answer,result=answer,syntax='complete_literal_request'))
                    # The entire request is solved; end its answer so a later
                    # neural continuation cannot round or replace the exact value.
                    self.plan=list(self.tok.encode(answer,add_special_tokens=False))+[int(eos)]
                    self._start=len(ids);self._first_call=False
                    return self._continue(ids,logits)
                c['request_refused']=c.get('request_refused',0)+1
        # Maintain the fixed device window even while an answer plan is active. Only
        # read its tiny trigger when a fresh parse could be needed, never its history.
        if self._gpu_cue_window is not None:
            gate = self.tiles.stash.get(self.tile.id)
            shape = self.tiles.stash_shape.get(self.tile.id)
            if gate is not None and shape and 0 <= self.row < shape[0]:
                trigger = self._gpu_cue_window.update(tokens, gate, self.row * shape[1] + shape[1] - 1)
                if not self._first_call and self.plan is None and not self._closing:
                    c["prefilter_checks"] = c.get("prefilter_checks", 0) + 1
                    if not bool(trigger.item()):
                        c["prefilter_skips"] = c.get("prefilter_skips", 0) + 1
                        return logits
        self._first_call = False
        c["history_materializations"] = c.get("history_materializations", 0) + 1
        ids = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        while ids and isinstance(ids[0], list):
            ids = [i for row in ids for i in row]
        if self._closing:
            # After the result: a "." or "," may follow (end of sentence, list), but no digit may
            # extend the number on the next step either ("17" must not become "17.5").
            self._closing = 0
            last = self._decode(ids[-1:]) if ids else ""
            if last in (".", ","):
                return self._ban(logits, self._digit_start_ids())
            if self.tile.parser == PARSER_V4 and last and not last.strip(" \t"):
                self._closing = 1
                return self._ban(logits, self._result_boundary_ids())
        if self.plan is not None:
            return self._continue(ids, logits)
        text = self._decode(ids[-self.context_tokens:])
        answer=None;info=None
        if self.tile.spec.get('complete_literal_requests',False) and text.rstrip().endswith('='):
            from . import amoeba_counting as counting
            line=text.rstrip()[:-1].strip().split('\n')[-1].strip()
            request=counting.parse_request(line)
            if request is not None:
                answer,info=counting.evaluate(request,self.tile.arithmetic_backend)
                if answer is not None:info=dict(**info,expr=line,result=answer,cued=True,syntax='literal_count')
        if info is None:
            answer, info = calculator_answer(text, truncated=len(ids) > self.context_tokens, parser=self.tile.parser, arithmetic_backend=self.tile.arithmetic_backend)
        if answer is not None and not info.get("cued", True) and self.prompt_length and len(ids) != self.prompt_length:
            # The question form ("What is 7 times 12?") only names the answer slot right after the
            # prompt; later it would re-read the gate on every step for any trailing question.
            return logits
        if answer is None:
            if info["refused"] != "no_expression":
                c["parse_refused"][info["refused"]] = c["parse_refused"].get(info["refused"], 0) + 1
            return logits
        c["cue_seen"] += 1
        if self.gate_override is not None:
            opened = self.gate_override(text)
        elif self.tile.read_point == "layer_output_capture":
            opened, why = self.tiles.calc_gate_at(self.tile.id, ids, bound_row)
            if opened is None:
                c[why] = c.get(why, 0) + 1
                return logits
        else:
            opened = self.tiles.calc_gate_row(self.tile.id, self.row, self._bound_forward)
            if opened is None:
                c["row_unbound"] = c.get("row_unbound", 0) + 1
                return logits
        # V3's exact two-row multiplication layout identifies its own answer slot.
        # The old neural probe was never trained on that separator. Preserve row binding
        # and explicit control overrides; only this complete grammar can open symbolically.
        if (not opened and self.gate_override is None and self.tile.parser in (PARSER_V3, PARSER_V4)
                and info.get("syntax") in ("latex_column", "latex_math", "literal_count")):
            opened = True
            key = {"latex_column":"latex_column_open","literal_count":"literal_count_open"}.get(info.get('syntax'),'latex_math_open')
            c[key] = c.get(key, 0) + 1
        if not opened:
            c["gate_closed"] += 1
            return logits
        c["gate_open"] += 1
        c["plans"] += 1
        execution = info.get('arithmetic_execution', 'unspecified')
        c['plan_executions'][execution] = c['plan_executions'].get(execution, 0) + 1
        self.events.append({"position": len(ids), **info, "answer": answer})
        if self.mode == "force":
            # Preserve the explicitly versioned v3 terminator. V4 leaves surrounding
            # units and reasoning to the model: forcing a newline changed correct GSM8K
            # continuations (e.g. minutes became hours) even when arithmetic was exact.
            if self.tile.parser == PARSER_V3 and info.get("cued") and info.get("syntax") != "latex_column":
                answer += "\n"
            self.plan = list(self.tok.encode(answer, add_special_tokens=False))
        else:
            self.plan = answer
        self.emitted = ""
        self._start = len(ids)
        return self._continue(ids, logits)

    def _continue(self, ids, logits):
        c = self.counts
        done = len(ids) - self._start
        if self.mode == "force":
            # Tokens emitted since the plan began must be the plan's prefix (they were forced).
            if ids[self._start:] != self.plan[:done]:
                c["plan_aborted"] += 1
                self.plan = None
                return logits
            if done >= len(self.plan):
                # The step after the result: no digit may extend the number ("2090" -> "20900").
                self.plan = None
                self._closing = 1
                c["closed"] = c.get("closed", 0) + 1
                return self._ban(logits, self._result_boundary_ids())
            if self._banned_upstream(logits, [self.plan[done]]):
                c["plan_aborted"] += 1
                c["aborted_upstream_mask"] = c.get("aborted_upstream_mask", 0) + 1
                self.plan = None
                return logits
            c["tokens_written"] += 1
            return self._mask_to(logits, [self.plan[done]])
        emitted = self._decode(ids[self._start:]) if done else ""
        target = self.plan
        if not target.startswith(emitted):
            c["plan_aborted"] += 1
            self.plan = None
            return logits
        rest = target[len(emitted):]
        if not rest:
            self.plan = None
            self._closing = 1
            c["closed"] = c.get("closed", 0) + 1
            return self._ban(logits, self._result_boundary_ids())
        allowed = [i for i, s in self._digit_vocab().items() if rest.startswith(s)]
        if not allowed or self._banned_upstream(logits, allowed):
            c["plan_aborted"] += 1
            self.plan = None
            return logits
        c["tokens_written"] += 1
        return self._mask_to(logits, allowed)


def write_manifest(out_dir, *, source, layers, tiles, tensors, rows_source=None, extra=None):
    """Write ``tiles.safetensors`` (+ optional ``rows.jsonl`` row provenance) and ``manifest.json``.

    ``tiles``: tile specs (id, kind, write_mode, read_layer, write_layer, gate/table fields); each
    tile's ``file`` is set here. ``tensors``: name -> mx.array, names prefixed ``<tile id>.``.
    ``rows_source``: list of dicts, one per stored table row (fact, relation, method, templates).
    """
    import mlx.core as mx
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"deterministic tiles: refusing to overwrite nonempty {out}")
    mx.save_safetensors(str(out / "tiles.safetensors"), dict(tensors))
    files = {"tiles.safetensors": file_sha256(out / "tiles.safetensors")}
    if rows_source is not None:
        (out / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows_source))
        files["rows.jsonl"] = file_sha256(out / "rows.jsonl")
    specs = [{**t, "file": "tiles.safetensors"} for t in tiles]
    manifest = {"schema": SCHEMA, "source": source, "layers": int(layers), "tiles": specs, "files": files,
                "qualification": "unqualified", **(extra or {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return out / "manifest.json"


# ----------------------------------------------------------------------------------------------
# Adapter helpers (model-agnostic; an adapter supplies its MLP declaration and tokenizer)
# ----------------------------------------------------------------------------------------------

# Legacy deterministic tiles refuse speculative routes. Resident calculators
# have a separately admitted external verifier contract below.
REFUSED_POLICY_KEYS = ("draft_model", "draft_models", "prompt_lookup", "target_verify_row_exact",
                       "q4_gate_three_rows", "tensorfold_prefill", "ast_graft_prompt_lookup",
                       "self_mtp_copy_draft", "mtp_ordinary_handoff")


def resident_external_manifest(manifest):
    """Only the device-validated calculator has an exact verify-state ledger."""
    tiles=manifest.get('tiles',())
    return bool(tiles) and all(t.get('kind')=='calculator' and t.get('parser')==PARSER_V5
        and t.get('write_mode')=='force' and t.get('arithmetic_backend')=='gpu'
        and t.get('gate',{}).get('kind')=='validated_syntax' for t in tiles)


def refuse_policy(manifest_path, policy, *, require_mtp=False, weight_streaming=None, allow_external=False):
    """Validate the tile/route combination before an adapter loads weights."""
    if type(manifest_path) is not str or not manifest_path:
        raise ValueError("amoeba_det_tiles must name a deterministic tile manifest path")
    if weight_streaming is not None:
        raise DetTileRefused("deterministic tiles cannot combine with weight streaming")
    if require_mtp:
        raise DetTileRefused("deterministic tiles run on the ordinary route only (native MTP requested)")
    external=bool(allow_external and policy.get('draft_model') and not policy.get('draft_models'))
    if external and not resident_external_manifest(load_manifest(manifest_path)):
        raise DetTileRefused('external draft decoding requires only resident GPU calculator tiles with exact verify ledgers')
    if external:
        for key in ('continuation_pool','continuation_strategy','ast_b1_tile_policy',
                    'ast_packed_pattern','ast_target_tree_oracle','ast_physical_target'):
            if policy.get(key):
                raise DetTileRefused(f'resident calculator has no verified processor-state contract for {key}')
    for key in REFUSED_POLICY_KEYS:
        if external and key=='draft_model':continue
        if policy.get(key):
            raise DetTileRefused(f"deterministic tiles refused with {key}: the ordinary route only")
    return 'external_draft' if external else 'ordinary'


def install_on_adapter(layers, manifest_path, declaration, *, source_fingerprint, host_tiles_sha256=None,
                       route="ordinary"):
    """Install, after every load-time projection law; checks the host the tiles were built on.

    A manifest built on a tiled host records that tile set's manifest digest
    (``host_tiles_manifest_sha256``); the live tile set must match it, and a manifest built on the
    full model refuses a tiled host (and vice versa), since the stored rows were solved for that
    host's downstream layers.
    """
    manifest = load_manifest(manifest_path)
    want = manifest.get("host_tiles_manifest_sha256")
    if want != host_tiles_sha256:
        raise ValueError(f"deterministic tiles were built on host tiles {want}, this route holds {host_tiles_sha256}")
    return install(layers, manifest_path, declaration, source_fingerprint=source_fingerprint, route=route)


REQUEST_FIELD = "det_tiles"


def request_toggles(tileset, request):
    """``{tile id: on}`` for one request from its ``det_tiles`` field (default: every tile on).

    ``det_tiles`` maps a tile id or a kind ("calculator") to a bool; a tile id wins over its kind.
    Unknown names, non-bool values, or turning on a tile the server did not load fail closed.
    """
    field = (request or {}).get(REQUEST_FIELD)
    if field is None:
        return {} if tileset is None else {t.id: True for t in tileset.tiles}
    if not isinstance(field, dict) or not all(isinstance(k, str) and type(v) is bool for k, v in field.items()):
        raise ValueError(f"{REQUEST_FIELD} must map tile ids or kinds to true/false")
    if tileset is None:
        on = sorted(k for k, v in field.items() if v)
        if on:
            raise DetTileRefused(f"{REQUEST_FIELD}: {', '.join(on)} requested on, but this server loaded "
                                 "no deterministic tiles")
        return {}
    ids, kinds = {t.id for t in tileset.tiles}, {t.kind for t in tileset.tiles}
    unknown = sorted(set(field) - ids - kinds)
    if unknown:
        raise DetTileRefused(f"{REQUEST_FIELD}: unknown tile(s) {', '.join(unknown)}; loaded "
                             f"{', '.join(sorted(ids))}")
    return {t.id: field.get(t.id, field.get(t.kind, True)) for t in tileset.tiles}


def processors_for(tileset, tokenizer, *, prompt_length=0, request=None):
    """One calculator processor per calculator tile the request leaves on (fresh per request).

    With ``request``, its ``det_tiles`` field selects tiles, and the host processors' counter ids
    are recorded in ``request["_mlx2_det_tile_uids"]`` for the request's receipt (resident device
    processors count into the process totals at a receipt instead).
    """
    if tileset is None:
        request_toggles(None, request)
        return ()
    toggles = request_toggles(tileset, request)
    tok = _calculator_tokenizer(tokenizer)
    from .amoeba_calc_requests import request_text_from_payload
    text = request_text_from_payload(request)
    from .amoeba_calc_resident import ResidentCalculatorProcessor
    if tileset.route=='external_draft':
        from .amoeba_calc_speculative import SpeculativeResidentCalculatorProcessor
        ResidentCalculatorProcessor=SpeculativeResidentCalculatorProcessor
    procs = tuple((ResidentCalculatorProcessor if t.parser==PARSER_V5 else CalculatorProcessor)(
                      tileset, t.id, tok, prompt_length=prompt_length, request_text=text)
                  for t in tileset.tiles if t.kind == "calculator" and toggles.get(t.id, True))
    if request is not None:
        request["_mlx2_det_tile_uids"] = (list(request.get("_mlx2_det_tile_uids") or [])
                                          + [p.uid for p in procs if hasattr(p, "uid")])
        request["_mlx2_det_tiles_on"] = {k: bool(v) for k, v in toggles.items()}
        if any(not hasattr(p, "uid") for p in procs):
            request["_mlx2_det_tile_resident"] = True
    return procs


def request_receipt(tileset, request):
    """The per-request det-tile receipt: what was on and what the calculator did (host only)."""
    if tileset is None:
        return None
    on = (request or {}).get("_mlx2_det_tiles_on")
    return {"schema": RECEIPT_SCHEMA, "loaded": [t.id for t in tileset.tiles],
            "requested": (request or {}).get(REQUEST_FIELD), "on": on,
            "qualification": "unqualified",
            # The loaded tile set joins the APCv2 namespace (numerics contract) for every request,
            # tile on or off: no prefix crosses tile sets or the base model.
            "apcv2_namespace": {"amoeba_det_tiles": tileset.identity["manifest_sha256"],
                                "applies_to": "every request while the tile set is loaded"},
            # Resident (device) processors have no per-request counters: they fold into the
            # tile set's process totals at a receipt, so a per-request zero would be untrue.
            **({"calculator": None,
                "calculator_scope": "process_cumulative_only (resident device counters)"}
               if (request or {}).get("_mlx2_det_tile_resident") else
               {"calculator": tileset.request_receipt((request or {}).get("_mlx2_det_tile_uids"))})}


def install_layer_output_reader(layers, tileset, capture):
    """Hand each output-read layer's output and input ids to ``tileset.on_layer_output``.

    The layer's class is swapped for a subclass whose ``__call__`` runs the original and then
    stashes the gate: same object, parameters and attributes, so fused paths that inspect the
    layer keep working. The layer's call signature must be ``(x, input_ids, ...)``. Paths that do
    not call the layer (packed mixed forwards) stash nothing, and the processor then fails closed.
    """
    for l in tileset.output_read_layers():
        layer = layers[l]
        if getattr(layer, "amoeba_det_reader", False):
            raise ValueError(f"deterministic tiles: layer {l} already has an output reader")
        base = type(layer)

        def __call__(self, x, input_ids, *args, _l=l, _base=base, **kwargs):
            out = _base.__call__(self, x, input_ids, *args, **kwargs)
            tileset.on_layer_output(_l, out, input_ids, capture)
            return out

        layer.__class__ = type(base.__name__, (base,), {"__call__": __call__, "amoeba_det_reader": True,
                                                       "__module__": base.__module__})
    return tileset


def numerics_contract_with_det_tiles(contract, tileset):
    if tileset is None:
        return contract
    return {**(contract or {}), "amoeba_det_tiles": tileset.identity}
