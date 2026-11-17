# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Bounded, original LaTeX arithmetic frontend; no TeX execution or MLX imports.

Parsing builds syntax only. ``evaluate_exact`` is an explicit CPU reference;
``to_deferred`` constructs backend IR without computing numerical answers.
See docs/research/2026-11-17-calculator-latex.md for the grammar and capabilities.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial

MAX_SOURCE = 4096
MAX_TOKENS = 1024
MAX_NODES = 512
MAX_DEPTH = 48
MAX_DIGITS = 60
MAX_EXPONENT = 64
MAX_TERMS = 128
MAX_FACTORIAL = 100
MAX_BINOMIAL = 256
CORE_OPS = frozenset({"literal", "+", "-", "*", "/", "^", "neg"})


class MathError(ValueError):
    """Stable machine-readable refusal, shared by syntax/domain/capability checks."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class MathExpr:
    op: str
    args: tuple[MathExpr, ...] = ()
    text: str | None = None


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str


_NUMBER = re.compile(r"(?:[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
_COMMAND = re.compile(r"\\([A-Za-z]+|.)", re.DOTALL)
_COMMANDS = {
    "frac": "frac", "dfrac": "frac", "tfrac": "frac", "cfrac": "frac",
    "binom": "binom", "dbinom": "binom", "tbinom": "binom",
    "sqrt": "root", "sum": "sum", "prod": "prod", "bmod": "mod",
    "times": "*", "cdot": "*", "div": "/", "left": "left", "right": "right",
    "{": "{", "}": "}", "lbrace": "{", "rbrace": "}",
    "lbrack": "[", "rbrack": "]", "lparen": "(", "rparen": ")",
}
_SPACING = {",", ":", ";", "!", " ", "quad", "qquad", "enspace", "thinspace",
            "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle"}
_OPEN = {"(": ")", "[": "]", "{": "}"}
_ATOMS = {"number", "name", "(", "[", "{", "left", "frac", "binom", "root", "sum", "prod"}


def _unwrap(text: str) -> str:
    text = text.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left):
            if len(text) < len(left) + len(right) or not text.endswith(right):
                raise MathError("unbalanced_math_delimiters")
            return text[len(left):-len(right)].strip()
    return text


def _tokens(text: str) -> list[_Token]:
    out = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char.isspace():
            pos += 1
            continue
        m = _NUMBER.match(text, pos)
        if m:
            # A stray/incorrectly grouped comma remains a foreign token, so the
            # complete parse refuses rather than changing an operand silently.
            raw = m.group().replace(',', '')
            # Canonicalization changes spelling only, not numerical value.
            raw = "0" + raw if raw.startswith(".") else raw
            parts = raw.split(".")
            if len(raw.replace(".", "").lstrip("0")) > MAX_DIGITS or (len(parts) == 2 and len(parts[1]) > MAX_DIGITS):
                raise MathError("too_large")
            out.append(_Token("number", raw))
            pos = m.end()
        elif char == "\\":
            m = _COMMAND.match(text, pos)
            if m is None:
                raise MathError("unknown_latex_command")
            command = m[1]
            pos = m.end()
            if command in _SPACING:
                continue
            if command not in _COMMANDS:
                raise MathError("unsupported_latex_command:" + command)
            out.append(_Token(_COMMANDS[command], command))
        elif char in "+-*/^!_={}()[]":
            out.append(_Token(char, char))
            pos += 1
        elif char in "×÷−":
            out.append(_Token({"×": "*", "÷": "/", "−": "-"}[char], char))
            pos += 1
        elif char.isascii() and char.isalpha():
            out.append(_Token("name", char))
            pos += 1
        else:
            raise MathError("unsupported_latex_character")
        if len(out) > MAX_TOKENS:
            raise MathError("token_bound")
    out.append(_Token("eof", ""))
    return out


class _Parser:
    def __init__(self, text: str):
        self.tokens = _tokens(text)
        self.i = self.nodes = self.depth = 0
        self.bound: list[str] = []

    def peek(self, kind: str) -> bool:
        return self.tokens[self.i].kind == kind

    def take(self, kind: str | None = None) -> _Token:
        tok = self.tokens[self.i]
        if kind is not None and tok.kind != kind:
            raise MathError("expected_" + kind)
        if tok.kind == "eof":
            raise MathError("incomplete_expression")
        self.i += 1
        return tok

    def node(self, op: str, *args: MathExpr, text: str | None = None) -> MathExpr:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise MathError("node_bound")
        return MathExpr(op, args, text)

    def expr(self) -> MathExpr:
        value = self.term()
        while self.peek("+") or self.peek("-"):
            op = self.take().kind
            value = self.node(op, value, self.term())
        return value

    def term(self) -> MathExpr:
        value = self.unary()
        division_seen = False
        while True:
            if self.tokens[self.i].kind in {"*", "/", "mod"}:
                op = self.take().kind
                division_seen |= op == "/"
                value = self.node(op, value, self.unary())
            elif self.tokens[self.i].kind in _ATOMS:
                # Adjacent numerals and mixed-number notation are ambiguous.
                numeral = value.op == "literal" or (value.op == "neg" and value.args[0].op == "literal")
                if division_seen or self.peek("number") or (numeral and self.peek("frac")):
                    raise MathError("ambiguous_implicit_multiplication")
                value = self.node("*", value, self.unary())
            else:
                return value

    def unary(self) -> MathExpr:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise MathError("depth_bound")
        try:
            if self.peek("+") or self.peek("-"):
                op = self.take().kind
                value = self.unary()
                return value if op == "+" else self.node("neg", value)
            value = self.atom()
            if self.peek("!"):
                self.take()
                value = self.node("factorial", value)
                if self.peek("!"):
                    raise MathError("unsupported_double_factorial")
            if self.peek("^"):
                self.take()
                if self.peek("{"):
                    exponent = self.group("{")
                elif self.peek("number") and len(self.tokens[self.i].text) == 1:
                    exponent = self.node("literal", text=self.take().text)
                elif self.peek("name"):
                    exponent = self.atom()
                else:
                    raise MathError("exponent_requires_braces")
                value = self.node("^", value, exponent)
                if self.peek("^"):
                    raise MathError("double_superscript")
            return value
        finally:
            self.depth -= 1

    def group(self, opening: str, sized: bool = False) -> MathExpr:
        self.take(opening)
        value = self.expr()
        if sized:
            self.take("right")
        self.take(_OPEN[opening])
        return value

    def argument(self) -> MathExpr:
        if not self.peek("{"):
            raise MathError("argument_requires_braces")
        return self.group("{")

    def atom(self) -> MathExpr:
        kind = self.tokens[self.i].kind
        if kind == "number":
            return self.node("literal", text=self.take().text)
        if kind == "name":
            name = self.take().text
            if name not in self.bound:
                raise MathError("unbound_variable")
            return self.node("variable", text=name)
        if kind in _OPEN:
            return self.group(kind)
        if kind == "left":
            self.take()
            opening = self.tokens[self.i].kind
            if opening not in _OPEN:
                raise MathError("unsupported_sized_delimiter")
            return self.group(opening, sized=True)
        if kind in {"frac", "binom"}:
            self.take()
            return self.node("/" if kind == "frac" else "binomial", self.argument(), self.argument())
        if kind == "root":
            self.take()
            degree = self.group("[") if self.peek("[") else self.node("literal", text="2")
            return self.node("root", self.argument(), degree)
        if kind in {"sum", "prod"}:
            self.take()
            self.take("_")
            self.take("{")
            name = self.take("name").text
            self.take("=")
            lower = self.expr()
            self.take("}")
            self.take("^")
            upper = self.argument()
            # Bounds are deliberately literal: no hidden evaluation before GPU dispatch.
            lo, hi = _literal_integer(lower), _literal_integer(upper)
            _check_terms(lo, hi)
            self.bound.append(name)
            try:
                # A term (not an additive expression) is the unbraced summand.
                body = self.term()
            finally:
                self.bound.pop()
            return self.node(kind, lower, upper, body, text=name)
        raise MathError("not_an_expression")


def parse_latex(text: str) -> MathExpr:
    """Parse ONE complete arithmetic formula, optionally wrapped in math delimiters.

    No prose, equality/result cue, arbitrary TeX macro, or free variable is accepted.
    No numerical operation is performed. Raises ``MathError`` on refusal.
    """
    if not isinstance(text, str):
        raise MathError("not_text")
    if len(text) > MAX_SOURCE:
        raise MathError("source_bound")
    parser = _Parser(_unwrap(text))
    result = parser.expr()
    if not parser.peek("eof"):
        raise MathError("trailing_latex_tokens")
    _validate(result)
    return result


def canonical_expression(expr: MathExpr) -> str:
    """Serialize a parsed AST compactly, preserving its full operation tree.

    Use ``len(canonical_expression(parse_latex(span)))`` for a semantic formula
    cap AFTER strict parsing and the independent raw/truncation checks. Styling,
    spacing, wrappers and sized delimiters disappear; literal digits and scoped
    loop indices do not. No numerical evaluation, macro erasure or prose/cue
    extraction occurs here. Parentheses preserve intermediate evaluation order,
    including the right operand of associative operations.
    """
    _validate(expr)
    precedence = {"+": 10, "-": 10, "*": 20, "/": 20, "mod": 20,
                  "neg": 30, "^": 40, "factorial": 50, "sum": 0, "prod": 0}

    def render(node: MathExpr, minimum: int = 0) -> str:
        op = node.op
        priority = precedence.get(op, 60)
        if op in {"literal", "variable"}:
            text = node.text
        elif op in {"+", "-", "*", "/", "mod"}:
            operator = r"\bmod " if op == "mod" else op
            # Equality on the right needs parentheses even for addition and
            # multiplication: intermediate digit/domain bounds are observable.
            text = render(node.args[0], priority) + operator + render(node.args[1], priority + 1)
        elif op == "neg":
            text = "-" + render(node.args[0], priority)
        elif op == "^":
            exponent = node.args[1]
            power = render(exponent)
            if not (exponent.op == "variable" or (exponent.op == "literal" and len(power) == 1)):
                power = "{" + power + "}"
            text = render(node.args[0], 50) + "^" + power
        elif op == "factorial":
            text = render(node.args[0], 60) + "!"
        elif op == "binomial":
            text = r"\binom{" + render(node.args[0]) + "}{" + render(node.args[1]) + "}"
        elif op == "root":
            degree = node.args[1]
            index = "" if degree.op == "literal" and degree.text == "2" else "[" + render(degree) + "]"
            text = r"\sqrt" + index + "{" + render(node.args[0]) + "}"
        elif op in {"sum", "prod"}:
            text = ("\\" + op + "_{" + node.text + "=" + render(node.args[0])
                    + "}^{" + render(node.args[1]) + "}{" + render(node.args[2]) + "}")
        else:
            raise MathError("unsupported_math_operation:" + op)
        return "(" + text + ")" if priority < minimum else text

    return render(expr)


def _literal_integer(expr: MathExpr) -> int:
    if expr.op == "literal" and expr.text is not None and expr.text.isascii() and expr.text.isdigit():
        return int(expr.text)
    if expr.op == "neg" and len(expr.args) == 1:
        return -_literal_integer(expr.args[0])
    raise MathError("bound_requires_integer_literal")


def _check_terms(lo: int, hi: int) -> None:
    if max(0, hi - lo + 1) > MAX_TERMS:
        raise MathError("term_bound")


def _validate(expr: MathExpr) -> None:
    count = 0
    arity = {"literal": 0, "variable": 0, "neg": 1, "factorial": 1,
             "+": 2, "-": 2, "*": 2, "/": 2, "^": 2, "mod": 2,
             "binomial": 2, "root": 2, "sum": 3, "prod": 3}

    def visit(node: MathExpr, depth: int) -> None:
        nonlocal count
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise MathError("expression_bound")
        if (not isinstance(node, MathExpr) or not isinstance(node.op, str)
                or node.op not in arity or not isinstance(node.args, tuple)
                or len(node.args) != arity[node.op]):
            raise MathError("invalid_math_ast")
        if node.op == "literal":
            if not isinstance(node.text, str) or _NUMBER.fullmatch(node.text) is None or len(node.text) > MAX_DIGITS * 2 + 2:
                raise MathError("invalid_math_literal")
            parts = node.text.split(".")
            if len(node.text.replace(".", "").lstrip("0")) > MAX_DIGITS or (len(parts) == 2 and len(parts[1]) > MAX_DIGITS):
                raise MathError("too_large")
        if node.op in {"variable", "sum", "prod"} and (not isinstance(node.text, str) or re.fullmatch("[A-Za-z]", node.text) is None):
            raise MathError("invalid_math_variable")
        for child in node.args:
            visit(child, depth + 1)

    visit(expr, 1)


def _bounded(value: Fraction) -> Fraction:
    if abs(value.numerator) >= 10 ** MAX_DIGITS or value.denominator >= 10 ** MAX_DIGITS:
        raise MathError("too_large")
    return value


def _integer(value: Fraction, reason: str) -> int:
    if value.denominator != 1:
        raise MathError(reason)
    return value.numerator


def _exact_root(value: int, degree: int) -> int:
    negative = value < 0
    if negative and degree % 2 == 0:
        raise MathError("root_domain")
    magnitude = abs(value)
    low, high = 0, 1 << ((magnitude.bit_length() + degree - 1) // degree)
    while low <= high:
        middle = (low + high) // 2
        powered = middle ** degree
        if powered == magnitude:
            return -middle if negative else middle
        if powered < magnitude:
            low = middle + 1
        else:
            high = middle - 1
    raise MathError("non_perfect_root")


def evaluate_exact(expr: MathExpr) -> Fraction:
    """Explicit CPU exact-rational reference; never called by ``to_deferred``.

    Bounds apply to every intermediate. Irrational roots and noninteger powers
    are refused, not rounded. ``0^0`` follows the calculator convention of 1.
    """
    _validate(expr)
    work = 0

    def run(node: MathExpr, env: dict[str, Fraction]) -> Fraction:
        nonlocal work
        work += 1
        if work > 8192:
            raise MathError("evaluation_work_bound")
        op = node.op
        if op == "literal":
            return _bounded(Fraction(node.text))
        if op == "variable":
            if node.text not in env:
                raise MathError("unbound_variable")
            return env[node.text]
        if op in {"sum", "prod"}:
            lo, hi = _literal_integer(node.args[0]), _literal_integer(node.args[1])
            _check_terms(lo, hi)
            value = Fraction(0 if op == "sum" else 1)
            for index in range(lo, hi + 1):
                term = run(node.args[2], {**env, node.text: Fraction(index)})
                value = _bounded(value + term if op == "sum" else value * term)
            return value
        args = [run(child, env) for child in node.args]
        a = args[0]
        if op == "neg":
            return -a
        if op == "factorial":
            n = _integer(a, "factorial_domain")
            if n < 0:
                raise MathError("factorial_domain")
            if n > MAX_FACTORIAL:
                raise MathError("factorial_bound")
            return _bounded(Fraction(factorial(n)))
        b = args[1]
        if op == "+":
            value = a + b
        elif op == "-":
            value = a - b
        elif op == "*":
            value = a * b
        elif op == "/":
            if b == 0:
                raise MathError("division_by_zero")
            value = a / b
        elif op == "^":
            exponent = _integer(b, "exponent_not_small_integer")
            if abs(exponent) > MAX_EXPONENT:
                raise MathError("exponent_not_small_integer")
            if a == 0 and exponent < 0:
                raise MathError("division_by_zero")
            value = a ** exponent
        elif op == "mod":
            numerator = _integer(a, "modulo_domain")
            modulus = _integer(b, "modulo_domain")
            if modulus <= 0:
                raise MathError("modulo_domain")
            value = Fraction(numerator % modulus)
        elif op == "binomial":
            n, k = _integer(a, "binomial_domain"), _integer(b, "binomial_domain")
            if n < 0 or k < 0 or k > n:
                raise MathError("binomial_domain")
            if n > MAX_BINOMIAL:
                raise MathError("binomial_bound")
            value = Fraction(comb(n, k))
        elif op == "root":
            degree = _integer(b, "root_degree")
            if not 2 <= degree <= MAX_EXPONENT:
                raise MathError("root_degree")
            value = Fraction(_exact_root(a.numerator, degree), _exact_root(a.denominator, degree))
        else:
            raise MathError("unsupported_math_operation:" + op)
        return _bounded(value)

    return run(expr, {})


def to_deferred(expr: MathExpr, number_type=None, *, supported_ops=CORE_OPS, max_nodes: int = 64):
    """Build backend IR, expanding bounded syntax but NEVER evaluating an answer.

    ``number_type`` must accept ``(op, tuple_of_args)`` and have ``literal(str)``.
    Defaults to ``amoeba_calc_gpu.Number`` (whose import does not initialize MLX).
    Literal factorials/binomials and literal-bound sums/products expand to core
    arithmetic. Other operations require explicit backend capability or refuse.
    ``max_nodes`` counts emitted IR instructions, including duplicated subtrees.
    """
    _validate(expr)
    if not isinstance(max_nodes, int) or not 1 <= max_nodes <= 8192:
        raise MathError("invalid_instruction_bound")
    if number_type is None:
        from .amoeba_calc_gpu import Number
        number_type = Number
    emitted = 0

    def emit(op, *args, text=None):
        nonlocal emitted
        emitted += 1
        if emitted > max_nodes:
            raise MathError("deferred_instruction_bound")
        if op not in supported_ops:
            raise MathError("unsupported_backend_operation:" + op)
        return number_type.literal(text) if op == "literal" else number_type(op, tuple(args))

    def integer_literal(number: int):
        if number < 0:
            return emit("neg", emit("literal", text=str(-number)))
        return emit("literal", text=str(number))

    def syntactic_integer(node: MathExpr, env: dict[str, int]) -> int:
        if node.op == "variable" and node.text in env:
            return env[node.text]
        if node.op == "neg":
            return -syntactic_integer(node.args[0], env)
        return _literal_integer(node)

    def lower(node: MathExpr, env: dict[str, int]):
        op = node.op
        if op == "literal":
            text = "0" + node.text if node.text.startswith(".") else node.text
            return emit(op, text=text)
        if op == "variable":
            if node.text not in env:
                raise MathError("unbound_variable")
            return integer_literal(env[node.text])
        if op in {"sum", "prod"}:
            lo, hi = _literal_integer(node.args[0]), _literal_integer(node.args[1])
            _check_terms(lo, hi)
            # Identity is emitted only for empty ranges, avoiding unused IR nodes.
            value = None
            for index in range(lo, hi + 1):
                term = lower(node.args[2], {**env, node.text: index})
                value = term if value is None else emit("+" if op == "sum" else "*", value, term)
            return integer_literal(0 if op == "sum" else 1) if value is None else value
        if op == "factorial" and op not in supported_ops:
            try:
                n = syntactic_integer(node.args[0], env)
            except MathError:
                raise MathError("unsupported_backend_operation:factorial") from None
            if n < 0:
                raise MathError("factorial_domain")
            if n > MAX_FACTORIAL:
                raise MathError("factorial_bound")
            value = integer_literal(1)
            for factor in range(2, n + 1):
                value = emit("*", value, integer_literal(factor))
            return value
        if op == "binomial" and op not in supported_ops:
            try:
                n, k = (syntactic_integer(child, env) for child in node.args)
            except MathError:
                raise MathError("unsupported_backend_operation:binomial") from None
            if n < 0 or k < 0 or k > n:
                raise MathError("binomial_domain")
            if n > MAX_BINOMIAL:
                raise MathError("binomial_bound")
            # Multiplicative recurrence; exact divisions stay deferred, with no
            # huge factorial intermediate. Symmetry reduces instruction count.
            k = min(k, n - k)
            value = integer_literal(1)
            for index in range(1, k + 1):
                value = emit("/", emit("*", value, integer_literal(n - k + index)), integer_literal(index))
            return value
        if op not in supported_ops:
            raise MathError("unsupported_backend_operation:" + op)
        return emit(op, *(lower(child, env) for child in node.args))

    return lower(expr, {})
