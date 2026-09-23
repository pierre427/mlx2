"""Bounded Lark-to-regex compilation for custom-tool grammars.

OpenAI Responses ``custom`` tools declare their freeform input language as a
``grammar`` with ``syntax`` ``lark`` or ``regex``.  Codex's ``apply_patch`` is
declared in Lark but is regular: no rule reaches itself.  This module accepts
exactly that non-recursive subset and lowers it to one regular expression for
mlx2's existing ``structured_output.compile_constraint(grammar=...)`` automaton.

Supported: rule/terminal definitions (``name:``, ``?name:``, ``!name:``,
``NAME.priority:``), ``|`` alternatives including leading-``|`` continuation
lines, sequences, double-quoted literals (optional ``i`` flag), ``/regex/``
terminals (flags ``i``, ``m``, ``s``, ``x`` scoped inline), parenthesised
groups, ``[optional]``, the ``? * +`` and ``~n`` / ``~n..m`` repetition
operators, ``-> alias`` (ignored: aliases do not change the language),
``//`` comments and ``%import common.NAME`` / ``%import common (A, B)`` for a
fixed table of common terminals.  Everything else (``%ignore``, ``%declare``,
templates, recursion, unknown imports) is rejected: silently widening or
narrowing a client's grammar would make validation meaningless.
"""

from __future__ import annotations

import re

MAX_GRAMMAR_CHARS = 16384
MAX_REGEX_CHARS = 4096
# ``~n`` lowers to a counted regex repeat, which ``regex`` unrolls into one
# node per required repeat; the same bound ``structured_output`` prices every
# compiled pattern against (nested repeats are priced there by their product).
MAX_REPEAT_COUNT = 4096

# Lark's ``common.lark`` terminals, restricted to the regular ones.
COMMON_TERMINALS = {
    "LF": r"\n",
    "CR": r"\r",
    "NEWLINE": r"(?:\r?\n)+",
    "WS": r"[ \t\f\r\n]+",
    "WS_INLINE": r"[ \t]+",
    "DIGIT": r"[0-9]",
    "HEXDIGIT": r"[a-fA-F0-9]",
    "INT": r"[0-9]+",
    "SIGNED_INT": r"[+-]?[0-9]+",
    "LCASE_LETTER": r"[a-z]",
    "UCASE_LETTER": r"[A-Z]",
    "LETTER": r"[A-Za-z]",
    "WORD": r"[A-Za-z]+",
    "CNAME": r"[_A-Za-z][_A-Za-z0-9]*",
}


class LarkGrammarError(ValueError):
    """The grammar is outside the bounded regular Lark subset."""


_TOKEN = re.compile(
    r"""
    (?P<ws>[ \t]+)
  | (?P<string>"(?:[^"\\\n]|\\.)*"i?)
  | (?P<regex>/(?:[^/\\\n]|\\.)+/[imsx]*)
  | (?P<arrow>->)
  | (?P<range>~\s*[0-9]+(?:\s*\.\.\s*[0-9]+)?)
  | (?P<name>[_A-Za-z][_A-Za-z0-9]*)
  | (?P<op>[|()\[\]?*+])
    """,
    re.VERBOSE,
)


def _strip_comment(line: str) -> str:
    out, quote, slash, index = [], False, False, 0
    while index < len(line):
        char = line[index]
        if char == "\\" and (quote or slash):
            out.append(line[index : index + 2])
            index += 2
            continue
        if char == '"' and not slash:
            quote = not quote
        elif char == "/" and not quote:
            if not slash and line.startswith("//", index):
                break
            slash = not slash
        out.append(char)
        index += 1
    return "".join(out).rstrip()


def _parse_definitions(source: str) -> dict[str, str]:
    if not isinstance(source, str) or not source.strip():
        raise LarkGrammarError("lark grammar definition must be nonempty text")
    if len(source) > MAX_GRAMMAR_CHARS:
        raise LarkGrammarError(
            f"lark grammar exceeds {MAX_GRAMMAR_CHARS} characters"
        )
    rules: dict[str, str] = {}
    current = None
    header = re.compile(r"^([?!]?)([_A-Za-z][_A-Za-z0-9]*)(?:\.-?[0-9]+)?\s*:(.*)$")
    for raw in source.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("%"):
            match = re.fullmatch(
                r"%import\s+common\s*(?:\.\s*([A-Z_]+)|\(\s*([A-Z_,\s]+)\))", stripped
            )
            if match is None:
                raise LarkGrammarError(f"unsupported lark directive: {stripped!r}")
            names = [match.group(1)] if match.group(1) else [
                item.strip() for item in match.group(2).split(",") if item.strip()
            ]
            for name in names:
                if name not in COMMON_TERMINALS:
                    raise LarkGrammarError(f"unsupported common import: {name}")
                rules.setdefault(name, None)
            current = None
            continue
        if stripped.startswith("|"):
            if current is None:
                raise LarkGrammarError("alternative continuation without a rule")
            rules[current] += " " + stripped
            continue
        match = header.match(stripped)
        if match is None:
            raise LarkGrammarError(f"cannot parse lark line: {stripped!r}")
        name = match.group(2)
        if rules.get(name) is not None:
            raise LarkGrammarError(f"duplicate lark rule: {name}")
        rules[name] = match.group(3).strip()
        current = name
    if "start" not in rules or rules["start"] is None:
        raise LarkGrammarError("lark grammar requires a start rule")
    return rules


def _tokens(text: str):
    position, result = 0, []
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise LarkGrammarError(f"unsupported lark syntax near {text[position:position + 20]!r}")
        position = match.end()
        kind = match.lastgroup
        if kind != "ws":
            result.append((kind, match.group(kind)))
    return result


def _literal(token: str) -> str:
    insensitive = token.endswith("i")
    body = token[1:-2] if insensitive else token[1:-1]
    try:
        value = body.encode("latin-1", "backslashreplace").decode("unicode_escape")
    except UnicodeDecodeError as error:
        raise LarkGrammarError(f"invalid lark string literal {token!r}") from error
    escaped = re.escape(value)
    return f"(?i:{escaped})" if insensitive else escaped


def _regex_terminal(token: str) -> str:
    end = token.rindex("/")
    body, flags = token[1:end], token[end + 1 :]
    try:
        re.compile(body)
    except re.error as error:
        raise LarkGrammarError(f"invalid lark regex {token!r}: {error}") from error
    return f"(?{flags}:{body})" if flags else f"(?:{body})"


def _bounded(size: int) -> int:
    """Refuse a lowering as soon as it passes the regex size cap.

    Every rule reference is inlined, so ``r0: r1 r1`` / ``r1: r2 r2`` / ...
    doubles the pattern per level: a 406-character grammar 32 levels deep
    would lower to ~2**32 copies.  Checking each sequence and alternation as
    it grows keeps what is built within about twice the cap.
    """
    if size > MAX_REGEX_CHARS:
        raise LarkGrammarError(
            f"lark grammar lowers to more than {MAX_REGEX_CHARS} regex characters"
        )
    return size


class _Compiler:
    def __init__(self, rules):
        self.rules = rules
        self.done: dict[str, str] = {}
        self.active: list[str] = []

    def rule(self, name: str) -> str:
        if name in self.done:
            return self.done[name]
        if name not in self.rules:
            raise LarkGrammarError(f"undefined lark rule or terminal: {name}")
        if name in self.active:
            cycle = " -> ".join([*self.active[self.active.index(name) :], name])
            raise LarkGrammarError(f"recursive lark grammar is not regular: {cycle}")
        if self.rules[name] is None:
            pattern = COMMON_TERMINALS[name]
        else:
            self.active.append(name)
            tokens = _tokens(self.rules[name])
            pattern, position = self.alternatives(tokens, 0)
            if position != len(tokens):
                raise LarkGrammarError(f"unbalanced lark rule: {name}")
            self.active.pop()
        self.done[name] = pattern
        return pattern

    def alternatives(self, tokens, position):
        options, size = [], 0
        while True:
            sequence, position = self.sequence(tokens, position)
            options.append(sequence)
            size = _bounded(size + len(sequence))
            if position < len(tokens) and tokens[position] == ("op", "|"):
                position += 1
                continue
            break
        pattern = options[0] if len(options) == 1 else "(?:" + "|".join(options) + ")"
        return pattern, position

    def sequence(self, tokens, position):
        items, size = [], 0
        while position < len(tokens):
            kind, value = tokens[position]
            if kind == "op" and value in {"|", ")", "]"}:
                break
            if kind == "arrow":
                position += 1
                if position >= len(tokens) or tokens[position][0] != "name":
                    raise LarkGrammarError("lark alias requires a name")
                position += 1
                continue
            atom, position = self.atom(tokens, position)
            atom, position = self.suffix(atom, tokens, position)
            items.append(atom)
            size = _bounded(size + len(atom))
        return "".join(items), position

    def atom(self, tokens, position):
        kind, value = tokens[position]
        if kind == "string":
            return _literal(value), position + 1
        if kind == "regex":
            return _regex_terminal(value), position + 1
        if kind == "name":
            return f"(?:{self.rule(value.lstrip('?!'))})", position + 1
        if kind == "op" and value in {"(", "["}:
            closing = ")" if value == "(" else "]"
            inner, position = self.alternatives(tokens, position + 1)
            if position >= len(tokens) or tokens[position] != ("op", closing):
                raise LarkGrammarError("unbalanced lark group")
            group = f"(?:{inner})"
            return (group + "?" if closing == "]" else group), position + 1
        raise LarkGrammarError(f"unexpected lark token {value!r}")

    def suffix(self, atom, tokens, position):
        if position < len(tokens):
            kind, value = tokens[position]
            if kind == "op" and value in {"?", "*", "+"}:
                return f"(?:{atom}){value}", position + 1
            if kind == "range":
                bounds = [int(item) for item in re.findall(r"[0-9]+", value)]
                if len(bounds) == 2 and bounds[0] > bounds[1]:
                    raise LarkGrammarError("lark repetition range is inverted")
                if max(bounds) > MAX_REPEAT_COUNT:
                    raise LarkGrammarError(
                        f"lark repetition count exceeds {MAX_REPEAT_COUNT}"
                    )
                spec = "{%d}" % bounds[0] if len(bounds) == 1 else "{%d,%d}" % tuple(bounds)
                return f"(?:{atom}){spec}", position + 1
        return atom, position


def lark_to_regex(source: str) -> str:
    """Lower a non-recursive Lark grammar to one anchored-free regex."""
    try:
        pattern = _Compiler(_parse_definitions(source)).rule("start")
    except RecursionError as error:
        # A long rule chain or deep parentheses fit the grammar size cap but
        # not the lowering's recursion; the client gets a grammar error.
        raise LarkGrammarError("lark grammar nests too deeply") from error
    if len(pattern) > MAX_REGEX_CHARS:
        raise LarkGrammarError(
            f"lark grammar lowers to more than {MAX_REGEX_CHARS} regex characters"
        )
    return pattern
