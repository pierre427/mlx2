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

A ``/regex/`` terminal is enforced by the ``regex`` module, which reads some
spellings differently from Python's ``re`` (fuzzy ``{e<=1}`` braces, POSIX
``[[:alpha:]]`` classes, verbose-mode ``{1, 2}``, ``\\m`` ``\\M`` ``\\G`` ``\\K``
``\\p`` escapes, ``(?|`` ``(?r)`` ``(?V1)`` groups).  A terminal is accepted only
when both engines compile it and it uses none of those, so what the client's
grammar says is what the server enforces (``_regex_terminal``).  The class
escapes ``\\w`` ``\\d`` ``\\s`` keep the enforcing engine's Unicode
definitions (UTS#18: marks and connector punctuation are word characters,
superscript digits are not), which are also llguidance's, the reference
enforcer of this grammar syntax; ``re``'s ``str.isalnum()`` reading is the
outlier and is not what a client's grammar means.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter

import regex
from regex import _regex_core

try:  # ``re``'s parser: private, stable since 3.11 (``sre_parse`` before)
    from re import _constants as _re_constants
    from re import _parser as _re_parser
except ImportError:  # pragma: no cover - older interpreters
    import sre_constants as _re_constants
    import sre_parse as _re_parser

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


# Escapes the ``regex`` module reads as extensions.  ``re`` refuses most of
# them too, but naming them keeps the refusal explicit should ``re`` learn one.
_REGEX_ONLY_ESCAPES = frozenset("mMGKpPXRhHeLg")
_RE_REPEATS = frozenset(
    getattr(_re_constants, name)
    for name in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")
    if hasattr(_re_constants, name)
)


def _regex_tree(pattern: str):
    """``regex``'s parse tree of ``pattern`` (the enforcing engine's reading)."""
    bits = regex.V0
    while True:
        source = _regex_core.Source(pattern)
        info = _regex_core.Info(bits, str, {})
        info.guess_encoding = _regex_core.UNICODE
        source.ignore_space = bool(info.flags & _regex_core.VERBOSE)
        try:
            return _regex_core._parse_pattern(source, info)
        except _regex_core._UnscopedFlagSet:
            bits = info.global_flags  # a global inline flag: parse again under it


def _regex_nodes(root):
    pending = [root]
    while pending:
        node = pending.pop()
        yield node
        for value in vars(node).values():
            if isinstance(value, _regex_core.RegexBase):
                pending.append(value)
            elif isinstance(value, (list, tuple)):
                pending.extend(item for item in value if isinstance(item, _regex_core.RegexBase))


def _re_repeat_bounds(subpattern) -> list:
    """The ``(min, max)`` of every repeat node in ``re``'s parse tree
    (``None`` for unbounded), in no particular order."""
    bounds = []
    for opcode, value in subpattern:
        if opcode in _RE_REPEATS:
            low, high = value[0], value[1]
            bounds.append((low, None if high is _re_constants.MAXREPEAT else high))
            bounds.extend(_re_repeat_bounds(value[2]))
        elif opcode is _re_constants.SUBPATTERN:
            bounds.extend(_re_repeat_bounds(value[3]))
        elif opcode is _re_constants.BRANCH:
            for branch in value[1]:
                bounds.extend(_re_repeat_bounds(branch))
        elif opcode in (_re_constants.ASSERT, _re_constants.ASSERT_NOT):
            bounds.extend(_re_repeat_bounds(value[1]))
        elif opcode is _re_constants.ATOMIC_GROUP:
            bounds.extend(_re_repeat_bounds(value))
        elif opcode is _re_constants.GROUPREF_EXISTS:
            bounds.extend(_re_repeat_bounds(value[1]))
            if value[2]:
                bounds.extend(_re_repeat_bounds(value[2]))
    return bounds


def _regex_repeat_bounds(tree) -> list:
    """The ``(min, max)`` of every repeat node in ``regex``'s parse tree.
    Lazy and possessive repeats subclass ``GreedyRepeat``, so they are
    counted by their bounds like any other."""
    bounds = []
    for node in _regex_nodes(tree):
        if isinstance(node, _regex_core.GreedyRepeat):
            high = node.max_count
            bounds.append((node.min_count, None if high in (None, _regex_core.UNLIMITED) else high))
    return bounds


_FLAG_GROUP = re.compile(r"\(\?([a-zA-Z]*)(?:-([a-zA-Z]*))?([:)])")


def _spaced_brace_in_verbose_scope(pattern: str) -> bool:
    """True when a ``{...}`` containing whitespace sits where verbose mode is
    in effect: ``regex`` drops the whitespace and reads a quantifier, ``re``
    keeps a literal.  Scope-aware: ``(?x:...)`` / ``(?-x:...)`` groups and
    ``(?x)`` for the rest of its group; escapes, character classes and
    verbose comments are skipped (a brace in those reads alike)."""
    stack = [False]
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "\\":
            i += 2
            continue
        if char == "[":
            j = i + 1
            if j < n and pattern[j] == "^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 2 if pattern[j] == "\\" else 1
            i = j + 1
            continue
        if char == "(":
            if pattern.startswith("(?#", i):
                end = pattern.find(")", i)
                i = n if end < 0 else end + 1
                continue
            match = _FLAG_GROUP.match(pattern, i)
            if match:
                on, off, close = match.group(1) or "", match.group(2) or "", match.group(3)
                verbose = stack[-1]
                if "x" in on:
                    verbose = True
                if "x" in off:
                    verbose = False
                if close == ":":
                    stack.append(verbose)
                else:
                    stack[-1] = verbose
                i = match.end()
                continue
            stack.append(stack[-1])
            i += 1
            continue
        if char == ")":
            if len(stack) > 1:
                stack.pop()
            i += 1
            continue
        if char == "#" and stack[-1]:
            end = pattern.find("\n", i)
            i = n if end < 0 else end + 1
            continue
        if char == "{" and stack[-1]:
            end = pattern.find("}", i)
            if end > i and any(ch.isspace() for ch in pattern[i + 1 : end]):
                return True
            i = end + 1 if end > i else i + 1
            continue
        i += 1
    return False


def _check_read_alike(pattern: str) -> None:
    """Refuse a terminal whose braces the two engines read differently.

    Both engines have compiled ``pattern`` (the terminal with its flags
    scoped).  ``regex`` reads ``{e<=1}``, ``{s}``, ``{i,d}`` as fuzzy matching
    where ``re`` reads a literal (a ``Fuzzy`` node in its tree); in a verbose
    scope, inline or scoped, ``regex`` reads ``{1, 2}`` as a quantifier where
    ``re`` keeps the literal, which shows as a repeat node with bounds in its
    tree that ``re``'s tree lacks (comments and disabled scopes read alike in
    both).  Asked
    of the two parsers, so the decision is exactly the engines' own.
    """
    tree = _regex_tree(pattern)
    if any(isinstance(node, _regex_core.Fuzzy) for node in _regex_nodes(tree)):
        raise LarkGrammarError("fuzzy matching braces are not portable in a lark terminal")
    # A brace with whitespace inside it, in a terminal that opens a verbose
    # scope anywhere, is refused outright: ``regex`` reads it as a quantifier
    # and ``re`` as a literal, and a nested shape can balance the repeat
    # bounds compared below (``(?x:(?:a{0, 1}?){1,2})``), so the parse-tree
    # comparison alone is not a proof.  Fail closed on the spelling instead.
    if _spaced_brace_in_verbose_scope(pattern):
        raise LarkGrammarError(
            "a brace expression in a verbose lark terminal is a quantifier to the "
            "enforcing engine only: write it without spaces, or escape it"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the same future-syntax warnings as compiling
        parsed = _re_parser.parse(pattern)
    # Compare the bounds, not the node count: a lazy or possessive brace
    # (``a{1, 2}?``) would otherwise be read as one repeat by each engine.
    if Counter(_regex_repeat_bounds(tree)) != Counter(_re_repeat_bounds(parsed)):
        raise LarkGrammarError(
            "a brace expression in a verbose lark terminal is a quantifier to the "
            "enforcing engine only: write it without spaces, or escape it"
        )


def _check_portable(body: str) -> None:
    """Refuse the set syntax the two engines read differently.

    A ``[:name:]`` inside a bracket set is a POSIX class to ``regex`` (V0)
    and plain members to ``re``.  Spellings the two engines read alike
    (``[[a]]``, ``[.a.]``, a top-level ``[:alpha:]``, set-operator lookalikes
    under V0) are accepted; braces are checked on the parse trees
    (``_check_read_alike``).
    """
    position, length, in_set = 0, len(body), False
    while position < length:
        char = body[position]
        if char == "\\":
            escaped = body[position + 1 : position + 2]
            if escaped in _REGEX_ONLY_ESCAPES:
                raise LarkGrammarError(
                    f"regex-module extension \\{escaped} is not portable in a lark terminal"
                )
            position += 2
            continue
        if in_set:
            if char == "]":
                in_set = False
            elif char == "[" and body[position + 1 : position + 2] == ":":
                raise LarkGrammarError(
                    "POSIX character classes are not portable in a lark terminal"
                )
            position += 1
            continue
        if char == "[":
            in_set = True
            position += 1
            if body[position : position + 1] == "^":
                position += 1
            if body[position : position + 1] == "]":
                position += 1  # a leading ``]`` is literal in both engines
            continue
        position += 1


def _regex_terminal(token: str) -> str:
    end = token.rindex("/")
    body, flags = token[1:end], token[end + 1 :]
    _check_portable(body)
    # The terminal as it is lowered: its flags scoped to it.  A global inline
    # flag inside the body (``(?x)``) is refused by ``re`` here; ``regex``
    # would apply it to the whole lowered grammar.
    pattern = f"(?{flags}:{body})" if flags else f"(?:{body})"
    try:
        with warnings.catch_warnings():
            # ``re`` warns about set syntax a future version may read
            # differently; the enforcing engine (V0) reads it as ``re`` does.
            warnings.simplefilter("ignore")
            re.compile(pattern)
        regex.compile(pattern)
    except (re.error, regex.error) as error:
        raise LarkGrammarError(f"invalid lark regex {token!r}: {error}") from error
    _check_read_alike(pattern)
    return pattern


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
