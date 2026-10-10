"""Lark ``/regex/`` terminals mean the same thing to ``re`` and to ``regex``."""
import re

import pytest
import regex

from mlx2.lark_regex import LarkGrammarError, lark_to_regex
from mlx2.structured_output import compile_constraint


@pytest.mark.parametrize(
    "grammar, message",
    [
        ("start: /(?:abc){e<=1}/", "fuzzy"),  # ``regex`` fuzzy match; literal for ``re``
        ("start: /abc{i<=1,d<=1}/", "fuzzy"),
        ("start: /x{e}/", "fuzzy"),
        ("start: /x{s}/", "fuzzy"),
        ("start: /x{i,d}/", "fuzzy"),
        ("start: /a{1, 2}/x", "brace"),  # a quantifier only to ``regex`` in verbose mode
        ("start: /(?x-i:a{1, 2})/", "brace"),  # a scoped verbose group
        ("start: /(?x:a { 1 , 2 })/", "brace"),
        # Integration review: a lazy or possessive verbose brace is one repeat
        # node to ``regex`` and none to ``re``; the old count comparison missed
        # it when another repeat balanced the count.
        ("start: /(?x:a{1, 2}?)/", "brace"),
        ("start: /(?x:a{1, 2}+)/", "brace"),
        ("start: /a{1, 2}?/x", "brace"),
        ("start: /(?x:a{1, 2}?)b*/", "brace"),  # ``b*`` keeps the counts equal
        # Integration review round 2: nested shapes balance the repeat
        # bounds too; the spelling is refused wherever a verbose scope opens.
        ("start: /(?x:(?:a{0, 1}?){1,2})/", "brace"),
        ("start: /(?x:(?:a{0, 1}+){1,2})/", "brace"),
        ("start: /(?x:(?:a{0, 1}){1,2})/", "brace"),
        # A global inline flag inside a terminal would apply to the whole
        # lowered grammar under ``regex``: refused as such, not read.
        ("start: /(?x)a{1, 2}/", "global flags"),
        ("start: /(?x) a { 1 , 2 }/", "global flags"),
        ("start: /[[:alpha:]]+/", "POSIX"),  # a POSIX class only to ``regex``
        ("start: /[a[:alpha:]]+/", "POSIX"),
        ("start: /\\mfoo/", "extension"),  # ``regex``-only escapes and groups
        ("start: /foo\\M/", "extension"),
        ("start: /\\Gfoo/", "extension"),
        ("start: /foo\\Kbar/", "extension"),
        ("start: /\\p{L}+/", "extension"),
        ("start: /(?|a|b)/", "extension"),
        ("start: /(?r)abc/", "extension"),
        ("start: /(?V1)abc/", "extension"),
        ("start: /(?<name>a)/", "extension"),
        ("start: /(?e)abc/", "extension"),
    ],
)
def test_regex_module_only_constructs_are_refused_in_terminals(grammar, message):
    with pytest.raises(LarkGrammarError, match=message):
        lark_to_regex(grammar)


@pytest.mark.parametrize(
    "body, texts",
    [
        (r"[a-z]+", ["abc", "ABC", ""]),
        (r"(?:ab){2,3}", ["abab", "ababab", "ab"]),
        (r"a{2}b{1,}c{,2}", ["aabcc", "aab", "abc"]),
        (r"[{}\[\]]+", ["{}", "[]", "{e<=1}"]),
        (r"\{e<=1\}", ["{e<=1}", "abc"]),
        # Spellings both engines read alike under ``regex``'s default V0.
        (r"x{foo}", ["x{foo}", "x", "xfoo"]),
        (r"x{,}|y{1, 2}", ["x{,}", "y{1, 2}", "yy"]),
        (r"(?x:a){1, 2}", ["a{1, 2}", "aa"]),  # the brace is outside the verbose scope
        (r"(?x:a(?-x:b{1, 2}))", ["ab{1, 2}", "abb"]),
        (r"(?x: a{1,2} b )", ["ab", "aab", "a b"]),
        (r"[:alpha:]+", ["alpha", ":", "b", "x"]),
        (r"[[.a.]]+", ["[.a.]", "a]", "b"]),
        (r"[[a]]+", ["[a]", "a]", "b"]),
        (r"[a&&b]+", ["a&b", "c"]),
        (r"[.:]+", [".:", "a"]),
        (r"(?P<n>x)(?P=n)", ["xx", "x"]),
        (r"\d+\.\d*|\w+", ["1.5", "1.", "abc", "é"]),
        (r"[^\]]+", ["abc", "]"]),
        (r"(?i:AbC)|\<\>", ["abc", "ABC", "<>"]),
        (r"a++b|(?>c)d", ["aab", "cd", "ab"]),
    ],
)
def test_accepted_terminals_agree_between_the_two_engines(body, texts):
    import warnings

    pattern = lark_to_regex(f"start: /{body}/")
    enforced = compile_constraint(grammar=pattern)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ``re``'s future-syntax warnings
        validated = re.compile(body)
    for text in texts:
        assert (enforced.fullmatch(text) is not None) == (
            validated.fullmatch(text) is not None
        ), (body, text)


def test_fuzzy_terminal_never_reaches_the_enforcing_engine():
    # Before the fix the lowering was accepted and ``regex`` enforced a fuzzy
    # match that admitted ``abx`` against a grammar declaring only ``abc``.
    with pytest.raises(LarkGrammarError):
        pattern = lark_to_regex("start: /(?:abc){e<=1}/")
        assert regex.fullmatch(pattern, "abx") is None
