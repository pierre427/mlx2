"""Exact token-level automaton: language, mask and lifecycle differentials
against the regex scanner it replaces."""

import os
import random
import string
import time
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2 import structured_automaton as sa
from mlx2 import structured_output as so
from mlx2.structured_automaton import AutomatonUnsupported, compile_pattern
from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

QWEN_TOKENIZER = os.path.expanduser("~/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp")

RICH_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "kind": {"enum": ["alpha", "beta", 3, None, "a\"q\\"]},
        "score": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "inner": {
            "type": "object",
            "properties": {
                "flag": {"type": "boolean"},
                "maybe": {"type": ["null", "string"]},
                "deep": {
                    "type": "object",
                    "properties": {"n": {"type": "array", "items": {"type": "number"}}},
                    "required": ["n"],
                    "additionalProperties": False,
                },
            },
            "required": ["flag"],
            "additionalProperties": False,
        },
        "note": {"type": "string"},
        "extra": {"const": "fixed"},
    },
    "required": ["name", "age", "kind"],
    "additionalProperties": False,
}
OPTIONAL_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}, "c": {"type": "boolean"}},
    "additionalProperties": False,
}


def _schema_format(schema):
    return {"type": "json_schema", "json_schema": {"name": "t", "strict": True, "schema": schema}}


CONSTRAINTS = {
    "json_object": ({"type": "json_object"}, None),
    "rich_schema": (_schema_format(RICH_SCHEMA), None),
    "optional_schema": (_schema_format(OPTIONAL_ONLY_SCHEMA), None),
    "array_schema": (_schema_format({"type": "array", "items": {"type": "integer"}}), None),
    "list_grammar": (None, r"[a-z]+(,[a-z]+)*"),
    "redos_grammar": (None, r"(a|aa)+b"),
    "class_grammar": (None, r"\d{2,3}-\w+?\s?[^x-z\n]{0,2}.(?P<tail>yes|no)?"),
    "lazy_counted": (None, r"(?:ab){2,}?c{,2}[\-\]\\]"),
}
DOCUMENTS = {
    "json_object": [
        '{"a":[1,true,{"b":null}], "c": "x\\u00e9\\n\\"q" , "d": {"e": {"f": [[[{"g": -1.5e+3}]]]}}}',
        '{ }',
        '{"long": "' + "lorem ipsum dolor sit amet " * 8 + '", "n": [0, -0.5, 1E9, []], "z": {}}',
    ],
    "rich_schema": [
        '{"name":"bob","age":3,"kind":null,"score":-1.5e3,"tags":["x","y"],"inner":{"flag":true,"maybe":"q","deep":{"n":[1,2.5]}},"note":"n","extra":"fixed"}',
        '{ "name" : "a\\\\b\\u00e9" , "age" : -12 , "kind" : "a\\"q\\\\", "inner": {"flag": false}, "extra": "fixed" }',
        '{"name":"' + "long text " * 12 + '","age":0,"kind":3,"tags":[]}',
    ],
    "optional_schema": ['{}', '{"a":"x","c":true}', '{"b":7}', '{"a":"x","b":1,"c":false}'],
    "array_schema": ['[]', '[1, -2,3 ]'],
    "list_grammar": ["abc,de,f", "z"],
    "redos_grammar": ["aaaaab", "ab"],
    "class_grammar": ["12-ab_9 q!.yes", "123-é٣\tno", "99-w.."],
    "lazy_counted": ["ababc-", "abababcc]", "abab\\"],
}
MUTATIONS = '{}[]",:1a \\eE-.0tn\n'


def _compile(name):
    response_format, grammar = CONSTRAINTS[name]
    return compile_constraint(response_format, grammar)


@pytest.mark.parametrize("name", sorted(CONSTRAINTS))
def test_automaton_language_equals_regex(name):
    constraint = _compile(name)
    automaton = compile_pattern(constraint.pattern.pattern)
    rng = random.Random(name)
    checked = 0
    for document in DOCUMENTS[name]:
        assert constraint.fullmatch(document) is not None, document
        texts = [document]
        for _ in range(60):
            cut = rng.randrange(len(document) + 1)
            texts.append(document[:cut] + rng.choice(MUTATIONS) + document[cut + rng.randrange(2) :])
        for text in texts:
            for cut in range(len(text) + 1):
                prefix = text[:cut]
                assert automaton.fullmatch(prefix, partial=True) == (
                    constraint.fullmatch(prefix, partial=True) is not None
                ), prefix
                assert automaton.fullmatch(prefix) == (constraint.fullmatch(prefix) is not None), prefix
                checked += 1
    assert checked > 100


def test_class_parser_agrees_with_regex_enumeration():
    import regex

    sources = [
        r"[\x20\x09\x0a\x0d]", r'[^"\\\x00-\x1f]', r'["\\/bfnrt]', r"[0-9a-fA-F]", r"[1-9]",
        r"[eE]", r"[+-]", r"[]a]", r"[^]a]", r"[a\-z]", r"[-a]", r"[\d_x]", r"[^\w\s]",
        r"[\u00e9-\u00ff\U0001F600]", r"[\b\t]", r".", r"\S", r"\p{Lu}", r"[\P{L}a-c]",
    ]
    for source in sources:
        parsed = sa._Parser(source).parse()
        assert parsed[0] == "set"
        assert parsed[1] == sa._enumerate_charset(source), source
        assert regex.compile(source)  # the oracle itself accepts the source


@pytest.mark.parametrize(
    "pattern",
    [
        r"a(?=b)b", r"a(?!b).", r"(?<=a)b", r"(?<!a)b", r"(a)\1", r"(?P<x>a)(?P=x)", r"^ab", r"ab$",
        r"\bab", r"a\Z", r"(?>a+)a", r"a++b", r"a*+b", r"(?i)ab", r"(?i:ab)", r"(?#c)ab",
        r"(?|(a)|(b))", r"(a)(?(1)b|c)", r"(?R)?a", r"(a(?1)?)", r"(?:ab){e<=1}", r"[[:alpha:]]",
        r"[a[bc]]", r"a{b", r"a}", r"a]", r"\N{BULLET}", r"\X", r"a{2}{3}", r"\01",
        # Recursion that is not LL(1)-deterministic or not properly nested.
        r"(?&v)(?(DEFINE)(?P<v>a(?&v)?\s*))",
        r"(?&v)(?(DEFINE)(?P<v>(?&v)a|b))",
        r"(?&v)(?(DEFINE)(?P<v>(?:x(?&v)y)?))",
        r"(?&missing)",
    ],
)
def test_unsupported_patterns_are_refused(pattern):
    with pytest.raises(AutomatonUnsupported):
        compile_pattern(pattern)


def test_supported_recursion_beyond_json():
    import regex

    source = r"(?&expr)(?(DEFINE)(?P<expr>\((?:(?&expr)|[a-z])(?:,(?:(?&expr)|[a-z]))*\)))"
    oracle = regex.compile(source)
    automaton = compile_pattern(source)
    assert automaton.recursive
    for text in ("(a)", "((a),b,((c)))", "((a,b)", "(a,)", "()", "((((((((a))))))))", "(a)(b)"):
        for cut in range(len(text) + 1):
            prefix = text[:cut]
            assert automaton.fullmatch(prefix) == (oracle.fullmatch(prefix) is not None), prefix
            assert automaton.fullmatch(prefix, partial=True) == (
                oracle.fullmatch(prefix, partial=True) is not None
            ), prefix


def test_state_cap_refuses_exponential_patterns(monkeypatch):
    # (a|b)*a(a|b){n} needs 2**(n+1) DFA states.
    monkeypatch.setattr(sa, "MAX_DFA_STATES", 500)
    with pytest.raises(AutomatonUnsupported, match="DFA cap"):
        compile_pattern(r"(?:a|b)*a(?:a|b){12}")
    assert compile_pattern(r"(?:a|b)*a(?:a|b){4}").state_count == 32
    monkeypatch.setattr(sa, "MAX_NFA_STATES", 2000)
    with pytest.raises(AutomatonUnsupported, match="NFA state cap"):
        compile_pattern(r"(?:(?:ab){50}){50}")


def _synthetic_tokenizer(pieces, eos=(0,)):
    return SimpleNamespace(
        vocab_size=len(pieces),
        eos_token_ids=list(eos),
        decode=lambda ids, **_kw: "".join(pieces[i] for i in ids if i < len(pieces)),
    )


def _synthetic_pieces(seed=13, count=3000):
    rng = random.Random(seed)
    alphabet = string.ascii_lowercase + string.digits + ' {}[]":,.-\n\\eEtrufalsn'
    singles = sorted(set(alphabet + string.ascii_uppercase + "é٣_!\t+/"))
    pieces = ["<eos>"] + singles + [
        "".join(rng.choice(alphabet) for _ in range(rng.choice([2, 2, 3, 4, 6])))
        for _ in range(count)
    ]
    pieces += [
        '"}', '"}]', "}]}", "]]", "}}", '":', '",', '": "', '", "', "}\n", " " * 16, "\n\n", "\\u00", "\\n",
        '{"', '[{', "[[", "[]", "{}", "true", "false", "null", '"name"', '"age":', "-1.5e+3", "<|special|>",
        "a\ufffd", "\ufffd", "",
    ]
    pieces[40] = pieces[41]  # duplicate decode
    return pieces


def _encode_chars(pieces, text):
    lookup = {}
    for token, piece in enumerate(pieces):
        if len(piece) == 1 and token != 0:
            lookup.setdefault(piece, token)
    return [lookup[char] for char in text]


def _scanner_allowed(processor, constraint, prefix):
    try:
        return set(processor._allowed(constraint.canonicalize(prefix)))
    except ValueError:
        return set()


def _automaton_allowed(processor, ids):
    try:
        return set(np.flatnonzero(processor._automaton_allowed(ids)).tolist())
    except ValueError:
        return set()


@pytest.mark.parametrize("name", ["json_object", "rich_schema", "optional_schema", "array_schema", "list_grammar", "class_grammar"])
def test_automaton_mask_equals_exhaustive_scanner_mask(name, monkeypatch):
    monkeypatch.setattr(so, "_ALLOWED_BUDGET_SECONDS", 600.0)
    monkeypatch.setattr(so, "_MATCH_TIMEOUT_SECONDS", 10.0)
    pieces = _synthetic_pieces()
    tokenizer = _synthetic_tokenizer(pieces)
    constraint = _compile(name)
    processor = StructuredOutputProcessor(tokenizer, 0, constraint)
    assert processor.engine == "automaton", processor.automaton_refusal
    rng = random.Random(name)
    prefixes = 0
    for document in DOCUMENTS[name]:
        cuts = set(range(min(len(document), 40) + 1)) | {len(document)}
        cuts |= {rng.randrange(len(document) + 1) for _ in range(25)}
        for cut in sorted(cuts):
            prefix = document[:cut]
            ids = _encode_chars(pieces, prefix)
            assert _automaton_allowed(processor, ids) == _scanner_allowed(processor, constraint, prefix), prefix
            prefixes += 1
    assert prefixes >= 10
    # Visits are memoized per automaton state, not per prefix.
    assert processor._trie.walks + processor._trie.hits >= prefixes - 5
    assert processor._trie.walks <= processor._automaton.state_count * 4


def test_deep_nesting_and_multi_frame_pops_match_the_scanner(monkeypatch):
    monkeypatch.setattr(so, "_ALLOWED_BUDGET_SECONDS", 600.0)
    monkeypatch.setattr(so, "_MATCH_TIMEOUT_SECONDS", 10.0)
    pieces = _synthetic_pieces()
    tokenizer = _synthetic_tokenizer(pieces)
    constraint = compile_constraint({"type": "json_object"})
    processor = StructuredOutputProcessor(tokenizer, 0, constraint)
    shapes = [
        '{"a":' * 12 + "1",
        '{"a":[' * 9 + '{"k": "v"}',
        '{"a":[[{"b":[{"c":[1',
        '{"a":{"b":[[2]',
        '{"a":[{"b":{"c":"x"',
        '{"a":[[[[',
    ]
    token_by_piece = {piece: token for token, piece in enumerate(pieces)}
    for prefix in shapes:
        ids = _encode_chars(pieces, prefix)
        got = _automaton_allowed(processor, ids)
        assert got == _scanner_allowed(processor, constraint, prefix), prefix
    # Same top state, different frames underneath: "}]}" pops three frames and
    # is admissible only when they are obj, arr, obj in that order.
    inside_obj_arr_obj = _encode_chars(pieces, '{"a":[{"b":1')
    inside_obj_obj_obj = _encode_chars(pieces, '{"a":{"q":{"b":1')
    closer = token_by_piece["}]}"]
    assert closer in _automaton_allowed(processor, inside_obj_arr_obj)
    assert closer not in _automaton_allowed(processor, inside_obj_obj_obj)
    assert token_by_piece["}}"] in _automaton_allowed(processor, inside_obj_obj_obj)


def test_processor_tracks_state_incrementally_and_survives_rollback():
    pieces = _synthetic_pieces()
    tokenizer = _synthetic_tokenizer(pieces)
    constraint = _compile("rich_schema")
    document = DOCUMENTS["rich_schema"][0]
    ids = _encode_chars(pieces, document)
    live = StructuredOutputProcessor(tokenizer, 0, constraint)
    walked = []

    class CountingAutomaton:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def advance(self, config, text):
            walked.append(len(text))
            return self._inner.advance(config, text)

    live._automaton = CountingAutomaton(live._automaton)
    for step in range(len(ids) + 1):
        fresh = StructuredOutputProcessor(tokenizer, 0, constraint)
        assert _automaton_allowed(live, ids[:step]) == _automaton_allowed(fresh, ids[:step])
    # One character of text per step: the output was never re-walked.
    assert max(walked) == 1 and len(walked) == len(ids)
    # Roll back (speculative verification), diverge, and come back.
    fresh = StructuredOutputProcessor(tokenizer, 0, constraint)
    assert _automaton_allowed(live, ids[:10]) == _automaton_allowed(fresh, ids[:10])
    detour = ids[:8] + _encode_chars(pieces, "zzz")  # {"name":zzz
    assert _automaton_allowed(live, detour) == set()  # dead text admits nothing
    assert _automaton_allowed(live, ids[:30]) == _automaton_allowed(fresh, ids[:30])
    # A multi-token jump forward, then a query inside the jumped span.
    assert _automaton_allowed(live, ids[:50]) == _automaton_allowed(fresh, ids[:50])
    assert _automaton_allowed(live, ids[:44]) == _automaton_allowed(fresh, ids[:44])


def test_processor_call_masks_exactly_with_eos_only_when_accepting():
    import mlx.core as mx

    pieces = ["<eos>", "{", "}", '"', "a", ":", "1", " ", '{"a"', "\ufffd", ""]
    tokenizer = _synthetic_tokenizer(pieces)
    processor = StructuredOutputProcessor(
        tokenizer, 2, compile_constraint({"type": "json_object"}), top_k=20, top_p=0.8
    )
    assert processor.engine == "automaton"
    width = len(pieces) + 5  # padded logits
    logits = mx.zeros((1, width), dtype=mx.bfloat16)

    def admitted(generated):
        out = processor(mx.array([7, 7] + generated, dtype=mx.uint32), logits)
        assert out.dtype == mx.bfloat16 and out.shape == logits.shape
        return set(np.flatnonzero(np.isfinite(np.array(out.astype(mx.float32))[0])).tolist())

    assert admitted([]) == {1, 8}
    assert admitted([8]) == {5, 7}
    assert admitted([8, 5, 6]) == {2, 6, 7}  # "}" or more digits or whitespace; no EOS yet
    assert admitted([8, 5, 6, 2]) == {0}  # complete: EOS and nothing else
    assert processor.failure is None
    assert processor.tail_mass_bound == 0.0 and processor.parallel_scans == 0
    # EOS ids may sit beyond the tokenizer vocabulary (Qwen's do).
    far = SimpleNamespace(vocab_size=len(pieces), eos_token_ids=[len(pieces) + 3], decode=tokenizer.decode)
    processor = StructuredOutputProcessor(far, 0, compile_constraint({"type": "json_object"}))
    out = processor(mx.array([1, 2], dtype=mx.uint32), logits)
    assert set(np.flatnonzero(np.isfinite(np.array(out.astype(mx.float32))[0])).tolist()) == {len(pieces) + 3}


def test_automaton_dead_end_fails_closed_like_the_scanner():
    import mlx.core as mx

    pieces = ["<eos>", "a", "b", "ab", "c"]
    processor = StructuredOutputProcessor(_synthetic_tokenizer(pieces), 0, compile_constraint(None, "abx"))
    assert processor.engine == "automaton"
    logits = mx.zeros((1, len(pieces)))
    masked = processor(mx.array([3], dtype=mx.uint32), logits)
    assert processor.failure is not None and "no valid token continuation" in processor.failure
    assert mx.array_equal(masked, logits)


def test_unsupported_grammar_falls_back_to_the_scanner(monkeypatch):
    import mlx.core as mx

    pieces = ["<eos>", "a", "b", "ab", "c"]
    tokenizer = _synthetic_tokenizer(pieces)
    processor = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, r"a(?=b)b+"))
    assert processor.engine == "scanner" and "group construct" in processor.automaton_refusal
    out = processor(mx.array([1], dtype=mx.uint32), mx.zeros((1, len(pieces))))
    assert set(np.flatnonzero(np.isfinite(np.array(out)[0])).tolist()) == {2}
    assert processor.failure is None
    # Kill switch: a supported grammar is forced onto the scanner.
    monkeypatch.setenv("MLX2_STRUCTURED_AUTOMATON", "0")
    forced = StructuredOutputProcessor(tokenizer, 0, compile_constraint({"type": "json_object"}))
    assert forced.engine == "scanner" and forced._automaton is None
    monkeypatch.delenv("MLX2_STRUCTURED_AUTOMATON")
    assert StructuredOutputProcessor(tokenizer, 0, compile_constraint({"type": "json_object"})).engine == "automaton"
    # Constraint objects that are not compiled patterns stay on the scanner.
    duck = SimpleNamespace(fullmatch=lambda *a, **k: object(), canonicalize=lambda p: p)
    assert StructuredOutputProcessor(tokenizer, 0, duck).engine == "scanner"


def test_mask_memo_is_bounded_by_bytes_and_stays_correct(monkeypatch):
    pieces = _synthetic_pieces()
    tokenizer = _synthetic_tokenizer(pieces)
    constraint = _compile("rich_schema")
    processor = StructuredOutputProcessor(tokenizer, 0, constraint)
    trie = processor._trie
    entry = -(-len(pieces) // 8)
    monkeypatch.setattr(sa, "MASK_MEMO_BYTES", entry * 4)
    ids = _encode_chars(pieces, DOCUMENTS["rich_schema"][0])
    first_pass = [_automaton_allowed(processor, ids[:step]) for step in range(len(ids) + 1)]
    assert trie.memo_bytes <= entry * 4 and len(trie.memo) <= 4
    assert trie.memo_bytes == sum(packed.nbytes for packed in trie.memo.values())
    walks = trie.walks
    second = StructuredOutputProcessor(tokenizer, 0, constraint)
    assert [_automaton_allowed(second, ids[:step]) for step in range(len(ids) + 1)] == first_pass
    assert trie.walks > walks  # evicted states were recomputed, not misread


def test_redos_shaped_grammar_is_linear():
    import mlx.core as mx

    pieces = ["<eos>", "a", "aa", "aaaa", "b", "ab", "c", "a" * 64]
    tokenizer = _synthetic_tokenizer(pieces)
    processor = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, r"(a|aa)+b"))
    assert processor.engine == "automaton" and processor._automaton.state_count <= 4
    ids = [7] * 80  # 5120 characters of "a": catastrophic for a backtracking matcher
    logits = mx.zeros((1, len(pieces)))
    started = time.perf_counter()
    for step in range(0, len(ids) + 1):
        out = processor(mx.array(ids[:step], dtype=mx.uint32), logits)
    elapsed = time.perf_counter() - started
    assert processor.failure is None
    assert set(np.flatnonzero(np.isfinite(np.array(out)[0])).tolist()) == {1, 2, 3, 4, 5, 7}
    assert elapsed < 2.0, elapsed
    assert processor._trie.walks <= 3


@pytest.mark.skipif(not os.path.isdir(QWEN_TOKENIZER), reason="Qwen3.6 tokenizer artifact is not present")
def test_real_qwen_vocabulary_mask_equals_untimed_scanner(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    monkeypatch.setattr(so, "_ALLOWED_BUDGET_SECONDS", 600.0)
    # The scanner drops pieces whose match times out (10 ms); the oracle here
    # is the same exhaustive walk with that timeout out of the way.
    monkeypatch.setattr(so, "_MATCH_TIMEOUT_SECONDS", 30.0)
    hf = AutoTokenizer.from_pretrained(QWEN_TOKENIZER, local_files_only=True)
    tokenizer = SimpleNamespace(vocab_size=hf.vocab_size, eos_token_ids=[hf.eos_token_id], decode=hf.decode)
    cases = [
        ({"type": "json_object"}, '{"name": "Ada", "notes": ["first", {"born": 1815, "ok": true}]}', (0, 1, 3, 9, 14, 17, -2, -1)),
        (_schema_format(RICH_SCHEMA), '{"name": "Ada", "age": 36, "kind": "alpha", "score": 9.75, "tags": ["x"]}', (0, 1, 2, 3, 8, 13, 16, -2, -1)),
        (None, r"(?:yes|no|maybe)(?:, (?:yes|no|maybe))*", (0, 1, 2)),
    ]
    for response_format, text_or_grammar, cuts in cases:
        grammar = text_or_grammar if response_format is None else None
        text = "yes, no" if response_format is None else text_or_grammar
        constraint = compile_constraint(response_format, grammar)
        processor = StructuredOutputProcessor(tokenizer, 0, constraint)
        assert processor.engine == "automaton", processor.automaton_refusal
        ids = hf.encode(text, add_special_tokens=False)
        assert hf.decode(ids) == text
        for cut in cuts:
            position = cut % (len(ids) + 1)
            started = time.perf_counter()
            got = _automaton_allowed(processor, ids[:position])
            first_visit = time.perf_counter() - started
            assert first_visit < 0.15, first_visit  # first-visit budget on the 248K vocabulary
            expected = _scanner_allowed(processor, constraint, hf.decode(ids[:position]))
            assert got == expected, (text[:20], position, len(got), len(expected))
        assert hf.eos_token_id in _automaton_allowed(processor, ids)


def test_pushdown_depth_is_capped_and_history_is_thinned(monkeypatch):
    import mlx2.structured_output as structured

    pieces = _synthetic_pieces()
    tokenizer = _synthetic_tokenizer(pieces)
    constraint = _compile("json_object")
    monkeypatch.setattr(structured, "_MAX_PUSHDOWN_DEPTH", 12)
    monkeypatch.setattr(structured, "_TRACK_HISTORY", 8)
    processor = StructuredOutputProcessor(tokenizer, 0, constraint)
    shallow = _encode_chars(pieces, '{"a":' + "[" * 8)
    fresh = StructuredOutputProcessor(tokenizer, 0, constraint)
    for step in range(len(shallow) + 1):
        _automaton_allowed(processor, shallow[:step])
    assert _automaton_allowed(processor, shallow) == _automaton_allowed(fresh, shallow)
    # Only the start and the trailing window keep a configuration...
    kept = [i for i, c in enumerate(processor._track_configs) if c is not False]
    assert kept[0] == 0 and kept[1:] == list(range(len(shallow) - 7, len(shallow) + 1))
    assert len(kept) == 9
    # ...and a rollback past the window is re-derived, not wrong.
    assert _automaton_allowed(processor, shallow[:3]) == _automaton_allowed(fresh, shallow[:3])
    deep = _encode_chars(pieces, '{"a":' + "[" * 40)
    with pytest.raises(ValueError, match="nests deeper than 12"):
        processor._automaton_config(deep)


def test_processor_survives_a_scheduler_deepcopy_snapshot():
    """External-draft rounds deep-copy lane state, processors included."""
    import copy

    pieces = _synthetic_pieces()
    tokenizer = _synthetic_tokenizer(pieces)
    constraint = _compile("json_object")
    processor = StructuredOutputProcessor(tokenizer, 0, constraint)
    ids = _encode_chars(pieces, '{"a":[1,')
    before = _automaton_allowed(processor, ids)
    lane = {"processors": [processor], "tokens": list(ids)}
    snapshot = copy.deepcopy(lane)
    clone = snapshot["processors"][0]
    assert clone is not processor and clone._trie is processor._trie
    assert clone._automaton is processor._automaton
    # The live lane moves on; the snapshot keeps answering for its own prefix.
    longer = ids + _encode_chars(pieces, "2]")
    _automaton_allowed(processor, longer)
    assert _automaton_allowed(clone, ids) == before
    assert clone._track_ids == ids and processor._track_ids == longer
    scanner = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, r"(?=a)ab"))
    assert scanner.engine == "scanner"
    assert copy.deepcopy(scanner).engine == "scanner"
