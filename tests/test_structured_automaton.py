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
            # First visits take 2-7 ms on an idle host; the regression this
            # guards against (a per-call scan of the 248K vocabulary) costs
            # seconds.  0.15 s failed at 0.355 s on a host busy with parallel
            # test suites, so the bound keeps a wide margin for contention.
            assert first_visit < 1.0, first_visit
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


def test_concurrent_requests_for_a_new_pattern_share_one_compile(monkeypatch):
    """Serving compiles each request's grammar on the request's own thread,
    so a burst of requests carrying one new schema asked for it at once.
    Each compiled it (seconds apiece for a large ``maxLength``, all fighting
    the generation worker for the interpreter lock); one compile serves all."""
    import threading
    import uuid

    import regex

    calls = []
    original = sa.compile_pattern

    def slow_compile(source):
        calls.append(source)
        time.sleep(0.3)
        return original(source)

    monkeypatch.setattr(sa, "compile_pattern", slow_compile)
    pattern = regex.compile(f"(?:yes|no|z{uuid.uuid4().hex})")
    unsupported = regex.compile(f"(?:(?=a)a{uuid.uuid4().hex})")
    results = []
    lock = threading.Lock()

    def request(compiled):
        try:
            outcome = sa.automaton_for(compiled)
        except AutomatonUnsupported as exc:
            outcome = str(exc)
        with lock:
            results.append((compiled.pattern, outcome))

    threads = [
        threading.Thread(target=request, args=(compiled,))
        for compiled in [pattern] * 4 + [unsupported] * 3
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert sorted(calls) == sorted([pattern.pattern, unsupported.pattern])
    automata = {id(outcome) for source, outcome in results if source == pattern.pattern}
    assert len(automata) == 1
    refusals = {outcome for source, outcome in results if source == unsupported.pattern}
    assert len(refusals) == 1 and isinstance(next(iter(refusals)), str)
    assert len(results) == 7 and not sa._COMPILING


def _reference_minimize(table, accepting):
    """Textbook Moore refinement, kept naive on purpose as the oracle.

    States stay together while they agree on acceptance and on the block (or
    the dead state, -1) every symbol leads to.  Blocks are numbered in order
    of first appearance, the canonical numbering ``_minimize`` produces, so
    the two results must be equal array for array, not merely isomorphic.
    """
    rows = table.tolist()
    block = [int(flag) for flag in accepting.tolist()]
    count = len(set(block))
    while True:
        signatures = {}
        block = [
            signatures.setdefault(
                (block[state], tuple(block[t] if t >= 0 else -1 for t in row)),
                len(signatures),
            )
            for state, row in enumerate(rows)
        ]
        if len(signatures) == count:
            break
        count = len(signatures)
    first = {}
    for state, number in enumerate(block):
        first.setdefault(number, state)
    representatives = [first[number] for number in range(count)]
    reduced = [[block[t] if t >= 0 else -1 for t in rows[state]] for state in representatives]
    return (
        np.array(reduced, dtype=np.int32).reshape(count, table.shape[1]),
        accepting[representatives],
    )


def _bounded_string_schema(maximum):
    return _schema_format({
        "type": "object",
        "properties": {"note": {"type": "string", "maxLength": maximum}},
        "required": ["note"],
        "additionalProperties": False,
    })


def _tool_grammar_tools(maximum=None):
    """A strict and a non-strict tool; ``maximum`` adds a bounded string
    parameter.  Qwen's XML string values need a lookahead the automaton
    refuses, so its grammars here use the strict tool without one."""
    properties = {
        "mode": {"enum": ["append", "replace", "create"]},
        "lines": {"type": "integer"},
        "meta": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                 "required": ["ok"], "additionalProperties": False},
    }
    if maximum is not None:
        properties = {"path": {"type": "string", "maxLength": maximum}, **properties}
    return [
        {"type": "function", "function": {"name": "write", "strict": True, "parameters": {
            "type": "object",
            "properties": properties,
            "required": [name for name in ("path", "mode") if name in properties],
            "additionalProperties": False,
        }}},
        {"type": "function", "function": {"name": "search", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        }}},
    ]


def _minimization_corpus():
    from mlx2.adapters.muse_glimmer_output import constrained_tool_grammar as muse_grammar
    from mlx2.runtime.tool_parsers.qwen3_coder import constrained_tool_grammar as qwen_grammar

    tools = _tool_grammar_tools(24)
    return {
        **{f"json maxLength={n}": compile_constraint(_bounded_string_schema(n)).pattern.pattern
           for n in (1, 8, 40)},
        **{name: _compile(name).pattern.pattern for name in sorted(CONSTRAINTS)},
        "json enum": compile_constraint(_schema_format({"enum": ["red", "read", "reed", 12, 125, None]})).pattern.pattern,
        "muse tools": muse_grammar(tools, "required"),
        "muse strict single": muse_grammar(tools[:1], "required", parallel_tool_calls=False),
        "qwen strict parallel": qwen_grammar(_tool_grammar_tools()[:1], "required"),
        "qwen strict single": qwen_grammar(_tool_grammar_tools()[:1], "required", parallel_tool_calls=False),
        # Subset constructions of these are not minimal.
        "suffix window": r"(?:a|b)*a(?:a|b){3}",
        "shared suffix": r"ac|bc|dc?",
        "shared middle": r"x(?:ab|cb)*y",
        "overlapping splits": r"(?:ab|a)(?:bc|c)",
        "numbers": r"-?(?:\d+|\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?",
    }


def test_minimization_equals_textbook_refinement_on_grammars(monkeypatch):
    """``_minimize`` refines partitions Hopcroft's way instead of one Moore
    round (a sort of the whole table) per distinguishing step.  Every grammar
    here, JSON schemas with ``maxLength``, enums, nested objects, the
    recursive ``json_object`` rules and the Muse and Qwen tool-call wires,
    must compile to exactly the automaton the textbook refinement gives, and
    so to the same token masks."""
    corpus = _minimization_corpus()
    merged = []

    def reference(table, accepting):
        result = _reference_minimize(table, accepting)
        merged.append(result[0].shape[0] < table.shape[0])
        return result

    pieces = _synthetic_pieces()
    trie = sa.TokenTrie(pieces, [0])
    for name, source in corpus.items():
        fast = compile_pattern(source)
        with monkeypatch.context() as patch:
            patch.setattr(sa, "_minimize", reference)
            oracle = compile_pattern(source)
        assert fast.table.dtype == oracle.table.dtype == np.int32, name
        np.testing.assert_array_equal(fast.table, oracle.table, err_msg=name)
        assert fast.accepting == oracle.accepting, name
        assert fast.final == oracle.final, name
        assert fast.calls == oracle.calls, name
        assert fast.first == oracle.first and fast.rule_start == oracle.rule_start, name
        for document in DOCUMENTS.get(name, ()):
            for cut in range(len(document) + 1):
                config = fast.advance(fast.start, document[:cut])
                assert config == oracle.advance(oracle.start, document[:cut])
                if config is not None:
                    np.testing.assert_array_equal(
                        trie.allowed(fast, config), trie.allowed(oracle, config)
                    )
    # The corpus has to exercise merging, not only already-minimal tables.
    assert len(merged) > len(corpus) and sum(merged) >= 8, merged


def test_minimization_equals_textbook_refinement_on_random_tables():
    """Random partial tables add what grammars rarely produce: unreachable
    states, real dead-end states (which stay distinct from the implicit dead
    state -1), all-accepting and all-rejecting automata."""
    rng = np.random.default_rng(20260923)
    cases = [
        (np.full((1, 1), -1, dtype=np.int32), np.array([False])),
        (np.full((1, 3), -1, dtype=np.int32), np.array([True])),
        (np.zeros((3, 2), dtype=np.int32), np.ones(3, dtype=bool)),
        (np.full((4, 2), -1, dtype=np.int32), np.array([False, True, False, True])),
    ]
    for _ in range(1500):
        count = int(rng.integers(1, 48))
        symbols = int(rng.integers(1, 6))
        table = rng.integers(0, int(rng.integers(1, count + 1)), size=(count, symbols)).astype(np.int32)
        table[rng.random((count, symbols)) < rng.random()] = -1
        cases.append((table, rng.random(count) < rng.random()))
    # A long chain and an equivalent copy of it reached from the start state:
    # the copy merges into the original only after ~length rounds of Moore.
    length = 300
    chain = np.full((length, 3), -1, dtype=np.int32)
    chain[:-1, 0] = np.arange(1, length)
    chain[::7, 1] = 1
    copy = np.where(chain >= 0, chain + length, -1)
    table = np.vstack((chain, copy)).astype(np.int32)
    table[0, 2] = length
    accepting = np.zeros(2 * length, dtype=bool)
    accepting[[length - 1, 2 * length - 1]] = True
    cases.append((table, accepting))
    for table, accepting in cases:
        got_table, got_accepting = sa._minimize(table, accepting)
        want_table, want_accepting = _reference_minimize(table, accepting)
        assert got_table.dtype == np.int32
        np.testing.assert_array_equal(got_table, want_table)
        np.testing.assert_array_equal(got_accepting, want_accepting)
    assert sa._minimize(table, accepting)[0].shape[0] == length + 1


def test_long_bounded_strings_compile_fast():
    """A string ``maxLength`` of n unrolls into a chain n states deep, and
    Moore refinement took one whole-table sort per chain step: a
    ``maxLength: 4096`` JSON schema compiled in 9.2 s (8.3 s minimizing) and
    a Muse strict tool with two string parameters in 34 s, all spent by the
    requesting client before its first token.  Both now compile in 0.05 s and
    0.25 s on the same host.  Each bound sits ~6x under the old time and
    20-30x over the new one, room for a host busy with parallel suites."""
    from mlx2.adapters.muse_glimmer_output import constrained_tool_grammar as muse_grammar

    tools = [{"type": "function", "function": {"name": "write", "strict": True, "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
        "additionalProperties": False,
    }}}]
    cases = [
        (compile_constraint(_bounded_string_schema(4096)).pattern.pattern, 4096, 1.5),
        (muse_grammar(tools, "required", parallel_tool_calls=False), 2 * 4096, 5.0),
    ]
    for source, depth, bound in cases:
        started = time.perf_counter()
        automaton = compile_pattern(source)
        elapsed = time.perf_counter() - started
        assert automaton.state_count > depth  # the unrolled chain is there
        assert automaton.fullmatch(source[:0], partial=True)
        assert elapsed < bound, (source[:40], elapsed)
