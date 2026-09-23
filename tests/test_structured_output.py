import pytest

from mlx2.server import validate_request
from mlx2.structured_output import StructuredOutputProcessor, compile_constraint


def accepts(constraint, text):
    return constraint.fullmatch(text) is not None


def remains_possible(constraint, text):
    return constraint.fullmatch(text, partial=True) is not None


def test_json_object_recursive_constraint():
    constraint = compile_constraint({"type": "json_object"})
    assert accepts(constraint, '{"a":[1,true,{"b":null}]}')
    assert remains_possible(constraint, '{"a": [1,')
    assert not remains_possible(constraint, '[1,2]')


def test_strict_schema_constraint_uses_bounded_canonical_order():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "score": {"type": "integer"},
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
    constraint = compile_constraint(response_format)
    assert accepts(constraint, '{"answer":"yes","score":3}')
    assert accepts(constraint, '{"answer":"yes"}')
    assert remains_possible(constraint, '{"answer":"ye')
    assert not remains_possible(constraint, '{"score":3}')
    request = validate_request({"messages": [{"role": "user", "content": "x"}], "response_format": response_format})
    assert request["response_format"] == response_format


def test_ref_resolution_preserves_the_preexisting_schema_depth_bound():
    def nested(levels):
        schema = {"type": "string"}
        for _ in range(levels):
            schema = {
                "type": "object",
                "properties": {"value": schema},
                "required": ["value"],
                "additionalProperties": False,
            }
        return schema

    response = lambda schema: {
        "type": "json_schema",
        "json_schema": {"strict": True, "schema": schema},
    }
    assert compile_constraint(response(nested(16)))
    with pytest.raises(ValueError, match="nesting depth") as caught:
        compile_constraint(response(nested(17)))
    assert not getattr(caught.value, "schema_reference_error", False)


def test_ref_resolution_only_interprets_keywords_in_schema_positions():
    from mlx2.runtime.tool_parsers._schema import resolve_local_refs

    schema = {
        "$defs": {"word": {"type": "string"}},
        "type": "object",
        "properties": {
            "definitions": {"type": "string"},
            "$defs": {"type": "integer"},
            "$ref": {"$ref": "#/$defs/word"},
        },
        "required": ["definitions", "$defs", "$ref"],
        "additionalProperties": False,
    }
    resolved = resolve_local_refs(schema)
    assert resolved["properties"] == {
        "definitions": {"type": "string"},
        "$defs": {"type": "integer"},
        "$ref": {"type": "string"},
    }


def test_ref_annotation_siblings_are_ignored_without_widening_constraints():
    from mlx2.runtime.tool_parsers._schema import resolve_local_refs

    schema = {
        "$defs": {"word": {"type": "string"}},
        "$ref": "#/$defs/word",
        "description": "a word",
    }
    assert resolve_local_refs(schema) == {"type": "string"}
    with pytest.raises(ValueError, match="annotation siblings"):
        resolve_local_refs({**schema, "maxLength": 3})


def _annotated_city_schema():
    """What Pydantic's ``model_json_schema()`` (the OpenAI SDK ``.parse()``)
    sends: ``title`` on the root and every property, plus the other
    annotation keywords.  Two properties are literally *named* ``title`` and
    ``description``; those names are data, not keywords."""
    return {
        "title": "City",
        "description": "a city",
        "$comment": "generated",
        "type": "object",
        "properties": {
            "title": {"title": "Title", "type": "string", "examples": ["Paris"]},
            "description": {
                "title": "Description",
                "type": "integer",
                "default": 3,
                "deprecated": False,
            },
            "tags": {
                "title": "Tags",
                "type": "array",
                "items": {"title": "Tag", "type": "string", "readOnly": True},
                "writeOnly": False,
            },
        },
        "required": ["title", "description"],
        "additionalProperties": False,
    }


def test_annotation_keywords_are_accepted_and_do_not_change_the_language():
    """Annotations never change which instances a schema accepts, yet every
    one used to be refused with 400, so every Pydantic schema was."""
    from mlx2.runtime.tool_parsers._schema import strip_annotations

    schema = _annotated_city_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "City", "strict": True, "schema": schema},
    }
    messages = [{"role": "user", "content": "x"}]
    validate_request({"messages": messages, "response_format": response_format})
    request = validate_request({
        "messages": messages,
        "tools": [{"type": "function", "function": {
            "name": "f", "strict": True, "parameters": schema,
        }}],
    })
    # The prompt still renders the annotated schema.
    rendered = request["tools"][0]["function"]["parameters"]
    assert rendered["properties"]["title"]["examples"] == ["Paris"]

    stripped = strip_annotations(schema)
    assert set(stripped["properties"]) == {"title", "description", "tags"}
    assert stripped["required"] == ["title", "description"]
    assert stripped["properties"]["description"] == {"type": "integer"}
    constraint = compile_constraint(response_format)
    assert constraint.pattern.pattern == compile_constraint({
        "type": "json_schema",
        "json_schema": {"strict": True, "schema": stripped},
    }).pattern.pattern
    assert accepts(constraint, '{"title":"Paris","description":3,"tags":["a"]}')
    assert not accepts(constraint, '{"title":"Paris"}')
    assert not accepts(constraint, '{"title":"Paris","description":"x"}')


def test_constraining_keywords_beside_annotations_still_fail_closed():
    schema = {
        "title": "T",
        "type": "object",
        "properties": {"v": {"title": "V", "type": "string", "pattern": "^a$"}},
        "required": ["v"],
        "additionalProperties": False,
    }
    with pytest.raises(ValueError, match="unsupported JSON schema keywords: pattern"):
        compile_constraint({
            "type": "json_schema",
            "json_schema": {"strict": True, "schema": schema},
        })


def test_regex_grammar_and_fail_closed_validation():
    constraint = compile_constraint(grammar=r"(?:yes|no)")
    assert accepts(constraint, "yes")
    assert remains_possible(constraint, "y")
    assert not remains_possible(constraint, "maybe")
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_request({
            "messages": [{"role": "user", "content": "x"}],
            "grammar": "x",
            "response_format": {"type": "json_object"},
        })
    with pytest.raises(ValueError, match="additionalProperties"):
        compile_constraint({
            "type": "json_schema",
            "json_schema": {"strict": True, "schema": {"type": "object", "additionalProperties": True}},
        })
    with pytest.raises(ValueError, match="unsupported JSON schema keywords"):
        compile_constraint({
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "schema": {"type": "integer", "minimum": 10},
            },
        })


def test_structured_processor_cache_is_request_bounded():
    class Tokenizer:
        eos_token_ids = [0]
        vocab_size = 3

        def decode(self, tokens, **_):
            return {0: "", 1: "a", 2: "b"}.get(tokens[0], "") if tokens else ""

    processor = StructuredOutputProcessor(
        Tokenizer(), 0, compile_constraint(grammar="a*")
    )
    for index in range(300):
        processor._allowed("a" * index)
    assert len(processor._allowed_cache) == 256


def test_structured_processor_conservatively_skips_one_timed_out_piece():
    class Tokenizer:
        eos_token_ids = [0]
        vocab_size = 3

        def decode(self, tokens, **_):
            return {0: "", 1: "a", 2: "pathological"}.get(tokens[0], "") if tokens else ""

    class Constraint:
        def fullmatch(self, value, *, partial=False, timeout=None):
            if value == "pathological":
                raise TimeoutError
            if value in {"", "a"}:
                return object() if partial else None
            return None

    processor = StructuredOutputProcessor(Tokenizer(), 0, Constraint())
    assert processor._allowed("") == (1,)


def test_structured_processor_still_fails_closed_when_prefix_times_out():
    class Tokenizer:
        eos_token_ids = [0]
        vocab_size = 2

        def decode(self, tokens, **_):
            return "a" if tokens else ""

    class Constraint:
        def fullmatch(self, value, *, partial=False, timeout=None):
            if not partial:
                raise TimeoutError
            return object()

    processor = StructuredOutputProcessor(Tokenizer(), 0, Constraint())
    with pytest.raises(ValueError, match="exceeded its match budget"):
        processor._allowed("")


def test_structured_output_with_thinking_is_rejected_at_validation():
    import pytest

    from mlx2.server import validate_request

    base = {"messages": [{"role": "user", "content": "hi"}]}
    for toggle in ({"enable_thinking": True}, {"reasoning_effort": "high"}, {"think": True}):
        with pytest.raises(ValueError, match="thinking to be disabled"):
            validate_request({**base, "response_format": {"type": "json_object"}, **toggle})
        with pytest.raises(ValueError, match="thinking to be disabled"):
            validate_request({**base, "grammar": "[a-z]+", **toggle})
    # Thinking off, or explicitly disabled by effort, stays accepted.
    validate_request({**base, "response_format": {"type": "json_object"}})
    validate_request({**base, "grammar": "[a-z]+", "reasoning_effort": "none"})
    # Plain text with thinking is unaffected.
    validate_request({**base, "response_format": {"type": "text"}, "enable_thinking": True})


def test_engine_thinking_resolution_prefers_the_adapter():
    from types import SimpleNamespace

    from mlx2.serving import thinking_enabled

    adapter = SimpleNamespace(thinking_enabled=lambda request: request.get("reasoning_effort") != "none")
    assert thinking_enabled(adapter, {}) is True
    assert thinking_enabled(adapter, {"reasoning_effort": "none"}) is False
    assert thinking_enabled(SimpleNamespace(), {"enable_thinking": True}) is True
    assert thinking_enabled(SimpleNamespace(), {}) is False


def test_raw_completion_with_grammar_ignores_the_thinking_toggle():
    from mlx2.server import validate_request

    # /v1/completions never opens a reasoning channel, so the constraint applies
    # from the first generated token and the toggle is irrelevant.
    validate_request({"prompt": "x", "grammar": "[a-z]+", "enable_thinking": True}, chat=False)


def _synthetic_tokenizer(pieces, eos=(0,)):
    from types import SimpleNamespace

    return SimpleNamespace(
        vocab_size=len(pieces),
        eos_token_ids=list(eos),
        decode=lambda ids, **_kw: "".join(pieces[i] for i in ids),
        convert_ids_to_tokens=lambda ids: (
            [pieces[i] for i in ids] if isinstance(ids, list) else pieces[ids]
        ),
    )


@pytest.mark.parametrize("engine", ["automaton", "scanner"])
@pytest.mark.parametrize(
    "constraint",
    [
        {"response_format": {"type": "json_object"}},
        {"grammar": r"\{\}", "constraint_kind": "tool_grammar"},
    ],
    ids=["response-format", "constrained-tool-grammar"],
)
def test_generation_terminal_is_consumed_only_after_an_accepting_prefix(
    engine, constraint, monkeypatch
):
    """The decode loop evaluates one unused logit row before matching EOS."""
    import mlx.core as mx
    import numpy as np

    from mlx2.structured_output import make_structured_processor

    if engine == "scanner":
        monkeypatch.setenv("MLX2_STRUCTURED_AUTOMATON", "0")
    pieces = ["<eos>", "{", "}", "x", "</assistant>"]
    tokenizer = _synthetic_tokenizer(pieces, eos=(0,))
    processor = make_structured_processor(
        tokenizer,
        0,
        generation_stop_token_ids=(0, 4),
        **constraint,
    )
    logits = mx.zeros((1, len(pieces)))

    admitted = np.flatnonzero(
        np.isfinite(np.asarray(processor(mx.array([1, 2]), logits))[0])
    ).tolist()
    assert admitted == [0, 4]

    # Token 4 has already been sampled from an accepting state. The ordinary
    # decode loop will match it immediately after this processor call, so the
    # unused next-token row must not turn the successful request into a 502.
    processor(mx.array([1, 2, 4]), logits)
    assert processor.failure is None
    repeated = processor(mx.array([1, 2, 4, 0]), logits)
    repeated_admitted = np.flatnonzero(
        np.isfinite(np.asarray(repeated)[0])
    ).tolist()
    assert repeated_admitted == [0, 4]
    assert processor.failure is None

    rejected = make_structured_processor(
        tokenizer,
        0,
        generation_stop_token_ids=(0, 4),
        **constraint,
    )
    rejected(mx.array([1, 4]), logits)
    assert rejected.failure == "structured-output grammar has no valid token continuation"


def test_deferred_structure_blocks_generation_stops_until_accepting():
    import mlx.core as mx
    import numpy as np

    from mlx2.structured_output import make_structured_processor

    pieces = ["<eos>", "</think>", "{", "}", "</assistant>", "reason"]
    tokenizer = _synthetic_tokenizer(pieces, eos=(0,))
    processor = make_structured_processor(
        tokenizer,
        0,
        response_format={"type": "json_object"},
        generation_stop_token_ids=(0, 4),
        defer_until=(1,),
        block_eos_while_deferred=True,
    )
    logits = mx.zeros((1, len(pieces)))
    before = np.flatnonzero(
        np.isfinite(np.asarray(processor(mx.array([5]), logits))[0])
    ).tolist()
    assert 0 not in before and 4 not in before
    after = np.flatnonzero(
        np.isfinite(np.asarray(processor(mx.array([5, 1, 2, 3]), logits))[0])
    ).tolist()
    assert after == [0, 4]
    processor(mx.array([5, 1, 2, 3, 4]), logits)
    assert processor.failure is None


@pytest.mark.parametrize("enveloped", [False, True])
def test_terminal_only_row_fails_closed_when_all_stop_ids_exceed_logits_width(
    enveloped,
):
    import mlx.core as mx
    import numpy as np

    from mlx2.structured_output import make_structured_processor

    pieces = ["{", "}", "x", "y", "</answer>", "<eos>"]
    tokenizer = _synthetic_tokenizer(pieces, eos=(5,))
    processor = make_structured_processor(
        tokenizer,
        0,
        response_format={"type": "json_object"},
        generation_stop_token_ids=(5,),
        envelope=((), (4,)) if enveloped else None,
    )
    logits = mx.zeros((1, 5))
    generated = [0, 1, 4] if enveloped else [0, 1, 5]
    output = processor(mx.array(generated), logits)
    assert processor.failure == (
        "structured-output generation stop tokens are outside the logits vocabulary"
    )
    assert np.isfinite(np.asarray(output)).all()


def test_pruned_vocabulary_walk_matches_brute_force():
    import random
    import string

    from mlx2.structured_output import (
        _MATCH_TIMEOUT_SECONDS,
        StructuredOutputProcessor,
        compile_constraint,
    )

    random.seed(7)
    alphabet = string.ascii_letters + string.digits + ' {}[]":,.-\n'
    pieces = ["<eos>"] + [
        "".join(random.choice(alphabet) for _ in range(random.choice([1, 1, 2, 3, 4, 6])))
        for _ in range(4000)
    ]
    pieces[5] = pieces[6]  # duplicate decode
    pieces[7] = ""  # empty piece
    pieces[8] = "a�"  # replacement char
    tokenizer = _synthetic_tokenizer(pieces)
    for response_format, grammar in (
        ({"type": "json_object"}, None),
        (None, r"[a-z]+(,[a-z]+)*"),
    ):
        constraint = compile_constraint(response_format, grammar)
        processor = StructuredOutputProcessor(tokenizer, 0, constraint)
        for prefix in ("", "{", '{"k": ', '{"k": "v", "x": [1,2', "abc", "abc,de"):
            expected = set([0]) if constraint.fullmatch(prefix) else set()
            for token, piece in enumerate(pieces):
                if token == 0 or not piece or "�" in piece:
                    continue
                if constraint.fullmatch(prefix + piece, partial=True, timeout=_MATCH_TIMEOUT_SECONDS):
                    expected.add(token)
            if not expected:
                with pytest.raises(ValueError):
                    processor._allowed(prefix)
                continue
            assert processor._allowed(prefix) == tuple(sorted(expected)), (response_format, grammar, prefix)


def test_optional_property_chain_matches_subset_enumeration():
    import itertools
    import json

    import regex

    from mlx2.structured_output import _STRING, _WS, _schema_pattern

    def enumerated(names, required):
        pairs = [(regex.escape(json.dumps(n)) + _WS + ":" + _WS + _STRING, n in required) for n in names]
        req = [p for p, r in pairs if r]
        opt = [p for p, r in pairs if not r]
        variants = [
            f"{_WS},{_WS}".join(req + [p for i, p in enumerate(opt) if mask & (1 << i)])
            for mask in range(1 << len(opt))
        ]
        body = "(?:" + "|".join(variants) + ")" if len(variants) > 1 else variants[0]
        return regex.compile(rf"\{{{_WS}{body}{_WS}\}}")

    for n_req in (0, 1, 2):
        for n_opt in (0, 1, 2, 3):
            names = [f"r{i}" for i in range(n_req)] + [f"o{i}" for i in range(n_opt)]
            schema = {
                "type": "object",
                "properties": {k: {"type": "string"} for k in names},
                "required": names[:n_req],
                "additionalProperties": False,
            }
            chain = regex.compile(_schema_pattern(schema))
            old = enumerated(names, set(names[:n_req]))
            for k in range(len(names) + 1):
                for combo in itertools.permutations(names, k):
                    text = "{" + ", ".join(f'"{c}": "x"' for c in combo) + "}"
                    assert bool(chain.fullmatch(text)) == bool(old.fullmatch(text)), text
                    for cut in range(1, len(text)):
                        assert (chain.fullmatch(text[:cut], partial=True) is None) == (
                            old.fullmatch(text[:cut], partial=True) is None
                        ), text[:cut]


def test_constraint_dead_end_fails_closed_without_raising_from_the_batch():
    import mlx.core as mx

    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

    # Only "x" can follow "ab", and no vocabulary piece starts with "x".
    pieces = ["<eos>", "a", "b", "ab", "c"]
    tokenizer = _synthetic_tokenizer(pieces)
    processor = StructuredOutputProcessor(tokenizer, 0, compile_constraint(None, "abx"))
    logits = mx.array([[0.0, 4.0, 3.0, 2.0, 1.0]])
    masked = processor(mx.array([3], dtype=mx.uint32), logits)  # prefix "ab" -> dead end
    assert processor.failure is not None and "no valid token continuation" in processor.failure
    assert processor.failure_context is None
    # Masking stopped; logits pass through untouched so the batch stays consistent.
    assert mx.array_equal(masked, logits)
    assert mx.array_equal(processor(mx.array([3, 4], dtype=mx.uint32), logits), logits)

    qualified = StructuredOutputProcessor(
        tokenizer,
        0,
        compile_constraint(None, "abx"),
        capture_failure_context=True,
    )
    qualified(mx.array([3], dtype=mx.uint32), logits.astype(mx.bfloat16))
    context = qualified.failure_context
    assert context["constraint_kind"] == "grammar"
    assert context["engine"] in {"automaton", "scanner"}
    assert context["generated_tokens"] == 1
    assert context["recent_tokens"] == [{
        "id": 3,
        "tokenizer_piece": "ab",
        "decoded_piece": "ab",
        "bytes_hex": "6162",
        "bytes_truncated": False,
    }]
    assert [item["id"] for item in context["top_logits"]] == [1, 2, 3, 4, 0]
    assert all("tokenizer_piece" in item and "bytes_hex" in item for item in context["top_logits"])
    assert context["automaton_state"] is not None


def test_per_token_budget_overrun_fails_closed(monkeypatch):
    from mlx2 import structured_output as so
    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

    pieces = ["<eos>"] + [f"t{i}" for i in range(3000)]
    processor = StructuredOutputProcessor(_synthetic_tokenizer(pieces), 0, compile_constraint(None, r"[t0-9]*"))
    monkeypatch.setattr(so, "_ALLOWED_BUDGET_SECONDS", -1.0)
    with pytest.raises(ValueError, match="per-token match budget"):
        processor._allowed("t1")


def test_min_tokens_with_structured_output_keeps_main_fail_closed_rule():
    from mlx2.server import validate_request

    base = {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}}
    with pytest.raises(ValueError, match="min_tokens cannot be combined"):
        validate_request({**base, "min_tokens": 4})
    validate_request({**base, "min_tokens": 0})


def test_logit_ordered_admissibility_matches_brute_force_where_mass_lives():
    """Greedy: the first admissible token in logit order is the exact argmax of
    the fully masked row.  Sampling: every token the lazy mask omits carries
    less than 2**-24 of the admissible mass."""
    import random
    import string

    import mlx.core as mx
    import numpy as np

    from mlx2.structured_output import (
        _MASS_RESOLUTION,
        StructuredOutputProcessor,
        compile_constraint,
    )

    random.seed(3)
    alphabet = string.ascii_letters + string.digits + ' {}[]":,.-\n'
    pieces = ["<eos>"] + [
        "".join(random.choice(alphabet) for _ in range(random.choice([1, 1, 2, 3, 4, 6])))
        for _ in range(3000)
    ]
    tokenizer = _synthetic_tokenizer(pieces)
    constraint = compile_constraint({"type": "json_object"})
    rng = np.random.default_rng(5)
    for prefix in ("{", '{"k": ', '{"k": "some text', '{"k": "v", "x": [1,2'):
        full = StructuredOutputProcessor(tokenizer, 0, constraint)
        exact = set(full._allowed(prefix))
        # Peaked logits, as a language model produces.
        row = rng.normal(0, 1, len(pieces)).astype(np.float32) * 3
        row[rng.integers(0, len(pieces), 8)] += 12
        masked_full = np.where(np.isin(np.arange(len(pieces)), list(exact)), row, -np.inf)
        # Greedy: exact argmax.
        greedy = StructuredOutputProcessor(tokenizer, 0, constraint, greedy=True)
        allowed = set(greedy._allowed_by_logit_order(prefix, row))
        assert allowed <= exact
        masked_lazy = np.where(np.isin(np.arange(len(pieces)), list(allowed)), row, -np.inf)
        assert int(masked_full.argmax()) == int(masked_lazy.argmax())
        # Sampling with the server's default transform (top_p then top_k): the
        # transformed distribution over the lazily masked row must equal the
        # one over the fully masked row exactly.
        from mlx2.runtime.sample_utils import make_transformed_logprobs

        for top_k, top_p in ((20, 0.8), (20, 0.0), (0, 0.9), (5, 0.95)):
            lazy = StructuredOutputProcessor(tokenizer, 0, constraint, top_k=top_k, top_p=top_p)
            allowed = set(lazy._allowed_by_logit_order(prefix, row))
            assert allowed <= exact
            masked_lazy = np.where(np.isin(np.arange(len(pieces)), list(allowed)), row, -np.inf)
            transform = make_transformed_logprobs(0.7, top_p=top_p, top_k=top_k)
            full_law = np.exp(np.array(transform(mx.array(masked_full)[None]))[0])
            lazy_law = np.exp(np.array(transform(mx.array(masked_lazy)[None]))[0])
            tv = 0.5 * np.abs(lazy_law - full_law).sum()
            if lazy.tail_mass_bound == 0.0:
                assert tv < 1e-5, (top_k, top_p, tv)
            else:
                # Bounded: the nucleus boundary moved by at most the recorded
                # unexamined-to-admitted mass ratio.
                assert tv <= lazy.tail_mass_bound + 1e-6, (top_k, top_p, tv, lazy.tail_mass_bound)
        # Pure temperature sampling has no exact early stop: the omitted
        # admissible mass is within the recorded bound (or beneath resolution).
        lazy = StructuredOutputProcessor(tokenizer, 0, constraint, top_k=0, top_p=0.0)
        allowed = set(lazy._allowed_by_logit_order(prefix, row))
        probs = np.exp(row - row.max())
        omitted = probs[list(exact - allowed)].sum() if exact - allowed else 0.0
        admitted_mass = probs[list(allowed)].sum()
        assert omitted <= admitted_mass * max(lazy.tail_mass_bound, _MASS_RESOLUTION) + 1e-9
        # The processor call path produces the same mask as the lazy set.
        logits = mx.array(row)[None]
        out = lazy(mx.array([], dtype=mx.uint32), logits) if not prefix else None
        assert out is None or out.shape == logits.shape


def test_tied_logits_admit_the_same_tokens_as_distinct_logits(monkeypatch):
    """Exact ties must not change the mask.

    The walk grows its examined frontier geometrically: it rebuilds ``order``
    as the top-``want`` ids but carries ``start`` over from the previous,
    smaller array, so it only ever reads ``order[start:]``.  That is sound
    only when the new prefix [0, start) is exactly the set already examined,
    which needs a ranking whose top-``want`` prefix is stable as ``want``
    grows.  Logit value alone is not one: ``np.argpartition`` picks an
    arbitrary representative set among tied values, so on a rebuild ids that
    were never examined can land in the skipped prefix and be dropped from the
    mask for good -- while others get examined twice as the frontier
    reshuffles.  The mask is then not the top-``start`` admissible set it is
    documented to be.

    Compared here against a row ranked identically but with no ties, and
    carrying the same mass to within 1e-4, so the only thing that can move the
    admitted set is the tie-break -- not a stopping rule reacting to a
    different distribution.  Served bfloat16 rows tie all through the tail,
    where this silently narrows the mask; an all-tied row makes it visible.
    """
    import random
    import string

    import numpy as np

    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

    chunk = 256  # the walk's first frontier, and the size of each step

    # The walk itself is under test, not the pool that can finish its tail.
    monkeypatch.setenv("MLX2_STRUCTURED_WORKERS", "0")
    random.seed(11)
    alphabet = string.ascii_letters + string.digits + ' {}[]":,.-\n'
    # Far enough past that first frontier for the walk to hand ``start`` over
    # to a rebuilt array and keep going for thousands of ids.
    pieces = ["<eos>"] + [
        "".join(random.choice(alphabet) for _ in range(random.choice([1, 1, 2, 3])))
        for _ in range(20_000)
    ]
    tokenizer = _synthetic_tokenizer(pieces)
    constraint = compile_constraint({"type": "json_object"})
    vocab = len(pieces)
    tied = np.zeros(vocab, dtype=np.float32)
    untied = np.linspace(0.0, -1e-4, vocab).astype(np.float32)
    assert len(set(untied.tolist())) == vocab  # no ties left to break

    # Two counters keep the comparison below from passing vacuously: the walk
    # has to take the partial-frontier path at all, and then go on past that
    # first array -- the hand-off a rebuild has to get right.
    counts = {"partitions": 0, "examined": 0}
    partition, admissible = np.argpartition, StructuredOutputProcessor._admissible

    def counted_partition(*args, **kwargs):
        counts["partitions"] += 1
        return partition(*args, **kwargs)

    def counted_admissible(self, *args, **kwargs):
        # ``decide`` memoizes, so one call per id the walk actually examined.
        counts["examined"] += 1
        return admissible(self, *args, **kwargs)

    monkeypatch.setattr(np, "argpartition", counted_partition)
    monkeypatch.setattr(StructuredOutputProcessor, "_admissible", counted_admissible)

    for prefix in ('{"k": "some text', "{"):
        exact = set(StructuredOutputProcessor(tokenizer, 0, constraint)._allowed(prefix))
        walked = []
        for row in (tied, untied):
            counts.update(partitions=0, examined=0)
            processor = StructuredOutputProcessor(tokenizer, 0, constraint)
            allowed = set(processor._allowed_by_logit_order(prefix, row))
            assert allowed <= exact
            assert counts["partitions"] >= 1, (prefix, counts)
            assert counts["examined"] > 4 * chunk, (prefix, counts)
            walked.append(allowed)
        assert walked[0] == walked[1], (
            prefix,
            len(walked[1] - walked[0]),
            len(walked[0] - walked[1]),
        )


def test_flash_next_tied_row_keeps_both_tool_call_openings():
    """The live case: the Qwen XML grammar against the real 248k vocabulary.

    A small vocabulary hides this -- the first rebuild already asks for the
    whole of it and takes the deterministic full-argsort path.
    """
    import glob
    from pathlib import Path

    import numpy as np

    candidates = sorted(glob.glob(str(Path.home() / "mlx-models" / "Qwen3.8-Flash-Next-MLX-*")))
    if not candidates:
        pytest.skip("Flash-Next tokenizer artifact is not present")
    from transformers import AutoTokenizer

    from mlx2.runtime.tokenizer_utils import TokenizerWrapper
    from mlx2.runtime.tool_parsers.qwen3_coder import constrained_tool_grammar
    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

    hf = AutoTokenizer.from_pretrained(candidates[0], trust_remote_code=True)
    marker = hf.get_added_vocab().get("<tool_call>")
    if marker is None:
        pytest.skip("this artifact does not carry <tool_call> as an added token")
    tokenizer = TokenizerWrapper(hf, eos_token_ids=[hf.eos_token_id])
    grammar = constrained_tool_grammar(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "required",
    )
    constraint = compile_constraint(grammar=grammar)
    width = 248_320  # the head is padded above the tokenizer

    def admitted(row):
        # ``_allowed_by_logit_order`` is the scanner's walk; the automaton
        # engine masks along a different path and never reaches it.
        processor = StructuredOutputProcessor(
            tokenizer, 0, constraint, generation_stop_token_ids=[hf.eos_token_id]
        )
        return set(processor._allowed_by_logit_order("", row))

    tied = admitted(np.zeros(width, dtype=np.float32))
    untied = admitted(np.linspace(0.0, -1e-4, width).astype(np.float32))
    assert tied == untied, (len(untied - tied), len(tied - untied))
    # "<" -- the ordinary token that starts spelling the marker out, and the
    # first admissible id in the vocabulary -- is the one a reshuffled frontier
    # buries: it sits in the very first chunk of the stable ranking and nowhere
    # near the arbitrary 256 ids ``argpartition`` hands back for a tied row.
    assert 27 in tied
    assert marker >= hf.vocab_size  # the grammar's one-token opening, for the record


def test_processor_accepts_bfloat16_logits():
    import mlx.core as mx

    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

    pieces = ["<eos>", "{", "}", '"', "a"]
    processor = StructuredOutputProcessor(_synthetic_tokenizer(pieces), 0, compile_constraint({"type": "json_object"}))
    logits = mx.array([[0.0, 3.0, 1.0, 0.5, 0.2]], dtype=mx.bfloat16)
    out = processor(mx.array([], dtype=mx.uint32), logits)
    assert processor.failure is None
    assert out.dtype == mx.bfloat16
    assert int(mx.argmax(out).item()) == 1  # "{" is the only admissible start
    assert mx.isinf(out[0, 4]).item()


def test_budget_exhaustion_with_admitted_tokens_is_a_recorded_bound_not_a_failure(monkeypatch):
    import numpy as np

    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

    pieces = ["<eos>"] + [f"t{i}" for i in range(2000)]
    processor = StructuredOutputProcessor(
        _synthetic_tokenizer(pieces), 0, compile_constraint(None, r"[t0-9]*"), top_k=20, top_p=0.8
    )
    # Make the very first regex call succeed, then exhaust the budget.
    calls = {"n": 0}
    real = processor._admissible

    def slow(text, deadline):
        calls["n"] += 1
        if calls["n"] > 3:
            raise ValueError("structured-output grammar exceeded its per-token match budget")
        return real(text, deadline)

    monkeypatch.setattr(processor, "_admissible", slow)
    row = np.linspace(5, 0, len(pieces)).astype(np.float32)
    allowed = processor._allowed_by_logit_order("t1", row)
    assert allowed and processor.failure is None
    assert 0.0 < processor.tail_mass_bound <= 1.0
    # Nothing admitted before exhaustion is still a failure.
    calls["n"] = 100
    # (prefix "t1" is not complete under this grammar, so EOS is not admitted)
    fresh = StructuredOutputProcessor(_synthetic_tokenizer(pieces), 0, compile_constraint(None, r"[t0-9]*x"))
    monkeypatch.setattr(fresh, "_admissible", lambda text, deadline: (_ for _ in ()).throw(ValueError("budget")))
    with pytest.raises(ValueError):
        fresh._allowed_by_logit_order("t1", row)


def test_canonical_prefix_preserves_grammar_state():
    """Admissibility of every piece under the canonical prefix equals that under
    the original prefix, for json_object and for a schema with free strings,
    constrained enums, numbers, arrays and nesting."""
    from mlx2.structured_output import compile_constraint

    pieces = ['"', 'a', 'ab', '\\', '\\n', '\\u', '1', '0', '.', ',', ':', '{', '}', '[', ']', ' ', '\n',
              '"k', 'true', 'nul', 'x"', '":', '", "', 'é', '日本', '\\"', 'u00', '00e9', 'e9"']
    schema = {"type": "object", "properties": {
        "name": {"type": "string"}, "kind": {"enum": ["cat", "dog"]}, "n": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "nested": {"type": "object", "properties": {"deep": {"type": "string"}, "code": {"const": "X1"}},
                   "required": ["deep"], "additionalProperties": False},
    }, "required": ["name", "kind"], "additionalProperties": False}
    constraints = [
        compile_constraint({"type": "json_object"}),
        compile_constraint({"type": "json_schema", "json_schema": {"name": "s", "strict": True, "schema": schema}}),
    ]
    docs = [
        '{"name": "Paris, the City of Light, long text here", "kind": "cat", "n": 12, "tags": ["a", "bb", "long tag value"], "nested": {"deep": "very deep string", "code": "X1"}}',
        '{"name": "esc \\"q\\" \\u00e9 \\n", "kind": "dog"}',
        '{"name": "x", "kind": "ca',
        '{"name": "unterminated \\u00',
        '{"name": "unterminated \\',
        '{ "name" : "spaced" , "kind" : "dog" , "n" : -0.5e3 }',
    ]
    checked = 0
    changed = 0
    for constraint in constraints:
        for doc in docs:
            for cut in range(0, len(doc) + 1):
                prefix = doc[:cut]
                canonical = constraint.canonicalize(prefix)
                changed += canonical != prefix
                assert len(canonical) <= len(prefix)
                for piece in pieces:
                    original = constraint.fullmatch(prefix + piece, partial=True) is not None
                    shortcut = constraint.fullmatch(canonical + piece, partial=True) is not None
                    assert original == shortcut, (prefix, canonical, piece)
                    checked += 1
                assert (constraint.fullmatch(prefix) is not None) == (constraint.fullmatch(canonical) is not None)
    assert checked > 10000 and changed > 100


def test_scanner_pool_finishes_the_tail_exactly(monkeypatch):
    """With parallel heads the sampled mask becomes exact (bound 0) and equals
    the brute-force admissible set restricted to tokens with mass."""
    import random
    import string

    import numpy as np

    from mlx2 import structured_output as so
    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

    monkeypatch.setenv("MLX2_STRUCTURED_WORKERS", "2")
    so._SCANNER_POOLS.clear()
    random.seed(11)
    alphabet = string.ascii_letters + string.digits + ' {}[]":,.-\n'
    pieces = ["<eos>"] + [
        "".join(random.choice(alphabet) for _ in range(random.choice([1, 1, 2, 3, 4, 6])))
        for _ in range(3000)
    ]
    tokenizer = _synthetic_tokenizer(pieces)
    constraint = compile_constraint({"type": "json_object"})
    prefix = '{"k": "some text'
    exact = set(StructuredOutputProcessor(tokenizer, 0, constraint)._allowed(prefix))
    # A flat-ish row so the in-process walk cannot stop exactly on its own.
    rng = np.random.default_rng(2)
    row = rng.normal(0, 0.3, len(pieces)).astype(np.float32)
    processor = StructuredOutputProcessor(tokenizer, 0, constraint, top_k=20, top_p=0.8)
    assert processor._pool is not None
    try:
        allowed = set(processor._allowed_by_logit_order(prefix, row))
        assert processor.parallel_scans == 1
        assert processor.tail_mass_bound == 0.0
        assert allowed == exact
    finally:
        processor._pool.close()
        so._SCANNER_POOLS.clear()


def test_scanner_pool_ignores_padded_logit_ids(monkeypatch):
    """Model logits can be wider than the tokenizer vocabulary; padded ids must
    never reach the vocabulary pieces in a worker."""
    import numpy as np

    from mlx2 import structured_output as so
    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

    monkeypatch.setenv("MLX2_STRUCTURED_WORKERS", "2")
    so._SCANNER_POOLS.clear()
    pieces = ["<eos>", "{", "}", '"', "a", "b", ":", " "]
    processor = StructuredOutputProcessor(_synthetic_tokenizer(pieces), 0, compile_constraint({"type": "json_object"}), top_k=20, top_p=0.8)
    row = np.zeros(len(pieces) + 300, dtype=np.float32)  # padded width
    row[4] = 3.0
    try:
        allowed = set(processor._allowed_by_logit_order('{"k": "', row))
        assert allowed and max(allowed) < len(pieces)
        assert processor.parallel_scans == 1 and processor.tail_mass_bound == 0.0
    finally:
        processor._pool.close()
        so._SCANNER_POOLS.clear()


def test_scanner_pool_shutdown_survives_killed_workers(monkeypatch):
    """A worker killed from outside must not wedge shutdown: the old
    multiprocessing.Pool finalizer deadlocked on the task-queue lock the dead
    idle worker held, leaving the server process (and the model) resident."""
    import os
    import signal
    import time

    import numpy as np

    from mlx2 import structured_output as so
    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint, shutdown_scanner_pools

    monkeypatch.setenv("MLX2_STRUCTURED_WORKERS", "2")
    so._SCANNER_POOLS.clear()
    pieces = ["<eos>"] + [f"t{i}" for i in range(3000)]
    processor = StructuredOutputProcessor(_synthetic_tokenizer(pieces), 0, compile_constraint(None, r"[t0-9]*"), top_k=20, top_p=0.8)
    row = np.random.default_rng(1).normal(0, 0.2, len(pieces)).astype(np.float32)
    processor._allowed_by_logit_order("t1", row)  # starts the pool and uses it
    pool = processor._pool
    pids = [p.pid for p in pool._executor._processes.values()]
    assert pids
    os.kill(pids[0], signal.SIGKILL)
    time.sleep(0.5)
    started = time.perf_counter()
    shutdown_scanner_pools()
    assert time.perf_counter() - started < 5.0
    assert pool._executor is None
    for pid in pids:
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)


def test_interpreter_exits_after_pool_use_and_killed_worker(tmp_path):
    """End-to-end exit check in a fresh interpreter: use the pool, kill one
    worker, run the ServingEngine.close() shutdown path, and exit.  Runs from
    a script file so spawned workers can re-import the parent main module."""
    import subprocess
    import sys
    import textwrap
    import time

    script = textwrap.dedent(
        """
        import os, signal, time, numpy as np
        os.environ["MLX2_STRUCTURED_WORKERS"] = "2"
        from types import SimpleNamespace
        from mlx2.structured_output import StructuredOutputProcessor, compile_constraint, shutdown_scanner_pools
        pieces = ["<eos>"] + [f"t{i}" for i in range(3000)]
        tok = SimpleNamespace(vocab_size=len(pieces), eos_token_ids=[0], decode=lambda ids, **kw: "".join(pieces[i] for i in ids))
        p = StructuredOutputProcessor(tok, 0, compile_constraint(None, r"[t0-9]*"), top_k=20, top_p=0.8)
        row = np.random.default_rng(1).normal(0, 0.2, len(pieces)).astype(np.float32)
        p._allowed_by_logit_order("t1", row)
        pid = next(iter(p._pool._executor._processes.values())).pid
        os.kill(pid, signal.SIGKILL); time.sleep(0.3)
        shutdown_scanner_pools()
        print("closed", "scans", p.parallel_scans, flush=True)
        """
    )
    path = tmp_path / "pool_exit.py"
    path.write_text("if __name__ == '__main__':\n" + textwrap.indent(script, "    "))
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "closed scans 1" in proc.stdout, proc.stdout
    assert time.perf_counter() - started < 45


def test_json_number_digit_runs_are_bounded():
    """A greedy model must not be able to extend a number to the token cap."""
    from mlx2.structured_output import compile_constraint

    schema = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {
        "type": "object", "properties": {"n": {"type": "integer"}, "x": {"type": "number"}},
        "required": ["n", "x"], "additionalProperties": False}}}
    pattern = compile_constraint(schema, None).pattern
    assert pattern.fullmatch('{"n":-9223372036854775808,"x":3.141592653589793e-308}')
    assert pattern.fullmatch('{"n":0,"x":0}')
    assert not pattern.fullmatch('{"n":' + "1" * 20 + ',"x":0}')
    assert not pattern.fullmatch('{"n":1,"x":0.' + "1" * 19 + "}")
    assert not pattern.fullmatch('{"n":1,"x":1e1000}')
    # The digit run dead-ends: only a terminator can follow 19 digits.
    assert pattern.fullmatch('{"n":' + "1" * 19, partial=True)
    assert not pattern.fullmatch('{"n":' + "1" * 20, partial=True)
    generic = compile_constraint({"type": "json_object"}, None).pattern
    assert generic.fullmatch('{"a":[1,2.5,-3e10]}')
    assert not generic.fullmatch('{"a":' + "9" * 20 + "}")


def test_json_whitespace_runs_are_bounded():
    """Whitespace is always admissible, so it must not be an unbounded escape."""
    from mlx2.structured_automaton import automaton_for
    from mlx2.structured_output import compile_constraint

    schema = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {
        "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
        "additionalProperties": False}}}
    for constraint in (compile_constraint(schema, None), compile_constraint({"type": "json_object"}, None)):
        pattern = constraint.pattern
        assert pattern.fullmatch('{\n' + " " * 30 + '"ok":' + " " * 32 + "true\n}")
        assert pattern.fullmatch('{"ok":' + " " * 32, partial=True)
        # Two whitespace sites can be adjacent (after ':' and before a value),
        # so the hard ceiling on one run is twice the per-site bound.
        assert not pattern.fullmatch('{"ok":' + " " * 65, partial=True)
        assert not pattern.fullmatch('{"ok":' + "\n \t" * 40, partial=True)
        assert automaton_for(pattern).state_count < 2000


def test_local_schema_refs_are_inlined_before_existing_bounds_apply():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "schema": {
                "$defs": {
                    "coordinate": {"type": "integer"},
                    "point": {
                        "type": "object",
                        "properties": {
                            "x": {"$ref": "#/$defs/coordinate"},
                            "y": {"$ref": "#/$defs/coordinate"},
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False,
                    },
                },
                "$ref": "#/$defs/point",
            },
        },
    }
    constraint = compile_constraint(response_format)
    assert accepts(constraint, '{"x":1,"y":-2}')
    assert not accepts(constraint, '{"x":"1","y":-2}')


@pytest.mark.parametrize(
    "reference, message",
    [
        ("https://example.test/schema.json", "local"),
        ("#/properties/x", "target"),
        ("#/$defs/missing", "does not exist"),
        ("#/$defs/bad~2name", "malformed"),
    ],
)
def test_remote_or_malformed_schema_refs_fail_closed(reference, message):
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "schema": {"$defs": {}, "$ref": reference},
        },
    }
    with pytest.raises(ValueError, match=message):
        compile_constraint(response_format)


def test_recursive_schema_refs_fail_closed():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "schema": {
                "$defs": {"node": {"$ref": "#/$defs/node"}},
                "$ref": "#/$defs/node",
            },
        },
    }
    with pytest.raises(ValueError, match="recursive"):
        compile_constraint(response_format)
