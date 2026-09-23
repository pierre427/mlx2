import pytest

from mlx2.adapters.muse_glimmer_output import (
    constrained_tool_grammar as muse_grammar,
    parse_atem,
)
from mlx2.adapters.north_output import constrained_tool_grammar as north_grammar
from mlx2.runtime.tool_parsers._schema import schema_value_matches
from mlx2.runtime.tool_parsers.qwen3_coder import (
    constrained_tool_grammar as qwen_grammar,
    parse_tool_call,
)
from mlx2.server import validate_request
from mlx2.structured_output import compile_constraint


def _tools(*, strict=True):
    return [{
        "type": "function",
        "function": {
            "name": "sum",
            "strict": strict,
            "parameters": {
                "$defs": {"count": {"type": "integer"}},
                "type": "object",
                "properties": {
                    "x": {"$ref": "#/$defs/count"},
                    "label": {"type": "string"},
                },
                "required": ["x"],
                "additionalProperties": False,
            },
        },
    }]


def _matches(grammar, text):
    return compile_constraint(grammar=grammar).fullmatch(text) is not None


def test_validation_accepts_required_and_named_tools_and_resolves_refs():
    base = {"messages": [{"role": "user", "content": "add"}], "tools": _tools()}
    required = validate_request(
        {**base, "tool_choice": "required", "parallel_tool_calls": False}
    )
    assert required["tool_choice"] == "required"
    assert required["parallel_tool_calls"] is False
    assert required["tools"][0]["function"]["parameters"]["properties"]["x"] == {
        "type": "integer"
    }
    named = validate_request({
        **base,
        "tool_choice": {"type": "function", "function": {"name": "sum"}},
    })
    assert named["tool_choice"]["function"]["name"] == "sum"


def test_non_strict_tool_reference_resolution_is_best_effort():
    messages = [{"role": "user", "content": "inspect"}]
    sibling = {
        "type": "function",
        "function": {
            "name": "sibling",
            "parameters": {
                "$defs": {"word": {"type": "string"}},
                "type": "object",
                "properties": {
                    "value": {
                        "$ref": "#/$defs/word",
                        "description": "annotation",
                    }
                },
            },
        },
    }
    recursive = {
        "type": "function",
        "function": {
            "name": "tree",
            "parameters": {
                "$defs": {
                    "node": {
                        "type": "object",
                        "properties": {
                            "children": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/node"},
                            }
                        },
                    }
                },
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/node"}},
            },
        },
    }
    accepted = validate_request({"messages": messages, "tools": [sibling]})
    assert accepted["tools"][0]["function"]["parameters"] == sibling[
        "function"
    ]["parameters"]
    strict_sibling = {
        **sibling,
        "function": {**sibling["function"], "strict": True},
    }
    assert validate_request(
        {
            "messages": messages,
            "tools": [strict_sibling],
            "tool_choice": "required",
        }
    )
    accepted = validate_request({"messages": messages, "tools": [recursive]})
    assert "$defs" in accepted["tools"][0]["function"]["parameters"]

    recursive["function"]["strict"] = True
    with pytest.raises(ValueError, match="recursive JSON schema"):
        validate_request(
            {"messages": messages, "tools": [recursive], "tool_choice": "required"}
        )


@pytest.mark.parametrize(
    "schema,candidates",
    [
        ({"type": "string"}, ["", "x", '"quoted"']),
        ({"type": "string", "enum": ["red", "blue"]}, ["red", "blue", "green"]),
        ({"type": "string", "const": "fixed"}, ["fixed", "other"]),
        ({"type": "string", "minLength": 1, "maxLength": 3}, ["", "a", "abc", "abcd"]),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, ['"x"', "12", "true"]),
        ({"type": ["string", "null"]}, ['"hello"', "null", "12"]),
    ],
)
def test_adapter_strict_string_grammars_only_parse_schema_valid_values(
    schema, candidates
):
    import regex

    tools = [{
        "type": "function",
        "function": {
            "name": "f",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"value": schema},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }]
    # The HTTP boundary accepts every schema exercised by the adapter property
    # check, so direct grammar coverage cannot bypass stricter request rules.
    validate_request({
        "messages": [{"role": "user", "content": "call"}],
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
    })
    if "anyOf" in schema:
        with pytest.raises(ValueError, match="post-generation validator"):
            validate_request(
                {
                    "messages": [{"role": "user", "content": "call"}],
                    "tools": tools,
                    "tool_choice": "required",
                },
                constrained_tool_grammar=True,
            )
        with pytest.raises(ValueError, match="post-generation validator"):
            qwen_grammar(tools, "required", parallel_tool_calls=False)
        return
    qwen = qwen_grammar(tools, "required", parallel_tool_calls=False)
    muse = muse_grammar(tools, "required", parallel_tool_calls=False)
    for candidate in candidates:
        qwen_text = (
            "<tool_call>\n<function=f>\n<parameter=value>\n"
            + candidate
            + "\n</parameter>\n</function>\n</tool_call>"
        )
        if regex.fullmatch(qwen, qwen_text):
            parsed = parse_tool_call(
                qwen_text.split("<tool_call>\n", 1)[1].rsplit("\n</tool_call>", 1)[0],
                tools,
            )
            assert schema_value_matches(parsed["arguments"]["value"], schema)

        muse_text = (
            '<atem:function_calls><atem:invoke name="f">'
            '<atem:parameter name="value">'
            + candidate
            + "</atem:parameter></atem:invoke></atem:function_calls>"
        )
        if regex.fullmatch(muse, muse_text):
            parsed = parse_atem(
                muse_text[len("<atem:function_calls>") : -len("</atem:function_calls>")],
                tools,
            )[0]
            assert schema_value_matches(parsed["arguments"]["value"], schema)


def test_raw_string_patterns_fail_closed_for_strict_tool_adapters():
    tools = [{
        "type": "function",
        "function": {
            "name": "f",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": "^[a-z]+$"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }]
    request = {
        "messages": [{"role": "user", "content": "call"}],
        "tools": tools,
        "tool_choice": "required",
    }
    with pytest.raises(ValueError, match="unsupported JSON schema keywords"):
        validate_request(request)
    with pytest.raises(ValueError, match="pattern is unsupported"):
        qwen_grammar(tools, "required")
    with pytest.raises(ValueError, match="pattern is unsupported"):
        muse_grammar(tools, "required")


def test_strict_tool_schemas_with_annotations_parse_and_constrain():
    """Pydantic-style ``title``/``description`` annotations on a strict tool
    used to make every well-formed call fail to parse and the forced grammar
    raise; a property may itself be named ``title`` or ``description``."""
    parameters = {
        "title": "Args",
        "type": "object",
        "properties": {
            "title": {"title": "Title", "type": "string", "description": "a name"},
            "description": {"title": "Description", "type": "integer"},
        },
        "required": ["title", "description"],
        "additionalProperties": False,
    }
    tools = [{"type": "function", "function": {
        "name": "f", "strict": True, "parameters": parameters,
    }}]
    validate_request({
        "messages": [{"role": "user", "content": "call"}],
        "tools": tools,
        "tool_choice": "required",
    })
    qwen_body = (
        "<function=f>\n<parameter=title>\nParis\n</parameter>"
        "\n<parameter=description>\n3\n</parameter>\n</function>"
    )
    assert parse_tool_call(qwen_body, tools)["arguments"] == {
        "title": "Paris", "description": 3,
    }
    qwen = qwen_grammar(tools, "required", parallel_tool_calls=False)
    assert _matches(qwen, f"<tool_call>\n{qwen_body}\n</tool_call>")
    assert not _matches(qwen, f"<tool_call>\n{qwen_body.replace('3', 'x')}\n</tool_call>")
    muse_body = (
        '<atem:invoke name="f"><atem:parameter name="title">Paris</atem:parameter>'
        '<atem:parameter name="description">3</atem:parameter></atem:invoke>'
    )
    assert parse_atem(muse_body, tools)[0]["arguments"] == {
        "title": "Paris", "description": 3,
    }
    muse = muse_grammar(tools, "required", parallel_tool_calls=False)
    assert _matches(muse, f"<atem:function_calls>{muse_body}</atem:function_calls>")
    north = north_grammar(tools, "required", parallel_tool_calls=False)
    assert _matches(
        north,
        '<|START_ACTION|>[{"tool_name":"f","parameters":'
        '{"title":"Paris","description":3}}]<|END_ACTION|>',
    )


def test_qwen_strict_number_grammar_excludes_nonfinite_float_literals():
    tools = [{
        "type": "function",
        "function": {
            "name": "f",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }]
    grammar = qwen_grammar(tools, "required", parallel_tool_calls=False)

    def call(value):
        return (
            "<tool_call>\n<function=f>\n<parameter=value>\n"
            f"{value}\n</parameter>\n</function>\n</tool_call>"
        )

    assert _matches(grammar, call("1e99"))
    assert not _matches(grammar, call("1e400"))


def test_validation_rejects_unsafe_or_conflicting_tool_constraint_modes():
    base = {"messages": [{"role": "user", "content": "add"}], "tools": _tools()}
    assert validate_request(base)
    with pytest.raises(ValueError, match="declared function"):
        validate_request({
            **base,
            "tool_choice": {"type": "function", "function": {"name": "missing"}},
        })
    assert validate_request({
        **base,
        "tool_choice": "required",
        "response_format": {"type": "json_object"},
    })
    with pytest.raises(ValueError, match="parallel_tool_calls"):
        validate_request({**base, "tool_choice": "required", "parallel_tool_calls": 1})


def test_qwen_required_strict_grammar_uses_xml_wire_schema_and_parallel_bound():
    one = (
        "<tool_call>\n<function=sum>\n<parameter=x>\n3\n</parameter>"
        "\n<parameter=label>\nok\n</parameter>\n</function>\n</tool_call>"
    )
    grammar = qwen_grammar(_tools(), "required", parallel_tool_calls=False)
    assert _matches(grammar, one)
    assert not _matches(grammar, one.replace("\n3\n", "\nthree\n"))
    assert not _matches(grammar, one + "\n" + one)
    parallel = qwen_grammar(_tools(), "required", parallel_tool_calls=True)
    assert _matches(parallel, one + "\n" + one)


def test_named_grammars_cover_muse_atem_and_north_action_json():
    choice = {"type": "function", "function": {"name": "sum"}}
    muse = (
        '<atem:function_calls><atem:invoke name="sum">'
        '<atem:parameter name="x">4</atem:parameter>'
        '</atem:invoke></atem:function_calls>'
    )
    assert _matches(muse_grammar(_tools(), choice, parallel_tool_calls=False), muse)
    north = (
        '<|START_ACTION|>[{"tool_name":"sum","parameters":{"x":4}}]'
        '<|END_ACTION|>'
    )
    assert _matches(north_grammar(_tools(), choice, parallel_tool_calls=False), north)


def test_non_strict_north_required_grammar_keeps_recursive_json_arguments():
    grammar = north_grammar(_tools(strict=False), "required", parallel_tool_calls=False)
    text = (
        '<|START_ACTION|>[{"tool_name":"sum","parameters":'
        '{"x":4,"nested":{"items":[1,true,null]}}}]<|END_ACTION|>'
    )
    assert _matches(grammar, text)


@pytest.mark.parametrize("wire", ["qwen", "muse"])
def test_non_strict_forced_grammars_only_admit_calls_their_parser_accepts(wire):
    """The optional-parameter block admitted any name, a repeated one
    included, and typed values were free text; the parser rejects both, so a
    call the grammar forced still failed with 502."""
    tools = [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "scale": {"type": ["number", "null"]} if wire == "qwen" else {"type": "number"},
            "metric": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": ["city", "days"],
    }}}]
    if wire == "qwen":
        grammar = qwen_grammar(tools, "required", parallel_tool_calls=False)

        def call(*params):
            body = "".join(
                f"\n<parameter={name}>\n{value}\n</parameter>" for name, value in params
            )
            return f"<tool_call>\n<function=f>{body}\n</function>\n</tool_call>"

        def parse(text):
            return parse_tool_call(text[len("<tool_call>"):-len("</tool_call>")], tools)
    else:
        grammar = muse_grammar(tools, "required", parallel_tool_calls=False)

        def call(*params):
            body = "".join(
                f'<atem:parameter name="{name}">{value}</atem:parameter>'
                for name, value in params
            )
            return (
                f'<atem:function_calls><atem:invoke name="f">{body}'
                "</atem:invoke></atem:function_calls>"
            )

        def parse(text):
            (parsed,) = parse_atem(
                text[len("<atem:function_calls>"):-len("</atem:function_calls>")], tools
            )
            return parsed

    good = call(("city", "Paris"), ("days", "3"), ("scale", "1.5"), ("metric", "true"))
    assert _matches(grammar, good)
    assert parse(good)["arguments"] == {
        "city": "Paris", "days": 3, "scale": 1.5, "metric": True,
    }
    rejected = [
        call(("city", "Paris"), ("city", "Lyon"), ("days", "3")),
        call(("city", "Paris"), ("days", "3"), ("note", "a"), ("note", "b")),
        call(("city", "Paris"), ("days", "ten")),
        call(("city", "Paris"), ("days", "3"), ("metric", "yes")),
        call(("city", "Paris"), ("days", "3"), ("scale", "1e400")),
    ]
    for text in rejected:
        assert not _matches(grammar, text), text
        with pytest.raises(ValueError):
            parse(text)
    if wire == "qwen":
        nullable = call(("city", "Paris"), ("days", "3"), ("scale", "null"))
        assert _matches(grammar, nullable)
        assert parse(nullable)["arguments"]["scale"] is None
    else:
        # Free values are unbounded runs: a ``{0,4096}`` run per parameter made
        # the exact automaton take tens of seconds to compile one call.
        from mlx2.structured_automaton import compile_pattern

        assert len(compile_pattern(grammar).rows) < 1000


@pytest.mark.parametrize(
    ("declared", "good", "bad"),
    [
        (["string", "null"], [('"Paris"', "Paris"), ("null", None)], ["Paris"]),
        (["integer", "null"], [("3", 3), ("null", None)], ["ten", "None"]),
        (["number", "null"], [("1.5", 1.5), ("null", None)], ["ten", "1e400", "NaN"]),
        (["boolean", "null"], [("true", True), ("null", None)], ["yes", "True"]),
        ("object", [('{"a": [1, null]}', {"a": [1, None]})], ["ten", "[1]"]),
        ("array", [('[1, "b"]', [1, "b"])], ["ten", "{}"]),
    ],
)
def test_muse_non_strict_grammar_admits_only_values_parse_atem_decodes(
    declared, good, bad
):
    """A list-form type (``["integer", "null"]``) or an object/array type
    left the Muse value free text, but ``parse_atem`` decodes every declared
    type other than a lone ``"string"`` as JSON: the grammar admitted
    ``ten``, the parser raised, and the forced call failed with 502."""
    from mlx2.structured_automaton import compile_pattern

    tools = [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object",
        "properties": {"x": {"type": declared}},
        "required": ["x"],
    }}}]
    grammar = muse_grammar(tools, "required", parallel_tool_calls=False)
    compile_pattern(grammar)

    def call(value):
        return (
            '<atem:function_calls><atem:invoke name="f">'
            f'<atem:parameter name="x">{value}</atem:parameter>'
            "</atem:invoke></atem:function_calls>"
        )

    def parse(text):
        (parsed,) = parse_atem(
            text[len("<atem:function_calls>"):-len("</atem:function_calls>")], tools
        )
        return parsed["arguments"]["x"]

    for value, expected in good:
        assert _matches(grammar, call(value)), value
        assert parse(call(value)) == expected
    for value in bad:
        assert not _matches(grammar, call(value)), value
        with pytest.raises(ValueError):
            parse(call(value))


def _server_admits(grammar, text):
    """Whether serving admits ``text``: the scanner compiles a server tool
    grammar without the client cap, and the exact automaton, when the
    pattern compiles to one, must agree."""
    from mlx2.structured_automaton import AutomatonUnsupported, automaton_for
    from mlx2.structured_output import _request_constraint

    pattern = _request_constraint(None, None, grammar, None).pattern
    admitted = pattern.fullmatch(text) is not None
    try:
        automaton = automaton_for(pattern)
    except AutomatonUnsupported:
        return admitted  # serving falls back to the scanner alone
    assert automaton.fullmatch(text) == admitted, text
    return admitted


def _single_parameter_tool(schema, *, strict):
    return [{"type": "function", "function": {
        "name": "f",
        "strict": strict,
        "parameters": {
            "type": "object",
            "properties": {"x": schema},
            "required": ["x"],
            "additionalProperties": False,
        },
    }}]


def _muse_call(value):
    return (
        '<atem:function_calls><atem:invoke name="f">'
        f'<atem:parameter name="x">{value}</atem:parameter>'
        "</atem:invoke></atem:function_calls>"
    )


def _muse_value(text, tools):
    """The argument the streaming parser serves for one forced call."""
    import json

    from mlx2.adapters.muse_glimmer_output import MuseOutputParser

    parser = MuseOutputParser(chat=True, tools=tools)
    events = parser.push(" to=f<|message|>" + text) + parser.push("", final=True)
    (call,) = [event["tool_calls"][0] for event in events if "tool_calls" in event]
    return json.loads(call["function"]["arguments"])["x"]


@pytest.mark.parametrize(
    ("schema", "strict", "finite", "overflowing"),
    [
        (
            {"type": "number"}, True,
            ["1e300", "-1.7976931348623157e308", "2.5e-400", "1E+289"],
            ["1e400", "9e308", "-1.8e308", "1.797693134862315808e308"],
        ),
        (
            {"type": "array", "items": {"type": "number"}}, True,
            ["[1e300, 1.797693134862315807e308]"], ["[1, 1e400]"],
        ),
        ({"type": "object"}, False, ['{"a": [1e300]}'], ['{"a": [1e400]}']),
        (["number", "null"], False, ["1e300"], ["1e400", "9e308"]),
    ],
)
def test_muse_grammars_admit_only_numbers_the_parser_can_serve(
    schema, strict, finite, overflowing
):
    """A number past float64's range (``1e400``, ``9e308``) is valid JSON
    text, but it decodes to infinity and ``parse_atem`` refuses to serialize
    it: the strict lowering and the shared recursive JSON rules admitted it,
    and the forced call failed with 502.  Finite values keep every exponent
    up to 308."""
    if isinstance(schema, list):
        schema = {"type": schema}
    tools = _single_parameter_tool(schema, strict=strict)
    grammar = muse_grammar(tools, "required", parallel_tool_calls=False)
    for value in finite:
        assert _server_admits(grammar, _muse_call(value)), value
        _muse_value(_muse_call(value), tools)
    for value in overflowing:
        assert not _server_admits(grammar, _muse_call(value)), value
        with pytest.raises(ValueError):
            _muse_value(_muse_call(value), tools)


@pytest.mark.parametrize(
    ("schema", "strict", "finite", "overflowing"),
    [
        ({"type": "number"}, True, ["1e300", "-1.7976931348623157e308"], ["1e400", "9e308"]),
        (
            {"type": "array", "items": {"type": "number"}}, True,
            ["[1e300]"], ["[1, 1e400]"],
        ),
        ({"type": "object"}, False, ['{"a": [1e300]}'], ['{"a": [1e400]}']),
        ({}, False, ["1e300"], ["9e308"]),
    ],
)
def test_north_grammars_admit_only_numbers_the_parser_can_serve(
    schema, strict, finite, overflowing
):
    """The North strict lowering and its recursive JSON rules admitted
    ``1e400``; ``parse_actions`` decodes it to infinity and refuses to
    serialize it, so the forced call failed with 502."""
    import json

    from mlx2.adapters.north_output import NorthOutputParser

    tools = _single_parameter_tool(schema, strict=strict)
    grammar = north_grammar(tools, "required", parallel_tool_calls=False)

    def action(value):
        return (
            '<|START_ACTION|>[{"tool_name":"f","parameters":{"x":'
            + value + "}}]<|END_ACTION|>"
        )

    def serve(text):
        parser = NorthOutputParser(chat=True, thinking=False, tools=tools)
        events = parser.push(text) + parser.push("", final=True)
        (call,) = [event["tool_calls"][0] for event in events if "tool_calls" in event]
        return json.loads(call["function"]["arguments"])["x"]

    for value in finite:
        assert _server_admits(grammar, action(value)), value
        serve(action(value))
    for value in overflowing:
        assert not _server_admits(grammar, action(value)), value
        with pytest.raises(ValueError):
            serve(action(value))


def test_finite_number_language_keeps_a_spelling_for_every_finite_value():
    """``_FINITE_NUMBER`` admits no literal that decodes to infinity, and
    every finite value ``_NUMBER`` can spell keeps an admitted spelling."""
    import math
    import random

    import regex

    from mlx2.structured_output import _FINITE_NUMBER, _NUMBER

    number, finite = regex.compile(_NUMBER), regex.compile(_FINITE_NUMBER)
    digits = "7976931348623158079"
    rng = random.Random(0)

    def draw(count):
        return "".join(rng.choice("0123456789") for _ in range(count))

    checked = 0
    for _ in range(20000):
        integer = rng.choice(["0", "1", str(rng.randint(2, 9)), "1" + draw(rng.randint(1, 18))])
        fraction = ""
        if rng.random() < 0.8:
            # Near the overflow threshold's digits, where the boundary is.
            size = rng.randint(1, 18)
            keep = rng.randint(0, size)
            fraction = "." + (digits[:keep] + draw(size - keep))[:size]
        exponent = ""
        if rng.random() < 0.9:
            power = rng.choice([rng.randint(0, 999), rng.randint(280, 320)])
            exponent = rng.choice(["e", "E+", "e-"]) + rng.choice(
                [str(power), f"{power:03d}"]
            )
        literal = rng.choice(["", "-"]) + integer + fraction + exponent
        if not number.fullmatch(literal):
            continue
        checked += 1
        value = float(literal)
        if finite.fullmatch(literal):
            assert math.isfinite(value), literal
        elif math.isfinite(value):
            mantissa, _, power = f"{abs(value):.17e}".partition("e")
            spelling = ("-" if value < 0 else "") + mantissa + "e" + str(int(power))
            assert finite.fullmatch(spelling) and float(spelling) == value, literal
    assert checked > 10000


def test_non_strict_tool_blocks_compose_with_a_json_answer():
    """Tool blocks carry the finite-number JSON rules; composing one with a
    JSON answer must still merge the two rule sets into one."""
    import regex

    from mlx2.structured_automaton import automaton_for
    from mlx2.tool_grammar import plan_tool_grammar

    tools = _single_parameter_tool({"type": "object"}, strict=False)
    for builder, marker in (
        (muse_grammar, "<atem:function_calls>"),
        (north_grammar, "<|START_ACTION|>"),
    ):
        pattern, status, _ = plan_tool_grammar(
            {"tools": tools, "response_format": {"type": "json_object"}},
            lambda request, builder=builder: builder(
                request["tools"], request["tool_choice"]
            ),
            open_marker=marker,
        )
        assert status == "engaged"
        language = regex.compile(rf"(?:{pattern})")
        automaton_for(language)
        assert language.fullmatch('{"a": 1e300}')
        assert not language.fullmatch('{"a": 1e400}')


_OBJECT = {
    "type": "object",
    "properties": {"s": {"type": "string"}},
    "required": ["s"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    ("schema", "strict", "value"),
    [
        (_OBJECT, True, '{"s": "a</atem:parameter>b"}'),
        ({"type": ["string", "null"]}, True, '"<b>bold</b></atem:parameter>"'),
        ({"type": "object"}, False, '{"s": "</atem:invoke></atem:function_calls>x"}'),
        ({"type": "array"}, False, '["<atem:parameter name=\\"y\\">", "a<b"]'),
        ({"type": ["string", "null"]}, False, '"x < y</atem:parameter>"'),
    ],
)
def test_muse_json_values_keep_raw_angle_brackets_and_quoted_tags(
    schema, strict, value
):
    """Muse writes JSON values with a raw ``<`` (its template's ``tojson``
    does not escape it) and the grammar admits one inside a JSON string,
    ``</atem:parameter>`` included, but the parser cut the value at the first
    closer, quoted or not: the admitted call failed with 502.  A JSON value
    now ends at the first closer outside its strings, however the text is
    chunked."""
    import json

    from mlx2.adapters.muse_glimmer_output import MuseOutputParser

    tools = _single_parameter_tool(schema, strict=strict)
    grammar = muse_grammar(tools, "required", parallel_tool_calls=False)
    text = _muse_call(value)
    assert _server_admits(grammar, text)
    for split in (len(text), 1, 7):
        parser = MuseOutputParser(chat=True, tools=tools)
        events = parser.push(" to=f<|message|>")
        for start in range(0, len(text), split):
            events += parser.push(text[start : start + split])
        events += parser.push("<|eot|>", final=True)
        (call,) = [event["tool_calls"][0] for event in events if "tool_calls" in event]
        assert json.loads(call["function"]["arguments"]) == {"x": json.loads(value)}
        assert not [event for event in events if event.get("content")]


@pytest.mark.parametrize("properties", [True, False])
@pytest.mark.parametrize(
    ("value", "served"),
    [
        ("NaN", "NaN"),
        ("Infinity", "Infinity"),
        ("-Infinity", "-Infinity"),
        ("1e400", "1e400"),
        ("[1, NaN]", "[1, NaN]"),
        ('{"a": -1e400}', '{"a": -1e400}'),
        ("hello", "hello"),
        ("[1, 2.5]", [1, 2.5]),
        ("1e300", 1e300),
    ],
)
def test_muse_untyped_values_serve_non_finite_json_as_text(properties, value, served):
    """An untyped non-strict value is free text decoded best-effort, and
    Python's decoder takes ``NaN``, ``Infinity`` and ``1e400`` to non-finite
    floats that the arguments cannot carry: the grammar admitted them and the
    call failed with 502.  Like other text that is not JSON they stay the
    raw string; finite JSON still decodes."""
    parameters = (
        {"type": "object", "properties": {"x": {}}, "required": ["x"]}
        if properties
        # No declared properties: argument names are free.
        else {"type": "object", "required": ["x"]}
    )
    tools = [{"type": "function", "function": {"name": "f", "parameters": parameters}}]
    grammar = muse_grammar(tools, "required", parallel_tool_calls=False)
    assert _server_admits(grammar, _muse_call(value))
    assert _muse_value(_muse_call(value), tools) == served


@pytest.mark.parametrize(
    "schema", [{"type": "object"}, {"description": "any value"}, {"type": "tuple"}]
)
@pytest.mark.parametrize(
    ("value", "served"),
    [
        ('{"a": NaN}', '{"a": NaN}'),
        ('{"a": -1e400}', '{"a": -1e400}'),
        ("[Infinity]", "[Infinity]"),
        ("{1, 2}", "{1, 2}"),
        ('{"a": [1e300]}', {"a": [1e300]}),
    ],
)
def test_qwen_free_values_the_arguments_cannot_carry_stay_text(schema, value, served):
    """Qwen's non-strict object, untyped and unknown-type values are free
    text decoded best-effort (JSON, then a Python literal), which can yield a
    non-finite float or a set.  The arguments serialize with
    ``allow_nan=False``, so the grammar admitted the call and serving it
    failed with 502; such text now stays the string the model wrote, and
    finite JSON still decodes."""
    import json

    from mlx2.output import OutputParser

    tools = [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"x": schema}, "required": ["x"],
    }}}]
    grammar = qwen_grammar(tools, "required", parallel_tool_calls=False)
    text = (
        f"<tool_call>\n<function=f>\n<parameter=x>\n{value}\n</parameter>"
        "\n</function>\n</tool_call>"
    )
    assert _server_admits(grammar, text)
    parser = OutputParser(
        chat=True, tools=tools, parse_tool=parse_tool_call, constrained_tools=True
    )
    events = parser.push(text) + parser.push("", final=True)
    (call,) = [event["tool_calls"][0] for event in events if "tool_calls" in event]
    assert json.loads(call["function"]["arguments"])["x"] == served


def test_non_strict_named_tool_grammars_enforce_required_parameters():
    # sglang #40051: a non-strict tool body of optional-only parameters let
    # greedy decoding close a forced call with no arguments.
    choice = {"type": "function", "function": {"name": "sum"}}
    tools = _tools(strict=False)
    qwen = qwen_grammar(tools, choice, parallel_tool_calls=False)
    empty = "<tool_call>\n<function=sum>\n</function>\n</tool_call>"
    label_only = (
        "<tool_call>\n<function=sum>\n<parameter=label>\nok\n</parameter>"
        "\n</function>\n</tool_call>"
    )
    with_x = (
        "<tool_call>\n<function=sum>\n<parameter=x>\n3\n</parameter>"
        "\n<parameter=label>\nok\n</parameter>\n</function>\n</tool_call>"
    )
    assert not _matches(qwen, empty)
    assert not _matches(qwen, label_only)
    assert _matches(qwen, with_x)
    assert parse_tool_call(with_x, tools)["arguments"]["x"] == 3
    # Optional arguments are the declared ones (the schema forbids others).
    assert not _matches(qwen, with_x.replace("=label>", "=extra>"))

    muse = muse_grammar(tools, choice, parallel_tool_calls=False)
    wrap = '<atem:function_calls><atem:invoke name="sum">{}</atem:invoke></atem:function_calls>'
    assert not _matches(muse, wrap.format(""))
    assert _matches(muse, wrap.format(
        '<atem:parameter name="x">4</atem:parameter>'
        '<atem:parameter name="label">ok</atem:parameter>'
    ))

    north = north_grammar(tools, choice, parallel_tool_calls=False)
    action = '<|START_ACTION|>[{{"tool_name":"sum","parameters":{}}}]<|END_ACTION|>'
    assert not _matches(north, action.format("{}"))
    assert not _matches(north, action.format('{"label":"ok"}'))
    assert _matches(north, action.format('{"x":[1,{"a":null}],"label":"ok"}'))
