"""CPU-only port contracts. Never import the tensor model or MLX."""

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.adapters.muse_glimmer import (
    MUSE_GLIMMER,
    MuseGlimmerAdapter,
    MuseRecipientProcessor,
    inspect_artifact,
    normalize_external_policy,
    normalize_messages,
)
from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.adapters.muse_glimmer_output import MuseOutputParser, parse_atem
from mlx2.contracts import Capability

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "functions.echo",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "data": {"type": "object"},
                },
            },
        },
    }
]
ATEM = '<atem:invoke name="functions.echo"><atem:parameter name="text">  a<b  </atem:parameter><atem:parameter name="count">2</atem:parameter></atem:invoke>'


def collect(text, split=1, **kwargs):
    parser = MuseOutputParser(chat=True, **kwargs)
    events = []
    for start in range(0, len(text), split):
        events.extend(parser.push(text[start : start + split]))
    events.extend(parser.push("", final=True))
    return parser, events


def test_import_has_no_tensor_side_effects():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mlx2.adapters.muse_glimmer; import mlx2.adapters.muse_glimmer_output; assert 'mlx.core' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_default_topology_and_layout():
    args = ModelArgs()
    assert args.layer_types.count("sliding_attention") == 39
    assert args.layer_types.count("full_attention") == 13
    assert args.sliding_window == 2048
    assert ModelArgs(sliding_window=1024).cache_layout != args.cache_layout


def test_row_exact_external_policy_is_explicit_and_boolean():
    base = {"draft_model": "/draft", "proposal_composition": False}
    assert "target_verify_row_exact" not in normalize_external_policy(base)
    selected = normalize_external_policy(
        {**base, "target_verify_row_exact": True}
    )
    assert selected["target_verify_row_exact"] is True
    with pytest.raises(ValueError, match="must be a boolean"):
        normalize_external_policy({**base, "target_verify_row_exact": 1})


def test_progressive_external_policy_requires_row_exact_host_chain():
    base = {"draft_model": "/draft", "target_verify_row_exact": True}
    selected = normalize_external_policy(
        {
            **base,
            "num_draft": 15,
            "progressive_verification_tile": 3,
            "progressive_multilane_draft_cap": 3,
        }
    )
    assert selected["progressive_verification_tile"] == 3
    assert selected["progressive_multilane_draft_cap"] == 3
    assert selected["proposal_composition"] is False
    with pytest.raises(ValueError, match="positive integer"):
        normalize_external_policy(
            {**base, "progressive_verification_tile": True}
        )
    with pytest.raises(ValueError, match="requires target_verify_row_exact"):
        normalize_external_policy(
            {
                "draft_model": "/draft",
                "progressive_verification_tile": 3,
            }
        )
    with pytest.raises(ValueError, match="cannot combine"):
        normalize_external_policy(
            {
                **base,
                "progressive_verification_tile": 3,
                "proposal_composition": {},
            }
        )
    with pytest.raises(ValueError, match="requires progressive verification"):
        normalize_external_policy(
            {**base, "progressive_multilane_draft_cap": 3}
        )
    with pytest.raises(ValueError, match="requires progressive verification"):
        normalize_external_policy(
            {
                **base,
                "num_draft": 15,
                "progressive_verification_tile": 3,
                "progressive_multilane_draft_cap": 16,
            }
        )


_PROGRESSIVE_BASE = {"draft_model": "/draft", "target_verify_row_exact": True}


@pytest.mark.parametrize(
    "policy",
    [
        # The default num_draft is 3: a tile of 3 can never split a proposal.
        {**_PROGRESSIVE_BASE, "progressive_verification_tile": 3},
        {**_PROGRESSIVE_BASE, "num_draft": 4, "progressive_verification_tile": 4},
        {**_PROGRESSIVE_BASE, "num_draft": 4, "progressive_verification_tile": 9},
        {**_PROGRESSIVE_BASE, "num_draft": "15", "progressive_verification_tile": 3},
    ],
)
def test_progressive_tile_must_be_below_num_draft(policy):
    # ExternalDraftBatchGenerator requires 1 <= tile < num_draft; the adapter
    # must refuse the policy before the target and drafter weights load.
    with pytest.raises(ValueError, match="below num_draft"):
        normalize_external_policy(policy)


@pytest.mark.parametrize("cap", [1, 2])
def test_multilane_cap_respects_dflash2_proposal_floor(cap):
    # The generator floors the cap at DFlash2's minimum proposal length (3).
    with pytest.raises(ValueError, match="requires progressive verification"):
        normalize_external_policy(
            {
                **_PROGRESSIVE_BASE,
                "num_draft": 15,
                "progressive_verification_tile": 3,
                "progressive_multilane_draft_cap": cap,
            }
        )


def test_shipped_progressive_profiles_still_accepted():
    root = Path(__file__).resolve().parents[1] / "qualification/policies"
    for name in (
        "muse-dflash2-progressive.json",
        "muse-dflash2-progressive-b1-m3.json",
    ):
        policy = normalize_external_policy(json.loads((root / name).read_text()))
        assert policy["progressive_verification_tile"] == 3
    accepted = normalize_external_policy(
        {**_PROGRESSIVE_BASE, "progressive_verification_tile": 2}
    )
    assert accepted["progressive_verification_tile"] == 2


def test_muse_cache_projection_keeps_rolling_boundaries():
    from mlx2.runtime.state_boundaries import (
        BoundaryPurpose,
        budget_state_boundaries,
        plan_state_boundaries,
    )

    adapter = object.__new__(MuseGlimmerAdapter)
    adapter._config = {"dtype": "bfloat16", "text_config": vars(ModelArgs())}
    budget = adapter.cache_budget(mtp=False)
    assert (budget.global_layers, budget.sliding_layers) == (13, 39)
    assert budget.item_bytes == 2
    assert budget.project(4096) > budget.project(1024) > 0
    planned = plan_state_boundaries(
        prompt_tokens=4096, cached_tokens=0, rolling_interval=512
    )
    kept, charged = budget_state_boundaries(
        planned, available_bytes=2 * budget.project(4096),
        cache_projection=budget.project,
    )
    assert charged > 0
    assert any(bound.purpose == BoundaryPurpose.ROLLING for bound in kept)
    with pytest.raises(ValueError, match="no native MTP"):
        adapter.cache_budget(mtp=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_types": ["full_attention"]},
        {"hidden_activation": "gelu"},
        {"sliding_window": 0},
        {"num_hidden_layers": 0},
        {"layer_rope_theta": [1.0] * 52},
        {"layer_types": ["unknown"] * 52},
        {"final_logit_softcapping": 0},
        {"num_attention_heads": 31},
        {"head_dim": 0},
        {"hidden_size": 0},
        {"intermediate_size": 0},
        {"vocab_size": 0},
        {"max_position_embeddings": 0},
    ],
)
def test_invalid_topology_fails_closed(kwargs):
    with pytest.raises(ValueError):
        ModelArgs(**kwargs)


def test_nested_configuration_is_target_identity():
    args = ModelArgs.from_dict(
        {
            "model_type": "muse_glimmer",
            "text_config": {
                "model_type": "muse_glimmer_text",
                "num_hidden_layers": 4,
            },
        }
    )
    assert args.model_type == "muse_glimmer"
    assert args.layer_types == ["sliding_attention"] * 3 + ["full_attention"]


def test_inspector_rejects_drafter_and_missing_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "dflash_config": {}})
    )
    with pytest.raises(ValueError, match="target artifact"):
        inspect_artifact(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_glimmer"}))
    with pytest.raises(FileNotFoundError):
        inspect_artifact(tmp_path)


def test_identity_bound_to_topology_and_tokenizer(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_glimmer"}))
    (tmp_path / "model.safetensors").write_bytes(b"metadata-only fixture")
    (tmp_path / "tokenizer.json").write_text("{}")
    first = inspect_artifact(tmp_path)
    (tmp_path / "tokenizer.json").write_text('{"changed":true}')
    assert inspect_artifact(tmp_path)["fingerprint"] != first["fingerprint"]
    assert first["qualification"] == "pending"


@pytest.mark.parametrize("weights", [{}, [], None])
def test_empty_or_invalid_index_rejected_before_tensor_load(tmp_path, weights):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_glimmer"}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    with pytest.raises(ValueError, match="nonempty weight index"):
        inspect_artifact(tmp_path)


def test_direct_inspection_rejects_drafter_metadata(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "muse_glimmer", "dflash_config": {}})
    )
    with pytest.raises(ValueError, match="drafter"):
        inspect_artifact(tmp_path)


def test_qualification_not_inferred_and_self_mtp_refused():
    assert MUSE_GLIMMER.metadata["qualification"] == "pending"
    assert Capability.MTP not in MUSE_GLIMMER.capabilities
    assert Capability.SEGMENTED_MTP not in MUSE_GLIMMER.capabilities
    assert Capability.APC_V2 in MUSE_GLIMMER.capabilities
    assert MuseGlimmerAdapter.profile_name(False) == "muse-glimmer-apcv2-ordinary"
    with pytest.raises(ValueError, match="DFlash2"):
        MuseGlimmerAdapter.profile_name(True)


def test_execution_policy_is_ordinary_and_no_speculation():
    adapter = MuseGlimmerAdapter.__new__(MuseGlimmerAdapter)
    config = adapter.execution_config(max_lanes=4, prefill_step=2048)
    assert config["num_draft"] == 0
    assert config["segment_aware_live_tip"] is False


def test_execution_policy_rejected_before_artifact_or_tensor_loading():
    with pytest.raises(ValueError, match="policy overrides"):
        MuseGlimmerAdapter("/not/a/model", execution_policy={"num_draft": 2})


def test_external_policy_defaults_pld_and_allows_explicit_opt_out():
    base = {"draft_model": "/draft", "num_draft": 4}
    defaulted = normalize_external_policy(base)
    assert defaulted["proposal_composition"] == {
        "prompt_lookup": True,
        "ngram_min": 3,
        "ngram_max": 6,
        "lookback": 256,
        "min_context_match": 4,
        "max_sources": 8,
    }
    disabled = normalize_external_policy({**base, "proposal_composition": False})
    assert disabled["proposal_composition"] is False
    pairwise = normalize_external_policy(
        {**base, "pairwise_selection": "batched"}
    )
    assert "proposal_composition" not in pairwise
    with pytest.raises(ValueError, match="cannot combine"):
        normalize_external_policy(
            {
                **base,
                "pairwise_selection": "batched",
                "proposal_composition": {"prompt_lookup": True},
            }
        )


def test_tool_history_is_normalized_without_mutation():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "functions.echo",
                        "arguments": '{"text":"hello"}',
                    }
                }
            ],
        }
    ]
    before = copy.deepcopy(messages)
    normalized = normalize_messages(messages)
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {"text": "hello"}
    assert messages == before


@pytest.mark.parametrize(
    ("choice", "thinking", "with_tools", "suffix"),
    [
        ("none", False, True, " to=user<|message|>"),
        ("none", True, True, ""),
        ("auto", False, True, ""),
        ("auto", True, True, ""),
        ("required", False, True, ""),
        ("required", True, True, ""),
        (
            {"type": "function", "function": {"name": "functions.echo"}},
            False,
            True,
            " to=functions.echo<|message|>",
        ),
        (
            {"type": "function", "function": {"name": "functions.echo"}},
            True,
            True,
            "",
        ),
        ("auto", False, False, " to=user<|message|>"),
    ],
)
def test_tool_choice_and_thinking_render_matrix(choice, thinking, with_tools, suffix):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.strength = kwargs["reasoning_strength"]
            self.tools = kwargs["tools"]
            assert kwargs["tokenize"] is False
            return "<|start|>assistant"

        def encode(self, prompt, **kwargs):
            self.prompt = prompt
            return [1, 2]

    adapter = MuseGlimmerAdapter.__new__(MuseGlimmerAdapter)
    adapter.tokenizer = Tokenizer()
    request = {
        "messages": [],
        "tool_choice": choice,
        "enable_thinking": thinking,
    }
    if with_tools:
        request["tools"] = TOOLS
    adapter.prompt_tokens(request)
    assert adapter.tokenizer.strength == ("high" if thinking else "low")
    assert adapter.tokenizer.prompt == "<|start|>assistant" + suffix
    assert bool(adapter.tokenizer.tools) is (with_tools and choice != "none")


def test_reasoning_effort_mapping_remains_native_prompt_policy():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.strength = kwargs["reasoning_strength"]
            return "<|start|>assistant"

        def encode(self, prompt, **kwargs):
            self.prompt = prompt
            return [1, 2]

    adapter = MuseGlimmerAdapter.__new__(MuseGlimmerAdapter)
    adapter.tokenizer = Tokenizer()
    adapter.prompt_tokens({"messages": [], "reasoning_effort": "none"})
    assert adapter.tokenizer.strength == "low"
    assert adapter.tokenizer.prompt.endswith(" to=user<|message|>")
    adapter.prompt_tokens({"messages": [], "reasoning_effort": "ultra"})
    assert adapter.tokenizer.strength == "high"
    assert adapter.tokenizer.prompt == "<|start|>assistant"


def test_recipient_processor_is_history_pure_and_probe_safe():
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.processor_probe import isolated_logits_processor

    processor = MuseRecipientProcessor(
        2,
        [
            (10, 20, 99),
            (10, 21, 30, 99),
            (10, 21, 31, 99),
        ],
    )
    logits = mx.arange(128, dtype=mx.float32)[None, :]

    def finite(history):
        masked = processor(mx.array([1, 2, *history]), logits)
        return set(np.flatnonzero(np.isfinite(np.asarray(masked[0]))).tolist())

    assert finite([]) == {10}
    assert finite([10]) == {20, 21}
    assert finite([10, 21]) == {30, 31}
    assert finite([10, 21, 30]) == {99}
    assert finite([55]) == set()
    completed = processor(mx.array([1, 2, 10, 20, 99]), logits)
    assert mx.array_equal(completed, logits)
    first = processor(mx.array([1, 2, 10]), logits)
    second = processor(mx.array([1, 2, 10]), logits)
    assert mx.array_equal(first, second)
    assert (
        isolated_logits_processor(processor)(mx.array([1, 2, 10]), logits).tolist()
        == first.tolist()
    )


def test_adapter_recipient_processors_select_only_requested_channels():
    class Tokenizer:
        headers = {
            " to=user<|message|>": [10, 20, 99],
            " to=functions.echo<|message|>": [10, 21, 30, 99],
        }

        def encode(self, text, **kwargs):
            return self.headers[text]

    adapter = MuseGlimmerAdapter.__new__(MuseGlimmerAdapter)
    adapter.tokenizer = Tokenizer()
    base = {
        "messages": [],
        "tools": TOOLS,
        "enable_thinking": False,
    }
    required = adapter.request_logits_processors(
        {**base, "tool_choice": "required"}, prompt_length=7
    )[0]
    automatic = adapter.request_logits_processors(
        {**base, "tool_choice": "auto"}, prompt_length=7
    )[0]
    assert required.prompt_length == automatic.prompt_length == 7
    assert required.headers == ((10, 21, 30, 99),)
    assert automatic.headers == ((10, 20, 99), (10, 21, 30, 99))
    named = {"type": "function", "function": {"name": "functions.echo"}}
    assert (
        adapter.request_logits_processors(
            {**base, "tool_choice": named}, prompt_length=7
        )
        == ()
    )
    assert (
        adapter.request_logits_processors(
            {**base, "tool_choice": "required", "enable_thinking": True},
            prompt_length=7,
        )
        == ()
    )


def test_required_decode_grammar_composes_recipient_header_with_atem_body():
    adapter = MuseGlimmerAdapter.__new__(MuseGlimmerAdapter)
    request = {
        "messages": [],
        "tools": TOOLS,
        "tool_choice": "required",
        "enable_thinking": False,
    }
    grammar = adapter.tool_constraint(request)
    body = "<atem:function_calls>" + ATEM.replace("a<b", "ab") + "</atem:function_calls>"
    # The object parameter's value calls the shared recursive JSON rules,
    # which only the ``regex`` engine the server compiles with can read.
    import regex

    assert regex.fullmatch(grammar, " to=functions.echo<|message|>" + body)
    assert regex.fullmatch(grammar, body) is None


def test_multimodal_content_refused():
    with pytest.raises(ValueError, match="text only"):
        normalize_messages([{"content": [{"type": "image_url", "image_url": "x"}]}])


@pytest.mark.parametrize("split", [1, 2, 3, 11, 4096])
def test_recipient_channels_are_chunk_boundary_safe(split):
    _, events = collect(
        "to=self<|message|>reason<|eom|><|start|>assistant to=user<|message|>answer<|eot|>",
        split=split,
    )
    assert "".join(e.get("reasoning_content", "") for e in events) == "reason"
    assert "".join(e.get("content", "") for e in events) == "answer"


def test_plain_content_and_bare_user_header():
    for raw, expected in [
        ("Hi.", "Hi."),
        ("to=user<|message|>Hi.", "Hi."),
        ("to=do list: Hi.", "to=do list: Hi."),
    ]:
        _, events = collect(raw)
        assert "".join(e.get("content", "") for e in events) == expected


def test_atem_tools_chunked_and_string_whitespace_preserved():
    parser, events = collect(
        "to=functions.echo<|message|><atem:function_calls>"
        + ATEM
        + "</atem:function_calls><|eot|>",
        tools=TOOLS,
    )
    assert parser.tool_count == 1
    tool = next(e["tool_calls"][0] for e in events if "tool_calls" in e)
    assert tool["function"]["name"] == "functions.echo"
    assert json.loads(tool["function"]["arguments"]) == {"text": "  a<b  ", "count": 2}


def test_muse_auto_parallel_false_drops_a_second_call():
    body = (
        "<atem:function_calls>"
        + ATEM
        + ATEM
        + "</atem:function_calls>"
    )
    parser, events = collect(body, tools=TOOLS, parallel_tool_calls=False)
    calls = [call for event in events for call in event.get("tool_calls", ())]
    assert len(calls) == 1
    assert parser.tool_call_constraint_truncations == 1


@pytest.mark.parametrize(
    "body",
    [
        ATEM.replace("functions.echo", "functions.other"),
        ATEM.replace(">2</atem:parameter>", ">true</atem:parameter>"),
        ATEM.replace('name="count"', 'name="text"'),
        "bad" + ATEM,
        ATEM + "bad",
        "",
    ],
)
def test_malformed_or_undeclared_tool_calls_fail_closed(body):
    with pytest.raises(ValueError):
        parse_atem(body, TOOLS)


def test_stop_spans_chunk_boundary():
    parser, events = collect("to=user<|message|>helloSTOPignored", stops=["STOP"])
    assert parser.stopped
    assert "".join(e.get("content", "") for e in events) == "hello"


def test_incomplete_tool_and_header_fail_closed():
    for raw in [
        "to=user<|mess",
        "to=functions.echo<|message|><atem:function_calls>" + ATEM,
    ]:
        with pytest.raises(ValueError):
            collect(raw, tools=TOOLS)


def test_tensor_source_is_standalone_and_ordinary_cache_topology():
    path = Path(__file__).parents[1] / "src/mlx2/runtime/models/muse_glimmer.py"
    tree = ast.parse(path.read_text())
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any(
        module and module.startswith(("mlx_lm", "mlx_vlm")) for module in imports
    )
    assert {"RotatingKVCache", "KVCache"} <= {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }


MUSE_TOKENIZER = Path.home() / "mlx-models" / "Muse-Glimmer-30B-mlx-4bit"


@pytest.mark.parametrize(
    "engine, strict, choice, text",
    [
        # The forced grammar opens with `` to=<name><|message|>``.
        ("automaton", False, "required", "call"),
        # The auto grammar (strict tools): header in the call block, and the
        # user header inside free text.
        ("scanner", True, "auto", "call"),
        ("scanner", True, "auto", "user"),
    ],
)
def test_real_tokenizer_tool_grammars_take_the_special_message_header(
    engine, strict, choice, text, monkeypatch
):
    """``<|message|>`` is a special token in the Muse tokenizer.  Refused by
    the structured mask while the recipient processor insisted on it, every
    forced call (and every ``auto`` grammar answer) failed closed; admitted
    without advancing the grammar (before b044155a) the header was spelled a
    second time and the parser delivered a stray ``<|message|>``."""
    if not (MUSE_TOKENIZER / "tokenizer.json").exists():
        pytest.skip("Muse tokenizer artifact is not present")
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
    from mlx2.structured_output import make_structured_processor
    from mlx2.tool_grammar import plan_tool_grammar

    monkeypatch.setenv("MLX2_STRUCTURED_AUTOMATON", "1" if engine == "automaton" else "0")
    monkeypatch.setenv("MLX2_STRUCTURED_WORKERS", "0")
    hf = AutoTokenizer.from_pretrained(MUSE_TOKENIZER, local_files_only=True)
    eot = hf.convert_tokens_to_ids("<|eot|>")
    tokenizer = TokenizerWrapper(
        hf, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids={eot}
    )
    adapter = MuseGlimmerAdapter.__new__(MuseGlimmerAdapter)
    adapter.tokenizer = tokenizer
    message = hf.convert_tokens_to_ids("<|message|>")
    assert message in hf.all_special_ids
    assert adapter.structured_special_token_ids() == (message,)
    tools = [{"type": "function", "function": {
        "name": "weather",
        "strict": strict,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }}]
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "tool_choice": choice,
    }
    grammar, status, _ = plan_tool_grammar(
        request, adapter.tool_constraint, open_marker=adapter.tool_call_open_marker
    )
    assert status == "engaged"
    prompt = [1, 2, 3]
    # The serving order: adapter processors, then the structured processor.
    recipient = adapter.request_logits_processors(request, prompt_length=len(prompt))[0]
    structured = make_structured_processor(
        tokenizer, len(prompt), server_grammar=grammar, greedy=True,
        generation_stop_token_ids=[eot],
        structural_token_ids=adapter.structured_special_token_ids(),
    )
    assert structured.engine == engine
    call = (
        ' to=weather<|message|><atem:function_calls><atem:invoke name="weather">'
        '<atem:parameter name="city">Paris</atem:parameter></atem:invoke>'
        "</atem:function_calls>"
    )
    wanted = call if text == "call" else " to=user<|message|>Hello there."
    target = hf.encode(wanted, add_special_tokens=False) + [eot]
    assert target.count(message) == 1
    generated = []
    for token in target:
        # The model wants exactly the target; only a mask can stop it.
        logits = mx.full((1, len(hf)), -5.0)
        logits[0, token] = 10.0
        history = mx.array(prompt + generated)
        row = structured(history, recipient(history, logits))[0]
        assert structured.failure is None, (hf.decode(generated), structured.failure)
        assert int(mx.argmax(row).item()) == token, hf.decode(generated)
        generated.append(token)
    decoded = hf.decode(generated[:-1], skip_special_tokens=False)
    parser = MuseOutputParser(chat=True, tools=tools)
    events = parser.push(decoded) + parser.push("", final=True)
    if text == "call":
        assert [event.get("content") for event in events if "content" in event] == []
        (only,) = [event for event in events if "tool_calls" in event]
        assert only["tool_calls"][0]["function"]["name"] == "weather"
        assert json.loads(only["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    else:
        assert events == [{"content": "Hello there."}]


def _finish(text, split, finish_reason, **kwargs):
    parser = MuseOutputParser(chat=True, tools=TOOLS, **kwargs)
    events = parser.push(text[:split]) + parser.finish(text[split:], finish_reason)
    content = "".join(e.get("content", "") for e in events)
    reasoning = "".join(e.get("reasoning_content", "") for e in events)
    tool_calls = [call for event in events for call in event.get("tool_calls", ())]
    return parser, content, reasoning, tool_calls


@pytest.mark.parametrize("cut", [len("<atem:function_calls>"), 30, len(ATEM) // 2, len(ATEM)])
def test_max_tokens_inside_an_atem_call_drops_the_partial_call(cut):
    # A length finish is a requested stop, not malformed output: drop the
    # unclosed block (as OutputParser and North do); EOS still fails closed.
    text = (
        "to=self<|message|>plan<|eom|>to=user<|message|>Sure.<|eom|>"
        "to=functions.echo<|message|>" + ("<atem:function_calls>" + ATEM)[:cut]
    )
    for split in range(len(text) + 1):
        parser, content, reasoning, tool_calls = _finish(text, split, "length")
        assert (content, reasoning, tool_calls) == ("Sure.", "plan", [])
        assert parser.tool_count == 0
        if cut > len("<atem:function_calls>"):
            with pytest.raises(ValueError, match="incomplete"):
                _finish(text, split, "stop")


@pytest.mark.parametrize("split", [0, len("to=functions.echo<|message|><atem:function_calls>")])
def test_eos_after_empty_atem_opener_fails_closed(split):
    text = "to=functions.echo<|message|><atem:function_calls>"
    with pytest.raises(ValueError, match="incomplete ATEM tool call"):
        _finish(text, split, "stop")
    parser, content, reasoning, calls = _finish(text, split, "length")
    assert (content, reasoning, calls, parser.tool_count) == ("", "", [], 0)


def test_max_tokens_inside_a_channel_header_is_dropped():
    text = "to=user<|message|>Sure.<|eom|>to=functions.echo<|mess"
    for split in range(len(text) + 1):
        _, content, _, tool_calls = _finish(text, split, "length")
        assert (content, tool_calls) == ("Sure.", [])
        with pytest.raises(ValueError, match="incomplete"):
            _finish(text, split, "stop")


def test_max_tokens_after_a_complete_atem_call_keeps_that_call():
    text = (
        "to=functions.echo<|message|><atem:function_calls>" + ATEM
        + "</atem:function_calls><|eom|>to=functions.echo<|message|>"
        "<atem:function_calls>" + ATEM[:40]
    )
    for split in range(len(text) + 1):
        parser, content, _, tool_calls = _finish(text, split, "length")
        assert content == "" and parser.tool_count == 1
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"text": "  a<b  ", "count": 2}


def test_client_stop_string_inside_an_atem_call_drops_the_partial_call():
    # Muse matches stop strings over the raw stream, markup included; a stop
    # that lands inside a call is a requested stop, as in OutputParser.
    text = "to=functions.echo<|message|><atem:function_calls>" + ATEM
    parser, events = collect(text, tools=TOOLS, stops=["count"])
    assert parser.stopped and parser.tool_count == 0
    assert not any("tool_calls" in event for event in events)
