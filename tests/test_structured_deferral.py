"""Structured output combined with thinking: the grammar is deferred past the
adapter's thinking-close marker."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mlx2.structured_output import (
    StructuredOutputProcessor,
    ThinkingBudgetProcessor,
    compile_constraint,
    make_structured_processor,
    structured_receipt,
)

PIECES = [
    "<eos>", "<pad>", "Let", " me", " think", "</think>", "\n\n", "{", '"a"', ":", "1", "}",
    "hello", " ", "</", "think", ">", "\n\n{", "yes", "no",
    "<tool_call>\n<function=sum>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
    "<tool_call><function=missing></function></tool_call>",
]
EOS, THINK_CLOSE, BLANK, HELLO = 0, 5, 6, 12
TOOL_CALL, MALFORMED_TOOL_CALL = len(PIECES) - 2, len(PIECES) - 1
SPLIT_MARKER = (14, 15, 16)  # "</" "think" ">"


def _tokenizer():
    return NS(
        vocab_size=len(PIECES),
        eos_token_ids=[EOS],
        decode=lambda ids, **_kw: "".join(PIECES[i] for i in ids),
        encode=lambda text, **_kw: [PIECES.index(text)],
    )


def _admitted(processor, generated, prompt=(1, 1)):
    import mlx.core as mx

    logits = mx.zeros((1, len(PIECES)))
    out = processor(mx.array(list(prompt) + list(generated), dtype=mx.uint32), logits)
    return set(np.flatnonzero(np.isfinite(np.array(out)[0])).tolist())


EVERYTHING = set(range(len(PIECES)))


def test_thinking_budget_forces_a_multi_token_close_from_history_only():
    processor = ThinkingBudgetProcessor(2, 2, SPLIT_MARKER)
    assert _admitted(processor, []) == EVERYTHING
    assert _admitted(processor, [2]) == EVERYTHING
    assert _admitted(processor, [2, 3]) == {SPLIT_MARKER[0]}
    assert _admitted(processor, [2, 3, SPLIT_MARKER[0]]) == {SPLIT_MARKER[1]}
    assert _admitted(processor, [2, 3, *SPLIT_MARKER[:2]]) == {SPLIT_MARKER[2]}
    assert _admitted(processor, [2, 3, *SPLIT_MARKER]) == EVERYTHING
    assert processor.fired is True
    # A speculative rollback below the boundary recomputes the no-op state.
    assert _admitted(processor, [2]) == EVERYTHING
    assert processor.fired is False


def test_thinking_budget_continues_a_marker_started_just_before_the_boundary():
    processor = ThinkingBudgetProcessor(2, 2, SPLIT_MARKER)
    # The first marker token arrived naturally as the final budgeted token.
    assert _admitted(processor, [2, SPLIT_MARKER[0]]) == {SPLIT_MARKER[1]}
    assert _admitted(processor, [2, *SPLIT_MARKER[:2]]) == {SPLIT_MARKER[2]}
    committed = [1, 1, 2, *SPLIT_MARKER]
    assert processor.fired_for_tokens(committed) is True


def test_thinking_budget_only_materializes_the_generated_suffix():
    class Tokens:
        def __init__(self, prompt_length, generated):
            self.prompt_length = prompt_length
            self.generated = generated
            self.slices = []

        def __getitem__(self, item):
            self.slices.append(item)
            assert item == slice(self.prompt_length, None)
            return NS(tolist=lambda: list(self.generated))

    processor = ThinkingBudgetProcessor(100_000, 8, SPLIT_MARKER)
    tokens = Tokens(100_000, [2, 3, 4])
    logits = object()
    assert processor(tokens, logits) is logits
    assert processor.fired_for_tokens(tokens) is False
    assert tokens.slices == [slice(100_000, None), slice(100_000, None)]


@pytest.mark.parametrize("engine", ["automaton", "scanner"])
def test_deferred_passthrough_then_activation_exactly_after_the_marker(engine, monkeypatch):
    if engine == "scanner":
        monkeypatch.setenv("MLX2_STRUCTURED_AUTOMATON", "0")
    processor = make_structured_processor(
        _tokenizer(), 2, response_format={"type": "json_object"}, defer_until=(THINK_CLOSE,)
    )
    assert processor.engine == engine and processor.deferred is True
    # While thinking, nothing is masked: not EOS, not JSON-looking reasoning.
    assert _admitted(processor, []) == EVERYTHING
    assert _admitted(processor, [2, 3, 7, 8]) == EVERYTHING
    assert processor.constraining is False
    assert structured_receipt(processor, completion_tokens=5) == {
        "engine": engine, "tail_mass_bound": 0.0, "parallel_scans": 0,
        "deferred": True, "deferred_tokens": 5,
    }
    # The marker itself was generated unconstrained; the very next token is
    # the first constrained one, from an empty constrained prefix.  Reasoning
    # text that looked like JSON does not count as constrained output.
    thinking = [2, 3, 7, 8, THINK_CLOSE]
    assert _admitted(processor, thinking) == {BLANK, 7, 13, 17}
    assert processor.constraining is True and processor.deferred_tokens == 5
    # Whitespace after the marker is tolerated (Qwen emits a blank line) ...
    assert _admitted(processor, thinking + [BLANK]) == {BLANK, 7, 13, 17}
    assert _admitted(processor, thinking + [BLANK, 7]) == {BLANK, 8, 11, 13}
    # ... and the grammar then runs to completion, EOS only at the end.
    assert _admitted(processor, thinking + [BLANK, 7, 8, 9, 10]) == {BLANK, 10, 11, 13}
    assert _admitted(processor, thinking + [BLANK, 7, 8, 9, 10, 11]) == {EOS}
    assert _admitted(processor, thinking + [17, 11]) == {EOS}
    assert processor.failure is None
    assert structured_receipt(processor, completion_tokens=11)["deferred_tokens"] == 5
    # No blank line is required either.
    assert _admitted(processor, thinking + [7, 8, 9, 10, 11]) == {EOS}
    # Rolling back before the marker returns to passthrough.
    assert _admitted(processor, [2, 3]) == EVERYTHING and processor.constraining is False


def test_multi_token_marker_activates_only_on_the_full_sequence():
    processor = make_structured_processor(
        _tokenizer(), 2, response_format={"type": "json_object"}, defer_until=SPLIT_MARKER
    )
    assert _admitted(processor, [2, 14, 15]) == EVERYTHING  # "</think" is not the marker
    assert _admitted(processor, [2, 14, 15, 3, 16]) == EVERYTHING  # interrupted
    assert _admitted(processor, [2, 14, 14, 15, 16]) == {BLANK, 7, 13, 17}
    assert processor.deferred_tokens == 5
    assert _admitted(processor, [2, 14, 14, 15, 16, 7]) == {BLANK, 8, 11, 13}


def test_leading_whitespace_is_exact_and_json_only():
    # JSON kinds: deferral widens the language by leading JSON whitespace only.
    deferred = compile_constraint({"type": "json_object"}, leading_whitespace=True)
    plain = compile_constraint({"type": "json_object"})
    assert deferred.fullmatch('\n\n {"a": 1}') and not plain.fullmatch('\n\n{"a": 1}')
    assert deferred.fullmatch('{"a": 1}') and not deferred.fullmatch('x{"a": 1}')
    assert not deferred.fullmatch("\n\n")
    schema = {"type": "json_schema", "json_schema": {"strict": True, "schema": {"type": "integer"}}}
    assert compile_constraint(schema, leading_whitespace=True).fullmatch("\n\n-12")
    # The canonicalizer stays sound with the widened grammar.
    for prefix in ("", "\n\n", '\n\n{"k', '\n {"k": [1, {"z": "long text'):
        canonical = deferred.canonicalize(prefix)
        for piece in PIECES[1:]:
            assert bool(deferred.fullmatch(prefix + piece, partial=True)) == bool(
                deferred.fullmatch(canonical + piece, partial=True)
            ), (prefix, piece)
    # A raw grammar is the client's exact language: never widened.
    processor = make_structured_processor(_tokenizer(), 2, grammar="yes|no", defer_until=(THINK_CLOSE,))
    assert _admitted(processor, [2, THINK_CLOSE]) == {18, 19}
    assert _admitted(processor, [2, THINK_CLOSE, 18]) == {EOS}
    # Without deferral nothing changes: whitespace before the object is refused.
    assert _admitted(make_structured_processor(_tokenizer(), 2, response_format={"type": "json_object"}), []) == {7}


def test_never_activated_run_masks_nothing_and_reports_every_token_deferred():
    processor = StructuredOutputProcessor(
        _tokenizer(), 2, compile_constraint({"type": "json_object"}, leading_whitespace=True),
        defer_until=(THINK_CLOSE,),
    )
    generated = []
    for token in (2, 3, 4, 4, 4, 4):
        assert _admitted(processor, generated) == EVERYTHING  # EOS included
        generated.append(token)
    assert processor.failure is None and processor.constraining is False
    receipt = structured_receipt(processor, completion_tokens=len(generated))
    assert receipt["deferred"] is True and receipt["deferred_tokens"] == 6
    # Not deferred at all: the receipt says so.
    plain = StructuredOutputProcessor(_tokenizer(), 2, compile_constraint({"type": "json_object"}))
    assert structured_receipt(plain, 9)["deferred"] is False
    assert structured_receipt(plain, 9)["deferred_tokens"] == 0


def test_validation_accepts_thinking_only_when_the_adapter_declares_the_marker():
    from mlx2.server import validate_request

    base = {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}}
    with pytest.raises(ValueError, match="thinking to be disabled"):
        validate_request({**base, "enable_thinking": True})
    accepted = validate_request({**base, "reasoning_effort": "high"}, structured_thinking=True)
    assert accepted["enable_thinking"] is True
    validate_request(
        {"messages": base["messages"], "grammar": "[a-z]+", "think": True}, structured_thinking=True
    )
    # Main deliberately rejects min_tokens with structured output because the
    # two EOS masks can otherwise produce empty support.
    with pytest.raises(ValueError, match="min_tokens cannot be combined"):
        validate_request(
            {**base, "enable_thinking": True, "min_tokens": 2},
            structured_thinking=True,
        )
    # Raw completions are unchanged.
    validate_request({"prompt": "x", "grammar": "[a-z]+", "enable_thinking": True}, chat=False)
    accepted = validate_request(
        {**base, "response_format": {"type": "text"}, "think": True,
         "options": {"thinking_budget": 0}},
        structured_thinking=True,
    )
    assert accepted["thinking_budget"] == 0
    with pytest.raises(ValueError, match="integer between 0 and 2097152"):
        validate_request({**base, "thinking_budget": True})
    # Main defers adapter-marker availability to the loaded-route check.
    assert validate_request(
        {"messages": base["messages"], "think": True, "thinking_budget": 4}
    )["thinking_budget"] == 4


def test_adapter_marker_accessor_requires_an_exact_nonempty_tokenization():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.serving import thinking_close_token_ids

    adapter = FlashNextAdapter.__new__(FlashNextAdapter)
    adapter.tokenizer = NS(encode=lambda text, **_kw: [77], decode=lambda ids, **_kw: "</think>")
    assert adapter.thinking_close_token_ids() == (77,)
    assert thinking_close_token_ids(adapter) == (77,)
    adapter.tokenizer = NS(encode=lambda text, **_kw: [1, 2, 3], decode=lambda ids, **_kw: "</think>")
    assert adapter.thinking_close_token_ids() is None
    adapter.tokenizer = NS(encode=lambda text, **_kw: [9], decode=lambda ids, **_kw: "<unk>")
    assert adapter.thinking_close_token_ids() is None
    assert thinking_close_token_ids(NS()) is None  # adapters without the accessor fail closed


@pytest.fixture
def scripted_engine(monkeypatch):
    """A real ServingEngine loop over a scripted model that honours processors."""
    import mlx.core as mx

    from mlx2 import memory, serving, thinking_calibration
    from mlx2.contracts import Capability
    from mlx2.runtime import apc_v2, generate, os_memory, pld

    state = {
        "script": [],
        "declare_marker": True,
        "declare_tool_constraint": True,
        "thinking_direction": None,
        "poison_budget_flag": False,
    }

    class APC:
        def __init__(self, **kw): self.apc_stats = {}
        def key(self, *a, **kw): return "key"
        def lookup(self, key, tokens, **kw):
            return NS(cache=[NS(nbytes=0)], cached_tokens=len(tokens) - 1, remaining_tokens=[1],
                      sidecar=None, miss_reason=None)
        def store(self, *a, **kw): pass
        def spill_idle_entries(self): pass
        def evict_oldest_unleased(self): return False
        def clear(self): pass
        def __len__(self): return 0

    class Batch:
        scheduler_stats = {}

        @staticmethod
        def validate_policy(policy):
            return dict(policy)

        def __init__(self, *a, **kw):
            self.lanes = {}
            self.uid = 0
            self.num_draft = 1

        def insert(self, prompts, max_tokens=None, logits_processors=None, **kw):
            uid = self.uid
            self.uid += 1
            self.lanes[uid] = {"generated": [], "max": max_tokens[0], "processors": logits_processors[0]}
            state.setdefault("inserted", []).append(kw)
            return [uid]

        def next(self):
            responses = []
            for uid, lane in list(self.lanes.items()):
                step = len(lane["generated"])
                # The model wants "hello" most and the scripted token second;
                # only a grammar can stop it from saying hello.
                row = np.zeros(len(PIECES), dtype=np.float32)
                row[HELLO], row[11], row[7] = 4.0, 3.0, 2.0  # hello, then "}", then "{"
                if step < len(state["script"]):
                    row[state["script"][step]] = 8.0
                logits = mx.array(row)[None]
                context = mx.array([1, 1] + lane["generated"], dtype=mx.uint32)
                for processor in lane["processors"]:
                    logits = processor(context, logits)
                if state["poison_budget_flag"]:
                    for processor in lane["processors"]:
                        if isinstance(processor, ThinkingBudgetProcessor):
                            # Simulate a speculative/uncommitted row mutating
                            # the old diagnostic flag after the committed mask.
                            processor.fired = True
                token = int(mx.argmax(logits[0]).item())
                lane["generated"].append(token)
                finish = "stop" if token == EOS else "length" if len(lane["generated"]) >= lane["max"] else None
                if finish:
                    del self.lanes[uid]
                responses.append(NS(uid=uid, execution_width=1, finish_reason=finish, token=token,
                                    mtp_state=None,
                                    all_tokens=[1, 1] + lane["generated"] if finish else None,
                                    prompt_cache=[], mtp_receipt=None, logprobs=None))
            return [], responses

        def remove(self, uids):
            for uid in uids:
                self.lanes.pop(uid, None)

        def close(self): pass

    class Detokenizer:
        def __init__(self): self.segment = ""
        def reset(self): self.segment = ""
        def add_token(self, token): self.segment += PIECES[token]
        def finalize(self): pass

        @property
        def last_segment(self):
            segment, self.segment = self.segment, ""
            return segment

    class Adapter:
        max_context = 2000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None

        def __init__(self, path):
            self.tokenizer = _tokenizer()
            self.tokenizer.detokenizer = Detokenizer()
            self.descriptor = NS(capabilities=frozenset({Capability.PROMPT_LOOKUP}))
            if state["declare_marker"]:
                self.thinking_close_token_ids = lambda: (THINK_CLOSE,)
            if not state["declare_tool_constraint"]:
                self.tool_constraint = None
            self._thinking_direction = state["thinking_direction"]
            if state.get("sampling_defaults") is not None:
                self.sampling_defaults = state["sampling_defaults"]

        def profile_name(self, mtp): return "fake"
        def execution_config(self, **kw): return {"num_draft": 0}
        def prompt_tokens(self, request): return [1, 1]
        def thinking_commit_direction(self): return self._thinking_direction

        def output_parser(self, request):
            from mlx2.output import OutputParser, constrained_tool_choice
            from mlx2.runtime.tool_parsers.qwen3_coder import parse_tool_call

            return OutputParser(
                chat="messages" in request,
                thinking=request.get("enable_thinking", False),
                tools=request.get("tools"),
                parse_tool=parse_tool_call,
                constrained_tools=constrained_tool_choice(request),
                parallel_tool_calls=request.get("parallel_tool_calls", True),
                tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
            )

        def tool_constraint(self, request):
            from mlx2.runtime.tool_parsers.qwen3_coder import constrained_tool_grammar

            return constrained_tool_grammar(
                request["tools"],
                request["tool_choice"],
                parallel_tool_calls=request.get("parallel_tool_calls", True),
            )

        def diagnostics(self): return {}
        def close(self): pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    monkeypatch.setattr(pld, "PromptLookupBatchGenerator", Batch)
    monkeypatch.setattr(mx, "synchronize", lambda: None)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    monkeypatch.setattr(
        thinking_calibration,
        "supports_calibration",
        lambda _adapter: state["thinking_direction"] is not None,
    )
    monkeypatch.setattr(
        thinking_calibration,
        "resolve_direction",
        lambda *_args, **_kwargs: (
            state["thinking_direction"],
            {"state": "calibrated", "source": "test"},
        ),
    )

    engines = []

    def build(
        *, declare_marker, declare_tool_constraint=True, execution_policy=None,
        thinking_direction=None, thinking_steer_alpha=None,
        default_max_tokens=65_536, qualification_mode=True, prompt_lookup=False,
        **engine_kwargs,
    ):
        state["declare_marker"] = declare_marker
        state["declare_tool_constraint"] = declare_tool_constraint
        state["thinking_direction"] = thinking_direction
        engine = serving.ServingEngine(
            "fake", adapter_factory=Adapter, qualification_mode=qualification_mode, mtp=False,
            execution_policy=execution_policy,
            thinking_steer_alpha=thinking_steer_alpha,
            default_max_tokens=default_max_tokens,
            prompt_lookup=prompt_lookup,
            **engine_kwargs,
        )
        assert engine.ready.wait(5)
        with engine.lock:
            engine.route_capabilities = frozenset(engine.route_capabilities | {Capability.GRAMMAR})
        engines.append(engine)
        return engine

    yield build, state
    for engine in engines:
        engine.close()
        assert not engine.error


def _collect(job):
    reasoning, content = [], []
    while True:
        event = job.events.get(timeout=10)
        if "delta" in event:
            reasoning.append(event["delta"].get("reasoning_content", ""))
            content.append(event["delta"].get("content", ""))
        if "finish_reason" in event or "error" in event:
            return "".join(reasoning), "".join(content), event


def test_null_constraint_fields_mean_absent_and_keep_the_worker_alive(scripted_engine):
    from mlx2.server import validate_request

    build, state = scripted_engine
    engine = build(declare_marker=True)
    base = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 2}
    for field in ("response_format", "grammar"):
        assert field not in validate_request({**base, field: None})
    both = validate_request({**base, "response_format": None, "grammar": None})
    assert "response_format" not in both and "grammar" not in both
    assert validate_request({**base, "response_format": None, "grammar": "yes|no"})["grammar"] == "yes|no"
    # A concurrent unconstrained lane that is still decoding when the null
    # requests finish: it must not be taken down with them.
    state["script"] = [3] * 40
    other = engine.submit({**base, "messages": [{"role": "user", "content": "x"}], "max_tokens": 50})
    for field in ("response_format", "grammar"):
        # The engine is also reachable without the HTTP validator; a null
        # constraint there is no constraint and the receipt must say so.
        for body in (validate_request({**base, field: None}), {**base, field: None}):
            _, _, final = _collect(engine.submit(body))
            assert final.get("finish_reason") == "length", final
            assert final["receipt"]["request_controls"]["structured_output"] is None
    _, _, other_final = _collect(other)
    assert other_final.get("finish_reason") == "length", other_final
    assert engine.thread.is_alive() and engine.error is None
    _, _, later = _collect(engine.submit(dict(base)))
    assert later.get("finish_reason") == "length", later


def test_a_failing_terminal_receipt_fails_only_its_own_request(scripted_engine, monkeypatch):
    from mlx2 import structured_output

    build, state = scripted_engine
    engine = build(declare_marker=True)

    def broken_receipt(*_args, **_kwargs):
        raise RuntimeError("injected receipt fault")

    monkeypatch.setattr(structured_output, "structured_receipt", broken_receipt)
    state["script"] = [7, 8, 9, 10, 11] + [3] * 40
    other = engine.submit(
        {"messages": [{"role": "user", "content": "x"}], "temperature": 0, "max_tokens": 30}
    )
    _, _, final = _collect(
        engine.submit(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": 8,
            }
        )
    )
    assert final["status"] == 500 and "finish_reason" not in final, final
    _, _, other_final = _collect(other)
    assert other_final.get("finish_reason") == "length", other_final
    assert engine.thread.is_alive() and engine.error is None
    assert engine.counts["terminal_receipt_failures"] == 1
    with engine.lock:
        assert not engine.jobs
    assert engine.slots.acquire(blocking=False)
    engine.slots.release()


def test_serving_receipt_records_effective_output_limit_and_defaulting(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["script"] = [EOS]
    _, _, defaulted = _collect(
        engine.submit(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0,
                "top_k": 5,
            }
        )
    )
    controls = defaulted["receipt"]["request_controls"]
    assert controls["max_tokens"] == 1998
    assert controls["max_tokens_defaulted"] is True
    assert controls["default_max_tokens"] == 65_536
    assert engine.status()["settings"]["default_max_tokens"] == 65_536

    state["script"] = [EOS]
    _, _, explicit = _collect(
        engine.submit(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                "temperature": 0,
                "top_k": 5,
            }
        )
    )
    controls = explicit["receipt"]["request_controls"]
    assert controls["max_tokens"] == 4
    assert controls["max_tokens_defaulted"] is False
    assert controls["default_max_tokens"] == 65_536

    lowered = build(declare_marker=True, default_max_tokens=512)
    assert lowered.status()["settings"]["default_max_tokens"] == 512
    state["script"] = [EOS]
    _, _, old_default = _collect(
        lowered.submit(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0,
                "top_k": 5,
            }
        )
    )
    controls = old_default["receipt"]["request_controls"]
    assert controls["max_tokens"] == 512
    assert controls["max_tokens_defaulted"] is True
    assert controls["default_max_tokens"] == 512

    _, _, overflow = _collect(
        engine.submit(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1999,
                "temperature": 0,
                "top_k": 5,
            }
        )
    )
    assert overflow["status"] == 400
    assert overflow["code"] == "context_length_exceeded"
    assert "maximum context length" in overflow["error"]


THINKING_JSON_REQUEST = {
    "messages": [{"role": "user", "content": "hi"}],
    "response_format": {"type": "json_object"},
    "enable_thinking": True,
    "max_tokens": 16,
    "temperature": 0,
    "top_k": 5,
}


def test_serving_defers_the_grammar_and_splits_reasoning_from_json(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    assert engine.status()["structured_output"]["thinking_deferral"] is True
    # Think, close, blank line, then the model only ever wants to say "hello".
    state["script"] = [2, 3, 4, HELLO, THINK_CLOSE, BLANK]
    reasoning, content, final = _collect(engine.submit(dict(THINKING_JSON_REQUEST)))
    assert "error" not in final, final
    assert reasoning == "Let me thinkhello"  # unconstrained, hello included
    assert json.loads(content) == {} and content.startswith("\n\n")
    assert final["finish_reason"] == "stop"
    receipt = final["receipt"]["request_controls"]
    assert receipt["thinking"] is True
    assert receipt["structured_output"] == {
        "kind": "json_object", "enforced": True, "engine": "automaton",
        "tail_mass_bound": 0.0, "parallel_scans": 0, "deferred": True, "deferred_tokens": 5,
    }
    # Thinking off: constrained from the first token, not deferred.
    state["script"] = [7, 8, 9, 10, 11]
    reasoning, content, final = _collect(
        engine.submit({**THINKING_JSON_REQUEST, "enable_thinking": False})
    )
    assert reasoning == "" and json.loads(content) == {"a": 1}
    structured = final["receipt"]["request_controls"]["structured_output"]
    assert structured["deferred"] is False and structured["deferred_tokens"] == 0
    assert structured["engine"] == "automaton"


def test_thinking_budget_closes_then_deferred_grammar_engages(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["script"] = [2, 3, 4, 4, 4, 4]
    reasoning, content, final = _collect(
        engine.submit({
            **THINKING_JSON_REQUEST,
            "thinking_budget": 2,
            "thinking_budget_mode": "history",
        })
    )
    assert "error" not in final, final
    assert reasoning == "Let me"
    assert json.loads(content) == {}
    controls = final["receipt"]["request_controls"]
    assert controls["thinking_budget"] == 2
    assert controls["thinking_budget_fired"] is True
    assert controls["thinking_mechanisms"] == {
        "state_aware_guard": True,
        "history_budget": True,
        "steering": False,
    }
    assert controls["structured_output"]["deferred_tokens"] == 3
    assert engine.counts["thinking_budget_forced_closes"] == 1


def test_history_budget_keeps_alpha_steering_and_the_state_guard(scripted_engine):
    import mlx.core as mx

    build, state = scripted_engine
    engine = build(
        declare_marker=True,
        thinking_direction={
            "layer": 1,
            "vector": mx.ones((1,)),
            "source": "test",
        },
        thinking_steer_alpha=0.2,
    )
    state["script"] = [2, 3, 4, 4]
    _, content, final = _collect(
        engine.submit(
            {
                **THINKING_JSON_REQUEST,
                "thinking_budget": 2,
                "thinking_budget_mode": "history",
            }
        )
    )
    assert json.loads(content) == {}
    controls = final["receipt"]["request_controls"]
    assert controls["thinking_mechanisms"] == {
        "state_aware_guard": True,
        "history_budget": True,
        "steering": True,
    }
    assert controls["thinking_guard"]["budget"] is None
    assert controls["thinking_guard"]["steering"]["alpha"] == 0.2
    assert engine.status()["settings"]["thinking_steer"]["calibration"] == {
        "state": "calibrated",
        "source": "test",
    }


def test_thinking_budget_receipt_uses_committed_tokens_not_mutable_flag(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["poison_budget_flag"] = True
    # The marker is emitted naturally before the boundary. The fake poisons
    # ``processor.fired`` after every row, but committed terminal history is
    # authoritative for both the receipt and counter.
    state["script"] = [THINK_CLOSE, BLANK, 7, 8, 9, 10, 11]
    _, content, final = _collect(
        engine.submit({
            **THINKING_JSON_REQUEST,
            "thinking_budget": 2,
            "thinking_budget_mode": "history",
        })
    )
    assert json.loads(content) == {"a": 1}
    assert final["receipt"]["request_controls"]["thinking_budget_fired"] is False
    assert engine.counts["thinking_budget_forced_closes"] == 0


def test_forced_strict_tool_call_defers_past_budgeted_thinking(scripted_engine):
    build, state = scripted_engine
    engine = build(
        declare_marker=True,
        execution_policy={"constrained_tool_grammar": True},
    )
    state["script"] = [2, 3, 4, TOOL_CALL, EOS]
    request = {
        "messages": [{"role": "user", "content": "add"}],
        "enable_thinking": True,
        "thinking_budget": 2,
        "thinking_budget_mode": "history",
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "tools": [{
            "type": "function",
            "function": {
                "name": "sum",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
            },
        }],
        "max_tokens": 8,
        "temperature": 0,
        "top_k": 5,
    }
    job = engine.submit(request)
    calls = []
    reasoning = []
    while True:
        event = job.events.get(timeout=10)
        if "delta" in event:
            reasoning.append(event["delta"].get("reasoning_content", ""))
            calls.extend(event["delta"].get("tool_calls", ()))
        if "finish_reason" in event or "error" in event:
            final = event
            break
    assert "error" not in final, final
    assert "".join(reasoning) == "Let me"
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "sum"
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}
    controls = final["receipt"]["request_controls"]
    assert controls["thinking_budget_fired"] is True
    assert controls["structured_output"]["kind"] == "tool_choice"
    assert controls["structured_output"]["deferred"] is True
    assert controls["tool_choice"] == {
        "mode": "required",
        "name": None,
        "strict": True,
        "parallel": False,
        "enforced": True,
        "decode_grammar": "engaged",
    }


def test_unconstrained_tool_parse_fallback_is_counted_in_serving(scripted_engine):
    build, state = scripted_engine
    engine = build(
        declare_marker=True,
        execution_policy={"tolerant_tool_markers": True},
    )
    state["script"] = [MALFORMED_TOOL_CALL, EOS]
    request = {
        "messages": [{"role": "user", "content": "answer"}],
        "tools": [{
            "type": "function",
            "function": {"name": "sum", "parameters": {}},
        }],
        "tool_choice": "auto",
        "enable_thinking": False,
        "max_tokens": 4,
        "temperature": 0,
        "top_k": 5,
    }
    _, content, final = _collect(engine.submit(request))
    assert "error" not in final
    assert content == PIECES[MALFORMED_TOOL_CALL]
    assert engine.counts["tool_call_parse_fallbacks"] == 1


def test_auto_parallel_false_truncates_a_second_call(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["script"] = [TOOL_CALL, TOOL_CALL]
    request = {
        "messages": [{"role": "user", "content": "answer"}],
        "tools": [{
            "type": "function",
            "function": {"name": "sum", "parameters": {}},
        }],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "enable_thinking": False,
        "max_tokens": 4,
        "temperature": 0,
        "top_k": 5,
    }
    job = engine.submit(request)
    calls, final = [], None
    while True:
        event = job.events.get(timeout=10)
        if "delta" in event:
            calls.extend(event["delta"].get("tool_calls", ()))
        if "finish_reason" in event or "error" in event:
            final = event
            break
    # Pre-fix this lane finished with {"error": ..., "status": 502}, which every
    # surface shaped as a server fault (Chat 502, Responses/Anthropic in-stream
    # error) for something only the model did.  The bound is honoured instead.
    assert "error" not in final
    assert len(calls) == 1
    assert engine.counts["tool_call_constraint_failures"] == 0
    assert engine.counts["tool_call_constraint_truncations"] == 1


def test_serving_length_or_eos_before_the_marker_is_not_a_failure(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    # Length: the budget ends inside the reasoning channel.
    state["script"] = [2, 3, 4, 4, 4, 4, 4, 4]
    reasoning, content, final = _collect(engine.submit({**THINKING_JSON_REQUEST, "max_tokens": 4}))
    assert final.get("finish_reason") == "length" and "error" not in final
    assert reasoning == "Let me think think" and content == ""
    # No grammar has engaged yet, so a generation terminal keeps main's normal
    # stop semantics instead of forcing the model to continue thinking.
    state["script"] = [2, 3, EOS]
    reasoning, content, final = _collect(engine.submit(dict(THINKING_JSON_REQUEST)))
    assert final.get("finish_reason") == "stop" and "error" not in final
    assert reasoning == "Let me" and content == ""
    assert engine.counts["structured_output_failures"] == 0


def test_prompt_lookup_structured_stop_inside_thinking_finishes_normally(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True, prompt_lookup=True)
    assert engine.status()["settings"]["speculation"] == "prompt_lookup"
    state["script"] = [2, 3, EOS]
    reasoning, content, final = _collect(engine.submit(dict(THINKING_JSON_REQUEST)))
    assert final.get("finish_reason") == "stop" and "error" not in final
    assert reasoning == "Let me" and content == ""
    assert engine.counts["structured_output_failures"] == 0


def test_serving_dead_end_receipt_is_qualification_only(scripted_engine, caplog):
    build, state = scripted_engine
    request = {
        "prompt": "x",
        "grammar": r"\{x",
        "max_tokens": 4,
        "temperature": 0,
    }

    qualified = build(declare_marker=False, qualification_mode=True)
    assert qualified.status()["counts"]["structured_output_dead_ends"] == 0
    state["script"] = [7, HELLO]
    _, _, qualified_event = _collect(qualified.submit(request))
    assert qualified_event["status"] == 502
    context = qualified_event["mlx2"]["structured_output_failure"]
    assert context["schema"] == "mlx2.structured-output-dead-end.v1"
    assert context["generated_tokens"] == 1
    assert context["recent_tokens"][0]["id"] == 7
    assert len(context["recent_tokens"]) <= 64
    assert len(context["top_logits"]) <= 8
    assert qualified.counts["structured_output_dead_ends"] == 1
    assert "structured-output dead end receipt=" in caplog.text

    normal = build(declare_marker=False)
    normal.qualification_mode = False
    state["script"] = [7, HELLO]
    _, _, normal_event = _collect(normal.submit(request))
    assert normal_event["status"] == 502
    assert "mlx2" not in normal_event
    assert normal.counts["structured_output_dead_ends"] == 1


def test_required_tool_choice_is_left_for_main_post_generation_validation(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["script"] = [2, 3, EOS]
    request = {
        "messages": [{"role": "user", "content": "call"}],
        "enable_thinking": True,
        "tool_choice": "required",
        "tools": [{
            "type": "function",
            "function": {"name": "sum", "parameters": {}},
        }],
        "max_tokens": 4,
        "temperature": 0,
        "top_k": 5,
    }
    _, _, final = _collect(engine.submit(request))
    assert final["finish_reason"] == "stop"
    assert final["receipt"]["request_controls"]["structured_output"] is None

    # A client stop retains the engine's main behavior too; the HTTP boundary
    # applies enforce_tool_contract after collecting the completed message.
    state["script"] = [2, 3]
    _, _, final = _collect(engine.submit({**request, "stop": " me"}))
    assert final["finish_reason"] in {"stop", "length"}
    assert "error" not in final
    assert engine.counts["structured_output_failures"] == 0


def test_opt_in_tool_grammar_skips_unsupported_adapter_and_conflicts(scripted_engine):
    build, state = scripted_engine
    engine = build(
        declare_marker=True,
        declare_tool_constraint=False,
        execution_policy={"constrained_tool_grammar": True},
    )
    request = {
        "messages": [{"role": "user", "content": "call"}],
        "enable_thinking": False,
        "tool_choice": "required",
        "tools": [{"type": "function", "function": {"name": "sum", "parameters": {}}}],
        "max_tokens": 4,
        "temperature": 0,
        "top_k": 5,
    }
    state["script"] = [HELLO, EOS]
    _, content, final = _collect(engine.submit(request))
    assert "error" not in final and content == "hello"
    assert engine.counts["constrained_tool_grammar_skips"] == 1
    assert final["receipt"]["request_controls"]["tool_choice"][
        "decode_grammar"
    ] == "skipped_adapter_unsupported"

    engine = build(
        declare_marker=True,
        execution_policy={"constrained_tool_grammar": True},
    )
    state["script"] = [7, 8, 9, 10, 11]
    _, content, final = _collect(
        engine.submit({
            **request,
            "response_format": {"type": "json_object"},
            "max_tokens": 8,
        })
    )
    assert json.loads(content) == {"a": 1}
    assert engine.counts["constrained_tool_grammar_skips"] == 1
    assert final["receipt"]["request_controls"]["tool_choice"][
        "decode_grammar"
    ] == "skipped_request_combination"


def test_serving_behavior_policy_flags_require_exact_booleans():
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="constrained_tool_grammar must be boolean"):
        ServingEngine("unused", execution_policy={"constrained_tool_grammar": 1})
    with pytest.raises(ValueError, match="tolerant_tool_markers must be boolean"):
        ServingEngine("unused", execution_policy={"tolerant_tool_markers": "yes"})


def test_serving_still_rejects_thinking_for_adapters_without_a_marker(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=False)
    assert engine.status()["structured_output"]["thinking_deferral"] is False
    state["script"] = [2, THINK_CLOSE]
    _, _, final = _collect(engine.submit(dict(THINKING_JSON_REQUEST)))
    assert final["status"] == 400
    assert final["error"] == "structured output requires thinking to be disabled"
    # Text response_format with thinking is not structured output: unaffected.
    state["script"] = [2, THINK_CLOSE, HELLO, EOS]
    reasoning, content, final = _collect(
        engine.submit({**THINKING_JSON_REQUEST, "response_format": {"type": "text"}})
    )
    assert "error" not in final and reasoning == "Let" and content == "hello"


def test_qualifier_thinking_probe_outcomes():
    import importlib.util
    import io
    from pathlib import Path
    from urllib.error import HTTPError

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("qualify_serving_probe", root / "scripts" / "qualify_serving.py")
    qualify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualify)

    def response(content, finish="stop", **receipt):
        return {
            "choices": [{"finish_reason": finish, "message": {"content": content, "reasoning_content": "r"}}],
            "mlx2": {"request_controls": {"structured_output": receipt}},
        }

    sent = []

    def ok(body):
        sent.append(body)
        if len(sent) == 1:  # the cap interrupted reasoning: retried with a larger budget
            return response("", finish="length", deferred=True, deferred_tokens=512, engine="automaton")
        return response("\n\n{\"answer\": 221}", deferred=True, deferred_tokens=40, engine="automaton")

    passed, evidence = qualify.run_structured_thinking_probe(ok, {"structured_output": {"thinking_deferral": True}})
    assert passed and evidence["outcome"] == "deferred" and evidence["engine"] == "automaton"
    assert len(sent) == 2 and sent[0]["enable_thinking"] is True
    assert sent[0]["response_format"] == {"type": "json_object"}
    # Valid JSON that was not deferred, or deferred non-objects, do not pass.
    assert not qualify.run_structured_thinking_probe(lambda body: response("{}", deferred=False), {})[0]
    assert not qualify.run_structured_thinking_probe(lambda body: response("[1]", deferred=True), {})[0]

    def refuse(body):
        raise HTTPError("u", 400, "Bad Request", {}, io.BytesIO(b'{"error": "structured output requires thinking to be disabled"}'))

    passed, evidence = qualify.run_structured_thinking_probe(refuse, {"structured_output": {"thinking_deferral": False}})
    assert passed and evidence["outcome"] == "skipped_unsupported"
    passed, evidence = qualify.run_structured_thinking_probe(refuse, {})  # older status: pass with a note
    assert passed and evidence["note"]
    # An adapter that declares the marker must not answer 400.
    assert not qualify.run_structured_thinking_probe(refuse, {"structured_output": {"thinking_deferral": True}})[0]

    def overloaded(body):
        raise HTTPError("u", 429, "Too Many", {}, io.BytesIO(b"busy"))

    passed, evidence = qualify.run_structured_thinking_probe(overloaded, {})
    assert not passed and evidence["outcome"] == "http_error"


# --- Answer-channel envelope (North frames answers as <|START_TEXT|>...<|END_TEXT|>).
ENVELOPE_PIECES = PIECES + ["<|START_TEXT|>", "<|END_TEXT|>"]
TEXT_OPEN, TEXT_CLOSE = len(PIECES), len(PIECES) + 1


def _envelope_processor(engine, monkeypatch, **kw):
    if engine == "scanner":
        monkeypatch.setenv("MLX2_STRUCTURED_AUTOMATON", "0")
    tokenizer = NS(
        vocab_size=len(ENVELOPE_PIECES), eos_token_ids=[EOS],
        decode=lambda ids, **_kw: "".join(ENVELOPE_PIECES[i] for i in ids),
        encode=lambda text, **_kw: [ENVELOPE_PIECES.index(text)],
    )
    processor = make_structured_processor(
        tokenizer, 2, response_format={"type": "json_object"},
        envelope=((TEXT_OPEN,), (TEXT_CLOSE,)), **kw,
    )
    assert processor.engine == engine

    def admitted(generated):
        import mlx.core as mx

        logits = mx.zeros((1, len(ENVELOPE_PIECES)))
        out = processor(mx.array([1, 1] + list(generated), dtype=mx.uint32), logits)
        return set(np.flatnonzero(np.isfinite(np.array(out)[0])).tolist())

    return processor, admitted


@pytest.mark.parametrize("engine", ["automaton", "scanner"])
def test_envelope_markers_are_admitted_only_where_the_framing_allows(engine, monkeypatch):
    processor, admitted = _envelope_processor(engine, monkeypatch)
    brace, key, colon, one, close = 7, 8, 9, 10, 11
    start = admitted([])
    assert TEXT_OPEN in start and brace in start and TEXT_CLOSE not in start and EOS not in start
    after_open = admitted([TEXT_OPEN])  # one opener only; the grammar is unchanged by it
    assert TEXT_OPEN not in after_open and brace in after_open and TEXT_CLOSE not in after_open
    middle = admitted([TEXT_OPEN, brace, key, colon])
    assert one in middle and not {TEXT_OPEN, TEXT_CLOSE, EOS} & middle
    complete = admitted([TEXT_OPEN, brace, key, colon, one, close])
    assert {TEXT_CLOSE, EOS} <= complete and TEXT_OPEN not in complete
    # After the closer only end-of-turn may follow.
    assert admitted([TEXT_OPEN, brace, key, colon, one, close, TEXT_CLOSE]) == {EOS}
    admitted([TEXT_OPEN, brace, key, colon, one, close, TEXT_CLOSE, EOS])
    assert processor.failure is None
    # The framing is optional: the same answer without it is still enforced.
    assert {TEXT_CLOSE, EOS} <= admitted([brace, key, colon, one, close])
    admitted([brace, key, colon, one, close, EOS])
    assert processor.failure is None


def test_envelope_composes_with_thinking_deferral(monkeypatch):
    processor, admitted = _envelope_processor("automaton", monkeypatch, defer_until=(THINK_CLOSE,))
    everything = set(range(len(ENVELOPE_PIECES)))
    assert admitted([2, 3, 4]) == everything  # still reasoning: unconstrained
    after_marker = admitted([2, 3, 4, THINK_CLOSE])
    assert TEXT_OPEN in after_marker and 7 in after_marker and HELLO not in after_marker
    assert 7 in admitted([2, 3, 4, THINK_CLOSE, TEXT_OPEN]) and HELLO not in admitted([2, 3, 4, THINK_CLOSE, TEXT_OPEN])
    receipt = structured_receipt(processor, 5)
    assert receipt["deferred"] is True and receipt["deferred_tokens"] == 4
