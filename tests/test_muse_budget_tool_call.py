"""A Muse thinking budget bounds reasoning, never a tool call or an answer.

Muse reasons in ``to=self`` messages and then either answers
(``<|eom|><|start|>assistant to=user<|message|>``), calls a tool
(``<|eom|><|start|>assistant to=<tool><|message|><atem:function_calls>...``),
or skips reasoning and opens its reply with a ``to=user`` or tool header.
The budget release forces the to=user switch, but only the first shape writes
it: the guard and the history-mode budget counted a call or a direct answer as
reasoning and, at the budget, forced the switch into the middle of the ATEM
arguments or the answer text (state-aware guard and Anthropic
``budget_tokens`` history mode alike).
"""
import mlx.core as mx
import pytest
from route_harness import CAPS, PIECES, make_engine, patch_host, run, tiny_qwen38_mtp

from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
from mlx2.contracts import Capability, ModelDescriptor, StatePlane

RELEASE = "<|eom|><|start|>assistant to=user<|message|>"
CALL = (
    '<atem:function_calls><atem:invoke name="f"><atem:parameter name="x">'
    + "0123456789" * 8
    + "</atem:parameter></atem:invoke></atem:function_calls>"
)
# The harness tokenizer is one token per character; digits hold no marker
# character.  The budget falls inside the call or answer, past the reasoning.
SCRIPTS = {
    "tool_call": " to=self<|message|>call it<|eom|><|start|>assistant to=f<|message|>" + CALL,
    "direct_call": " to=f<|message|>" + CALL,
    "direct_answer": " to=user<|message|>" + "0123456789" * 12,
}
BUDGET = 120
TOOLS = [{"type": "function", "function": {
    "name": "f", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}}]


def scripted_muse(script):
    class ScriptedMuse:
        """Muse's real thinking contract over the harness's char-level vocabulary."""

        descriptor = ModelDescriptor(
            model_type="tiny", family="tiny", variant="v",
            state_planes=frozenset({StatePlane.ATTENTION_KV}),
            capabilities=frozenset(CAPS | {Capability.REASONING, Capability.TOOLS}),
            cache_layout="tiny-layout",
        )
        thinking_enabled = staticmethod(MuseGlimmerAdapter.thinking_enabled)
        structured_answer_token_ids = MuseGlimmerAdapter.structured_answer_token_ids
        thinking_release_token_ids = MuseGlimmerAdapter.thinking_release_token_ids

        def __getattr__(self, name):
            # Every other thinking hook the Muse adapter declares.
            if name.startswith("thinking_") and hasattr(MuseGlimmerAdapter, name):
                return getattr(MuseGlimmerAdapter, name).__get__(self)
            raise AttributeError(name)

        def request_logits_processors(self, request, *, prompt_length):
            ids = self.tokenizer.encode(script) + [0]  # 0 is the harness EOS

            def scripted(tokens, logits):
                # The model "wants" the next script token; anything the
                # serving stack forces in between is skipped when matching.
                position = 0
                for token in tokens[prompt_length:].tolist():
                    if position < len(ids) and int(token) == ids[position]:
                        position += 1
                target = ids[min(position, len(ids) - 1)]
                bias = mx.where(mx.arange(logits.shape[-1]) == target, 1000.0, 0.0)
                return logits + bias.astype(logits.dtype)

            return (scripted,)

    return ScriptedMuse


def serve(monkeypatch, script, mode, *, tools=True, budget=BUDGET):
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    request = {
        "messages": [{"role": "user", "content": "x"}],
        "tokens": [(7 * i + 3) % 30 + 33 for i in range(24)],
        "enable_thinking": True, "max_tokens": 400, "temperature": 0,
        "thinking_budget": budget, "thinking_budget_mode": mode,
    }
    if tools:
        request["tools"] = TOOLS
    engine = make_engine(model, vocab, mtp=False, eos=(0,), adapter_mixin=scripted_muse(script))
    try:
        result = run(engine, request)
    finally:
        engine.close()
    assert "error" not in result, result["error"]
    return result, "".join(PIECES[t] for t in result["tokens"] if t)


@pytest.mark.parametrize("mode", ["state_aware", "history"])
@pytest.mark.parametrize("shape", sorted(SCRIPTS))
def test_budget_never_writes_the_answer_switch_after_reasoning_ended(monkeypatch, mode, shape):
    result, text = serve(monkeypatch, SCRIPTS[shape], mode, tools=shape != "direct_answer")
    assert RELEASE not in text, text
    assert text == SCRIPTS[shape], text
    if mode == "state_aware":
        guard = result["receipt"]["request_controls"]["thinking_guard"]
        assert guard["forced_close"] is False
        assert guard["think_tokens"] < BUDGET


@pytest.mark.parametrize("mode", ["state_aware", "history"])
def test_run_on_reasoning_is_still_released_at_the_budget(monkeypatch, mode):
    script = " to=self<|message|>" + "0123456789" * 20 + "<|eom|><|start|>assistant to=f<|message|>" + CALL
    result, text = serve(monkeypatch, script, mode)
    assert text.index(RELEASE) == BUDGET, text
    if mode == "state_aware":
        guard = result["receipt"]["request_controls"]["thinking_guard"]
        assert guard["forced_close"] is True and guard["released_at"] == BUDGET


# Review r1: a budget that falls while the model writes a recipient header.
# Every header opens with " to=" (Muse's shared " to" token), and " to=user"
# also opens a tool named user_x; the budget forced its release into them.
SECOND = SCRIPTS["tool_call"].index(" to=f")
USER_X = SCRIPTS["tool_call"][:SECOND] + " to=user_x<|message|>" + CALL.replace('"f"', '"user_x"')
CHOICES = {
    "direct_call": (SCRIPTS["direct_call"], 1),
    "direct_answer": (SCRIPTS["direct_answer"], 1),
    "tool_after_reasoning": (SCRIPTS["tool_call"], SECOND + len(" t")),
    "user_prefixed_tool": (USER_X, SECOND + len(" to=user")),
}


@pytest.mark.parametrize("mode", ["state_aware", "history"])
@pytest.mark.parametrize("shape", sorted(CHOICES))
def test_budget_inside_a_recipient_header_leaves_the_choice_to_the_model(monkeypatch, mode, shape):
    script, budget = CHOICES[shape]
    result, text = serve(monkeypatch, script, mode, tools=shape != "direct_answer", budget=budget)
    assert text == script, text
    if mode == "state_aware":
        assert result["receipt"]["request_controls"]["thinking_guard"]["forced_close"] is False


@pytest.mark.parametrize("mode", ["state_aware", "history"])
def test_budget_1_forces_the_release_once_the_reasoning_header_is_complete(monkeypatch, mode):
    own = " to=self<|message|>"
    result, text = serve(monkeypatch, own + "0123456789" * 3, mode, budget=1)
    assert text == own + RELEASE + "0123456789" * 3, text
    if mode == "state_aware":
        guard = result["receipt"]["request_controls"]["thinking_guard"]
        assert guard["forced_close"] is True and guard["released_at"] == len(own)


def templated_muse(script):
    class TemplatedMuse(scripted_muse(script)):
        """Muse's own prompt path (request admission) over the harness vocabulary."""

        prompt_tokens = MuseGlimmerAdapter.prompt_tokens

        def __init__(self, path):
            super().__init__(path)

            class Tokenizer(type(self.tokenizer)):
                def apply_chat_template(self, messages, **_kw):
                    return "<|start|>assistant"

            self.tokenizer = Tokenizer()

    return TemplatedMuse


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("name", ["self", "user"])
def test_a_tool_named_like_a_protocol_recipient_is_refused(monkeypatch, name, thinking):
    # Review r1: " to=self" always reads as reasoning (the budget counted the
    # call's ATEM body and forced the answer switch into it) and " to=user"
    # as the answer; no tool can be addressed by either name.
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, eos=(0,),
                         adapter_mixin=templated_muse(" to=user<|message|>ok"))
    request = {
        "messages": [{"role": "user", "content": "x"}], "max_tokens": 8,
        "temperature": 0, "enable_thinking": thinking, "thinking_budget": 4 if thinking else 0,
        "tools": [*TOOLS, {"type": "function", "function": {"name": name}}],
    }
    try:
        refused = run(engine, request)
        admitted = run(engine, {**request, "tools": [*TOOLS, {"type": "function",
                                                               "function": {"name": name + "_x"}}]})
    finally:
        engine.close()
    assert refused.get("status") == 400, refused
    assert f"'{name}'" in refused["error"] and "reserved" in refused["error"]
    assert "error" not in admitted, admitted


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("name", ["self", "user"])
def test_a_reserved_tool_name_is_inert_under_tool_choice_none(monkeypatch, name, thinking):
    # Review r2: tool_choice "none" keeps every tool out of the template, so a
    # stable registry holding a tool by a reserved name offers nothing the
    # model could address; only an offered one is refused.
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, eos=(0,),
                         adapter_mixin=templated_muse(" to=user<|message|>ok"))
    request = {
        "messages": [{"role": "user", "content": "x"}], "max_tokens": 8,
        "temperature": 0, "enable_thinking": thinking, "thinking_budget": 4 if thinking else 0,
        "tools": [*TOOLS, {"type": "function", "function": {"name": name}}],
    }
    try:
        served = run(engine, {**request, "tool_choice": "none"})
        offered = run(engine, {**request, "tool_choice": "auto"})
    finally:
        engine.close()
    assert "error" not in served, served
    assert served["tokens"]
    assert offered.get("status") == 400, offered
    assert f"'{name}'" in offered["error"] and "reserved" in offered["error"]
