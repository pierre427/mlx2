"""Muse bounds reasoning with a forced switch to the user's answer.

qualify-1007-extra (Muse-Glimmer-30B-mlx-8bit, every route): a request with a
``thinking_budget`` failed 400 "thinking_budget needs an adapter-declared
thinking-close token" while ``/v1/status`` reported ``thinking_deferral``.
Muse defers a grammar to its answer header (``structured_answer_token_ids``),
not to a thinking-close marker, and the budget path only read the latter.
"""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
from mlx2.serving import thinking_close_token_ids, thinking_release_token_ids

SWITCH = "<|eom|><|start|>assistant to=user<|message|>"
# <|eom|>, <|start|>, "assistant", " to", "=user", <|message|> in the Muse vocabulary
SWITCH_IDS = (200007, 200022, 140680, 328, 76976, 200023)


def _adapter(encode, decode):
    adapter = NS(tokenizer=NS(encode=encode, decode=decode))
    adapter.thinking_release_token_ids = (
        lambda: MuseGlimmerAdapter.thinking_release_token_ids(adapter)
    )
    return adapter


def test_muse_declares_the_switch_to_the_users_answer_as_its_release():
    adapter = _adapter(
        lambda text, **_kw: list(SWITCH_IDS) if text == SWITCH else [],
        lambda ids, **_kw: SWITCH if tuple(ids) == SWITCH_IDS else "",
    )
    assert thinking_release_token_ids(adapter) == SWITCH_IDS
    # The grammar deferral contract is unchanged: Muse waits for its answer
    # header, and a tool call (which never writes it) keeps its own grammar.
    assert thinking_close_token_ids(adapter) is None
    assert not hasattr(MuseGlimmerAdapter, "thinking_close_token_ids")


def test_a_switch_that_does_not_round_trip_is_not_declared():
    adapter = _adapter(lambda text, **_kw: [1, 2], lambda ids, **_kw: "other")
    assert thinking_release_token_ids(adapter) is None


@pytest.mark.parametrize("name", ["Muse-Glimmer-30B-mlx-8bit", "Muse-Glimmer-30B-mlx-4bit"])
def test_switch_ids_match_the_real_tokenizer(name):
    path = Path.home() / "mlx-models" / name / "tokenizer.json"
    if not path.is_file():
        pytest.skip(f"{name} tokenizer not present")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))

    def encode(text, **_kw):
        return tokenizer.encode(text, add_special_tokens=False).ids

    def decode(ids, **_kw):
        return tokenizer.decode(list(ids), skip_special_tokens=False)

    assert thinking_release_token_ids(_adapter(encode, decode)) == SWITCH_IDS


def test_thinking_budget_request_is_accepted_and_forces_the_release(monkeypatch):
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    marker = (97, 98, 99)

    class MuseShaped:
        # Muse's contract shape: thinking on (by default here; the tiny route
        # declares no REASONING capability for an explicit enable_thinking), an answer header for grammars,
        # a release switch for the budget, and no thinking-close marker.
        def thinking_enabled(self, request):
            return "messages" in request and request.get("enable_thinking", True)

        def structured_answer_token_ids(self, request):
            return (60, 61)

        def thinking_release_token_ids(self):
            return marker

    request = {"messages": [{"role": "user", "content": "x"}],
               "tokens": [(7 * i + 3) % (vocab - 2) + 1 for i in range(40)],
               "max_tokens": 12, "temperature": 0, "thinking_budget": 1}
    engine = make_engine(model, vocab, mtp=False, adapter_mixin=MuseShaped)
    try:
        assert engine.status()["structured_output"]["thinking_deferral"] is True
        result = run(engine, request)
        history = run(engine, {**request, "thinking_budget_mode": "history"})
    finally:
        engine.close()
    assert "error" not in result, result.get("error")
    guard = result["receipt"]["request_controls"]["thinking_guard"]
    assert guard["budget"] == 1 and guard["forced_close"] is True
    assert tuple(result["tokens"][1:4]) == marker
    assert "error" not in history, history.get("error")
    assert tuple(history["tokens"][1:4]) == marker
