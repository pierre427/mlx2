"""A grammar failure on a row past a client stop does not fail the request.

Prompt lookup processes every accepted verify row of a round before the
serving loop delivers the round's first token, and ordinary decode evaluates
the unused next-token row before it returns the current token.  Serving
checked the latched ``structured.failure`` before the output parser saw the
token being delivered, so a grammar dead end on a later row turned a normal
client-stop completion into a 502: on prompt lookup for a dead end any number
of rows past the stop token, on every route for the row right after it.

A token drawn from a masked row is grammatical whatever a later row latched;
only a token at or past the failure row fails the request closed.
"""

import pytest

from route_harness import PIECES, make_engine, patch_host, run, tiny_qwen38_mtp

A, B, C = PIECES.index("a"), PIECES.index("b"), PIECES.index("c")
P, Q, R = 40, 41, 42
# "p q r a b c" repeats, so the prompt-lookup suffix match proposes "a b c ...".
PROMPT = [P, Q, R, A, B, C, 50, 51, P, Q, R, A, B, C, 52, 53, P, Q, R]
ROUTES = (
    ("ordinary", {"mtp": False}),
    ("prompt_lookup", {"mtp": False, "prompt_lookup": True, "num_draft": 4}),
)


class StopParser:
    """The real client-stop handling over the harness's id-text detokenizer."""

    def output_parser(self, request):
        from mlx2.output import OutputParser

        return OutputParser(stops=request.get("stop", ()))


def _serve(monkeypatch, request):
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    results = {}
    for route, kwargs in ROUTES:
        engine = make_engine(model, vocab, adapter_mixin=StopParser, **kwargs)
        try:
            results[route] = run(engine, request)
            assert engine.thread.is_alive() and engine.error is None, route
        finally:
            engine.close()
    return results


# The harness vocabulary cannot spell "é", so the automaton dead-ends on the
# row after the ASCII prefix: generated row 1 for "aé", row 2 for "abé".
@pytest.mark.parametrize("grammar", ["aé", "abé", "abcé"])
def test_grammar_dead_end_past_a_client_stop_keeps_the_stop_completion(
    monkeypatch, grammar
):
    request = {
        "tokens": PROMPT, "max_tokens": 8, "temperature": 0,
        "grammar": grammar, "stop": [f"{A} "],
    }
    results = _serve(monkeypatch, request)
    for route, _ in ROUTES:
        result = results[route]
        assert result.get("finish") == "stop", (
            route, result.get("status"), result.get("error"),
        )
        assert result["tokens"] == [], route  # the stop string is withheld


@pytest.mark.parametrize(
    "grammar, stop, delivered",
    [
        # No client stop: the tokens before the dead end stream, then the
        # first token drawn from the unmasked row fails the request closed.
        ("abé", None, [A, B]),
        # A dead end on the first row taints the first token itself, even
        # when that token would complete the client stop.
        ("é", f"{A} ", []),
    ],
)
def test_a_token_at_or_past_the_failure_row_still_fails_closed(
    monkeypatch, grammar, stop, delivered
):
    request = {"tokens": PROMPT, "max_tokens": 8, "temperature": 0, "grammar": grammar}
    if stop is not None:
        request["stop"] = [stop]
    results = _serve(monkeypatch, request)
    for route, _ in ROUTES:
        result = results[route]
        assert result.get("status") == 502, (route, result)
        assert "structured output failed closed" in result["error"], route
        assert result["tokens"] == delivered, route
