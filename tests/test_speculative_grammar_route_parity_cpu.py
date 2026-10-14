"""Every speculative route keeps a grammar constraint exactly (CPU, tiny models).

Peer failures this guards against (2026-10-14 survey):

* Rapid-MLX #4615: MTP desynchronized the grammar matcher and silently
  dropped the constraint; the fix routed grammar requests to ordinary decode.
* sglang #43518: a stale grammar mask was applied to DFlash/DSpark verify rows.
* vLLM #61557: grammar-invalid drafts were dropped before rejection sampling,
  biasing the sampled distribution.

mlx2 does not refuse: every speculative route runs the request's processors
on each verify row with that row's own history (history-pure grammar state),
draws processor-aware drafts from the law it hands the verifier, and rewinds
the bookkeeping of rows past the accept point.  These tests hold each route to
the ordinary-decode reference: under greedy the tokens must be identical (with
the thinking-close marker landing inside draft blocks, and with two lanes in
one batch); under sampling every answer must satisfy the grammar.  The last
test plants the Rapid-MLX defect class (verify rows without processors) and
shows the parity check catches it.
"""

import copy
import re

import pytest

from route_harness import (
    PIECES,
    collect,
    make_engine,
    make_external_engine,
    patch_host,
    tiny_muse_dflash,
    tiny_qwen38_mtp,
)

EOS, CLOSE = 0, 127
GRAMMAR = r"[a-c]{2}(xy|zq)[0-9]{3}"
A, B, C = PIECES.index("a"), PIECES.index("b"), PIECES.index("c")
# Repeated n-grams give prompt lookup something to copy.
PROMPTS = (
    [40, 41, 42, A, B, C, 50, 51, 40, 41, 42, A, B, C, 52, 53, 40, 41, 42],
    [60, 61, 62, A, A, C, 70, 71, 60, 61, 62, A, A, C, 72, 73, 60, 61],
)

SELF_MTP_ROUTES = {
    "native_mtp": dict(num_draft=3),
    "copy_draft": dict(
        num_draft=3, execution_policy={"self_mtp_copy_draft": {"enabled": True}}
    ),
    "ordinary_handoff": dict(
        num_draft=2,
        execution_policy={"mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 1}},
    ),
    "fly_verification": dict(
        num_draft=3, execution_policy={"fly_verification": {"enabled": True, "window": 1}}
    ),
    "draft_loop": dict(
        num_draft=5, extra={"draft_loop": {"boundaries": [2, 5], "threshold": -1e9}}
    ),
    "adaptive_depth": dict(num_draft=3, adaptive_mtp_depth={}),
}
ROUTES = (*SELF_MTP_ROUTES, "prompt_lookup", "external_dflash2")


def _wide_selector_dflash2():
    """The harness Muse/DFlash2 pair with a full-vocabulary selector.

    The harness drafter keeps 4 of 128 selector candidates; a random one
    almost never ranks a grammar-legal token there, so constrained rounds
    would propose nothing and parity would hold vacuously.
    """
    import mlx.core as mx

    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel

    muse, small, vocab = tiny_muse_dflash()
    config = copy.deepcopy(small.config)
    config.selector_top_k = vocab
    mx.random.seed(9)
    draft = DFlash2DraftModel(config).bind(muse)
    mx.eval(draft.parameters())
    return muse, draft, vocab


@pytest.fixture(scope="module")
def models():
    target, vocab = tiny_qwen38_mtp()
    return {"qwen": (target, vocab), "muse": _wide_selector_dflash2()}


def _engine(models, route, **kw):
    if route == "external_dflash2":
        muse, draft, vocab = models["muse"]
        return make_external_engine(muse, draft, vocab, num_draft=3, eos=(EOS,), **kw)
    target, vocab = models["qwen"]
    if route == "ordinary":
        return make_engine(target, vocab, mtp=False, eos=(EOS,), **kw)
    if route == "muse_ordinary":
        muse, _, vocab = models["muse"]
        return make_engine(muse, vocab, mtp=False, eos=(EOS,), **kw)
    if route == "prompt_lookup":
        return make_engine(
            target, vocab, mtp=False, prompt_lookup=True, num_draft=4, eos=(EOS,), **kw
        )
    return make_engine(target, vocab, eos=(EOS,), **SELF_MTP_ROUTES[route], **kw)


def _requests(*, thinking=False, temperature=0.0, seed=None):
    requests = []
    for index, prompt in enumerate(PROMPTS):
        request = {
            "tokens": prompt, "max_tokens": 14, "temperature": temperature,
            "grammar": GRAMMAR,
        }
        if thinking:
            request["messages"] = [{"role": "user", "content": "x"}]
            request["thinking_budget"] = 6
        if seed is not None:
            request["seed"] = seed + index
        requests.append(request)
    return requests


def _serve(models, route, requests, *, thinking=False):
    engine = _engine(models, route, max_lanes=2, close_id=CLOSE if thinking else None)
    try:
        jobs = [engine.submit(dict(request)) for request in requests]
        results = [collect(job) for job in jobs]
        assert engine.thread.is_alive() and engine.error is None, route
        return results
    finally:
        engine.close()


def _answer(tokens, thinking):
    if thinking:
        assert CLOSE in tokens, tokens
        tokens = tokens[tokens.index(CLOSE) + 1:]
    return "".join(PIECES[token] for token in tokens)


def _reference(models, route, requests, thinking):
    base = "muse_ordinary" if route == "external_dflash2" else "ordinary"
    return _serve(models, base, requests, thinking=thinking)


def _engaged(route, results):
    """Whether speculation actually ran (a parity pass must not be vacuous)."""
    if route in SELF_MTP_ROUTES:
        return sum(
            ((result["receipt"].get("mtp") or {}).get("stats") or {}).get(
                "draft_proposed", 0
            )
            for result in results
        ) > 0
    key = "speculation"
    return sum(
        (result["receipt"].get(key) or {}).get("proposed", 0) for result in results
    ) > 0


@pytest.mark.parametrize("thinking", [False, True], ids=["answer", "thinking"])
@pytest.mark.parametrize("route", ROUTES)
def test_greedy_grammar_output_equals_ordinary_decode(monkeypatch, models, route, thinking):
    patch_host(monkeypatch)
    requests = _requests(thinking=thinking)
    reference = _reference(models, route, requests, thinking)
    results = _serve(models, route, requests, thinking=thinking)
    for want, got in zip(reference, results):
        assert want.get("error") is None and got.get("error") is None, (route, got)
        assert got["tokens"] == want["tokens"], route
        assert re.fullmatch(GRAMMAR, _answer(got["tokens"], thinking)), route
    if route == "ordinary_handoff":
        return  # two live lanes hand off to ordinary decode by design
    assert _engaged(route, results), route


@pytest.mark.parametrize("route", ROUTES)
def test_sampled_grammar_output_always_satisfies_the_grammar(monkeypatch, models, route):
    patch_host(monkeypatch)
    for seed in (11, 23, 37):
        results = _serve(models, route, _requests(temperature=1.0, seed=seed))
        for result in results:
            assert result.get("error") is None, (route, seed, result.get("error"))
            text = _answer(result["tokens"], False)
            assert re.fullmatch(GRAMMAR, text), (route, seed, text)


def test_parity_check_has_teeth_against_unconstrained_verify_rows(monkeypatch, models):
    """Plant the Rapid-MLX #4615 defect: self-MTP verify rows skip processors."""
    from mlx2.runtime import hybrid_speculative

    patch_host(monkeypatch)
    requests = _requests()
    reference = _serve(models, "ordinary", requests)
    original = hybrid_speculative._apply_logits_processors

    def drop_constraint(processors, y, logits):
        kept = [p for p in processors if type(p).__name__ != "StructuredOutputProcessor"]
        return original(kept, y, logits)

    monkeypatch.setattr(hybrid_speculative, "_apply_logits_processors", drop_constraint)
    planted = _serve(models, "native_mtp", requests)
    assert any(
        got.get("error") is not None or got["tokens"] != want["tokens"]
        for want, got in zip(reference, planted)
    )


# -- tree15 (bounded best-first tree verify), executor level ------------------

TREE_GRAMMAR = r"[0-3]{2}(12|45)[5-9]{3}"


def _tree_case(monkeypatch):
    from route_harness import make_adapter
    from test_dflash_pair_select import generator, tiny

    for name in ("MLX2_DFLASH_TOPOLOGY", "MLX2_QWEN_TARGET_EXECUTION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    # Every id the grammar spells ('0'..'9' = 48..57) fits a 64-token vocab.
    target, draft = tiny(vocab=64, top_k=64, block_size=8)
    tokenizer = make_adapter(None, 64, eos=(EOS,)).tokenizer
    return target, draft, tokenizer, generator


def _grammar(tokenizer, prompt_length, greedy):
    from mlx2.structured_output import make_structured_processor

    return make_structured_processor(
        tokenizer, prompt_length, grammar=TREE_GRAMMAR, greedy=greedy,
        generation_stop_token_ids=[EOS],
    )


def _ordinary_greedy(target, tokenizer, prompt, max_tokens):
    import mlx.core as mx

    processor = _grammar(tokenizer, len(prompt), True)
    tokens = list(prompt)
    for _ in range(max_tokens):
        logits = target(mx.array([tokens]))[:, -1].astype(mx.float32)
        logits = processor(mx.array(tokens), logits)
        token = int(mx.argmax(logits[0]).item())
        tokens.append(token)
        if token == EOS:
            break
    return tokens[len(prompt):]


def _tree_run(target, draft, tokenizer, generator, prompt, *, temp, seed=123, max_tokens=12):
    import mlx.core as mx

    batch = generator(target, draft, stop_tokens=[[EOS]])
    batch.insert(
        [list(prompt)], max_tokens=[max_tokens],
        logits_processors=[[_grammar(tokenizer, len(prompt), temp == 0)]],
        sampling_configs=[{"sampling_temp": temp}],
        lane_rngs=[type("Seed", (), {"key": mx.array([seed])})()],
    )
    tokens = []
    for _ in range(200):
        _, responses = batch.next()
        tokens.extend(response.token for response in responses)
        if not batch.lanes:
            return tokens, dict(batch.scheduler_stats)
    raise AssertionError("tree15 scheduler stalled")


def _text(tokens):
    return "".join(PIECES[token] for token in tokens if token != EOS)


def test_tree15_greedy_grammar_output_equals_ordinary_decode(monkeypatch):
    target, draft, tokenizer, generator = _tree_case(monkeypatch)
    for prompt in ([1, 2, 3], [9, 4, 8, 4]):
        tokens, stats = _tree_run(target, draft, tokenizer, generator, prompt, temp=0.0)
        assert stats["external_tree_rounds"] > 0 and stats["external_tree_nodes"] > 0
        assert tokens == _ordinary_greedy(target, tokenizer, prompt, 12)
        assert re.fullmatch(TREE_GRAMMAR, _text(tokens)), _text(tokens)


def test_tree15_sampled_grammar_output_satisfies_the_grammar(monkeypatch):
    target, draft, tokenizer, generator = _tree_case(monkeypatch)
    accepted = 0
    for seed in range(6):
        tokens, stats = _tree_run(
            target, draft, tokenizer, generator, [1, 2, 3], temp=1.0, seed=seed
        )
        accepted += stats["external_tree_accepted_edges"]
        assert re.fullmatch(TREE_GRAMMAR, _text(tokens)), (seed, _text(tokens))
    assert accepted > 0
