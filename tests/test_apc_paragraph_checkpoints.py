"""APCv2 paragraph checkpoint: the first new question on a cached document.

A hybrid prompt cache resumes only where an exact recurrent state was
captured.  A junction checkpoint learns the branch point of a document from a
*second* divergence, so the first different question asked about a cached
document re-prefills from the last rolling or interior state.  With
``apc_paragraph_checkpoints`` admission plans one exact boundary after the
last paragraph break before the final user question; a second request that
asks a different question after the same break resumes there on its first
try.

The end-to-end tests drive the real ``ServingEngine`` on CPU with the tiny
random-weight Qwen3.8 hybrid (GDN + full attention) used by the junction
tests, behind a word-level toy tokenizer and chat template.
"""

import re

import pytest

from mlx2 import qualification, serving
from mlx2.prometheus import _ENGINE_EVENTS, _OPTIONAL_ENGINE_EVENTS
from mlx2.runtime.interior_placement import (
    PARAGRAPH_MIN_CONTENT_TOKENS,
    paragraph_checkpoint_position,
)

from test_06_junction_snapshots import _adapter as _junction_adapter
from test_06_junction_snapshots import host, qwen38  # noqa: F401 - fixtures
from test_apc_interior_route_selection import _UnsupportedInteriorAdapter

# --------------------------------------------------------------------------
# Toy tokenizer and chat template
# --------------------------------------------------------------------------

IM_START, IM_END, USER, ASSISTANT, NL, NL2 = 1, 2, 3, 4, 5, 6
WORD0, WORDS = 10, 100  # ids 10..109 are words w0..w99
MERGED0 = 110  # ids 110..119 are "w{i}\n\n" (break merged into the word)


class ToyTokenizer:
    """Word-level tokenizer; ``mode`` picks how a paragraph break encodes.

    ``one``    "\\n\\n" is one token (NL2).
    ``two``    every "\\n" is its own token, so "\\n\\n" is NL NL.
    ``merged`` a word directly followed by "\\n\\n" is one token (".\\n\\n"
               style, as byte-level BPEs with punctuation+newline merges).
    ``prefix`` like ``one``, but a text's first word gets a distinct
               "dummy prefix" id when encoded alone (SentencePiece style).
    """

    vocab_size = 128
    eos_token_ids = []

    def __init__(self, mode="one"):
        self.mode = mode

    @property
    def detokenizer(self):
        from test_apc_hits_hybrid_gdn_self_mtp import Detok

        return Detok()

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in re.findall(r"\n+|[^\s]+", text):
            if piece.startswith("\n"):
                if self.mode == "two":
                    ids.extend([NL] * len(piece))
                elif (
                    self.mode == "merged"
                    and len(piece) == 2
                    and ids
                    and WORD0 <= ids[-1] < WORD0 + 10
                ):
                    ids[-1] = MERGED0 + ids[-1] - WORD0
                else:
                    ids.extend([NL2] * (len(piece) // 2) + [NL] * (len(piece) % 2))
            else:
                ids.append(WORD0 + int(piece[1:]) % WORDS)
        if self.mode == "prefix" and ids and WORD0 <= ids[0] < WORD0 + WORDS:
            ids[0] = 120  # the standalone "▁w" form never seen in context
        return ids

    def decode(self, ids):
        out = ""
        for token in ids:
            token = int(token)
            if token == NL:
                out += "\n"
            elif token == NL2:
                out += "\n\n"
            elif MERGED0 <= token < MERGED0 + 10:
                out += (" " if out and not out.endswith("\n") else "") + (
                    f"w{token - MERGED0}\n\n"
                )
            elif WORD0 <= token < WORD0 + WORDS:
                out += (" " if out and not out.endswith("\n") else "") + (
                    f"w{token - WORD0}"
                )
            else:
                out += "<s>"
        return out


def render(tokenizer, messages):
    """ChatML-shaped toy template: 3 head tokens, 5 tail tokens."""
    ids = []
    for message in messages:
        role = USER if message["role"] == "user" else ASSISTANT
        content = tokenizer.encode(message["content"])
        if tokenizer.mode == "prefix" and content and content[0] == 120:
            # In context the first word follows a newline: no dummy prefix.
            content[0] = WORD0 + int(message["content"].split()[0][1:]) % WORDS
        ids += [IM_START, role, NL] + content + [IM_END, NL]
    return ids + [IM_START, ASSISTANT, NL]


def document(words=300, seed=3, last_word=None):
    body = [f"w{(seed * i + 7) % WORDS}" for i in range(words)]
    if last_word is not None:
        body[-1] = last_word
    return " ".join(body[: words // 2]) + "\n" + " ".join(body[words // 2 :])


def question(seed, words=24):
    return " ".join(f"w{(seed * (i + 1) + 11) % WORDS}" for i in range(words))


def messages(doc, q):
    return [{"role": "user", "content": doc + "\n\n" + q}]


# --------------------------------------------------------------------------
# Selection rule (pure)
# --------------------------------------------------------------------------


def _plan(mode, doc, q, **kwargs):
    tokenizer = ToyTokenizer(mode)
    request = {"messages": messages(doc, q)}
    tokens = render(tokenizer, request["messages"])
    return tokens, paragraph_checkpoint_position(tokens, request, tokenizer, **kwargs)


@pytest.mark.parametrize("mode,break_tokens", [("one", 1), ("two", 2)])
def test_position_follows_the_break_for_one_and_two_token_breaks(mode, break_tokens):
    doc = document()
    tokens, (position, reason) = _plan(mode, doc, question(5))
    doc_tokens = len(ToyTokenizer(mode).encode(doc))
    assert reason == "planned"
    assert position == 3 + doc_tokens + break_tokens
    assert tokens[position - break_tokens : position] == (
        [NL2] if mode == "one" else [NL, NL]
    )
    # The question starts exactly at the boundary.
    assert tokens[position : position + 24] == ToyTokenizer(mode).encode(question(5))


def test_break_merged_into_the_preceding_word_is_found():
    doc = document(last_word="w4")
    tokens, (position, reason) = _plan("merged", doc, question(5))
    assert reason == "planned"
    assert tokens[position - 1] == MERGED0 + 4
    assert ToyTokenizer().decode(tokens[position - 1 : position]).endswith("\n\n")


def test_dummy_prefix_on_the_standalone_question_is_tolerated():
    doc = document()
    tokens, (position, reason) = _plan("prefix", doc, question(5))
    assert reason == "planned"
    assert tokens[position - 1] == NL2


def test_last_break_is_chosen_and_trailing_breaks_are_ignored():
    tokenizer = ToyTokenizer("one")
    doc = document() + "\n\n" + document(words=20, seed=7)
    content = doc + "\n\n" + question(5) + "\n\n"
    request = {"messages": [{"role": "user", "content": content}]}
    tokens = render(tokenizer, request["messages"])
    position, reason = paragraph_checkpoint_position(tokens, request, tokenizer)
    assert reason == "planned"
    assert position == 3 + len(tokenizer.encode(doc)) + 1


@pytest.mark.parametrize(
    "name,content,kwargs,reason",
    [
        ("no break", document() + " " + question(5), {}, "no_break"),
        ("only a single newline", document() + "\n" + question(5), {}, "no_break"),
        ("short document", document(words=200) + "\n\n" + question(5), {},
         "short_document"),
        # 24-word question + 5 template tokens = 29 < 40 tokens before the end.
        ("break too close to the end", document() + "\n\n" + question(5),
         {"min_tail_tokens": 40}, "near_end"),
        ("break too close to the end (default)", document() + "\n\n" + question(5, 4),
         {}, "near_end"),
        ("inside a media span", document() + "\n\n" + question(5),
         {"floor_tokens": 400}, "inside_media"),
    ],
)
def test_plans_nothing(name, content, kwargs, reason):
    tokenizer = ToyTokenizer("one")
    request = {"messages": [{"role": "user", "content": content}]}
    tokens = render(tokenizer, request["messages"])
    assert paragraph_checkpoint_position(tokens, request, tokenizer, **kwargs) == (
        None, reason
    ), name


def test_plans_nothing_without_a_final_user_text_message():
    tokenizer = ToyTokenizer("one")
    tokens = render(tokenizer, messages(document(), question(5)))
    for request in (
        {"tokens": tokens},
        {"messages": messages(document(), question(5))
         + [{"role": "assistant", "content": "w1"}]},
        {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
    ):
        assert paragraph_checkpoint_position(tokens, request, tokenizer) == (
            None, "no_user_text"
        )


def test_plans_nothing_when_the_prompt_does_not_contain_the_question():
    # The adapter rewrote the content (or the template does not render it
    # verbatim): never guess a position.
    tokenizer = ToyTokenizer("one")
    request = {"messages": messages(document(), question(5))}
    tokens = render(tokenizer, messages(document(), question(9)))
    assert paragraph_checkpoint_position(tokens, request, tokenizer) == (
        None, "unmatched"
    )


def test_newline_absorbed_into_the_question_is_rejected():
    class Absorbing(ToyTokenizer):
        # "\n\nw5" encodes as NL, <"\nw5">: the break is not at a token edge.
        def encode(self, text, add_special_tokens=False):
            return [NL if t == NL2 else t for t in super().encode(text)]

    tokenizer = Absorbing("one")
    request = {"messages": messages(document(), question(5))}
    tokens = render(tokenizer, request["messages"])
    position, reason = paragraph_checkpoint_position(tokens, request, tokenizer)
    assert (position, reason) == (None, "unmatched")


# --------------------------------------------------------------------------
# End to end through ServingEngine
# --------------------------------------------------------------------------


def _adapter(model, vocab, mode):
    base = _junction_adapter(model, vocab)

    class Adapter(base):
        tokenizer = ToyTokenizer(mode)

        def prompt_tokens(self, request):
            if "tokens" in request:
                return list(request["tokens"])
            return render(self.tokenizer, request["messages"])

    return Adapter


def _engine(model, vocab, *, paragraph, mode="one", mtp=False):
    policy = {"apc_paragraph_checkpoints": True} if paragraph else None
    engine = serving.ServingEngine(
        "tiny", adapter_factory=_adapter(model, vocab, mode),
        qualification_mode=True, mtp=mtp, max_lanes=1, prefill_step=16,
        execution_policy=policy,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _ask(engine, doc, q, max_tokens=6):
    job = engine.submit(
        {"messages": messages(doc, q), "max_tokens": max_tokens, "temperature": 0}
    )
    text = ""
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], event.get("receipt") or {}, job


def _expected(mode, doc):
    return 3 + len(ToyTokenizer(mode).encode(doc)) + (1 if mode == "one" else 2)


@pytest.mark.parametrize(
    "mode,mtp",
    [("one", False), ("two", False), ("one", True)],
    ids=["one_token_break", "two_token_break", "one_token_break_self_mtp"],
)
def test_first_new_question_resumes_at_the_paragraph_break_exactly(
    host, qwen38, mode, mtp  # noqa: F811
):
    model, vocab = qwen38
    doc = document()
    boundary = _expected(mode, doc)
    assert boundary - 3 >= PARAGRAPH_MIN_CONTENT_TOKENS
    engine = _engine(model, vocab, paragraph=True, mode=mode, mtp=mtp)
    try:
        _out1, _r1, j1 = _ask(engine, doc, question(5))
        assert int(j1.cached_tokens or 0) == 0
        assert j1.apc_paragraph_position == boundary
        assert boundary in j1.prefill_cold_cuts
        out2, r2, j2 = _ask(engine, doc, question(9))
        counts = dict(engine.counts)
        settings = dict(engine.snapshot["settings"])
    finally:
        engine.close()
    # The first different question resumes at the break, not short of it.
    assert int(j2.cached_tokens) == boundary
    assert r2.get("cache_checkpoint_role") == "interior_checkpoint"
    assert j2.slice_aligned
    # Planned by the first request only: the second is already cached there.
    assert counts["apc_paragraph_checkpoints_planned"] == 1
    assert counts["apc_paragraph_checkpoints_hits"] == 1
    assert counts.get("apc_paragraph_checkpoints_degraded", 0) == 0
    assert settings["apc_paragraph_checkpoints"] is True

    # Exactness: resuming at the break equals a cold prefill of the same
    # prompt on the same policy (same cold cuts, APC-3) and with it off.
    for paragraph in (True, False):
        cold = _engine(model, vocab, paragraph=paragraph, mode=mode, mtp=mtp)
        try:
            out_cold, _r, jcold = _ask(cold, doc, question(9))
        finally:
            cold.close()
        assert int(jcold.cached_tokens or 0) == 0
        assert out2 == out_cold, paragraph


def test_default_off_plans_nothing_and_settings_are_unchanged(host, qwen38):  # noqa: F811
    model, vocab = qwen38
    doc = document()
    engine = _engine(model, vocab, paragraph=False)
    try:
        _ask(engine, doc, question(5))
        _out, _r, j2 = _ask(engine, doc, question(9))
        counts = dict(engine.counts)
        settings = dict(engine.snapshot["settings"])
    finally:
        engine.close()
    assert j2.apc_paragraph_position is None
    assert int(j2.cached_tokens or 0) < _expected("one", doc)
    assert "apc_paragraph_checkpoints" not in settings
    assert not any(key.startswith("apc_paragraph_") for key in counts)


def test_no_break_plans_nothing_and_is_counted(host, qwen38):  # noqa: F811
    model, vocab = qwen38
    engine = _engine(model, vocab, paragraph=True)
    try:
        job = engine.submit({
            "messages": [{"role": "user", "content": document() + " " + question(5)}],
            "max_tokens": 4, "temperature": 0,
        })
        while "finish_reason" not in job.events.get(timeout=120):
            pass
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert job.apc_paragraph_position is None
    assert counts["apc_paragraph_checkpoints_skipped_no_break"] == 1
    assert counts.get("apc_paragraph_checkpoints_planned", 0) == 0


def test_paragraph_degrades_when_its_snapshot_does_not_fit(host, qwen38, monkeypatch):  # noqa: F811
    model, vocab = qwen38
    doc = document()
    boundary = _expected("one", doc)
    base = _adapter(model, vocab, "one")
    inner_budget = base.cache_budget

    class Adapter(base):
        def cache_budget(self, *, mtp):
            budget = inner_budget(self, mtp=mtp)
            project = budget.project
            budget.project = lambda tokens: (
                1 << 60 if tokens == boundary else project(tokens)
            )
            return budget

    engine = serving.ServingEngine(
        "tiny", adapter_factory=Adapter, qualification_mode=True, mtp=False,
        max_lanes=1, prefill_step=16,
        execution_policy={"apc_paragraph_checkpoints": True},
    )
    assert engine.ready.wait(60), engine.error
    try:
        _ask(engine, doc, question(5))
        _out, _r, j2 = _ask(engine, doc, question(9))
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert int(j2.cached_tokens or 0) < boundary
    assert counts["apc_paragraph_checkpoints_degraded"] == 2
    assert counts.get("apc_paragraph_checkpoints_planned", 0) == 0


@pytest.mark.parametrize("value", [1, "yes", None, {"enabled": True}])
def test_policy_is_strictly_boolean(host, qwen38, value):  # noqa: F811
    model, vocab = qwen38
    with pytest.raises(ValueError, match="apc_paragraph_checkpoints"):
        serving.ServingEngine(
            "tiny", adapter_factory=_adapter(model, vocab, "one"),
            qualification_mode=True, mtp=False,
            execution_policy={"apc_paragraph_checkpoints": value},
        )


@pytest.mark.parametrize(
    ("prompt_lookup", "backend", "route"),
    [(True, None, "prompt lookup"), (False, "external_draft", "external draft")],
)
def test_fails_closed_on_routes_that_cannot_capture(
    monkeypatch, prompt_lookup, backend, route
):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})

    class Adapter(_UnsupportedInteriorAdapter):
        pass

    Adapter.backend = backend
    engine = serving.ServingEngine(
        "fixture", adapter_factory=Adapter, qualification_mode=True, mtp=False,
        prompt_lookup=prompt_lookup,
        execution_policy={"apc_paragraph_checkpoints": True},
    )
    try:
        engine.thread.join(5)
        assert not engine.ready.is_set()
        assert "APCv2 paragraph checkpoints" in (engine.error or "")
        assert route in engine.error
    finally:
        engine.close()


def test_counters_are_optional_and_feature_check_is_selected():
    from mlx2.adapters.registry import (
        ADAPTER_DEFAULT_POLICY_KEYS,
        STATE_CHECKPOINT_POLICY_KEYS,
    )

    for event in ("planned", "degraded", "hits", "skipped_no_break"):
        assert _OPTIONAL_ENGINE_EVENTS["apc_paragraph_checkpoints_" + event] == (
            "apcv2_paragraph", event
        )
        assert "apc_paragraph_checkpoints_" + event not in _ENGINE_EVENTS
    assert "apc_paragraph_checkpoints" in serving.SERVER_OWNED_EXECUTION_POLICY_KEYS
    assert "apc_paragraph_checkpoints" in ADAPTER_DEFAULT_POLICY_KEYS
    assert "apc_paragraph_checkpoints" in STATE_CHECKPOINT_POLICY_KEYS
    assert "feature_apc_paragraph_checkpoints" in qualification._route_feature_checks(
        {"apc_paragraph_checkpoints": True}
    )
    assert "feature_apc_paragraph_checkpoints" not in (
        qualification._route_feature_checks({})
    )
