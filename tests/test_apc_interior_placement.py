"""Interior checkpoint placement (turn boundaries, nested tail, auto) and the
serving mechanism: a preamble shared across sessions is reused exactly from an
interior hybrid checkpoint.

CPU only.  The engine tests drive the real ServingEngine with tiny hybrid GDN
models (Qwen4-exp with MTP and a Qwen3.8-style hybrid) and assert the
mechanism ran: interior hits > 0 at the turn boundary, greedy output equals a
cold engine.
"""

import time
from pathlib import Path

import pytest

from mlx2 import serving
from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.generate import interior_checkpoint_positions
from mlx2.runtime.interior_placement import (
    detect_turn_marker_ids,
    plan_interior_positions,
    tail_positions,
    turn_boundary_positions,
)
from mlx2.serving import APC_INTERIOR_AUTO_POLICY, apc_interior_checkpoint_policy

from test_apc_hits_hybrid_gdn_self_mtp import (  # noqa: F401 - fixture
    host,
    make_adapter,
    run,
    tiny_qwen38_mtp,
    tiny_qwen4_mtp,
)

M = 999


def _conversation(preamble, *turns):
    tokens = list(range(1, preamble + 1))
    for body in turns:
        tokens += [M] + list(body)
    return tokens


# ----------------------------------------------------------------- placement


def test_pow2_placement_is_byte_identical_to_legacy_lattice():
    for length in (3, 9, 40, 257, 1000, 70000):
        tokens = list(range(length))
        for count, stride in ((1, 1), (2, 64), (3, 4), (32, 16)):
            for cached in (0, 5, 40, 300):
                legacy = tuple(
                    p
                    for p in interior_checkpoint_positions(
                        length, count=count, min_stride=stride
                    )
                    if p > cached
                )
                planned, sources = plan_interior_positions(
                    tokens, count=count, min_stride=stride, cached_tokens=cached
                )
                assert planned == legacy
                assert sources["lattice"] == len(legacy)


def test_turn_boundaries_are_marker_positions_inside_the_prompt():
    tokens = [M, 1, 2, M, 3, 4, M, 5, M]
    # Position 0 (nothing to cache) and P-1 (the generation boundary) excluded.
    assert turn_boundary_positions(tokens, (M,)) == (3, 6)
    assert turn_boundary_positions(tokens, ()) == ()


def test_tail_lattice_is_nested_across_prompts_sharing_a_prefix():
    a = tail_positions(16282, min_stride=256)
    assert a == (16128, 15360, 12288)
    # A second prompt sharing the first 16,200 tokens computes points on the
    # same absolute lattice, so its shallow points coincide with the first's.
    b = tail_positions(16290, min_stride=256)
    assert set(b) & set(a) == {16128, 15360, 12288}
    for length in (300, 5000, 123457):
        for point in tail_positions(length, min_stride=64):
            assert 0 < point < length - 1


def test_auto_prefers_preamble_then_branch_point_then_spaced_tail():
    tokens = _conversation(2000, range(10, 400), range(10, 3000), range(10, 50))
    turns = turn_boundary_positions(tokens, (M,))
    positions, sources = plan_interior_positions(
        tokens, count=4, min_stride=256, placement="auto", marker_ids=(M,)
    )
    assert turns[0] in positions  # end of the shared preamble
    assert turns[-2] in positions  # start of the final user turn (branch point)
    assert sources["turn"] >= 2 and sources["tail"] >= 1
    assert len(positions) == 4
    ordered = sorted(positions)
    spaced = [p for p in ordered if p not in (turns[0], turns[-2])]
    for point in spaced:
        assert all(abs(point - other) >= 256 for other in ordered if other != point)


def test_auto_without_marker_degrades_to_tail_and_respects_cache_and_media():
    tokens = list(range(5000))
    positions, sources = plan_interior_positions(
        tokens, count=4, min_stride=256, placement="auto", marker_ids=()
    )
    assert sources["turn"] == 0 and sources["tail"] == len(positions) > 0
    floor, _ = plan_interior_positions(
        tokens, count=4, min_stride=256, placement="auto", cached_tokens=4500
    )
    assert all(p > 4500 for p in floor)
    media, _ = plan_interior_positions(
        tokens, count=4, min_stride=256, placement="tail", floor_tokens=4800
    )
    assert all(p > 4800 for p in media)


def test_turns_placement_keeps_deepest_boundaries():
    tokens = _conversation(10, *[range(20, 30)] * 6)
    positions, sources = plan_interior_positions(
        tokens, count=2, min_stride=1, placement="turns", marker_ids=(M,)
    )
    assert positions == turn_boundary_positions(tokens, (M,))[-2:]
    assert sources == {"turn": 2, "tail": 0, "lattice": 0}


# -------------------------------------------------------------------- policy


def test_policy_auto_preset_and_legacy_identity():
    assert apc_interior_checkpoint_policy(None) == {"count": 0, "min_stride": 1}
    # Legacy settings keep their exact receipt identity.
    assert apc_interior_checkpoint_policy({"count": 2, "min_stride": 64}) == {
        "count": 2,
        "min_stride": 64,
    }
    assert apc_interior_checkpoint_policy(
        {"count": 2, "min_stride": 64, "placement": "pow2", "headroom_fraction": 1.0}
    ) == {"count": 2, "min_stride": 64}
    assert apc_interior_checkpoint_policy("auto") == APC_INTERIOR_AUTO_POLICY
    # "auto" is pinned to the arm that carried the 2026-09-19/20 GPU
    # qualification (interior-ckpt-20260919 flashnext-all-gated /
    # qwen38-27b-shared-rag).  headroom_fraction 0.25 starved RAG and
    # min_uncached_fraction 0 cost +3.9%/+7.2% on the linear control, so a
    # silent drift back to either would void the qualified numbers.
    assert APC_INTERIOR_AUTO_POLICY == {
        "count": 4,
        "min_stride": 256,
        "placement": "auto",
        "headroom_fraction": 0.5,
        "min_uncached_fraction": 0.5,
    }
    # ...and it is still only reachable when an execution policy asks for it.
    assert apc_interior_checkpoint_policy(None)["count"] == 0
    for invalid in (
        "on",
        {"placement": "fibonacci"},
        {"headroom_fraction": 0},
        {"headroom_fraction": 1.5},
        {"headroom_fraction": True},
        {"headroom_fraction": float("nan")},
    ):
        with pytest.raises(ValueError):
            apc_interior_checkpoint_policy(invalid)


# ------------------------------------------------------------ marker probing


class _FakeTemplateTokenizer:
    all_special_ids = [100, 101]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        out = []
        for message in messages:
            out += [100, 7, 8, 101, 9]
        if add_generation_prompt:
            out += [100, 5, 6]
        return {"input_ids": out}


def test_detect_turn_marker_from_generation_prompt_suffix():
    assert detect_turn_marker_ids(_FakeTemplateTokenizer()) == (100,)

    class NotSpecial(_FakeTemplateTokenizer):
        all_special_ids = [101]

    assert detect_turn_marker_ids(NotSpecial()) == ()
    assert detect_turn_marker_ids(object()) == ()


@pytest.mark.parametrize(
    "path",
    [
        "~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP",
        "~/mlx-models/Qwen3.8-27B-oQ4e-mtp",
    ],
)
def test_detect_turn_marker_on_real_qwen_template(path):
    if not Path(path, "tokenizer_config.json").exists():
        pytest.skip("tokenizer not present")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    markers = detect_turn_marker_ids(tokenizer)
    assert [tokenizer.convert_ids_to_tokens(m) for m in markers] == ["<|im_start|>"]


# ------------------------------------------------------------------ eviction


def test_reused_interior_checkpoint_is_no_longer_evicted_first():
    from test_apcv2_lifecycle import KVCache, _state

    apc = APCv2(max_size=4, max_interior_entries=2, layout_name="interior-promotion-v1")
    key = APCKey("promotion")
    apc.store(key, [3], [_state(KVCache(), 1)], retention_role="interior_checkpoint")
    hit = apc.lookup(key, [3, 4])
    assert hit.hit and hit.retention_role == "interior_checkpoint"
    hit.cache.close()
    assert apc.apc_stats["interior"]["lifetime_reused_entries"] == 1
    assert apc.apc_stats["interior"]["reused_entries"] == 1
    assert apc.apc_stats["interior"]["entries"] == 1
    apc.store(key, [5], [_state(KVCache(), 1)], retention_role="interior_checkpoint")
    apc.store(key, [6], [_state(KVCache(), 1)], retention_role="interior_checkpoint")
    # Unpromoted, the oldest interior entry ([3]) would be evicted first; once
    # reused it outranks fresh interiors and the least-recent unreused one
    # ([5]) goes instead.
    survivor = apc.lookup(key, [3, 4])
    assert survivor.hit and survivor.retention_role == "interior_checkpoint"
    survivor.cache.close()
    assert not apc.lookup(key, [5, 4]).hit
    assert apc.lookup(key, [6, 4]).hit
    assert apc.apc_stats["lifetime"]["interior_hits"] == 3
    apc.clear(release_memory=False)


# ------------------------------------------------------ serving mechanism


class _TinyCacheBudget:
    def project(self, context_tokens):
        return 4096 + 1024 * int(context_tokens)

    def as_dict(self):
        return {"schema": "tiny-test-budget"}


def _engine(model, vocab, marker, policy, *, mtp=True):
    base = make_adapter(model, vocab)

    class Adapter(base):
        def apc_turn_marker_ids(self):
            return (marker,)

        def cache_budget(self, *, mtp):
            # Interior capture is fail-closed without a cache projection.
            args = getattr(model, "args", None)
            if getattr(args, "model_type", None) == "qwen3_5":
                from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

                return Qwen38CacheBudget.from_config(dict(vars(args)), mtp=mtp)
            return _TinyCacheBudget()

    engine = serving.ServingEngine(
        "tiny",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=mtp,
        max_lanes=1,
        prefill_step=16,
        execution_policy={"apc_interior_checkpoints": policy},
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _interior_apc(engine):
    deadline = time.monotonic() + 10
    while True:
        status = engine.status()
        found = {}

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "apcv2" and isinstance(value, dict):
                        found.update(value)
                    walk(value)

        walk(status)
        lifetime = dict(found.get("lifetime") or {})
        hits = int(found.get("interior_hits", 0)) + int(lifetime.get("interior_hits", 0))
        if hits > 0 or time.monotonic() > deadline:
            return hits, found
        time.sleep(0.2)


def _session_prompts(vocab):
    marker = vocab - 1
    preamble = [(7 * i + 3) % (vocab - 2) + 1 for i in range(60)]
    user_a = [(5 * i + 1) % (vocab - 2) + 1 for i in range(20)]
    user_b = [(11 * i + 2) % (vocab - 2) + 1 for i in range(20)]
    assert user_a[0] != user_b[0]
    tail = [marker, 3, 4]
    a = preamble + [marker] + user_a + tail
    b = preamble + [marker] + user_b + tail
    return marker, len(preamble), a, b


CASES = [
    pytest.param(tiny_qwen4_mtp, True, id="qwen4_exp_mtp"),
    pytest.param(tiny_qwen38_mtp, True, id="qwen38_hybrid_mtp"),
    pytest.param(tiny_qwen38_mtp, False, id="qwen38_hybrid_ordinary"),
]


@pytest.mark.parametrize(("factory", "mtp"), CASES)
def test_shared_preamble_resumes_from_turn_boundary_interior_checkpoint(
    host, factory, mtp
):
    model, vocab = factory()
    marker, preamble, a, b = _session_prompts(vocab)
    policy = {"count": 4, "min_stride": 16, "placement": "auto"}
    warm = _engine(model, vocab, marker, policy, mtp=mtp)
    try:
        run(warm, a)
        out_warm, receipt, job = run(warm, b)
        hits, apc = _interior_apc(warm)
        counts = dict(warm.counts)
        settings = warm.status().get("settings", {})
    finally:
        warm.close()
    # Mechanism: the second session resumed exactly at the preamble boundary
    # from an interior checkpoint, and every counter along the path moved.
    assert int(job.cached_tokens) == preamble
    assert receipt.get("cache_checkpoint_role") == "interior_checkpoint"
    assert hits > 0
    assert counts["apc_interior_positions_planned_turn"] > 0
    assert counts["apc_interior_checkpoints_published"] > 0
    assert counts["apc_interior_hits"] > 0
    assert counts["apc_interior_hits_turn_boundary"] > 0
    assert counts["apc_interior_hit_tokens"] >= preamble
    assert apc["interior"]["lifetime_reused_entries"] >= 1
    if settings:
        assert settings["apc_interior_checkpoints"]["placement"] == "auto"

    cold = _engine(model, vocab, marker, {"count": 0, "min_stride": 1}, mtp=mtp)
    try:
        out_cold, _, cold_job = run(cold, b)
        assert int(cold_job.cached_tokens or 0) == 0
    finally:
        cold.close()
    assert out_warm == out_cold


def test_legacy_pow2_lattice_misses_the_preamble_boundary(host):
    """Regression: the old lattice resumes short of the shared preamble."""
    model, vocab = tiny_qwen38_mtp()
    marker, preamble, a, b = _session_prompts(vocab)
    warm = _engine(model, vocab, marker, {"count": 4, "min_stride": 16})
    try:
        run(warm, a)
        _out, _receipt, job = run(warm, b)
        counts = dict(warm.counts)
    finally:
        warm.close()
    assert int(job.cached_tokens) == 32 < preamble
    assert counts["apc_interior_positions_planned_lattice"] > 0
    assert counts["apc_interior_positions_planned_turn"] == 0
    assert counts["apc_interior_hits_turn_boundary"] == 0


def test_headroom_fraction_caps_checkpoints(host):
    model, vocab = tiny_qwen38_mtp()
    marker, _preamble, a, _b = _session_prompts(vocab)
    # Tiny headroom fraction: every candidate is dropped, counted as capped.
    warm = _engine(
        model,
        vocab,
        marker,
        {"count": 4, "min_stride": 16, "placement": "auto", "headroom_fraction": 1e-12},
    )
    try:
        run(warm, a)
        counts = dict(warm.counts)
    finally:
        warm.close()
    assert counts["apc_interior_positions_planned_turn"] > 0
    assert counts["apc_interior_positions_headroom_capped"] > 0
    assert counts.get("apc_interior_checkpoints_published", 0) == 0


def test_min_uncached_fraction_skips_capture_on_a_continuation_turn(host):
    """A deep-hit continuation pays nothing: no candidates, no publication."""
    model, vocab = tiny_qwen38_mtp()
    marker, preamble, a, b = _session_prompts(vocab)
    policy = {
        "count": 4, "min_stride": 16, "placement": "auto",
        "min_uncached_fraction": 0.9,
    }
    warm = _engine(model, vocab, marker, policy)
    try:
        run(warm, a)          # fresh prefill: captures
        first = dict(warm.counts)
        run(warm, a + [vocab - 3, vocab - 2])  # continuation of its own prefix
        counts = dict(warm.counts)
    finally:
        warm.close()
    assert first["apc_interior_checkpoints_published"] > 0
    assert counts["apc_interior_requests_skipped_continuation"] == 1
    assert (
        counts["apc_interior_checkpoints_published"]
        == first["apc_interior_checkpoints_published"]
    )
    assert (
        counts["apc_interior_positions_planned_turn"]
        == first["apc_interior_positions_planned_turn"]
    )


def test_min_uncached_fraction_policy_round_trip():
    from mlx2.serving import apc_interior_checkpoint_policy

    # "auto" now carries the qualified continuation skip; an explicit policy
    # that does not ask for it keeps the historical identity (key absent).
    assert apc_interior_checkpoint_policy("auto")["min_uncached_fraction"] == 0.5
    assert "min_uncached_fraction" not in apc_interior_checkpoint_policy(
        {"count": 4, "min_stride": 256, "placement": "auto"}
    )
    assert "min_uncached_fraction" not in apc_interior_checkpoint_policy(
        {"count": 4, "min_stride": 256, "min_uncached_fraction": 0.0}
    )
    policy = apc_interior_checkpoint_policy(
        {"count": 4, "min_stride": 256, "placement": "auto",
         "min_uncached_fraction": 0.5}
    )
    assert policy["min_uncached_fraction"] == 0.5
    for bad in (1.0, -0.1, True, "half"):
        with pytest.raises(ValueError):
            apc_interior_checkpoint_policy(
                {"count": 4, "min_stride": 256, "min_uncached_fraction": bad}
            )


# ------------------------------------------ generation-prompt turn boundary


class _ChatTemplateTokenizer:
    """A Qwen-style chat template over tiny-model token ids.

    ``keeps_think_block=False`` is the Qwen3.6 shape: a finished assistant
    turn re-renders without the ``<think>`` block its generation prompt
    ended with. ``True`` is the Qwen3.8 shape, which keeps it.
    """

    def __init__(self, vocab, *, keeps_think_block):
        (self.start, self.end, self.think, self.unthink, self.nl, self.nn) = range(
            vocab - 8, vocab - 2
        )
        self.roles = {"system": 1, "user": 2, "assistant": 3}
        self.keeps_think_block = keeps_think_block
        self.all_special_ids = [self.start, self.end, self.think, self.unthink]

    def _think(self, thinking):
        if thinking:
            return [self.think, self.nl]
        return [self.think, self.nn, self.unthink, self.nn]

    def apply_chat_template(
        self, messages, *, add_generation_prompt, tokenize=True, **flags
    ):
        thinking = flags.get("enable_thinking", False)
        out = []
        for message in messages:
            body = [
                int(word) if word.isdigit() else 1 + ord(word[0]) % 50
                for word in str(message["content"]).split()
            ]
            out += [self.start, self.roles[message["role"]], self.nl]
            if message["role"] == "assistant" and self.keeps_think_block:
                out += self._think(False)
            out += body + [self.end, self.nl]
        if add_generation_prompt:
            out += [self.start, self.roles["assistant"], self.nl]
            out += self._think(thinking)
        return out


def test_generation_prompt_suffix_is_detected_only_where_history_drops_it():
    from mlx2.runtime.interior_placement import (
        detect_generation_prompt_suffixes,
        generation_prompt_boundary,
    )

    dropping = _ChatTemplateTokenizer(128, keeps_think_block=False)
    suffixes = detect_generation_prompt_suffixes(dropping)
    t = dropping
    # Thinking off and thinking on each drop their own suffix.
    assert suffixes == (
        (t.start, 3, t.nl, t.think, t.nn, t.unthink, t.nn),
        (t.start, 3, t.nl, t.think, t.nl),
    )
    prompt = t.apply_chat_template(
        [{"role": "user", "content": "5 6 7"}], add_generation_prompt=True
    )
    assert generation_prompt_boundary(prompt, suffixes) == len(prompt) - 7
    assert generation_prompt_boundary(prompt[:-1], suffixes) is None
    # The Qwen3.8 shape re-renders the whole generation prompt: the ``P-1``
    # boundary already serves the next turn, so nothing is added.
    keeping = _ChatTemplateTokenizer(128, keeps_think_block=True)
    assert detect_generation_prompt_suffixes(keeping) == ()
    assert detect_generation_prompt_suffixes(object()) == ()


def _chat_engine(model, vocab, *, keeps_think_block, mtp, policy=None):
    base = make_adapter(model, vocab)
    template = _ChatTemplateTokenizer(vocab, keeps_think_block=keeps_think_block)

    class Adapter(base):
        def __init__(self, path):
            super().__init__(path)
            detokenizer = type(self).tokenizer

            class Tokenizer:
                vocab_size = vocab
                eos_token_ids = []
                all_special_ids = template.all_special_ids
                apply_chat_template = staticmethod(template.apply_chat_template)

                @property
                def detokenizer(self):
                    return detokenizer.detokenizer

            self.tokenizer = Tokenizer()

        def prompt_tokens(self, request):
            return template.apply_chat_template(
                request["messages"],
                add_generation_prompt=True,
                enable_thinking=request.get("enable_thinking", False),
            )

        def cache_budget(self, *, mtp):
            args = getattr(model, "args", None)
            if getattr(args, "model_type", None) == "qwen3_5":
                from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

                return Qwen38CacheBudget.from_config(dict(vars(args)), mtp=mtp)
            return _TinyCacheBudget()

    engine = serving.ServingEngine(
        "tiny",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=mtp,
        max_lanes=1,
        prefill_step=16,
        **({} if policy is None else {"execution_policy": policy}),
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _chat(engine, messages, max_tokens=6):
    job = engine.submit(
        {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
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


def _words(vocab, seed, count):
    return " ".join(str((seed * 7 + 11 * i) % (vocab - 10) + 1) for i in range(count))


@pytest.mark.parametrize(("factory", "mtp"), CASES)
@pytest.mark.parametrize("policy", [None, "auto"], ids=["default", "auto"])
def test_multiturn_chat_resumes_before_the_dropped_generation_prompt(
    host, factory, mtp, policy
):
    """X3-2: a template that drops the generation prompt from history.

    The Qwen3.6 template re-renders a finished assistant turn without the
    ``<think></think>`` its generation prompt ended with, so the stored
    ``P-1`` boundary and finished lane both diverge a few tokens before
    their end and the next turn re-prefilled everything. Serving now plans
    one exact boundary just before that suffix, by default and past the
    ``auto`` preset's continuation skip, on the MTP route too: the next turn
    re-prefills only its own turn plus the suffix region, with greedy output
    identical to a cold engine.
    """
    model, vocab = factory()
    suffix = 7
    messages = [
        {"role": "system", "content": _words(vocab, 1, 60)},
        {"role": "user", "content": _words(vocab, 2, 20)},
    ]
    template = _ChatTemplateTokenizer(vocab, keeps_think_block=False)
    engine = _chat_engine(
        model,
        vocab,
        keeps_think_block=False,
        mtp=mtp,
        policy=None if policy is None else {"apc_interior_checkpoints": policy},
    )
    try:
        previous = None
        for turn in range(3):
            prompt = template.apply_chat_template(messages, add_generation_prompt=True)
            out, receipt, job = _chat(engine, messages)
            if previous is not None:
                cached = int(job.cached_tokens or 0)
                assert cached == len(previous) - suffix, (turn, cached)
                assert prompt[:cached] == previous[:cached]
                assert receipt.get("cache_checkpoint_role") == "interior_checkpoint"
            previous = prompt
            messages = messages + [
                {"role": "assistant", "content": " ".join(map(str, out))},
                {"role": "user", "content": _words(vocab, 3 + turn, 20)},
            ]
        last_out = out
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert counts["apc_interior_positions_planned_generation_prompt"] >= 2
    assert counts["apc_interior_hits"] >= 2
    cold = _chat_engine(model, vocab, keeps_think_block=False, mtp=mtp)
    try:
        cold_out, _receipt, cold_job = _chat(cold, messages[:-2])
        assert int(cold_job.cached_tokens or 0) == 0
    finally:
        cold.close()
    assert cold_out == last_out


def test_template_that_keeps_the_generation_prompt_adds_no_boundary(host):
    """The Qwen3.8 shape already resumes from ``P-1``: nothing new is stored."""
    model, vocab = tiny_qwen38_mtp()
    messages = [
        {"role": "system", "content": _words(vocab, 1, 60)},
        {"role": "user", "content": _words(vocab, 2, 20)},
    ]
    template = _ChatTemplateTokenizer(vocab, keeps_think_block=True)
    engine = _chat_engine(model, vocab, keeps_think_block=True, mtp=True)
    try:
        first = template.apply_chat_template(messages, add_generation_prompt=True)
        out, _receipt, _job = _chat(engine, messages)
        messages += [
            {"role": "assistant", "content": " ".join(map(str, out))},
            {"role": "user", "content": _words(vocab, 3, 20)},
        ]
        _out, _receipt, job = _chat(engine, messages)
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert int(job.cached_tokens) >= len(first) - 1
    assert counts.get("apc_interior_positions_planned_generation_prompt", 0) == 0
    assert counts.get("apc_interior_checkpoints_published", 0) == 0
