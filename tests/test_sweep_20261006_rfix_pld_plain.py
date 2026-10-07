"""Plain (proposal-free) PLD rounds on hybrid cohorts cost an ordinary step.

GPU evidence (Qwen3.8-27B hybrid GDN, B4 summarisation): 95% of PLD rounds
proposed nothing, yet each cost about twice an ordinary B4 decode step: the
round built its forward, waited for it, and only then built the next, while
ordinary decode builds step k+1 while the device runs step k.  A plain round
of an eligible hybrid cohort is now dispatched before the previous one is
read (its anchors are that round's lazy tokens); a lane that finishes on the
read round is released from the following one before its cache is frozen.
"""
import mlx.core as mx
import pytest
from test_segmented_mtp import _tiny_hybrid_mtp_model
from test_sweep_20261006_spec_pld_hybrid import PROMPTS, _greedy

from mlx2.runtime.generate import StopSequenceMatcher
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.pld import PromptLookupBatchGenerator
from mlx2.runtime.sample_utils import LaneRNG

# No repeated n-gram: prompt lookup mostly finds nothing to propose.
RANDOM_PROMPTS = [
    [(17 * i + 31 * lane + 7) % 120 + 1 for i in range(40)] for lane in range(4)
]


class _Forwards:
    def __init__(self, model):
        self.model = model
        self.log = []

    def __call__(self, tokens, cache=None, **kwargs):
        self.log.append(tuple(tokens.shape))
        return self.model(tokens, cache=cache, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _run(prompts, max_tokens, *, model=None, samplers=None, stop=None,
         remove_after=None, polls=800, **policy):
    model = model or _tiny_hybrid_mtp_model()
    forwards = _Forwards(model)
    generator = PromptLookupBatchGenerator(
        forwards, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup={"num_draft": 3, "ngram_min": 2, "ngram_max": 3,
                       "adaptive": False, **policy},
    )
    uids = generator.insert(
        prompts, max_tokens=list(max_tokens),
        caches=[model.make_cache() for _ in prompts], samplers=samplers,
        stop_matchers=None if stop is None else [StopSequenceMatcher(s) for s in stop],
    )
    tokens = {uid: [] for uid in uids}
    finals = {}
    removed = False
    try:
        for poll in range(polls):
            _prompts, responses = generator.next()
            for response in responses:
                tokens[response.uid].append(response.token)
                if response.finish_reason:
                    finals[response.uid] = response
            if remove_after is not None and not removed and poll >= remove_after[1]:
                generator.remove([uids[remove_after[0]]])
                removed = True
            if not generator.lanes:
                break
    finally:
        generator.close()
    return model, [tokens[uid] for uid in uids], finals, generator.scheduler_stats, forwards.log


def _greedy_cache(model, prompt, tokens):
    """B1 ordinary cache over ``prompt + tokens[:-1]`` (what a finish freezes)."""
    cache = model.make_cache()
    model(mx.array([prompt]), cache=cache)
    for token in tokens[:-1]:
        model(mx.array([[token]]), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def _assert_same_cache(actual, expected):
    for got, want in zip(actual, expected):
        if isinstance(want, ArraysCache):
            for a, b in zip(got.cache, want.cache):
                assert mx.allclose(a, b, atol=1e-5).item()
        else:
            assert isinstance(want, KVCache)
            assert int(got.offset) == int(want.offset)
            keys, values = got.keys_and_values()
            want_keys, want_values = want.keys_and_values()
            assert mx.allclose(keys, want_keys, atol=1e-5).item()
            assert mx.allclose(values, want_values, atol=1e-5).item()


def test_plain_rounds_pipeline_and_match_greedy_decode():
    model, outputs, finals, stats, _log = _run(RANDOM_PROMPTS, [24, 17, 30, 21])
    assert stats["pld_pipelined_rounds"] > 0
    # Every booked token was delivered: no round ran for a finished lane.
    assert stats["pld_accepted"] + stats["pld_bonus"] == sum(map(len, outputs))
    for prompt, output, count in zip(RANDOM_PROMPTS, outputs, [24, 17, 30, 21]):
        assert output == _greedy(model, prompt, count)
    # Lanes finishing by length never ran a released row: the pipeline
    # drains before the round that would finish a lane.
    assert stats["pld_pipeline_released_rows"] == 0
    # Every finished lane froze exactly its committed boundary.
    for uid, prompt, output in zip(sorted(finals), RANDOM_PROMPTS, outputs):
        _assert_same_cache(finals[uid].prompt_cache, _greedy_cache(model, prompt, output))


def test_pipelined_rounds_are_one_forward_and_labelled_plain():
    _model, _outputs, finals, stats, log = _run(RANDOM_PROMPTS, [40] * 4)
    # One prefill forward per prompt, then exactly one forward per round:
    # dispatching ahead never runs a round twice or throws one away.
    assert len(log) == len(RANDOM_PROMPTS) + stats["pld_batched_rounds"]
    assert stats["pld_pipelined_rounds"] * 2 > stats["pld_batched_rounds"]
    receipt = finals[min(finals)].speculative_receipt
    assert receipt["pipelined_plain_rounds"] > 0
    # Pipelined rounds are plain decode steps, never verify rows.
    assert receipt["verify_path_rounds"] + receipt["pipelined_plain_rounds"] <= receipt["cycles"]


def test_disabled_pipeline_never_dispatches_ahead():
    model, outputs, _finals, stats, _log = _run(
        RANDOM_PROMPTS, [20] * 4, pipelined_plain=False
    )
    assert stats["pld_pipelined_rounds"] == 0
    for prompt, output in zip(RANDOM_PROMPTS, outputs):
        assert output == _greedy(model, prompt, 20)


def test_stop_on_a_read_round_releases_the_following_row():
    model = _tiny_hybrid_mtp_model()
    reference = [_greedy(model, prompt, 30) for prompt in RANDOM_PROMPTS]
    # Each lane stops on a token it emits mid-stream (its 12th + lane).
    stops = [[[ref[11 + lane]]] for lane, ref in enumerate(reference)]
    _model, outputs, finals, stats, _log = _run(
        RANDOM_PROMPTS, [30] * 4, model=model, stop=stops
    )
    assert stats["pld_pipeline_released_rows"] > 0
    assert stats["pld_accepted"] + stats["pld_bonus"] == sum(map(len, outputs))
    for lane, (output, ref) in enumerate(zip(outputs, reference)):
        end = ref.index(stops[lane][0][0]) + 1
        assert output == ref[:end]
    for uid, prompt, output in zip(sorted(finals), RANDOM_PROMPTS, outputs):
        _assert_same_cache(finals[uid].prompt_cache, _greedy_cache(model, prompt, output))


def test_removing_a_lane_mid_pipeline_keeps_its_peers_exact():
    model, outputs, _finals, stats, _log = _run(
        RANDOM_PROMPTS, [30] * 4, remove_after=(1, 12)
    )
    assert stats["pld_pipelined_rounds"] > 0
    for lane, (prompt, output) in enumerate(zip(RANDOM_PROMPTS, outputs)):
        reference = _greedy(model, prompt, 30)
        if lane == 1:
            assert output == reference[: len(output)] and len(output) < 30
        else:
            assert output == reference


def _seeded_samplers(seed):
    def sampler(rng):
        def draw(logprobs):
            return mx.random.categorical(logprobs * 1.25, key=rng.next_key())
        return draw

    return [sampler(LaneRNG(seed + lane)) for lane in range(4)]


@pytest.mark.parametrize("prompts", [RANDOM_PROMPTS, PROMPTS], ids=["plain", "proposing"])
def test_pipelining_takes_the_same_draws_from_each_lane_stream(prompts):
    """One draw per emitted token either way: dispatching the anchor's draw
    early does not move any lane's stream, with or without proposals."""
    model = _tiny_hybrid_mtp_model()
    _m, piped, _f, stats, _l = _run(
        prompts, [24] * 4, model=model, samplers=_seeded_samplers(5)
    )
    _m, serial, _f, _s, _l = _run(
        prompts, [24] * 4, model=model, samplers=_seeded_samplers(5),
        pipelined_plain=False,
    )
    assert stats["pld_pipelined_rounds"] > 0
    assert piped == serial


class _Fault:
    def __init__(self, mlp):
        self.mlp = mlp
        self.armed = False
        self.fired = 0

    def __call__(self, x):
        if self.armed:
            self.armed = False
            self.fired += 1
            raise RuntimeError("injected plain-forward fault")
        return self.mlp(x)


def test_failed_pipelined_dispatch_leaves_every_lane_exact():
    model = _tiny_hybrid_mtp_model()
    layer = model.language_model.model.layers[3]
    fault = _Fault(layer.mlp)
    layer.mlp = fault
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup={"num_draft": 3, "ngram_min": 2, "ngram_max": 3, "adaptive": False},
    )
    uids = generator.insert(RANDOM_PROMPTS, max_tokens=[24] * 4,
                            caches=[model.make_cache() for _ in RANDOM_PROMPTS])
    tokens = {uid: [] for uid in uids}
    errors = []
    try:
        for _ in range(400):
            if generator._inflight is not None and not fault.fired and len(tokens[uids[0]]) > 6:
                # The next poll dispatches the following round before it
                # reads this one; that dispatch fails part-way.
                fault.armed = True
            try:
                _prompts, responses = generator.next()
            except RuntimeError as error:
                errors.append(error)
                continue
            for response in responses:
                tokens[response.uid].append(response.token)
            if not generator.lanes:
                break
    finally:
        generator.close()
    assert fault.fired == 1 and len(errors) == 1
    for uid, prompt in zip(uids, RANDOM_PROMPTS):
        assert tokens[uid] == _greedy(model, prompt, 24)


def test_invalid_token_on_a_read_round_drops_only_that_lane():
    model = _tiny_hybrid_mtp_model()
    calls = [0]

    def argmax(logprobs):
        return mx.argmax(logprobs, axis=-1)

    def failing(logprobs):
        calls[0] += 1
        token = mx.argmax(logprobs, axis=-1)
        # Sampled at dispatch: the 10th draw goes out of vocabulary.
        return mx.full(token.shape, 4096, token.dtype) if calls[0] == 10 else token

    argmax.deterministic = failing.deterministic = True
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup={"num_draft": 3, "ngram_min": 2, "ngram_max": 3, "adaptive": False},
    )
    uids = generator.insert(
        RANDOM_PROMPTS, max_tokens=[24] * 4,
        caches=[model.make_cache() for _ in RANDOM_PROMPTS],
        samplers=[argmax, failing, argmax, argmax],
    )
    tokens = {uid: [] for uid in uids}
    try:
        for _ in range(400):
            _prompts, responses = generator.next()
            for response in responses:
                tokens[response.uid].append(response.token)
            if not generator.lanes:
                break
        failures = generator.take_lane_failures()
        stats = generator.scheduler_stats
    finally:
        generator.close()
    assert [failure["uid"] for failure in failures] == [uids[1]]
    assert stats["pld_pipelined_rounds"] > 0
    assert stats["pld_pipeline_released_rows"] >= 1
    for lane, (uid, prompt) in enumerate(zip(uids, RANDOM_PROMPTS)):
        reference = _greedy(model, prompt, 24)
        if lane == 1:
            assert tokens[uid] == reference[: len(tokens[uid])] and len(tokens[uid]) < 24
        else:
            assert tokens[uid] == reference


# ----------------------------- views kept across a cohort's rounds
def _counting_builds(monkeypatch):
    from mlx2.runtime import segmented_batch_cache

    builds, joins = [0], [0]
    real_build = segmented_batch_cache.build_segmented_batch_cache_group
    real_refresh = segmented_batch_cache.SegmentedBatchArraysCache._refresh_state

    def build(*args, **kwargs):
        builds[0] += 1
        return real_build(*args, **kwargs)

    def refresh(self):
        joins[0] += 1
        return real_refresh(self)

    monkeypatch.setattr(segmented_batch_cache, "build_segmented_batch_cache_group", build)
    monkeypatch.setattr(segmented_batch_cache.SegmentedBatchArraysCache, "_refresh_state", refresh)
    return builds, joins


def test_plain_cohort_builds_its_views_once_and_never_rejoins(monkeypatch):
    builds, joins = _counting_builds(monkeypatch)
    model = _tiny_hybrid_mtp_model()
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup={"num_draft": 3, "ngram_min": 2, "ngram_max": 3, "adaptive": False},
    )
    generator.insert(RANDOM_PROMPTS, max_tokens=[60] * 4,
                     caches=[model.make_cache() for _ in RANDOM_PROMPTS])
    try:
        while any(lane.anchor is None for lane in generator.lanes.values()):
            generator.next()
        for _ in range(4):
            generator.next()  # the cohort is whole and pipelined
        builds[0] = joins[0] = 0
        rounds0 = generator.scheduler_stats["pld_batched_rounds"]
        for _ in range(12):
            generator.next()
        rounds = generator.scheduler_stats["pld_batched_rounds"] - rounds0
        stats = dict(generator.scheduler_stats)
    finally:
        generator.close()
    assert rounds == 12 and stats["pld_proposed"] == 0
    # The cohort's views were built before this window; plain rounds reuse
    # them and their joined recurrent state (no per-round concatenation).
    assert builds[0] == 0 and joins[0] == 0


def test_kept_views_rejoin_after_an_abort_and_stay_exact():
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    model = _tiny_hybrid_mtp_model()
    rows, references = [], []
    for prompt in RANDOM_PROMPTS[:3]:
        cache = model.make_cache()
        model(mx.array([prompt]), cache=cache)
        rows.append(cache)
        reference = model.make_cache()
        model(mx.array([prompt]), cache=reference)
        references.append(reference)
    owner = HybridVerifyRows(rows)
    steps = [[3, 5, 7], [11, 13, 17], [19, 23, 29], [31, 37, 41]]
    views = None
    for step, tokens in enumerate(steps):
        transaction = owner.begin([1, 1, 1], plain_single_token=True)
        if views is not None:
            assert transaction.caches is views  # kept across rounds
        views = transaction.caches
        model(mx.array([[t] for t in tokens]), cache=transaction.caches)
        if step == 2:
            transaction.abort()  # rows rewound under the kept views
            assert owner._views is None
            views = None
            continue
        transaction.commit([1, 1, 1])
        for reference, token in zip(references, tokens):
            model(mx.array([[token]]), cache=reference)
    for row, reference in zip(rows, references):
        mx.eval([c.state for c in row], [c.state for c in reference])
        _assert_same_cache(row, reference)


# ------------------- dropped proposals still feed the adaptive gate
def test_refuted_dropped_proposals_latch_a_lane_off_retrieval(monkeypatch):
    """Proposals that only ever surface in pipelined rounds are dropped, but
    each round's own token tests the first draft.  A lane whose drafts are
    always wrong reaches the adaptive gate and stops proposing, as it would
    from verify rounds; without that it would drain the pipeline forever."""
    from mlx2.runtime import prompt_lookup

    model = _tiny_hybrid_mtp_model()
    reference = [_greedy(model, prompt, 60) for prompt in RANDOM_PROMPTS]
    wrong = next(t for t in range(1, 128) if all(t not in ref for ref in reference))
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup={"num_draft": 3, "ngram_min": 2, "ngram_max": 3,
                       "adaptive": True, "adaptive_warmup": 0,
                       "admission_window": 3, "adaptive_gate": 0.5},
    )
    real_steps = generator._round_steps

    def steps(lane, plain=False):
        lane._in_plain = plain
        return real_steps(lane, plain=plain)

    calls = {}

    def propose(self, *args, **kwargs):
        calls[id(self)] = calls.get(id(self), 0) + 1
        lane = next(l for l in generator.lanes.values() if l.proposer is self)
        # A wrong draft, only ever found by an already-dispatched plain round.
        return [wrong, wrong] if lane._in_plain and calls[id(self)] % 2 else []

    generator._round_steps = steps
    monkeypatch.setattr(prompt_lookup.IndexedPromptLookup, "propose", propose)
    uids = generator.insert(RANDOM_PROMPTS, max_tokens=[60] * 4,
                            caches=[model.make_cache() for _ in RANDOM_PROMPTS])
    tokens = {uid: [] for uid in uids}
    try:
        for _ in range(600):
            _prompts, responses = generator.next()
            for response in responses:
                tokens[response.uid].append(response.token)
            if not generator.lanes:
                break
        stats = generator.scheduler_stats
    finally:
        generator.close()
    assert stats["pld_retrieval_cycles"] == 0  # never verified
    assert stats["pld_pipeline_refuted_proposals"] >= 3 * len(uids)
    assert stats["pld_fallbacks"] == len(uids)  # every lane latched
    for uid, ref in zip(uids, reference):
        assert tokens[uid] == ref
