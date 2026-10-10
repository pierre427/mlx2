"""State-aware release of a run-on reasoning channel."""
import mlx.core as mx
import numpy as np
import pytest

from mlx2.thinking_guard import ThinkingGuard

CLOSE, VOCAB = 7, 16


def _call(guard, generated, prompt=(1, 2)):
    out = guard(mx.array(list(prompt) + list(generated), dtype=mx.uint32), mx.zeros((1, VOCAB)))
    return np.array(out)[0]


def test_silent_below_the_soft_budget_then_ramps_then_forces_the_close():
    guard = ThinkingGuard(2, (CLOSE,), budget=10, soft_ratio=0.5, ramp_nats=2.0)
    novel = list(range(8, 16)) + [3, 4, 5, 6]
    assert not _call(guard, novel[:4]).any()  # untouched logits
    ramp = [_call(guard, novel[:n])[CLOSE] for n in (5, 6, 7)]
    assert ramp == [2.0, 4.0, 6.0] and guard.receipt()["tripped"] == "budget_soft"
    assert _call(guard, novel[:7])[3] == 0.0  # only the close token is biased
    forced = _call(guard, novel[:10])
    assert np.isfinite(forced[CLOSE]) and np.isinf(forced[np.arange(VOCAB) != CLOSE]).all()
    assert guard.receipt()["forced_close"] is True
    # Once the channel is closed the guard is out of the way for good.
    assert not _call(guard, novel[:6] + [CLOSE, 9, 9, 9, 9, 9, 9, 9, 9]).any()
    assert guard.receipt()["released_at"] == 6 and guard.receipt()["think_tokens"] == 6


def test_run_on_alarm_releases_a_loop_long_before_the_budget():
    guard = ThinkingGuard(2, (CLOSE,), budget=4000, tau=2.5, ngram=6)
    loop = [8, 9, 10, 11, 12, 13, 14, 15] * 8
    biases = [_call(guard, loop[:n])[CLOSE] for n in range(1, len(loop) + 1)]
    first = next(i for i, b in enumerate(biases) if b > 0)
    assert 16 < first < 40 and guard.receipt()["tripped"] == "run_on"
    assert biases[first + 3] > biases[first]
    fresh = ThinkingGuard(2, (CLOSE,), budget=4000)
    novel = [(i * 7 + 3) % 5 + 8 + (i % 3) for i in range(12)]
    assert not _call(fresh, list(range(8, 16)) + [3, 4, 5, 6]).any() and fresh.receipt()["tripped"] is None


def test_guard_is_a_pure_function_of_the_ids_across_rollbacks():
    ids = list(range(8, 16)) + [3, 4, 5, 6]
    live = ThinkingGuard(2, (CLOSE,), budget=10, soft_ratio=0.5)
    _call(live, ids[:9])
    _call(live, ids[:4] + [15, 15, 15])       # speculative row that diverges
    rolled_back = _call(live, ids[:7])        # verify rejected it
    fresh = _call(ThinkingGuard(2, (CLOSE,), budget=10, soft_ratio=0.5), ids[:7])
    np.testing.assert_array_equal(rolled_back, fresh)
    with pytest.raises(ValueError, match="single-token"):
        ThinkingGuard(2, (1, 2), budget=10, direction={"layer": 0, "vector": mx.ones((4,))}, alpha=0.2)


# A GPT-OSS-shaped marker: <|end|><|start|>assistant<|channel|>final<|message|>.
SWITCH = (3, 4, 5, 6)


def _multi(budget=10, **config):
    return ThinkingGuard(2, SWITCH, budget=budget, soft_ratio=0.5, ramp_nats=2.0, **config)


def _masked_to(row):
    finite = np.flatnonzero(np.isfinite(row))
    return int(finite[0]) if finite.size == 1 else None


def test_multi_token_close_is_forced_token_by_token_after_the_budget():
    guard = _multi()
    novel = list(range(8, 16)) + [9, 10]
    assert not _call(guard, novel[:4]).any()
    # The release boosts the first marker token until the model writes it.
    assert _call(guard, novel[:6])[SWITCH[0]] == 4.0 and not _call(guard, novel[:6])[SWITCH[1]]
    ids = novel[:10]
    for expected in SWITCH:
        row = _call(guard, ids)
        assert _masked_to(row) == expected
        ids.append(expected)
    assert guard.receipt()["forced_close"] is True
    assert not _call(guard, ids + [9, 9]).any()  # closed: out of the way
    assert guard.receipt()["released_at"] == 10 and guard.receipt()["think_tokens"] == 10
    settled = _multi()
    settled.settle(ids + [9])
    assert settled.receipt()["forced_close"] is True


def test_multi_token_release_boosts_the_next_marker_token_after_a_written_prefix():
    guard = _multi(budget=40)
    ids = list(range(8, 16)) + [9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12]
    ids = ids[:21]  # past the soft budget (20)
    assert _call(guard, ids)[SWITCH[0]] > 0
    row = _call(guard, ids + [SWITCH[0]])
    assert row[SWITCH[1]] > 0 and row[SWITCH[0]] == 0.0
    row = _call(guard, ids + list(SWITCH[:3]))
    assert row[SWITCH[3]] > 0


def test_forced_close_continues_a_marker_prefix_the_model_already_wrote():
    guard = _multi()
    ids = list(range(8, 16)) + [9, SWITCH[0]]  # the budget lands after <|end|>
    assert _masked_to(_call(guard, ids)) == SWITCH[1]  # not a second <|end|>


def test_a_partial_or_different_switch_does_not_close_reasoning():
    guard = _multi(budget=None)
    # <|end|><|start|>assistant<|channel|>analysis: another analysis message.
    ids = [8, 9] + list(SWITCH[:3]) + [11, 12, 13]
    _call(guard, ids)
    assert guard.receipt()["released_at"] is None and not guard.dormant(np.array([1, 2] + ids))
    closed = [1, 2, 8, 9] + list(SWITCH) + [11]
    assert guard.dormant(np.array(closed))
    _call(guard, closed[2:])
    assert guard.receipt()["released_at"] == 2 and guard.receipt()["forced_close"] is False


def test_multi_token_guard_matches_a_fresh_guard_under_random_rollbacks():
    rng = np.random.default_rng(1)
    marker = (3, 4, 5)

    def make():
        return ThinkingGuard(2, marker, budget=150, soft_ratio=0.8, tau=2.5,
                             ngram=4, rewrite_window=32)
    live, ids = make(), []
    for _ in range(600):
        if ids and rng.random() < 0.2:
            del ids[len(ids) - int(rng.integers(1, min(len(ids), 12) + 1)):]
        ids += [int(token) for token in rng.integers(3, 12, size=int(rng.integers(1, 4)))]
        ours = _call(live, ids)
        fresh_guard = make()
        np.testing.assert_array_equal(ours, _call(fresh_guard, ids))
        assert live.receipt()["released_at"] == fresh_guard.receipt()["released_at"]


def test_rejected_close_restores_steering_and_release_receipt():
    direction = {"layer": 1, "vector": np.ones(4)}
    guard = ThinkingGuard(2, (CLOSE,), budget=20, direction=direction, alpha=0.2)
    _call(guard, [8, CLOSE])
    assert guard.released_at == 1 and guard.residual_steer(8) is None
    _call(guard, [8, 9])
    assert guard.released_at is None
    assert guard.receipt()["released_at"] is None
    assert guard.residual_steer(9) is not None


def test_rejected_hard_budget_clears_forced_receipt():
    guard = ThinkingGuard(2, (CLOSE,), budget=3, soft_ratio=0.5)
    _call(guard, [8, 9, 10])
    assert guard.receipt()["forced_close"]
    _call(guard, [8, 9])
    assert not guard.receipt()["forced_close"]


class _CountingTokens:
    """Token context that records how many ids the guard reads per step."""

    def __init__(self, ids):
        self.array = mx.array(ids, dtype=mx.uint32)
        self.read = 0

    @property
    def size(self):
        return self.array.size

    def __getitem__(self, key):
        return _CountingTokens(self.array[key].tolist())._counted(self)

    def _counted(self, parent):
        parent.read += self.array.size
        return self

    def tolist(self):
        return self.array.tolist()


def test_guard_step_reads_only_the_tail_of_a_long_reasoning_channel():
    # vllm #52677: rescanning every generated id made each step O(n).
    prompt, window = (1, 2), 64
    guard = ThinkingGuard(2, (CLOSE,), budget=None, rewrite_window=window)
    ids = [8 + (i * 5 + i // 9) % 8 for i in range(4096)]
    reads = []
    for n in range(1, len(ids) + 1):
        tokens = _CountingTokens(list(prompt) + ids[:n])
        guard(tokens, mx.zeros((1, VOCAB)))
        reads.append(tokens.read)
    assert max(reads) <= window + 1
    assert guard.receipt()["think_tokens"] == len(ids)


def test_incremental_guard_matches_a_fresh_guard_under_random_rollbacks():
    rng = np.random.default_rng(0)
    def make():
        return ThinkingGuard(2, (CLOSE,), budget=300, soft_ratio=0.8, tau=2.5,
                             ngram=4, rewrite_window=32)
    live, ids = make(), []
    for _ in range(600):
        if ids and rng.random() < 0.2:
            del ids[len(ids) - int(rng.integers(1, min(len(ids), 12) + 1)):]
        ids += [int(token) for token in rng.integers(8, 12, size=int(rng.integers(1, 4)))]
        if rng.random() < 0.01:
            ids.append(CLOSE)
        ours = _call(live, ids)
        fresh_guard = make()
        np.testing.assert_array_equal(ours, _call(fresh_guard, ids))
        if CLOSE in ids:  # a fresh guard never runs the alarm past a close
            continue
        mine, theirs = live.receipt(), fresh_guard.receipt()
        for key in ("think_tokens", "tripped", "tripped_at"):
            assert mine[key] == theirs[key], key


def test_juice_ladder_scales_the_anchor_by_effort_and_explicit_budgets_win():
    from mlx2.thinking_guard import resolve_thinking_budget

    assert resolve_thinking_budget({}, 0) == 0  # no anchor, no request: no guard
    assert resolve_thinking_budget({"thinking_budget": 300}, 0) == 300
    assert resolve_thinking_budget({"thinking_budget": 300, "reasoning_effort": "high"}, 1000) == 300
    assert resolve_thinking_budget({"reasoning_effort": "low"}, 1000) == 375
    assert resolve_thinking_budget({"reasoning_effort": "medium"}, 1000) == 1000
    assert resolve_thinking_budget({"reasoning_effort": "high"}, 1000) == 4000
    assert resolve_thinking_budget({}, 1000) == 4000  # unspecified effort thinks at "high"
    assert resolve_thinking_budget({"reasoning_effort": "ultra"}, 4000) == 8192  # API ceiling
    assert resolve_thinking_budget({"reasoning_effort": "minimal"}, 100) == 64


# --- alpha actuator: calibrated commit-direction steering -------------------
def _tiny_north(seed=5):
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    mx.random.seed(seed)
    model = Model(ModelArgs(
        hidden_size=16, head_dim=4, num_hidden_layers=4, intermediate_size=8,
        prefix_dense_intermediate_size=24, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=32, num_experts=4, num_experts_per_tok=2, first_k_dense_replace=1,
        sliding_window=6,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    ))
    model.eval()
    return model


def test_residual_taps_steer_only_the_rows_that_ask_and_capture_residuals():
    model = _tiny_north()
    taps = model.model.residual_taps
    tokens = mx.array([[1, 2, 3], [4, 5, 6]])
    plain = model(tokens)
    vector = mx.zeros((2, 1, 16))
    vector[0] = 3.0  # lane 0 steered, lane 1 a zero row
    taps.steer = (1, vector)
    taps.capture = {1: None, 3: None}
    try:
        steered = model(tokens)
        captured = dict(taps.capture)
    finally:
        taps.steer = taps.capture = None
    assert not mx.allclose(steered[0], plain[0]).item()
    assert mx.allclose(steered[1], plain[1], atol=1e-5).item()
    assert captured[1].shape == captured[3].shape == (2, 3, 16)
    assert mx.array_equal(model(tokens), plain).item()  # taps off: the plain forward again


def test_guard_steers_while_reasoning_is_open_and_hammers_after_the_alarm():
    direction = {"layer": 2, "vector": mx.ones((16,)), "source": "test"}
    guard = ThinkingGuard(2, (CLOSE,), budget=10, soft_ratio=0.5, direction=direction, alpha=0.2, hammer=0.8)
    layer, vector = guard.residual_steer(9)
    assert layer == 2 and abs(float(vector[0]) - 0.2) < 1e-6
    _call(guard, list(range(8, 16)))                    # past the soft budget: tripped
    assert abs(float(guard.residual_steer(9)[1][0]) - 0.8) < 1e-6
    assert guard.residual_steer(CLOSE) is None          # the close token just went in
    assert guard.residual_steer(9) is None              # and steering never resumes
    receipt = guard.receipt()["steering"]
    assert receipt["steered_steps"] == 2 and receipt["alpha"] == 0.2 and receipt["source"] == "test"
    assert ThinkingGuard(2, (CLOSE,), budget=10).residual_steer(9) is None  # no direction, no steering
    assert ThinkingGuard(2, (CLOSE,), budget=10).receipt()["steering"] is None


def test_generator_applies_per_lane_steering_and_always_clears_the_tap():
    from mlx2.runtime.generate import BatchGenerator

    model = _tiny_north()

    class Steer:
        def __init__(self):
            self.calls = 0

        def residual_steer(self, _token):
            self.calls += 1
            return 1, (mx.arange(16) * 9.0 - 60.0) * (1 if self.calls % 2 else -1)

        def __call__(self, _tokens, logits):
            return logits

    def run(processors):
        batch = BatchGenerator(model, completion_batch_size=2, prefill_batch_size=2, prefill_step_size=8)
        uids = batch.insert([[1, 2, 3, 4], [5, 6, 7]], max_tokens=[6, 6], logits_processors=processors)
        out = {uid: [] for uid in uids}
        for _ in range(60):
            _prompts, responses = batch.next()
            for response in responses:
                out[response.uid].append(response.token)
            assert model.model.residual_taps.steer is None
            if all(len(v) >= 6 for v in out.values()):
                break
        return [out[uid] for uid in uids]

    plain = run([[], []])
    steer = Steer()
    steered = run([[steer], []])
    assert steer.calls > 0
    assert steered[1] == plain[1]   # the unsteered lane shares the forward, untouched
    assert steered[0] != plain[0]   # a strong vector changes the steered lane


def test_prompt_lookup_batched_verify_steers_per_lane_and_skips_rounds_past_the_close():
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    model = _tiny_north()

    class Steer:
        close_ids = (31,)

        def __init__(self):
            self.calls = 0

        def residual_steer(self, _token):
            self.calls += 1
            return 1, (mx.arange(16) * 9.0 - 60.0)

        def __call__(self, _tokens, logits):
            return logits

    def run(processors):
        generator = PromptLookupBatchGenerator(
            model, completion_batch_size=2, prefill_step_size=5,
            prompt_lookup={"num_draft": 4, "ngram_min": 2, "ngram_max": 3, "adaptive": False, "deferred_admission": False},
        )
        uids = generator.insert([[1, 2, 3, 1, 2, 3, 1, 2], [7, 8, 9, 7, 8, 9, 7]], max_tokens=[10, 10], logits_processors=processors)
        out = {uid: [] for uid in uids}
        for _ in range(400):
            _prompts, responses = generator.next()
            for response in responses:
                out[response.uid].append(response.token)
            assert model.model.residual_taps.steer is None
            if not generator.lanes:
                break
        return [out[uid] for uid in uids], generator.scheduler_stats

    plain, _stats = run([[], []])
    steer = Steer()
    steered, stats = run([[steer], []])
    assert steer.calls > 0 and stats["pld_batched_rounds"] > 0
    assert steered[1] == plain[1] and len(steered[0]) == 10
    # Per position, as ordinary decode steers each step: the anchor and the
    # rows before the close token are steered, the close and later rows not.
    generator = PromptLookupBatchGenerator(model, completion_batch_size=1, prefill_step_size=5, prompt_lookup={"num_draft": 4})
    guard = ThinkingGuard(2, (31,), budget=None, direction={"layer": 1, "vector": mx.ones((16,))}, alpha=0.5)
    lane = type("Lane", (), {"processors": [guard], "history": [1, 2]})()
    taps, (layer, rows), commits = generator._verify_steer([lane], [[5, 6, 31, 7]])
    assert layer == 1 and rows.shape == (1, 4, 16)
    assert rows[0, :, 0].tolist() == [0.5, 0.5, 0.0, 0.0]
    assert guard.steered_steps == 0  # nothing counted until the round commits
    commits[0](3)
    assert guard.steered_steps == 2
    # A generated close anchor closes the channel for the whole block.
    assert generator._verify_steer([lane], [[31, 6]])[:2] == (None, None)


_NORTH_CLOSE = 31


def _north_steering_guard(prompt_length, **overrides):
    config = {
        "budget": None,
        "direction": {"layer": 1, "vector": mx.arange(16, dtype=mx.float32) * 9.0 - 60.0},
        "alpha": 1.0,
    }
    config.update(overrides)
    return ThinkingGuard(prompt_length, (_NORTH_CLOSE,), **config)


def _drain_one(generator):
    tokens = []
    for _ in range(200):
        _prompts, responses = generator.next()
        tokens.extend(response.token for response in responses)
        if any(response.finish_reason for response in responses):
            return tokens
    raise AssertionError("generator stalled")


@pytest.mark.parametrize("policy", [{}, {"batched_verify": False}, {"rotating_replay": True}])
@pytest.mark.parametrize(
    "prompt,overrides,close_in_proposal",
    [
        ([5, 6, _NORTH_CLOSE, 7, 5, 6], {}, True),
        ([4, 18, 27] * 3, {}, False),
        # The alarm trips mid-run: the hammer takes over, then the close lands.
        ([4, 18, 27] * 3, {"budget": 12, "soft_ratio": 0.3, "hammer": 3.0, "alpha": 0.2}, True),
    ],
)
def test_prompt_lookup_steering_matches_ordinary_steered_greedy(policy, prompt, overrides, close_in_proposal):
    """Batched and per-lane PLD steer every verify row as ordinary decode does.

    A block holding the close token used to go wholly unsteered (anchor
    included), and the per-lane path never steered at all.
    """
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    model = _tiny_north()

    def ordinary(guard):
        generator = BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=8)
        generator.insert([prompt], max_tokens=[10], logits_processors=[[guard] if guard else []])
        try:
            return _drain_one(generator)
        finally:
            generator.close()

    reference = ordinary(_north_steering_guard(len(prompt), **overrides))
    assert reference != ordinary(None)  # the steering is load-bearing here
    guard = _north_steering_guard(len(prompt), **overrides)
    blocks = []
    real_block = guard.residual_steer_block

    def spy(context, inputs):
        blocks.append(list(inputs))
        return real_block(context, inputs)

    guard.residual_steer_block = spy
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=1, prefill_step_size=5,
        prompt_lookup={"num_draft": 4, "ngram_min": 2, "ngram_max": 3, "adaptive": False,
                       "deferred_admission": False, **policy},
    )
    generator.insert([prompt], max_tokens=[10], logits_processors=[[guard]])
    try:
        tokens = _drain_one(generator)
        stats = dict(generator.scheduler_stats)
    finally:
        generator.close()
    assert tokens == reference
    assert stats["pld_proposed"] > 0
    assert any(_NORTH_CLOSE in block[1:] for block in blocks) is close_in_proposal
    # The receipt counts committed steered steps: every fed input before the
    # first close token (the last prompt token feeds the first step).
    fed = [prompt[-1]] + tokens[:-1]
    expected = fed.index(_NORTH_CLOSE) if _NORTH_CLOSE in fed else len(fed)
    assert guard.steered_steps == expected


def test_north_ships_the_guard_with_steering_off_and_no_direction_asset():
    from pathlib import Path

    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter

    adapter = object.__new__(NorthMiniCodeAdapter)
    # 2026-09-23: with North's layer 0 rotated as the reference does, the
    # held-out grid closed 16/16 unsteered in 2,761 reasoning tokens, and the
    # recalibrated alpha 0.2 direction took 3,971 (one prompt looped).  The
    # budget and run-on alarm stay on; steering is off by default.
    assert adapter.thinking_guard_defaults() == {
        "thinking_budget": 512, "thinking_steer_alpha": 0.0, "thinking_steer_hammer": 0.0}
    # A shipped direction must beat no steering on held-out prompts; none
    # does on the corrected body, so none ships and none may linger to bind.
    assets = adapter.commit_direction_assets()
    assert assets["paths"] == []
    import mlx2.adapters.north_mini_code as module

    asset_dir = Path(module.__file__).with_name("assets")
    assert not (asset_dir / NorthMiniCodeAdapter.COMMIT_DIRECTION_ASSET).exists()


def test_explicit_zero_budget_turns_the_guard_off_for_a_request():
    from mlx2.thinking_guard import resolve_thinking_budget

    assert resolve_thinking_budget({"thinking_budget": 0}, 512) == 0
    assert resolve_thinking_budget({}, 512) == 2048


@pytest.mark.parametrize("overrides,expected,source", [
    # The fake adapter cannot be calibrated (no residual taps), so the adapter's
    # steering default is dropped while its budget default stays.
    ({}, (512, 0.0, 0.0), "adapter"),
    ({"thinking_steer_alpha": 0.0}, (512, 0.0, 0.0), "operator"),
    ({"thinking_budget": 0}, (0, 0.0, 0.0), "operator"),
])
def test_engine_takes_adapter_defaults_unless_the_operator_sets_a_lever(monkeypatch, overrides, expected, source):
    from types import SimpleNamespace as NS

    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory

    class APC:
        def __init__(self, **_kw):
            self.apc_stats = {}

        def key(self, *_a, **_kw):
            return "key"

        def spill_idle_entries(self):
            pass

        def clear(self):
            pass

    class Batch:
        scheduler_stats = {}

        def __init__(self, *_a, **_kw):
            pass

        def next(self):
            return [], []

        def close(self):
            pass

    class Adapter:
        max_context = 1000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tokenizer = NS(vocab_size=10, eos_token_ids=[])

        def __init__(self, _path):
            pass

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, **_kw):
            return {"num_draft": 0}

        def thinking_guard_defaults(self):
            return {"thinking_budget": 512, "thinking_steer_alpha": 0.2, "thinking_steer_hammer": 0.0}

        def diagnostics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine("fake", adapter_factory=Adapter, qualification_mode=True, mtp=False,
                                   max_lanes=2, max_inflight=2, **overrides)
    try:
        assert engine.ready.wait(5), engine.error
        settings = engine.status()["settings"]
    finally:
        engine.close()
    assert (engine.thinking_budget, engine.thinking_steer_alpha, engine.thinking_steer_hammer) == expected
    assert settings["thinking_budget"] == expected[0] and settings["thinking_steer"]["alpha"] == expected[1]
    assert settings["thinking_steer"]["calibration"]["state"] == "unsupported"
    assert settings["thinking_defaults_source"] == source


def _guard_routes():
    segmented = {"segment_aware_live_tip": True, "segment_aware_cohort_size": 1}
    return {
        "ordinary": {"mtp": False},
        "native_mtp": {"mtp": True, "extra": segmented},
        "native_mtp_physical": {"mtp": True},
        "prompt_lookup": {"mtp": False, "prompt_lookup": True},
    }


def test_guard_receipt_describes_the_committed_stream_on_every_route(monkeypatch):
    # Native MTP reported the guard as the last verify row left it (a row
    # drafted past the stop token: one think token too many, a soft trip the
    # committed stream never reached); prompt lookup never showed the guard
    # the final committed token (one too few).
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(40)]
    requests = [
        {"messages": [{"role": "user", "content": "x"}], "tokens": prompt,
         "max_tokens": 30, "temperature": 0, "thinking_budget": budget}
        for budget in (8, 9)
    ]
    outputs = {}
    for route, options in _guard_routes().items():
        engine = make_engine(model, vocab, eos=(24,), close_id=99, **options)
        try:
            outputs[route] = [run(engine, request) for request in requests]
        finally:
            engine.close()
    for route, results in outputs.items():
        for result, reference in zip(results, outputs["ordinary"]):
            assert result["tokens"] == reference["tokens"], route
            guard = result["receipt"]["request_controls"]["thinking_guard"]
            expected = reference["receipt"]["request_controls"]["thinking_guard"]
            assert guard == expected, route
    first = outputs["ordinary"][0]["receipt"]["request_controls"]["thinking_guard"]
    assert first["tripped"] == "budget_soft"  # the case exercises a trip


def test_forced_close_receipt_ignores_rows_drafted_past_the_stop(monkeypatch):
    # The lane stops on EOS two tokens before the budget; only verify rows
    # drafted past EOS reach it, so no close was ever forced.
    from route_harness import (
        install_mtp_oracle, make_engine, patch_host, run, tiny_qwen38_mtp,
    )

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    eos, close = 24, 127
    request = {"messages": [{"role": "user", "content": "x"}],
               "tokens": [(25 * i + 5) % (vocab - 2) + 1 for i in range(30)],
               "max_tokens": 40, "temperature": 0, "thinking_budget": 8}
    engine = make_engine(model, vocab, mtp=False, eos=(eos,), close_id=close)
    try:
        ordinary = run(engine, request)
    finally:
        engine.close()
    assert ordinary["finish"] == "stop" and len(ordinary["tokens"]) == 5
    install_mtp_oracle(monkeypatch, {tuple(request["tokens"]): ordinary["tokens"] + [eos, 5, 6, 7]})
    engine = make_engine(model, vocab, mtp=True, eos=(eos,), close_id=close, num_draft=3)
    try:
        speculative = run(engine, request)
    finally:
        engine.close()
    assert speculative["tokens"] == ordinary["tokens"]
    guard = speculative["receipt"]["request_controls"]["thinking_guard"]
    assert guard == ordinary["receipt"]["request_controls"]["thinking_guard"]
    assert guard["forced_close"] is False and guard["think_tokens"] == 6


def test_settle_reports_a_forced_close_only_when_one_was_committed():
    # Ordinary decode's last guard call is a lookahead: it evaluates the row
    # after the final committed token, which is never sampled.  Settling on
    # that call latched forced_close for a stream that ended exactly at the
    # budget (max_tokens == thinking_budget, or a stop on that token).
    budget, reasoning = 4, [8, 9, 10, 11]
    guard = ThinkingGuard(2, (CLOSE,), budget=budget, soft_ratio=0.5)
    for length in range(budget + 1):  # every ordinary step, lookahead included
        _call(guard, reasoning[:length])
    guard.settle(reasoning)
    receipt = guard.receipt()
    assert receipt["forced_close"] is False
    assert receipt["tripped"] == "budget_soft" and receipt["released_at"] is None
    assert receipt["think_tokens"] == budget
    # The forcing row admits only the close, which lands at the budget.
    forced = reasoning + [CLOSE]
    assert np.isinf(_call(guard, reasoning)[np.arange(VOCAB) != CLOSE]).all()
    guard.settle(forced)
    receipt = guard.receipt()
    assert receipt["forced_close"] is True and receipt["released_at"] == budget
    # A route that never showed the guard some committed ids settles alike.
    fresh = ThinkingGuard(2, (CLOSE,), budget=budget, soft_ratio=0.5)
    fresh.settle(forced)
    assert fresh.receipt() == receipt
    # A close the model chose before the budget was not forced.
    guard.settle(reasoning[:3] + [CLOSE])
    assert guard.receipt()["forced_close"] is False
    assert guard.receipt()["released_at"] == 3


def test_forced_close_receipt_is_exact_and_identical_on_every_route(monkeypatch):
    # The model never closes the channel on its own.  Ending exactly at the
    # budget forces nothing; running past it forces the close at the budget.
    # Every route reported the lookahead row ordinary decode evaluates after
    # the last token (forced_close true at the budget), or before 7032610e
    # prompt lookup and native MTP reported the stream one token short.
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    close = 99
    penalty = mx.where(mx.arange(vocab) == close, -1e4, 0.0).astype(mx.float32)
    mx.eval(penalty)  # the engine thread cannot evaluate this thread's graph
    base = type(model)

    class NeverCloses(base):
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs) + penalty

        def logits(self, hidden):
            return super().logits(hidden) + penalty

    model.__class__ = NeverCloses
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(40)]
    budget = 8
    requests = [
        {"messages": [{"role": "user", "content": "x"}], "tokens": prompt,
         "max_tokens": max_tokens, "temperature": 0, "thinking_budget": budget}
        for max_tokens in (budget, budget + 3)
    ]
    outputs = {}
    for route, options in _guard_routes().items():
        engine = make_engine(model, vocab, close_id=close, **options)
        try:
            outputs[route] = [run(engine, request) for request in requests]
        finally:
            engine.close()
    for route, (at_budget, past_budget) in outputs.items():
        assert close not in at_budget["tokens"] and len(at_budget["tokens"]) == budget
        guard = at_budget["receipt"]["request_controls"]["thinking_guard"]
        assert guard["forced_close"] is False, route
        assert guard["think_tokens"] == budget and guard["released_at"] is None, route
        assert past_budget["tokens"].index(close) == budget, route
        guard = past_budget["receipt"]["request_controls"]["thinking_guard"]
        assert guard["forced_close"] is True and guard["released_at"] == budget, route
        for result, reference in zip((at_budget, past_budget), outputs["ordinary"]):
            assert result["tokens"] == reference["tokens"], route
            assert (
                result["receipt"]["request_controls"]["thinking_guard"]
                == reference["receipt"]["request_controls"]["thinking_guard"]
            ), route


def _guard_state(guard):
    import copy

    return copy.deepcopy({key: value for key, value in vars(guard).items() if key != "_direction"})


def test_probe_matches_a_copied_call_and_leaves_the_guard_untouched():
    # Draft probing deep-copied the guard (history, n-gram table, undo log)
    # for every drafted position; probe answers on the live guard and
    # rewinds it instead.
    import copy

    rng = np.random.default_rng(3)

    def make():
        return ThinkingGuard(2, (CLOSE,), budget=120, soft_ratio=0.5, tau=2.5,
                             ngram=4, rewrite_window=32)

    guard, ids = make(), []
    for _ in range(400):
        if ids and rng.random() < 0.2:
            del ids[len(ids) - int(rng.integers(1, min(len(ids), 6) + 1)):]
        ids += [int(token) for token in rng.integers(8, 12, size=int(rng.integers(1, 3)))]
        if rng.random() < 0.01:
            ids.append(CLOSE)
        _call(guard, ids)
        for depth in range(4):
            back = int(rng.integers(0, min(len(ids), 3) + 1))
            drafted = ids[: len(ids) - back] + [
                int(token) for token in rng.integers(7, 12, size=depth)
            ]
            tokens = mx.array([1, 2] + drafted, dtype=mx.uint32)
            logits = mx.array(rng.normal(size=(1, VOCAB)).astype(np.float32))
            before = _guard_state(guard)
            expected = copy.deepcopy(guard)(tokens, logits)
            np.testing.assert_array_equal(np.array(guard.probe(tokens, logits)), np.array(expected))
            after = _guard_state(guard)
            assert after.keys() == before.keys()
            for key in before:
                assert after[key] == before[key], key


def test_probing_a_long_reasoning_channel_does_not_copy_its_history():
    import tracemalloc

    from mlx2.runtime.processor_probe import probe_logits_processors

    peaks = []
    for length in (2000, 20000):
        guard = ThinkingGuard(2, (CLOSE,), budget=None)
        ids = [8 + (i * 5 + i // 9) % 8 for i in range(length)]
        _call(guard, ids)
        tokens = mx.array([1, 2] + ids + [9, 10], dtype=mx.uint32)
        logits = mx.zeros((1, VOCAB))
        tracemalloc.start()
        probe_logits_processors([guard], tokens, logits)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    # A copy of the guard's history, n-gram table and undo log at 20000
    # generated ids is several megabytes; the probe stays bounded.
    assert max(peaks) < 256 << 10, peaks


def test_recovery_snapshots_share_a_guard_that_resyncs_after_rollback():
    from types import SimpleNamespace

    from mlx2.runtime.hybrid_speculative import _snapshot_segmented_recovery_row
    from mlx2.runtime.models.cache import KVCache

    plain = ThinkingGuard(2, (CLOSE,), budget=100)
    steering = ThinkingGuard(
        2, (CLOSE,), budget=100, direction={"layer": 1, "vector": np.ones(4)}, alpha=0.2
    )
    _call(plain, [8, 9, 10])
    lane = SimpleNamespace(logits_processors=[plain, steering], generated=3)
    pair = SimpleNamespace(target=[KVCache()], draft=[KVCache()])
    fields, _target, _draft, _borrowed = _snapshot_segmented_recovery_row((lane, pair))
    shared, copied = fields["logits_processors"]
    assert shared is plain
    # Steering counts committed positions, which the ids do not determine.
    assert copied is not steering and copied.steered_steps == steering.steered_steps


def test_multi_token_forced_close_is_the_same_committed_stream_on_every_route(monkeypatch):
    # A GPT-OSS-style switch sequence: every route commits the same stream,
    # and a forced close writes the whole marker, token by token.
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    marker = (97, 98, 99)
    base = {"messages": [{"role": "user", "content": "x"}],
            "tokens": [(7 * i + 3) % (vocab - 2) + 1 for i in range(40)],
            "max_tokens": 20, "temperature": 0}
    requests = [{**base, "thinking_budget": budget} for budget in (1, 12)]
    outputs = {}
    for route, options in _guard_routes().items():
        engine = make_engine(model, vocab, close_id=marker, **options)
        try:
            outputs[route] = [run(engine, request) for request in requests]
        finally:
            engine.close()
    forced, released = outputs["ordinary"]
    guard = forced["receipt"]["request_controls"]["thinking_guard"]
    assert tuple(forced["tokens"][1:4]) == marker, forced["tokens"]
    assert guard["forced_close"] is True and guard["think_tokens"] == 1
    late = released["receipt"]["request_controls"]["thinking_guard"]
    at = late["think_tokens"]
    assert late["tripped"] and at <= 12 and tuple(released["tokens"][at:at + 3]) == marker
    for route, results in outputs.items():
        for result, reference in zip(results, outputs["ordinary"]):
            assert result["tokens"] == reference["tokens"], route
            receipt = result["receipt"]["request_controls"]["thinking_guard"]
            assert receipt == reference["receipt"]["request_controls"]["thinking_guard"], route


def test_adapter_hidden_budget_guards_a_thinking_off_request(monkeypatch):
    # A model that reasons even with thinking off (GPT-OSS Puzzle) bounds that
    # hidden reasoning through the adapter; others get no guard.
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    marker = (97, 98, 99)

    class Hidden:
        def hidden_thinking_budget(self, request):
            return 1

    request = {"messages": [{"role": "user", "content": "x"}], "enable_thinking": False,
               "tokens": [(7 * i + 3) % (vocab - 2) + 1 for i in range(40)],
               "max_tokens": 16, "temperature": 0}
    engine = make_engine(model, vocab, mtp=False, close_id=marker, adapter_mixin=Hidden)
    try:
        hidden = run(engine, request)
    finally:
        engine.close()
    guard = hidden["receipt"]["request_controls"]["thinking_guard"]
    assert guard is not None and guard["budget"] == 1 and guard["forced_close"] is True
    assert tuple(hidden["tokens"][1:4]) == marker
    # History mode defers to a visible channel; hidden reasoning is still
    # bounded, by the state-aware guard (codex review).
    engine = make_engine(model, vocab, mtp=False, close_id=marker, adapter_mixin=Hidden)
    try:
        history = run(engine, {**request, "thinking_budget_mode": "history"})
    finally:
        engine.close()
    assert history["tokens"] == hidden["tokens"]
    assert history["receipt"]["request_controls"]["thinking_guard"]["budget"] == 1
    engine = make_engine(model, vocab, mtp=False, close_id=marker)
    try:
        plain = run(engine, request)
    finally:
        engine.close()
    assert plain["receipt"]["request_controls"]["thinking_guard"] is None


NUDGE = (12, 13, 14)


def test_soft_landing_nudge_is_written_at_the_soft_budget_then_the_model_may_close():
    guard = ThinkingGuard(2, SWITCH, budget=20, soft_ratio=0.5, nudge_ids=NUDGE)
    ids = [8, 9, 10, 11, 15] * 2  # 10 = the soft budget
    for expected in NUDGE:
        assert _masked_to(_call(guard, ids)) == expected
        ids.append(expected)
    row = _call(guard, ids)
    assert _masked_to(row) is None and row[SWITCH[0]] > 0  # back to the ramp
    assert guard.receipt()["nudged"] is True and guard.receipt()["forced_close"] is False
    # The model closes on its own after the nudge: no forced close.
    closed = ids + list(SWITCH) + [9]
    _call(guard, closed)
    assert guard.receipt()["released_at"] == 13 and guard.receipt()["nudged"] is True
    settled = ThinkingGuard(2, SWITCH, budget=20, soft_ratio=0.5, nudge_ids=NUDGE)
    settled.settle(closed)
    assert settled.receipt()["nudged"] is True and settled.receipt()["forced_close"] is False


def test_nudge_is_skipped_when_it_and_the_marker_do_not_fit_the_budget():
    guard = ThinkingGuard(2, SWITCH, budget=12, soft_ratio=0.5, nudge_ids=NUDGE)  # 6+3+4 > 12
    ids = [8, 9, 10, 11, 15, 8]
    assert _masked_to(_call(guard, ids)) is None
    assert guard.receipt()["nudged"] is False


def test_nudge_after_a_natural_close_or_on_a_stale_branch_is_not_written():
    guard = ThinkingGuard(2, SWITCH, budget=20, soft_ratio=0.5, nudge_ids=NUDGE)
    closed = [8, 9] + list(SWITCH) + [10, 11, 15, 8]
    assert not _call(guard, closed).any()
    fresh = ThinkingGuard(2, SWITCH, budget=20, soft_ratio=0.5, nudge_ids=NUDGE)
    stale = [8, 9, 10, 11, 15, 8, 9, 10, 11, 15] + [NUDGE[0], 9]  # diverged mid-nudge
    assert _masked_to(_call(fresh, stale)) is None


def test_nudge_guard_matches_a_fresh_guard_under_random_rollbacks():
    rng = np.random.default_rng(2)

    def make():
        return ThinkingGuard(2, (3, 4, 5), budget=60, soft_ratio=0.5, tau=2.5, ngram=4,
                             rewrite_window=32, nudge_ids=(6, 7, 6, 8))
    live, ids = make(), []
    for _ in range(600):
        if ids and rng.random() < 0.15:
            del ids[len(ids) - int(rng.integers(1, min(len(ids), 10) + 1)):]
        row = _call(live, ids)
        fresh = make()
        np.testing.assert_array_equal(row, _call(fresh, ids))
        assert live.receipt()["nudged"] == fresh.receipt()["nudged"]
        if np.isinf(row).any() and rng.random() < 0.5:
            ids.append(int(np.flatnonzero(np.isfinite(row))[0]))
        else:
            ids += [int(token) for token in rng.integers(3, 12, size=int(rng.integers(1, 3)))]


# Self-addressed reasoning, Muse-shaped: <|eom|><|start|>assistant separates
# messages, " to=self<|message|>" addresses one to the model itself and the
# release switch opens the user's answer.  " to=f<|message|>" is a tool call.
SEP, SELF, USER, TOOL = (3, 4), (5, 6, 9), (5, 7, 9), (5, 8, 9)
RELEASE = SEP + USER
MESSAGE = (SEP, SELF)


def test_self_addressed_reasoning_closes_at_any_other_message():
    from mlx2.thinking_guard import reasoning_closed_at

    def closed(ids):
        return reasoning_closed_at(list(ids), SEP, SELF, release=RELEASE)

    assert closed(SELF + (10, 11) + SEP + SELF + (12,)) is None  # reasoning only
    assert closed(SELF[:2]) is None  # an undecided first header
    assert closed(SELF + (10,) + SEP + TOOL + (12,)) == (4, 7)  # a tool call
    assert closed(TOOL + (12,)) == (0, 1)  # a call without reasoning
    assert closed(USER + (12,)) == (0, 1)  # a direct answer
    # The release closes once complete, so a forced one is finished.
    assert closed(SELF + (10,) + RELEASE[:4]) is None
    assert closed(SELF + (10,) + RELEASE) == (4, 8)
    assert closed(SELF + (10,) + RELEASE[:3] + (8,)) == (4, 7)
    # ``start`` only skips decisions the caller already examined.
    ids = list(SELF + (10, 11) + SEP + SELF + (12,) * 20 + SEP + TOOL)
    assert reasoning_closed_at(ids, SEP, SELF, release=RELEASE, start=len(ids) - 2) == (30, 33)


def test_guard_stops_at_a_tool_call_and_never_forces_the_release_into_it():
    guard = ThinkingGuard(2, RELEASE, budget=12, soft_ratio=0.5, reasoning_message=MESSAGE)
    ids = list(SELF + (10, 11) + SEP + TOOL)
    for _ in range(20):
        assert not _call(guard, ids).any()  # out of the way past the budget
        ids.append(12)
    receipt = guard.receipt()
    assert receipt["released_at"] == 5 and receipt["think_tokens"] == 5
    assert receipt["forced_close"] is False
    assert guard.dormant(np.array([1, 2] + ids))
    guard.settle(ids)
    assert guard.receipt()["forced_close"] is False


def test_guard_still_forces_the_whole_release_after_run_on_reasoning():
    guard = ThinkingGuard(2, RELEASE, budget=8, soft_ratio=0.5, reasoning_message=MESSAGE)
    ids = list(SELF) + [10, 11, 12, 13, 10]
    for expected in RELEASE:
        assert _masked_to(_call(guard, ids)) == expected
        ids.append(expected)
    assert not _call(guard, ids).any()
    assert guard.receipt()["released_at"] == 8 and guard.receipt()["forced_close"] is True
    settled = ThinkingGuard(2, RELEASE, budget=8, soft_ratio=0.5, reasoning_message=MESSAGE)
    settled.settle(ids + [12])
    assert settled.receipt()["forced_close"] is True


@pytest.mark.parametrize(("budget", "seed"), [(40, 4), (1, 5), (7, 6), (12, 7)])
def test_self_addressed_guard_matches_a_fresh_guard_under_random_rollbacks(budget, seed):
    import copy

    rng = np.random.default_rng(seed)
    chunks = [SEP, SELF, USER, TOOL, RELEASE, RELEASE[:3], (10,), (11, 12), (13,), (9,),
              SELF[:1], SELF[:2]]

    def make():
        return ThinkingGuard(2, RELEASE, budget=budget, soft_ratio=0.5, tau=2.5, ngram=4,
                             rewrite_window=32, reasoning_message=MESSAGE)
    live, ids = make(), list(SELF)
    for step in range(800):
        if rng.random() < 0.02:
            live, ids = make(), list(SELF)
        if len(ids) > 3 and rng.random() < 0.2:
            del ids[len(ids) - int(rng.integers(1, min(len(ids) - 3, 10) + 1)):]
        row = _call(live, ids)
        fresh = make()
        np.testing.assert_array_equal(row, _call(fresh, ids))
        for key in ("released_at", "think_tokens"):
            assert live.receipt()[key] == fresh.receipt()[key], key
        tokens = np.array([1, 2] + ids)
        assert live.dormant(tokens) == (fresh.receipt()["released_at"] is not None)
        # forced_close latches on the forcing rows a guard evaluated; a fresh
        # guard saw none of them, so both answer from the ids via settle.
        settled = copy.deepcopy(live)
        settled.settle(ids)
        fresh.settle(ids)
        assert settled.receipt()["forced_close"] == fresh.receipt()["forced_close"]
        if step % 7 == 0:
            drafted = mx.array([1, 2] + ids + [10, 3], dtype=mx.uint32)
            logits = mx.zeros((1, VOCAB))
            before = _guard_state(live)
            expected = copy.deepcopy(live)(drafted, logits)
            np.testing.assert_array_equal(np.array(live.probe(drafted, logits)), np.array(expected))
            assert _guard_state(live) == before
        if np.isinf(row).any() and rng.random() < 0.7:
            ids.append(_masked_to(row))
        else:
            ids += list(chunks[int(rng.integers(len(chunks)))])


def test_guard_leaves_a_budget_1_recipient_choice_to_the_model():
    # Review r1: at budget 1 only the shared " to" (5) is written.  It opens
    # a reasoning message, an answer and a tool call alike; forcing the
    # release there broke the header of a direct answer or tool call.
    for recipient in (USER, TOOL):
        guard = ThinkingGuard(2, RELEASE, budget=1, reasoning_message=MESSAGE)
        ids = list(recipient[:1])
        for token in recipient[1:] + (12, 12):
            assert not _call(guard, ids).any()  # the model chooses
            ids.append(token)
        guard.settle(ids)
        assert guard.receipt()["released_at"] == 0
        assert guard.receipt()["forced_close"] is False
    # A message to itself completes, then the whole release is forced.
    guard = ThinkingGuard(2, RELEASE, budget=1, reasoning_message=MESSAGE)
    ids = list(SELF[:1])
    for token in SELF[1:]:
        assert not _call(guard, ids).any()
        ids.append(token)
    for expected in RELEASE:
        assert _masked_to(_call(guard, ids)) == expected
        ids.append(expected)
    assert not _call(guard, ids).any()
    assert guard.receipt()["released_at"] == 3 and guard.receipt()["forced_close"] is True
    settled = ThinkingGuard(2, RELEASE, budget=1, reasoning_message=MESSAGE)
    settled.settle(ids + [12])
    assert settled.receipt()["forced_close"] is True


def test_guard_defers_to_a_header_the_model_opened_by_the_budget():
    # Natural separator, then the budget falls on the next recipient choice.
    reasoning = list(SELF) + [10]
    for budget in (6, 7):  # the empty header, or after the shared " to"
        guard = ThinkingGuard(2, RELEASE, budget=budget, soft_ratio=0.5, reasoning_message=MESSAGE)
        ids = reasoning + list(SEP) + list(TOOL[:budget - 6])
        for token in TOOL[budget - 6:] + (12,):
            assert not _call(guard, ids).any()
            ids.append(token)
        assert guard.receipt()["released_at"] == 4 and guard.receipt()["forced_close"] is False
        # The model writing the answer switch itself is not a forced close.
        answer = reasoning + list(RELEASE) + [12]
        natural = ThinkingGuard(2, RELEASE, budget=budget, soft_ratio=0.5, reasoning_message=MESSAGE)
        for length in range(budget, len(answer)):
            assert not _call(natural, answer[:length]).any()
        natural.settle(answer)
        assert natural.receipt()["forced_close"] is False
    # A switch the budget began (mid-separator) is still finished.
    guard = ThinkingGuard(2, RELEASE, budget=5, soft_ratio=0.5, reasoning_message=MESSAGE)
    ids = reasoning + [SEP[0]]
    for expected in RELEASE[1:]:
        assert _masked_to(_call(guard, ids)) == expected
        ids.append(expected)
    guard.settle(ids)
    assert guard.receipt()["forced_close"] is True


def test_release_ramp_stays_out_of_every_header_the_model_writes():
    guard = ThinkingGuard(2, RELEASE, budget=40, soft_ratio=0.25, reasoning_message=MESSAGE)
    reasoning = list(SELF) + [10, 11, 12, 13] * 3
    assert _call(guard, reasoning)[RELEASE[0]] > 0  # tripped, ramping
    assert _call(guard, reasoning + [SEP[0]])[SEP[1]] > 0  # the separator too
    # The recipient is the model's: " to" also opens a tool call, and after
    # " to=self" the ramp's next marker id (<|eom|>) would break the header.
    for header in ((), (5,), (5, 6), (5, 7)):
        assert not _call(guard, reasoning + list(SEP) + list(header)).any(), header
    assert _call(guard, reasoning + list(SEP) + list(SELF))[RELEASE[0]] > 0
    # Nor inside the reply's first header.
    early = ThinkingGuard(2, RELEASE, budget=2, soft_ratio=0.5, reasoning_message=MESSAGE)
    assert not _call(early, [5]).any()


def test_budget_never_completes_the_answer_header_over_a_tool_sharing_it():
    # A tool named like "user_x" opens " to" "=user" "_x": the release's
    # recipient ids are a prefix of its header.
    shared = (5, 7, 14, 9)
    for start in ([], list(SELF) + [10] + list(SEP)):
        budget = len(start) + 2  # right after " to=user"
        guard = ThinkingGuard(2, RELEASE, budget=budget, soft_ratio=0.5, reasoning_message=MESSAGE)
        ids = start + list(shared[:2])
        for token in shared[2:] + (12,):
            assert not _call(guard, ids).any()
            ids.append(token)
        guard.settle(ids)
        assert guard.receipt()["forced_close"] is False


def _receipt_keys(guard):
    return {
        key: guard.receipt().get(key)
        for key in ("forced_close", "released_at", "tripped", "tripped_at", "think_tokens")
    }


def test_receipt_is_the_same_whether_the_close_was_observed_stepwise_or_at_once():
    """P5: the receipt is a function of the ids.  Observing the close marker
    token by token advanced the alarm over the marker's own leading tokens
    (tripping there) and never undid it once the marker completed, so a
    stepwise guard and a fresh guard fed the same ids disagreed on
    ``tripped`` / ``tripped_at`` / ``forced_close`` while their logits agreed."""
    close = (3, 4, 5)

    def guard(budget):
        return ThinkingGuard(2, close, budget=budget, soft_ratio=0.5, tau=1.0, ngram=3, rewrite_window=4)

    # The soft budget (12) lands on the marker's second token.
    before = [8, 7, 10, 9, 7, 7, 7, 9, 4, 4, 3, 4]
    final = before + [5]
    live = guard(24)
    _call(live, before)
    assert live.receipt()["tripped"] == "budget_soft"  # the step at 12 saw no close yet
    live_row = _call(live, final)
    fresh = guard(24)
    fresh_row = _call(fresh, final)
    np.testing.assert_array_equal(live_row, fresh_row)
    assert _receipt_keys(live) == _receipt_keys(fresh)
    assert live.receipt()["tripped"] is None and live.receipt()["released_at"] == 10
    # A close forced at the budget reads as forced on both paths.
    final = [8, 3, 4, 5, 7, 7, 8]
    live = guard(2)
    for length in (2, 3, 4, 7):
        _call(live, final[:length])
    fresh = guard(2)
    _call(fresh, final)
    assert _receipt_keys(live) == _receipt_keys(fresh)
    assert live.receipt()["forced_close"] is True and live.receipt()["released_at"] == 1


def test_receipt_purity_under_rollback_fuzz():
    """The lane's differential fuzz (seed 2): stepwise observation with
    rollbacks against one fresh guard on the final ids, 148/400 receipts
    differed before the fix with identical logits."""
    import random

    close, alphabet = (3, 4, 5), [3, 4, 5, 6, 7, 8, 9, 10]

    def sequence(rng, n):
        out = []
        while len(out) < n:
            r = rng.random()
            if r < 0.15:
                out += list(close)
            elif r < 0.4:
                out += [rng.choice([7, 8])] * rng.randrange(1, 4)
            else:
                out.append(rng.choice(alphabet))
        return out[:n]

    rng = random.Random(2)
    mismatches = []
    for trial in range(400):
        final = sequence(rng, rng.randrange(3, 40))
        budget, window = rng.randrange(2, 30), rng.choice([4, 8, 256])

        def make():
            return ThinkingGuard(
                2, close, budget=budget, soft_ratio=0.5, tau=1.0, ngram=3, rewrite_window=window
            )

        live, seen = make(), 0
        while seen < len(final):
            if seen > 0 and rng.random() < 0.4:
                seen -= rng.randrange(1, min(window, seen) + 1)
                _call(live, final[:seen] + sequence(rng, rng.randrange(1, 4)))
            seen = min(len(final), seen + rng.randrange(1, 4))
            _call(live, final[:seen])
        live_row = _call(live, final)
        fresh = make()
        fresh_row = _call(fresh, final)
        np.testing.assert_array_equal(live_row, fresh_row)
        if _receipt_keys(live) != _receipt_keys(fresh):
            mismatches.append((trial, budget, window, final, _receipt_keys(live), _receipt_keys(fresh)))
    assert not mismatches, mismatches[:3]


def test_probe_restores_the_alarm_when_a_draft_completes_a_marker_before_the_sync_window():
    """A probed draft row that completes a close marker whose start lies
    before the sync window truncated the alarm below the window; the restore
    then replayed only the window's ids, leaving a gap in the alarm input."""
    close = (3, 4, 5)
    guard = ThinkingGuard(2, close, budget=None, tau=1.0, ngram=2, rewrite_window=1)
    committed = [8, 9, 8, 9, 3, 4]  # a marker prefix the model wrote itself
    _call(guard, committed)
    probe = mx.array([1, 2] + committed + [5], dtype=mx.uint32)
    guard.probe(probe, mx.zeros((1, VOCAB)))
    fresh = ThinkingGuard(2, close, budget=None, tau=1.0, ngram=2, rewrite_window=1)
    _call(fresh, committed)
    assert guard._ids == fresh._ids == committed
    assert guard.receipt() == fresh.receipt()
    np.testing.assert_array_equal(_call(guard, committed + [8]), _call(fresh, committed + [8]))
    assert guard.receipt() == fresh.receipt()


def test_forced_close_requires_the_marker_prefix_after_the_forcing_step():
    """A forcing row admits only the next marker token, so ids past the first
    forcing step that are not the marker's continuation were not written by
    the guard (a processor bypass, a vocabulary past the marker id): the
    receipt fails closed instead of reporting a forced close it never made."""
    guard = ThinkingGuard(2, (3, 4, 5), budget=2, soft_ratio=0.5)
    guard.settle([8, 9, 10])
    assert guard.receipt()["forced_close"] is False and guard.receipt()["released_at"] is None
    guard.settle([8, 9, 3, 4])  # a genuine forced marker cut short by max_tokens
    assert guard.receipt()["forced_close"] is True and guard.receipt()["released_at"] is None
    guard.settle([8, 9, 3, 9])
    assert guard.receipt()["forced_close"] is False
    guard.settle([8, 9, 3, 4, 5, 7])
    assert guard.receipt()["forced_close"] is True and guard.receipt()["released_at"] == 2
    # A later complete marker does not make ids the mask would have refused forced.
    guard.settle([8, 9, 10, 3, 4, 5])
    assert guard.receipt()["forced_close"] is False and guard.receipt()["released_at"] == 3
    guard.settle([8, 9, 10, 11, 3, 4, 5, 7])
    assert guard.receipt()["forced_close"] is False and guard.receipt()["released_at"] == 4
    # The model wrote a marker prefix before the budget; forcing continues it.
    guard.settle([8, 3, 4])
    assert guard.receipt()["forced_close"] is True
    guard.settle([8, 3, 8])
    assert guard.receipt()["forced_close"] is False


def test_a_natural_message_boundary_close_is_never_reported_forced():
    """Self-addressed reasoning that ended in a message to the user or a tool
    closed naturally; a release marker the model spells later (inside that
    message) is not a forced close, so the first close must be the marker
    itself for the receipt to say forced."""
    separator, self_header, user_header = (3, 4), (5, 6, 9), (5, 7, 9)
    release = separator + user_header

    def guard():
        return ThinkingGuard(
            2, release, budget=3, soft_ratio=0.5, reasoning_message=(separator, self_header)
        )

    for first_message in (user_header, (5, 8, 9)):  # to the user, to a tool
        settled = guard()
        settled.settle(list(first_message + release))
        receipt = settled.receipt()
        assert receipt["released_at"] == 0 and receipt["think_tokens"] == 0
        assert receipt["forced_close"] is False, first_message
    # Reasoning to itself that the budget released (the marker from the
    # forcing step on) reads as forced.
    settled = guard()
    settled.settle(list(self_header) + list(release))
    assert settled.receipt()["forced_close"] is True and settled.receipt()["released_at"] == 3
