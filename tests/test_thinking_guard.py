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
        ThinkingGuard(2, (1, 2), budget=10)


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


def test_north_ships_guard_and_steering_defaults_and_an_identity_bound_asset():
    import json
    from pathlib import Path

    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
    from mlx2.thinking_calibration import SCHEMA

    adapter = object.__new__(NorthMiniCodeAdapter)
    assert adapter.thinking_guard_defaults() == {
        "thinking_budget": 512, "thinking_steer_alpha": 0.2, "thinking_steer_hammer": 0.0}
    assets = adapter.commit_direction_assets()
    (npz,) = assets["paths"]
    meta = json.loads(Path(npz).with_suffix(".json").read_text())
    assert assets["layer"] == meta["layer"] == 32
    assert len(meta["artifact_identity"]) == 64 and meta["artifact"].endswith("4bit")
    # 2026-09-20: re-shipped after the normalization correction.  The previous
    # asset was calibrated on 2026-09-18 against a North body that ran
    # mean-centred LayerNorm where the reference uses RMSNorm; a commit
    # direction is a property of that residual geometry, so it was invalid
    # under the corrected norm.  The artifact-file identity hash cannot see a
    # code change and would still have matched, so SCHEMA is what failed it
    # closed -- which is why this asset must now declare the CURRENT schema and
    # not a pinned literal: the next body change bumps SCHEMA and this asset
    # must go stale with it rather than keep binding.
    assert meta["schema"] == SCHEMA
    # The replacement is a gated calibration, not just a fresh one: it beat no
    # steering on the held-out set and a same-norm random direction did not.
    assert meta["gate_failures"] == []
    assert meta["validation"]["calibrated"]["think_tokens"] < meta["validation"]["off"]["think_tokens"]
    assert meta["validation"]["calibrated"]["think_tokens"] < meta["validation"]["random"]["think_tokens"]
    assert meta["validation"]["calibrated"]["correct"] >= meta["validation"]["off"]["correct"]
    assert meta["supersedes"]["schema"] == "mlx2.commit-direction.v1"


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
