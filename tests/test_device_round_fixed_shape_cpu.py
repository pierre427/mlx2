"""device_round slice 1a-5: fixed-shape rounds (``fixed_shape``), CPU only.

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md sections 3.5 and 4.7;
Metal replay P6b (docs/research/2026-11-26-metal-replay-feasibility-probes.md) counted 17
capture layouts in 24 Raon rounds because the verify width and the per-row decision chain
followed the draft count. Pins, on the CPU device:
  * every runner round verifies ``num_draft + 1`` rows;
  * greedy tokens and the lane RNG equal the variable-width device round's, plain, with a
    presence penalty (windowed and whole-history), and through a composed drafter;
  * sampled lanes and other processor kinds keep the variable-width round, counted;
  * ``fixed_shape`` needs lazy_draft + decide and is refused with device_commit.
"""

import mlx.core as mx
import pytest
from test_device_round_dflash2_cpu import (PROMPT, _decode, _decode_composed, _decode_processed,
                                           _presence)
from test_external_dflash2_cpu import generator, tiny

from mlx2.runtime.speculative_device_round import FIXED_SHAPE_SKIPS, DeviceRoundPolicy

mx.set_default_device(mx.cpu)

BOTH = {"lazy_draft": True, "decide": True}
FIXED = {**BOTH, "fixed_shape": True}
PROC = {**BOTH, "processors": True}
PROC_FIXED = {**PROC, "fixed_shape": True}


def _widths(monkeypatch, steps=16, policy=FIXED, sampling=None):
    """Verify widths of every target forward the runner issues."""
    from mlx2.runtime import external_speculative as es

    seen = []
    original = es.ExternalDraftBatchGenerator._cohort_target_forward

    def spy(self, inputs, *args, **kwargs):
        if kwargs.get("site") == "standard_verify":
            seen.append(int(inputs.shape[1]))
        return original(self, inputs, *args, **kwargs)

    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_cohort_target_forward", spy)
    model, draft = tiny()
    batch = generator(model, draft, device_round=policy)
    try:
        batch.insert([list(PROMPT)], max_tokens=[steps],
                     sampling_configs=[sampling or {"sampling_temp": 0}])
        for _ in range(600):
            batch.next()
            if not batch.lanes:
                break
        return seen, int(batch.num_draft), dict(batch.scheduler_stats)
    finally:
        batch.close()


def test_policy_needs_lazy_draft_and_decide():
    with pytest.raises(ValueError, match="fixed_shape requires lazy_draft and decide"):
        DeviceRoundPolicy.from_value({"lazy_draft": True, "fixed_shape": True})
    # Since the commit-side extension, fixed_shape combines with device_commit and the pipeline
    # (tests/test_device_round_fixed_shape_commit_cpu.py).
    assert DeviceRoundPolicy.from_value({**FIXED, "device_commit": True, "lagged_width": True,
                                         "pipeline": True}).fixed_shape


def test_greedy_tokens_and_rng_equal_the_variable_width_round():
    var, var_stats = _decode(steps=16, device_round=BOTH)
    fix, fix_stats = _decode(steps=16, device_round=FIXED)
    assert fix == var
    assert list(fix_stats["_rng"].values()) == list(var_stats["_rng"].values())
    assert fix_stats["external_device_round_fixed_shape_rounds"] > 0
    assert fix_stats["external_device_round_fixed_shape_rounds"] == fix_stats["external_device_round_rounds"]
    assert fix_stats["external_device_round_host_reads"] == fix_stats["external_device_round_rounds"]


def test_every_runner_round_verifies_the_menu_width(monkeypatch):
    seen, menu, stats = _widths(monkeypatch)
    runner = stats["external_device_round_rounds"]
    assert runner > 0
    # Every runner round at the menu width; the only other forward is the final
    # zero-draft round, which the chain body serves outside the runner.
    assert seen.count(menu + 1) == runner
    assert set(seen) - {menu + 1} <= {1}
    var, _, var_stats = _widths(monkeypatch, policy=BOTH)
    assert len(set(var) - {1}) > 1, "the variable-width round should vary here"


@pytest.mark.parametrize("context", [20, 3, 0])
def test_presence_penalty_lane_equals_the_variable_width_round(context):
    def make(prompt, context=context):
        from mlx2.runtime.sample_utils import make_logits_processors

        return make_logits_processors(presence_penalty=1.5, presence_context_size=context,
                                      penalty_generation_start=len(prompt))

    var, var_rng, _ = _decode_processed(steps=18, make=make, device_round=PROC)
    fix, fix_rng, stats = _decode_processed(steps=18, make=make, device_round=PROC_FIXED)
    assert fix == var and fix_rng == var_rng
    assert stats["external_device_round_fixed_shape_rounds"] > 0
    assert stats["external_device_round_declines"] == 0


def test_composed_drafter_equals_the_variable_width_round():
    var, var_state, _ = _decode_composed(device_round=BOTH)
    fix, fix_state, stats = _decode_composed(device_round=FIXED)
    assert fix == var and fix_state == var_state
    assert stats["external_device_round_fixed_shape_rounds"] > 0


def test_option_off_engine_is_unchanged_by_the_switch_existing():
    off, _ = _decode(steps=16)
    fix, _ = _decode(steps=16, device_round=FIXED)
    assert fix == off


def test_stats_install_only_with_the_switch():
    _, stats = _decode(steps=6, device_round=BOTH)
    assert "external_device_round_fixed_shape_rounds" not in stats
    assert FIXED_SHAPE_SKIPS not in stats


def test_presence_helper_is_the_public_processor():
    # The engine-built processor declares the window the fixed plan reads.
    processor = _presence(PROMPT)[0]
    assert processor.presence_window == (1.5, 20, len(PROMPT))


# -- future-proofing: the fixed-shape path is pinned to the variable-width definitions ----------
#
# ``_fixed_values`` re-implements, in one [rows, vocab] shape, what ``_row_value`` computes per
# row, and ``_lane_plan_fixed`` re-implements ``_lane_plan``'s greedy decision. These tests pin
# them to each other on randomized lanes, so a change to either side (a new processor kind, a
# device-held history, a different presence window rule) fails here instead of drifting.

import random
from types import SimpleNamespace

import numpy as np

from mlx2.runtime.speculative_device_round import DeviceRound

VOCAB, MENU = 48, 5


def _owner():
    return SimpleNamespace(mx=mx, num_draft=MENU,
                           scheduler_stats={FIXED_SHAPE_SKIPS: {}})


def _runtime(policy=FIXED):
    runtime = DeviceRound.__new__(DeviceRound)
    runtime.policy = DeviceRoundPolicy.from_value(policy)
    runtime.copy = None
    return runtime


def _presence_processors(rng, prompt_len, stacked):
    from mlx2.runtime.sample_utils import make_logits_processors

    out = []
    for _ in range(2 if stacked else 1):
        out += make_logits_processors(
            presence_penalty=rng.choice([0.5, 1.5, 2.25]),
            presence_context_size=rng.choice([0, 1, 3, 7, 20, 100]),
            penalty_generation_start=rng.choice([0, prompt_len, prompt_len + 3]))
    return out


def _random_row(rng, *, stacked=False, emit=True):
    prompt_len = rng.randint(1, 8)
    generated = rng.randint(0, 25)
    # Small token range: duplicates inside the history and between history and drafts.
    history = [rng.randrange(0, 12) for _ in range(prompt_len + generated)]
    count = rng.randint(0, MENU)
    drafts = [rng.randrange(0, 12) for _ in range(count)]
    lane = SimpleNamespace(history=history, anchor=rng.randrange(0, 12),
                           processors=_presence_processors(rng, prompt_len, stacked),
                           sampling={"sampling_temp": 0, "emit_logprobs": emit})
    row = SimpleNamespace(lane=lane, count=count, drafts=mx.array(drafts, dtype=mx.int32),
                          host=list(drafts))
    logits = mx.random.normal((MENU + 1, VOCAB), key=mx.random.key(rng.randrange(1 << 30)))
    return row, logits.astype(rng.choice([mx.float32, mx.bfloat16]))


@pytest.mark.parametrize("seed", range(40))
def test_fixed_values_equal_row_value_on_every_kept_row(seed):
    rng = random.Random(seed)
    row, logits = _random_row(rng, stacked=seed % 3 == 0)
    owner = _owner()
    fixed = DeviceRound._fixed_values(owner, row, logits)
    assert fixed.shape == (MENU + 1, VOCAB)
    for j in range(row.count + 1):
        want = DeviceRound._row_value(owner, row, logits, j)[0]
        assert mx.array_equal(fixed[j], want).item(), (seed, j)


@pytest.mark.parametrize("seed", range(40))
def test_fixed_plan_decides_like_the_per_row_plan(seed):
    rng = random.Random(1000 + seed)
    row, logits = _random_row(rng, stacked=seed % 4 == 0)
    # Make the drafts agree with the target for a random prefix, so acceptance varies.
    owner, runtime = _owner(), _runtime()
    values = DeviceRound._fixed_values(owner, row, logits)
    greedy = [int(v) for v in mx.argmax(values, axis=-1).tolist()]
    keep = rng.randint(0, row.count)
    drafts = greedy[:keep] + [(t + 1) % 12 for t in greedy[keep:row.count]]
    row.drafts, row.host = mx.array(drafts, dtype=mx.int32), drafts
    var = runtime._lane_plan(owner, row, logits[: row.count + 1])
    fix = runtime._lane_plan_fixed(owner, row, logits)
    mx.eval(var.packet, fix.packet)
    v, f = np.asarray(var.packet).tolist(), np.asarray(fix.packet).tolist()
    assert f[:3] == v[:3]                               # accepted, finite, last
    assert f[3: 3 + row.count] == v[3: 3 + row.count]   # the real drafts
    assert len(f) == 3 + MENU                           # one packet shape for every count
    assert len(fix.response) == len(var.response) == row.count + 1
    for a, b in zip(fix.response, var.response, strict=True):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-6)


def test_fixed_width_counts_each_reason_to_keep_the_variable_round():
    from mlx2.runtime.sample_utils import make_logits_processors

    owner, runtime = _owner(), _runtime()
    rng = random.Random(7)

    def rows(**lane_changes):
        row, _ = _random_row(rng)
        for key, value in lane_changes.items():
            setattr(row.lane, key, value)
        return [row]

    assert runtime._fixed_width(owner, [object()], rows()) == MENU + 1
    assert runtime._fixed_width(owner, [object()], rows(sampling={"sampling_temp": 0.7})) is None
    # Real processors without a presence_window: minimum tokens (branch-safe, so the device
    # round itself admits it) and the frequency penalty.
    from mlx2.serving import minimum_tokens_processor

    minimum = minimum_tokens_processor(mx, [2], 4, 3)
    frequency = make_logits_processors(frequency_penalty=0.5, penalty_generation_start=4)
    assert runtime._fixed_width(owner, [object()], rows(processors=[minimum])) is None
    assert runtime._fixed_width(owner, [object()], rows(processors=frequency)) is None
    wide, _ = _random_row(rng)
    wide.count = MENU + 2
    assert runtime._fixed_width(owner, [object()], [wide]) is None
    skips = owner.scheduler_stats[FIXED_SHAPE_SKIPS]
    assert skips == {"sampled_lane": 1, "processor_kind": 2, "wider_than_menu": 1}


def test_every_switch_is_either_compatible_or_refused():
    # A new device_round switch must be classified here: if it parses beside fixed_shape the
    # fixed path must have been taught about it (or a refusal added to from_value).
    from mlx2.runtime.speculative_device_round import _SWITCHES

    # Compatible: the fixed path (and, for the commit switches, the pipeline seam in
    # speculative_device_pipeline.dispatch/book/serve) handles them. Refused: none today.
    compatible = {"lazy_draft", "decide", "fixed_shape", "processors", "submit_drafts",
                  "sampled_drafts", "nucleus", "device_commit", "lagged_width", "pipeline",
                  "lagged_composition"}
    refused = set()
    unknown = set(_SWITCHES) - compatible - refused
    assert not unknown, f"classify new device_round switches for fixed_shape: {sorted(unknown)}"


# --- Drafter padding (2026-11-27). The 1a-5c GPU check (qualification/runs/specround-1a5c-
# 20261127/) found the remaining P6b variation was the drafter's block width on budget-limited
# rounds (depth min(num_draft, budget) < menu). Under fixed_shape those rounds now draft a full
# menu block (``pad_to``); the proposals stay the first ``depth`` tokens.


def _drafter_widths(monkeypatch, steps, policy):
    """Block widths of every lazy draft (anchor + mask rows) the runner asks the drafter for."""
    from mlx2.runtime.drafters import dflash2

    seen = []
    original = dflash2._draft_lazy

    def spy(self, anchor, hidden, cache, proposal_length, *args, pad_to=None, **kwargs):
        out = original(self, anchor, hidden, cache, proposal_length, *args, pad_to=pad_to,
                       **kwargs)
        seen.append(int(out.shape[0]) + 1)
        return out

    monkeypatch.setattr(dflash2.DFlash2DraftModel, "draft_lazy", spy)
    tokens, stats = _decode(steps=steps, device_round=policy)
    return seen, tokens, stats


@pytest.mark.parametrize("steps", [7, 9, 12, 16])
def test_budget_limited_rounds_draft_a_full_menu_block(monkeypatch, steps):
    fixed, fix_tokens, stats = _drafter_widths(monkeypatch, steps, FIXED)
    var, var_tokens, var_stats = _drafter_widths(monkeypatch, steps, BOTH)
    model, draft = tiny()
    batch = generator(model, draft)
    menu = int(batch.num_draft)
    batch.close()
    assert fixed and set(fixed) == {menu + 1}
    assert fix_tokens == var_tokens
    assert list(stats["_rng"].values()) == list(var_stats["_rng"].values())
    short = sum(width < menu + 1 for width in var)
    assert short == stats["external_device_round_fixed_shape_padded_drafts"]
    assert "external_device_round_fixed_shape_padded_drafts" not in var_stats
    # Every padded draft was verified at the menu width (no skip took it).
    assert stats["external_device_round_fixed_shape_rounds"] == stats["external_device_round_rounds"]
    assert stats[FIXED_SHAPE_SKIPS] == {}


@pytest.mark.parametrize("steps", [7, 12])
def test_padded_fixed_round_equals_the_unpadded_fixed_round(monkeypatch, steps):
    from mlx2.runtime.speculative_device_round import DeviceRound

    padded, padded_stats = _decode(steps=steps, device_round=FIXED)
    monkeypatch.setattr(DeviceRound, "_pad_depth", lambda self, owner, lane, depth: None)
    plain, plain_stats = _decode(steps=steps, device_round=FIXED)
    assert padded == plain
    assert list(padded_stats["_rng"].values()) == list(plain_stats["_rng"].values())
    assert padded_stats["external_device_round_fixed_shape_padded_drafts"] > 0
    assert plain_stats["external_device_round_fixed_shape_padded_drafts"] == 0


def test_padded_rounds_with_presence_and_composition_equal_the_variable_round():
    for steps in (7, 11):
        var, var_rng, _ = _decode_processed(steps=steps, make=_presence_make, device_round=PROC)
        fix, fix_rng, stats = _decode_processed(steps=steps, make=_presence_make,
                                                device_round=PROC_FIXED)
        assert fix == var and fix_rng == var_rng
        assert stats["external_device_round_fixed_shape_padded_drafts"] > 0
        var, var_state, _ = _decode_composed(steps=steps, device_round=BOTH)
        fix, fix_state, stats = _decode_composed(steps=steps, device_round=FIXED)
        assert fix == var and fix_state == var_state
        assert stats["external_device_round_fixed_shape_padded_drafts"] > 0


def _presence_make(prompt):
    from mlx2.runtime.sample_utils import make_logits_processors

    return make_logits_processors(presence_penalty=1.5, presence_context_size=20,
                                  penalty_generation_start=len(prompt))


@pytest.mark.parametrize("processed", [False, True])
def test_draft_lazy_pad_to_contract(processed):
    """``pad_to`` returns a menu block, consumes the unpadded uniforms, and is a no-op when
    the proposal is already that long."""
    from mlx2.runtime.speculative_sampling import RequestRNG

    _, draft = tiny()
    extra = {}
    if processed:
        extra = {"logits_processors": _presence_make([1, 2, 3]), "processor_history": [1, 2, 3]}
    hidden = mx.random.normal((1, 5, draft.config.hidden_size * len(draft.config.target_layer_ids)),
                              key=mx.random.key(3))
    for depth in (1, 2, 3):
        rng_a, rng_b = RequestRNG(7), RequestRNG(7)
        plain = draft.draft_lazy(4, hidden, draft.make_cache(), depth, rng=rng_a, **extra)
        padded = draft.draft_lazy(4, hidden, draft.make_cache(), depth, rng=rng_b, pad_to=4,
                                  **extra)
        assert padded.shape == (4,) and padded.dtype == mx.int32
        assert rng_a.snapshot() == rng_b.snapshot()
        assert all(0 <= int(t) for t in padded.tolist()[:depth])
        rng_c = RequestRNG(7)
        same = draft.draft_lazy(4, hidden, draft.make_cache(), depth, rng=rng_c, pad_to=depth,
                                **extra)
        assert same.tolist() == plain.tolist()


def test_pairwise_walk_steps_mark_each_invalid_position():
    from mlx2.runtime.drafters.dflash2 import pairwise_walk

    rng = np.random.default_rng(5)
    candidates = mx.array(rng.integers(0, 50, (2, 4, 3)), dtype=mx.int32)
    scores = rng.normal(size=(2, 4, 3, 3)).astype(np.float32)
    scores[0, 2] = np.nan
    scores[1, 3] = -np.inf
    uniforms = rng.uniform(size=(2, 4)).tolist()
    tokens, laws, invalid = pairwise_walk(candidates, mx.array(scores), uniforms, [0.0, 0.0])
    t2, l2, i2, steps = pairwise_walk(candidates, mx.array(scores), uniforms, [0.0, 0.0],
                                      steps=True)
    assert tokens.tolist() == t2.tolist() and invalid.tolist() == i2.tolist()
    assert steps.tolist() == [[False, False, True, False], [False, False, False, True]]
    assert mx.any(steps, axis=1).tolist() == invalid.tolist()


@pytest.mark.parametrize("bad, depth, invalid", [(3, 2, False), (1, 2, True), (0, 4, True)])
def test_only_real_positions_mark_a_padded_draft_invalid(monkeypatch, bad, depth, invalid):
    """A NaN pair-score row at a padded position never invalidates the draft (-1 tokens);
    one at a real position still does, padded or not."""
    _, draft = tiny()
    original = type(draft).pairwise_score_table

    def poisoned(self, anchors, features, candidates, unary):
        scores = original(self, anchors, features, candidates, unary)
        position = mx.arange(scores.shape[1]).reshape(1, -1, 1, 1) == bad
        return mx.where(position, mx.array(float("nan"), dtype=scores.dtype), scores)

    monkeypatch.setattr(type(draft), "pairwise_score_table", poisoned)
    hidden = mx.random.normal((1, 5, draft.config.hidden_size * len(draft.config.target_layer_ids)),
                              key=mx.random.key(11))
    padded = draft.draft_lazy(4, hidden, draft.make_cache(), depth, pad_to=4)
    assert (min(padded.tolist()[:depth]) == -1) is invalid
    if depth == 4:  # pad_to equal to the depth: the same masked graph, every position real
        assert padded.tolist() == [-1] * 4
