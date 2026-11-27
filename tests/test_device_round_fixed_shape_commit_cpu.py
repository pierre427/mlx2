"""device_round.fixed_shape with device_commit and the DC-5 pipeline (slice 1a-5), CPU only.

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md (1a-5 row; DC-4/DC-5/DC-6
in section 4.9). The 2026-11-26 GPU check of fixed_shape alone left 13 capture layouts in 16
Raon rounds, because the commit side (drafter context, recurrent rollback, KV trim) followed the
previous round's accepted count. Under device_commit the transaction holds the menu width, so
the commit side has one shape too. Pins:
  * DeviceHistory.apply_rows equals apply on every row (bitwise);
  * every device round writes ``num_draft + 1`` rows, serial and pipelined;
  * pipelined rounds equal the serial device-commit round bitwise under fixed_shape (tokens,
    RNG, receipts, delivered caches), with presence windows, stops, budget ends and lagged
    composition, because both write the same rows and so read the same lagged width;
  * against device_commit without fixed_shape, greedy tokens and the RNG agree on this route
    (padding moves the read width: on the GPU near ties are possible, not on this fp32 route).
"""

import random

import mlx.core as mx
import numpy as np
import pytest
from test_device_commit_raon_cpu import NUM_DRAFT, PROMPT, _decode, _reference
from test_device_pipeline_raon_cpu import DC4, DC4_EXACT, PIPE, _assert_like_dc4, _decode_presence

from mlx2.runtime.speculative_device_pipeline import SKIPS
from mlx2.runtime.speculative_device_presence import DeviceHistory

mx.set_default_device(mx.cpu)

FIX = {"fixed_shape": True}
DC4_EXACT_FIX = {**DC4_EXACT, **FIX}
DC4_FIX = {**DC4, **FIX}
PIPE_FIX = {**PIPE, **FIX}
DC4_PROC_FIX = {**DC4_FIX, "processors": True}
PIPE_PROC_FIX = {**PIPE_FIX, "processors": True}


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


# -- DeviceHistory.apply_rows ---------------------------------------------------------------

@pytest.mark.parametrize("seed", range(30))
def test_apply_rows_equals_apply_on_every_row(seed):
    from mlx2.runtime.sample_utils import make_logits_processors

    rng = random.Random(seed)
    vocab, menu = 40, 5
    host = [rng.randrange(0, 10) for _ in range(rng.randint(2, 30))]
    k = rng.randint(1, menu)
    prev = mx.array([rng.randrange(0, 10) for _ in range(k)], dtype=mx.int32)
    kept = mx.array([rng.randint(0, k)], dtype=mx.int32)
    last = mx.array([rng.randrange(0, 10)], dtype=mx.int32)
    history = DeviceHistory(mx, host, rng.randrange(0, 10), prev, kept, last)
    processors = []
    for _ in range(1 + seed % 2):
        processors += make_logits_processors(
            presence_penalty=rng.choice([0.5, 1.5]),
            presence_context_size=rng.choice([0, 1, 3, 20]),
            penalty_generation_start=rng.choice([None, 0, len(host) // 2]))
    count = rng.randint(0, menu)
    drafts = [rng.randrange(0, 10) for _ in range(count)] + [0] * (menu - count)
    drafts = mx.array(drafts, dtype=mx.int32)
    values = mx.random.normal((menu + 1, vocab), key=mx.random.key(seed)).astype(mx.float32)
    rows = history.apply_rows(processors, drafts, count, values)
    for j in range(count + 1):
        want = history.apply(processors, drafts[:j], values[j][None])[0]
        assert mx.array_equal(rows[j], want).item(), (seed, j)


# -- the engine -------------------------------------------------------------------------------

def _widths(monkeypatch):
    from mlx2.runtime import external_speculative as es

    seen = []
    original = es.ExternalDraftBatchGenerator._cohort_target_forward

    def spy(self, inputs, *args, **kwargs):
        if kwargs.get("site") == "standard_verify":
            seen.append(int(inputs.shape[1]))
        return original(self, inputs, *args, **kwargs)

    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_cohort_target_forward", spy)
    return seen


@pytest.mark.parametrize("policy", [DC4_EXACT_FIX, DC4_FIX, PIPE_FIX], ids=["exact", "lagged", "pipe"])
def test_every_device_round_writes_the_menu_width(monkeypatch, policy):
    seen = _widths(monkeypatch)
    _, _, _, stats = _decode(policy, oracle=monkeypatch)
    rounds = stats["external_device_round_rounds"]
    assert rounds > 3 and stats["external_device_round_fixed_shape_rounds"] == rounds
    assert stats["external_device_round_device_commit_rounds"] == rounds
    # Every runner round at the menu width; the only other forward is the zero-draft
    # final round the chain body serves.
    assert seen.count(NUM_DRAFT + 1) == rounds
    assert set(seen) - {NUM_DRAFT + 1} <= {1}


@pytest.mark.parametrize("oracle", [False, True], ids=["random-drafter", "oracle-drafts"])
def test_fixed_shape_device_commit_tokens_equal_device_commit(monkeypatch, oracle):
    for base in (DC4_EXACT, DC4):
        var = _decode(base, oracle=monkeypatch if oracle else None)
        fix = _decode({**base, **FIX}, oracle=monkeypatch if oracle else None)
        assert fix[0] == var[0] and fix[1] == var[1]  # tokens, lane RNG
        assert fix[2].finish_reason == var[2].finish_reason


@pytest.mark.parametrize("oracle", [False, True], ids=["random-drafter", "oracle-drafts"])
def test_pipelined_equals_serial_under_fixed_shape(monkeypatch, oracle):
    serial = _decode(DC4_FIX, oracle=monkeypatch if oracle else None)
    piped = _decode(PIPE_FIX, oracle=monkeypatch if oracle else None)
    stats = _assert_like_dc4(serial, piped)
    assert stats["external_device_round_pipeline_rounds"] > 3
    assert stats["external_device_round_pipeline_followers"] > 2
    assert stats["external_device_round_fixed_shape_rounds"] == stats["external_device_round_rounds"]


def test_a_stop_discards_the_following_round_under_fixed_shape(monkeypatch):
    stop = _reference(PROMPT, 40)[17]
    serial = _decode(DC4_FIX, oracle=monkeypatch, stops=[stop])
    piped = _decode(PIPE_FIX, oracle=monkeypatch, stops=[stop])
    stats = _assert_like_dc4(serial, piped)
    assert piped[0][-1] == stop and stats["external_device_round_pipeline_discards"] == 1
    # The discarded round left no lane trace: per-lane fixed-shape rounds equal the booked rounds.
    lane_counts = piped[2].speculative_receipt.get("device_round", {}).get("lane", {})
    if lane_counts:
        assert lane_counts.get("fixed_shape_rounds") == lane_counts.get("rounds")


@pytest.mark.parametrize("steps", [7, 12, 13, 19])
def test_budget_ends_under_fixed_shape(monkeypatch, steps):
    serial = _decode(DC4_FIX, oracle=monkeypatch, steps=steps)
    piped = _decode(PIPE_FIX, oracle=monkeypatch, steps=steps)
    _assert_like_dc4(serial, piped)


@pytest.mark.parametrize("context,start", [(20, len(PROMPT)), (3, len(PROMPT)), (0, len(PROMPT)),
                                           (5, None), (1, 0)])
def test_presence_lane_pipelines_bitwise_under_fixed_shape(monkeypatch, context, start):
    serial = _decode_presence(DC4_PROC_FIX, monkeypatch, context=context, start=start)
    piped = _decode_presence(PIPE_PROC_FIX, monkeypatch, context=context, start=start)
    stats = _assert_like_dc4(serial, piped)
    assert stats["external_device_round_pipeline_followers"] > 3
    assert stats[SKIPS].get("processors", 0) == 0
    assert stats["external_device_round_fixed_shape_rounds"] == stats["external_device_round_rounds"]


@pytest.mark.parametrize("presence", [False, True], ids=["external", "external-presence"])
def test_lagged_composition_under_fixed_shape(presence):
    serial = _decode({**DC4_FIX, "processors": True}, composed=True, presence=presence, steps=30)
    piped = _decode({**PIPE_PROC_FIX, "lagged_composition": True}, composed=True,
                    presence=presence, steps=30)
    assert piped[0] == serial[0] and piped[1][0] == serial[1][0]
    stats = piped[3]
    assert stats["external_device_round_pipeline_followers"] > 5
    assert stats["external_device_round_fixed_shape_rounds"] == stats["external_device_round_rounds"]


def test_nax_route_is_bitwise_under_fixed_shape(monkeypatch):
    """The absorbed NAX decode reads the true device length whatever the verify width (DC-5),
    so on that route the menu width changes nothing: tokens and delivered caches are bitwise
    against the plain device round, serial and pipelined."""
    from test_device_commit_raon_cpu import _arrays, _same_arrays
    from test_device_pipeline_raon_cpu import _nax_engine_run

    plain = _nax_engine_run(monkeypatch, {"lazy_draft": True, "decide": True})
    for policy in (DC4_FIX, PIPE_FIX):
        tokens, final, counts, stats = _nax_engine_run(monkeypatch, policy)
        assert tokens == plain[0]
        _same_arrays(_arrays(plain[1].prompt_cache), _arrays(final.prompt_cache))
        assert counts["latent_flash_device_offset"] > 0
        assert stats["external_device_round_fixed_shape_rounds"] == stats["external_device_round_rounds"]
