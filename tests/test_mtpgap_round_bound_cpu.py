"""A contended self-MTP round must fit its stall target whole (mtpgap 2026-10-07).

GPU (Qwen3.8-27B, a 16K prompt prefilling beside a decoding lane, loop
trace in qualification/runs/mtpgap-20261007/diag): the round that finished
the long prompt ran a 273-row slice to the planned turn boundary (486 ms)
and then, in the same round, prepared the lane from the 7-token
generation-prompt residual (127 ms: target forward plus draft head and
first-token sampling) -- 634 ms of prefill work in one neighbour gap.  The
stall bound had sized the residual as one ``mtp`` forward without the
preparation's own cost, and ``_prepare_mtp_rows`` splits it at the boundary
into two forwards.

Here the kernels are faked onto a clock: a prefill forward costs
``fixed + per_token * rows``, a lane preparation additionally pays
``prepare_overhead``.  Every ``next`` call's prefill work beside the live
decode lane must stay within the 100 ms stall target.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_decode_first_publish import tiny_model
from test_mtp_cohort_formation import SELF_MTP, _admission

from mlx2.runtime import generate as G
from mlx2.runtime import hybrid_speculative as H
from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
from mlx2.runtime.generate import BatchGenerator

TARGET = 0.100


def _clock(monkeypatch, *, fixed, per_token, prepare_overhead):
    now = [100.0]
    clock = lambda: now[0]
    monkeypatch.setattr(
        G, "time", SimpleNamespace(perf_counter=clock, monotonic=clock, time=clock)
    )
    from mlx2.runtime import adaptive_policy as A

    monkeypatch.setattr(A, "time", SimpleNamespace(monotonic=clock, perf_counter=clock))
    stalls = [0.0]
    forwards = [[]]
    real_prepare = H.prepare_self_mtp_lane
    real_advance = H.advance_self_mtp_prefill
    real_next = G.MTPGenerationBatch.next
    real_plain_next = G.GenerationBatch.next

    def prepare(prompt, *args, **kwargs):
        out = real_prepare(prompt, *args, **kwargs)
        cost = fixed + prepare_overhead + per_token * int(prompt.shape[0])
        now[0] += cost
        stalls[-1] += cost
        forwards[-1].append(("prepare", int(prompt.shape[0])))
        return out

    def advance(prompt, *args, max_tokens, **kwargs):
        out = real_advance(prompt, *args, max_tokens=max_tokens, **kwargs)
        cost = fixed + per_token * int(out[3])
        now[0] += cost
        stalls[-1] += cost
        forwards[-1].append(("slice", int(out[3])))
        return out

    def next_(self):
        out = real_next(self)
        now[0] += 0.020
        return out

    def plain_next(self):
        out = real_plain_next(self)
        now[0] += 0.020
        return out

    monkeypatch.setattr(H, "prepare_self_mtp_lane", prepare)
    monkeypatch.setattr(H, "advance_self_mtp_prefill", advance)
    monkeypatch.setattr(G.MTPGenerationBatch, "next", next_)
    monkeypatch.setattr(G.GenerationBatch, "next", plain_next)
    return stalls, forwards


def _run(monkeypatch, *, prompt_tokens, boundary):
    stalls, forwards = _clock(
        monkeypatch, fixed=0.010, per_token=0.001, prepare_overhead=0.045
    )
    gen = BatchGenerator(
        tiny_model(),
        completion_batch_size=8,
        # Large enough that the stall bound, not the step, sizes slices.
        prefill_step_size=512,
        self_mtp=dict(SELF_MTP),
        mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
            {"enabled": True, "max_mtp_width": 4}
        ),
        mtp_admission=_admission(),
        decode_time_fairness={
            "enabled": True,
            "stall_target_ms": TARGET * 1000,
            "grid": 8,
            "floor": 8,
        },
    )
    gen.prefill_step_autoscale = False
    try:
        gen.insert([[3, 4, 5, 6, 7]], max_tokens=[2000])
        for _ in range(4):
            gen.next()
        # Calibrate the cost model on a few contended short preparations
        # and one sliced prompt, as earlier traffic would.
        for k in range(3):
            gen.insert([[k + 1, k + 2, k + 3, k + 4]], max_tokens=[2])
            for _ in range(4):
                gen.next()
        long_prompt = [(7 * i) % 50 + 1 for i in range(prompt_tokens)]
        positions = [[boundary]] if boundary is not None else None
        gen.insert(
            [long_prompt], max_tokens=[2],
            **({"apc_interior_positions": positions} if positions else {}),
        )
        uid = gen._uid_count - 1
        stalls.clear()
        forwards.clear()
        for _ in range(400):
            stalls.append(0.0)
            forwards.append([])
            gen.next()
            if not any(s[0] == uid for s in gen._unprocessed_sequences):
                break
        assert not any(s[0] == uid for s in gen._unprocessed_sequences)
        # The decoding neighbour stayed live for the whole prefill.
        assert gen._has_active_decode()
        return stalls, forwards
    finally:
        gen.close()


def test_final_turn_boundary_and_lane_preparation_fit_one_stall_target(monkeypatch):
    stalls, forwards = _run(monkeypatch, prompt_tokens=207, boundary=200)
    worst = max(range(len(stalls)), key=stalls.__getitem__)
    assert stalls[worst] <= TARGET + 1e-9, (stalls[worst], forwards[worst])


def test_final_residual_preparation_fits_one_stall_target(monkeypatch):
    stalls, forwards = _run(monkeypatch, prompt_tokens=207, boundary=None)
    worst = max(range(len(stalls)), key=stalls.__getitem__)
    assert stalls[worst] <= TARGET + 1e-9, (stalls[worst], forwards[worst])


def test_preparation_overhead_is_charged_to_preparations_only():
    """GPU 2026-10-07: 7-row preparations (127 ms, against ~58 ms for a
    7-row slice) sat in the ``mtp`` slice ring, so later 4K-deep slices
    were bounded at 64 rows (per-row 7.1 ms instead of ~1.2 ms)."""
    from mlx2.runtime.adaptive_policy import DecodeTimeFairness

    policy = DecodeTimeFairness(enabled=True, stall_target_ms=500.0)
    for _ in range(8):
        policy.observe_decode(0.050)
    for depth in (4096, 4416, 4736, 5056):
        policy.observe_prefill(320, 0.450, contended=True, depth=depth, kind="mtp")
    slice_bound = policy.stall_bound(512, depth=5376, kind="mtp")
    for depth in (7469, 7472, 7475):
        policy.observe_prefill(7, 0.127, contended=True, depth=depth, kind="mtp_prepare")
    # Slices are sized as before the preparations ran.
    assert policy.stall_bound(512, depth=5376, kind="mtp") == slice_bound == 320
    # A preparation pays its measured overhead on top of the slice model.
    (fixed, per_row, _ceiling) = policy.cost.estimate("mtp_prepare", 5376)
    (slice_fixed, slice_per_row, _ceiling) = policy.cost.estimate("mtp", 5376)
    assert per_row == slice_per_row
    assert 0.060 < fixed - slice_fixed < 0.120
    assert policy.stall_bound(512, depth=5376, kind="mtp_prepare") < slice_bound


def test_host_time_already_in_the_gap_shrinks_or_defers_the_slice(monkeypatch):
    """Serving-loop work since the last decode (token delivery, a prompt
    boundary store that spilled 0.6 s to disk) is part of the neighbour's
    gap: the next slice gets only the rest of the stall target, and a gap
    already half spent decodes first (once per gap)."""
    from mlx2.runtime import adaptive_policy as A

    now = [0.0]
    monkeypatch.setattr(
        A, "time", SimpleNamespace(monotonic=lambda: now[0], perf_counter=lambda: now[0])
    )
    policy = A.DecodeTimeFairness(
        enabled=True, stall_target_ms=500.0, charge_host_gap=True
    )
    for _ in range(4):
        policy.observe_decode(0.050)
    for depth in (1024, 1344, 1664, 1984):
        policy.observe_prefill(320, 0.450, contended=True, depth=depth, kind="mtp")
        now[0] += 0.450
        policy.observe_decode(0.050)
        now[0] += 0.050
    policy.debt_seconds = 0.0
    policy.observe_decode(0.050)
    fresh = policy.stall_bound(512, depth=2304, kind="mtp")
    now[0] += 0.150
    assert policy.may_prefill(contended=True)
    shrunk = policy.stall_bound(512, depth=2304, kind="mtp")
    assert shrunk < fresh
    now[0] += 0.150  # 300 ms of host work in this gap
    assert not policy.may_prefill(contended=True)
    assert policy.counters["host_gap_deferrals"] == 1
    # Once per gap: the next poll proceeds, with at most half the target
    # charged, so a stale decode clock cannot pin slices to the floor.
    assert policy.may_prefill(contended=True)
    now[0] += 5.0
    assert policy.stall_bound(512, depth=2304, kind="mtp") >= policy.grid
    policy.observe_decode(0.050)
    assert policy.stall_bound(512, depth=2304, kind="mtp") == fresh
    # Off by default: the external-draft and PLD generators are unchanged.
    policy.charge_host_gap = False
    now[0] += 0.400
    assert policy.may_prefill(contended=True)
    assert policy.stall_bound(512, depth=2304, kind="mtp") == fresh


def test_ordinary_and_self_mtp_generator_charges_host_gap():
    gen = BatchGenerator(tiny_model(), decode_time_fairness={"enabled": True})
    try:
        assert gen.decode_time_fairness.charge_host_gap
    finally:
        gen.close()


def test_routine_host_time_does_not_cost_a_slice_tile(monkeypatch):
    """Flash-Next confirm-20261007: with every few-ms serving-loop pass
    charged, a bound just above 512 rows fell a whole 64-row tile (the
    contended histogram moved from 512- to 448-row slices).  Host time
    within the cost model's margin is not charged."""
    from mlx2.runtime import adaptive_policy as A

    now = [0.0]
    monkeypatch.setattr(
        A, "time", SimpleNamespace(monotonic=lambda: now[0], perf_counter=lambda: now[0])
    )
    policy = A.DecodeTimeFairness(
        enabled=True, stall_target_ms=500.0, charge_host_gap=True
    )
    for _ in range(4):
        policy.observe_decode(0.010)
    for depth in (1024, 1536, 2048, 2560):
        # ~0.85 ms/row: the bound lands just above 512 rows.
        policy.observe_prefill(512, 0.440, contended=True, depth=depth, kind="ordinary")
        now[0] += 0.440
        policy.observe_decode(0.010)
        now[0] += 0.010
    policy.debt_seconds = 0.0
    policy.observe_decode(0.010)
    fresh = policy.stall_bound(1024, depth=3072, kind="ordinary")
    assert fresh == 512
    now[0] += 0.020  # delivery, status snapshot: routine host work
    assert policy.stall_bound(1024, depth=3072, kind="ordinary") == fresh
    now[0] += 0.180  # a disk-spilling APC store is still charged
    assert policy.stall_bound(1024, depth=3072, kind="ordinary") < fresh
