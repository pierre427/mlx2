"""A self-MTP lane preparation's overhead is per call, not a whole prefill.

``PrefillCostModel`` charges a preparation (``mtp_prepare``) as the ``mtp``
slice model plus a measured per-call overhead.  Before any slice was
measured, ``_observe_overhead`` predicted the preparation's base cost as the
decode-seeded fixed cost alone, so an idle one-call preparation of a whole
prompt (a fresh server's first chat turns) stored its entire prefill as
"overhead".  Once the first contended slice gave the base model a sample,
every contended preparation for the next 60 s was bounded with base cost
plus that whole prefill: the 0.9-quantile overhead ring held ~0.7 s where
the preparation pays ~70 ms, the preparation's stall bound fell to the
64-row grid, and co-arriving short prompts beside a decoding lane were
prepared one per round.  The same samples recorded slice-first gave the
expected bounds.
"""

from types import SimpleNamespace

from mlx2.runtime import adaptive_policy as A
from mlx2.runtime.adaptive_policy import DecodeTimeFairness


def _policy(monkeypatch, order):
    now = [100.0]
    monkeypatch.setattr(
        A, "time", SimpleNamespace(monotonic=lambda: now[0], perf_counter=lambda: now[0])
    )
    policy = DecodeTimeFairness(enabled=True, stall_target_ms=500.0)
    for _ in range(4):
        policy.observe_decode(0.030)
    for (kind, rows, seconds) in order:
        policy.observe_prefill(rows, seconds, contended=False, depth=0, kind=kind)
        now[0] += 1.0
    return policy


# 30 ms fixed + 0.43 ms/row, plus 70 ms per preparation.
_PREPARE = ("mtp_prepare", 1500, 0.030 + 0.00043 * 1500 + 0.070)
_SLICE = ("mtp", 512, 0.030 + 0.00043 * 512)


def test_idle_preparations_before_any_slice_do_not_inflate_the_overhead(monkeypatch):
    idle_first = _policy(monkeypatch, [_PREPARE] * 3 + [_SLICE])
    slice_first = _policy(monkeypatch, [_SLICE] + [_PREPARE] * 3)

    # The preparation's per-call overhead is ~70 ms either way, never the
    # 1500-row prefill it ran.
    assert slice_first.cost.overhead("mtp_prepare") < 0.150
    assert idle_first.cost.overhead("mtp_prepare") < 0.150
    assert idle_first.counters.get("cost_prepare_overhead_us", 0) < 150_000

    # A contended preparation is bounded by the slice model plus that
    # overhead, not pinned to the 64-row grid.
    reference = slice_first.stall_bound(2048, depth=0, kind="mtp_prepare")
    assert reference >= 512
    assert idle_first.stall_bound(2048, depth=0, kind="mtp_prepare") >= reference

    # A 273-row boundary slice plus a 7-row preparation fit one 500 ms target.
    forwards = [(273, 0, "mtp"), (7, 273, "mtp_prepare")]
    assert slice_first.burst_keep(forwards) == 2
    assert idle_first.burst_keep(forwards) == 2


def test_preparations_measured_before_a_slice_still_charge_their_overhead(
    monkeypatch,
):
    # The measurement is kept, not discarded: once a slice exists, the idle
    # preparations are charged against it (~70 ms each), as if recorded later.
    idle_first = _policy(monkeypatch, [_PREPARE] * 3 + [_SLICE])
    assert 0.050 < idle_first.cost.overhead("mtp_prepare") < 0.150


def test_contended_short_preparations_after_an_idle_first_turn_share_a_round(
    monkeypatch,
):
    """Through the self-MTP scheduler: a fresh generator's first (idle) turn,
    then a long prompt beside it, then three short prompts arriving
    together.  Their three preparations (~120 ms each) fit one 450 ms
    budget and must not be spread over three rounds."""
    from test_decode_first_publish import tiny_model
    from test_mtp_cohort_formation import SELF_MTP, _admission
    from test_mtpgap_round_bound_cpu import _clock

    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import BatchGenerator

    (stalls, forwards) = _clock(
        monkeypatch, fixed=0.030, per_token=0.00043, prepare_overhead=0.070
    )
    gen = BatchGenerator(
        tiny_model(),
        completion_batch_size=8,
        prefill_step_size=512,
        self_mtp=dict(SELF_MTP),
        mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
            {"enabled": True, "max_mtp_width": 4}
        ),
        mtp_admission=_admission(),
        decode_time_fairness={"enabled": True, "stall_target_ms": 500.0},
    )
    gen.prefill_step_autoscale = False
    try:
        gen.insert([[(3 * i) % 50 + 1 for i in range(480)]], max_tokens=[4000])
        for _ in range(4):
            gen.next()
        gen.insert([[(7 * i) % 50 + 1 for i in range(2000)]], max_tokens=[2])
        uid = gen._uid_count - 1
        for _ in range(400):
            gen.next()
            if not any(s[0] == uid for s in gen._unprocessed_sequences):
                break
        assert not any(s[0] == uid for s in gen._unprocessed_sequences)
        assert gen._has_active_decode()

        gen.insert(
            [[(k * 5 + i) % 50 + 1 for i in range(40)] for k in range(3)],
            max_tokens=[2, 2, 2],
        )
        uids = set(range(gen._uid_count - 3, gen._uid_count))
        stalls.clear()
        forwards.clear()
        for _ in range(100):
            stalls.append(0.0)
            forwards.append([])
            gen.next()
            if not any(s[0] in uids for s in gen._unprocessed_sequences):
                break
        prepared = [f for f in forwards if f]
        assert prepared == [[("prepare", 40)] * 3], prepared
        assert max(stalls) <= 0.500
        assert "mtp_stall_budget_deferred_rows" not in gen.scheduler_stats
    finally:
        gen.close()
