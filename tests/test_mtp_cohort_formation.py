"""MTP cohort formation under planned boundaries and staggered arrivals.

Regression for the 27B options sweep (2026-10-02, item 4).  With interior
checkpoints on (the native-MTP adapter default), cold short prompts that
arrived together were admitted one per round: the first decoded alone and
locked the segmented cohort at width 1, later lanes waited out the width
lock and went plain beside it, and the handoff never fired because its width
test ignored the plain lanes.  Every round then ran two forwards.
"""

import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_batched_mtp import _tiny_qwen4_model
from test_segmented_mtp import _detached

from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
from mlx2.runtime.generate import (
    BatchGenerator,
    MTPGenerationBatch,
    StopSequenceMatcher,
)
from mlx2.runtime.memory_policy import (
    SelfMTPLaneAdmissionController,
    _make_self_mtp_admission_callback,
)
from mlx2.runtime.sample_utils import LaneRNG
from mlx2.runtime.segmented_self_mtp import segmented_self_mtp_stats

SELF_MTP = {
    "num_draft": 2,
    "persistent": True,
    "segment_aware_live_tip": True,
    "segment_aware_cohort_size": 8,
    "segment_aware_async_qsa_promotion": False,
}


@pytest.fixture(scope="module")
def model():
    mx.random.seed(7)
    return _tiny_qwen4_model()


def _admission():
    return _make_self_mtp_admission_callback(
        SelfMTPLaneAdmissionController(saturation_lane_cap=8),
        free_memory=lambda: 100.0,
        max_draft=2,
    )


def _generator(model, **kwargs):
    return BatchGenerator(
        model,
        completion_batch_size=8,
        prefill_step_size=64,
        self_mtp=dict(SELF_MTP),
        mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
            {"enabled": True, "max_mtp_width": 4}
        ),
        **kwargs,
    )


def _widths(gen):
    batch = gen._generation_batch
    return (len(batch.state.lanes), len(batch._paused), len(gen._plain_fallback_batch))


# -- (b) one-chunk prompts with planned boundaries form one cohort ---------


def test_cold_short_prompts_with_interior_boundaries_prepare_together(model):
    segmented_self_mtp_stats(reset=True)
    gen = _generator(
        model,
        apc_interior_checkpoints={"count": 2, "min_stride": 2},
        mtp_admission=_admission(),
    )
    try:
        prompts = [[1 + i, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] for i in range(4)]
        uids = gen.insert(prompts, max_tokens=[20] * 4)
        gen.next()
        # One round prepares all four rows; the cohort starts at full width.
        assert _widths(gen) == (4, 0, 0)
        assert not gen._unprocessed_sequences
        stats = gen.scheduler_stats
        # Lattice min_stride 2, count 2, P 12: boundaries 4 and 8 per row.
        assert stats["apc_interior_checkpoints_captured"] == 8
        assert stats["mtp_one_chunk_boundary_advances"] == 8
        assert stats.get("mtp_short_prefill_interleaved", 0) == 0
        for uid, prompt in zip(uids, prompts):
            checkpoints = gen.pop_interior_checkpoints(uid)
            assert [c["covered_tokens"] for c in checkpoints] == [4, 8]
            assert [c["tokens"] for c in checkpoints] == [prompt[:4], prompt[:8]]
        for _ in range(60):
            gen.next()
        assert segmented_self_mtp_stats()["width_lock_plain_fallbacks"] == 0
        assert segmented_self_mtp_stats()["live_width_change_deferrals"] == 0
    finally:
        gen.close()


def test_generation_prompt_style_boundary_alone_does_not_serialise_admission(model):
    """The Qwen3.6 trigger: interior lattice off, one planned boundary each.

    The server plans a generation-prompt boundary for every chat request on
    a template that drops that suffix from history, independent of the
    interior lattice.  Any planned boundary used to make the prompt look
    multi-slice.
    """
    segmented_self_mtp_stats(reset=True)
    gen = _generator(model, mtp_admission=_admission())
    try:
        prompts = [[1 + i, 2, 3, 4, 5, 6, 7, 8, 9, 10] for i in range(8)]
        gen.insert(prompts, max_tokens=[20] * 8, apc_interior_positions=[[7]] * 8)
        gen.next()
        assert len(gen._generation_batch.state.lanes) + len(
            gen._generation_batch._plain_ready
        ) + len(gen._plain_fallback_batch) == 8
        assert not gen._unprocessed_sequences
        assert gen.scheduler_stats["apc_interior_checkpoints_captured"] == 8
        assert segmented_self_mtp_stats()["width_lock_plain_fallbacks"] == 0
    finally:
        gen.close()


def test_batched_boundary_capture_restores_exactly(model):
    """A checkpoint captured in the batched path resumes token-exactly."""
    shared = [5, 9, 13, 17, 21, 25, 29, 33]
    config = {"sampling_temp": 0.0, "num_draft": 2}
    gen = _generator(model)
    try:
        prompts = [shared + [40 + i, 41, 42, 43] for i in range(4)]
        uids = gen.insert(
            prompts,
            max_tokens=[6] * 4,
            lane_rngs=[LaneRNG(3) for _ in prompts],
            self_mtp_configs=[dict(config) for _ in prompts],
            apc_interior_positions=[[8]] * 4,
        )
        gen.next()
        assert len(gen._generation_batch.state.lanes) == 4
        (checkpoint,) = gen.pop_interior_checkpoints(uids[0])
        assert checkpoint["covered_tokens"] == 8 and checkpoint["tokens"] == shared
    finally:
        gen.close()

    follow = shared + [50, 51, 52]

    def run(**insert):
        g = BatchGenerator(
            model, self_mtp={"num_draft": 2, "persistent": True}, prefill_step_size=64
        )
        try:
            uid = g.insert(
                max_tokens=[6], lane_rngs=[LaneRNG(3)],
                self_mtp_configs=[dict(config)], **insert,
            )[0]
            out = []
            for _ in range(100):
                _p, responses = g.next()
                for response in responses:
                    if response.uid == uid:
                        out.append(response.token)
                        if response.finish_reason:
                            return out
            raise AssertionError("request did not finish")
        finally:
            g.close()

    cold = run(prompts=[follow], mtp_states=[None])
    warm = run(
        prompts=[follow[8:]],
        caches=[checkpoint["target_cache"]],
        all_tokens=[shared],
        mtp_states=[checkpoint["mtp_state"]],
    )
    assert warm == cold


# -- (a) the handoff width counts lanes already decoding ordinary ----------


def _locked_single_lane(max_width, plain):
    policy = MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": max_width}
    )
    active = MTPGenerationBatch(
        object(), [_detached(0)], [None], [StopSequenceMatcher()],
        segmented_live_tip=True, ordinary_handoff_policy=policy,
        scheduler_stats={},
    )
    active._segmented_compute_width_locked = True
    active.plain_width_probe = lambda: plain
    return active, policy


def test_width_locked_join_counts_plain_lanes_beside_the_cohort():
    segmented_self_mtp_stats(reset=True)
    active, policy = _locked_single_lane(4, plain=6)
    arriving = MTPGenerationBatch(
        object(), [_detached(9)], [None], [StopSequenceMatcher()],
        segmented_live_tip=True, ordinary_handoff_policy=policy,
    )
    active.extend(arriving)
    # 1 MTP + 1 joining + 6 plain = 8 > 4: one ordinary forward, not two.
    assert active._ordinary_handoff_latched
    assert not active.state.lanes and not active._paused
    assert [p.detached.lane.uid for p in active._plain_ready] == [0, 9]
    assert active._plain_ready[0].handoff_receipt["projected_width"] == 8
    assert segmented_self_mtp_stats()["live_width_change_deferrals"] == 0
    active.close()


def test_width_locked_join_still_defers_within_the_width():
    segmented_self_mtp_stats(reset=True)
    active, policy = _locked_single_lane(4, plain=2)
    arriving = MTPGenerationBatch(
        object(), [_detached(9)], [None], [StopSequenceMatcher()],
        segmented_live_tip=True, ordinary_handoff_policy=policy,
    )
    active.extend(arriving)
    assert not active._ordinary_handoff_latched
    assert list(active._paused) == [9]
    assert segmented_self_mtp_stats()["live_width_change_deferrals"] == 1
    active.close()


@pytest.mark.parametrize("plain, fires", [(3, False), (4, True), (7, True)])
def test_active_cohort_handoff_counts_plain_lanes(plain, fires):
    active, _policy = _locked_single_lane(4, plain=plain)
    assert active._maybe_handoff_active_cohort() is fires
    assert active._ordinary_handoff_latched is fires
    active.close()


def test_generator_wires_its_plain_batch_into_the_handoff_width(model):
    gen = _generator(model)
    try:
        probe = gen._generation_batch.plain_width_probe
        assert probe is not None and probe() == 0
    finally:
        gen.close()


def test_staggered_arrivals_hand_off_instead_of_locking_width_one(model):
    """One lane decodes alone; seven more arrive slower than the deferral bound.

    Before: every late lane waited out the width lock and went plain, the
    projected width never exceeded 2, and the run ended as [1 MTP, 7 plain]
    with two forwards per round and no handoff.
    """
    segmented_self_mtp_stats(reset=True)
    gen = _generator(model, mtp_admission=_admission())
    try:
        gen.insert([[1, 2, 3, 4, 5, 6]], max_tokens=[200])
        mixed = []
        for r in range(60):
            if 3 <= r < 3 + 6 * 7 and r % 6 == 3:
                gen.insert([[7 + r, 2, 3, 4, 5]], max_tokens=[200])
            gen.next()
            (mtp, _paused, plain) = _widths(gen)
            if mtp:
                mixed.append(mtp + plain)
        assert gen.scheduler_stats["mtp_ordinary_handoff_events"] == 1
        # MTP never shares a round with more plain rows than the width allows.
        assert max(mixed) <= 4
        assert _widths(gen) == (0, 0, 8)
    finally:
        gen.close()


# -- (d) multi-chunk co-arrivals form one cohort (sweep 2026-10-06, SS-1) ---


def test_concurrent_multichunk_prompts_form_one_cohort(model):
    """Four 150-token prompts at step 64 (three slices each), arriving together.

    Before: one lane advanced per round, the first to finish decoded alone
    and locked the cohort at width 1, and the other three went plain beside
    it after the width-lock deferrals: (1, 0, 3) for the rest of the run.
    """
    segmented_self_mtp_stats(reset=True)
    gen = _generator(model, mtp_admission=_admission())
    try:
        prompts = [[(7 * i + j) % 50 + 1 for j in range(150)] for i in range(4)]
        gen.insert(prompts, max_tokens=[40] * 4)
        seen = []
        for _ in range(80):
            gen.next()
            seen.append(_widths(gen))
        assert max(w[0] for w in seen) == 4, seen
        # The cohort starts at full width: no width-1 round precedes it.
        assert [w for w in seen if w != (0, 0, 0)][0] == (4, 0, 0), seen
        assert segmented_self_mtp_stats()["width_lock_plain_fallbacks"] == 0
        assert gen.scheduler_stats["mtp_coarrival_holds"] >= 3
    finally:
        gen.close()


def test_autoscaled_600_token_prompts_form_one_cohort(model):
    """The 27B serving shape: no adapter step, so autoscale picks 512 rows.

    Every prompt over 513 tokens is then multi-chunk; four 600-token
    co-arrivals used to settle at one MTP lane plus three plain lanes.
    """
    segmented_self_mtp_stats(reset=True)
    gen = BatchGenerator(
        model,
        completion_batch_size=8,
        prefill_step_size=8192,
        prefill_step_autoscale=True,
        self_mtp=dict(SELF_MTP, prefill_step_size=8192),
        mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
            {"enabled": True, "max_mtp_width": 4}
        ),
        mtp_admission=_admission(),
    )
    try:
        prompts = [[(11 * i + j) % 50 + 1 for j in range(600)] for i in range(4)]
        gen.insert(prompts, max_tokens=[40] * 4)
        seen = []
        for _ in range(60):
            gen.next()
            seen.append(_widths(gen))
        assert max(w[0] for w in seen) == 4, seen
        assert segmented_self_mtp_stats()["width_lock_plain_fallbacks"] == 0
    finally:
        gen.close()


def test_coarrival_hold_is_bounded_by_sibling_prefill(model):
    """A short prompt does not wait for a sibling far longer than itself.

    150 tokens beside 2000: the sibling's remaining prefill exceeds the hold
    factor times the short prompt, so the short lane starts decoding while
    the long one is still prefilling (the omlx#3726 TTFT guarantee).
    """
    segmented_self_mtp_stats(reset=True)
    gen = _generator(model, mtp_admission=_admission())
    try:
        short = [(3 * j) % 50 + 1 for j in range(150)]
        long_ = [(5 * j) % 50 + 1 for j in range(2000)]
        gen.insert([short, long_], max_tokens=[40, 40])
        first_mtp = None
        for r in range(20):
            gen.next()
            if first_mtp is None and _widths(gen)[0]:
                first_mtp = r
                break
        assert first_mtp is not None
        # The long prompt is still queued for prefill when the short decodes.
        assert len(gen._unprocessed_sequences) == 1
        assert gen.scheduler_stats.get("mtp_coarrival_holds", 0) == 0
    finally:
        gen.close()


def test_eight_multichunk_coarrivals_form_one_cohort(model):
    """The 27B serving shape at B8: every prompt spans two prefill steps.

    The hold compared the siblings' summed residual with four times the
    row's own prompt, so eight equal prompts (seven siblings) released the
    first lane alone at width 1.  Four equal prompts formed one cohort.
    """
    segmented_self_mtp_stats(reset=True)
    gen = BatchGenerator(
        model,
        completion_batch_size=8,
        prefill_step_size=64,
        self_mtp=dict(SELF_MTP),
        mtp_admission=_admission(),
    )
    try:
        prompts = [[(7 * i + j) % 50 + 1 for j in range(100)] for i in range(8)]
        gen.insert(prompts, max_tokens=[40] * 8)
        seen = []
        for _ in range(40):
            gen.next()
            seen.append(_widths(gen))
        assert [w for w in seen if w != (0, 0, 0)][0] == (8, 0, 0), seen
        assert segmented_self_mtp_stats()["width_lock_plain_fallbacks"] == 0
    finally:
        gen.close()
