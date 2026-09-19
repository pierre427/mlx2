from collections import deque
from types import MethodType, SimpleNamespace

import mlx.core as mx
from test_batched_mtp import _tiny_qwen4_model

import mlx2.runtime.generate as generate_runtime
import mlx2.runtime.hybrid_speculative as hybrid_speculative
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.memory_policy import (
    SelfMTPLaneAdmissionController,
    _make_self_mtp_admission_callback,
)


class _EmptyMTPBatch:
    has_deferred_lanes = False
    uids = ()

    def __init__(self, active=False):
        self.active = active

    def mtp_cycle_state(self):
        return ((99,),) if self.active else ()

    def next(self):
        return []

    def __len__(self):
        return 0

    def remove_uids(self, _uids):
        return None

    def extend(self, _batch):
        return None


class _EmptyPlainBatch:
    def __len__(self):
        return 0


class _SizedCache:
    def __init__(self, nbytes, offset):
        self.nbytes = int(nbytes)
        self.offset = int(offset)


class _StarvedMTPBatch:
    mtp_admission = object()
    _plain_ready = []

    def __init__(self):
        self._paused = {7: object(), 8: object()}
        self.state = SimpleNamespace(lanes=[])

    def demote_oldest_paused_to_plain(self):
        uid = next(iter(self._paused))
        self._plain_ready.append(self._paused.pop(uid))
        return uid


def _scheduler_only_mtp_batch(prompt_tokens, *, step=4):
    batch = BatchGenerator.__new__(BatchGenerator)
    batch._generation_batch = _EmptyMTPBatch()
    batch._plain_fallback_batch = _EmptyPlainBatch()
    batch._unprocessed_sequences = deque(
        [
            (
                7,
                [range(prompt_tokens)],
                8,
                [],
                [],
                None,
                [],
                None,
                0.0,
            )
        ]
    )
    batch.completion_batch_size = 1
    batch.self_mtp = {"segment_aware_live_tip": False}
    batch._mtp_configs = {}
    batch.adaptive_prefill = True
    batch.prefill_step_size = step
    batch._last_decode_completed_s = None
    batch._last_decode_duration_ms = None
    batch._last_decode_interval_ms = None
    batch._budget_admissible = lambda n: n
    batch._admit_mtp_joining = lambda n: n
    batch._migrate_plain_fallbacks = list
    batch._mtp_states = {7: None}
    batch._mtp_prefill_resident = set()
    batch._mtp_prefill_projection_bytes = {}
    batch._prompt_tokens_counter = 0
    batch._prompt_time_counter = 0.0
    batch._prefill_ms_per_token_ewma = None
    batch.scheduler_stats = {
        "prefill_rounds": 0,
        "adaptive_prefill_release_rounds": 0,
        "adaptive_prefill_chunk_histogram": {},
    }
    return batch


def test_cheaper_tail_runs_when_memory_policy_rejects_queue_head():
    mx.random.seed(19)
    model = _tiny_qwen4_model()

    def admit(rows):
        chosen = rows[-1][0]
        return {row[0]: 2 if row[0] == chosen else "queue" for row in rows}

    batch = BatchGenerator(
        model,
        completion_batch_size=2,
        prefill_step_size=4,
        self_mtp={"persistent": True, "num_draft": 2, "segment_aware_live_tip": True},
        mtp_admission=admit,
    )
    try:
        uids = batch.insert([[1, 2, 3, 4], [5, 6, 7]], max_tokens=[4, 4])
        finished = []
        for _ in range(20):
            _, responses = batch.next()
            finished.extend(r.uid for r in responses if r.finish_reason)
            if len(finished) == 2:
                break
        assert finished == list(reversed(uids))
    finally:
        batch.close()


def _atomic_joining_batch(decisions):
    batch = BatchGenerator.__new__(BatchGenerator)
    batch._generation_batch = _EmptyMTPBatch()
    batch._unprocessed_sequences = deque(
        (uid, [[1, 2]], 4, [], [], None, [], None, 0.0)
        for uid in range(20)
    )
    batch.self_mtp = {"persistent": True, "num_draft": 2}
    cohort = {"tenant_id": "tenant-a", "id": "live-b20", "size": 20}
    batch._mtp_configs = {
        uid: {"batch_cohort": dict(cohort)} for uid in range(20)
    }
    batch._mtp_states = {uid: None for uid in range(20)}
    batch._mtp_prefill_resident = set()
    batch._mtp_prefill_projection_bytes = {}
    batch.prefill_step_size = 2048
    batch.mtp_admission = lambda _rows: dict(decisions)
    batch.scheduler_stats = {}
    batch._atomic_cohort_failures = []
    return batch


def test_declared_mtp_cohort_fails_whole_group_on_partial_lane_admission():
    decisions = {uid: 2 if uid < 2 else "queue" for uid in range(20)}
    batch = _atomic_joining_batch(decisions)

    assert batch._admit_mtp_joining(20) == 0
    assert [sequence[0] for sequence in batch._unprocessed_sequences] == list(
        range(20)
    )
    assert batch.take_atomic_cohort_failures() == [
        {
            "uids": tuple(range(20)),
            "cohort": {
                "tenant_id": "tenant-a",
                "id": "live-b20",
                "size": 20,
            },
            "reason": (
                "declared batch cohort was not wholly admitted for "
                "speculative decoding"
            ),
        }
    ]
    assert batch.scheduler_stats["atomic_cohort_admission_failures"] == 1


def test_declared_mtp_cohort_fails_if_state_budget_exposes_only_prefix():
    batch = _atomic_joining_batch({uid: 2 for uid in range(20)})

    assert batch._admit_mtp_joining(2) == 0
    failure = batch.take_atomic_cohort_failures()
    assert failure[0]["uids"] == tuple(range(20))
    assert "scheduler admission boundary" in failure[0]["reason"]


def test_declared_mtp_cohort_admits_all_members_as_one_boundary():
    batch = _atomic_joining_batch({uid: 2 for uid in range(20)})

    assert batch._admit_mtp_joining(20) == 20
    assert batch.take_atomic_cohort_failures() == []


def test_declared_mtp_cohort_uses_uniform_lower_depth_admission():
    batch = _atomic_joining_batch({})

    def ordinary(_rows):
        return {uid: 2 if uid < 19 else "queue" for uid in range(20)}

    ordinary.atomic = lambda _rows: {uid: 1 for uid in range(20)}
    batch.mtp_admission = ordinary

    assert batch._admit_mtp_joining(20) == 20
    assert {
        config["num_draft"] for config in batch._mtp_configs.values()
    } == {1}
    assert {
        config["requested_num_draft"] for config in batch._mtp_configs.values()
    } == {2}
    assert {
        config["cohort_admission_stage"] for config in batch._mtp_configs.values()
    } == {"lower_k"}
    assert batch.take_atomic_cohort_failures() == []


def test_starved_mtp_lane_demotes_to_plain_after_eight_closed_boundaries():
    batch = BatchGenerator.__new__(BatchGenerator)
    batch._generation_batch = _StarvedMTPBatch()
    batch._plain_fallback_batch = _EmptyPlainBatch()
    batch._starved_mtp_boundaries = 0
    batch.scheduler_stats = {}

    for _ in range(generate_runtime.MTP_STARVED_BOUNDARIES_BEFORE_PLAIN - 1):
        assert batch._demote_starved_mtp_lane() is None
    assert batch._demote_starved_mtp_lane() == 7
    assert list(batch._generation_batch._paused) == [8]
    assert batch.scheduler_stats["starved_mtp_plain_fallbacks"] == 1


def test_idle_uncached_long_mtp_prefill_yields_before_lane_preparation():
    batch = _scheduler_only_mtp_batch(13, step=4)
    calls = []

    def advance(_self, index, max_tokens):
        calls.append(("advance", index, max_tokens))
        return None, [(7, (4, 13), False, False)]

    def prepare(_self, _n):
        raise AssertionError("the first scheduler turn consumed the whole prompt")

    batch._advance_mtp_prefill = MethodType(advance, batch)
    batch._make_mtp_batch = MethodType(prepare, batch)
    batch._adaptive_prefill_decision = lambda _now: (_ for _ in ()).throw(
        AssertionError("idle prefill must not use decode-latency deferral")
    )

    prompts, responses = batch._next_mtp()

    assert responses == []
    assert prompts == [(7, (4, 13), False, False)]
    assert calls == [("advance", 0, 4)]


def test_short_idle_mtp_prompt_keeps_direct_lane_preparation():
    batch = _scheduler_only_mtp_batch(5, step=4)
    calls = []

    def advance(_self, _index, _max_tokens):
        raise AssertionError("one-chunk prompt should not be split")

    def prepare(_self, n):
        calls.append(("prepare", n))
        batch._unprocessed_sequences.popleft()
        return _EmptyMTPBatch(), [(7, (5, 5), True, True)]

    batch._advance_mtp_prefill = MethodType(advance, batch)
    batch._make_mtp_batch = MethodType(prepare, batch)

    prompts, responses = batch._next_mtp()

    assert responses == []
    assert prompts == [(7, (5, 5), True, True)]
    assert calls == [("prepare", 1)]


def test_mtp_prefill_bound_uses_lane_step_override():
    batch = _scheduler_only_mtp_batch(5, step=4)
    batch._mtp_configs[7] = {"prefill_step_size": 2}
    calls = []
    batch._advance_mtp_prefill = MethodType(
        lambda _self, index, max_tokens: (
            calls.append((index, max_tokens)) or (None, [(7, (2, 5), False, False)])
        ),
        batch,
    )
    batch._make_mtp_batch = MethodType(
        lambda _self, _n: (_ for _ in ()).throw(
            AssertionError("lane-specific multi-chunk prompt must yield")
        ),
        batch,
    )

    prompts, _ = batch._next_mtp()

    assert prompts == [(7, (2, 5), False, False)]
    assert calls == [(0, 2)]


def test_mtp_incremental_selection_skips_one_chunk_candidate():
    batch = _scheduler_only_mtp_batch(5, step=4)
    batch._unprocessed_sequences.append(
        (8, [range(13)], 8, [], [], None, [], None, 0.0)
    )
    batch.completion_batch_size = 2
    calls = []
    batch._advance_mtp_prefill = MethodType(
        lambda _self, index, max_tokens: (
            calls.append((index, max_tokens)) or (None, [(8, (4, 13), False, False)])
        ),
        batch,
    )
    batch._make_mtp_batch = MethodType(
        lambda _self, _n: (_ for _ in ()).throw(
            AssertionError("long candidate must take the bounded turn")
        ),
        batch,
    )

    prompts, _ = batch._next_mtp()

    assert prompts == [(8, (4, 13), False, False)]
    assert calls == [(1, 4)]

    # The next round belongs to the one-chunk request (omlx#3726): it must not
    # wait for every chunk of the long prefill.
    made = []
    batch._make_mtp_batch = MethodType(
        lambda _self, n: made.append(
            (n, _self._unprocessed_sequences[0][0])
        ) or ([], []),
        batch,
    )
    batch._generation_batch.extend = lambda _batch: None
    batch._migrate_plain_fallbacks = lambda: []
    batch._next_mtp()
    assert made == [(1, 7)]
    assert calls == [(1, 4)]


def test_long_mtp_prefill_adaptive_defer_only_with_active_decode():
    batch = _scheduler_only_mtp_batch(13, step=4)
    batch._generation_batch = _EmptyMTPBatch(active=True)
    batch.completion_batch_size = 2
    decisions = []
    batch._adaptive_prefill_decision = lambda now: (
        decisions.append(now) or (True, 0, False)
    )
    batch._advance_mtp_prefill = MethodType(
        lambda _self, _index, _max_tokens: (_ for _ in ()).throw(
            AssertionError("deferred prefill must not advance")
        ),
        batch,
    )
    batch._make_mtp_batch = MethodType(
        lambda _self, _n: (_ for _ in ()).throw(
            AssertionError("deferred prefill must not prepare")
        ),
        batch,
    )

    prompts, responses = batch._next_mtp()

    assert prompts == []
    assert responses == []
    assert len(decisions) == 1


def test_near_limit_idle_mtp_prefill_returns_control_every_bounded_slice():
    total = 262_051
    step = 2_048
    batch = _scheduler_only_mtp_batch(total, step=step)
    visible = []
    prepared = []

    def advance(_self, index, max_tokens):
        assert index == 0
        sequence = batch._unprocessed_sequences[0]
        remaining = len(sequence[1][0])
        processed = min(max_tokens, remaining - 1)
        after = remaining - processed
        history = len(sequence[4]) + processed
        batch._unprocessed_sequences[0] = (
            sequence[0],
            [range(after)],
            *sequence[2:4],
            range(history),
            *sequence[5:],
        )
        visible.append((history, total))
        return None, [(sequence[0], visible[-1], False, False)]

    def prepare(_self, n):
        prepared.append((n, len(batch._unprocessed_sequences[0][1][0])))
        batch._unprocessed_sequences.popleft()
        return _EmptyMTPBatch(), [(7, (total, total), True, True)]

    batch._advance_mtp_prefill = MethodType(advance, batch)
    batch._make_mtp_batch = MethodType(prepare, batch)

    turns = 0
    while batch._unprocessed_sequences:
        prompts, responses = batch._next_mtp()
        turns += 1
        assert responses == []
        assert prompts

    # The serving loop regains control after every prefill chunk, so its
    # cancellation and stalled-request watchdog checks can run between turns.
    assert turns == 128
    assert len(visible) == 127
    assert visible[0] == (step, total)
    assert all(b[0] > a[0] for a, b in zip(visible, visible[1:]))
    assert prepared == [(1, total - 127 * step)]


def test_successful_bounded_prefill_slice_marks_cache_as_owned(monkeypatch):
    batch = _scheduler_only_mtp_batch(9, step=4)
    batch.model = object()
    monkeypatch.setattr(generate_runtime, "_prefetch_known_mtp_tail", lambda *_: None)

    def advance(prompt, _model, **_kwargs):
        return (
            prompt[4:],
            [_SizedCache(1 << 20, 4)],
            ([_SizedCache(1 << 18, 3)], mx.zeros((1, 1, 1))),
            4,
        )

    monkeypatch.setattr(hybrid_speculative, "advance_self_mtp_prefill", advance)

    result, progress = batch._advance_mtp_prefill(0, 4)

    assert result is None
    assert len(progress) == 1
    assert (progress[0].uid, progress[0].progress) == (7, (4, 9))
    assert not progress[0].end_of_prompt
    assert batch._mtp_prefill_resident == {7}
    assert batch._unprocessed_sequences[0][4] == [0, 1, 2, 3]


def test_sliced_mtp_continuation_progresses_as_headroom_falls():
    total = 260_001
    gib = 1 << 30
    free = [40.0]
    controller = SelfMTPLaneAdmissionController(
        cache_estimator=lambda tokens: int(tokens * 63_000)
    )
    observed = []
    policy = _make_self_mtp_admission_callback(
        controller, free_memory=lambda: free[0]
    )

    def admit(rows):
        observed.append(rows[-1])
        return policy(rows)

    batch = _scheduler_only_mtp_batch(total, step=2_048)
    del batch._admit_mtp_joining
    batch.mtp_admission = admit
    batch._mtp_prefill_resident = {7}
    for covered, available in ((200_000, 40.0), (220_000, 35.8), (240_000, 31.0)):
        target = _SizedCache(10 * gib * covered / 200_000, covered)
        draft = _SizedCache(2 * gib * covered / 200_000, covered - 1)
        batch._unprocessed_sequences[0] = (
            7,
            [range(total - covered)],
            8,
            [target],
            range(covered),
            None,
            [],
            None,
            0.0,
        )
        batch._mtp_states = {7: ([draft], None)}
        free[0] = available
        assert batch._admit_mtp_joining(1) == 1

    assert [row[3] for row in observed] == [True, True, True]
    assert all(row[4] > 0.0 and row[5] > 0.0 for row in observed)
    assert observed[-1][5] < observed[0][5]


def test_sliced_mtp_pending_reservation_is_monotone_across_capacity_jump():
    total = 260_001
    gib = 1 << 30
    observed = []

    def admit(rows):
        observed.append(rows[-1])
        return {7: 2}

    admit.cache_projection_bytes = lambda tokens: int(32 * gib * tokens / total)
    batch = _scheduler_only_mtp_batch(total, step=2_048)
    del batch._admit_mtp_joining
    batch.mtp_admission = admit
    batch._mtp_prefill_resident = {7}

    for covered, target_gib, draft_gib in (
        (200_000, 10, 2),
        (202_048, 20, 4),
        (240_000, 22, 5),
    ):
        batch._unprocessed_sequences[0] = (
            7,
            [range(total - covered)],
            8,
            [_SizedCache(target_gib * gib, covered)],
            range(covered),
            None,
            [],
            None,
            0.0,
        )
        batch._mtp_states = {
            7: ([_SizedCache(draft_gib * gib, covered - 1)], None)
        }
        assert batch._admit_mtp_joining(1) == 1

    pending = [row[5] for row in observed]
    assert pending == [20.0, 8.0, 5.0]
    assert pending == sorted(pending, reverse=True)
    assert batch._mtp_prefill_projection_bytes == {7: 32 * gib}


def test_sliced_mtp_projection_bound_violation_queues_fail_closed():
    total = 260_001
    gib = 1 << 30

    def admit(rows):
        return {row[0]: 2 for row in rows}

    admit.cache_projection_bytes = lambda _tokens: 16 * gib
    batch = _scheduler_only_mtp_batch(total, step=2_048)
    del batch._admit_mtp_joining
    batch.mtp_admission = admit
    batch._mtp_prefill_resident = {7}
    batch._mtp_prefill_projection_bytes = {7: 16 * gib}
    batch._unprocessed_sequences[0] = (
        7,
        [range(total - 202_048)],
        8,
        [_SizedCache(20 * gib, 202_048)],
        range(202_048),
        None,
        [],
        None,
        0.0,
    )
    batch._mtp_states = {7: ([_SizedCache(4 * gib, 202_047)], None)}

    assert batch._admit_mtp_joining(1) == 0
    assert batch._mtp_prefill_projection_bytes == {7: 16 * gib}


def test_sliced_mtp_projection_ownership_clears_on_remove_and_close():
    batch = _scheduler_only_mtp_batch(9, step=4)
    batch._mtp_prefill_resident = {7}
    batch._mtp_prefill_projection_bytes = {7: 1234}
    batch._mtp_states = {7: None}
    batch._mtp_lane_rngs = {7: None}
    batch._mtp_configs = {7: {}}
    batch._prompt_boundaries = {7: {}}
    batch._prompt_batch = _EmptyPlainBatch()
    batch._currently_processing = []
    batch._find_uids = lambda _uids: {7: (0, 0)}

    batch.remove([7])

    assert not batch._mtp_prefill_resident
    assert not batch._mtp_prefill_projection_bytes
    assert not batch._mtp_states
    assert not batch._mtp_lane_rngs
    assert not batch._mtp_configs
    assert not batch._prompt_boundaries

    batch._mtp_prefill_resident = {8}
    batch._mtp_prefill_projection_bytes = {8: 5678}
    batch._old_wired_limit = None
    batch.close()

    assert not batch._mtp_prefill_resident
    assert not batch._mtp_prefill_projection_bytes


def test_sliced_mtp_projection_ownership_clears_at_final_preparation(monkeypatch):
    batch = _scheduler_only_mtp_batch(5, step=4)
    batch.model = object()
    batch.mtp_admission = None
    batch._mtp_prefill_resident = {7}
    batch._mtp_prefill_projection_bytes = {7: 1234}
    batch._mtp_states = {7: None}
    batch._mtp_lane_rngs = {7: None}
    batch._mtp_configs = {7: {}}
    batch._prompt_boundaries = {}
    monkeypatch.setattr(generate_runtime, "_prefetch_known_mtp_tail", lambda *_: None)

    detached = SimpleNamespace(
        lane=SimpleNamespace(max_tokens=8, ntoks=0),
    )
    monkeypatch.setattr(
        hybrid_speculative,
        "prepare_self_mtp_lane",
        lambda *_args, **_kwargs: (detached, "first"),
    )
    class FakePreparedBatch:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(generate_runtime, "MTPGenerationBatch", FakePreparedBatch)

    prepared, progress = batch._make_mtp_batch(1)

    assert isinstance(prepared, FakePreparedBatch)
    assert progress[0].end_of_prompt
    assert not batch._mtp_prefill_resident
    assert not batch._mtp_prefill_projection_bytes


def test_mtp_admission_still_queues_cold_or_genuinely_unsafe_prefill():
    total = 260_001
    gib = 1 << 30
    free = [35.8]
    controller = SelfMTPLaneAdmissionController(
        cache_estimator=lambda tokens: int(tokens * 63_000)
    )
    policy = _make_self_mtp_admission_callback(
        controller, free_memory=lambda: free[0]
    )
    batch = _scheduler_only_mtp_batch(total, step=2_048)
    del batch._admit_mtp_joining
    batch.mtp_admission = policy
    batch._mtp_prefill_resident = set()
    batch._mtp_states = {7: None}

    # A cold 260K lane still has to fit its complete adapter-projected cache.
    assert batch._admit_mtp_joining(1) == 0

    # Once a slice owns live cache state, only future growth is charged, but
    # that growth and the next verify transient must still fit above reserve.
    covered = 200_000
    batch._unprocessed_sequences[0] = (
        7,
        [range(total - covered)],
        8,
        [_SizedCache(10 * gib, covered)],
        range(covered),
        None,
        [],
        None,
        0.0,
    )
    batch._mtp_states = {7: ([_SizedCache(2 * gib, covered - 1)], None)}
    batch._mtp_prefill_resident.add(7)
    free[0] = 20.0
    assert batch._admit_mtp_joining(1) == 0


def _scheduler_only_mtp_cohort(monkeypatch, checkpoints=None):
    """A 5/13-token declared cohort driving the real prefill + admission path.

    Only the model kernel is faked: it consumes ``min(max_tokens, len - 1)``
    tokens exactly like ``advance_self_mtp_prefill``.  ``_make_mtp_batch``
    records the uids it would prepare together.
    """

    def advance(prompt, _model, *, prompt_cache, mtp_state, max_tokens, **_):
        n = min(int(max_tokens), int(prompt.size) - 1)
        return prompt[n:], prompt_cache, mtp_state, n

    monkeypatch.setattr(hybrid_speculative, "advance_self_mtp_prefill", advance)
    monkeypatch.setattr(generate_runtime, "_prefetch_known_mtp_tail", lambda *_: 0)
    batch = _scheduler_only_mtp_batch(5, step=4)
    batch._unprocessed_sequences.append(
        (8, [range(13)], 8, [], [], None, [], None, 0.0)
    )
    batch.completion_batch_size = 2
    batch.adaptive_prefill = False
    batch.model = None
    batch.mtp_admission = None
    cohort = {"tenant_id": "t", "id": "c", "size": 2}
    batch._mtp_configs = {
        uid: {"batch_cohort": cohort, "num_draft": 2} for uid in (7, 8)
    }
    batch._mtp_states = {7: None, 8: None}
    batch._interior_checkpoint_positions = {
        uid: deque(positions) for uid, positions in (checkpoints or {}).items()
    }
    batch._capture_mtp_interior_checkpoint = lambda *_: False
    batch._validate_mtp_config = lambda _config: None
    batch._fairness = lambda: SimpleNamespace(
        enabled=False, observe_prefill=lambda *_, **__: None
    )
    del batch._admit_mtp_joining  # use the real atomic admission gate
    prepared = []

    def make(self, n):
        prepared.append(
            [self._unprocessed_sequences.popleft()[0] for _ in range(n)]
        )
        return _EmptyMTPBatch(), []

    batch._make_mtp_batch = MethodType(make, batch)
    return batch, prepared


def _drain_mtp_queue(batch):
    for _ in range(10):
        if not batch._unprocessed_sequences:
            return
        batch._next_mtp()
    raise AssertionError("cohort never prepared")


def test_mixed_length_cohort_prefill_prepares_members_together(monkeypatch):
    batch, prepared = _scheduler_only_mtp_cohort(monkeypatch)
    _drain_mtp_queue(batch)
    assert prepared == [[7, 8]]


def test_cohort_member_reaching_final_token_is_held_for_siblings(monkeypatch):
    # The final-chunk exit of ``_advance_mtp_prefill`` used to prepare the
    # member alone (B1 decode) while its sibling was still queued.
    batch, prepared = _scheduler_only_mtp_cohort(monkeypatch)
    held, prompts = batch._advance_mtp_prefill(0, 4)

    assert held is None and prepared == []
    assert [(r.uid, r.progress, r.end_of_prompt) for r in prompts] == [
        (7, (4, 5), False)
    ]
    assert [sequence[0] for sequence in batch._unprocessed_sequences] == [7, 8]
    assert batch._unprocessed_sequences[0][1] == [[4]]
    assert batch.scheduler_stats["atomic_cohort_prefill_holds"] == 1

    _drain_mtp_queue(batch)
    assert prepared == [[7, 8]]


def test_cohort_member_with_single_token_prompt_is_held(monkeypatch):
    batch, prepared = _scheduler_only_mtp_cohort(monkeypatch)
    batch._unprocessed_sequences[0] = (7, [[4]], 8, [], [], None, [], None, 0.0)
    assert batch._advance_mtp_prefill(0, 4) == (None, [])
    assert prepared == []
    assert batch.scheduler_stats["atomic_cohort_prefill_holds"] == 1


def test_make_mtp_batch_fails_closed_on_a_cohort_subset(monkeypatch):
    batch, _ = _scheduler_only_mtp_cohort(monkeypatch)
    before = list(batch._unprocessed_sequences)
    try:
        BatchGenerator._make_mtp_batch(batch, 1)
    except RuntimeError as error:
        assert "whole batch" in str(error)
    else:
        raise AssertionError("a proper cohort subset was prepared")
    assert list(batch._unprocessed_sequences) == before


def test_ungrouped_head_never_admits_a_cohort_prefix():
    batch = BatchGenerator.__new__(BatchGenerator)
    batch._unprocessed_sequences = deque(
        (uid, [[1, 2]], 4, [], [], None, [], None, 0.0) for uid in (6, 7, 8)
    )
    cohort = {"tenant_id": "t", "id": "c", "size": 2}
    batch._mtp_configs = {7: {"batch_cohort": cohort}, 8: {"batch_cohort": cohort}}
    batch.mtp_admission = None
    batch.scheduler_stats = {}
    assert batch._admit_mtp_joining(2) == 1
    assert batch._admit_mtp_joining(3) == 1


def test_mtp_short_interleave_never_splits_a_declared_cohort():
    batch = _scheduler_only_mtp_batch(5, step=4)
    batch._unprocessed_sequences.append(
        (8, [range(13)], 8, [], [], None, [], None, 0.0)
    )
    batch.completion_batch_size = 2
    cohort = {"tenant_id": "t", "id": "c", "size": 2}
    batch._mtp_configs = {7: {"batch_cohort": cohort}, 8: {"batch_cohort": cohort}}
    calls = []
    batch._advance_mtp_prefill = MethodType(
        lambda _self, index, max_tokens: (
            calls.append((index, max_tokens)) or (None, [])
        ),
        batch,
    )
    batch._make_mtp_batch = MethodType(
        lambda _self, _n: (_ for _ in ()).throw(
            AssertionError("a cohort member must not be admitted alone")
        ),
        batch,
    )
    batch._next_mtp()
    batch._next_mtp()
    assert calls == [(1, 4), (1, 4)]
