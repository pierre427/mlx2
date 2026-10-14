"""APCv2 against peer prefix-cache failure shapes (PR survey 2026-10-14).

Each test pins one shape a peer engine shipped (gufo, splash, TensorFold,
Rapid-MLX, Strata, mlx-serve, vLLM) so mlx2 fails here if it ever grows the
same defect.  CPU only; see provenance/prsurvey-1014-apcv2-audit.json.
"""

from __future__ import annotations

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.models.cache import ArraysCache, BatchKVCache, KVCache


def _kv(length, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def _recurrent(length):
    cache = ArraysCache(1)
    cache[0] = mx.ones((1, 4), dtype=mx.float32)
    cache.lengths = mx.array([length], dtype=mx.int32)
    cache._host_lengths = (cache.lengths, [length])
    mx.eval(cache.state)
    return cache


def _hybrid(length, seed=0):
    return [_recurrent(length), _kv(length, seed)]


def _sidecar(covered, seed=0):
    return MTPAPCSidecar(
        ([_kv(covered - 1, seed=seed)], mx.ones((1, 1, 4))), covered_tokens=covered
    )


def _paths(apc):
    with apc._apc_lock:
        return {
            tuple(tokens): bool(entry.prompt_cache)
            for _key, tokens, entry in apc._entry_records_locked()
        }


def _close(hit):
    close = getattr(hit.cache, "close", None)
    if callable(close):
        close()


KEY = APCKey("prsurvey-1014")


# ----------------------------------------------------------------- item 1
# gufo #436: a short RAM-tier hit masked a deeper disk-tier checkpoint.


def _shallow_resident_deep_disk(tmp_path, *, sidecar):
    clock = [0.0]
    apc = APCv2(
        max_size=16, max_bytes=1 << 24, layout_name="prsurvey",
        idle_disk_seconds=1.0, idle_disk_dir=str(tmp_path),
        now_fn=lambda: clock[0],
    )
    deep = list(range(64))
    apc.store(
        KEY, deep, _hybrid(64, seed=1),
        sidecar=_sidecar(64, seed=10) if sidecar else None,
        retention_role="committed_prompt_boundary",
    )
    clock[0] = 10.0
    assert apc.spill_idle_entries() == 1
    shallow = deep[:16]
    apc.store(
        KEY, shallow, _hybrid(16, seed=2),
        sidecar=_sidecar(16, seed=20) if sidecar else None,
        retention_role="committed_prompt_boundary",
    )
    assert _paths(apc) == {tuple(deep): False, tuple(shallow): True}
    return apc, deep + [99, 98]


def test_admission_hit_reports_a_deeper_disk_checkpoint(tmp_path):
    apc, prompt = _shallow_resident_deep_disk(tmp_path, sidecar=False)
    admission = apc.lookup(KEY, prompt, allow_disk_restore=False)
    assert admission.hit and admission.cached_tokens == 16
    # The resident hit stands, but admission learns a deeper restore exists.
    assert admission.deeper_disk_tokens == 64
    _close(admission)
    gated = apc.lookup(KEY, prompt)
    assert gated.cached_tokens == 64 and gated.deeper_disk_tokens == 0
    _close(gated)
    apc.close()


def test_sidecar_admission_hit_reports_a_deeper_disk_sidecar(tmp_path):
    apc, prompt = _shallow_resident_deep_disk(tmp_path, sidecar=True)
    admission = apc.lookup(
        KEY, prompt, allow_disk_restore=False, require_sidecar=True
    )
    assert admission.hit_kind == "mtp_sidecar"
    assert admission.cached_tokens == 16
    assert admission.deeper_disk_tokens == 64
    _close(admission)
    gated = apc.lookup(KEY, prompt, require_sidecar=True)
    assert gated.hit_kind == "mtp_sidecar" and gated.cached_tokens == 64
    _close(gated)
    apc.close()


def test_shallower_disk_checkpoint_does_not_ask_for_a_restore(tmp_path):
    clock = [0.0]
    apc = APCv2(
        max_size=16, max_bytes=1 << 24, layout_name="prsurvey",
        idle_disk_seconds=1.0, idle_disk_dir=str(tmp_path),
        now_fn=lambda: clock[0],
    )
    tokens = list(range(64))
    apc.store(KEY, tokens[:16], _hybrid(16, seed=1))
    clock[0] = 10.0
    assert apc.spill_idle_entries() == 1
    apc.store(KEY, tokens, _hybrid(64, seed=2))
    hit = apc.lookup(KEY, tokens + [99], allow_disk_restore=False)
    assert hit.cached_tokens == 64 and hit.deeper_disk_tokens == 0
    _close(hit)
    apc.close()


# --------------------------------------------------------- items 2, 7, 10
# gufo #553 / Strata #2325 / TensorFold #735: a shared root (system-prompt
# checkpoint) that conversations resume from must not be the first victim
# once it has been reused.


def _budgeted_lru(entries):
    clock = [0.0]
    one = sum(cache.nbytes for cache in _hybrid(32))
    apc = APCv2(
        max_size=64, max_bytes=entries * one + one // 2,
        layout_name="prsurvey", now_fn=lambda: clock[0],
    )
    return apc, clock


def test_reused_shared_preamble_is_evicted_by_recency_not_first():
    apc, clock = _budgeted_lru(3)
    unrelated = list(range(100, 132))
    preamble = list(range(32))
    conversation = preamble + list(range(40, 48))
    apc.store(KEY, unrelated, _hybrid(32, seed=1))
    clock[0] = 1.0
    apc.store(KEY, preamble, _hybrid(32, seed=2), retention_role="interior_checkpoint")
    clock[0] = 2.0
    hit = apc.lookup(KEY, conversation + [7])
    assert hit.cached_tokens == 32
    _close(hit)
    clock[0] = 3.0
    apc.store(
        KEY, conversation, _hybrid(40, seed=3),
        retention_role="committed_prompt_boundary",
    )
    clock[0] = 4.0
    apc.store(
        KEY, list(range(200, 232)), _hybrid(32, seed=4),
        retention_role="committed_prompt_boundary",
    )
    paths = _paths(apc)
    assert tuple(preamble) in paths, "reused preamble evicted before older entries"
    assert tuple(unrelated) not in paths


def test_unreused_interior_checkpoint_still_goes_first():
    apc, clock = _budgeted_lru(2)
    older = list(range(100, 132))
    interior = list(range(32))
    apc.store(KEY, older, _hybrid(32, seed=1))
    clock[0] = 1.0
    apc.store(KEY, interior, _hybrid(32, seed=2), retention_role="interior_checkpoint")
    clock[0] = 2.0
    apc.store(KEY, list(range(200, 232)), _hybrid(32, seed=3))
    paths = _paths(apc)
    assert tuple(interior) not in paths and tuple(older) in paths


# ---------------------------------------------------------------- item 12
# vllm #61282: the boundary the next turn needs was evicted while its
# request still ran.  mlx2's turn-end boundary (before a dropped generation
# prompt) is the next turn's only exact reuse point on a hybrid cache.


def test_next_turn_boundary_outlives_older_entries_before_its_first_hit():
    apc, clock = _budgeted_lru(3)
    older = list(range(100, 132))
    apc.store(KEY, older, _hybrid(32, seed=1))
    clock[0] = 1.0
    turn_end = list(range(32))
    apc.store(
        KEY, turn_end, _hybrid(32, seed=2),
        retention_role="interior_checkpoint", next_turn_boundary=True,
    )
    clock[0] = 2.0
    apc.store(KEY, list(range(200, 232)), _hybrid(32, seed=3))
    clock[0] = 3.0
    apc.store(KEY, list(range(300, 332)), _hybrid(32, seed=4))
    paths = _paths(apc)
    assert tuple(turn_end) in paths
    assert tuple(older) not in paths
    hit = apc.lookup(KEY, turn_end + [5, 6])
    assert hit.cached_tokens == 32 and hit.retention_role == "interior_checkpoint"
    _close(hit)


def test_next_turn_boundary_survives_a_disk_round_trip(tmp_path):
    clock = [0.0]
    apc = APCv2(
        max_size=16, max_bytes=1 << 24, layout_name="prsurvey",
        idle_disk_seconds=1.0, idle_disk_dir=str(tmp_path),
        now_fn=lambda: clock[0],
    )
    tokens = list(range(32))
    apc.store(
        KEY, tokens, _hybrid(32), retention_role="interior_checkpoint",
        next_turn_boundary=True,
    )
    clock[0] = 10.0
    assert apc.spill_idle_entries() == 1
    hit = apc.lookup(KEY, tokens + [1])
    assert hit.cached_tokens == 32
    _close(hit)
    entry = apc._trie.get(KEY, tokens)
    assert APCv2._entry_retention_rank(entry) == 2
    apc.close()


# ----------------------------------------------------------------- item 5
# TensorFold #835: a kept state that goes stale must leave the index at
# once; otherwise every lookup "hits" it, fails, and hides older entries.


def test_stale_entry_is_dropped_and_the_shallower_prefix_served():
    apc = APCv2(max_size=8, layout_name="prsurvey")
    tokens = list(range(48))
    apc.store(KEY, tokens[:16], _hybrid(16, seed=1))
    apc.store(KEY, tokens[:32], _hybrid(32, seed=2))
    stale = apc._trie.get(KEY, tokens[:32])
    stale.prompt_cache.cow_owner.invalidate()
    first = apc.lookup(KEY, tokens)
    assert first.hit and first.cached_tokens == 16
    _close(first)
    assert tuple(tokens[:32]) not in _paths(apc)
    assert apc.nbytes == sum(cache.nbytes for cache in _hybrid(16))
    assert apc.apc_stats["stale_entry_drops"] == 1
    assert apc.apc_stats["lookups"] == 1 and apc.apc_stats["hits"] == 1


def test_stale_sidecar_entry_is_dropped_too():
    apc = APCv2(max_size=8, layout_name="prsurvey")
    tokens = list(range(48))
    apc.store(KEY, tokens[:16], _hybrid(16, seed=1), sidecar=_sidecar(16, seed=3))
    apc.store(KEY, tokens[:32], _hybrid(32, seed=2), sidecar=_sidecar(32, seed=4))
    apc._trie.get(KEY, tokens[:32]).prompt_cache.cow_owner.invalidate()
    hit = apc.lookup(KEY, tokens, require_sidecar=True)
    assert hit.hit_kind == "mtp_sidecar" and hit.cached_tokens == 16
    _close(hit)
    assert tuple(tokens[:32]) not in _paths(apc)


def test_stale_resident_keeps_its_verified_disk_copy(tmp_path):
    apc = APCv2(
        max_size=8, layout_name="prsurvey", idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    tokens = list(range(48))
    apc.store(KEY, tokens[:32], _hybrid(32, seed=2))
    entry = apc._trie.get(KEY, tokens[:32])
    with apc._apc_lock:
        assert apc._spill_entry_locked(
            KEY, tokens[:32], entry, reason="persist", keep_resident=True
        )
    entry.prompt_cache.cow_owner.invalidate()
    hit = apc.lookup(KEY, tokens)
    assert hit.cached_tokens == 32  # restored from the disk copy
    _close(hit)
    assert apc.apc_stats["stale_entry_drops"] == 1
    assert apc.apc_stats["idle_disk"]["restores"] == 1
    apc.close()


# ----------------------------------------------------------------- item 3
# gufo #423: RAM evictions must spill to the disk tier, not vanish.


def test_pressure_evictions_spill_and_every_follow_up_restores(tmp_path):
    one = sum(cache.nbytes for cache in _hybrid(32))
    apc = APCv2(
        max_size=64, max_bytes=2 * one + one // 2, layout_name="prsurvey",
        idle_disk_seconds=180, idle_disk_dir=str(tmp_path),
    )
    conversations = [[1000 * n + i for i in range(32)] for n in range(1, 7)]
    for seed, tokens in enumerate(conversations):
        apc.store(KEY, tokens, _hybrid(32, seed=seed))
    assert apc.apc_stats["idle_disk"]["pressure_spills"] >= 4
    for tokens in conversations:
        hit = apc.lookup(KEY, tokens + [7])
        assert hit.cached_tokens == 32
        _close(hit)


# ----------------------------------------------------------------- item 6
# TensorFold #722: a kept prompt state holds its KV at the kept length.


def test_extracted_lane_cache_holds_exactly_its_rows():
    batch = BatchKVCache([2, 0])
    rows = mx.ones((2, 1, 9, 4), dtype=mx.float32)
    batch.update_and_fetch(rows, rows)
    lane = batch.extract(0)
    assert lane.offset == 7
    assert lane.keys.shape[2] == lane.values.shape[2] == 7
    assert lane.nbytes == 2 * 7 * 4 * 4


# ----------------------------------------------------------------- item 8
# Rapid-MLX #4626: a 2048 grid rounded a 748-token boundary down to 0.


def test_sub_chunk_turn_end_boundary_is_planned_exactly():
    from mlx2.runtime.interior_placement import (
        cold_prefill_cuts,
        generation_prompt_boundary,
        plan_interior_positions,
    )

    suffix = (151644, 77091, 198, 151667, 271, 151668, 271)
    prompt = list(range(1, 749)) + list(suffix)
    assert len(prompt) == 755
    assert generation_prompt_boundary(prompt, [suffix]) == 748
    policy = {"count": 4, "min_stride": 256, "placement": "auto"}
    assert 748 in cold_prefill_cuts(prompt, policy=policy, generation_suffixes=[suffix])
    turns = [7, 300]
    marked = [5000 if index in turns else token for index, token in enumerate(prompt)]
    positions, _ = plan_interior_positions(
        marked, count=4, min_stride=256, placement="auto", marker_ids=[5000]
    )
    assert 7 in positions and 300 in positions


# ----------------------------------------------------------------- item 7
# TensorFold #735: a boundary resumed again after a longer state extended it
# is a branch point; value retention must rank it by its own hits.


def test_resumed_superseded_boundary_is_valued_by_its_hits():
    clock = [0.0]
    apc = APCv2(
        max_size=16, layout_name="prsurvey", retention_policy="value",
        now_fn=lambda: clock[0],
    )
    boundary = list(range(32))
    turn = boundary + list(range(100, 116))
    apc.store(KEY, boundary, _hybrid(32, seed=1), retention_role="committed_prompt_boundary")
    apc.store(KEY, turn, _hybrid(48, seed=2))
    entry = apc._trie.get(KEY, boundary)
    with apc._apc_lock:
        assert apc._boundary_superseded_locked(KEY, boundary, entry)
        before = apc._entry_retention_value_locked(KEY, boundary, entry, clock[0])
    for _ in range(2):
        hit = apc.lookup(KEY, boundary + [200, 201])
        assert hit.cached_tokens == 32
        _close(hit)
    with apc._apc_lock:
        after = apc._entry_retention_value_locked(KEY, boundary, entry, clock[0])
    assert after > 10 * before


# ------------------------------------------------- item 1, through serving


def _wait_until(predicate, seconds=20.0):
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


import pytest  # noqa: E402

from test_apc_hits_hybrid_gdn_self_mtp import (  # noqa: E402
    host,  # noqa: F401 - fixture
    make_adapter,
    run,
    tiny_qwen38_mtp,
)


@pytest.mark.parametrize("mtp", [True, False], ids=["self_mtp", "plain"])
def test_serving_restores_a_deeper_disk_checkpoint_over_a_shallow_resident_hit(
    host, tmp_path, mtp  # noqa: F811
):
    """A shorter conversation's resident boundary must not mask the deeper
    checkpoint of the conversation this request continues, which the idle
    scan moved to disk."""
    import time

    from mlx2 import serving

    model, vocab = tiny_qwen38_mtp()
    deep = [(7 * i + 3) % (vocab - 2) + 1 for i in range(96)]
    shallow = deep[:48]
    follow_up = deep + [(5 * i + 1) % (vocab - 2) + 1 for i in range(24)]
    engine = serving.ServingEngine(
        "tiny", adapter_factory=make_adapter(model, vocab),
        qualification_mode=True, mtp=mtp, max_lanes=1, prefill_step=16,
        cache_dir=str(tmp_path),
    )
    assert engine.ready.wait(60), engine.error
    try:
        run(engine, deep)
        run(engine, shallow)
        apc = engine.apc

        def disk_state():
            with apc._apc_lock:
                return {
                    len(tokens): bool(entry.prompt_cache)
                    for _key, tokens, entry in apc._entry_records_locked()
                }

        # Age the deep conversation's entries past the idle-disk threshold;
        # the worker's own idle scan spills them (MLX streams are per thread).
        with apc._apc_lock:
            for _key, tokens, entry in apc._entry_records_locked():
                if len(tokens) > len(shallow) + 16:
                    entry._apc_last_access_at = time.monotonic() - 10_000
            apc._last_idle_scan = float("-inf")
        assert _wait_until(
            lambda: all(
                not resident
                for length, resident in disk_state().items()
                if length > len(shallow) + 16
            )
        ), disk_state()
        assert disk_state()[len(shallow) - 1] is True
        out, _receipt, job = run(engine, follow_up)
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert int(job.cached_tokens) == len(deep) - 1
    assert counts["apc_deeper_disk_restore_retries"] == 1
    cold = serving.ServingEngine(
        "tiny", adapter_factory=make_adapter(model, vocab),
        qualification_mode=True, mtp=mtp, max_lanes=1, prefill_step=16,
    )
    assert cold.ready.wait(60), cold.error
    try:
        cold_out, _receipt, cold_job = run(cold, follow_up)
    finally:
        cold.close()
    assert int(cold_job.cached_tokens or 0) == 0
    assert out == cold_out


# ------------------------------------------------ item 12, through serving


@pytest.mark.parametrize("mtp", [True, False], ids=["self_mtp", "plain"])
def test_serving_publishes_the_turn_end_boundary_as_a_next_turn_point(
    host, mtp  # noqa: F811
):
    from test_apc_interior_placement import (
        _ChatTemplateTokenizer,
        _chat,
        _chat_engine,
        _words,
    )

    model, vocab = tiny_qwen38_mtp()
    messages = [
        {"role": "system", "content": _words(vocab, 1, 60)},
        {"role": "user", "content": _words(vocab, 2, 20)},
    ]
    template = _ChatTemplateTokenizer(vocab, keeps_think_block=False)
    prompt = template.apply_chat_template(messages, add_generation_prompt=True)
    engine = _chat_engine(model, vocab, keeps_think_block=False, mtp=mtp)
    try:
        _chat(engine, messages)
        apc = engine.apc
        with apc._apc_lock:
            ranks = {
                len(tokens): (
                    getattr(entry, "_apc_retention_role", None),
                    APCv2._entry_retention_rank(entry),
                )
                for _key, tokens, entry in apc._entry_records_locked()
            }
        counts = dict(engine.counts)
    finally:
        engine.close()
    turn_end = len(prompt) - 7
    assert ranks[turn_end] == ("interior_checkpoint", 2), ranks
    assert counts["apc_next_turn_boundaries_published"] == 1


# ----------------------------------------------------------------- item 4
# splash #420: a lane's checkpoint retires only once its successor is
# published; a refused successor leaves the predecessor as the retry point.


def test_rolling_checkpoint_retires_only_after_its_successor_is_published():
    from collections import Counter
    from types import SimpleNamespace

    from mlx2.runtime.state_boundaries import BoundaryPurpose
    from mlx2.serving import ServingEngine

    engine = object.__new__(ServingEngine)
    engine.counts = Counter()
    outcomes = []
    engine._publish_checkpoint = lambda *a, **k: outcomes.pop(0)
    retired = []

    class Apc:
        def retire(self, key, tokens, *, role):
            retired.append((key, tuple(tokens), role))
            return True

    previous = ("k", tuple(range(64)))
    owner = SimpleNamespace(
        request={}, tenant_id=None, approximate_kv_applied=False,
        state_boundaries_published={}, apc_inflight_planned_bytes={},
        uid=1, rolling_checkpoint=previous,
    )
    active = {1: owner}

    class Batch:
        def __init__(self, position):
            self.position = position

        def drain_state_checkpoints(self):
            return [(1, {
                "purpose": int(BoundaryPurpose.ROLLING),
                "tokens": list(range(self.position)),
                "target_cache": [], "covered_tokens": self.position,
            })]

    def publish(position, stored):
        outcomes.append(stored)
        engine._publish_state_checkpoints(
            Batch(position), Apc(), active,
            lambda tenant, scope: "k", lambda job: None, None,
        )

    publish(128, False)
    assert retired == [] and owner.rolling_checkpoint == previous
    publish(128, True)
    assert retired == [("k", tuple(range(64)), "prefill_rolling")]
    assert owner.rolling_checkpoint == ("k", tuple(range(128)))


# ----------------------------------------------------------------- item 9
# vllm #61589 / #52244: with MTP drafting, prefix hits lost a block, and a
# hybrid prompt whose length was an exact multiple of the hash unit got zero
# cached tokens.  mlx2 has no hash grid: the self-MTP boundary sits at P-1
# at every length, including multiples of the prefill chunk.


def test_self_mtp_replay_reaches_p_minus_one_at_chunk_multiples(host):  # noqa: F811
    from mlx2 import serving

    model, vocab = tiny_qwen38_mtp()
    lengths = (32, 33, 47, 48, 64)
    prompts = {
        length: [(3 * i + length) % (vocab - 2) + 1 for i in range(length)]
        for length in lengths
    }
    extended = {
        length: prompt + [(11 * i + 5) % (vocab - 2) + 1 for i in range(5)]
        for length, prompt in prompts.items()
    }

    def engine():
        built = serving.ServingEngine(
            "tiny", adapter_factory=make_adapter(model, vocab),
            qualification_mode=True, mtp=True, max_lanes=1, prefill_step=16,
        )
        assert built.ready.wait(60), built.error
        return built

    warm = engine()
    replay, follow = {}, {}
    try:
        for length, prompt in prompts.items():
            run(warm, prompt)
            _out, receipt, job = run(warm, prompt)
            assert receipt["mtp"]["route"] == "continuous_batched_self_mtp"
            replay[length] = int(job.cached_tokens)
            out, _receipt, job = run(warm, extended[length])
            follow[length] = (int(job.cached_tokens), out)
        counts = dict(warm.counts)
    finally:
        warm.close()
    assert replay == {length: length - 1 for length in lengths}
    assert {length: cached for length, (cached, _) in follow.items()} == {
        length: length - 1 for length in lengths
    }
    assert counts.get("mtp_sidecar_missing_misses", 0) == 0
    cold = engine()
    try:
        for length, tokens in extended.items():
            out, _receipt, job = run(cold, tokens)
            assert int(job.cached_tokens or 0) == 0
            assert out == follow[length][1], length
    finally:
        cold.close()
