"""Qwen3.8 device RoPE position + device_round.pipeline (spec-round DC-5), CPU only.

A pipelined round (DC-5) is verified while the round ahead of it is still in
flight: its device-offset KV caches carry slack, so ``offset`` is a bound, not
the fill. Qwen3.8's full attention takes its rotary offset from
``DeviceOffsetKVCache.rope_offset`` (the device fill ``pos`` while there is
slack) and the target declares ``device_commit_kv_device_position``, which the
pipeline gate admits. On a tiny Qwen3.8-27B-shaped target
(``tests/test_qwen38_dflash2_cpu.tiny_pair``, GDN + full attention, DFlash2
drafter) this pins:

* ``mx.fast.rope`` with an int32 array offset is bitwise the int offset;
* a target forward over slack KV (pipelined state) is bitwise the serial
  DC-4 forward at the same read width, and the old bound-offset rotation is not;
* the declaration and the gate: admitted as declared, refused (serial round,
  ``position_encoding``) for class-swapped attention, another rotary module, an
  installed row-exact executor;
* engine rounds: pipelined and lagged-composition rounds are token-identical to
  the serial DC-4 round and to ordinary greedy, with acceptance varied round by
  round (rejections leave slack), through stops and budget ends, and under
  ``fixed_shape``; the delivered target caches are bitwise; the serial
  engine round still rotates at the host int.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from test_device_commit_raon_cpu import _arrays, _same_arrays
from test_qwen38_dflash2_cpu import VOCAB, ordinary_greedy, tiny_pair, tiny_target

from mlx2.runtime import external_speculative as es
from mlx2.runtime.drafters import dflash2
from mlx2.runtime.drafters.device_ring_kv import DeviceRingKVCache
from mlx2.runtime.models import qwen38_27b
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.device_offset_kv import DeviceOffsetKVCache
from mlx2.runtime.speculative_device_pipeline import SKIPS

mx.set_default_device(mx.cpu)

PROMPT = [3, 14, 15, 9, 26, 5, 35, 8]
DC4 = {"lazy_draft": True, "decide": True, "device_commit": True, "lagged_width": True}
PIPE = {**DC4, "pipeline": True}
DC4_PROC = {**DC4, "processors": True}
PIPE_COMP = {**PIPE, "processors": True, "lagged_composition": True}
NUM_DRAFT = 3
_DRAFT_LAZY = dflash2.DFlash2DraftModel.draft_lazy


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _np(array):
    return np.asarray(array.astype(mx.float32) if array.dtype == mx.bfloat16 else array)


# -- the device RoPE position ---------------------------------------------------------

@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_fast_rope_takes_an_array_offset_bitwise(dtype):
    x = mx.random.normal((1, 2, 5, 8)).astype(dtype)
    rope = nn.RoPE(4, traditional=False, base=10000000.0)
    for offset in (0, 1, 7, 513, 70000):
        want = rope(x, offset=offset)
        got = rope(x, offset=mx.array([offset], dtype=mx.int32))
        assert np.array_equal(_np(want), _np(got)), offset


def test_rope_offset_is_the_fill_in_every_mode():
    cache = DeviceOffsetKVCache()
    keys = mx.zeros((1, 2, 6, 8))
    cache.update_and_fetch(keys, keys)
    assert cache.rope_offset == 6 and isinstance(cache.rope_offset, int)  # host mode
    cache.enter_device_mode()
    assert cache.rope_offset == 6 and isinstance(cache.rope_offset, int)  # exact bound
    cache.update_and_fetch(keys[:, :, :4], keys[:, :, :4])
    cache.device_commit(mx.array([2], dtype=mx.int32))
    cache.settle(8, 2)  # the pipelined state: 2 slack rows past the fill
    assert cache.offset == 10
    position = cache.rope_offset
    assert isinstance(position, mx.array) and position.tolist() == [8]
    cache.observe(8)
    assert cache.rope_offset == 8 and isinstance(cache.rope_offset, int)


def _round_logits(model, rounds, *, pipelined):
    """Device rounds over a tiny Qwen3.8 target: serial DC-4 (observe after each
    commit) or the pipelined state (the bound keeps the previous round's slack).

    Every round reads ``lagged_width``'s shared base (previous start + rows), so
    both arms attend over the same columns; only the position source differs.
    """
    caches = [DeviceOffsetKVCache() if type(c) is KVCache else c for c in model.make_cache()]
    model(mx.array([PROMPT]), cache=caches)
    kv = [c for c in caches if type(c) is DeviceOffsetKVCache]
    gdn = [c for c in caches if type(c) is ArraysCache]
    for c in kv:
        c.enter_device_mode()
    length, base, out = len(PROMPT), len(PROMPT), []
    for rows, kept in rounds:
        for c in kv:
            c.set_read_base(base)
        for c in gdn:
            c.start_speculation()
        logits = model(mx.array([rows]), cache=caches)
        mx.eval(logits)
        out.append(_np(logits))
        count = mx.array([kept], dtype=mx.int32)
        for c in kv:
            c.device_commit(count)
        for c in gdn:
            c.device_rollback(count)
        base = length + len(rows)
        length += kept
        for c in kv:
            if pipelined:
                # The host has not read this round: its rows past the fill stay slack.
                c.settle(length, base - length) if base > length else c.observe(length)
            else:
                c.observe(length)
    return out


ROUNDS = [([11, 12, 13, 14], 1), ([20, 21, 22, 23], 3), ([30, 31, 32, 33], 4),
          ([40, 41, 42, 43], 2), ([50, 51, 52, 53], 1), ([60, 61, 62, 63], 4)]


def test_a_slack_forward_is_the_serial_forward_bitwise():
    model = tiny_target()
    serial = _round_logits(model, ROUNDS, pipelined=False)
    piped = _round_logits(model, ROUNDS, pipelined=True)
    for a, b in zip(serial, piped, strict=True):
        assert np.array_equal(a, b)


def test_the_bound_offset_rotates_at_the_wrong_position(monkeypatch):
    """Sensitivity: rotating at ``offset`` (the slack bound) changes the logits."""
    model = tiny_target()
    serial = _round_logits(model, ROUNDS, pipelined=False)
    monkeypatch.setattr(DeviceOffsetKVCache, "rope_offset", property(lambda self: self.offset))
    piped = _round_logits(model, ROUNDS, pipelined=True)
    assert np.array_equal(serial[0], piped[0])  # nothing in flight yet
    assert any(not np.array_equal(a, b) for a, b in zip(serial[1:], piped[1:]))


def test_host_caches_and_exact_bounds_keep_the_int_offset(monkeypatch):
    """Nothing changes off the pipeline: every rotation off slack reads an int."""
    seen = []
    real = nn.RoPE.__call__

    def spy(self, x, offset=0):
        seen.append(type(offset))
        return real(self, x, offset=offset)

    monkeypatch.setattr(nn.RoPE, "__call__", spy)
    model = tiny_target()
    ordinary_greedy(model, PROMPT, 3)
    _round_logits(model, ROUNDS, pipelined=False)
    assert seen and all(kind is int for kind in seen)
    seen.clear()
    _round_logits(model, ROUNDS, pipelined=True)
    assert mx.array in seen


# -- the declaration and the gate -----------------------------------------------------

def test_qwen38_declares_the_device_position():
    target = tiny_target()
    assert target.device_commit_kv_device_position is True
    assert getattr(target, "device_commit_kv_position_free", False) is False


def test_the_declaration_refuses_attention_that_reads_offset_itself():
    target = tiny_target()
    attn = next(l.self_attn for l in target.language_model.model.layers if not l.is_linear)

    class Swapped(qwen38_27b.Qwen3NextAttention):
        pass

    original = type(attn)
    attn.__class__ = Swapped  # a route class swap (QSA, row-exact)
    assert target.device_commit_kv_device_position is False
    attn.__class__ = original
    rope = attn.rope
    attn.rope = qwen38_27b.initialize_rope(4, base=10000.0, traditional=False,
                                           scaling_config={"type": "dynamic", "factor": 2.0},
                                           max_position_embeddings=512)
    assert target.device_commit_kv_device_position is False
    attn.rope = rope
    object.__setattr__(attn, "_attend", lambda *a, **k: None)
    assert target.device_commit_kv_device_position is False
    del attn.__dict__["_attend"]
    object.__setattr__(target, "_mlx2_qwen_hybrid_row_exact", object())
    assert target.device_commit_kv_device_position is False
    del target.__dict__["_mlx2_qwen_hybrid_row_exact"]
    assert target.device_commit_kv_device_position is True


# -- engine rounds --------------------------------------------------------------------

def _oracle(monkeypatch, sequence):
    """Drafts = the target's continuation, cut at a position-dependent depth.

    The anchor's position is read from the drafter's committed context (a
    ring's ``pos`` plus its pending count, or the host offset plus the tail),
    never from the host history: a pipelined round is dispatched before the
    round ahead of it is booked. Every arm sees the same drafts at the same
    position (``test_device_commit_raon_cpu._oracle``).
    """
    def position(hidden, cache):
        first = cache[0]
        rows = int(hidden.shape[1])
        if isinstance(first, DeviceRingKVCache):
            pending = first.pending_count
            return int(first.pos.item()) + (int(pending.item()) if rows and pending is not None else 0)
        return int(first.offset) + rows

    def drafted(self, anchor, hidden, cache, count, *args, **kwargs):
        start = position(hidden, cache) + 1
        tokens = _DRAFT_LAZY(self, anchor, hidden, cache, count, *args, **kwargs)
        want = list(sequence[start:start + count])
        if len(want) < count:
            return tokens
        cut = start % (count + 1)
        if cut < count:
            want[cut] = (want[cut] + 1) % VOCAB
        return mx.array(want, dtype=mx.int32)

    monkeypatch.setattr(dflash2.DFlash2DraftModel, "draft_lazy", drafted)


def _decode(policy, *, steps=40, oracle=None, stops=None, composed=False, prompt=PROMPT):
    target, draft = tiny_pair()
    reference = ordinary_greedy(target, prompt, steps + NUM_DRAFT + 2)
    if oracle is not None:
        _oracle(oracle, list(prompt) + reference)
    if composed:
        from mlx2.adapters.external_draft_policy import DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION
        from mlx2.runtime.proposal_composition import ComposedDraftModel

        draft = ComposedDraftModel(draft, dict(DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION))
    engine = es.ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="test", num_draft=NUM_DRAFT,
        prefill_step_size=4, device_round=policy,
    )
    if stops is not None:
        engine.stops = set(stops)
    try:
        uid = engine.insert([list(prompt)], max_tokens=[steps],
                            sampling_configs=[{"sampling_temp": 0.0}])[0]
        got, rng, final = [], None, None
        for _ in range(400):
            lane = engine.lanes.get(uid)
            if lane is not None:
                rng = lane.rng.snapshot()
            _, responses = engine.next()
            for response in responses:
                if response.uid == uid:
                    got.append(response.token)
                    if response.finish_reason:
                        final = response
            if not engine.lanes:
                break
        return got, rng, final, dict(engine.scheduler_stats), reference[:steps]
    finally:
        engine.close()


def _assert_like_dc4(serial, piped):
    (a, a_rng, a_final, _, reference), (b, b_rng, b_final, stats, _) = serial, piped
    assert b == a and b_rng == a_rng
    assert a_final.finish_reason == b_final.finish_reason
    assert a_final.all_tokens == b_final.all_tokens
    for cache in b_final.prompt_cache:
        assert type(cache) is not DeviceOffsetKVCache
    _same_arrays(_arrays(a_final.prompt_cache), _arrays(b_final.prompt_cache))
    assert SKIPS not in stats or "position_encoding" not in stats[SKIPS]
    return stats


@pytest.mark.parametrize("oracle", [False, True], ids=["random-drafter", "oracle-drafts"])
def test_pipelined_rounds_equal_dc4_and_ordinary_greedy(monkeypatch, oracle):
    serial = _decode(DC4, oracle=monkeypatch if oracle else None)
    piped = _decode(PIPE, oracle=monkeypatch if oracle else None)
    stats = _assert_like_dc4(serial, piped)
    assert piped[0] == piped[4]  # ordinary greedy
    assert stats["external_device_round_pipeline_rounds"] > 5
    assert stats["external_device_round_pipeline_followers"] > 3
    if oracle:
        # Acceptance varies round by round: rejections leave slack rows.
        assert piped[2].speculative_receipt["accepted"] > 10


@pytest.mark.parametrize("steps", [7, 12, 13, 19])
def test_budget_ends(monkeypatch, steps):
    serial = _decode(DC4, oracle=monkeypatch, steps=steps)
    piped = _decode(PIPE, oracle=monkeypatch, steps=steps)
    _assert_like_dc4(serial, piped)
    assert piped[0] == piped[4]


def test_a_stop_discards_the_following_round(monkeypatch):
    target = tiny_target()
    stop = ordinary_greedy(target, PROMPT, 40)[17]
    serial = _decode(DC4, oracle=monkeypatch, stops=[stop])
    piped = _decode(PIPE, oracle=monkeypatch, stops=[stop])
    stats = _assert_like_dc4(serial, piped)
    assert piped[0][-1] == stop and piped[2].finish_reason == "stop"
    assert stats["external_device_round_pipeline_discards"] == 1


def test_an_undeclared_target_keeps_the_serial_round(monkeypatch):
    serial = _decode(DC4, oracle=monkeypatch)
    monkeypatch.setattr(qwen38_27b.Model, "device_commit_kv_device_position", False)
    gated = _decode(PIPE, oracle=monkeypatch)
    assert gated[0] == serial[0] == serial[4]
    stats = gated[3]
    assert stats[SKIPS].get("position_encoding", 0) > 0
    assert stats.get("external_device_round_pipeline_rounds", 0) == 0


@pytest.mark.parametrize("presence", [False, True], ids=["composed", "composed-presence"])
def test_lagged_composition_pipelines_a_composed_qwen38_lane(monkeypatch, presence):
    """The default composition, pipelined with its lagged plan (DC-6)."""
    from mlx2.runtime import sample_utils

    if presence:
        real = sample_utils.make_logits_processors
        original_insert = es.ExternalDraftBatchGenerator.insert

        def insert(self, prompts, *args, **kwargs):
            kwargs["logits_processors"] = [real(presence_penalty=1.5, presence_context_size=20,
                                                penalty_generation_start=len(PROMPT))]
            return original_insert(self, prompts, *args, **kwargs)

        monkeypatch.setattr(es.ExternalDraftBatchGenerator, "insert", insert)
    serial = _decode(DC4_PROC, composed=True, steps=30)
    piped = _decode(PIPE_COMP, composed=True, steps=30)
    assert piped[0] == serial[0]
    assert piped[1] == serial[1]
    if not presence:
        assert piped[0] == piped[4]
    stats = piped[3]
    assert stats["external_device_round_pipeline_rounds"] > 10
    assert stats["external_device_round_pipeline_followers"] > 10
    assert stats[SKIPS].get("composition", 0) == 0
    assert stats[SKIPS].get("position_encoding", 0) == 0
    assert stats.get("external_composed_external_rounds", 0) > 0


@pytest.mark.parametrize("steps", [12, 19, 40])
def test_fixed_shape_pipelined_rounds_equal_fixed_shape_dc4(monkeypatch, steps):
    serial = _decode({**DC4, "fixed_shape": True}, oracle=monkeypatch, steps=steps)
    piped = _decode({**PIPE, "fixed_shape": True}, oracle=monkeypatch, steps=steps)
    stats = _assert_like_dc4(serial, piped)
    assert piped[0] == piped[4]
    assert stats["external_device_round_pipeline_rounds"] > 0
    assert stats["external_device_round_fixed_shape_rounds"] > 0


def _spy_rope_offsets(monkeypatch):
    seen = []
    real = qwen38_27b._rope_offset

    def spy(cache):
        value = real(cache)
        seen.append(type(value))
        return value

    monkeypatch.setattr(qwen38_27b, "_rope_offset", spy)
    return seen


@pytest.mark.parametrize("composed", [False, True], ids=["plain", "composed"])
def test_the_serial_engine_round_rotates_at_an_int(monkeypatch, composed):
    """Off the pipeline the engine's serial DC-4 rounds still rotate at the host int."""
    seen = _spy_rope_offsets(monkeypatch)
    _decode(DC4_PROC if composed else DC4, oracle=None if composed else monkeypatch,
            composed=composed)
    assert seen and all(kind is int for kind in seen)
    seen.clear()
    _decode(PIPE, oracle=monkeypatch)
    assert mx.array in seen and int in seen
