"""CPU fake proof for the default-off Qwen3 packed request boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters.qwen3_paged_candidate import PackedLane, Qwen3PackedCandidate
from mlx2.runtime.paged_attention_metal import paged_attention_cpu_reference
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner

PROFILE = TokenKVProfile(2, 128, "float16")


class FakeArena:
    def __init__(self, capacity):
        self.plane_bytes = capacity * PROFILE.page_bytes
        self.keys = bytearray(self.plane_bytes)
        self.values = bytearray(self.plane_bytes)
        self.events = []

    def validate_sources(self, key, value):
        return (type(key) is np.ndarray and type(value) is np.ndarray and
                key.dtype == value.dtype == np.uint8 and key.ndim == value.ndim == 1)

    def write(self, key, value, offset, byte_count, epoch):
        self.keys[offset:offset + byte_count] = key.tobytes()
        self.values[offset:offset + byte_count] = value.tobytes()
        self.events.append((epoch, True))
        return object()

    def poll_completions(self):
        events, self.events = self.events, []
        return events


class FakeModel:
    args = SimpleNamespace(model_type="qwen3", num_experts=0, rope_scaling=None,
                           head_dim=128, num_attention_heads=4, num_key_value_heads=2)
    layers = (object(), object())

    def paged_embed(self, tokens):
        return np.asarray(tokens, dtype=np.float16)

    def paged_project(self, index, hidden, counts, offsets):
        rows = len(hidden)
        q = np.broadcast_to(hidden[:, None, None] / 100, (rows, 4, 128)).copy()
        k = np.broadcast_to(hidden[:, None, None] / 50, (rows, 2, 128)).copy()
        v = np.broadcast_to((hidden + index)[:, None, None] / 10,
                            (rows, 2, 128)).copy()
        return q, k, v

    def paged_finish_layer(self, index, hidden, attended):
        return hidden + attended[:, 0, 0]

    def paged_logits(self, hidden):
        return hidden


class FakeBackend:
    def __init__(self, arena, capacity):
        self.arena, self.capacity = arena, capacity
        self.reads = []

    def append_completed(self, owners, keys, values, row_counts):
        begin = 0
        for owner, count in zip(owners, row_counts):
            spans = owner.planned_spans(count)
            chunks = []
            for plane in (keys, values):
                chunks.append(tuple(np.frombuffer(
                    plane[begin + span.source_token_offset:
                          begin + span.source_token_offset + span.token_count,
                          span.kv_head].tobytes(), dtype=np.uint8).copy()
                    for span in spans))
            owner.append(*chunks, token_count=count)
            assert owner.poll_completions()
            begin += count

    def read_completed(self, use, queries, *, scale):
        use.mark_submitted()
        try:
            k = np.frombuffer(self.arena.keys, dtype=np.float16).reshape(
                self.capacity, 2, 64, 128)
            v = np.frombuffer(self.arena.values, dtype=np.float16).reshape(
                self.capacity, 2, 64, 128)
            result = paged_attention_cpu_reference(use.plan, queries, k, v, scale=scale)
            self.reads.append(tuple(span.row_count for span in use.plan.spans))
            return result
        finally:
            use.complete_after_proof()


def fixture():
    pool = PagedKVPool(8)
    arena = FakeArena(pool.capacity)
    writer = PagedKVWriteOwner(pool, arena, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    lanes = tuple(PackedLane(tokens, tuple(
        PagedKVTokenOwner(writer, PROFILE, permit_candidate=True) for _ in range(2)))
        for tokens in ((2, 3, 4), (7,)))
    return pool, lanes, FakeBackend(arena, pool.capacity)


def test_unequal_packed_rows_publish_each_layer_and_close_read_pins():
    pool, lanes, backend = fixture()
    logits, receipt = Qwen3PackedCandidate(FakeModel(), backend).forward(
        lanes, permit_candidate=True)
    assert logits.shape == (4,)
    assert backend.reads == [(3, 1), (3, 1)]
    assert [[owner.offset for owner in lane.layers] for lane in lanes] == [[3, 3], [1, 1]]
    assert receipt["selected"] and not receipt["qualified"] and not receipt["observed_used"]
    assert pool.free_count == 4  # Four private layer/sequence pages; no read pins remain.


def test_disabled_and_unsupported_routes_refuse_before_writes():
    _, lanes, backend = fixture()
    candidate = Qwen3PackedCandidate(FakeModel(), backend)
    with pytest.raises(RuntimeError, match="explicit enablement"):
        candidate.forward(lanes)
    with pytest.raises(ValueError, match="text only"):
        candidate.forward(lanes, permit_candidate=True,
                          requested_capabilities=frozenset({"text", "vision"}))
    with pytest.raises(ValueError, match="one owner per layer"):
        candidate.forward((PackedLane((1,), lanes[0].layers[:1]),), permit_candidate=True)
    assert all(owner.offset == 0 for lane in lanes for owner in lane.layers)
    model = FakeModel()
    model.args = SimpleNamespace(**{**vars(FakeModel.args), "model_type": "qwen3_moe"})
    with pytest.raises(ValueError, match="dense full-attention Qwen3"):
        Qwen3PackedCandidate(model, backend).forward(lanes, permit_candidate=True)


def test_backend_must_publish_before_read():
    _, lanes, backend = fixture()
    backend.append_completed = lambda *args: None
    with pytest.raises(RuntimeError, match="before KV publication"):
        Qwen3PackedCandidate(FakeModel(), backend).forward(lanes, permit_candidate=True)


def test_layer_offset_mismatch_refuses_before_first_write():
    _, lanes, backend = fixture()
    lanes[0].layers[1]._accepted_tokens = 1
    with pytest.raises(ValueError, match="layer offsets"):
        Qwen3PackedCandidate(FakeModel(), backend).forward(lanes, permit_candidate=True)
    assert all(owner.offset == 0 for lane in lanes for owner in lane.layers[:1])


def test_prepared_read_is_aborted_if_backend_raises_before_submit():
    pool, lanes, backend = fixture()

    def fail_read(use, queries, *, scale):
        raise RuntimeError("read refused")

    backend.read_completed = fail_read
    with pytest.raises(RuntimeError, match="read refused"):
        Qwen3PackedCandidate(FakeModel(), backend).forward(lanes, permit_candidate=True)
    assert pool.free_count == 6  # Two accepted owner pages, no stranded read pin.
