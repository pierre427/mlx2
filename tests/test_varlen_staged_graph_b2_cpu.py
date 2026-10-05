"""CPU ownership proof for one physical two-span staged Qwen3 forward."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
import sys

import mlx.core as mx
import pytest

from mlx2.adapters.qwen3_paged_candidate import PackedLane, Qwen3PackedCandidate
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner
from mlx2.runtime.paged_attention_pack import prepare_staged_token_read
from mlx2.runtime.paged_native_atomic_owner import (NativeAtomicRequestOwner,
                                                     complete_staged_read_after_event)
from mlx2.runtime.paged_request_transaction import CandidateRequest
from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation
from mlx2.runtime.paged_native_graph_group import run_research_graph_b2
from mlx2.runtime.generate import BatchGenerator, StopSequenceMatcher
from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend
import mlx2.runtime.qwen3_paged_native_backend as native_backend_module
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
from varlen_staged_graph_request_driver import retained_native_state_diagnostics


PROFILE = TokenKVProfile(2, 128, "float16")


@pytest.fixture(autouse=True)
def cpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class FakeArena:
    def __init__(self, capacity):
        self.plane_bytes = capacity * PROFILE.page_bytes
        self.events = []
        self.stream = mx.default_stream(mx.cpu)

    def validate_sources(self, key, value):
        return (type(key) is mx.array and type(value) is mx.array and
                key.dtype == value.dtype == mx.uint8 and key.ndim == value.ndim == 1)

    def write(self, _key, _value, _offset, _count, epoch):
        self.events.append((epoch, True))
        return mx.array([0], dtype=mx.uint8)

    def poll_completions(self):
        events, self.events = self.events, []
        return events

    def wait_completions(self, _timeout):
        return self.poll_completions()


class FakeModel:
    args = SimpleNamespace(model_type="qwen3", num_experts=0, rope_scaling=None,
                           head_dim=128, num_attention_heads=4, num_key_value_heads=2)
    layers = (object(), object())

    def paged_embed(self, tokens):
        return mx.array(tokens, dtype=mx.float16)

    def paged_project(self, index, hidden, counts, offsets, *, vector_q1_rope=False):
        if vector_q1_rope:
            assert counts == (1, 1)
            self.vector_calls += 1
        rows = len(hidden)
        query = mx.broadcast_to(hidden[:, None, None] / 100, (rows, 4, 128))
        key = mx.broadcast_to(hidden[:, None, None] / 50, (rows, 2, 128))
        value = mx.broadcast_to((hidden + index)[:, None, None] / 10,
                                (rows, 2, 128))
        return query, key, value

    def paged_finish_layer(self, _index, hidden, attended):
        return hidden + attended[:, 0, 0]

    def paged_logits(self, hidden):
        return hidden


@pytest.mark.parametrize("counts", [(1, 1), (32, 127)])
def test_two_branches_share_one_writer_and_one_two_span_read_per_layer(counts):
    pool = PagedKVPool(64)
    arena = FakeArena(pool.capacity)
    writer = PagedKVWriteOwner(pool, arena, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    owners = tuple(NativeAtomicRequestOwner(
        f"r{lane}", tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                          for _ in FakeModel.layers), {},
        supported_planes=("kv",), enabled=True) for lane in (1, 2))
    branches = tuple(owner.begin(CandidateRequest(lane, f"r{lane}", count, ("kv",)))
                     for lane, (owner, count) in enumerate(zip(owners, counts), 1))
    backend = object.__new__(NativeQwen3PagedBackend)
    backend.writer = writer
    backend.read_submissions = 0
    backend.terminal_successes = 0
    physical_spans = []

    def append_staged(self, layers, _keys, _values, row_counts):
        tickets = []
        for layer, rows in zip(layers, row_counts):
            spans = layer.planned_spans(rows)
            chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8)
                           for span in spans)
            tickets.extend(layer.append_staged(chunks, chunks, token_count=rows))
        return tuple(tickets)

    def read_staged(self, use, queries, _tickets, *, scale):
        assert scale > 0 and use.writer is writer
        physical_spans.append(tuple(span.row_count for span in use.plan.spans))
        use.mark_submitted()
        self.read_submissions += 1
        return mx.broadcast_to(mx.ones((len(queries), 1, 1), dtype=mx.float16),
                               queries.shape)

    def drain_staged(self, layers, uses):
        for layer in layers:
            layer.poll_completions()
        proofs = tuple(complete_staged_read_after_event(
            use, (use.lease.epoch, True)) for use in uses)
        self.terminal_successes += len(proofs)
        return proofs

    backend.append_staged = MethodType(append_staged, backend)
    backend.read_staged = MethodType(read_staged, backend)
    backend.drain_staged = MethodType(drain_staged, backend)
    model = FakeModel()
    model.vector_calls = 0
    candidate = Qwen3PackedCandidate(model, backend)
    candidate._vector_q1_rope = counts == (1, 1)
    lanes = tuple(PackedLane((8 + index,) * count, branch.layers)
                  for index, (branch, count) in enumerate(zip(branches, counts)))
    with pytest.raises(ValueError, match="native branch lanes"):
        candidate.forward_staged(lanes, list(branches), permit_candidate=True)
    logits, receipt = candidate.forward_staged(lanes, branches, permit_candidate=True)
    assert model.vector_calls == (len(model.layers) if counts == (1, 1) else 0)
    assert getattr(candidate, "_vector_q1_rope_calls", 0) == model.vector_calls
    assert tuple(logits.shape) == (sum(counts),)
    mx.eval(logits)
    assert float(logits[0].item()) != float(logits[-1].item())
    assert physical_spans == [counts, counts]
    assert receipt["packed_lanes"] == 2 and receipt["native_read_calls"] == 2
    assert receipt["terminal_successes"] == 2 and not receipt["selected"]
    for branch, count in zip(branches, counts):
        branch.prepare(count).publish()
    for owner, count in zip(owners, counts):
        with owner.snapshot() as public:
            assert public.offset == count and public.generation == 1
        owner.close()
        assert owner.reap_retired() == 2
        assert owner.fully_retired
    assert not writer.pending_epochs and writer.ledger.pending_count == 0


def test_native_dependency_join_refuses_duplicate_read_terminal(monkeypatch):
    pool = PagedKVPool(8)
    arena = FakeArena(pool.capacity)
    writer = PagedKVWriteOwner(pool, arena, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    layer = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    spans = layer.planned_spans(1)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
    tickets = layer.append_staged(chunks, chunks, token_count=1)
    use = prepare_staged_token_read((layer,), (1,), query_heads=4,
                                    permit_candidate=True)
    backend = object.__new__(NativeQwen3PagedBackend)
    backend.writer = writer
    backend.timeout_s = 0.1
    backend._failed = False
    backend.read_submissions = 0
    backend.terminal_successes = 0
    backend.staged_read_spans = []
    observed_dependencies = []

    def fake_read(packed, _arena, query, dependency, *, scale, permit_candidate):
        assert packed is use and scale > 0 and permit_candidate
        mx.eval(dependency)
        observed_dependencies.append((dependency.shape, dependency.dtype,
                                      dependency.tolist()))
        packed.mark_submitted()
        return mx.zeros(query.shape, dtype=mx.float16)

    monkeypatch.setattr(native_backend_module, "native_paged_attention_read_fp16", fake_read)
    query = mx.zeros((1, 4, 128), dtype=mx.float16)
    backend.read_staged(use, query, tickets, scale=128 ** -0.5)
    assert observed_dependencies == [((1,), mx.uint8, [1])]
    monkeypatch.setattr(native_backend_module, "poll_native_paged_read_events",
                        lambda _backend: ((use.lease.epoch, True),
                                          (use.lease.epoch, True)))
    with pytest.raises(RuntimeError, match="duplicate staged read"):
        backend.drain_staged((layer,), (use,))
    assert writer.poisoned and backend._failed
    assert use.state == "closed" and use.terminal_succeeded is True
    assert layer.offset == 0  # No private KV publication on ambiguous proof.


@pytest.mark.parametrize("malformed", [False, True])
def test_direct_grouped_fence_is_exact_write_output_and_retains_terminal_proof(
        monkeypatch, malformed):
    pool = PagedKVPool(8)
    arena = FakeArena(pool.capacity)
    dependency = mx.array([1, 1] if malformed else [1], dtype=mx.uint8)

    def grouped_write(_keys, _values, _pages, _slots, _heads, _dim, epoch):
        arena.events.append((epoch, True))
        return dependency

    arena.grouped_q1_write = grouped_write
    writer = PagedKVWriteOwner(pool, arena, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    keys = values = mx.zeros((2, PROFILE.kv_heads, PROFILE.head_dim),
                             dtype=mx.float16)
    destinations = tuple(owner.stage_grouped_q1(keys, values) for owner in owners)
    ticket = writer.submit_grouped_q1_write(
        tuple(handle for handle, _slot in destinations),
        tuple(slot for _handle, slot in destinations), keys=keys, values=values,
        kv_heads=PROFILE.kv_heads, dim=PROFILE.head_dim)
    assert ticket.grouped_q1 is True and ticket.dependency is dependency
    for owner in owners:
        owner.adopt_grouped_q1(ticket)
    use = prepare_staged_token_read(owners, (1, 1), query_heads=4,
                                    permit_candidate=True)
    backend = object.__new__(NativeQwen3PagedBackend)
    backend.writer = writer
    backend._failed = False
    backend.profile_host = False
    backend.direct_grouped_fence = True
    backend.read_submissions = 0
    backend.staged_read_spans = []
    observed = []

    def fake_read(packed, _arena, query, fence, *, scale, permit_candidate):
        assert packed is use and scale > 0 and permit_candidate
        observed.append(fence)
        packed.mark_submitted()
        return mx.zeros(query.shape, dtype=mx.float16)

    monkeypatch.setattr(native_backend_module, "native_paged_attention_read_fp16", fake_read)
    monkeypatch.setattr(mx, "depends", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(AssertionError("joined fence was built")))
    query = mx.zeros((2, 4, 128), dtype=mx.float16)
    if malformed:
        with pytest.raises(ValueError, match="one native byte"):
            backend.read_staged(use, query, (ticket,), scale=128 ** -0.5)
        assert use.state == "closed" and writer.poisoned and backend._failed
        assert all(owner.offset == 0 for owner in owners)
        assert writer.pending_epochs  # Submitted write lease remains pinned.
        return
    backend.read_staged(use, query, (ticket,), scale=128 ** -0.5)
    assert observed == [dependency]
    assert backend.direct_grouped_fence_reads == 1
    writer.poll_completions()
    for owner in owners:
        owner.poll_completions()
        assert owner.offset == 1
    complete_staged_read_after_event(use, (use.lease.epoch, True))
    assert not writer.pending_epochs and writer.ledger.pending_count == 0


@pytest.mark.parametrize("reuse_private_tail", [False, True])
@pytest.mark.parametrize("serving", [False, True])
@pytest.mark.parametrize("direct_fence", [False, True])
def test_research_group_samples_two_responses_after_one_packed_forward(reuse_private_tail,
                                                                        serving,
                                                                        direct_fence):
    events = []

    class Branch:
        layers = (SimpleNamespace(offset=2), SimpleNamespace(offset=2))

        def prepare(self, rows):
            events.append(("prepare", rows))
            return self

        def publish(self):
            events.append(("publish",))

        def rollback(self):
            events.append(("rollback",))

    owners = []
    for index in range(2):
        owner = object.__new__(NativeAtomicRequestOwner)
        owner.supported_planes = ("kv",)
        owner._enabled = True
        owner._reuse_private_tail = reuse_private_tail

        @contextmanager
        def snapshot(index=index):
            try:
                yield SimpleNamespace(revision="r1", offset=1, generation=1,
                                      layer_owners=(object(), object()))
            finally:
                events.append(("snapshot_closed", index))

        def begin(_request):
            assert events.count(("snapshot_closed", 0)) == int(reuse_private_tail)
            assert events.count(("snapshot_closed", 1)) == int(reuse_private_tail)
            return Branch()

        owner.snapshot = snapshot
        owner.begin = begin
        owner.reap_retired = lambda: events.append(("reap",))
        owners.append(owner)
    backend = SimpleNamespace(read_submissions=0, terminal_successes=0,
                              staged_read_spans=[],
                              direct_grouped_fence=direct_fence,
                              direct_grouped_fence_reads=0)
    counters = {"grouped_q1_writes": 0, "native_write_dispatches": 0,
                    "q1_tile_dispatches": 0, "q1_metadata_dispatches": 0,
                    "grouped_write_async_evals": 0,
                    "staged_read_async_evals": 0,
                    "deferred_q1_write_roots": 0,
                    "deferred_q1_read_roots": 0,
                    "deferred_q1_final_evals": 0,
                    "deferred_q1_failure_flushes": 0}
    counters.update(q1_gather_dispatches=0, stock_sdpa_graph_calls=0)
    counters.update({f"q1_stripe_dispatches_{stripes}": 0
                     for stripes in (8, 16, 32)})
    backend.profile_counters_snapshot = lambda: dict(counters)
    candidate = Qwen3PackedCandidate(SimpleNamespace(layers=(object(), object())), backend)
    candidate._research_staged_graph = True
    candidate._serving_b2 = serving
    candidate._b2_profile_id = "cpu-b2"

    def forward(lanes, branches, *, permit_candidate):
        assert permit_candidate and len(lanes) == len(branches) == 2
        backend.read_submissions += 2
        backend.terminal_successes += 2
        backend.staged_read_spans.extend((2, 2))
        if serving:
            counters["grouped_q1_writes"] += 2
            counters["q1_tile_dispatches"] += 2
            counters["grouped_write_async_evals"] += 2
            counters["staged_read_async_evals"] += 2
            if direct_fence:
                backend.direct_grouped_fence_reads += 2
        return mx.array([[1, 2, 3], [3, 2, 1]], dtype=mx.float16), {"packed_lanes": 2}

    candidate.forward_staged = forward
    continuations = tuple(NativeQwen3Continuation(
        uid=uid, revision="r1", prompt_tokens=(7,), first_logits=None,
        owner=owner, candidate=candidate, maximum=2,
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        processors=[], matcher=StopSequenceMatcher(), research_only=not serving)
        for uid, owner in zip((1, 2), owners))
    for continuation in continuations:
        continuation._pending_token = 8
    responses = run_research_graph_b2(continuations)
    assert [response.uid for response in responses] == [1, 2]
    assert [response.token for response in responses] == [2, 0]
    assert all(response.execution_width == 2 and
               response.mtp_receipt["research_executed"] == (not serving) and
               response.mtp_receipt["published_layer_offsets"] == [2, 2] and
               response.mtp_receipt["selected"] == serving and
               response.mtp_receipt["observed_used"] == serving for response in responses)
    if serving:
        assert all(response.mtp_receipt["route"] == "native_qwen3_paged_b2" and
                   response.mtp_receipt["admission_profile"] == "cpu-b2" and
                   response.mtp_receipt["direct_grouped_fence_reads"] ==
                   (2 if direct_fence else 0) and
                   response.mtp_receipt["physical_dispatches"] == {
                       "grouped_q1_writes": 2, "native_write_dispatches": 0,
                       "q1_tile_dispatches": 2} for response in responses)
    assert events.count(("prepare", 1)) == events.count(("publish",)) == 2
    assert events.count(("reap",)) == 2
    # Both terminal responses can close their owners before the harness runs
    # diagnostics. The emitted receipts are the sole retained evidence.
    for owner, continuation in zip(owners, continuations):
        continuation.closed = True
        owner.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("closed owner"))
    retained = {response.uid: [response.mtp_receipt] for response in responses}
    assert retained_native_state_diagnostics(retained, (1, 2)) == [
        {"pre_step_reader_offset": 1, "published_layer_offsets": [2, 2]},
        {"pre_step_reader_offset": 1, "published_layer_offsets": [2, 2]},
    ]


@pytest.mark.parametrize("fault", ["sampler", "second_publish"])
def test_graph_fault_suppresses_both_responses_and_retains_owners(monkeypatch, fault):
    events = []
    published = 0
    sampler_calls = 0
    releasable = {"value": False}

    class Branch:
        layers = (SimpleNamespace(offset=2), SimpleNamespace(offset=2))

        def prepare(self, rows):
            assert rows == 1
            return self

        def publish(self):
            nonlocal published
            published += 1
            events.append(("publish", published))
            if fault == "second_publish" and published == 2:
                raise RuntimeError("second owner publish failed")

        def rollback(self):
            events.append(("rollback",))

    owners = []
    for _ in range(2):
        owner = object.__new__(NativeAtomicRequestOwner)
        owner.supported_planes = ("kv",)
        owner._enabled = True
        owner.snapshot = lambda: nullcontext(SimpleNamespace(
            revision="r1", offset=1, generation=1,
            layer_owners=(object(), object())))
        owner.begin = lambda _request: Branch()
        owner.close = lambda: events.append(("close",))
        owner.reap_retired = lambda: 0
        owner.reap_quarantine = lambda: 0
        owners.append(owner)
    backend = SimpleNamespace(read_submissions=0, terminal_successes=0,
                              staged_read_spans=[])
    candidate = Qwen3PackedCandidate(SimpleNamespace(layers=(object(), object())), backend)
    candidate._research_staged_graph = True

    def forward(_lanes, _branches, *, permit_candidate):
        assert permit_candidate
        backend.read_submissions += 2
        backend.terminal_successes += 2
        backend.staged_read_spans.extend((2, 2))
        return mx.array([[1, 2, 3], [3, 2, 1]], dtype=mx.float16), {"packed_lanes": 2}

    def sampler(logprobs):
        nonlocal sampler_calls
        sampler_calls += 1
        if fault == "sampler":
            raise RuntimeError("sampler failed after publish")
        return mx.argmax(logprobs, axis=-1)

    candidate.forward_staged = forward
    continuations = tuple(NativeQwen3Continuation(
        uid=uid, revision="r1", prompt_tokens=(7,), first_logits=None,
        owner=owner, candidate=candidate, maximum=2, sampler=sampler,
        processors=[], matcher=StopSequenceMatcher(), research_only=True)
        for uid, owner in zip((1, 2), owners))
    for continuation in continuations:
        continuation._pending_token = 8
    monkeypatch.setattr(NativeQwen3Continuation, "can_release",
                        property(lambda self: releasable["value"]))
    generator = BatchGenerator(None)
    try:
        generator._native_continuations.update(
            (lane.uid, lane) for lane in continuations)
        prompt, responses = generator.next()
        assert not prompt and not responses
        assert len(generator.take_lane_failures()) == 2
        assert not generator._native_continuations
        assert all(lane.closed and lane in generator._native_retiring
                   for lane in continuations)
        assert events.count(("close",)) == 2
        assert not generator._native_prompt_responses
        assert not generator._unprocessed_sequences
        prompt, responses = generator.next()
        assert not prompt and not responses  # No ordinary replay or later partial output.
        if fault == "sampler":
            assert published == 2 and sampler_calls == 1
        else:
            assert published == 2 and sampler_calls == 0
    finally:
        releasable["value"] = True
        generator._reap_native_retiring()
        generator.close()
