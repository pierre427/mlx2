"""Default-dry-run, bounded multi-page paged-KV ownership diagnostic.

--cpu-fake exercises host state with a bytearray backend. --execute-gpu is an
opt-in, one-stream M5 native byte-copy cell; neither mode selects serving.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

SCHEMA = "mlx2.varlen-multipage-gate.v1"
CASES = (63, 64, 65)


def dry_run() -> dict:
    return {
        "schema": SCHEMA, "mode": "dry-run", "gpu_executed": False,
        "planned_boundary_bytes": list(CASES),
        "native_scope": ["same-stream two-page K/V write-read dependency",
                         "terminal successful callbacks", "generation reuse after reader and writer retirement"],
        "cpu_fake_scope": ["63/64/65-token metadata boundaries", "unrelated lane churn",
                           "shared-tail COW refusal", "failed terminal event retirement"],
        "not_proven": ["native command-buffer failure retirement", "native COW byte copy",
                       "cross-stream ordering", "serving or adapter parity"],
    }


class FakeBackend:
    """Deterministic host byte store; events are injected, never GPU callbacks."""

    def __init__(self, plane_bytes: int) -> None:
        self.plane_bytes = plane_bytes
        self.keys = bytearray(plane_bytes)
        self.values = bytearray(plane_bytes)
        self.events: list[tuple[int, bool]] = []
        self.writes: list[tuple[int, int, int]] = []

    def write(self, key: bytes, value: bytes, offset: int, byte_count: int, epoch: int) -> object:
        assert len(key) == len(value) == byte_count
        self.keys[offset:offset + byte_count] = key
        self.values[offset:offset + byte_count] = value
        self.writes.append((offset, byte_count, epoch))
        return object()

    def poll_completions(self) -> list[tuple[int, bool]]:
        events, self.events = self.events, []
        return events


def cpu_fake() -> dict:
    from mlx2.runtime.paged_kv_cache import PagedKVPrivateCache
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_write import PagedKVWriteOwner

    token_bytes = 256  # one KV head, d128, fp16
    page_bytes = 64 * token_bytes
    cases = []
    for count in CASES:
        pool = PagedKVPool(4)
        backend = FakeBackend(pool.capacity * page_bytes)
        writer = PagedKVWriteOwner(pool, backend, page_bytes=page_bytes, permit_candidate=True)
        cache = PagedKVPrivateCache(writer, kv_heads=1, head_dim=128,
                                    dtype="float16", permit_candidate=True)
        keys, values = bytes([count]) * (count * token_bytes), bytes([count + 1]) * (count * token_bytes)
        tickets = cache.append(keys, values)
        assert len(tickets) == (1 if count <= 64 else 2)
        assert cache.offset == 0
        assert [length for _, length, _ in backend.writes] == (
            [count * token_bytes] if count <= 64 else [page_bytes, token_bytes])
        backend.events.extend((ticket.epoch, True) for ticket in reversed(tickets))
        assert cache.poll_completions() and cache.offset == count
        assert cache.export_exact().key_bytes == keys
        use = cache.attention(row_count=1, query_heads=2)
        assert len(use.plan.page_table) == len(tickets)
        use.mark_submitted()
        cache.close()
        assert pool.free_count < pool.capacity
        writer.ledger.complete(use.lease)  # fake terminal reader event only
        assert pool.free_count == pool.capacity
        reused = pool.reserve()[0]
        assert reused.generation == (2 if reused.page_id in {h.page_id for h in use.plan.page_table} else 1)
        cases.append({"tokens": count, "pages": len(tickets), "accepted_after_events": True})

    # Another lane's allocation/release cannot mutate a pinned lane's page or plan.
    pool = PagedKVPool(3)
    backend = FakeBackend(pool.capacity * page_bytes)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=page_bytes, permit_candidate=True)
    primary = PagedKVPrivateCache(writer, kv_heads=1, head_dim=128, dtype="float16", permit_candidate=True)
    ticket = primary.append(b"a" * token_bytes, b"b" * token_bytes)[0]
    backend.events.append((ticket.epoch, True))
    assert primary.poll_completions()
    use = primary.attention(row_count=1, query_heads=2)
    pinned = use.plan.page_table[0]
    churn = pool.reserve()[0]
    pool.release((churn,), after_epoch=writer.ledger.completed_epoch)
    pool.retire(writer.ledger.completed_epoch)
    again = pool.reserve()[0]
    assert pool.live_generations()[pinned.page_id] == pinned.generation
    assert again.page_id != pinned.page_id
    pool.release((again,), after_epoch=writer.ledger.completed_epoch)
    pool.retire(writer.ledger.completed_epoch)
    use.abort_before_submit()

    # A shared tail is refused before metadata or native write mutation.
    pool.retain((pinned,))
    old_offset, old_writes = primary.offset, len(backend.writes)
    try:
        primary.append(b"c" * token_bytes, b"d" * token_bytes)
    except ValueError as exc:
        assert "shared partial tail requires staged native COW" in str(exc)
    else:
        raise AssertionError("shared tail was writable")
    assert primary.offset == old_offset and len(backend.writes) == old_writes
    pool.release((pinned,), after_epoch=writer.ledger.completed_epoch)
    primary.close()

    # Failure on page two retires the callback's lease, poisons the owner,
    # and never publishes the staged 65-token append.
    pool = PagedKVPool(2)
    backend = FakeBackend(pool.capacity * page_bytes)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=page_bytes, permit_candidate=True)
    cache = PagedKVPrivateCache(writer, kv_heads=1, head_dim=128, dtype="float16", permit_candidate=True)
    tickets = cache.append(b"k" * (65 * token_bytes), b"v" * (65 * token_bytes))
    cache.close()
    backend.events.append((tickets[1].epoch, False))
    assert not cache.poll_completions() and writer.poisoned
    assert writer.ledger.completed_epoch == 0 and pool.free_count == 0
    backend.events.append((tickets[0].epoch, True))
    assert not cache.poll_completions()
    # A failed native terminal epoch permanently quarantines its touched page.
    # The independently successful page retires normally, but the poisoned
    # writer refuses all further submissions through this arena.
    assert writer.ledger.completed_epoch == tickets[1].epoch
    assert pool.free_count == 1 and pool.quarantined_count == 1
    return {"schema": SCHEMA, "mode": "cpu-fake", "gpu_executed": False,
            "boundary_cases": cases, "unrelated_lane_churn": True,
            "shared_tail_cow_refused_before_write": True,
            "injected_failed_terminal_retired_without_publication": True,
            "failed_page_quarantined_successful_page_reusable": True,
            "native_command_failure_tested": False}


def _wait_writes(owner, tickets, timeout_s: float = 5.0) -> None:
    remaining = {ticket.epoch for ticket in tickets}
    deadline = time.monotonic() + timeout_s
    while remaining and time.monotonic() < deadline:
        for completion in owner.poll_completions():
            if completion.ticket.epoch not in remaining or not completion.succeeded:
                raise AssertionError("unexpected or failed native terminal write")
            remaining.remove(completion.ticket.epoch)
        if remaining:
            time.sleep(0.05)
    if remaining:
        raise TimeoutError(f"native terminal writes missing: {sorted(remaining)}")


def gpu_native() -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner

    stream = mx.default_stream(mx.gpu)
    results = []
    for count in CASES:
        pool = PagedKVPool(2)
        backend = NativeWriteBackend(128, stream, permit_candidate=True)
        owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
        handles = pool.reserve(2 if count == 65 else 1)
        key = np.arange(1, count + 1, dtype=np.uint8)
        value = np.arange(101, 101 + count, dtype=np.uint8)
        tickets = []
        for page_index, handle in enumerate(handles):
            begin, end = page_index * 64, min(count, (page_index + 1) * 64)
            tickets.append(owner.submit_write(handle, within_page_offset=0,
                byte_count=end - begin, key_bytes=mx.array(key[begin:end]),
                value_bytes=mx.array(value[begin:end])))
        # The last dependency and explicit stream order cover the preceding
        # write. A separate read lease protects all involved generations.
        reader = owner.ledger.prepare(handles)
        owner.ledger.submit(reader)
        read_pairs = [backend.diagnostic_read(tickets[-1].dependency, i * 64,
                      min(64, count - i * 64), permit_diagnostic=True)
                      for i in range(len(handles))]
        mx.eval(*(array for pair in read_pairs for array in pair))
        mx.synchronize(stream)
        observed_k = np.concatenate([np.array(pair[0]) for pair in read_pairs])
        observed_v = np.concatenate([np.array(pair[1]) for pair in read_pairs])
        if not np.array_equal(observed_k, key) or not np.array_equal(observed_v, value):
            raise AssertionError(f"native K/V mismatch at {count} bytes")
        pool.release(handles, after_epoch=reader.epoch)
        assert pool.free_count == 2 - len(handles)
        _wait_writes(owner, tickets)
        assert pool.free_count == 2 - len(handles)
        owner.ledger.complete(reader)  # synchronized diagnostic read proof
        assert pool.free_count == 2
        reused = pool.reserve(len(handles))
        old_generations = {handle.page_id: handle.generation for handle in handles}
        assert all(new.generation == old_generations[new.page_id] + 1 for new in reused)
        results.append({"bytes": count, "pages": len(handles), "write_epochs": [t.epoch for t in tickets],
                        "read_epoch": reader.epoch, "generation_reuse": True,
                        "key_bytes": observed_k.tolist(), "value_bytes": observed_v.tolist()})
    return {"schema": SCHEMA, "mode": "gpu-native", "host": platform.node(),
            "mlx_version": mx.__version__, "gpu_executed": True,
            "same_stream_two_page_dependency_observed": True,
            "boundary_cases": results, "native_command_failure_tested": False,
            "native_cow_copy_tested": False, "serving_route_selected": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--cpu-fake", action="store_true")
    choice.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = gpu_native() if args.execute_gpu else cpu_fake() if args.cpu_fake else dry_run()
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
