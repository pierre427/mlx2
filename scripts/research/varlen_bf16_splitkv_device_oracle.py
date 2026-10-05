"""Dry-by-default BF16 paged Q1 hardware oracle; root owns GPU execution.

This measures one attention layer and native ownership, never model performance
or serving qualification. Runtime imports occur only after explicit execution,
GPU lease validation and frozen binary identity validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path('/tmp/mlx2-native-bf16-storage-build-1004/_paged_kv_native.cpython-312-darwin.so')
BINARY_SHA = 'f220d9391ea4c48c56c5fb11c4e0ab37a62298dcc9d9d270a9d0394a894577f1'
CONTEXTS = ((63, 129), (127, 128), (128, 129), (4096, 8192))
PARTITIONS = (128, 256)
MAX_SECONDS = 60
MAX_RSS = 4 << 30
FAILURE_ROOTS = []


def case_plan(contexts, partition):
    if (type(contexts) is not tuple or len(contexts) != 2 or
            any(type(n) is not int or not 32 <= n <= 8192 for n in contexts) or
            type(partition) is not int or partition not in PARTITIONS):
        raise ValueError('bounded B2 D256 oracle plan required')
    split = max(contexts) > 128
    partitions = (max(contexts) + partition - 1) // partition if split else 0
    capacity = sum((n + 63) // 64 + 2 for n in contexts)
    return dict(contexts=list(contexts), partition=partition, split=split,
                expected_partial=int(split), expected_reduce=int(split),
                expected_tile=int(not split), expected_grouped_write=1,
                capacity_pages=capacity, plane_bytes=capacity * 64 * 4 * 256 * 2,
                scratch_bytes=2 * 24 * partitions * 258 * 4,
                scratch_scope='calculated one-primitive FP32 allocation; retained until command completion',
                scratch_allocation_measured=False)


def preflight():
    return dict(schema='mlx2.bf16-splitkv-device-oracle.v1', status='planned',
                gpu_executed=False, qualified=False, measurement_scope='attention_layer_device_oracle',
                dtype='bfloat16', batch=2, head_dim=256, query_heads=24, kv_heads=4,
                seed=1004, hard_seconds=MAX_SECONDS, max_rss_bytes=MAX_RSS,
                native_binary=str(BINARY), native_sha256=BINARY_SHA,
                reference='stock SDPA independently per logical unpadded lane with same BF16 QKV',
                cases=[case_plan(c, p) for p in PARTITIONS for c in CONTEXTS])


def check_bounds(start):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin': rss *= 1024
    if rss > MAX_RSS: raise MemoryError('4 GiB oracle RSS ceiling')
    if time.monotonic() - start >= MAX_SECONDS: raise TimeoutError('60 second oracle wall deadline')
    return rss


def execute_case(mx, plan, start, receipt):
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend
    from mlx2.runtime.paged_attention_pack import prepare_staged_token_read
    from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner
    from mlx2.runtime.paged_request_transaction import CandidateRequest
    from mlx2.runtime.paged_native_retirement import reap_native_request_owner
    import _paged_kv_native as native
    os.environ.update(MLX2_PAGED_Q1_SPLIT_KV=str(plan['partition']),
                      MLX2_PAGED_Q1_SIMD_TILE='1', MLX2_PAGED_Q1_SIMD_STRIPES='16',
                      MLX2_PAGED_GROUPED_Q1_WRITE='1', MLX2_PAGED_Q1_STOCK_SDPA='0')
    stream = mx.default_stream(mx.gpu)
    profile = TokenKVProfile(4, 256, 'bfloat16')
    pool = PagedKVPool(plan['capacity_pages'])
    arena = NativeWriteBackend(plan['plane_bytes'], stream, permit_candidate=True, storage_dtype='bfloat16')
    writer = PagedKVWriteOwner(pool, arena, page_bytes=profile.page_bytes, permit_candidate=True)
    backend = NativeQwen3PagedBackend(writer, timeout_s=5.0, permit_candidate=True, profile_host=True)
    backend.direct_grouped_fence = True
    layers = tuple(PagedKVTokenOwner(writer, profile, permit_candidate=True) for _ in range(2))
    owners = []; branches = []; prepared = []; use = None
    keys = values = queries = output = reference = None
    clean = False
    try:
        if native.storage_dtype(arena._arena) != 'bfloat16': raise RuntimeError('native dtype receipt differs')
        with mx.stream(stream):
            mx.random.seed(1004)
            keys = tuple(mx.random.uniform(-.25, .25, shape=(n, 4, 256)).astype(mx.bfloat16) for n in plan['contexts'])
            values = tuple(mx.random.uniform(-.25, .25, shape=(n, 4, 256)).astype(mx.bfloat16) for n in plan['contexts'])
            queries = mx.random.uniform(-.25, .25, shape=(2, 24, 256)).astype(mx.bfloat16)
            mx.eval(queries, *keys, *values)
            check_bounds(start)
            prefix_counts = tuple(n - 1 for n in plan['contexts'])
            tick = time.perf_counter()
            # Existing pack_head_spans uses contiguous(...).view(uint8): raw
            # same-dtype BF16 payloads, with no numerical cast at the arena.
            backend.append_completed(layers, mx.concatenate([k[:-1] for k in keys]),
                                     mx.concatenate([v[:-1] for v in values]), prefix_counts)
            receipt['completed_prefix_write_seconds'] = time.perf_counter() - tick
            if tuple(layer.offset for layer in layers) != prefix_counts: raise RuntimeError('prefix publication differs')
            revision = 'bf16-device-oracle-1004'
            owners = [NativeAtomicRequestOwner(revision, (layer,), {}, supported_planes=('kv',),
                                              enabled=True, reuse_private_tail=True) for layer in layers]
            branches = [owner.begin(CandidateRequest(i, revision, 1, ('kv',))) for i, owner in enumerate(owners)]
            private = tuple(branch.layers[0] for branch in branches)
            before = backend.profile_counters_snapshot()
            tick = time.perf_counter()
            tickets = backend.append_staged(private, mx.stack([k[-1] for k in keys]),
                                           mx.stack([v[-1] for v in values]), (1, 1))
            use = prepare_staged_token_read(private, (1, 1), query_heads=24, permit_candidate=True)
            output = backend.read_staged(use, queries, tickets, scale=256 ** -.5)
            mx.eval(output)
            proofs = backend.drain_staged(private, (use,))
            receipt['native_append_read_terminal_seconds'] = time.perf_counter() - tick
            after = backend.profile_counters_snapshot()
            counter_fields = ('q1_split_partial_dispatches', 'q1_split_reduce_dispatches',
                              'q1_tile_dispatches', 'grouped_q1_writes', 'native_write_dispatches')
            delta = {name: after[name] - before[name] for name in counter_fields}
            receipt['physical_counter_delta'] = delta
            expected = (plan['expected_partial'], plan['expected_reduce'], plan['expected_tile'], 1, 0)
            if tuple(delta[name] for name in counter_fields) != expected: raise RuntimeError('physical dispatch proof differs')
            if len(proofs) != 1 or proofs[0].event != (use.lease.epoch, True): raise RuntimeError('terminal proof differs')
            if tuple(owner.offset for owner in private) != tuple(plan['contexts']): raise RuntimeError('private suffix publication differs')
            for lane, branch in enumerate(branches): branch.prove_staged_layer_read(0, lane, proofs[0])
            prepared = [branch.prepare(1) for branch in branches]
            for state in prepared: state.publish()
            branches = []; prepared = []
            raw_checks = 0
            for lane, owner in enumerate(owners):
                with_view = owner.snapshot()
                try:
                    if with_view.offset != plan['contexts'][lane] or with_view.generation != 1:
                        raise RuntimeError('atomic public generation proof differs')
                    table = with_view.layer_owners[0].accepted_handles()
                    # Hold the public reader while copying raw first/page-edge/
                    # appended-last BF16 payloads. Synchronize before closing it.
                    for token in sorted({0, min(63, plan['contexts'][lane] - 1), plan['contexts'][lane] - 1}):
                        for head in (0, 3):
                            offset = table[token // 64].page_id * profile.page_bytes + (head * 64 + token % 64) * 256 * 2
                            raw = arena.diagnostic_read(tickets[-1].dependency, offset, 256 * 2, permit_diagnostic=True)
                            expected_raw = tuple(mx.contiguous(plane[lane][token, head]).view(mx.uint8).reshape(-1)
                                                 for plane in (keys, values))
                            mx.eval(*raw, *expected_raw)
                            if any(not bool(mx.all(a == b).item()) for a, b in zip(raw, expected_raw)):
                                raise RuntimeError('raw BF16 arena payload differs')
                            raw_checks += 2
                    mx.synchronize(stream)
                finally: with_view.close()
                owner.reap_retired()
            receipt['raw_bf16_payload_checks'] = raw_checks
            receipt['raw_bf16_payloads_equal'] = True
            receipt['atomic_published_offsets'] = list(plan['contexts'])
            receipt['read_terminal_epoch'] = use.lease.epoch
            receipt['read_terminal_success'] = use.terminal_succeeded
            tick = time.perf_counter()
            reference = mx.concatenate([
                mx.fast.scaled_dot_product_attention(
                    queries[i:i + 1, :, None, :], keys[i].transpose(1, 0, 2)[None, :, :, :],
                    values[i].transpose(1, 0, 2)[None, :, :, :], scale=256 ** -.5)
                for i in range(2)], axis=0).reshape(2, 24, 256)
            mx.eval(reference)
            receipt['stock_unpadded_b1_pair_seconds'] = time.perf_counter() - tick
            if output.dtype != mx.bfloat16 or reference.dtype != mx.bfloat16: raise RuntimeError('BF16 output dtype differs')
            diff = mx.abs(output.astype(mx.float32) - reference.astype(mx.float32))
            finite = bool(mx.all(mx.isfinite(output)).item()) and bool(mx.all(mx.isfinite(reference)).item())
            receipt.update(output_dtype=str(output.dtype), reference_dtype=str(reference.dtype), finite=finite,
                           max_abs_error=[float(mx.max(diff[i]).item()) for i in range(2)],
                           atol=.00025, rtol=.01,
                           exact_elements_fraction=float(mx.mean((output == reference).astype(mx.float32)).item()),
                           max_reference_abs=[float(mx.max(mx.abs(reference[i].astype(mx.float32))).item()) for i in range(2)],
                           allclose=bool(mx.all(diff <= .00025 + .01 * mx.abs(reference.astype(mx.float32))).item()))
            if not finite or not receipt['allclose']: raise RuntimeError('BF16 native vs stock SDPA parity failed')
            check_bounds(start)
            clean = True
    finally:
        if clean:
            for owner in owners:
                owner.close(); reap_native_request_owner(owner, writer, backend)
            if writer.pending_epochs or writer.ledger.pending_count or not all(owner.fully_retired for owner in owners):
                raise RuntimeError('oracle retained native leases')
            arena.close_after_terminal()
            receipt.update(pending_write_epochs=0, pending_ledger=0, owners_retired=True,
                           free_pages=pool.free_count, scratch_device_access_terminal=True)
            if pool.free_count != pool.capacity: raise RuntimeError('oracle leaked physical pages')
        else:
            # Timeout or ambiguous submission never authorizes freeing/reusing
            # pages. Keep every graph/state root until the process exits.
            FAILURE_ROOTS.append((arena, writer, backend, layers, owners, branches, prepared,
                                  use, keys, values, queries, output, reference))
    return receipt


def execute(result):
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('oracle hard deadline')))
    signal.alarm(MAX_SECONDS)
    from varlen_pack_price_bench import _gpuq_owner
    result['gpuq_owner'] = _gpuq_owner()
    if hashlib.sha256(BINARY.read_bytes()).hexdigest() != BINARY_SHA: raise RuntimeError('frozen native binary hash differs')
    sys.path.insert(0, str(BINARY.parent)); sys.path.insert(0, str(ROOT / 'src'))
    import _paged_kv_native as native
    if Path(native.__file__).resolve() != BINARY.resolve(): raise RuntimeError('native import path differs')
    import mlx.core as mx
    result.update(gpu_executed=True, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  device=dict(mx.device_info()), cases=[])
    try:
        for partition in PARTITIONS:
            for contexts in CONTEXTS:
                check_bounds(start)
                plan = case_plan(contexts, partition)
                receipt = dict(plan)
                result['cases'].append(receipt)
                execute_case(mx, plan, start, receipt)
        result.update(status='passed', total_seconds=time.monotonic() - start,
                      peak_rss_bytes=check_bounds(start), qualified=False)
    finally: signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = preflight(); exit_code = 0
    try:
        if args.execute: execute(result)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}',
                      retained_failure_roots=len(FAILURE_ROOTS), qualified=False)
        exit_code = 1
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return exit_code

if __name__ == '__main__': raise SystemExit(main())
