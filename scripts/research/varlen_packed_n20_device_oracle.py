"""Source-bound default-off N20 packed KV/read Metal oracle; root owns GPU lease.

This proves one-layer raw BF16 bytes and terminal retirement, not model serving.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = ((256,), (256, 257), (256, 257, 258), tuple(256 + i % 2 for i in range(20)))
LARGE_CASE = tuple(7000 + i % 3 for i in range(20))
MAX_SECONDS = 90
MAX_RSS = 4 << 30
FAILURE_ROOTS: list[object] = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_plan(counts: tuple[int, ...]) -> dict:
    if (type(counts) is not tuple or not 1 <= len(counts) <= 20 or
            any(type(n) is not int or not 256 <= n <= 8192 for n in counts) or
            sum(counts) > 163840):
        raise ValueError('N20 oracle requires 1..20 bounded cold rows')
    pages = sum((n + 63) // 64 for n in counts)
    return {'counts': list(counts), 'rows': sum(counts), 'pages': pages,
            'plane_bytes': pages * 4 * 64 * 256 * 2,
            'global_score_scratch_bytes': 0}


def bounds(start: float) -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin': rss *= 1024
    if rss > MAX_RSS: raise MemoryError(f'N20 oracle RSS cap {MAX_RSS} exceeded')
    if time.monotonic() - start > MAX_SECONDS: raise TimeoutError('N20 oracle deadline')
    return rss


def preflight(source: str, binary: Path, binary_sha: str) -> dict:
    if (not source or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                               text=True).strip() != source or
            subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT,
                                    text=True).strip()):
        raise RuntimeError('N20 oracle source must be exact and clean')
    if not binary.is_file() or sha(binary) != binary_sha:
        raise RuntimeError('N20 native binary identity differs')
    return {'schema': 'mlx2.packed-n20-device-oracle.v1', 'status': 'preflight_passed',
            'gpu_executed': False, 'qualified': False, 'default_off': True,
            'source_commit': source, 'binary_path': str(binary),
            'native_sha256': binary_sha, 'hard_seconds': MAX_SECONDS,
            'max_rss_bytes': MAX_RSS, 'cases': [case_plan(c) for c in CASES]}


def run_case(mx, native, counts: tuple[int, ...], start: float, result: dict) -> None:
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend

    plan = case_plan(counts)
    capacity = plan['pages'] + len(counts)
    stream = mx.default_stream(mx.gpu)
    pool = PagedKVPool(capacity)
    profile = TokenKVProfile(4, 256, 'bfloat16')
    arena = NativeWriteBackend(capacity * profile.page_bytes, stream,
                               permit_candidate=True, storage_dtype='bfloat16')
    result['native_arena_kv_plane_bytes'] = 2 * arena.plane_bytes
    writer = PagedKVWriteOwner(pool, arena, page_bytes=profile.page_bytes,
                               permit_candidate=True)
    backend = NativeQwen3PagedBackend(writer, timeout_s=10, permit_candidate=True)
    owners = tuple(PagedKVTokenOwner(writer, profile, permit_candidate=True)
                   for _ in counts)
    roots = [arena, writer, backend, owners]
    use = None
    proved = False
    result.update(counts=list(counts), status='running',
                  expected_grouped_rows=sum(counts))
    try:
        with mx.stream(stream):
            mx.random.seed(1004 + len(counts))
            keys = tuple(mx.random.uniform(-.25, .25, shape=(n, 4, 256)).astype(mx.bfloat16)
                         for n in counts)
            values = tuple(mx.random.uniform(-.25, .25, shape=(n, 4, 256)).astype(mx.bfloat16)
                           for n in counts)
            query = mx.random.uniform(-.25, .25,
                                      shape=(sum(counts), 24, 256)).astype(mx.bfloat16)
            mx.eval(query, *keys, *values)
            packed_k = mx.concatenate(keys, axis=0)
            packed_v = mx.concatenate(values, axis=0)
            roots.extend((keys, values, query, packed_k, packed_v))
            before = (native.grouped_n20_write_count(arena._arena),
                      native.grouped_n20_row_count(arena._arena),
                      native.prefill_long_n20_dispatch_count(arena._arena))
            tickets = backend.append_packed_multirow_n20(
                owners, packed_k, packed_v, counts, permit_candidate=True)
            use = backend.prepare_read_n20(owners, counts, query_heads=24,
                                           permit_candidate=True)
            output = backend.read_staged(use, query, tickets, scale=256 ** -.5)
            roots.extend((tickets, use, output))
            mx.eval(output)
            proofs = backend.drain_staged(owners, (use,))
            delta = (native.grouped_n20_write_count(arena._arena) - before[0],
                     native.grouped_n20_row_count(arena._arena) - before[1],
                     native.prefill_long_n20_dispatch_count(arena._arena) - before[2])
            result.update(grouped_write_delta=delta[0], grouped_row_delta=delta[1],
                          long_read_delta=delta[2], terminal_success=use.terminal_succeeded,
                          read_proofs=len(proofs), pending_epochs=writer.pending_epochs,
                          pending_leases=writer.ledger.pending_count,
                          owner_offsets=[o.offset for o in owners])
            if (delta != (1, sum(counts), 1) or len(proofs) != 1 or
                    proofs[0].event != (use.lease.epoch, True) or
                    tuple(o.offset for o in owners) != counts or
                    writer.pending_epochs or writer.ledger.pending_count):
                raise RuntimeError('N20 dispatch or terminal proof differs')
            proved = True
            reference = []
            row = 0
            for n, k, v in zip(counts, keys, values):
                q = query[row:row+n].transpose(1, 0, 2)[None]
                # This is the per-span forced-fused stock reference used by
                # ordinary segmented attention on the same real rows.
                ref = mx.fast.scaled_dot_product_attention(
                    q, k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None],
                    scale=256 ** -.5, mask='causal', force_fused=True)
                reference.append(ref[0].transpose(1, 0, 2))
                row += n
            expected = mx.concatenate(reference, axis=0)
            roots.extend((expected, reference))
            mx.eval(expected)
            raw = bool(mx.all(output.view(mx.uint16) == expected.view(mx.uint16)).item())
            finite = bool(mx.all(mx.isfinite(output)).item())
            result.update(raw_bit_equal=raw, finite=finite,
                          max_abs_error=float(mx.max(mx.abs(output.astype(mx.float32) -
                                                          expected.astype(mx.float32))).item()),
                          exact_fraction=float(mx.mean((output == expected).astype(mx.float32)).item()))
            if not raw or not finite:
                raise RuntimeError('N20 raw BF16 per-span forced-fused stock oracle differs')
        result['status'] = 'passed'
        result['rss_bytes'] = bounds(start)
        return None
    finally:
        try:
            mx.synchronize(stream)
            if proved and not writer.poisoned and not writer.pending_epochs and not writer.ledger.pending_count:
                for owner in owners: owner.close()
                arena.close_after_terminal()
                result.update(owners_retired=True, pending_epochs=0, pending_leases=0,
                              free_pages=pool.free_count,
                              all_pages_retired=pool.free_count == pool.capacity)
                if pool.free_count != pool.capacity:
                    raise RuntimeError('N20 oracle pages remain after terminal retirement')
            else:
                FAILURE_ROOTS.append(tuple(roots))
                result.update(owners_retired=False, failure_roots_retained=True)
        except BaseException:
            FAILURE_ROOTS.append(tuple(roots))
            result.update(owners_retired=False, failure_roots_retained=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--native-path', type=Path, required=True)
    parser.add_argument('--native-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--large-read', action='store_true',
                        help='separate 20x~7000 raw read, 100s/12GiB cap')
    args = parser.parse_args()
    global MAX_SECONDS, MAX_RSS
    if args.large_read:
        MAX_SECONDS = 100
        MAX_RSS = 12 << 30
    receipt = {'schema': 'mlx2.packed-n20-device-oracle.v1', 'status': 'failed',
               'gpu_executed': False, 'cases_completed': []}
    try:
        receipt.update(preflight(args.source_commit, args.native_path,
                                 args.native_sha256))
        if args.large_read:
            receipt['cases'] = [case_plan(LARGE_CASE)]
            receipt['scope'] = '20-lane 140k-row full paged NAX attention read, 12GiB/100s'
        if not args.preflight_only:
            from varlen_pack_price_bench import _gpuq_owner
            receipt['gpuq_owner'] = _gpuq_owner()
            for key, value in {'MLX2_PAGED_PACKED_N20': '1',
                               'MLX2_PAGED_PREFILL_MATRIX': '1',
                               'MLX2_PAGED_PREFILL_NAX_LONG_FUSED': '1',
                               'MLX2_PAGED_PREFILL_NAX_EXACT': '0',
                               'MLX2_PAGED_Q1_STOCK_LONG': '1'}.items():
                os.environ[key] = value
            sys.path.insert(0, str(args.native_path.parent))
            sys.path.insert(0, str(ROOT / 'src'))
            import _paged_kv_native as native
            if Path(native.__file__).resolve() != args.native_path.resolve():
                raise RuntimeError('N20 native import differs')
            cap = native.packed_n20_capability()
            if (cap.get('version') != 1 or cap.get('max_spans') != 20 or
                    cap.get('max_total_rows') != 163840 or
                    cap.get('selector') != 'MLX2_PAGED_PACKED_N20'):
                raise RuntimeError('N20 native capability differs')
            import mlx.core as mx
            signal.signal(signal.SIGALRM,
                          lambda *_: (_ for _ in ()).throw(TimeoutError('N20 oracle hard cap')))
            signal.alarm(MAX_SECONDS)
            start = time.monotonic()
            for counts in ((LARGE_CASE,) if args.large_read else CASES):
                receipt['gpu_executed'] = True
                case = case_plan(counts)
                case['status'] = 'running'
                receipt['cases_completed'].append(case)
                try:
                    run_case(mx, native, counts, start, case)
                except BaseException as exc:
                    case['status'] = 'failed'
                    case['error'] = f'{type(exc).__name__}: {exc}'
                    raise
            receipt['status'] = 'passed'
            receipt['elapsed_seconds'] = time.monotonic() - start
    except BaseException as exc:
        receipt['status'] = 'failed'
        receipt['error_type'] = type(exc).__name__
        receipt['error'] = str(exc)
        raise
    finally:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
