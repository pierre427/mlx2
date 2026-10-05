"""GPUQ-owned exact-byte 16-layer paged arena boundary oracle.

The command is diagnostic only. It validates native allocation and byte addressing,
not attention math or serving. The parent task owns the GPU lease.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PLANE_BYTES = 4_802_478_080
PAGE_BYTES = 4 * 64 * 256 * 2
LAYERS = 16
PAGES_PER_LAYER = PLANE_BYTES // PAGE_BYTES // LAYERS
FAILURE_ROOTS: list[object] = []
HARD_SECONDS = 100
MAX_BYTES = 12 << 30


def check_bounds(start: float, mx=None) -> None:
    if time.monotonic() - start > HARD_SECONDS:
        raise TimeoutError('large-plane oracle deadline exceeded')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    if rss > MAX_BYTES:
        raise MemoryError('large-plane oracle RSS cap exceeded')
    if mx is not None and mx.get_active_memory() > MAX_BYTES:
        raise MemoryError('large-plane oracle MLX active-memory cap exceeded')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpuq_owner() -> dict:
    session = os.environ.get('GPUQ_SESSION')
    lease = os.environ.get('GPUQ_LEASE')
    if not session or not lease:
        raise RuntimeError('GPUQ_SESSION and GPUQ_LEASE are required')
    owners = []
    for lock in (Path('/Users/Shared/mlxuag/gpu.lock'), Path('/tmp/gpu.lock')):
        if not lock.is_dir() or lock.is_symlink():
            raise RuntimeError(f'GPU lock is not an owned directory: {lock}')
        owner = json.loads((lock / 'owner.json').read_text())
        if (owner.get('session') != session or owner.get('lease_id') != lease or
                type(owner.get('pid')) is not int or owner['pid'] < 1):
            raise RuntimeError(f'GPU lock owner differs: {lock}')
        os.kill(owner['pid'], 0)
        owners.append(owner)
    if owners[0] != owners[1]:
        raise RuntimeError('GPU lock owner receipts disagree')
    return owners[0]


def preflight(source: str, binary: Path, expected_sha: str) -> dict:
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != source:
        raise RuntimeError('source revision differs')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('source tree is dirty')
    if not binary.is_file() or sha(binary) != expected_sha:
        raise RuntimeError('native binary differs')
    assert PLANE_BYTES == LAYERS * PAGES_PER_LAYER * PAGE_BYTES
    return {'schema': 'mlx2.n20-large-plane-oracle.v1', 'status': 'preflight_passed',
            'source_commit': source, 'native_sha256': expected_sha,
            'plane_bytes': PLANE_BYTES, 'total_arena_bytes': 2 * PLANE_BYTES,
            'layers': LAYERS, 'pages_per_layer': PAGES_PER_LAYER,
            'page_bytes': PAGE_BYTES, 'expected_write_terminals': 2 * LAYERS,
            'hard_seconds': HARD_SECONDS, 'max_rss_bytes': MAX_BYTES,
            'max_mlx_active_bytes': MAX_BYTES,
            'qualified': False, 'price_usable': False, 'gpu_executed': False}


def _run_device(result: dict, binary: Path) -> None:
    started = time.monotonic()
    sys.path.insert(0, str(binary.parent))
    import mlx.core as mx
    import _paged_kv_native as native

    if Path(native.__file__).resolve() != binary.resolve():
        raise RuntimeError('loaded native extension path differs')
    capability = dict(native.arena_storage_capability())
    result['capability'] = capability
    if (capability.get('version') != 1 or
        capability.get('layout') != 'contiguous_uint8_2d_large' or
        capability.get('large_plane_alignment_bytes') != 4096 or
        capability.get('max_plane_bytes', 0) < PLANE_BYTES or
        not capability.get('exact_byte_allocation')):
        raise RuntimeError('bounded large-plane native capability absent')
    stream = mx.default_stream(mx.gpu)
    arena = native.create_arena(PLANE_BYTES, storage_dtype='bfloat16')
    check_bounds(started, mx)
    roots: list[object] = [arena]
    result['gpu_executed'] = True
    result['reported_plane_bytes'] = native.plane_bytes(arena)
    result['plane_storage_shape'] = list(native.plane_storage_shape(arena))
    if result['reported_plane_bytes'] != PLANE_BYTES:
        raise RuntimeError('native plane byte identity differs')
    if result['plane_storage_shape'] != [PLANE_BYTES // 4096, 4096]:
        raise RuntimeError('native large-plane storage shape differs')
    checked = []
    writes = []
    try:
        with mx.stream(stream):
            for layer in range(LAYERS):
                for at_end in (False, True):
                    page = layer * PAGES_PER_LAYER + (PAGES_PER_LAYER - 1 if at_end else 0)
                    offset = page * PAGE_BYTES + (PAGE_BYTES - 16 if at_end else 0)
                    key = bytes(((layer * 17 + (7 if at_end else 3) + j) & 255) for j in range(16))
                    value = bytes(((layer * 29 + (13 if at_end else 5) + j) & 255) for j in range(16))
                    k = mx.array(list(key), dtype=mx.uint8)
                    v = mx.array(list(value), dtype=mx.uint8)
                    epoch = len(writes) + 1
                    ticket = native.write(arena, k, v, offset, 16, epoch, stream)
                    writes.append(ticket)
                    checked.append((layer, at_end, offset, key, value, ticket))
                    roots.extend((k, v, ticket))
            mx.eval(*writes)
        mx.synchronize(stream)
        check_bounds(started, mx)
        terminals = native.poll_completions(arena)
        result['write_terminal_events'] = [(int(epoch), bool(ok)) for epoch, ok in terminals]
        if sorted(result['write_terminal_events']) != [(i, True) for i in range(1, 2 * LAYERS + 1)]:
            raise RuntimeError('write terminal count or status differs')
        reads = []
        with mx.stream(stream):
            for layer, at_end, offset, key, value, ticket in checked:
                actual_k, actual_v = native.diagnostic_read(arena, ticket, offset, 16, stream,
                                                            permit_diagnostic=True)
                reads.append((layer, at_end, offset, key, value, actual_k, actual_v))
                roots.extend((actual_k, actual_v))
            mx.eval(*(arr for _, _, _, _, _, k, v in reads for arr in (k, v)))
        mx.synchronize(stream)
        check_bounds(started, mx)
        for layer, at_end, offset, key, value, actual_k, actual_v in reads:
            if bytes(actual_k.tolist()) != key or bytes(actual_v.tolist()) != value:
                raise RuntimeError(f'raw byte mismatch at layer={layer} last={at_end} offset={offset}')
        result['checked_boundary_pages'] = len(reads)
        result['highest_checked_offset'] = max(item[2] for item in reads)
        result['raw_bytes_exact'] = True
        result['write_dispatches'] = native.write_dispatch_count(arena)
        if result['write_dispatches'] != 2 * LAYERS:
            raise RuntimeError('physical write dispatch count differs')
        result['mlx_active_bytes'] = mx.get_active_memory()
        result['mlx_peak_bytes'] = mx.get_peak_memory()
        result['mlx_cache_bytes'] = mx.get_cache_memory()
        result['elapsed_seconds'] = time.monotonic() - started
    except BaseException:
        FAILURE_ROOTS.append(tuple(roots))
        result['failure_roots_retained'] = True
        raise
    finally:
        try:
            mx.synchronize(stream)
        except BaseException:
            FAILURE_ROOTS.append(tuple(roots))
            result['failure_roots_retained'] = True
            raise


def run(result: dict, binary: Path) -> None:
    result['gpuq_owner_before'] = gpuq_owner()
    import mlx.core as mx
    baseline_active = mx.get_active_memory()
    result['baseline_mlx_active_bytes'] = baseline_active
    _run_device(result, binary)
    # _run_device has returned only after all 32 write terminals and reads
    # completed. Its local graph roots and arena capsule can now be destroyed.
    gc.collect()
    mx.synchronize(mx.default_stream(mx.gpu))
    mx.clear_cache()
    gc.collect()
    post_active = mx.get_active_memory()
    result['post_release_mlx_active_bytes'] = post_active
    result['post_release_mlx_cache_bytes'] = mx.get_cache_memory()
    result['gpuq_owner_after'] = gpuq_owner()
    result['large_allocation_retired'] = post_active <= baseline_active + (32 << 20)
    if not result['large_allocation_retired']:
        raise RuntimeError('large arena allocation remains active after terminal release')
    if result['gpuq_owner_after'] != result['gpuq_owner_before']:
        raise RuntimeError('GPUQ owner changed during large-plane oracle')
    result['status'] = 'passed'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--native-path', required=True, type=Path)
    parser.add_argument('--native-sha256', required=True)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    result = preflight(args.source_commit, args.native_path, args.native_sha256)
    try:
        if not args.preflight_only:
            run(result, args.native_path)
    except BaseException as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        raise
    finally:
        args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
