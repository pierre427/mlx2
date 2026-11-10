#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded actual PP DOWN replay under one paired CPG/radio ownership window.

The parent operator owns execution. This controller never selects a serving
route. See provenance/deepseek-v41-source-down-actual.json.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/research')]
from mlx2.desktop_campaign_worker import atomic, check_ownership
from mlx2.desktop_runner import source_manifest, digest

MEMORY_LIMIT = 6 << 30
HOST_RESERVE = 9 << 30
HOST_INITIAL = 15 << 30
MAX_ARRAY_BYTES = 64 << 20
MAX_CASES = 10
PARSER = 'ds41_down_actual_capture.py'
SCHEMA = 'mlx2.ds41-down-actual-owned.v1'


def checked_json(path, *, maximum=8 << 20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('Unsafe or oversized DOWN metadata file: '+path.name)
    return json.loads(path.read_text())


def copy_capture(unit, out):
    """Copy only the one sealed manifest, referenced NPZs and small metadata."""
    unit = Path(unit)
    if unit.is_symlink():
        raise ValueError('DOWN capture unit symlink')
    unit = unit.resolve(strict=True)
    spec = checked_json(unit/'spec.json')
    result = checked_json(unit/'results.json')
    release = checked_json(unit/'release-verified.json')
    if (result.get('status') != 'completed' or result.get('passed') is not True
            or release.get('owners_absent') is not True or release.get('source_unchanged') is not True):
        raise ValueError('DOWN capture unit incomplete or not released')
    manifests = sorted(unit.glob('down-actual-capture-*.json'))
    if len(manifests) != 1:
        raise ValueError('DOWN capture requires exactly one PP99 manifest')
    manifest = checked_json(manifests[0])
    parser = unit/'harness'/PARSER
    seal = spec.get('harness_sha256', {}).get(PARSER)
    if (parser.is_symlink() or not parser.is_file() or parser.stat().st_size > 128 << 10
            or not seal or hashlib.sha256(parser.read_bytes()).hexdigest() != seal
            or manifest.get('helper_sha256') != seal):
        raise ValueError('DOWN capture parser seal mismatch')
    result_identity = dict(checkpoint=result.get('checkpoint'),
        source_revision=result.get('source', {}).get('commit'),
        source_snapshot_sha256=result.get('source', {}).get('source_snapshot_sha256'))
    for key in ('checkpoint', 'source_revision', 'source_snapshot_sha256'):
        if not spec.get(key) or manifest.get(key) != spec[key] or result_identity.get(key) != spec[key]:
            raise ValueError('DOWN capture model/spec/manifest identity mismatch')
    cases = manifest.get('cases', [])
    if not 2 <= len(cases) <= MAX_CASES:
        raise ValueError('DOWN capture case count')
    names = [case['path'] for case in cases]
    if len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+\.npz', name) for name in names):
        raise ValueError('Unsafe or duplicate DOWN NPZ filename')
    sources = []
    total = 0
    for case in cases:
        path = unit/case['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size > (6 << 20)+4096:
            raise ValueError('Unsafe or oversized DOWN NPZ')
        total += path.stat().st_size
        if total > MAX_ARRAY_BYTES+MAX_CASES*4096:
            raise ValueError('DOWN capture aggregate copy ceiling')
        if hashlib.sha256(path.read_bytes()).hexdigest() != case['sha256']:
            raise ValueError('DOWN NPZ source hash mismatch')
        sources.append(path)
    metadata = [unit/name for name in ('spec.json', 'results.json', 'release-verified.json', 'request.json', 'policy.json') if (unit/name).exists()]
    for path in metadata:
        checked_json(path)
    if sum(path.stat().st_size for path in metadata) > 32 << 20:
        raise ValueError('DOWN metadata aggregate ceiling')
    # All paths/hashes are checked before importing the sealed parser or
    # creating a capture destination. Never copy unrelated globs/payloads.
    dest = Path(out)/'reference-capture'
    (dest/'harness').mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parser, dest/'harness'/PARSER)
    if hashlib.sha256((dest/'harness'/PARSER).read_bytes()).hexdigest() != seal:
        raise ValueError('Copied DOWN parser changed')
    copied = []
    for path in [*metadata, manifests[0], *sources]:
        target = dest/path.name
        shutil.copyfile(path, target)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        after = hashlib.sha256(target.read_bytes()).hexdigest()
        if before != after:
            raise ValueError('DOWN capture changed during copy')
        copied.append(dict(name=path.name, bytes=path.stat().st_size, sha256=after))
    loader = importlib.util.spec_from_file_location('_owned_down_capture_parser', dest/'harness'/PARSER)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    report, loaded = module.load_capture(dest, checkpoint=spec['checkpoint'], source_snapshot=spec['source_snapshot_sha256'])
    if len(loaded) != len(cases):
        raise ValueError('Copied DOWN case inventory changed')
    atomic(Path(out)/'capture-copy.json', dict(source_unit=str(unit), source_spec_sha256=hashlib.sha256((unit/'spec.json').read_bytes()).hexdigest(),
        files=copied, parser_sha256=seal, copied_npz_bytes=total, validated_cases=len(loaded),
        captured_array_bytes=report['captured_array_bytes'], capture_checkpoint=report['checkpoint']))
    return dest


def host_admission(minimum):
    """CPU-only live check; never modifies runtime reserve or memory policy."""
    from mlx2.runtime.os_memory import host_memory_snapshot
    host = host_memory_snapshot()
    if host is None or host.available_bytes < minimum:
        raise MemoryError('DOWN component live host admission refused: '+str(minimum)+' bytes required')
    return host


def validate_spec(spec, out):
    if (spec.get('schema') != SCHEMA or Path(spec.get('out', '')).resolve() != out.resolve()
            or spec.get('job_id') != out.name or spec.get('memory_limit_bytes') != MEMORY_LIMIT
            or spec.get('host_live_reserve_bytes') != HOST_RESERVE or spec.get('host_initial_floor_bytes') != HOST_INITIAL
            or spec.get('down_actual_options') != {'repeats': 7, 'include_original_e1': True}
            or spec.get('rate_eligible') is not False or spec.get('numerical_environment') != {'MLX_ENABLE_TF32': '0'}):
        raise ValueError('Invalid sealed bounded DOWN component spec')
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        raise ValueError('DOWN component numerical environment mismatch')


def component(out):
    out = Path(out)
    spec = checked_json(out/'spec.json')
    report = {**spec, 'passed': False, 'methodological_passed': False,
              'numerical_passed': False, 'qualification': False, 'performance_qualified': False}
    mx = None
    try:
        validate_spec(spec, out)
        report['ownership'] = check_ownership(spec)
        if digest(source_manifest(ROOT)) != spec['source_snapshot_sha256']:
            raise RuntimeError('Frozen DOWN source changed')
        host = host_admission(HOST_INITIAL)
        report['host_available_before_component'] = host.available_bytes
        from probe_deepseek_v4_decode_compare import sample_environment, HID_PROFILE
        report['before'] = sample_environment(HID_PROFILE)
        if [p for p in report['before']['foreign_model_processes'] if p['pid'] != os.getpid()]:
            raise RuntimeError('Foreign model; DOWN component yields')
        import mlx.core as mx
        import numpy as np
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(MEMORY_LIMIT)
        mx.set_cache_limit(128 << 20)
        from ds41_down_actual_components import run_components
        from probe_deepseek_v41_source_math import compare

        def guarded_write(path, value):
            # Probe helper writes between bounded coefficient loads/oracles and
            # timing cells. The parent concurrently renews/checks the lease.
            host = host_admission(HOST_RESERVE)
            report['minimum_live_host_available_bytes'] = min(report.get('minimum_live_host_available_bytes', host.available_bytes), host.available_bytes)
            atomic(path, value)

        result = run_components(mx, np, spec['down_actual_capture_path'], out, report, compare, guarded_write)
        report['peak_mlx_bytes'] = max(int(mx.get_peak_memory()), report.get('component_peak_mlx_bytes', 0))
        report['after'] = sample_environment(HID_PROFILE)
        report['ownership_after'] = check_ownership(spec)
        host_admission(HOST_RESERVE)
        report['methodological_passed'] = bool(result['complete'] and result['harness_completed'] and result['required_reference_controls_exact'])
        required = {'actual_reference_replay', 'all_candidate_raw_and_owner_exact', 'q1_adversarials_exact'}
        verdicts = result.get('verdicts', {})
        report['numerical_verdicts'] = verdicts
        report['numerical_passed'] = set(verdicts) == required and all(verdicts[key] is True for key in required)
        report['passed'] = report['methodological_passed'] and report['numerical_passed']
        report['passed_scope'] = 'Bounded actual reference replay plus all candidate raw/owner/adversarial bars; no model-quality or controlled-performance claim'
        if report['peak_mlx_bytes'] > MEMORY_LIMIT:
            raise MemoryError('DOWN component allocation ceiling exceeded')
        if digest(source_manifest(ROOT)) != spec['source_snapshot_sha256']:
            raise RuntimeError('Frozen DOWN source changed')
    except BaseException as error:
        report.update(passed=False, error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
    finally:
        if mx is not None:
            try:
                mx.synchronize()
                gc.collect()
                mx.clear_cache()
            except BaseException as error:
                report.update(passed=False, cleanup_error=dict(type=type(error).__name__, message=str(error)))
        atomic(out/'results.json', report)
    print(json.dumps(dict(passed=report['passed'], error=report.get('error'))), flush=True)
    return 0 if report['passed'] else 1


def freeze_source(out):
    manifest = source_manifest(ROOT)
    sha = digest(manifest)
    source = out/'source'
    source.mkdir()
    atomic(out/'source-inputs.json', dict(source_sha256=sha, files=manifest))
    for rel in manifest:
        target = source/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
    frozen = source_manifest(source)
    if digest(frozen) != sha:
        atomic(out/'source-freeze-failure.json', dict(expected_sha256=sha, actual_sha256=digest(frozen),
            changed_files=[k for k in sorted(set(manifest)|set(frozen)) if manifest.get(k) != frozen.get(k)], gpu_lease_acquired=False))
        raise RuntimeError('DOWN source freeze inconsistent')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    atomic(source/'desktop-source.json', dict(source_revision=revision, source_sha256=sha, files=manifest))
    return source, sha, revision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('component', 'unit'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-unit', type=Path)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if args.mode == 'unit':
        if args.capture_unit is not None:
            parser.error('unit reads only its sealed spec; --capture-unit is forbidden')
        return component(out)
    if args.capture_unit is None:
        parser.error('component requires --capture-unit')
    if out.exists():
        raise FileExistsError('Refusing existing DOWN evidence directory: '+str(out))
    host = host_admission(HOST_INITIAL)
    out.mkdir(parents=True, exist_ok=False)
    capture = copy_capture(args.capture_unit, out)
    capture_spec = checked_json(capture/'spec.json')
    source, sha, revision = freeze_source(out)
    spec = dict(schema=SCHEMA, **{k: capture_spec[k] for k in ('checkpoint', 'model', 'context_cap', 'prompt_tokens')},
        out=str(out), job_id=out.name, source_revision=revision, source_snapshot_sha256=sha,
        capture_source_snapshot_sha256=capture_spec['source_snapshot_sha256'],
        down_actual_capture_path=str(capture), down_actual_options={'repeats': 7, 'include_original_e1': True},
        memory_limit_bytes=MEMORY_LIMIT, host_live_reserve_bytes=HOST_RESERVE, host_initial_floor_bytes=HOST_INITIAL,
        host_available_before=host.available_bytes, numerical_environment={'MLX_ENABLE_TF32': '0'},
        mcp_url='http://127.0.0.1:8766/mcp', thermal_gating='User waived; live admission and paired ownership retained',
        scope='Actual PP DOWN stage component diagnostic; no throughput/model-quality/qualification claim', rate_eligible=False)
    atomic(out/'spec.json', spec)
    # CPU checks immediately before lease: no idle ownership while another
    # model or an existing waiter needs the device. The paired wrapper still
    # enforces the actual queue/foreign-process/lock order atomically.
    from probe_deepseek_v4_decode_compare import sample_environment, HID_PROFILE, foreign_waiters
    before = sample_environment(HID_PROFILE)
    waiters = foreign_waiters('')
    atomic(out/'prelease-environment.json', dict(environment=before, foreign_waiters=waiters))
    if before['foreign_model_processes'] or waiters:
        atomic(out/'admission-refused.json', dict(reason='foreign model or queued waiter', gpu_lease_acquired=False))
        raise RuntimeError('DOWN component yields to foreign model/waiter before lease')
    host_admission(HOST_INITIAL)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    result = 1
    child = None
    try:
        with DesktopCPGLease(out, out.name, checkpoint=spec['checkpoint'], source_revision=revision) as lease:
            command = [sys.executable, str(source/'scripts/run_with_gpu_locks.py'), '--session', lease.owner['cpg_session'],
                '--label', out.name, '--desktop-cpg-lease', str(lease.path), '--mcp-url', spec['mcp_url'],
                '--receipt', str(out/'paired-locks.json'), '--', sys.executable,
                str(source/'scripts/research/probe_ds41_down_actual.py'), 'unit', '--output', str(out)]
            try:
                with (out/'process.log').open('x') as log:
                    env = dict(os.environ, PYTHONPATH=str(source/'src'), MLX_ENABLE_TF32='0')
                    child = spawn(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    atomic(out/'process.json', dict(pid=child.pid, command=command))
                    deadline = time.monotonic()+600
                    while child.poll() is None:
                        lease.check()
                        if time.monotonic() > deadline:
                            raise TimeoutError('Bounded DOWN component deadline')
                        time.sleep(1)
                    result = child.returncode
                    lease.success = result == 0
            finally:
                if child is not None:
                    child.stop(timeout=20)
    finally:
        atomic(out/'release-verified.json', dict(finished_at=time.time(),
            owners_absent=all(not Path(p).exists() for p in ('/Users/Shared/mlxuag/gpu.lock/owner.json', '/tmp/gpu.lock/owner.json')),
            source_unchanged=digest(source_manifest(source)) == sha))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
