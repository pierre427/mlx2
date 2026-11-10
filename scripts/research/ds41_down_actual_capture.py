# SPDX-License-Identifier: MIT
"""Bounded reference-returning actual PP DOWN capture; caller owns GPU.

Import and load_capture are CPU only. The hook observes existing leased expert
objects inside the batched routed provider; it never acquires another expert.
Serialization/evaluation are interventions, not ordinary latency measurements.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

SCHEMA = 'mlx2.source-down-actual-capture.v1'
MAX_CASES = 10
MAX_ARRAY_BYTES = 64 << 20
MAX_CASE_BYTES = 6 << 20
_ACTIVE = False
_EVIDENCE_SHA256 = '7b90f06e4acd3017fd53c06eaec8209ecc0c893056946e0d436bbf751d60f40f'
ALL_STRATA = tuple([f'original-m{a}-{b}' for a, b in ((1, 1), (2, 3), (4, 7), (8, 12), (13, 32), (33, 96), (97, 99))]
                   + [f'q1-m{a}-{b}' for a, b in ((1, 1), (2, 3), (4, 99))])


@lru_cache(maxsize=1)
def _evidence():
    from mlx2 import research_evidence
    if hashlib.sha256(Path(research_evidence.__file__).read_bytes()).hexdigest() != _EVIDENCE_SHA256:
        raise ValueError('DOWN research evidence dependency seal differs')
    return research_evidence


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def band(kind, rows):
    if kind not in ('original', 'q1') or type(rows) is not int or not 1 <= rows <= 99:
        raise ValueError('Unsupported actual DOWN kind/row count')
    limits = ((1, 1), (2, 3), (4, 7), (8, 12), (13, 32), (33, 96), (97, 99)) if kind == 'original' else ((1, 1), (2, 3), (4, 99))
    low, high = next((a, b) for a, b in limits if a <= rows <= b)
    return f'{kind}-m{low}-{high}'


def coefficient_record(coefficients, kind):
    expected = {'qweight': ((5120, 288), np.uint32), 'scales': ((5120, 72), np.uint8)}
    if kind == 'q1':
        expected = {'qweight': ((5120, 72), np.uint32), 'scales': ((5120, 36), np.float16), 'biases': ((5120, 36), np.float16)}
    elif kind != 'original':
        raise ValueError('Unknown DOWN coefficient kind')
    if set(coefficients) != set(expected):
        raise ValueError('DOWN coefficient inventory')
    result = {}
    for key, (shape, dtype) in expected.items():
        value = np.asarray(coefficients[key])
        if value.shape != shape or value.dtype != dtype:
            raise ValueError('DOWN coefficient geometry/dtype')
        result[key] = dict(shape=list(shape), dtype=str(value.dtype), sha256=digest(value))
    return result


def save_case(unit, name, *, layer, expert, kind, activation, raw, boundary, coefficients, route_rows, route_locations):
    """CPU serializer. Saves no expert weights and refuses existing paths."""
    if type(layer) is not int or not 0 <= layer < 40 or type(expert) is not int or not 0 <= expert < 384:
        raise ValueError('DOWN expert identity')
    rows = activation.shape[0]
    label = band(kind, rows)
    arrays = {'activation': activation, 'reference_raw': raw, 'reference_boundary': boundary}
    for key, value in arrays.items():
        shape = (rows, 2304 if key == 'activation' else 5120)
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('DOWN capture array shape/dtype/finite contract')
    if any(np.any(arrays[key].view(np.uint32) & 0xffff) for key in ('activation', 'reference_boundary')):
        raise ValueError('DOWN activation/owner is not on BF16 grid')
    raw_bits = raw.view(np.uint32)
    rounded = (raw_bits + np.uint32(0x7fff) + ((raw_bits >> 16) & 1)) & np.uint32(0xffff0000)
    if rounded.tobytes() != boundary.tobytes():
        raise ValueError('DOWN captured owner differs from raw BF16 round')
    if (route_rows.dtype != np.uint32 or route_locations.dtype != np.uint32 or route_rows.shape != (rows,)
            or route_locations.shape != (rows,) or np.any(route_rows >= 99)
            or np.any(route_locations >= 99*6) or not np.array_equal(route_rows, route_locations//6)
            or len(np.unique(route_locations)) != rows or np.any(np.diff(route_locations.astype(np.int64)) <= 0)):
        raise ValueError('DOWN CPU job route metadata')
    total = sum(v.nbytes for v in arrays.values())
    if total > MAX_CASE_BYTES:
        raise ValueError('DOWN per-case array budget')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.npz', name):
        raise ValueError('Unsafe DOWN capture filename')
    record = coefficient_record(coefficients, kind)
    path = Path(unit)/name
    with path.open('xb') as file:
        np.savez(file, **arrays)
    return dict(layer=layer, expert=expert, kind=kind, rows=rows, band=label,
                input_width=2304, output_width=5120, path=name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=digest(v)) for k, v in arrays.items()},
                coefficients=record, captured_array_bytes=total,
                route_rows=route_rows.tolist(), route_locations=route_locations.tolist(),
                input_layout={k: dict(export_byte_strides=list(v.strides), export_c_contiguous=bool(v.flags.c_contiguous),
                                     device_strides_observed=False) for k, v in arrays.items()},
                input_bf16_roundtrip_exact=True, source_result_always_returned=True)


def _validate_case(case, arrays):
    rows = case.get('rows')
    if band(case.get('kind'), rows) != case.get('band') or case.get('input_width') != 2304 or case.get('output_width') != 5120:
        raise ValueError('DOWN capture geometry/band')
    if type(case.get('layer')) is not int or not 0 <= case['layer'] < 40 or type(case.get('expert')) is not int or not 0 <= case['expert'] < 384:
        raise ValueError('DOWN capture expert identity')
    if set(arrays) != {'activation', 'reference_raw', 'reference_boundary'}:
        raise ValueError('DOWN capture array inventory')
    for key, value in arrays.items():
        if (value.dtype != np.float32 or value.shape != (rows, 2304 if key == 'activation' else 5120)
                or not np.isfinite(value).all() or case['arrays'].get(key) != dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest(value))):
            raise ValueError('DOWN capture array mismatch')
    if sum(v.nbytes for v in arrays.values()) != case.get('captured_array_bytes') or sum(v.nbytes for v in arrays.values()) > MAX_CASE_BYTES:
        raise ValueError('DOWN capture per-case budget')
    # Reuse the independent owner/grid/route validator without writing data.
    if any(np.any(arrays[k].view(np.uint32) & 0xffff) for k in ('activation', 'reference_boundary')):
        raise ValueError('DOWN capture BF16 grid')
    bits = arrays['reference_raw'].view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    if rounded.tobytes() != arrays['reference_boundary'].tobytes():
        raise ValueError('DOWN capture owner round')
    if any(not isinstance(case.get(key), list) or any(type(v) is not int for v in case[key]) for key in ('route_locations', 'route_rows')):
        raise ValueError('DOWN capture CPU route integer inventory')
    locations = np.asarray(case['route_locations'], dtype=np.int64)
    selected = np.asarray(case['route_rows'], dtype=np.int64)
    if (locations.shape != (rows,) or selected.shape != (rows,) or np.any(locations < 0) or np.any(locations >= 594)
            or not np.array_equal(selected, locations//6) or np.any(np.diff(locations) <= 0)):
        raise ValueError('DOWN capture route metadata')
    expected = {'qweight': ((5120, 288), 'uint32'), 'scales': ((5120, 72), 'uint8')}
    if case['kind'] == 'q1':
        expected = {'qweight': ((5120, 72), 'uint32'), 'scales': ((5120, 36), 'float16'), 'biases': ((5120, 36), 'float16')}
    if set(case['coefficients']) != set(expected):
        raise ValueError('DOWN coefficient record inventory')
    for key, (shape, dtype) in expected.items():
        value = case['coefficients'][key]
        if value.get('shape') != list(shape) or value.get('dtype') != dtype or not re.fullmatch('[0-9a-f]{64}', value.get('sha256', '')):
            raise ValueError('DOWN coefficient record geometry/hash')


def load_capture(path, *, checkpoint=None, source_snapshot=None):
    """Validate bounded manifest and NPZs; no MLX import or device operations."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('down-actual-capture-*.json'))
        if len(files) != 1:
            raise ValueError('DOWN capture directory needs exactly one manifest')
        path = files[0]
    if path.is_symlink():
        raise ValueError('DOWN manifest symlink')
    report = json.loads(path.read_text())
    if (report.get('schema') != SCHEMA or report.get('status') != 'completed'
            or report.get('model_forward_completed') is not True or report.get('source_result_always_returned') is not True
            or report.get('candidate_used_by_model') is not False or report.get('input_rows') != 99 or report.get('frontier') != 0):
        raise ValueError('Incomplete/nonreference DOWN capture')
    for key, expected in (('checkpoint', checkpoint), ('source_snapshot_sha256', source_snapshot)):
        if not report.get(key) or expected is not None and report[key] != expected:
            raise ValueError('DOWN capture identity mismatch: '+key)
    if not report.get('source_revision') or report.get('observed_layers') != list(range(40)):
        raise ValueError('DOWN capture requires complete 40-layer scan')
    cases = report.get('cases', [])
    if not 2 <= len(cases) <= MAX_CASES or len({c['band'] for c in cases}) != len(cases):
        raise ValueError('DOWN case count/duplicate stratum')
    if {c['band'] for c in cases} != set(report.get('stratum_call_counts', {})) or {c['kind'] for c in cases} != {'original', 'q1'}:
        raise ValueError('DOWN capture misses observed strata/kinds')
    if any(type(v) is not int or v <= 0 for v in report['stratum_call_counts'].values()):
        raise ValueError('DOWN census counts')
    if report.get('observed_down_calls') != sum(report['stratum_call_counts'].values()):
        raise ValueError('DOWN census total mismatch')
    if report.get('missing_strata') != sorted(set(ALL_STRATA)-set(report['stratum_call_counts'])):
        raise ValueError('DOWN observed/missing stratum receipt mismatch')
    if report.get('expert_budget_bytes') != 75_000_000_000 or report.get('original_hot_bytes') != 60_000_000_000:
        raise ValueError('DOWN capture pool budget changed')
    loaded = []
    total = 0
    for case in cases:
        name = case['path']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.npz', name):
            raise ValueError('Unsafe DOWN capture path')
        file = path.parent/name
        arrays = _evidence().load_array_archive(
            file, specs={key: ((case['rows'], 2304 if key == 'activation' else 5120), np.float32)
                         for key in ('activation', 'reference_raw', 'reference_boundary')},
            sha256=case['sha256'],
            identity={key: report[key] for key in ('checkpoint', 'source_snapshot_sha256')},
            expected_identity={'checkpoint': checkpoint or report['checkpoint'],
                               'source_snapshot_sha256': source_snapshot or report['source_snapshot_sha256']},
            max_archive_bytes=MAX_CASE_BYTES+4096, max_container_bytes=MAX_CASE_BYTES+4096,
            max_array_bytes=min(MAX_CASE_BYTES, MAX_ARRAY_BYTES-total), backend=np,
            allowed_versions=((1, 0), (2, 0), (3, 0)))
        _validate_case(case, arrays)
        total += sum(v.nbytes for v in arrays.values())
        if total > MAX_ARRAY_BYTES:
            raise ValueError('DOWN aggregate array budget')
        loaded.append((case, arrays))
    if total != report.get('captured_array_bytes'):
        raise ValueError('DOWN aggregate byte receipt mismatch')
    return report, loaded


@contextmanager
def instrument(model, unit, label, index, *, input_rows, frontier):
    """Observe existing routed acquires and QMM calls; always return source."""
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError('Nested/concurrent DOWN capture refused')
    if input_rows != 99 or frontier != 0 or type(index) is not int or index < 0:
        raise ValueError('DOWN capture is PP99/frontier0 only')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', label):
        raise ValueError('Unsafe DOWN capture label')
    import mlx.core as mx
    from mlx2.runtime.gguf import source_q1
    from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs
    from probe_deepseek_v4_packed import atomic
    unit = Path(unit)
    output = unit/f'down-actual-capture-{index:04d}.json'
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads((unit/'spec.json').read_text())
    policy = spec.get('expert_policy', {})
    if (policy.get('budget_bytes') != 75_000_000_000 or policy.get('hot_bytes') != 60_000_000_000
            or policy.get('mode') != 'hot_original_q1'
            or spec.get('extra_source_policy', {}).get('deepseek_source_prefill_batch') != 'gate_up'):
        raise ValueError('DOWN capture requires unchanged75GB60/15 gate_up profile')
    weights = model.weights
    stream = weights.expert_stream
    before_provider = weights._prefill_batched_gate_up
    before_acquire = stream.acquire
    before_qmm = mx.quantized_matmul
    before_q1 = source_q1.matmul
    local_provider = '_prefill_batched_gate_up' in vars(weights)
    local_acquire = 'acquire' in vars(stream)
    active = None
    seen = set()
    census = Counter()
    layers = set()
    report = dict(schema=SCHEMA, status='running', label=label, forward_index=index,
                  input_rows=input_rows, frontier=frontier, source_result_always_returned=True,
                  candidate_used_by_model=False, model_forward_completed=False, performance_measurement=False,
                  rate_eligible=False, intervention='Evaluated selected operands and CPU copies; reference output returned; no benchmark rate',
                  checkpoint=spec['checkpoint'], source_revision=spec['source_revision'],
                  source_snapshot_sha256=spec['source_snapshot_sha256'], expert_budget_bytes=policy['budget_bytes'],
                  original_hot_bytes=policy['hot_bytes'], expert_policy=policy,
                  selected_policies=spec.get('extra_source_policy', {}),
                  original_artifact_fingerprint=stream.artifact['identity']['fingerprint'],
                  q1_bank_binding=stream.q1_binding, hot_map_binding=stream.hot_binding,
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cases=[], captured_array_bytes=0, maximum_capture_array_bytes=MAX_ARRAY_BYTES,
                  selection='First existing routed DOWN in every observed representation/row band',
                  expected_route_scope='Complete PP99, 40 routed provider windows; no extra acquire')

    def provider(x, values, scores, prefix, order):
        nonlocal active
        if active is not None:
            raise RuntimeError('Reentrant routed DOWN provider')
        if values.shape != (99, 6) or np.any(np.diff(np.sort(values, axis=1), axis=1) == 0):
            raise ValueError('DOWN scope requires99rows with six distinct routed experts')
        layer = int(prefix.split('.')[1])
        if layer in layers:
            raise ValueError('Duplicate routed provider layer')
        kinds = {int(e): stream.kind((layer, int(e))) for e in np.unique(values)}
        plan = plan_jobs(values, kinds)
        active = dict(layer=layer, prefix=prefix, jobs={j.expert_id: j for j in plan.jobs}, tags={}, acquired=set(), observed=set())
        try:
            result = before_provider(x, values, scores, prefix, order)
            if active['acquired'] != set(active['jobs']) or active['observed'] != set(active['jobs']):
                raise ValueError('DOWN capture did not observe exactly one existing acquire/down per job')
            layers.add(layer)
            return result
        finally:
            active = None

    def acquire(prefix, expert):
        if active is not None and (prefix != active['prefix'] or expert not in active['jobs'] or expert in active['acquired']):
            raise ValueError('Unexpected routed DOWN acquire ownership')
        data = before_acquire(prefix, expert)
        if active is not None:
            job = active['jobs'][expert]
            kind = 'q1' if 'down.biases' in data else 'original'
            if kind != job.kind:
                raise ValueError('Acquired DOWN kind differs from CPU plan')
            active['acquired'].add(expert)
            # References belong to the existing provider lease, not a new cache.
            active['tags'][id(data['down.qweight'])] = (job, data)
        return data

    def capture(x, weight, scales, biases, result, kind):
        if active is None:
            return result
        tagged = active['tags'].get(id(weight))
        if tagged is None:
            return result  # Dense/shared/lookalike geometry is never sufficient.
        job, data = tagged
        if (data['down.qweight'] is not weight or data['down.scales'] is not scales
                or (data.get('down.biases') is not biases) or job.kind != kind
                or x.shape != (job.count, 2304) or x.dtype != mx.float32
                or job.expert_id in active['observed']):
            raise ValueError('DOWN operand differs from leased job identity/shape')
        active['observed'].add(job.expert_id)
        stratum = band(kind, job.count)
        census[stratum] += 1
        if stratum in seen:
            return result
        owner = result.astype(mx.bfloat16).astype(mx.float32)
        mx.eval(x, result, owner, weight, scales, *([biases] if biases is not None else []))
        arrays = [np.array(v) for v in (x, result, owner)]
        total = sum(v.nbytes for v in arrays)
        if report['captured_array_bytes']+total > MAX_ARRAY_BYTES or len(report['cases']) >= MAX_CASES:
            raise ValueError('DOWN capture aggregate ceiling')
        coefficients = dict(qweight=np.array(weight), scales=np.array(scales))
        if biases is not None:
            coefficients['biases'] = np.array(biases)
        name = f'down-actual-{label}-{index:04d}-{stratum}.npz'
        record = save_case(unit, name, layer=active['layer'], expert=job.expert_id, kind=kind,
                           activation=arrays[0], raw=arrays[1], boundary=arrays[2], coefficients=coefficients,
                           route_rows=job.rows, route_locations=job.locations)
        report['cases'].append(record)
        report['captured_array_bytes'] += total
        seen.add(stratum)
        return result

    def qmm(x, weight, *args, **kwargs):
        result = before_qmm(x, weight, *args, **kwargs)
        if active is not None and id(weight) in active['tags']:
            if args or kwargs.get('mode') != 'mxfp4' or kwargs.get('transpose', True) is not True or kwargs.get('biases') is not None:
                raise ValueError('Unexpected original DOWN QMM signature')
            return capture(x, weight, kwargs['scales'], None, result, 'original')
        return result

    def q1(x, weight, scales, biases):
        return capture(x, weight, scales, biases, before_q1(x, weight, scales, biases), 'q1')

    started = time.perf_counter()
    _ACTIVE = True
    weights._prefill_batched_gate_up = provider
    stream.acquire = acquire
    mx.quantized_matmul = qmm
    source_q1.matmul = q1
    try:
        yield report
        if layers != set(range(40)):
            raise ValueError('DOWN capture incomplete40-layer provider scan')
        if {case['kind'] for case in report['cases']} != {'original', 'q1'} or seen != set(census):
            raise ValueError('DOWN capture missing observed precision strata')
        report.update(status='completed', model_forward_completed=True)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        mx.quantized_matmul = before_qmm
        source_q1.matmul = before_q1
        if local_provider:
            weights._prefill_batched_gate_up = before_provider
        else:
            del weights._prefill_batched_gate_up
        if local_acquire:
            stream.acquire = before_acquire
        else:
            del stream.acquire
        _ACTIVE = False
        report.update(wall_seconds=time.perf_counter()-started, observed_layers=sorted(layers),
                      stratum_call_counts=dict(sorted(census.items())), observed_down_calls=sum(census.values()),
                      missing_strata=sorted(set(ALL_STRATA)-set(census)))
        atomic(output, report)
