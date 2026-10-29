# SPDX-License-Identifier: MIT
"""Reference-returning complete original expert capture; no GPU work on import.

Import, selection, serialization and load_capture are CPU-only. The caller
owns the model/window. No additional expert acquisition or coefficient copy.
Provenance: provenance/deepseek-v41-native-ffn-capture.json.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import zipfile

import numpy as np

SCHEMA = 'mlx2.native-ffn-actual-capture.v1'
MAX_ARRAY_BYTES = 32 << 20
MAX_CASES = 2
_LOCK = threading.Lock()


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def shapes(rows):
    result = {'shared_gu_input': (99, 5120), 'gu_input': (rows, 5120), 'down_raw': (rows, 5120), 'down_owned': (rows, 5120),
        'assembly_output': (rows, 5120), 'assembly_parts': (rows, 6, 5120),
        'values': (99, 6), 'scores': (99, 6), 'order': (594,)}
    result.update({key: (rows, 2304) for key in ('gate_raw', 'up_raw', 'gate_owned', 'up_owned',
        'weighted_raw', 'weighted_owned', 'middle', 'middle_reference')})
    result.update({key: (rows,) for key in ('route_rows', 'route_locations', 'route_scores', 'assembly_rank')})
    return result


UINT_ARRAYS = frozenset(('values', 'order', 'route_rows', 'route_locations', 'assembly_rank'))


def estimated_bytes(rows):
    return sum(int(np.prod(shape))*4 for shape in shapes(rows).values())


def choose_jobs(jobs, layer, *, remaining_cases=MAX_CASES, remaining_bytes=MAX_ARRAY_BYTES):
    """Dynamic jobs, preferred IDs only if actually eligible; no row fiction."""
    eligible = [(i, job) for i, job in enumerate(jobs) if job.kind == 'original' and 16 <= job.count <= 99]
    preferred = {38: 0, 50: 1} if layer == 0 else {}
    eligible.sort(key=lambda item: (preferred.get(item[1].expert_id, 2), item[1].expert_id))
    result = []
    for index, job in eligible:
        size = estimated_bytes(job.count)
        if len(result) < remaining_cases and size <= remaining_bytes:
            result.append(index)
            remaining_bytes -= size
    return tuple(result)


def bf16(value):
    bits = np.ascontiguousarray(value).view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def validate_arrays(case, arrays):
    rows = case.get('rows')
    if (type(rows) is not int or not 16 <= rows <= 99 or case.get('kind') != 'original'
            or type(case.get('layer')) is not int or not 0 <= case['layer'] < 40
            or type(case.get('expert')) is not int or not 0 <= case['expert'] < 384):
        raise ValueError('Native FFN actual case identity/geometry')
    expected = shapes(rows)
    if set(arrays) != set(expected):
        raise ValueError('Native FFN array inventory')
    for key, shape in expected.items():
        a = arrays[key]
        if (a.shape != shape or a.dtype != (np.uint32 if key in UINT_ARRAYS else np.float32)
                or not np.isfinite(a).all()):
            raise ValueError('Native FFN array geometry/dtype/finite: '+key)
    for raw, owner in (('gate_raw', 'gate_owned'), ('up_raw', 'up_owned'),
                       ('weighted_raw', 'weighted_owned'), ('down_raw', 'down_owned')):
        if not _same(bf16(arrays[raw]), arrays[owner]):
            raise ValueError('Native FFN BF16 owner boundary: '+raw)
    for key in ('shared_gu_input', 'gu_input', 'middle', 'gate_owned', 'up_owned', 'weighted_owned', 'down_owned'):
        if np.any(arrays[key].view(np.uint32) & 0xffff):
            raise ValueError('Native FFN expected BF16-grid values: '+key)
    if not _same(arrays['middle'], arrays['middle_reference']):
        raise ValueError('Native FFN actual middle differs from ordinary scored QAT replay')
    values, order = arrays['values'], arrays['order']
    locations, selected = arrays['route_locations'], arrays['route_rows']
    if (np.any(values > 383) or np.any(np.diff(np.sort(values, axis=1), axis=1) == 0)
            or not np.array_equal(order, np.argsort(values.reshape(-1), kind='stable'))
            or np.any(locations >= 594) or np.any(selected >= 99)
            or not np.array_equal(selected, locations//6)
            or np.any(np.diff(locations.astype(np.int64)) <= 0)
            or not np.all(values.reshape(-1)[locations] == case['expert'])
            or not _same(arrays['route_scores'], arrays['scores'].reshape(-1)[locations])):
        raise ValueError('Native FFN actual score/route mapping')
    expected_locations = np.flatnonzero(values.reshape(-1) == case['expert']).astype(np.uint32)
    if not np.array_equal(locations, expected_locations):
        raise ValueError('Native FFN split/omitted whole expert rows')
    if not _same(arrays['gu_input'], arrays['shared_gu_input'][selected]):
        raise ValueError('Native FFN actual shared-input gather differs from admitted GU input')
    expert_order, counts = np.unique(values, return_counts=True)
    if (case.get('expert_order') != expert_order.tolist() or case.get('expert_row_counts') != counts.tolist()
            or type(case.get('part_index')) is not int or not 0 <= case['part_index'] < len(expert_order)
            or int(expert_order[case['part_index']]) != case['expert'] or case.get('same_forward') is not True):
        raise ValueError('Native FFN complete assembly part placement')
    ranks = np.sum(values[selected] < case['expert'], axis=1).astype(np.uint32)
    if not np.array_equal(arrays['assembly_rank'], ranks):
        raise ValueError('Native FFN assembly rank')
    chosen = arrays['assembly_parts'][np.arange(rows), ranks]
    if not _same(chosen, arrays['down_owned']):
        raise ValueError('Native FFN selected assembly contribution differs from down owner')
    total = np.zeros((rows, 5120), np.float32)
    for slot in range(6):
        np.add(total, arrays['assembly_parts'][:, slot], out=total)
    if not _same(total, arrays['assembly_output']):
        raise ValueError('Native FFN ordered F32 assembly replay')
    size = sum(a.nbytes for a in arrays.values())
    if size != estimated_bytes(rows) or size > MAX_ARRAY_BYTES:
        raise ValueError('Native FFN bounded capture bytes')
    return size


def save_case(unit, name, case, arrays):
    """CPU-only immutable serialization; coefficients are never included."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.npz', name):
        raise ValueError('Native FFN unsafe filename')
    size = validate_arrays(case, arrays)
    path = Path(unit)/name
    with path.open('xb') as output:
        np.savez(output, **arrays)
    return dict(case, path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        captured_array_bytes=size, arrays={k: dict(shape=list(a.shape), dtype=str(a.dtype), sha256=digest(a)) for k, a in arrays.items()})


def load_capture(path, *, checkpoint=None, source_snapshot=None):
    """Strict bounded parser, no MLX import or tensor/device creation."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Native FFN capture symlink')
    if path.is_dir():
        files = sorted(path.glob('native-ffn-capture-*.json'))
        if len(files) != 1:
            raise ValueError('Native FFN directory requires one manifest')
        path = files[0]
    if path.is_symlink() or path.parent.is_symlink() or path.stat().st_size > 2 << 20:
        raise ValueError('Native FFN unsafe manifest')
    report = json.loads(path.read_text())
    if (report.get('schema') != SCHEMA or report.get('status') != 'completed'
            or report.get('model_forward_completed') is not True or report.get('source_result_always_returned') is not True
            or report.get('candidate_used_by_model') is not False or report.get('rate_eligible') is not False
            or report.get('input_rows') != 99 or report.get('frontier') != 0
            or report.get('observed_layers') != list(range(40)) or report.get('extra_expert_acquisitions') != 0
            or report.get('extra_coefficient_copy_bytes') != 0 or report.get('expert_budget_bytes') != 75_000_000_000
            or report.get('original_hot_bytes') != 60_000_000_000):
        raise ValueError('Incomplete or nonreference native FFN capture')
    for key, wanted in (('checkpoint', checkpoint), ('source_snapshot_sha256', source_snapshot)):
        if not report.get(key) or wanted is not None and report[key] != wanted:
            raise ValueError('Native FFN capture identity mismatch: '+key)
    if not report.get('source_revision') or not re.fullmatch('[0-9a-f]{64}', report.get('helper_sha256', '')):
        raise ValueError('Native FFN missing source/helper identity')
    if report.get('capture_policy') != {'prefill_batch': 'gate_up', 'expert_math': 'routed_fused_fp32',
            'wide_math': 'reference', 'wide_dtype': 'reference'}:
        raise ValueError('Native FFN owning execution policy')
    if any(not re.fullmatch('[0-9a-f]{64}', report.get(key, '')) for key in
            ('original_artifact_fingerprint', 'q1_bank_binding', 'hot_map_binding')):
        raise ValueError('Native FFN artifact/precision binding')
    cases = report.get('cases', [])
    if not 1 <= len(cases) <= MAX_CASES or len({(c['layer'], c['expert']) for c in cases}) != len(cases):
        raise ValueError('Native FFN actual case count/duplicate')
    loaded, names, total = [], set(), 0
    for case in cases:
        name = case['path']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.npz', name) or name in names:
            raise ValueError('Native FFN unsafe/duplicate case path')
        names.add(name)
        file = path.parent/name
        if file.is_symlink() or file.stat().st_size > MAX_ARRAY_BYTES+16384:
            raise ValueError('Native FFN file size/symlink')
        if hashlib.sha256(file.read_bytes()).hexdigest() != case['sha256']:
            raise ValueError('Native FFN NPZ hash')
        expected_names = {k+'.npy' for k in shapes(case['rows'])}
        with zipfile.ZipFile(file) as archive:
            entries = archive.infolist()
            if (len(entries) != len(expected_names) or {e.filename for e in entries} != expected_names
                    or sum(e.file_size for e in entries) > MAX_ARRAY_BYTES+16384):
                raise ValueError('Native FFN decompression inventory/budget')
            # Reject a tiny malformed file claiming a huge allocation before
            # NumPy creates an array. ZIP byte ceilings alone cannot do this.
            for entry in entries:
                key = entry.filename[:-4]
                with archive.open(entry) as value:
                    version = np.lib.format.read_magic(value)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(value, max_header_size=4096)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(value, max_header_size=4096)
                    else:
                        raise ValueError('Native FFN unsupported NPY version')
                    if (shape != shapes(case['rows'])[key] or dtype != np.dtype(np.uint32 if key in UINT_ARRAYS else np.float32)
                            or entry.file_size-value.tell() != int(np.prod(shape))*4):
                        raise ValueError('Native FFN NPY header geometry/dtype/payload')
        with np.load(file, allow_pickle=False) as archive:
            arrays = {k: np.array(archive[k], copy=True) for k in archive.files}
        size = validate_arrays(case, arrays)
        records = {k: dict(shape=list(a.shape), dtype=str(a.dtype), sha256=digest(a)) for k, a in arrays.items()}
        if records != case.get('arrays') or size != case.get('captured_array_bytes'):
            raise ValueError('Native FFN array hashes/byte receipt')
        binding = case.get('source_binding', {})
        if (binding.get('live_lease_verified') is not True or binding.get('kind') != 'original'
                or binding.get('weight_payload_exported') is not False
                or binding.get('source_fingerprint') != report['original_artifact_fingerprint']
                or any(not re.fullmatch('[0-9a-f]{64}', binding.get(key, '')) for key in ('source_seals_sha256', 'layout_sha256'))):
            raise ValueError('Native FFN source/lease binding')
        total += size
        if total > MAX_ARRAY_BYTES:
            raise ValueError('Native FFN aggregate budget')
        loaded.append((case, arrays))
    if total != report.get('captured_array_bytes'):
        raise ValueError('Native FFN total byte receipt')
    return report, loaded


def source_binding(weights, stream, layer, job, data):
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_source_weights import expert_names
    from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert, canonical_layout
    key = (layer, job.expert_id)
    if (not stream._active or stream._leased.get(key) is not data
            or not isinstance(data, PackedExpert) or data.layout != canonical_layout('original')
            or data.backing.dtype != mx.uint8 or data.backing.shape != (data.layout.nbytes,)
            or set(data) != {t.name for t in data.layout.tensors}):
        raise ValueError('Native FFN expert is not the exact existing original lease')
    dtypes = {'<u4': mx.uint32, '|u1': mx.uint8}
    if any(data[t.name].shape != t.shape or data[t.name].dtype != dtypes[t.dtype] for t in data.layout.tensors):
        raise ValueError('Native FFN leased typed views differ from canonical backing layout')
    artifact = weights.artifact
    files = sorted({artifact['tensors'][name]['file'] for name in expert_names(*key)})
    seals = {name: artifact['seals'][name] for name in files}
    layout = [dict(name=t.name, offset=t.offset, dtype=t.dtype, shape=list(t.shape)) for t in data.layout.tensors]
    encoded = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(kind='original', source_fingerprint=artifact['identity']['fingerprint'],
        files=files, source_seals_sha256=encoded(seals), layout_sha256=encoded(layout),
        tensor_layout=layout, backing_nbytes=data.layout.nbytes, backing_object_id=id(data.backing),
        live_lease_verified=True, weight_payload_exported=False,
        hash_scope='Checked provider source/file seals and canonical layout, not a GPU weight-payload readback')


@contextmanager
def instrument(model, unit, label, index, *, input_rows, frontier):
    """Observe source GU/middle/DOWN/assembly; every source object is returned."""
    if (input_rows != 99 or frontier != 0 or type(index) is not int or index < 0
            or not re.fullmatch(r'[A-Za-z0-9_.-]+', label)):
        raise ValueError('Native FFN capture is bounded PP99/frontier0 only')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Nested/concurrent native FFN capture refused')
    patches, report, output = [], None, None
    started = time.perf_counter()
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs
        from mlx2.adapters import deepseek_v41_source_math as source_math
        from mlx2.runtime.gguf import source_middle_fusion as source_middle
        from mlx2.runtime.models.deepseek_v41_quant import quantize
        from probe_deepseek_v4_packed import atomic
        unit = Path(unit)
        output = unit/f'native-ffn-capture-{index:04d}.json'
        if output.exists():
            raise FileExistsError(output)
        spec = json.loads((unit/'spec.json').read_text())
        helper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        policy = spec.get('expert_policy', {})
        weights, stream = model.weights, model.weights.expert_stream
        if (spec.get('native_ffn_capture') is not True or spec.get('resident_discovery') is not True
                or spec.get('rate_eligible') is not False
                or any(spec.get(key) is True for key in ('woa_shadow', 'small_m_shadow', 'down_actual_capture', 'q1_down_shadow'))
                or spec.get('harness_sha256', {}).get('ds41_native_ffn_capture.py') != helper_sha
                or policy.get('budget_bytes') != 75_000_000_000 or policy.get('hot_bytes') != 60_000_000_000
                or policy.get('mode') != 'hot_original_q1'
                or weights.source_prefill_batch_policy != 'gate_up' or weights.expert_math_policy != 'routed_fused_fp32'
                or getattr(weights, 'source_prefill_wide_math', 'reference') != 'reference'
                or getattr(weights, 'source_prefill_wide_dtype', 'reference') != 'reference'
                or getattr(weights, 'source_prefill_down_policy', 'reference') != 'reference'
                or getattr(weights, 'source_ffn_input_policy', 'reference') != 'reference'
                or getattr(weights, 'source_fp8_pp_policy', None) != 'native_bf16_qb_wob_v1'
                or not getattr(stream, 'packed_buffers', False)):
            raise ValueError('Native FFN capture requires sealed reference-wide75GB60/15 discovery')
        identity = {k: spec.get(k) for k in ('checkpoint', 'source_revision', 'source_snapshot_sha256')}
        if any(not isinstance(v, str) or not v for v in identity.values()):
            raise ValueError('Native FFN missing source/checkpoint identity')
        report = dict(schema=SCHEMA, **identity, status='running', label=label, forward_index=index,
            helper_sha256=helper_sha, input_rows=99, frontier=0, rate_eligible=False,
            source_result_always_returned=True, candidate_used_by_model=False, model_forward_completed=False,
            expert_budget_bytes=policy['budget_bytes'], original_hot_bytes=policy['hot_bytes'],
            original_artifact_fingerprint=stream.artifact['identity']['fingerprint'],
            q1_bank_binding=stream.q1_binding, hot_map_binding=stream.hot_binding,
            extra_expert_acquisitions=0, extra_coefficient_copy_bytes=0,
            extra_stock_middle_replay_graphs=0,
            capture_activation_concat_bytes=0, captured_array_bytes=0, maximum_array_bytes=MAX_ARRAY_BYTES,
            capture_policy={'prefill_batch': 'gate_up', 'expert_math': 'routed_fused_fp32',
                'wide_math': 'reference', 'wide_dtype': 'reference'},
            intervention='Synchronized activation capture/replay only; original returned; no throughput/model-quality claim',
            diagnostic_frontiers=['gate_owned', 'up_owned', 'weighted_raw', 'weighted_owned', 'middle_reference'],
            frontier_scope='Diagnostic owners/scored intermediates reify the fused epilogue value boundaries; they are not claims of standalone serving buffers. Actual middle/down/assembly are captured from the source graph.',
            shared_input_scope='Actual layer-shared prepared QAT input; layer-shared QAT is outside this expert-stage capture. Row gather, conversion and complete expert math can be replayed from these values.',
            selection='Up to2 actual whole original expertsM>=16, prefer L0E38/E50 when eligible, bounded by32MiB',
            cases=[], observed_layers=[], observed_acquire_jobs=0)
        active, layers = None, set()
        before_provider, before_acquire = weights._prefill_batched_gate_up, stream.acquire
        before_qmm, before_middle, before_combine = mx.quantized_matmul, source_middle.middle, source_math.ordered_combine
        def patch(obj, name, fn):
            patches.append((obj, name, name in vars(obj), getattr(obj, name)))
            setattr(obj, name, fn)
        def provider(x, values, scores, prefix, order):
            nonlocal active
            layer = int(prefix.split('.')[1])
            if active is not None or layer in layers or values.shape != (99, 6):
                raise ValueError('Native FFN provider scope/reentrancy')
            kinds = {int(e): stream.kind((layer, int(e))) for e in np.unique(values)}
            plan = plan_jobs(values, kinds)
            chosen = choose_jobs(plan.jobs, layer, remaining_cases=MAX_CASES-len(report['cases']),
                remaining_bytes=MAX_ARRAY_BYTES-report['captured_array_bytes'])
            active = dict(layer=layer, prefix=prefix, jobs=plan.jobs, x=x, values=values, scores=scores,
                order=order, chosen=set(chosen), acquired={}, tags={}, raw_tags={}, stages={})
            try:
                result = before_provider(x, values, scores, prefix, order)
                if len(active['acquired']) != len(plan.jobs) or any(j not in active['stages'] or not active['stages'][j].get('saved') for j in chosen):
                    raise ValueError('Native FFN did not observe all existing acquires/complete selected stages')
                layers.add(layer)
                return result
            finally:
                active = None
        def acquire(prefix, expert):
            if active is not None and (prefix != active['prefix'] or expert in active['acquired']):
                raise ValueError('Native FFN extra/unexpected existing acquire')
            data = before_acquire(prefix, expert)
            if active is not None:
                indexes = [j for j, job in enumerate(active['jobs']) if job.expert_id == expert]
                if len(indexes) != 1:
                    raise ValueError('Native FFN acquired expert absent from plan')
                j = indexes[0]
                active['acquired'][expert] = data
                report['observed_acquire_jobs'] += 1
                if j in active['chosen']:
                    job = active['jobs'][j]
                    active['stages'][j] = dict(binding=source_binding(weights, stream, active['layer'], job, data))
                    for projection in ('gate', 'up', 'down'):
                        active['tags'][id(data[projection+'.qweight'])] = (j, projection, data)
            return data
        def qmm(x, weight, *args, **kwargs):
            result = before_qmm(x, weight, *args, **kwargs)
            tagged = active['tags'].get(id(weight)) if active is not None else None
            if tagged is not None:
                j, projection, data = tagged
                job, stage = active['jobs'][j], active['stages'][j]
                if (args or kwargs.get('mode') != 'mxfp4' or kwargs.get('transpose', True) is not True
                        or kwargs.get('scales') is not data[projection+'.scales'] or projection+'_raw' in stage
                        or x.dtype != mx.float32 or x.shape != (job.count, 2304 if projection == 'down' else 5120)):
                    raise ValueError('Native FFN actual projection operand/signature')
                if projection == 'down':
                    if x is not stage.get('middle'):
                        raise ValueError('Native FFN DOWN did not consume actual same-frame middle')
                elif 'gu_input' in stage and x is not stage['gu_input']:
                    raise ValueError('Native FFN gate/up did not share admitted input object')
                else:
                    stage['gu_input'] = x
                stage[projection+'_raw'] = result
                active['raw_tags'][id(result)] = (j, projection)
            return result
        def middle(gate, up, scores, slots):
            result = before_middle(gate, up, scores, slots)
            tagged = active['raw_tags'].get(id(gate)) if active is not None else None
            if tagged is not None:
                j, projection = tagged
                if projection != 'gate' or active['raw_tags'].get(id(up)) != (j, 'up'):
                    raise ValueError('Native FFN mismatched GU middle producer')
                stage = active['stages'][j]
                if 'middle' in stage:
                    raise ValueError('Native FFN duplicate selected middle')
                stage.update(middle=result, middle_scores=scores, middle_slots=slots)
            return result
        def combine(parts, values, order):
            result = before_combine(parts, values, order)
            if active is None or not active['chosen']:
                return result
            if (values is not active['values'] or order is not active['order'] or len(parts) != len(active['jobs'])):
                raise ValueError('Native FFN assembly is not current provider ordered parts')
            packed = mx.concatenate(parts, axis=0)
            report['capture_activation_concat_bytes'] += values.size*5120*4
            offsets = source_math.ordered_offsets(values, order)
            for j in sorted(active['chosen']):
                job, stage = active['jobs'][j], active['stages'][j]
                if any(key not in stage for key in ('gu_input', 'gate_raw', 'up_raw', 'middle', 'down_raw')):
                    raise ValueError('Native FFN selected stage incomplete before assembly')
                data = active['acquired'][job.expert_id]
                source_binding(weights, stream, active['layer'], job, data)
                owned = lambda value: value.astype(mx.bfloat16).astype(mx.float32)
                g, u = owned(stage['gate_raw']), owned(stage['up_raw'])
                clipped_g, clipped_u = mx.minimum(g, 10.), mx.clip(u, -10., 10.)
                selected_scores = stage['middle_scores'][stage['middle_slots']]
                weighted = (nn.silu(clipped_g)*clipped_u)*selected_scores[:, None]
                weighted_owned = owned(weighted)
                replay_middle = quantize(weighted_owned, 'fp8')
                report['extra_stock_middle_replay_graphs'] += 1
                contributions = packed[mx.array(offsets[job.rows], mx.uint32)]
                selected_output = result[mx.array(job.rows, mx.uint32)]
                tensors = dict(shared_gu_input=active['x'], gu_input=stage['gu_input'], gate_raw=stage['gate_raw'], up_raw=stage['up_raw'],
                    gate_owned=g, up_owned=u, weighted_raw=weighted, weighted_owned=weighted_owned,
                    middle=stage['middle'], middle_reference=replay_middle, down_raw=stage['down_raw'],
                    down_owned=parts[j], assembly_parts=contributions, assembly_output=selected_output,
                    scores=active['scores'], route_scores=selected_scores)
                mx.eval(*tensors.values(), stage['middle_slots'])
                arrays = {key: np.array(value) for key, value in tensors.items()}
                slots = np.array(stage['middle_slots'])
                if not np.array_equal(slots, job.locations):
                    raise ValueError('Native FFN middle slots differ from actual job')
                arrays.update(values=np.array(values, dtype=np.uint32), order=np.array(order, dtype=np.uint32),
                    route_rows=np.array(job.rows, dtype=np.uint32), route_locations=np.array(slots, dtype=np.uint32),
                    assembly_rank=np.sum(values[job.rows] < job.expert_id, axis=1).astype(np.uint32))
                case = dict(layer=active['layer'], expert=job.expert_id, kind='original', rows=job.count,
                    part_index=j, source_binding=stage['binding'], same_forward=True,
                    expert_order=[job.expert_id for job in active['jobs']],
                    expert_row_counts=[job.count for job in active['jobs']],
                    numerical_contract='GU BF16 each; gate upper10/up±10; F32 compiled SiLU×up×actualscore; BF16; FP8group32; original DOWN; BF16; ascending expert F32 +0 assembly')
                size = validate_arrays(case, arrays)
                if report['captured_array_bytes']+size > MAX_ARRAY_BYTES:
                    raise MemoryError('Native FFN aggregate32MiB ceiling')
                name = f'native-ffn-{label}-{index:04d}-l{active["layer"]}-e{job.expert_id}.npz'
                record = save_case(unit, name, case, arrays)
                report['cases'].append(record)
                report['captured_array_bytes'] += size
                stage['saved'] = True
            return result
        patch(weights, '_prefill_batched_gate_up', provider)
        patch(stream, 'acquire', acquire)
        patch(mx, 'quantized_matmul', qmm)
        patch(source_middle, 'middle', middle)
        patch(source_math, 'ordered_combine', combine)
        yield report
        if layers != set(range(40)) or not report['cases']:
            raise ValueError('Native FFN incomplete40-layer scan or no eligible actual expert')
        report.update(status='completed', model_forward_completed=True)
    except BaseException as error:
        if report is not None:
            report.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        try:
            for obj, name, local, original in reversed(patches):
                if local: setattr(obj, name, original)
                else: delattr(obj, name)
            if report is not None and output is not None:
                report.update(wall_seconds=time.perf_counter()-started, observed_layers=sorted(layers))
                atomic(output, report)
        finally:
            _LOCK.release()
