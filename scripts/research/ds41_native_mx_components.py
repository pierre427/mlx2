# SPDX-License-Identifier: MIT
"""Actual original expert native-MX components; caller owns GPU and admission.

No GPU import or work at module import. Native reductions are an explicit
component profile; exact stock replay is a control, not its numerical law.
Provenance: provenance/deepseek-v41-native-mx-owned-probe.json.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import hashlib
import json
from pathlib import Path
import time

LIMIT = 6 << 30
ACTUAL_ROWS = (34, 1, 3, 5, 9, 32, 98)


def options(report):
    value = report.get('native_mx_options')
    expected = {'repeats': 7, 'projection': 'down', 'actual_rows': list(ACTUAL_ROWS)}
    if (type(value) is not dict or value != expected
            or any(type(value.get(k)) is not type(v) for k, v in expected.items())
            or any(type(row) is not int for row in value['actual_rows'])):
        raise ValueError('Invalid native MX bounded options')
    return value


def select_cases(cases, rows=ACTUAL_ROWS):
    """All requested shapes are separate actual captures, never tiled repeats."""
    selected = {}
    for case, arrays in cases:
        if case.get('kind') != 'original' or case.get('rows') not in rows:
            continue
        count = case['rows']
        if type(count) is not int or count in selected:
            raise ValueError('Ambiguous actual native MX row stratum')
        selected[count] = (case, arrays)
    if set(selected) != set(rows):
        raise ValueError('Missing actual original native MX row strata')
    return [selected[count] for count in rows]


def half_roundtrip(np, x):
    if x.dtype != np.float32 or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Native MX activation must be finite F32 matrix')
    with np.errstate(over='ignore', invalid='ignore'):
        half = x.astype(np.float16)
        restored = half.astype(np.float32)
    return dict(finite=bool(np.isfinite(half).all()),
        bitwise_exact=bool(x.tobytes() == restored.tobytes()),
        input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        restored_sha256=hashlib.sha256(restored.tobytes()).hexdigest(),
        half_bytes=int(half.nbytes), scope='Actual activation value/range gate; not native reduction parity')


def coefficient_gate(np, q, scales):
    """Inspect every actual coefficient in <=16-row CPU tiles, not a bank copy."""
    from ds41_native_mx_operands import decode_rows
    ff = int((scales == 255).sum())
    if ff:
        return dict(eligible=False, scale_ff_bytes=ff,
            reason='Installed F32 ff=+inf and native UE8M0 ff=NaN differ; native eligibility refused')
    changed, nonfinite, maximum_tile_bytes = 0, 0, 0
    for start in range(0, q.shape[0], 16):
        value = decode_rows(np, q, scales, tuple(range(start, min(start+16, q.shape[0]))))
        maximum_tile_bytes = max(maximum_tile_bytes, int(value.nbytes))
        with np.errstate(over='ignore', invalid='ignore'):
            restored = value.astype(np.float16).astype(np.float32)
        nonfinite += int(np.count_nonzero(~np.isfinite(value) | ~np.isfinite(restored)))
        changed += int(np.count_nonzero(value.view(np.uint32) != restored.view(np.uint32)))
    return dict(eligible=changed == 0 and nonfinite == 0, scale_ff_bytes=0,
        changed_half_roundtrip_words=changed, nonfinite_words=nonfinite,
        inspected_coefficients=int(q.shape[0]*q.shape[1]*8), maximum_decoded_cpu_tile_bytes=maximum_tile_bytes,
        scope='Conservative actual coefficient representation gate; no F32 reduction parity claim')


def verdicts(cases, required_rows=34):
    ran = [case for case in cases if case.get('native_dispatched') is True]
    return dict(actual_reference_replay=bool(cases) and all(case['reference_exact'] for case in cases),
        coefficient_identity=bool(cases) and all(case['coefficient_identity'] and case['coefficient_gate']['eligible'] for case in cases),
        half_roundtrip=bool(cases) and all(case['half_roundtrip']['finite'] and case['half_roundtrip']['bitwise_exact'] for case in cases),
        native_finite=bool(ran) and all(case['native_finite'] for case in ran),
        native_stable=bool(ran) and all(case['native_stable'] and case['canary_passed'] for case in ran),
        actual_fp64_bound=bool(ran) and all(case['fp64_oracle']['reference']['within_bound'] and case['fp64_oracle']['native']['within_bound'] for case in ran),
        native_dispatched=any(case['capture']['rows'] == required_rows for case in ran))


def dot_bounds(np, x, q, scales, *, input_rows, output_rows):
    """Conservative F32 dot forward bound for selected actual operand windows."""
    from ds41_native_mx_operands import decode_rows
    weights = decode_rows(np, q, scales, output_rows).astype(np.float64)
    inputs = x[list(input_rows)].astype(np.float64)
    sum_abs = np.stack([np.sum(np.abs(inputs*w), axis=1) for w in weights], axis=1)
    eps = float(np.finfo(np.float32).eps)  # 2^-23, more conservative than unit-roundoff.
    k = x.shape[1]
    if k*eps >= 1:
        raise ValueError('Native MX F32 arithmetic bound is undefined')
    gamma = (k*eps)/(1-k*eps)
    floor = float(np.finfo(np.float32).tiny)
    return gamma*sum_abs+floor, dict(k=k, eps=eps, gamma_k=gamma, absolute_floor=floor,
        formula='gamma_K*sum_abs_products + F32tiny; gamma_K=(K*eps)/(1-K*eps)',
        precision_scope='Lossless half inputs and coefficients required; conservative F32 dot bound on selected actual windows, not raw parity or model quality')


def _error(np, reference64, candidate, bound):
    delta = candidate.astype(np.float64)-reference64
    return dict(max_abs=float(np.max(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta*delta))),
                within_bound=bool(np.isfinite(delta).all() and np.all(np.abs(delta) <= bound)),
                maximum_error_over_bound=float(np.max(np.abs(delta)/bound)),
                scope='Bounded FP64 mathematical dot oracle; not stock reduction or model-quality oracle')


def run_components(mx, np, capture_dir, out, report, compare, atomic):
    """Only caller-selected original expert backings, one resident at a time."""
    if mx.default_device() != mx.gpu:
        raise RuntimeError('Native MX component requires caller-owned GPU')
    if 'native_mx_components' in report:
        raise ValueError('Refusing native MX evidence overwrite')
    cfg = options(report)
    from ds41_down_actual_components import capture_module, check_coefficients, _materialize
    from ds41_native_mx_operands import inspect_case, oracle
    from ds41_prefill_wide_components import raw_details
    from ds41_attention_components import timed_interleaved
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed
    from mlx2.runtime.gguf import source_native_mx
    parser, path, capture_spec = capture_module(capture_dir)
    manifest, captured = parser.load_capture(path)
    for key in ('checkpoint', 'source_revision'):
        if report.get(key) != manifest[key]:
            raise ValueError('Native MX capture/component identity mismatch')
    cases = select_cases(captured)
    artifact = inspect_artifact(capture_spec['model'])
    if artifact['identity']['fingerprint'] != manifest['original_artifact_fingerprint']:
        raise ValueError('Native MX original artifact fingerprint changed')
    hot = Path(capture_spec['expert_policy']['hotset_path'])
    if hot.is_symlink() or hashlib.sha256(hot.read_bytes()).hexdigest() != manifest['hot_map_binding']:
        raise ValueError('Native MX frozen precision map changed')
    hot_keys = frozenset(map(tuple, json.loads(hot.read_text())['experts']))
    section = report['native_mx_components'] = dict(schema='mlx2.ds41-native-mx-components.v1',
        checkpoint=manifest['checkpoint'], source_revision=manifest['source_revision'],
        component_source_snapshot_sha256=report['source_snapshot_sha256'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        capture_path=str(path), capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_artifact_fingerprint=manifest['original_artifact_fingerprint'],
        q1_bank_binding=manifest['q1_bank_binding'], hot_map_binding=manifest['hot_map_binding'],
        numerical_profile='Native half activation × original MXFP4, F32 output; changed reduction allowed and exposed',
        scope='Actual original expert DOWN only; no fabricated gate inputs, no serving/quality/rate claim',
        native_used_by_model=False, qualification=False, performance_qualified=False,
        maximum_allocator_bytes=LIMIT, loaded_experts_concurrently=1, extra_expert_acquisitions=0,
        repetitions=cfg['repeats'], cases=[], complete=False, harness_completed=False,
        required_reference_controls_exact=False, verdicts={})
    out = Path(out)
    rounded = lambda x: x.astype(mx.bfloat16).astype(mx.float32)

    def write():
        peak = int(mx.get_peak_memory())
        section['peak_mlx_bytes'] = max(section.get('peak_mlx_bytes', 0), peak)
        report['component_peak_mlx_bytes'] = max(report.get('component_peak_mlx_bytes', 0), peak)
        atomic(out/'results.json', report)
        if peak > LIMIT:
            raise MemoryError('Native MX component exceeded6GiB allocator ceiling')

    write()
    for case, arrays in cases:
        if (case['layer'], case['expert']) not in hot_keys:
            raise ValueError('Native MX original expert outside frozen hot precision map')
        start = time.perf_counter()
        host = read_original_packed(artifact, case['layer'], case['expert'])
        views = host.views()
        check_coefficients(np, case, views)
        cpu_receipt = inspect_case(np, case, arrays, views, projection='down')
        coefficients = coefficient_gate(np, views['down.qweight'], views['down.scales'])
        half = half_roundtrip(np, arrays['activation'])
        row = dict(capture=case, coefficient_identity=True, cpu_operands=cpu_receipt,
            half_roundtrip=half, coefficient_gate=coefficients, native_dispatched=False, reference_exact=False,
            timing_scope='Synchronized component interventions, not model critical path',
            coefficient_host_read_seconds=time.perf_counter()-start)
        section['cases'].append(row)
        if not coefficients['eligible']:
            write()
            raise ValueError('Actual native MX coefficient eligibility gate failed')
        if not half['finite'] or not half['bitwise_exact']:
            write()
            raise ValueError('Actual native MX F16 activation roundtrip gate failed')
        start = time.perf_counter()
        expert = _materialize(mx, host)
        x = mx.array(arrays['activation'], mx.float32)
        mx.eval(x, expert.backing)
        row['resident_upload_seconds'] = time.perf_counter()-start
        row['staged_host_bytes'] = int(host.host_allocation_nbytes)
        q, s = expert['down.qweight'], expert['down.scales']
        stock = lambda: mx.quantized_matmul(x, q, scales=s, mode='mxfp4')
        raw = stock()
        owner = rounded(raw)
        mx.eval(raw, owner)
        reference_cpu, owner_cpu = np.array(raw), np.array(owner)
        row['reference_controls'] = dict(raw=raw_details(np, arrays['reference_raw'], reference_cpu),
            owner=raw_details(np, arrays['reference_boundary'], owner_cpu))
        row['reference_exact'] = all(check['exact'] for check in row['reference_controls'].values())
        if not row['reference_exact']:
            write()
            raise ValueError('Actual native MX stock replay differs from sealed capture')
        try:
            row['plan'] = source_native_mx.plan('down', case['rows'])
        except ValueError as error:
            row['unsupported_reason'] = str(error)
            write()
            mx.synchronize()
            expert=None;x=None;q=None;s=None;stock=None;del raw,owner,host,views
            mx.clear_cache()
            continue
        start = time.perf_counter()
        prepared = source_native_mx.prepare_input(x)
        row['prepare_input_seconds'] = time.perf_counter()-start
        row['preparation_receipt'] = prepared.receipt
        native = source_native_mx.project(prepared, expert, projection='down')
        native_owner = rounded(native)
        repeated = source_native_mx.project(prepared, expert, projection='down')
        mx.eval(native, native_owner, repeated)
        native_cpu, native_owner_cpu = np.array(native), np.array(native_owner)
        row.update(native_dispatched=True, native_finite=bool(np.isfinite(native_cpu).all() and np.isfinite(native_owner_cpu).all()),
            native_stable=raw_details(np, native_cpu, np.array(repeated))['exact'],
            native_raw=raw_details(np, reference_cpu, native_cpu),
            native_owner=raw_details(np, owner_cpu, native_owner_cpu))
        zero_prepared = source_native_mx.prepare_input(mx.zeros_like(x))
        zero = source_native_mx.project(zero_prepared, expert, projection='down')
        again = source_native_mx.project(prepared, expert, projection='down')
        mx.eval(zero, again)
        row['canary_passed'] = bool(np.all(np.array(zero) == 0) and np.any(native_cpu != 0)
            and raw_details(np, native_cpu, np.array(again))['exact'])
        input_rows = tuple(sorted({0, case['rows']//2, case['rows']-1}))
        output_rows = (0, 1, 31, 32, 5119)
        target64 = oracle(np, arrays['activation'], views['down.qweight'], views['down.scales'],
                          input_rows=input_rows, output_rows=output_rows)
        bounds, bound_metadata = dot_bounds(np, arrays['activation'], views['down.qweight'], views['down.scales'],
                                           input_rows=input_rows, output_rows=output_rows)
        ix = np.ix_(input_rows, output_rows)
        row['fp64_oracle'] = dict(input_rows=list(input_rows), output_rows=list(output_rows),
            bounds=bounds.tolist(), bound_metadata=bound_metadata,
            reference=_error(np, target64, reference_cpu[ix], bounds), native=_error(np, target64, native_cpu[ix], bounds))
        if not row['native_finite'] or not row['native_stable'] or not row['canary_passed']:
            write()
            raise ValueError('Native MX finite/stability/consecutive-input canary failed')
        if not row['fp64_oracle']['reference']['within_bound'] or not row['fp64_oracle']['native']['within_bound']:
            write()
            raise ValueError('Actual native MX bounded FP64 arithmetic/geometry gate failed')
        # This explicit new profile is timed after engineering gates even when
        # stock raw/owner differences exist. The full preparation API includes
        # its own checks; that cost is deliberately visible, never hidden.
        def native_complete():
            return rounded(source_native_mx.project(source_native_mx.prepare_input(x), expert, projection='down'))
        functions = dict(reference=lambda: rounded(stock()),
            native_prepared=lambda: rounded(source_native_mx.project(prepared, expert, projection='down')),
            native_complete_preparation=native_complete)
        row['timing'] = timed_interleaved(mx, np, functions, repeats=cfg['repeats'])
        row['timing_preparation_scope'] = 'Complete preparation API including diagnostic roundtrip checks; prepared variant excludes that API cost'
        write()
        mx.synchronize()
        expert=None;x=None;q=None;s=None;stock=None;prepared=None;del raw,owner,host,views,native,native_owner,repeated
        del zero_prepared, zero, again, functions, native_complete
        mx.clear_cache()
    section['verdicts'] = verdicts(section['cases'])
    section['required_reference_controls_exact'] = section['verdicts']['actual_reference_replay']
    ran = [row for row in section['cases'] if row.get('native_dispatched')]
    section['all_native_raw_and_owner_exact'] = bool(ran) and all(row['native_raw']['exact'] and row['native_owner']['exact'] for row in ran)
    section.update(complete=True, harness_completed=True)
    write()
    return section
