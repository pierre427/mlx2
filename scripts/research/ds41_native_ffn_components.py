# SPDX-License-Identifier: MIT
"""Complete actual expert FFN replay; caller owns GPU, locks and admission.

No GPU imports/work on import. Native dot differences are a distinct numerical
profile; exact stock replay and bounded arithmetic controls precede timing.
Provenance: provenance/deepseek-v41-native-ffn-owned-probe.json.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

LIMIT = 6 << 30
PARSER = 'ds41_native_ffn_capture.py'
FRONTIERS = ('shared_gu_input', 'gu_input', 'gate_raw', 'up_raw', 'gate_owned', 'up_owned',
    'weighted_raw', 'weighted_owned', 'middle', 'middle_reference', 'down_raw', 'down_owned',
    'assembly_parts', 'assembly_output')
CONSUMERS = ('gate_owned', 'up_owned', 'weighted_owned', 'middle', 'down_owned', 'assembly_output')
VERDICTS = frozenset(('stock_capture_replay', 'coefficient_identity', 'native_eligibility',
    'finite_stable', 'actual_fp64_bounds', 'zero_aba', 'native_stage_dispatched'))


def options(report):
    value = report.get('native_ffn_options')
    expected = {'repeats': 7, 'tile': [16, 32], 'stage': 'complete_original_expert_v1'}
    if (type(value) is not dict or value != expected
            or type(value['repeats']) is not int or type(value['tile']) is not list
            or any(type(v) is not int for v in value['tile'])):
        raise ValueError('Invalid complete native FFN options')
    return value


def capture_module(capture_dir):
    path = Path(capture_dir)
    if path.is_symlink():
        raise ValueError('Native FFN capture symlink')
    if path.is_dir():
        manifests = sorted(path.glob('native-ffn-capture-*.json'))
        if len(manifests) != 1:
            raise ValueError('Native FFN requires one complete manifest')
        path = manifests[0]
    unit = path.parent
    from probe_ds41_native_ffn import checked_json
    spec, manifest = checked_json(unit/'spec.json'), checked_json(path)
    if spec.get('native_ffn_capture') is not True or spec.get('resident_discovery') is not True or spec.get('rate_eligible') is not False:
        raise ValueError('Native FFN capture is not sealed untimed discovery')
    parser = unit/'harness'/PARSER
    seal = spec.get('harness_sha256', {}).get(PARSER)
    if (parser.is_symlink() or not parser.is_file() or parser.stat().st_size > 128 << 10
            or not seal or manifest.get('helper_sha256') != seal
            or hashlib.sha256(parser.read_bytes()).hexdigest() != seal):
        raise ValueError('Native FFN sealed parser mismatch')
    for key in ('checkpoint', 'source_revision', 'source_snapshot_sha256'):
        if not spec.get(key) or manifest.get(key) != spec[key]:
            raise ValueError('Native FFN capture identity mismatch')
    loader = importlib.util.spec_from_file_location('_native_ffn_sealed_parser', parser)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module, path, spec


def source_coefficients(artifact, case, host, hot_keys):
    """CPU-only bind reader output to captured original source/precision/layout."""
    if (case['layer'], case['expert']) not in hot_keys:
        raise ValueError('Native FFN original expert outside frozen precision map')
    binding = case['source_binding']
    if binding['source_fingerprint'] != artifact['identity']['fingerprint']:
        raise ValueError('Native FFN original artifact changed')
    from mlx2.adapters.deepseek_v41_source_weights import expert_names
    files = sorted({artifact['tensors'][name]['file'] for name in expert_names(case['layer'], case['expert'])})
    seals = {name: artifact['seals'][name] for name in files}
    layout = [dict(name=t.name, offset=t.offset, dtype=t.dtype, shape=list(t.shape)) for t in host.layout.tensors]
    sha = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (files != binding['files'] or sha(seals) != binding['source_seals_sha256']
            or sha(layout) != binding['layout_sha256'] or host.layout.kind != 'original'):
        raise ValueError('Native FFN original file seals/canonical layout changed')
    return dict(source_seals_exact=True, layout_exact=True, original_hot_membership=True,
        host_payload_sha256=hashlib.sha256(memoryview(host.data)).hexdigest(),
        hash_scope='Fresh CPU checked reader original expert payload; captured binding uses source seals/layout, no captured weight dump')


def validate_frame_inventory(cases):
    """Cases from one layer must share their actual frame, never be spliced."""
    if not 1 <= len(cases) <= 2:
        raise ValueError('Native FFN complete case inventory')
    layers = {}
    for case, arrays in cases:
        previous = layers.get(case['layer'])
        if previous is not None:
            for key in ('shared_gu_input', 'values', 'scores', 'order'):
                a, b = previous[key], arrays[key]
                if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                    raise ValueError('Native FFN same-layer cases splice different live frames: '+key)
        layers[case['layer']] = arrays


def compare_frontiers(np, expected, actual):
    from ds41_prefill_wide_components import raw_details
    if any(key not in actual or key not in expected for key in FRONTIERS):
        raise ValueError('Native FFN complete frontier inventory')
    return {key: raw_details(np, expected[key], actual[key]) for key in FRONTIERS}


def assembly_fold(np, parts):
    if parts.dtype != np.float32 or parts.ndim != 3 or parts.shape[1] != 6:
        raise ValueError('Native FFN independent ordered assembly geometry')
    total = np.zeros((parts.shape[0], parts.shape[2]), np.float32)
    for slot in range(6):
        np.add(total, parts[:, slot], out=total)
    return total


def engineering_verdicts(cases):
    return dict(stock_capture_replay=bool(cases) and all(c['reference_exact'] for c in cases),
        coefficient_identity=bool(cases) and all(c['coefficient_identity'] for c in cases),
        native_eligibility=bool(cases) and all(c['admission']['eligible'] is True for c in cases),
        finite_stable=bool(cases) and all(c['native_finite'] and c['native_stable']
            and all(v['exact'] for v in c['normal_graph_checks'].values())
            and all(v['exact'] for v in c['assembly_fold_checks'].values()) for c in cases),
        actual_fp64_bounds=bool(cases) and all(c['all_fp64_bounds_passed'] for c in cases),
        zero_aba=bool(cases) and all(c['zero_aba_passed'] for c in cases),
        native_stage_dispatched=bool(cases) and all(c['native_stage_dispatched'] for c in cases))


def first_changed(checks, names):
    return next((dict(frontier=name, **checks[name]['first_raw_mismatch']) for name in names
        if not checks[name]['exact']), None)


def run_components(mx, np, capture_dir, out, report, compare, atomic):
    if mx.default_device() != mx.gpu:
        raise RuntimeError('Complete native FFN requires caller-owned GPU')
    if 'native_ffn_components' in report:
        raise ValueError('Refusing complete native FFN evidence overwrite')
    cfg = options(report)
    from ds41_down_actual_components import _materialize
    from ds41_native_mx_components import coefficient_gate, dot_bounds, _error
    from ds41_native_mx_operands import oracle
    from ds41_attention_components import timed_interleaved
    from ds41_prefill_wide_components import raw_details
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed
    from mlx2.adapters.deepseek_v41_source_lfu import load_q1_manifest
    from ds41_native_ffn_math import admit, reference_graph, native_graph
    parser, path, capture_spec = capture_module(capture_dir)
    manifest, cases = parser.load_capture(path)
    validate_frame_inventory(cases)
    if any(report.get(key) != manifest[key] for key in ('checkpoint', 'source_revision')):
        raise ValueError('Native FFN component/capture identity mismatch')
    artifact = inspect_artifact(capture_spec['model'])
    if artifact['identity']['fingerprint'] != manifest['original_artifact_fingerprint']:
        raise ValueError('Native FFN original artifact fingerprint changed')
    policy = capture_spec['expert_policy']
    hot = Path(policy['hotset_path'])
    if hot.is_symlink() or hashlib.sha256(hot.read_bytes()).hexdigest() != manifest['hot_map_binding']:
        raise ValueError('Native FFN frozen hot precision map changed')
    hot_keys = frozenset(map(tuple, json.loads(hot.read_text())['experts']))
    _, _, q1_binding = load_q1_manifest(policy['q1_root'], manifest['original_artifact_fingerprint'])
    if q1_binding != manifest['q1_bank_binding']:
        raise ValueError('Native FFN Q1 derivative identity changed')
    section = report['native_ffn_components'] = dict(schema='mlx2.native-ffn-components.v1',
        checkpoint=manifest['checkpoint'], source_revision=manifest['source_revision'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        component_source_snapshot_sha256=report['source_snapshot_sha256'],
        capture_path=str(path), capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_artifact_fingerprint=manifest['original_artifact_fingerprint'], q1_bank_binding=q1_binding,
        hot_map_binding=manifest['hot_map_binding'], scope='Same-frame actual original expert complete-stage replay; no model quality/throughput claim',
        stage='shared prepared input gather→GU→Source scored/QAT middle→DOWN→owner→owned selected-row fold',
        shared_layer_qat_included=False, cpu_route_sort_included=False, coefficient_load_in_timing=False,
        global_all_expert_output_packing_included=False, other_expert_parts_prepacked_outside_timing=True,
        complete_moe_layer_cost_claimed=False,
        native_used_by_model=False, qualification=False, performance_qualified=False,
        maximum_allocator_bytes=LIMIT, loaded_experts_concurrently=1, repetitions=cfg['repeats'],
        cases=[], complete=False, harness_completed=False, required_reference_controls_exact=False, verdicts={})
    out = Path(out)
    def write():
        peak = int(mx.get_peak_memory())
        section['peak_mlx_bytes'] = max(section.get('peak_mlx_bytes', 0), peak)
        report['component_peak_mlx_bytes'] = max(report.get('component_peak_mlx_bytes', 0), peak)
        atomic(out/'results.json', report)
        if peak > LIMIT:
            raise MemoryError('Complete native FFN allocator ceiling exceeded')
    def evaluate(graph):
        mx.eval(*graph.values())
        return {name: np.array(value) for name, value in graph.items()}
    write()
    for case, arrays in cases:
        start = time.perf_counter()
        host = read_original_packed(artifact, case['layer'], case['expert'])
        source = source_coefficients(artifact, case, host, hot_keys)
        views = host.views()
        coeff = {name: coefficient_gate(np, views[name+'.qweight'], views[name+'.scales']) for name in ('gate', 'up', 'down')}
        row = dict(capture=case, source_binding=source, coefficient_gates=coeff,
            coefficient_identity=all(g['eligible'] for g in coeff.values()), reference_exact=False,
            coefficient_read_and_inspection_seconds=time.perf_counter()-start,
            staged_host_bytes=int(host.host_allocation_nbytes), native_stage_dispatched=False)
        section['cases'].append(row)
        if not row['coefficient_identity']:
            write()
            raise ValueError('Complete native FFN actual coefficient eligibility failed')
        expert = _materialize(mx, host)
        start = time.perf_counter()
        admission = admit(mx, np, {'case': case, 'arrays': arrays}, expert, tile=tuple(cfg['tile']))
        row['initial_admission_compile_eval_seconds'] = time.perf_counter()-start
        row['initial_admission_scope'] = 'One-time preparation, actual backing/typed-view CPU inspection and native input checks, including initial compilation/evaluation; not pure kernel compile latency'
        row['admission'] = admission.receipt
        start = time.perf_counter()
        reference = evaluate(reference_graph(admission, frontiers=True))
        row['first_reference_graph_build_eval_seconds'] = time.perf_counter()-start
        row['reference_checks'] = compare_frontiers(np, arrays, reference)
        row['reference_exact'] = all(c['exact'] for c in row['reference_checks'].values())
        write()
        if not row['reference_exact']:
            raise ValueError('Complete stock FFN frontiers differ from actual capture; timing refused')
        start = time.perf_counter()
        native = evaluate(native_graph(admission, frontiers=True))
        row['first_native_graph_build_eval_seconds'] = time.perf_counter()-start
        row['first_graph_scope'] = 'Diagnostic full-frontier graph; initial admission may already compile/evaluate shared native operators; steady fresh graphs separately timed'
        row['native_stage_dispatched'] = True
        row['native_checks'] = compare_frontiers(np, reference, native)
        row['first_changed_frontier'] = first_changed(row['native_checks'], FRONTIERS)
        row['first_changed_consumer_frontier'] = first_changed(row['native_checks'], CONSUMERS)
        repeated = evaluate(native_graph(admission, frontiers=True))
        stable = compare_frontiers(np, native, repeated)
        row['native_stability_checks'] = stable
        row['native_stable'] = all(c['exact'] for c in stable.values())
        row['native_finite'] = all(np.isfinite(value).all() for value in native.values())
        zero = evaluate(native_graph(admission, frontiers=True, canary=True))
        zero_reference = evaluate(reference_graph(admission, frontiers=True, canary=True))
        again = evaluate(native_graph(admission, frontiers=True))
        row['zero_aba'] = dict(zero_math=all(np.all(zero[name] == 0) for name in ('gu_input', 'gate_raw', 'up_raw', 'middle', 'down_raw', 'down_owned')),
            zero_assembly=compare_frontiers(np, zero_reference, zero)['assembly_output'],
            actual_after_zero=compare_frontiers(np, native, again))
        row['zero_aba_passed'] = bool(row['zero_aba']['zero_math'] and row['zero_aba']['zero_assembly']['exact']
            and all(c['exact'] for c in row['zero_aba']['actual_after_zero'].values()))
        row['assembly_fold_checks'] = {name: raw_details(np, assembly_fold(np, graph['assembly_parts']), graph['assembly_output'])
            for name, graph in (('reference', reference), ('native', native), ('zero_reference', zero_reference), ('zero_native', zero))}
        normal_checks, first_normal = {}, {}
        for name, graph, expected in (('reference', reference_graph, reference), ('native', native_graph, native)):
            start = time.perf_counter()
            value = graph(admission)
            mx.eval(value)
            first_normal[name] = time.perf_counter()-start
            normal_checks[name] = raw_details(np, expected['assembly_output'], np.array(value))
            del value
        normal_zero = native_graph(admission, canary=True)
        normal_again = native_graph(admission)
        mx.eval(normal_zero, normal_again)
        normal_checks['native_zero'] = raw_details(np, zero['assembly_output'], np.array(normal_zero))
        normal_checks['native_after_zero'] = raw_details(np, native['assembly_output'], np.array(normal_again))
        row['normal_graph_checks'] = normal_checks
        row['first_normal_graph_build_eval_seconds'] = first_normal
        row['normal_graph_scope'] = 'Normal single-output graph versus diagnostic owner/assembly, including normal zero/ABA; first normal replacement-kernel compilation may occur here, excluded from steady repetitions'
        del normal_zero, normal_again
        linear = {}
        input_rows = tuple(sorted({0, case['rows']//2, case['rows']-1}))
        for name in ('gate', 'up', 'down'):
            output_rows = (0, 1, 31, 32, 5119 if name == 'down' else 2303)
            q, scales = views[name+'.qweight'], views[name+'.scales']
            checks = {}
            for profile, actual in (('reference', reference), ('native', native)):
                operand = actual['middle' if name == 'down' else 'gu_input']
                truth = oracle(np, operand, q, scales, input_rows=input_rows, output_rows=output_rows)
                bounds, metadata = dot_bounds(np, operand, q, scales, input_rows=input_rows, output_rows=output_rows)
                checks[profile] = _error(np, truth, actual[name+'_raw'][np.ix_(input_rows, output_rows)], bounds)
                checks[profile].update(bound_metadata=metadata, bounds=bounds.tolist(),
                    input_scope='Each profile actual DOWN input; native middle may differ from stock' if name == 'down' else 'Same lossless actual gathered GU values')
            linear[name] = dict(input_rows=list(input_rows), output_rows=list(output_rows), checks=checks)
        row['fp64_linear_windows'] = linear
        row['all_fp64_bounds_passed'] = all(value['within_bound'] for entry in linear.values() for value in entry['checks'].values())
        write()
        if (admission.receipt.get('eligible') is not True or not row['native_finite'] or not row['native_stable']
                or not row['zero_aba_passed'] or not row['all_fp64_bounds_passed']
                or not all(c['exact'] for c in row['assembly_fold_checks'].values())
                or not all(c['exact'] for c in normal_checks.values())):
            raise ValueError('Complete native FFN eligibility/finite/stability/ABA/F64 gate failed')
        functions = {'reference_complete': lambda: reference_graph(admission),
                     'native_complete': lambda: native_graph(admission)}
        row['timing'] = timed_interleaved(mx, np, functions, repeats=cfg['repeats'])
        row['timing_scope'] = 'Fresh single-expert gather/casts/GU/existing middle/DOWN/owner/selected-row fold. Other five contributions are prepacked outside timing; layer-global QAT, CPU route planning and global all-expert output packing/concatenation are excluded. Source serving concatenation is not removed or measured here. No eval/item/readbacks inside graph builders; admission/residency excluded. Not complete MoE-layer cost or model rate.'
        write()
        mx.synchronize()
        admission=None;del host,views,expert,reference,native,repeated,zero,zero_reference,again,functions,q,scales
        mx.clear_cache()
    section['verdicts'] = engineering_verdicts(section['cases'])
    section['required_reference_controls_exact'] = section['verdicts']['stock_capture_replay']
    section['all_native_frontiers_exact'] = all(c['exact'] for row in section['cases'] for c in row['native_checks'].values())
    section.update(complete=True, harness_completed=True)
    write()
    return section
