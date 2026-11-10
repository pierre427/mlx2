# SPDX-License-Identifier: MIT
"""Actual DOWN replay and owner-store probes; caller owns GPU and admission.

Only caller-selected actual expert objects load, one at a time. No all-dense
provider, bank repack, serving integration or precision selection is present.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import hashlib
import importlib.util
import json
from pathlib import Path

LIMIT = 6 << 30
_UPLOAD_QUARANTINE = []


def options(report):
    value = report.get('down_actual_options', {})
    if not isinstance(value, dict) or set(value)-{'repeats', 'include_original_e1'}:
        raise ValueError('Unknown actual DOWN component options')
    repeats = value.get('repeats', 3)
    original = value.get('include_original_e1', True)
    if type(repeats) is not int or not 3 <= repeats <= 7 or type(original) is not bool:
        raise ValueError('Invalid actual DOWN repeats/original control option')
    return repeats, original


def capture_module(capture_dir):
    path = Path(capture_dir)
    if path.is_symlink():
        raise ValueError('DOWN capture path symlink')
    if path.is_dir():
        manifests = sorted(path.glob('down-actual-capture-*.json'))
        if len(manifests) != 1:
            raise ValueError('DOWN capture directory requires one manifest')
        path = manifests[0]
    unit = path.parent
    if path.is_symlink() or (unit/'spec.json').is_symlink():
        raise ValueError('DOWN capture manifest/spec symlink')
    manifest = json.loads(path.read_text())
    spec = json.loads((unit/'spec.json').read_text())
    parser = unit/'harness/ds41_down_actual_capture.py'
    seal = spec.get('harness_sha256', {}).get(parser.name)
    if (parser.is_symlink() or not seal or manifest.get('helper_sha256') != seal
            or hashlib.sha256(parser.read_bytes()).hexdigest() != seal):
        raise ValueError('DOWN capture parser seal mismatch')
    for key in ('checkpoint', 'source_revision', 'source_snapshot_sha256'):
        if not manifest.get(key) or manifest[key] != spec.get(key):
            raise ValueError('DOWN capture source identity mismatch')
    loader = importlib.util.spec_from_file_location('_ds41_sealed_actual_down', parser)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module, path, spec


def check_coefficients(np, case, data):
    fields = ('qweight', 'scales', 'biases') if case['kind'] == 'q1' else ('qweight', 'scales')
    if ('down.biases' in data) != (case['kind'] == 'q1'):
        raise ValueError('DOWN replay precision kind mismatch')
    for field in fields:
        value = np.array(data['down.'+field])
        record = dict(shape=list(value.shape), dtype=str(value.dtype),
                      sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        if record != case['coefficients'][field]:
            raise ValueError('Actual DOWN coefficient hash/geometry mismatch')


def _materialize(mx, host):
    from mlx2.adapters.deepseek_v41_source_buffers import materialize
    packed = None
    try:
        packed = materialize(host)
        mx.eval(packed.backing, *packed.values())
        return packed
    except BaseException:
        try:
            mx.synchronize()
        except BaseException:
            _UPLOAD_QUARANTINE.append((host, packed))
        raise


def load_expert_factory(mx, artifact, manifest, spec):
    """Return bounded loader, retaining only Q1 metadata, never a GPU bank."""
    from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed, borrow_q1_packed
    from mlx2.adapters.deepseek_v41_source_lfu import load_q1_manifest, q1_read
    if artifact['identity']['fingerprint'] != manifest['original_artifact_fingerprint']:
        raise ValueError('DOWN original artifact fingerprint changed')
    policy = spec['expert_policy']
    hot_file = Path(policy['hotset_path'])
    if hot_file.is_symlink() or hashlib.sha256(hot_file.read_bytes()).hexdigest() != manifest['hot_map_binding']:
        raise ValueError('DOWN hot precision map changed')
    hot_map = json.loads(hot_file.read_text())
    hot = frozenset(map(tuple, hot_map['experts']))
    source = artifact['identity']['fingerprint']
    root, records, fingerprint = load_q1_manifest(policy['q1_root'], source)
    if fingerprint != manifest['q1_bank_binding']:
        raise ValueError('DOWN Q1 derivative fingerprint changed')

    def load(case):
        key = case['layer'], case['expert']
        if (key in hot) != (case['kind'] == 'original'):
            raise ValueError('DOWN captured precision differs from frozen map')
        host = (read_original_packed(artifact, *key) if case['kind'] == 'original'
                else borrow_q1_packed(q1_read(root, records[key])))
        # _materialize fences before the host owner leaves this frame.
        return _materialize(mx, host)
    return load


def run_components(mx, np, captureDir, out, report, compare, atomic):
    """Replay raw first; exact variants time complete DOWN+BF16 owner stage."""
    if mx.default_device() != mx.gpu:
        raise RuntimeError('Actual DOWN components require caller-owned GPU')
    if 'down_actual_components' in report:
        raise ValueError('Refusing to overwrite actual DOWN evidence')
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    from mlx2.runtime.gguf import source_q1, source_q1_down_owner, source_down_prefill
    from ds41_prefill_wide_components import raw_details
    from ds41_attention_components import timed_interleaved
    repeats, original_e1 = options(report)
    parser, path, spec = capture_module(captureDir)
    manifest, cases = parser.load_capture(path)
    for key in ('checkpoint', 'source_revision'):
        if report.get(key) != manifest[key]:
            raise ValueError('DOWN component/capture identity mismatch')
    artifact = inspect_artifact(spec['model'])
    load_expert = load_expert_factory(mx, artifact, manifest, spec)
    section = report['down_actual_components'] = dict(
        schema='mlx2.source-down-actual-components.v1', checkpoint=manifest['checkpoint'],
        source_revision=manifest['source_revision'], capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        capture_path=str(path), capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_artifact_fingerprint=manifest['original_artifact_fingerprint'],
        q1_bank_binding=manifest['q1_bank_binding'], hot_map_binding=manifest['hot_map_binding'],
        expert_budget_bytes=75_000_000_000, original_hot_bytes=60_000_000_000,
        scope='Actual PP99 DOWN activations/coefficient hashes; selected actual row strata; no serving candidate or model-quality claim',
        candidate_used_by_model=False, qualified=False, performance_qualified=False,
        maximum_allocator_bytes=LIMIT, loaded_experts_concurrently=1,
        timing_scope='Resident actual DOWN plus owning BF16→F32 boundary; rotated interleaved samples, graphbuild/submission/completion separately; not model critical path',
        repetitions=repeats, original_e1_scope='Existing E1 pp_down control, not selection of the historical full batch route',
        cases=[], adversarials=[], complete=False, harness_completed=False,
        required_reference_controls_exact=False, verdicts={})
    out = Path(out)

    def write():
        atomic(out/'results.json', report)

    def observe():
        peak = int(mx.get_peak_memory())
        section['peak_mlx_bytes'] = max(section.get('peak_mlx_bytes', 0), peak)
        report['component_peak_mlx_bytes'] = max(report.get('component_peak_mlx_bytes', 0), peak)
        if peak > LIMIT:
            write()
            raise MemoryError('Actual DOWN exceeded6GiB allocator ceiling')

    rounded = lambda x: x.astype(mx.bfloat16).astype(mx.float32)
    adversarials_done = False
    write()
    for case, arrays in cases:
        data = load_expert(case)
        check_coefficients(np, case, data)
        x = mx.array(arrays['activation'], mx.float32)
        mx.eval(x)
        q, s = data['down.qweight'], data['down.scales']
        b = data.get('down.biases')
        stock = (lambda: source_q1.matmul(x, q, s, b)) if case['kind'] == 'q1' else (
            lambda: mx.quantized_matmul(x, q, scales=s, mode='mxfp4'))
        raw = stock()
        owner = rounded(raw)
        mx.eval(raw, owner)
        row = dict(capture=case, reference_controls={
            'raw': raw_details(np, arrays['reference_raw'], np.array(raw)),
            'owner': raw_details(np, arrays['reference_boundary'], np.array(owner))}, variants={})
        section['cases'].append(row)
        observe()
        write()
        if not all(v['exact'] for v in row['reference_controls'].values()):
            raise ValueError('Actual DOWN stock replay differs from capture; candidate timing refused')
        functions = {'reference': lambda: rounded(stock())}
        if case['kind'] == 'q1':
            candidate = lambda: source_q1_down_owner.project(x, q, s, b)
            projected, preround = source_q1_down_owner.project(x, q, s, b, return_raw=True)
            normal = candidate()
            mx.eval(projected, preround, normal)
            checks = dict(raw=raw_details(np, arrays['reference_raw'], np.array(preround)),
                          diagnostic_owner=raw_details(np, arrays['reference_boundary'], np.array(projected)),
                          serving_shape_owner=raw_details(np, arrays['reference_boundary'], np.array(normal)))
            row['variants']['q1_owner_store'] = dict(checks=checks, exact=all(v['exact'] for v in checks.values()),
                changed_dot_program=False, explicit_raw_intermediate_output=False,
                logical_owner_intermediate_bytes_omitted=case['rows']*5120*4,
                traffic_scope='Logical graph allocation/store omission only; not measured physical DRAM traffic')
            if row['variants']['q1_owner_store']['exact']:
                functions['q1_owner_store'] = candidate
            if not adversarials_done:
                synthetic = [('signed_zero', np.full((1, 2304), -0., np.float32)),
                             ('cancellation', np.tile(np.array([32., -32., 1., -1.], np.float32), (3, 576))),
                             ('small_values', np.tile(np.array([2**-12, -2**-12, 2**-8, -2**-8], np.float32), (2, 576)))]
                for name, cpu in synthetic:
                    a = mx.array(cpu, mx.float32)
                    reference = source_q1.matmul(a, q, s, b)
                    result, candidate_raw = source_q1_down_owner.project(a, q, s, b, return_raw=True)
                    direct = source_q1_down_owner.project(a, q, s, b)
                    expected = rounded(reference)
                    mx.eval(reference, expected, result, candidate_raw, direct)
                    checks = dict(raw=raw_details(np, np.array(reference), np.array(candidate_raw)),
                                  owner=raw_details(np, np.array(expected), np.array(result)),
                                  normal=raw_details(np, np.array(expected), np.array(direct)))
                    section['adversarials'].append(dict(name=name, scope='Synthetic activation, same actual Q1 coefficients; not model trajectory',
                                                       checks=checks, exact=all(v['exact'] for v in checks.values())))
                    del a, reference, result, candidate_raw, direct, expected
                adversarials_done = True
            if any(not value['exact'] for value in section['adversarials']):
                functions.pop('q1_owner_store', None)
                row['variants']['q1_owner_store']['timing_refused'] = 'An independent actual-coefficient adversarial gate failed'
            del projected, preround, normal
        elif original_e1 and case['rows'] <= 12:
            candidate = lambda: source_down_prefill.pp_down(x, [data], (case['rows'],))
            projected, preround = source_down_prefill.pp_down(x, [data], (case['rows'],), return_raw=True)
            normal = candidate()
            mx.eval(projected, preround, normal)
            checks = dict(raw=raw_details(np, arrays['reference_raw'], np.array(preround)),
                          diagnostic_owner=raw_details(np, arrays['reference_boundary'], np.array(projected)),
                          normal_owner=raw_details(np, arrays['reference_boundary'], np.array(normal)))
            row['variants']['original_e1_control'] = dict(checks=checks, exact=all(v['exact'] for v in checks.values()),
                scope='Existing one-expert down worklist control only; no new batch math or full-model selection')
            if row['variants']['original_e1_control']['exact']:
                functions['original_e1_control'] = candidate
            del projected, preround, normal
        row['timing'] = timed_interleaved(mx, np, functions, repeats=repeats)
        observe()
        write()
        # Caller owns this bounded window; all work must finish before the
        # immutable actual backing and input references are released.
        mx.synchronize()
        data=None;q=None;s=None;b=None;x=None;stock=None;del raw,owner,functions
        if 'candidate' in locals():
            del candidate
        mx.clear_cache()
    section['required_reference_controls_exact'] = all(v['exact'] for c in section['cases'] for v in c['reference_controls'].values())
    section['verdicts'] = dict(
        actual_reference_replay=section['required_reference_controls_exact'],
        all_candidate_raw_and_owner_exact=all(v['exact'] for c in section['cases'] for v in c['variants'].values()),
        q1_adversarials_exact=bool(section['adversarials']) and all(v['exact'] for v in section['adversarials']))
    section.update(complete=True, harness_completed=True)
    write()
    return section
