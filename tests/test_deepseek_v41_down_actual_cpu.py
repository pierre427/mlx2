"""CPU-only actual DOWN evidence/ownership/source-law tests; no GPU work."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('_test_'+name, ROOT/'scripts/research'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load('ds41_down_actual_capture')
component = load('ds41_down_actual_components')


def coefficients(kind):
    if kind == 'original':
        return dict(qweight=np.zeros((5120, 288), np.uint32), scales=np.zeros((5120, 72), np.uint8))
    return dict(qweight=np.zeros((5120, 72), np.uint32), scales=np.zeros((5120, 36), np.float16), biases=np.zeros((5120, 36), np.float16))


@pytest.fixture
def evidence(tmp_path):
    cases = []
    for kind in ('original', 'q1'):
        x = np.zeros((1, 2304), np.float32)
        raw = np.zeros((1, 5120), np.float32)
        cases.append(capture.save_case(tmp_path, kind+'.npz', layer=1, expert=3, kind=kind,
            activation=x, raw=raw, boundary=raw, coefficients=coefficients(kind),
            route_rows=np.array([0], np.uint32), route_locations=np.array([0], np.uint32)))
    code = (ROOT/'scripts/research/ds41_down_actual_capture.py').read_bytes()
    manifest = dict(schema=capture.SCHEMA, status='completed', model_forward_completed=True,
        source_result_always_returned=True, candidate_used_by_model=False, input_rows=99, frontier=0,
        checkpoint='synthetic-down@fixture', source_revision='fixture-revision', source_snapshot_sha256='fixture-source',
        observed_layers=list(range(40)), stratum_call_counts={c['band']: 1 for c in cases}, observed_down_calls=2,
        missing_strata=sorted(set(capture.ALL_STRATA)-{c['band'] for c in cases}),
        cases=cases, captured_array_bytes=sum(c['captured_array_bytes'] for c in cases),
        expert_budget_bytes=75_000_000_000, original_hot_bytes=60_000_000_000,
        helper_sha256=hashlib.sha256(code).hexdigest(), original_artifact_fingerprint='fixture-artifact',
        q1_bank_binding='fixture-q1', hot_map_binding='fixture-hot')
    path = tmp_path/'down-actual-capture-0000.json'
    path.write_text(json.dumps(manifest))
    harness = tmp_path/'harness'
    harness.mkdir()
    (harness/'ds41_down_actual_capture.py').write_bytes(code)
    spec = {k: manifest[k] for k in ('checkpoint', 'source_revision', 'source_snapshot_sha256')}
    spec['harness_sha256'] = {'ds41_down_actual_capture.py': manifest['helper_sha256']}
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    return path, manifest


def test_cpu_roundtrip_and_sealed_component_parser(evidence):
    path, manifest = evidence
    result, cases = capture.load_capture(path.parent, checkpoint=manifest['checkpoint'], source_snapshot='fixture-source')
    assert len(cases) == 2 and result['observed_down_calls'] == 2
    parser, saved, spec = component.capture_module(path.parent)
    assert parser.load_capture(saved)[0] == result
    assert spec['checkpoint'] == manifest['checkpoint']
    for case, arrays in cases:
        data = {'down.'+k: v for k, v in coefficients(case['kind']).items()}
        component.check_coefficients(np, case, data)
        data['down.qweight'][0, 0] = 1
        with pytest.raises(ValueError, match='hash'):
            component.check_coefficients(np, case, data)


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(status='failed'), lambda d: d.update(frontier=1),
    lambda d: d.update(candidate_used_by_model=True), lambda d: d.update(source_result_always_returned=False),
    lambda d: d.update(model_forward_completed=False), lambda d: d.update(observed_layers=[0]),
    lambda d: d.update(expert_budget_bytes=74_000_000_000), lambda d: d.update(original_hot_bytes=59_000_000_000),
    lambda d: d.update(observed_down_calls=3), lambda d: d['cases'][0].update(path='../bad.npz'),
    lambda d: d['cases'][0].update(sha256='bad'), lambda d: d['cases'][0].update(layer=40),
    lambda d: d['cases'][0].update(rows=100), lambda d: d['cases'][0].update(band='other'),
    lambda d: d['cases'][0].update(route_locations=[594]), lambda d: d['cases'][0]['coefficients']['qweight'].update(sha256='bad'),
    lambda d: d['cases'][0].update(route_locations=[0.5]), lambda d: d['cases'][0].update(route_rows=[False]),
    lambda d: d['cases'][0]['arrays']['activation'].update(sha256='bad'),
    lambda d: d.update(captured_array_bytes=1), lambda d: d['cases'].pop(),
    lambda d: d['stratum_call_counts'].update(other=1),
    lambda d: d.update(missing_strata=[]),
])
def test_malformed_evidence_refused(evidence, mutate):
    path, manifest = evidence
    mutate(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        capture.load_capture(path)


@pytest.mark.parametrize('mutation', ['parser', 'seal', 'spec_source'])
def test_parser_seal_before_execution(evidence, mutation):
    path, manifest = evidence
    spec = json.loads((path.parent/'spec.json').read_text())
    if mutation == 'parser':
        (path.parent/'harness/ds41_down_actual_capture.py').write_text("raise AssertionError('must not execute')")
    elif mutation == 'seal':
        spec['harness_sha256']['ds41_down_actual_capture.py'] = 'bad'
    else:
        spec['source_snapshot_sha256'] = 'another'
    (path.parent/'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='seal|identity'):
        component.capture_module(path)


def test_no_overwrite_and_identity_mismatch(evidence):
    path, manifest = evidence
    with pytest.raises(ValueError, match='identity'):
        capture.load_capture(path, source_snapshot='another')
    case, arrays = capture.load_capture(path)[1][0]
    with pytest.raises(FileExistsError):
        capture.save_case(path.parent, case['path'], layer=1, expert=3, kind='original',
            activation=arrays['activation'], raw=arrays['reference_raw'], boundary=arrays['reference_boundary'],
            coefficients=coefficients('original'), route_rows=np.array([0], np.uint32), route_locations=np.array([0], np.uint32))


@pytest.mark.parametrize('kind,rows,expected', [
    ('original', 1, 'original-m1-1'), ('original', 3, 'original-m2-3'), ('original', 4, 'original-m4-7'),
    ('original', 12, 'original-m8-12'), ('original', 13, 'original-m13-32'), ('original', 32, 'original-m13-32'),
    ('original', 33, 'original-m33-96'), ('original', 96, 'original-m33-96'), ('original', 99, 'original-m97-99'),
    ('q1', 1, 'q1-m1-1'), ('q1', 3, 'q1-m2-3'), ('q1', 99, 'q1-m4-99')])
def test_strata(kind, rows, expected):
    assert capture.band(kind, rows) == expected


@pytest.mark.parametrize('kind,rows', [('other', 1), ('q1', 0), ('original', 100), ('q1', True), ('original', 1.0)])
def test_invalid_strata(kind, rows):
    with pytest.raises(ValueError):
        capture.band(kind, rows)


@pytest.mark.parametrize('change', ['owner', 'input_grid', 'nonfinite', 'route', 'weight'])
def test_serializer_rejects_bad_owner_input_or_coefficients(tmp_path, change):
    x = np.zeros((2, 2304), np.float32)
    raw = np.zeros((2, 5120), np.float32)
    owner = raw.copy()
    route = np.array([0, 6], np.uint32)
    weights = coefficients('q1')
    if change == 'owner': owner[0, 0] = 1
    if change == 'input_grid': x[0, 0] = np.nextafter(np.float32(1), np.float32(2))
    if change == 'nonfinite': raw[0, 0] = np.nan
    if change == 'route': route[1] = 0
    if change == 'weight': weights['scales'] = weights['scales'].astype(np.float32)
    with pytest.raises(ValueError):
        capture.save_case(tmp_path, 'case.npz', layer=0, expert=0, kind='q1', activation=x, raw=raw, boundary=owner,
            coefficients=weights, route_rows=np.array([0, 1], np.uint32), route_locations=route)


def test_hook_exact_provider_scope_and_cleanup(tmp_path, monkeypatch):
    from mlx2.runtime.gguf import source_q1
    mx = ModuleType('mlx.core')
    mx.float32 = np.dtype('float32')
    mx.bfloat16 = np.dtype('float32')  # All-zero fixture owner, no fake BF16 arithmetic claim.
    mx.eval = lambda *a: None
    returned = []
    def dot(x, *args, **kw):
        value = np.zeros((x.shape[0], 5120), np.float32)
        returned.append(value)
        return value
    mx.quantized_matmul = dot
    package = ModuleType('mlx')
    package.core = mx
    monkeypatch.setitem(sys.modules, 'mlx', package)
    monkeypatch.setitem(sys.modules, 'mlx.core', mx)
    monkeypatch.setattr(source_q1, 'matmul', dot)
    writer = ModuleType('probe_deepseek_v4_packed')
    writer.atomic = lambda p, d: p.write_text(json.dumps(d))
    monkeypatch.setitem(sys.modules, 'probe_deepseek_v4_packed', writer)
    objects = {}
    for expert in range(6):
        # Distinct array identities even if the fixture uses identical zeros.
        objects[expert] = {'down.'+k: v for k, v in coefficients('original' if expert == 0 else 'q1').items()}
    class Stream:
        calls = 0
        artifact = {'identity': {'fingerprint': 'fixture-original'}}
        q1_binding = 'fixture-q1'
        hot_binding = 'fixture-hot'
        def kind(self, key): return 'original' if key[1] == 0 else 'q1'
        def acquire(self, prefix, expert):
            self.calls += 1
            return objects[expert]
    stream = Stream()
    unrelated = np.zeros((5120, 288), np.uint32)
    class Weights:
        expert_stream = stream
        def _prefill_batched_gate_up(self, x, values, scores, prefix, order):
            data = [stream.acquire(prefix, e) for e in range(6)]
            # Same geometry outside the identity set must not be captured.
            mx.quantized_matmul(x, unrelated, scales=objects[0]['down.scales'], mode='mxfp4')
            for e, d in enumerate(data):
                y = (mx.quantized_matmul(x, d['down.qweight'], scales=d['down.scales'], mode='mxfp4') if e == 0
                     else source_q1.matmul(x, d['down.qweight'], d['down.scales'], d['down.biases']))
                assert y is returned[-1]
            return y
    weights = Weights()
    model = SimpleNamespace(weights=weights)
    spec = dict(checkpoint='synthetic-down@fixture', source_revision='fixture', source_snapshot_sha256='fixture-source',
        expert_policy=dict(mode='hot_original_q1', budget_bytes=75_000_000_000, hot_bytes=60_000_000_000),
        extra_source_policy={'deepseek_source_prefill_batch': 'gate_up'})
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    values = np.tile(np.arange(6, dtype=np.uint32), (99, 1))
    x = np.zeros((99, 2304), np.float32)
    with capture.instrument(model, tmp_path, 'capture', 0, input_rows=99, frontier=0):
        with pytest.raises(RuntimeError, match='Nested'):
            with capture.instrument(model, tmp_path, 'nested', 1, input_rows=99, frontier=0): pass
        for layer in range(40):
            weights._prefill_batched_gate_up(x, values, None, f'blk.{layer}', None)
    assert stream.calls == 240 and mx.quantized_matmul is dot and source_q1.matmul is dot
    assert '_prefill_batched_gate_up' not in vars(weights) and 'acquire' not in vars(stream) and not capture._ACTIVE
    result, cases = capture.load_capture(tmp_path/'down-actual-capture-0000.json')
    assert len(cases) == 2 and result['observed_down_calls'] == 240
    assert result['stratum_call_counts'] == {'original-m97-99': 40, 'q1-m4-99': 200}
    with pytest.raises(RuntimeError, match='fixture'):
        with capture.instrument(model, tmp_path, 'failure', 2, input_rows=99, frontier=0):
            raise RuntimeError('fixture')
    assert mx.quantized_matmul is dot and not capture._ACTIVE
    assert json.loads((tmp_path/'down-actual-capture-0002.json').read_text())['status'] == 'failed'
    with pytest.raises(ValueError, match='incomplete'):
        with capture.instrument(model, tmp_path, 'empty', 3, input_rows=99, frontier=0): pass
    assert not capture._ACTIVE

    # Duplicate caller acquire is refused before a second underlying acquire.
    def duplicate(x, values, scores, prefix, order):
        stream.acquire(prefix, 0)
        stream.acquire(prefix, 0)
    weights._prefill_batched_gate_up = duplicate
    before = stream.calls
    with pytest.raises(ValueError, match='ownership'):
        with capture.instrument(model, tmp_path, 'duplicate', 4, input_rows=99, frontier=0):
            weights._prefill_batched_gate_up(x, values, None, 'blk.0', None)
    assert stream.calls == before+1 and weights._prefill_batched_gate_up is duplicate
    assert 'acquire' not in vars(stream) and not capture._ACTIVE

    # A same-weight but foreign scale pointer is not accepted by geometry.
    def wrong_scale(x, values, scores, prefix, order):
        data = stream.acquire(prefix, 0)
        mx.quantized_matmul(x, data['down.qweight'], scales=data['down.scales'].copy(), mode='mxfp4')
    weights._prefill_batched_gate_up = wrong_scale
    with pytest.raises(ValueError, match='leased job'):
        with capture.instrument(model, tmp_path, 'scales', 5, input_rows=99, frontier=0):
            weights._prefill_batched_gate_up(x, values, None, 'blk.0', None)
    assert weights._prefill_batched_gate_up is wrong_scale and not capture._ACTIVE


@pytest.mark.parametrize('value', [None, {'repeats': 2}, {'repeats': True}, {'repeats': 8}, {'extra': 1}, {'include_original_e1': 1}])
def test_bad_component_options(value):
    with pytest.raises(ValueError):
        component.options({'down_actual_options': value})


def test_q1_candidate_dot_source_is_identical_except_final_store():
    from mlx2.runtime.gguf import source_q1, source_q1_down_owner as owner
    stock = (source_q1.SOURCE.replace('GROUP_SIZE', '64').replace('OUTPUTS_PER_THREADGROUP', '8')
             .replace('OUTPUTS_PER_SIMDGROUP', '4').replace('ROW_VALID', 'output_row < output_dims'))
    for raw in (False, True):
        candidate = owner.kernel_spec(return_raw=raw)
        store = 'float(bfloat(accumulators[row]));'
        if raw:
            store += '\n            raw[input_row * output_dims + output_row] = accumulators[row];'
        assert candidate['source'].replace(store, 'static_cast<T>(accumulators[row]);') == stock
        assert candidate['threadgroup'] == (64, 1, 1) and candidate['dot_body_modifications'] == 0
        assert 'atomic' not in candidate['source'] and 'threadgroup_barrier' not in candidate['source']


def test_source_pin_and_prelaunch_geometry_refusal(monkeypatch):
    import mlx.core as mx
    from mlx2.runtime.gguf import source_q1_down_owner as owner
    monkeypatch.setattr(owner, 'SOURCE', owner.SOURCE+'\n')
    with pytest.raises(ValueError, match='requires review'):
        owner.kernel_spec()
    x = SimpleNamespace(ndim=2, shape=(100, 2304), dtype=mx.float32)
    data = [SimpleNamespace(shape=(5120, 72), dtype=mx.uint32),
            SimpleNamespace(shape=(5120, 36), dtype=mx.float16), SimpleNamespace(shape=(5120, 36), dtype=mx.float16)]
    with pytest.raises(ValueError, match='geometry'):
        owner.plan(x, *data)


@pytest.mark.parametrize('rows', [1, 3, 7, 12, 99])
def test_q1_geometry_cpu_only(rows):
    import mlx.core as mx
    from mlx2.runtime.gguf import source_q1_down_owner as owner
    data = coefficients('q1')
    x = SimpleNamespace(ndim=2, shape=(rows, 2304), dtype=mx.float32)
    fields = [SimpleNamespace(shape=v.shape, dtype=mx.uint32 if k == 'qweight' else mx.float16) for k, v in data.items()]
    result = owner.plan(x, *fields)
    assert result['grid'] == (40960, rows, 1) and result['logical_output_bytes'] == rows*5120*4


@pytest.mark.parametrize('raw', [False, True])
def test_q1_offline_compile_no_gpu(raw, tmp_path):
    import mlx.core as mx
    from mlx2.runtime.gguf import source_q1_down_owner as owner
    if not shutil.which('xcrun'):
        pytest.skip('Offline Metal compiler unavailable')
    source = tmp_path/'q1.metal'
    source.write_text(owner.compilation_source(return_raw=raw))
    command = ['xcrun', '-sdk', 'macosx', 'metal', '-std=metal3.2', '-I', str(Path(mx.__file__).parent/'include'),
               '-c', str(source), '-o', str(tmp_path/'q1.air')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.parametrize('candidate_exact', [True, False])
def test_component_actual_oracles_gate_timing_cpu(evidence, tmp_path, monkeypatch, candidate_exact):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/research'))
    from mlx2.runtime.gguf import source_q1, source_q1_down_owner, source_down_prefill
    from mlx2.adapters import deepseek_v41_source_artifact as artifact
    import ds41_attention_components as clocks
    path, manifest = evidence
    spec = json.loads((path.parent/'spec.json').read_text())
    spec['model'] = 'synthetic-model'
    (path.parent/'spec.json').write_text(json.dumps(spec))
    monkeypatch.setattr(artifact, 'inspect_artifact', lambda path: {'identity': {'fingerprint': 'fixture-artifact'}})
    loaded = []
    def loader(case):
        loaded.append((case['layer'], case['expert'], case['kind']))
        return {'down.'+k: v for k, v in coefficients(case['kind']).items()}
    monkeypatch.setattr(component, 'load_expert_factory', lambda *args: loader)
    mx = SimpleNamespace(gpu='fixture-gpu', default_device=lambda: 'fixture-gpu',
        float32=np.float32, bfloat16=np.float32, array=lambda x, *a, **kw: np.array(x),
        eval=lambda *a: None, synchronize=lambda: None, clear_cache=lambda: None, get_peak_memory=lambda: 0,
        quantized_matmul=lambda x, *a, **kw: np.zeros((len(x), 5120), np.float32))
    monkeypatch.setattr(source_q1, 'matmul', mx.quantized_matmul)
    def candidate(x, *args, return_raw=False):
        y = np.full((len(x), 5120), 0 if candidate_exact else 1, np.float32)
        return (y, y) if return_raw else y
    monkeypatch.setattr(source_q1_down_owner, 'project', candidate)
    monkeypatch.setattr(source_down_prefill, 'pp_down', lambda x, *args, return_raw=False: (
        (np.zeros((len(x), 5120), np.float32),)*2 if return_raw else np.zeros((len(x), 5120), np.float32)))
    timed = []
    def timing(mx, np, functions, *, repeats):
        timed.append(tuple(functions))
        assert repeats == 3
        for name, fn in functions.items():
            assert not np.any(fn()), name
        return {name: {'fixture_clock': True} for name in functions}
    monkeypatch.setattr(clocks, 'timed_interleaved', timing)
    out = tmp_path/'component'
    out.mkdir()
    report = dict(checkpoint=manifest['checkpoint'], source_revision=manifest['source_revision'], source_snapshot_sha256='component-fixture')
    result = component.run_components(mx, np, path.parent, out, report, None, lambda p, d: p.write_text(json.dumps(d)))
    assert result['complete'] and result['harness_completed'] and result['required_reference_controls_exact']
    assert result['verdicts']['all_candidate_raw_and_owner_exact'] is candidate_exact
    assert result['verdicts']['q1_adversarials_exact'] is candidate_exact
    assert timed[0] == ('reference', 'original_e1_control')
    assert timed[1] == (('reference', 'q1_owner_store') if candidate_exact else ('reference',))
    assert len(loaded) == 2 and result['loaded_experts_concurrently'] == 1


def test_upload_failure_quarantines_host_only_if_completion_fails(monkeypatch):
    from mlx2.adapters import deepseek_v41_source_buffers as buffers
    host = object()
    packed = SimpleNamespace(backing=object(), values=lambda: [])
    monkeypatch.setattr(buffers, 'materialize', lambda h: packed)
    def fail(*args): raise RuntimeError('fixture completion failure')
    mx = SimpleNamespace(eval=fail, synchronize=fail)
    component._UPLOAD_QUARANTINE.clear()
    with pytest.raises(RuntimeError, match='completion'):
        component._materialize(mx, host)
    assert component._UPLOAD_QUARANTINE == [(host, packed)]
    component._UPLOAD_QUARANTINE.clear()
    mx.synchronize = lambda: None
    with pytest.raises(RuntimeError):
        component._materialize(mx, host)
    assert component._UPLOAD_QUARANTINE == []


def test_loader_actual_precision_identity_and_no_implicit_bank(tmp_path, monkeypatch):
    from mlx2.adapters import deepseek_v41_source_buffers as buffers
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    hot = tmp_path/'hot.json'
    hot.write_text(json.dumps({'experts': [[1, 3]]}))
    manifest = dict(original_artifact_fingerprint='fixture-artifact', q1_bank_binding='fixture-q1',
                    hot_map_binding=hashlib.sha256(hot.read_bytes()).hexdigest())
    spec = {'expert_policy': {'hotset_path': str(hot), 'q1_root': 'fixture-bank'}}
    calls = []
    def q1_manifest(root, source):
        assert root == 'fixture-bank' and source == 'fixture-artifact'
        return root, {(1, 4): 'q1-record'}, 'fixture-q1'
    monkeypatch.setattr(lfu, 'load_q1_manifest', q1_manifest)
    monkeypatch.setattr(lfu, 'q1_read', lambda root, record: calls.append(('q1', record)) or 'q1-host')
    monkeypatch.setattr(buffers, 'read_original_packed', lambda artifact, layer, expert: calls.append(('original', layer, expert)) or 'original-host')
    monkeypatch.setattr(buffers, 'borrow_q1_packed', lambda host: host)
    monkeypatch.setattr(component, '_materialize', lambda mx, host: host)
    load = component.load_expert_factory(None, {'identity': {'fingerprint': 'fixture-artifact'}}, manifest, spec)
    assert calls == []
    assert load({'layer': 1, 'expert': 3, 'kind': 'original'}) == 'original-host'
    assert load({'layer': 1, 'expert': 4, 'kind': 'q1'}) == 'q1-host'
    before = list(calls)
    with pytest.raises(ValueError, match='precision'):
        load({'layer': 1, 'expert': 3, 'kind': 'q1'})
    assert calls == before
    with pytest.raises(ValueError, match='artifact'):
        component.load_expert_factory(None, {'identity': {'fingerprint': 'other'}}, manifest, spec)
    hot.write_text('changed')
    with pytest.raises(ValueError, match='precision map'):
        component.load_expert_factory(None, {'identity': {'fingerprint': 'fixture-artifact'}}, manifest, spec)
