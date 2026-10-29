"""Synthetic complete-stage serialization/protocol fixtures; no GPU."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT/'scripts/research/ds41_native_ffn_capture.py'
    spec = importlib.util.spec_from_file_location('_native_ffn_capture_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_binding():
    return dict(kind='original', live_lease_verified=True, weight_payload_exported=False,
        source_fingerprint='a'*64, source_seals_sha256='b'*64, layout_sha256='c'*64)


def fixture(module, rows=16):
    arrays = {k: np.zeros(shape, np.uint32 if k in module.UINT_ARRAYS else np.float32)
              for k, shape in module.shapes(rows).items()}
    values = np.tile(np.arange(6, dtype=np.uint32), (99, 1))
    values[:rows, 0] = 38
    arrays['values'] = values
    arrays['scores'].fill(.125)
    arrays['route_rows'] = np.arange(rows, dtype=np.uint32)
    arrays['route_locations'] = np.arange(rows, dtype=np.uint32)*6
    arrays['route_scores'].fill(.125)
    arrays['order'] = np.argsort(values.reshape(-1), kind='stable').astype(np.uint32)
    arrays['assembly_rank'].fill(5)
    arrays['assembly_parts'][:, 0] = np.float32(1e10)
    arrays['assembly_parts'][:, 1] = np.float32(-1e10)
    arrays['assembly_parts'][:, 2] = 1.
    arrays['assembly_parts'][:, 3] = 2.
    arrays['assembly_parts'][:, 4] = 3.
    arrays['assembly_output'].fill(6.)
    unique, counts = np.unique(values, return_counts=True)
    case = dict(layer=0, expert=38, kind='original', rows=rows, part_index=6,
        expert_order=unique.tolist(), expert_row_counts=counts.tolist(), same_forward=True,
        source_binding=source_binding())
    return case, arrays


def manifest(module, cases):
    return dict(schema=module.SCHEMA, status='completed', checkpoint='synthetic-native-ffn@'+'f'*64,
        source_revision='89e6938ea58a0060026815c94e0ed4d33729d775', source_snapshot_sha256='d'*64,
        helper_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        model_forward_completed=True, source_result_always_returned=True, candidate_used_by_model=False,
        rate_eligible=False, input_rows=99, frontier=0, observed_layers=list(range(40)),
        extra_expert_acquisitions=0, extra_coefficient_copy_bytes=0,
        expert_budget_bytes=75_000_000_000, original_hot_bytes=60_000_000_000,
        original_artifact_fingerprint='a'*64, q1_bank_binding='e'*64, hot_map_binding='f'*64,
        capture_policy={'prefill_batch': 'gate_up', 'expert_math': 'routed_fused_fp32', 'wide_math': 'reference', 'wide_dtype': 'reference'},
        cases=cases, captured_array_bytes=sum(c['captured_array_bytes'] for c in cases))


@pytest.fixture
def saved(tmp_path):
    module = load()
    case, arrays = fixture(module)
    record = module.save_case(tmp_path, 'case.npz', case, arrays)
    path = tmp_path/'native-ffn-capture-0000.json'
    path.write_text(json.dumps(manifest(module, [record])))
    return module, path, arrays


def test_complete_stage_roundtrip_keeps_actual_scores_and_assembly(saved):
    module, path, arrays = saved
    report, cases = module.load_capture(path.parent)
    case, restored = cases[0]
    assert set(restored) == set(module.shapes(16))
    assert all(module._same(arrays[k], restored[k]) for k in arrays)
    assert report['captured_array_bytes'] == module.estimated_bytes(16) < 32 << 20
    assert case['source_binding']['weight_payload_exported'] is False


def test_actual_case_selection_prefers_present_ids_without_invented_counts():
    module = load()
    jobs = [SimpleNamespace(kind='original', expert_id=e, count=m) for e, m in ((0, 98), (38, 34), (50, 32))]
    assert module.choose_jobs(jobs, 0) == (1, 2)
    assert module.choose_jobs(jobs, 0, remaining_cases=1) == (1,)
    assert module.choose_jobs(jobs, 0, remaining_bytes=module.estimated_bytes(32)) == (2,)
    jobs[1].count = 3
    jobs[2].kind = 'q1'
    assert module.choose_jobs(jobs, 0) == (0,)
    assert module.estimated_bytes(99) < 32 << 20


@pytest.mark.parametrize('bad', ['gate_owner', 'up_owner', 'weighted_owner', 'down_owner', 'score',
    'route', 'split', 'order', 'middle', 'assembly_rank', 'assembly_selected', 'assembly_order',
    'part_index', 'expert_order', 'same_forward', 'shared_input', 'nonfinite', 'dtype', 'missing', 'extra'])
def test_array_contract_refuses_changed_boundaries_routes_or_assembly(bad):
    module = load()
    case, arrays = fixture(module)
    if bad.endswith('_owner'):
        key = {'gate_owner': 'gate_owned', 'up_owner': 'up_owned', 'weighted_owner': 'weighted_owned', 'down_owner': 'down_owned'}[bad]
        arrays[key][0, 0] = 1.
    elif bad == 'score': arrays['route_scores'][0] = .25
    elif bad == 'route': arrays['route_rows'][0] = 98
    elif bad == 'split': arrays['values'][98, 0] = 38  # An omitted called row.
    elif bad == 'order': arrays['order'][0], arrays['order'][1] = arrays['order'][1], arrays['order'][0]
    elif bad == 'middle': arrays['middle_reference'][0, 0] = 1.
    elif bad == 'assembly_rank': arrays['assembly_rank'][0] = 0
    elif bad == 'assembly_selected': arrays['assembly_parts'][0, 5, 0] = 1.
    elif bad == 'assembly_order': arrays['assembly_parts'][:, [1, 2]] = arrays['assembly_parts'][:, [2, 1]]
    elif bad == 'part_index': case['part_index'] = 0
    elif bad == 'expert_order': case['expert_order'].reverse()
    elif bad == 'same_forward': case['same_forward'] = False
    elif bad == 'shared_input': arrays['shared_gu_input'][0, 0] = .25
    elif bad == 'nonfinite': arrays['gate_raw'][0, 0] = np.nan
    elif bad == 'dtype': arrays['route_scores'] = arrays['route_scores'].astype(np.float64)
    elif bad == 'missing': arrays.pop('scores')
    else: arrays['weights'] = np.zeros(1, np.float32)
    with pytest.raises(ValueError): module.validate_arrays(case, arrays)


@pytest.mark.parametrize('key,value', [('status', 'running'), ('model_forward_completed', False),
    ('source_result_always_returned', False), ('candidate_used_by_model', True), ('rate_eligible', True),
    ('frontier', 1), ('input_rows', 98), ('extra_expert_acquisitions', 1), ('extra_coefficient_copy_bytes', 4),
    ('expert_budget_bytes', 90_000_000_000), ('observed_layers', [0]), ('capture_policy', {}),
    ('q1_bank_binding', ''), ('captured_array_bytes', 0)])
def test_manifest_contract_refuses_incomplete_or_changed_identity(saved, key, value):
    module, path, _ = saved
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): module.load_capture(path)


@pytest.mark.parametrize('bad', ['npz_hash', 'array_hash', 'npz_symlink', 'path', 'duplicate', 'lease', 'payload', 'source_binding'])
def test_seals_paths_and_lease_binding_are_required(saved, bad):
    module, path, _ = saved
    data = json.loads(path.read_text())
    case = data['cases'][0]
    if bad == 'npz_hash': case['sha256'] = '0'*64
    elif bad == 'array_hash': case['arrays']['scores']['sha256'] = '0'*64
    elif bad == 'npz_symlink':
        original = path.parent/'case.npz'
        moved = path.parent/'moved.npz'
        original.rename(moved)
        original.symlink_to(moved)
    elif bad == 'path': case['path'] = '../case.npz'
    elif bad == 'duplicate': data['cases'].append(case)
    elif bad == 'lease': case['source_binding']['live_lease_verified'] = False
    elif bad == 'payload': case['source_binding']['weight_payload_exported'] = True
    else: case['source_binding']['source_fingerprint'] = '0'*64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): module.load_capture(path)


def test_npy_declared_huge_allocation_refused_before_numpy_load(saved, monkeypatch):
    module, path, _ = saved
    data = json.loads(path.read_text())
    file = path.parent/'case.npz'
    with zipfile.ZipFile(file) as archive: contents = {i.filename: archive.read(i) for i in archive.infolist()}
    body = io.BytesIO()
    np.lib.format.write_array_header_1_0(body, dict(descr='<f4', fortran_order=False, shape=(1 << 40,)))
    contents['scores.npy'] = body.getvalue()
    with zipfile.ZipFile(file, 'w') as archive:
        for name, value in contents.items(): archive.writestr(name, value)
    data['cases'][0]['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(data))
    def forbidden(*args, **kwargs): raise AssertionError('NumPy allocation path reached')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='NPY header'): module.load_capture(path)


def test_existing_npz_and_wrong_checkpoint_or_source_refused(saved):
    module, path, arrays = saved
    case = json.loads(path.read_text())['cases'][0]
    with pytest.raises(FileExistsError): module.save_case(path.parent, 'case.npz', case, arrays)
    with pytest.raises(ValueError): module.load_capture(path, checkpoint='other@identity')
    with pytest.raises(ValueError): module.load_capture(path, source_snapshot='other-source')


def test_import_and_bad_scope_are_cpu_only_and_nested_scope_refused():
    module = load()
    tree = ast.parse(Path(module.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import): assert not any(a.name.startswith('mlx') for a in node.names)
        if isinstance(node, ast.ImportFrom): assert not (node.module or '').startswith('mlx')
    with pytest.raises(ValueError):
        with module.instrument(None, None, 'fixture', 0, input_rows=1, frontier=100): pass
    module._LOCK.acquire()
    try:
        with pytest.raises(RuntimeError, match='Nested'):
            with module.instrument(None, None, 'fixture', 0, input_rows=99, frontier=0): pass
    finally: module._LOCK.release()


@pytest.mark.parametrize('bad', [None, 'inactive', 'foreign', 'q1', 'backing', 'views'])
def test_source_binding_uses_live_canonical_lease_without_payload_export(monkeypatch, bad):
    capture = load()
    mx = ModuleType('mlx.core')
    mx.uint8, mx.uint32 = np.uint8, np.uint32
    package = ModuleType('mlx')
    package.core = mx
    monkeypatch.setitem(sys.modules, 'mlx', package)
    monkeypatch.setitem(sys.modules, 'mlx.core', mx)
    from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert, canonical_layout
    from mlx2.adapters.deepseek_v41_source_weights import expert_names
    layout = canonical_layout('q1' if bad == 'q1' else 'original')
    types = {'<u4': np.uint32, '|u1': np.uint8, '<f2': np.float16}
    tensors = {t.name: SimpleNamespace(shape=t.shape, dtype=types[t.dtype]) for t in layout.tensors}
    backing = SimpleNamespace(dtype=np.uint8, shape=(layout.nbytes,))
    data = PackedExpert(backing, layout, tensors)
    if bad == 'backing': backing.shape = (1,)
    if bad == 'views': tensors['down.scales'].shape = (1,)
    stream = SimpleNamespace(_active=bad != 'inactive', _leased={(0, 38): object() if bad == 'foreign' else data})
    artifact = dict(identity={'fingerprint': 'a'*64}, tensors={n: {'file': 'actual-fixture-shard'} for n in expert_names(0, 38)},
        seals={'actual-fixture-shard': {'inode': 1, 'size': 2}})
    owner, job = SimpleNamespace(artifact=artifact), SimpleNamespace(expert_id=38)
    if bad:
        with pytest.raises(ValueError): capture.source_binding(owner, stream, 0, job, data)
    else:
        result = capture.source_binding(owner, stream, 0, job, data)
        assert result['live_lease_verified'] and result['weight_payload_exported'] is False
        assert result['backing_object_id'] == id(backing) and len(result['tensor_layout']) == 6


@pytest.mark.parametrize('failure', [None, 'projection', 'cancel'])
def test_live_hook_uses_current_source_provider_returns_original_and_restores(tmp_path, monkeypatch, failure):
    """Numpy-backed fake GPU, actual current Source provider control flow."""
    capture = load()
    class Tensor(np.ndarray):
        def __new__(cls, value, dtype=None): return np.asarray(value, dtype=dtype).view(cls)
        def astype(self, dtype, *args, **kwargs):
            if dtype == 'fake_bf16': return Tensor(capture.bf16(np.asarray(self)))
            return Tensor(np.asarray(self).astype(dtype, *args, **kwargs))
    mx = ModuleType('mlx.core')
    mx.array = Tensor
    mx.float32, mx.uint32, mx.bfloat16 = np.float32, np.uint32, 'fake_bf16'
    mx.eval = lambda *args: None
    mx.minimum = lambda a, b: Tensor(np.minimum(a, b))
    mx.clip = lambda a, lo, hi: Tensor(np.clip(a, lo, hi))
    mx.concatenate = lambda parts, axis=0: Tensor(np.concatenate(parts, axis=axis))
    nn = ModuleType('mlx.nn')
    nn.silu = lambda a: Tensor(np.zeros_like(a))  # All synthetic raw GU are zero.
    package = ModuleType('mlx')
    package.core, package.nn = mx, nn
    monkeypatch.setitem(sys.modules, 'mlx', package)
    monkeypatch.setitem(sys.modules, 'mlx.core', mx)
    monkeypatch.setitem(sys.modules, 'mlx.nn', nn)
    quant = ModuleType('mlx2.runtime.models.deepseek_v41_quant')
    quant.quantize = lambda value, kind: value
    monkeypatch.setitem(sys.modules, quant.__name__, quant)
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    from mlx2.adapters import deepseek_v41_source_math as math
    from mlx2.runtime.gguf import source_middle_fusion as middle, source_mxfp4_pair as pair
    from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs
    monkeypatch.setattr(pair, '_device_supported', lambda: True)
    original_results = []
    def combine(parts, values, order):
        offsets = math.ordered_offsets(values, order)
        packed = np.concatenate(parts)
        total = np.zeros((99, 5120), np.float32)
        for slot in range(6): np.add(total, packed[offsets[:, slot]], out=total)
        result = Tensor(total)
        original_results.append(result)
        return result
    def normal_middle(gate, up, scores, slots): return Tensor(np.zeros_like(gate))
    monkeypatch.setattr(math, 'ordered_combine', combine)
    monkeypatch.setattr(middle, 'middle', normal_middle)
    def qmm(x, weight, **kwargs):
        if failure == 'projection': raise RuntimeError('synthetic source projection error')
        return Tensor(np.zeros((len(x), weight.width), np.float32))
    mx.quantized_matmul = qmm
    class Stream:
        packed_buffers = True
        artifact = {'identity': {'fingerprint': 'a'*64}}
        q1_binding, hot_binding = 'e'*64, 'f'*64
        def __init__(self): self._active, self._leased, self.calls, self.stats = True, {}, [], {'lookups': 0}
        def kind(self, key): return 'original'
        def acquire(self, prefix, expert):
            layer = int(prefix.split('.')[1])
            self.calls.append((layer, expert))
            data = {}
            for name in ('gate', 'up', 'down'):
                data[name+'.qweight'] = SimpleNamespace(width=5120 if name == 'down' else 2304)
                data[name+'.scales'] = object()
            self._leased[(layer, expert)] = data
            return data
    class Weights:
        _prefill_batched_gate_up = SourceV41Weights._prefill_batched_gate_up
        source_prefill_batch_policy, expert_math_policy = 'gate_up', 'routed_fused_fp32'
        source_prefill_wide_math = source_prefill_wide_dtype = 'reference'
        source_prefill_down_policy = source_ffn_input_policy = 'reference'
        source_fp8_pp_policy = 'native_bf16_qb_wob_v1'
        def __init__(self): self.expert_stream, self.scheduled_expert_calls = Stream(), {}
    weights = Weights()
    stream = weights.expert_stream
    def live_binding(owner, leased, layer, job, data):
        assert leased._active and leased._leased[(layer, job.expert_id)] is data
        return source_binding()
    monkeypatch.setattr(capture, 'source_binding', live_binding)
    spec = dict(checkpoint='synthetic-native-ffn@'+'f'*64,
        source_revision='89e6938ea58a0060026815c94e0ed4d33729d775', source_snapshot_sha256='d'*64,
        native_ffn_capture=True, resident_discovery=True, rate_eligible=False,
        harness_sha256={'ds41_native_ffn_capture.py': hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest()},
        expert_policy={'budget_bytes':75_000_000_000, 'hot_bytes':60_000_000_000, 'mode':'hot_original_q1'})
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    values = np.tile(np.arange(6, dtype=np.uint32), (99, 1))
    values[:34, 0] = 38
    values[34:66, 1] = 50
    order = np.argsort(values.reshape(-1), kind='stable')
    x, scores = Tensor(np.full((99, 5120), .25, np.float32)), Tensor(np.full((99, 6), .125, np.float32))
    count = len(plan_jobs(values, {int(e): 'original' for e in np.unique(values)}).jobs)
    try:
        with capture.instrument(SimpleNamespace(weights=weights), tmp_path, 'fixture', 0, input_rows=99, frontier=0):
            if failure == 'cancel': raise KeyboardInterrupt('synthetic cancellation')
            for layer in range(40):
                result = weights._prefill_batched_gate_up(x, values, scores, f'layers.{layer}.ffn', order)
                assert result is original_results[-1]
    except (RuntimeError, KeyboardInterrupt):
        assert failure is not None
    assert mx.quantized_matmul is qmm and math.ordered_combine is combine and middle.middle is normal_middle
    assert '_prefill_batched_gate_up' not in vars(weights) and 'acquire' not in vars(stream)
    assert capture._LOCK.acquire(blocking=False)
    capture._LOCK.release()
    report = json.loads((tmp_path/'native-ffn-capture-0000.json').read_text())
    if failure is None:
        assert report['status'] == 'completed' and report['observed_acquire_jobs'] == count*40
        report, captured = capture.load_capture(tmp_path)
        assert [(c['expert'], c['rows']) for c, _ in captured] == [(38, 34), (50, 32)]
        assert report['captured_array_bytes'] < 32 << 20
    else:
        assert report['status'] == 'failed' and report['model_forward_completed'] is False
        with pytest.raises(ValueError): capture.load_capture(tmp_path)
