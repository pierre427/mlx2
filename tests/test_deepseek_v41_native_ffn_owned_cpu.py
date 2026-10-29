"""Owned native MX controller fail-closed CPU fixtures; no real GPU or leases."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    source = ROOT/'scripts/research'/f'{name}.py'
    spec = importlib.util.spec_from_file_location('_owned_test_'+name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def capture_unit(tmp_path):
    helper = load('ds41_native_ffn_capture')
    unit = tmp_path/'capture'
    (unit/'harness').mkdir(parents=True)
    code = (ROOT/'scripts/research/ds41_native_ffn_capture.py').read_bytes()
    (unit/'harness/ds41_native_ffn_capture.py').write_bytes(code)
    values = np.tile(np.arange(6, dtype=np.uint32), (99,1))
    values[:16,0] = 38; values[16:32,0] = 50
    experts, counts = np.unique(values, return_counts=True)
    cases=[]
    for i,expert in enumerate((38,50)):
        arrays={key:np.zeros(shape,np.uint32 if key in helper.UINT_ARRAYS else np.float32) for key,shape in helper.shapes(16).items()}
        locations=np.flatnonzero(values.reshape(-1)==expert).astype(np.uint32)
        arrays.update(values=values,order=np.argsort(values.reshape(-1),kind='stable').astype(np.uint32),
            route_locations=locations,route_rows=locations//6,assembly_rank=np.full(16,5,np.uint32))
        case=dict(layer=0,expert=expert,kind='original',rows=16,part_index=int(np.flatnonzero(experts==expert)[0]),
            expert_order=experts.tolist(),expert_row_counts=counts.tolist(),same_forward=True,
            source_binding=dict(kind='original',live_lease_verified=True,weight_payload_exported=False,
                source_fingerprint='a'*64,source_seals_sha256='b'*64,layout_sha256='c'*64))
        cases.append(helper.save_case(unit,('a.npz','b.npz')[i],case,arrays))
    identity=dict(checkpoint='synthetic-native-ffn@'+'f'*64,source_revision='fixture-revision',source_snapshot_sha256='d'*64)
    seal=hashlib.sha256(code).hexdigest()
    spec=dict(**identity,harness_sha256={'ds41_native_ffn_capture.py':seal},model='fixture-model',context_cap=16384,prompt_tokens=100,
        native_ffn_capture=True,resident_discovery=True,rate_eligible=False)
    manifest=dict(**identity,schema=helper.SCHEMA,status='completed',model_forward_completed=True,
        source_result_always_returned=True,candidate_used_by_model=False,rate_eligible=False,input_rows=99,frontier=0,
        observed_layers=list(range(40)),helper_sha256=seal,cases=cases,captured_array_bytes=sum(c['captured_array_bytes']for c in cases),
        expert_budget_bytes=75_000_000_000,original_hot_bytes=60_000_000_000,extra_expert_acquisitions=0,extra_coefficient_copy_bytes=0,
        original_artifact_fingerprint='a'*64,q1_bank_binding='e'*64,hot_map_binding='f'*64,
        capture_policy={'prefill_batch':'gate_up','expert_math':'routed_fused_fp32','wide_math':'reference','wide_dtype':'reference'})
    for name,value in [('spec.json',spec),('native-ffn-capture-0000.json',manifest),
        ('results.json',dict(checkpoint=identity['checkpoint'],source={'commit':identity['source_revision'],'source_snapshot_sha256':identity['source_snapshot_sha256']},status='completed',passed=True)),
        ('release-verified.json',dict(owners_absent=True,source_unchanged=True)),('request.json',{}),('policy.json',{})]:
        (unit/name).write_text(json.dumps(value))
    return unit


def test_capture_copy_exact_inventory_and_parser(capture_unit, tmp_path):
    controller = load('probe_ds41_native_ffn')
    out = tmp_path/'component'
    out.mkdir()
    # Unrelated payloads, including a symlink, are not enumerated/copied.
    (capture_unit/'foreign.npz').symlink_to('/never/read/foreign')
    dest = controller.copy_capture(capture_unit, out)
    assert {p.name for p in dest.glob('*.npz')} == {'a.npz', 'b.npz'}
    assert not (dest/'foreign.npz').exists()
    receipt = json.loads((out/'capture-copy.json').read_text())
    assert receipt['validated_cases'] == 2 and receipt['captured_array_bytes'] < 64 << 20
    assert len(receipt['files']) == 8


@pytest.mark.parametrize('failure', ['release', 'result', 'source', 'parser_hash', 'parser_symlink', 'npz_hash', 'npz_symlink', 'path', 'duplicate', 'two_manifests'])
def test_copy_refuses_bad_release_identity_paths_or_seals_before_dest(capture_unit, tmp_path, failure):
    controller = load('probe_ds41_native_ffn')
    unit = capture_unit
    manifest_path = unit/'native-ffn-capture-0000.json'
    manifest = json.loads(manifest_path.read_text())
    if failure == 'release': (unit/'release-verified.json').write_text(json.dumps({'owners_absent': False, 'source_unchanged': True}))
    elif failure == 'result': (unit/'results.json').write_text(json.dumps({'status': 'failed', 'passed': False}))
    elif failure == 'source': manifest['source_snapshot_sha256'] = 'other'
    elif failure == 'parser_hash': (unit/'harness/ds41_native_ffn_capture.py').write_text("raise AssertionError('must not execute')")
    elif failure == 'parser_symlink':
        p = unit/'harness/ds41_native_ffn_capture.py'
        saved = unit/'saved.py'
        saved.write_bytes(p.read_bytes())
        p.unlink()
        p.symlink_to(saved)
    elif failure == 'npz_hash': (unit/'a.npz').write_bytes(b'changed')
    elif failure == 'npz_symlink':
        p = unit/'a.npz'
        saved = unit/'saved.npz'
        saved.write_bytes(p.read_bytes())
        p.unlink()
        p.symlink_to(saved)
    elif failure == 'path': manifest['cases'][0]['path'] = '../a.npz'
    elif failure == 'duplicate': manifest['cases'][1]['path'] = manifest['cases'][0]['path']
    elif failure == 'two_manifests': (unit/'native-ffn-capture-0001.json').write_text('{}')
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path/'component'
    out.mkdir()
    with pytest.raises(ValueError):
        controller.copy_capture(unit, out)
    assert not (out/'reference-capture').exists()


def sealed_spec(controller, out):
    return dict(schema=controller.SCHEMA, out=str(out), job_id=out.name,
        source_snapshot_sha256='expected', memory_limit_bytes=6 << 30, host_live_reserve_bytes=9 << 30,
        host_initial_floor_bytes=15 << 30, native_ffn_options={'repeats': 7, 'tile': [16, 32], 'stage': 'complete_original_expert_v1'},
        numerical_environment={'MLX_ENABLE_TF32': '0'}, rate_eligible=False, native_ffn_capture_path='fixture-capture')


@pytest.mark.parametrize('failure', ['ownership', 'source', 'host', 'foreign'])
def test_child_refuses_before_gpu_preparation(tmp_path, monkeypatch, failure):
    controller = load('probe_ds41_native_ffn')
    monkeypatch.setenv('MLX_ENABLE_TF32', '0')
    (tmp_path/'spec.json').write_text(json.dumps(sealed_spec(controller, tmp_path)))
    def reject(*args): raise RuntimeError('paired ownership refused')
    monkeypatch.setattr(controller, 'check_ownership', reject if failure == 'ownership' else lambda *args: {})
    monkeypatch.setattr(controller, 'source_manifest', lambda *args: {})
    monkeypatch.setattr(controller, 'digest', lambda *args: 'changed' if failure == 'source' else 'expected')
    def host(minimum):
        if failure == 'host': raise MemoryError('live host refused')
        return SimpleNamespace(available_bytes=100 << 30)
    monkeypatch.setattr(controller, 'host_admission', host)
    environment = ModuleType('probe_deepseek_v4_decode_compare')
    environment.HID_PROFILE = 'fixture'
    environment.sample_environment = lambda *args: {'foreign_model_processes': [{'pid': -1}]}
    monkeypatch.setitem(sys.modules, 'probe_deepseek_v4_decode_compare', environment)
    assert controller.component(tmp_path) == 1
    result = json.loads((tmp_path/'results.json').read_text())
    assert result['passed'] is False and 'cleanup_error' not in result
    assert ('ownership' in result['error']) if failure == 'ownership' else True
    assert ('source' in result['error']) if failure == 'source' else True
    assert ('host' in result['error']) if failure == 'host' else True
    assert ('Foreign model' in result['error']) if failure == 'foreign' else True


@pytest.mark.parametrize('field,value', [('memory_limit_bytes', 7 << 30), ('host_live_reserve_bytes', 8 << 30),
    ('host_initial_floor_bytes', 14 << 30), ('rate_eligible', True), ('native_ffn_options', {'repeats': 3, 'tile': [16, 32], 'stage': 'complete_original_expert_v1'}),
    ('numerical_environment', {'MLX_ENABLE_TF32': '1'})])
def test_hard_spec_bounds_fail_closed(tmp_path, monkeypatch, field, value):
    controller = load('probe_ds41_native_ffn')
    monkeypatch.setenv('MLX_ENABLE_TF32', '0')
    spec = sealed_spec(controller, tmp_path)
    spec[field] = value
    with pytest.raises(ValueError): controller.validate_spec(spec, tmp_path)


def test_host_admission_is_cpu_only_and_exact_floor(monkeypatch):
    controller = load('probe_ds41_native_ffn')
    from mlx2.runtime import os_memory
    monkeypatch.setattr(os_memory, 'host_memory_snapshot', lambda: SimpleNamespace(available_bytes=15 << 30))
    assert controller.host_admission(15 << 30).available_bytes == 15 << 30
    with pytest.raises(MemoryError): controller.host_admission((15 << 30)+1)
    monkeypatch.setattr(os_memory, 'host_memory_snapshot', lambda: None)
    with pytest.raises(MemoryError): controller.host_admission(9 << 30)


def test_sealed_bool_row_does_not_bypass_integer_shape_validation(tmp_path, monkeypatch):
    controller = load('probe_ds41_native_ffn')
    monkeypatch.setenv('MLX_ENABLE_TF32', '0')
    spec = sealed_spec(controller, tmp_path)
    spec['native_ffn_options']['tile'][0] = True
    with pytest.raises(ValueError, match='bounded native FFN'):
        controller.validate_spec(spec, tmp_path)


@pytest.mark.parametrize('failure', [None, 'numerical', 'live', 'cleanup', 'peak'])
def test_child_control_flow_and_cleanup_are_distinct_cpu(tmp_path, monkeypatch, failure):
    controller = load('probe_ds41_native_ffn')
    monkeypatch.setenv('MLX_ENABLE_TF32', '0')
    spec = sealed_spec(controller, tmp_path)
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    monkeypatch.setattr(controller, 'check_ownership', lambda *args: {'fixture_ownership': True})
    monkeypatch.setattr(controller, 'source_manifest', lambda *args: {})
    monkeypatch.setattr(controller, 'digest', lambda *args: 'expected')
    floors = []
    def host(minimum):
        floors.append(minimum)
        if failure == 'live' and minimum == 9 << 30: raise MemoryError('live reserve refused')
        return SimpleNamespace(available_bytes=100 << 30)
    monkeypatch.setattr(controller, 'host_admission', host)
    environment = ModuleType('probe_deepseek_v4_decode_compare')
    environment.HID_PROFILE = 'fixture'
    environment.sample_environment = lambda *args: {'foreign_model_processes': []}
    monkeypatch.setitem(sys.modules, 'probe_deepseek_v4_decode_compare', environment)
    cleanup = []
    mx = ModuleType('mlx.core')
    mx.gpu = 'fixture'
    mx.set_default_device = lambda value: None
    mx.set_memory_limit = lambda value: cleanup.append(('limit', value))
    mx.set_cache_limit = lambda value: None
    mx.get_peak_memory = lambda: 7 << 30 if failure == 'peak' else 0
    def synchronize():
        cleanup.append(('synchronize', True))
        if failure == 'cleanup': raise RuntimeError('fixture fence failed')
    mx.synchronize = synchronize
    mx.clear_cache = lambda: cleanup.append(('clear', True))
    package = ModuleType('mlx')
    package.core = mx
    monkeypatch.setitem(sys.modules, 'mlx', package)
    monkeypatch.setitem(sys.modules, 'mlx.core', mx)
    helper = ModuleType('ds41_native_ffn_components')
    def run(mx, np, capture, out, report, compare, atomic):
        assert capture == 'fixture-capture' and report['native_ffn_options']['repeats'] == 7
        atomic(out/'helper.json', {'fixture': True})
        return dict(complete=True, harness_completed=True, required_reference_controls_exact=True,
            all_native_frontiers_exact=False, verdicts={'stock_capture_replay': True, 'coefficient_identity': True, 'native_eligibility': True, 'finite_stable': failure != 'numerical', 'actual_fp64_bounds': True, 'zero_aba': True, 'native_stage_dispatched': True})
    helper.run_components = run
    monkeypatch.setitem(sys.modules, 'ds41_native_ffn_components', helper)
    comparison = ModuleType('probe_deepseek_v41_source_math')
    comparison.compare = lambda *args: None
    monkeypatch.setitem(sys.modules, 'probe_deepseek_v41_source_math', comparison)
    assert controller.component(tmp_path) == (0 if failure is None else 1)
    result = json.loads((tmp_path/'results.json').read_text())
    assert ('synchronize', True) in cleanup and ('limit', 6 << 30) in cleanup
    assert floors[0] == 15 << 30 and 9 << 30 in floors
    if failure == 'numerical':
        assert result['methodological_passed'] is True and result['engineering_passed'] is False and result['numerical_passed'] is None
        assert 'error' not in result
    if failure == 'cleanup': assert result['cleanup_error']['type'] == 'RuntimeError'
    if failure is None:
        assert result['native_frontiers_exact'] is False and result['passed'] is True
        assert result['numerical_passed'] is None  # Different reduction is not automatic quality failure.


def test_cli_refuses_unknown_or_existing_shapes_before_admission(tmp_path, monkeypatch):
    controller = load('probe_ds41_native_ffn')
    with pytest.raises(SystemExit): controller.main(['component', '--output', str(tmp_path/'out')])
    with pytest.raises(SystemExit): controller.main(['unit', '--capture-unit', 'unused', '--output', str(tmp_path/'out')])
    def forbidden(*args): raise AssertionError('admission touched for existing evidence')
    monkeypatch.setattr(controller, 'host_admission', forbidden)
    with pytest.raises(FileExistsError):
        controller.main(['component', '--capture-unit', 'unused', '--output', str(tmp_path)])


def test_source_freeze_mismatch_writes_cpu_failure_receipt(tmp_path, monkeypatch):
    controller = load('probe_ds41_native_ffn')
    root = tmp_path/'root'
    (root/'src').mkdir(parents=True)
    (root/'src/test.py').write_text('original')
    out = tmp_path/'unit'
    out.mkdir()
    monkeypatch.setattr(controller, 'ROOT', root)
    monkeypatch.setattr(controller, 'source_manifest', lambda path: {'src/test.py': hashlib.sha256((path/'src/test.py').read_bytes()).hexdigest()})
    original_copy = controller.shutil.copyfile
    def copy(source, target):
        original_copy(source, target)
        target.write_text('changed')
    monkeypatch.setattr(controller.shutil, 'copyfile', copy)
    with pytest.raises(RuntimeError, match='freeze inconsistent'):
        controller.freeze_source(out)
    failure = json.loads((out/'source-freeze-failure.json').read_text())
    assert failure['gpu_lease_acquired'] is False and failure['changed_files'] == ['src/test.py']
