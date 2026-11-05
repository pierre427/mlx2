"""Host-only bound-spec and sampled-resource regression checks."""
import hashlib
import json
from pathlib import Path

import pytest
from test_kimi_lora_serving_cpu import harness, FakeClient, basic_spec, status, PROGRAM


def full_spec(tmp_path):
    original = tmp_path / 'original' / 'Model'
    model = tmp_path / 'copy' / 'Model'
    original.mkdir(parents=True); model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    (model / 'config.json').chmod(0o444); model.chmod(0o555)
    out = tmp_path / 'out'; out.mkdir()
    return {**basic_spec(), 'model': str(model), 'original_model': str(original), 'out': str(out),
            'source_revision': 'c' * 40, 'selected_candidate_receipt': 'selection.json',
            'selected_candidate_receipt_sha256': 'd' * 64}


@pytest.mark.parametrize('field,value', [('max_metal_bytes', 0), ('max_metal_bytes', 101 << 30),
    ('max_metal_bytes', True), ('max_metal_bytes', 1.0), ('system_reserve_bytes', 19 << 30),
    ('system_reserve_bytes', 0), ('system_reserve_bytes', True)])
def test_resource_limits_cannot_be_invalid_or_loosened(tmp_path, field, value):
    spec = full_spec(tmp_path)
    try:
        with pytest.raises(ValueError): harness.validate_spec({**spec, field: value})
    finally:
        Path(spec['model']).chmod(0o755)


def test_resource_sample_has_actual_values_and_explicit_sampling_limit():
    result = harness.resource_evidence({'metal_peak_bytes': 60 << 30}, {}, available_bytes=40 << 30)
    assert result['metal_peak_bytes_last_published'] == 60 << 30
    assert result['host_available_bytes'] == 40 << 30
    assert result['max_metal_bytes'] == 100 << 30 and result['system_reserve_bytes'] == 20 << 30
    assert result['sampling'] == 'lifecycle_boundary_not_continuous'
    harness.resource_evidence({'metal_peak_bytes': 100 << 30}, {}, available_bytes=20 << 30)


@pytest.mark.parametrize('snapshot,available,exception', [({}, 40 << 30, ValueError),
    ({'metal_peak_bytes': True}, 40 << 30, ValueError),
    ({'metal_peak_bytes': 101 << 30}, 40 << 30, MemoryError),
    ({'metal_peak_bytes': 60 << 30}, (20 << 30) - 1, MemoryError)])
def test_missing_or_overbound_resource_sample_refuses(snapshot, available, exception):
    with pytest.raises(exception): harness.resource_evidence(snapshot, {}, available_bytes=available)


def test_every_lifecycle_boundary_invokes_resource_guard():
    stages = []
    def guard(snapshot, stage):
        assert snapshot['healthy']
        stages.append(stage)
    result = harness.run_lifecycle(FakeClient(), basic_spec(), PROGRAM,
                                  ownership_check=lambda: None, resource_check=guard)
    assert result['passed']
    assert stages == ['startup', 'base_before', 'base_cold_after', 'base_warm_after',
                      'adapter_load_after', 'adapter_before', 'adapter_cold_after',
                      'adapter_warm_after', 'adapter_unload_after', 'restored_base_before',
                      'restored_base_cold_after', 'restored_base_warm_after']


def test_resource_guard_failure_keeps_response_before_raising():
    record = {}
    def guard(snapshot, stage):
        if stage == 'adapter_cold_after': raise MemoryError('synthetic bound')
    with pytest.raises(MemoryError, match='synthetic bound'):
        harness.run_lifecycle(FakeClient(), basic_spec(), PROGRAM,
                             ownership_check=lambda: None, resource_check=guard, record=record)
    assert record['arms'][1]['rows'][0]['response']
    assert not record.get('passed')


def test_main_hashes_the_spec_snapshot_it_parsed(tmp_path, monkeypatch):
    import sys
    from mlx2 import desktop_campaign_worker, desktop_campaign_process
    sys.path.insert(0, str(harness.ROOT / 'scripts/research'))
    import kimi_lora_round
    spec = full_spec(tmp_path)
    path = tmp_path / 'spec.json'
    payload = (json.dumps(spec) + '\n').encode()
    path.write_bytes(payload)
    real_read, real_text, calls = Path.read_bytes, Path.read_text, []
    def replacing_read(target):
        value = real_read(target)
        if target == path:
            calls.append(target); target.write_bytes(b'{"different":"spec after snapshot"}\n')
        return value
    def replacing_text(target, *args, **kwargs):
        value = real_text(target, *args, **kwargs)
        if target == path:
            calls.append(target); target.write_bytes(b'{"different":"spec after snapshot"}\n')
        return value
    monkeypatch.setattr(Path, 'read_bytes', replacing_read)
    monkeypatch.setattr(Path, 'read_text', replacing_text)
    stopped = []
    class Server:
        pid, returncode = 4242, -15
        def stop(self, timeout): stopped.append(timeout)
        def live(self): return []
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): pass
        def getsockname(self): return ('127.0.0.1', 19001)
    monkeypatch.setattr(harness.socket, 'socket', Socket)
    monkeypatch.setattr(desktop_campaign_worker, 'check_ownership', lambda value: {'fixture': True})
    monkeypatch.setattr(desktop_campaign_process, 'spawn', lambda *args, **kwargs: Server())
    monkeypatch.setattr(kimi_lora_round, 'candidate_selection', lambda value: PROGRAM)
    monkeypatch.setattr(harness.subprocess, 'check_output', lambda argv, **kwargs: 'c' * 40 + '\n' if argv[0] == 'git' else 'p4242\n')
    monkeypatch.setattr(harness, 'wait_ready', lambda *args, **kwargs: status())
    monkeypatch.setattr(harness.Client, 'call', lambda *args, **kwargs: '# synthetic metrics\n')
    def lifecycle(*args, **kwargs): kwargs['record']['passed'] = True
    monkeypatch.setattr(harness, 'run_lifecycle', lifecycle)
    try:
        assert harness.main(['--spec', str(path)]) == 0
        receipt = json.loads((Path(spec['out']) / 'serving/receipt.json').read_text())
        assert receipt['spec'] == spec
        assert receipt['spec_sha256'] == hashlib.sha256(payload).hexdigest()
        assert calls == [path] and real_read(path) != payload
        assert stopped == [20] and receipt['server_stopped'] is True
        assert not (Path(spec['out']) / 'serving/admin-token').exists()
    finally:
        Path(spec['model']).chmod(0o755)


def test_program_construction_revision_is_distinct_from_serving_revision(tmp_path):
    import sys
    import mlx.core as mx
    previous = mx.default_device(); mx.set_default_device(mx.cpu)
    try:
        import numpy as np
        sys.path.insert(0, str(harness.ROOT / 'scripts/research'))
        import kimi_lora_round
        from mlx2.experimental.weight_space.programs import create_program, export_program
        from mlx2.runtime.lora import read_lora_config, inspect_lora_program, lora_fingerprint
        checkpoint = 'Model@' + 'a' * 64
        program = create_program({'model.layers.0.attention.q_proj':
            (np.ones((4, 2), np.float32), np.ones((2, 4), np.float32))},
            checkpoint=checkpoint, source_revision='b' * 40, construction='synthetic_test', provenance={'test': True})
        path = tmp_path / 'adapter'; export_program(program, path)
        _, config = read_lora_config(path)
        payload = (path / 'adapters.safetensors').read_bytes()
        manifest = inspect_lora_program(path, config, base_checkpoint=checkpoint, payload=payload)
        fingerprint = lora_fingerprint(payload, config['rank'], config['scale'], config['keys'], manifest)
        development = tmp_path / 'dev.json'
        development.write_text(json.dumps({'status': 'completed', 'checkpoint': checkpoint,
            'source_revision': 'b' * 40, 'adapter_fingerprint': fingerprint,
            'spec': {'eval_split': 'dev'}, 'evaluations': {'adapter': {'cases': 1}}}))
        selected = {'schema': 'mlx2.kimi-lora-candidate-selection.v1', 'checkpoint': checkpoint,
            'adapter_path': str(path), 'adapter_fingerprint': fingerprint, 'program_id': program.program_id,
            'source_revision': 'b' * 40,
            'development_receipt': {'path': str(development), 'sha256': harness.sha(development)}}
        selection = tmp_path / 'selection.json'; selection.write_text(json.dumps(selected))
        spec = {'eval_split': 'final', 'artifact_name': 'Model', 'base_fingerprint': 'a' * 64,
            'source_revision': 'd' * 40, 'adapter': str(path), 'selected_candidate_receipt': str(selection),
            'selected_candidate_receipt_sha256': harness.sha(selection)}
        result = kimi_lora_round.candidate_selection(spec)
        assert result['source_revision'] == 'b' * 40 != spec['source_revision']
        # Do not rewrite historical construction identity to the current harness revision.
        selected['source_revision'] = spec['source_revision']; selection.write_text(json.dumps(selected))
        spec['selected_candidate_receipt_sha256'] = harness.sha(selection)
        with pytest.raises(ValueError, match='source identity changed'):
            kimi_lora_round.candidate_selection(spec)
    finally:
        mx.set_default_device(previous)
