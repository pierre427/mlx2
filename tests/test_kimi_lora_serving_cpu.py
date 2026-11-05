"""Host-only lifecycle oracles; never starts a model server or uses the GPU."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('kimi_lora_serving_tests', ROOT / 'scripts/research/kimi_lora_serving.py')
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)

PROGRAM = {'program_id': 'sha256:' + 'b' * 64, 'checkpoint': 'Model@' + 'a' * 64,
           'source_revision': 'c' * 40, 'construction': 'synthetic',
           'numerical_policy': 'unmerged_float32_factors', 'approximate': False}
POLICY = {'mla': 'auto', 'kda': 'kernel'}


def status(active=None, revision=0, applications=0):
    return {'model': 'Model', 'artifact': 'a' * 64, 'base_artifact': 'a' * 64,
            'runtime': {'source_sha256': 'source', 'mlx_native_sha256': 'native'},
            'build_runtime': {'source_sha256': 'full-source'}, 'profile': 'ordinary-profile',
            'apcv2': {'version': 2, 'layout_name': 'kimi-layout'},
            'settings': {'route': 'ordinary', 'max_lanes': 1, 'max_inflight': 1},
            'execution': {'route': 'ordinary', 'policy': POLICY, 'artifact': {'checkpoint': 'hub@rev'}},
            'healthy': True, 'inflight': 0, 'queue_depth': 0, 'accepted_lifecycles': 0,
            'quiesce': {'terminal_cleanup_pending': 0}, 'model_revision': revision,
            'lora': {'enabled': True, 'active': active, **({'weight_program': {
                **PROGRAM, 'implemented': True, 'qualified': False, 'selected': True,
                'observed_used': applications > 0, 'adapter_delta_applications': applications}} if active else {})},
            'multi_lora': {'enabled': False}, 'recent_operation_receipts': []}


def response(cached=0, *, active=False, tokens=None):
    tokens = [7, 11, 13] if tokens is None else tokens
    return {'choices': [{'message': {'content': 'KL-7319'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 64, 'completion_tokens': len(tokens), 'prompt_tokens_details': {'cached_tokens': cached}},
            'mlx2': {'cache': 'apcv2', 'route': 'ordinary', 'cached_tokens': cached, 'qualification': 'unqualified',
                     **({'weight_program': {**PROGRAM, 'selected': True, 'qualified': False, 'observed_used': True}} if active else {}),
                     'token_trace': {'schema': 'mlx2.generated-token-trace.v1', 'source': 'generation_response',
                         'start_completion_token': 1, 'token_ids': tokens, 'execution_widths': [1] * len(tokens),
                         'ordinary_execution_widths': [1] * len(tokens), 'completion_tokens': len(tokens), 'truncated': False}}}


class FakeClient:
    def __init__(self, *, invalidate=True, count=True, drift=False, restore_drift=False):
        self.active = None
        self.revision = 0
        self.applications = 0
        self.cached = False
        self.operations = []
        self.calls = []
        self.invalidate, self.count = invalidate, count
        self.drift, self.restore_drift = drift, restore_drift

    def call(self, path, body=None, **kwargs):
        self.calls.append((path, body))
        if path == '/v1/status':
            result = status(self.active, self.revision, self.applications)
            result['recent_operation_receipts'] = copy.deepcopy(self.operations)
            return result
        if path in {'/v1/load_lora_adapter', '/v1/unload_lora_adapter'}:
            load = path.endswith('load_lora_adapter') and not path.endswith('unload_lora_adapter')
            self.active = body['lora_name'] if load else None
            self.applications = 0
            if self.invalidate:
                self.cached = False
            self.revision += 1
            self.operations.append({'operation': 'lora_load' if load else 'lora_unload', 'status': 'completed',
                                    'model_revision': self.revision})
            return {'status': 'loaded' if load else 'unloaded', 'model_revision': self.revision}
        assert path == '/v1/chat/completions'
        if self.active and self.count:
            self.applications += 10
        tokens = [7, 11, 13]
        if self.drift and self.cached:
            tokens = [7, 12, 13]
        if self.restore_drift and self.revision == 2:
            tokens = [7, 11, 14]
        result = response(32 if self.cached else 0, active=bool(self.active), tokens=tokens)
        self.cached = True
        return result


def basic_spec():
    return {'job_id': 'fixture', 'artifact_name': 'Model', 'base_fingerprint': 'a' * 64,
            'execution_policy': POLICY, 'adapter': '/fixture/adapter'}


def test_full_lifecycle_requires_real_invalidation_replay_engagement_and_restoration():
    client = FakeClient()
    checks = []
    result = harness.run_lifecycle(client, basic_spec(), PROGRAM, ownership_check=lambda: checks.append(True))
    assert result['passed'] and result['base_restored_exactly']
    assert len(result['arms']) == 3 and len(result['transitions']) == 2
    assert result['arms'][1]['adapter_application_delta'] == 20
    assert len(checks) == 8
    assert [t['revision_after'] for t in result['transitions']] == [1, 2]
    assert client.active is None
    assert all('return_token_trace' in body for path, body in client.calls if path.endswith('completions'))
    assert not any('promote' in path for path, _ in client.calls)


@pytest.mark.parametrize('change', [{'invalidate': False}, {'drift': True}, {'restore_drift': True}])
def test_cache_leakage_replay_drift_or_failed_base_restore_cannot_pass(change):
    result = harness.run_lifecycle(FakeClient(**change), basic_spec(), PROGRAM, ownership_check=lambda: None)
    assert result['passed'] is False
    if change.get('drift'):
        assert result['arms'][0]['replay']['first_divergence_zero_based'] == 1
        assert result['arms'][0]['replay']['classification'] == 'unclassified'


def test_adapter_counter_must_increase_and_failed_response_is_retained():
    record = {}
    with pytest.raises(ValueError, match='counter did not increase'):
        harness.run_lifecycle(FakeClient(count=False), basic_spec(), PROGRAM, ownership_check=lambda: None, record=record)
    assert len(record['arms'][1]['rows']) == 2
    assert record['arms'][1]['rows'][0]['response']['mlx2']['weight_program']


@pytest.mark.parametrize('field,value', [('route', 'prompt_lookup'), ('cache', 'legacy'), ('cached_tokens', 9)])
def test_response_route_and_cache_contracts_are_enforced(field, value):
    result = response()
    result['mlx2'][field] = value
    with pytest.raises(ValueError):
        harness.response_evidence(result, expected='KL-7319')


def test_exact_program_identity_and_usage_are_required():
    for change in ({'program_id': 'wrong'}, {'observed_used': False}, {'qualified': True}, {'source_revision': 'wrong'}):
        result = response(active=True)
        result['mlx2']['weight_program'].update(change)
        with pytest.raises(ValueError):
            harness.response_evidence(result, expected='KL-7319', program=PROGRAM)
    with pytest.raises(ValueError, match='unexpectedly'):
        harness.response_evidence(response(active=True), expected='KL-7319')


def test_copy_permissions_and_basename_are_checked_without_changing_them(tmp_path):
    original = tmp_path / 'original' / 'Model'
    copied = tmp_path / 'copy' / 'Model'
    original.mkdir(parents=True); copied.mkdir(parents=True)
    file = copied / 'config.json'; file.write_text('{}')
    copied.chmod(0o555)
    with pytest.raises(ValueError, match='writable'):
        harness.readonly_manifest(copied)
    file.chmod(0o444)
    manifest = harness.readonly_manifest(copied)
    assert manifest['file_count'] == 1
    assert file.stat().st_mode & 0o222 == 0
    copied.chmod(0o755)
    spec = {**basic_spec(), 'model': str(copied), 'original_model': str(original), 'out': str(tmp_path / 'out'),
            'source_revision': 'c' * 40, 'selected_candidate_receipt': 'selected.json', 'selected_candidate_receipt_sha256': 'd' * 64}
    harness.validate_spec(spec)
    with pytest.raises(ValueError, match='basename'):
        harness.validate_spec({**spec, 'model': str(tmp_path / 'copy' / 'Renamed')})
    with pytest.raises(ValueError, match='separate'):
        harness.validate_spec({**spec, 'model': str(original)})


def test_server_flags_force_ephemeral_single_lane_ordinary_and_no_multilora(tmp_path):
    spec = {**basic_spec(), 'model': str(tmp_path / 'Model')}
    command = harness.server_command(spec, tmp_path / 'serving', 19000, tmp_path / 'admin-token')
    for flag, expected in [('--host', '127.0.0.1'), ('--port', '19000'), ('--max-lanes', '1'),
                           ('--max-inflight', '1'), ('--max-loras', '0'), ('--lora-dir', '/fixture/adapter')]:
        assert command[command.index(flag) + 1] == expected
    assert '--ordinary' in command and '--no-auto-route' in command
    assert '--prompt-lookup' not in command and '--native-mtp' not in command


def test_status_transition_requires_revision_and_completed_operation_receipt():
    before, after = status(), status('fixture-adapter', 1)
    with pytest.raises(ValueError, match='missing revision-bound'):
        harness.verify_transition(before, after, operation='lora_load', active='fixture-adapter', program=PROGRAM)
    after['recent_operation_receipts'] = [{'operation': 'lora_load', 'status': 'completed', 'model_revision': 1}]
    harness.verify_transition(before, after, operation='lora_load', active='fixture-adapter', program=PROGRAM)
    after['execution']['policy'] = {'mla': 'expanded'}
    with pytest.raises(ValueError, match='changed'):
        harness.verify_transition(before, after, operation='lora_load', active='fixture-adapter', program=PROGRAM)


def test_native_active_program_counter_preserves_qualification_states():
    import mlx.core as mx
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        from mlx2.serving import ServingEngine
        absent = SimpleNamespace(lora_session=None)
        assert ServingEngine._active_weight_program_receipt(absent) is None
        session = SimpleNamespace(weight_program=PROGRAM, program_application_count=0)
        engine = SimpleNamespace(lora_session=session)
        first = ServingEngine._active_weight_program_receipt(engine)
        assert first['adapter_delta_applications'] == 0 and first['observed_used'] is False
        assert first['implemented'] is True and first['qualified'] is False and first['selected'] is True
        session.program_application_count = 12
        second = ServingEngine._active_weight_program_receipt(engine)
        assert second['adapter_delta_applications'] == 12 and second['observed_used'] is True
        assert second['qualified'] is False and second['program_id'] == PROGRAM['program_id']
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize('fail_cleanup', [False, True])
def test_main_stops_owned_descendants_and_removes_token_on_failure(tmp_path, monkeypatch, fail_cleanup):
    import sys
    from mlx2 import desktop_campaign_worker, desktop_campaign_process
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import kimi_lora_round
    original = tmp_path / 'original' / 'Model'
    copied = tmp_path / 'copy' / 'Model'
    original.mkdir(parents=True); copied.mkdir(parents=True)
    (copied / 'config.json').write_text('{}'); (copied / 'config.json').chmod(0o444); copied.chmod(0o555)
    out = tmp_path / 'out'; out.mkdir()
    specification = {**basic_spec(), 'model': str(copied), 'original_model': str(original), 'out': str(out),
                     'source_revision': 'c' * 40, 'selected_candidate_receipt': 'selected.json',
                     'selected_candidate_receipt_sha256': 'd' * 64}
    path = tmp_path / 'spec.json'; path.write_text(json.dumps(specification))
    stopped = []
    class FakeServer:
        pid, returncode = 4242, -15
        def stop(self, timeout):
            stopped.append(timeout)
            if fail_cleanup:
                raise RuntimeError('synthetic survivor')
        def live(self):
            return []
    class FakeSocket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): assert address == ('127.0.0.1', 0)
        def getsockname(self): return ('127.0.0.1', 19001)
    monkeypatch.setattr(harness.socket, 'socket', FakeSocket)
    monkeypatch.setattr(desktop_campaign_worker, 'check_ownership', lambda spec: {'synthetic': True})
    monkeypatch.setattr(desktop_campaign_process, 'spawn', lambda *a, **k: FakeServer())
    monkeypatch.setattr(kimi_lora_round, 'candidate_selection', lambda spec: PROGRAM)
    monkeypatch.setattr(harness.subprocess, 'check_output', lambda argv, **kwargs: 'c' * 40 + '\n' if argv[0] == 'git' else 'p4242\n')
    monkeypatch.setattr(harness, 'wait_ready', lambda *a, **k: status())
    def failed_request(*args, **kwargs):
        raise ValueError('synthetic HTTP failure')
    monkeypatch.setattr(harness, 'run_lifecycle', failed_request)
    try:
        assert harness.main(['--spec', str(path)]) == 1
        assert stopped == [20]
        record = json.loads((out / 'serving/receipt.json').read_text())
        assert record['passed'] is False and record['readonly_copy_unchanged'] is True
        assert 'synthetic HTTP failure' in record['error']
        assert not (out / 'serving/admin-token').exists()
        assert bool(record.get('cleanup_error')) is fail_cleanup
    finally:
        copied.chmod(0o755)
