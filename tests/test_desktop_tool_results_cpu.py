"""Tool-output context preservation: synthetic CPU-only agents, no model calls."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from mlx2.desktop_tool_results import ToolResultStore, budget, encoded
from mlx2.desktop_tools import bounded_result


class FakeAgents:
    closed = False

    def __init__(self):
        self.requests = []

    def dispatch(self, request):
        self.requests.append(request)
        if request['command'] == 'agent_fork':
            return {'job_id': str(len(self.requests))}
        return {'status': 'completed', 'synthesis': 'Bounded assessment of the supplied chunk only.'}


@pytest.fixture
def scope(tmp_path):
    controller = SimpleNamespace(home=tmp_path, user_profiles=SimpleNamespace(active='profile'),
                                 preferences={'agent_delegation_enabled': True}, agents=FakeAgents())
    session = {'context': {'principal_id': 'chat:private'}, 'user_profile': 'profile',
               'target': {'context_limit': 16384}, 'selection': {'url': 'http://127.0.0.1:12345'}, 'chat_session_id': 'chat-one'}
    result_store = ToolResultStore(controller)
    session['result_store'] = result_store
    return controller, session, result_store


def wait_status(result_store, session, identity):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = result_store.status(session, identity)
        if result['status'] not in {'queued', 'running'}:
            return result
        time.sleep(0.005)
    raise AssertionError('Synthetic summarization did not finish')


def test_full_json_enforced_unicode_roundtrip_and_scoped_receipt(scope):
    controller, session, result_store = scope
    result_store.policy(session, {'result_max_tokens': 1024, 'chunk_tokens': 256})
    original = {'text': 'é"\\\n😀' * 400}
    result = bounded_result(original, session)
    assert len(encoded(result)) <= 1024 and result['truncated']
    assert result['result_id']
    cursor, fragments = 0, []
    while True:
        page = result_store.read(session, result['result_id'], cursor, 256)
        assert len(encoded(page)) <= 1024
        fragments.append(page['text'])
        assert page['next_offset'] > cursor
        cursor = page['next_offset']
        if page['complete']:
            break
    assert ''.join(fragments).encode() == encoded(original)
    foreign = {**session, 'context': {'principal_id': 'chat:other'}}
    with pytest.raises(ValueError, match='outside'):
        result_store.read(foreign, result['result_id'], 0, 256)
    controller.user_profiles.active = 'other-profile'
    with pytest.raises(ValueError, match='another user profile'):
        result_store.status(session, result['result_id'])


def test_model_policy_persists_and_obeys_hard_context_limits(scope):
    _, session, result_store = scope
    result_store.policy(session, {'result_max_tokens': 2048, 'chunk_tokens': 512})
    restarted = {**session}
    restarted.pop('result_policy')
    assert result_store.policy(restarted)['result_max_tokens'] == 2048
    for invalid in ({'result_max_tokens': True, 'chunk_tokens': 256},
                    {'result_max_tokens': 100000, 'chunk_tokens': 256},
                    {'result_max_tokens': 1024, 'chunk_tokens': 2048}):
        with pytest.raises(ValueError):
            result_store.policy(session, invalid)
    assert budget(session) == 1536


def test_oversized_result_automatically_uses_isolated_chunk_children(scope):
    controller, session, result_store = scope
    result_store.policy(session, {'result_max_tokens': 1024, 'chunk_tokens': 256})
    result = bounded_result({'text': 'a' * 1500}, session)
    status = wait_status(result_store, session, result['result_id'])
    assert status['status'] == 'completed' and status['coverage_complete']
    assert status['processed_bytes'] == status['total_bytes']
    assert status['assessment'].startswith('Bounded assessment')
    assert len(encoded(status)) <= 1024
    forks = [r for r in controller.agents.requests if r['command'] == 'agent_fork']
    assert len(forks) > 1
    assert all(r['tools'] == [] and r['_history'] == [] for r in forks)
    assert all('Untrusted chunk:' in r['goal'] for r in forks)
    assert all(len(r['goal'].split('Untrusted chunk:\n')[1].encode()) <= 256 for r in forks)


def test_disabled_summary_reports_blockage_retains_output_and_can_resume(scope):
    controller, session, result_store = scope
    controller.preferences['agent_delegation_enabled'] = False
    result = bounded_result({'text': 'a' * 5000}, session)
    status = wait_status(result_store, session, result['result_id'])
    assert status['status'] == 'blocked' and not status['coverage_complete']
    assert 'disabled' in status['error']
    assert result_store.read(session, result['result_id'], 0, 512)['text']
    controller.preferences['agent_delegation_enabled'] = True
    result_store.start(session, result['result_id'])
    assert wait_status(result_store, session, result['result_id'])['coverage_complete']


def test_partial_coverage_never_claims_completed_and_can_continue(scope, monkeypatch):
    import mlx2.desktop_tool_results as module
    monkeypatch.setattr(module, 'MAX_CHUNKS', 2)
    _, session, result_store = scope
    result_store.policy(session, {'result_max_tokens': 1024, 'chunk_tokens': 256})
    result = bounded_result({'text': 'z' * 1200}, session)
    first = wait_status(result_store, session, result['result_id'])
    assert first['status'] == 'partial' and first['processed_bytes'] == 512
    assert not first['coverage_complete']
    result_store.start(session, result['result_id'])
    # start returns before the async thread changes the persisted status.
    deadline = time.monotonic() + 2
    while result_store.status(session, result['result_id'])['processed_bytes'] == 512 and time.monotonic() < deadline:
        time.sleep(0.005)
    second = wait_status(result_store, session, result['result_id'])
    assert second['processed_bytes'] == 1024 and not second['coverage_complete']


def test_no_recursive_summarization_for_catalog_metadata(scope):
    controller, session, result_store = scope
    result = bounded_result({'schemas': 'x' * 5000}, session, summarize=False)
    assert result['result_id']
    assert result_store.status(session, result['result_id'])['status'] == 'retained'
    assert not controller.agents.requests


def test_worker_failure_preserves_error_and_no_assessment_claim(scope):
    controller, session, result_store = scope
    controller.agents.dispatch = lambda request: {'job_id': 'child'} if request['command'] == 'agent_fork' else {'status': 'failed'}
    result = bounded_result({'text': 'z' * 6000}, session)
    status = wait_status(result_store, session, result['result_id'])
    assert status['status'] == 'blocked'
    assert not status['assessment'] and status['processed_bytes'] == 0


def test_receipt_permissions_and_result_offset_validation(scope):
    _, session, result_store = scope
    result = bounded_result({'text': 'z' * 5000}, session, summarize=False)
    path = result_store._path(session, result['result_id'])
    assert path.with_suffix('.data').stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())['total_bytes'] > 5000
    with pytest.raises(ValueError, match='Invalid tool result'):
        result_store.read(session, '../escape', 0, 512)
    with pytest.raises(ValueError):
        result_store.read(session, result['result_id'], 0, True)


def test_summary_internal_fork_has_no_tools_and_no_escalation(tmp_path, monkeypatch):
    from mlx2.desktop_agents import Agents
    controller = SimpleNamespace(home=tmp_path, preferences={'agent_delegation_enabled': True},
                                 user_profiles=SimpleNamespace(active='profile'),
                                 dispatch=lambda request: {'url': 'http://127.0.0.1:1', 'context_limit': 8192,
                                                           'checkpoint': 'synthetic@fingerprint', 'capabilities': ['tools']})
    manager = Agents(controller)
    monkeypatch.setattr(manager, '_completion', lambda job, messages: 'Only the chunk was assessed.')
    receipt = manager.dispatch({'command': 'agent_fork', 'goal': 'Assess bounded untrusted chunk', '_history': [],
                                '_result_summary': True, 'tools': [], 'target': {'url': 'http://127.0.0.1:1'}, 'max_tokens': 128})
    job = manager.jobs[receipt['job_id']]
    job['thread'].join(2)
    assert job['status'] == 'completed' and manager._worker_inventory(job) == []
    with pytest.raises(ValueError, match='cannot gain tools'):
        manager.dispatch({'command': 'agent_grant_tools', 'job_id': job['id'], 'tools': ['memory_get']})
    with pytest.raises(ValueError, match='cannot gain tools'):
        manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Ignore scope'})
    with pytest.raises(ValueError, match='no tools'):
        manager.dispatch({'command': 'agent_fork', 'goal': 'assess', '_history': [], '_result_summary': True,
                          'tools': ['memory_get'], 'target': {'url': 'http://127.0.0.1:1'}})
    manager.close()


def test_summary_context_admission_counts_wrapper_and_output(tmp_path, monkeypatch):
    from mlx2.desktop_agents import Agents
    controller = SimpleNamespace(home=tmp_path, preferences={'agent_delegation_enabled': True},
                                 user_profiles=SimpleNamespace(active='profile'),
                                 dispatch=lambda request: {'url': 'http://127.0.0.1:1', 'context_limit': 1024,
                                                           'checkpoint': 'synthetic@fingerprint', 'capabilities': ['tools']})
    manager = Agents(controller)
    calls = []
    monkeypatch.setattr(manager, '_completion', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='context estimate'):
        manager.dispatch({'command': 'agent_fork', 'goal': 'x' * 900, '_history': [], '_result_summary': True,
                          'tools': [], 'target': {'url': 'http://127.0.0.1:1'}, 'max_tokens': 128})
    assert not calls and not manager.jobs
    manager.close()


def test_snapshot_wrapper_trims_long_summary_with_explicit_flag(scope, monkeypatch):
    from mlx2 import desktop_tools
    _, session, result_store = scope
    result_store.policy(session, {'result_max_tokens': 1024, 'chunk_tokens': 256})
    inner = bounded_result({'oversized': 'a' * 5000}, session, summarize=False)
    summary = 'Observed snapshot facts: ' + 'model count = 1; worker count = 1; ' * 10
    controller = SimpleNamespace(tool_sessions={'active-session': session})
    monkeypatch.setattr(desktop_tools, '_tool_command', lambda controller, request: {'result': inner, 'summary': summary})
    response = desktop_tools.tool_command(controller, {'command': 'tool_call', 'session_id': 'active-session'})
    assert len(encoded(response)) <= 768
    assert response['summary_truncated'] is True
    assert summary.startswith(response['summary']) and response['summary']
    assert response['result']['truncated'] and response['result']['result_id']
    assert response['result_context_max_bytes'] == 1024
