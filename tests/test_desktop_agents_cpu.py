"""Agent isolation/bounds contracts, using synthetic model completions only."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2 import desktop_agents as agents


def controller(tmp_path):
    return SimpleNamespace(home=tmp_path, preferences={}, dispatch=lambda request: {"url": "http://127.0.0.1:9999", "api_key": "private-key", "checkpoint": "synthetic@fingerprint", "source_revision": "synthetic-source"})


def delegate(manager, **kw):
    return manager.dispatch({"command": "agent_delegate", "target": {"url": "http://127.0.0.1:9999", "pid": 1}, "goals": ["goal alpha", "goal beta"], **kw})


def test_independent_workers_and_separate_synthesis(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    calls = []
    lock = threading.Lock()
    def completion(job, messages):
        with lock:
            calls.append(messages)
        if messages[0]["content"].startswith("You are the orchestrator"):
            assert "goal alpha" in messages[2]["content"] and "goal beta" in messages[2]["content"]
            return "Synthesis with uncertainty."
        assert len(messages) == 2
        assert messages[1]["content"] in {"goal alpha", "goal beta"}
        return "Independent report for " + messages[1]["content"]
    monkeypatch.setattr(manager, "_completion", completion)
    result = delegate(manager, concurrency=2)
    job = manager.jobs[result["job_id"]]
    job["thread"].join(3)
    assert job["status"] == "completed"
    assert len(calls) == 3
    assert job["synthesis"] == "Synthesis with uncertainty."
    assert len(set(result["worker_ids"])) == 2
    assert all(w["report"]["task_id"] == job["id"] for w in job["workers"])
    assert all(w["report"]["target_identity"]["checkpoint"] == "synthetic@fingerprint" for w in job["workers"])
    receipt = json.loads((Path(job["workspace"]) / "receipt.json").read_text())
    assert "private-key" not in json.dumps(receipt)
    assert "private-key" not in json.dumps(manager.snapshot())


@pytest.mark.parametrize("changes", [{"goals": []}, {"goals": ["x"] * 17}, {"concurrency": 5}, {"concurrency": True}, {"max_tokens": 999999}, {"target": {"url": "x", "api_key": "injected"}}])
def test_delegate_refuses_unbounded_or_injected_arguments(tmp_path, changes):
    manager = agents.Agents(controller(tmp_path))
    with pytest.raises(ValueError):
        delegate(manager, **changes)
    assert not manager.jobs


def test_worker_failure_preserves_reports_without_false_synthesis(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    def completion(job, messages):
        if messages[1]["content"] == "goal beta":
            raise ValueError("private-key")
        return "successful report"
    monkeypatch.setattr(manager, "_completion", completion)
    result = delegate(manager)
    job = manager.jobs[result["job_id"]]
    job["thread"].join(3)
    assert job["status"] == "failed" and job["synthesis"] is None
    assert [w["status"] for w in job["workers"]] == ["completed", "failed"]
    assert "private-key" not in json.dumps(manager.snapshot())


def test_engine_failure_retains_safe_http_diagnostic_without_model_fallback(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    selected = []

    def completion(job, messages):
        selected.append(job['target']['checkpoint'])
        raise agents.EngineCompletionFailure(500)

    monkeypatch.setattr(manager, '_completion', completion)
    result = delegate(manager, goals=['Report ok'])
    job = manager.jobs[result['job_id']]
    job['thread'].join(3)
    assert job['status'] == 'failed' and selected == ['synthetic@fingerprint']
    failure = job['workers'][0]['failure']
    assert failure['http_status'] == 500
    assert failure['kind'] == 'engine_completion_rejected'
    assert 'Do not select a different model' in failure['guidance']
    assert 'private-key' not in json.dumps(manager.snapshot())


def test_cancellation_blocks_synthesis_and_new_work_while_active(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    def completion(job, messages):
        entered.set()
        release.wait(2)
        if job["cancel"].is_set():
            raise InterruptedError()
        return "report"
    monkeypatch.setattr(manager, "_completion", completion)
    result = delegate(manager, concurrency=1)
    assert entered.wait(1)
    with pytest.raises(ValueError, match="active"):
        delegate(manager)
    manager.dispatch({"command": "agent_cancel", "job_id": result["job_id"]})
    release.set()
    job = manager.jobs[result["job_id"]]
    job["thread"].join(3)
    assert job["status"] == "cancelled"
    assert job["synthesis"] is None
    assert all(w["status"] == "cancelled" for w in job["workers"])


def test_planning_records_claims_without_auto_verification_and_restart(tmp_path, monkeypatch):
    from mlx2.desktop_planning import Planning
    control = controller(tmp_path)
    control.planning = Planning(control)
    manager = agents.Agents(control)
    monkeypatch.setattr(manager, "_completion", lambda job, messages: "synthetic conclusion")
    result = delegate(manager)
    job = manager.jobs[result["job_id"]]
    job["thread"].join(3)
    assert job["status"] == "completed"
    graph = control.planning.dispatch({"command": "plan_get", "session_id": job["planning_session_id"]})
    assert all(task["status"] == "claimed" and task["verification"] is None for task in graph["tasks"])
    restored = agents.Agents(control)
    assert restored.dispatch({"command": "agent_result", "job_id": job["id"]})["synthesis"] == "synthetic conclusion"
    control.planning.close()


def test_actual_scoped_sibling_mailbox_tools_and_capsule_receipts(tmp_path, monkeypatch):
    from mlx2.desktop_memory import MemoryController
    control = controller(tmp_path)
    original_dispatch = control.dispatch
    control.dispatch = lambda request: {**original_dispatch(request), "capabilities": ["tools"]}
    control.memory = MemoryController(control)
    manager = agents.Agents(control)
    posted = threading.Barrier(2)
    received = {}
    def request(job, messages, tools=None):
        assert {d["function"]["name"] for d in tools} == {"memory_search", "memory_get", "message_post", "message_read", "message_ack", "agent_request_tools"}
        goal = messages[1]["content"]
        assert sum(m["role"] == "user" for m in messages) == 1
        worker = next(w for w in job["workers"] if w["goal"] == goal)
        sibling = next(w for w in job["workers"] if w["id"] != worker["id"])
        responses = [m for m in messages if m["role"] == "tool"]
        if not responses:
            function, arguments = "message_post", {"session_id": job["id"], "recipient": sibling["id"], "body": "Observation from " + goal,
                "capsule": {"id": "synthetic-capsule", "revision": 2, "engaged": True}}
        elif len(responses) == 1:
            posted.wait(2)
            function, arguments = "message_read", {"session_id": job["id"], "limit": 5}
        else:
            mailbox = json.loads(responses[-1]["content"])["messages"]
            received[goal] = mailbox
            assert any(m["sender"] == sibling["id"] and m["recipient"] == worker["id"] for m in mailbox)
            return {"content": "Report after explicitly reading scoped sibling message", "_output_tokens": 10}
        return {"content": "", "_output_tokens": 10, "tool_calls": [{"id": worker["id"] + str(len(responses)), "type": "function",
            "function": {"name": function, "arguments": json.dumps(arguments)}}]}
    monkeypatch.setattr(manager, "_request", request)
    monkeypatch.setattr(manager, "_completion", lambda job, messages: "Synthesis of reports")
    result = delegate(manager, concurrency=2, tools=["memory_search", "memory_get", "message_post", "message_read", "message_ack"])
    job = manager.jobs[result["job_id"]]
    job["thread"].join(4)
    assert job["status"] == "completed", job
    assert set(received) == {"goal alpha", "goal beta"}
    assert all(w["tool_receipts"][0]["capsule"]["engaged"] is True for w in job["workers"])
    assert all(w["report"]["tool_calls"][0]["name"] == "message_post" for w in job["workers"])
    control.memory.close()


def fork(manager, history=None, **changes):
    return manager.dispatch({'command': 'agent_fork', 'goal': 'Explore one alternative',
                             '_history': history if history is not None else [{'role': 'user', 'content': 'Private parent state'}],
                             'parent_id': 'parent-conversation', 'target': {'url': 'http://127.0.0.1:9999', 'pid': 1}, **changes})


def test_agent_fork_seed_isolated_persisted_metadata_not_history(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    seed = [{'role': 'system', 'content': 'Parent preferences'}, {'role': 'user', 'content': [
        {'type': 'text', 'text': 'Private seed content'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,c3ludGhldGlj'}}]}]
    captured = []
    def completion(job, messages):
        captured.append(messages)
        assert messages[1]['content'].startswith('Parent system prompt (reference data)')
        assert messages[2]['content'][1]['type'] == 'image_url'  # actual structured media survives seeding
        assert messages[-1]['content'] == 'Explore one alternative'
        messages[2]['content'][0]['text'] = 'Child-only mutation'
        return 'Independent child report'
    monkeypatch.setattr(manager, '_completion', completion)
    result = fork(manager, seed)
    job = manager.jobs[result['job_id']]
    job['thread'].join(3)
    assert job['status'] == 'completed' and len(captured) == 1  # no redundant synthesis inference
    assert seed[1]['content'][0]['text'] == 'Private seed content'
    assert result['branch_id'] == job['id'] and result['parent_id'] == 'parent-conversation'
    assert job['workers'][0]['report']['branch_id'] == job['id']
    receipt = json.loads((Path(job['workspace']) / 'receipt.json').read_text())
    assert '_history' not in receipt and 'Private seed content' not in json.dumps(receipt)
    assert 'Private seed content' not in json.dumps(manager.snapshot())
    assert '_history' not in job
    assert agents.Agents(controller(tmp_path)).jobs[job['id']]['branch_id'] == job['id']


@pytest.mark.parametrize('seed', [[], [{'role': 'assistant', 'tool_calls': [{'id': 'pending'}]}],
                                 [{'role': 'tool', 'tool_call_id': 'unknown', 'content': '{}'}],
                                 [{'role': 'user', 'content': 'x'}] * 257])
def test_agent_fork_rejects_invalid_or_incomplete_context(tmp_path, seed):
    manager = agents.Agents(controller(tmp_path))
    with pytest.raises(ValueError):
        fork(manager, seed)
    assert manager.jobs == {}


def test_agent_fork_four_active_limit_and_delegate_exclusivity(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    release = threading.Event()
    monkeypatch.setattr(manager, '_completion', lambda job, messages: (release.wait(3), 'Bounded report')[1])
    results = [fork(manager) for _ in range(4)]
    try:
        with pytest.raises(ValueError, match='four'):
            fork(manager)
        with pytest.raises(ValueError, match='active'):
            delegate(manager)
        assert all(manager.jobs[row['job_id']]['concurrency'] == 1 for row in results)
    finally:
        release.set()
        manager.close()


def test_agent_fork_refuses_context_overflow_before_start(tmp_path):
    control = controller(tmp_path)
    control.dispatch = lambda request: {'url': 'http://127.0.0.1:9999', 'context_limit': 1024, 'output_tokens': 128}
    manager = agents.Agents(control)
    with pytest.raises(ValueError, match='context estimate'):
        fork(manager, [{'role': 'user', 'content': 'x' * 2000}], max_tokens=128)
    assert not manager.jobs
    assert not (tmp_path / 'agent-tasks').exists()


@pytest.mark.parametrize('writes,expected', [(False, {'skills_list', 'skill_view'}), (True, {'skills_list', 'skill_view', 'skill_manage'})])
def test_worker_skills_policy_exact_preadmission_and_no_operator_authority(tmp_path, monkeypatch, writes, expected):
    control = controller(tmp_path)
    control.user_profiles = SimpleNamespace(active='alice')
    control.dispatch = lambda request: {'url': 'http://127.0.0.1:9999', 'capabilities': ['tools'], 'context_limit': 100000}
    invocations = []
    control.skills = SimpleNamespace(policy=lambda **kw: {'enabled': True, 'autonomous_writes': writes},
        dispatch=lambda request, **kw: invocations.append((request, kw)) or {'skills': []})
    manager = agents.Agents(control)
    admissions = []
    original_admit = manager._admit
    def admit(job, messages, tools=None):
        admissions.append((json.loads(json.dumps(messages)), tools))
        return original_admit(job, messages, tools)
    monkeypatch.setattr(manager, '_admit', admit)
    def request(job, messages, tools=None):
        assert {d['function']['name'] for d in tools} == expected | {'agent_request_tools'}
        if len(messages) == 3:
            assert messages == admissions[0][0]
            assert tools == admissions[0][1]
            return {'content': '', '_output_tokens': 10, 'tool_calls': [
                {'id': 'read-skills', 'type': 'function', 'function': {'name': 'skills_list', 'arguments': '{}'}}]}
        return {'content': 'Evidence-bound report', '_output_tokens': 10}
    monkeypatch.setattr(manager, '_request', request)
    result = fork(manager, tools=sorted(expected))
    job = manager.jobs[result['job_id']]
    job['thread'].join(2)
    assert job['status'] == 'completed'
    assert invocations == [({'command': 'skills_list'}, {'operator': False, 'profile_id': 'alice'})]


def test_profile_change_blocks_worker_tool_write_and_hides_reports(tmp_path, monkeypatch):
    control = controller(tmp_path)
    control.user_profiles = SimpleNamespace(active='alice')
    control.dispatch = lambda request: {'url': 'http://127.0.0.1:9999', 'capabilities': ['tools']}
    control.skills = SimpleNamespace(policy=lambda **kw: {'enabled': True, 'autonomous_writes': True},
        dispatch=lambda *args, **kwargs: pytest.fail('No writes after profile change'))
    manager = agents.Agents(control)
    def request(job, messages, tools=None):
        control.user_profiles.active = 'bob'
        return {'content': '', '_output_tokens': 1, 'tool_calls': [{'id': 'write', 'function': {
            'name': 'skill_manage', 'arguments': json.dumps({'action': 'archive', 'name': 'test', 'expected_revision': 1})}}]}
    monkeypatch.setattr(manager, '_request', request)
    result = fork(manager, tools=['skill_manage'])
    job = manager.jobs[result['job_id']]
    job['thread'].join(2)
    assert job['status'] == 'cancelled'
    assert manager.snapshot()['jobs'] == []
    for command in ('agent_result', 'agent_cancel'):
        with pytest.raises(ValueError, match='Unknown'):
            manager.dispatch({'command': command, 'job_id': job['id']})
    manager.close()


def test_child_requests_do_not_grant_tools_and_parent_can_revive_exact_continuation(tmp_path, monkeypatch):
    control = controller(tmp_path)
    control.dispatch = lambda request: {'url': 'http://127.0.0.1:9999', 'checkpoint': 'exact@one', 'capabilities': ['tools'], 'context_limit': 100000}
    accessed = []
    control.memory = SimpleNamespace(dispatch=lambda request, **kw: accessed.append(request) or {'note': 'reviewed fact'})
    manager = agents.Agents(control)
    seen = []
    def request(job, messages, tools=None):
        names = {d['function']['name'] for d in tools}
        seen.append((names, json.loads(json.dumps(messages))))
        latest_user = max(i for i, message in enumerate(messages) if message['role'] == 'user')
        if not any(message['role'] == 'tool' for message in messages[latest_user + 1:]):
            name = 'memory_get' if 'memory_get' in names else 'agent_request_tools'
            args = {'id': 'fact'} if name == 'memory_get' else {'tools': ['memory_get'], 'reason': 'Need a reviewed fact'}
            return {'content': '', '_output_tokens': 1, 'tool_calls': [{'id': 'call-' + str(len(seen)), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}
        return {'content': 'Private child report', '_output_tokens': 1}
    monkeypatch.setattr(manager, '_request', request)
    result = fork(manager)
    job = manager.jobs[result['job_id']]
    job['thread'].join(2)
    assert job['status'] == 'completed'
    assert seen[0][0] == {'agent_request_tools'}
    assert job['tools'] == [] and not accessed
    assert job['tool_requests'][0]['tools'] == ['memory_get']
    with pytest.raises(ValueError, match='requested'):
        manager.dispatch({'command': 'agent_grant_tools', 'job_id': job['id'], 'tools': ['skill_manage']})
    manager.dispatch({'command': 'agent_grant_tools', 'job_id': job['id'], 'tools': ['memory_get']})
    manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Continue using the reviewed fact'})
    job['thread'].join(2)
    assert job['status'] == 'completed'
    assert seen[2][0] == {'agent_request_tools', 'memory_get'}
    assert any(message.get('content') == 'Private child report' for message in seen[2][1])
    assert accessed == [{'command': 'memory_get', 'id': 'fact'}]
    receipt = (Path(job['workspace']) / 'receipt.json').read_text()
    assert 'Private parent state' not in receipt
    private = Path(job['workspace']) / 'continuation.json'
    assert private.stat().st_mode & 0o777 == 0o600
    control.dispatch = lambda request: {'url': 'http://127.0.0.1:9999', 'checkpoint': 'different@two'}
    with pytest.raises(ValueError, match='identity changed'):
        manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Changed engine'})


def test_omitted_child_grants_hide_memory_and_reject_ungranted_execution(tmp_path, monkeypatch):
    control = controller(tmp_path)
    control.dispatch = lambda request: {'url': 'http://127.0.0.1:9999', 'capabilities': ['tools']}
    control.memory = SimpleNamespace(dispatch=lambda *args, **kwargs: pytest.fail('Unadmitted tool executed'))
    manager = agents.Agents(control)
    def request(job, messages, tools=None):
        assert [tool['function']['name'] for tool in tools] == ['agent_request_tools']
        return {'content': '', '_output_tokens': 1, 'tool_calls': [{'id': 'bad', 'function': {'name': 'memory_get', 'arguments': '{}'}}]}
    monkeypatch.setattr(manager, '_request', request)
    result = fork(manager)
    job = manager.jobs[result['job_id']]
    job['thread'].join(2)
    assert job['status'] == 'failed'


def test_resume_budget_rejection_preserves_completed_report(tmp_path, monkeypatch):
    control = controller(tmp_path)
    target = {'url': 'http://127.0.0.1:9999', 'checkpoint': 'exact@one', 'runtime': {'source_sha256': 'same-runtime'}, 'context_limit': 100000}
    control.dispatch = lambda request: dict(target)
    manager = agents.Agents(control)
    monkeypatch.setattr(manager, '_completion', lambda job, messages: 'Preserved report')
    result = fork(manager)
    job = manager.jobs[result['job_id']]
    job['thread'].join(2)
    before = json.loads(json.dumps(job['workers'][0]))
    target['context_limit'] = 32
    with pytest.raises(ValueError, match='budget'):
        manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Cannot fit'})
    assert job['status'] == 'completed'
    assert job['synthesis'] == 'Preserved report'
    assert job['workers'][0] == before
    target['context_limit'] = 100000
    target['runtime']['source_sha256'] = 'changed-dirty-source'
    with pytest.raises(ValueError, match='identity changed'):
        manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Cannot reuse another runtime'})


def test_oversized_new_history_invalidates_prior_continuation(tmp_path, monkeypatch):
    control = controller(tmp_path)
    manager = agents.Agents(control)
    monkeypatch.setattr(manager, '_completion', lambda job, messages: 'First completed report')
    result = fork(manager)
    job = manager.jobs[result['job_id']]
    job['thread'].join(2)
    path = Path(job['workspace']) / 'continuation.json'
    assert path.is_file() and job['continuation_available']
    monkeypatch.setattr(agents, 'LIMIT', 5000)
    monkeypatch.setattr(manager, '_completion', lambda job, messages: 'x' * 10000)
    manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Larger completion'})
    job['thread'].join(2)
    assert job['status'] == 'completed'
    assert not path.exists() and not job['continuation_available']
    with pytest.raises(ValueError, match='continuation'):
        manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Must not use stale history'})


def test_failed_resume_cannot_reuse_history_from_before_that_attempt(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    monkeypatch.setattr(manager, '_completion', lambda job, messages: 'First report')
    result = fork(manager)
    job = manager.jobs[result['job_id']]
    job['thread'].join(3)
    path = Path(job['workspace']) / 'continuation.json'
    assert path.is_file()

    def fail(job, messages):
        raise agents.EngineCompletionFailure(500)

    monkeypatch.setattr(manager, '_completion', fail)
    manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Second attempt'})
    job['thread'].join(3)
    assert job['status'] == 'failed'
    assert not path.exists()
    assert job['continuation_available'] is False
    with pytest.raises(ValueError, match='continuation'):
        manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Continue second attempt'})


def test_resume_waits_for_previous_orchestrator_cleanup(tmp_path, monkeypatch):
    host = controller(tmp_path)
    manager = agents.Agents(host)
    generating, finish_generation = threading.Event(), threading.Event()
    cleaning, finish_cleanup = threading.Event(), threading.Event()

    def completion(job, messages):
        generating.set()
        assert finish_generation.wait(3)
        return 'Completed report'

    original_time = agents.time.time

    def cleanup_clock():
        # Pause the terminal orchestrator before its final locked save. This
        # seam exists without unpublished research-plan callback extensions.
        if (threading.current_thread() is job.get('thread')
                and job['status'] == 'completed'):
            cleaning.set()
            assert finish_cleanup.wait(3)
        return original_time()

    monkeypatch.setattr(manager, '_completion', completion)
    result = fork(manager)
    job = manager.jobs[result['job_id']]
    old_thread = job['thread']
    try:
        assert generating.wait(2)
        monkeypatch.setattr(agents.time, 'time', cleanup_clock)
        finish_generation.set()
        assert cleaning.wait(2)
        assert job['status'] == 'completed'
        with pytest.raises(ValueError, match='finishing'):
            manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Continue'})
        assert job['thread'] is old_thread
    finally:
        finish_generation.set()
        finish_cleanup.set()
        old_thread.join(3)
        job['thread'].join(3)
    manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Continue after cleanup'})
    job['thread'].join(3)
    assert job['status'] == 'completed'


def test_async_completion_is_durable_fenced_and_scoped(tmp_path, monkeypatch):
    host = controller(tmp_path)
    host.user_profiles = SimpleNamespace(active='alice')
    manager = agents.Agents(host)
    entered, release = threading.Event(), threading.Event()
    def completion(job, messages):
        entered.set()
        assert release.wait(3)
        return 'hi'
    monkeypatch.setattr(manager, '_completion', completion)
    receipt = delegate(manager, goals=['say hi'], parent_id='parent-chat', _notify_parent=True, _delegation_key='same-task')
    assert entered.wait(1)
    assert manager.jobs[receipt['job_id']]['status'] == 'running'
    assert manager.snapshot()['deliveries'] == []
    duplicate = delegate(manager, goals=['say hi'], parent_id='parent-chat', _notify_parent=True, _delegation_key='same-task')
    assert duplicate['reused'] and duplicate['job_id'] == receipt['job_id']
    release.set()
    manager.jobs[receipt['job_id']]['thread'].join(3)
    assert len(manager.snapshot()['deliveries']) == 1
    recovered = agents.Agents(host)
    with pytest.raises(ValueError, match='Unknown completion'):
        recovered.dispatch({'command': 'agent_completion_claim', 'job_id': receipt['job_id'], 'parent_id': 'sibling'})
    host.user_profiles.active = 'bob'
    assert recovered.snapshot()['deliveries'] == []
    host.user_profiles.active = 'alice'
    claim = recovered.dispatch({'command': 'agent_completion_claim', 'job_id': receipt['job_id'], 'parent_id': 'parent-chat'})
    assert claim['synthesis'] == 'hi' and claim['status'] == 'completed'
    with pytest.raises(ValueError, match='no longer pending'):
        recovered.dispatch({'command': 'agent_completion_claim', 'job_id': receipt['job_id'], 'parent_id': 'parent-chat'})
    with pytest.raises(ValueError, match='Stale'):
        recovered.dispatch({'command': 'agent_completion_complete', 'job_id': receipt['job_id'], 'parent_id': 'parent-chat', 'delivery_fence': 'bad', 'success': True})
    restarted = agents.Agents(host)
    assert restarted.jobs[receipt['job_id']]['delivery_state'] == 'interrupted'
    assert restarted.snapshot()['deliveries'] == []  # A possibly consumed callback never silently replays.
    recovered.dispatch({'command': 'agent_completion_complete', 'job_id': receipt['job_id'], 'parent_id': 'parent-chat', 'delivery_fence': claim['delivery_fence'], 'success': True})
    assert recovered.snapshot()['deliveries'] == []


def test_failed_delegation_is_reused_across_restart_until_request_changes(tmp_path, monkeypatch):
    host = controller(tmp_path)
    manager = agents.Agents(host)
    def fail(*args):
        raise agents.EngineCompletionFailure(500)
    monkeypatch.setattr(manager, '_completion', fail)
    first = delegate(manager, parent_id='parent', _delegation_key='unchanged', _notify_parent=True)
    manager.jobs[first['job_id']]['thread'].join(3)
    recovered = agents.Agents(host)
    again = delegate(recovered, parent_id='parent', _delegation_key='unchanged', _notify_parent=True)
    assert again['reused'] and again['job_id'] == first['job_id'] and len(recovered.jobs) == 1
    claim = recovered.dispatch({'command': 'agent_completion_claim', 'job_id': first['job_id'], 'parent_id': 'parent'})
    assert claim['status'] == 'failed'
    assert all(w['failure']['http_status'] == 500 for w in claim['workers'])


def test_operator_agent_sessions_cover_running_finished_workers_restart_and_profile_scope(tmp_path, monkeypatch):
    host = controller(tmp_path)
    host.user_profiles = SimpleNamespace(active='alice')
    manager = agents.Agents(host)
    entered, release = threading.Event(), threading.Event()

    def completion(job, messages):
        if messages[0]['content'].startswith('You are the orchestrator'):
            return 'Synthesis'
        entered.set()
        release.wait(3)
        return 'Report: ' + messages[-1]['content']

    monkeypatch.setattr(manager, '_completion', completion)
    receipt = delegate(manager, concurrency=2)
    job = manager.jobs[receipt['job_id']]
    assert entered.wait(2)
    request = {'id': 'native-bridge-request', 'command': 'agent_session', 'job_id': job['id'], 'worker_id': receipt['worker_ids'][0]}
    active = manager.dispatch(request)
    assert active['worker_status'] == 'running'
    assert active['messages'] == [{'role': 'user', 'content': 'goal alpha'}]
    with pytest.raises(ValueError, match='Unknown worker'):
        manager.dispatch({**request, 'worker_id': 'not-a-member'})
    host.user_profiles.active = 'bob'
    with pytest.raises(ValueError, match='Unknown agent'):
        manager.dispatch(request)
    host.user_profiles.active = 'alice'
    release.set()
    job['thread'].join(4)
    page = manager.dispatch({**request, 'limit': 1})
    assert page['next_offset'] == 1 and page['total'] == 2
    assert manager.dispatch({**request, 'offset': 1})['messages'][0]['content'] == 'Report: goal alpha'
    other = manager.dispatch({**request, 'worker_id': receipt['worker_ids'][1]})
    assert other['messages'][0]['content'] == 'goal beta'
    assert 'messages' not in manager.snapshot()['jobs'][0]
    recovered = agents.Agents(host)
    assert recovered.dispatch(request)['messages'] == manager.dispatch(request)['messages']
    for changes in ({'offset': -1}, {'limit': 51}, {'path': '/etc/passwd'}, {'limit': True}):
        with pytest.raises(ValueError):
            recovered.dispatch({**request, **changes})


def test_operator_fork_session_omits_parent_seed_redacts_secrets_and_survives_resume(tmp_path, monkeypatch):
    host = controller(tmp_path)
    manager = agents.Agents(host)
    monkeypatch.setattr(manager, '_completion', lambda job, messages: 'private-key ' + 'x' * 20000)
    receipt = fork(manager, history=[{'role': 'system', 'content': 'PARENT-SECRET-SYSTEM'},
                                     {'role': 'user', 'content': 'PARENT-SECRET-HISTORY'}])
    job = manager.jobs[receipt['job_id']]
    job['thread'].join(3)
    request = {'command': 'agent_session', 'job_id': job['id'], 'worker_id': receipt['worker_ids'][0]}
    session = manager.dispatch(request)
    text = json.dumps(session)
    assert 'PARENT-SECRET' not in text and 'private-key' not in text
    assert session['messages'][0]['content'] == 'Explore one alternative'
    assert session['messages'][1]['truncated']
    assert len(json.dumps(session['messages'][1], ensure_ascii=False).encode()) <= 16384
    monkeypatch.setattr(manager, '_completion', lambda job, messages: 'Follow-up report')
    manager.dispatch({'command': 'agent_resume', 'job_id': job['id'], 'message': 'Continue own work'})
    job['thread'].join(3)
    continued = manager.dispatch(request)['messages']
    assert [row['role'] for row in continued] == ['user', 'assistant', 'user', 'assistant']
    assert continued[-2]['content'] == 'Continue own work' and continued[-1]['content'] == 'Follow-up report'


def test_operator_session_failure_is_durable_without_raw_error_leak(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))

    def fail(job, messages):
        raise ValueError('private-key')

    monkeypatch.setattr(manager, '_completion', fail)
    receipt = delegate(manager, goals=['Attempt bounded work'])
    job = manager.jobs[receipt['job_id']]
    job['thread'].join(3)
    session = manager.dispatch({'command': 'agent_session', 'job_id': job['id']})
    assert session['worker_status'] == 'failed'
    assert session['messages'][-1]['failure']['kind'] == 'worker_execution_failed'
    assert 'private-key' not in json.dumps(session)


def test_operator_session_updates_after_complete_tool_round_without_parent_history(tmp_path, monkeypatch):
    host = controller(tmp_path)
    host.dispatch = lambda request: {'url': 'http://127.0.0.1:9999', 'checkpoint': 'synthetic@fingerprint',
                                     'context_limit': 100000, 'capabilities': ['tools']}
    manager = agents.Agents(host)
    entered, release = threading.Event(), threading.Event()

    def request(job, messages, tools=None):
        if messages[-1]['role'] == 'tool':
            entered.set()
            release.wait(3)
            return {'content': 'Blocked pending parent grant', '_output_tokens': 1}
        return {'content': '', '_output_tokens': 1, 'tool_calls': [{'id': 'request-tool', 'type': 'function',
                'function': {'name': 'agent_request_tools', 'arguments': json.dumps({'tools': ['memory_get'], 'reason': 'Need a fact'})}}]}

    monkeypatch.setattr(manager, '_request', request)
    receipt = fork(manager, history=[{'role': 'user', 'content': 'DO-NOT-DISPLAY-PARENT'}])
    assert entered.wait(2)
    command = {'command': 'agent_session', 'job_id': receipt['job_id']}
    active = manager.dispatch(command)
    assert [row['role'] for row in active['messages']] == ['user', 'assistant', 'tool']
    assert active['messages'][1]['tool_calls'][0]['function']['name'] == 'agent_request_tools'
    assert 'DO-NOT-DISPLAY-PARENT' not in json.dumps(active)
    release.set()
    job = manager.jobs[receipt['job_id']]
    job['thread'].join(3)
    assert manager.dispatch(command)['messages'][-1]['content'] == 'Blocked pending parent grant'
    path = manager._session_path(job, job['workers'][0])
    assert path.stat().st_mode & 0o777 == 0o600


def test_generic_research_worker_reads_sources_before_researching_and_has_bounded_extra_rounds(tmp_path, monkeypatch):
    host = controller(tmp_path)
    host.dispatch = lambda request: {'url': 'http://127.0.0.1:9999', 'checkpoint': 'synthetic@fingerprint',
                                     'capabilities': ['tools'], 'context_limit': 100000}
    network = []

    def web(request):
        network.append(request)
        if request['command'] == 'web_search':
            return {'results': [{'url': 'https://example.com/' + request['query'], 'title': 'Actual source lead'}]}
        return {'url': request['url'], 'status': '200', 'content': 'Fetched source evidence'}

    host.web = SimpleNamespace(dispatch=web)
    manager = agents.Agents(host)
    operations = [('web_search', {'query': 'one'}), ('web_search', {'query': 'do-not-execute'}),
                  ('fetch_page', {'url': 'https://example.com/one'}), ('web_search', {'query': 'two'}),
                  ('fetch_page', {'url': 'https://example.com/two'})]
    rounds = []

    def request(job, messages, tools=None):
        if tools is None:
            return {'content': 'Source-bound synthesis'}
        rounds.append(messages)
        index = len(rounds) - 1
        if index == 2:
            feedback = json.loads(messages[-1]['content'])
            assert feedback['error'] == 'read_sources_before_searching_again' and not feedback['executed']
            assert feedback['next_tool'] == 'fetch_page'
        if index == len(operations):
            return {'content': 'Evidence from https://example.com/one and https://example.com/two', '_output_tokens': 1}
        name, arguments = operations[index]
        return {'content': '', '_output_tokens': 1, 'tool_calls': [{'id': f'call-{index}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}

    monkeypatch.setattr(manager, '_request', request)
    receipt = delegate(manager, goals=['Research the evidence'], tools=['web_search', 'fetch_page'])
    job = manager.jobs[receipt['job_id']]
    job['thread'].join(4)
    assert job['status'] == 'completed', job
    assert job['max_rounds'] == 12 and job['max_tool_calls'] == 16 and job['max_tokens'] == 4096
    assert len(rounds) == 6  # The formerly hardcoded four-round ceiling no longer interrupts research.
    assert [row['command'] for row in network] == ['web_search', 'fetch_page', 'web_search', 'fetch_page']
    assert all(row.get('query') != 'do-not-execute' for row in network)
    assert 'Fetched source evidence' in json.dumps(rounds[-1])


def test_unattempted_completion_returns_to_pending_and_interrupted_has_explicit_retry(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    monkeypatch.setattr(manager, "_completion", lambda job, messages: "report")
    result = manager.dispatch({"command": "agent_fork", "goal": "child", "_history": [{"role": "user", "content": "seed"}], "parent_id": "parent",
                               "target": {"url": "http://127.0.0.1:9999"}, "_notify_parent": True})
    job = manager.jobs[result["job_id"]]
    job["thread"].join(3)
    base = {"job_id": job["id"], "parent_id": "parent"}
    claim = manager.dispatch({"command": "agent_completion_claim", **base})
    done = manager.dispatch({"command": "agent_completion_complete", **base, "delivery_fence": claim["delivery_fence"], "success": False, "attempted": False})
    assert done["delivery_state"] == "pending"
    job["delivery_retry_after"] = 0
    claim = manager.dispatch({"command": "agent_completion_claim", **base})
    manager.dispatch({"command": "agent_completion_complete", **base, "delivery_fence": claim["delivery_fence"], "success": False})
    assert job["delivery_state"] == "interrupted"
    manager.dispatch({"command": "agent_completion_retry", **base})
    assert manager.snapshot()["deliveries"]
    with pytest.raises(ValueError, match="Unknown completion"):
        manager.dispatch({"command": "agent_completion_retry", **base, "parent_id": "other"})


def test_fork_continuation_excludes_inherited_seed_and_keeps_child_turns_on_resume(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    monkeypatch.setattr(manager, "_request", lambda *args, **kwargs: {"content": "child report", "_output_tokens": 5})
    result = manager.dispatch({"command": "agent_fork", "goal": "child goal", "_history": [{"role": "user", "content": "PARENT PRIVATE SEED"}], "parent_id": "parent", "target": {"url": "http://127.0.0.1:9999"}})
    job = manager.jobs[result["job_id"]]
    job["thread"].join(3)
    path = Path(job["workspace"]) / "continuation.json"
    assert job["status"] == "completed"
    assert "PARENT PRIVATE SEED" not in path.read_text()
    manager.dispatch({"command": "agent_resume", "job_id": job["id"], "message": "follow up"})
    job["thread"].join(3)
    text = path.read_text()
    assert "PARENT PRIVATE SEED" not in text
    assert "child goal" in text and "follow up" in text


GRANTED_TARGET = {"url": "http://127.0.0.1:9999", "checkpoint": "c@f", "capabilities": ["tools"], "context_limit": 32768, "output_tokens": 8192}


def granted_controller(tmp_path):
    web_calls = []
    control = SimpleNamespace(home=tmp_path, preferences={"agent_controls": True, "memory_enabled": False, "agent_delegation_enabled": True},
                              user_profiles=SimpleNamespace(active="alice"), dispatch=lambda r: dict(GRANTED_TARGET),
                              web=SimpleNamespace(dispatch=lambda r: web_calls.append(r) or {"content": "x" * r.get("max_chars", 8000), "status": "200"}))
    return agents.Agents(control), web_calls


def test_granted_fetch_page_recomputes_research_limits_and_clamps_result(tmp_path, monkeypatch):
    manager, web_calls = granted_controller(tmp_path)
    def request(job, messages, tools=None):
        names = {t["function"]["name"] for t in tools or []}
        responses = [m for m in messages if m["role"] == "tool"]
        fresh_turn = messages[-1]["role"] == "user"
        if "fetch_page" in names and fresh_turn:
            call = {"name": "fetch_page", "arguments": json.dumps({"url": "https://example.com/big", "max_chars": 200000})}
        elif "fetch_page" not in names and fresh_turn:
            call = {"name": "agent_request_tools", "arguments": json.dumps({"tools": ["fetch_page"], "reason": "need the page"})}
        else:
            return {"content": "report", "_output_tokens": 5}
        return {"content": "", "_output_tokens": 5, "tool_calls": [{"id": "c" + str(len(responses)), "type": "function", "function": call}]}
    monkeypatch.setattr(manager, "_request", request)
    receipt = manager.dispatch({"command": "agent_fork", "goal": "g", "_history": [{"role": "user", "content": "x"}], "parent_id": "conv", "target": {"url": "http://127.0.0.1:9999"}, "tools": []})
    job = manager.jobs[receipt["job_id"]]
    job["thread"].join(3)
    assert job["status"] == "completed" and job["tool_requests"][0]["tools"] == ["fetch_page"]
    manager.dispatch({"command": "agent_grant_tools", "job_id": job["id"], "tools": ["fetch_page"]})
    manager.dispatch({"command": "agent_resume", "job_id": job["id"], "message": "now fetch it"})
    job["thread"].join(3)
    assert job["status"] == "completed"
    assert job["research_worker"] and job["max_rounds"] == 12 and job["max_tool_calls"] == 16
    assert web_calls[0]["max_chars"] == 4096
    assert "failure" not in job["workers"][0]


def test_restore_removes_legacy_continuation_with_inherited_seed(tmp_path, monkeypatch):
    manager = agents.Agents(controller(tmp_path))
    monkeypatch.setattr(manager, '_completion', lambda *args: 'child report')
    result = manager.dispatch({'command': 'agent_fork', 'goal': 'child', '_history': [{'role': 'user', 'content': 'PRIVATE PARENT'}],
                               'parent_id': 'parent', 'target': {'url': 'http://127.0.0.1:9999'}})
    job = manager.jobs[result['job_id']]
    job['thread'].join(3)
    continuation = Path(job['workspace']) / 'continuation.json'
    assert continuation.exists()
    assert agents.Agents(controller(tmp_path)).jobs[job['id']]['continuation_available']
    receipt = Path(job['workspace']) / 'receipt.json'
    legacy = json.loads(receipt.read_text()); legacy.pop('continuation_scope')
    receipt.write_text(json.dumps(legacy))
    continuation.write_text(json.dumps([{'role': 'user', 'content': 'PRIVATE PARENT'}]))
    restored = agents.Agents(controller(tmp_path))
    assert not continuation.exists()
    assert not restored.jobs[job['id']]['continuation_available']
    assert not json.loads(receipt.read_text())['continuation_available']
