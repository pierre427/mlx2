"""Desktop integration and model tool-boundary tests; synthetic CPU-only inputs.

No downloads, model inference, GPU reservations or network/MCP calls occur.
"""
from __future__ import annotations

import json
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2 import desktop, desktop_badges, desktop_control, desktop_tools


class FakeCollector:
    def __init__(self, **_kwargs):
        self.closed = False

    def sample(self):
        return ()

    def close(self):
        self.closed = True


class FakeBadges:
    def __init__(self, _root):
        pass

    def summarize(self, *_args):
        return {"label": "Untested", "qualified": False}


@pytest.fixture
def controller(tmp_path, monkeypatch):
    home = tmp_path / "app"
    root = tmp_path / "models"
    home.mkdir()
    root.mkdir()
    artifact = root / "synthetic-text"
    artifact.mkdir()
    (artifact / "config.json").write_text(json.dumps({"model_type": "synthetic", "max_position_embeddings": 16384}))
    desktop.atomic_json(home / "desktop-settings.json", {
        "root": str(root), "port": 18285, "context": 0, "lanes": 1,
        "route": "auto", "sticky_port": True})
    monkeypatch.setattr(desktop, "HostCollector", FakeCollector)
    monkeypatch.setattr(desktop, "InferenceCollector", FakeCollector)
    monkeypatch.setattr(desktop_badges, "BadgeResolver", FakeBadges)
    instance = desktop_control.DesktopController(home, tmp_path / "never-executed-helper")
    instance.startup_thread.join(2)
    deadline = time.monotonic() + 2
    while instance.primary.library.scanning and time.monotonic() < deadline:
        time.sleep(0.01)
    original_dispatch = instance.dispatch
    def dispatch(request):
        if request["command"] == "chat_target":
            return {"url": request["url"], "pid": request.get("pid"), "api_key": "synthetic-private-token",
                    "model": "synthetic", "checkpoint": request.get("checkpoint", "synthetic@fingerprint"),
                    "capabilities": ["text", "tools"], "context_limit": 16384}
        return original_dispatch(request)
    monkeypatch.setattr(instance, "dispatch", dispatch)
    yield instance
    instance.close()


def profile(controller, **changes):
    model = controller.primary.library.root / "synthetic-text"
    return {"id": "synthetic-profile", "path": str(model), "port": 18400, "context": 8192,
            "lanes": 1, "route": "ordinary", "autostart": False, "share": False,
            "sticky_port": True, **changes}


def chat_session(controller, *, url="http://127.0.0.1:18285", pid=42):
    return controller.dispatch({"command": "chat_tools", "target": {"url": url, "pid": pid}})["session_id"]


def tool(controller, session, name, arguments, **outside):
    return controller.dispatch({"command": "tool_call", "session_id": session,
                               "name": "mlx2__" + name, "arguments": arguments, **outside})


def test_large_chat_tool_inventory_uses_discovery_and_preserves_approval_and_scope(controller):
    from mlx2.server import validate_request

    controller.preferences.update(agent_controls=True, built_in_planning_enabled=True,
                                  agent_delegation_enabled=True, helper_enabled=True, memory_enabled=True)
    result = controller.dispatch({"command": "chat_tools", "target": {"url": "http://127.0.0.1:18285", "pid": 42}, "chat_session_id": "discovery-branch"})
    session = result["session_id"]
    assert len(controller.tool_sessions[session]["catalog"]) > 64
    assert {t["function"]["name"] for t in result["tools"]} == {"mlx2_tool_search", "mlx2_tool_call", "spawn_subagent", "mlx2__tool_search"}
    validate_request({"messages": [{"role": "user", "content": "hi"}], "tools": result["tools"]}, True)

    def invoke(name, arguments, **extra):
        return controller.dispatch({"command": "tool_call", "session_id": session, "name": name, "arguments": arguments, **extra})

    found = invoke("mlx2_tool_search", {"query": "snapshot"})["result"]
    assert any(t["function"]["name"] == "mlx2__snapshot" for t in found["tools"])
    for offset in (0, 3, 6, 18):
        overview = invoke("mlx2_tool_search", {"query": "", "offset": offset})["result"]
        assert overview["complete"] and overview["next_offset"] is None
        assert all(name in overview["summary"] for name in controller.tool_sessions[session]["catalog"])
        assert "tools" not in overview  # Argument schemas are fetched only for specific tasks.
    page = invoke("mlx2_tool_search", {"query": "model"})["result"]
    assert 1 <= len(page["tools"]) <= 3 and page["next_offset"] == len(page["tools"])
    assert invoke("mlx2_tool_call", {"name": "mlx2__stop", "arguments": {}})["requires_approval"]
    with pytest.raises(ValueError, match="outside this conversation"):
        invoke("mlx2_tool_call", {"name": "mlx2__plan_get", "arguments": {"session_id": "foreign"}})
    with pytest.raises(ValueError, match="not exposed"):
        invoke("mlx2_tool_call", {"name": "mlx2_tool_call", "arguments": {}})
    controller.preferences["agent_controls"] = False
    assert invoke("mlx2_tool_search", {"query": "mlx2__stop"})["result"]["tools"] == []
    with pytest.raises(ValueError, match="disabled"):
        invoke("mlx2_tool_call", {"name": "mlx2__stop", "arguments": {}}, approved=True)


def test_media_schema_names_configured_providers_without_mutating_global_schema(controller, monkeypatch):
    monkeypatch.setattr(controller.media, 'profiles', list)
    controller.preferences['media_services'] = [{'id': 'synthetic-image-provider'}]
    session = chat_session(controller)
    schema = controller.tool_sessions[session]['catalog']['mlx2__media_generate']['function']['parameters']
    assert schema['properties']['provider_id']['enum'] == ['synthetic-image-provider']
    assert 'enum' not in desktop_tools.OPERATIONS['media_generate'][1]['provider_id']


@pytest.mark.parametrize(('query', 'expected'), [
    ('picture', 'world_image_search'), ('pictures', 'world_image_search'),
    ('photo', 'world_image_search'), ('photographs', 'world_image_search'),
    ('penguin image', 'world_image_search'), ('image search', 'world_image_search'),
    ('image generation', 'media_generate'), ('generate image', 'media_generate'),
    ('generating images', 'media_generate'), ('music generation', 'media_generate'),
    ('video generation', 'media_generate'), ('audio generation', 'media_generate'),
    ('mlx2__snapshot', 'snapshot'),
])
def test_tool_discovery_ranks_capabilities_without_requiring_subject_words(controller, query, expected):
    session = chat_session(controller)
    result = controller.dispatch({'command': 'tool_call', 'session_id': session,
                                  'name': 'mlx2_tool_search', 'arguments': {'query': query}})['result']
    assert result['tools'][0]['function']['name'] == 'mlx2__' + expected


def test_tool_discovery_whole_words_and_no_match_guidance(controller):
    session = chat_session(controller)
    result = controller.dispatch({'command': 'tool_call', 'session_id': session,
                                  'name': 'mlx2_tool_search', 'arguments': {'query': 'art'}})['result']
    assert not result['tools']  # "start" and "artifact" are not whole-word art matches.
    assert result['next_offset'] is None
    assert 'empty-query overview' in result['summary']
    assert 'not the subject' in result['summary']
    catalog = controller.tool_sessions[session]['catalog']
    exact = catalog['mlx2__world_image_search']
    assert desktop_tools._search_score('mlx2__world_image_search', exact) > desktop_tools._search_score('image', exact)


def test_tool_discovery_keeps_large_schemas_explicitly_withheld_at_small_budget(controller):
    session = chat_session(controller)
    controller.tool_sessions[session]['result_policy'] = {'result_max_tokens': 1024}
    result = controller.dispatch({'command': 'tool_call', 'session_id': session,
                                  'name': 'mlx2_tool_search', 'arguments': {'query': 'mlx2__media_generate'}})['result']
    assert result['tools'] == [] and result['schema_withheld']
    assert result['next_offset'] is None
    assert 'Increase result_max_tokens' in result['summary']
    assert 'Do not guess arguments' in result['summary']


def test_tool_discovery_native_prefix_typo_returns_only_authorized_canonical_schema(controller):
    session = chat_session(controller)

    def search(query):
        return controller.dispatch({'command': 'tool_call', 'session_id': session,
                                    'name': 'mlx2_tool_search', 'arguments': {'query': query}})['result']['tools']

    canonical = search('mlx2__tool_result_policy')
    assert canonical[0]['function']['name'] == 'mlx2__tool_result_policy'
    assert search('mlx2_tool_result_policy') == canonical
    assert search('mlx2_nonexistent_tool') == []
    controller.preferences['agent_controls'] = False
    assert search('mlx2_stop') == []
    assert search('mlx2__stop') == []


@pytest.mark.parametrize('query', ['planning tool', 'planner function', 'planned API', 'session planning task'])
def test_planning_discovery_prefers_domain_entry_points_over_generic_tool_operations(controller, query):
    controller.preferences['built_in_planning_enabled'] = True
    session = chat_session(controller)
    result = controller.dispatch({'command': 'tool_call', 'session_id': session, 'name': 'mlx2_tool_search',
                                  'arguments': {'query': query}})['result']
    names = [definition['function']['name'] for definition in result['tools']]
    assert names[:2] == ['mlx2__plan_get', 'mlx2__plan_create']
    assert 'For a capability question, answer from this metadata' in result['summary']
    assert "only when the user's task requires action" in result['summary']


def test_bounded_model_inventory_preserves_exact_paths_without_full_snapshot(controller, monkeypatch):
    rows = [{'name': f'model-{index:02}', 'path': f'/exact/models/model-{index:02}',
             'identity': f'exact:{index}', 'kind': 'Serving candidate', 'can_start': True,
             'oversized_details': 'x' * 100000} for index in range(12)]
    monkeypatch.setattr(controller.primary.library, 'snapshot', lambda: {'rows': rows, 'scanning': False})
    monkeypatch.setattr(controller, 'snapshot', lambda: pytest.fail('Full app snapshot must not be needed'))
    session = chat_session(controller)
    first = tool(controller, session, 'library_models', {'limit': 3})['result']
    assert first['models'][0]['path'] == '/exact/models/model-00'
    assert first['models'][0]['identity'] == 'exact:0'
    assert first['total'] == 12 and first['next_offset'] == 3
    assert 'oversized_details' not in first['models'][0]
    second = tool(controller, session, 'library_models', {'offset': first['next_offset'], 'limit': 3})['result']
    assert second['models'][0]['path'] == '/exact/models/model-03'
    filtered = tool(controller, session, 'library_models', {'query': 'model-11'})['result']
    assert filtered['total'] == 1 and filtered['models'][0]['identity'] == 'exact:11'
    for arguments in ({'offset': -1}, {'limit': 21}, {'limit': True}, {'query': 'x' * 257}):
        with pytest.raises(ValueError, match='Model inventory needs'):
            tool(controller, session, 'library_models', arguments)


def test_agent_launch_guidance_rejects_invented_library_urls(controller):
    controller.preferences['agent_delegation_enabled'] = True
    session = chat_session(controller)
    description = controller.tool_sessions[session]['catalog']['mlx2__agent_launch']['function']['description']
    assert 'A capability question needs an answer' in description
    with pytest.raises(ValueError, match='Do not invent library:// URLs'):
        tool(controller, session, 'agent_launch', {'path': 'library://agent/agent-v1', 'identity': 'invented',
             'goal': 'Check ability', 'instance': 'new', 'tools': []}, approved=True)


def test_harness_tool_usage_guidance_covers_native_and_dynamic_schemas_without_mutation(controller):
    assert set(desktop_tools.TOOL_USAGE) == set(desktop_tools.OPERATIONS)
    controller.preferences.update(memory_enabled=True, agent_controls=True, agent_delegation_enabled=True)
    result = controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                  'chat_session_id': 'documented-tools'})
    catalog = controller.tool_sessions[result['session_id']]['catalog']
    for name in desktop_tools.OPERATIONS:
        if 'mlx2__' + name in catalog:
            assert desktop_tools.TOOL_USAGE[name] in catalog['mlx2__' + name]['function']['description']
    for name, expected in [('memory_search', 'scoped facts'), ('message_post', 'admitted agent mailbox'),
                           ('capsule_get', 'exact scoped memory checkpoints'),
                           ('mlx2__session_compact', 'latest revision/digest')]:
        assert expected in catalog[name]['function']['description']
    schema = catalog['mlx2__agent_launch']['function']['parameters']['properties']
    assert 'copy unchanged' in schema['identity']['description']
    assert 'description' not in desktop_tools.OPERATIONS['agent_launch'][1]['identity']
    provider = {'type': 'function', 'function': {'name': 'custom_remote', 'description': 'Provider instructions',
                                               'parameters': {'properties': {}}}}
    assert desktop_tools._document_tool(provider) == provider


def test_prompt_preset_tools_return_metadata_and_require_approval_for_install(controller):
    custom = {'id': 'custom-instructions', 'name': 'Custom', 'text': 'Keep this custom prompt.'}
    controller.configure_preferences({'system_prompts': [custom], 'base_system_prompt_id': custom['id']})
    session = chat_session(controller)
    catalog = tool(controller, session, 'prompt_presets', {})
    assert not catalog.get('requires_approval')
    assert {row['preset'] for row in catalog['result']['presets']} == {'full', 'compact'}
    assert all('text' not in row and row['estimated_tokens'] > 0 for row in catalog['result']['presets'])
    assert len(json.dumps(catalog['result'])) < 2048
    arguments = {'preset': 'compact', 'activate': True}
    assert tool(controller, session, 'prompt_install', arguments)['requires_approval']
    assert controller.preferences['base_system_prompt_id'] == custom['id']
    installed = tool(controller, session, 'prompt_install', arguments, approved=True)['result']
    assert installed['activated'] and installed['applies_to'] == 'future chat turns'
    assert custom in controller.preferences['system_prompts']
    assert controller.preferences['base_system_prompt_id'] == installed['prompt_id']
    with pytest.raises(ValueError, match='missing control arguments'):
        tool(controller, session, 'prompt_install', {'preset': 'compact'}, approved=True)


def test_default_agent_delegation_binds_current_model_without_loading_another(controller, monkeypatch):
    controller.preferences['agent_delegation_enabled'] = True
    calls = []
    monkeypatch.setattr(controller.agents, 'dispatch', lambda request: calls.append(request) or {'job_id': 'same-engine-child'})
    session = chat_session(controller)
    schema = controller.tool_sessions[session]['catalog']['mlx2__agent_delegate']['function']['parameters']
    assert schema['required'] == ['goals']
    arguments = {'goals': ['Report ok']}
    assert tool(controller, session, 'agent_delegate', arguments)['requires_approval']
    assert not calls
    result = tool(controller, session, 'agent_delegate', arguments, approved=True)
    assert result['result']['job_id'] == 'same-engine-child'
    assert calls[0]['target'] == controller.tool_sessions[session]['selection']
    assert '_history' not in calls[0]  # Independent fresh context, same serving engine.
    with pytest.raises(ValueError, match='same serving model'):
        tool(controller, session, 'agent_delegate', {**arguments, 'target': {'url': 'http://127.0.0.1:29999'}}, approved=True)
    assert len([call for call in calls if call['command'] == 'agent_delegate']) == 1
    controller.tool_sessions[session]['target']['checkpoint'] = 'other@checkpoint'
    with pytest.raises(ValueError, match='will not fall back'):
        tool(controller, session, 'agent_delegate', arguments, approved=True)
    assert len([call for call in calls if call['command'] == 'agent_delegate']) == 1


def test_delegation_retries_reuse_job_across_direct_and_wrapper_calls_and_preserve_reports(controller, monkeypatch):
    controller.preferences.update(agent_delegation_enabled=True, memory_enabled=True)
    created, jobs = [], {}

    def dispatch(request):
        if request['command'] == 'agent_delegate':
            created.append(request)
            identity = f'child-{len(created)}'
            jobs[identity] = {'id': identity, 'status': 'running', 'workers': []}
            return {'job_id': identity}
        assert request['command'] == 'agent_result'
        return jobs[request['job_id']]

    monkeypatch.setattr(controller.agents, 'dispatch', dispatch)
    session = chat_session(controller)
    first = tool(controller, session, 'agent_delegate', {'goals': ['  Report ok  ', 'Report ok'], 'concurrency': 1,
                                                       'max_tokens': 128}, approved=True)['result']
    wrapped = controller.dispatch({'command': 'tool_call', 'session_id': session, 'name': 'mlx2_tool_call',
        'arguments': {'name': 'mlx2__agent_delegate', 'arguments': {'goals': ['Report ok', 'Report ok'],
                      'concurrency': 4, 'max_tokens': 4096}}, 'approved': True})['result']
    assert first['job_id'] == wrapped['job_id'] == 'child-1'
    assert not first['reused'] and wrapped['reused'] and len(created) == 1
    assert len(created[0]['goals']) == 2  # Explicit goal multiplicity survives deduplication.
    assert 'do not poll' in wrapped['next_step']['instruction']
    assert 'arguments' not in wrapped['next_step']
    assert set(created[0]['tools']) == {'message_post', 'message_read', 'message_ack'}
    jobs['child-1'].update(status='completed', synthesis='ok', workers=[{'report': {'content': 'ok'}}])
    completed = tool(controller, session, 'agent_delegate', {'goals': ['Report ok', 'Report ok']}, approved=True)['result']
    assert completed['reused'] and completed['status'] == 'completed'
    assert completed['synthesis'] == 'ok' and completed['workers'][0]['report']['content'] == 'ok'
    assert len(created) == 1
    for arguments in ({'goals': ['Report ok']}, {'goals': ['Report ok'], 'tools': ['memory_get']},
                      {'goals': ['A', 'B']}, {'goals': ['B', 'A']}, {'goals': ['Report  ok', 'Report ok']},
                      {'goals': ['if ok:\n    report()']}, {'goals': ['if ok:\n  report()']}):
        tool(controller, session, 'agent_delegate', arguments, approved=True)
    assert len(created) == 8  # Order, grants and meaningful internal whitespace remain distinct.
    tool(controller, chat_session(controller), 'agent_delegate', {'goals': ['Report ok']}, approved=True)
    assert len(created) == 9  # A new submitted turn may deliberately repeat work.


def test_snapshot_facts_survive_bounded_detail_truncation(controller, monkeypatch):
    # Automatic overflow assessment must never reach a real engine in this CPU fixture.
    controller.preferences["agent_delegation_enabled"] = False
    monkeypatch.setattr(controller, 'snapshot', lambda: {'workers': [{'process_running': True}], 'library': {'rows': [{}, {}]}, 'oversized': 'x' * 100000})
    result = tool(controller, chat_session(controller), 'snapshot', {})
    assert result['result']['truncated']
    assert 'worker count = 1' in result['summary']
    assert 'running process = 1' in result['summary']
    assert 'library entries = 2' in result['summary']


def test_named_engine_failure_remains_visible_in_snapshot(controller):
    engine = SimpleNamespace(worker={'state': 'Error', 'name': 'synthetic'}, child=None,
                             snapshot=lambda: {'servers': [], 'error': 'Synthetic startup admission failure'})
    controller.engines['failed-profile'] = engine
    try:
        row = next(row for row in controller.snapshot()['workers'] if row['profile_id'] == 'failed-profile')
        assert row['state'] == 'Error' and row['error'] == 'Synthetic startup admission failure'
    finally:
        controller.engines.pop('failed-profile')


def test_startup_and_snapshot_have_no_implicit_jobs_or_model_owners(controller):
    assert not controller.startup_thread.is_alive()
    assert not controller.campaign.running
    assert controller.pool.users == 0
    assert controller.primary.child is None
    assert controller.store.jobs() == []
    snapshot = controller.snapshot()
    assert snapshot["preferences"]["download_parallelism"] == 5
    assert snapshot["preferences"]["download_bytes_per_second"] == 0
    assert snapshot["workers"] == []
    assert snapshot["jobs"] == []
    assert "application_started" in controller.logs()["text"]
    path = controller.home / "synthetic-worker.log"
    path.write_text("Synthetic model load log\n")
    controller.primary.log_path = path
    assert "Synthetic model load log" in controller.logs("worker")["text"]


@pytest.mark.parametrize("lanes", [1, 2, 3, 4])
def test_configure_accepts_native_lane_range_and_sticky_port(controller, lanes):
    settings = {**controller.primary.settings, "lanes": lanes, "sticky_port": False}
    result = controller.dispatch({"command": "configure", "settings": settings})
    assert result["lanes"] == lanes
    assert result["sticky_port"] is False
    assert json.loads((controller.home / "desktop-settings.json").read_text())["lanes"] == lanes


@pytest.mark.parametrize("changes", [{"lanes": 0}, {"lanes": 5}, {"lanes": True},
                                      {"sticky_port": "true"}, {"context": -1}, {"port": 80}])
def test_configure_rejects_untyped_or_unsupported_ui_settings(controller, changes):
    with pytest.raises((ValueError, TypeError)):
        controller.dispatch({"command": "configure", "settings": {**controller.primary.settings, **changes}})


def test_profile_settings_typed_and_running_profile_cannot_be_rebound(controller):
    saved = profile(controller, lanes=4, sticky_port=False)
    controller.configure_preferences({"profiles": [saved]})
    assert controller.profile(saved["id"])["lanes"] == 4
    for changes in ({"lanes": 0}, {"lanes": 5}, {"lanes": True}, {"sticky_port": 1}, {"autostart": "yes"}):
        with pytest.raises(ValueError):
            controller.configure_preferences({"profiles": [profile(controller, **changes)]})
    controller.engines[saved["id"]] = SimpleNamespace(child=object(), preparing=lambda: False)
    try:
        with pytest.raises(ValueError, match="Stop this profile"):
            controller.configure_preferences({"profiles": [{**saved, "port": saved["port"] + 1}]})
        with pytest.raises(ValueError, match="Stop a running profile"):
            controller.configure_preferences({"profiles": []})
    finally:
        controller.engines.pop(saved["id"])


def test_global_and_model_context_limits_clamp_actual_nested_model_config(controller):
    path = controller.primary.library.root / "synthetic-text"
    controller.configure_preferences({"global_context_tokens": 65536})
    assert controller.context_limit(path) == 16384
    controller.configure_preferences({"model_context_tokens": {str(path): 8192}})
    assert controller.context_limit(path) == 8192
    (path / "config.json").write_text(json.dumps({"model_type": "synthetic", "text_config": {"max_position_embeddings": 4096}}))
    assert controller.context_limit(path) == 4096
    controller.configure_preferences({"model_context_tokens": {str(path): 0}, "global_context_tokens": 2048})
    assert controller.context_limit(path) == 2048


def test_interactive_pool_releases_only_after_last_owned_child(monkeypatch):
    calls = []
    class Reservation:
        def __init__(self):
            self.owner = {"synthetic": True}
        def acquire(self):
            calls.append("acquire")
        def release(self):
            calls.append("release")
    monkeypatch.setattr(desktop_control, "GPUReservation", Reservation)
    pool = desktop_control.ReservationPool()
    first, second = pool.ticket(), pool.ticket()
    first.acquire()
    second.acquire()
    assert pool.users == 2
    assert calls == ["acquire"]
    first.release()
    first.release()
    assert pool.users == 1
    assert calls == ["acquire"]
    second.release()
    assert pool.users == 0 and pool.reservation is None
    assert calls == ["acquire", "release"]


def test_download_configuration_default_parallelism_and_global_throttle(controller):
    assert controller.downloads.pool._max_workers == 5
    controller.configure_preferences({"download_parallelism": 3, "download_bytes_per_second": 123456})
    assert controller.downloads.pool._max_workers == 3
    assert controller.downloads.throttle.rate == 123456
    controller.downloads.cancellations["synthetic-active-job"] = threading.Event()
    try:
        with pytest.raises(ValueError, match="downloads to finish"):
            controller.configure_preferences({"download_parallelism": 2})
    finally:
        controller.downloads.cancellations.clear()
    for changes in ({"download_parallelism": True}, {"download_parallelism": 0},
                    {"download_bytes_per_second": -1}, {"download_bytes_per_second": 0.5}):
        with pytest.raises(ValueError):
            controller.configure_preferences(changes)


def test_missing_saved_model_does_not_block_unrelated_settings(controller):
    saved = profile(controller)
    controller.configure_preferences({'profiles': [saved]})
    Path(saved['path']).rename(Path(saved['path']).with_name('moved-model'))
    controller.configure_preferences({'global_context_tokens': 8192})
    assert controller.preferences['global_context_tokens'] == 8192
    assert controller.preferences['profiles'] == [saved]
    with pytest.raises(FileNotFoundError):
        controller.configure_preferences({'profiles': [{**saved, 'lanes': 2}]})


def test_shared_disk_budget_counts_outstanding_reservations(tmp_path, monkeypatch):
    from mlx2 import desktop_disk
    free = {"bytes": 1000}
    monkeypatch.setattr(desktop_disk.shutil, "disk_usage", lambda path: SimpleNamespace(free=free["bytes"]))
    disk = desktop_disk.DiskBudget()
    disk.reserve("first", tmp_path, 600, reserve=100)
    with pytest.raises(ValueError, match="Insufficient free space"):
        disk.reserve("second", tmp_path, 400, reserve=100)
    disk.consumed("first", 200)
    free["bytes"] -= 200
    with pytest.raises(ValueError, match="Insufficient free space"):
        disk.reserve("second", tmp_path, 400, reserve=100)
    disk.release("first")
    disk.reserve("second", tmp_path, 400, reserve=100)
    free["bytes"] = 99
    with pytest.raises(ValueError, match="safety reserve"):
        disk.check("second")


def test_model_cannot_self_approve_mutation_through_tool_arguments(controller):
    session = chat_session(controller)
    changes = {"global_context_tokens": 8192}
    pending = tool(controller, session, "preferences", {"changes": changes})
    assert pending["requires_approval"]
    assert controller.preferences["global_context_tokens"] == 32768
    with pytest.raises(ValueError, match="Unexpected or missing"):
        tool(controller, session, "preferences", {"changes": changes, "approved": True})
    assert controller.preferences["global_context_tokens"] == 32768
    tool(controller, session, "preferences", {"changes": changes}, approved=True)
    assert controller.preferences["global_context_tokens"] == 8192


def test_plan_scope_and_grants_survive_new_turn_for_same_principal(controller):
    first = chat_session(controller)
    created = tool(controller, first, "plan_create", {"goal": "Synthetic bounded goal"})["result"]["session_id"]
    outsider = controller.planning.dispatch({"command": "plan_create", "goal": "Private unrelated operator plan"})["session_id"]
    with pytest.raises(ValueError, match="outside this conversation"):
        tool(controller, first, "plan_get", {"session_id": outsider})
    second = chat_session(controller)
    assert tool(controller, second, "plan_get", {"session_id": created})["result"]["session"]["id"] == created
    different = chat_session(controller, pid=43)
    with pytest.raises(ValueError, match="outside this conversation"):
        tool(controller, different, "plan_get", {"session_id": created})
    snapshot = tool(controller, first, "snapshot", {})["result"]
    assert "planning" not in snapshot and "agents" not in snapshot and "helpers" not in snapshot
    assert "session:" + created in controller.tool_sessions[second]["context"]["allowed_scopes"]


def test_native_branches_have_distinct_tool_grants_on_same_engine(controller):
    def session(branch):
        return controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                    'chat_session_id': branch})['session_id']
    first = session('first-branch')
    created = tool(controller, first, 'plan_create', {'goal': 'Private branch plan'})['result']['session_id']
    same = session('first-branch')
    assert tool(controller, same, 'plan_get', {'session_id': created})['result']['session']['id'] == created
    other = session('other-branch')
    with pytest.raises(ValueError, match='outside this conversation'):
        tool(controller, other, 'plan_get', {'session_id': created})
    assert controller.tool_sessions[first]['context']['principal_id'] != controller.tool_sessions[other]['context']['principal_id']


def test_skill_tools_gate_writes_and_progressively_disclose_content(controller):
    session = chat_session(controller)
    arguments = {'action': 'create', 'name': 'synthetic-procedure', 'expected_revision': 0,
                 'description': 'A synthetic reusable procedure', 'body': '# Procedure\nCheck the exact source first.'}
    assert tool(controller, session, 'skill_manage', arguments)['requires_approval']
    assert controller.skills.snapshot()['total'] == 0
    created = tool(controller, session, 'skill_manage', arguments, approved=True)['result']
    assert created['revision'] == 1
    listed = tool(controller, session, 'skills_list', {})['result']
    assert listed['skills'][0]['name'] == arguments['name'] and 'body' not in str(listed)
    viewed = tool(controller, session, 'skill_view', {'name': arguments['name']})['result']
    assert viewed['content'] == arguments['body'] and viewed['reference_only']
    assert tool(controller, session, 'skill_policy', {'changes': {'autonomous_writes': True}})['requires_approval']
    tool(controller, session, 'skill_policy', {'changes': {'autonomous_writes': True}}, approved=True)
    patched = tool(controller, session, 'skill_manage', {'action': 'patch', 'name': arguments['name'],
                   'expected_revision': 1, 'old_text': 'first.', 'new_text': 'and checkpoint first.'})['result']
    assert patched['revision'] == 2
    controller.dispatch({'command': 'skill_policy', 'changes': {'enabled': False}})
    controller.tool_sessions[session]['target']['context_limit'] = 131072
    assert tool(controller, session, 'snapshot', {})['result']['skills']['skills'] == []
    with pytest.raises(ValueError, match='disabled'):
        tool(controller, session, 'skill_view', {'name': arguments['name']})


def test_model_snapshot_omits_disabled_skill_metadata(controller):
    controller.dispatch({'command': 'skill_manage', 'action': 'create', 'name': 'private-disabled',
                         'expected_revision': 0, 'description': 'DISABLED-DESCRIPTION-CANARY', 'body': 'Private instructions', 'enabled': False})
    controller.dispatch({'command': 'skill_manage', 'action': 'create', 'name': 'public-enabled',
                         'expected_revision': 0, 'description': 'Enabled procedure', 'body': 'Procedure', 'enabled': True})
    session = chat_session(controller)
    controller.tool_sessions[session]['target']['context_limit'] = 131072
    model_snapshot = tool(controller, session, 'snapshot', {})['result']['skills']
    assert model_snapshot['total'] == 1 and model_snapshot['skills'][0]['name'] == 'public-enabled'
    assert 'DISABLED-DESCRIPTION-CANARY' not in json.dumps(model_snapshot)


def test_model_snapshot_respects_disabled_memory_and_capsules(controller, monkeypatch):
    session = chat_session(controller)
    controller.tool_sessions[session]['target']['context_limit'] = 131072
    monkeypatch.setattr(controller.memory, 'snapshot', lambda **kwargs: {'private': 'OFF-MEMORY-CANARY'})
    monkeypatch.setattr(controller.capsules, 'snapshot', lambda **kwargs: {'private': 'OFF-CAPSULE-CANARY'})
    controller.preferences['memory_enabled'] = False
    result = tool(controller, session, 'snapshot', {})['result']
    assert result['memory'] == {'enabled': False} and result['capsules'] == {'enabled': False}
    assert 'OFF-MEMORY-CANARY' not in json.dumps(result) and 'OFF-CAPSULE-CANARY' not in json.dumps(result)


def test_mcp_header_secret_is_private_single_line_and_absent_from_snapshot(controller):
    saved = controller.dispatch({"command": "mcp_save_secret", "headers": {"Authorization": "Bearer SYNTHETIC-PRIVATE-HEADER"}})
    secret = controller.home / "mcp-secrets" / (saved["header_ref"] + ".json")
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    connection = {"id": "synthetic", "url": "http://127.0.0.1:12345/mcp", "enabled": False,
                  "allowed_tools": [], "auto_approve": False, "header_ref": saved["header_ref"]}
    controller.configure_preferences({"mcp_servers": [connection]})
    assert "SYNTHETIC-PRIVATE-HEADER" not in json.dumps(controller.snapshot())
    assert "SYNTHETIC-PRIVATE-HEADER" not in controller.logs()["text"]
    for headers in ({"Authorization": "a\r\nb"}, {"X-Header\nInjected": "value"}):
        with pytest.raises(ValueError, match="single-line"):
            controller.dispatch({"command": "mcp_save_secret", "headers": headers})


def test_changing_external_mcp_settings_invalidates_old_tool_session(controller, monkeypatch):
    executed = []
    instance = SimpleNamespace(execute=lambda executor, args: executed.append(args) or {"ok": True})
    monkeypatch.setattr(desktop_tools, "backend", lambda control: (instance, [], {"external": object()}, {"external": False}))
    session = chat_session(controller)
    request = {"command": "tool_call", "session_id": session, "name": "external", "arguments": {}}
    assert controller.dispatch(request)["result"] == {"ok": True}
    connection = {"id": "changed", "url": "http://127.0.0.1:12345/mcp", "enabled": False,
                  "allowed_tools": [], "auto_approve": False, "header_ref": ""}
    controller.configure_preferences({"mcp_servers": [connection]})
    with pytest.raises(ValueError, match="MCP settings changed"):
        controller.dispatch(request)
    assert executed == [{}]


def test_self_lifecycle_uses_private_session_target_and_forbids_injected_target(controller, monkeypatch):
    session = chat_session(controller)
    dispatched = []
    monkeypatch.setattr(controller.lifecycle, "dispatch", lambda request: dispatched.append(request) or {"terminal_action": True})
    with pytest.raises(ValueError, match="Unexpected or missing"):
        tool(controller, session, "engine_unload_self", {"_target": {"url": "http://127.0.0.1:9999"}}, approved=True)
    pending = tool(controller, session, "engine_unload_self", {})
    assert pending["requires_approval"] and not dispatched
    result = tool(controller, session, "engine_unload_self", {}, approved=True)
    assert result["result"]["terminal_action"]
    assert dispatched[0]["_target"]["url"] == "http://127.0.0.1:18285"
    assert dispatched[0]["_target"]["checkpoint"] == "synthetic@fingerprint"


def test_fleet_advertises_only_opted_in_managed_and_exact_discovered_engines(controller, monkeypatch):
    from mlx2 import inference_monitor
    from mlx2.desktop_fleet import DesktopFleet
    fleet = DesktopFleet(controller)
    overlays = {}
    fleet.node = SimpleNamespace(_desktop_set_shares=lambda client_id, entries: overlays.update({client_id: tuple(entries)}))
    engine = SimpleNamespace(child=SimpleNamespace(pid=77), worker={"url": "http://127.0.0.1:18400", "api_key_file": "/synthetic/key"})
    controller.engines["synthetic-profile"] = engine
    controller.preferences["profiles"] = [profile(controller, share=True)]
    first = SimpleNamespace(url="http://127.0.0.1:18401", pid=78)
    foreign = SimpleNamespace(url="http://127.0.0.1:18402", pid=79)
    controller.preferences["shared_servers"] = [{"url": first.url, "pid": first.pid}]
    monkeypatch.setattr(inference_monitor, "discover_servers", lambda: [first, foreign])
    monkeypatch.setattr(inference_monitor, "local_server_api_key", lambda endpoint: "SYNTHETIC-FLEET-SECRET")
    try:
        fleet.sync()
        rows = overlays[fleet.client_id]
        assert {row["pid"] for row in rows} == {77, 78}
        copied = Path(next(row["api_key_file"] for row in rows if row["pid"] == 78))
        assert stat.S_IMODE(copied.stat().st_mode) == 0o600
        assert copied.read_text() == "SYNTHETIC-FLEET-SECRET"
        controller.preferences["profiles"][0]["share"] = False
        controller.preferences["shared_servers"][0]["pid"] = 999
        fleet.sync()
        assert overlays[fleet.client_id] == ()
    finally:
        controller.engines.pop("synthetic-profile")


def test_fleet_three_machine_inventory_exact_remote_model_and_departure_cleanup(controller, monkeypatch):
    from mlx2 import desktop_fleet
    peers = {
        "second": SimpleNamespace(candidate=SimpleNamespace(nid="second"), state="alive", models=("remote-A",), hostname="second-host"),
        "third": SimpleNamespace(candidate=SimpleNamespace(nid="third"), state="alive", models=("remote-B",), hostname="third-host"),
    }
    class Collector:
        def __init__(self, *, urls, **kwargs):
            self.url = urls[0]
            self.api_key = kwargs["api_key"]
            self.closed = False
        def sample(self):
            return [{"endpoint": {"url": self.url}, "reachable": True}]
        def close(self):
            self.closed = True
    monkeypatch.setattr(desktop_fleet, "InferenceCollector", Collector)
    fleet = desktop_fleet.DesktopFleet(controller)
    fleet.proxy = SimpleNamespace(server_port=54321)
    fleet.community = SimpleNamespace(nid="self", community_id="community")
    fleet.owns_node = True
    def nodes():
        return [{"nid": "self", "state": "alive", "models": [], "hostname": "self-host"},
                *[{"nid": identity, "state": peer.state, "models": peer.models, "hostname": peer.hostname} for identity, peer in peers.items()]]
    fleet.node = SimpleNamespace(_lock=threading.RLock(), _peers=peers,
        _peer_view=lambda peer: peer, _nodes=nodes,
        actual_front_port=8290, community=fleet.community)
    result = fleet.snapshot()
    assert len(result["nodes"]) == 3
    assert {row["endpoint"]["nid"] for row in result["servers"]} == {"second", "third"}
    assert all(row["endpoint"]["remote"] for row in result["servers"])
    target = fleet.chat_target("second", "remote-A")
    assert target["api_key"] == fleet.key
    assert target["url"].startswith("http://127.0.0.1:54321/peer/second/")
    with pytest.raises(ValueError, match="no longer available"):
        fleet.chat_target("second", "remote-B")
    old = fleet.collectors[("third", "remote-B")]
    peers["third"].state = "dead"
    result = fleet.snapshot()
    assert len(result["servers"]) == 1
    assert old.closed
    with pytest.raises(ValueError, match="no longer available"):
        fleet.chat_target("third", "remote-B")


def test_fleet_connection_requires_existing_admitted_community(controller, monkeypatch):
    from mlx2 import desktop_fleet
    calls = []
    def absent(home):
        raise FileNotFoundError("Synthetic unadmitted community")
    monkeypatch.setattr(desktop_fleet, "load_community", absent)
    monkeypatch.setattr(desktop_fleet, "FabricNode", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(FileNotFoundError, match="unadmitted"):
        desktop_fleet.DesktopFleet(controller).start()
    assert calls == []


def test_reviewed_preflight_input_pins_reject_mutation_missing_and_escape(tmp_path, monkeypatch):
    import hashlib

    from mlx2 import desktop_preflight_inputs
    relative = "qualification/runs/synthetic-historical/fixture.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    content = b'{"synthetic": true}\n'
    path.write_bytes(content)
    monkeypatch.setattr(desktop_preflight_inputs, "PREFLIGHT_INPUT_SHA256", {relative: hashlib.sha256(content).hexdigest()})
    assert list(desktop_preflight_inputs.preflight_inputs(tmp_path)) == [path]
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        list(desktop_preflight_inputs.preflight_inputs(tmp_path))
    path.unlink()
    with pytest.raises(ValueError, match="Missing"):
        list(desktop_preflight_inputs.preflight_inputs(tmp_path))
    foreign = tmp_path.parent / (tmp_path.name + "-foreign-fixture.json")
    foreign.write_bytes(content)
    try:
        path.symlink_to(foreign)
        with pytest.raises(ValueError, match="escaped"):
            list(desktop_preflight_inputs.preflight_inputs(tmp_path))
    finally:
        foreign.unlink()


def test_same_url_mcp_connections_keep_distinct_operator_approval(controller, monkeypatch):
    first = SimpleNamespace(url="http://127.0.0.1:12345/mcp")
    second = SimpleNamespace(url="http://127.0.0.1:12345/mcp")
    class Backend:
        def __init__(self, config):
            self.clients = {"automatic": first, "manual": second}
        def prepare(self, definitions):
            return [], {"mcp__automatic__read": (first, "read"), "mcp__manual__write": (second, "write")}
    row = {"url": first.url, "enabled": True, "allowed_tools": [], "header_ref": ""}
    controller.preferences["mcp_servers"] = [{**row, "id": "automatic", "auto_approve": True}, {**row, "id": "manual", "auto_approve": False}]
    monkeypatch.setattr(desktop_tools, "ConfiguredToolBackend", Backend)
    _, _, _, approvals = desktop_tools.backend(controller)
    assert approvals == {"mcp__automatic__read": False, "mcp__manual__write": True}


def test_self_lifecycle_revalidates_original_checkpoint_before_dispatch(controller, monkeypatch):
    session = chat_session(controller)
    original = controller.dispatch
    dispatched = []
    def changed(request):
        if request["command"] == "chat_target":
            return {"url": request["url"], "checkpoint": "different-model@different-fingerprint"}
        if request["command"] == "engine_unload_self":
            dispatched.append(request)
            return {"terminal_action": True}
        return original(request)
    monkeypatch.setattr(controller, "dispatch", changed)
    with pytest.raises(ValueError, match="engine changed"):
        tool(controller, session, "engine_unload_self", {}, approved=True)
    assert dispatched == []


def test_model_snapshot_hides_other_research_and_private_dream_metadata(controller, monkeypatch):
    session = chat_session(controller)
    controller.tool_sessions[session]["target"]["context_limit"] = 131072
    controller.research.sources["private-source"] = {"id": "private-source", "searched_by": "chat:someone-else", "title": "PRIVATE-RESEARCH-TOPIC"}
    controller.dream.policy["operator_preferences"] = "PRIVATE-DREAM-BOUNDARY"
    controller.dream.policy["goal"] = "PRIVATE-DREAM-GOAL"
    result = tool(controller, session, "snapshot", {})["result"]
    assert result["research"]["sources"] == []
    assert result["dream"]["private_details_in_operator_ui"]
    encoded = json.dumps(result)
    assert "PRIVATE-RESEARCH-TOPIC" not in encoded
    assert "PRIVATE-DREAM-BOUNDARY" not in encoded
    assert "PRIVATE-DREAM-GOAL" not in encoded
    assert "operator_preferences" in controller.snapshot()["dream"]["policy"]


def test_user_profiles_real_controller_capsule_settings_and_chat_isolation(controller, monkeypatch):
    c = controller
    first = c.user_profiles.active
    c.capsules.configure({'autonomy': 'situational', 'allow_promote': True})
    c.user_profiles.autosave()
    c.dispatch({'command': 'user_profile_create', 'name': 'Research'})
    second = c.user_profiles.active
    c.capsules.configure({'autonomy': 'off', 'allow_promote': False})
    c.dispatch({'command': 'user_profile_chat_set', 'profile_id': second, 'document': {'messages': ['private synthetic chat']}})
    c.configure_preferences({'global_context_tokens': 8192})
    c.dispatch({'command': 'user_profile_switch', 'profile_id': first})
    assert c.capsules.policy['autonomy'] == 'situational'
    assert c.capsules.policy['allow_promote'] is True
    assert c.preferences['global_context_tokens'] == 32768
    assert c.dispatch({'command': 'user_profile_chat_get', 'profile_id': first})['document'] == {}
    prior_revision = c.capsules.policy['revision']
    c.dispatch({'command': 'user_profile_switch', 'profile_id': second})
    assert c.capsules.policy['autonomy'] == 'off'
    assert c.capsules.policy['revision'] > prior_revision
    assert c.dispatch({'command': 'user_profile_chat_get', 'profile_id': second})['document']['messages'] == ['private synthetic chat']
    document = c.dispatch({'command': 'user_profile_export', 'profile_id': first})['document']
    assert 'revision' not in document['capsule_policy']
    result = c.dispatch({'command': 'user_profile_import', 'document': document})
    imported = next(row for row in result['profiles'] if row['imported'])
    c.dispatch({'command': 'user_profile_switch', 'profile_id': imported['id']})
    assert c.capsules.policy['autonomy'] == 'off' and not c.capsules.policy['allow_promote']
    # A synthetic preparing flag is enough to enforce the real controller's launch guard; no engine starts.
    row = c.user_profiles._read(second)
    row['settings']['context'] = 2048
    c.user_profiles._write(row)
    monkeypatch.setattr(c.primary, 'preparing', lambda: True)
    try:
        with pytest.raises(ValueError, match='Stop running engines'):
            c.dispatch({'command': 'user_profile_switch', 'profile_id': second})
    finally:
        monkeypatch.setattr(c.primary, 'preparing', lambda: False)


def test_no_model_engine_commands_reject_unknown_profiles_without_work(controller, monkeypatch):
    monkeypatch.setattr(controller.primary.runner_manager, "ensure", lambda *_args, **_kwargs: pytest.fail("Runner preparation must not start"))
    for command in ("profile_start", "profile_stop"):
        with pytest.raises(ValueError):
            controller.dispatch({"command": command, "profile_id": "missing-profile"})
    assert controller.primary.child is None
    assert not controller.primary.preparing()
    assert controller.pool.users == 0
    assert controller.store.jobs() == []


def test_library_empty_missing_and_stale_paths_fail_without_preparation(controller, tmp_path, monkeypatch):
    monkeypatch.setattr(controller.primary.runner_manager, "ensure", lambda *_args, **_kwargs: pytest.fail("Runner preparation must not start"))
    library = controller.primary.library
    for raw, identity in ((str(library.root / "missing"), "stale"), (str(library.root), "stale"),
                          (str(library.root / "synthetic-text"), "stale")):
        with pytest.raises(ValueError):
            controller.primary.begin_start(raw, identity)
    assert controller.primary.preparation_thread is None
    empty = tmp_path / "empty-library"
    empty.mkdir()
    library.refresh(empty)
    deadline = time.monotonic() + 2
    while library.scanning and time.monotonic() < deadline:
        time.sleep(0.01)
    assert library.snapshot()["rows"] == []
    assert library.snapshot()["error"] is None
    library.refresh(tmp_path / "absent-library")
    deadline = time.monotonic() + 2
    while library.scanning and time.monotonic() < deadline:
        time.sleep(0.01)
    assert library.snapshot()["rows"] == []
    assert library.snapshot()["error"]
    assert not controller.primary.preparing()
    assert controller.primary.child is None
    assert controller.pool.users == 0


def test_fleet_rotation_restricts_existing_key_file_permissions(controller, monkeypatch):
    import hashlib

    from mlx2 import inference_monitor
    from mlx2.desktop_fleet import DesktopFleet
    fleet = DesktopFleet(controller)
    fleet.node = SimpleNamespace(_desktop_set_shares=lambda *_args: None)
    endpoint = SimpleNamespace(url="http://127.0.0.1:18401", pid=78)
    controller.preferences["shared_servers"] = [{"url": endpoint.url, "pid": endpoint.pid}]
    monkeypatch.setattr(inference_monitor, "discover_servers", lambda: [endpoint])
    monkeypatch.setattr(inference_monitor, "local_server_api_key", lambda _endpoint: "ROTATED-SYNTHETIC-KEY")
    folder = controller.home / "community-keys"
    folder.mkdir()
    key = folder / hashlib.sha256(f"{endpoint.url}|{endpoint.pid}".encode()).hexdigest()
    key.write_text("old")
    key.chmod(0o644)
    fleet.sync()
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert key.read_text() == "ROTATED-SYNTHETIC-KEY"


def test_wakeup_tool_requires_approval_and_is_bound_to_native_branch(controller):
    target = {'url': 'http://127.0.0.1:18285', 'pid': 42}
    session = controller.dispatch({'command': 'chat_tools', 'target': target, 'chat_session_id': 'synthetic-branch'})['session_id']
    controller.primary.worker = {'url': target['url'], 'checkpoint': 'synthetic@fingerprint', 'resume_lineage_id': 'synthetic-lineage', 'state': 'Ready',
                                'execution_kind': 'autoregressive', 'apc_persist_dir': str(controller.home / 'synthetic-apc')}
    args = {'message': 'Synthetic wakeup', 'every_seconds': 3600, 'at': None, 'suspend_after': True}
    pending = tool(controller, session, 'wakeup_schedule', args)
    assert pending['requires_approval'] is True
    assert controller.wakeups.snapshot()['total'] == 0
    accepted = tool(controller, session, 'wakeup_schedule', args, approved=True)
    assert accepted['result']['session_id'] == 'synthetic-branch'
    assert controller.wakeups.snapshot()['total'] == 1
    with pytest.raises(ValueError):
        tool(controller, session, 'wakeup_claim', {'wakeup_id': accepted['result']['wakeup_id']}, approved=True)
    other = controller.dispatch({'command': 'chat_tools', 'target': target, 'chat_session_id': 'other-branch'})['session_id']
    assert tool(controller, other, 'wakeup_list', {'offset': 0, 'limit': 10})['result']['total'] == 0
    with pytest.raises(ValueError):
        tool(controller, other, 'wakeup_cancel', {'wakeup_id': accepted['result']['wakeup_id']}, approved=True)


def test_session_tools_never_expose_sibling_archive_or_private_upload(controller):
    branch, sibling = 'synthetic-self', 'synthetic-sibling'
    profile_id = controller.user_profiles.active
    own = controller.sessions.archive(profile_id, branch, {'messages': ['SELF-CANARY sharedterm']})
    controller.sessions.archive(profile_id, sibling, {'messages': ['SIBLING-CANARY sharedterm']})
    session = controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42}, 'chat_session_id': branch})['session_id']
    results = tool(controller, session, 'session_search', {'query': 'sharedterm'})
    assert 'SELF-CANARY' in json.dumps(results) and 'SIBLING-CANARY' not in json.dumps(results)
    with pytest.raises(ValueError):
        tool(controller, session, 'session_get', {'session_id': sibling})
    with pytest.raises(ValueError):
        tool(controller, session, 'session_archive_begin', {'session_id': branch, 'total_bytes': 2, 'sha256': 'a' * 64}, approved=True)
    from mlx2.desktop_sessions import FIELDS
    summary = {field: 'Synthetic compact checkpoint' for field in FIELDS}
    compacted = tool(controller, session, 'session_compact', {'expected_revision': own['revision'], 'expected_digest': own['digest'], 'summary': summary})
    assert compacted['native_action'] == 'session_compact'
    assert compacted['checkpoint']['original_retained'] is True
    assert controller.sessions.latest(profile_id, branch)['revision'] == own['revision']


def test_new_engine_controls_refuse_unknown_owned_instances(controller):
    for command in ('engine_suspend', 'engine_resume', 'engine_cache_branches'):
        with pytest.raises(ValueError, match='Unknown app-managed'):
            controller.dispatch({'command': command, 'profile_id': 'not-owned'})
    assert controller.primary.child is None


def test_named_lora_chat_target_requires_and_returns_runtime_fingerprint(controller, monkeypatch):
    from mlx2 import inference_monitor
    status = {'model': 'base', 'artifact': 'base-fingerprint', 'runtime': {'source_sha256': 'source'},
              'multi_lora': {'enabled': True, 'registered': ['adapter'], 'registered_fingerprints': {'adapter': 'adapter-fingerprint'}}}
    monkeypatch.setattr(inference_monitor, '_get', lambda *args: json.dumps(status).encode())
    target = controller.chat_target({'url': 'http://127.0.0.1:18285', 'pid': 42, 'requested_model': 'adapter'})
    assert target['model'] == 'adapter' and target['lora_fingerprint'] == 'adapter-fingerprint'
    status['multi_lora'].pop('registered_fingerprints')
    with pytest.raises(ValueError, match='fingerprint'):
        controller.chat_target({'url': 'http://127.0.0.1:18285', 'pid': 42, 'requested_model': 'adapter'})


@pytest.mark.parametrize("inherit", [True, False])
def test_agent_launch_tool_consumes_context_option_before_manager(controller, monkeypatch, inherit):
    session = controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                   'chat_session_id': 'parent-branch'})['session_id']
    calls = []
    def capture(request, *, principal):
        assert 'inherit_context' not in request
        assert request['parent_id'] == 'parent-branch'
        assert ('_history' in request) is inherit
        assert principal
        calls.append(request)
        return {'job_id': 'synthetic-launch'}
    monkeypatch.setattr(controller.agent_launches, 'dispatch', capture)
    arguments = {'path': '/synthetic/model', 'identity': 'synthetic-identity', 'goal': 'One bounded goal',
                 'instance': 'new', 'inherit_context': inherit}
    history = [{'role': 'user', 'content': 'Synthetic parent prompt'}]
    pending = tool(controller, session, 'agent_launch', arguments, history=history)
    assert pending['requires_approval'] and not calls
    result = tool(controller, session, 'agent_launch', arguments, history=history, approved=True)
    assert result['result']['job_id'] == 'synthetic-launch' and len(calls) == 1


def test_child_tools_cannot_exceed_parent_inventory_or_other_branch_grants(controller, monkeypatch):
    first = chat_session(controller)
    arguments = {'path': '/synthetic/model', 'identity': 'synthetic-identity', 'goal': 'One goal',
                 'instance': 'new', 'inherit_context': False, 'tools': ['skill_manage']}
    with pytest.raises(ValueError, match="parent's"):
        tool(controller, first, 'agent_launch', arguments, approved=True)
    controller.preferences['memory_enabled'] = False
    with pytest.raises(ValueError, match="parent's"):
        tool(controller, first, 'agent_launch', {**arguments, 'tools': ['memory_get']}, approved=True)
    monkeypatch.setattr(controller.agent_launches, 'dispatch', lambda *a, **k: {
        'id': 'launch', 'agent': {'job_id': 'child', 'session_id': 'child', 'orchestrator_id': 'orchestrator:child', 'worker_ids': ['worker']}})
    tool(controller, first, 'agent_launch_status', {'job_id': 'launch'})
    assert 'child' in controller.tool_sessions[first]['context']['agent_job_ids']
    assert controller.tool_sessions[first]['context']['mailboxes']['child']['principal_id'] == 'orchestrator:child'
    other = controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                 'chat_session_id': 'unrelated-branch'})['session_id']
    with pytest.raises(ValueError, match='outside this conversation'):
        tool(controller, other, 'agent_resume', {'job_id': 'child', 'message': 'Continue'}, approved=True)


def test_plain_profile_never_inherits_primary_lora_settings(controller, tmp_path):
    adapter = tmp_path / "primary-lora"
    adapter.mkdir()
    controller.primary.settings.update(route="ordinary", lora_root=str(adapter), max_loras=4, max_lora_rank=64)
    saved = profile(controller, route="ordinary")
    controller.configure_preferences({"profiles": [saved]})
    engine = SimpleNamespace(child=None, worker=None, preparing=lambda: False,
        begin_start=lambda path, identity: {"state": "Preparing"})
    controller.engines[saved["id"]] = engine
    try:
        controller.start_profile(saved["id"])
        assert not {"lora_root", "max_loras", "max_lora_rank"} & engine.settings.keys()
    finally:
        controller.engines.pop(saved["id"])


def test_sage_canonical_spawn_inventory_and_approval(controller):
    session = controller.dispatch({"command": "chat_tools", "target": {"url": "http://127.0.0.1:18285", "pid": 42}})
    tools = session["tools"]
    spawn = next(tool for tool in tools if tool["function"]["name"] == "spawn_subagent")
    assert spawn["function"]["parameters"]["properties"]["tasks"]["maxItems"] == 5
    request = {"command": "tool_call", "session_id": session["session_id"], "name": "spawn_subagent",
               "arguments": {"tasks": [{"task": "x", "context": "c", "expected_output": "e"}]}}
    receipt = controller.dispatch(request)
    assert receipt["requires_approval"] is True
    assert len(tools) <= 64, f"Native chat exposes {len(tools)} tools but the server accepts at most 64"


def test_native_tool_discovery_pages_without_losing_spawn(controller):
    session = controller.dispatch({"command": "chat_tools", "target": {"url": "http://127.0.0.1:18285", "pid": 42}})
    assert session["total_tools"] > len(session["tools"])
    result = controller.dispatch({"command": "tool_call", "session_id": session["session_id"],
        "name": "mlx2__tool_search", "arguments": {"query": "world"}})
    assert result["native_action"] == "tool_inventory"
    names = {t["function"]["name"] for t in result["tools"]}
    assert "spawn_subagent" in names and "mlx2__world_export" in names
    assert len(names) <= 64
    assert not result.get("requires_approval")


def startup_control(*, resumed=False):
    instance = desktop_control.DesktopController.__new__(desktop_control.DesktopController)
    instance.lifecycle = SimpleNamespace(startup_resume=lambda: resumed)
    instance.preferences = {'community_enabled': False, 'profiles': []}
    instance.stopping = threading.Event()
    instance.engines = {}
    instance.startup_events = []
    instance.event = lambda action, **fields: instance.startup_events.append({'action': action, **fields})
    return instance


@pytest.mark.parametrize('resumed', [False, True])
def test_saved_app_autostart_runs_after_normal_or_restart_startup(resumed):
    instance = startup_control(resumed=resumed)
    instance.preferences['profiles'] = [{'id': 'startup', 'autostart': True}, {'id': 'manual', 'autostart': False}]
    starts = []
    def start(identity):
        starts.append(identity)
        engine = SimpleNamespace(worker={'state': 'Loading'}, worker_error=None, child=None, preparing=lambda: False)
        engine.snapshot = lambda: engine.worker.update(state='Ready')
        instance.engines[identity] = engine
    instance.start_profile = start
    instance.startup()
    assert starts == ['startup']
    assert instance.startup_events == [{'action': 'autostart_ready', 'profile_id': 'startup'}]


def test_restart_restored_and_explicitly_suspended_profiles_are_not_started_twice():
    instance = startup_control(resumed=True)
    instance.preferences['profiles'] = [{'id': key, 'autostart': True} for key in ['restored', 'paused', 'other']]
    instance.engines = {'restored': SimpleNamespace(worker={'state': 'Ready'}, child=object(), preparing=lambda: False),
                        'paused': SimpleNamespace(worker={'state': 'Suspended'}, child=None, preparing=lambda: False)}
    starts = []
    def start(identity):
        starts.append(identity)
        instance.engines[identity] = SimpleNamespace(worker={'state': 'Ready'}, worker_error=None,
            preparing=lambda: False, snapshot=lambda: None)
    instance.start_profile = start
    instance.startup()
    assert starts == ['other']


def test_startup_observes_readiness_without_ui_and_reports_async_failure_before_next_profile():
    instance = startup_control()
    instance.preferences['profiles'] = [{'id': key, 'autostart': True} for key in ['first', 'second']]
    order = []
    def start(identity):
        order.append('start:' + identity)
        engine = SimpleNamespace(worker={'state': 'Loading'}, worker_error='Synthetic runner admission failure' if identity == 'first' else None,
            preparing=lambda: False)
        def snapshot():
            order.append('snapshot:' + identity)
            engine.worker['state'] = 'Error' if identity == 'first' else 'Ready'
        engine.snapshot = snapshot
        instance.engines[identity] = engine
    instance.start_profile = start
    instance.startup()
    assert order == ['start:first', 'snapshot:first', 'start:second', 'snapshot:second']
    assert instance.startup_events == [
        {'action': 'autostart_failed', 'profile_id': 'first', 'error': 'Synthetic runner admission failure'},
        {'action': 'autostart_ready', 'profile_id': 'second'}]


def test_startup_timeout_defers_remaining_models_without_starting_them(monkeypatch):
    instance = startup_control()
    instance.preferences['profiles'] = [{'id': key, 'autostart': True} for key in ['first', 'second']]
    starts = []
    def start(identity):
        starts.append(identity)
        instance.engines[identity] = SimpleNamespace(worker={'state': 'Loading'}, worker_error=None,
            preparing=lambda: False, snapshot=lambda: None)
    instance.start_profile = start
    ticks = iter([0, 700])
    monkeypatch.setattr(desktop_control.time, 'monotonic', lambda: next(ticks))
    instance.startup()
    assert starts == ['first']
    assert instance.startup_events[0]['action'] == 'autostart_failed'
    assert 'timed out' in instance.startup_events[0]['error']


def test_tool_result_policy_caps_full_wire_and_persists_for_branch(controller, monkeypatch):
    controller.preferences['agent_delegation_enabled'] = False
    first = controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                 'chat_session_id': 'bounded-results'})['session_id']
    result = tool(controller, first, 'tool_result_policy', {'result_max_tokens': 1024, 'chunk_tokens': 256})
    assert result['result_context_max_bytes'] == 1024
    monkeypatch.setattr(controller, 'snapshot', lambda: {'oversized': '\u00e9' * 50000, 'workers': [], 'library': {'rows': []}})
    response = tool(controller, first, 'snapshot', {})
    assert len(json.dumps(response, ensure_ascii=False).encode()) <= 1024
    assert response['summary'].startswith('Application snapshot retrieved successfully.')
    assert response['result']['truncated'] is True
    identity = response['result']['result_id']
    page = tool(controller, first, 'tool_result_read', {'result_id': identity, 'offset': 0, 'max_tokens': 99999})
    assert len(json.dumps(page, ensure_ascii=False).encode()) <= 1024
    assert page['result']['next_offset'] <= 256
    second = controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                  'chat_session_id': 'bounded-results'})['session_id']
    assert controller.tool_sessions[second]['result_policy']['result_max_tokens'] == 1024
    third = controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                 'chat_session_id': 'other-branch'})['session_id']
    with pytest.raises(ValueError, match='outside'):
        tool(controller, third, 'tool_result_read', {'result_id': identity, 'offset': 0, 'max_tokens': 256})


def test_unrelated_preferences_and_prompt_install_survive_stale_share(controller, monkeypatch):
    from mlx2 import inference_monitor
    stale = {"url": "http://127.0.0.1:18289", "pid": 999999}
    controller.preferences["shared_servers"] = [stale]
    monkeypatch.setattr(inference_monitor, "discover_servers", list)
    result = controller.dispatch({"command": "prompt_install", "preset": "full", "activate": True})
    assert result["activated"]
    assert controller.preferences["shared_servers"] == [stale]
    controller.configure_preferences({"built_in_planning_enabled": False})
    with pytest.raises(ValueError, match="currently discovered"):
        controller.configure_preferences({"shared_servers": [stale, {"url": stale["url"], "pid": 999998}]})
    controller.configure_preferences({"shared_servers": []})
    assert not controller.preferences["shared_servers"]


def test_builtin_web_tools_require_no_mcp_or_approval(controller, monkeypatch):
    controller.preferences.update(agent_controls=True, agent_delegation_enabled=True, mcp_servers=[])
    calls = []
    monkeypatch.setattr(controller.web, 'dispatch', lambda request: pytest.fail('Parent must not perform web requests'))
    def agents(request):
        calls.append(request)
        return {'job_id': 'research-child'} if request['command'] == 'agent_delegate' else {'status': 'running'}
    monkeypatch.setattr(controller.agents, 'dispatch', agents)
    session = chat_session(controller)
    controller.tool_sessions[session]['task_goal'] = 'Explain isinstance in one sentence.'
    for name, arguments in [('web_search', {'query': 'synthetic'}), ('fetch_page', {'url': 'https://example.com'})]:
        result = tool(controller, session, name, arguments)
        assert 'requires_approval' not in result
        assert result['result']['research_delegated'] and result['result']['job_id'] == 'research-child'
    delegates = [call for call in calls if call['command'] == 'agent_delegate']
    assert len(delegates) == 1  # A different follow-up query cannot spawn another child in this turn.
    assert 'Explain isinstance in one sentence.' in delegates[0]['goals'][0]
    assert all(call['target'] == controller.tool_sessions[session]['selection'] for call in delegates)
    assert all(set(call['tools']) == {'web_search', 'fetch_page'} and call['_research'] for call in delegates)
    with pytest.raises(ValueError, match='HTTP'):
        tool(controller, session, 'fetch_page', {'url': 'file:///etc/passwd'})
    controller.preferences['agent_delegation_enabled'] = False
    with pytest.raises(ValueError, match='No parent network fallback'):
        tool(controller, session, 'web_search', {'query': 'synthetic'})


def test_operator_can_read_finished_child_history_with_delegation_disabled(controller, monkeypatch):
    monkeypatch.setattr(controller.agents, '_completion', lambda *args: 'finished report')
    monkeypatch.setattr(controller.agents, '_request', lambda *args: {'content': 'finished report', '_output_tokens': 1})
    receipt = controller.agents.dispatch({'command': 'agent_delegate', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                         'goals': ['Report own work']})
    controller.agents.jobs[receipt['job_id']]['thread'].join(3)
    controller.preferences['agent_delegation_enabled'] = False
    session = controller.dispatch({'command': 'agent_session', 'job_id': receipt['job_id']})
    assert session['messages'][-1]['content'] == 'finished report'
    with pytest.raises(ValueError, match='disabled'):
        controller.dispatch({'command': 'agent_delegate', 'goals': ['New work']})
    controller.user_profiles.active = 'another-user'
    with pytest.raises(ValueError, match='Unknown agent'):
        controller.dispatch({'command': 'agent_session', 'job_id': receipt['job_id']})


def test_parent_research_delegates_once_same_model_and_child_fetches_sources(controller, monkeypatch):
    controller.preferences.update(agent_controls=True, agent_delegation_enabled=True)
    response = controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                    'chat_session_id': 'research-parent'})
    session = response['session_id']
    requests, network = [], []

    def web(request):
        assert threading.current_thread().name.startswith('mlx2-worker')
        network.append(request)
        if request['command'] == 'web_search':
            return {'results': [{'url': 'https://example.com/source', 'title': 'Source lead'}]}
        return {'url': request['url'], 'status': '200', 'content': 'Fetched evidence'}

    def completion(job, messages, tools=None):
        requests.append(messages)
        assert {row['function']['name'] for row in tools} == {'web_search', 'fetch_page'}
        index = len(requests) - 1
        if index == 3:
            assert 'Fetched evidence' in messages[-1]['content']
            return {'content': 'Fetched evidence: https://example.com/source', '_output_tokens': 1}
        name, arguments = [('web_search', {'query': 'source'}), ('web_search', {'query': 'blocked repeat'}),
                           ('fetch_page', {'url': 'https://example.com/source'})][index]
        return {'content': '', '_output_tokens': 1, 'tool_calls': [{'id': f'call-{index}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}

    monkeypatch.setattr(controller.web, 'dispatch', web)
    monkeypatch.setattr(controller.agents, '_request', completion)
    first = tool(controller, session, 'web_search', {'query': 'source'})['result']
    job = controller.agents.jobs[first['job_id']]
    job['thread'].join(4)
    assert job['status'] == 'completed', job
    assert job['target_identity']['checkpoint'] == controller.tool_sessions[session]['target']['checkpoint']
    assert job['target_identity']['engine'] == controller.tool_sessions[session]['selection']
    assert job['delivery_state'] == 'pending' and job['research_plan']
    assert [row['command'] for row in network] == ['web_search', 'fetch_page']
    repeated = tool(controller, session, 'web_search', {'query': 'source'})['result']
    assert repeated['reused'] and repeated['synthesis'].startswith('Fetched evidence')
    changed = tool(controller, session, 'web_search', {'query': 'different research'})['result']
    assert changed['job_id'] == first['job_id'] and changed['request_changed']
    assert len(controller.agents.jobs) == 1 and len(network) == 2


def test_parent_encyclopedia_read_delegates_with_existing_principal_scope(controller, monkeypatch):
    controller.preferences.update(agent_controls=True, agent_delegation_enabled=True)
    session = chat_session(controller)
    principal = controller.tool_sessions[session]['context']['principal_id']
    identity = 'a' * 32
    controller.research.sources[identity] = {'id': identity, 'searched_by': principal, 'url': 'https://en.wikipedia.org/?curid=1'}
    calls = []

    def research(request, **kwargs):
        assert threading.current_thread().name.startswith('mlx2-worker')
        calls.append((request, kwargs))
        return {'source_id': identity, 'extract': 'Read source evidence'}

    def completion(job, messages, tools=None):
        if messages[-1]['role'] == 'tool':
            return {'content': 'Read source evidence', '_output_tokens': 1}
        return {'content': '', '_output_tokens': 1, 'tool_calls': [{'id': 'read', 'type': 'function',
                'function': {'name': 'research_read', 'arguments': json.dumps({'source_id': identity})}}]}

    monkeypatch.setattr(controller.research, 'dispatch', research)
    monkeypatch.setattr(controller.agents, '_request', completion)
    result = tool(controller, session, 'research_read', {'source_id': identity}, approved=True)['result']
    job = controller.agents.jobs[result['job_id']]
    job['thread'].join(3)
    assert job['status'] == 'completed' and calls[0][1]['principal'] == principal
    controller.research.sources[identity]['searched_by'] = 'different-principal'
    with pytest.raises(PermissionError, match='different principal'):
        tool(controller, session, 'research_read', {'source_id': identity}, approved=True)


def test_parent_and_child_use_scoped_builtin_mailboxes(controller, monkeypatch):
    controller.preferences.update(agent_delegation_enabled=True, memory_enabled=True)
    release = threading.Event()
    monkeypatch.setattr(controller.agents, '_completion', lambda *args: release.wait(3) and 'done')
    session = chat_session(controller)
    try:
        receipt = tool(controller, session, 'agent_delegate', {'goals': ['bounded work']}, approved=True)['result']
        job = controller.agents.jobs[receipt['job_id']]
        worker = job['workers'][0]
        context = controller.agents._worker_context(job, worker, [{'role': 'system', 'content': ''}])
        posted = controller.memory.dispatch({'command': 'message_post', 'session_id': job['id'],
            'recipient': receipt['orchestrator_id'], 'body': 'Need a source URL'}, context=context)
        messages = controller.dispatch({'command': 'tool_call', 'session_id': session, 'name': 'message_read', 'arguments': {'session_id': job['id']}})['result']
        assert 'Need a source URL' in str(messages)
        controller.dispatch({'command': 'tool_call', 'session_id': session, 'name': 'message_post', 'arguments': {'session_id': job['id'], 'recipient': worker['id'],
            'body': 'Use the admitted source', 'reply_to': posted['id']}})
        child = controller.memory.dispatch({'command': 'message_read', 'session_id': job['id']}, context=context)
        assert 'Use the admitted source' in str(child)
    finally:
        release.set()


def test_canonical_spawn_returns_before_completion_and_schedules_callback(controller, monkeypatch):
    controller.preferences.update(agent_delegation_enabled=True, memory_enabled=True)
    entered, release = threading.Event(), threading.Event()
    def completion(*args):
        entered.set()
        assert release.wait(3)
        raise RuntimeError("synthetic child failure")
    monkeypatch.setattr(controller.agents, '_request', completion)
    session = controller.dispatch({'command': 'chat_tools', 'target': {'url': 'http://127.0.0.1:18285', 'pid': 42},
                                   'chat_session_id': 'async-spawn-parent'})['session_id']
    request = {'command': 'tool_call', 'session_id': session, 'name': 'spawn_subagent', 'approved': True,
               'arguments': {'tasks': [{'task': 'Bounded work', 'context': 'Synthetic input', 'expected_output': 'Report'}]}}
    try:
        response = controller.dispatch(request)
        assert entered.wait(1)
        job = controller.agents.jobs[response['job_id']]
        assert job['status'] == 'running' and response['result']['completion_delivery'] == 'automatic'
        duplicate = controller.dispatch(request)
        assert duplicate['job_id'] == response['job_id'] and duplicate['result']['reused']
        assert {'message_post', 'message_read', 'message_ack'} <= set(job['tools'])
    finally:
        release.set()
    job['thread'].join(3)
    assert job['status'] == 'failed' and job['delivery_state'] == 'pending'
    claim = controller.dispatch({'command': 'agent_completion_claim', 'job_id': job['id'], 'parent_id': 'async-spawn-parent'})
    assert claim['status'] == 'failed' and claim['workers'][0]['failure']['cause'] == 'RuntimeError'


def test_pending_automatic_child_read_yields_parent_for_callback(controller, monkeypatch):
    controller.preferences['agent_delegation_enabled'] = True
    session = chat_session(controller)
    context = controller.tool_sessions[session]
    context['chat_session_id'] = 'chat:parent'
    context['context'].setdefault('agent_job_ids', []).append('child')
    job = {'id': 'child', 'parent_id': 'chat:parent', 'notify_parent': True,
           'status': 'running', 'workers': []}
    monkeypatch.setattr(controller.agents, 'dispatch', lambda request: dict(job))
    result = tool(controller, session, 'agent_result', {'job_id': 'child'})
    assert result['native_action'] == 'wait_for_agent'
    job['status'] = 'completed'
    assert 'native_action' not in tool(controller, session, 'agent_result', {'job_id': 'child'})
    job.update(status='running', parent_id='chat:other')
    assert 'native_action' not in tool(controller, session, 'agent_result', {'job_id': 'child'})


def test_chat_target_clamps_reasoning_after_context_reduces_output(controller, monkeypatch):
    from mlx2 import inference_monitor
    status = {"model": "synthetic", "max_context": 65536, "capabilities": [],
              "chat_template_options": {"recommended_chat": {"output_tokens": 65536, "thinking_budget": 32768}}}
    monkeypatch.setattr(inference_monitor, '_get', lambda *args: json.dumps(status))
    target = controller.chat_target({'url': 'http://127.0.0.1:18285'})
    assert target['chat_options']['thinking_budget'] <= max(0, target['output_tokens'] - 1024)


def test_shutdown_releases_all_owners_even_when_first_subsystem_raises():
    from mlx2.desktop_control import DesktopController
    calls = []
    def component(name, fails=False):
        def close():
            calls.append(name)
            if fails:
                raise RuntimeError('synthetic failure')
        return SimpleNamespace(close=close)
    fake = SimpleNamespace(stopping=threading.Event(), startup_thread=SimpleNamespace(join=lambda **kwargs: None))
    names = ('wakeups web agent_launches user_profiles dream lifecycle downloads agents research_plans skills '
             'sessions planning capsules memory code helpers campaign media node store').split()
    for name in names:
        setattr(fake, name, component(name, name == 'wakeups'))
    fake.engines = {'one': component('engine')}
    with pytest.raises(RuntimeError, match='shutdown failed'):
        DesktopController.close(fake)
    assert set(calls) == set(names) | {'engine'}
