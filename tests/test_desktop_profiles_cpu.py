import copy
import json
import stat
from types import SimpleNamespace

import pytest

from mlx2.desktop_choices import Choices
from mlx2.desktop_profiles import UserProfiles
from mlx2.desktop_world import Worlds


def controller(tmp_path):
    c = SimpleNamespace(home=tmp_path, preferences={
        'hf_token_file': '/secret/token', 'mcp_servers': [{'token': 'SECRET'}],
        'media_services': [], 'shared_servers': [], 'profiles': [], 'community_enabled': True,
        'default_share': True, 'helper_enabled': True, 'helper_codex_enabled': True,
        'helper_claude_enabled': True, 'system_prompts': [], 'global_context_tokens': 32768,
    }, engines={})
    c.world = Worlds(c)
    c.choices = Choices(c)
    policy = {'enabled': False, 'target': None}
    def dream_dispatch(r):
        policy.update(r['changes'])
    c.dream = SimpleNamespace(policy=policy, validate_policy=lambda v: v, dispatch=dream_dispatch)
    def configure(changes):
        if changes.get('global_context_tokens', 32768) < 1024:
            raise ValueError('context budget')
        c.preferences.update(copy.deepcopy(changes))
    c.configure_preferences = configure
    return c


def test_create_autosave_switch_restart_and_private_files(tmp_path):
    c = controller(tmp_path)
    profiles = UserProfiles(c)
    first = profiles.active
    profiles.dispatch({'command': 'user_profile_create', 'name': 'Writing'})
    second = profiles.active
    c.preferences['global_context_tokens'] = 8192
    profiles.autosave()
    profiles.close()  # flushes pending debounce
    assert stat.S_IMODE((profiles.home / (second + '.json')).stat().st_mode) == 0o600
    restored = UserProfiles(c)
    assert restored.active == second
    restored.dispatch({'command': 'user_profile_switch', 'profile_id': first})
    assert c.preferences['global_context_tokens'] == 32768
    restored.dispatch({'command': 'user_profile_switch', 'profile_id': second})
    assert c.preferences['global_context_tokens'] == 8192
    restored.close()


def test_portable_no_credentials_paths_or_execution_authority(tmp_path):
    c = controller(tmp_path)
    c.world.active['objects'][0]['resource'] = '/Users/private/personal.txt'
    profiles = UserProfiles(c)
    document = profiles.dispatch({'command': 'user_profile_export', 'profile_id': profiles.active})['document']
    encoded = json.dumps(document)
    assert 'SECRET' not in encoded and '/secret' not in encoded and '/Users/private' not in encoded
    result = profiles.dispatch({'command': 'user_profile_import', 'document': document})
    imported = next(r for r in result['profiles'] if r['imported'])
    assert profiles.active != imported['id']  # import is staged only
    profiles.dispatch({'command': 'user_profile_switch', 'profile_id': imported['id']})
    assert not c.preferences['community_enabled'] and not c.preferences['helper_enabled']
    assert c.preferences['mcp_servers'] == [] and c.preferences['hf_token_file'] == ''
    assert not c.world.policy['autonomous_scene'] and not c.dream.policy['enabled']
    requested = profiles._read(imported['id'])['requested_import']['preferences']
    assert requested['community_enabled'] is True
    profiles.close()


def test_import_rejects_credentials_and_traversal(tmp_path):
    profiles = UserProfiles(controller(tmp_path))
    doc = profiles.dispatch({'command': 'user_profile_export', 'profile_id': profiles.active})['document']
    doc['preferences']['hf_token_file'] = '/secret'
    with pytest.raises(ValueError):
        profiles.dispatch({'command': 'user_profile_import', 'document': doc})
    with pytest.raises(ValueError):
        profiles.dispatch({'command': 'user_profile_delete', 'profile_id': '../escape'})
    profiles.close()


def test_recoverable_archive_and_active_delete_refusal(tmp_path):
    profiles = UserProfiles(controller(tmp_path))
    first = profiles.active
    with pytest.raises(ValueError, match='active'):
        profiles.dispatch({'command': 'user_profile_delete', 'profile_id': first})
    profiles.dispatch({'command': 'user_profile_create', 'name': 'Second'})
    profiles.dispatch({'command': 'user_profile_delete', 'profile_id': first})
    assert first in profiles.snapshot()['archived']
    profiles.dispatch({'command': 'user_profile_restore', 'profile_id': first})
    assert len(profiles.snapshot()['profiles']) == 2
    profiles.close()


def test_switch_running_engine_profile_guard_and_validation_rollback(tmp_path):
    c = controller(tmp_path)
    profiles = UserProfiles(c)
    first = profiles.active
    profiles.dispatch({'command': 'user_profile_create', 'name': 'Other'})
    c.preferences['profiles'] = [{'id': 'different'}]
    c.engines = {'default': SimpleNamespace(child=object(), preparing=lambda: False)}
    with pytest.raises(ValueError, match='Stop running'):
        profiles.dispatch({'command': 'user_profile_switch', 'profile_id': first})
    c.engines = {}
    row = profiles._read(first)
    row['preferences']['global_context_tokens'] = 1
    profiles._write(row)
    before = copy.deepcopy(c.preferences)
    with pytest.raises(ValueError, match='context budget'):
        profiles.dispatch({'command': 'user_profile_switch', 'profile_id': first})
    assert c.preferences == before
    profiles.close()


def test_chat_is_private_profile_scoped_and_export_excluded(tmp_path):
    profiles = UserProfiles(controller(tmp_path))
    first = profiles.active
    profiles.dispatch({'command': 'user_profile_chat_set', 'profile_id': first, 'document': {'messages': ['PRIVATE CHAT']}})
    assert profiles.dispatch({'command': 'user_profile_chat_get', 'profile_id': first})['document']['messages'] == ['PRIVATE CHAT']
    export = profiles.dispatch({'command': 'user_profile_export', 'profile_id': first})
    assert 'PRIVATE CHAT' not in json.dumps(export)
    profiles.dispatch({'command': 'user_profile_create', 'name': 'Other'})
    with pytest.raises(ValueError, match='different active'):
        profiles.dispatch({'command': 'user_profile_chat_set', 'profile_id': first, 'document': {}})
    profiles.close()


@pytest.mark.parametrize('document', [[], None, 'broken', 7])
def test_malformed_active_marker_recovers_without_changing_settings(tmp_path, document):
    c = controller(tmp_path)
    home = tmp_path / 'user-profiles'
    home.mkdir()
    (home / 'active.json').write_text(json.dumps(document))
    before = copy.deepcopy(c.preferences)
    profiles = UserProfiles(c)
    assert profiles.active
    assert 'could not be recovered' in profiles.error
    assert c.preferences == before
    profiles.close()


@pytest.mark.parametrize('document', [[], None, 'broken', 7])
def test_malformed_profile_is_skipped_and_active_profile_recovers(tmp_path, document):
    c = controller(tmp_path)
    profiles = UserProfiles(c)
    identity = profiles.active
    profiles.close()
    (profiles.home / (identity + '.json')).write_text(json.dumps(document))
    recovered = UserProfiles(c)
    assert recovered.active != identity
    assert len(recovered.snapshot()['profiles']) == 1
    assert 'could not be recovered' in recovered.error
    recovered.close()


def test_malformed_private_chat_is_an_explicit_restore_error(tmp_path):
    profiles = UserProfiles(controller(tmp_path))
    (profiles.home / (profiles.active + '.chat')).write_text('[]')
    with pytest.raises(ValueError, match='Invalid private chat document'):
        profiles.dispatch({'command': 'user_profile_chat_get', 'profile_id': profiles.active})
    profiles.close()


def test_portable_profile_with_local_lora_settings_roundtrips_without_adapter_paths(tmp_path):
    c = controller(tmp_path)
    c.primary = SimpleNamespace(settings={'root': '/private/models', 'port': 8285, 'context': 8192, 'lanes': 1, 'route': 'ordinary',
        'sticky_port': True, 'lora_root': '/private/adapters', 'max_loras': 1, 'max_lora_rank': 64})
    profiles = UserProfiles(c)
    document = profiles.dispatch({'command': 'user_profile_export', 'profile_id': profiles.active})['document']
    assert '/private/adapters' not in json.dumps(document)
    assert 'lora_root' not in document['settings']
    before = {row['id'] for row in profiles.snapshot()['profiles']}
    result = profiles.dispatch({'command': 'user_profile_import', 'document': document})
    imported = next(row['id'] for row in result['profiles'] if row['id'] not in before)
    settings = profiles._read(imported)['settings']
    assert not {'lora_root', 'max_loras', 'max_lora_rank'} & settings.keys()
    assert settings['route'] == 'ordinary'
    profiles.close()


@pytest.mark.parametrize('manager,state_key,state', [
    ('agents', 'status', 'queued'), ('agents', 'status', 'running'), ('agents', 'status', 'synthesizing'),
    ('agent_launches', 'state', 'queued'), ('agent_launches', 'state', 'loading'), ('agent_launches', 'state', 'dispatching'), ('agent_launches', 'state', 'running'),
])
def test_active_agent_work_blocks_profile_switch_and_creation(tmp_path, manager, state_key, state):
    c = controller(tmp_path)
    profiles = UserProfiles(c)
    first = profiles.active
    profiles.dispatch({'command': 'user_profile_create', 'name': 'Second'})
    second = profiles.active
    setattr(c, manager, SimpleNamespace(jobs={'worker': {state_key: state}}))
    for request in ({'command': 'user_profile_switch', 'profile_id': first}, {'command': 'user_profile_create', 'name': 'Third'}):
        with pytest.raises(ValueError, match='active agent work'):
            profiles.dispatch(request)
        assert profiles.active == second
    getattr(c, manager).jobs['worker'][state_key] = 'completed'
    profiles.dispatch({'command': 'user_profile_switch', 'profile_id': first})
    assert profiles.active == first
    profiles.close()
