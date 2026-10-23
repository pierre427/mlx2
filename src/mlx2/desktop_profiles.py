"""Private named user settings; portable exports never carry execution authority."""
from __future__ import annotations

import copy
import json
import re
import threading
import time
import uuid

from .desktop import atomic_json

MAX_BYTES = 16 * 1024 * 1024
PORTABLE = {
    'system_prompts', 'base_system_prompt_id', 'global_context_tokens', 'context_reserve_tokens',
    'memory_enabled', 'memory_retrieval_tokens', 'built_in_planning_enabled', 'agent_delegation_enabled',
    'download_parallelism', 'download_bytes_per_second', 'agent_controls', 'helper_enabled',
    'helper_codex_enabled', 'helper_claude_enabled', 'helper_timeout', 'adapter_helper',
    'community_enabled', 'community_front_port', 'community_fabric_port', 'default_share',
}
PORTABLE_SETTINGS = {'port', 'context', 'lanes', 'route', 'sticky_port'}
DISABLED = {'community_enabled', 'default_share', 'helper_enabled', 'helper_codex_enabled', 'helper_claude_enabled'}


def bounded(value):
    data = json.dumps(value, allow_nan=False).encode()
    if len(data) > MAX_BYTES:
        raise ValueError('User profile exceeds the 16 MiB document budget')
    return copy.deepcopy(value)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{32}', value):
        raise ValueError('Invalid user profile identifier')
    return value


def name_text(value):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 100:
        raise ValueError('User profile name must contain 1–100 characters')
    return value.strip()


class UserProfiles:
    def __init__(self, controller):
        self.controller = controller
        self.home = controller.home / 'user-profiles'
        self.home.mkdir(mode=0o700, exist_ok=True)
        self.lock = threading.RLock()
        self.timer = None
        self.closed = False
        self.applying = False
        self.error = ''
        self.active = ''
        marker = self.home / 'active.json'
        if marker.exists():
            try:
                if marker.is_symlink() or marker.stat().st_size > MAX_BYTES:
                    raise ValueError('Invalid active user profile file')
                state = json.loads(marker.read_text())
                if not isinstance(state, dict):
                    raise TypeError('Invalid active user profile format')
                candidate = identifier(state['id'])
                self._read(candidate)
                self.active = candidate
            except (ValueError, TypeError, KeyError, OSError):
                self.error = 'Previous user profile could not be recovered; current settings retained'
        if not self.active:
            self.active = uuid.uuid4().hex
            self._write(self._capture(self.active, 'Default'))
            atomic_json(marker, {'id': self.active})

    def _read(self, identity):
        path = self.home / (identifier(identity) + '.json')
        if path.is_symlink() or path.stat().st_size > MAX_BYTES:
            raise ValueError('Invalid user profile file')
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or row.get('id') != identity or row.get('version') != 1:
            raise ValueError('Invalid user profile format')
        name_text(row.get('name'))
        return bounded(row)

    def _write(self, value):
        atomic_json(self.home / (identifier(value['id']) + '.json'), bounded(value))

    def _capture(self, identity, name):
        c = self.controller
        return bounded({'version': 1, 'id': identity, 'name': name, 'updated': time.time(),
                        'preferences': copy.deepcopy(c.preferences),
                        'settings': copy.deepcopy(getattr(getattr(c, 'primary', None), 'settings', {})),
                        'world': {'world': copy.deepcopy(c.world.active), 'assets': {}}, 'world_policy': dict(c.world.policy),
                        'choice_policy': dict(c.choices.policy), 'dream_policy': copy.deepcopy(c.dream.policy),
                        'capsule_policy': {k: copy.deepcopy(v) for k, v in getattr(getattr(c, 'capsules', None), 'policy', {}).items() if k != 'revision'}})

    def _save(self):
        with self.lock:
            if self.applying:
                return
            try:
                row = self._read(self.active)
                current = self._capture(self.active, row['name'])
                if 'requested_import' in row:
                    current['requested_import'] = row['requested_import']
                self._write(current)
                self.error = ''
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.error = str(exc)

    def autosave(self):
        with self.lock:
            if self.closed or self.applying:
                return
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(0.4, self._save)
            self.timer.daemon = True
            self.timer.start()

    def close(self):
        with self.lock:
            self.closed = True
            if self.timer:
                self.timer.cancel()
            self._save()

    def snapshot(self):
        with self.lock:
            rows = []
            for path in sorted(self.home.glob('*.json')):
                if path.stem == 'active':
                    continue
                try:
                    row = self._read(path.stem)
                    rows.append({k: row[k] for k in ('id', 'name', 'updated')})
                    rows[-1]['imported'] = 'requested_import' in row
                    rows[-1]['requested_import'] = row.get('requested_import', {})
                except (ValueError, OSError, KeyError, TypeError):
                    continue
            return {'active_id': self.active, 'profiles': rows, 'error': self.error,
                    'autosave': True, 'archived': [p.stem for p in self.home.glob('*.archived')]}

    @staticmethod
    def _portable_world(bundle):
        result = copy.deepcopy(bundle)
        world = result.get('world', result)
        for obj in world.get('objects', []):
            if not obj.get('resource', '').startswith('asset:'):
                obj['resource'] = ''
        return result

    def _guard_agent_activity(self):
        for attribute, active in (("agents", {"queued", "running", "synthesizing"}),
                                  ("agent_launches", {"queued", "loading", "dispatching", "running"})):
            manager = getattr(self.controller, attribute, None)
            # Copy only references and inspect state; do not take a job-manager
            # lock while holding the profile lock (workers may autosave).
            jobs = list(getattr(manager, "jobs", {}).values())
            if any(job.get("status", job.get("state")) in active for job in jobs):
                raise ValueError("Wait for or cancel active agent work before changing user profiles")

    def _apply(self, row):
        self._guard_agent_activity()
        c = self.controller
        changes = row['preferences']
        settings = row.get('settings', {})
        primary = getattr(c, 'primary', None)
        launch_changed = bool(primary and settings and settings != primary.settings)
        busy = any(getattr(engine, 'child', None) is not None or (getattr(engine, 'worker', None) or {}).get('state') in {'Suspended', 'Suspending', 'Resuming'} or (engine.preparing() if callable(getattr(engine, 'preparing', None)) else getattr(engine, 'preparing', False)) for engine in c.engines.values())
        if busy and (launch_changed or changes.get('profiles', c.preferences.get('profiles')) != c.preferences.get('profiles')):
            raise ValueError('Stop running engines before switching serving profile settings')
        if primary and settings:
            primary.valid_settings(settings)
            if not 1 <= settings['lanes'] <= 4:
                raise ValueError('Choose 1–4 serving lanes')
        # Validate policy and world before touching preferences. World import stores only digest-checked assets.
        world = c.world.import_bundle(row['world'])
        c.dream.validate_policy(row['dream_policy'])
        capsules = getattr(c, 'capsules', None)
        capsule_policy = row.get('capsule_policy', {})
        if capsules and capsule_policy:
            capsules.validate({**capsules.policy, **capsule_policy})
        previous = self._capture(self.active, self._read(self.active)['name'])
        self.applying = True
        try:
            c.configure_preferences(changes)
            if launch_changed:
                primary.configure(settings)
            c.choices.dispatch({'command': 'choice_policy', 'changes': row['choice_policy']})
            c.world.dispatch({'command': 'world_save', 'world': world})
            c.world.dispatch({'command': 'world_policy', 'changes': row['world_policy']})
            c.dream.dispatch({'command': 'dream_configure', 'changes': row['dream_policy']})
            if capsules and capsule_policy:
                capsules.configure(capsule_policy)
        except Exception:
            # Existing validators remain authoritative; best-effort restore never launches engines.
            c.configure_preferences(previous['preferences'])
            if launch_changed:
                primary.configure(previous['settings'])
            c.choices.dispatch({'command': 'choice_policy', 'changes': previous['choice_policy']})
            c.world.dispatch({'command': 'world_save', 'world': c.world.import_bundle(previous['world'])})
            c.world.dispatch({'command': 'world_policy', 'changes': previous['world_policy']})
            c.dream.dispatch({'command': 'dream_configure', 'changes': previous['dream_policy']})
            if capsules and previous.get('capsule_policy'):
                capsules.configure(previous['capsule_policy'])
            raise
        finally:
            self.applying = False
        wakeups = getattr(c, 'wakeups', None)
        if wakeups:
            wakeups.profile_departing(self.active)
        self.active = row['id']
        for key in ('tool_sessions', 'principal_contexts'):
            getattr(c, key, {}).clear()
        atomic_json(self.home / 'active.json', {'id': self.active})

    def dispatch(self, request):
        with self.lock:
            command = request['command']
            if command in {'user_profile_chat_get', 'user_profile_chat_set'}:
                identity = identifier(request['profile_id'])
                if identity != self.active:
                    raise ValueError('Chat history belongs to a different active user profile')
                path = self.home / (identity + '.chat')
                if command == 'user_profile_chat_set':
                    document = bounded(request['document'])
                    if not isinstance(document, dict):
                        raise ValueError('Chat history must be an object')
                    atomic_json(path, document)
                if path.is_symlink() or (path.exists() and path.stat().st_size > MAX_BYTES):
                    raise ValueError('Invalid private chat document')
                document = json.loads(path.read_text()) if path.exists() else {}
                if not isinstance(document, dict):
                    raise ValueError('Invalid private chat document')
                return {'profile_id': identity, 'document': document}
            if command == 'user_profile_export':
                self._save()
                row = self._read(request['profile_id'])
                document = {'format': 'mlx2-user-profile', 'version': 1, 'name': row['name'],
                            'preferences': {k: v for k, v in row['preferences'].items() if k in PORTABLE},
                            'world': self._portable_world(self.controller.world.bundle(self.controller.world.import_bundle(row['world']))), 'choice_policy': row['choice_policy'],
                            'settings': {k: v for k, v in row.get('settings', {}).items() if k in PORTABLE_SETTINGS},
                            'capsule_policy': row.get('capsule_policy', {})}
                # Prompt content is user-authored and may itself contain private text; UI labels this clearly.
                return {'document': bounded(document)}
            if command == 'user_profile_create':
                self._guard_agent_activity()
                if len(self.snapshot()['profiles']) >= 32:
                    raise ValueError('At most 32 user profiles are supported')
                self._save()
                row = self._capture(uuid.uuid4().hex, name_text(request['name']))
                self._write(row)
                wakeups = getattr(self.controller, 'wakeups', None)
                if wakeups:
                    wakeups.profile_departing(self.active)
                self.active = row['id']
                for key in ('tool_sessions', 'principal_contexts'):
                    getattr(self.controller, key, {}).clear()
                atomic_json(self.home / 'active.json', {'id': self.active})
            elif command == 'user_profile_rename':
                row = self._read(request['profile_id'])
                row.update(name=name_text(request['name']), updated=time.time())
                self._write(row)
            elif command == 'user_profile_switch':
                self._save()
                self._apply(self._read(request['profile_id']))
            elif command == 'user_profile_delete':
                identity = identifier(request['profile_id'])
                if identity == self.active:
                    raise ValueError('Switch to another user profile before deleting the active one')
                self._read(identity)
                (self.home / (identity + '.json')).replace(self.home / (identity + '.archived'))
            elif command == 'user_profile_restore':
                identity = identifier(request['profile_id'])
                if len(self.snapshot()['profiles']) >= 32:
                    raise ValueError('At most 32 user profiles are supported')
                path = self.home / (identity + '.archived')
                if path.is_symlink() or path.stat().st_size > MAX_BYTES:
                    raise ValueError('Invalid archived user profile')
                if (self.home / (identity + '.json')).exists():
                    raise ValueError('User profile already exists')
                path.replace(self.home / (identity + '.json'))
            elif command == 'user_profile_import':
                doc = bounded(request['document'])
                if not isinstance(doc, dict) or set(doc) != {'format', 'version', 'name', 'preferences', 'world', 'choice_policy', 'settings', 'capsule_policy'} or doc['format'] != 'mlx2-user-profile' or doc['version'] != 1:
                    raise ValueError('Invalid portable user profile')
                if not isinstance(doc['preferences'], dict) or set(doc['preferences']) - PORTABLE:
                    raise ValueError('Portable profiles cannot contain credentials, local paths or connection definitions')
                if len(self.snapshot()['profiles']) >= 32:
                    raise ValueError('At most 32 user profiles are supported')
                row = self._capture(uuid.uuid4().hex, name_text(doc['name']))
                if not isinstance(doc['settings'], dict) or set(doc['settings']) - PORTABLE_SETTINGS:
                    raise ValueError('Portable profiles cannot contain model directory paths')
                for key in ('lora_root', 'max_loras', 'max_lora_rank'):
                    row['settings'].pop(key, None)
                row['settings'].update(doc['settings'])
                row['preferences'].update(doc['preferences'])
                for key in DISABLED:
                    row['preferences'][key] = False
                row['preferences']['mcp_servers'] = []
                row['preferences']['media_services'] = []
                row['preferences']['shared_servers'] = []
                row['preferences']['profiles'] = []
                row['preferences']['hf_token_file'] = ''
                row['world'] = doc['world']
                row['choice_policy'] = doc['choice_policy']
                row['world_policy'] = {'autonomous_scene': False}
                row['dream_policy'].update(enabled=False, target=None)
                if not isinstance(doc['capsule_policy'], dict) or 'revision' in doc['capsule_policy']:
                    raise ValueError('Invalid portable capsule policy')
                row['capsule_policy'] = {**doc['capsule_policy'], 'autonomy': 'off', 'allow_promote': False}
                capsules = getattr(self.controller, 'capsules', None)
                if capsules:
                    capsules.validate({**capsules.policy, **row['capsule_policy']})
                row['requested_import'] = {'preferences': doc['preferences'], 'capsule_policy': doc['capsule_policy'], 'note': 'Imported execution and sharing permissions require operator re-enablement'}
                # Full validators run on explicit switch; importing never applies or executes settings.
                self.controller.world.import_bundle(row['world'])
                self._write(row)
            else:
                raise ValueError('Unknown user profile command')
            return self.snapshot()
