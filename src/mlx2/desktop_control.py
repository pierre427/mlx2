"""Typed native/agent control plane. No generic command or shell execution API."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .desktop import GPUReservation, Supervisor, atomic_json, json_safe, read_config
from .desktop_records import RecordStore

ACTIVE = {"Preparing", "Loading", "Ready", "Busy", "Stopping", "Unreachable"}


class ReservationPool:
    """One paired interactive reservation for the app's explicitly managed engines."""
    def __init__(self):
        self.lock = threading.RLock()
        self.reservation = None
        self.users = 0

    def ticket(self):
        pool = self
        class Ticket:
            acquired = False
            @property
            def owner(self):
                return pool.reservation.owner
            def acquire(self):
                with pool.lock:
                    if pool.reservation is None:
                        reservation = GPUReservation()
                        reservation.acquire()
                        pool.reservation = reservation
                    pool.users += 1
                    self.acquired = True
            def release(self):
                with pool.lock:
                    if not self.acquired:
                        return
                    if pool.users == 1:
                        pool.reservation.release()
                        pool.reservation = None
                    pool.users -= 1
                    self.acquired = False
        return Ticket()


class DesktopController:
    def __init__(self, home: Path, trash_helper: Path):
        self.home = home
        self.trash_helper = trash_helper
        self.pool = ReservationPool()
        self.primary = Supervisor(home, trash_helper,
            reservation_factory=self.pool.ticket, owned_pids=self.owned_pids, context_resolver=self.context_limit)
        self.engines = {"default": self.primary}
        self.store = RecordStore(home)
        from .desktop_badges import BadgeResolver
        self.badges = BadgeResolver(self.primary.runner_manager.source)
        self.lock = threading.RLock()
        from .desktop_disk import DiskBudget
        self.disk = DiskBudget()
        self.events = home / "events.jsonl"
        self.preferences = {
            "agent_controls": True,
            "mcp_servers": [],
            "profiles": [],
            "community_enabled": False,
            "community_home": str(Path.home() / ".mlx2/fabric"),
            "community_front_port": 8290,
            "community_fabric_port": 8291,
            "studio_url": "",
            "media_services": [],
            "conversion_python": "",
            "conversion_omlx_source": "",
            "conversion_reserve_bytes": 10 * 1024**3,
            "default_share": False,
            "shared_servers": [],
            "helper_enabled": False,
            "helper_claude_enabled": False,
            "helper_codex_enabled": False,
            "adapter_model_path": "",
            "adapter_helper": "codex",
            "helper_workspace": str(home / "helper-workspaces"),
            "helper_timeout": 900,
            "download_parallelism": 5,
            "download_bytes_per_second": 0,
            "hf_token_file": "",
            "system_prompts": [],
            "base_system_prompt_id": "",
            "panel_system_prompt_ids": {},
            "built_in_planning_enabled": True,
            "agent_delegation_enabled": True,
            "memory_enabled": True,
            "memory_retrieval_tokens": 1024,
            "global_context_tokens": 32768,
            "model_context_tokens": {},
            "model_chat_options": {},
            "context_reserve_tokens": 4096,
        }
        path = home / "desktop-preferences.json"
        if path.exists():
            self.preferences.update(read_config(path))
        self.node = None
        self.tool_sessions = {}
        self.campaign = None
        self.event("application_started")
        # Source-only imports: these modules never load a model on construction.
        from .desktop_agents import Agents
        from .desktop_campaign import CampaignQueue
        from .desktop_downloads import Downloads
        from .desktop_helpers import Helpers
        from .desktop_lifecycle import Lifecycle
        from .desktop_media import MediaController
        from .desktop_memory import MemoryController
        from .desktop_planning import Planning
        self.campaign = CampaignQueue(self)
        self.media = MediaController(self)
        self.helpers = Helpers(self)
        from .desktop_code import CodeWorkspace
        self.code = CodeWorkspace(self)
        self.downloads = Downloads(self)
        self.agents = Agents(self)
        self.planning = Planning(self)
        self.memory = MemoryController(self)
        from .desktop_capsules import CapsuleController
        self.capsules = CapsuleController(self)
        from .desktop_world import Worlds
        self.world = Worlds(self)
        from .desktop_research import Research
        self.research = Research(self)
        from .desktop_research_plans import ResearchPlans
        self.research_plans = ResearchPlans(self)
        from .desktop_web import WebTools
        self.web = WebTools()
        from .desktop_choices import Choices
        self.choices = Choices(self)
        from .desktop_dream import Dream
        self.dream = Dream(self)
        from .desktop_profiles import UserProfiles
        self.user_profiles = UserProfiles(self)
        from .desktop_skills import Skills
        self.skills = Skills(self)
        from .desktop_sessions import SessionArchive
        self.sessions = SessionArchive(self)
        from .desktop_agent_launch import AgentLaunch
        self.agent_launches = AgentLaunch(self)
        self.lifecycle = Lifecycle(self)
        self.stopping = threading.Event()
        # Restore retained profile metadata without loading a model at login.
        for profile in self.preferences["profiles"]:
            identity = profile["id"]
            folder = self.home / "engines" / identity
            if (folder / "desktop-suspension.json").is_file():
                engine = Supervisor(folder, self.trash_helper,
                    reservation_factory=self.pool.ticket, owned_pids=self.owned_pids, context_resolver=self.context_limit)
                engine.library = self.primary.library
                self.engines[identity] = engine
        from .desktop_wakeups import Wakeups
        self.wakeups = Wakeups(self)
        self.startup_thread = threading.Thread(target=self.startup, daemon=True, name="mlx2-startup")
        self.startup_thread.start()

    def event(self, action, **fields):
        # Never record prompts, tool arguments, headers or credentials.
        entry = {"time": time.time(), "action": action, **fields}
        fd = os.open(self.events, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")

    def owned_pids(self):
        return {s.child.pid for s in getattr(self, "engines", {}).values() if s.child is not None}

    def startup(self):
        # A one-use restart plan restores its own listeners; it must not
        # suppress unrelated profiles explicitly selected for app startup.
        self.lifecycle.startup_resume()
        if self.preferences["community_enabled"]:
            try:
                self.community_start()
            except Exception as error:  # noqa: BLE001 -- retain startup errors without killing the UI
                self.event("community_start_failed", error=type(error).__name__)
        for profile in list(self.preferences["profiles"]):
            if self.stopping.is_set():
                break
            if not profile.get("autostart"):
                continue
            existing = self.engines.get(profile["id"])
            if existing and (self.retained(existing) or existing.child is not None or existing.preparing()):
                continue
            try:
                self.start_profile(profile["id"])
                # Avoid overlapping runner preparation and model loads at login.
                engine = self.engines[profile["id"]]
                deadline = time.monotonic() + 660
                while not self.stopping.is_set() and time.monotonic() < deadline:
                    if engine.preparing():
                        time.sleep(0.2)
                        continue
                    # Readiness advances from the worker receipt even when the
                    # details window is closed and no UI snapshot is requested.
                    engine.snapshot()
                    if not engine.worker or engine.worker["state"] not in {"Preparing", "Loading"}:
                        break
                    time.sleep(0.2)
                if self.stopping.is_set():
                    break
                state = (engine.worker or {}).get("state")
                if engine.preparing() or state in {"Preparing", "Loading"}:
                    self.event("autostart_failed", profile_id=profile["id"], error="Model startup timed out; remaining startup models were deferred")
                    break
                if state not in {"Ready", "Busy"}:
                    raise RuntimeError(engine.worker_error or "Model startup ended in state " + str(state))
                self.event("autostart_ready", profile_id=profile["id"])
            except Exception as error:  # noqa: BLE001 -- individual startup failure must not skip later profiles
                self.event("autostart_failed", profile_id=profile["id"], error=str(error))

    def profile(self, identity):
        return next((p for p in self.preferences["profiles"] if p["id"] == identity), None)

    def start_profile(self, identity):
        profile = self.profile(identity)
        if profile is None:
            raise ValueError("Serving profile no longer exists")
        if self.campaign and self.campaign.running:
            raise ValueError("Wait for or cancel the qualification queue before loading a model")
        with self.lock:
            engine = self.engines.get(identity)
            if engine and self.retained(engine):
                raise ValueError("Resume or stop this suspended profile before starting it")
            if engine is None:
                folder = self.home / "engines" / identity
                engine = Supervisor(folder, self.trash_helper,
                    reservation_factory=self.pool.ticket, owned_pids=self.owned_pids, context_resolver=self.context_limit)
                self.engines[identity] = engine
            engine.settings = self.primary.valid_settings({
                **{k: v for k, v in self.primary.settings.items() if k not in {"lora_root", "max_loras", "max_lora_rank"}},
                **{k: profile[k] for k in ("port", "context", "lanes", "route")},
                **{k: profile[k] for k in ("lora_root", "max_loras", "max_lora_rank") if k in profile},
                "sticky_port": profile.get("sticky_port", True),
            })
            engine.library = self.primary.library
            path = Path(profile["path"])
            from .desktop import artifact_row
            row = artifact_row(path)
            result = engine.begin_start(str(path), row["identity"])
            self.event("model_start_requested", profile_id=identity, port=profile["port"])
            return result

    def configure_preferences(self, changes):
        if not isinstance(changes, dict) or set(changes) - self.preferences.keys():
            raise ValueError("Unknown application settings")
        value = {**self.preferences, **changes}
        for flag in ("agent_controls", "community_enabled", "helper_enabled", "helper_claude_enabled", "helper_codex_enabled", "default_share", "built_in_planning_enabled", "agent_delegation_enabled", "memory_enabled"):
            if type(value[flag]) is not bool:
                raise ValueError(f"{flag} must be a boolean")
        profiles = value["profiles"]
        if not isinstance(profiles, list) or len(profiles) > 32:
            raise ValueError("At most 32 serving profiles are supported")
        ids, ports = set(), set()
        for profile in profiles:
            required = {"id", "path", "port", "context", "lanes", "route", "autostart", "share"}
            if not isinstance(profile, dict) or not required <= set(profile) or set(profile) - (required | {"sticky_port", "lora_root", "max_loras", "max_lora_rank"}):
                raise ValueError("Invalid serving profile fields")
            if type(profile.get("sticky_port", True)) is not bool or type(profile["lanes"]) is not int or not 1 <= profile["lanes"] <= 4:
                raise ValueError("Choose 1–4 serving lanes and a boolean sticky-port setting")
            identity = profile["id"]
            if not isinstance(identity, str) or not identity or len(identity) > 64 or not all(c.isalnum() or c == '-' for c in identity):
                raise ValueError("Invalid profile identifier")
            if identity == "default" or identity in ids or (profile.get("sticky_port", True) and profile["port"] in ports):
                raise ValueError("Serving profiles need unique identifiers and ports")
            if type(profile["autostart"]) is not bool or type(profile["share"]) is not bool:
                raise ValueError("Startup and sharing flags must be boolean")
            old = self.profile(identity)
            if old != profile:
                path = Path(profile["path"]).expanduser().resolve(strict=True)
                if path.parent != self.primary.library.root.resolve() or not path.is_dir():
                    raise ValueError("Choose a model in the current library")
                self.primary.valid_settings({"root": str(path.parent), **{k: profile[k] for k in ("port", "context", "lanes", "route")},
                    **{k: profile[k] for k in ("lora_root", "max_loras", "max_lora_rank") if k in profile}})
            engine = self.engines.get(identity)
            launch_keys = ("path", "port", "context", "lanes", "route", "sticky_port", "lora_root", "max_loras", "max_lora_rank")
            changed_launch = old is None or any(old.get(k, True if k == "sticky_port" else None) != profile.get(k, True if k == "sticky_port" else None) for k in launch_keys)
            if changed_launch and engine and (engine.child is not None or engine.preparing() or self.retained(engine)):
                raise ValueError("Stop this profile before editing its launch settings")
            ids.add(identity)
            if profile.get("sticky_port", True):
                ports.add(profile["port"])
        for identity, engine in self.engines.items():
            if identity != "default" and identity not in ids and (engine.child is not None or engine.preparing() or self.retained(engine)):
                raise ValueError("Stop a running profile before removing it")
        from .desktop_tools import validate_mcp_settings
        validate_mcp_settings(value["mcp_servers"])
        from .desktop_media import validate_services
        validate_services(value["media_services"])
        for name in ("conversion_python", "conversion_omlx_source"):
            if not isinstance(value[name], str):
                raise TypeError("Conversion runtime settings must be paths")
        if type(value["conversion_reserve_bytes"]) is not int or value["conversion_reserve_bytes"] < 0:
            raise ValueError("Conversion free-space reserve must be a nonnegative byte count")
        if not isinstance(value["shared_servers"], list):
            raise TypeError("Shared servers must be a list")
        previous_shares = self.preferences.get("shared_servers", [])
        for server in value["shared_servers"]:
            if not isinstance(server, dict) or set(server) != {"url", "pid"} or not isinstance(server["url"], str) or type(server["pid"]) is not int or server["pid"] <= 0:
                raise ValueError("Shared engines require a URL and positive process identifier")
        added_shares = [server for server in value["shared_servers"] if server not in previous_shares]
        if added_shares:
            from .inference_monitor import discover_servers
            discovered = {(e.url, e.pid) for e in discover_servers()}
            if any((server["url"], server["pid"]) not in discovered for server in added_shares):
                raise ValueError("Only currently discovered local engines can be shared")
        # Previously admitted PID bindings may go stale after a restart. Keep
        # unrelated settings editable; fleet routing rechecks every live binding.
        if type(value["download_parallelism"]) is not int or not 1 <= value["download_parallelism"] <= 32:
            raise ValueError("Choose between 1 and 32 parallel file downloads")
        if type(value["download_bytes_per_second"]) is not int or value["download_bytes_per_second"] < 0:
            raise ValueError("Download throttle must be a nonnegative byte count per second")
        if not isinstance(value["hf_token_file"], str):
            raise TypeError("HF token file must be a path")
        if value["hf_token_file"]:
            from .http_security import load_secret_file
            load_secret_file(value["hf_token_file"], flag="HF token file")
        if type(value["helper_timeout"]) is not int or not 30 <= value["helper_timeout"] <= 7200:
            raise ValueError("Helper timeout must be 30–7200 seconds")
        if value["adapter_helper"] not in {"codex", "claude", "local"}:
            raise ValueError("Choose a local, Claude or Codex adapter helper")
        if Path(value["helper_workspace"]).expanduser().resolve() != (self.home / "helper-workspaces").resolve():
            raise ValueError("Helper workspaces must stay in this app's isolated workspace directory")
        if not isinstance(value["system_prompts"], list) or len(value["system_prompts"]) > 64:
            raise ValueError("Save at most 64 system prompts")
        prompt_ids = set()
        for prompt in value["system_prompts"]:
            if not isinstance(prompt, dict) or set(prompt) != {"id", "name", "text"} or any(not isinstance(v, str) for v in prompt.values()):
                raise ValueError("System prompts need an identifier, name and text")
            if not prompt["id"] or len(prompt["id"]) > 64 or prompt["id"] in prompt_ids or len(prompt["name"]) > 200 or len(prompt["text"]) > 65536:
                raise ValueError("Invalid or duplicate system prompt")
            prompt_ids.add(prompt["id"])
        if value["base_system_prompt_id"] not in prompt_ids | {""}:
            raise ValueError("The base system prompt must name a saved prompt")
        if not isinstance(value["panel_system_prompt_ids"], dict) or len(value["panel_system_prompt_ids"]) > 1000 or any(not isinstance(k, str) or v not in prompt_ids | {""} for k, v in value["panel_system_prompt_ids"].items()):
            raise ValueError("Panel instructions must select saved system prompts")
        for name, low, high in (("global_context_tokens", 1024, 1048576), ("context_reserve_tokens", 128, 32768), ("memory_retrieval_tokens", 64, 8192)):
            if type(value[name]) is not int or not low <= value[name] <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        if not isinstance(value["model_context_tokens"], dict) or any(not isinstance(k, str) or type(v) is not int or (v != 0 and not 1024 <= v <= 1048576) for k, v in value["model_context_tokens"].items()):
            raise ValueError("Model context limits must be 0 (global default) or 1024–1048576 tokens")
        from .agentic_chat import validate_chat_options
        if not isinstance(value["model_chat_options"], dict):
            raise TypeError("Per-model chat options must be an object")
        for path, options in value["model_chat_options"].items():
            if not isinstance(path, str):
                raise TypeError("Per-model chat options need a model path")
            validate_chat_options(options)
        for key in ("community_front_port", "community_fabric_port"):
            if type(value[key]) is not int or not 1024 <= value[key] <= 65535:
                raise ValueError("Community ports must be between 1024 and 65535")
        if value["community_front_port"] == value["community_fabric_port"]:
            raise ValueError("Community front and fabric ports must differ")
        if not isinstance(value["community_home"], str):
            raise TypeError("Community directory must be text")
        community_fields = ("community_home", "community_front_port", "community_fabric_port")
        if self.node and any(value[k] != self.preferences[k] for k in community_fields):
            raise ValueError("Disconnect the community before changing its connection settings")
        if not isinstance(value["studio_url"], str) or (value["studio_url"] and not value["studio_url"].startswith(("http://127.0.0.1:", "https://"))):
            raise ValueError("Studio URL must use local HTTP or HTTPS")
        if hasattr(self, "downloads"):
            self.downloads.configure(self.preferences, value)
        atomic_json(self.home / "desktop-preferences.json", value)
        self.preferences = value
        self.event("settings_updated", fields=sorted(changes))
        if hasattr(self, "user_profiles"):
            self.user_profiles.autosave()
        return self.public_preferences()

    def public_preferences(self):
        # Auth headers are kept in mode-0600 files, not application snapshots.
        return self.preferences

    def community_start(self):
        from .desktop_fleet import DesktopFleet
        if self.node is None:
            node = DesktopFleet(self)
            node.start()
            self.node = node
        return self.node.snapshot()

    def snapshot(self):
        value = self.primary.snapshot()
        workers = []
        for identity, engine in list(self.engines.items()):
            if identity == "default":
                part = value
            else:
                part = engine.snapshot()
                value["servers"].extend(part["servers"])
            if engine.worker:
                workers.append({**json_safe(engine.worker), "profile_id": identity,
                                "error": part.get("error"),
                                "process_running": engine.child is not None,
                                "share": bool((self.profile(identity) or {}).get("share"))})
        unique = {s["endpoint"]["url"]: s for s in value["servers"]}
        value["servers"] = list(unique.values())
        for server in value["servers"]:
            endpoint = server["endpoint"]
            owner = next((engine for engine in self.engines.values()
                          if engine.child is not None and engine.worker
                          and engine.child.pid == endpoint.get("pid")
                          and engine.worker.get("url") == endpoint.get("url")), None)
            if owner and owner.worker.get("resume_lineage_id") and not owner.worker["resume_lineage_id"].startswith("pending:") and owner.worker.get("state") in {"Ready", "Busy"}:
                endpoint.update(managed=True, checkpoint=owner.worker.get("checkpoint"),
                                resume_lineage_id=owner.worker["resume_lineage_id"])
                for key in ("resume_previous_pid", "resume_previous_url", "resume_previous_checkpoint"):
                    if key in owner.worker:
                        endpoint[key] = owner.worker[key]
        if self.node:
            self.node.sync()
            fleet = self.node.snapshot()
            value["community"] = fleet
            value["servers"].extend(fleet.get("servers", []))
        value["workers"] = workers
        value["preferences"] = self.public_preferences()
        value["jobs"] = self.store.jobs()
        value["qualification_records"] = self.store.records()
        from .desktop_campaign import comparison_rows
        value["performance_comparisons"] = comparison_rows(value["qualification_records"])
        value["campaign"] = self.campaign.status()
        value["media"] = self.media.snapshot()
        value["helpers"] = self.helpers.snapshot()
        value["code"] = self.code.snapshot()
        value["agents"] = self.agents.snapshot()
        value["planning"] = self.planning.snapshot()
        value["memory"] = self.memory.snapshot()
        value["capsules"] = self.capsules.snapshot()
        if self.world.sync_artifacts(value["jobs"]):
            self.user_profiles.autosave()
        value["world"] = self.world.snapshot()
        value["research"] = self.research.snapshot()
        value["research_plans"] = self.research_plans.snapshot()
        value["choices"] = self.choices.snapshot()
        value["dream"] = self.dream.snapshot()
        value["user_profiles"] = self.user_profiles.snapshot()
        value["wakeups"] = self.wakeups.snapshot()
        value["session_archive"] = self.sessions.snapshot()
        value["agent_launches"] = self.agent_launches.snapshot()
        value["skills"] = self.skills.snapshot()
        value["downloads"] = dict(self.downloads.progress)
        # Badge aggregation is performed on exactly bound records, never from chat success.
        records = value["qualification_records"]
        for row in value["library"]["rows"]:
            row["qualification"] = self.badges.summarize(row, records, self.preferences["profiles"])
        return value

    def logs(self, source="application", profile_id="default"):
        if source == "worker":
            engine = self.engines.get(profile_id)
            if engine is None:
                raise ValueError("Unknown engine")
            return engine.logs()
        if source == "campaign":
            return self.campaign.logs()
        paths = [self.events, self.home / "desktop.log"]
        text = []
        for path in paths:
            if path.is_file():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 64000))
                    text.append(f"--- {path.name} ---\n" + stream.read(64000).decode("utf-8", "replace"))
        return {"text": "\n".join(text) or "No application events yet."}

    @staticmethod
    def retained(engine):
        return (engine.worker or {}).get("state") in {"Suspending", "Suspended", "Resuming", "ResumeError", "SuspensionRecoveryRequired"}

    def cache_branches(self, identity):
        engine = self.engines.get(identity)
        if engine is None or not engine.worker:
            raise ValueError("Choose an app-managed model")
        worker = engine.worker
        if worker.get("state") == "Suspended":
            return {**worker.get("apc_branches", {}), "clock_frozen": True, "state": "Suspended"}
        child = engine.child
        if child is None or child.poll() is not None:
            raise ValueError("This model is not running")
        # Resolve this exact listener again before sending its private admin key.
        engine.chat_target(worker["url"], child.pid)
        from .http_security import load_secret_file
        from .inference_monitor import _get
        token = load_secret_file(Path(worker["admin_token_file"]), flag="--admin-token-file")
        return json.loads(_get(worker["url"], "/v1/admin/apc/branches?limit=200", token, 5))

    def dispatch(self, request):
        command = request.get("command")
        if command and command.startswith("code_"):
            return self.code.dispatch(request)
        if command and (command.startswith("skill_") or command == "skills_list"):
            return self.skills.dispatch(request, operator=True)
        if command == "artifact_document":
            import re
            name, text = request["name"], request["text"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,120}\.(md|txt|json)", name) or ".." in name:
                raise ValueError("Choose a simple .md, .txt or .json document name")
            if not isinstance(text, str) or not 1 <= len(text.encode("utf-8")) <= 262144:
                raise ValueError("Documents must contain between one byte and 256 KiB of text")
            if name.endswith(".json"):
                json.loads(text)
            identity = self.store.enqueue("document", {"name": name, "capability": "document"})
            directory = self.home / "artifacts" / identity
            directory.mkdir(parents=True, mode=0o700)
            path = directory / name
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(text)
            result = {"job_id": identity, "path": str(path), "name": name}
            self.store.update(identity, "completed", "Document saved", result)
            self.world.sync_artifacts(self.store.jobs())
            self.user_profiles.autosave()
            return result
        if command and command.startswith("user_profile_"):
            return self.user_profiles.dispatch(request)
        if command and command.startswith("wakeup_"):
            return self.wakeups.dispatch(request)
        if command and command.startswith("session_"):
            session_id = request.get("session_id")
            return self.sessions.dispatch(request, context={"profile_id": self.user_profiles.active,
                "session_id": session_id, "archive_operator": True})
        if command in {"agent_launch", "agent_launch_status", "agent_launch_cancel"}:
            return self.agent_launches.dispatch(request)
        if command and command.startswith("dream_"):
            result = self.dream.dispatch(request)
            if command == "dream_configure":
                self.user_profiles.autosave()
            return result
        if command and command.startswith("choice_"):
            result = self.choices.dispatch(request)
            if command == "choice_policy":
                self.user_profiles.autosave()
            return result
        if command in {"web_search", "fetch_page"}:
            return self.web.dispatch(request)
        if command == "research_plan_finish":
            return self.research_plans.finish(request.get("session_id"), request.get("outcome"), parent_id=request.get("parent_id"))
        if command and command.startswith("research_"):
            return self.research.dispatch(request)
        if command and command.startswith("world_"):
            result = self.world.dispatch(request)
            if command not in {"world_get", "world_export", "world_image_search", "world_asset_fetch", "world_asset_import"}:
                self.user_profiles.autosave()
            return result
        if command and command.startswith("capsule_"):
            if not self.preferences["memory_enabled"]:
                raise ValueError("Agent memory is disabled")
            result = self.capsules.dispatch(request)
            if command == "capsule_policy":
                self.user_profiles.autosave()
            return result
        if command == "snapshot":
            return self.snapshot()
        if command in {"prompt_presets", "prompt_install"}:
            from . import desktop_prompt_presets
            if command == "prompt_presets":
                return desktop_prompt_presets.catalog()
            return desktop_prompt_presets.install(self, request.get("preset"), request.get("activate"))
        if command == "preferences":
            return self.configure_preferences(request["changes"])
        if command == "profile_start":
            return self.start_profile(request["profile_id"])
        if command == "profile_stop":
            engine = self.engines.get(request["profile_id"])
            if engine is None:
                raise ValueError("Unknown engine")
            result = engine.stop()
            self.event("model_stopped", profile_id=request["profile_id"])
            return result
        if command in {"engine_suspend", "engine_resume", "engine_cache_branches"}:
            identity = request.get("profile_id", "default")
            engine = self.engines.get(identity)
            if engine is None:
                raise ValueError("Unknown app-managed engine")
            if command == "engine_cache_branches":
                return self.cache_branches(identity)
            if command == "engine_resume" and self.campaign.running:
                raise ValueError("Wait for or cancel the qualification queue before resuming a model")
            result = engine.begin_suspend() if command == "engine_suspend" else engine.begin_resume(request.get("suspension_id"))
            self.event(command, profile_id=identity)
            return result
        if command == "logs":
            return self.logs(request.get("source", "application"), request.get("profile_id", "default"))
        if command in {"engine_unload_self", "engine_restart_self", "ui_close", "ui_restart", "lifecycle_prepare_restart"}:
            return self.lifecycle.dispatch(request)
        if command.startswith("campaign_"):
            return self.campaign.dispatch(request)
        if command.startswith(("media_", "service_", "conversion_")):
            return self.media.dispatch(request)
        if command.startswith(("helper_", "adapter_", "document_")):
            return self.helpers.dispatch(request)
        if command.startswith("agent_"):
            if command != "agent_session" and not self.preferences["agent_delegation_enabled"]:
                raise ValueError("Local agent delegation is disabled")
            return self.agents.dispatch(request)
        if command.startswith("plan_"):
            if not self.preferences["built_in_planning_enabled"]:
                raise ValueError("Built-in planning is disabled")
            return self.planning.dispatch(request)
        if command.startswith(("memory_", "message_")):
            if not self.preferences["memory_enabled"]:
                raise ValueError("Agent memory and messaging are disabled")
            return self.memory.dispatch(request)
        if command.startswith("download_") or command == "hf_save_token":
            return self.downloads.dispatch(request)
        if command.startswith("community_"):
            if command == "community_connect":
                return self.community_start()
            if command == "community_disconnect":
                if self.node:
                    self.node.close()
                    self.node = None
                return {"connected": False}
            from .desktop_fleet import membership_command
            return membership_command(self, request)
        if command in {"chat_tools", "tool_call", "tool_discover", "mcp_save_secret"}:
            from .desktop_tools import tool_command
            return tool_command(self, request)
        if command == "chat_target" and request.get("remote"):
            if self.node is None:
                raise ValueError("Community is disconnected")
            return self.chat_target(self.node.chat_target(request["nid"], request["model"]))
        if command == "chat_target":
            target = self.primary.chat_target(request["url"], request["pid"])
            if "model" in request:
                target["requested_model"] = request["model"]
            return self.chat_target(target)
        if command == "start" and self.campaign.running:
            raise ValueError("Wait for or cancel the qualification queue before loading a model")
        if command == "configure" and not 1 <= request["settings"].get("lanes", 0) <= 4:
            raise ValueError("Choose between 1 and 4 serving lanes")
        result = self.primary.dispatch(request)
        if command == "configure":
            self.user_profiles.autosave()
        if command in {"start", "stop", "configure", "trash"}:
            self.event(command)
        return result

    def close(self):
        self.stopping.set()
        errors = []
        def close_one(component):
            try:
                component.close()
            except Exception as error:  # Finish releasing every independent owner.
                errors.append(error)
        for component in (self.wakeups, self.web, self.agent_launches, self.user_profiles,
                          self.dream, self.lifecycle):
            close_one(component)
        self.startup_thread.join(timeout=10)
        for component in (self.downloads, self.agents, self.research_plans, self.skills,
                          self.sessions, self.planning, self.capsules, self.memory,
                          self.code, self.helpers, self.campaign, self.media, self.node,
                          *reversed(list(self.engines.values())), self.store):
            if component is not None:
                close_one(component)
        if errors:
            raise RuntimeError("Desktop shutdown failed in " + str(len(errors)) + " subsystem(s)") from errors[0]

    def context_limit(self, path, explicit=0):
        preferences = getattr(self, "preferences", {})
        from .agentic_chat import model_chat_defaults
        defaults = model_chat_defaults(path)
        defaults.update(preferences.get("model_chat_options", {}).get(str(path), {}))
        chosen = explicit or preferences.get("model_context_tokens", {}).get(str(path), 0) or defaults.get("context_limit") or preferences.get("global_context_tokens", 32768)
        try:
            config = read_config(Path(path) / "config.json")
            text_config = config.get("text_config") or config
            maximum = text_config.get("max_position_embeddings")
            if type(maximum) is int and maximum > 0:
                chosen = min(chosen, maximum)
        except (OSError, ValueError, TypeError):
            pass
        return chosen

    def chat_target(self, target):
        from .inference_monitor import _get
        status = json.loads(_get(target["url"], "/v1/status", target.get("api_key"), 5))
        selected_model = target.get("requested_model")
        registered = (status.get("multi_lora") or {}).get("registered") or []
        if selected_model is not None and (not isinstance(selected_model, str) or selected_model != status.get("model") and (not (status.get("multi_lora") or {}).get("enabled") or selected_model not in registered)):
            raise ValueError("The requested model or LoRA is not loaded by this serving instance")
        lora_fingerprint = None
        if selected_model is not None and selected_model != status.get("model"):
            lora_fingerprint = ((status.get("multi_lora") or {}).get("registered_fingerprints") or {}).get(selected_model)
            if not isinstance(lora_fingerprint, str) or not lora_fingerprint:
                raise ValueError("The requested LoRA has no verified runtime fingerprint")
        capabilities = set(status.get("selected_capabilities") or status.get("capabilities") or [])
        modalities = ["text"]
        if (status.get("media_route") or {}).get("media") != "refused":
            modalities.extend(name for name, cap in (("image", "vision"), ("audio", "audio"), ("video", "video")) if cap in capabilities)
        artifact = status.get("artifact")
        fingerprint = (artifact.get("fingerprint") if isinstance(artifact, dict) else artifact) or status.get("artifact_fingerprint")
        checkpoint = status.get("checkpoint") or ((str(status.get("model", "model")) + "@" + str(fingerprint)) if fingerprint else None)
        worker = next((engine.worker for engine in self.engines.values() if engine.worker and engine.worker.get("url") == target["url"]), {})
        advertised_chat = dict((status.get("chat_template_options") or {}).get("recommended_chat") or {})
        requested_context = self.preferences.get("model_context_tokens", {}).get(worker.get("path", ""), 0)
        requested_context = requested_context or self.preferences.get("model_chat_options", {}).get(worker.get("path", ""), {}).get("context_limit", 0)
        context_limit = min(status.get("max_context") or 32768,
            requested_context or advertised_chat.get("context_limit") or self.context_limit(worker.get("path", "")))
        chat_options = dict((status.get("chat_template_options") or {}).get("recommended_chat") or {})
        chat_options.update(self.preferences.get("model_chat_options", {}).get(worker.get("path", ""), {}))
        if chat_options.get("verification") and "verification" not in (status.get("chat_template_options") or {}):
            raise ValueError("Selected engine does not advertise verification template support")
        output_tokens = min(chat_options.get("output_tokens", self.preferences["context_reserve_tokens"]), max(128, context_limit // 4))
        if chat_options.get("thinking_budget") is not None:
            chat_options["thinking_budget"] = min(chat_options["thinking_budget"], max(0, output_tokens - 1024))
        managed = {key: worker[key] for key in ("resume_lineage_id",) if worker.get(key) and not worker[key].startswith("pending:") and worker.get("state") in {"Ready", "Busy"}}
        return {**target, **managed, "model": selected_model or status.get("model"), "managed": bool(managed), "input_modalities": modalities, "capabilities": sorted(capabilities), "checkpoint": checkpoint,
                "artifact_fingerprint": fingerprint, "source_revision": (status.get("runtime") or {}).get("source_revision"),
                "runtime": status.get("runtime"), "lora_fingerprint": lora_fingerprint, "context_limit": context_limit,
                "chat_options": chat_options,
                "chat_template_options": status.get("chat_template_options") or {},
                "output_tokens": output_tokens}
