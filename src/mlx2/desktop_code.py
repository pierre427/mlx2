"""Profile-scoped coding sessions and bounded, explicitly submitted CLI work.

The local model coordinates through a narrow tool set. Fixed provider argv,
workspace identity checks and the provider's normal permission boundary remain
in force; this is not an arbitrary executable or shell control API.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .desktop import atomic_json, stop_group
from .desktop_helpers import executable

OUTPUT_LIMIT = 2 * 1024 * 1024
DOCUMENT_LIMIT = 8 * 1024 * 1024
ACTIVE = {"queued", "running"}
PLAN_OPERATIONS = {"plan_create", "plan_add", "plan_get", "plan_claim", "plan_report", "plan_cancel"}


def workspace_identity(raw):
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise ValueError("Choose an existing absolute work folder")
    path = Path(raw).resolve(strict=True)
    if not path.is_dir() or path == Path(path.anchor):
        raise ValueError("Choose an existing work folder below the filesystem root")
    stat = path.stat()
    return {"path": str(path), "device": stat.st_dev, "inode": stat.st_ino}


def checked_workspace(identity):
    path = Path(identity["path"])
    for component in [path, *path.parents]:
        if component.is_symlink():
            raise ValueError("Work folder traversal changed; choose the folder again")
    actual = workspace_identity(str(path))
    if actual != identity:
        raise ValueError("Work folder identity changed; choose the folder again")
    return path


def code_command(provider, binary, workspace, effort):
    if effort not in {"low", "medium", "high"}:
        raise ValueError("Choose low, medium or high effort")
    if provider == "codex":
        return [binary, "exec", "--ignore-user-config", "--sandbox", "workspace-write",
                "--skip-git-repo-check", "--ephemeral", "--color", "never", "--json",
                "-c", 'approval_policy="on-request"', "-c", f'model_reasoning_effort="{effort}"',
                "-c", 'web_search="disabled"', "-C", str(workspace), "-"]
    if provider == "claude":
        return [binary, "--print", "--restricted", "--tools", "Read,Edit,Write,Glob,Grep,Bash",
                "--permission-mode", "acceptEdits", "--permission-prompts", "none",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
                "--output-format", "stream-json", "--verbose", "--effort", effort]
    raise ValueError("Choose codex or claude")


def redact(text):
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[redacted]", text)
    text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{8,})\b", "[redacted]", text)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CODEX_API_KEY"):
        value = os.environ.get(name, "")
        if len(value) >= 8:
            text = text.replace(value, "[redacted]")
    return text


class CodeWorkspace:
    def __init__(self, controller):
        self.controller = controller
        self.home = controller.home / "code"
        self.home.mkdir(mode=0o700, exist_ok=True)
        if self.home.is_symlink():
            raise ValueError("Code state directory must not be a symbolic link")
        self.lock = threading.RLock()
        self.sessions, self.jobs, self.tools = {}, {}, {}
        self.closed = False
        for path in self.home.glob("session-*.json"):
            try:
                if path.is_symlink() or path.stat().st_size > DOCUMENT_LIMIT + 65536:
                    continue
                value = json.loads(path.read_text())
                if path.name != "session-" + value["id"] + ".json":
                    continue
                value.pop("document", None)
                value.setdefault("max_iterations", 3); value.setdefault("iterations", 0)
                value.setdefault("cycle_id", uuid.uuid4().hex); value.setdefault("loop_stopped", False)
                self.sessions[value["id"]] = value
            except (ValueError, OSError, KeyError, TypeError):
                continue
        for path in self.home.glob("job-*.json"):
            try:
                if path.is_symlink() or path.stat().st_size > 131072:
                    continue
                value = json.loads(path.read_text())
                if path.name != "job-" + value["id"] + ".json":
                    continue
                if value["status"] in ACTIVE:
                    value.update(status="interrupted", error="Application exited before this job completed.",
                                 delivery_status="pending" if value.get("chat_session_id") else "none")
                    self._save_job(value)
                value["cancel"] = threading.Event()
                self.jobs[value["id"]] = value
            except (ValueError, OSError, KeyError, TypeError):
                continue

    def _profile(self):
        return self.controller.user_profiles.active

    def _session(self, identity, *, runnable=False):
        value = self.sessions.get(identity)
        if not value or value.get("profile_id") != self._profile():
            raise ValueError("Unknown code session for this profile")
        if runnable and (value["archived"] or value["deleted"]):
            raise ValueError("Restore this code session before running work")
        return value

    def _job(self, identity, code_session_id=None):
        value = self.jobs.get(identity)
        if not value or value.get("profile_id") != self._profile() or (code_session_id and value["code_session_id"] != code_session_id):
            raise ValueError("Unknown code job for this session/profile")
        return value

    def _session_path(self, identity):
        return self.home / ("session-" + identity + ".json")

    def _save_session(self, value, document=None):
        path = self._session_path(value["id"])
        if document is None:
            document = json.loads(path.read_text()).get("document", {}) if path.exists() else {}
        atomic_json(path, {**value, "document": document})

    @staticmethod
    def _public_job(value):
        return {key: value for key, value in value.items() if key not in {"cancel", "child", "thread", "prompt", "workspace_identity", "claim_id", "claim_expires", "summary"}}

    def _save_job(self, value):
        atomic_json(self.home / ("job-" + value["id"] + ".json"),
                    {key: item for key, item in value.items() if key not in {"cancel", "child", "thread", "prompt"}})

    def snapshot(self):
        with self.lock:
            profile = self._profile()
            sessions = [dict(value) for value in self.sessions.values() if value["profile_id"] == profile and not value["deleted"]]
            jobs = [self._public_job(value) for value in self.jobs.values() if value["profile_id"] == profile]
            return {"sessions": sorted(sessions, key=lambda value: -value["updated"]), "jobs": sorted(jobs, key=lambda value: -value["created"])[:100],
                    "pending_deliveries": sum(value.get("delivery_status") in {"pending", "claimed"} for value in jobs),
                    "available": {provider: executable(provider) is not None for provider in ("codex", "claude")}}

    def _settings(self, request, existing=None):
        value = dict(existing or {})
        for key in ("title", "provider", "effort", "max_iterations"):
            if key in request:
                value[key] = request[key]
        value.setdefault("title", "New code session")
        value.setdefault("provider", "codex")
        value.setdefault("effort", "medium")
        value.setdefault("max_iterations", 3)
        if type(value["max_iterations"]) is not int or not 1 <= value["max_iterations"] <= 16:
            raise ValueError("Code iteration limit must be 1–16")
        if not isinstance(value["title"], str) or not value["title"].strip() or len(value["title"]) > 200:
            raise ValueError("Session title must be 1–200 characters")
        if value["provider"] not in {"codex", "claude"} or value["effort"] not in {"low", "medium", "high"}:
            raise ValueError("Choose codex/claude and low/medium/high effort")
        if "workspace" in request:
            value["workspace_identity"] = workspace_identity(request["workspace"])
            value["workspace"] = value["workspace_identity"]["path"]
        if "workspace_identity" not in value:
            raise ValueError("Choose a work folder")
        return value

    def dispatch(self, request):
        command = request.get("command")
        with self.lock:
            if self.closed:
                raise ValueError("Code workspace is shutting down")
            if command == "code_session_create":
                value = self._settings(request)
                value.update(id=uuid.uuid4().hex, profile_id=self._profile(), created=time.time(), updated=time.time(), archived=False, deleted=False, plan_ids=[], iterations=0, cycle_id=uuid.uuid4().hex, loop_stopped=False)
                self.sessions[value["id"]] = value
                self._save_session(value, {})
                return {"session": dict(value)}
            if command == "code_session_list":
                return {"sessions": [dict(value) for value in self.sessions.values() if value["profile_id"] == self._profile()
                                     and (request.get("include_archived") is True or not value["archived"] and not value["deleted"])]}
            if command in {"code_tools", "code_tool_call"}:
                return self._tools(request)
            if command in {"code_completion_claim", "code_completion_complete", "code_completion_retry"}:
                return self._delivery(request)
            if command in {"code_status", "code_result", "code_cancel"}:
                job = self._job(request.get("job_id"))
                if command == "code_cancel":
                    job["cancel"].set()
                    return {"job_id": job["id"], "cancelling": job["status"] in ACTIVE}
                return self._result(job, request)
            value = self._session(request.get("code_session_id"))
            if command == "code_cycle_start":
                self._session(value["id"], runnable=True)
                if any(job["code_session_id"] == value["id"] and job["status"] in ACTIVE for job in self.jobs.values()):
                    raise ValueError("Stop or finish the active code cycle before submitting another goal")
                for job in self.jobs.values():
                    if job["code_session_id"] == value["id"] and job.get("delivery_status") in {"pending", "claimed"}:
                        job["delivery_status"] = "superseded"
                        self._save_job(job)
                value.update(cycle_id=uuid.uuid4().hex, iterations=0, loop_stopped=False, updated=time.time())
                self._save_session(value)
                return {"session": dict(value)}
            if command == "code_stop_loop":
                value["loop_stopped"] = True
                for job in self.jobs.values():
                    if job["code_session_id"] == value["id"] and job.get("cycle_id") == value["cycle_id"]:
                        job["cancel"].set()
                        if job.get("delivery_status") in {"pending", "claimed"}:
                            job["delivery_status"] = "superseded"
                            self._save_job(job)
                self._save_session(value)
                return {"session": dict(value), "stopped": True}
            if command == "code_session_get":
                document = json.loads(self._session_path(value["id"]).read_text()).get("document", {})
                return {"session": dict(value), "document": document,
                        "jobs": [self._public_job(job) for job in self.jobs.values() if job["code_session_id"] == value["id"]][-20:]}
            if command == "code_session_save":
                if request.get("profile_id", self._profile()) != self._profile():
                    raise ValueError("User profile changed before saving")
                document = request.get("document")
                if not isinstance(document, dict) or len(json.dumps(document, allow_nan=False).encode()) > DOCUMENT_LIMIT:
                    raise ValueError("Coordinator state must be bounded JSON of at most 8 MiB")
                self._save_session(value, document)
                return {"saved": value["id"]}
            if command == "code_session_update":
                if any(job["code_session_id"] == value["id"] and job["status"] in ACTIVE for job in self.jobs.values()):
                    raise ValueError("Wait for or cancel this session's active code job before changing settings")
                value.update(self._settings(request, value), updated=time.time())
            elif command in {"code_session_archive", "code_session_delete", "code_session_restore"}:
                if any(job["code_session_id"] == value["id"] and job["status"] in ACTIVE for job in self.jobs.values()):
                    raise ValueError("Cancel active work before archiving this session")
                value.update(archived=command != "code_session_restore", deleted=command == "code_session_delete", updated=time.time())
            elif command in {"code_start", "code_run"}:
                return self._start(value, request["prompt"], request.get("_binding"))
            else:
                raise ValueError("Unknown code workspace operation")
            self._save_session(value)
            return {"session": dict(value)}

    def _start(self, session, prompt, binding=None):
        self._session(session["id"], runnable=True)
        workspace = checked_workspace(session["workspace_identity"])
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 65536:
            raise ValueError("Code prompt must contain 1–65536 bytes")
        if session["loop_stopped"] or session["iterations"] >= session["max_iterations"]:
            raise ValueError("Code cycle stopped or reached its iteration limit; a new user goal is required")
        if binding and binding.get("cycle_id") != session["cycle_id"]:
            raise ValueError("Code tool context belongs to an older goal cycle")
        prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()
        stamp = self._workspace_stamp(workspace)
        if any(job["code_session_id"] == session["id"] and job["status"] == "failed" and job.get("prompt_digest") == prompt_digest
               and job.get("provider") == session["provider"] and job.get("effort") == session["effort"]
               and (stamp is None or job.get("workspace_stamp") == stamp) for job in self.jobs.values()):
            raise ValueError("The same prompt already failed without a verified workspace change; inspect the failure and change the request before retrying")
        binary = executable(session["provider"])
        if not binary:
            raise ValueError("Install and authenticate the selected CLI first")
        running = [job for job in self.jobs.values() if job["status"] in ACTIVE]
        if len(running) >= 2 or any(job["workspace"] == str(workspace) for job in running):
            raise ValueError("Wait for active code work in this folder; at most two distinct folders run concurrently")
        if len(self.jobs) >= 4096:
            raise ValueError("Code job archive reached its 4096-job limit")
        identity = uuid.uuid4().hex
        job = {"id": identity, "code_session_id": session["id"], "profile_id": session["profile_id"],
               "workspace": str(workspace), "workspace_identity": dict(session["workspace_identity"]),
               "cycle_id": session["cycle_id"], "prompt_digest": prompt_digest, "workspace_stamp": stamp,
               "provider": session["provider"], "effort": session["effort"], "status": "queued", "created": time.time(),
               "prompt": prompt, "cancel": threading.Event(), "output_bytes": 0,
               "chat_session_id": (binding or {}).get("chat_session_id"), "target": (binding or {}).get("public_target", {}),
               "delivery_status": "none", "error": None, "exit_code": None}
        self.jobs[identity] = job
        session["iterations"] += 1
        self._save_session(session)
        self._save_job(job)
        thread = threading.Thread(target=self._run, args=(job, binary), daemon=True, name="mlx2-code-cli")
        job["thread"] = thread
        thread.start()
        return {"job_id": identity, "status": "queued", "completion_delivery": "automatic" if job["chat_session_id"] else "manual"}

    @staticmethod
    def _workspace_stamp(workspace):
        digest = hashlib.sha256()
        count = 0
        try:
            for root, directories, files in os.walk(workspace, followlinks=False):
                directories[:] = sorted(name for name in directories if name not in {".git", ".venv", "node_modules", "__pycache__"})
                for name in sorted(files):
                    path = Path(root) / name
                    stat = path.lstat()
                    digest.update(f"{path.relative_to(workspace)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
                    count += 1
                    if count > 4096:
                        return None
            return digest.hexdigest()
        except OSError:
            return None

    def _result(self, job, request):
        offset, limit = request.get("offset", 0), request.get("limit", 8192)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 32768:
            raise ValueError("Use a nonnegative output offset and limit 1–32768")
        path = self.home / ("output-" + job["id"] + ".txt")
        output = ""
        if path.is_file() and not path.is_symlink():
            try:
                with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
                    data = stream.read(OUTPUT_LIMIT + 1)
                if len(data) <= OUTPUT_LIMIT:
                    output = data.decode(errors="replace")
            except OSError:
                pass
        text = output[offset:offset + limit].encode()[:limit].decode(errors="ignore")
        return {"job": self._public_job(job), "output": text, "next_offset": offset + len(text) if offset + len(text) < len(output) else None,
                "total_characters": len(output), "summary": job.get("summary", ""), "reference_only": True}

    @staticmethod
    def _event_result(text):
        try:
            value = json.loads(text)
        except ValueError:
            return False, ""
        if not isinstance(value, dict):
            return False, ""
        failure = (value.get("is_error") is True or bool(value.get("permission_denials"))
                   or value.get("type") in {"error", "turn.failed"}
                   or value.get("type") == "result" and str(value.get("subtype", "")).startswith("error"))
        summary = value.get("result") if value.get("type") == "result" else None
        item = value.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            summary = item.get("text")
        return failure, summary if isinstance(summary, str) else ""

    def _run(self, job, binary):
        child = None
        try:
            with self.lock:
                if job["cancel"].is_set() or job["profile_id"] != self._profile():
                    raise InterruptedError("Code work cancelled or profile changed before launch")
                workspace = checked_workspace(job["workspace_identity"])
                job.update(status="running", started=time.time())
                self._save_job(job)
            command = code_command(job["provider"], binary, Path("."), job["effort"])
            environment = {key: value for key, value in os.environ.items() if key in {
                "HOME", "PATH", "LANG", "LC_ALL", "TMPDIR", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CODEX_API_KEY"}}
            incoming = self.home / ("input-" + job["id"] + ".txt")
            with os.fdopen(os.open(incoming, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w") as stream:
                stream.write(job["prompt"])
            output_path = self.home / ("output-" + job["id"] + ".txt")
            with incoming.open("rb") as source, os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as output:
                output_path.chmod(0o600)
                # Open the checked cwd and pass the descriptor, preventing a
                # rename/symlink swap between the identity check and spawn.
                descriptor = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    stat = os.fstat(descriptor)
                    if (stat.st_dev, stat.st_ino) != (job["workspace_identity"]["device"], job["workspace_identity"]["inode"]):
                        raise ValueError("Work folder changed immediately before launch")
                    launcher = [sys.executable, "-I", "-c",
                                "import os,sys; fd=int(sys.argv[1]); os.fchdir(fd); os.close(fd); os.execvpe(sys.argv[2],sys.argv[2:],os.environ)",
                                str(descriptor), *command]
                    child = subprocess.Popen(launcher, pass_fds=(descriptor,), env=environment,
                                             stdin=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
                finally:
                    os.close(descriptor)
                with self.lock:
                    job["child"] = child
                selector = selectors.DefaultSelector()
                selector.register(child.stdout, selectors.EVENT_READ)
                os.set_blocking(child.stdout.fileno(), False)
                deadline, total = time.monotonic() + 1800, 0
                pending = b""
                provider_failed = False
                exited_at = None
                try:
                    while True:
                        if job["cancel"].is_set() or job["profile_id"] != self._profile():
                            raise InterruptedError("Code job cancelled or user profile changed")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Code job exceeded its 30-minute limit")
                        if child.poll() is not None:
                            exited_at = exited_at or time.monotonic()
                            if time.monotonic() - exited_at >= 0.5:
                                if pending:
                                    decoded = redact(pending.decode(errors="replace"))
                                    failure, summary = self._event_result(decoded)
                                    provider_failed = provider_failed or failure
                                    with self.lock:
                                        if summary:
                                            job["summary"] = summary.encode()[:8192].decode(errors="ignore")
                                    clean = decoded.encode()
                                    if output.tell() + len(clean) > OUTPUT_LIMIT:
                                        raise ValueError("Code output exceeded its 2 MiB limit")
                                    output.write(clean); output.flush()
                                break
                        events = selector.select(0.1)
                        if events:
                            chunk = os.read(child.stdout.fileno(), 4096)
                            if not chunk:
                                if pending:
                                    decoded = redact(pending.decode(errors="replace"))
                                    failure, summary = self._event_result(decoded)
                                    provider_failed = provider_failed or failure
                                    if summary:
                                        with self.lock:
                                            job["summary"] = summary.encode()[:8192].decode(errors="ignore")
                                    clean = decoded.encode()
                                    if output.tell() + len(clean) > OUTPUT_LIMIT:
                                        raise ValueError("Code output exceeded its 2 MiB limit")
                                    output.write(clean); output.flush()
                                break
                            total += len(chunk)
                            if total > OUTPUT_LIMIT:
                                raise ValueError("Code output exceeded its 2 MiB limit")
                            # Retain bounded JSONL/text privately; model results
                            # are paginated and explicitly marked reference data.
                            pending += chunk
                            while b"\n" in pending:
                                line, pending = pending.split(b"\n", 1)
                                decoded = redact(line.decode(errors="replace"))
                                failure, summary = self._event_result(decoded)
                                provider_failed = provider_failed or failure
                                if summary:
                                    with self.lock:
                                        job["summary"] = summary.encode()[:8192].decode(errors="ignore")
                                clean = (decoded + "\n").encode()
                                if output.tell() + len(clean) > OUTPUT_LIMIT:
                                    raise ValueError("Code output exceeded its 2 MiB limit")
                                output.write(clean); output.flush()
                            with self.lock:
                                job["output_bytes"] = output.tell()
                        elif child.poll() is not None:
                            continue
                    job["exit_code"] = child.wait(timeout=5)
                finally:
                    selector.close()
                    child.stdout.close()
            if job["cancel"].is_set() or job["profile_id"] != self._profile():
                raise InterruptedError("Code job cancelled or user profile changed")
            job["output_bytes"] = output_path.stat().st_size
            if provider_failed:
                raise ValueError("The coding provider reported a failure or denied permission; inspect the bounded output")
            if job["exit_code"] != 0:
                raise ValueError("The coding CLI exited unsuccessfully; inspect its bounded output and permission errors")
            with self.lock:
                job["status"] = "completed"
        except InterruptedError as error:
            with self.lock:
                job.update(status="cancelled", error=str(error))
        except Exception as error:  # noqa: BLE001 - isolate one submitted CLI job
            with self.lock:
                job.update(status="failed", error=redact(str(error))[:1024])
        finally:
            if child is not None:
                stop_group(child)
            (self.home / ("input-" + job["id"] + ".txt")).unlink(missing_ok=True)
            with self.lock:
                job.pop("child", None)
                job.pop("prompt", None)
                session = self.sessions.get(job["code_session_id"], {})
                if job["status"] == "failed":
                    job["workspace_stamp"] = self._workspace_stamp(Path(job["workspace"]))
                deliver = job["chat_session_id"] and not session.get("loop_stopped") and session.get("cycle_id") == job.get("cycle_id")
                job.update(finished=time.time(), delivery_status="pending" if deliver else "none")
                self._save_job(job)

    def _tools(self, request):
        if request["command"] == "code_tools":
            code = self._session(request.get("code_session_id"), runnable=True)
            selection = request.get("target")
            if not isinstance(selection, dict) or selection.get("remote"):
                raise ValueError("Choose a local running coordinator model")
            selection = {key: selection[key] for key in ("url", "pid", "model") if key in selection}
            fresh = self.controller.dispatch({"command": "chat_target", **selection})
            chat = request.get("chat_session_id")
            if not isinstance(chat, str) or not 1 <= len(chat) <= 512:
                raise ValueError("A coordinator conversation identity is required")
            now = time.time()
            self.tools = {token: context for token, context in self.tools.items() if now - context["created"] < 3600}
            if len(self.tools) >= 256:
                raise ValueError("Too many active code coordinator contexts; wait for older contexts to expire")
            token = uuid.uuid4().hex
            public = {key: fresh[key] for key in ("url", "model", "checkpoint") if key in fresh}
            public["pid"] = selection.get("pid")
            self.tools[token] = {"code_session_id": code["id"], "profile_id": self._profile(), "chat_session_id": chat,
                                 "selection": selection, "public_target": public, "created": time.time(), "cycle_id": code["cycle_id"]}
            from .desktop_tools import OPERATIONS
            definitions = {
                "code_run": ("Delegate implementation in this code session's chosen work folder and CLI. Completion automatically returns here; continue independent planning or end the turn instead of polling.", {"prompt": {"type": "string"}}),
                "code_result": ("Inspect a bounded page of a retained job report only when needed; completion delivery is automatic.", {"job_id": {"type": "string"}}),
                "code_cancel": ("Cancel a code job created in this exact session.", {"job_id": {"type": "string"}}),
                **{name: OPERATIONS[name] for name in PLAN_OPERATIONS}}
            tools = [{"type": "function", "function": {"name": "mlx2__" + name, "description": description,
                      "parameters": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}}
                     for name, (description, properties) in definitions.items()]
            return {"session_id": token, "tools": tools}
        context = self.tools.get(request.get("session_id"))
        if not context or context["profile_id"] != self._profile() or time.time() - context["created"] >= 3600:
            raise ValueError("Code tool context expired or user profile changed")
        code = self._session(context["code_session_id"], runnable=True)
        fresh = self.controller.dispatch({"command": "chat_target", **context["selection"]})
        if any(fresh.get(key) != context["public_target"].get(key) for key in ("url", "model", "checkpoint")):
            raise ValueError("Coordinator model changed; reconnect this code session")
        if not isinstance(request.get("name"), str):
            raise TypeError("Code tool name must be text")
        name = request["name"].removeprefix("mlx2__")
        arguments = request.get("arguments")
        if name not in {"code_run", "code_result", "code_cancel"} | PLAN_OPERATIONS or not isinstance(arguments, dict):
            raise ValueError("Tool is outside the code coordinator's grant")
        from .desktop_tools import OPERATIONS
        properties = {"prompt"} if name == "code_run" else {"job_id"} if name in {"code_result", "code_cancel"} else set(OPERATIONS[name][1])
        if set(arguments) != properties:
            raise ValueError("Unexpected or missing code-tool arguments")
        if name in {"code_run", "code_cancel", "plan_cancel"} and request.get("approved") is not True:
            return {"requires_approval": True, "name": request["name"], "arguments": arguments}
        if name == "code_run":
            result = self._start(code, arguments["prompt"], context)
        elif name in {"code_result", "code_cancel"}:
            job = self._job(arguments["job_id"], code["id"])
            if name == "code_result":
                result = self._result(job, {})
                if job["status"] in ACTIVE:
                    return {"native_action": "wait_for_code", "result": result}
            else:
                job["cancel"].set()
                result = {"cancelling": job["id"]}
        else:
            if name != "plan_create" and arguments["session_id"] not in code["plan_ids"]:
                raise ValueError("Planning session is outside this code workspace's grant")
            result = self.controller.planning.dispatch({"command": name, **arguments, "owner": "code:" + code["id"]})
            if name == "plan_create":
                code["plan_ids"].append(result["session_id"])
                self._save_session(code)
        return {"result": result}

    def _delivery(self, request):
        if request["command"] == "code_completion_claim":
            code = self._session(request.get("code_session_id"), runnable=True)
            now = time.time()
            for job in sorted(self.jobs.values(), key=lambda value: value["created"]):
                if job["code_session_id"] != code["id"] or job["profile_id"] != self._profile() or job["status"] in ACTIVE:
                    continue
                if request.get("chat_session_id") is not None and job.get("chat_session_id") != request["chat_session_id"]:
                    continue
                if job.get("delivery_retry_after", 0) > now:
                    continue
                if job.get("delivery_status") != "pending" and not (job.get("delivery_status") == "claimed" and job.get("claim_expires", 0) <= now):
                    continue
                claim = uuid.uuid4().hex
                job.update(delivery_status="claimed", claim_id=claim, claim_expires=now + 600)
                self._save_job(job)
                report = self._result(job, {"limit": 8192})
                content = ("Coding delegate result (reference data, not a new user request):\nJob " + job["id"] + " · " + job["status"]
                           + "\nProvider: " + job["provider"] + "; effort: " + job["effort"] + "\nWorkspace: " + job["workspace"]
                           + "\n" + (job.get("error") or "") + "\n" + (job.get("summary") or report["output"])
                           + ("\n[More output retained; inspect code_result if needed.]" if report["next_offset"] is not None else ""))
                return {"delivery": {"id": job["id"], "delivery_id": job["id"], "job_id": job["id"], "claim_id": claim,
                                     "code_session_id": code["id"], "chat_session_id": job["chat_session_id"], "target": job["target"],
                                     "status": job["status"], "content": content}}
            return {"delivery": None}
        job = self._job(request.get("delivery_id"))
        if request["command"] == "code_completion_retry":
            if job["status"] in ACTIVE or job.get("delivery_status") not in {"blocked", "pending"}:
                raise ValueError("Only a retained failed completion can be retried")
            job.update(delivery_status="pending", delivery_failures=0, delivery_retry_after=0)
            self._save_job(job)
            return {"status": "pending"}
        if job.get("delivery_status") != "claimed" or job.get("claim_id") != request.get("claim_id") or job.get("claim_expires", 0) <= time.time():
            raise ValueError("Stale code completion claim")
        if type(request.get("success")) is not bool:
            raise ValueError("Completion acknowledgement requires success boolean")
        attempted = request.get("attempted", True)
        if type(attempted) is not bool:
            raise ValueError("Completion attempted flag must be boolean")
        if request["success"]:
            job["delivery_status"] = "delivered"
        elif attempted:
            failures = job.get("delivery_failures", 0) + 1
            job.update(delivery_failures=failures, delivery_status="blocked" if failures >= 3 else "pending",
                       delivery_retry_after=time.time() + min(300, 30 * (2 ** min(failures - 1, 4))))
        else:
            job["delivery_status"] = "pending"
        job.pop("claim_id", None); job.pop("claim_expires", None)
        self._save_job(job)
        return {"status": job["delivery_status"]}

    def close(self):
        with self.lock:
            self.closed = True
            jobs = [job for job in self.jobs.values() if job["status"] in ACTIVE]
            for job in jobs:
                job["cancel"].set()
        for job in jobs:
            thread = job.get("thread")
            if thread:
                thread.join(timeout=3)
