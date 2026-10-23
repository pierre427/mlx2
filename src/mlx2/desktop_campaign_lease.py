"""Real CPG/RADIO lease for bounded desktop GPU jobs, independent of UI polling.

Use inside one job, then launch the job through run_with_gpu_locks.py. This
lease does not replace the paired host locks. Call check() in the child monitor
loop and terminate the entire child process group on any failure.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .desktop_campaign_worker import Radio, atomic, verify_claim


class DesktopCPGLease:
    def __init__(self, directory: Path, job_id: str, *, mcp_url="http://127.0.0.1:8766/mcp", checkpoint="desktop-job", source_revision="unavailable", radio=None):
        self.directory = Path(directory)
        self.path = self.directory / "desktop-lease.json"
        self.radio = radio or Radio(mcp_url)
        self.job_id = job_id
        self.checkpoint = checkpoint
        self.agent = "mlx2-desktop-" + job_id
        self.owner = {"agent": "mlx2_desktop_campaign", "agent_id": self.agent, "pid": os.getpid(),
                      "job_id": job_id, "source_revision": source_revision}
        self.task = self.worker = self.generation = None
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.success = False

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.exists():
            raise ValueError("Refusing to overwrite an existing CPG ownership receipt")
        try:
            plan = self.radio.call("plan_session", {"goal": "One bounded desktop GPU window",
                "tasks": [{"title": "GPU: " + self.checkpoint,
                           "description": "One artifact under paired host locks; cleanup before releasing ownership"}],
                "agent_id": self.agent, "archive_previous": False})
            self.task = plan["task_ids"][0]
            workflow = self.radio.call("start_workflow", {"name": "Desktop job " + self.job_id, "task_ids": [self.task]})["workflow_id"]
            self.worker = self.radio.call("register_worker", {"workflow_id": workflow, "label": "desktop-artifact", "role": "executor"})["worker_id"]
            claim = self.radio.call("claim_task", {"task_id": self.task, "lease_seconds": 180})
            self.generation = verify_claim(claim, task=self.task, agent=self.agent)
            self.owner.update(cpg_task=self.task, cpg_session=plan["session_id"], cpg_worker=self.worker,
                cpg_generation=self.generation, claim=claim, renewed_at=time.time(), expires=claim["lease_expires_at"])
            atomic(self.path, self.owner)
            self.thread = threading.Thread(target=self._renew, daemon=True, name="mlx2-desktop-radio")
            self.thread.start()
            return self
        except BaseException:
            self.close()
            raise

    def _renew(self):
        while not self.stop_event.wait(30):
            try:
                reply = self.radio.call("renew_lease", {"task_id": self.task, "lease_generation": self.generation, "lease_seconds": 180})
                verify_claim(reply, task=self.task, generation=self.generation, renewed=True)
                self.radio.call("worker_heartbeat", {"worker_id": self.worker})
                self.owner.update(renewal=reply, expires=reply["lease_expires_at"], renewed_at=time.time())
                atomic(self.path, self.owner)
            except Exception as error:  # noqa: BLE001 -- monitor must fail closed, never silently keep using GPU
                self.error = error
                return

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"CPG GPU lease renewal failed: {self.error}")
        if self.stop_event.is_set() or self.owner.get("expires", 0) <= time.time() + 15:
            raise RuntimeError("CPG GPU lease is no longer live")
        return dict(self.owner)

    def close(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=35)
        for name, arguments in (
            ("release_task", {"task_id": self.task, "lease_generation": self.generation, "status": "done" if self.success else "failed"}),
            ("complete_worker", {"worker_id": self.worker, "status": "completed" if self.success else "failed"}),
        ):
            if (name == "release_task" and self.generation is None) or (name == "complete_worker" and self.worker is None):
                continue
            try:
                self.owner[name] = self.radio.call(name, arguments)
            except Exception as error:  # noqa: BLE001 -- cleanup failure must remain visible in durable receipt
                self.owner[name] = {"error": str(error)}
        if self.task or self.worker:
            self.owner["finished_at"] = time.time()
            atomic(self.path, self.owner)

    def __exit__(self, kind, value, traceback):
        self.success = kind is None and self.success
        self.close()
