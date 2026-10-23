"""Retain campaign subprocess identities until every owned descendant has exited.

Signals address psutil.Process identities (PID plus creation time), never an
unverified process group. A reaped leader does not end descendant ownership.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time

import psutil


class OwnedProcess:
    def __init__(self, process):
        self.process = process
        self.pid = process.pid
        self.members = {}
        self.created = psutil.Process(self.pid).create_time()
        self.session = os.getsid(self.pid)
        if self.session != self.pid:
            raise ValueError("Campaign process must start its own session")
        self.capture()

    def __getattr__(self, key):
        return getattr(self.process, key)

    def capture(self):
        candidates = []
        original_session = True
        try:
            leader = psutil.Process(self.pid)
            if leader.create_time() != self.created:
                # Never adopt the replacement leader or its session.
                original_session = False
            else:
                candidates = [leader, *leader.children(recursive=True)]
        except psutil.NoSuchProcess:
            pass
        # Retained children may create further independent sessions after their
        # original leader exits. Follow only their verified identities.
        for member in tuple(self.members.values()):
            try:
                if member.is_running():
                    candidates.extend(member.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
        # Orphaned same-session children remain attributable after their leader
        # exits. A session ID cannot be recreated without a leader with that PID;
        # the creation-time check above rejects that replacement.
        for candidate in psutil.process_iter() if original_session else ():
            try:
                if os.getsid(candidate.pid) == self.session and candidate.create_time() >= self.created:
                    candidates.append(candidate)
            except (ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        for candidate in candidates:
            try:
                self.members[(candidate.pid, candidate.create_time())] = candidate
            except psutil.NoSuchProcess:
                continue

    def poll(self):
        self.capture()  # preserve identities before Popen.poll reaps the leader
        return self.process.poll()

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.process.args, timeout)
            time.sleep(0.05)
        return self.process.returncode

    def live(self):
        self.capture()
        result = []
        for member in self.members.values():
            try:
                if member.is_running() and member.status() != psutil.STATUS_ZOMBIE:
                    result.append(member)
            except psutil.NoSuchProcess:
                pass
        return result

    def stop(self, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            live = self.live()
            if not live:
                break
            signum = signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL
            for member in live:
                try:
                    member.send_signal(signum)  # psutil refuses PID reuse
                except psutil.NoSuchProcess:
                    pass
            if time.monotonic() >= deadline + 10:
                raise RuntimeError("Campaign descendants are still alive; cleanup incomplete")
            time.sleep(0.05)
        return self.process.wait(timeout=10)


def spawn(*args, **kwargs):
    if kwargs.get("start_new_session") is not True:
        raise ValueError("Owned campaign subprocesses require a new session")
    process = subprocess.Popen(*args, **kwargs)
    return OwnedProcess(process)
