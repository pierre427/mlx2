"""First-in, first-out request lane for the desktop companion process.

Each request runs on its own thread, but requests take the lane in arrival
order and run one at a time, exactly as the earlier single-threaded loop did.
A section that only waits on the network and touches state guarded by its own
lock may step out of the lane with ``released()``: later requests run while it
waits, and it then rejoins at the back of the queue.
"""
from __future__ import annotations

import contextlib
import threading


class SerialLane:
    def __init__(self):
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving = 0
        self._holder = None

    def ticket(self):
        """Reserve the next place in line; every ticket must later be held once."""
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            return ticket

    def _acquire(self, ticket):
        with self._condition:
            self._condition.wait_for(lambda: self._serving == ticket)
            self._holder = threading.get_ident()

    def _release(self):
        with self._condition:
            self._holder = None
            self._serving += 1
            self._condition.notify_all()

    @contextlib.contextmanager
    def held(self, ticket):
        self._acquire(ticket)
        try:
            yield
        finally:
            self._release()

    def owned(self):
        return self._holder == threading.get_ident()

    @contextlib.contextmanager
    def released(self):
        """Let later requests run during a network wait. A no-op outside the lane."""
        if not self.owned():
            yield
            return
        self._release()
        try:
            yield
        finally:
            self._acquire(self.ticket())


LANE = SerialLane()
