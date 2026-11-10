# SPDX-License-Identifier: MIT
"""Ranked ReadPool: demand reads overtake queued speculative reads.

Mined from the Kimi-K3.1-Pro expert-drafting simulation (lab evidence,
provenance/deepseek-v41-source-expert-lookahead.json): a FIFO channel lost
because demand reads queued behind predictions; the priority channel (demand
first, stale predictions dropped) kept 80-85% of the oracle's gain.

Ownership, deadlines, quarantine and close semantics are inherited unchanged
from :class:`ReadPool`; only the order in which queued (never running) calls
start differs. A running syscall is never preempted.
"""
from __future__ import annotations

import heapq
import itertools
import math
import threading

from .io_ownership import ReadPool

DEMAND_RANK = 0


class _RankedQueue:
    """SimpleQueue-compatible put/get over a (rank, seq) heap.

    ``promote`` re-files a queued future at a better rank; the superseded heap
    entry is skipped lazily. The close sentinel (None) ranks last, matching
    FIFO drain-then-exit behaviour for already queued work.
    """

    def __init__(self, pool):
        self._pool = pool
        self._heap = []
        self._seq = itertools.count()
        self._live = {}  # future -> seq of its current entry
        self._cond = threading.Condition()

    def put(self, task):
        rank = math.inf if task is None else self._pool._take_rank()
        with self._cond:
            seq = next(self._seq)
            if task is not None:
                self._live[task[0]] = seq
            heapq.heappush(self._heap, (rank, seq, task))
            self._cond.notify()

    def get(self):
        with self._cond:
            while True:
                while not self._heap:
                    self._cond.wait()
                rank, seq, task = heapq.heappop(self._heap)
                if task is None:
                    return None
                if self._live.get(task[0]) != seq:
                    continue  # superseded by a promotion
                del self._live[task[0]]
                return task

    def promote(self, future, rank):
        with self._cond:
            seq = self._live.get(future)
            if seq is None:
                return False
            for item in self._heap:
                if item[1] == seq:
                    if item[0] <= rank:
                        return False
                    new = next(self._seq)
                    self._live[future] = new
                    heapq.heappush(self._heap, (rank, new, item[2]))
                    self._cond.notify()
                    return True
            return False

    def queued(self):
        with self._cond:
            return sum(not future.cancelled() for future in self._live)


class PriorityReadPool(ReadPool):
    """ReadPool whose queued calls start in (rank, submission) order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue = _RankedQueue(self)
        self._rank_lock = threading.Lock()
        self._next_rank = DEMAND_RANK
        self.priority_counters = dict(priority_demand_submitted=0, priority_speculative_submitted=0,
                                      priority_promotions=0)

    def _take_rank(self):
        # Called from ReadPool.submit while _rank_lock is held by submit_ranked.
        return self._next_rank

    def submit_ranked(self, rank, function, *args):
        if type(rank) is not int or rank < DEMAND_RANK:
            raise ValueError('read rank must be a non-negative integer')
        with self._rank_lock:
            self._next_rank = rank
            try:
                future = super().submit(function, *args)
            finally:
                self._next_rank = DEMAND_RANK
        self.priority_counters['priority_demand_submitted' if rank == DEMAND_RANK
                               else 'priority_speculative_submitted'] += 1
        return future

    def submit(self, function, *args):
        return self.submit_ranked(DEMAND_RANK, function, *args)

    def promote(self, future, rank=DEMAND_RANK):
        """Move a still-queued call ahead; a running or finished call is unaffected."""
        moved = self._queue.promote(future, rank)
        if moved:
            self.priority_counters['priority_promotions'] += 1
        return moved

    def counters(self):
        result = super().counters()
        result.update(self.priority_counters)
        result['priority_queued'] = self._queue.queued()
        return result
