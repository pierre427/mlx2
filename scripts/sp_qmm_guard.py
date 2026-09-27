"""Swap guard for the sp-qmm GPU harnesses (2026-09-25).

A background thread samples ``vm_stat`` Swapouts once a second. During the
load phase the process may add up to ``load_limit_mb`` of swapouts; once
``timed()`` is called the baseline resets and a rise of ``timed_limit_mb`` or
more kills the process with exit code 3. The machine is shared: a swapping
benchmark spoils every other session's numbers too.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

PAGE = 16384


def swapouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    raise RuntimeError("vm_stat has no Swapouts line")


class SwapGuard:
    def __init__(self, load_limit_mb: float = 1536, timed_limit_mb: float = 256):
        self.load_limit = load_limit_mb * 1e6
        self.timed_limit = timed_limit_mb * 1e6
        self.base = swapouts()
        self.limit = self.load_limit
        self.phase = "load"
        self.max_delta_mb = {"load": 0.0, "timed": 0.0}
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def timed(self):
        self.base = swapouts()
        self.limit = self.timed_limit
        self.phase = "timed"

    def _run(self):
        while not self._stop:
            d = (swapouts() - self.base) * PAGE
            self.max_delta_mb[self.phase] = max(self.max_delta_mb[self.phase], d / 1e6)
            if d >= self.limit:
                print(f"SWAP GUARD: {self.phase} swapouts +{d / 1e6:.0f} MB "
                      f">= {self.limit / 1e6:.0f} MB; aborting", file=sys.stderr, flush=True)
                os._exit(3)
            time.sleep(1.0)

    def report(self):
        return {k: round(v, 1) for k, v in self.max_delta_mb.items()}
