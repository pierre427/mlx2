# SPDX-License-Identifier: Apache-2.0
"""Per-build parity canaries for exact multi-row operations.

An exact multi-row op claims that every row of a ``[1, R, ...]`` strip gets
the bits its own one-row call would produce.  That claim depends on how the
installed MLX build dispatches and accumulates each shape, so it is checked,
not assumed: the first use of an (op, shape, dtype) on a (device, MLX build)
runs the op at several row counts against its one-row reference on
deterministic inputs and caches the verdict for the process.  A failed canary
makes the op take its per-row form, which is the one-row call itself.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from threading import RLock

import mlx.core as mx

from .. import row_exact_verify as REV

PROBE_ROWS = (2, 3, 4, 5, 8)
_BIT_VIEWS = {
    mx.bfloat16: mx.uint16,
    mx.float16: mx.uint16,
    mx.float32: mx.uint32,
}
_PROBES_ALLOWED = ContextVar("mlx2_row_exact_canary_probes_allowed", default=True)


class NumericalCanaryUnavailable(RuntimeError):
    """A row-exact verdict was absent while tensor probes were forbidden."""


@contextmanager
def probe_policy(*, allow: bool):
    """Control whether a cache miss may execute a numerical parity probe.

    Offline direct probes use ``allow=True`` to populate verdicts. A physical
    target uses ``allow=False`` so a reset or stale module verdict fails closed
    instead of putting synthetic tensor work in request execution.
    """

    if type(allow) is not bool:
        raise TypeError("row-exact canary probe policy must be boolean")
    token = _PROBES_ALLOWED.set(allow)
    try:
        yield
    finally:
        _PROBES_ALLOWED.reset(token)


def bit_mismatches(actual: mx.array, expected: mx.array) -> int:
    """Elements whose bit patterns differ (indices compare by value)."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return max(int(actual.size), int(expected.size), 1)
    view = _BIT_VIEWS.get(actual.dtype)
    if view is None:
        return int(mx.sum(actual != expected).item())
    return int(mx.sum(actual.view(view) != expected.view(view)).item())


def probe_input(rows: int, width: int, dtype, *, seed: int = 4217) -> mx.array:
    """A deterministic ``[1, rows, width]`` probe that leaves the global RNG alone."""
    values = mx.random.normal((1, rows, width), key=mx.random.key(seed + rows))
    return (values * 0.5).astype(dtype)


@lru_cache(maxsize=1)
def _installed_build() -> str:
    # The installed MLX build cannot change inside a process, and reading
    # package metadata costs ~0.14 ms: per lookup that was ~60 ms of host
    # time per Nemotron verify window (11 ops x 40 MoE layers).
    from ..mlx_build import installed_mlx_build

    return installed_mlx_build()


def build_identity() -> tuple[str, str]:
    return _installed_build(), str(mx.default_device())


class ParityCanaries:
    """Process-wide verdicts keyed by build, device, generation and op shape.

    One re-entrant lock covers lookup, probe and store, so a first use is
    single-flight (two threads never probe one key at once, and none returns
    a verdict another is about to contradict) and ``reset`` is a barrier (it
    waits for an in-flight probe; no verdict from before it survives).
    Probes run on first use only, so serialising them costs nothing on the
    hot path.
    """

    def __init__(self) -> None:
        self._results: dict[tuple, dict] = {}
        self._inconclusive = 0
        self._private = {"checked": 0, "failed": [], "inconclusive": 0}
        self._rows_probed: set[int] = set()
        self._lock = RLock()
        # Bumped by ``reset``: verdicts cached outside the registry (on a
        # module, for the hot verify path) are valid only for one generation.
        self.generation = 0

    def _run(self, probe, probe_rows):
        """``(mismatches, conclusive)`` of ``probe`` at ``probe_rows``."""
        mismatches = 0
        conclusive = True
        # A probe is not part of the verify window that triggered it: a
        # module whose class swap consults the window (an installed row-exact
        # projection) must run its plain one-row call here and must not note
        # routes on, or fail, the caller's window.
        with REV.window(None):
            for count in probe_rows:
                multi, reference = probe(count)
                mx.eval(multi, reference)
                mismatches += bit_mismatches(multi, reference)
                # An all-zero reference (zero weights, a zero bias) cannot
                # tell two forms apart.
                conclusive = conclusive and bool(mx.any(reference != 0).item())
        self._rows_probed.update(int(count) for count in probe_rows)
        return mismatches, conclusive

    def exact(
        self,
        key: tuple[Hashable, ...],
        probe: Callable[[int], tuple[mx.array, mx.array]],
        *,
        rows: int | None = None,
    ) -> bool:
        """Whether ``probe(rows)`` = (multi-row, one-row reference) agree bitwise.

        For value-independent forms (row-independent ops, selection, combine
        laws on synthetic inputs): the verdict is shared by every caller with
        the same key.  With ``rows`` it is for exactly that row count (part of
        the key; the probe runs at it): a dispatch exact through eight rows
        may change tiling at seventeen.  Without it the probe runs at
        ``PROBE_ROWS``.  An inconclusive probe (all-zero reference) answers
        False and is not stored.  Weight-bound forms use ``exact_private``.
        """
        probe_rows = PROBE_ROWS if rows is None else (int(rows),)
        with self._lock:
            generation = self.generation
            full = (*build_identity(), generation, probe_rows, *key)
            result = self._results.get(full)
            if result is None:
                if not _PROBES_ALLOWED.get():
                    raise NumericalCanaryUnavailable(
                        "row-exact shared numerical verdict is unavailable"
                    )
                mismatches, conclusive = self._run(probe, probe_rows)
                if self.generation != generation:
                    return False  # reset during the probe: fail closed, keep nothing
                if not conclusive:
                    self._inconclusive += 1
                    return False
                result = {"exact": mismatches == 0, "mismatches": mismatches}
                earlier = self._results.get(full)  # stored by a re-entrant probe
                if earlier is not None and not earlier["exact"]:
                    result = earlier  # a failing verdict wins
                self._results[full] = result
            return result["exact"]

    def exact_private(
        self,
        key: tuple[Hashable, ...],
        probe: Callable[[int], tuple[mx.array, mx.array]],
        *,
        rows: int,
    ) -> bool:
        """A verdict for one module's own arrays, never shared.

        Exactness of a form can depend on the values it is fed (a zero bias
        hides a fused-bias difference), so a weight-bound op probes each
        module for itself and its caller caches the answer next to the arrays
        it was earned on.  Counted in ``snapshot`` under ``private``.
        """
        with self._lock:
            if not _PROBES_ALLOWED.get():
                raise NumericalCanaryUnavailable(
                    "row-exact private numerical verdict is unavailable"
                )
            generation = self.generation
            mismatches, conclusive = self._run(probe, (int(rows),))
            if self.generation != generation:
                return False  # reset during the probe: fail closed
            if not conclusive:
                self._private["inconclusive"] += 1
                return False
            self._private["checked"] += 1
            if mismatches:
                self._private["failed"].append(
                    {
                        "key": [str(part) for part in key],
                        "rows": int(rows),
                        "mismatches": mismatches,
                    }
                )
            return mismatches == 0

    def snapshot(self) -> dict:
        with self._lock:
            items = list(self._results.items())
            return {
                "probe_rows": sorted(self._rows_probed),
                "checked": len(items),
                "inconclusive": self._inconclusive,
                "failed": [
                    {"key": [str(part) for part in key], **dict(result)}
                    for key, result in items
                    if not result["exact"]
                ],
                "private": {
                    "checked": self._private["checked"],
                    "inconclusive": self._private["inconclusive"],
                    "failed": list(self._private["failed"]),
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._results.clear()
            self._inconclusive = 0
            self._private = {"checked": 0, "failed": [], "inconclusive": 0}
            self._rows_probed.clear()
            self.generation += 1


CANARIES = ParityCanaries()
