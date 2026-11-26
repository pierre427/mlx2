"""Opt-in per-round host attribution for decode rounds (Phase 0 measurement).

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md, Phase 0
(§4.1). Diagnostic only, default off. With ``MLX2_ROUND_PROBE`` unset nothing
is patched and every call site pays one module-attribute check
(``round_probe.ACTIVE``).

With ``MLX2_ROUND_PROBE=1`` the process installs count-and-time wrappers on

* the explicit sync points ``mlx.core.eval``, ``mlx.core.async_eval`` and
  ``mlx.core.synchronize`` (module attributes, so every ``import mlx.core as
  mx`` caller is seen, unlike ``_TimedMx``, which sees only the external
  executor's own ``self.mx``), and
* the implicit device-to-host reads on ``mlx.core.array``: ``item``,
  ``tolist``, ``__array__``, ``__buffer__`` (``np.asarray``/``memoryview``),
  ``__int__``, ``__float__``, ``__bool__`` and ``__index__``.

Only calls made inside a :func:`scope` (one decode round on the executor's
thread) are counted, attributed to that round's route label. Each counted call
is split into

* ``encode_submit``: an ``mx.async_eval`` of the same arguments first, which
  builds the Metal command buffers and commits them without waiting, and
* ``wait``: the original call, which now only waits for the device and copies.

So a round's wall time divides into ``encode_submit_ns`` + ``wait_ns`` +
``host_python_ns`` (everything else on the host: Python graph construction,
NumPy acceptance, bookkeeping). The split itself adds one ``async_eval`` call
per sync, so the probe is attribution evidence, never performance evidence;
receipts say so (``performance_claim: False``). GPU busy time is not visible
from Python: ``scripts/spec_round_phase0.py`` measures it with a bounded
Metal System Trace and joins it to these counters.

``host_python_ns`` is not graph construction alone: it also holds NumPy
acceptance, commit bookkeeping and receipts. ``async_eval`` can block on MLX's
own scheduler backpressure, so ``encode_submit`` may include some waiting.
Under ``MLX2_EXTERNAL_ROUND_TIMING`` the executor's own ``self.mx`` calls
resolve to ``_TimedMx._timed`` as their call site. ``syncs`` counts host syncs
(``async_eval`` excluded); ``blocking_syncs`` counts those that waited more
than ``BLOCKING_NS``, so ``mx.eval(t); t.tolist()`` books two syncs but one
blocking sync.

``MLX2_ROUND_PROBE_SITES`` (default 16) bounds the per-route table of call
sites (``file:line`` of the caller) kept for each sync kind; 0 disables it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import Counter

ENV = "MLX2_ROUND_PROBE"
SITES_ENV = "MLX2_ROUND_PROBE_SITES"
# MLX's own command-buffer limits (read by libmlx at device creation); the
# Phase 0 harness sweeps them and the probe echoes what the process saw.
BUFFER_ENVS = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")
EXPLICIT = ("eval", "async_eval", "synchronize")
IMPLICIT = (
    "item", "tolist", "__array__", "__buffer__",
    "__int__", "__float__", "__bool__", "__index__",
)

# A counted read whose wait exceeds this is booked as blocking: it waited on
# the device (or a copy). A read of an already-evaluated array stays below it.
BLOCKING_NS = 20_000

ACTIVE = False
_installed = False
_originals: dict = {}
_lock = threading.Lock()
_local = threading.local()
_routes: dict = {}
_site_limit = 16
_installed_at = None


def _env_on(value) -> bool:
    return value not in (None, "", "0", "false", "False", "off")


def enabled_by_env(environ=None) -> bool:
    return _env_on((os.environ if environ is None else environ).get(ENV))


def _route_row(route):
    row = _routes.get(route)
    if row is None:
        row = _routes[route] = {
            "rounds": 0,
            "tokens": 0,
            "wall_ns": 0,
            "syncs": {},
            "sites": {},
        }
    return row


def _account(kind, encode_ns, wait_ns, site):
    """Book one sync into the open round; merged into its route at exit.

    Per-round accumulation keeps a round's syncs and wall time under one
    label even when :func:`retag` refines it after some syncs ran.
    """
    frame = getattr(_local, "frame", None)
    if frame is None:
        return
    frame["encode_ns"] += encode_ns
    frame["wait_ns"] += wait_ns
    blocking = wait_ns > BLOCKING_NS
    if kind != "async_eval":  # submits without waiting: not a host sync
        frame["syncs"] += 1
        frame["blocking"] += blocking
    sync = frame["kinds"].get(kind)
    if sync is None:
        sync = frame["kinds"][kind] = [0, 0, 0, 0]
    sync[0] += 1
    sync[1] += encode_ns
    sync[2] += wait_ns
    sync[3] += blocking
    if site is not None:
        frame["sites"][(kind, site)] = frame["sites"].get((kind, site), 0) + 1


def _merge(row, frame):
    for kind, (calls, encode, wait, blocking) in frame["kinds"].items():
        sync = row["syncs"].get(kind)
        if sync is None:
            sync = row["syncs"][kind] = {"calls": 0, "blocking": 0, "encode_submit_ns": 0, "wait_ns": 0}
        sync["calls"] += calls
        sync["blocking"] += blocking
        sync["encode_submit_ns"] += encode
        sync["wait_ns"] += wait
    for (kind, site), count in frame["sites"].items():
        sites = row["sites"].setdefault(kind, Counter())
        if site in sites or len(sites) < _site_limit:
            sites[site] += count
        else:
            sites["<other>"] += count


def _caller_site():
    if _site_limit <= 0:
        return None
    try:
        frame = sys._getframe(2)
    except ValueError:  # pragma: no cover - shallow stack
        return None
    code = frame.f_code
    path = code.co_filename
    marker = path.rfind("/mlx2/")
    if marker >= 0:
        path = path[marker + 1:]
    else:
        path = os.path.basename(path)
    return f"{path}:{frame.f_lineno}"


def _counting():
    """True only inside a round on this thread, outside our own calls."""
    return getattr(_local, "route", None) is not None and not getattr(_local, "busy", False)


def _wrap_explicit(kind, original, async_eval):
    def wrapper(*args, **kwargs):
        if not _counting():
            return original(*args, **kwargs)
        site = _caller_site()
        _local.busy = True
        try:
            started = time.perf_counter_ns()
            if kind == "eval" and args:
                try:
                    async_eval(*args)
                except Exception:  # the original call raises it properly
                    pass
            submitted = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                finished = time.perf_counter_ns()
                if kind == "async_eval":
                    # async_eval only encodes and commits.
                    encode, wait = finished - started, 0
                elif kind == "synchronize":
                    encode, wait = 0, finished - started
                else:
                    encode, wait = submitted - started, finished - submitted
                _account(kind, encode, wait, site)
        finally:
            _local.busy = False

    wrapper.__name__ = getattr(original, "__name__", kind)
    wrapper.__doc__ = getattr(original, "__doc__", None)
    wrapper._round_probe_original = original
    return wrapper


def _wrap_implicit(kind, original, async_eval):
    def wrapper(self, *args, **kwargs):
        if not _counting():
            return original(self, *args, **kwargs)
        site = _caller_site()
        _local.busy = True
        try:
            started = time.perf_counter_ns()
            try:
                async_eval(self)
            except Exception:
                pass
            submitted = time.perf_counter_ns()
            try:
                return original(self, *args, **kwargs)
            finally:
                _account(kind, submitted - started, time.perf_counter_ns() - submitted, site)
        finally:
            _local.busy = False

    wrapper.__name__ = kind
    wrapper._round_probe_original = original
    return wrapper


def install(mx=None) -> bool:
    """Install the wrappers once per process; returns whether the probe is active."""
    global ACTIVE, _installed, _site_limit, _installed_at
    with _lock:
        if _installed:
            return ACTIVE
        if mx is None:
            import mlx.core as mx
        try:
            _site_limit = max(0, int(os.environ.get(SITES_ENV, "16")))
        except ValueError:
            _site_limit = 16
        async_eval = mx.async_eval
        for kind in EXPLICIT:
            original = getattr(mx, kind)
            _originals[("module", kind)] = (mx, original)
            setattr(mx, kind, _wrap_explicit(kind, original, async_eval))
        array = mx.array
        for kind in IMPLICIT:
            original = getattr(array, kind, None)
            if original is None:
                continue
            _originals[("array", kind)] = (array, original)
            setattr(array, kind, _wrap_implicit(kind, original, async_eval))
        _installed = True
        _installed_at = time.time()
        ACTIVE = True
        return True


def uninstall() -> None:
    """Restore the originals (tests); counters are kept until :func:`reset`."""
    global ACTIVE, _installed
    with _lock:
        for (_, kind), (owner, original) in _originals.items():
            setattr(owner, kind, original)
        _originals.clear()
        _installed = False
        ACTIVE = False


def reset() -> None:
    with _lock:
        _routes.clear()


def install_from_env() -> bool:
    if enabled_by_env():
        return install()
    return False


class _NullScope:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def tokens(self, count):
        return None


_NULL = _NullScope()


class _Scope:
    __slots__ = ("route", "frame", "started", "outer", "_tokens")

    def __init__(self, route):
        self.route = route
        self._tokens = 0

    def __enter__(self):
        self.outer = getattr(_local, "route", None) is not None
        if self.outer:
            return self  # nested: the enclosing round owns the time
        self.frame = {"encode_ns": 0, "wait_ns": 0, "syncs": 0, "blocking": 0, "kinds": {}, "sites": {}}
        _local.frame = self.frame
        _local.route = self.route
        self.started = time.perf_counter_ns()
        return self

    def tokens(self, count):
        self._tokens += int(count)

    def __exit__(self, *exc):
        if self.outer:
            return False
        wall = time.perf_counter_ns() - self.started
        route = _local.route  # retag() may have refined the label
        _local.route = None
        _local.frame = None
        with _lock:
            row = _route_row(route)
            row["rounds"] += 1
            row["tokens"] += self._tokens
            row["wall_ns"] += wall
            row.setdefault("encode_submit_ns", 0)
            row.setdefault("wait_ns", 0)
            row.setdefault("round_syncs", 0)
            row["encode_submit_ns"] += self.frame["encode_ns"]
            row["wait_ns"] += self.frame["wait_ns"]
            row["round_syncs"] += self.frame["syncs"]
            row["blocking_syncs"] = row.get("blocking_syncs", 0) + self.frame["blocking"]
            _merge(row, self.frame)
        return False


def scope(route):
    """Context manager for one decode round; a no-op unless the probe is active."""
    if not ACTIVE:
        return _NULL
    return _Scope(str(route))


def retag(route) -> None:
    """Re-label the current round (the dispatcher learns its branch late)."""
    if ACTIVE and getattr(_local, "route", None) is not None:
        _local.route = str(route)


def _route_summary(row):
    rounds = row["rounds"]
    wall = row["wall_ns"]
    encode = row.get("encode_submit_ns", 0)
    wait = row.get("wait_ns", 0)
    explicit = {k: dict(v) for k, v in row["syncs"].items() if k in EXPLICIT}
    implicit = {k: dict(v) for k, v in row["syncs"].items() if k not in EXPLICIT}
    syncs_total = row.get("round_syncs", 0)
    per_round = (lambda value: round(value / rounds, 4)) if rounds else (lambda value: None)
    return {
        "rounds": rounds,
        "tokens": row["tokens"],
        "wall_ns": wall,
        "encode_submit_ns": encode,
        "wait_ns": wait,
        "host_python_ns": max(0, wall - encode - wait),
        # Host syncs (everything but async_eval); blocking = waited > BLOCKING_NS.
        "syncs": syncs_total,
        "blocking_syncs": row.get("blocking_syncs", 0),
        "explicit_syncs": dict(sorted(explicit.items())),
        "implicit_syncs": dict(sorted(implicit.items())),
        "per_round": {
            "wall_ms": per_round(wall / 1e6),
            "encode_submit_ms": per_round(encode / 1e6),
            "wait_ms": per_round(wait / 1e6),
            "host_python_ms": per_round(max(0, wall - encode - wait) / 1e6),
            "syncs": per_round(syncs_total),
            "blocking_syncs": per_round(row.get("blocking_syncs", 0)),
            "explicit_syncs": per_round(sum(v["calls"] for k, v in explicit.items() if k != "async_eval")),
            "implicit_syncs": per_round(sum(v["calls"] for v in implicit.values())),
            "tokens": per_round(row["tokens"]),
        },
        "sites": {
            kind: dict(counter.most_common())
            for kind, counter in sorted(row["sites"].items())
        },
    }


def status() -> dict:
    """``/v1/status`` and receipt block; ``{"enabled": False}`` when off."""
    if not ACTIVE:
        return {"enabled": False}
    with _lock:
        routes = {route: _route_summary(row) for route, row in sorted(_routes.items())}
    return {
        "enabled": True,
        "schema": "mlx2.round-probe.v1",
        "scope": "process_cumulative_since_install",
        "installed_at": _installed_at,
        "clock": "host_wall_split_eval",
        "performance_claim": False,
        "command_buffer_env": {name: os.environ.get(name) for name in BUFFER_ENVS},
        "routes": routes,
    }


install_from_env()
