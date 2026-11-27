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

Replay feasibility probes (P0/P3 of
docs/research/2026-11-26-metal-replay-feasibility-probes.md), both default off
and only with the probe on:

* ``MLX2_ROUND_PROBE_GPU=1`` loads the staged native step-replay module
  (``MLX2_STEP_REPLAY_MODULE``, default ``build/step_replay/stage``), installs
  its hooks and turns on its GPU timeline: every MLX command buffer files its
  ``GPUStartTime``/``GPUEndTime`` in the host clock. Each round records its own
  host window and the windows of its counted syncs (the ``async_eval`` split
  included), and rounds are joined with the timeline once all their command
  buffers completed. Per route it adds ``gpu_rounds`` (attributed rounds),
  ``gpu_wall_ns`` (their wall time), ``gpu_busy_ns`` (union of GPU busy time
  inside the round), ``gpu_eval_ns``/``gpu_eval_busy_ns`` (the same inside the
  sync windows), ``gpu_eval_tail_ns`` (from the last GPU end in a sync window
  to its return: completion latency, not removable) and
  ``gpu_command_buffers`` (committed inside the round). GPU idle inside a sync
  window, less its tail, is what replaying the round's command stream could
  remove; GPU idle outside the sync windows is host Python, which replay only
  removes if graph construction moves into the recorded program.
* ``MLX2_ROUND_PROBE_RECORD=N`` (needs the module) records the compute
  commands of N rounds per route with ``step_replay`` after each round the
  probe sees after more than ``MLX2_ROUND_PROBE_RECORD_GAP_S`` (default 0.5)
  without rounds, skipping the first round after the gap, and writes each
  recording (summary, dispatch table, pipeline reflection) as JSON under
  ``MLX2_ROUND_PROBE_RECORD_DIR``. ``MLX2_ROUND_PROBE_RECORD_MAX`` (default 4)
  caps the recordings per process. With the Phase 0 harness, N=2 and the cap
  of 4 put the recordings in the cold warm-up requests of the two quiet cells
  (each follows a prefill gap); the contended cells' load-ramp gaps come after
  the cap. Recorded rounds are slower (capture and a
  drain); they are booked under ``recorded_rounds`` and left out of the GPU
  attribution, but stay in the host counters.
* ``MLX2_ROUND_PROBE_SUBSTITUTE=N`` (P4, needs recording) replays every
  dispatch of the first N recordings with ``step_replay.substitution_check``
  right after the drained recording, on the executor thread with MLX idle: the
  recorded ``setBytes`` payloads against the same payloads bound from a
  constants buffer, directly and through a one-command indirect command buffer,
  compared bit for bit on private copies of the written buffers, once per
  alignment in ``MLX2_ROUND_PROBE_SUBSTITUTE_ALIGN`` (default ``256,4``).
  Results go to ``substitution-<pid>-<n>.json`` beside the recordings. It takes
  seconds per recording, inside the request that hosts the recorded round.
  Only *clean* dispatches are replayed: every buffer the dispatch reads is
  not written again later in the round, and every buffer it writes is last
  written by it. The pass runs after the round, so any other dispatch would
  read post-round contents. On 2026-11-26, 3,393 of a Raon round's 3,803
  dispatches were not clean: MLX reuses buffers within a round, and a
  replayed scatter whose int64 index buffer held a later int32 op's data
  wrote outside its private copy and changed later outputs. Those dispatches
  are booked ``skipped_stale_inputs``, never replayed.
  ``MLX2_ROUND_PROBE_FLAG_AB=0,2`` (P2, needs recording) times, for the listed
  recordings, every distinct kernel on two pipelines rebuilt from MLX's
  descriptor with ``supportIndirectCommandBuffers`` off and on (a short
  ranking pass, then 20+20 interleaved blocks of 100 dispatches for the 10
  kernels with the most GPU time per round and every NAX kernel), with
  ``flag_ab_instance`` choosing a replay-safe dispatch per kernel; results
  in ``flag-ab-<pid>-<n>.json``.
  ``MLX2_ROUND_PROBE_RECORD_ADDRESSES=1`` (P5a) adds every recorded buffer's
  ``gpuAddress``, length, storage mode and heap flag to each recording.
  Side-effect isolation: ``MLX2_ROUND_PROBE_SUBSTITUTE_ARMS`` picks the arms
  (``A,A2,B,C``, default all; ``none`` makes the copies only, no GPU work;
  partial runs compare nothing), and ``MLX2_ROUND_PROBE_SUBSTITUTE_CHECKSUM=1``
  hashes every host-visible recorded buffer before and after the pass and
  lists the ones that changed, with the dispatches that bind them.
"""

from __future__ import annotations

import bisect
import json
import os
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

ENV = "MLX2_ROUND_PROBE"
SITES_ENV = "MLX2_ROUND_PROBE_SITES"
GPU_ENV = "MLX2_ROUND_PROBE_GPU"
MODULE_ENV = "MLX2_STEP_REPLAY_MODULE"
RECORD_ENV = "MLX2_ROUND_PROBE_RECORD"
RECORD_DIR_ENV = "MLX2_ROUND_PROBE_RECORD_DIR"
RECORD_GAP_ENV = "MLX2_ROUND_PROBE_RECORD_GAP_S"
RECORD_MAX_ENV = "MLX2_ROUND_PROBE_RECORD_MAX"
SUBSTITUTE_ENV = "MLX2_ROUND_PROBE_SUBSTITUTE"
SUBSTITUTE_ALIGN_ENV = "MLX2_ROUND_PROBE_SUBSTITUTE_ALIGN"
SUBSTITUTE_ARMS_ENV = "MLX2_ROUND_PROBE_SUBSTITUTE_ARMS"
FLAG_AB_ENV = "MLX2_ROUND_PROBE_FLAG_AB"
# P5a: each recording also lists every recorded buffer's gpuAddress and length.
ADDRESSES_ENV = "MLX2_ROUND_PROBE_RECORD_ADDRESSES"
# Kernels that may write at addresses read from buffer contents: replayed for
# timing only from a clean instance (record-time inputs). Data-dependent reads
# (gather, gather_qmv) are allowed: a stale index can only fault the probe's
# own command buffer, since every written buffer is a private copy.
_WRITE_UNSAFE = ("scatter", "sort", "custom_kernel")
SUBSTITUTE_CHECKSUM_ENV = "MLX2_ROUND_PROBE_SUBSTITUTE_CHECKSUM"
_ARM_BITS = {"A": 1, "A2": 2, "B": 4, "C": 8}


def _arms_mask(value):
    if value is None or value.strip() == "":
        return 15
    if value.strip().lower() == "none":
        return 0
    return sum(_ARM_BITS[name.strip()] for name in value.split(",") if name.strip())
GPU_KEYS = (
    "gpu_rounds", "gpu_wall_ns", "gpu_busy_ns", "gpu_eval_ns", "gpu_eval_busy_ns",
    "gpu_eval_tail_ns", "gpu_command_buffers", "recorded_rounds",
)
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
# Replay feasibility probes (module docstring); None while off.
_native = None
_gpu = {"enabled": False, "error": None, "module": None}
_pending = deque()  # finished rounds awaiting the GPU timeline
_PENDING_LIMIT = 50_000
_iv_start: list = []  # GPU intervals sorted by start (host seconds)
_iv = []  # (start, end, commit), same order
_iv_max = 0.0  # longest interval seen, for the window lookup
_timeline_counts = {"intervals": 0, "errored": 0, "unattributed_rounds_dropped": 0}
_record = {"per_gap": 0, "dir": None, "gap_s": 0.5, "max": 4, "written": 0,
           "substitute": 0, "substitute_align": (256, 4),
           "errors": [], "left": 0, "last_round_end": None, "skip": False}


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


def _account(kind, encode_ns, wait_ns, site, window=None):
    """Book one sync into the open round; merged into its route at exit.

    Per-round accumulation keeps a round's syncs and wall time under one
    label even when :func:`retag` refines it after some syncs ran.
    """
    frame = getattr(_local, "frame", None)
    if frame is None:
        return
    frame["encode_ns"] += encode_ns
    frame["wait_ns"] += wait_ns
    if window is not None and "windows" in frame:
        frame["windows"].append(window)
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
            h0 = _native.host_now() if _native is not None else None
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
                window = (h0, _native.host_now()) if h0 is not None else None
                _account(kind, encode, wait, site, window)
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
            h0 = _native.host_now() if _native is not None else None
            started = time.perf_counter_ns()
            try:
                async_eval(self)
            except Exception:
                pass
            submitted = time.perf_counter_ns()
            try:
                return original(self, *args, **kwargs)
            finally:
                finished = time.perf_counter_ns()
                window = (h0, _native.host_now()) if h0 is not None else None
                _account(kind, submitted - started, finished - submitted, site, window)
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
    if _env_on(os.environ.get(GPU_ENV)):
        _install_native()
    return True


def _module_path():
    configured = os.environ.get(MODULE_ENV)
    if configured:
        return Path(configured)
    import sysconfig
    root = Path(__file__).resolve().parents[3]
    return root / "build" / "step_replay" / "stage" / ("_step_replay" + sysconfig.get_config_var("EXT_SUFFIX"))


def _install_native():
    """Load step_replay, install its hooks and start the GPU timeline (P0/P3)."""
    global _native
    from .native_extension import load_extension

    path = _module_path()
    _gpu["module"] = str(path)
    try:
        module = load_extension("mlx2_step_replay._step_replay", str(path))
        hooks = module.install_hooks()
        if not hooks.get("installed"):
            raise RuntimeError(f"step_replay hooks not installed: {hooks.get('error')}")
        if "timeline" not in module.CAPABILITIES:
            raise RuntimeError("step_replay module has no timeline capability")
        module.set_timeline(True)
    except Exception as error:  # diagnostic: report, never break serving
        _gpu["error"] = f"{type(error).__name__}: {error}"
        return
    _gpu.update(enabled=True, gpu=module.gpu_identity(), mlx=module.MLX_VERSION,
                capabilities=list(module.CAPABILITIES))
    try:
        per_gap = max(0, int(os.environ.get(RECORD_ENV, "0") or 0))
    except ValueError:
        per_gap = 0
    if per_gap:
        directory = os.environ.get(RECORD_DIR_ENV)
        if not directory:
            _record["errors"].append(f"{RECORD_DIR_ENV} unset; recording off")
        else:
            Path(directory).mkdir(parents=True, exist_ok=True)
            _record.update(per_gap=per_gap, dir=directory)
            try:
                _record["gap_s"] = float(os.environ.get(RECORD_GAP_ENV, "0.5"))
                _record["max"] = int(os.environ.get(RECORD_MAX_ENV, "4"))
                _record["substitute"] = int(os.environ.get(SUBSTITUTE_ENV, "0") or 0)
                _record["substitute_align"] = tuple(
                    int(v) for v in os.environ.get(SUBSTITUTE_ALIGN_ENV, "256,4").split(",") if v)
            except ValueError:
                pass
            if _record["substitute"] and "substitution_check" not in module.CAPABILITIES:
                _record["errors"].append("module has no substitution_check; P4 off")
                _record["substitute"] = 0
    _native = module


def _merged(lo, hi, a, b):
    """Union of GPU intervals lo..hi clipped to [a, b], as sorted disjoint spans."""
    spans = []
    for index in range(lo, hi):
        start, end, _ = _iv[index]
        start, end = max(start, a), min(end, b)
        if end <= start:
            continue
        if spans and start <= spans[-1][1]:
            if end > spans[-1][1]:
                spans[-1][1] = end
        else:
            spans.append([start, end])
    return spans


def _busy(spans, a, b):
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in spans)


def _attribute_locked(drain_all=False):
    """Join finished rounds with the GPU timeline (caller holds ``_lock``)."""
    global _iv_max
    if _native is None:
        return
    entries = _native.timeline_drain()
    if entries:
        for commit, start, end, status in sorted(entries, key=lambda e: e[1]):
            _timeline_counts["intervals"] += 1
            if status != 4 or end <= start:  # 4: MTLCommandBufferStatusCompleted
                _timeline_counts["errored"] += 1
                continue
            if not _iv_start or start >= _iv_start[-1]:
                _iv_start.append(start)
                _iv.append((start, end, commit))
            else:
                index = bisect.bisect_right(_iv_start, start)
                _iv_start.insert(index, start)
                _iv.insert(index, (start, end, commit))
            _iv_max = max(_iv_max, end - start)
    settled = drain_all or _native.timeline_pending() == 0
    horizon = _native.host_now() - 0.25
    while _pending:
        route, t0, t1, windows = _pending[0]
        if not settled and t1 > horizon:
            break
        _pending.popleft()
        lo = bisect.bisect_left(_iv_start, t0 - _iv_max)
        hi = bisect.bisect_left(_iv_start, t1)
        spans = _merged(lo, hi, t0, t1)
        row = _route_row(route)
        row["gpu_rounds"] = row.get("gpu_rounds", 0) + 1
        row["gpu_wall_ns"] = row.get("gpu_wall_ns", 0) + int((t1 - t0) * 1e9)
        row["gpu_busy_ns"] = row.get("gpu_busy_ns", 0) + int(_busy(spans, t0, t1) * 1e9)
        commits = sum(1 for i in range(lo, hi) if t0 <= _iv[i][2] <= t1)
        row["gpu_command_buffers"] = row.get("gpu_command_buffers", 0) + commits
        for a, b in windows:
            row["gpu_eval_ns"] = row.get("gpu_eval_ns", 0) + int((b - a) * 1e9)
            row["gpu_eval_busy_ns"] = row.get("gpu_eval_busy_ns", 0) + int(_busy(spans, a, b) * 1e9)
            last = max((min(e, b) for s, e in spans if s < b and e > a), default=None)
            if last is not None:
                row["gpu_eval_tail_ns"] = row.get("gpu_eval_tail_ns", 0) + int((b - last) * 1e9)
    # Keep only intervals a pending round can still overlap.
    floor = (_pending[0][1] if _pending else _native.host_now()) - _iv_max - 1.0
    cut = bisect.bisect_left(_iv_start, floor)
    if cut:
        del _iv_start[:cut]
        del _iv[:cut]


def _write_recording(route, handle, started):
    """One step_replay recording as JSON (P3 input); failures are reported, not raised."""
    try:
        summary = handle.summary()
        table = handle.dispatch_table()
        # ``pipelines`` is reflection looked up by function name, in the order of
        # ``summary["kernels"]``. It is not indexed by ``dispatch["pipeline"]``: name
        # the dispatch with ``summary["kernels"][dispatch["pipeline"]]`` (two pipelines
        # can share a name, and an unreflected one has no entry here).
        order = {name: i for i, name in reversed(list(enumerate(summary.get("kernels") or [])))}
        reflection = sorted((p for p in _native.pipelines() if p.get("function") in order),
                            key=lambda p: order[p.get("function")])
        index = _record["written"]
        path = Path(_record["dir"]) / f"round-{os.getpid()}-{index:04d}.json"
        previous = _record.get("previous")
        compared = None
        if previous is not None and previous[0] == route:
            # Structural equality with the previous recording of this route
            # (kernels, geometry, buffers, inline bytes): P4/P6 input.
            compared = _native.compare_recordings(previous[1], handle)
        _record["previous"] = (route, handle)
        document = {
            "schema": "mlx2.round-probe.recording.v1",
            "route": route, "index": index, "host_time": started, "wall_time": time.time(),
            "gpu": _native.gpu_identity(), "mlx": _native.MLX_VERSION,
            "compare_with_previous": compared,
            "summary": summary, "dispatches": table, "pipelines": reflection,
        }
        if _env_on(os.environ.get(ADDRESSES_ENV)) and "buffer_addresses" in _native.CAPABILITIES:
            document["buffers"] = [list(row) for row in _native.buffer_addresses(handle)]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(document))
        tmp.replace(path)
        _record["written"] = index + 1
        if index < _record["substitute"]:
            _substitution_pass(route, handle, index, summary)
        if index in _flag_ab_indices():
            _flag_ab_pass(route, handle, index, summary)
    except Exception as error:
        if len(_record["errors"]) < 16:
            _record["errors"].append(f"{type(error).__name__}: {error}")


def clean_dispatches(table, kernels, reflection):
    """Indices of dispatches whose replay after the round sees record-time inputs.

    ``reflection``: kernel name -> {buffer slot: writes}. A dispatch is clean
    when no buffer it reads is written by a later dispatch, and every buffer
    it writes is written last by it (so its post-round contents are its own
    output, which the replay arms copy before running).

    A recorded dispatch carries the encoder's whole binding table, including
    slots an earlier kernel bound that this kernel does not use (MLX never
    clears bindings). Only the slots in the kernel's reflection count; for an
    unreflected kernel every slot counts (access unknown).
    """
    def name(entry):
        pipeline = entry["pipeline"]
        return kernels[pipeline] if pipeline < len(kernels) else None

    last_write = {}
    for i, entry in enumerate(table):
        slots = reflection.get(name(entry))
        for slot in entry["slots"]:
            # Unreflected kernels: access unknown, so every binding may write.
            # Leftover bindings (not in the reflection) are not this kernel's.
            if slot[1] == "buffer" and (slots is None or slots.get(slot[0], False)):
                last_write[slot[2]] = i
    clean = []
    for i, entry in enumerate(table):
        slots = reflection.get(name(entry))
        if slots is None:
            continue  # unreflected: access unknown
        ok = True
        for slot in entry["slots"]:
            if slot[1] != "buffer":
                continue
            last = last_write.get(slot[2], -1)
            if slot[0] not in slots:
                continue  # left over from an earlier kernel in this encoder
            if (slots.get(slot[0]) and last != i) or (not slots.get(slot[0]) and last > i):
                ok = False
                break
        if ok:
            clean.append(i)
    return clean


def _flag_ab_indices():
    value = os.environ.get(FLAG_AB_ENV, "")
    try:
        return {int(v) for v in value.split(",") if v.strip()}
    except ValueError:
        return set()


def flag_ab_instance(table, kernels, clean):
    """One replay-safe dispatch per kernel for the P2 timing: the first clean
    instance (record-time inputs). Kernels with no clean instance are not
    timed: a stale input can steer a write (scatter indices, the destination
    offset of MLX's ``copy_gg_dynamic`` slice update) outside the private
    copies, so name lists cannot make a non-clean replay safe (review,
    2026-11-26)."""
    clean = set(clean)
    chosen = {}
    for i, entry in enumerate(table):
        name = kernels[entry["pipeline"]] if entry["pipeline"] < len(kernels) else None
        if name is None or entry.get("form") == "indirect" or i not in clean:
            continue
        chosen.setdefault(name, (i, True))
    return chosen


def _bootstrap_ratio(plain, flag, draws=2000, seed=0):
    import random
    ratios = [f / p for p, f in zip(plain, flag) if p > 0]
    if not ratios:
        return None
    rng = random.Random(seed)
    meds = sorted(statistics_median([rng.choice(ratios) for _ in ratios]) for _ in range(draws))
    return {"median": round(statistics_median(ratios), 4),
            "ci95": [round(meds[int(0.025 * draws)], 4), round(meds[int(0.975 * draws) - 1], 4)],
            "pairs": len(ratios)}


def statistics_median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def _flag_ab_pass(route, handle, index, summary):
    """P2: rank kernels by GPU time per round, then A/B the hot ones (module docstring)."""
    if not summary.get("drained"):
        _record["errors"].append(f"P2 refused recording {index}: not drained")
        return
    started = time.perf_counter()
    kernels = summary.get("kernels") or []
    table = handle.dispatch_table()
    reflection = {p.get("function"): {s[0]: s[2] for s in p.get("slots", []) if s[1] == "buffer"}
                  for p in _native.pipelines()}
    chosen = flag_ab_instance(table, kernels, clean_dispatches(table, kernels, reflection))
    counts = Counter(kernels[e["pipeline"]] for e in table if e["pipeline"] < len(kernels))
    ranking, failures = [], []
    for name, (i, clean) in sorted(chosen.items()):
        try:
            row = _native.flag_ab(handle, i, blocks=2, per_block=20)
        except Exception as error:
            row = {"status": "exception", "errors": f"{type(error).__name__}: {error}"}
        if row.get("status") != "timed":
            failures.append({"kernel": name, "dispatch": i, **{k: row.get(k) for k in ("status", "errors", "build_errors")}})
            continue
        us = statistics_median(list(row["plain_us"]))
        ranking.append({"kernel": name, "dispatch": i, "clean": clean, "count": counts[name],
                        "plain_us": round(us, 3), "round_us": round(us * counts[name], 1),
                        "build_plain": row.get("build_plain"), "build_flag": row.get("build_flag")})
    ranking.sort(key=lambda r: -r["round_us"])
    hot = [r for r in ranking[:10]] + [r for r in ranking[10:] if "nax" in r["kernel"]]
    results = []
    for r in hot:
        try:
            row = _native.flag_ab(handle, r["dispatch"], blocks=20, per_block=100)
        except Exception as error:
            row = {"status": "exception", "errors": f"{type(error).__name__}: {error}"}
        entry = {**r, "status": row.get("status"), "errors": row.get("errors")}
        if row.get("status") == "timed":
            plain, flag = list(row["plain_us"]), list(row["flag_us"])
            stats = _bootstrap_ratio(plain, flag)
            entry.update(plain_median_us=round(statistics_median(plain), 3),
                         flag_median_us=round(statistics_median(flag), 3), ratio=stats,
                         slower_beyond_3pct=bool(stats and stats["median"] > 1.03 and stats["ci95"][0] > 1.0))
        results.append(entry)
    total = sum(r["round_us"] for r in ranking)
    document = {
        "schema": "mlx2.round-probe.flag-ab.v1", "probe": "P2", "route": route,
        "recording_index": index, "distinct_kernels": len(set(kernels)),
        "kernels_timed": len(ranking), "kernels_untimed": failures,
        "untimed_no_clean_instance": sorted(n for n in set(kernels) if n not in chosen),
        "builds_failed": [r["kernel"] for r in ranking if not (r["build_plain"] and r["build_flag"])],
        "ranked_round_us_total": round(total, 1),
        "hot_share_of_ranked": round(sum(r["round_us"] for r in ranking[:10]) / total, 4) if total else None,
        "ranking": ranking, "ab": results,
        "verdict": "kill" if any(r.get("slower_beyond_3pct") for r in results) else "pass",
        "kill_criterion": "a hot kernel is slower with the flag by more than 3% with the 95% CI above 1.0",
        "seconds": round(time.perf_counter() - started, 3),
    }
    path = Path(_record["dir"]) / f"flag-ab-{os.getpid()}-{index:04d}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(document))
    tmp.replace(path)


def _substitution_pass(route, handle, index, summary):
    """P4: every dispatch of one recording, once per alignment (module docstring)."""
    if not summary.get("drained"):
        _record["errors"].append(f"P4 refused recording {index}: not drained")
        return
    kernels = summary.get("kernels") or []
    started = time.perf_counter()
    passes = []
    arms = _arms_mask(os.environ.get(SUBSTITUTE_ARMS_ENV))
    checksum = _env_on(os.environ.get(SUBSTITUTE_CHECKSUM_ENV))
    before = _native.buffer_checksums(handle) if checksum else None
    checksum_s = time.perf_counter() - started
    table = handle.dispatch_table()
    reflection = {p.get("function"): {s[0]: s[2] for s in p.get("slots", []) if s[1] == "buffer"}
                  for p in _native.pipelines()}
    clean = set(clean_dispatches(table, kernels, reflection))
    for align in _record["substitute_align"]:
        rows = []
        for i in range(summary.get("dispatches", 0)):
            if i not in clean:
                rows.append({"index": i, "kernel": kernels[table[i]["pipeline"]] if table[i]["pipeline"] < len(kernels) else "?",
                             "status": "skipped_stale_inputs"})
                continue
            try:
                rows.append(_native.substitution_check(handle, i, align=align, icb=True, arms=arms))
            except Exception as error:
                rows.append({"index": i, "status": "exception", "errors": f"{type(error).__name__}: {error}"})
        statuses = Counter(row.get("status") for row in rows)
        by_kernel = {}
        for row in rows:
            entry = by_kernel.setdefault(row.get("kernel", "?"), Counter())
            entry[row.get("status")] += 1
        passes.append({
            "align": align, "dispatches": len(rows), "statuses": dict(statuses),
            "with_bytes": sum(1 for row in rows if row.get("bytes_slots")),
            "wrote": sum(1 for row in rows if row.get("wrote")),
            "icb_ran": sum(1 for row in rows if row.get("icb_ran")),
            "kernels": {k: dict(v) for k, v in sorted(by_kernel.items())},
            "not_equal": [row for row in rows
                          if row.get("status") not in ("equal", "partial", "skipped_stale_inputs")][:64],
            "clean": len(clean),
            "kernels_with_clean_dispatch": len({r.get("kernel") for r in rows if r.get("status") != "skipped_stale_inputs"}),
            "copy_bytes_total": sum(row.get("copy_bytes_total") or 0 for row in rows),
            "largest_copies": sorted(
                ({"index": row["index"], "kernel": row.get("kernel"), "bytes": row.get("copy_bytes_total")}
                 for row in rows if row.get("copy_bytes_total")),
                key=lambda e: -e["bytes"])[:16],
        })
    changed = None
    if checksum:
        after = _native.buffer_checksums(handle)
        changed = []
        for (i, length, h0), (_, _, h1) in zip(before, after):
            if h0 != h1:
                users = []
                for d_index, d in enumerate(table):
                    for slot in d["slots"]:
                        if slot[1] == "buffer" and slot[2] == i:
                            name = kernels[d["pipeline"]] if d["pipeline"] < len(kernels) else "?"
                            users.append({"dispatch": d_index, "kernel": name, "slot": slot[0],
                                          "reflected_writes": reflection.get(name, {}).get(slot[0])})
                changed.append({"buffer": i, "length": length, "private": h0 is None,
                                "users": users[:12], "user_count": len(users)})
    document = {
        "schema": "mlx2.round-probe.substitution.v1", "probe": "P4", "route": route,
        "recording_index": index, "distinct_kernels": len(set(kernels)),
        "seconds": round(time.perf_counter() - started, 3), "passes": passes,
        "arms_mask": arms, "checksum": checksum, "checksum_seconds": round(checksum_s, 3),
        "buffers_changed_by_pass": changed,
        "buffers_checked": None if before is None else len(before),
        "private_buffers": None if before is None else sum(1 for b in before if b[2] is None),
    }
    path = Path(_record["dir"]) / f"substitution-{os.getpid()}-{index:04d}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(document))
    tmp.replace(path)


def uninstall() -> None:
    """Restore the originals (tests); counters are kept until :func:`reset`."""
    global ACTIVE, _installed
    with _lock:
        for (_, kind), (owner, original) in _originals.items():
            setattr(owner, kind, original)
        _originals.clear()
        _installed = False
        ACTIVE = False
    _detach_native()


def _detach_native():
    """Stop using the native module (its hooks stay installed for the process)."""
    global _native, _iv_max
    if _native is not None:
        try:
            _native.set_timeline(False)
        except Exception:
            pass
    _native = None
    _iv_max = 0.0
    _gpu.update(enabled=False, error=None)
    _record.update(per_gap=0, written=0, left=0, last_round_end=None, skip=False, previous=None)
    _record["errors"] = []


def reset() -> None:
    with _lock:
        _routes.clear()
        _pending.clear()
        _iv_start.clear()
        _iv.clear()


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
    __slots__ = ("route", "frame", "started", "outer", "_tokens", "_h0", "_recording")

    def __init__(self, route):
        self.route = route
        self._tokens = 0

    def __enter__(self):
        self.outer = getattr(_local, "route", None) is not None
        if self.outer:
            return self  # nested: the enclosing round owns the time
        self.frame = {"encode_ns": 0, "wait_ns": 0, "syncs": 0, "blocking": 0, "kinds": {}, "sites": {}}
        self._h0 = None
        self._recording = False
        if _native is not None:
            self.frame["windows"] = []
            self._h0 = _native.host_now()
            if _record["per_gap"]:
                self._arm_recording()
        _local.frame = self.frame
        _local.route = self.route
        self.started = time.perf_counter_ns()
        return self

    def _arm_recording(self):
        rec = _record
        if _native.recording_open():
            # A recording left open on this thread (an exit on another thread,
            # or an exception before record_end): end it before it grows.
            try:
                _native.record_end(0.0)
            except Exception:
                pass
            rec["leaked_recordings"] = rec.get("leaked_recordings", 0) + 1
        last = rec["last_round_end"]
        if last is not None and self._h0 - last > rec["gap_s"]:
            rec["left"] = rec["per_gap"]
            rec["skip"] = True  # the first round after a gap is not recorded
        elif rec["skip"]:
            rec["skip"] = False
            if rec["left"] > 0 and rec["written"] < rec["max"]:
                try:
                    _native.record_begin()
                    self._recording = True
                    rec["left"] -= 1
                except Exception as error:
                    if len(rec["errors"]) < 16:
                        rec["errors"].append(f"{type(error).__name__}: {error}")
        elif rec["left"] > 0 and rec["written"] < rec["max"]:
            try:
                _native.record_begin()
                self._recording = True
                rec["left"] -= 1
            except Exception as error:
                if len(rec["errors"]) < 16:
                    rec["errors"].append(f"{type(error).__name__}: {error}")

    def tokens(self, count):
        self._tokens += int(count)

    def __exit__(self, *exc):
        if self.outer:
            return False
        wall = time.perf_counter_ns() - self.started
        route = _local.route  # retag() may have refined the label
        _local.route = None
        _local.frame = None
        h1 = _native.host_now() if self._h0 is not None else None
        handle = None
        if self._recording:
            try:
                handle = _native.record_end(5.0)
            except Exception as error:
                if len(_record["errors"]) < 16:
                    _record["errors"].append(f"{type(error).__name__}: {error}")
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
            if h1 is not None:
                if self._recording:
                    row["recorded_rounds"] = row.get("recorded_rounds", 0) + 1
                elif len(_pending) < _PENDING_LIMIT:
                    _pending.append((route, self._h0, h1, tuple(self.frame["windows"])))
                else:
                    _timeline_counts["unattributed_rounds_dropped"] += 1
        if handle is not None:
            _write_recording(route, handle, self._h0)
        if h1 is not None:
            _record["last_round_end"] = _native.host_now() if handle is not None else h1
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
        **_gpu_summary(row),
    }


def _gpu_summary(row):
    if _native is None:
        return {}
    out = {key: row.get(key, 0) for key in GPU_KEYS}
    wall = out["gpu_wall_ns"]
    if out["gpu_rounds"] and wall:
        idle = wall - out["gpu_busy_ns"]
        eval_idle = out["gpu_eval_ns"] - out["gpu_eval_busy_ns"]
        out["gpu_per_round"] = {
            "wall_ms": round(wall / out["gpu_rounds"] / 1e6, 4),
            "busy_ms": round(out["gpu_busy_ns"] / out["gpu_rounds"] / 1e6, 4),
            "command_buffers": round(out["gpu_command_buffers"] / out["gpu_rounds"], 3),
            "idle_share": round(idle / wall, 4),
            "eval_idle_share": round(eval_idle / wall, 4),
            "eval_idle_less_tail_share": round(max(0, eval_idle - out["gpu_eval_tail_ns"]) / wall, 4),
            "outside_eval_idle_share": round(max(0, idle - eval_idle) / wall, 4),
        }
    return out


def status() -> dict:
    """``/v1/status`` and receipt block; ``{"enabled": False}`` when off."""
    if not ACTIVE:
        return {"enabled": False}
    if _native is not None:
        # Completed handlers run just after the device finishes; give the last
        # round's a moment so its request's receipt carries it.
        deadline = time.perf_counter() + 0.02
        while _native.timeline_pending() and time.perf_counter() < deadline:
            time.sleep(0.001)
    with _lock:
        _attribute_locked()
        routes = {route: _route_summary(row) for route, row in sorted(_routes.items())}
        gpu = _gpu_status_locked()
    return {
        "enabled": True,
        "schema": "mlx2.round-probe.v1",
        "scope": "process_cumulative_since_install",
        "installed_at": _installed_at,
        "clock": "host_wall_split_eval",
        "performance_claim": False,
        "command_buffer_env": {name: os.environ.get(name) for name in BUFFER_ENVS},
        "routes": routes,
        **({"gpu_timeline": gpu} if gpu else {}),
    }


def _gpu_status_locked():
    if not (_gpu["enabled"] or _gpu["error"]):
        return None
    out = dict(_gpu)
    if _native is not None:
        out.update(_timeline_counts, pending_rounds=len(_pending),
                   pending_command_buffers=_native.timeline_pending(),
                   dropped_entries=_native.timeline_dropped(),
                   recordings={k: _record.get(k) for k in ("per_gap", "dir", "gap_s", "max", "written", "errors",
                                                            "leaked_recordings")},
                   open_recordings=_native.open_recordings() if hasattr(_native, "open_recordings") else None)
    return out


install_from_env()
