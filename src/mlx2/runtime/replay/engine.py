"""Model-agnostic record/replay API for fixed-shape step functions.

A *site* is one place in a decode loop that offers a step function
``f(inputs) -> outputs`` (lists of MLX arrays whose launches depend only on
the geometry key). ``StepSite.call`` always returns the step's outputs: from
the MLX function, or, once a program for the key is live, from a replay of
the recorded launches. Every call is counted by route, and every call that
did not replay is counted with its reason.

Per geometry key the site runs the bit-check protocol:

    new -> warming -> recording -> checking -> live
                         |            |
                         +--> resting +--> refused (named code, permanent)

* warming: the first ``warm_calls`` calls are served by MLX, not recorded;
  no key is recorded or checked inside a request's first ``quiet_calls``
  calls (a key that is already live serves at once);
* recording: two recordings in a row must be structurally equal (or the
  longer equal to the shorter plus copy launches: the shorter is sealed and
  needs two checks); other differences retry, then rest, then refuse
  (``signature_unstable``);
* checking: MLX serves the call; the program also replays on private copies
  of every written input, and each output is compared bit for bit through
  integer views (signed zeros and NaN payloads count); any difference
  refuses the key (``check_mismatch``);
* live: the program serves; a run-time legality refusal falls back to MLX
  for that call only. ``mode="shadow"`` checks every live call,
  ``mode="mixed"`` serves randomly from either path (QA), and
  ``mode="record_only"`` never seals.

The backend does the device work (recording, sealing, replay). The native
backend wraps the optional native module (``loader.load``); without it the
engine is unavailable and every call declines with the loader's reason
(``module_missing`` when no module is installed). Tests drive the protocol
with an in-process backend.
"""

from __future__ import annotations

import random
import threading
from collections import OrderedDict
from typing import NamedTuple

import mlx.core as mx

from . import loader

ROUTES = ("native", "recording", "checking", "mlx")
STATES = ("warming", "recording", "checking", "live", "resting", "refused")
# Named refusal codes (a refused key falls back to MLX permanently).
REFUSALS = (
    "unsupported_call", "unreflected_kernel", "non_buffer_argument",
    "unbound_argument", "cpu_bridge", "pinned_written", "aliased_written_inputs",
    "undeclared_read", "output_not_produced", "misaligned_output_view",
    "malformed_program", "signature_unstable", "check_mismatch", "check_blocked",
    "replay_failed", "replay_lost",
)


class Refusal(Exception):
    """A backend's named refusal of a recording or program."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class ReplayLost(RuntimeError):
    """A committed replay wrote into live inputs and produced no outputs (or an
    in-place check found writes outside its window). The state cannot be
    trusted: the caller must end the request. Never silently recovered."""


class GeometryKey(NamedTuple):
    caller: tuple
    weights: str
    layout: tuple
    mlx_version: str
    gpu: str


def _int_dtype(itemsize):
    return {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[itemsize]


def bit_view(a):
    """``a`` reinterpreted as unsigned integers of its item size."""
    if a.dtype == mx.bool_:
        return a.astype(mx.uint8)
    if a.dtype in (mx.uint8, mx.uint16, mx.uint32, mx.uint64):
        return a
    return mx.view(a, _int_dtype(a.dtype.size))


def bitwise_equal(a, b):
    """True when ``a`` and ``b`` hold the same bytes (dtype and shape too)."""
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    return bool(mx.array_equal(bit_view(a), bit_view(b)).item())


def private_copy(a):
    """A copy that no float operation touched (-0.0 and NaN payloads kept)."""
    if a.dtype == mx.bool_:
        return mx.array(a.astype(mx.uint8) + 0).astype(mx.bool_)
    ints = bit_view(a) + 0
    return ints if ints.dtype == a.dtype else mx.view(ints, a.dtype)


def default_layout(array):
    """Python-visible layout: dtype and shape (the native backend adds the
    strides of extent > 1 dimensions)."""
    return (str(array.dtype), tuple(array.shape))


class _KeyState:
    __slots__ = (
        "attempts",
        "calls",
        "check_blocks",
        "checks",
        "checks_needed",
        "program",
        "recordings",
        "refusal",
        "rest_until",
        "rests",
        "state",
    )

    def __init__(self):
        self.state = "warming"
        self.calls = 0
        self.recordings = []
        self.attempts = 0
        self.rests = 0
        self.rest_until = 0
        self.program = None
        self.checks = 0
        self.checks_needed = 1
        self.refusal = None
        self.check_blocks = 0


class Engine:
    """One per process: the backend plus its availability."""

    def __init__(self, backend=None, reason=None):
        self.backend = backend
        self.reason = None if backend is not None else (reason or "module_missing")
        self.lock = threading.Lock()

    @property
    def available(self):
        return self.backend is not None

    @classmethod
    def native(cls):
        module, reason = loader.load()
        if module is None:
            return cls(None, reason)
        caps = set(getattr(module, "CAPABILITIES", ()))
        if not {"record", "seal", "replay"} <= caps:
            # The built module records and introspects; sealing and replay
            # are not implemented yet (native/step_replay/step_replay.mm).
            return cls(None, "replay_unimplemented")
        return cls(NativeBackend(module), None)


_DEFAULT = None
_DEFAULT_LOCK = threading.Lock()


def default_engine():
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = Engine.native()
        return _DEFAULT


class NativeBackend:
    """Thin adapter over the native module's functions (device work in C++)."""

    def __init__(self, module):
        self.module = module
        self.mlx_version = module.MLX_VERSION

    def layout(self, array):
        info = self.module.array_layout(array)
        strides = tuple(s for s, n in zip(info["strides"], info["shape"]) if n > 1)
        return (info["dtype"], tuple(info["shape"]), strides)

    def gpu_identity(self):
        return self.module.gpu_identity()

    def begin_recording(self):
        self.module.record_begin()

    def end_recording(self):
        return self.module.record_end()

    def compare(self, a, b):
        return self.module.compare_recordings(a, b)

    def seal(self, recording, inputs, outputs, written, pinned):
        return self.module.seal(recording, inputs, outputs, list(written), pinned)

    def legality(self, program, inputs):
        return self.module.legality(program, inputs)

    def replay(self, program, inputs):
        return self.module.replay(program, inputs)

    def busy(self):
        return self.module.busy()

    def stuck(self):
        return self.module.stuck()


class StepSite:
    """One decode-loop site: owns its programs (LRU, bounded) and counters."""

    def __init__(self, name, *, engine=None, weights_identity="unknown", warm_calls=2,
                 quiet_calls=96, record_attempts=3, rest_calls=64, max_rests=2,
                 max_programs=16, max_check_blocks=4, mode="serve", seed=0):
        if mode not in ("serve", "shadow", "mixed", "record_only"):
            raise ValueError(f"replay site mode {mode!r}")
        self.name = name
        self._engine = engine
        self.weights_identity = weights_identity
        self.warm_calls = warm_calls
        self.quiet_calls = quiet_calls
        self.record_attempts = record_attempts
        self.rest_calls = rest_calls
        self.max_rests = max_rests
        self.max_programs = max_programs
        self.max_check_blocks = max_check_blocks
        self.mode = mode
        self.keys = OrderedDict()
        self.counts = {route: 0 for route in ROUTES}
        self.declines = {}
        self.widths = {}
        self.refusals = {}
        self.since_boundary = 0
        self.last_route = None
        self._random = random.Random(seed)

    @property
    def engine(self):
        if self._engine is None:
            self._engine = default_engine()
        return self._engine

    # ----- bookkeeping -----

    def note_request_boundary(self):
        self.since_boundary = 0

    def _route(self, route, reason=None):
        self.counts[route] += 1
        if reason is not None:
            self.declines[reason] = self.declines.get(reason, 0) + 1
        self.last_route = (route, reason)

    def _mlx(self, fn, inputs, reason):
        outputs = fn(inputs)
        self._route("mlx", reason)
        return outputs

    def _refuse(self, st, code):
        st.state, st.refusal, st.program = "refused", code, None
        st.recordings.clear()
        self.refusals[code] = self.refusals.get(code, 0) + 1

    def _key(self, backend, geometry, inputs):
        layout = tuple(backend.layout(a) for a in inputs)
        return GeometryKey(tuple(geometry) if isinstance(geometry, (tuple, list)) else
                           (geometry,), self.weights_identity, layout,
                           backend.mlx_version, backend.gpu_identity())

    def _state(self, key):
        st = self.keys.get(key)
        if st is None:
            st = self.keys[key] = _KeyState()
            while len(self.keys) > self.max_programs:
                self.keys.popitem(last=False)
        else:
            self.keys.move_to_end(key)
        return st

    # ----- the call -----

    def call(self, fn, inputs, geometry, *, written=(), pinned=()):
        """Outputs of ``fn(inputs)`` from MLX or a live program.

        ``written``: indices of inputs the step writes (commit-fused form).
        ``pinned``: long-lived arrays the program may read but never write."""
        self.since_boundary += 1
        engine = self.engine
        if not engine.available:
            return self._mlx(fn, inputs, engine.reason)
        backend = engine.backend
        # Every input is evaluated first: the layout key reads the strides of
        # materialized arrays, and no earlier producer of an input may run
        # inside a recording or race a replay.
        mx.eval(inputs)
        key = self._key(backend, geometry, inputs)
        st = self._state(key)
        st.calls += 1
        if st.state == "refused":
            return self._mlx(fn, inputs, "refused:" + st.refusal)
        if st.state == "live":
            width = geometry[1] if isinstance(geometry, (tuple, list)) and len(geometry) > 1 \
                and isinstance(geometry[1], int) else None
            return self._live(fn, inputs, st, backend, written, width)
        if backend.busy():
            return self._mlx(fn, inputs, "busy")
        if backend.stuck():
            return self._mlx(fn, inputs, "stuck")
        if st.state == "warming":
            if st.calls <= self.warm_calls:
                return self._mlx(fn, inputs, "warming")
            st.state = "recording"
        if self.since_boundary <= self.quiet_calls:
            return self._mlx(fn, inputs, "quiet_start")
        if st.state == "resting":
            if st.calls < st.rest_until:
                return self._mlx(fn, inputs, "resting")
            st.state = "recording"
        if st.state == "recording":
            return self._record(fn, inputs, st, backend, written, pinned)
        return self._check(fn, inputs, st, backend, written)

    def _record(self, fn, inputs, st, backend, written, pinned):
        # MLX's queue drains before the window opens, so the capture holds only
        # the step's own launches (the inputs were evaluated in ``call``).
        mx.synchronize()
        backend.begin_recording()
        try:
            outputs = fn(inputs)
            mx.eval(outputs)
        finally:
            recording = backend.end_recording()
        self._route("recording")
        st.recordings.append((recording, tuple(inputs), outputs, tuple(written), tuple(pinned)))
        if len(st.recordings) < 2:
            return outputs
        earlier, latest = st.recordings[-2:]
        verdict = backend.compare(earlier[0], latest[0])
        if verdict == "different":
            st.recordings = st.recordings[-1:]
            st.attempts += 1
            if st.attempts >= self.record_attempts:
                st.attempts, st.recordings = 0, []
                st.rests += 1
                if st.rests > self.max_rests:
                    self._refuse(st, "signature_unstable")
                else:
                    st.state, st.rest_until = "resting", st.calls + self.rest_calls
            return outputs
        if self.mode == "record_only":
            st.recordings = st.recordings[-1:]
            return outputs
        sealed, st.checks_needed = latest, 1
        if verdict == "copy_superset_a":  # a = b + copies: b is the program
            st.checks_needed = 2
        elif verdict == "copy_superset_b":
            sealed, st.checks_needed = earlier, 2
        try:
            st.program = backend.seal(*sealed)
        except Refusal as exc:
            self._refuse(st, exc.code)
            return outputs
        st.recordings = []
        st.state = "checking"
        return outputs

    def _replay_checked(self, fn, inputs, st, backend, written):
        """MLX serves; the program replays on private copies and must match
        bit for bit. Returns (mlx outputs, passed)."""
        # Snapshot before the reference step can update a written input.
        trial = [private_copy(a) if i in written else a for i, a in enumerate(inputs)]
        mx.eval(trial)
        outputs = fn(inputs)
        mx.eval(outputs)
        reason = backend.legality(st.program, trial)
        if reason is not None:
            return outputs, reason
        try:
            replayed = backend.replay(st.program, trial)
        except ReplayLost:
            raise
        except Refusal as exc:
            return outputs, "refusal:" + exc.code
        if len(replayed) != len(outputs) or not all(
                bitwise_equal(x, y) for x, y in zip(replayed, outputs)):
            return outputs, False
        return outputs, True

    def _check(self, fn, inputs, st, backend, written):
        try:
            outputs, verdict = self._replay_checked(fn, inputs, st, backend, written)
        except ReplayLost:
            self._refuse(st, "replay_lost")
            raise
        self._route("checking")
        if verdict is True:
            st.checks += 1
            if st.checks >= st.checks_needed:
                st.state = "live"
        elif verdict is False:
            self._refuse(st, "check_mismatch")
        else:
            st.check_blocks += 1
            self.declines["check_blocked:" + verdict] = (
                self.declines.get("check_blocked:" + verdict, 0) + 1)
            if st.check_blocks > self.max_check_blocks:
                self._refuse(st, "check_blocked")
        return outputs

    def _live(self, fn, inputs, st, backend, written, width=None):
        if self.mode == "shadow":
            try:
                outputs, verdict = self._replay_checked(fn, inputs, st, backend, written)
            except ReplayLost:
                self._refuse(st, "replay_lost")
                raise
            if verdict is False:
                self._refuse(st, "check_mismatch")
            self._route("checking", None if verdict is True else "shadow:" + str(verdict))
            return outputs
        if self.mode == "mixed" and self._random.random() < 0.5:
            return self._mlx(fn, inputs, "mixed")
        if backend.busy():
            return self._mlx(fn, inputs, "busy")
        if backend.stuck():
            return self._mlx(fn, inputs, "stuck")
        reason = backend.legality(st.program, inputs)
        if reason is not None:
            return self._mlx(fn, inputs, "legality:" + reason)
        try:
            outputs = backend.replay(st.program, inputs)
        except ReplayLost:
            self._refuse(st, "replay_lost")
            raise
        except Refusal as exc:
            self._refuse(st, "replay_failed")
            return self._mlx(fn, inputs, "replay_failed:" + exc.code)
        self._route("native")
        self.widths[width] = self.widths.get(width, 0) + 1
        return outputs

    def receipt(self):
        states = {}
        for st in self.keys.values():
            states[st.state] = states.get(st.state, 0) + 1
        engine = self._engine
        return {"site": self.name, "mode": self.mode,
                "engine_available": bool(engine and engine.available),
                "engine_reason": None if engine is None else engine.reason,
                "routes": dict(self.counts), "declines": dict(self.declines),
                "refusals": dict(self.refusals), "keys": states,
                "native_widths": {str(k): v for k, v in self.widths.items()},
                "last_route": self.last_route}
