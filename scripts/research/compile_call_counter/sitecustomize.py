"""Count calls of every ``mx.compile`` function in a process (diagnostic only).

Put this directory first on ``PYTHONPATH`` and set ``MLX2_COMPILE_COUNTS_OUT``
to a JSON path.  ``mlx.core.compile`` is wrapped before any mlx2 module is
imported, so decorators and call-time ``mx.compile(...)`` sites alike return a
counting wrapper around the real compiled function.  A daemon thread rewrites
the file every two seconds (a server stopped by signal keeps the last
snapshot): ``{"sites": {"module:qualname": {"compiled": n, "calls": m}}}``.

Used by the MLX upgrade requalification (ml-explore/mlx #4953 changes what
``mx.compile`` fuses) as the receipt that compiled functions actually ran on
the route under test.  It changes no computed value: the wrapper calls the
compiled function with the same arguments.  Without the environment variable
it does nothing.

``calls`` counts Python-level calls.  A compiled function called inside
another compiled function is counted only while the outer one traces (its
replays bypass Python), so the outer site is the engagement receipt.  The
file is also written at exit and on SIGTERM (chaining any earlier handler).
"""

import atexit
import functools
import json
import os
import signal
import threading
import time

_OUT = os.environ.get("MLX2_COMPILE_COUNTS_OUT")

if _OUT:
    import mlx.core as _mx

    _real_compile = _mx.compile
    _lock = threading.Lock()
    _sites: dict = {}

    def _site_name(fun) -> str:
        target = getattr(fun, "func", fun)  # functools.partial
        module = getattr(target, "__module__", None) or "?"
        name = getattr(target, "__qualname__", None) or type(target).__name__
        code = getattr(target, "__code__", None)
        line = f"@{code.co_firstlineno}" if code is not None and "<lambda>" in name else ""
        return f"{module}:{name}{line}"

    class _Counted:
        """Callable without ``__get__``: like MLX's compiled object, it does
        not bind ``self`` when stored as a class attribute."""

        __slots__ = ("_compiled", "_site", "__wrapped__")

        def __init__(self, compiled, site, fun):
            self._compiled, self._site, self.__wrapped__ = compiled, site, fun

        def __call__(self, *call_args, **call_kwargs):
            with _lock:
                _sites[self._site]["calls"] += 1
            return self._compiled(*call_args, **call_kwargs)

    def _counting_compile(fun=None, *args, **kwargs):
        if fun is None:
            return functools.partial(_counting_compile, *args, **kwargs)
        compiled = _real_compile(fun, *args, **kwargs)
        site = _site_name(fun)
        with _lock:
            entry = _sites.setdefault(site, {"compiled": 0, "calls": 0})
            entry["compiled"] += 1

        return _Counted(compiled, site, fun)

    _mx.compile = _counting_compile

    def _dump() -> None:
        with _lock:
            payload = {"pid": os.getpid(), "time": time.time(),
                       "mlx": getattr(_mx, "__version__", "?"),
                       "compile_disabled_env": "MLX_DISABLE_COMPILE" in os.environ,
                       "sites": {k: dict(v) for k, v in sorted(_sites.items())}}
        tmp = f"{_OUT}.tmp.{os.getpid()}"
        with open(tmp, "w") as stream:
            json.dump(payload, stream, indent=1)
        os.replace(tmp, _OUT)

    def _loop() -> None:
        while True:
            time.sleep(2.0)
            try:
                _dump()
            except Exception:  # noqa: BLE001 - diagnostic only
                pass

    threading.Thread(target=_loop, name="compile-call-counter", daemon=True).start()
    atexit.register(_dump)
    _previous = signal.getsignal(signal.SIGTERM)

    def _on_term(signum, frame):
        try:
            _dump()
        finally:
            if callable(_previous):
                _previous(signum, frame)
            else:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, _on_term)
