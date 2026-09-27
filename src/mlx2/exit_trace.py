"""Leave a trace for every exit of the serving process.

A server that stops must say why in its own log.  On 2026-09-25 a Qwen3.8 27B
server went away mid-measurement with no log line, traceback or crash report:
another tool's ``pkill -f mlx2[.]server`` sent SIGTERM, and the signal
controller shut down cleanly without saying so (see
``qualification/runs/silent-death-20260926``).  This module makes each exit
path leave one line:

* a signal the process can catch (SIGTERM, SIGINT, SIGHUP) is logged with
  the pid, parent pid and uptime when it arrives, during startup as well as
  after the engine is ready;
* an uncaught exception in the main thread or any other thread is logged
  through ``logging`` with its traceback;
* the interpreter's exit is logged by an ``atexit`` line carrying the reason;
* a fatal signal (SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL) dumps every
  thread's Python stack through :mod:`faulthandler`, to stderr or to a file.

SIGKILL (including a kernel jetsam kill) cannot be caught by any process; the
last line in the log is then whatever preceded it, and the caller's exit
status (-9) or a JetsamEvent report is the only trace.

Normal behaviour is unchanged: startup signals still kill the process with
the default action (the same exit status), and after the engine is ready the
server's shutdown controller still decides what a signal does.
"""

from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import signal
import sys
import threading
import time

LOG = logging.getLogger("mlx2.server")

#: Catchable signals whose default action ends the process.
TERMINATING_SIGNALS = ("SIGTERM", "SIGINT", "SIGHUP")


def signal_name(signum) -> str:
    try:
        return signal.Signals(signum).name
    except (TypeError, ValueError):
        return str(signum)


class ExitTrace:
    """Record why the process is exiting and say so on the way out."""

    def __init__(self, *, logger: logging.Logger = LOG, clock=time.monotonic):
        self.logger = logger
        self._clock = clock
        self.started = clock()
        self.reason: str | None = None
        self.fault_file = None
        self._installed_signals: list[int] = []
        self._previous_hooks = None

    # -- bookkeeping -------------------------------------------------------
    def uptime(self) -> float:
        return self._clock() - self.started

    def set_reason(self, reason: str) -> None:
        """Record the first reason given; later ones do not overwrite it."""
        if self.reason is None:
            self.reason = reason

    def log_signal(self, signum, action: str) -> None:
        name = signal_name(signum)
        self.set_reason(f"signal {name}")
        self.logger.warning(
            "received %s (pid %d, parent pid %d, uptime %.1fs); %s",
            name, os.getpid(), os.getppid(), self.uptime(), action,
        )

    # -- installation ------------------------------------------------------
    def install(self, fault_log: str | os.PathLike | None = None) -> "ExitTrace":
        self._install_faulthandler(fault_log)
        self._install_excepthooks()
        atexit.register(self.at_exit)
        self._install_startup_signals()
        return self

    def _install_faulthandler(self, fault_log) -> None:
        if fault_log is not None:
            # Line-buffered and kept open for the process lifetime: the
            # handler writes to the descriptor from inside a signal handler.
            self.fault_file = open(fault_log, "a", buffering=1)  # noqa: SIM115
            self.fault_file.write(
                f"# mlx2 faulthandler armed: pid {os.getpid()} at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            faulthandler.enable(file=self.fault_file, all_threads=True)
        elif not faulthandler.is_enabled() and sys.stderr is not None:
            faulthandler.enable(file=sys.stderr, all_threads=True)

    def _install_excepthooks(self) -> None:
        self._previous_hooks = (sys.excepthook, threading.excepthook)
        sys.excepthook = self._sys_excepthook
        threading.excepthook = self._thread_excepthook

    def _sys_excepthook(self, exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            self.set_reason("KeyboardInterrupt")
        else:
            self.set_reason(f"uncaught {exc_type.__name__} in main thread")
        self.logger.critical(
            "uncaught exception in the main thread", exc_info=(exc_type, exc, tb)
        )

    def _thread_excepthook(self, args) -> None:
        if args.exc_type is SystemExit:
            return  # threading's default hook ignores it too
        name = args.thread.name if args.thread is not None else "<unknown>"
        self.logger.critical(
            "uncaught exception in thread %s",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    def _install_startup_signals(self) -> None:
        """Log a terminating signal that arrives before the server is ready.

        The default action (death by that signal) is kept: the handler logs,
        restores the default and re-delivers the signal.  A signal the
        process inherited as ignored (``nohup`` ignores SIGHUP) stays ignored.
        """
        if threading.current_thread() is not threading.main_thread():
            return
        for name in TERMINATING_SIGNALS:
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            current = signal.getsignal(signum)
            if current is signal.SIG_IGN:
                continue
            if name == "SIGINT" and current is not signal.default_int_handler:
                continue  # someone else owns SIGINT; leave it
            signal.signal(signum, self._startup_signal)
            self._installed_signals.append(signum)

    def _startup_signal(self, signum, frame=None) -> None:
        self.log_signal(signum, "exiting before the server was ready")
        self._flush_logs()
        if signum == signal.SIGINT:
            # Python's own default: KeyboardInterrupt in the main thread.
            signal.signal(signum, signal.default_int_handler)
            signal.default_int_handler(signum, frame)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def shutdown_handler(self, stop):
        """Wrap a ready server's shutdown controller so each signal is logged."""

        def handler(signum=None, frame=None):
            self.log_signal(signum, "requesting server shutdown")
            return stop(signum, frame)

        return handler

    def install_shutdown_signals(self, stop) -> list[int]:
        """Route the terminating signals to ``stop`` once the server is ready.

        SIGHUP joins SIGTERM and SIGINT unless it was inherited as ignored;
        its default action would otherwise end the process with no trace.
        """
        handler = self.shutdown_handler(stop)
        routed = []
        for name in TERMINATING_SIGNALS:
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            if name == "SIGHUP" and signal.getsignal(signum) is signal.SIG_IGN:
                continue
            signal.signal(signum, handler)
            routed.append(signum)
        return routed

    # -- exit --------------------------------------------------------------
    def at_exit(self) -> None:
        exc = getattr(sys, "last_value", None)
        reason = self.reason or (
            f"uncaught {type(exc).__name__}" if exc is not None else "normal return"
        )
        self.logger.info(
            "mlx2 server exiting: reason=%s pid=%d uptime=%.1fs",
            reason, os.getpid(), self.uptime(),
        )
        self._flush_logs()

    def _flush_logs(self) -> None:
        for handler in list(logging.getLogger().handlers) + list(self.logger.handlers):
            try:
                handler.flush()
            except Exception:  # noqa: BLE001 - best effort on the way out
                pass

    def uninstall(self) -> None:
        """Undo :meth:`install` (tests)."""
        try:
            atexit.unregister(self.at_exit)
        except Exception:  # noqa: BLE001
            pass
        if self._previous_hooks is not None:
            sys.excepthook, threading.excepthook = self._previous_hooks
            self._previous_hooks = None
        if threading.current_thread() is threading.main_thread():
            for signum in self._installed_signals:
                default = signal.default_int_handler if signum == signal.SIGINT else signal.SIG_DFL
                signal.signal(signum, default)
        self._installed_signals = []
        if self.fault_file is not None:
            faulthandler.disable()
            self.fault_file.close()
            self.fault_file = None
