"""Every exit of the serving process leaves a line in its log (CPU only)."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from mlx2.exit_trace import ExitTrace
from mlx2.server import build_parser

SRC = str(Path(__file__).resolve().parents[1] / "src")

PRELUDE = """
import logging, os, signal, sys, threading, time
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
from mlx2.exit_trace import ExitTrace
"""


def spawn(body: str, *, preexec=None) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("MLX2_FAULT_LOG", None)
    return subprocess.Popen(
        [sys.executable, "-u", "-c", PRELUDE + textwrap.dedent(body)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        preexec_fn=preexec,
    )


def wait_ready(proc: subprocess.Popen) -> None:
    line = proc.stdout.readline()
    assert line.strip() == "READY", (line, proc.stderr.read() if proc.poll() is not None else "")


def test_sigterm_during_startup_is_logged_and_keeps_the_default_exit():
    proc = spawn("""
        ExitTrace().install()
        print("READY", flush=True)
        time.sleep(60)
    """)
    wait_ready(proc)
    proc.send_signal(signal.SIGTERM)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == -signal.SIGTERM
    assert "received SIGTERM" in err
    assert "exiting before the server was ready" in err


def test_sigint_during_startup_still_raises_keyboard_interrupt():
    proc = spawn("""
        ExitTrace().install()
        print("READY", flush=True)
        time.sleep(60)
    """)
    wait_ready(proc)
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(timeout=30)
    # Python's own exit for an uncaught KeyboardInterrupt: re-raised SIGINT.
    assert proc.returncode == -signal.SIGINT
    assert "received SIGINT" in err
    assert "KeyboardInterrupt" in err
    assert "mlx2 server exiting: reason=signal SIGINT" in err


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_signal_after_ready_is_logged_and_the_exit_line_names_it(signame):
    # The incident shape: a ready server receives SIGTERM from another tool's
    # ``pkill -f mlx2[.]server`` and shuts down cleanly.  Before this module
    # nothing in its log said so.
    proc = spawn("""
        trace = ExitTrace().install()
        stopped = threading.Event()
        trace.install_shutdown_signals(lambda signum=None, frame=None: stopped.set())
        print("READY", flush=True)
        stopped.wait(60)
        trace.set_reason("server shutdown")
    """)
    wait_ready(proc)
    proc.send_signal(getattr(signal, signame))
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 0
    assert f"received {signame}" in err
    assert "requesting server shutdown" in err
    assert f"mlx2 server exiting: reason=signal {signame}" in err


def test_sighup_inherited_as_ignored_stays_ignored():
    proc = spawn("""
        trace = ExitTrace().install()
        trace.install_shutdown_signals(lambda signum=None, frame=None: None)
        print("READY", flush=True)
        print(signal.getsignal(signal.SIGHUP) is signal.SIG_IGN, flush=True)
    """, preexec=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
    out, err = proc.communicate(timeout=30)
    assert out.split() == ["READY", "True"], err
    assert proc.returncode == 0


def test_fatal_signal_dumps_all_thread_stacks_to_the_fault_log(tmp_path):
    fault_log = tmp_path / "fault.log"
    proc = spawn(f"""
        ExitTrace().install({str(fault_log)!r})
        import faulthandler
        threading.Thread(target=time.sleep, args=(60,), name="idle-worker", daemon=True).start()
        faulthandler._sigsegv()
    """)
    proc.communicate(timeout=30)
    assert proc.returncode == -signal.SIGSEGV
    text = fault_log.read_text()
    assert "mlx2 faulthandler armed" in text
    assert "Fatal Python error: Segmentation fault" in text
    assert "Current thread 0x" in text and "\nThread 0x" in text  # every thread


def test_uncaught_main_thread_exception_is_logged_with_the_exit_reason():
    proc = spawn("""
        ExitTrace().install()
        raise RuntimeError("boom")
    """)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 1
    assert "CRITICAL uncaught exception in the main thread" in err
    assert "RuntimeError: boom" in err
    assert "mlx2 server exiting: reason=uncaught RuntimeError in main thread" in err


def test_thread_exception_is_logged(caplog):
    trace = ExitTrace()
    trace._install_excepthooks()
    try:
        with caplog.at_level(logging.CRITICAL, logger="mlx2.server"):
            worker = threading.Thread(target=lambda: 1 / 0, name="probe-thread")
            worker.start()
            worker.join()
            quiet = threading.Thread(target=sys.exit, name="exit-thread")
            quiet.start()
            quiet.join()
    finally:
        trace.uninstall()
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["uncaught exception in thread probe-thread"]
    assert caplog.records[0].exc_info[0] is ZeroDivisionError


def test_first_reason_wins_and_the_exit_line_reports_it(caplog):
    clock = iter([0.0, 12.5]).__next__
    trace = ExitTrace(clock=clock)
    trace.set_reason("signal SIGTERM")
    trace.set_reason("server shutdown")
    with caplog.at_level(logging.INFO, logger="mlx2.server"):
        trace.at_exit()
    assert caplog.records[-1].getMessage().startswith(
        "mlx2 server exiting: reason=signal SIGTERM"
    )
    assert "uptime=12.5s" in caplog.records[-1].getMessage()


def test_uninstall_restores_hooks_and_signals():
    before = (sys.excepthook, threading.excepthook, signal.getsignal(signal.SIGTERM))
    trace = ExitTrace().install()
    assert signal.getsignal(signal.SIGTERM) == trace._startup_signal
    trace.uninstall()
    assert (sys.excepthook, threading.excepthook, signal.getsignal(signal.SIGTERM)) == (
        before[0], before[1], signal.SIG_DFL,
    )
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_fault_log_flag_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MLX2_FAULT_LOG", raising=False)
    assert build_parser().parse_args(["--model", "m"]).fault_log is None
    monkeypatch.setenv("MLX2_FAULT_LOG", str(tmp_path / "f.log"))
    assert str(build_parser().parse_args(["--model", "m"]).fault_log) == str(tmp_path / "f.log")
