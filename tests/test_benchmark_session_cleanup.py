"""Exercise the benchmark supervisor with local processes. No model or GPU."""
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_guarded_benchmark import owns_listener


def test_crashed_driver_reaps_reparented_server_group(tmp_path):
    pidfile = tmp_path / "server.pid"
    code = """
import os, pathlib, signal, subprocess, sys
server = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)'], process_group=0)
pathlib.Path(sys.argv[1]).write_text(str(server.pid))
os.kill(os.getpid(), signal.SIGKILL)
"""
    foreign = subprocess.Popen(["sleep", "90"], start_new_session=True)
    try:
        result = subprocess.run([sys.executable, SCRIPTS / "run_guarded_benchmark.py", "--",
                                 sys.executable, "-c", code, str(pidfile)], timeout=20)
        assert result.returncode == 128 + signal.SIGKILL
        assert foreign.poll() is None
        pid = int(pidfile.read_text())
        row = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
        assert not row.stdout.strip() or row.stdout.strip().startswith("Z")
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(not shutil.which("lsof"), reason="lsof required by Maxwell launchers")
def test_foreign_listener_never_satisfies_readiness():
    foreign = subprocess.Popen(["sleep", "60"])
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            port = sock.getsockname()[1]
            assert owns_listener(os.getpid(), port)
            assert not owns_listener(foreign.pid, port)
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)
