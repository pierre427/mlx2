"""CPU regressions for launcher ownership and benchmark verdicts. No model is loaded."""
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "research"))

import ling32_air_prefill_ab as prefill
import ling32_air_vision_smoke as vision
import maxwell_abba_ladder as maxwell
import raon1_mini_prefill_ladder as raon
import served_cmdbuf_probe as cmdbuf


def test_guard_preserves_child_exit_status():
    result = subprocess.run([sys.executable, SCRIPTS / "run_guarded_benchmark.py",
                             "--", sys.executable, "-c", "raise SystemExit(17)"])
    assert result.returncode == 17


def test_guard_cancellation_reaches_server_in_separate_session(tmp_path):
    marker = tmp_path / "child"
    parent = """
import pathlib, subprocess, sys, time
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)'], start_new_session=True)
pathlib.Path(sys.argv[1]).write_text(str(p.pid))
p.wait()
"""
    guard = subprocess.Popen([sys.executable, SCRIPTS / "run_guarded_benchmark.py",
                              "--", sys.executable, "-c", parent, str(marker)])
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        pid = int(marker.read_text())
        guard.terminate()
        assert guard.wait(timeout=40) == 128 + signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if guard.poll() is None:
            guard.terminate()
            guard.wait(timeout=40)


@pytest.mark.skipif(not shutil.which("zsh"), reason="zsh qualification launchers")
def test_server_cleanup_leaves_unrelated_child_alive():
    script = """
source "$1"
sleep 60 & foreign=$!
trap 'kill $foreign 2>/dev/null; wait $foreign 2>/dev/null' EXIT
owned_server_exec sleep 60 & SRV=$!
for i in {1..50}; do kill -0 -- -$SRV 2>/dev/null && break; sleep .02; done
owned=$SRV
stop_server
! kill -0 $owned 2>/dev/null && kill -0 $foreign
"""
    subprocess.run(["zsh", "-fc", script, "zsh", SCRIPTS / "owned_server.zsh"], check=True)


def test_maxwell_unrelated_tied_pairs_and_empty_error_traces_do_not_pass():
    def arm(token, top):
        return {"found": True, "tokens": [{"id": token, "top2": top}]}

    a = arm(1, [[1, -1], [2, -1]])
    b = arm(3, [[3, -1], [4, -1.1]])
    assert maxwell.classify(a, b)["verdict"] == "unclassified"
    tied = arm(2, [[2, -1], [4, -1.1]])
    assert maxwell.classify(a, tied)["verdict"] == "near_tie_equivalent"
    failed = {"tokens": [], "found": False, "error": "request failed"}
    assert maxwell.classify(failed, failed)["verdict"] == "unclassified"


@pytest.mark.parametrize("module", [raon, cmdbuf])
def test_truncated_stream_refuses_partial_measurement(module, monkeypatch):
    delta = {"choices": [{"delta": {"content": "partial"}}], "usage": {"completion_tokens": 1},
             "mlx2": {"route": "ordinary"}}
    payload = ("data: " + json.dumps(delta) + "\n").encode()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(ValueError, match="incomplete"):
        module.stream("http://localhost", {})


def test_prefill_compares_keyed_cells_in_any_order_and_rejects_missing_rows(tmp_path):
    def write(name, rows):
        path = tmp_path / name
        path.write_text(json.dumps({"checkpoint": "fixture@abc", "rows": rows}))
        return path

    rows = [{"target_tokens": n, "repeat": 0, "token_ids": [n], "prefill_tok_s": n}
            for n in (100, 200)]
    args = SimpleNamespace(reference=write("a", rows), candidate=write("b", rows[::-1]),
                           output=tmp_path / "comparison")
    prefill.compare(args)
    assert all(row["identical_tokens"] for row in json.loads(args.output.read_text())["rows"])
    write("b", rows[:1])
    with pytest.raises(ValueError, match="same checkpoint and cells"):
        prefill.compare(args)


def test_vision_refusal_rejects_an_unrelated_server_error(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "served_model", lambda *a: "test")
    monkeypatch.setattr(vision, "call", lambda *a: (500, {"error": "internal server error"}))
    args = SimpleNamespace(base_url="http://localhost", cases="image", checkpoint="fixture@abc",
                           output=tmp_path / "result")
    assert vision.run_refusal(args) == 1


@pytest.mark.parametrize("override", ["--port=8282", "--por=8282", "--host 0.0.0.0"])
def test_raon_extra_args_refuse_socket_override_before_start(tmp_path, monkeypatch, override):
    argv = ["ladder", "--model", str(tmp_path), "--cpg-lock", str(tmp_path),
            "--out", str(tmp_path / "evidence"), "--extra-args=" + override]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as result:
        raon.main()
    assert result.value.code == 2
    assert not (tmp_path / "evidence").exists()
