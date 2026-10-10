import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from run_profile import (
    expand_server_argv,
    ladder_argv,
    validate_ladder_options,
    wait_ready,
)


def test_server_command_binds_reserved_port_and_localhost_only():
    argv = expand_server_argv(
        ["python", "-m", "mlx2.server", "--host", "127.0.0.1", "--port", "@PORT@"],
        23456,
    )
    assert argv[-1] == "23456"


def test_server_command_rejects_missing_or_nonlocal_bind():
    with pytest.raises(ValueError, match="127.0.0.1"):
        expand_server_argv(["python", "x", "--port", "@PORT@", "--host", "0.0.0.0"], 10)
    with pytest.raises(ValueError, match="exactly one"):
        expand_server_argv(["python", "x", "--host", "127.0.0.1", "--port", "1"], 10)


def test_ladder_command_uses_profile_args_without_shell():
    profile = {
        "model": "Qwen3.8",
        "model-id": "qwen",
        "route": "self_mtp",
        "artifact-identity": "fingerprint",
        "max-context": 8192,
        "cache-bytes": 4096,
        "apc-persistence": "off",
        "apc-persist-on-shutdown": "off",
        "max-lanes": 2,
        "max-inflight": 4,
        "prefill-step": 1024,
        "prefill-policy": {
            "prefill_depth_budget": None,
            "prefill_step_autoscale": False,
        },
        "mtp-policy": {"enabled": True},
        "draft-loop-policy": {
            "draft_loop": "auto",
            "draft_loop_threshold": 0.2,
            "draft_loop_widths": "1,2",
        },
        "wide": 2,
    }
    command = ladder_argv(
        profile, url="http://127.0.0.1:1234", output=Path("x.json"), server_pid=12
    )
    assert "http://127.0.0.1:1234" in command
    assert "--cache-bytes" in command and "4096" in command
    assert command[command.index("--server-pid") + 1] == "12"
    assert not any(";" in item or "$(" in item for item in command)
    with pytest.raises(ValueError, match="unsupported"):
        ladder_argv(
            {**profile, "unknown": "x"},
            url="http://127.0.0.1:1234",
            output=Path("x.json"),
            server_pid=12,
        )


def test_mock_local_status_readiness_is_cpu_only():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"healthy": True, "state": "ready"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class Alive:
        def poll(self):
            return None

    try:
        status = wait_ready(
            f"http://127.0.0.1:{server.server_port}/v1/status", Alive(), 2, 0.01
        )
        assert status["healthy"] is True
        assert status["state"] == "ready"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_profile_bounds_reject_duplicate_width_and_unbounded_repetitions():
    options = {
        "max-lanes": 2,
        "max-inflight": 4,
        "wide": 2,
        "cache-bytes": 4096,
        "prefill-step": 1024,
        "max-context": 8192,
        "runs": 3,
        "apc-persistence": "off",
        "apc-persist-on-shutdown": "off",
    }
    validate_ladder_options(options)
    with pytest.raises(ValueError, match="profile bounds"):
        validate_ladder_options({**options, "wide": 1})
    with pytest.raises(ValueError, match="3 repetitions"):
        validate_ladder_options({**options, "runs": 5})
