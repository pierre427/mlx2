"""Sweep 2026-10-09 (ops-cli#2): startup configuration errors are usage errors.

A nonexistent ``--model``, a malformed or non-object ``--execution-policy``
file and out-of-range engine limits used to escape ``main()`` as interpreter
tracebacks (exit 1, logged by the fault handler as an uncaught exception)
because ``main()`` never ran ``ServingEngine.validate_arguments`` and guarded
the model/policy reads for ``ValueError`` only.  Every other flag ends in a
one-line ``mlx2-serve: error: ...`` with exit status 2; these must too.
"""

import socket
import sys
import types

import pytest


class _FakeEngine:
    """Enough of ServingEngine for main() past the bind, without weights."""

    instances = []
    reasoning_signer = None
    validate_arguments = None  # bound to the real check in _run_main

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.instances.append(self)

    def close(self):
        self.closed = True


def _run_main(monkeypatch, capsys, argv, *, fake_model=False, fake_engine=False):
    from mlx2 import exit_trace, server
    from mlx2.adapters import registry

    monkeypatch.setattr(exit_trace.ExitTrace, "install", lambda self, fault_log=None: self)
    if fake_model:
        monkeypatch.setattr(
            registry,
            "inspect_model",
            lambda path: types.SimpleNamespace(artifact_fingerprint="sha256:fake"),
        )
        monkeypatch.setattr(
            server,
            "resolve_route_selection",
            lambda args, policy, resolution: types.SimpleNamespace(
                native_mtp=False, route="ordinary", source="engine_argument"
            ),
        )
        monkeypatch.setattr(
            server, "resolve_execution_policy_defaults", lambda policy, *a, **k: policy
        )
    if fake_engine:
        _FakeEngine.validate_arguments = staticmethod(server.ServingEngine.validate_arguments)
        _FakeEngine.instances.clear()
        monkeypatch.setattr(server, "ServingEngine", _FakeEngine)
    bound = []
    real_bind = socket.socket.bind
    monkeypatch.setattr(
        socket.socket, "bind", lambda self, *a: (bound.append(a), real_bind(self, *a))[1]
    )
    monkeypatch.setattr(sys, "argv", ["mlx2.server", *argv])
    with pytest.raises(SystemExit) as exc:
        server.main()
    captured = capsys.readouterr()
    assert exc.value.code == 2, captured.err
    assert "Traceback" not in captured.err
    assert "error:" in captured.err
    if not fake_engine:
        assert not bound, "a usage error must be reported before the port is taken"
    return captured.err


def test_missing_model_directory_is_a_usage_error(monkeypatch, capsys, tmp_path):
    err = _run_main(monkeypatch, capsys, ["--model", str(tmp_path / "absent"), "--port", "0"])
    assert "absent" in err


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("[]", "JSON object"),  # inspect_model raises TypeError for this
        ('{"a": ', "Expecting"),
        ('{"model_type": "nope"}', "No mlx2 adapter"),
    ],
)
def test_unusable_model_config_is_a_usage_error(monkeypatch, capsys, tmp_path, text, fragment):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(text)
    err = _run_main(monkeypatch, capsys, ["--model", str(model), "--port", "0"])
    assert "cannot inspect --model" in err and fragment in err


@pytest.mark.parametrize(
    "text, fragment",
    [
        ('{"a": 1,}', "execution-policy"),
        ("[1, 2]", "JSON object"),
        ('"policy"', "JSON object"),
    ],
)
def test_malformed_execution_policy_file_is_a_usage_error(
    monkeypatch, capsys, tmp_path, text, fragment
):
    policy = tmp_path / "policy.json"
    policy.write_text(text)
    err = _run_main(
        monkeypatch,
        capsys,
        ["--model", "unused", "--execution-policy", str(policy), "--port", "0"],
        fake_model=True,
    )
    assert fragment in err


def test_missing_execution_policy_file_is_a_usage_error(monkeypatch, capsys, tmp_path):
    err = _run_main(
        monkeypatch,
        capsys,
        ["--model", "unused", "--execution-policy", str(tmp_path / "none.json"), "--port", "0"],
        fake_model=True,
    )
    assert "execution-policy" in err


@pytest.mark.parametrize(
    "flags, fragment",
    [
        (["--max-lanes", "0"], "limits must be positive"),
        (["--max-inflight", "0"], "limits must be positive"),
        (["--coalesce-window-ms", "-1"], "coalescing window"),
        (["--gpu-keep-warm-interval", "0.01"], "gpu_keep_warm"),
        (["--cache-bytes", "-1"], "limits must be positive"),
        (["--host-prompt-cache-entries", "-5"], "host prompt cache"),
        (["--apc-session-max-ttl-seconds", "0"], "apc_session_max_ttl_seconds"),
        (["--max-loras", "-1"], "max_loras"),
        # A wrongly shaped value: the validate-only prefix raises TypeError.
        (["--lane-policy", "[]"], "lane policy must be a JSON object"),
    ],
)
def test_out_of_range_engine_flags_are_usage_errors(monkeypatch, capsys, flags, fragment):
    err = _run_main(
        monkeypatch, capsys, ["--model", "unused", "--port", "0", *flags], fake_model=True
    )
    assert fragment in err


@pytest.mark.parametrize(
    "flag, text, fragment",
    [
        ("--tool-backend-config", None, "tool-backend-config"),
        ("--tool-backend-config", "[1]", "servers object"),
        ("--tool-backend-config", '{"servers": {"a": {"server_url": "ftp://x"}}}', "allowlisted"),
        ("--agent-compat-tenants", None, "agent-compat-tenants"),
        ("--agent-compat-tenants", "[1]", "JSON object"),
        ("--agent-compat-tenants", '{"t": {"agent_compat": "maybe"}}', "on or off"),
    ],
)
def test_operator_json_files_are_read_before_the_bind(
    monkeypatch, capsys, tmp_path, flag, text, fragment
):
    # The same shape as the engine limits: these files were read after the
    # port was taken and the engine built, and a bad one was a traceback.
    path = tmp_path / "config.json"
    if text is not None:
        path.write_text(text)
    err = _run_main(
        monkeypatch,
        capsys,
        ["--model", "unused", "--port", "0", flag, str(path)],
        fake_model=True,
    )
    assert fragment in err


def test_unusable_semantic_memory_closes_the_engine_and_the_server(
    monkeypatch, capsys, tmp_path
):
    # Semantic memory needs the engine's bridge, so it starts after the bind;
    # a bad root or neural artifact must still be a usage error that releases
    # the engine and the listening socket instead of a traceback.
    from mlx2 import server

    closed = []
    real_close = server.BoundedHTTPServer.server_close
    monkeypatch.setattr(
        server.BoundedHTTPServer,
        "server_close",
        lambda self: (closed.append(self), real_close(self))[1],
    )
    err = _run_main(
        monkeypatch,
        capsys,
        [
            "--model", "unused", "--port", "0", "--semantic-memory",
            "--api-state-dir", str(tmp_path / "state"),
            "--qualification-mode", "--semantic-bridge", "neural",
            "--neural-concept-artifact", str(tmp_path / "no-such-artifact"),
        ],
        fake_model=True,
        fake_engine=True,
    )
    assert "cannot start semantic memory" in err
    assert [engine.closed for engine in _FakeEngine.instances] == [True]
    assert len(closed) == 1


def test_unreadable_lane_policy_file_is_a_usage_error(monkeypatch, capsys, tmp_path):
    # validate_arguments reads a --lane-policy file; a read failure is an
    # OSError the preflight must report like the shape errors.
    policy = tmp_path / "lane.json"
    policy.write_text('{"backend": "simd"}')
    policy.chmod(0)
    try:
        err = _run_main(
            monkeypatch,
            capsys,
            ["--model", "unused", "--port", "0", "--lane-policy", str(policy)],
            fake_model=True,
        )
    finally:
        policy.chmod(0o600)
    assert "Permission denied" in err


@pytest.mark.parametrize(
    "manifest, fragment",
    [
        ("[]", "JSON object"),
        ('{"schema": "x"}', "schema"),
    ],
)
def test_malformed_neural_manifest_closes_the_engine_and_the_server(
    monkeypatch, capsys, tmp_path, manifest, fragment
):
    from mlx2 import server

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(manifest)
    closed = []
    real_close = server.BoundedHTTPServer.server_close
    monkeypatch.setattr(
        server.BoundedHTTPServer,
        "server_close",
        lambda self: (closed.append(self), real_close(self))[1],
    )
    err = _run_main(
        monkeypatch,
        capsys,
        [
            "--model", "unused", "--port", "0", "--semantic-memory",
            "--api-state-dir", str(tmp_path / "state"),
            "--qualification-mode", "--semantic-bridge", "neural",
            "--neural-concept-artifact", str(artifact),
        ],
        fake_model=True,
        fake_engine=True,
    )
    assert "cannot start semantic memory" in err and fragment in err
    assert [engine.closed for engine in _FakeEngine.instances] == [True]
    assert len(closed) == 1


def test_post_bind_abort_closes_the_socket_even_when_the_engine_close_raises(
    monkeypatch, capsys, tmp_path
):
    from mlx2 import server

    class ExplodingEngine(_FakeEngine):
        def close(self):
            super().close()
            raise RuntimeError("close failed")

    closed = []
    real_close = server.BoundedHTTPServer.server_close
    monkeypatch.setattr(
        server.BoundedHTTPServer,
        "server_close",
        lambda self: (closed.append(self), real_close(self))[1],
    )
    monkeypatch.setattr(server, "ServingEngine", ExplodingEngine)
    ExplodingEngine.validate_arguments = staticmethod(server.ServingEngine.validate_arguments)
    _FakeEngine.instances.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx2.server", "--model", "unused", "--port", "0", "--semantic-memory",
            "--api-state-dir", str(tmp_path / "state"),
            "--qualification-mode", "--semantic-bridge", "neural",
            "--neural-concept-artifact", str(tmp_path / "no-such-artifact"),
        ],
    )
    from mlx2 import exit_trace
    from mlx2.adapters import registry

    monkeypatch.setattr(exit_trace.ExitTrace, "install", lambda self, fault_log=None: self)
    monkeypatch.setattr(
        registry, "inspect_model",
        lambda path: types.SimpleNamespace(artifact_fingerprint="sha256:fake"),
    )
    monkeypatch.setattr(
        server, "resolve_route_selection",
        lambda args, policy, resolution: types.SimpleNamespace(
            native_mtp=False, route="ordinary", source="engine_argument"
        ),
    )
    monkeypatch.setattr(
        server, "resolve_execution_policy_defaults", lambda policy, *a, **k: policy
    )
    with pytest.raises(RuntimeError, match="close failed"):
        server.main()
    assert len(closed) == 1  # the listening socket was still released
