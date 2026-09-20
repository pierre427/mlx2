"""Item 14: persistent reasoning signing key and the ``mlx2`` agent launcher."""

from __future__ import annotations

import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mlx2 import cli, clients
from mlx2.reasoning_signatures import (
    PERSISTENT_KEY_NAME,
    ReasoningSigner,
    ensure_persistent_key,
)
from mlx2.server import build_parser, load_api_key_file, resolve_reasoning_signing_key


# --- persistent reasoning signing key ------------------------------------


def test_persistent_key_created_owner_only_and_reused(tmp_path):
    path = ensure_persistent_key(tmp_path / "state" / PERSISTENT_KEY_NAME)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    first = path.read_bytes()
    assert len(first) == 64
    assert ensure_persistent_key(path) == path
    assert path.read_bytes() == first
    # No temporary files left behind.
    assert sorted(p.name for p in path.parent.iterdir()) == [PERSISTENT_KEY_NAME]


def test_signatures_verify_across_restart(tmp_path):
    path = ensure_persistent_key(tmp_path / PERSISTENT_KEY_NAME)
    before = ReasoningSigner.configured(key_file=path)
    signature = before.sign_anthropic(model="m", tenant="t", text="thought")
    token = before.sign_responses(model="m", tenant="t", text="thought")
    after = ReasoningSigner.configured(key_file=ensure_persistent_key(path))
    assert not after.ephemeral and after.key_id == before.key_id
    assert after.verify_anthropic(signature, model="m", tenant="t", text="thought")
    assert after.verify_responses(token, model="m", tenant="t") == "thought"
    # The per-process default is exactly what used to break this.
    assert not ReasoningSigner().verify_anthropic(
        signature, model="m", tenant="t", text="thought"
    )


def test_persistent_key_rejects_broad_mode_and_symlink(tmp_path):
    path = ensure_persistent_key(tmp_path / PERSISTENT_KEY_NAME)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0400 or 0600"):
        ensure_persistent_key(path)
    path.chmod(0o600)
    link = tmp_path / "link.key"
    link.symlink_to(path)
    with pytest.raises(OSError):
        ensure_persistent_key(link)


def test_concurrent_creation_keeps_first_key(tmp_path, monkeypatch):
    path = tmp_path / PERSISTENT_KEY_NAME
    real_link = os.link

    def racing_link(source, target):
        # Another server published its key between our probe and our link.
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, b"a" * 64)
        os.close(fd)
        return real_link(source, target)

    monkeypatch.setattr(os, "link", racing_link)
    ensure_persistent_key(path)
    assert path.read_bytes() == b"a" * 64
    assert sorted(p.name for p in tmp_path.iterdir()) == [PERSISTENT_KEY_NAME]


def _args(*extra):
    return build_parser().parse_args(["--model", "fixture", *extra])


def test_api_state_dir_defaults_to_persistent_key(tmp_path):
    args = _args("--api-state-dir", str(tmp_path))
    path = resolve_reasoning_signing_key(args)
    assert path == tmp_path.resolve() / PERSISTENT_KEY_NAME
    assert args.reasoning_signing_key_file == str(path)
    # A second start reuses the same file and key.
    again = _args("--api-state-dir", str(tmp_path))
    resolve_reasoning_signing_key(again)
    assert (
        ReasoningSigner.configured(key_file=again.reasoning_signing_key_file).key_id
        == ReasoningSigner.configured(key_file=path).key_id
    )


def test_persistent_key_opt_outs_and_default_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNING", "secret")
    for extra in (
        (),
        ("--api-state-dir", str(tmp_path), "--reasoning-signing-ephemeral"),
        ("--api-state-dir", str(tmp_path), "--reasoning-signing-key-env", "SIGNING"),
    ):
        args = _args(*extra)
        assert resolve_reasoning_signing_key(args) is None
        assert args.reasoning_signing_key_file is None
    assert not (tmp_path / PERSISTENT_KEY_NAME).exists()
    with pytest.raises(SystemExit):
        _args("--reasoning-signing-ephemeral", "--reasoning-signing-key-env", "X")


# --- API key -------------------------------------------------------------


def test_api_key_sources(tmp_path):
    assert clients.resolve_api_key({}) == "local"
    assert clients.resolve_api_key({"MLX2_API_KEY": "env-key"}) == "env-key"
    key = tmp_path / "api.key"
    key.write_text("file-key\n")
    key.chmod(0o600)
    assert load_api_key_file(key) == "file-key"
    assert clients.resolve_api_key({"MLX2_API_KEY": "env-key"}, key) == "file-key"
    key.chmod(0o644)
    with pytest.raises(clients.ClientError, match="--api-key-file permissions"):
        clients.resolve_api_key({}, key)


# --- per-client argv / environment ---------------------------------------


CALLER = {
    "PATH": "/usr/bin",
    "ANTHROPIC_API_KEY": "cloud",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "OPENCODE_CONFIG_CONTENT": json.dumps(
        {
            "theme": "dark",
            "agent": {"build": {"temperature": 0.2}},
            "provider": {
                "mlx2": {"models": {"m": {"variants": {"fast": {"x": 1}}}}}
            },
        }
    ),
}


def _command(name, **kwargs):
    caller = dict(CALLER)
    argv, env = clients.command(
        name, f"/bin/{name}", "http://127.0.0.1:8285/", "m", 1000, "k", caller, **kwargs
    )
    assert caller == CALLER  # caller environment is never mutated
    return argv, env


def test_claude_command():
    argv, env = _command("claude", client_args=("--resume",))
    assert argv == [
        "/bin/claude", "--disallowedTools", "WebSearch", "--model", "m",
        "--permission-mode", "default", "--resume",
    ]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8285"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "k"
    assert "ANTHROPIC_API_KEY" not in env
    for key in (
        "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        assert env[key] == "m"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1000"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "1000"
    assert all(env[flag] == "0" for flag in clients.CLAUDE_PROVIDER_FLAGS)
    assert env["PATH"] == "/usr/bin"


def test_opencode_command_merges_user_config():
    argv, env = _command("opencode", client_args=("run", "hi"), vision=True)
    assert argv == ["/bin/opencode", "run", "hi"]
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["theme"] == "dark"
    assert config["model"] == config["small_model"] == "mlx2/m"
    assert config["agent"]["build"] == {"temperature": 0.2, "model": "mlx2/m"}
    assert config["agent"]["compaction"]["model"] == "mlx2/m"
    provider = config["provider"]["mlx2"]
    assert provider["options"] == {"baseURL": "http://127.0.0.1:8285/v1", "apiKey": "k"}
    model = provider["models"]["m"]
    assert model["variants"]["fast"] == {"x": 1}
    assert model["variants"]["high"] == {"reasoningEffort": "high"}
    assert model["limit"] == {"context": 1000, "input": 750, "output": 250}
    assert model["modalities"]["input"] == ["text", "image"]
    _, text_only = clients.command(
        "opencode", "/bin/opencode", "http://h", "m", 1000, "k", {}
    )
    model = json.loads(text_only["OPENCODE_CONFIG_CONTENT"])["provider"]["mlx2"]
    assert model["models"]["m"]["modalities"]["input"] == ["text"]
    with pytest.raises(clients.ClientError, match="JSON object"):
        clients.command(
            "opencode", "/bin/opencode", "http://h", "m", 1, "k",
            {"OPENCODE_CONFIG_CONTENT": "[1]"},
        )


def test_codex_command_keeps_overrides_at_root():
    argv, env = _command(
        "codex", client_args=("exec", "-c", "model_reasoning_effort=high", "--", "-c")
    )
    assert env["MLX2_API_KEY"] == "k"
    assert argv[:2] == ["/bin/codex", "-c"]
    pairs = argv[1:argv.index("exec")]
    config = [pairs[i + 1] for i in range(0, len(pairs), 2)]
    assert all(pairs[i] == "-c" for i in range(0, len(pairs), 2))
    assert config == [
        'model="m"',
        'web_search="disabled"',
        'model_provider="mlx2"',
        'model_providers.mlx2={name="mlx2",base_url="http://127.0.0.1:8285/v1",'
        'env_key="MLX2_API_KEY",wire_api="responses"}',
        "model_context_window=1000",
        "model_auto_compact_token_limit=900",
        "model_reasoning_effort=high",
    ]
    assert argv[argv.index("exec"):] == ["exec", "--", "-c"]


def test_command_validates_server_report():
    with pytest.raises(clients.ClientError, match="model"):
        clients.command("claude", "/bin/claude", "http://h", "", 1, "k", {})
    with pytest.raises(clients.ClientError, match="context"):
        clients.command("claude", "/bin/claude", "http://h", "m", 0, "k", {})
    with pytest.raises(clients.ClientError, match="unknown"):
        clients.command("hermes", "/bin/hermes", "http://h", "m", 1, "k", {})


# --- discovery against a live HTTP endpoint ------------------------------


@pytest.fixture
def fake_server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            body = {
                "/v1/models": {
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen-fixture",
                            "object": "model",
                            "owned_by": "mlx2",
                            "capabilities": ["text", "vision"],
                        }
                    ],
                },
                "/v1/status": {"max_context": 65536, "model": "Qwen-fixture"},
            }.get(self.path)
            payload = json.dumps(body).encode()
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def test_discover_uses_models_and_status(fake_server):
    url, seen = fake_server
    assert clients.discover(url, "k") == ("Qwen-fixture", 65536, True)
    assert seen == [("/v1/models", "Bearer k"), ("/v1/status", "Bearer k")]


def test_discover_errors_are_actionable():
    with pytest.raises(clients.ClientError, match="no ready mlx2 server"):
        clients.discover("http://127.0.0.1:9", "k")
    with pytest.raises(clients.ClientError, match="identify"):
        clients.discover(
            "http://h", "k", fetch=lambda *_a, **_k: {"data": [{"owned_by": "x"}]}
        )


def test_cli_execs_client_with_private_env(fake_server, monkeypatch, capsys):
    url, _ = fake_server
    monkeypatch.setattr(clients, "find_executable", lambda name: f"/bin/{name}")
    calls = []
    caller = {"PATH": "/usr/bin", "MLX2_URL": url, "MLX2_API_KEY": "env-key"}
    snapshot = dict(caller)
    status = cli.main(
        ["claude", "--model", "other", "-p", "hello"],
        environment=caller,
        execvpe=lambda *call: calls.append(call),
    )
    assert status == 0 and caller == snapshot
    [(path, argv, env)] = calls
    assert path == "/bin/claude"
    assert argv[-4:] == ["--model", "other", "-p", "hello"]
    assert argv[argv.index("--model") + 1] == "Qwen-fixture"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "env-key"
    assert "65,536 context tokens" in capsys.readouterr().out


def test_cli_parse_splits_at_agent_name():
    args = cli.parse_args(
        ["--model", "x", "codex", "--", "-c", "a=b"], {"MLX2_URL": "http://example:1"}
    )
    assert (args.url, args.model, args.command) == ("http://example:1", "x", "codex")
    assert args.client_args == ["-c", "a=b"]
    assert cli.parse_args(["opencode"], {}).url == clients.DEFAULT_URL
    assert cli.parse_args(["--url", "http://u", "claude"], {}).url == "http://u"
    with pytest.raises(SystemExit):
        cli.parse_args([], {})
