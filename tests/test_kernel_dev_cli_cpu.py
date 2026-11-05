"""CLI request construction; no model, external service or GPU is used."""

import json

import pytest

from mlx2 import kernel_dev


@pytest.fixture
def client(tmp_path, monkeypatch):
    token = tmp_path / "admin.key"
    token.write_text("secret\n")
    token.chmod(0o600)
    seen = []

    def http(*args, **kwargs):
        seen.append((args, kwargs))
        return {"status": "completed", "apcv2_purged": True}

    monkeypatch.setattr(kernel_dev, "_http", http)
    return ["--admin-token-file", str(token)], seen


def test_cli_status_and_reload_request(client, capsys):
    flags, seen = client
    assert kernel_dev._main(flags + ["status"]) == 0
    assert seen[-1][0] == ("http://127.0.0.1:8285", "GET", "/v1/admin/kernels")
    assert seen[-1][1]["token"] == "secret"
    assert kernel_dev._main(flags + [
        "reload", "--modules", "pkg.kernel", "pkg.consumer",
        "--settings", '[{"target":"pkg.kernel:MODE","value":"candidate"}]',
        "--calls", '[{"target":"pkg.kernel:set_mode","args":["candidate"]}]',
        "--drain-timeout-seconds", "2",
    ]) == 0
    args, kwargs = seen[-1]
    assert args[-1] == "/v1/admin/kernels/reload"
    assert kwargs["body"]["modules"] == ["pkg.kernel", "pkg.consumer"]
    assert kwargs["body"]["drain_timeout_seconds"] == 2
    assert kwargs["timeout"] == 122
    assert "secret" not in capsys.readouterr().out


def test_cli_file_supports_repeated_variant_selection(client, tmp_path):
    flags, seen = client
    path = tmp_path / "variant.json"
    path.write_text(json.dumps({"settings": [{"target": "pkg.kernel:MODE", "value": "candidate"}]}))
    assert kernel_dev._main(flags + ["reload", "--file", str(path)]) == 0
    assert seen[-1][1]["body"]["settings"][0]["value"] == "candidate"
    path.write_text(json.dumps({"settings": [{"target": "pkg.kernel:MODE", "value": "reference"}]}))
    assert kernel_dev._main(flags + ["reload", "--file", str(path)]) == 0
    assert seen[-1][1]["body"]["settings"][0]["value"] == "reference"


@pytest.mark.parametrize("args", [
    ["reload"], ["reload", "--settings", "{}"],
    ["reload", "--settings", '[{"target":"pkg.kernel:MODE","value":NaN}]'],
    ["reload", "--modules", "pkg.kernel", "--drain-timeout-seconds", "nan"],
    ["--url", "https://example.com", "status"],
    ["--url", "http://user:secret@127.0.0.1", "status"],
    ["--timeout", "0", "status"],
])
def test_cli_invalid_request_never_contacts_server(client, args):
    flags, seen = client
    with pytest.raises(SystemExit) as error:
        kernel_dev._main(flags + args)
    assert error.value.code == 2
    assert seen == []


def test_cli_file_does_not_silently_override_operation_flags(client, tmp_path):
    flags, seen = client
    path = tmp_path / "variant.json"
    path.write_text('{"modules":["pkg.kernel"]}')
    with pytest.raises(SystemExit):
        kernel_dev._main(flags + ["reload", "--file", str(path), "--modules", "pkg.other"])
    assert seen == []
