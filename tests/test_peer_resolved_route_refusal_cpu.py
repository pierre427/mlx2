"""Resolved route incompatibilities fail before model allocation."""

import json
import sys
from types import SimpleNamespace

import pytest

from mlx2 import server
from mlx2.adapters import registry
from mlx2.contracts import Capability


@pytest.mark.parametrize("kind", ["draft_model", "draft_models", "cli_draft_model", "default_pld"])
def test_resolved_speculation_refuses_approximate_kv_before_allocation(
    monkeypatch, tmp_path, capsys, kind,
):
    resolution = SimpleNamespace(
        default_route="ordinary",
        adapter_type=SimpleNamespace(supports_draft_portfolio=True),
        descriptor=SimpleNamespace(capabilities={Capability.PROMPT_LOOKUP}),
        default_execution_policy=lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(registry, "inspect_model", lambda _path: resolution)

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("incompatible route reached model allocation")

    validate = server.ServingEngine.validate_arguments

    class ForbiddenEngine:
        validate_arguments = staticmethod(validate)

        def __init__(self, *args, **kwargs):
            forbidden_allocation()

    monkeypatch.setattr(server, "ServingEngine", ForbiddenEngine)
    monkeypatch.setattr(server, "BoundedHTTPServer", forbidden_allocation)
    argv = [
        "mlx2-serve", "--model", "not-loaded", "--qualification-mode",
        "--approximate-kv", json.dumps({"enabled": True, "operation": "kv_k8v4"}),
    ]
    if kind == "cli_draft_model":
        argv.extend(["--external-draft", "--draft-model", "not-loaded-draft"])
    elif kind != "default_pld":
        policy = tmp_path / "policy.json"
        value = "not-loaded-draft" if kind == "draft_model" else ["not-loaded-draft"]
        policy.write_text(json.dumps({kind: value}))
        argv.extend(["--external-draft", "--execution-policy", str(policy)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        server.main()
    assert error.value.code == 2
    assert "--approximate-kv requires the --ordinary route" in capsys.readouterr().err
