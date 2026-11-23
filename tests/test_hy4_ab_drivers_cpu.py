"""Host-only regression checks for the Hy4 research A/B drivers."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def driver(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,foreign,errors,expected", [
    ("knowledge_20x20_ab", True, None, 1),
    ("spec_copy_prose_ab", True, None, 1),
    ("knowledge_20x20_ab", False, 2, 1),
    ("knowledge_20x20_ab", False, 0, 0),
])
def test_benchmark_needs_own_listener_and_successful_requests(tmp_path, monkeypatch, name, foreign, errors, expected):
    module = driver(name)
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps([{"name": "probe", "tree": str(module.ROOT)}]))
    out = tmp_path / "out"
    proc = SimpleNamespace(pid=999, poll=lambda: None, wait=lambda timeout: 0)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: proc)
    monkeypatch.setattr(module.os, "killpg", lambda *a: None)
    monkeypatch.setattr(module, "port_listeners", lambda port: {333 if foreign else proc.pid})

    if name == "knowledge_20x20_ab":
        monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="test_revision"))
        monkeypatch.setattr(module, "status", lambda url: {"healthy": True})

        def run_benchmark(argv):
            assert not foreign
            dest = Path(argv[2])
            dest.mkdir()
            (dest / "report.json").write_text(json.dumps({"errors": errors}))
            return 0

        monkeypatch.setattr(module.subprocess, "call", run_benchmark)
        output_option, receipt = "--out", "summary.json"
    else:
        monkeypatch.setattr(module.Client, "call", lambda *a, **kw: {"healthy": True})
        monkeypatch.setattr(module, "revision", lambda *a: "test_revision")
        monkeypatch.setattr(module, "run_cells", lambda *a: pytest.fail("sent requests to a foreign listener"))
        output_option, receipt = "--output", "receipt.json"
    assert module.main(["--model", "test_fixture", "--arms", str(arms), output_option, str(out)]) == expected
    row = json.loads((out / receipt).read_text())["arms"][0]
    if foreign:
        assert "different process" in row["error"]
    else:
        assert row["request_errors"] == errors
