"""CPU checks for the bounded Gemma 4 HTTP video diagnostic runner."""

import importlib.util
from pathlib import Path


def _runner():
    path = (Path(__file__).resolve().parents[1] / "qualification" / "runs"
            / "gemma4-defaults-20260925" / "smoke_http.py")
    spec = importlib.util.spec_from_file_location("gemma4_smoke_http", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_video_diagnostic_binds_prompt_cap_and_apc_reuse(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(runner, "video_url", lambda: "data:video/mp4;base64,fixture")

    class Client:
        def __init__(self):
            self.bodies = []

        def post(self, path, body):
            assert path == "/v1/chat/completions"
            self.bodies.append(body)
            return 200, {
                "choices": [{"message": {"content": "Red, Green"},
                             "finish_reason": "stop"}],
                "mlx2": {"prompt_tokens": 602,
                         "cached_tokens": 601 if len(self.bodies) == 2 else 0},
            }, 0.1

    client = Client()
    result = runner.video_diagnostic(
        client, max_tokens=96, prompt="Name the colors in order.")

    assert len(client.bodies) == 2
    assert all(body["max_tokens"] == 96 and body["temperature"] == 0
               for body in client.bodies)
    assert all(body["messages"][0]["content"][0]["type"] == "input_video"
               for body in client.bodies)
    assert result["video_apc_reuse"]["full_prefix_reused"] is True
    assert result["video_apc_reuse"]["same_text"] is True
