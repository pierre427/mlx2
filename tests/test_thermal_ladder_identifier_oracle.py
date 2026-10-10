"""Exact semantic identifier checks for the exploratory thermal ladder."""

import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmark_results/2026-09-29/scripts/series/thermal_ladder.py"


def _ladder_module():
    spec = importlib.util.spec_from_file_location("current_thermal_ladder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = os.environ.get("MLX2_CAMPAIGN_ROOT")
    os.environ["MLX2_CAMPAIGN_ROOT"] = str(ROOT)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop("MLX2_CAMPAIGN_ROOT", None)
        else:
            os.environ["MLX2_CAMPAIGN_ROOT"] = previous
    return module


def _row(content, *, cached_tokens=0, prompt_tokens=32):
    return {
        "done": True,
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 4,
        "cached_tokens": cached_tokens,
        "server_ttft_seconds": 1.0,
        "client_ttft_seconds": 1.0,
        "prefill_tokens_per_second": 1.0,
        "decode_tokens_per_second": 1.0,
    }


def test_needle_oracle_accepts_only_typographic_dash_substitution():
    ladder = _ladder_module()
    cold = {"rows": [_row("code NEEDLE‑ABC123‑0")]}
    warm = {"rows": [_row("code NEEDLE-ABC123-0")]}
    cold["aggregate_completion_tokens_per_second"] = 1.0
    summary = ladder.summarize_run(cold, warm, ["NEEDLE-ABC123-0"])
    assert summary["needle_correct"] == 2

    cold["rows"][0]["content"] = "code needle‑ABC123‑0"
    summary = ladder.summarize_run(cold, warm, ["NEEDLE-ABC123-0"])
    assert summary["needle_correct"] == 1


def test_warm_hit_excludes_chat_preamble_only_reuse():
    ladder = _ladder_module()
    cold = {"rows": [_row("NEEDLE-X-0", prompt_tokens=32_000)]}
    cold["aggregate_completion_tokens_per_second"] = 1.0
    preamble = {"rows": [_row(
        "NEEDLE-X-0", cached_tokens=54, prompt_tokens=32_000
    )]}
    assert ladder.summarize_run(cold, preamble, ["NEEDLE-X-0"])["warm_apc_hits"] == 0
    full = {"rows": [_row(
        "NEEDLE-X-0", cached_tokens=31_999, prompt_tokens=32_000
    )]}
    summary = ladder.summarize_run(cold, full, ["NEEDLE-X-0"])
    assert summary["warm_apc_hits"] == 1
    assert summary["warm_equals_cold"] == 1


def test_stream_logit_diagnostic_is_explicit_and_preserves_rows(monkeypatch):
    ladder = _ladder_module()
    client = ladder.Stream("http://127.0.0.1:1", 1, "fixture")
    client.top_logprobs = 2
    client.verify_bitexact = True
    client.min_tokens_supported = False
    captured = {}

    class Response:
        def __enter__(self):
            return iter([
                b'data: {"choices":[{"delta":{"content":"A"},"logprobs":{"content":[{"id":1,"logprob":-0.1,"top_logprobs":[{"id":2,"logprob":-0.2}]}]}}],"usage":{"prompt_tokens":3,"completion_tokens":1},"mlx2":{"cached_tokens":2}}\n',
                b'data: [DONE]\n',
            ])

        def __exit__(self, *_args):
            return False

    def urlopen(request, timeout):
        captured.update(json.loads(request.data))
        assert timeout == 1
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    row = client.request("prompt", 1)
    assert captured["logprobs"] is True
    assert captured["top_logprobs"] == 2
    assert captured["verify_bitexact"] is True
    assert row["content"] == "A"
    assert row["logprobs"][0]["top_logprobs"][0]["id"] == 2
