from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _benchmark_module():
    spec = spec_from_file_location(
        "bench_incremental_tokenizer_cache",
        ROOT / "scripts" / "bench_incremental_tokenizer_cache.py",
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nemotron_benchmark_renderer_defaults_thinking_on():
    adapter = object.__new__(_benchmark_module()._Adapter)
    adapter.renderer = "nemotron-thinking-on"

    assert adapter._request({"messages": []})["enable_thinking"] is True
    assert (
        adapter._request({"messages": [], "enable_thinking": False})[
            "enable_thinking"
        ]
        is False
    )


def test_qwen_benchmark_renderer_preserves_request():
    adapter = object.__new__(_benchmark_module()._Adapter)
    adapter.renderer = "qwen-thinking-off"
    request = {"messages": []}

    assert adapter._request(request) is request
