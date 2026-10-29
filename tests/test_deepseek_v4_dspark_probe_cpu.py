"""Probe guard checks only; never opens a GPU or model artifact."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def probe(monkeypatch):
    directory = ROOT / "scripts/research"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "ds4_support_probe_test", directory / "probe_deepseek_v4_dspark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path, **changes):
    values = {
        "run_gpu": True,
        "ownership_window": tmp_path / "window",
        "output": tmp_path / "fresh",
        "max_tokens": 8,
        "prefill_step": 128,
        "memory_limit_gib": 108,
        "execution_policy": {},
    }
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "change",
    [
        {"run_gpu": False},
        {"ownership_window": None},
        {"max_tokens": 65},
        {"prefill_step": 129},
        {"memory_limit_gib": 120},
        {"execution_policy": {"x": True}},
    ],
)
def test_refuses_unowned_or_unbounded_execution(probe, tmp_path, change):
    with pytest.raises(ValueError):
        probe.validate_args(arguments(tmp_path, **change))


def test_refuses_existing_evidence(probe, tmp_path):
    args = arguments(tmp_path)
    args.output.mkdir()
    with pytest.raises(ValueError, match="existing"):
        probe.validate_args(args)


def test_same_values_compares_full_nested_planes(probe):
    original = (np.array([1.0, 2.0]), None, (np.array([3]),))
    assert probe.same_values(original, (np.array([1.0, 2.0]), None, (np.array([3]),)))
    assert not probe.same_values(
        original, (np.array([1.0, 2.0]), None, (np.array([4]),))
    )
    assert not probe.same_values(original, (np.array([1.0, 2.0]),))


def test_request_json_is_bounded_text_and_hash_bound(probe, tmp_path):
    import json

    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Count to twenty."}],
                "enable_thinking": False,
            }
        )
    )
    request, binding = probe.load_request(path)
    assert request["messages"][0]["content"] == "Count to twenty."
    assert len(binding["sha256"]) == 64
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": [{"image": "x"}]}]})
    )
    with pytest.raises(ValueError, match="plain-text"):
        probe.load_request(path)


def test_adapter_external_flag_selects_real_bounded_policy(probe, tmp_path):
    args = arguments(
        tmp_path,
        adapter_external=True,
        verification_mode="causal_slab",
        sidecar_path=tmp_path / "support.gguf",
        execution_policy={"deepseek_kv": "fp8_mixed"},
    )
    selected = probe.selected_adapter_policy(args)
    assert selected == {
        "deepseek_kv": "fp8_mixed",
        "draft_model": str(args.sidecar_path),
        "num_draft": 5,
        "deepseek_verifier": "causal_slab",
    }
    args.adapter_external = False
    assert probe.selected_adapter_policy(args) == {"deepseek_kv": "fp8_mixed"}
    args.execution_policy["draft_model"] = "foreign"
    with pytest.raises(ValueError, match="adapter-external"):
        probe.selected_adapter_policy(args)
