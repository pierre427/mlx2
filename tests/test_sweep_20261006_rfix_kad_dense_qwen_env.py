"""rfix-kad 2026-10-07: the dense Qwen constructor (Qwen3.8 27B, inherited by
Qwen3.6 27B and dense Qwen3.5 9B/4B) applied its profile environment outside
``guarded_construction``, so a failed load after that point left the
profile's cleared/replaced variables behind for the next load."""

from __future__ import annotations

import importlib
import os

import pytest

DENSE = [
    ("qwen38_27b", "Qwen3827BAdapter"),
    ("qwen36_27b", "Qwen3627BAdapter"),
    ("qwen35_9b", "Qwen359BAdapter"),
    ("qwen35_4b", "Qwen354BAdapter"),
]
FAKE_ARTIFACT = {
    "identity": {"path": "/nonexistent-mlx2-artifact", "fingerprint": "x", "files": []},
    "config": {}, "has_mtp": False, "weight_map": {}, "mtp_path": None,
}


@pytest.mark.parametrize("module_name,name", DENSE)
def test_failed_dense_load_restores_environment(module_name, name, monkeypatch):
    from mlx2.runtime.models import import_env

    cls = getattr(importlib.import_module(f"mlx2.adapters.{module_name}"), name)
    monkeypatch.setattr(cls, "artifact_inspector", staticmethod(lambda *a, **k: dict(FAKE_ARTIFACT)))
    monkeypatch.setenv("MLX_LM_OTHER_ADAPTER_PIN", "before")
    before = dict(os.environ)
    applied = []

    def refuse(owner):
        applied.append(dict(os.environ) != before)
        raise RuntimeError(f"{owner}: import-order conflict (test)")

    monkeypatch.setattr(import_env, "assert_profile_applied", refuse)
    with pytest.raises(RuntimeError, match="import-order conflict"):
        cls("/nonexistent-mlx2-artifact")
    assert applied == [True]  # the profile had edited the environment
    assert dict(os.environ) == before
