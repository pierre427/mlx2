"""The opt-in host-available floor applies to parallel-sample admission too."""

from __future__ import annotations

from types import SimpleNamespace

from mlx2 import memory
from mlx2.serving import ServingEngine

GIB = 1 << 30


def _engine(policy):
    engine = object.__new__(ServingEngine)
    engine.host_memory_signals_policy = policy
    return engine


def test_parallel_samples_and_lanes_share_the_floor(monkeypatch):
    seen = []

    def fake(host_signals=False, minimum_host_available_bytes=0):
        seen.append((host_signals, minimum_host_available_bytes))
        return 0

    monkeypatch.setattr(memory, "execution_headroom", fake)
    fn = _engine({"enabled": True, "minimum_host_available_gib": 4.0}).admission_headroom_fn()
    fn()
    assert seen == [(True, 4 * GIB)]


def test_disabled_policy_keeps_plain_headroom(monkeypatch):
    monkeypatch.setattr(memory, "execution_headroom", lambda **k: SimpleNamespace(**k))
    fn = _engine({"enabled": False}).admission_headroom_fn()
    assert vars(fn()) == {}
