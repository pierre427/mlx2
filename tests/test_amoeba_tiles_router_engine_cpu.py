"""Amoeba router open sets through a real ServingEngine (CPU): preemption replay and a generator
rebuild never leave a stale uid holding a request's open set.

A tiny hybrid Qwen3.8 text model with quantized MLPs carries router tiles (every box on the router
law). Request B (restricted open set) is preempted in prefill and replayed under a new uid while
request A (unrestricted) decodes; every forward's bound rows are checked: only B's current uid
may carry a restricted set. Then one injected device OOM rebuilds the generator (its uid counter
restarts) and an unrestricted request must equal its pre-rebuild reference.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from test_amoeba_tiles_spec_cpu import _write_source
from test_suspend_replay import POLICY, A_PROMPT, B_PROMPT, collect, host, make_adapter, tiny_model  # noqa: F401

from mlx2 import serving
from mlx2.runtime.os_memory import PressureLevel
from mlx2.runtime import generate

ROOT = Path(__file__).resolve().parents[1]
OOM = RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory "
                   "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).")


def _tiled_model(tmp_path):
    from mlx2.runtime.amoeba_tiles import install
    sys.path.insert(0, str(ROOT / "scripts"))
    export = importlib.import_module("amoeba27b_export")
    model = tiny_model()
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and ".mlp." in f".{p}.")
    mx.eval(model.parameters())
    layers = list(model.model.layers)
    _write_source(layers, tmp_path / "src")
    n, n_layers = 64, len(layers)
    rng = np.random.default_rng(5)
    owner = rng.integers(0, 4, size=(n_layers, n))              # 0 = core, 1..3 = boxes (a partition)
    core = (owner == 0).astype(np.float32)
    boxes = {f"b{j}": (owner == j).astype(np.float32) for j in (1, 2, 3)}
    export.write_tiles(tmp_path / "t", tmp_path / "src", "fp-tiny", "fake", core, [0], boxes, {},
                       n_layers, n, print, "t", router={b: [] for b in boxes})
    decl = {"mlp": "mlp", "projections": ("gate_proj", "up_proj", "down_proj"), "activation_fn": nn.silu}
    tiles = install(model, layers, tmp_path / "t" / "manifest.json", decl, source_fingerprint="fp-tiny")
    return model, tiles


def test_preemption_and_rebuild_leave_no_stale_open_set(host, tmp_path, monkeypatch):  # noqa: F811
    model, tiles = _tiled_model(tmp_path)
    router = tiles.router
    adapter_cls = make_adapter(model)
    adapter_cls.amoeba_tiles = tiles
    engine = serving.ServingEngine("tiny", adapter_factory=adapter_cls, qualification_mode=True, mtp=False,
                                   max_lanes=2, prefill_step=8, execution_policy=POLICY)
    violations, jobs = [], {}
    original_bind = router.bind_rows

    def bind_rows(uids):
        b = jobs.get("b")
        current = None if b is None else b.uid
        restricted = [uid for uid in uids if uid in router.uids]
        if any(uid != current for uid in restricted) or any(uid != current for uid in router.uids):
            violations.append((list(uids), dict(router.uids), current))
        return original_bind(uids)
    router.bind_rows = bind_rows
    try:
        assert engine.ready.wait(60), engine.error
        greedy = {"temperature": 0}
        reference = collect(engine.submit({"tokens": A_PROMPT, "max_tokens": 8, **greedy}))[0]
        a = engine.submit({"tokens": A_PROMPT, "max_tokens": 24, **greedy})
        while a.uid is None and not a.events.qsize():
            time.sleep(0.001)
        b = jobs["b"] = engine.submit({"tokens": B_PROMPT, "max_tokens": 12, "mlx2_amoeba_open": {"tiles": ["b1"]},
                                       **greedy})
        fired = threading.Event()

        def level():
            if not fired.is_set() and b.uid is not None and b.completion_tokens == 0:
                return PressureLevel.CRITICAL
            if engine.counts["memory_preemptions"]:
                fired.set()
            return PressureLevel.NORMAL
        engine.memory_pressure_level = level
        _, a_receipt = collect(a)
        _, b_receipt = collect(b)
        assert engine.counts["memory_preemptions"] == 1 and engine.counts["preempted_replays"] == 1
        assert b_receipt["preemption"]["replays"] == 1
        assert b_receipt["amoeba_open"]["open_tiles"] == ["b1"] and b_receipt["amoeba_open"]["bound_forwards"] > 0
        assert a_receipt["amoeba_open"]["all_router_tiles_open"] is True
        assert len(b.amoeba_open_record["uids"]) == 0                # released at finish
        assert not router.uids and not violations, violations
        engine.memory_pressure_level = lambda: PressureLevel.NORMAL

        # One injected OOM in prefill: the generator is rebuilt and its uids restart.
        state = {"armed": True}
        original_next = generate.BatchGenerator.next

        def next_(self):
            if state["armed"] and getattr(self, "_prompt_batch", None) is not None:
                state["armed"] = False
                raise OOM
            return original_next(self)
        monkeypatch.setattr(generate.BatchGenerator, "next", next_)
        resets = router.resets
        failed = engine.submit({"tokens": B_PROMPT, "max_tokens": 4, "mlx2_amoeba_open": {"tiles": ["b2"]},
                                **greedy})
        with pytest.raises(AssertionError):
            collect(failed)                                           # 503: device out of memory
        deadline = time.monotonic() + 60
        while not engine.counts.get("device_oom_recoveries") and time.monotonic() < deadline:
            time.sleep(0.01)                                          # recovery follows the error event
        assert engine.counts["device_oom_recoveries"] == 1 and router.resets == resets + 1
        assert not router.uids
        again = collect(engine.submit({"tokens": A_PROMPT, "max_tokens": 8, **greedy}))[0]
        assert again == reference and not violations
    finally:
        router.bind_rows = original_bind
        engine.close()
    assert not engine.error
