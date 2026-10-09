"""sp_qmm is an explicitly selected projection owner: lane auto steps aside.

On an M5 (lane backend "mpp") the default --lane-matmul auto resolves to
crossover on dense models and swaps every eligible QuantizedLinear to
LaneQuantizedLinear before sp_qmm.apply runs.  sp_qmm then finds no exact
nn.QuantizedLinear and startup failed with "no eligible 4/5-bit gs64
projections" although the artifact had them.
"""

import json
import time

import mlx.core as mx
import pytest
import route_harness
from mlx import nn


def _emulate_m5(monkeypatch):
    # The only patch: the lane backend this host reports.  No kernel runs.
    import mlx2.runtime.lane.installer as inst
    import mlx2.runtime.lane.matmul as lm
    from mlx2.runtime import lane

    monkeypatch.setattr(lm, "backend", lambda: "mpp")
    monkeypatch.setattr(inst, "backend", lambda: "mpp")
    monkeypatch.setattr(inst, "available", lambda: True)
    monkeypatch.setattr(lane, "available", lambda: True)


def _q4_model():
    model, vocab = route_harness.tiny_qwen38_mtp()
    model.set_dtype(mx.bfloat16)
    nn.quantize(
        model, group_size=64, bits=4,
        class_predicate=lambda _p, m: isinstance(m, nn.Linear)
        and m.weight.shape[-1] % 64 == 0 and m.weight.shape[0] % 8 == 0,
    )
    mx.eval(model.parameters())
    return model, vocab


def _sp_qmm_engine(tmp_path, monkeypatch, lane_matmul):
    from mlx2.serving import ServingEngine

    route_harness.patch_host(monkeypatch)
    _emulate_m5(monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    model, vocab = _q4_model()
    engine = ServingEngine(
        str(tmp_path),
        adapter_factory=route_harness.make_adapter(model, vocab),
        qualification_mode=True, mtp=False, max_lanes=1, max_inflight=4,
        prefill_step=16, lane_matmul=lane_matmul,
        execution_policy={"sp_qmm": True},
    )
    for _ in range(1200):
        if engine.error or engine.ready.is_set():
            break
        time.sleep(0.1)
    return engine


def test_lane_auto_steps_aside_for_an_explicit_sp_qmm(tmp_path, monkeypatch):
    engine = _sp_qmm_engine(tmp_path, monkeypatch, "auto")
    try:
        assert not engine.error, engine.error
        status = engine.status()
        assert status["settings"]["lane_matmul_stepped_aside"] == "auto->off (sp_qmm)"
        assert status["sp_qmm"]["modules"] > 0
        assert not (engine.lane_matmul_receipt or {}).get("covered")
    finally:
        engine.close()


@pytest.mark.parametrize("mode", ["crossover", "exact"])
def test_explicit_lane_mode_refuses_sp_qmm_by_name(tmp_path, monkeypatch, mode):
    engine = _sp_qmm_engine(tmp_path, monkeypatch, mode)
    try:
        error = str(engine.error)
        assert "sp_qmm" in error and "lane_matmul" in error, error
        assert "no eligible" not in error, error
    finally:
        engine.close()
