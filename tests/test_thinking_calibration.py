"""Steering is bound to the exact artifact it was calibrated on, or it is off."""
import json
from types import SimpleNamespace as NS

import mlx.core as mx
import numpy as np
import pytest

from mlx2 import thinking_calibration as tc


def _artifact(root, *, weights=b"w" * 4096, template="T"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"hidden_size": 8}')
    (root / "chat_template.jinja").write_text(template)
    (root / "model-00001.safetensors").write_bytes(weights)
    return root


def _store(path, identity, *, hidden=8, layer=2, unit=None, schema=tc.SCHEMA):
    unit = np.eye(hidden, dtype=np.float32)[0] if unit is None else unit
    np.savez(path, **{f"v_{layer}": unit, f"rms_{layer}": np.array(3.0)})
    path.with_suffix(".json").write_text(json.dumps({"schema": schema, "artifact_identity": identity, "layer": layer, "origin": "shipped"}))
    return path


def test_identity_is_content_bound_and_host_independent(tmp_path):
    a = tc.artifact_identity(_artifact(tmp_path / "a"))
    assert a == tc.artifact_identity(_artifact(tmp_path / "moved" / "copy"))  # path and mtime do not matter
    assert a != tc.artifact_identity(_artifact(tmp_path / "quant", weights=b"x" * 4096))
    assert a != tc.artifact_identity(_artifact(tmp_path / "tmpl", template="other template"))


def test_a_direction_measured_on_another_artifact_is_never_loaded(tmp_path):
    mine, other = "a" * 64, "b" * 64
    foreign = _store(tmp_path / "foreign.npz", other)
    assert tc.load_bound_direction(mine, [foreign], hidden_size=8, num_layers=4) is None
    bound = _store(tmp_path / "bound.npz", mine)
    direction = tc.load_bound_direction(mine, [foreign, bound], hidden_size=8, num_layers=4)
    assert direction["layer"] == 2 and direction["source"] == "bound.npz:L2"
    assert abs(float(direction["vector"][0]) - 3.0) < 1e-6  # rms * unit
    # Right identity is still not enough: shape, layer range, unit norm and schema must hold.
    assert tc.load_bound_direction(mine, [bound], hidden_size=16, num_layers=4) is None
    assert tc.load_bound_direction(mine, [bound], hidden_size=8, num_layers=2) is None
    assert tc.load_bound_direction(mine, [_store(tmp_path / "scaled.npz", mine, unit=np.full(8, 2.0, dtype=np.float32))], hidden_size=8, num_layers=4) is None
    assert tc.load_bound_direction(mine, [_store(tmp_path / "old.npz", mine, schema="v0")], hidden_size=8, num_layers=4) is None
    assert tc.load_bound_direction(mine, [tmp_path / "missing.npz"], hidden_size=8, num_layers=4) is None


def test_every_gate_fails_closed():
    good_report = {"traces_used": 12, "layers": {"5": {"consistency": 0.45}}}
    good = {"off": {"correct": 7, "closed": 7, "think_tokens": 900}, "calibrated": {"correct": 8, "closed": 8, "think_tokens": 400},
            "random": {"correct": 6, "closed": 6, "think_tokens": 1200}}
    assert tc.gates(good_report, 5, good) == []
    assert "usable closed traces" in tc.gates({**good_report, "traces_used": 3}, 5, good)[0]
    assert "consistency" in tc.gates({"traces_used": 12, "layers": {"5": {"consistency": 0.1}}}, 5, good)[0]
    assert "consistency" in tc.gates(good_report, None, None)[0]
    worse = {**good, "calibrated": {"correct": 6, "closed": 8, "think_tokens": 400}}
    assert any("lost accuracy" in f for f in tc.gates(good_report, 5, worse))
    longer = {**good, "calibrated": {"correct": 8, "closed": 8, "think_tokens": 950}}
    assert any("did not shorten" in f for f in tc.gates(good_report, 5, longer))
    undirected = {**good, "random": {"correct": 8, "closed": 8, "think_tokens": 390}}
    assert any("not directional" in f for f in tc.gates(good_report, 5, undirected))


def _north_like(max_tokens=32):
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    mx.random.seed(3)
    model = Model(ModelArgs(
        hidden_size=16, head_dim=4, num_hidden_layers=8, intermediate_size=8, prefix_dense_intermediate_size=24,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32, num_experts=4, num_experts_per_tok=2,
        first_k_dense_replace=1, sliding_window=6, layer_types=["full_attention"] + ["sliding_attention"] * 7))
    model.eval()
    return NS(model=model, tokenizer=NS(eos_token_id=31, decode=lambda ids: " ".join(map(str, ids))),
              prompt_tokens=lambda request: [1, 2, 3, 4], thinking_close_token_ids=lambda: (30,))


def test_automatic_calibration_that_fails_its_gates_writes_nothing(tmp_path, monkeypatch):
    """A random model never closes its reasoning: no usable traces, so no file, so no steering."""
    adapter = _north_like()
    monkeypatch.setattr(tc, "CALIBRATION_PROMPTS", tc.CALIBRATION_PROMPTS[:3])
    npz, report = tc.auto_calibrate(adapter, "c" * 64, tmp_path / "store", max_think=12)
    assert npz is None and report["status"] == "failed"
    assert any("usable closed traces" in failure for failure in report["failures"])
    # Only the verdict is remembered; there is no direction file to pick up later.
    assert sorted(item.name for item in (tmp_path / "store").iterdir()) == ["c" * 64 + ".rejected.json"]
    assert tc.load_bound_direction("c" * 64, [tmp_path / "store" / ("c" * 64 + ".npz")], hidden_size=16, num_layers=8) is None
    assert adapter.model.model.residual_taps.steer is None and adapter.model.model.residual_taps.capture is None
    assert tc.supports_calibration(adapter)
    assert not tc.supports_calibration(NS(model=None))


def test_resolve_prefers_bound_assets_then_the_store_then_calibrates(tmp_path, monkeypatch):
    adapter = _north_like()
    model_path = _artifact(tmp_path / "model")
    identity = tc.artifact_identity(model_path)
    calls = []

    def failing(adapter, identity, out_dir, **_kw):
        calls.append(identity)
        return None, {"status": "failed", "failures": ["steering did not shorten held-out reasoning"]}

    monkeypatch.setattr(tc, "auto_calibrate", failing)
    foreign = _store(tmp_path / "foreign.npz", "f" * 64, hidden=16)
    direction, status = tc.resolve_direction(adapter, model_path, cache_dir=tmp_path / "cache", shipped=[foreign], allow_auto=False)
    assert direction is None and status["state"] == "uncalibrated" and not calls  # never calibrates when told not to
    direction, status = tc.resolve_direction(adapter, model_path, cache_dir=tmp_path / "cache", shipped=[foreign])
    assert direction is None and calls == [identity]
    assert status["auto_calibration"]["failures"] == ["steering did not shorten held-out reasoning"]
    store = tc.calibration_cache_dir(tmp_path / "cache")
    store.mkdir(parents=True)
    # A remembered rejection stops the server recalibrating on every start.
    (store / f"{identity}.rejected.json").write_text(json.dumps({"status": "failed", "failures": ["not directional"]}))
    direction, status = tc.resolve_direction(adapter, model_path, cache_dir=tmp_path / "cache", shipped=[foreign])
    assert direction is None and calls == [identity] and status["auto_calibration"]["failures"] == ["not directional"]
    assert "cached_verdict" in status["auto_calibration"]
    (store / f"{identity}.rejected.json").unlink()
    _store(store / f"{identity}.npz", identity, hidden=16, layer=5)
    direction, status = tc.resolve_direction(adapter, model_path, cache_dir=tmp_path / "cache", shipped=[foreign])
    assert direction["layer"] == 5 and status["state"] == "calibrated" and calls == [identity]  # stored: no recalibration


def test_server_refuses_explicit_steering_without_a_bound_direction(tmp_path, monkeypatch):
    from mlx2 import serving

    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.model_path, engine.cache_dir = str(_artifact(tmp_path / "model")), str(tmp_path / "cache")
    engine.thinking_auto_calibration = True
    monkeypatch.setattr(tc, "auto_calibrate", lambda *a, **k: (None, {"status": "failed", "failures": ["consistency +0.05 below 0.3"]}))
    adapter = _north_like()
    # Adapter default: steering quietly stays off, the guard budget is untouched.
    engine._thinking_overrides = {"thinking_steer_alpha": None}
    engine.thinking_steer_alpha = 0.2
    engine._resolve_commit_direction(adapter)
    assert engine.thinking_steer_alpha == 0.0 and engine._commit_direction is None
    assert engine.thinking_steer_status["state"] == "uncalibrated"
    # Operator asked for it: refuse to start rather than serve unsteered.
    engine._thinking_overrides = {"thinking_steer_alpha": 0.2}
    engine.thinking_steer_alpha = 0.2
    with pytest.raises(ValueError, match="no commit direction is calibrated for this artifact: consistency"):
        engine._resolve_commit_direction(adapter)
    with pytest.raises(ValueError, match="residual taps"):
        engine._resolve_commit_direction(NS(model=None))
