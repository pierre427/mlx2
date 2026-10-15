"""CPU-only serving integration for adapter-owned Core ML projections."""

import hashlib
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mlx2.api_resources import CapabilityUnavailable
from mlx2.contracts import Capability, ModelDescriptor
from mlx2.runtime import ane_projection as ap
from mlx2.serving import ServingEngine


@pytest.fixture
def executor(tmp_path, monkeypatch):
    model = tmp_path / "projection.mlmodelc"
    model.mkdir()
    (model / "model.mil").write_bytes(b"serving integration fixture")
    digest = ap.compiled_model_digest(model)
    state = NS(predictions=0, fail=False)
    policy = object()

    class Model:
        def __init__(self, _path, *, compute_units):
            assert compute_units is policy

        def predict(self, values):
            state.predictions += 1
            if state.fail:
                raise RuntimeError("native failure")
            return {"projection": values["x"][:, :2] * np.float16(2)}

    runtime = NS(
        models=NS(CompiledMLModel=Model),
        ComputeUnit=NS(CPU_AND_NE=policy),
        __version__="fake-serving-contract",
    )
    monkeypatch.setattr(ap, "_coreml_runtime", lambda: runtime)
    spec = ap.FixedProjection(
        "dense_fixture:v1", hashlib.sha256(b"artifact").hexdigest(), 4, 2, 1
    )
    instance = ap.CoreMLProjectionExecutor(spec, model, declared_sha256=digest)
    try:
        yield instance, state
    finally:
        instance.close()


def binding(executor, reference=None):
    instance, _state = executor
    reference = reference or (lambda values: values[:, :2] * np.float16(2))
    return ap.AdapterProjection(
        instance,
        reference,
        absolute_tolerance=0,
        relative_tolerance=0,
    )


def adapter(projection, *, variant="v1", fingerprint=None):
    descriptor = ModelDescriptor(
        model_type="dense_fixture",
        family="dense_fixture",
        variant=variant,
        state_planes=frozenset(),
        capabilities=frozenset({Capability.TEXT}),
    )
    return NS(
        descriptor=descriptor,
        identity={
            "fingerprint": fingerprint
            or hashlib.sha256(b"artifact").hexdigest()
        },
        coreml_projection=projection,
    )


def engine_for(value):
    engine = ServingEngine.__new__(ServingEngine)
    engine.adapter = value
    calls = []

    def exclusive(operation, callback, **kwargs):
        calls.append((operation, kwargs))
        return callback(engine.adapter)

    engine._exclusive_adapter_operation = exclusive
    return engine, calls


def test_engine_executes_adapter_owned_projection_and_reports_reference_gate(executor):
    projection = binding(executor)
    engine, calls = engine_for(adapter(projection))
    values = np.array([[1, 2, 3, 4]], dtype=np.float16)

    output, receipt = engine.project_fixed(values)

    assert output.tolist() == [[2, 4]]
    assert calls == [("coreml_projection", {"admission_class": "projection", "admitted": False})]
    assert receipt["route"] == "coreml_projection"
    assert receipt["adapter_owned"] is receipt["ordinary_reference"] is True
    assert receipt["prediction_completed"] is receipt["parity"] is True
    assert receipt["selected"] is True
    assert receipt["qualified"] is receipt["observed_used"] is False
    assert receipt["execution_device_observed"] is False
    status = projection.status()
    assert status["prediction_count"] == 1
    # Status describes availability, not a new request's selection.
    assert status["selected"] is False
    assert engine._coreml_projection_state(engine.adapter) == {
        "coreml_projection": status
    }


def test_reference_mismatch_refuses_publication_without_second_execution(executor):
    projection = binding(executor, lambda _values: np.zeros((1, 2), dtype=np.float16))
    engine, _calls = engine_for(adapter(projection))

    with pytest.raises(RuntimeError, match="differs from the ordinary reference"):
        engine.project_fixed(np.ones((1, 4), dtype=np.float16))

    assert executor[1].predictions == 1
    assert projection.status()["prediction_count"] == 0


def test_native_failure_does_not_fall_back_to_ordinary_reference(executor):
    reference_calls = []
    projection = binding(executor, lambda values: reference_calls.append(values) or values[:, :2])
    engine, _calls = engine_for(adapter(projection))
    executor[1].fail = True

    with pytest.raises(RuntimeError, match="native failure"):
        engine.project_fixed(np.ones((1, 4), dtype=np.float16))

    assert reference_calls == []
    assert projection.status()["prediction_count"] == 0


def test_missing_or_foreign_adapter_binding_fails_closed(executor):
    values = np.ones((1, 4), dtype=np.float16)
    engine, _calls = engine_for(adapter(None))
    with pytest.raises(CapabilityUnavailable, match="no fixed Core ML projection"):
        engine.project_fixed(values)

    engine.adapter = adapter(binding(executor), variant="foreign")
    with pytest.raises(RuntimeError, match="identity differs"):
        engine.project_fixed(values)
    assert executor[1].predictions == 0

    engine.adapter = adapter(
        binding(executor), fingerprint=hashlib.sha256(b"foreign").hexdigest()
    )
    with pytest.raises(RuntimeError, match="artifact differs"):
        engine.project_fixed(values)
    assert executor[1].predictions == 0


def test_reference_receives_independent_read_only_input(executor):
    seen = []

    def reference(values):
        seen.append(values)
        assert not values.flags.writeable
        return values[:, :2] * np.float16(2)

    values = np.ones((1, 4), dtype=np.float16)
    output, _receipt = binding(executor, reference).predict(values)
    values[:] = 9
    assert output.tolist() == [[2, 2]]
    assert seen[0] is not values
