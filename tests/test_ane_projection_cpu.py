import hashlib

import pytest

from mlx2.runtime.ane_projection import (
    FixedProjection,
    bind_compiled_model,
    record_compute_plan,
    verify_projection_output,
)

DIGEST = hashlib.sha256(b"artifact").hexdigest()


def test_fixed_projection_accounts_all_declared_buffers():
    spec = FixedProjection("fixture.adapter", DIGEST, 8, 4, 2)
    receipt = spec.receipt()
    assert receipt["allowed_compute_units"] == "cpu_and_ne"
    assert receipt["bytes"] == {"input": 32, "output": 16, "weights": 64, "minimum_live": 112}
    assert receipt["observed_used"] is False


def test_compiled_model_binding_is_revision_bound(tmp_path):
    model = tmp_path / "projection.mlmodelc"
    model.mkdir()
    (model / "model.mil").write_bytes(b"fixed")
    digest = hashlib.sha256()
    digest.update(b"model.mil")
    digest.update(bytes.fromhex(hashlib.sha256(b"fixed").hexdigest()))
    spec = FixedProjection("fixture.adapter", DIGEST, 8, 4, 2)
    receipt = bind_compiled_model(spec, model, declared_sha256=digest.hexdigest())
    assert receipt["compiled_model_path"] == str(model)
    (model / "model.mil").write_bytes(b"changed")
    with pytest.raises(ValueError, match="reviewed identity"):
        bind_compiled_model(spec, model, declared_sha256=digest.hexdigest())


def test_compiled_model_binding_rejects_symlinked_content(tmp_path):
    model = tmp_path / "projection.mlmodelc"
    model.mkdir()
    target = tmp_path / "outside"
    target.write_bytes(b"fixed")
    (model / "model.mil").symlink_to(target)
    spec = FixedProjection("fixture.adapter", DIGEST, 8, 4, 2)
    with pytest.raises(ValueError, match="regular files"):
        bind_compiled_model(spec, model, declared_sha256="0" * 64)


def test_full_projection_parity_and_nonfinite_refusal():
    spec = FixedProjection("fixture.adapter", DIGEST, 3, 2, 2)
    receipt = verify_projection_output(spec, [[1, 2], [3, 4.001]], [[1, 2], [3, 4]],
                                       absolute_tolerance=.002, relative_tolerance=0)
    assert receipt["parity"] is True and receipt["execution_device_observed"] is False
    with pytest.raises(ValueError, match="finite"):
        verify_projection_output(spec, [[1, 2], [3, float("nan")]], [[1, 2], [3, 4]],
                                 absolute_tolerance=.01, relative_tolerance=0)


def test_compute_plan_never_claims_observed_execution():
    spec = FixedProjection("fixture.adapter", DIGEST, 3, 2, 2)
    receipt = spec.receipt()
    planned = record_compute_plan(receipt, operations=[
        {"name": "linear", "anticipated_device": "neural_engine"},
        {"name": "output", "anticipated_device": "cpu"},
    ])
    assert planned["observed_used"] is False
    assert planned["execution_device_observed"] is False
    assert planned["compute_plan"][0]["anticipated_device"] == "neural_engine"


@pytest.mark.parametrize("kwargs", [
    {"input_width": True}, {"output_width": 0}, {"batch_rows": -1}, {"dtype": "float32"},
])
def test_invalid_projection_contracts_fail_closed(kwargs):
    values = {
        "adapter": "fixture.adapter",
        "artifact_sha256": DIGEST,
        "input_width": 3,
        "output_width": 2,
        "batch_rows": 1,
        "dtype": "float16",
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        FixedProjection(**values)
