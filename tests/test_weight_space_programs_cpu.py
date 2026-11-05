"""CPU algebra/serialization evidence; no model checkpoint was executed."""

import json

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from mlx2.experimental.weight_space.programs import (
    WeightProgram,
    analytical_directional_repair,
    compose_programs,
    create_program,
    export_program,
    import_lora,
    inspect_program,
    load_program,
    truncate_program,
    validate_program_artifact,
)

CHECKPOINT = "synthetic-cpu-fixture@" + "a" * 64
REVISION = "b" * 40


def program(factors=None, **overrides):
    args = {
        "checkpoint": CHECKPOINT,
        "source_revision": REVISION,
        "construction": "synthetic_cpu_test",
        "provenance": {"source": "synthetic algebra fixture, no model execution"},
    }
    args.update(overrides)
    return create_program(
        factors or {"layers.0.mlp.down_proj": (np.eye(3, 2), np.eye(2, 4))}, **args,
    )


def test_factors_and_identity_are_immutable_and_content_bound():
    a = np.eye(3, 2, dtype=np.float32)
    p = program({"layer": (a, np.eye(2, 4))})
    identity = p.program_id
    a[0, 0] = 12
    assert p.program_id == identity
    with pytest.raises(ValueError):
        p.factors[0].a.flags.writeable = True
    changed_provenance = p.provenance
    changed_provenance["source"] = "changed"
    assert p.program_id == identity
    assert program({"layer": (a, np.eye(2, 4))}).program_id != identity
    assert program(p.factor_map, provenance={"source": "different receipt"}).program_id != identity
    assert WeightProgram.from_dict(p.to_dict()).program_id == identity


def test_directional_repair_matches_sparse_weight_edit():
    w = np.arange(20, dtype=np.float32).reshape(4, 5) / 10
    u = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float32)
    p = analytical_directional_repair(
        {"layer": w}, {"layer": u}, {"layer": [1, 4]}, strength=6,
        checkpoint=CHECKPOINT, source_revision=REVISION, provenance={"source": "test"},
    )
    a, b = p.factor_map["layer"]
    expected = -6 * np.outer(u, (u @ w) * np.array([0, 1, 0, 0, 1]))
    np.testing.assert_allclose((a @ b).T, expected, atol=1e-6)
    x = np.arange(10, dtype=np.float32).reshape(2, 5) / 10
    np.testing.assert_allclose(x @ a @ b, x @ expected.T, atol=1e-5)
    assert p.rank == 1
    assert not p.approximate
    assert p.manifest()["status"]["qualified"] is False


def test_composition_handles_missing_targets_and_scales_once():
    one = program({"first": (np.eye(3, 2), np.eye(2, 4))})
    two = program({"first": (np.ones((3, 1)), np.ones((1, 4))), "second": (np.ones((2, 1)), np.ones((1, 2)))})
    combined = compose_programs([one, two], [2, -3])
    assert combined.rank == 3
    for target, (a, b) in combined.factor_map.items():
        expected = np.zeros((a.shape[0], b.shape[1]))
        for p, scale in [(one, 2), (two, -3)]:
            if target in p.factor_map:
                old_a, old_b = p.factor_map[target]
                expected += scale * old_a @ old_b
        np.testing.assert_allclose(a @ b, expected)
    assert combined.provenance["parents"] == [one.program_id, two.program_id]


def test_svd_truncation_is_explicit_and_measures_discarded_energy():
    p = program({"layer": (np.eye(3), np.diag([5.0, 2.0, 1.0]))})
    reduced = truncate_program(p, 1)
    a, b = reduced.factor_map["layer"]
    np.testing.assert_allclose(a @ b, np.diag([5.0, 0.0, 0.0]), atol=1e-6)
    assert reduced.approximate
    assert reduced.provenance["errors"]["layer"]["discarded_frobenius_norm"] == pytest.approx(np.sqrt(5))
    assert reduced.provenance["parent"] == p.program_id
    with pytest.raises(ValueError, match="below"):
        truncate_program(p, p.rank)


def test_norm_inspection_uses_factor_product():
    p = program()
    a, b = next(iter(p.factor_map.values()))
    stats = inspect_program(p)["factor_statistics"][0]
    assert stats["delta_frobenius_norm"] == pytest.approx(np.linalg.norm(a @ b))
    assert stats["stored_bytes"] == a.nbytes + b.nbytes


def test_roundtrip_and_expected_bindings(tmp_path):
    p = program()
    output = export_program(p, tmp_path / "program")
    restored = load_program(output)
    assert restored.program_id == p.program_id
    assert validate_program_artifact(output, expected_checkpoint=CHECKPOINT, expected_source_revision=REVISION) == p.manifest()
    config = json.loads((output / "adapter_config.json").read_text())
    assert config["lora_parameters"]["scale"] == 1
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        validate_program_artifact(output, expected_checkpoint="different@" + "a" * 64)
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_program_artifact(output, expected_source_revision="c" * 40)


def test_export_refuses_nonempty_and_can_publish_into_empty(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    export_program(program(), output)
    original = (output / "adapter_config.json").read_bytes()
    with pytest.raises(ValueError, match="empty directory"):
        export_program(program(), output)
    assert (output / "adapter_config.json").read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp-*"))


@pytest.mark.parametrize("part", ["factors", "manifest", "config"])
def test_modified_artifact_is_rejected(tmp_path, part):
    output = export_program(program(), tmp_path / "program")
    if part == "factors":
        tensors = load_file(str(output / "adapters.safetensors"))
        next(iter(tensors.values()))[0, 0] += 0.125
        save_file(tensors, str(output / "adapters.safetensors"))
    elif part == "manifest":
        path = output / "weight_program.json"
        payload = json.loads(path.read_text())
        payload["status"]["qualified"] = True
        path.write_text(json.dumps(payload))
    else:
        path = output / "adapter_config.json"
        payload = json.loads(path.read_text())
        payload["lora_parameters"]["scale"] = 2
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mismatch|match"):
        load_program(output)


def test_snapshot_validation_checks_consumed_tensor_bytes(tmp_path):
    p = program()
    output = export_program(p, tmp_path / "program")
    original_bytes = (output / "adapters.safetensors").read_bytes()
    (output / "adapters.safetensors").write_bytes(b"invalid changed file")
    assert validate_program_artifact(output, weights_payload=original_bytes)["program_id"] == p.program_id
    with pytest.raises(ValueError, match="safetensors"):
        validate_program_artifact(output, weights_payload=b"invalid")


def test_plain_lora_import_preserves_effective_scale(tmp_path):
    path = tmp_path / "legacy"
    path.mkdir()
    a, b = np.ones((3, 2), dtype=np.float32), np.ones((2, 4), dtype=np.float32)
    save_file({"layer.lora_a": a, "layer.lora_b": b}, str(path / "adapters.safetensors"))
    config = {"fine_tune_type": "lora", "lora_parameters": {"rank": 2, "scale": 0.5, "dropout": 0, "keys": ["layer"]}}
    (path / "adapter_config.json").write_text(json.dumps(config))
    imported = import_lora(path, checkpoint=CHECKPOINT, source_revision=REVISION, provenance={"source": "test trained adapter"})
    imported_a, imported_b = imported.factor_map["layer"]
    np.testing.assert_allclose(imported_a @ imported_b, 0.5 * a @ b)
    output = export_program(imported, tmp_path / "compiled")
    assert json.loads((output / "adapter_config.json").read_text())["lora_parameters"]["scale"] == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e100])
def test_nonfinite_or_float32_overflow_rejected(bad):
    with pytest.raises(ValueError, match="finite|overflow"):
        program({"layer": (np.array([[bad]]), np.ones((1, 1)))})


@pytest.mark.parametrize("overrides", [
    {"checkpoint": "*"}, {"checkpoint": "model@main"},
    {"source_revision": "HEAD"}, {"provenance": {}},
    {"provenance": {"number": np.nan}}, {"approximate": "yes"},
])
def test_invalid_identity_and_metadata_rejected(overrides):
    with pytest.raises(ValueError):
        program(**overrides)


def test_incompatible_composition_fails_closed():
    with pytest.raises(ValueError, match="same checkpoint"):
        compose_programs([program(), program(checkpoint="other@" + "a" * 64)])
    with pytest.raises(ValueError, match="dimensions"):
        compose_programs([program({"layer": (np.eye(3), np.eye(3))}), program({"layer": (np.eye(4), np.eye(4))})])


def test_embedded_factors_reject_tampering():
    payload = program().to_dict()
    next(iter(payload["factors"].values()))["a"][0][0] = 123
    with pytest.raises(ValueError, match="identity mismatch"):
        WeightProgram.from_dict(payload)


def test_derived_programs_preserve_session_memory_scope():
    private = program(provenance={"contains_private_memory": True, "promotion_scope": "session"})
    composed = compose_programs([program(), private])
    truncated = truncate_program(composed, 1)
    for derived in (composed, truncated):
        assert derived.provenance["contains_private_memory"] is True
        assert derived.provenance["promotion_scope"] == "session"
        assert derived.provenance["data_scope"] == "session"
        assert derived.provenance["parent_provenance"]


def test_stripped_marker_with_leftover_manifest_is_not_imported_as_plain_lora(tmp_path):
    p = program(approximate=True)
    output = export_program(p, tmp_path / "compiled")
    config = json.loads((output / "adapter_config.json").read_text())
    config.pop("weight_program")
    (output / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="adapter config"):
        import_lora(output, checkpoint=CHECKPOINT, source_revision=REVISION, provenance={"source": "relabelled"})
