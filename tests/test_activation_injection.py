"""Host-only evidence for adapter-owned activation capsule preparation."""

from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.runtime.activation_injection import (
    ACTIVATION_CAPSULE_REVISION,
    apply_activation_injection_numpy,
    bind_activation_injection_bridge,
    compose_deep_concept_memory,
    prepare_activation_injection,
    prepare_loaded_activation_injection,
    step_persistent_deep_memory,
)


BINDINGS = {"model": "qwen35-9b@a", "tokenizer": "tok@b", "runtime": "mlx2@c"}
DIGEST = "a" * 64


def _bridge():
    artifact = SimpleNamespace(
        fingerprint="projector-v1",
        hidden_dim=2,
        state_dim=2,
        manifest={
            "bindings": BINDINGS,
            "activation_injection_layer": 2,
            "activation_capsule_revision": ACTIVATION_CAPSULE_REVISION,
        },
        arrays={
            "key_projection": np.eye(2, dtype=np.float32),
            "value_projection": np.eye(2, dtype=np.float32),
            "direction_projection": np.eye(2, dtype=np.float32),
        },
    )
    return bind_activation_injection_bridge(artifact, hidden_dim=2, layer_count=4)


def _snapshot(operation="directional_residual", **updates):
    result = {
        "capsule_digest": DIGEST,
        "capsule_revision": ACTIVATION_CAPSULE_REVISION,
        "projector_fingerprint": "projector-v1",
        "bindings": BINDINGS,
        "hidden_dim": 2,
        "injection_layer": 2,
        "operation": operation,
        "ordered": True,
        "gate": 0.5,
        "states": [[1.0, 0.0], [0.0, 1.0]],
    }
    result.update(updates)
    return result


def _prepare(snapshot):
    return prepare_activation_injection(
        snapshot, _bridge(), tokens=[1, 2], prefill_step=8
    )


def _loaded(kind, tensors, *, gate=0.25, **manifest_updates):
    manifest = {
        "schema": "mlx2-activation-capsule-v1",
        "payload_kind": kind,
        "bindings": {
            "source_model": "qwen35-9b@a",
            "target_model": BINDINGS["model"],
            "tokenizer": BINDINGS["tokenizer"],
            "runtime": BINDINGS["runtime"],
            "projector_revision": "projector-v1",
        },
        "tap": {
            "source_layer": 1,
            "source_tap": "post_block_residual",
            "target_layer": 2,
            "target_injection": "qwen35-post-block-residual-v1",
        },
        "geometry": {"hidden_width": 2, "sequence_length": 2},
        "metadata": {
            "representation_space": (
                "target_projected"
                if kind == "cross_attention_bank"
                else "target_hidden"
            )
        },
        "bounds": {
            "maximum_relative_norm": 0.5,
            "minimum_gate": 0.0,
            "maximum_gate": 0.5,
        },
    }
    manifest.update(manifest_updates)
    return prepare_loaded_activation_injection(
        manifest,
        tensors,
        _bridge(),
        capsule_digest=DIGEST,
        gate=gate,
        tokens=[1, 2],
        prefill_step=8,
    )


def test_directional_capsule_preserves_order_and_sign():
    forward = _prepare(_snapshot()).memory["values"]
    reversed_states = _prepare(
        _snapshot(states=[[0.0, 1.0], [1.0, 0.0]])
    ).memory["values"]
    sign_flipped = _prepare(
        _snapshot(direction_weights=[1.0, -1.0])
    ).memory["values"]

    assert not np.allclose(forward, reversed_states)
    assert not np.allclose(forward, sign_flipped)
    assert np.sign(sign_flipped[0, 1]) == -1


def test_continuous_prefix_is_external_ordered_memory_not_token_kv():
    hidden = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    forward = _prepare(_snapshot("continuous_prefix")).memory
    reverse = _prepare(
        _snapshot("continuous_prefix", states=[[0.0, 1.0], [1.0, 0.0]])
    ).memory

    assert forward["operation"] == "continuous_prefix"
    assert "order_bias" in forward
    assert "token_ids" not in forward and "positions" not in forward
    assert not np.allclose(
        apply_activation_injection_numpy(hidden, forward),
        apply_activation_injection_numpy(hidden, reverse),
    )


def test_zero_gate_is_exact_object_identity_but_still_validates_binding():
    prepared = _prepare(_snapshot(gate=0.0))
    hidden = np.asarray([[[0.25, -0.5]]], dtype=np.float32)

    assert apply_activation_injection_numpy(hidden, prepared.memory) is hidden
    assert prepared.receipt["status"] == "prepared"
    assert prepared.receipt["preparation_outcome"] == "identity"
    assert prepared.receipt["engaged"] is False
    with pytest.raises(ValueError, match="binding mismatch"):
        _prepare(_snapshot(gate=0.0, bindings={**BINDINGS, "runtime": "wrong"}))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("projector_fingerprint", "other", "projector fingerprint"),
        ("capsule_revision", "v2", "revision mismatch"),
        ("hidden_dim", 3, "hidden geometry"),
        ("injection_layer", 1, "injection layer"),
    ],
)
def test_snapshot_binding_mismatch_fails_closed(field, value, message):
    with pytest.raises(ValueError, match=message):
        _prepare(_snapshot(**{field: value}))


@pytest.mark.parametrize("route", ["prompt_lookup", "native_mtp", "external_draft", "pld"])
def test_speculative_routes_are_refused(route):
    with pytest.raises(ValueError, match=f"{route} route cannot apply"):
        prepare_activation_injection(
            _snapshot(), _bridge(), tokens=[1], prefill_step=8, route=route
        )


def test_persistent_direction_schedule_advances_stops_and_composes():
    prepared = _prepare(
        _snapshot(
            states=[[1.0, 0.0]],
            decode_states=[[1.0, 0.0], [0.0, 1.0]],
            decode_gates=[0.25, 0.75],
        )
    )
    neural = {
        "layer": 2,
        "keys": "neural-keys",
        "values": "neural-values",
        "gate": 0.5,
    }
    composed = compose_deep_concept_memory(neural, prepared.memory)

    first = step_persistent_deep_memory(composed, 0)
    second = step_persistent_deep_memory(composed, 1)
    after = step_persistent_deep_memory(composed, 2)
    assert len(first["components"]) == 2
    assert first["components"][1]["gate"] == 0.25
    assert np.array_equal(first["components"][1]["values"], [[1.0, 0.0]])
    assert second["components"][1]["gate"] == 0.75
    # The unscheduled neural component remains persistent after the capsule
    # trajectory ends; it is not overwritten by activation memory.
    assert after is neural


def test_receipt_reports_observed_engagement_and_state_boundaries():
    receipt = _prepare(_snapshot(gate=0.25)).receipt

    assert receipt["selected"] is True
    assert receipt["engaged"] is False
    assert receipt["observed_used"] is False
    assert receipt["status"] == "prepared"
    assert receipt["preparation_outcome"] == "nonzero_gate"
    assert receipt["capsule_digest"] == DIGEST
    assert receipt["projector_fingerprint"] == "projector-v1"
    assert receipt["operation"] == "directional_residual"
    assert receipt["chronological_kv"] is False
    assert receipt["recurrent_state_edit"] is False
    assert receipt["route"] == "ordinary" and receipt["batch_size"] == 1


@pytest.mark.parametrize(
    ("kind", "tensors", "operation"),
    [
        (
            "directional_residual",
            {"residual": np.asarray([1.0, -1.0], dtype=np.float32)},
            "directional_residual",
        ),
        (
            "continuous_prefix",
            {"prefix": np.eye(2, dtype=np.float32)},
            "continuous_prefix",
        ),
        (
            "cross_attention_bank",
            {
                "keys": np.eye(2, dtype=np.float32),
                "values": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            },
            "cross_attention_memory",
        ),
        (
            "ordered_trajectory",
            {
                "anchor": np.asarray([0.0, 0.0], dtype=np.float32),
                "basis": np.eye(2, dtype=np.float32),
                "coefficients": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            },
            "continuous_prefix",
        ),
    ],
)
def test_core_loaded_payload_kinds_map_to_model_operations(kind, tensors, operation):
    prepared = _loaded(kind, tensors)

    assert prepared.memory["operation"] == operation
    assert prepared.receipt["payload_kind"] == kind
    assert prepared.receipt["projector_fingerprint"] == "projector-v1"


def test_core_loaded_payload_rechecks_projector_tap_and_gate_bounds():
    tensors = {"residual": np.asarray([1.0, 0.0], dtype=np.float32)}
    with pytest.raises(ValueError, match="model/projector binding mismatch"):
        _loaded(
            "directional_residual",
            tensors,
            bindings={
                "target_model": BINDINGS["model"],
                "tokenizer": BINDINGS["tokenizer"],
                "runtime": BINDINGS["runtime"],
                "projector_revision": "wrong",
            },
        )
    with pytest.raises(ValueError, match="target tap mismatch"):
        _loaded(
            "directional_residual",
            tensors,
            tap={"target_layer": 1, "target_injection": "wrong"},
        )
    with pytest.raises(ValueError, match="exceeds adapter bounds"):
        _loaded("directional_residual", tensors, gate=0.75)
    with pytest.raises(ValueError, match="exceeds adapter bounds"):
        _loaded(
            "directional_residual",
            tensors,
            gate=0.25,
            bounds={
                "maximum_relative_norm": 0.1,
                "minimum_gate": 0.0,
                "maximum_gate": 0.5,
            },
        )


def test_loaded_zero_gate_preserves_exact_identity_after_contract_validation():
    prepared = _loaded(
        "directional_residual",
        {"residual": np.asarray([1.0, 0.0], dtype=np.float32)},
        gate=0.0,
    )
    hidden = np.asarray([[[1.0, 2.0]]], dtype=np.float32)

    assert apply_activation_injection_numpy(hidden, prepared.memory) is hidden
    assert prepared.receipt["status"] == "prepared"
    assert prepared.receipt["preparation_outcome"] == "identity"
    with pytest.raises(ValueError, match="geometry mismatch"):
        _loaded(
            "directional_residual",
            {"residual": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)},
            gate=0.0,
        )


def test_loaded_raw_state_refuses_cross_model_or_undeclared_representation():
    residual = {"residual": np.asarray([1.0, 0.0], dtype=np.float32)}
    with pytest.raises(ValueError, match="same-model target-hidden"):
        _loaded(
            "directional_residual",
            residual,
            bindings={
                "source_model": "different-source",
                "target_model": BINDINGS["model"],
                "tokenizer": BINDINGS["tokenizer"],
                "runtime": BINDINGS["runtime"],
                "projector_revision": "projector-v1",
            },
        )
    with pytest.raises(ValueError, match="same-model target-hidden"):
        _loaded("continuous_prefix", {"prefix": np.eye(2)}, metadata={})
    with pytest.raises(ValueError, match="not target-projected"):
        _loaded(
            "cross_attention_bank",
            {"keys": np.eye(2), "values": np.eye(2)},
            metadata={"representation_space": "target_hidden"},
        )
