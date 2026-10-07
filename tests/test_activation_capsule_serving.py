"""Host-only serving contracts for trusted activation-capsule requests."""

import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.runtime import capsule_bus
from mlx2.serving import (
    HostPromptCache,
    ServingEngine,
    activation_capsule_request,
    observe_activation_capsule_forward,
    prepare_semantic_prefill_inputs,
    request_apc_scope,
)


FINGERPRINT = "b" * 64
DIGEST = "a" * 64


def _wrapped(*, gate=0.25, fingerprint=FINGERPRINT, digest=DIGEST):
    return capsule_bus.TrustedActivationRequest(
        capsule_digest=digest,
        manifest={
            "schema": "mlx2-activation-capsule-v1",
            "payload_kind": "directional_residual",
        },
        tensors={"residual": np.asarray([1.0, 0.0], dtype=np.float32)},
        gate=gate,
        semantic_fingerprint=fingerprint,
        _capability=capsule_bus._TRUSTED_ACTIVATION_CAPABILITY,
    )


def _request(**updates):
    result = {
        "messages": [{"role": "user", "content": "hello"}],
        "_mlx2_semantic_fingerprint": FINGERPRINT,
        "_mlx2_activation_capsule": _wrapped(),
    }
    result.update(updates)
    return result


def test_public_mapping_cannot_fabricate_runtime_capsule():
    request = _request()
    request["_mlx2_activation_capsule"] = {
        "capsule_digest": DIGEST,
        "manifest": {"schema": "mlx2-activation-capsule-v1"},
        "tensors": {"residual": [1.0, 0.0]},
        "gate": 0.25,
        "semantic_fingerprint": FINGERPRINT,
    }

    with pytest.raises(ValueError, match="public requests cannot provide"):
        activation_capsule_request(request)

    request["_mlx2_activation_capsule"] = capsule_bus.TrustedActivationRequest(
        capsule_digest=DIGEST,
        manifest={"schema": "mlx2-activation-capsule-v1"},
        tensors={"residual": np.asarray([1.0, 0.0])},
        gate=0.25,
        semantic_fingerprint=FINGERPRINT,
        _capability=object(),
    )
    with pytest.raises(ValueError, match="public requests cannot provide"):
        activation_capsule_request(request)


def test_trusted_capsule_binds_apc_scope_and_host_token_cache_stays_serializable():
    request = _request()
    payload = activation_capsule_request(request)

    assert payload["capsule_digest"] == DIGEST
    assert request_apc_scope(request)[-4:] == (
        "activation-capsule-bus-v1",
        FINGERPRINT,
        DIGEST,
        0.25,
    )
    first = HostPromptCache.key(request)
    other = _request(
        _mlx2_semantic_fingerprint="c" * 64,
        _mlx2_activation_capsule=_wrapped(
            fingerprint="c" * 64, digest="d" * 64
        ),
    )
    assert first != HostPromptCache.key(other)


class _Adapter:
    def __init__(self):
        self.calls = []

    def neural_concept_prefill(self, tokens, payload, *, prefill_step):
        self.calls.append("neural")
        return {
            "deep_concept_memory": {"source": "neural"},
            "receipt": {"schema": "neural-v1"},
        }

    def activation_capsule_prefill(
        self, tokens, payload, *, prefill_step, route, batch_size
    ):
        self.calls.append("activation")
        assert set(payload) == {
            "capsule_digest",
            "manifest",
            "tensors",
            "gate",
            "semantic_fingerprint",
        }
        return {
            "deep_concept_memory": {"source": "activation"},
            "receipt": {
                "schema": "activation-v1",
                "status": "applied",
                "engaged": True,
                "relative_gate": payload["gate"],
            },
        }

    def compose_semantic_prefill_inputs(self, *prepared):
        self.calls.append("compose")
        return {
            "deep_concept_memory": {
                "components": tuple(item["deep_concept_memory"] for item in prepared)
            },
            # Serving must not forward arbitrary receipt containers to model.
            "receipts": tuple(item["receipt"] for item in prepared),
        }


def test_neural_and_activation_prefill_compose_without_receipt_kwargs():
    adapter = _Adapter()
    request = _request(_mlx2_neural_concepts={"concepts": [{"id": "c"}]})

    model_input, neural_receipt, activation_receipt = (
        prepare_semantic_prefill_inputs(
            adapter,
            request,
            [1, 2],
            prefill_step=8,
            route="ordinary",
        )
    )

    assert adapter.calls == ["neural", "activation", "compose"]
    # Final-row memory is never a prefill kwarg: it rides the prompt-tail
    # decode step with a one-step lifetime.
    assert set(model_input) == {"_mlx2_persistent_decode_inputs"}
    components = model_input["_mlx2_persistent_decode_inputs"][
        "deep_concept_memory"
    ]["components"]
    assert tuple(item["source"] for item in components) == ("neural", "activation")
    assert tuple(item["_mlx2_semantic_source"] for item in components) == (
        "neural_concept",
        "activation_capsule",
    )
    assert all(item["persistent_steps"] == 1 for item in components)
    assert neural_receipt["schema"] == "neural-v1"
    assert activation_receipt["schema"] == "activation-v1"


@pytest.mark.parametrize("route", ["native_mtp", "prompt_lookup", "external_draft"])
def test_speculative_route_refuses_before_any_adapter_engagement(route):
    adapter = _Adapter()

    with pytest.raises(ValueError, match=f"{route} route cannot apply"):
        prepare_semantic_prefill_inputs(
            adapter, _request(), [1], prefill_step=8, route=route
        )
    assert adapter.calls == []


def test_multimodal_prefill_refuses_before_any_adapter_engagement():
    adapter = _Adapter()

    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_semantic_prefill_inputs(
            adapter,
            _request(),
            [1],
            prefill_step=8,
            route="ordinary",
            prefill_input={"pixel_values": object()},
        )
    assert adapter.calls == []


def test_forward_observation_preserves_observed_implies_engaged():
    positive = SimpleNamespace(
        id="positive",
        activation_capsule_receipt={
            "status": "prepared",
            "engaged": False,
            "observed_used": False,
            "relative_gate": 0.25,
            "prepared_for_uid": 1,
            "prepared_request_id": "positive",
        }
    )
    zero = SimpleNamespace(
        id="zero",
        activation_capsule_receipt={
            "status": "prepared",
            "engaged": False,
            "observed_used": False,
            "relative_gate": 0.0,
            "prepared_for_uid": 2,
            "prepared_request_id": "zero",
        }
    )
    counts = Counter()

    def tail(gate):
        return {
            "deep_concept_memory": {
                "gate": gate,
                "_mlx2_semantic_source": "activation_capsule",
            }
        }

    observe_activation_capsule_forward(
        positive, counts, uid=1, prompt_tail_inputs=tail(0.25)
    )
    observe_activation_capsule_forward(
        zero, counts, uid=2, prompt_tail_inputs=tail(0.0)
    )

    assert positive.activation_capsule_receipt["observed_used"] is True
    assert positive.activation_capsule_receipt["engaged"] is True
    assert zero.activation_capsule_receipt["status"] == "identity"
    assert zero.activation_capsule_receipt["engaged"] is False
    assert zero.activation_capsule_receipt["observed_used"] is False
    assert counts["activation_capsule_bridge_forward_validated"] == 2
    assert counts["activation_capsule_bridge_engagements"] == 1
    assert counts["activation_capsule_bridge_observed_used"] == 1

    no_evidence = SimpleNamespace(
        id="no-evidence",
        activation_capsule_receipt={
            "status": "prepared",
            "engaged": False,
            "observed_used": False,
            "relative_gate": 0.5,
            "prepared_for_uid": 3,
            "prepared_request_id": "no-evidence",
        },
    )
    observe_activation_capsule_forward(no_evidence, counts, uid=3)
    assert no_evidence.activation_capsule_receipt["status"] == "forward_completed"
    assert no_evidence.activation_capsule_receipt["engaged"] is False
    assert no_evidence.activation_capsule_receipt["observed_used"] is False
    assert counts["activation_capsule_bridge_forward_completed"] == 1

    mismatch = SimpleNamespace(
        id="mismatch",
        activation_capsule_receipt={
            "status": "prepared",
            "relative_gate": 1.0,
            "prepared_for_uid": 9,
            "prepared_request_id": "mismatch",
        },
    )
    observe_activation_capsule_forward(
        mismatch,
        counts,
        uid=10,
        prompt_tail_inputs=tail(1.0),
    )
    assert mismatch.activation_capsule_receipt["status"] == "prepared"
    assert counts["activation_capsule_bridge_forward_identity_mismatches"] == 1


def _engine(route="ordinary"):
    configured = []

    class Adapter:
        def configure_activation_capsule_bridge(self, artifact=None):
            configured.append(artifact)

        def diagnostics(self):
            return {"activation_capsule_bridge": {"state": "configured"}}

    engine = object.__new__(ServingEngine)
    engine.ready = threading.Event()
    engine.ready.set()
    engine.thread = SimpleNamespace(is_alive=lambda: True)
    engine.error = None
    engine.prompt_lock = threading.Lock()
    engine.lock = threading.Lock()
    engine.adapter = Adapter()
    engine.snapshot = {"settings": {"route": route}, "execution": {}}
    return engine, configured


def test_default_off_worker_configuration_is_explicit_and_ordinary_only():
    engine, configured = _engine()
    engine.configure_activation_capsule_bridge("artifact", timeout=0.1)

    assert configured == ["artifact"]
    assert engine.snapshot["execution"]["activation_capsule_bridge"]["state"] == (
        "configured"
    )
    refused, configured = _engine("native_mtp")
    with pytest.raises(ValueError, match="needs the ordinary route"):
        refused.configure_activation_capsule_bridge("artifact", timeout=0.1)
    assert configured == []


def test_semantic_startup_composes_activation_bridge_with_neural_artifact():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "mlx2" / "server.py"
    ).read_text()
    neural = source.index("engine.configure_neural_concept_bridge(")
    activation = source.index("engine.configure_activation_capsule_bridge(", neural)

    assert activation > neural
