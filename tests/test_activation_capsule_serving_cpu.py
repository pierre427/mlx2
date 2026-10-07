"""Tiny CPU integration: trusted activation memory reaches ordinary prefill."""

import pytest

from mlx2.runtime import capsule_bus


def _payload(gate=0.25):
    fingerprint = "b" * 64
    return {
        "_mlx2_semantic_fingerprint": fingerprint,
        "_mlx2_activation_capsule": capsule_bus.TrustedActivationRequest(
            capsule_digest="a" * 64,
            manifest={"schema": "mlx2-activation-capsule-v1"},
            tensors={"residual": (1.0, 0.0)},
            gate=gate,
            semantic_fingerprint=fingerprint,
            _capability=capsule_bus._TRUSTED_ACTIVATION_CAPABILITY,
        ),
    }


def test_trusted_activation_is_observed_only_after_ordinary_model_forward(monkeypatch):
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    inner, vocab = tiny_qwen38_mtp()
    seen = []

    class Model:
        def __init__(self, target):
            self.target = target

        def __call__(self, inputs, cache=None, deep_concept_memory=None, **kwargs):
            if deep_concept_memory is not None:
                seen.append(deep_concept_memory)
            return self.target(inputs, cache=cache, **kwargs)

        def __getattr__(self, name):
            return getattr(self.target, name)

    class Bridge:
        def activation_capsule_prefill(
            self, tokens, payload, *, prefill_step, route, batch_size
        ):
            assert route == "ordinary" and batch_size == 1
            return {
                "deep_concept_memory": {
                    "capsule": payload["capsule_digest"],
                    "gate": payload["gate"],
                },
                "receipt": {
                    "schema": "tiny.activation.v1",
                    "status": "applied",
                    "engaged": True,
                    "relative_gate": payload["gate"],
                },
            }

    engine = make_engine(Model(inner), vocab, mtp=False, adapter_mixin=Bridge)
    request = {
        "tokens": [1, 2, 3, 4],
        "max_tokens": 2,
        "temperature": 0,
        **_payload(),
    }
    try:
        output = run(engine, request)
        counts = dict(engine.counts)
    finally:
        engine.close()

    receipt = output["receipt"]["activation_capsule_bridge"]
    assert seen
    assert receipt["status"] == "applied"
    assert receipt["engaged"] is True
    assert receipt["observed_used"] is True
    assert receipt["forward_evidence"] == "evaluated_prompt_tail_deep_concept_memory"
    assert receipt["read_position"] == "final_prompt_row"
    # The memory reached the model only on the prompt-tail forward.
    assert len(seen) == 1
    assert counts["activation_capsule_bridge_prepared"] == 1
    assert counts["activation_capsule_bridge_engagements"] == 1
    assert counts["activation_capsule_bridge_observed_used"] == 1
    assert counts["apcv2_write_suppressed_requests"] == 1


def test_zero_gate_forward_is_identity_not_observed_use(monkeypatch):
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    inner, vocab = tiny_qwen38_mtp()

    class Bridge:
        def activation_capsule_prefill(
            self, tokens, payload, *, prefill_step, route, batch_size
        ):
            return {
                "deep_concept_memory": {"identity": True, "gate": 0.0},
                "receipt": {
                    "schema": "tiny.activation.v1",
                    "status": "identity",
                    "engaged": False,
                    "relative_gate": 0.0,
                },
            }

    class Model:
        def __init__(self, target):
            self.target = target

        def __call__(self, inputs, cache=None, deep_concept_memory=None, **kwargs):
            return self.target(inputs, cache=cache, **kwargs)

        def __getattr__(self, name):
            return getattr(self.target, name)

    engine = make_engine(Model(inner), vocab, mtp=False, adapter_mixin=Bridge)
    try:
        output = run(
            engine,
            {
                "tokens": [1, 2, 3],
                "max_tokens": 1,
                "temperature": 0,
                **_payload(gate=0.0),
            },
        )
        counts = dict(engine.counts)
    finally:
        engine.close()

    receipt = output["receipt"]["activation_capsule_bridge"]
    assert receipt["status"] == "identity"
    assert receipt["engaged"] is False
    assert receipt["observed_used"] is False
    assert counts["activation_capsule_bridge_forward_validated"] == 1
    assert counts.get("activation_capsule_bridge_engagements", 0) == 0
    assert counts.get("activation_capsule_bridge_observed_used", 0) == 0


def test_public_or_speculative_activation_refuses_before_bridge(monkeypatch):
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    inner, vocab = tiny_qwen38_mtp()
    calls = []

    class Bridge:
        def activation_capsule_prefill(self, *args, **kwargs):
            calls.append(True)
            raise AssertionError("bridge must not run")

    public = {
        "capsule_digest": "a" * 64,
        "manifest": {"schema": "mlx2-activation-capsule-v1"},
        "tensors": {"residual": [1.0, 0.0]},
        "gate": 0.25,
        "semantic_fingerprint": "b" * 64,
    }
    ordinary = make_engine(inner, vocab, mtp=False, adapter_mixin=Bridge)
    speculative = make_engine(inner, vocab, mtp=True, adapter_mixin=Bridge)
    try:
        with pytest.raises(ValueError, match="public requests cannot provide"):
            ordinary.submit(
                {
                    "tokens": [1, 2],
                    "max_tokens": 1,
                    "_mlx2_semantic_fingerprint": "b" * 64,
                    "_mlx2_activation_capsule": public,
                }
            )
        with pytest.raises(ValueError, match="native_mtp route cannot apply"):
            speculative.submit(
                {"tokens": [1, 2], "max_tokens": 1, **_payload()}
            )
    finally:
        ordinary.close()
        speculative.close()

    assert calls == []
