"""Sweep 2026-10-06 (capsule lane, CAP-2/3/7): served semantic memory acts
where the model reads it.

Serving prefills ``tokens[:-1]`` and feeds the final prompt token in the first
decode step.  ``_apply_deep_concept_memory`` amends the last row of the
forward it is given, so a capsule passed as a prefill kwarg landed on prompt
row n-2.  At the last decoder layer nothing reads that row, yet the receipt
said ``observed_used``.  The memory now rides the prompt-tail forward, the
seam the M3 gate measured (full-prompt forward, last row).
"""

import mlx.core as mx

mx.set_default_device(mx.cpu)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mlx2.runtime import capsule_bus  # noqa: E402
from mlx2.runtime.activation_injection import (  # noqa: E402
    compose_deep_concept_memory,
    step_persistent_deep_memory,
)

PROMPT = [3, 9, 14, 27, 5, 41, 8, 19]


def _memory(direction, layer, gate=1.0):
    return {
        "operation": "directional_residual",
        "values": mx.array(direction[None, :], dtype=mx.float32),
        "layer": layer,
        "gate": gate,
    }


def _flipping_direction(model, layer):
    """An lm-head row whose direction flips the full-prompt argmax."""
    ordinary = model(mx.array([PROMPT]))[0, -1]
    base = int(mx.argmax(ordinary).item())
    weight = np.asarray(model.lm_head.weight, dtype=np.float32)
    for token in np.argsort(-np.asarray(ordinary))[1:]:
        direction = weight[int(token)] / np.linalg.norm(weight[int(token)])
        logits = model(
            mx.array([PROMPT]), deep_concept_memory=_memory(direction, layer)
        )[0, -1]
        if int(mx.argmax(logits).item()) != base:
            return direction, base, int(mx.argmax(logits).item())
    pytest.skip("no flipping direction for this seed")


def _trusted(gate):
    fingerprint = "b" * 64
    return {
        "_mlx2_semantic_fingerprint": fingerprint,
        "_mlx2_activation_capsule": capsule_bus.TrustedActivationRequest(
            capsule_digest="a" * 64,
            manifest={"schema": "mlx2-activation-capsule-v1"},
            tensors={"residual": (1.0,)},
            gate=gate,
            semantic_fingerprint=fingerprint,
            _capability=capsule_bus._TRUSTED_ACTIVATION_CAPABILITY,
        ),
    }


def _run_engine(monkeypatch, memory=None, *, gate=1.0, neural=None, prompt=PROMPT):
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()

    class Bridge:
        def activation_capsule_prefill(self, tokens, payload, *, prefill_step, route, batch_size):
            return {
                "deep_concept_memory": {**memory, "gate": payload["gate"]},
                "receipt": {
                    "schema": "mlx2-activation-capsule-injection-receipt-v1",
                    "status": "prepared",
                    "relative_gate": payload["gate"],
                    "states": 1,
                    "decode_steps": 0,
                },
            }

        def neural_concept_prefill(self, tokens, payload, *, prefill_step):
            return {
                "deep_concept_memory": dict(neural),
                "receipt": {
                    "schema": "mlx2-neural-concept-prefill-v2",
                    "engaged": True,
                    "relative_gate": neural["gate"],
                },
            }

    engine = make_engine(model, vocab, mtp=False, adapter_mixin=Bridge)
    request = {
        "tokens": list(prompt),
        "max_tokens": 4,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 3,
    }
    if memory is not None:
        request.update(_trusted(gate))
    if neural is not None:
        request["_mlx2_neural_concepts"] = {"concepts": [{"id": "c"}]}
    try:
        output = run(engine, request)
        counts = dict(engine.counts)
    finally:
        engine.close()
    output["logprobs"] = [event["logprob"] for event in output["events"] if "logprob" in event]
    return output, counts


@pytest.fixture
def last_layer_flip():
    from route_harness import tiny_qwen38_mtp

    probe, _ = tiny_qwen38_mtp()
    last = len(probe.layers) - 1
    direction, base, flipped = _flipping_direction(probe, last)
    return _memory(direction, last), base, flipped


def test_served_capsule_at_last_layer_matches_full_prompt_seam(monkeypatch, last_layer_flip):
    memory, base, flipped = last_layer_flip
    ordinary, _ = _run_engine(monkeypatch)
    served, counts = _run_engine(monkeypatch, memory, gate=1.0)
    receipt = served["receipt"]["activation_capsule_bridge"]

    assert ordinary["tokens"][0] == base
    assert served["tokens"][0] == flipped
    assert served["logprobs"][0] != ordinary["logprobs"][0]
    assert receipt["status"] == "applied"
    assert receipt["engaged"] is True and receipt["observed_used"] is True
    assert receipt["forward_evidence"] == "evaluated_prompt_tail_deep_concept_memory"
    assert receipt["read_position"] == "final_prompt_row"
    assert counts["activation_capsule_bridge_observed_used"] == 1
    # One-shot: later positions are ordinary decode of the new prefix.
    assert counts["apcv2_write_suppressed_requests"] == 1


def test_zero_gate_capsule_is_bit_identical_to_ordinary(monkeypatch, last_layer_flip):
    memory, _base, _flipped = last_layer_flip
    ordinary, _ = _run_engine(monkeypatch)
    served, counts = _run_engine(monkeypatch, memory, gate=0.0)
    receipt = served["receipt"]["activation_capsule_bridge"]

    assert served["tokens"] == ordinary["tokens"]
    assert served["logprobs"] == ordinary["logprobs"]
    assert receipt["status"] == "identity"
    assert receipt["engaged"] is False and receipt["observed_used"] is False
    assert counts["activation_capsule_bridge_forward_validated"] == 1
    assert counts.get("activation_capsule_bridge_observed_used", 0) == 0


def test_single_token_prompt_still_applies_the_capsule(monkeypatch, last_layer_flip):
    memory, _base, _flipped = last_layer_flip
    ordinary, _ = _run_engine(monkeypatch, prompt=PROMPT[-1:])
    served, _ = _run_engine(monkeypatch, memory, gate=1.0, prompt=PROMPT[-1:])
    receipt = served["receipt"]["activation_capsule_bridge"]

    assert served["logprobs"][0] != ordinary["logprobs"][0]
    assert receipt["observed_used"] is True


def test_neural_one_shot_memory_engages_after_the_forward(monkeypatch, last_layer_flip):
    memory, base, flipped = last_layer_flip
    served, counts = _run_engine(monkeypatch, neural=memory)
    receipt = served["receipt"]["neural_concept_bridge"]

    assert served["tokens"][0] == flipped != base
    assert receipt["status"] == "applied"
    assert receipt["engaged"] is True and receipt["observed_used"] is True
    assert counts["neural_concept_bridge_prepared"] == 1
    assert counts["neural_concept_bridge_engagements"] == 1


def test_composed_memory_lifetimes_follow_their_schedule():
    neural = {
        "operation": "cross_attention_memory",
        "keys": mx.ones((1, 4)),
        "values": mx.ones((1, 4)),
        "decode_values": mx.ones((2, 4)),
        "layer": 1,
        "gate": 0.3,
        "temperature": 0.5,
        "tag": "neural",
    }
    activation = {
        "operation": "continuous_prefix",
        "keys": mx.ones((3, 4)),
        "values": mx.ones((3, 4)),
        "order_bias": mx.zeros((3,)),
        "layer": 1,
        "gate": 0.2,
        "temperature": 0.5,
        "tag": "activation",
    }

    def tags(memory):
        if memory is None:
            return []
        return [item["tag"] for item in memory.get("components", (memory,))]

    composed = compose_deep_concept_memory(neural, activation)
    # CAP-3: no explicit lifetime means not persistent.
    assert {step: tags(step_persistent_deep_memory(composed, step)) for step in (0, 1, 2, 500)} == {
        0: ["neural"], 1: ["neural"], 2: [], 500: []
    }
    # Serving's final-row lifetime: the one-shot capsule rides step 0 only.
    from mlx2.runtime.activation_injection import final_row_decode_memory

    lived = final_row_decode_memory(composed)
    assert {step: tags(step_persistent_deep_memory(lived, step)) for step in (0, 1, 2, 500)} == {
        0: ["neural", "activation"], 1: ["neural"], 2: [], 500: []
    }


def test_scheduled_latent_attention_with_several_concepts_validates():
    from mlx2.runtime.models.qwen38_27b import _validate_deep_concept_component

    memory = {
        "keys": mx.ones((3, 8)),
        "values": mx.ones((3, 8)),
        "decode_values": mx.ones((2, 8)),
        "layer": 1,
        "gate": 0.3,
        "temperature": 0.5,
    }
    stepped = step_persistent_deep_memory(memory, 0)
    assert _validate_deep_concept_component(mx.zeros((1, 1, 8)), stepped, layer_count=4) == 1


def test_scheduled_step_equals_one_key_cross_attention():
    from mlx2.runtime.models.qwen38_27b import _apply_deep_concept_memory

    mx.random.seed(3)
    hidden = mx.random.normal((1, 1, 8))
    schedule = mx.random.normal((2, 8))
    cross = {
        "keys": mx.random.normal((1, 8)),
        "values": schedule[:1],
        "layer": 0,
        "gate": 0.4,
        "temperature": 0.5,
    }
    stepped = step_persistent_deep_memory({**cross, "decode_values": schedule}, 0)
    assert mx.array_equal(
        _apply_deep_concept_memory(hidden, stepped),
        _apply_deep_concept_memory(hidden, cross),
    ).item()


def test_schedule_index_survives_a_long_lived_generation_batch():
    """A capsule lane admitted after ordinary traffic starts at schedule row 0."""
    from route_harness import tiny_qwen38_mtp

    from mlx2.runtime.generate import BatchGenerator

    class Recording:
        def __init__(self, target):
            self.target = target
            self.rows = []

        def __call__(self, inputs, cache=None, deep_concept_memory=None, **kwargs):
            if deep_concept_memory is not None:
                self.rows.append(deep_concept_memory["values"])
            return self.target(inputs, cache=cache, **kwargs)

        def __getattr__(self, name):
            return getattr(self.target, name)

    inner, _vocab = tiny_qwen38_mtp()
    model = Recording(inner)
    width = inner.model.embed_tokens.weight.shape[1]
    schedule = mx.eye(width, dtype=mx.float32)[:3]
    memory = {"values": schedule[:1], "decode_values": schedule, "layer": 0, "gate": 0.1,
              "operation": "directional_residual"}
    generator = BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                               prefill_step_size=8, prefill_batch_window=1)
    generator.insert([[3, 4]], max_tokens=[6])
    done = set()
    for _ in range(40):
        _prompts, responses = generator.next()
        done |= {response.uid for response in responses if response.finish_reason}
        if done:
            break
    capsule = generator.insert([[5, 6]], max_tokens=[3],
                               prefill_inputs=[{"_mlx2_persistent_decode_inputs": {"deep_concept_memory": memory}}])[0]
    for _ in range(40):
        _prompts, responses = generator.next()
        if any(response.uid == capsule and response.finish_reason for response in responses):
            break
    assert len(model.rows) == 3
    for row, expected in zip(model.rows, (0, 1, 2)):
        assert mx.array_equal(row, schedule[expected : expected + 1]).item()


def test_capsule_request_reuses_the_exact_ordinary_prefix(monkeypatch, last_layer_flip):
    """CAP-6: rows [:-1] are ordinary state, so a warm capsule run is exact."""
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    memory, _base, _flipped = last_layer_flip
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()

    class Bridge:
        def activation_capsule_prefill(self, tokens, payload, *, prefill_step, route, batch_size):
            return {
                "deep_concept_memory": {**memory, "gate": payload["gate"]},
                "receipt": {"status": "prepared", "relative_gate": payload["gate"]},
            }

    engine = make_engine(model, vocab, mtp=False, adapter_mixin=Bridge)
    prompt = list(range(5, 45))
    request = {"tokens": prompt, "max_tokens": 3, "temperature": 0}
    try:
        cold = run(engine, {**request, **_trusted(1.0)})
        run(engine, request)  # publishes the ordinary prompt boundary
        warm = run(engine, {**request, **_trusted(1.0)})
        counts = dict(engine.counts)
    finally:
        engine.close()

    assert cold["receipt"]["cached_tokens"] == 0
    assert warm["receipt"]["cached_tokens"] == len(prompt) - 1
    assert warm["tokens"] == cold["tokens"]
    assert warm["receipt"]["activation_capsule_bridge"]["observed_used"] is True
    assert counts["activation_capsule_exact_prefix_hits"] == 1
    # Capsule requests never publish: only the ordinary request wrote.
    assert counts["apcv2_write_suppressed_requests"] == 2


def test_conditioned_identity_binds_the_gate_and_lookup_is_ordinary():
    from mlx2.serving import request_apc_lookup_scope, request_apc_scope

    low = {"tokens": [1, 2], **_trusted(0.25)}
    high = {"tokens": [1, 2], **_trusted(0.5)}
    assert request_apc_scope(low) != request_apc_scope(high)
    assert request_apc_scope(low)[-1] == 0.25
    ordinary = request_apc_scope({"tokens": [1, 2]})
    assert request_apc_lookup_scope(low) == request_apc_lookup_scope(high) == ordinary
