"""CPU quantization, target isolation and matched-perplexity contracts."""

import hashlib
import threading

import numpy as np
import pytest

from mlx2.experimental.weight_space.overlays import (
    InMemoryOverlayBank,
    QuantizationSpec,
    TensorTarget,
    compare_quality,
    pack_codes,
    quantize_tensor,
    reconstruction_metrics,
    teacher_forced_quality,
    unpack_codes,
)


@pytest.mark.parametrize("bits", range(1, 16))
@pytest.mark.parametrize("length", [1, 7, 4097])
def test_all_q_widths_pack_exact_codes_including_chunk_and_byte_tails(bits, length):
    codes = np.random.default_rng(bits).integers(0, 1 << bits, length, dtype=np.uint16)
    packed = pack_codes(codes, bits)
    assert len(packed) == (bits * length + 7) // 8
    np.testing.assert_array_equal(unpack_codes(packed, bits, length), codes)


@pytest.mark.parametrize("bits", range(1, 16))
def test_all_q_widths_reconstruct_bounded_affine_weights(bits):
    weights = np.random.default_rng(17).normal(size=(13, 17)).astype("f4")
    quantized = quantize_tensor(weights, QuantizationSpec(bits, 32))
    restored = quantized.dequantize()
    error = reconstruction_metrics(weights, restored)
    assert error["maximum_absolute_error"] <= float(quantized.scales.max()) / 2 + 2e-6
    assert quantized.source_sha256 == hashlib.sha256(weights.tobytes()).hexdigest()
    assert len(quantized.packed) == (weights.size * bits + 7) // 8
    with pytest.raises(ValueError):
        quantized.scales.flags.writeable = True


def test_constant_zero_and_partial_groups_preserve_values():
    weights = np.array([[0, 0, 0, 2, 2, 2, -3]], dtype="f4")
    result = quantize_tensor(weights, QuantizationSpec(1, 3))
    np.testing.assert_array_equal(result.dequantize(), weights)


def test_fp4_has_explicit_format_and_exact_codebook():
    weights = np.array([[0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]], dtype="f4")
    result = quantize_tensor(weights, QuantizationSpec(4, 16, "fp4_e2m1_f32_scale"))
    np.testing.assert_array_equal(result.dequantize(), weights)
    assert result.spec.format != "mxfp4"
    with pytest.raises(ValueError, match="exactly 4"):
        QuantizationSpec(3, 16, "fp4_e2m1_f32_scale")


@pytest.mark.parametrize("bits", [0, 16, True, 4.0])
def test_bad_precision_is_refused(bits):
    with pytest.raises(ValueError):
        QuantizationSpec(bits)


@pytest.mark.parametrize("checkpoint,source", [("artifact@main", "commit"), ("artifact@hash", "HEAD")])
def test_overlay_identity_refuses_mutable_aliases(checkpoint, source):
    with pytest.raises(ValueError):
        InMemoryOverlayBank(checkpoint=checkpoint, source_revision=source, budget_bytes=1024)


def bank(**kwargs):
    return InMemoryOverlayBank(checkpoint="synthetic-experts@" + "a" * 64,
                               source_revision="e448f070b84642bfc5d11dc04719e40336326a68",
                               budget_bytes=kwargs.pop("budget_bytes", 65536), **kwargs)


def test_one_expert_override_does_not_change_base_or_other_experts():
    registry = bank()
    target = TensorTarget(23, 123, "down_proj")
    weights = np.random.default_rng(3).normal(size=(16, 16)).astype("f4")
    original = weights.copy()
    info = registry.create("trial-q4", target, weights, source_format="fp8_e4m3fn_effective",
                           spec=QuantizationSpec(4, 16))
    np.testing.assert_array_equal(weights, original)
    assert info["in_memory_only"] and not info["qualified"]
    with registry.acquire("trial-q4") as view:
        assert set(view.targets) == {target}
        assert TensorTarget(23, 122, "down_proj") not in view.targets
        with pytest.raises(TypeError):
            view.targets[target] = None
        with pytest.raises(ValueError, match="pinned"):
            registry.delete("trial-q4")
    registry.delete("trial-q4")
    assert registry.snapshot()["resident_bytes"] == 0
    np.testing.assert_array_equal(weights, original)


def test_budget_includes_scale_metadata_and_composites_share_storage():
    registry = bank(budget_bytes=128)
    weights = np.ones((8, 8), dtype="f4")
    with pytest.raises(ValueError, match="memory budget"):
        registry.create("too-much", TensorTarget(0, 0, "w1"), weights,
                        source_format="f32", spec=QuantizationSpec(1, 1))
    registry.create("a", TensorTarget(0, 0, "w1"), weights,
                    source_format="f32", spec=QuantizationSpec(1, 64))
    registry.create("b", TensorTarget(0, 1, "w1"), weights * 2,
                    source_format="f32", spec=QuantizationSpec(1, 64))
    before = registry.snapshot()["resident_bytes"]
    registry.compose("both", ["a", "b"])
    assert registry.snapshot()["resident_bytes"] == before
    registry.delete("a")
    assert registry.snapshot()["resident_bytes"] == before
    with registry.acquire("both") as view:
        assert len(view.targets) == 2


def test_separately_allocated_identical_tensors_are_charged_separately():
    registry = bank(budget_bytes=16)
    weights = np.ones((8, 8), dtype="f4")
    registry.create("one", TensorTarget(1, 0, "w1"), weights,
                    source_format="f32", spec=QuantizationSpec(1, 64))
    with pytest.raises(ValueError, match="memory budget"):
        registry.create("two", TensorTarget(1, 1, "w1"), weights,
                        source_format="f32", spec=QuantizationSpec(1, 64))


def test_early_budget_refusal_never_runs_quantization(monkeypatch):
    from mlx2.experimental.weight_space import overlays

    def forbidden(*args):
        raise AssertionError("quantizer ran before memory admission")

    monkeypatch.setattr(overlays, "quantize_tensor", forbidden)
    with pytest.raises(ValueError, match="memory budget"):
        bank(budget_bytes=1).create("oversized", TensorTarget(0, None, "w"), np.eye(4),
                                    source_format="f32", spec=QuantizationSpec(4))
    registry = bank(construction_budget_bytes=1024)
    with pytest.raises(ValueError, match="construction workspace"):
        registry.create("scratch", TensorTarget(0, None, "w"), np.eye(4),
                        source_format="f32", spec=QuantizationSpec(4))
    assert registry.snapshot()["construction_reserved_bytes"] == 0


def test_construction_reservation_is_observable_without_blocking_scrapes(monkeypatch):
    from mlx2.experimental.weight_space import overlays

    entered, release = threading.Event(), threading.Event()
    real = overlays.quantize_tensor

    def controlled(*args):
        entered.set()
        assert release.wait(timeout=5)
        return real(*args)

    monkeypatch.setattr(overlays, "quantize_tensor", controlled)
    registry = bank()
    errors = []

    def construct():
        try:
            registry.create("candidate", TensorTarget(0, None, "w"), np.eye(4),
                            source_format="f32", spec=QuantizationSpec(4))
        except (AssertionError, ValueError, RuntimeError, TypeError) as error:
            errors.append(error)

    thread = threading.Thread(target=construct)
    thread.start()
    try:
        assert entered.wait(timeout=2)
        snapshot = registry.snapshot()
        assert snapshot["construction_reserved_bytes"] > 0 and snapshot["payload_reserved_bytes"] > 0
        with pytest.raises(ValueError, match="pending"):
            registry.check_capacity("candidate", (4, 4), QuantizationSpec(4))
    finally:
        release.set()
        thread.join(timeout=5)
    assert not errors and not thread.is_alive()
    assert registry.snapshot()["construction_reserved_bytes"] == 0


def test_duplicate_target_composition_and_stale_source_fail_closed():
    valid = [True]
    registry = bank(source_validator=lambda target, fingerprint: valid[0])
    target = TensorTarget(1, 2, "w2")
    for name, bits in (("a", 3), ("b", 5)):
        registry.create(name, target, np.eye(4), source_format="f32",
                        spec=QuantizationSpec(bits), source_tensor_fingerprint="exact-source")
    with pytest.raises(ValueError, match="ambiguous"):
        registry.compose("invalid", ["a", "b"])
    valid[0] = False
    with pytest.raises(ValueError, match="revision changed"):
        registry.acquire("a")


def test_perplexity_uses_identical_explicit_next_token_rows():
    reference_logits = np.array([[2, 0], [0, 2]], dtype="f4")
    candidate_logits = np.array([[1, 0], [0, 1]], dtype="f4")
    labels = np.array([0, 1])
    opts = {"tokenizer_fingerprint": "toy-tokenizer", "input_digest": "same-tokens"}
    ref = teacher_forced_quality(reference_logits, labels, **opts)
    candidate = teacher_forced_quality(candidate_logits, labels, **opts)
    comparison = compare_quality(ref, candidate)
    assert ref["perplexity"] == pytest.approx(1 + np.exp(-2))
    assert comparison["delta_nll_nats"] > 0
    assert comparison["delta_perplexity"] > 0
    assert not comparison["qualified"]
    changed = teacher_forced_quality(candidate_logits, labels[::-1], **opts)
    with pytest.raises(ValueError, match="labels"):
        compare_quality(ref, changed)


def test_logits_inputs_are_bounded_finite_and_no_implicit_label_shift():
    opts = {"tokenizer_fingerprint": "t", "input_digest": "i"}
    with pytest.raises(ValueError, match="align"):
        teacher_forced_quality(np.zeros((2, 3)), np.array([1]), **opts)
    with pytest.raises(ValueError, match="finite"):
        teacher_forced_quality(np.array([[np.nan, 1]]), np.array([1]), **opts)
    with pytest.raises(ValueError, match="select"):
        teacher_forced_quality(np.zeros((2, 3)), np.array([1, 2]), mask=np.array([False, False]), **opts)


def test_nonzero_padding_and_out_of_range_codes_are_rejected():
    with pytest.raises(ValueError, match="padding"):
        unpack_codes(bytes([255]), 3, 1)
    with pytest.raises(ValueError, match="bit width"):
        pack_codes(np.array([8]), 3)


def test_perplexity_is_stable_for_large_tied_logits():
    opts = {"tokenizer_fingerprint": "toy-tokenizer", "input_digest": "huge-logits"}
    quality = teacher_forced_quality(np.array([[1e16, 1e16]]), np.array([0]), **opts)
    assert quality["mean_nll_nats"] == pytest.approx(np.log(2))


def test_failed_view_construction_does_not_leak_a_pin(monkeypatch):
    registry = bank()
    registry.create("trial", TensorTarget(0, 0, "w1"), np.ones((8, 8), dtype="f4"),
                    source_format="f32", spec=QuantizationSpec(4, 8))

    def exhausted(overrides):
        raise MemoryError("host allocation failed")

    monkeypatch.setattr(registry, "_fingerprint", exhausted)
    with pytest.raises(MemoryError):
        registry.acquire("trial")
    monkeypatch.undo()
    assert registry.snapshot()["active_requests"] == 0
    registry.delete("trial")
