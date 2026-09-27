"""Per-group RMSNorm (+1) convention: trunk vs MTP head, fail closed.

Synthetic CPU weights shaped like the real failure modes (calibrated in
adapters/norm_repair.py against the local checkpoints):

* oQ's per-tensor skip: converted trunk, most of the head shifted, but q_norm,
  k_norm and mtp.norm left raw (Jundot/Qwen3.6-27B-oQ4e-mtp);
* the same in fp16 / fp32 storage (Noctalin/...-oQ4-fp16-mtp);
* raw HF layout (conv1d last dim != 1), where everything is shifted once.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.adapters.norm_repair import (
    NormConventionError,
    UnshiftedNorm,
    decide_norm_convention,
    norm_family,
    resolve_norm_convention,
    tensor_sha256,
    trunk_family_means,
)
from mlx2.runtime.models import qwen4_exp, qwen38_27b

QWEN38_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    "mtp.norm.weight",
    ".pre_fc_norm_embedding.weight",
    ".pre_fc_norm_hidden.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

# Runtime (shifted) family means taken from Jundot/Qwen3.6-27B-oQ4e-mtp.
TRUNK = {
    "input_layernorm": 1.06,
    "post_attention_layernorm": 1.03,
    "self_attn.q_norm": 1.46,
    "self_attn.k_norm": 1.45,
}
HEAD = {
    "layers.0.input_layernorm": 1.04,
    "layers.0.post_attention_layernorm": 1.21,
    "layers.0.self_attn.q_norm": 1.755,
    "layers.0.self_attn.k_norm": 1.743,
    "norm": 2.274,
    "pre_fc_norm_embedding": 0.56,
    "pre_fc_norm_hidden": 0.83,
}
RAW_IN_HEAD = ("layers.0.self_attn.q_norm", "layers.0.self_attn.k_norm", "norm")


def _vec(mean, dtype, n=16, seed=0):
    # Deterministic spread around ``mean`` so per-tensor hashes differ.
    base = mx.linspace(-0.2, 0.2, n) + 0.01 * seed
    return (base - base.mean() + mean).astype(dtype)


def _checkpoint(dtype, *, raw_layout=False, raw_head=RAW_IN_HEAD, layers=8):
    """Stored checkpoint (``language_model.`` keys, as TextModel.sanitize sees them)."""
    shift = 1.0 if raw_layout else 0.0
    w = {}
    for i in range(layers):
        p = f"language_model.model.layers.{i}."
        w[p + "input_layernorm.weight"] = _vec(TRUNK["input_layernorm"] - shift, dtype, seed=i)
        w[p + "post_attention_layernorm.weight"] = _vec(
            TRUNK["post_attention_layernorm"] - shift, dtype, seed=i + 20
        )
        if i % 4 == 3:
            w[p + "self_attn.q_norm.weight"] = _vec(TRUNK["self_attn.q_norm"] - shift, dtype, seed=i)
            w[p + "self_attn.k_norm.weight"] = _vec(TRUNK["self_attn.k_norm"] - shift, dtype, seed=i)
        else:
            conv = (8, 1, 3) if raw_layout else (8, 3, 1)
            w[p + "linear_attn.conv1d.weight"] = mx.zeros(conv, dtype=dtype)
            # One-centered by construction and never folded.
            w[p + "linear_attn.norm.weight"] = _vec(0.95, dtype, seed=i)
    w["language_model.model.norm.weight"] = _vec(1.96 - shift, dtype, seed=99)
    for name, runtime in HEAD.items():
        stored_raw = raw_layout or name in raw_head
        w[f"language_model.mtp.{name}.weight"] = _vec(
            runtime - (1.0 if stored_raw else 0.0), dtype, seed=len(name)
        )
    return w


def _qwen38_sanitize(weights):
    stub = SimpleNamespace(mtp=object(), args=SimpleNamespace(tie_word_embeddings=False))
    out = qwen38_27b.TextModel.sanitize(stub, weights)
    return out, stub.norm_convention


def _mean(v):
    return float(v.astype(mx.float32).mean().item())


def _runtime_ok(weights):
    for name, runtime in HEAD.items():
        assert _mean(weights[f"language_model.mtp.{name}.weight"]) == pytest.approx(
            runtime, abs=0.02
        ), name
    assert _mean(weights["language_model.model.layers.0.input_layernorm.weight"]) == pytest.approx(
        TRUNK["input_layernorm"], abs=0.02
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32], ids=str)
def test_converted_trunk_raw_mtp_qk_norms_are_repaired_once(dtype):
    weights = _checkpoint(dtype)
    before = dict(weights)
    out, report = _qwen38_sanitize(weights)
    _runtime_ok(out)
    assert sorted(k.split("mtp.")[1] for k in report.repaired_head_keys) == sorted(
        f"{n}.weight" for n in RAW_IN_HEAD
    )
    for key in before:
        if not any(key.endswith(f"mtp.{n}.weight") for n in RAW_IN_HEAD):
            assert out[key] is before[key], key  # trunk and shifted head untouched
    for key in report.applied:
        assert out[key].dtype == dtype
    # Idempotent: a second pass over the repaired weights changes nothing.
    again = dict(out)
    out2, report2 = _qwen38_sanitize(again)
    assert report2.applied == []
    assert all(out2[k] is out[k] for k in out)


def test_all_shifted_checkpoint_is_unchanged():
    weights = _checkpoint(mx.bfloat16, raw_head=())
    before = dict(weights)
    out, report = _qwen38_sanitize(weights)
    assert report.applied == [] and report.repaired_head_keys == []
    assert all(out[k] is before[k] for k in before if "conv1d" not in k)


@pytest.mark.parametrize(
    "name, stored",
    [
        ("layers.0.self_attn.q_norm", 1.06),  # gap +0.40 to the 1.46 trunk family
        ("pre_fc_norm_hidden", 0.10),  # between the unpaired thresholds
    ],
)
def test_ambiguous_head_tensor_raises_and_names_it(name, stored):
    weights = _checkpoint(mx.bfloat16, raw_head=())
    weights[f"language_model.mtp.{name}.weight"] = _vec(stored, mx.bfloat16)
    with pytest.raises(NormConventionError, match=f"mtp.{name}.weight"):
        _qwen38_sanitize(weights)


def test_known_raw_hash_resolves_an_ambiguous_tensor_in_any_dtype():
    # A raw tensor whose statistics are ambiguous (the 35B post_attention case).
    raw = _vec(0.87, mx.bfloat16, seed=7)
    table = (UnshiftedNorm("mtp.layers.0.post_attention_layernorm.weight", tensor_sha256(raw), "test"),)
    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        weights = _checkpoint(dtype, raw_head=())
        weights["language_model.mtp.layers.0.post_attention_layernorm.weight"] = raw.astype(dtype)
        weights["language_model.model.layers.0.post_attention_layernorm.weight"] = _vec(1.31, dtype)
        for i in range(1, 8):
            weights[f"language_model.model.layers.{i}.post_attention_layernorm.weight"] = _vec(1.31, dtype)
        report = resolve_norm_convention(
            weights, fold_suffixes=QWEN38_SUFFIXES, trunk_raw=False, unshifted=table
        )
        assert report.repaired_head_keys == [
            "language_model.mtp.layers.0.post_attention_layernorm.weight"
        ]
        assert any(d.evidence.startswith("sha256 known raw") for d in report.decisions)
        # Without the hash the same tensor is ambiguous and must raise.
        weights["language_model.mtp.layers.0.post_attention_layernorm.weight"] = raw.astype(dtype)
        with pytest.raises(NormConventionError):
            decide_norm_convention(weights, fold_suffixes=QWEN38_SUFFIXES, trunk_raw=False, unshifted=())


def test_raw_hf_layout_shifts_trunk_and_head_exactly_once():
    weights = _checkpoint(mx.bfloat16, raw_layout=True)
    out, report = _qwen38_sanitize(weights)
    assert report.trunk_raw
    _runtime_ok(out)
    assert _mean(out["language_model.model.norm.weight"]) == pytest.approx(1.96, abs=0.02)
    # The GDN norm is never folded; conv1d is moved to the converted layout.
    assert _mean(out["language_model.model.layers.0.linear_attn.norm.weight"]) == pytest.approx(0.95, abs=0.02)
    assert out["language_model.model.layers.0.linear_attn.conv1d.weight"].shape == (8, 3, 1)
    # Re-sanitizing the output sees the converted layout and changes nothing.
    out2, report2 = _qwen38_sanitize(dict(out))
    assert not report2.trunk_raw and report2.applied == []


def test_raw_layout_with_an_already_shifted_head_raises():
    weights = _checkpoint(mx.bfloat16, raw_layout=True)
    weights["language_model.mtp.layers.0.self_attn.q_norm.weight"] = _vec(1.75, mx.bfloat16)
    with pytest.raises(NormConventionError, match="looks already shifted"):
        _qwen38_sanitize(weights)


def test_trunk_decision_is_verified():
    # Converted layout but an unshifted trunk: the pooled trunk check refuses.
    weights = _checkpoint(mx.bfloat16, raw_head=())
    for key in list(weights):
        if not key.startswith("language_model.mtp.") and key.endswith(QWEN38_SUFFIXES):
            weights[key] = weights[key] - 1.0
    with pytest.raises(NormConventionError, match="zero-centered"):
        _qwen38_sanitize(weights)


def test_separately_loaded_head_uses_supplied_trunk_means():
    full = _checkpoint(mx.float16)
    trunk_means = trunk_family_means(full, QWEN38_SUFFIXES, raw=False)
    head = {k: v for k, v in full.items() if k.startswith("language_model.mtp.")}
    report = resolve_norm_convention(
        head, fold_suffixes=QWEN38_SUFFIXES, trunk_raw=False, trunk_means=trunk_means
    )
    assert sorted(k.split("mtp.")[1] for k in report.repaired_head_keys) == sorted(
        f"{n}.weight" for n in RAW_IN_HEAD
    )
    # Without trunk statistics the paired head tensors cannot be decided.
    with pytest.raises(NormConventionError):
        decide_norm_convention(head, fold_suffixes=QWEN38_SUFFIXES, trunk_raw=False)


def test_family_pairs_head_with_trunk():
    assert norm_family("language_model.mtp.layers.0.self_attn.q_norm.weight") == norm_family(
        "language_model.model.layers.11.self_attn.q_norm.weight"
    )
    assert norm_family("mtp.norm.weight") == norm_family("language_model.model.norm.weight")
    assert norm_family("mtp.hyper_connection_mixer.hc_norm.weight") == norm_family(
        "language_model.model.hyper_connection_mixer.hc_norm.weight"
    )


# --- qwen4_exp (Flash-Next) -------------------------------------------------


QWEN4_RAW = ("layers.0.attn_hyper_connection.hc_norm", "layers.0.self_attn.indexer.q_layernorm")


def _qwen4_checkpoint(dtype, *, raw_head=QWEN4_RAW, head_qk=1.60):
    w = {}
    for i in range(12):
        p = f"language_model.model.layers.{i}."
        w[p + "attn_hyper_connection.hc_norm.weight"] = _vec(1.04, dtype, seed=i)
        w[p + "mlp_hyper_connection.hc_norm.weight"] = _vec(1.37, dtype, seed=i + 1)
        w[p + "self_attn.q_norm.weight"] = _vec(1.45, dtype, seed=i)
        w[p + "self_attn.k_norm.weight"] = _vec(1.45, dtype, seed=i + 2)
        w[p + "self_attn.indexer.q_layernorm.weight"] = _vec(0.95, dtype, seed=i + 3)
        w[p + "linear_attn.conv1d.weight"] = mx.zeros((8, 3, 1), dtype=dtype)
    w["language_model.model.hyper_connection_mixer.hc_norm.weight"] = _vec(3.75, dtype)
    head = {
        "hyper_connection_mixer.hc_norm": 4.0,
        "layers.0.attn_hyper_connection.hc_norm": 1.28,
        "layers.0.self_attn.indexer.q_layernorm": 1.10,
        "layers.0.self_attn.q_norm": head_qk,
        "layers.0.self_attn.k_norm": head_qk,
        "pre_fc_norm_embedding": 0.24,
        "pre_fc_norm_hidden": 0.67,
    }
    for name, runtime in head.items():
        w[f"mtp.{name}.weight"] = _vec(runtime - (1.0 if name in raw_head else 0.0), dtype, seed=3)
    return w


def _qwen4_sanitize(weights):
    stub = SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=False),
        NORM_CONVENTION_HINT=qwen4_exp.TextModel.NORM_CONVENTION_HINT,
    )
    out = qwen4_exp.TextModel.sanitize(stub, weights)
    return out, stub.norm_convention


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
def test_qwen4_mtp_only_misfold_is_repaired_not_diluted(dtype, monkeypatch):
    monkeypatch.delenv("MLX_QWEN4_NORM_CONVENTION", raising=False)
    monkeypatch.delenv("MLX_QWEN4_NORM_CONVENTION_UNCHECKED", raising=False)
    out, report = _qwen4_sanitize(_qwen4_checkpoint(dtype))
    assert sorted(report.repaired_head_keys) == [
        "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    ]
    assert _mean(out["mtp.layers.0.attn_hyper_connection.hc_norm.weight"]) == pytest.approx(1.28, abs=0.02)
    out2, report2 = _qwen4_sanitize(dict(out))
    assert report2.applied == []


def test_qwen4_all_shifted_unchanged_and_ambiguous_raises(monkeypatch):
    monkeypatch.delenv("MLX_QWEN4_NORM_CONVENTION", raising=False)
    monkeypatch.delenv("MLX_QWEN4_NORM_CONVENTION_UNCHECKED", raising=False)
    weights = _qwen4_checkpoint(mx.bfloat16, raw_head=())
    before = dict(weights)
    out, report = _qwen4_sanitize(weights)
    assert report.applied == []
    assert all(out[k] is before[k] for k in before if "conv1d" not in k)

    weights = _qwen4_checkpoint(mx.bfloat16, raw_head=())
    weights["mtp.layers.0.self_attn.q_norm.weight"] = _vec(1.05, mx.bfloat16)
    with pytest.raises(NormConventionError, match="q_norm"):
        _qwen4_sanitize(weights)
    # The documented escape hatch keeps the legacy follow-the-trunk behaviour.
    monkeypatch.setenv("MLX_QWEN4_NORM_CONVENTION_UNCHECKED", "1")
    weights = _qwen4_checkpoint(mx.bfloat16, raw_head=())
    weights["mtp.layers.0.self_attn.q_norm.weight"] = _vec(1.05, mx.bfloat16)
    _, report = _qwen4_sanitize(weights)
    assert report.applied == []


def test_qwen4_raw_trunk_without_fold_is_refused(monkeypatch):
    monkeypatch.delenv("MLX_QWEN4_NORM_CONVENTION_UNCHECKED", raising=False)
    monkeypatch.setenv("MLX_QWEN4_NORM_CONVENTION", "converted")
    weights = _qwen4_checkpoint(mx.bfloat16, raw_head=())
    for key in list(weights):
        if key.startswith("language_model.") and "norm" in key:
            weights[key] = weights[key] - 1.0
    with pytest.raises(NormConventionError, match="MLX_QWEN4_NORM_CONVENTION"):
        _qwen4_sanitize(weights)


def test_head_far_above_its_trunk_needs_a_pin(monkeypatch):
    """Flash-Next's head q/k_norm sit +2.2 above the trunk: stats cannot place
    them (an oQ-skipped raw copy would sit +1.2 above), so only a known hash
    decides; the shipped table pins the qualified Flash-Next values."""
    monkeypatch.delenv("MLX_QWEN4_NORM_CONVENTION", raising=False)
    monkeypatch.delenv("MLX_QWEN4_NORM_CONVENTION_UNCHECKED", raising=False)
    for stored in (3.68, 2.68):  # shifted, and an oQ-skipped raw copy
        weights = _qwen4_checkpoint(mx.bfloat16, raw_head=(), head_qk=stored)
        with pytest.raises(NormConventionError, match="too far above"):
            _qwen4_sanitize(weights)
    weights = _qwen4_checkpoint(mx.bfloat16, raw_head=(), head_qk=3.68)
    pins = tuple(
        UnshiftedNorm(k.removeprefix("language_model."), tensor_sha256(weights[k]), "test")
        for k in ("mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight")
    )
    report = decide_norm_convention(
        weights, fold_suffixes=qwen4_exp_suffixes(), trunk_raw=False, shifted=pins
    )
    assert report.raw_keys == []


def qwen4_exp_suffixes():
    return (
        ".hc_norm.weight",
        ".q_layernorm.weight",
        ".k_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        "hyper_connection_mixer.hc_norm.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    )
