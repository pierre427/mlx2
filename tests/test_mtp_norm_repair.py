"""MTP norm shift repair (omlx#3750 / MTPLX#511 class of bug)."""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.adapters.norm_repair import (
    QWEN36_35B_UNSHIFTED_MTP_NORMS,
    UnshiftedNorm,
    norm_means,
    repair_unshifted_norms,
    tensor_sha256,
)

ARTIFACT = Path.home() / (
    "mlx-models/"
    "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp"
)


def test_repair_is_byte_exact_and_idempotent():
    raw = mx.array([0.75, 0.8, 0.9], dtype=mx.bfloat16)
    other = mx.array([0.75, 0.8, 0.91], dtype=mx.bfloat16)
    table = (UnshiftedNorm("mtp.norm.weight", tensor_sha256(raw), "test"),)
    weights = {
        "language_model.mtp.norm.weight": raw,
        "language_model.mtp.layers.0.self_attn.q_norm.weight": other,
    }

    assert repair_unshifted_norms(weights, table) == ["language_model.mtp.norm.weight"]
    assert mx.array_equal(weights["language_model.mtp.norm.weight"], raw + 1.0)
    assert weights["language_model.mtp.norm.weight"].dtype == mx.bfloat16
    # Near-identical content under the same key is not touched.
    weights["language_model.mtp.norm.weight"] = other
    assert repair_unshifted_norms(weights, table) == []
    # A repaired tensor no longer matches, so a second pass cannot double-shift.
    weights["language_model.mtp.norm.weight"] = raw + 1.0
    assert repair_unshifted_norms(weights, table) == []


def test_norm_means_scopes_to_prefix():
    weights = {
        "language_model.mtp.norm.weight": mx.full((4,), 2.0),
        "language_model.model.norm.weight": mx.full((4,), 5.0),
        "language_model.mtp.fc.weight": mx.ones((2, 2)),
    }
    assert norm_means(weights, "mtp.") == {"language_model.mtp.norm.weight": 2.0}


@pytest.mark.skipif(not ARTIFACT.is_dir(), reason="35B oQ4e-mtp artifact not present")
def test_qwen36_35b_oq_artifact_mtp_norms_are_repaired():
    index = json.loads((ARTIFACT / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [k for k in index if k.startswith("language_model.mtp.") and "norm" in k]
    weights = {}
    for shard in sorted({index[k] for k in keys}):
        loaded = mx.load(str(ARTIFACT / shard))
        weights.update({k: loaded[k] for k in keys if index[k] == shard})
    assert len(weights) == 7

    repaired = repair_unshifted_norms(weights)
    assert sorted(k.removeprefix("language_model.") for k in repaired) == sorted(
        e.key for e in QWEN36_35B_UNSHIFTED_MTP_NORMS
    )
    # Runtime means are raw-HF + 1 for all seven (raw values measured against
    # Qwen/Qwen3.6-35B-A3B@995ad96).
    expected = {
        "mtp.layers.0.input_layernorm.weight": 0.9049,
        "mtp.layers.0.post_attention_layernorm.weight": 1.8686,
        "mtp.layers.0.self_attn.q_norm.weight": 1.7672,
        "mtp.layers.0.self_attn.k_norm.weight": 1.7418,
        "mtp.norm.weight": 2.9251,
        "mtp.pre_fc_norm_embedding.weight": 0.2734,
        "mtp.pre_fc_norm_hidden.weight": 0.4937,
    }
    means = {k.removeprefix("language_model."): v for k, v in norm_means(weights, "mtp.").items()}
    assert means == pytest.approx(expected, abs=2e-3)
    assert repair_unshifted_norms(weights) == []
