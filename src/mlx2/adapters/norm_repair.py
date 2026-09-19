"""Byte-exact repair of RMSNorm weights a converter left unshifted.

Qwen3.5/3.6-family checkpoints store RMSNorm gammas as ``w`` for a runtime
scale of ``1 + w``; MLX runtimes expect the shifted value. Some converters
decide the shift per tensor from its mean (oQ's ``add_if_mean_lt_0_5``), which
is ambiguous: a raw gamma whose mean is already >= 0.5 is left unshifted. On
Qwen3.6-35B-A3B that skips four of the seven MTP-head norms, and draft
acceptance drops to ~0.34 without any load error (cf. omlx#3750, MTPLX#511).

No rule on the stored values alone can tell a shifted gamma from an unshifted
one in that band, so the repair is keyed on content: a tensor is shifted only
when its stored bytes are exactly the known-unshifted bytes. A second load of
an already-repaired artifact, or any other artifact, is untouched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class UnshiftedNorm:
    key: str  # without any "language_model." prefix
    sha256: str  # of the stored little-endian element bytes
    reference: str


_QWEN36_35B_RAW = "Qwen/Qwen3.6-35B-A3B@995ad96 model-00026-of-00026.safetensors (raw HF)"

# Stored bytes equal the raw official tensor bit for bit (verified
# 2026-09-19 against the HF shard above) in
# Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp.
QWEN36_35B_UNSHIFTED_MTP_NORMS = (
    UnshiftedNorm(
        "mtp.layers.0.post_attention_layernorm.weight",
        "03f1d8971f0e2378806ee9269f4ef91115d5d97e2ff6460fe4af3ab9ff1fd12a",
        _QWEN36_35B_RAW,
    ),
    UnshiftedNorm(
        "mtp.layers.0.self_attn.q_norm.weight",
        "c57bdab3a6844454085b1df4991bc46443ae2495c4a620ec22e0d1b93c27fbba",
        _QWEN36_35B_RAW,
    ),
    UnshiftedNorm(
        "mtp.layers.0.self_attn.k_norm.weight",
        "6fb1be31f55e2efd79dd2d737a4a801c69ee28534e4465a6d7b39f9754087938",
        _QWEN36_35B_RAW,
    ),
    UnshiftedNorm(
        "mtp.norm.weight",
        "b56452e89857beaee0724b4f9c1341295754c015ba6e671cb929320f6ecd5271",
        _QWEN36_35B_RAW,
    ),
)


def tensor_sha256(value) -> str:
    import mlx.core as mx
    import numpy as np

    if value.dtype == mx.bfloat16:
        value = value.view(mx.uint16)
    return hashlib.sha256(np.array(value).tobytes()).hexdigest()


def repair_unshifted_norms(weights: dict, table=QWEN36_35B_UNSHIFTED_MTP_NORMS) -> list[str]:
    """Add 1.0 in place to every tensor matching ``table`` byte for byte.

    ``weights`` may carry a ``language_model.`` prefix. Returns the repaired
    keys (as named in ``weights``) so the adapter can record them.
    """
    wanted = {entry.key: entry.sha256 for entry in table}
    repaired = []
    for name in sorted(weights):
        digest = wanted.get(name.removeprefix("language_model."))
        value = weights[name]
        if digest is None or value.ndim != 1:
            continue
        if tensor_sha256(value) == digest:
            weights[name] = value + 1.0
            repaired.append(name)
    return repaired


def norm_means(weights: dict, prefix: str) -> dict[str, float]:
    """Mean runtime gamma of every 1-D ``*norm*`` tensor under ``prefix``."""
    import mlx.core as mx

    return {
        name: round(float(value.astype(mx.float32).mean().item()), 4)
        for name, value in sorted(weights.items())
        if name.removeprefix("language_model.").startswith(prefix)
        and "norm" in name
        and value.ndim == 1
    }
