"""Dtype gate for TensorFold tree tiles on Qwen3.5/3.6/3.8 hybrids.

The vendored tree kernels (``runtime/tensorfold_qwen38``) reproduce the
served bf16 arithmetic: the GDN conv-state buffers are typed ``bfloat16_t``
and the glue rounds through bf16.  An artifact whose norm, conv or gate
tensors are stored in another float dtype (GGUF conversions keep them in
F32; some exports are all F16) promotes the residual stream away from bf16,
so its tiles would neither compile nor match ordinary decoding.  Such a
target keeps its chain route.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

_TAILS = (
    "conv1d.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "dt_bias",
)
_SKIP = ("mtp.", "vision", "visual", "audio")


def _relevant(name: str) -> bool:
    return name.endswith(_TAILS) and not any(part in name for part in _SKIP)


def _issue(found: dict[str, str]) -> str | None:
    other = sorted({dtype for dtype in found.values() if dtype not in ("BF16", "bfloat16")})
    if not found or not other:
        return None
    example = next(name for name, dtype in found.items() if dtype in other)
    return f"tree tiles need bf16 norm/conv tensors; artifact stores {'/'.join(other)} (e.g. {example})"


def artifact_tree_tile_dtype_issue(model_path) -> str | None:
    """Read safetensors headers only; None when every relevant tensor is bf16."""
    found: dict[str, str] = {}
    for shard in sorted(Path(model_path).glob("*.safetensors")):
        try:
            with open(shard, "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(size))
        except (OSError, ValueError, struct.error):
            continue
        for name, entry in header.items():
            if name != "__metadata__" and _relevant(name):
                found[name] = str(entry.get("dtype"))
    return _issue(found)


def model_tree_tile_dtype_issue(model) -> str | None:
    """The same check on a loaded model's parameters."""
    from mlx.utils import tree_flatten

    if not callable(getattr(model, "parameters", None)):
        return None                       # not an MLX module (test doubles)
    found = {
        name: "BF16" if str(value.dtype) == "mlx.core.bfloat16" else str(value.dtype)
        for name, value in tree_flatten(model.parameters())
        if _relevant(name)
    }
    return _issue(found)
