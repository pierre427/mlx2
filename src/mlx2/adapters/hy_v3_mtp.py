"""Explicit, unqualified HY V3 embedded MTP candidate.

Ordinary serving never loads or selects this sidecar. The loader is for an
offline parity gate with a bound HY V3 target; it does not advertise MTP.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from .hy_v3 import inspect_artifact


MTP_REQUIRED = frozenset({
    "mtp.eh_proj.weight", "mtp.enorm.weight", "mtp.hnorm.weight",
    "mtp.final_layernorm.weight", "mtp.layer.self_attn.q_proj.weight",
    "mtp.layer.self_attn.k_proj.weight", "mtp.layer.self_attn.v_proj.weight",
    "mtp.layer.self_attn.o_proj.weight", "mtp.layer.mlp.router.gate.weight",
    "mtp.layer.mlp.switch_mlp.gate_proj.weight",
    "mtp.layer.mlp.switch_mlp.up_proj.weight",
    "mtp.layer.mlp.switch_mlp.down_proj.weight",
})


def inspect_mtp_candidate(model_path: str | Path) -> dict:
    artifact = inspect_artifact(model_path)
    config = artifact["config"]
    if config.get("num_nextn_predict_layers") != 1:
        raise ValueError("HY V3 candidate requires exactly one embedded MTP layer")
    if not MTP_REQUIRED <= artifact["weight_map"].keys():
        raise ValueError("HY V3 embedded MTP tensors are incomplete")
    if artifact["reap"] and config.get("mtp_num_experts") != 192:
        raise ValueError("REAP MTP must retain 192 experts")
    path = Path(artifact["identity"]["path"])
    shapes = {
        "mtp.eh_proj.weight": [4096, 1024],
        "mtp.enorm.weight": [4096],
        "mtp.layer.mlp.router.gate.weight": [192, 1024],
        "mtp.layer.mlp.switch_mlp.gate_proj.weight": [192, 1536, 512],
    }
    headers = {}
    for name, expected in shapes.items():
        shard = artifact["weight_map"][name]
        if shard not in headers:
            with (path / shard).open("rb") as stream:
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError("truncated MTP safetensors shard")
                length = struct.unpack("<Q", raw)[0]
                if not 0 < length <= 64 << 20:
                    raise ValueError("invalid MTP safetensors header")
                headers[shard] = json.loads(stream.read(length))
        if headers[shard].get(name, {}).get("shape") != expected:
            raise ValueError(f"MTP tensor shape mismatch: {name}")
    return {"identity": artifact["identity"], "config": config,
            "sidecar_tensors": artifact["mtp_tensor_count"],
            "qualified": False, "selected": False}


def load_mtp_candidate(target_adapter):
    """Attach a strict depth-1 sidecar to an already loaded ordinary target.

    Only a caller performing offline verification should invoke this. The
    serving resolver never reaches it and the target descriptor stays ordinary.
    """
    record = inspect_mtp_candidate(target_adapter.identity["path"])
    if record["identity"]["fingerprint"] != target_adapter.identity["fingerprint"]:
        raise ValueError("MTP sidecar and HY V3 target identities differ")
    import mlx.core as mx
    from mlx import nn
    from ..runtime.models.hy_v3 import HYV3MTP, ModelArgs
    from ..runtime.ubc_evict import load_shards_evicting

    path = Path(record["identity"]["path"])
    weight_map = inspect_artifact(path)["weight_map"]
    files = [path / name for name in sorted({weight_map[k] for k in weight_map if k.startswith("mtp.")})]
    weights = {key.removeprefix("mtp."): value for key, value in
               load_shards_evicting(files).items() if key.startswith("mtp.")}
    sidecar = HYV3MTP(ModelArgs.from_dict(record["config"]))
    quant = record["config"]["quantization"]
    nn.quantize(sidecar, group_size=quant["group_size"], bits=quant["bits"],
                mode=quant["mode"], class_predicate=lambda name, module:
                hasattr(module, "to_quantized") and f"{name}.scales" in weights)
    sidecar.load_weights(list(weights.items()), strict=True)
    sidecar.eval()
    mx.eval(sidecar.parameters())
    target_adapter.model.mtp = sidecar
    weights.clear()
    mx.clear_cache()
    return {**record, "loaded": True, "qualified": False, "selected": False}
