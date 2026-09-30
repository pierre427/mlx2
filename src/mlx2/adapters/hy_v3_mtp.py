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


