#!/usr/bin/env python3
"""Lite inputs for the DS4.1 Source expert token atlas: the BF16 embedding and, per layer, ffn_norm + router.

Reads exact tensor extents from the sealed official Source artifact (checked_file, stamp-bound).
usage: atlas_prep.py MODEL OUT_LITE
"""
import os, sys
import numpy as np
import mlx.core as mx
from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, checked_file

DT = {"BF16": (np.uint16, mx.bfloat16), "F32": (np.float32, None), "F8_E8M0": (np.uint8, None),
      "I8": (np.uint8, None), "F8_E4M3": (np.uint8, None)}


def read(art, name):
    t = art["tensors"][name]
    path = checked_file(art["root"], t["file"], art["seals"][t["file"]])
    with open(path, "rb") as f:
        f.seek(t["offset"]); raw = f.read(t["size"])
    np_dt, mx_dt = DT[t["dtype"]]
    a = mx.array(np.frombuffer(raw, np_dt).reshape(t["shape"]))
    return a.view(mx_dt) if mx_dt is not None else a


def main():
    model, out = sys.argv[1:3]
    art = inspect_artifact(model)
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{out}/embed.safetensors"):
        mx.save_safetensors(f"{out}/embed.safetensors", {"embed.weight": read(art, "embed.weight")})
    for L in range(art["args"].num_hidden_layers):
        p = f"{out}/layer_{L:02d}.safetensors"
        if os.path.exists(p):
            continue
        s = f"layers.{L}."
        mx.save_safetensors(p, {"ffn_norm.weight": read(art, s + "ffn_norm.weight"),
                                "gate.weight": read(art, s + "ffn.gate.weight"),
                                "gate.bias": read(art, s + "ffn.gate.bias")})
    print("prep done", len(os.listdir(out)), "fingerprint", art["identity"]["fingerprint"])


if __name__ == "__main__":
    main()
