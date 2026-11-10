"""Shared constants, provenance and atomic I/O for the DS4.1 Source expert-prefetch scheduler study.

CPU-only research code. No runtime/adapter changes; reads the routing capture read-only.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # scripts/research/ds41_atlas (cache_sim.read_capture)

CHECKPOINT = ("DeepSeek-V4.1-Flash-1017-Source@"
              "c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb")
BINDING = "c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb"
MLX2_COMMIT = "c18fef602 + uncommitted research scripts (scripts/research/ds41_atlas/sched/)"
CAPTURE = os.path.expanduser("~/mlx-models/.ds41-routing-capture-20261108")
LITE = "/Volumes/T7/ds41-atlas/lite"
OUT = "~/Desktop/mlx2/output/ds41-source-20261108/cache-design/sched"
LAYERS, EXPERTS, TOPK, HIDDEN = 40, 384, 6, 5120
EXPERT_BYTES = 18_800_640


def provenance(**params):
    return {"checkpoint": CHECKPOINT, "mlx2_commit": MLX2_COMMIT, "capture": CAPTURE,
            "cpu_only": True, "params": params}


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_trace():
    return dict(np.load(f"{OUT}/trace.npz", allow_pickle=False))


def softplus(z):
    return np.logaddexp(0.0, z)


def load_gate(L):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    lite = mx.load(f"{LITE}/layer_{L:02d}.safetensors")
    gate = np.array(lite["gate.weight"].astype(mx.float32))
    bias = np.array(lite["gate.bias"].astype(mx.float32))
    return gate, bias
