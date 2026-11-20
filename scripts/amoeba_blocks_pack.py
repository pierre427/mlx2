#!/usr/bin/env python3
"""Offline aMoEba block packer (CPU only): block tables for every layer, packed gate/up for some.

From an aMoEba partition (``partition/tiles.npz`` + ``partition.json``: exclusive per-tile neuron
masks, tile 0 the core), write

  OUT/manifest.json, OUT/blocks.npz     every layer's block tables and the tile graph
  OUT/packed-L<l>.safetensors           (--weights-layers) gate/up of layer l in packed order,
                                        checked to unpack to the source bit for bit

The packed gate/up are the source arrays with rows permuted (same names, dtypes and format); down
is not rewritten (it keeps its original layout).  Runs on the CPU stream; never the GPU.

  scripts/amoeba_blocks_pack.py --model fn --out DIR [--weights-layers 3,23,43] [--chunk 32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    import amoeba_blocks_bench as B
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="fn", choices=["fn"])
    ap.add_argument("--partition", default=str(B.FN_PARTITION))
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--weights-layers", default="", help="comma list of layers whose packed gate/up to write")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing nonempty {out}")
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx2.runtime.amoeba_blocks import layout as L
    from mlx2.runtime.amoeba_blocks import ops as O
    part = Path(a.partition)
    meta = json.loads((part / "partition.json").read_text())
    want = B.MODELS[a.model]["checkpoint"]
    if not want.endswith("@" + str(meta.get("fingerprint", ""))[:8]):
        sys.exit(f"partition fingerprint {meta.get('fingerprint')} is not {want}")
    z = np.load(part / "tiles.npz")
    names = ["core"] + [k for k in z.keys() if k.startswith(("box:", "pair:"))]
    graph = L.graph_from_partition_names(names)
    decl = B.declaration(a.model)
    layers = {}
    for i, layer in enumerate(z["layers"]):
        own = L.owner_from_masks({n: z[n][i] for n in names}, names)
        experts = 512 if decl["kind"] == "switch_glu" else 1
        lb = L.plan_layer(own.reshape(experts, -1), tiles=len(names), group_size=64, chunk=a.chunk)
        lb.validate()
        layers[int(layer)] = lb
    manifest = L.save_tables(out, layers, graph, source={
        "checkpoint": want, "partition": str(part), "partition_checkpoint": meta.get("checkpoint"),
        "partition_rule": meta.get("rule"), "declaration": decl})
    print(f"tables: {len(layers)} layers, {len(names)} tiles -> {out}")
    again, _, _ = L.load_tables(out)
    assert all(again[l].digest() == layers[l].digest() for l in layers)
    for layer in [int(v) for v in a.weights_layers.split(",") if v]:
        q = B.load_layer(a.model, layer)
        lb = layers[layer]
        tensors = {}
        prefix = decl["prefix"].format(layer=layer)
        for proj in ("gate_proj", "up_proj"):
            pk = O.pack_qweight(q[proj], lb)
            back = O.unpack_qweight(pk, lb)
            for x0, x1 in ((q[proj].weight, back.weight), (q[proj].scales, back.scales),
                           (q[proj].biases, back.biases)):
                if not mx.array_equal(x0, x1).item():
                    sys.exit(f"L{layer} {proj}: packed rows do not unpack to the source")
            squeeze = decl["kind"] != "switch_glu"
            for part_name, arr in (("weight", pk.weight), ("scales", pk.scales), ("biases", pk.biases)):
                tensors[f"{prefix}.{proj}.{part_name}"] = arr[0] if squeeze else arr
        mx.save_safetensors(str(out / f"packed-L{layer}.safetensors"), tensors,
                            metadata={"amoeba_blocks_layer_digest": lb.digest(),
                                      "packing_law": L.PACKING_LAW, "source": want})
        print(f"L{layer}: packed gate/up written and checked")
    print(json.dumps({"layers": len(manifest["layers"]), "graph_digest": manifest["graph_digest"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
