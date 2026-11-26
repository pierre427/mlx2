#!/usr/bin/env python3
"""Sizing table for a tile-paged amoeba image (research, CPU only, reads safetensors headers).

Reads the source checkpoint's tensor extents (by category) and a router tile set's per-layer
neuron rows (``L{l}.{tile}.rows``), and writes the bytes of every always-resident part, the core,
and every tile at q8 and q4 (affine, group 64), with and without a tile-own down block padded to
whole 64-neuron groups, plus KV / GDN state at a few contexts. Nothing is loaded onto a device.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import struct
import subprocess
from pathlib import Path



def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def category(k):
    if "visual" in k or "vision" in k:
        return "vision"
    if "mtp" in k:
        return "mtp"
    if ".mlp." in k:
        return "mlp"
    if "linear_attn" in k:
        return "gdn"
    if "self_attn" in k:
        return "attention"
    if "embed_tokens" in k:
        return "embedding"
    if "lm_head" in k:
        return "head"
    return "norms"


def bytes_per_neuron(d, bits, group=64, scale_bytes=2, bias_bytes=2, mats=3):
    """gate row + up row + down column of one neuron, affine quantized along the input dim.
    For gate/up a row's groups run along d; for the down column the group runs over 64 neurons, so
    per neuron it is d weights plus d/64 (scale, bias) shared by 64 neurons."""
    w = d * bits / 8
    gate_up = 2 * (w + d / group * (scale_bytes + bias_bytes))
    down = w + d * (scale_bytes + bias_bytes) / group
    return gate_up + down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tiles", required=True, help="router tile set dir (manifest.json + tiles.safetensors)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = Path(args.model).expanduser()
    cfg = json.loads((model / "config.json").read_text())
    t = cfg.get("text_config", cfg)
    d, N, L = t["hidden_size"], t["intermediate_size"], t["num_hidden_layers"]
    cats = collections.Counter()
    for f in sorted(model.glob("*.safetensors")):
        h, _ = header(f)
        for k, v in h.items():
            if k != "__metadata__":
                cats[category(k)] += v["data_offsets"][1] - v["data_offsets"][0]
    man = json.loads((Path(args.tiles) / "manifest.json").read_text())
    th, _ = header(Path(args.tiles) / "tiles.safetensors")
    ids = [man["core"]["id"]] + [b["id"] for b in man["boxes"]]
    per_layer = {i: [0] * L for i in ids}
    for l in range(L):
        for i in ids:
            k = f"L{l}.{i}.rows"
            if k in th:
                per_layer[i][l] = int(th[k]["shape"][0])
    total = sum(sum(v) for v in per_layer.values())
    bpn = {b: bytes_per_neuron(d, b) for b in (8, 4)}
    rows = []
    for i in ids:
        n = sum(per_layer[i])
        padded = sum(math.ceil(c / 64) * 64 for c in per_layer[i])
        rec = {"tile": i, "neurons": n, "frac_of_mlp": n / (N * L),
               "layers_present": sum(1 for c in per_layer[i] if c), "max_per_layer": max(per_layer[i]),
               "padded_neurons_own_down": padded}
        for b in (8, 4):
            rec[f"q{b}_bytes"] = n * bpn[b]
            rec[f"q{b}_bytes_own_down_padded"] = n * (bpn[b] - (d * b / 8 + d * 4 / 64)) + padded * (d * b / 8 + d * 4 / 64)
        rows.append(rec)
    # KV: full-attention layers only (GDN layers keep a fixed state)
    lt = t.get("layer_types") or []
    full = sum(1 for x in lt if x == "full_attention") or L
    kv_per_token = 2 * t["num_key_value_heads"] * t["head_dim"] * 2 * full      # fp16 K and V
    gdn_layers = sum(1 for x in lt if x == "linear_attention")
    gdn_state = gdn_layers * (t["linear_num_value_heads"] * t["linear_key_head_dim"] * t["linear_value_head_dim"] * 4
                              + (t["linear_num_key_heads"] * t["linear_key_head_dim"] * 2
                                 + t["linear_num_value_heads"] * t["linear_value_head_dim"]) * (t["linear_conv_kernel_dim"] - 1) * 2)
    kv = {str(c): {"kv_fp16": kv_per_token * c, "kv_q8": kv_per_token * c * (1 + 4 / 64) / 2,
                   "gdn_state": gdn_state} for c in (4096, 32768, 131072, 262144)}
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parent).stdout.strip()
    out = {"schema": "mlx2.amoeba-tile-paging-sizing.v1", "mlx2_commit": rev,
           "checkpoint": f"{model.name}@{man['source']['fingerprint'][:8]}",
           "tile_set": f"{man['name']}@{man['files']['tiles.safetensors'][:8]}",
           "geometry": {"d_model": d, "intermediate": N, "layers": L, "full_attention_layers": full,
                        "gdn_layers": gdn_layers, "neurons_total": N * L, "neurons_in_tiles": total},
           "checkpoint_bytes_by_category": dict(cats), "bytes_per_neuron": bpn,
           "tiles": rows, "kv_and_state": kv, "kv_fp16_bytes_per_token": kv_per_token,
           "status": "research, unqualified"}
    Path(args.out).write_text(json.dumps(out, indent=1))
    G = 1e9
    print(f"checkpoint {out['checkpoint']}  tiles {out['tile_set']}  commit {rev}")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {k:10s} {v / G:7.3f} GB")
    print(f"  total      {sum(cats.values()) / G:7.3f} GB; MLP neurons in tiles {total} / {N * L}")
    for r in sorted(rows, key=lambda r: -r["neurons"]):
        print(f"  {r['tile']:34s} {r['neurons']:7d} {r['frac_of_mlp']:6.3f}  q8 {r['q8_bytes'] / G:6.3f}  q4 {r['q4_bytes'] / G:6.3f}"
              f"  q8own {r['q8_bytes_own_down_padded'] / G:6.3f} q4own {r['q4_bytes_own_down_padded'] / G:6.3f}  layers {r['layers_present']}")
    for c, v in kv.items():
        print(f"  ctx {c:>6s}: KV fp16 {v['kv_fp16'] / G:6.2f} GB  q8 {v['kv_q8'] / G:6.2f}  GDN state {v['gdn_state'] / G:.3f}")


if __name__ == "__main__":
    main()
