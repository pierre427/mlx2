#!/usr/bin/env python3
"""CPU parity of mlx2's Laguna forward on poolside's per-expert (vendor) layout.

Two modes, both CPU float32:

``--tiny``  random tensors in the exact key layout of poolside/Laguna-XS-2.1
            bf16 (``mlp.experts.<e>.<proj>.weight``, ``mlp.gate.weight``,
            ``mlp.experts.e_score_correction_bias``) at reduced dimensions.
``--real``  the first ``--layers`` decoder layers of a real checkpoint, plus
            embeddings, final norm and lm_head, read tensor by tensor from
            the shards.  A 2-layer XS slice holds ~5 GB of float32 tensors, but
            the numpy, torch and mlx copies plus both expert stacks peak near
            15-18 GB of host memory.

The reference is transformers' in-tree ``LagunaForCausalLM`` (eager
attention); the checkpoint's own remote ``configuration_laguna.py`` does not
validate its per-layer ``rope_parameters`` under transformers 5.17.  Vendor
tensors are converted to its fused ``experts.gate_up_proj`` /
``experts.down_proj`` / ``shared_experts`` parameters here and loaded with
``strict=True``; mlx2 receives the vendor tensors unchanged and goes
through ``Model.sanitize``.  Prints max |logit diff| and top-1 agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def tiny_config(real: dict) -> dict:
    layers = 4
    config = dict(real)
    config.update(
        vocab_size=512, hidden_size=64, intermediate_size=128,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        sliding_window=8,
        layer_types=["full_attention" if i % 4 == 0 else "sliding_attention" for i in range(layers)],
        mlp_layer_types=["dense", *("sparse" for _ in range(layers - 1))],
        gating_types=["per_head"] * layers,
        num_attention_heads_per_layer=[4 if i % 4 == 0 else 8 for i in range(layers)],
    )
    return config


def tiny_weights(config: dict, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, hd, kv = config["hidden_size"], config["head_dim"], config["num_key_value_heads"]

    def w(*shape, scale=None):
        scale = scale if scale is not None else 1.0 / np.sqrt(shape[-1])
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    out = {
        "model.embed_tokens.weight": w(config["vocab_size"], d, scale=1.0),
        "model.norm.weight": 1.0 + w(d, scale=0.1),
        "lm_head.weight": w(config["vocab_size"], d),
    }
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}."
        heads = config["num_attention_heads_per_layer"][i]
        out.update({
            p + "input_layernorm.weight": 1.0 + w(d, scale=0.1),
            p + "post_attention_layernorm.weight": 1.0 + w(d, scale=0.1),
            p + "self_attn.q_proj.weight": w(heads * hd, d),
            p + "self_attn.k_proj.weight": w(kv * hd, d),
            p + "self_attn.v_proj.weight": w(kv * hd, d),
            p + "self_attn.o_proj.weight": w(d, heads * hd),
            p + "self_attn.g_proj.weight": w(heads, d),
            p + "self_attn.q_norm.weight": 1.0 + w(hd, scale=0.1),
            p + "self_attn.k_norm.weight": 1.0 + w(hd, scale=0.1),
        })
        if config["mlp_layer_types"][i] == "dense":
            m = config["intermediate_size"]
            out.update({p + "mlp.gate_proj.weight": w(m, d), p + "mlp.up_proj.weight": w(m, d),
                        p + "mlp.down_proj.weight": w(d, m)})
            continue
        m, s, e = config["moe_intermediate_size"], config["shared_expert_intermediate_size"], config["num_experts"]
        out[p + "mlp.gate.weight"] = w(e, d, scale=1.0)
        out[p + "mlp.experts.e_score_correction_bias"] = w(e, scale=0.05)
        out.update({p + "mlp.shared_expert.gate_proj.weight": w(s, d),
                    p + "mlp.shared_expert.up_proj.weight": w(s, d),
                    p + "mlp.shared_expert.down_proj.weight": w(d, s)})
        for x in range(e):
            out.update({p + f"mlp.experts.{x}.gate_proj.weight": w(m, d),
                        p + f"mlp.experts.{x}.up_proj.weight": w(m, d),
                        p + f"mlp.experts.{x}.down_proj.weight": w(d, m)})
    return out


def real_slice(model_dir: Path, layers: int) -> tuple[dict, dict[str, np.ndarray]]:
    import torch
    from safetensors import safe_open

    config = json.loads((model_dir / "config.json").read_text())
    for key in ("layer_types", "mlp_layer_types", "gating_types", "num_attention_heads_per_layer"):
        if key in config:
            config[key] = config[key][:layers]
    config["num_hidden_layers"] = layers
    config["mlp_only_layers"] = [i for i in config.get("mlp_only_layers", []) if i < layers]
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {k: v for k, v in weight_map.items()
              if not k.startswith("model.layers.") or int(k.split(".")[2]) < layers}
    out: dict[str, np.ndarray] = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(model_dir / shard, framework="pt") as handle:
            for key in (k for k, v in wanted.items() if v == shard):
                out[key] = handle.get_tensor(key).to(torch.float32).numpy()
    return config, out


def to_reference_state(config: dict, weights: dict[str, np.ndarray]):
    """Vendor checkpoint keys -> the remote modeling file's parameters."""
    import torch

    state, used = {}, set()
    for i in range(config["num_hidden_layers"]):
        if config["mlp_layer_types"][i] != "sparse":
            continue
        p = f"model.layers.{i}.mlp."
        gate_up, down = [], []
        for x in range(config["num_experts"]):
            names = [p + f"experts.{x}.{proj}.weight" for proj in PROJECTIONS]
            used.update(names)
            gate_up.append(np.concatenate([weights[names[0]], weights[names[1]]], axis=0))
            down.append(weights[names[2]])
        state[p + "experts.gate_up_proj"] = torch.from_numpy(np.stack(gate_up))
        state[p + "experts.down_proj"] = torch.from_numpy(np.stack(down))
        bias = p + "experts.e_score_correction_bias"
        state[p + "gate.e_score_correction_bias"] = torch.from_numpy(weights[bias].copy())
        used.add(bias)
    for key, value in weights.items():
        if key not in used:
            state[key.replace(".mlp.shared_expert.", ".mlp.shared_experts.")] = torch.from_numpy(value.copy())
    return state


def reference_logits(config: dict, weights, sequences):
    import torch
    from transformers import LagunaConfig, LagunaForCausalLM

    fields = {k: v for k, v in config.items() if k not in ("architectures", "auto_map", "model_type")}
    hf_config = LagunaConfig(**fields)
    hf_config._attn_implementation = "eager"
    model = LagunaForCausalLM(hf_config).to(torch.float32)
    model.load_state_dict(to_reference_state(config, weights), strict=True)
    model.eval()
    with torch.no_grad():
        return [model(torch.tensor([seq])).logits[0].float().numpy() for seq in sequences]


def mlx2_logits(config: dict, weights, sequences):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx2.runtime.models.laguna import Model, ModelArgs

    model = Model(ModelArgs.from_dict(config))
    sanitized = model.sanitize({k: mx.array(v) for k, v in weights.items()})
    model.load_weights(list(sanitized.items()), strict=True)
    model.eval()
    out = []
    for seq in sequences:
        logits = model(mx.array([seq]), cache=model.make_cache())
        out.append(np.array(logits[0].astype(mx.float32)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path.home() / "Models" / "Laguna-XS-2.1")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tiny", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    real_config = json.loads((args.model / "config.json").read_text())
    if args.tiny:
        config = tiny_config(real_config)
        weights = tiny_weights(config, args.seed)
    else:
        config, weights = real_slice(args.model, args.layers)
    rng = np.random.default_rng(args.seed + 1)
    vocab = config["vocab_size"]
    sequences = [[int(t) for t in rng.integers(0, vocab, n)] for n in (1, 5, 13, 24)]
    ref = reference_logits(config, weights, sequences)
    got = mlx2_logits(config, weights, sequences)
    report = {"mode": "tiny" if args.tiny else f"real-{args.layers}-layers", "cells": []}
    for seq, a, b in zip(sequences, ref, got):
        report["cells"].append({
            "length": len(seq),
            "max_abs_logit_diff": float(np.max(np.abs(a - b))),
            "logit_scale": float(np.max(np.abs(a))),
            "top1_agreement": float(np.mean(a.argmax(-1) == b.argmax(-1))),
        })
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
