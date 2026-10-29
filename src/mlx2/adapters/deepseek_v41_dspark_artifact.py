# SPDX-License-Identifier: MIT
"""Bounded admission for the draft embedded in the sealed official source."""

import hashlib
import json
import math
from dataclasses import dataclass, replace

REVISION = "mlx2.v41.embedded-dspark-source-greedy.v1"
WORKSPACE_BOUND_BYTES = 64 << 20


def dspark_workspace_bound(config):
    """Bound live B1/block5 scratch and explicit FP32 coefficient views.

    The Markov head alone exceeds64MiB at the official vocabulary/rank. The
    grouped wo_a projections are stored expanded BF16 but consumed as FP32;
    serial lazy stages may retain all three casts before their result is read.
    Embedding tables are indexed before conversion, so their complete FP32
    expansion is neither created nor reserved.
    """
    specs = tensor_specs(config)
    casts = sum(4 * math.prod(shape) for name, (dtype, shape) in specs.items()
                if (dtype == 'BF16' and name != 'mtp.2.markov_head.embed.weight')
                or name.endswith('attn.wo_a.weight'))
    return WORKSPACE_BOUND_BYTES + casts


@dataclass(frozen=True)
class DSparkConfig:
    stages: int = 3
    block_size: int = 5
    markov_rank: int = 256
    noise_token_id: int = 128799
    target_layers: tuple = (37, 38, 39)

    @classmethod
    def from_artifact(cls, artifact):
        t = artifact["config"]["text_config"]
        config = cls(
            t["num_nextn_predict_layers"],
            t["dspark_block_size"],
            t["dspark_markov_rank"],
            t["dspark_noise_token_id"],
            tuple(t["dspark_target_layer_ids"]),
        )
        if (
            config != cls()
            or t["dspark_n_routed_experts"] != 128
            or t["dspark_num_experts_per_tok"] != 3
        ):
            raise ValueError("unsupported official V4.1 DSpark geometry")
        return config


def draft_config(target):
    return replace(
        target,
        num_hidden_layers=3,
        n_routed_experts=128,
        num_experts_per_tok=3,
        compress_ratios=(0, 0, 0),
        kv_source_layer_ids=(),
        index_source_layer_ids=(),
        candidate_source_layer_id=-1,
        engram_layer_ids=(),
        engram_num_embeddings=(),
    )


def aliases():
    from ..runtime.models.deepseek_v41_config import DeepSeekV41Config
    from .deepseek_v41_source_artifact import aliases as target_aliases

    c = draft_config(DeepSeekV41Config())
    mapping = target_aliases(c)
    result = {
        k: v.replace("layers.", "mtp.", 1)
        for k, v in mapping.items()
        if k.startswith("blk.")
    }
    result.update(
        {
            "main_proj.weight": "mtp.0.main_proj.weight",
            "main_norm.weight": "mtp.0.main_norm.weight",
            "output_norm.weight": "mtp.2.norm.weight",
            "markov_embed.weight": "mtp.2.markov_head.embed.weight",
            "markov_head.weight": "mtp.2.markov_head.head.weight",
            "confidence.weight": "mtp.2.confidence_head.proj.weight",
        }
    )
    return result


def tensor_specs(c):
    d, h, q = c.hidden_size, c.hc_mult, c.q_lora_rank
    heads, hd, groups, rank = (
        c.num_attention_heads,
        c.head_dim,
        c.o_groups,
        c.o_lora_rank,
    )
    mixed = (2 + h) * h
    common = {
        "hc_attn_fn": ("F32", (mixed, h * d)),
        "hc_ffn_fn": ("F32", (mixed, h * d)),
        "hc_attn_scale": ("F32", (3,)),
        "hc_ffn_scale": ("F32", (3,)),
        "hc_attn_base": ("F32", (mixed,)),
        "hc_ffn_base": ("F32", (mixed,)),
        "attn.attn_sink": ("F32", (heads,)),
        "attn.q_norm.weight": ("BF16", (q,)),
        "attn.kv_norm.weight": ("BF16", (hd,)),
        "attn_norm.weight": ("BF16", (d,)),
        "ffn_norm.weight": ("BF16", (d,)),
        "ffn.gate.weight": ("BF16", (128, d)),
        "ffn.gate.bias": ("F32", (128,)),
        "ffn.gate.bias_vl": ("F32", (128,)),
    }
    projections = {
        "attn.wq_a": (q, d),
        "attn.wq_b": (heads * hd, q),
        "attn.wkv": (hd, d),
        "attn.wo_a": (groups * rank, heads * hd // groups),
        "attn.wo_b": (d, groups * rank),
        "ffn.shared_experts.w1": (c.moe_intermediate_size, d),
        "ffn.shared_experts.w3": (c.moe_intermediate_size, d),
        "ffn.shared_experts.w2": (d, c.moe_intermediate_size),
    }
    specs = {}
    for stage in range(3):
        p = f"mtp.{stage}."
        specs.update({p + n: s for n, s in common.items()})
        for n, (out, inp) in projections.items():
            specs[p + n + ".weight"] = ("F8_E4M3", (out, inp))
            specs[p + n + ".scale"] = ("F8_E8M0", ((out + 31) // 32, inp // 32))
        for expert in range(128):
            for name in ("w1", "w3", "w2"):
                out, inp = (
                    (d, c.moe_intermediate_size)
                    if name == "w2"
                    else (c.moe_intermediate_size, d)
                )
                specs[p + f"ffn.experts.{expert}.{name}.weight"] = (
                    "I8",
                    (out, inp // 2),
                )
                specs[p + f"ffn.experts.{expert}.{name}.scale"] = (
                    "F8_E8M0",
                    (out, inp // 32),
                )
    specs.update(
        {
            "mtp.0.main_proj.weight": ("F8_E4M3", (d, 3 * d)),
            "mtp.0.main_proj.scale": ("F8_E8M0", ((d + 31) // 32, 3 * d // 32)),
            "mtp.0.main_norm.weight": ("BF16", (d,)),
            "mtp.2.norm.weight": ("BF16", (d,)),
            "mtp.2.markov_head.embed.weight": ("BF16", (c.vocab_size, 256)),
            "mtp.2.markov_head.head.weight": ("BF16", (c.vocab_size, 256)),
            "mtp.2.confidence_head.proj.weight": ("BF16", (1, d + 256)),
        }
    )
    return specs


def dspark_admission_plan(artifact):
    config = DSparkConfig.from_artifact(artifact)
    c = artifact["args"]
    specs = tensor_specs(c)
    tensors = artifact["tensors"]
    actual = {n for n in tensors if n.startswith("mtp.")}
    if actual != set(specs):
        raise ValueError("embedded DSpark tensor inventory differs from source")
    for name, (dtype, shape) in specs.items():
        row = tensors[name]
        if row["dtype"] != dtype or tuple(row["shape"]) != shape:
            raise ValueError("embedded DSpark tensor precision/shape mismatch: " + name)
    payload = sum(tensors[n]["size"] for n in specs)
    # FP8 scales expand across32 output rows; grouped wo_a is expanded BF16.
    prepared = sum(
        math.prod(shape) // 32 for dtype, shape in specs.values() if dtype == "F8_E4M3"
    )
    prepared += sum(
        2 * math.prod(shape)
        for n, (dtype, shape) in specs.items()
        if n.endswith("attn.wo_a.weight")
    )
    staging = max(tensors[n]["size"] for n in specs) + (16 << 20)
    workspace = dspark_workspace_bound(c)
    binding = hashlib.sha256(
        json.dumps(
            {
                "source": artifact["identity"]["fingerprint"],
                "revision": REVISION,
                "config": config.__dict__,
                "tensors": [
                    (n, tensors[n]["dtype"], tensors[n]["shape"]) for n in sorted(specs)
                ],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "schema": REVISION,
        "binding": binding,
        "payload_bytes": payload,
        "prepared_bound_bytes": prepared,
        "staging_bound_bytes": staging,
        # Same bounded B1/block5 forward and Markov scratch charged by the
        # external draft owner, independently of the resident tensor payload.
        "workspace_bound_bytes": workspace,
        "reservation_bytes": payload + prepared + staging + workspace,
        "tensor_count": len(specs),
        "qualification": "unqualified",
    }
