# SPDX-License-Identifier: MIT
"""V4.1 geometry, independent of device imports. See V4.1 provenance."""

from dataclasses import dataclass, fields


@dataclass
class DeepSeekV41Config:
    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1280
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    n_routed_experts: int = 384
    num_experts_per_tok: int = 6
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2304
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    scoring_func: str = "sqrtsoftplus"
    rms_norm_eps: float = 1e-20
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    sliding_window: int = 128
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 1048576
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    swiglu_limit: float = 10.0
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20
    kv_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_head_dim: int = 256
    engram_max_ngram_size: int = 4
    engram_n_heads: int = 8
    engram_compressed_vocab_size: int = 99092
    simulate_quantization: bool = True
    # Tiny CPU equation fixtures may disable BF16 stores, never artifact admission.
    activation_dtype: str = "bfloat16"

    @property
    def intermediate_size(self):
        return self.moe_intermediate_size

    def __post_init__(self):
        for name in (
            "compress_ratios",
            "kv_source_layer_ids",
            "index_source_layer_ids",
            "engram_layer_ids",
            "engram_num_embeddings",
        ):
            setattr(self, name, tuple(getattr(self, name)))
        if len(self.compress_ratios) != self.num_hidden_layers or any(
            type(r) is not int or r not in (0, 1, 2) for r in self.compress_ratios
        ):
            raise ValueError("V4.1 requires exact ratio0/1/2 layer geometry")
        if (
            self.num_key_value_heads != 1
            or self.num_attention_heads % self.o_groups
            or self.head_dim % 2
        ):
            raise ValueError("invalid V4.1 shared KV/group geometry")
        if (
            not 0 < self.qk_rope_head_dim <= min(self.head_dim, self.index_head_dim)
            or self.qk_rope_head_dim % 2
        ):
            raise ValueError("invalid V4.1 rotary geometry")
        if (
            self.scoring_func != "sqrtsoftplus"
            or not 0 < self.num_experts_per_tok <= self.n_routed_experts
            or self.swiglu_limit != 10
        ):
            raise ValueError("unsupported V4.1 expert law")
        if self.activation_dtype not in ("bfloat16", "float32"):
            raise ValueError("unsupported V4.1 activation dtype")
        for ids in (
            self.kv_source_layer_ids,
            self.index_source_layer_ids,
            self.engram_layer_ids,
        ):
            if tuple(sorted(set(ids))) != ids or any(
                type(i) is not int or not 0 <= i < self.num_hidden_layers for i in ids
            ):
                raise ValueError("V4.1 source layers must be unique ordered valid IDs")
        if not set(self.kv_source_layer_ids) <= set(self.index_source_layer_ids):
            raise ValueError("every KV owner must publish index keys")
        for i, ratio in enumerate(self.compress_ratios):
            if ratio:
                self.kv_source_for(i)
                if not any(j <= i for j in self.index_source_layer_ids):
                    raise ValueError("compressed layer has no index source")
        if any(not self.compress_ratios[i] for i in self.index_source_layer_ids):
            raise ValueError("index sources require compressed geometry")
        if (
            self.candidate_source_layer_id >= 0
            and self.candidate_source_layer_id not in self.index_source_layer_ids
        ):
            raise ValueError("candidate source must be an index source")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("Engram table count mismatch")
        if self.candidate_topk_blocks < 1 or self.candidate_block_size < 1:
            raise ValueError("invalid candidate block geometry")

    def kv_source_for(self, layer):
        sources = [i for i in self.kv_source_layer_ids if i <= layer]
        if (
            not sources
            or self.compress_ratios[sources[-1]] != self.compress_ratios[layer]
        ):
            raise ValueError("V4.1 compressed consumer has no matching source")
        return sources[-1]

    @classmethod
    def from_gguf(cls, metadata):
        if metadata.get("general.architecture") != "deepseek41":
            raise ValueError("GGUF architecture is not deepseek41")
        names = {f.name for f in fields(cls)} - {
            "model_type",
            "simulate_quantization",
            "activation_dtype",
            "rope_scaling",
            "engram_layer_ids",
            "engram_num_embeddings",
            "engram_head_dim",
            "engram_max_ngram_size",
            "engram_n_heads",
            "engram_compressed_vocab_size",
        }
        values = {k: metadata["deepseek41." + k] for k in names}
        values.update(
            engram_layer_ids=metadata["deepseek41.engram.layer_ids"],
            engram_num_embeddings=metadata["deepseek41.engram.rows"],
            engram_compressed_vocab_size=metadata[
                "deepseek41.engram.compressed_vocab_size"
            ],
        )
        values["rope_scaling"] = {
            "type": "yarn",
            **{
                k: metadata["deepseek41.rope_scaling." + k]
                for k in (
                    "factor",
                    "beta_fast",
                    "beta_slow",
                    "original_max_position_embeddings",
                )
            },
        }
        return cls(**values)
