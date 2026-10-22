# Copyright 2026 Eagle AI Corp.
# Modified 2026-10-21 by mlx2 contributors: split the checkpoint config and
# tensor schema out of the MLX drafter so artifact inspection imports no model
# module.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Eagle 4 Source-Available License (not an OSI-approved open-source license):
# see third_party/eagle4/ (LICENSE, NOTICE, MODIFICATIONS.md, README.md).
"""EAGLE-4 checkpoint config and tensor schema (no MLX, no model modules).

The Qwen3.8 27B adapter inspects an external drafter before it pins its
import-time environment; importing ``runtime.drafters.eagle4`` there pulled in
``runtime.models`` and the adapter refused to load (ImportOrderError).
``runtime.drafters.eagle4`` re-exports these names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ARCHITECTURE = "LlamaForCausalLMEagle4"
EAGLE4_NUMERICS_VERSION = "mlx2.eagle4-dense.v1"


@dataclass
class Eagle4Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    sliding_window: int
    target_hidden_size: int
    num_target_layers: int
    aux_layer_ids: list[int]
    candidate_rank: int
    num_candidates: int
    num_hidden_layers: int = 1
    target_layer_ids: list[int] = field(default_factory=list)
    block_size: int = 16
    norm_before_residual: bool = False
    norm_output: bool = True
    fc_norm: bool = True
    has_embed_tokens: bool = True
    model_type: str = "eagle4"
    maximum_proposal_length: int = 31

    def __post_init__(self):
        if not self.target_layer_ids:
            # The release config names post-block outputs. mlx2 captures the
            # same post-block residual stream by zero-based layer index.
            self.target_layer_ids = list(self.aux_layer_ids)
        self.validate()

    @property
    def feature_width(self):
        return len(self.target_layer_ids) * self.target_hidden_size

    def validate(self):
        positive = (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "sliding_window",
            "target_hidden_size",
            "num_target_layers",
            "candidate_rank",
            "num_candidates",
            "num_hidden_layers",
            "block_size",
            "maximum_proposal_length",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) <= 0
            for name in positive
        ):
            raise ValueError("EAGLE-4 integer geometry must be positive")
        if self.num_hidden_layers != 1:
            raise ValueError("EAGLE-4 supports exactly one dense draft layer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("EAGLE-4 attention heads must divide KV heads")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("EAGLE-4 attention width must equal hidden_size")
        if len(self.aux_layer_ids) != 3 or self.aux_layer_ids != sorted(
            set(self.aux_layer_ids)
        ):
            raise ValueError("EAGLE-4 needs three increasing target auxiliary layers")
        if self.target_layer_ids != self.aux_layer_ids:
            raise ValueError(
                "EAGLE-4 target taps must be the declared post-block outputs"
            )
        if (
            self.aux_layer_ids[0] < 0
            or self.aux_layer_ids[-1] >= self.num_target_layers
        ):
            raise ValueError("EAGLE-4 auxiliary layer is outside the target")
        if not all(
            type(value) is bool
            for value in (
                self.norm_before_residual,
                self.norm_output,
                self.fc_norm,
                self.has_embed_tokens,
            )
        ):
            raise ValueError("EAGLE-4 structural switches must be boolean")
        if not self.fc_norm or not self.norm_output or not self.has_embed_tokens:
            raise ValueError(
                "The released dense EAGLE-4 contract requires fc_norm, norm_output and embeddings"
            )
        if not isinstance(self.rms_norm_eps, (int, float)) or self.rms_norm_eps <= 0:
            raise ValueError("EAGLE-4 rms_norm_eps must be positive")

    @classmethod
    def from_hf(cls, config: dict, *, num_target_layers: int):
        if config.get("architectures") != [ARCHITECTURE]:
            raise ValueError(f"Expected a {ARCHITECTURE} draft artifact")
        head = config.get("efficient_lm_head_config")
        eagle = config.get("eagle_config")
        rope = config.get("rope_parameters") or {}
        required = {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "mlp_bias": False,
            "num_hidden_layers": 1,
            "fc_norm": True,
            "norm_output": True,
            "use_sliding_window": True,
            "layer_types": ["sliding_attention"],
            "tie_word_embeddings": False,
        }
        for name, wanted in required.items():
            if config.get(name) != wanted:
                raise ValueError(f"EAGLE-4 {name} must be {wanted!r}")
        if not isinstance(eagle, dict) or eagle.get("use_aux_hidden_state") is not True:
            raise ValueError("EAGLE-4 requires auxiliary target hidden states")
        if set(eagle) != {"use_aux_hidden_state", "eagle_aux_hidden_state_layer_ids"}:
            raise ValueError("Unsupported EAGLE-4 eagle_config keys")
        if not isinstance(head, dict) or set(head) != {
            "type",
            "rank",
            "num_candidates",
        }:
            raise ValueError("Unsupported EAGLE-4 efficient head config")
        if head.get("type") != "low_rank":
            raise ValueError("EAGLE-4 requires a low_rank candidate head")
        if rope.get("rope_type", "default") != "default" or set(rope) - {
            "rope_type",
            "rope_theta",
        }:
            raise ValueError("Unsupported EAGLE-4 RoPE contract")
        if int(config.get("draft_vocab_size", config["vocab_size"])) != int(
            config["vocab_size"]
        ):
            raise ValueError("EAGLE-4 reduced draft vocabularies are not implemented")
        return cls(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            vocab_size=int(config["vocab_size"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(rope.get("rope_theta", config.get("rope_theta", 10000.0))),
            max_position_embeddings=int(config["max_position_embeddings"]),
            sliding_window=int(config["sliding_window"]),
            target_hidden_size=int(
                config.get("target_hidden_size", config["hidden_size"])
            ),
            num_target_layers=int(num_target_layers),
            aux_layer_ids=list(eagle["eagle_aux_hidden_state_layer_ids"]),
            candidate_rank=int(head["rank"]),
            num_candidates=int(head["num_candidates"]),
        )


def expected_weight_schema(config: Eagle4Config) -> dict[str, list[int]]:
    h, inter = config.hidden_size, config.intermediate_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    p = "layers.0."
    return {
        "efficient_lm_head.weight": [config.vocab_size, config.candidate_rank],
        "efficient_lm_head_down_proj.weight": [config.candidate_rank, h],
        "embed_tokens.weight": [config.vocab_size, h],
        "fc.weight": [h, config.feature_width],
        **{
            f"fc_norm.{index}.weight": [config.target_hidden_size] for index in range(3)
        },
        p + "hidden_norm.weight": [h],
        p + "input_layernorm.weight": [h],
        p + "mlp.down_proj.weight": [h, inter],
        p + "mlp.gate_proj.weight": [inter, h],
        p + "mlp.up_proj.weight": [inter, h],
        p + "post_attention_layernorm.weight": [h],
        p + "self_attn.k_proj.weight": [kv, 2 * h],
        p + "self_attn.o_proj.weight": [h, q],
        p + "self_attn.q_proj.weight": [q, 2 * h],
        p + "self_attn.v_proj.weight": [kv, 2 * h],
        "lm_head.weight": [config.vocab_size, h],
        "norm.weight": [h],
    }


# Proposal-tree shape (policy key ``eagle4_tree``).  The default reproduces the
# shape every EAGLE-4 measurement before 2026-10-21 ran: four children per
# expanded node, best-first, depth bounded only by the scheduler's node budget.
EAGLE4_TREE_DEFAULT = {"children": 4, "depth": None, "max_nodes": None}
EAGLE4_TREE_MAX_DEPTH = 31
EAGLE4_TREE_MAX_NODES = 64


def validate_eagle4_tree(value, *, num_candidates=None):
    """Normalized ``{"children", "depth", "max_nodes"}``; raises on any bad value.

    ``children``: candidates pushed per expanded node (1 is a pure chain).
    ``depth``: longest root-to-leaf path in proposal tokens (None: unbounded,
    i.e. the scheduler's node budget decides).  ``max_nodes``: a cap below the
    scheduler's per-round node budget (None: the budget).  Every EAGLE-4 tree
    level costs one sequential drafter step, so these trade proposal width
    for drafter time.
    """
    if value is None:
        return dict(EAGLE4_TREE_DEFAULT)
    if not isinstance(value, dict):
        raise ValueError("eagle4_tree must be an object")
    unknown = set(value) - set(EAGLE4_TREE_DEFAULT)
    if unknown:
        raise ValueError(f"eagle4_tree has unknown keys: {sorted(unknown)}")
    tree = {**EAGLE4_TREE_DEFAULT, **value}
    limit = num_candidates if num_candidates is not None else 128
    children = tree["children"]
    if type(children) is not int or not 1 <= children <= limit:
        raise ValueError(f"eagle4_tree.children must be an integer from 1 to {limit}")
    depth = tree["depth"]
    if depth is not None and (type(depth) is not int or not 1 <= depth <= EAGLE4_TREE_MAX_DEPTH):
        raise ValueError(
            f"eagle4_tree.depth must be null or an integer from 1 to {EAGLE4_TREE_MAX_DEPTH}"
        )
    nodes = tree["max_nodes"]
    if nodes is not None and (type(nodes) is not int or not 1 <= nodes <= EAGLE4_TREE_MAX_NODES):
        raise ValueError(
            f"eagle4_tree.max_nodes must be null or an integer from 1 to {EAGLE4_TREE_MAX_NODES}"
        )
    return tree
