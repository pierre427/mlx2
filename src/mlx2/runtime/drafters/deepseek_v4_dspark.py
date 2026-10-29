# SPDX-License-Identifier: MIT
"""DeepSeek V4 Vision-Exp DSpark GGUF primitives; distinct from Qwen DSpark.

Pinned equations/layouts: provenance/deepseek-v4-dspark.json. This module does
not register a serving route. Actual sidecar identity must be verified before
runtime construction; scheduler integration/qualification are separate gates.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass

VISION_REVISION = "e46e16bf6035c6f317eb2ac7458eb0362926d402"
TARGET_SHA256 = "66e47437ce7201546fa870a23ccdc094282660f1e52f3403ed5fb59604b2d894"
SIDECAR_SHA256 = "0807a67fd9ce5874bfc60d8d2461f50e11657e3dd94913d3473f85aa679bc877"
DENSE = frozenset((0, 1, 8))
PLAIN = frozenset((0, 1))
F32 = frozenset((0,))
ROUTED = frozenset((8, 10, 16))


class DSparkContractError(ValueError):
    pass


def _alias(metadata, field):
    keys = (f"deepseek4.dspark.{field}", f"deepseek4.dspark_{field}", f"dspark.{field}")
    values = [metadata[k] for k in keys if k in metadata]
    if not values or any(v != values[0] for v in values[1:]):
        raise DSparkContractError(f"missing or conflicting DSpark {field}")
    return values[0]


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]  # GGUF order: contiguous input dimension first.
    kinds: frozenset[int]


@dataclass(frozen=True)
class DeepSeekV4DSparkConfig:
    stages: int
    block_size: int
    markov_rank: int
    noise_token_id: int
    target_layers: tuple[int, ...]
    hidden_size: int = 4096
    vocab_size: int = 129280
    target_layer_count: int = 43
    hc_count: int = 4
    head_count: int = 64
    head_dim: int = 512
    q_lora_rank: int = 1024
    output_groups: int = 8
    output_rank: int = 1024
    expert_count: int = 256
    expert_hidden: int = 2048
    top_k: int = 6
    window: int = 128

    def __post_init__(self):
        for name in ("stages", "block_size", "markov_rank", "noise_token_id"):
            if type(getattr(self, name)) is not int:
                raise DSparkContractError(f"{name} must be an integer")
        if (
            not 3 <= self.stages <= 8
            or not 1 <= self.block_size <= 16
            or not 1 <= self.markov_rank <= self.hidden_size
        ):
            raise DSparkContractError(
                "DSpark stage/block/rank outside bounded implementation"
            )
        if not 0 <= self.noise_token_id < self.vocab_size:
            raise DSparkContractError("DSpark noise token out of vocabulary")
        if not 1 <= len(self.target_layers) <= 8 or any(
            type(x) is not int or not 0 <= x < self.target_layer_count
            for x in self.target_layers
        ):
            raise DSparkContractError("DSpark target layers outside target model")
        if tuple(sorted(set(self.target_layers))) != self.target_layers:
            raise DSparkContractError(
                "DSpark target layers must be strictly increasing"
            )

    @classmethod
    def from_index(cls, metadata, tensors):
        if (
            metadata.get("deepseek4.checkpoint_variant") != "vision-exp"
            or metadata.get("general.source.revision") != VISION_REVISION
        ):
            raise DSparkContractError(
                "sidecar is not the pinned Vision-Exp checkpoint; 0731/V4.1 are incompatible"
            )
        if metadata.get("dspark.target_architecture", "deepseek4") != "deepseek4":
            raise DSparkContractError("sidecar target architecture is not deepseek4")
        if metadata.get("general.architecture") not in (
            "deepseek4",
            "deepseek4-dspark",
        ):
            raise DSparkContractError("unsupported DSpark GGUF architecture")
        ids = {
            int(m.group(1))
            for name in tensors
            if (m := re.match(r"^mtp\.(\d+)\.", name))
        }
        if not ids or max(ids) >= 8 or ids != set(range(max(ids) + 1)):
            raise DSparkContractError("DSpark stage IDs are missing/noncontiguous")
        for name in ("dspark.stage_count", "dspark.n_layers"):
            if name in metadata and (
                type(metadata[name]) is not int or metadata[name] != len(ids)
            ):
                raise DSparkContractError(
                    "DSpark declared stage count conflicts with tensors"
                )
        layers = _alias(metadata, "target_layer_ids")
        if not isinstance(layers, (tuple, list)):
            raise DSparkContractError("target layer IDs must be an array")
        config = cls(
            len(ids),
            _alias(metadata, "block_size"),
            _alias(metadata, "markov_rank"),
            _alias(metadata, "noise_token_id"),
            tuple(layers),
        )
        config.validate_tensors(tensors)
        return config

    def validate_target(self, target):
        geometry = {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.target_layer_count,
            "hc_mult": self.hc_count,
            "num_attention_heads": self.head_count,
            "num_key_value_heads": 1,
            "head_dim": self.head_dim,
            "q_lora_rank": self.q_lora_rank,
            "o_groups": self.output_groups,
            "o_lora_rank": self.output_rank,
            "n_routed_experts": self.expert_count,
            "moe_intermediate_size": self.expert_hidden,
            "num_experts_per_tok": self.top_k,
            "qk_rope_head_dim": 64,
            "rope_theta": 10000.0,
            "swiglu_limit": 10.0,
            "routed_scaling_factor": 1.5,
            "rms_norm_eps": 1e-20,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "simulate_quantization": True,
        }
        for name, expected in geometry.items():
            actual = getattr(target, name, None)
            # GGUF stores these two epsilon metadata fields as IEEEfloat32.
            # Accept only the authored Python literal or its exact storedf32
            # representation; no tolerance, rounding of arbitrary inputs, or
            # normalization that would change model arithmetic is introduced.
            if name in ("rms_norm_eps", "hc_eps") and type(actual) is float:
                stored = struct.unpack("<f", struct.pack("<f", expected))[0]
                matches = actual == expected or actual == stored
            else:
                matches = actual == expected
            if not matches:
                raise DSparkContractError(
                    f"DSpark target {name} differs from pinned semantics"
                )

    def tensor_specs(self):
        d, h = self.hidden_size, self.hc_count
        mix = 2 * h + h * h
        q = self.head_count * self.head_dim
        low = self.output_groups * self.output_rank
        block = {
            "hc_attn_fn.weight": TensorSpec((d * h, mix), PLAIN),
            "hc_attn_scale.weight": TensorSpec((3,), F32),
            "hc_attn_base.weight": TensorSpec((mix,), F32),
            "attn_norm.weight": TensorSpec((d,), F32),
            "attn_q_a.weight": TensorSpec((d, self.q_lora_rank), DENSE),
            "attn_q_a_norm.weight": TensorSpec((self.q_lora_rank,), F32),
            "attn_q_b.weight": TensorSpec((self.q_lora_rank, q), DENSE),
            "attn_kv.weight": TensorSpec((d, self.head_dim), DENSE),
            "attn_kv_a_norm.weight": TensorSpec((self.head_dim,), F32),
            "attn_sinks.weight": TensorSpec((self.head_count,), F32),
            "attn_output_a.weight": TensorSpec(
                (self.head_dim * (self.head_count // self.output_groups), low), DENSE
            ),
            "attn_output_b.weight": TensorSpec((low, d), DENSE),
            "hc_ffn_fn.weight": TensorSpec((d * h, mix), PLAIN),
            "hc_ffn_scale.weight": TensorSpec((3,), F32),
            "hc_ffn_base.weight": TensorSpec((mix,), F32),
            "ffn_norm.weight": TensorSpec((d,), F32),
            "ffn_gate_inp.weight": TensorSpec((d, self.expert_count), DENSE),
            "exp_probs_b.bias": TensorSpec((self.expert_count,), F32),
            "ffn_gate_exps.weight": TensorSpec(
                (d, self.expert_hidden, self.expert_count), ROUTED
            ),
            "ffn_up_exps.weight": TensorSpec(
                (d, self.expert_hidden, self.expert_count), ROUTED
            ),
            "ffn_down_exps.weight": TensorSpec(
                (self.expert_hidden, d, self.expert_count), ROUTED
            ),
            "ffn_gate_shexp.weight": TensorSpec((d, self.expert_hidden), DENSE),
            "ffn_up_shexp.weight": TensorSpec((d, self.expert_hidden), DENSE),
            "ffn_down_shexp.weight": TensorSpec((self.expert_hidden, d), DENSE),
        }
        result = {
            f"mtp.{stage}.{name}": spec
            for stage in range(self.stages)
            for name, spec in block.items()
        }
        result["mtp.0.main_proj.weight"] = TensorSpec(
            (len(self.target_layers) * d, d), DENSE
        )
        result["mtp.0.main_norm.weight"] = TensorSpec((d,), F32)
        p = f"mtp.{self.stages - 1}."
        result.update(
            {
                p + "norm.weight": TensorSpec((d,), F32),
                p + "hc_head_base.weight": TensorSpec((h,), F32),
                p + "hc_head_fn.weight": TensorSpec((d * h, h), PLAIN),
                p + "hc_head_scale.weight": TensorSpec((1,), F32),
                p + "markov_head.markov_w1.weight": TensorSpec(
                    (self.markov_rank, self.vocab_size), DENSE
                ),
                p + "markov_head.markov_w2.weight": TensorSpec(
                    (self.markov_rank, self.vocab_size), DENSE
                ),
                p + "confidence_head.proj.weight": TensorSpec(
                    (d + self.markov_rank, 1), DENSE
                ),
            }
        )
        return result

    def validate_tensors(self, tensors):
        for name, spec in self.tensor_specs().items():
            tensor = tensors.get(name)
            if (
                tensor is None
                or tensor.shape != spec.shape
                or tensor.ggml_type not in spec.kinds
            ):
                raise DSparkContractError(f"DSpark tensor layout/type mismatch: {name}")
        for stage in range(self.stages):
            p = f"mtp.{stage}."
            if (
                tensors[p + "ffn_gate_exps.weight"].ggml_type
                != tensors[p + "ffn_up_exps.weight"].ggml_type
            ):
                raise DSparkContractError("DSpark gate/up quantization differs")


def verify_runtime_binding(
    config, *, target_sha256, sidecar_sha256, target_metadata, source_snapshot
):
    if target_sha256 != TARGET_SHA256 or sidecar_sha256 != SIDECAR_SHA256:
        raise DSparkContractError(
            "actual target/sidecar SHA256 verification required before DSpark load"
        )
    if (
        target_metadata.get("deepseek4.checkpoint_variant") != "vision-exp"
        or target_metadata.get("general.source.revision") != VISION_REVISION
    ):
        raise DSparkContractError("target is not pinned Vision-Exp source")
    if (
        target_metadata.get("general.architecture") != "deepseek4"
        or target_metadata.get("deepseek4.block_count") != config.target_layer_count
    ):
        raise DSparkContractError("target architecture/layer count mismatch")
    if not isinstance(source_snapshot, str) or not re.fullmatch(
        "[0-9a-f]{64}", source_snapshot
    ):
        raise DSparkContractError("exact mlx2 source snapshot required")
    return f"{TARGET_SHA256}:{SIDECAR_SHA256}:{source_snapshot}"


def mean_target_hc(value):
    import mlx.core as mx

    if value.ndim < 2:
        raise DSparkContractError("target feature must include HC and hidden axes")
    return mx.mean(value.astype(mx.float32), axis=-2)


def setup_tokens(seed, position, config):
    if (
        type(seed) is not int
        or not 0 <= seed < config.vocab_size
        or type(position) is not int
        or position < 1
    ):
        raise DSparkContractError("proposal needs valid seed and post-feature position")
    if position + config.block_size > 2**31 - 1:
        raise DSparkContractError("proposal position exceeds int32 reference contract")
    return (seed,) + (config.noise_token_id,) * (config.block_size - 1), tuple(
        range(position - 1, position + config.block_size)
    )


def markov_chain(
    base_logits,
    hidden_before_norm,
    first_prev_token,
    *,
    lookup_w1,
    project_w2,
    confidence_project,
    threshold=0.0,
):
    """Bounded deterministic chain. Confidence reads PRE-final-RMS hidden."""
    import mlx.core as mx

    if (
        base_logits.ndim != 2
        or hidden_before_norm.ndim != 2
        or base_logits.shape[0] != hidden_before_norm.shape[0]
    ):
        raise DSparkContractError("logit/hidden rows mismatch")
    vocab = base_logits.shape[-1]
    if (
        type(first_prev_token) is not int
        or not 0 <= first_prev_token < vocab
        or not 0 <= threshold <= 1
    ):
        raise DSparkContractError("invalid Markov seed/confidence threshold")
    tokens, confidences = [], []
    previous = first_prev_token
    for row in range(base_logits.shape[0]):
        state = lookup_w1(previous).astype(mx.float32).reshape(-1)
        features = mx.concatenate((hidden_before_norm[row].astype(mx.float32), state))
        confidence = confidence_project(features[None]).reshape(-1)[0]
        logits = base_logits[row].astype(mx.float32) + project_w2(state[None]).reshape(
            -1
        )
        mx.eval(confidence, logits)
        if not bool(mx.all(mx.isfinite(logits))) or not bool(mx.isfinite(confidence)):
            raise DSparkContractError("nonfinite DSpark Markov/confidence output")
        confidences.append(float(confidence.item()))
        if threshold > 0 and float(mx.sigmoid(confidence).item()) < threshold:
            break
        previous = int(mx.argmax(logits).item())
        tokens.append(previous)
    return tuple(tokens), tuple(confidences)


class DSparkStage:
    """One HC/MoE support block with noncausal draft attention, no compressor."""

    def __init__(self, target_config, weights, stage):
        from ..models.deepseek_v4 import HyperConnection, Rotary

        self.c, self.w, self.prefix = target_config, weights, f"mtp.{stage}."
        self.attn_hc = HyperConnection(target_config, weights, self.prefix + "hc_attn")
        self.ffn_hc = HyperConnection(target_config, weights, self.prefix + "hc_ffn")
        self.rope = Rotary(
            target_config.qk_rope_head_dim, target_config.rope_theta, scaling=None
        )
        self.attn_norm = weights.array(self.prefix + "attn_norm.weight")
        self.qa_norm = weights.array(self.prefix + "attn_q_a_norm.weight")
        self.kv_norm = weights.array(self.prefix + "attn_kv_a_norm.weight")
        self.sinks = weights.array(self.prefix + "attn_sinks.weight")
        self.ffn_norm = weights.array(self.prefix + "ffn_norm.weight")
        self.correction = weights.array(self.prefix + "exp_probs_b.bias")

    def linear(self, x, name):
        return self.w.linear(self.prefix + name + ".weight")(x)

    def target_kv(self, projected_main, positions):
        import mlx.core as mx

        from ..models.deepseek_v4 import simulate_fp8_kv

        kv = mx.fast.rms_norm(
            self.linear(projected_main, "attn_kv"), self.kv_norm, self.c.rms_norm_eps
        )
        kv = self.rope(kv[:, None], positions=positions)[:, 0]
        return (
            simulate_fp8_kv(kv, self.c.qk_rope_head_dim)
            if self.c.simulate_quantization
            else kv
        )

    def __call__(self, draft_hc, main_x, support_kv, position):
        import mlx.core as mx

        from ..models.deepseek_v4 import (
            attend,
            hc_expand,
            limited_swiglu,
            route_experts,
        )

        c = self.c
        if (
            draft_hc.ndim != 4
            or draft_hc.shape[0] != 1
            or main_x.shape != (1, 1, c.hidden_size)
        ):
            raise DSparkContractError(
                "DSpark stage requires B1 draft HC and one main feature"
            )
        count = draft_hc.shape[1]
        draft_positions = tuple(range(position, position + count))
        x, post, comb = self.attn_hc(draft_hc)
        normalized = mx.fast.rms_norm(x, self.attn_norm, c.rms_norm_eps)
        q = mx.fast.rms_norm(
            self.linear(normalized, "attn_q_a"), self.qa_norm, c.rms_norm_eps
        )
        q = (
            self.linear(q, "attn_q_b")
            .reshape(1, count, c.num_attention_heads, c.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q = mx.fast.rms_norm(q, None, c.rms_norm_eps)
        q = self.rope(q, positions=draft_positions)
        # The target feature bypasses stage HC mixing and attn_norm on KV.
        current = self.target_kv(
            mx.concatenate((main_x, normalized), axis=1),
            (position - 1,) + draft_positions,
        )
        kv = (
            current
            if support_kv is None
            else mx.concatenate((support_kv, current), axis=1)
        )
        mask = mx.ones((1, 1, count, kv.shape[1]), dtype=mx.bool_)
        heads = attend(q, kv[:, None], mask, sinks=self.sinks)
        heads = self.rope(heads, positions=draft_positions, inverse=True)
        grouped = heads.transpose(0, 2, 1, 3).reshape(1, count, c.o_groups, -1)
        low = self.w.grouped(
            grouped, self.prefix + "attn_output_a.weight", groups=c.o_groups
        )
        branch = self.linear(
            low.reshape(1, count, c.o_groups * c.o_lora_rank), "attn_output_b"
        )
        h = hc_expand(branch, draft_hc, post, comb)
        x, post, comb = self.ffn_hc(h)
        x = mx.fast.rms_norm(x, self.ffn_norm, c.rms_norm_eps)
        ids, scores = route_experts(
            self.linear(x, "ffn_gate_inp"),
            top_k=c.num_experts_per_tok,
            scale=c.routed_scaling_factor,
            correction=self.correction,
            normalize=c.norm_topk_prob,
        )
        routed = self.w.moe(
            x.reshape(-1, c.hidden_size),
            ids.reshape(-1, c.num_experts_per_tok),
            scores.reshape(-1, c.num_experts_per_tok),
            self.prefix[:-1],
        ).reshape(x.shape)
        shared = limited_swiglu(
            self.linear(x, "ffn_gate_shexp"),
            self.linear(x, "ffn_up_shexp"),
            c.swiglu_limit,
        )
        return hc_expand(routed + self.linear(shared, "ffn_down_shexp"), h, post, comb)


@dataclass(frozen=True)
class DSparkSnapshot:
    identity: str
    next_position: int
    start_position: int
    main_history: object
    stage_kv: tuple


class DSparkHistory:
    """Committed target features only; owned compact buffers, exact positions."""

    def __init__(self, identity, stages, window=128):
        if (
            type(stages) is not int
            or not 1 <= stages <= 8
            or type(window) is not int
            or window < 1
        ):
            raise DSparkContractError("invalid DSpark history bounds")
        self.identity, self.stages, self.window = identity, stages, window
        self.next_position = self.start_position = 0
        self.main_history = None
        self.stage_kv = (None,) * stages

    @staticmethod
    def _owned(main, stage_kv):
        import mlx.core as mx

        # array()/contiguous() can retain an entire prefill backing allocation.
        main = None if main is None else mx.asarray(main, copy=True)
        values = tuple(
            None if kv is None else mx.asarray(kv, copy=True) for kv in stage_kv
        )
        live = [value for value in (main, *values) if value is not None]
        if live:
            mx.eval(*live)
        return main, values

    def snapshot(self):
        main, values = self._owned(self.main_history, self.stage_kv)
        return DSparkSnapshot(
            self.identity, self.next_position, self.start_position, main, values
        )

    def restore(self, state):
        if (
            not isinstance(state, DSparkSnapshot)
            or state.identity != self.identity
            or len(state.stage_kv) != self.stages
        ):
            raise DSparkContractError("DSpark snapshot identity/stage mismatch")
        if any(
            type(value) is not int or value < 0
            for value in (state.start_position, state.next_position)
        ):
            raise DSparkContractError(
                "DSpark snapshot positions must be nonnegative integers"
            )
        length = state.next_position - state.start_position
        if not 0 <= length <= self.window:
            raise DSparkContractError("DSpark snapshot positions invalid")
        values = (state.main_history, *state.stage_kv)
        if length == 0:
            if any(value is not None for value in values):
                raise DSparkContractError(
                    "empty DSpark snapshot contains tensor values"
                )
        elif any(
            value is None
            or value.ndim != 3
            or value.shape[0] != 1
            or value.shape[1] != length
            or value.shape[2] < 1
            for value in values
        ):
            raise DSparkContractError(
                "DSpark snapshot tensor values/positions disagree"
            )
        main, stage_kv = self._owned(state.main_history, state.stage_kv)
        self.next_position, self.start_position = (
            state.next_position,
            state.start_position,
        )
        self.main_history, self.stage_kv = main, stage_kv

    def append(self, main, stage_kv, position, *, skipped_prefix=0):
        import mlx.core as mx

        if (
            type(position) is not int
            or position != self.next_position
            or main.ndim != 3
            or main.shape[0] != 1
            or not main.shape[1]
            or len(stage_kv) != self.stages
        ):
            raise DSparkContractError(
                "DSpark target history must append contiguous B1 features"
            )
        if (
            type(skipped_prefix) is not int
            or skipped_prefix < 0
            or (skipped_prefix and main.shape[1] != self.window)
        ):
            raise DSparkContractError(
                "skipped projected prefix requires a complete authoritative trailing window"
            )
        if any(kv.ndim != 3 or kv.shape[:2] != main.shape[:2] for kv in stage_kv):
            raise DSparkContractError("DSpark projected target/KV rows disagree")
        replace_window = skipped_prefix > 0 or self.main_history is None
        new_main = (
            main
            if replace_window
            else mx.concatenate((self.main_history, main), axis=1)
        )
        new_kv = tuple(
            value if replace_window else mx.concatenate((old, value), axis=1)
            for old, value in zip(self.stage_kv, stage_kv)
        )
        end = position + skipped_prefix + main.shape[1]
        # Evaluate all compact owned buffers before atomically publishing state.
        new_main, new_kv = self._owned(
            new_main[:, -self.window :],
            tuple(value[:, -self.window :] for value in new_kv),
        )
        self.main_history, self.stage_kv = new_main, new_kv
        self.next_position, self.start_position = end, end - new_main.shape[1]

    def crop(self, next_position):
        if (
            type(next_position) is not int
            or not self.start_position <= next_position <= self.next_position
        ):
            raise DSparkContractError(
                "DSpark crop outside retained exact history; restore snapshot required"
            )
        keep = next_position - self.start_position
        if keep:
            main, values = self._owned(
                self.main_history[:, :keep], tuple(kv[:, :keep] for kv in self.stage_kv)
            )
        else:
            main, values = None, (None,) * self.stages
        self.main_history, self.stage_kv = main, values
        self.next_position = next_position


class DeepSeekV4DSparkBackbone:
    """Unregistered B1 proposal primitive, gated on exact actual sidecar hash.

    Caller supplies verified target post-block HC-mean features. Proposing is
    read-only with respect to history. AST must verify every proposed token;
    confidence scores cannot replace target verification or exact rollback.
    """

    proposal_distribution = "deterministic_point_mass"
    supports_sampling = (
        False  # Until runtime target-distribution acceptance is wired/probed.
    )
    supports_ordered_alternatives = False
    qualified = False

    def __init__(
        self,
        config,
        target_config,
        sidecar_weights,
        target_weights,
        *,
        target_sha256=None,
        sidecar_sha256=None,
        target_metadata=None,
        source_snapshot=None,
    ):
        from ..models.deepseek_v4 import HyperConnection

        self.identity = verify_runtime_binding(
            config,
            target_sha256=target_sha256,
            sidecar_sha256=sidecar_sha256,
            target_metadata=target_metadata or {},
            source_snapshot=source_snapshot,
        )
        config.validate_target(target_config)
        actual_config = DeepSeekV4DSparkConfig.from_index(
            sidecar_weights.metadata, sidecar_weights.tensors
        )
        if actual_config != config:
            raise DSparkContractError(
                "DSpark configuration differs from verified sidecar metadata"
            )
        config.validate_tensors(sidecar_weights.tensors)
        self.config, self.target_config = config, target_config
        self.w, self.target_weights = sidecar_weights, target_weights
        self.stages = [
            DSparkStage(target_config, sidecar_weights, i) for i in range(config.stages)
        ]
        self.main_proj = sidecar_weights.linear("mtp.0.main_proj.weight")
        self.main_norm = sidecar_weights.array("mtp.0.main_norm.weight")
        self.final_prefix = f"mtp.{config.stages - 1}."
        self.head = HyperConnection(
            target_config, sidecar_weights, self.final_prefix + "hc_head", head=True
        )
        self.norm = sidecar_weights.array(self.final_prefix + "norm.weight")
        self.history = DSparkHistory(self.identity, config.stages, config.window)

    def parameters(self):
        """All resident sidecar planes, excluding separately owned target weights."""
        native_linear = getattr(self.w, "native_fp32_parameters", None)
        hc = {"head": self.head.static_parameters()}
        for index, stage in enumerate(self.stages):
            hc[f"mtp.{index}.hc_attn"] = stage.attn_hc.static_parameters()
            hc[f"mtp.{index}.hc_ffn"] = stage.ffn_hc.static_parameters()
        return {
            "native": dict(getattr(self.w, "_dense", {})),
            "packed": {
                name: value.data
                for name, value in getattr(self.w, "_packed", {}).items()
            },
            "native_linear_fp32": native_linear() if callable(native_linear) else {},
            "hc_static_fp32": hc,
            "rotary_inv": {
                f"mtp.{i}": stage.rope.inv for i, stage in enumerate(self.stages)
            },
        }

    def preload(self):
        """Materialize post-construction support arrays before admission/wiring."""
        import mlx.core as mx

        for name, tensor in self.w.tensors.items():
            if tensor.ggml_type in (0, 1, 30):
                self.w.array(name)
            else:
                self.w.packed(name)
        # Stage.linear resolves lazily during proposals. Register its exact
        # native FP32 operand caches now, not on the first served request.
        names = [
            "mtp.0.main_proj.weight",
            self.final_prefix + "markov_head.markov_w2.weight",
            self.final_prefix + "confidence_head.proj.weight",
        ]
        for index in range(self.config.stages):
            names.extend(
                f"mtp.{index}.{suffix}.weight"
                for suffix in (
                    "attn_q_a",
                    "attn_q_b",
                    "attn_kv",
                    "attn_output_b",
                    "ffn_gate_inp",
                    "ffn_gate_shexp",
                    "ffn_up_shexp",
                    "ffn_down_shexp",
                )
            )
        for name in names:
            self.w.linear(name)
        preload_native = getattr(self.w, "preload_native_linear_fp32", None)
        if callable(preload_native):
            preload_native()
        mx.eval(self.parameters())
        return self

    def record_features(
        self, captures: Mapping[int, object], position: int, *, history=None
    ):
        import mlx.core as mx

        if set(captures) != set(self.config.target_layers):
            raise DSparkContractError("missing or extra target layer feature capture")
        values = [captures[layer] for layer in self.config.target_layers]
        shape = values[0].shape
        if (
            len(shape) != 3
            or shape[0] != 1
            or shape[-1] != self.config.hidden_size
            or any(v.shape != shape for v in values)
        ):
            raise DSparkContractError(
                "DSpark features must be aligned B1 post-block HC means"
            )
        skip = max(0, shape[1] - self.config.window)
        # As pinned DS4 seed-cache path: project only the authoritative suffix,
        # not thousands of rows that the bounded support window will discard.
        values = [value[:, skip:] for value in values]
        main = mx.fast.rms_norm(
            self.main_proj(mx.concatenate(values, axis=-1)),
            self.main_norm,
            self.target_config.rms_norm_eps,
        )
        positions = tuple(range(position + skip, position + skip + main.shape[1]))
        stage_kv = tuple(stage.target_kv(main, positions) for stage in self.stages)
        history = self.history if history is None else history
        if history.identity != self.identity:
            raise DSparkContractError("DSpark per-request history binding differs")
        history.append(main, stage_kv, position, skipped_prefix=skip)

    def propose(
        self,
        seed: int,
        *,
        position: int,
        confidence_threshold: float = 0.0,
        history=None,
        max_proposals=None,
    ):
        import mlx.core as mx

        tokens, _ = setup_tokens(seed, position, self.config)
        history = self.history if history is None else history
        if history.identity != self.identity:
            raise DSparkContractError("DSpark per-request history binding differs")
        limit = self.config.block_size if max_proposals is None else max_proposals
        if type(limit) is not int or not 1 <= limit <= self.config.block_size:
            raise DSparkContractError(
                "proposal count must be within trained block size"
            )
        if history.next_position != position or history.main_history is None:
            raise DSparkContractError(
                "DSpark target feature history not at proposal boundary"
            )
        main = history.main_history[:, -1:]
        embedded = self.target_weights.embedding(
            mx.array([tokens], dtype=mx.uint32)
        ).astype(mx.float32)
        draft = mx.broadcast_to(
            embedded[:, :, None],
            (*embedded.shape[:2], self.config.hc_count, self.config.hidden_size),
        )
        for index, stage in enumerate(self.stages):
            support = history.stage_kv[index][:, :-1]
            draft = stage(draft, main, support, position)
        # Confidence consumes this value before final RMSNorm. The backbone
        # still sees its complete trained noncausal draft block.
        hidden = self.head(draft)[:, :limit]
        normalized = mx.fast.rms_norm(
            hidden, self.norm, self.target_config.rms_norm_eps
        )
        logits = self.target_weights.linear("output.weight")(normalized).reshape(
            limit, -1
        )
        p = self.final_prefix
        return markov_chain(
            logits,
            hidden.reshape(limit, -1),
            seed,
            lookup_w1=lambda token: self.w.embedding(
                mx.array([token], dtype=mx.uint32), p + "markov_head.markov_w1.weight"
            )[0],
            project_w2=self.w.linear(p + "markov_head.markov_w2.weight"),
            confidence_project=self.w.linear(p + "confidence_head.proj.weight"),
            threshold=confidence_threshold,
        )
