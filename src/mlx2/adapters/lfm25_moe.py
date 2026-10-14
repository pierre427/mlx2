"""Official 8-bit LFM2.5 24B-A2B adapter (ordinary decode, PLD, batching)."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import (
    GENERATION_CONFIG,
    SamplingDefaults,
    VendorSampling,
)
from ..process_env import (
    PROCESS_NUMERICS,
    assert_process_profile_applied,
    clear_inherited_profile,
    require_process_numerics,
)
from .lfm25_output import (
    TOOL_CALL_START,
    make_output_parser,
    make_tool_constraint,
    render_chat,
    validate_tool_request,
)
from .ordinary_artifact import inspect_indexed_artifact

CACHE_LAYOUT = "lfm25-24b-a2b-hybrid-layer-segments-v1"
SOURCE_REVISION = "1104ced19ed98800bdaf4ebcdca14bbdeb597c23"
MODEL_REVISION = "67ab417dd02d0c9545802a726df668620eae049d"
LAYER_TYPES = (
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
)

DESCRIPTOR = ModelDescriptor(
    model_type="lfm2_moe",
    family="lfm2.5-24b-a2b",
    variant="official-8bit-ordinary",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.REASONING,
            Capability.STREAMING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.PROMPT_LOOKUP,
            Capability.GRAMMAR,
            Capability.TOOLS,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.lfm25_moe.LFM25MoEAdapter",
        "implemented": True,
        "qualification": "pending",
        "selected": False,
        "observed_used": False,
        "route": "ordinary",
        "weights": "official affine 8-bit",
        "tools": "pythonic <|tool_call_start|> parser + forced-call grammar (unqualified)",
        "speculation": "prompt lookup (exact ShortConv rollback); no draft head",
        "batching": "mixed-length lanes (ragged ShortConv state)",
        "moe_experts": "shared switch_layers SwitchGLU (rhs pad floor; M5 NAX sorted gather default)",
        "fused_shortconv": "MLX2_LFM25_FUSED_SHORTCONV opt-in, one-token B1 BF16",
        "kv_quantization": "kv_q8 and kv_k8v4 declared, unqualified",
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
    },
)

SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=0.1,
        top_k=50,
        repetition_penalty=1.05,
        source=GENERATION_CONFIG,
    ),
    model="LiquidAI/LFM2.5-24B-A2B",
)


def _positive(config: dict, key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"LFM2.5 {key} must be a positive integer")
    return value


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate the exact official 8-bit topology without importing tensors."""
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    expected = {
        "model_type": "lfm2_moe",
        "architectures": ["Lfm2MoeForCausalLM"],
        "vocab_size": 128000,
        "hidden_size": 2048,
        "intermediate_size": 11776,
        "moe_intermediate_size": 1536,
        "num_hidden_layers": 40,
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "use_expert_bias": True,
        "num_dense_layers": 2,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        "routed_scaling_factor": 1.0,
        "tie_word_embeddings": True,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("LFM2.5 24B-A2B topology mismatch")
    if config.get("layer_types") != list(LAYER_TYPES):
        raise ValueError("LFM2.5 24B-A2B hybrid layer order mismatch")
    rope = config.get("rope_parameters")
    # The model builds plain ``nn.RoPE``: any scaling variant must refuse.
    if (
        not isinstance(rope, dict)
        or rope.get("rope_theta") != 10_000_000.0
        or rope.get("rope_type", "default") != "default"
        or set(rope) - {"rope_theta", "rope_type"}
        or config.get("rope_scaling") not in (None, {})
    ):
        raise ValueError("LFM2.5 24B-A2B rotary configuration mismatch")
    quant = {"group_size": 64, "bits": 8, "mode": "affine"}
    if (
        config.get("quantization") != quant
        or config.get("quantization_config") != quant
    ):
        raise ValueError(
            "LFM2.5 24B-A2B adapter requires official affine 8-bit weights"
        )
    for key, value in expected.items():
        if key not in {
            "model_type",
            "architectures",
            "norm_eps",
            "conv_bias",
            "tie_word_embeddings",
        }:
            _positive(config, key) if type(value) is int else None

    weights = artifact["weight_map"]
    required = {"model.embed_tokens.weight", "model.embedding_norm.weight"}
    for index, layer_type in enumerate(LAYER_TYPES):
        prefix = f"model.layers.{index}."
        required.update({prefix + "operator_norm.weight", prefix + "ffn_norm.weight"})
        required.add(
            prefix
            + (
                "self_attn.q_proj.weight"
                if layer_type == "full_attention"
                else "conv.conv.weight"
            )
        )
        if index < 2:
            required.update(
                {
                    prefix + "feed_forward.gate_proj.weight",
                    prefix + "feed_forward.up_proj.weight",
                    prefix + "feed_forward.down_proj.weight",
                }
            )
        else:
            required.update(
                {
                    prefix + "feed_forward.gate.weight",
                    prefix + "feed_forward.expert_bias",
                    prefix + "feed_forward.switch_mlp.gate_proj.weight",
                    prefix + "feed_forward.switch_mlp.up_proj.weight",
                    prefix + "feed_forward.switch_mlp.down_proj.weight",
                }
            )
            if any(
                prefix + f"feed_forward.gate.{suffix}" in weights
                for suffix in ("scales", "biases")
            ):
                raise ValueError("LFM2.5 routing gates must remain unquantized")
    missing = required - weights.keys()
    if missing:
        raise ValueError(
            "LFM2.5 24B-A2B indexed tensors are incomplete: "
            + ", ".join(sorted(missing)[:5])
        )
    if any(name.startswith(("mtp.", "draft.")) for name in weights):
        raise ValueError("LFM2.5 ordinary target must not contain a draft head")
    artifact.update(
        {
            "max_context": 131072,
            "attention_layers": 10,
            "conv_layers": 30,
            "quantization": quant,
            "qualification": "pending",
            "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION,
        }
    )
    return artifact


@dataclass(frozen=True)
class LFM25MoECacheBudget:
    attention_layers: int = 10
    conv_layers: int = 30
    kv_heads: int = 8
    head_dim: int = 64
    hidden_size: int = 2048
    conv_history: int = 2
    checkpoint_copies: int = 4
    item_bytes: int = 4
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    transient_gib_per_lane: float = 3.1

    @classmethod
    def from_config(cls, config: dict, *, mtp: bool):
        if mtp:
            raise ValueError("LFM2.5 24B-A2B has no implemented MTP route")
        if config.get("layer_types") != list(LAYER_TYPES):
            raise ValueError("unknown LFM2.5 24B-A2B cache topology")
        try:
            checkpoints = max(
                0, int(os.environ.get("MLX_LM_STATE_CHECKPOINT_MAX", "4"))
            )
        except ValueError:
            checkpoints = 4
        return cls(checkpoint_copies=checkpoints)

    @property
    def recurrent_live_bytes(self):
        return self.conv_layers * self.conv_history * self.hidden_size * self.item_bytes

    def project(self, context_tokens: int):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be a nonnegative integer")
        capacity = (
            math.ceil(context_tokens / self.allocation_step) + 1
        ) * self.allocation_step
        kv = (
            self.attention_layers
            * 2
            * self.kv_heads
            * self.head_dim
            * self.item_bytes
            * capacity
        )
        return (
            kv
            + (self.checkpoint_copies + 1) * self.recurrent_live_bytes
            + capacity * self.transcript_bytes_per_token
            + (self.attention_layers + self.conv_layers) * 4096
        )

    def prefill_transient_bytes(self, context_tokens: int, chunk_rows: int):
        if (
            type(context_tokens) is not int
            or context_tokens < 0
            or type(chunk_rows) is not int
            or chunk_rows < 0
        ):
            raise ValueError("invalid LFM2.5 prefill dimensions")
        kv_copy = (
            self.attention_layers
            * 2
            * self.kv_heads
            * self.head_dim
            * self.item_bytes
            * context_tokens
        )
        return int(kv_copy + chunk_rows * self.hidden_size * 16)

    def as_dict(self):
        return {
            "schema": "lfm25-24b-a2b-hybrid-cache-v1",
            **asdict(self),
            "bound": "fp32-KV-plus-shortconv-state-and-APCv2-checkpoints",
            "workspace": "provisional; no controlled performance measurement",
        }


class LFM25MoEAdapter:
    default_route = "ordinary"
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)

    def __init__(self, model_path: str, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self, lambda: self._init_lfm25(model_path, execution_policy=execution_policy)
        )

    def _init_lfm25(self, model_path: str, *, execution_policy=None):
        if execution_policy:
            raise ValueError("LFM2.5 24B-A2B accepts no execution policy")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT
        # Refuse an explicit TF32 value before the profile overwrites it.
        require_process_numerics("the LFM2.5 24B-A2B profile")
        clear_inherited_profile(())
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("the LFM2.5 24B-A2B profile")
        self.environment.update(
            source_revision=SOURCE_REVISION, model_revision=MODEL_REVISION
        )
        self._claim_moe_globals()
        path = Path(self.identity["path"])

        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.lfm25_moe import Model, ModelArgs
        from ..runtime.tokenizer_utils import (
            BPEStreamingDetokenizer,
            TokenizerWrapper,
        )
        from ..runtime.ubc_evict import load_shards_evicting

        self.config = artifact["config"]
        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = artifact["quantization"]

        def quantized_module(name, module):
            return hasattr(module, "to_quantized") and f"{name}.scales" in weights

        nn.quantize(
            self.model,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant["mode"],
            class_predicate=quantized_module,
        )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()

        from .lfm25_fused_shortconv import (
            ShortConvCounters,
            enabled as fused_shortconv_enabled,
            install_text as install_shortconv,
        )

        self.fused_shortconv_counters = ShortConvCounters()
        self.fused_shortconv_opt_in_requested = fused_shortconv_enabled()
        install_shortconv(self.model, self.fused_shortconv_counters)

        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        from .eos import artifact_eos_token_ids

        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=artifact_eos_token_ids(path, self.config, tokenizer),
        )
        self.max_context = artifact["max_context"]

    def _claim_moe_globals(self):
        """Claim the shared sorted-MoE globals this model's experts read.

        The experts are mlx2 ``switch_layers`` gathers: the rhs pad policy
        follows the operator (default floor).  The M5 NAX sorted gather is
        this adapter's default (``gather``; an explicit
        ``MLX2_MOE_NAX_GATHER`` wins): it is bit-identical to the stock sorted
        kernel, prefill-only, and a no-op off M5.  On an M5 Max it lifted
        prefill 14-19% at 4K-16K in an interleaved A/B with bit-identical
        greedy logits
        (qualification/runs/lfm25-24b-a2b-smoke-20261014/ab_nax_prefill.json;
        indicative, not a controlled performance run).  ``fused`` needs a
        fused gate/up expert module this model does not build.
        """
        from ..runtime.models import moe_nax_gather, switch_layers
        from .process_globals import MOE_NAX_GATHER, MOE_RHS_PAD_POLICY, claim

        nax = (
            moe_nax_gather.mode_from_env()
            if os.environ.get(moe_nax_gather.ENV_MODE, "").strip()
            else "gather"
        )
        if nax == "fused":
            raise ValueError(
                "LFM2.5 24B-A2B supports MLX2_MOE_NAX_GATHER=off or gather, not fused"
            )
        self.moe_nax_gather = nax
        self.moe_rhs_pad_policy = switch_layers._rhs_pad_policy()
        self._process_claim = claim(
            self,
            "the LFM2.5 24B-A2B adapter",
            {
                MOE_NAX_GATHER: (nax, moe_nax_gather.set_mode),
                MOE_RHS_PAD_POLICY: (
                    self.moe_rhs_pad_policy,
                    switch_layers.set_pad_policy,
                ),
            },
        )

    @staticmethod
    def spomin_backend(model, prompt_cache):
        return None

    def approximate_kv_operations(self):
        """KV quantization for the ten attention layers (implemented, unqualified).

        Attention layers allocate plain ``KVCache`` and attend through the
        shared ``scaled_dot_product_attention``; ShortConv layers hold
        ``ArraysCache`` state, which stays exact.
        """
        from ..runtime.approximate_kv import standard_kv_quantization_operations

        return standard_kv_quantization_operations(group_size=64)

    def attention_topology(self):
        """Revision-bound 10 full-KV / 30 recurrent ShortConv geometry."""
        from ..runtime.attention_topology import (
            FULL_KV,
            RECURRENT,
            AttentionTopology,
        )

        if self.config.get("layer_types") != list(LAYER_TYPES):
            raise ValueError("unknown LFM2.5 24B-A2B attention topology")
        return AttentionTopology(
            state_revision=self.identity["fingerprint"],
            cache_layout=self.layout,
            layers=tuple(
                FULL_KV if kind == "full_attention" else RECURRENT
                for kind in LAYER_TYPES
            ),
            recurrent_layout="lfm25-shortconv-2x2048-state-v1",
        )

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("LFM2.5 24B-A2B has no implemented MTP route")
        return "lfm25-24b-a2b-8bit-apcv2-ordinary"

    def execution_config(self, *, max_lanes, prefill_step):
        counters = getattr(self, "fused_shortconv_counters", None)
        return {
            "persistent": True,
            "num_draft": 0,
            "backend": "ordinary",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
            "lfm_shortconv": (
                "fused-metal-candidate"
                if counters is not None and counters.installed
                else "source"
            ),
            "moe_nax_gather": getattr(self, "moe_nax_gather", "off"),
        }

    def cache_budget(self, *, mtp):
        return LFM25MoECacheBudget.from_config(self.config, mtp=mtp)

    @staticmethod
    def thinking_enabled(request):
        return "messages" in request

    tool_call_open_marker = TOOL_CALL_START

    def _validate_request(self, request):
        if request.get("enable_thinking") is False:
            raise ValueError("LFM2.5 24B-A2B always reasons before its answer")
        validate_tool_request(request)

    def prompt_tokens(self, request):
        self._validate_request(request)
        if "messages" in request:
            return render_chat(self.tokenizer, request, tokenize=True)
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request):
        self._validate_request(request)
        if "messages" in request:
            return render_chat(self.tokenizer, request, tokenize=False)
        return request["prompt"]

    def thinking_close_token_ids(self):
        marker = "</think>"
        try:
            ids = list(self.tokenizer.encode(marker, add_special_tokens=False))
            if len(ids) != 1 or self.tokenizer.decode(ids) != marker:
                return None
        except Exception:  # noqa: BLE001 - marker absence disables the contract
            return None
        return (int(ids[0]),)

    def output_parser(self, request):
        self._validate_request(request)
        return make_output_parser(request)

    def tool_constraint(self, request):
        return make_tool_constraint(request)

    def diagnostics(self):
        return {
            "architecture": "lfm2_moe",
            "cache_layout": self.layout,
            "route": "ordinary",
            "quantization": self.config["quantization"],
            "moe_nax_gather": getattr(self, "moe_nax_gather", "off"),
            "moe_rhs_pad_policy": getattr(self, "moe_rhs_pad_policy", None),
            "fused_shortconv": vars(self.fused_shortconv_counters)
            if getattr(self, "fused_shortconv_counters", None) is not None
            else None,
            "implemented": True,
            "qualification": "pending",
            "selected": False,
            "observed_used": False,
            "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION,
            "tools": "implemented, unqualified",
        }

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
        self.tokenizer = None
