"""Agnes 3 Flash text-only, ordinary-decode candidate."""

from __future__ import annotations

import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import PROCESS_NUMERICS, require_process_numerics
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter

CACHE_LAYOUT = "agnes-hybrid-layer-segments-v1"
DESCRIPTOR = ModelDescriptor(
    model_type="agnes", family="agnes-3-flash", variant="preview-6bit-text-ordinary",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RECURRENT,
                            StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING,
                            Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE,
                            Capability.APC_V2, Capability.LAYERED_CACHE}),
    cache_layout=CACHE_LAYOUT,
    metadata={"execution": "mlx2.adapters.agnes_3_flash.Agnes3FlashAdapter",
              "qualification": "pending", "scope": "text-only", "mtp": "absent",
              "vision": "absent"},
)


def inspect_artifact(model_path: str | Path) -> dict:
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    text = config.get("text_config")
    if config.get("model_type") != "agnes" or not isinstance(text, dict):
        raise ValueError("artifact is not an Agnes conditional-generation checkpoint")
    expected = {"model_type": "agnes_text", "hidden_size": 5120,
                "num_hidden_layers": 72, "intermediate_size": 17408,
                "parallel_ffn_intermediate_size": 2048, "vocab_size": 248320,
                "num_attention_heads": 24, "num_key_value_heads": 4,
                "head_dim": 256, "linear_num_key_heads": 16,
                "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "attn_output_gate": True,
                "output_gate_type": "swish", "hidden_act": "silu",
                "attention_bias": False, "linear_conv_kernel_dim": 4,
                "mamba_ssm_dtype": "float32", "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
                "partial_rotary_factor": 0.25}
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("Agnes 3 Flash text topology mismatch")
    rope = text.get("rope_parameters")
    if (not isinstance(rope, dict) or rope.get("rope_type") != "default"
            or rope.get("rope_theta") != 10_000_000
            or rope.get("partial_rotary_factor") != 0.25
            or rope.get("mrope_section") != [11, 11, 10]
            or rope.get("mrope_interleaved") is not True):
        raise ValueError("Agnes 3 Flash rotary topology mismatch")
    layers = ["agnes_global_attention" if (i + 1) % 4 == 0
              else "agnes_delta_attention" for i in range(72)]
    if text.get("layer_types") != layers:
        raise ValueError("Agnes 3 Flash layer order mismatch")
    if config.get("quantization") != {"group_size": 64, "bits": 6, "mode": "affine"}:
        raise ValueError("Agnes 3 Flash requires affine 6-bit weights")
    weights = artifact["weight_map"]
    if any(key.startswith(("mtp.", "language_model.mtp.")) for key in weights):
        raise ValueError("embedded Agnes MTP is outside the ordinary text contract")
    required = {"language_model.model.embed_tokens.weight", "language_model.lm_head.weight",
                "language_model.model.norm.weight"}
    quantized = {"language_model.lm_head"}
    for i, kind in enumerate(layers):
        prefix = f"language_model.model.layers.{i}."
        required.update({prefix + "input_layernorm.weight",
                         prefix + "post_attention_layernorm.weight"})
        projections = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                       "mlp.parallel_ffn.gate_proj", "mlp.parallel_ffn.up_proj",
                       "mlp.parallel_ffn.down_proj"]
        if kind.endswith("global_attention"):
            projections += ["global_attn.q_proj", "global_attn.k_proj",
                            "global_attn.v_proj", "global_attn.o_proj"]
            required.update({prefix + "global_attn.q_norm.weight",
                             prefix + "global_attn.k_norm.weight"})
        else:
            projections += ["delta_attn.in_proj_qkv", "delta_attn.in_proj_z",
                            "delta_attn.in_proj_b", "delta_attn.in_proj_a",
                            "delta_attn.out_proj"]
            required.update({prefix + "delta_attn.A_log", prefix + "delta_attn.dt_bias",
                             prefix + "delta_attn.conv1d.weight",
                             prefix + "delta_attn.norm.weight"})
        quantized.update(prefix + name for name in projections)
    for prefix in quantized:
        required.update(prefix + "." + suffix for suffix in ("weight", "scales", "biases"))
    if not required <= weights.keys():
        raise ValueError("Agnes 3 Flash text tensors are incomplete")
    artifact["has_mtp"] = False
    artifact["text_weight_count"] = sum(key.startswith("language_model.") for key in weights)
    return artifact


def configure_environment() -> dict[str, str]:
    # Refuse an explicit TF32 value before the profile overwrites it
    # (sweep 1002 review item 6).
    require_process_numerics("the Agnes-3-Flash profile")
    profile = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               **PROCESS_NUMERICS, "MLX_GDN_PACKED": "1",
               "MLX_GDN_CORE": "0", "MLX_LM_COMPILED_DECODE": "0"}
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_", "MLX_AGNES_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


class Agnes3FlashAdapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "agnes-3-flash-6bit-apcv2-ordinary"

    def __init__(self, model_path: str, *, execution_policy=None):
        if execution_policy:
            raise ValueError("Agnes ordinary decode accepts no model execution policy")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT
        self.environment = configure_environment()
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.agnes import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        config = artifact["config"]
        self.model = Model(ModelArgs.from_dict(config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = config["quantization"]
        nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant["mode"], class_predicate=lambda name, module:
                    hasattr(module, "to_quantized") and f"{name}.scales" in weights)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                  trust_remote_code=False)
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        eos = config["eos_token_id"]
        self.tokenizer = TokenizerWrapper(tokenizer,
                                          detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=[eos] if isinstance(eos, int) else list(eos))
        self.max_context = int(config["text_config"]["max_position_embeddings"])

    def prefill_step_default(self):
        """Own the pinned Agnes text-prefill chunk independently of vision."""

        return 2048

    def exact_prefix_cascade_contract(self):
        """Declare Agnes's text-only exact-prefix ownership boundary.

        Ordering and sibling pruning are model-neutral. Reusing an accepted
        prefix is allowed only after the caller proves that the checkpoint is
        the exact Agnes autoregressive text layout, including every recurrent
        and attention plane. Conditional-generation inputs and vision prefill
        are deliberately outside this contract.
        """

        return {
            "schema": "mlx2.exact-prefix-cascade-contract.v1",
            "verification_order": "longest_first",
            "verification_law": "canonical_target_draw_prefix_match",
            "invalid_sibling_pruning": True,
            "accepted_prefix_state": "exact_text_decoder_hybrid_checkpoint",
            "shared_prefix_reuse": "suffix_only_without_common_token_replay",
            "required_cache_layout": CACHE_LAYOUT,
            "required_state_planes": ("attention_kv", "recurrent"),
            "state_scope": "autoregressive_text_decoder",
            "conditional_generation_prefill": False,
            "vision_prefill": False,
            "request_private_only": True,
            "apcv2_publication": False,
            "ordinary_reference_preserved": True,
            "implemented": True,
            "implementation_scope": "planner_and_adapter_contract",
            "qualified": False,
            "selected": False,
            "observed_used": False,
        }

    def plan_exact_prefix_cascade(
        self,
        paths,
        accepted_prefix=(),
        *,
        attempted=(),
        state_scope="autoregressive_text_decoder",
        cache_layout=None,
        exact_state_geometry=False,
    ):
        """Plan a text cascade, refusing unproved prefix-state reuse.

        A cold first stage does not reuse state. Every later stage consumes
        only its suffix, so it requires an exact hybrid checkpoint proof for
        this adapter layout. The explicit scope guard prevents an embedded
        vision/conditional-generation prefill from being mistaken for text
        decoder state.
        """

        prefix = tuple(accepted_prefix)
        if state_scope != "autoregressive_text_decoder":
            raise ValueError("Agnes exact-prefix cascades are text-decoder only")
        if prefix and (
            cache_layout != CACHE_LAYOUT or exact_state_geometry is not True
        ):
            raise ValueError(
                "accepted Agnes prefixes require exact text hybrid cache geometry"
            )
        from ..runtime.exact_prefix_cascade import next_cascade_stage

        return next_cascade_stage(paths, prefix, attempted=attempted)
