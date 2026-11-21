"""Strict artifact gate and ordinary-decode adapter for Maxwell-V1-Flash (``maxwell_moe``).

Scope: bf16 HF checkpoint, ordinary text decode, contexts inside the Block
Sparse GQA dense regime (<= 4,096 tokens; the model refuses longer).  No
native MTP, no APCv2, no speculation.  Unqualified.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .artifact_paths import shard_within_artifact

SOURCE_REPOSITORY = "Maxwell-V1-Flash (local download; model card names no hub revision)"
CACHE_LAYOUT = "maxwell-mhc-dense-kv-v1"

DESCRIPTOR = ModelDescriptor(
    model_type="maxwell_moe",
    family="maxwell",
    variant="v1-flash-ordinary",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
    # REASONING: the template opens <think> by default and the adapter honours
    # enable_thinking both ways; without it serving refused only "true".
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING, Capability.REASONING}),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.maxwell_moe.MaxwellMoeAdapter",
        "qualification": "unqualified",
        "scope": "ordinary text decode in the Block Sparse GQA dense regime (<= 4096 tokens); "
                 "no native MTP",
    },
)

_EXPECTED = {
    "architectures": ["MaxwellMoeForCausalLM"],
    "model_type": "maxwell_moe",
    "attention_variant": "block_sparse_gqa",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 45,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "attention_head_size": 128,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 1024,
    "num_experts": 256,
    "num_experts_per_tok": 12,
    "vocab_size": 200704,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "rope_theta": 20000000,
    "tie_word_embeddings": False,
    "qk_layernorm": True,
    "enable_hyper_connections": True,
    "num_residual_streams": 4,
    "mhc_sinkhorn_iterations": 20,
    "moe_router_enable_expert_bias": True,
    "router_score_function": "sigmoid",
    "block_sparse_gqa_indexer_n_heads": 4,
    "block_sparse_gqa_indexer_head_dim": 128,
    "block_sparse_gqa_block_size": 128,
    "block_sparse_gqa_prefix_window_size": 1024,
    "block_sparse_gqa_local_window_size": 1024,
    "block_sparse_gqa_num_remote_blocks": 16,
}
_IDENTITY_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json",
                   "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                   "special_tokens_map.json")


def _json(path: Path) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            out[key] = value
        return out

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must be an object")
    return value


def required_tensors(config: dict) -> set[str]:
    """Every trunk tensor the ordinary route loads (MTP tensors are optional, dropped)."""
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
             "model.hc_head.hc_head_fn", "model.hc_head.hc_head_base", "model.hc_head.hc_head_scale"}
    dense = set(config["mlp_only_layers"])
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        names.update(p + s for s in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_layernorm.weight", "self_attn.k_layernorm.weight",
            "self_attn.indexer.linear_q_proj.weight", "self_attn.indexer.linear_k_proj.weight",
            "self_attn.indexer.q_layernorm.weight", "self_attn.indexer.k_layernorm.weight"))
        for hc in ("self_attn_hyper_connection", "mlp_hyper_connection"):
            names.update(f"{p}{hc}.{s}" for s in ("alpha_pre", "alpha_post", "alpha_res", "bias",
                                                  "mapping_proj.weight"))
        moe = p + "block_sparse_moe."
        if layer in dense:
            names.update(f"{moe}{w}.weight" for w in ("w1", "w2", "w3"))
        else:
            names.update({moe + "gate.weight", moe + "expert_bias"})
            names.update(f"{moe}shared_expert.{w}.weight" for w in ("w1", "w2", "w3"))
            names.update(f"{moe}experts.{e}.{w}.weight"
                         for e in range(config["num_experts"]) for w in ("w1", "w2", "w3"))
    return names


def inspect_artifact(model_path: str | Path) -> dict:
    """Topology, index closure and shard containment; identity from the indexed files only.

    Only shards the index names are read or fingerprinted, so exFAT AppleDouble
    ``._*.safetensors`` siblings are never touched; an index naming one is refused.
    """
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    if any(config.get(k) != v for k, v in _EXPECTED.items()):
        bad = sorted(k for k, v in _EXPECTED.items() if config.get(k) != v)
        raise ValueError(f"Maxwell topology differs from the pinned V1-Flash release: {bad}")
    if config.get("block_sparse_gqa_layer_freq") not in (None, [1] * 45):
        raise ValueError("Maxwell requires Block Sparse GQA on every layer")
    if config.get("mlp_only_layers") != [0, 1, 2]:
        raise ValueError("Maxwell dense-layer order differs")
    if config.get("attention_dropout", 0.0) not in (0, 0.0) or config.get("rope_scaling") is not None:
        raise ValueError("Maxwell requires zero attention dropout and unscaled RoPE")
    if config.get("use_shared_expert_gate", False):
        raise ValueError("Maxwell shared expert must be ungated")
    if any(k in config for k in ("quantization", "quantization_config")):
        raise ValueError("Maxwell adapter admits the bf16 checkpoint only")
    weights = _json(path / "model.safetensors.index.json").get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Maxwell weight index is empty")
    files = []
    for name in sorted(set(weights.values())):
        if (not isinstance(name, str) or Path(name).name != name or name.startswith("._")
                or not name.endswith(".safetensors")):
            raise ValueError(f"Maxwell shard path is unsafe: {name!r}")
        shard = path / name
        if not shard.is_file() or not shard_within_artifact(path, shard.resolve()):
            raise ValueError(f"Maxwell shard is missing or escaped: {name}")
        st = shard.stat()
        files.append((name, st.st_size, st.st_mtime_ns))
    missing = required_tensors(config) - weights.keys()
    if missing:
        raise ValueError("Maxwell indexed tensor topology is incomplete: " + ", ".join(sorted(missing)[:5]))
    digest = hashlib.sha256()
    for name in _IDENTITY_FILES:
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for record in files:
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": weights,
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": files},
        "source_repository": SOURCE_REPOSITORY,
        "qualification": "unqualified",
    }


class MaxwellMoeAdapter:
    default_route = "ordinary"
    prompt_lookup_default = False
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "maxwell-ordinary"

    def __init__(self, model_path: str, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(self, lambda: self._init(model_path, execution_policy))

    def _init(self, model_path, execution_policy):
        if execution_policy not in (None, {}):
            raise ValueError("Maxwell supports ordinary decode only; it has no execution-policy overrides")
        artifact = inspect_artifact(model_path)
        self.identity, self.config, self.layout = artifact["identity"], artifact["config"], CACHE_LAYOUT
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        require_process_numerics("the Maxwell profile")
        clear_inherited_profile(())
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        os.environ.update(self.environment)
        assert_process_profile_applied("the Maxwell profile")
        from .process_globals import claim_stock_moe

        claim_stock_moe(self, "the Maxwell adapter")

        import mlx.core as mx
        from transformers import AutoTokenizer

        from ..runtime.models.maxwell_moe import Model, ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        path = Path(self.identity["path"])
        self.model = Model(ModelArgs.from_dict(self.config))
        weights = load_shards_evicting([path / name for name, _, _ in self.identity["files"]])
        weights = self.model.sanitize(weights, eager=True)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        generation = path / "generation_config.json"
        eos = (_json(generation).get("eos_token_id") if generation.is_file() else None) \
            or self.config.get("eos_token_id", tokenizer.eos_token_id)
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=eos if isinstance(eos, list) else [eos])
        self.max_context = self.model.args.dense_context_limit

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Maxwell native MTP is not implemented")
        return self.profile

    def execution_config(self, *, max_lanes, prefill_step):
        return {"persistent": True, "num_draft": 0, "rate_gate": False,
                "prefill_step_size": prefill_step, "segment_aware_live_tip": False,
                "segment_aware_cohort_size": max_lanes}

    def prefill_step_default(self):
        return 512

    def approximate_kv_operations(self):
        return {}

    @staticmethod
    def spomin_backend(model, prompt_cache):
        return None

    @staticmethod
    def _template_kwargs(request):
        kwargs = dict(request.get("chat_template_kwargs") or {})
        if set(kwargs) - {"enable_thinking"}:
            raise ValueError("Maxwell chat template accepts only enable_thinking")
        # The server folds every public thinking control (enable_thinking, think,
        # chat_template_kwargs.enable_thinking, reasoning_effort "none") into the
        # top-level enable_thinking and removes chat_template_kwargs before the
        # adapter sees the request.
        flag = request.get("enable_thinking")
        if type(flag) is bool:
            kwargs["enable_thinking"] = flag
        return kwargs

    def _template(self, request, tokenize):
        messages = copy.deepcopy(request["messages"])
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=tokenize,
                                                  **self._template_kwargs(request))

    def prompt_tokens(self, request: dict) -> list[int]:
        if request.get("tools"):
            raise ValueError("Maxwell adapter has no tool-call route")
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        return self._template(request, True)

    def render_prompt(self, request: dict) -> str:
        return request["prompt"] if "messages" not in request else self._template(request, False)

    @classmethod
    def thinking_enabled(cls, request: dict) -> bool:
        # The template opens <think> at the generation prompt unless enable_thinking is false.
        return "messages" in request and cls._template_kwargs(request).get("enable_thinking") is not False

    def output_parser(self, request):
        from ..output import OutputParser

        return OutputParser(chat="messages" in request, thinking=self.thinking_enabled(request),
                            tools=None, stops=request.get("stop", ()))

    def capability_receipts(self):
        return {
            "ordinary_reference": {"implemented": True, "qualified": False, "selected": None,
                                   "observed_used": False,
                                   "selection_authority": "serving_route_receipt",
                                   "profile": self.profile},
            "block_sparse_gqa": {"implemented": False, "qualified": False, "selected": False,
                                 "observed_used": False,
                                 "refusal": "dense-equivalent regime only (<= 4096 tokens)"},
            "native_mtp": {"implemented": False, "qualified": False, "selected": False,
                           "observed_used": False},
            "apcv2": {"implemented": False, "qualified": False, "selected": False,
                      "observed_used": False},
        }

    def diagnostics(self):
        return {
            "architecture": "maxwell-mhc-sparse-moe-gqa",
            "cache_layout": self.layout,
            "experts": {"total": 256, "active": 12, "shared": 1, "packed": True},
            "ordinary_reference": True,
            "qualification": "unqualified",
            "max_context": getattr(self, "max_context", None),
            "target_forward_observed": getattr(getattr(self, "model", None), "completed_forwards", 0) > 0,
            "capability_receipts": self.capability_receipts(),
        }

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
        self.tokenizer = None
