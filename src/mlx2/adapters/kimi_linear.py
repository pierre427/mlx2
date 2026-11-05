"""Pinned Kimi Linear REAP BF16 artifact and ordinary serving adapter."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter

REPO = "KimiRio/Huihui-Kimi-Linear-REAP-35B-A3B-Instruct-abliterated"
REVISION = "b42145adff0585a78379d1845b11d20431a40a86"
DEFAULT_CONTEXT = 1048576
CACHE_LAYOUT = "kimi-linear-nope-latent-kda-fp32-v1"
DESCRIPTOR = ModelDescriptor(
    model_type="kimi_linear", family="kimi-linear", variant="reap35b-bf16-ordinary",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RECURRENT,
                            StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING,
                            Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE,
                            Capability.APC_V2, Capability.LAYERED_CACHE}),
    cache_layout=CACHE_LAYOUT,
    metadata={"execution": "mlx2.adapters.kimi_linear.KimiLinearAdapter",
              "qualification": "unqualified", "implemented": True,
              "qualified": False, "selected": False, "observed_used": False,
              "scope": "ordinary text; 1M advertised capacity, long-context unqualified"},
)


def validate_config(config):
    expected = {"model_type": "kimi_linear", "architectures": ["KimiLinearForCausalLM"],
                "hidden_size": 2304, "vocab_size": 163840, "num_hidden_layers": 27,
                "num_experts": 180, "num_experts_per_token": 8, "num_shared_experts": 1,
                "num_expert_group": 1, "topk_group": 1, "first_k_dense_replace": 1,
                "moe_layer_freq": 1, "num_attention_heads": 32,
                "num_key_value_heads": 32, "kv_lora_rank": 512,
                "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                "intermediate_size": 9216, "moe_intermediate_size": 1024,
                "q_lora_rank": None, "mla_use_nope": True, "model_max_length": DEFAULT_CONTEXT,
                "tie_word_embeddings": False, "hidden_act": "silu",
                "moe_router_activation_func": "sigmoid", "moe_renormalize": True,
                "routed_scaling_factor": 2.446, "rms_norm_eps": 1e-5,
                "num_nextn_predict_layers": 0}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Kimi Linear REAP topology or math does not match the staged checkpoint")
    full = [4, 8, 12, 16, 20, 24, 27]
    linear = {"full_attn_layers": full, "kda_layers": [i for i in range(1, 28) if i not in full],
              "head_dim": 128, "num_heads": 32, "short_conv_kernel_size": 4}
    if config.get("linear_attn_config") != linear:
        raise ValueError("Kimi layer partition or KDA geometry differs")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("this staged artifact gate requires original BF16 tensors")


def inspect_artifact(model_path):
    artifact = inspect_indexed_artifact(model_path)
    validate_config(artifact["config"])
    path = Path(artifact["identity"]["path"])
    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    full = set(artifact["config"]["linear_attn_config"]["full_attn_layers"])
    for i in range(27):
        p = f"model.layers.{i}."
        required.update(p+n for n in ("input_layernorm.weight", "post_attention_layernorm.weight"))
        attn = ["q_proj.weight", "o_proj.weight"]
        if i+1 in full:
            attn += ["kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight"]
        else:
            attn += [n+".weight" for n in ("k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d",
                      "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "b_proj", "o_norm")]
            attn += ["A_log", "dt_bias"]
        required.update(p+"self_attn."+n for n in attn)
        if i == 0:
            required.update(p+"mlp."+n+".weight" for n in ("gate_proj", "up_proj", "down_proj"))
        else:
            required.update(p+"block_sparse_moe.experts."+str(e)+"."+n+".weight"
                            for e in range(180) for n in ("w1", "w2", "w3"))
            required.update(p+"block_sparse_moe.shared_experts."+n+".weight"
                            for n in ("gate_proj", "up_proj", "down_proj"))
            required.update(p+"block_sparse_moe.gate."+n for n in ("weight", "e_score_correction_bias"))
    if set(artifact["weight_map"]) != required:
        raise ValueError("Kimi tensor index differs from the complete staged topology")
    # Read-only header validation establishes the BF16 cache/conv budget's
    # dtype premise. safe_open validates offsets without loading tensor data.
    from safetensors import safe_open
    observed = set()
    for name in sorted(set(artifact["weight_map"].values())):
        with safe_open(path/name, framework="np") as shard:
            for key in shard.keys():  # noqa: SIM118 - safe_open is not an iterable mapping
                if artifact["weight_map"].get(key) != name:
                    raise ValueError("Kimi tensor index and shard headers disagree")
                expected_dtype = "F32" if key.endswith((".A_log", ".dt_bias")) else "BF16"
                if shard.get_slice(key).get_dtype() != expected_dtype:
                    raise ValueError(f"Kimi tensor dtype differs from the BF16/F32 contract: {key}")
                observed.add(key)
    if observed != required:
        raise ValueError("Kimi shard tensor headers are incomplete")
    # Include all tokenizer semantics, including the local tiktoken vocabulary.
    digest = hashlib.sha256(artifact["identity"]["fingerprint"].encode())
    for name in ("tiktoken.model", "special_tokens_map.json"):
        digest.update(name.encode())
        digest.update((path/name).read_bytes())
    artifact["identity"]["fingerprint"] = digest.hexdigest()
    artifact["qualification"] = "unqualified"
    return artifact


def parse_policy(policy=None):
    policy = dict(policy or {})
    # The requested 1M capacity needs a bounded attention workspace. Retain
    # expanded MLA as the explicit ordinary numerical reference.
    defaults = {"mla": "latent_tiled", "kda": "kernel", "fused_projections": False,
                "fused_norm": False, "router_combine": "checkpoint", "norm_rounding": "fla_fp32",
                "rms_law": "fused"}
    if policy.keys()-defaults.keys():
        raise ValueError(f"unsupported Kimi policy: {sorted(policy.keys()-defaults.keys())}")
    result = {**defaults, **policy}
    if result["mla"] not in {"expanded", "latent_tiled", "latent_fused", "latent_single", "auto"} or result["kda"] not in {"reference", "kernel", "chunk32", "chunk_wy"}:
        raise ValueError("unsupported Kimi MLA/KDA implementation")
    if result["router_combine"] not in {"checkpoint", "unbiased"}:
        raise ValueError("router_combine must name checkpoint or unbiased semantics")
    if result["rms_law"] not in {"fused", "reference"}:
        raise ValueError("rms_law must name fused or reference (KimiRMSNorm) rounding")
    if result["norm_rounding"] not in {"fla_fp32", "qwen_bf16"}:
        raise ValueError("norm_rounding must name fla_fp32 or qwen_bf16")
    if any(type(result[k]) is not bool for k in ("fused_projections", "fused_norm")):
        raise ValueError("Kimi fusion switches must be boolean")
    return result


class KimiLinearAdapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "kimi-linear-reap35b-bf16-ordinary"
    default_max_context = DEFAULT_CONTEXT
    # Existing live memory admission determines concurrent sessions; a lane
    # ceiling is not a promise to hold that many full 1M contexts.
    default_route = "ordinary"

    def __init__(self, model_path, *, execution_policy=None):
        from .process_globals import guarded_construction
        guarded_construction(self, lambda: self._load(model_path, execution_policy))

    def _load(self, model_path, execution_policy):
        import mlx.core as mx

        from ..process_env import PROCESS_NUMERICS, require_process_numerics
        from ..runtime.kimi_tokenizer import TikTokenTokenizer
        from ..runtime.models.kimi_linear import Model, ModelArgs
        from ..runtime.tokenizer_utils import TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting
        from .process_globals import claim_stock_moe

        require_process_numerics("Kimi Linear ordinary")
        self.environment = dict(PROCESS_NUMERICS)
        self.policy = parse_policy(execution_policy)
        artifact = inspect_artifact(model_path)
        self.config, self.identity = artifact["config"], artifact["identity"]
        # Bind all numerical profiles into the APCv2 state namespace.
        binding = hashlib.sha256(json.dumps(self.policy, sort_keys=True).encode()).hexdigest()
        self.layout = CACHE_LAYOUT+":"+binding
        self.descriptor = replace(DESCRIPTOR, cache_layout=self.layout,
                                  metadata={**DESCRIPTOR.metadata, "selected": True, "policy": self.policy})
        claim_stock_moe(self, "Kimi Linear BF16 ordinary")
        path = Path(self.identity["path"])
        self.model = Model(ModelArgs.from_dict(self.config))
        weights = self.model.sanitize(load_shards_evicting(
            [path/n for n in sorted(set(artifact["weight_map"].values()))]))
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        self.model.configure(self.policy)
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = TikTokenTokenizer.from_pretrained(path, local_files_only=True)
        tokenizer.chat_template = (path/"chat_template.jinja").read_text()
        self.tokenizer = TokenizerWrapper(tokenizer, eos_token_ids=[163586])
        self.max_context = DEFAULT_CONTEXT

    @staticmethod
    def planning_cache_budget(config, *, mtp):
        from .kimi_linear_memory import KimiLinearCacheBudget
        return KimiLinearCacheBudget.from_config(config, mtp=mtp)

    def cache_budget(self, *, mtp):
        from .kimi_linear_memory import KimiLinearCacheBudget
        return KimiLinearCacheBudget.from_config(self.config, mtp=mtp, mla=self.policy["mla"])

    def diagnostics(self):
        return {"route": "ordinary", "qualification": "unqualified", "policy": self.policy,
                "max_context": self.max_context, "counters": self.model.route_status(),
                **self.model.mechanism_status(),
                "nax": {"custom_quantized_gather": "not_applicable_original_bf16",
                        "dense_and_gather_mm": "installed MLX hardware dispatch; not observed by this receipt"}}
