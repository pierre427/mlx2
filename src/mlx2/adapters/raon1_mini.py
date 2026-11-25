"""KRAFTON Raon-1-mini (``raon1_mini``) artifacts and serving adapter.

Two artifact layouts share the 25-layer topology (provenance/raon1-mini.json):

* ``krafton_nvfp4``: KRAFTON/Raon-1-mini-mlx (the vendor MLX pack), NVFP4 g16
  everywhere except embeddings, norms, mHC selectors, router and ``kv_b_proj``
  (bf16) and an MXFP8 head.  Dense linears carry an output-side fp32 global
  scale; routed experts carry MLX's in-kernel per-expert scale.  The vendor's
  own MLX model file is NOT used (``runtime.models.raon1_mini`` is ours).
* ``mlx2_mxfp8e``: the mlx2 conversion of KRAFTON/Raon-1-mini (BF16 with
  MXFP8 routed experts): the experts' E4M3 codes and E8M0 scales repacked
  into MLX ``mxfp8`` byte for byte, everything else BF16 (the reference).

Routes: ordinary (default) and the official DFlash2 drafter
(KRAFTON/Raon-1-mini-dflash2) as an explicit ``draft_model`` external draft.
Text with reasoning and qwen3-coder XML tool calls.  Every route is
unqualified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter

SOURCE_REPOSITORY = "KRAFTON/Raon-1-mini"
SOURCE_REVISION = "6691a076e13a225148b1bce2abdd57a0055fae63"
PACK_REPOSITORY = "KRAFTON/Raon-1-mini-mlx"
PACK_REVISION = "3804c11410aa9d3e51d7667d6d6d5de596d3cc41"
CONVERSION_RECIPE = "raon1-mini-mxfp8e-v1"
AFFINE_RECIPE = "raon1-mini-affine-a4g64-h8g64-v1"
RECIPE_LAYOUTS = {CONVERSION_RECIPE: "mlx2_mxfp8e", AFFINE_RECIPE: "mlx2_affine"}
DEFAULT_CONTEXT = 262144
CACHE_LAYOUT = "raon1-mini-mhc4-rda-fp32-nope-latent-mla-v1"
EOS_TOKEN_IDS = (259202, 259200)

LAYOUTS = {
    "krafton_nvfp4": {"variant": "krafton-nvfp4-ordinary", "repo": PACK_REPOSITORY,
                      "revision": PACK_REVISION},
    "mlx2_affine": {"variant": "mlx2-a4g64-ordinary", "repo": SOURCE_REPOSITORY,
                    "revision": SOURCE_REVISION},
    "mlx2_mxfp8e": {"variant": "mlx2-mxfp8e-ordinary", "repo": SOURCE_REPOSITORY,
                    "revision": SOURCE_REVISION},
}

DESCRIPTOR = ModelDescriptor(
    model_type="raon1_mini", family="raon-1-mini", variant="krafton-nvfp4-ordinary",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RECURRENT,
                            StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING, Capability.REASONING,
                            Capability.TOOLS, Capability.CONTINUOUS_BATCH,
                            Capability.PREFIX_REUSE, Capability.APC_V2,
                            Capability.LAYERED_CACHE}),
    cache_layout=CACHE_LAYOUT,
    metadata={"execution": "mlx2.adapters.raon1_mini.RaonAdapter",
              "qualification": "unqualified", "implemented": True,
              "qualified": False, "selected": False, "observed_used": False,
              "residual": "raon-mhc-4-stream-helmert-chart",
              "scope": "text, reasoning, qwen3-coder XML tools; 256K advertised, long context unqualified"},
)

SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
        source=GENERATION_CONFIG,
        note="KRAFTON/Raon-1-mini generation_config.json (do_sample, presence_penalty 1.5)",
    ),
    model="KRAFTON/Raon-1-mini",
)

_TOPOLOGY = {
    "model_type": "raon1_mini", "architectures": ["Raon1MiniForCausalLM"],
    "hidden_size": 2560, "num_hidden_layers": 25, "num_attention_heads": 40, "head_dim": 128,
    "vocab_size": 259840, "tie_word_embeddings": False, "max_position_embeddings": DEFAULT_CONTEXT,
    "rms_norm_eps": 1e-05, "hidden_act": "situ", "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0, "intermediate_size": 12032, "first_k_dense_replace": 1,
    "moe_layer_freq": 1, "num_experts": 128, "num_experts_per_token": 6,
    "moe_intermediate_size": 1088, "num_shared_experts": 1, "moe_router_activation_func": "sigmoid",
    "moe_renormalize": True, "routed_scaling_factor": 1.0, "q_lora_rank": 576, "kv_lora_rank": 512,
    "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128, "mla_use_nope": True,
    "mla_use_output_gate": True, "raon_mhc_streams": 4,
}
_RDA = {"num_heads": 40, "head_dim": 128, "conv_kernel_size": 4, "full_rank_gate": True,
        "gate_lower_bound": -5.0, "mla_layers": [4, 8, 12, 16, 20, 24, 25],
        "rda_layers": [i for i in range(1, 26) if i not in (4, 8, 12, 16, 20, 24, 25)]}
STEERING_LAYERS = [12, 14, 16, 18, 20]


def validate_config(config: dict) -> str:
    """The artifact layout name, or raise for anything not staged."""
    wrong = sorted(k for k, v in _TOPOLOGY.items() if config.get(k) != v)
    if wrong:
        raise ValueError(f"Raon-1-mini topology differs from the pinned release: {wrong[:6]}")
    if config.get("rda_config") != _RDA:
        raise ValueError("Raon-1-mini RDA/MLA layer partition differs from the release")
    steering = config.get("reasoning_effort_steering") or {}
    if steering.get("layers") != STEERING_LAYERS or steering.get("default") not in steering.get("alpha", {}):
        raise ValueError("Raon-1-mini reasoning-effort steering table differs from the release")
    if config.get("rope_parameters") is not None:
        raise ValueError("Raon-1-mini is NoPE; a rope_parameters block is not supported")
    for key in ("use_grouped_topk", "latent_moe_use_norm"):
        if config.get(key):
            raise ValueError(f"Raon-1-mini switch {key} is not supported")
    for key in ("routed_expert_hidden_size", "attn_res_block_size"):
        if config.get(key) is not None:
            raise ValueError(f"Raon-1-mini switch {key} is not supported")
    conversion = config.get("mlx2_conversion") or {}
    if conversion:
        if (conversion.get("recipe") not in RECIPE_LAYOUTS
                or conversion.get("source_revision") != SOURCE_REVISION):
            raise ValueError("artifact is not a pinned mlx2 Raon-1-mini conversion")
        layout = RECIPE_LAYOUTS[conversion["recipe"]]
        if layout == "mlx2_affine":
            quant = config.get("quantization") or {}
            if ((quant.get("mode"), quant.get("bits"), quant.get("group_size")) != ("affine", 4, 64)
                    or quant.get("lm_head") != {"group_size": 64, "bits": 8}):
                raise ValueError("Raon affine artifact must declare a4g64 with an 8-bit g64 head")
        return layout
    quant = config.get("quantization") or {}
    if (quant.get("mode"), quant.get("bits"), quant.get("group_size")) == ("nvfp4", 4, 16):
        return "krafton_nvfp4"
    if (config.get("quantization_config") or {}).get("quant_method") == "mxfp8":
        raise ValueError("the KRAFTON BF16/MXFP8 source needs the mlx2 conversion "
                         "(scripts/convert_raon1_mini.py) before serving")
    raise ValueError("Raon-1-mini artifact layout is not staged")


def _required(layout: str) -> set[str]:
    """Tensor names every staged layout must index (quantization planes added per layout)."""
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    mla = {i - 1 for i in _RDA["mla_layers"]}
    for i in range(25):
        p = f"model.layers.{i}."
        names |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight"}
        for site in ("self_attn_raon_mhc", "mlp_raon_mhc"):
            names |= {f"{p}{site}.{n}" for n in ("phi_pre", "phi_post", "alpha_pre", "alpha_post",
                                                  "b_pre", "b_post", "res.phi_z", "res.alpha_z",
                                                  "res.b_z", "res.eta")}
        a = p + "self_attn."
        if i in mla:
            names |= {a + n for n in ("q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
                                      "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight",
                                      "kv_b_proj.weight", "g_proj.weight", "o_proj.weight")}
        else:
            names |= {a + n for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                      "f_a_proj.weight", "f_b_proj.weight", "b_proj.weight",
                                      "rda_w_proj.weight", "g_proj.weight", "o_proj.weight",
                                      "o_norm.weight", "A_log", "dt_bias")}
            if layout == "krafton_nvfp4":
                names.add(a + "qkv_conv1d.weight")
            else:
                names |= {a + f"{n}_conv1d.weight" for n in "qkv"}
        if i == 0:
            names |= {p + f"mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")}
        else:
            m = p + "moe."
            names |= {m + "gate.weight", m + "gate.e_score_correction_bias"}
            names |= {m + f"shared_experts.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")}
            names |= {m + f"switch_mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")}
            if i in STEERING_LAYERS:
                names.add(m + "reasoning_effort_steering")
    return names


def inspect_artifact(model_path):
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    layout = validate_config(config)
    weights = artifact["weight_map"]
    missing = sorted(_required(layout) - weights.keys())
    if missing:
        raise ValueError(f"Raon-1-mini tensors are incomplete: {missing[:6]}")
    if layout == "krafton_nvfp4":
        dense = set(config.get("raon_global_scaled_linears") or ())
        if len(dense) != 272:
            raise ValueError("Raon NVFP4 pack must list its 272 global-scaled linears")
        for path in sorted(dense):
            for part in ("weight", "scales", "global_scale"):
                if f"{path}.{part}" not in weights:
                    raise ValueError(f"Raon NVFP4 linear {path} lacks {part}")
    path = Path(artifact["identity"]["path"])
    digest = hashlib.sha256(artifact["identity"]["fingerprint"].encode())
    digest.update(layout.encode())
    artifact["identity"]["fingerprint"] = digest.hexdigest()
    artifact["layout"] = layout
    artifact["qualification"] = "unqualified"
    artifact["checkpoint"] = f"{path.name}@sha256:{artifact['identity']['fingerprint'][:16]}"
    return artifact


NUMERIC_POLICY_KEYS = frozenset({"moe_nax", "mla", "mla_flash", "rda", "fused_norm", "reasoning_effort", "mhc",
                                 "merged_projections", "rms_law"})


def split_policy(policy=None):
    policy = dict(policy or {})
    numerics = {key: policy.pop(key) for key in list(policy) if key in NUMERIC_POLICY_KEYS}
    return numerics, policy


def parse_policy(policy=None):
    policy = dict(policy or {})
    # mla "auto": expanded fused-causal SDPA for prompt chunks, absorbed
    # one-pass latent for decode/verify rows (Kimi Linear's measured law).
    # The NAX latent/flash192 kernels are fixed at Kimi's 32 heads and decline
    # Raon's 40, so mla_flash stays off.  rda "kernel": the gated-delta vector
    # Metal kernel; "chunk_wy" is a candidate prefill law (opt-in, unmeasured).
    # mhc "fused": two-launch mHC sites (runtime.models.raon1_mini_fused),
    # opt-in until the probe classifies it against the ops law and an A/B
    # shows a decode win.
    # mla_flash "expanded": Kimi's NAX flash192 prompt-chunk kernel (and its
    # key-window form past the expanded bound) admitted at Raon's 40 heads
    # behind a 40-head probe; opt-in, not bitwise to the fused D256 SDPA law,
    # unmeasured.  moe_nax "gather": mlx2's segmented NAX sorted expert gather
    # (bit-identical to MLX's sorted gather) for prompt chunks; affine
    # artifact only (it keys on QuantizedSwitchLinear); opt-in, unmeasured.
    defaults = {"moe_nax": "off", "mla": "auto", "mla_flash": "off", "rda": "kernel", "fused_norm": False,
                "reasoning_effort": None, "mhc": "ops",
                # One matmul for RDA q/k/v/b/f_a/g/w and MLP up/gate at <= 8
                # rows (resident merged copy, ~0.6 GB on the NVFP4 pack).
                # DEFAULT ON (2026-11-25, unqualified): bitwise-identical
                # logits and greedy tokens, decode +14 % ABBA on
                # Raon-1-mini-mlx@sha256:089f6390 (qualification/runs/
                # raon1-mini-20261125/levers-2c6ef2759/merged.json).
                "merged_projections": True,
                # Layer/final/MLA-latent RMSNorm rounding.  "mlx" (default):
                # mx.fast.rms_norm, normalize -> bf16 -> weight product in
                # bf16 (two roundings).  "fp32": the reference
                # Raon1MiniRMSNorm, fp32 weight product, one rounding.
                # Opt-in, untested on GPU (docs/research/
                # 2026-11-25-raon1-mini-pr-sweep.md); a default change is
                # Pierre's call after an oracle A/B.
                "rms_law": "mlx",
                # Expanded-K/V transient bound (GiB) for MLA prompt chunks.
                # Below it, chunks use fused padded-D256 causal SDPA; above it
                # Kimi's law falls back to the Python-tiled absorbed latent
                # loop (64-row x 2048-key tiles).  At Raon's 40 heads the 4 GiB
                # default is crossed at ~104.8K keys, which is the 128K prefill
                # cliff (week ladder 2026-11-25: 64K 1116 tok/s, 128K 341).
                # 12 keeps D256 through 256K (10.7 GB transient).  Opt-in,
                # unmeasured: needs a GPU A/B at 96K/128K/192K first.
                "mla_expanded_gib": 4}
    if policy.keys() - defaults.keys():
        raise ValueError(f"unsupported Raon policy: {sorted(policy.keys() - defaults.keys())}")
    result = {**defaults, **policy}
    if result["mla"] not in {"expanded", "latent_tiled", "latent_single", "auto"}:
        raise ValueError("unsupported Raon MLA law")
    if result["mla_flash"] not in {"off", "expanded"}:
        raise ValueError("mla_flash must be off or expanded (the absorbed NAX decode kernel is 32-head only)")
    if result["moe_nax"] not in {"off", "gather"}:
        raise ValueError("moe_nax must name off or gather")
    if result["rda"] not in {"reference", "kernel", "chunk_wy"}:
        raise ValueError("unsupported Raon RDA law")
    if type(result["mla_expanded_gib"]) is not int or not 1 <= result["mla_expanded_gib"] <= 24:
        raise ValueError("mla_expanded_gib must be an integer from 1 to 24")
    if result["rms_law"] not in {"mlx", "fp32"}:
        raise ValueError("rms_law must name mlx or fp32")
    if result["mhc"] not in {"ops", "fused"}:
        raise ValueError("mhc must name ops or fused")
    if type(result["merged_projections"]) is not bool:
        raise ValueError("merged_projections must be boolean")
    if type(result["fused_norm"]) is not bool:
        raise ValueError("fused_norm must be boolean")
    effort = result["reasoning_effort"]
    if effort is not None and effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("reasoning_effort must be none, low, medium, high, xhigh or max")
    return result


def _nax_status():
    from ..runtime.models import moe_nax_gather

    return moe_nax_gather.status()


class RaonAdapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    sampling_defaults = SAMPLING
    profile = "raon1-mini-krafton-nvfp4-ordinary"
    default_max_context = DEFAULT_CONTEXT
    default_route = "ordinary"
    prompt_lookup_default = False
    tool_call_open_marker = "<tool_call>"
    # The pack's drafter was trained at block 8; the vendor runtime drafts
    # 5 (7 at high acceptance).  Unmeasured here until the GSM8K run.
    EXTERNAL_DEFAULT_NUM_DRAFT = 5
    EXTERNAL_ROUTE_TAG = "external-raon1-mini-v1"
    EXTERNAL_PROFILE = "raon1-mini-external-draft"
    default_prefill_step = 2048

    def prefill_step_default(self):
        return int(type(self).default_prefill_step)

    def __init__(self, model_path, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(self, lambda: self._load(model_path, execution_policy))

    def _load(self, model_path, execution_policy):
        import mlx.core as mx
        from transformers import AutoTokenizer

        from ..process_env import PROCESS_NUMERICS, require_process_numerics
        from ..runtime.models import raon1_mini as R
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting
        from .process_globals import claim_stock_moe

        require_process_numerics("Raon-1-mini ordinary")
        self.environment = dict(PROCESS_NUMERICS)
        numerics, speculative = split_policy(execution_policy)
        self.policy = parse_policy(numerics)
        external = self._parse_external_policy(speculative, family="Raon-1-mini")
        artifact = inspect_artifact(model_path)
        self.config, self.identity = artifact["config"], artifact["identity"]
        self.artifact_layout = layout = artifact["layout"]
        self.checkpoint = artifact["checkpoint"]
        draft_record = None
        if external:
            from .raon1_mini_dflash2 import inspect_drafter

            draft_record = inspect_drafter(self.external_policy["draft_model"],
                                           target_path=self.identity["path"],
                                           target_config=self.config,
                                           target_fingerprint=self.identity["fingerprint"])
            self._check_num_draft(draft_record)
            self.draft_record_receipt = {k: draft_record[k] for k in (
                "checkpoint", "fingerprint", "config_sha256", "source_repository", "source_revision")}
        # rms_law joined the policy after APCv2 stores were written under the
        # default law; binding it only when it is not the default keeps those
        # stores' layout key while an "fp32" store can never be reused by an
        # "mlx" server (or the reverse).
        bound = {k: v for k, v in self.policy.items() if not (k == "rms_law" and v == "mlx")}
        binding = hashlib.sha256(json.dumps({**bound, "layout": layout},
                                            sort_keys=True).encode()).hexdigest()
        self.layout = CACHE_LAYOUT + ":" + binding
        self.descriptor = replace(DESCRIPTOR, cache_layout=self.layout, variant=LAYOUTS[layout]["variant"],
                                  metadata={**DESCRIPTOR.metadata, "selected": True, "policy": self.policy,
                                            "checkpoint": self.checkpoint})
        self.profile = "raon1-mini-" + LAYOUTS[layout]["variant"]
        if self.policy["moe_nax"] == "gather":
            if layout != "mlx2_affine":
                raise ValueError("moe_nax gather needs the affine artifact (QuantizedSwitchLinear experts)")
            from .process_globals import claim_moe

            claim_moe(self, "Raon-1-mini affine", nax_mode="gather")
        else:
            claim_stock_moe(self, "Raon-1-mini ordinary")
        path = Path(self.identity["path"])
        self.model = R.Model(R.ModelArgs.from_dict(self.config))
        names = artifact["weight_map"].keys()
        self.storage = R.install_for_layout(self.model, layout, self.config, names)
        weights = self.model.sanitize(load_shards_evicting(
            [path / n for n in sorted(set(artifact["weight_map"].values()))]))
        self.model.load_weights(list(weights.items()), strict=True)
        weights.clear()
        self.model.eval()
        self.model.configure(self.policy)
        self.effort_alpha = self.model.set_reasoning_effort(self.policy["reasoning_effort"])
        mx.eval(self.model.parameters())
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=list(EOS_TOKEN_IDS))
        self.max_context = DEFAULT_CONTEXT
        if draft_record is not None:
            from .raon1_mini_dflash2 import load_drafter

            self._bind_external_drafter(draft_record, load_drafter, self.descriptor)
            self.__dict__.pop("profile_name", None)
            stem = LAYOUTS[layout]["variant"].removesuffix("-ordinary")
            self.descriptor = replace(self.descriptor, variant=stem + "-external-dflash2",
                                      metadata={**self.descriptor.metadata, "qualification": "unqualified",
                                                "speculative_route": "external_raon_dflash2_chain"})

    # ------------------------------------------------------------ routes

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Raon-1-mini ships no MTP layer")
        if getattr(self, "draft_model", None) is not None:
            return self.profile.removesuffix("-ordinary") + "-external-dflash2"
        return self.profile

    def execution_config(self, *, max_lanes, prefill_step):
        if getattr(self, "draft_model", None) is None:
            return {**super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step),
                    "prefill_plan_exact": True}
        return {**self._external_execution_config(max_lanes=max_lanes, prefill_step=prefill_step),
                "prefill_plan_exact": True}

    def cache_budget(self, *, mtp):
        if getattr(self, "draft_model", None) is not None:
            return None
        from .raon1_mini_memory import RaonCacheBudget

        return RaonCacheBudget.from_config(self.config, mtp=mtp, mla=self.policy["mla"],
                                           expanded_bytes=self.policy["mla_expanded_gib"] << 30)

    @staticmethod
    def planning_cache_budget(config, *, mtp):
        from .raon1_mini_memory import RaonCacheBudget

        return RaonCacheBudget.from_config(config, mtp=mtp)

    # ------------------------------------------------------------ prompts

    @staticmethod
    def _template_kwargs(request):
        kwargs = dict(request.get("chat_template_kwargs") or {})
        if set(kwargs) - {"enable_thinking", "truncate_history_thinking"}:
            raise ValueError("Raon chat template accepts only enable_thinking and truncate_history_thinking")
        flag = request.get("enable_thinking")
        if type(flag) is bool:
            kwargs["enable_thinking"] = flag
        return kwargs

    def _template(self, request, tokenize):
        from ..output import callable_tools
        from .ordinary_text import _template_history

        tools = None if request.get("tool_choice") == "none" else callable_tools(request) or None
        return self.tokenizer.apply_chat_template(
            _template_history(request["messages"]), tools=tools, add_generation_prompt=True,
            tokenize=tokenize, **self._template_kwargs(request))

    def prompt_tokens(self, request):
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        return self._template(request, True)

    def render_prompt(self, request):
        return request["prompt"] if "messages" not in request else self._template(request, False)

    @classmethod
    def thinking_enabled(cls, request):
        # The template opens "<think>\n" unless enable_thinking is false.
        return "messages" in request and cls._template_kwargs(request).get("enable_thinking") is not False

    def thinking_close_token_ids(self):
        try:
            ids = list(self.tokenizer.encode("</think>", add_special_tokens=False))
        except (AttributeError, TypeError, ValueError):
            return None
        return (int(ids[0]),) if len(ids) == 1 else None

    def output_parser(self, request):
        from ..output import OutputParser, callable_tools, constrained_tool_choice
        from ..runtime.tool_parsers.qwen3_coder import parse_tool_call

        return OutputParser(
            chat="messages" in request, thinking=self.thinking_enabled(request),
            tools=callable_tools(request), parse_tool=parse_tool_call,
            stops=request.get("stop", ()), constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
            think_close_separator="\n")

    def tool_constraint(self, request):
        from ..output import constrained_tool_choice
        from ..runtime.tool_parsers.qwen3_coder import constrained_tool_grammar

        if not constrained_tool_choice(request):
            return None
        return constrained_tool_grammar(request["tools"], request["tool_choice"],
                                        parallel_tool_calls=request.get("parallel_tool_calls", True))

    # ------------------------------------------------------------ receipts

    def capability_receipts(self):
        status = self.model.mechanism_status()
        external = getattr(self, "draft_model", None) is not None
        unq = {"qualified": False, "selection_authority": "serving_route_receipt"}
        return {
            "ordinary_reference": {"implemented": True, "default": True,
                                   "selected": None if not external else False, **unq},
            "external_dflash2_chain": {"implemented": True, "selected": external,
                                       "observed_used": external and status["rda"].get("rollback_records", 0) > 0,
                                       "rollback": "RDA conv closed form + FP32 state one-token replay; MLA kv_trim",
                                       **unq},
            "reasoning_effort_steering": {
                "implemented": True, "scope": "per-model (load-time policy), not per-request",
                "effort": self.policy["reasoning_effort"] or "default",
                "alpha": status["effort_alpha"],
                # The reference (modeling_raon1_mini.set_reasoning_effort),
                # vLLM, llama.cpp and the vendor MLX server steer a request
                # whose prompt turns thinking off with the "none" alpha.  This
                # route applies the load-time alpha to every request, so a
                # thinking-off request matches the reference only when the
                # two are equal.
                "thinking_off": {"reference_alpha": self.model.effort_alpha("none"),
                                 "served_alpha": status["effort_alpha"],
                                 "matches_reference": status["effort_alpha"] == self.model.effort_alpha("none")},
                **unq},
            "mla_flash": {"implemented": "expanded prompt chunks only (flash192 / key windows at 40 heads)",
                          "selected": self.policy["mla_flash"], "qualified": False,
                          "observed_used": int(status["mla"].get("expanded_flash192", 0)
                                               + status["mla"].get("expanded_flash192_windows", 0)),
                          "declined": int(status["mla"].get("flash_declined", 0)),
                          "absorbed_decode": "not implemented: the NAX absorbed kernel's grid covers 32 heads"},
            "moe_nax_gather": {"implemented": "affine artifact only", "selected": self.policy["moe_nax"],
                               "qualified": False},
        }

    def diagnostics(self):
        layout = LAYOUTS[self.artifact_layout]
        external = getattr(self, "draft_model", None) is not None
        return {"route": "external_draft" if external else "ordinary",
                "qualification": "unqualified", "policy": self.policy,
                "artifact": {"layout": self.artifact_layout, "variant": layout["variant"],
                             "checkpoint": self.checkpoint,
                             "source": layout["repo"] + "@" + layout["revision"],
                             "storage": self.storage},
                "max_context": self.max_context, "counters": self.model.route_status(),
                "moe_nax_gather": _nax_status(),
                **self.model.mechanism_status(),
                **({"draft": self.draft_model.receipt_settings,
                    "draft_artifact": getattr(self, "draft_record_receipt", None)} if external else {}),
                "capability_receipts": self.capability_receipts()}

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
        self.tokenizer = None
