"""Pinned Kimi Linear artifacts (REAP BF16, 48B MLX 8-bit) and ordinary serving adapter."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .external_draft_policy import ExternalDraftAdapterMixin
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
                            Capability.APC_V2, Capability.LAYERED_CACHE,
                            # No-flag capability default (cost-latched PLD); the
                            # KDA/conv/MLA planes record exact rollback.
                            Capability.PROMPT_LOOKUP, Capability.TOOLS}),
    cache_layout=CACHE_LAYOUT,
    metadata={"execution": "mlx2.adapters.kimi_linear.KimiLinearAdapter",
              "qualification": "unqualified", "implemented": True,
              "qualified": False, "selected": False, "observed_used": False,
              "scope": "ordinary text; 1M advertised capacity, long-context unqualified"},
)


# Two exact artifact layouts share one topology:
#  hub_bf16   - original HF tensors (KimiRio REAP 35B, 180 experts, BF16/F32);
#  mlx_affine - mlx-lm conversion (mlx-community 48B, 256 experts), MLX
#               affine 8-bit group-64 weights, BF16 scales/biases/norms/convs.
LAYOUTS = {"hub_bf16": {"num_experts": 180, "variant": "reap35b-bf16-ordinary",
                        "repo": REPO, "revision": REVISION},
           "mlx_affine": {"num_experts": 256, "variant": "48b-mlx-q8-ordinary",
                          "repo": "mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit",
                          "revision": "553a7a5673491df2b2d38f7380188a9d8e9f9a66"}}
QUANTIZATION = {"group_size": 64, "bits": 8, "mode": "affine"}


def validate_config(config):
    """Return the artifact layout name, or raise for anything unstaged."""
    quant = config.get("quantization") or config.get("quantization_config")
    layout = "mlx_affine" if quant else "hub_bf16"
    expected = {"model_type": "kimi_linear", "architectures": ["KimiLinearForCausalLM"],
                "hidden_size": 2304, "vocab_size": 163840, "num_hidden_layers": 27,
                "num_experts": LAYOUTS[layout]["num_experts"], "num_experts_per_token": 8,
                "num_shared_experts": 1,
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
        raise ValueError("Kimi Linear topology or math does not match a staged checkpoint")
    full = [4, 8, 12, 16, 20, 24, 27]
    linear = {"full_attn_layers": full, "kda_layers": [i for i in range(1, 28) if i not in full],
              "head_dim": 128, "num_heads": 32, "short_conv_kernel_size": 4}
    if config.get("linear_attn_config") != linear:
        raise ValueError("Kimi layer partition or KDA geometry differs")
    if quant:
        # Global affine 8/64 plus per-module overrides, which must all agree.
        overrides = [v for v in quant.values() if isinstance(v, dict)]
        if (any(quant.get(k) != v for k, v in QUANTIZATION.items())
                or any(o != {"group_size": 64, "bits": 8} for o in overrides)):
            raise ValueError("Kimi quantized artifacts are staged only as MLX affine 8-bit group 64")
    return layout


def _quantized(prefix, name):
    return {prefix+name+suffix for suffix in (".weight", ".scales", ".biases")}


def _required(layout, full):
    if layout == "mlx_affine":
        required = (_quantized("model.", "embed_tokens") | _quantized("", "lm_head")
                    | {"model.norm.weight"})
    else:
        required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(27):
        p = f"model.layers.{i}."
        required.update(p+n for n in ("input_layernorm.weight", "post_attention_layernorm.weight"))
        if layout == "mlx_affine":
            a = p+"self_attn."
            linear = ["q_proj", "o_proj"]
            if i+1 in full:
                linear += ["kv_a_proj_with_mqa", "kv_b_proj"]
                required.add(a+"kv_a_layernorm.weight")
            else:
                linear += ["k_proj", "v_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "b_proj"]
                required.update(a+n for n in ("q_conv.conv.weight", "k_conv.conv.weight",
                                              "v_conv.conv.weight", "o_norm.weight", "A_log", "dt_bias"))
            for name in linear:
                required |= _quantized(a, name)
            m = p+"mlp."
            if i == 0:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    required |= _quantized(m, name)
            else:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    required |= _quantized(m+"switch_mlp.", name) | _quantized(m+"shared_experts.", name)
                required |= _quantized(m, "gate") | {m+"e_score_correction_bias"}
            continue
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
    return required


def _expected_dtype(layout, key, quantized):
    if key.endswith((".A_log", ".dt_bias")):
        return "F32"
    if layout == "mlx_affine" and key.endswith(".weight") and key[:-len(".weight")] in quantized:
        return "U32"
    return "BF16"


def inspect_artifact(model_path):
    artifact = inspect_indexed_artifact(model_path)
    layout = validate_config(artifact["config"])
    path = Path(artifact["identity"]["path"])
    required = _required(layout, set(artifact["config"]["linear_attn_config"]["full_attn_layers"]))
    if set(artifact["weight_map"]) != required:
        raise ValueError("Kimi tensor index differs from the complete staged topology")
    quantized = {k[:-len(".scales")] for k in required if k.endswith(".scales")}
    # Read-only header validation establishes the BF16 cache/conv budget's
    # dtype premise. safe_open validates offsets without loading tensor data.
    from safetensors import safe_open
    observed = set()
    for name in sorted(set(artifact["weight_map"].values())):
        with safe_open(path/name, framework="np") as shard:
            for key in shard.keys():  # noqa: SIM118 - safe_open is not an iterable mapping
                if artifact["weight_map"].get(key) != name:
                    raise ValueError("Kimi tensor index and shard headers disagree")
                if shard.get_slice(key).get_dtype() != _expected_dtype(layout, key, quantized):
                    raise ValueError(f"Kimi tensor dtype differs from the {layout} contract: {key}")
                observed.add(key)
    if observed != required:
        raise ValueError("Kimi shard tensor headers are incomplete")
    # Include all tokenizer semantics, including the local tiktoken vocabulary.
    digest = hashlib.sha256(artifact["identity"]["fingerprint"].encode())
    for name in ("tiktoken.model", "special_tokens_map.json"):
        digest.update(name.encode())
        digest.update((path/name).read_bytes())
    overlay = tokenizer_marker_overlay(path)
    if overlay:
        # The tokenizer names these K2 tool-call ids itself (REAP config).
        digest.update(b"k2_tool_marker_names")
        digest.update(json.dumps(overlay, sort_keys=True).encode())
    artifact["identity"]["fingerprint"] = digest.hexdigest()
    artifact["tokenizer_marker_overlay"] = overlay
    artifact["qualification"] = "unqualified"
    artifact["layout"] = layout
    return artifact


def tokenizer_marker_overlay(path):
    """K2 tool-call marker ids the artifact's tokenizer config leaves unnamed.

    :class:`mlx2.runtime.kimi_tokenizer.TikTokenTokenizer` supplies these
    names; the overlay is bound into the artifact fingerprint.
    """
    from ..runtime.tool_parsers.kimi import K2_TOOL_MARKER_IDS

    config = json.loads((Path(path)/"tokenizer_config.json").read_text())
    named = {int(k) for k in config.get("added_tokens_decoder") or {}}
    return {str(i): name for i, name in K2_TOOL_MARKER_IDS.items() if i not in named}


def checkpoint_name(model_path):
    """``<repo>@<revision>`` for receipts, from the artifact's own layout.

    A derived artifact (e.g. a residual-ablation bake records
    ``mlx2_residual_ablation`` in its config) is never labelled as its
    source: it reports its own directory and the checkpoint it came from.
    """
    config = json.loads((Path(model_path)/"config.json").read_text())
    layout = LAYOUTS[validate_config(config)]
    derived = config.get("mlx2_residual_ablation")
    if isinstance(derived, dict):
        return f"{Path(model_path).resolve().name}@derived-from:{derived.get('source_checkpoint', 'unknown')}"
    return layout["repo"]+"@"+layout["revision"]


NUMERIC_POLICY_KEYS = frozenset({"mla", "kda", "fused_projections", "fused_norm",
                                 "router_combine", "norm_rounding", "rms_law", "moe_nax",
                                 "mla_flash", "mla_flash_law", "tensorfold_qmv"})
PROMPT_LOOKUP_DRAFT_NAMES = frozenset({"prompt_lookup", "pld"})
# The generic bounded tree route (B1-B4 trees, B5+ chain verify).
TREE_ROUTE = "tree15_b1_b4_chain_b5plus_v1"
TREE_POLICY_KEYS = frozenset({"batch_size_route", "ast_topology", "ast_tiles", "ast_tile_rows",
                              "ast_adaptive_rows", "ast_target_executor", "ast_graft_prompt_lookup",
                              "tree_node_budget_by_lanes", "pairwise_selection",
                              "ast_multilane_tile_rows"})


def split_policy(policy=None):
    """``(numerics, speculative)``: Kimi math keys versus draft-route keys."""
    policy = dict(policy or {})
    numerics = {key: policy.pop(key) for key in list(policy) if key in NUMERIC_POLICY_KEYS}
    return numerics, policy


def validate_tree_policy(policy):
    """Tree-route controls (reached through ``--ast`` tree tiles only)."""
    route = policy.get("batch_size_route")
    if route is None:
        if set(policy) & (TREE_POLICY_KEYS - {"pairwise_selection"}):
            raise ValueError("Kimi AST tree controls require the tree batch_size_route (--ast)")
        return False
    if route != TREE_ROUTE:
        raise ValueError(f"Kimi external batch_size_route must be {TREE_ROUTE}")
    if policy.get("ast_target_executor", "packed") not in ("packed", "serial"):
        raise ValueError("Kimi AST target executor must be packed or serial")
    rows = policy.get("ast_tile_rows", 16)
    if type(rows) is not int or not 4 <= rows <= 16:
        raise ValueError("Kimi AST tile rows must be an integer from 4 to 16")
    if policy.get("ast_graft_prompt_lookup") not in (None, False, True):
        raise ValueError("Kimi AST prompt-lookup graft must be boolean")
    if policy.get("pairwise_selection", "host") != "host":
        raise ValueError("Kimi AST uses host pairwise selection")
    return True


def parse_policy(policy=None):
    policy = dict(policy or {})
    # The requested 1M capacity needs a bounded attention workspace. Retain
    # expanded MLA as the explicit ordinary numerical reference. Default
    # "auto" (Pierre, 2026-11-02, after the loopback A/B in
    # docs/research/kimi-linear-20261102/AUDIT-2026-11-02.md): fused expanded
    # prompt chunks, one-pass latent decode. latent_tiled stays selectable.
    defaults = {"mla": "auto", "kda": "kernel", "fused_projections": False,
                "fused_norm": False, "router_combine": "unbiased", "norm_rounding": "fla_fp32",
                # mla_flash "all" default: Pierre, 2026-11-02, after ab-def-q8
                # (decode @103K 43.7 -> 73.9 tok/s, prefill 1315 -> 1607).
                "rms_law": "fused", "moe_nax": "off", "mla_flash": "all",
                # flash192 prompt-chunk kernel law (kimi_linear_mla_flash.
                # EXPANDED_LAWS): v2 = 16-byte row loads + 16x32x32 ops after
                # the QSA NAX law v2 (provenance/kimi-mla-flash-law-v2.json).
                # NOT HELPFUL, left off (2026-11-05): v2 ran 0.39-0.51x v1's
                # speed (v1 ms / v2 ms; ~2-2.5x slower) at every case
                # (8K/32K/128K x 2048/8192 rows, equal error, both chunk-invariant; qualification/runs/
                # kimi-flash192-law-ab-20261105). v1 already sustains 35-43
                # TFLOP/s, so the load/op-width tuning that doubled the QSA
                # kernel (9 TFLOP/s) has no headroom here; the FP32-P x BF16
                # 16x32x32 PV op and the larger per-lane fragments are the
                # likely cost (not profiled).
                "mla_flash_law": "v1",
                # Opt-in TensorFold Flash q4/q8-g64 row matvec (1..17 rows) on
                # the eligible dense projections of the MLX affine artifact.
                "tensorfold_qmv": False}
    if policy.keys()-defaults.keys():
        raise ValueError(f"unsupported Kimi policy: {sorted(policy.keys()-defaults.keys())}")
    result = {**defaults, **policy}
    if result["mla"] not in {"expanded", "latent_tiled", "latent_fused", "latent_single", "auto"} or result["kda"] not in {"reference", "kernel", "chunk32", "chunk_wy"}:
        raise ValueError("unsupported Kimi MLA/KDA implementation")
    if result["mla_flash"] not in {"off", "expanded", "all"}:
        raise ValueError("mla_flash must name off, expanded or all (NAX flash under mla=auto)")
    if result["mla_flash_law"] not in {"v1", "v2"}:
        raise ValueError("mla_flash_law must name v1 or v2 (the flash192 kernel law)")
    if result["router_combine"] not in {"checkpoint", "unbiased"}:
        raise ValueError("router_combine must name checkpoint or unbiased semantics")
    if result["moe_nax"] not in {"off", "gather"}:
        raise ValueError("moe_nax must name off or gather")
    if result["rms_law"] not in {"fused", "reference"}:
        raise ValueError("rms_law must name fused or reference (KimiRMSNorm) rounding")
    if result["norm_rounding"] not in {"fla_fp32", "qwen_bf16"}:
        raise ValueError("norm_rounding must name fla_fp32 or qwen_bf16")
    if any(type(result[k]) is not bool for k in ("fused_projections", "fused_norm", "tensorfold_qmv")):
        raise ValueError("Kimi fusion switches must be boolean")
    return result


def _tensorfold_qmv_counters():
    from ..runtime.models.flash_tensorfold_qmv import counters

    return counters()


class KimiLinearAdapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    # Residual writers of the KDA/MLA hybrid decoder layer (``x + attention``,
    # then ``h + mlp``) for the experimental residual-ablation tooling.  MLA and
    # KDA both end in ``attention.o_proj``; layer 0 is a dense MLP, MoE layers
    # write through stacked routed experts plus the shared expert.  The loader
    # renames checkpoint ``self_attn`` to ``attention`` (sanitize).
    residual_ablation = {
        "writers": {"attention": ("attention",), "mlp": ("mlp",)},
        "linears": {"attention": ("attention.o_proj",),
                    "mlp": ("mlp.down_proj", "mlp.switch_mlp.down_proj", "mlp.shared_experts.down_proj")},
        "checkpoint_linears": {"attention": ("self_attn.o_proj",),
                               "mlp": ("mlp.down_proj", "mlp.switch_mlp.down_proj",
                                       "mlp.shared_experts.down_proj")},
    }
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "kimi-linear-reap35b-bf16-ordinary"
    default_max_context = DEFAULT_CONTEXT
    # Existing live memory admission determines concurrent sessions; a lane
    # ceiling is not a promise to hold that many full 1M contexts.
    default_route = "ordinary"
    # Cost-latched prompt lookup is the no-flag route (Pierre, 2026-11-02):
    # 48B q8 2.27x on copy and 0.97-0.99x on math/chat/extract
    # (docs/research/kimi-linear-speculative-20261102). DFlash and AST trees
    # stay explicit; --ordinary still selects plain decode. Unqualified.
    prompt_lookup_default = True
    # ``--ast`` tree tiles: prompt lookup or the DFlash head, verified by the
    # Kimi packed tree (or the serial one-token reference executor).  Kimi's
    # provider owns its KDA/MLA math; the Qwen TensorFold header gate does
    # not apply to this BF16 artifact.
    ast_tree_tiles = True
    ast_prompt_lookup_tiles = True
    ast_tree_tile_dtype_gate = False
    ast_tile_rows_max = 16
    default_ast_tile_rows = 16
    # Chain depth of an explicit --external-draft DFlash route.  Unmeasured
    # choice; every verify row is a full MoE forward of the target.
    EXTERNAL_DEFAULT_NUM_DRAFT = 7
    EXTERNAL_ROUTE_TAG = "external-kimi-linear-v1"
    EXTERNAL_PROFILE = "kimi-linear-external-draft"
    EXTERNAL_POLICY_EXTENSIONS = TREE_POLICY_KEYS
    # Flash-Next's measured prefill chunk (623b6975, fn-prefill-chunk-size-
    # reversal): fewer per-chunk MLA prefix re-expansions and fuller expert
    # batches. The vocabulary head projects one checkpoint row per chunk.
    default_prefill_step = 8192
    # rows * (depth + rows) per chunk (omlx#4149): 8192-row chunks through
    # ~128K of prefix, halving beyond so one attention dispatch stays well
    # inside the Metal watchdog at 1M (flash192: 166 ms at 128K x 2048).
    default_prefill_depth_budget = 8192 * (131072 + 8192)

    def prefill_step_default(self):
        """Adapter-preferred prefill chunk; an explicit engine setting wins."""
        return int(type(self).default_prefill_step)

    def prefill_depth_budget_default(self):
        return int(type(self).default_prefill_depth_budget)

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
        from mlx import nn

        from .process_globals import claim_moe, claim_stock_moe

        require_process_numerics("Kimi Linear ordinary")
        self.environment = dict(PROCESS_NUMERICS)
        numerics, speculative = split_policy(execution_policy)
        self.policy = parse_policy(numerics)
        external = self._parse_external_policy(speculative, family="Kimi Linear")
        draft = self.external_policy.get("draft_model") if external else None
        self.drafter_kind = (None if draft is None else "prompt_lookup"
                             if draft in PROMPT_LOOKUP_DRAFT_NAMES else "dflash")
        if self.drafter_kind == "prompt_lookup":
            # The model-free drafter exists here only as AST tree tiles
            # (``--ast`` writes no route for it); PLD chains are the
            # separate --prompt-lookup route.
            self.external_policy.setdefault("batch_size_route", TREE_ROUTE)
        self.tree_route = validate_tree_policy(self.external_policy) if external else False
        if self.drafter_kind == "prompt_lookup":
            if self.external_policy.get("ast_graft_prompt_lookup"):
                raise ValueError("Kimi prompt-lookup tiles have no base drafter to graft onto")
            # A copy source pays off as one long run.
            self.external_policy.setdefault("ast_topology", "chain")
        if self.tree_route:
            self.external_policy.setdefault("num_draft", 15)
        if self.drafter_kind == "dflash" and self.external_policy.get("ast_graft_prompt_lookup"):
            raise ValueError("Kimi DFlash prompt-lookup graft is not implemented")
        artifact = inspect_artifact(model_path)
        self.config, self.identity = artifact["config"], artifact["identity"]
        self.artifact_layout = layout = artifact["layout"]
        if layout != "mlx_affine" and self.policy["moe_nax"] != "off":
            raise ValueError("moe_nax gather applies only to the affine-quantized artifact")
        if layout != "mlx_affine" and self.policy["tensorfold_qmv"]:
            # Before any tensor loads: the kernel reads affine q4/q8-g64 only.
            raise ValueError("tensorfold_qmv reads affine q4/q8-g64 projections; "
                             "not applicable to the BF16 artifact")
        if layout == "mlx_affine" and self.policy["fused_projections"]:
            # Grouped KDA projections concatenate dense weights; quantized
            # banks would need their own grouped quantized matmul.
            raise ValueError("fused_projections requires the unquantized BF16 artifact")
        draft_record = None
        if self.drafter_kind == "dflash":
            # Header-only inspection, content pin and tokenizer pairing,
            # before any target tensor loads.
            from .kimi_dflash import inspect_drafter

            draft_record = inspect_drafter(draft, self.identity["path"], target_config=self.config)
            self._check_num_draft(draft_record)
            self.draft_record_receipt = {
                "checkpoint": draft_record["checkpoint"],
                "fingerprint": draft_record["fingerprint"],
                "tokenizer_pairing": draft_record["tokenizer_pairing"],
            }
        elif self.drafter_kind == "prompt_lookup":
            from .qwen38_27b import _prompt_lookup_record

            draft_record = _prompt_lookup_record(self.external_policy, self.identity["fingerprint"])
        # Bind all numerical profiles into the APCv2 state namespace.
        binding = hashlib.sha256(json.dumps(self.policy, sort_keys=True).encode()).hexdigest()
        self.layout = CACHE_LAYOUT+":"+binding
        self.descriptor = replace(DESCRIPTOR, cache_layout=self.layout, variant=LAYOUTS[layout]["variant"],
                                  metadata={**DESCRIPTOR.metadata, "selected": True, "policy": self.policy,
                                            "checkpoint": LAYOUTS[layout]["repo"]+"@"+LAYOUTS[layout]["revision"]})
        self.profile = "kimi-linear-"+LAYOUTS[layout]["variant"]
        if self.policy["moe_nax"] == "gather":
            claim_moe(self, "Kimi Linear MLX 8-bit ordinary", nax_mode="gather")
        else:
            claim_stock_moe(self, "Kimi Linear ordinary")
        path = Path(self.identity["path"])
        self.model = Model(ModelArgs.from_dict(self.config))
        weights = self.model.sanitize(load_shards_evicting(
            [path/n for n in sorted(set(artifact["weight_map"].values()))]))
        if layout == "mlx_affine":
            nn.quantize(self.model, **QUANTIZATION, class_predicate=lambda name, module:
                        hasattr(module, "to_quantized") and f"{name}.scales" in weights)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        self.model.configure(self.policy)
        self.tensorfold_qmv = None
        if self.policy["tensorfold_qmv"]:
            from ..runtime.models.flash_tensorfold_qmv import install as install_tensorfold_qmv

            # Eligible: K % 512 == 0 (o_proj, kv_b_proj, dense/shared down
            # projections). Every K=2304 input projection and the 163840 x
            # 2304 vocabulary head are outside the kernel's 512-column tile.
            self.tensorfold_qmv = install_tensorfold_qmv(self.model)
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = TikTokenTokenizer.from_pretrained(path, local_files_only=True)
        tokenizer.chat_template = (path/"chat_template.jinja").read_text()
        self.tokenizer = TokenizerWrapper(tokenizer, eos_token_ids=[163586])
        self.max_context = DEFAULT_CONTEXT
        if draft_record is not None:
            if self.drafter_kind == "dflash":
                from .kimi_dflash import load_drafter
            else:
                from ..runtime.drafters.prompt_lookup_draft import (
                    load_prompt_lookup_drafter as load_drafter,
                )
            self._bind_external_drafter(draft_record, load_drafter, self.descriptor)
            # Keep the per-artifact profile name, not the mixin's class constant.
            self.__dict__.pop("profile_name", None)
            stem = LAYOUTS[layout]["variant"].removesuffix("-ordinary")
            self.descriptor = replace(self.descriptor, variant=stem+"-"+self._speculative_route(),
                                      metadata={**self.descriptor.metadata, "qualification": "unqualified",
                                                "implemented": True, "qualified": False, "selected": True,
                                                "observed_used": False,
                                                "speculative_route": self._speculative_route()})

    def _speculative_route(self):
        if getattr(self, "draft_model", None) is None:
            return None
        kind = "kimi_dflash" if self.drafter_kind == "dflash" else "prompt_lookup"
        if not self.tree_route:
            return f"external_{kind}_chain"
        return f"ast_{kind}_tree_" + self.external_policy.get("ast_target_executor", "packed")

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Kimi Linear has no MTP route")
        if getattr(self, "draft_model", None) is not None:
            return self.profile.removesuffix("-ordinary")+"-external-draft"
        return self.profile

    def execution_config(self, *, max_lanes, prefill_step):
        # Exact idle-plan prompt slices: MLA auto/flash and any chunked KDA
        # choose by chunk rows, so load must not reshape a slice or batch
        # prompts, or APCv2 would publish load-dependent state bits.
        if getattr(self, "draft_model", None) is None:
            return {**super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step),
                    "prefill_plan_exact": True}
        config = {**self._external_execution_config(max_lanes=max_lanes, prefill_step=prefill_step),
                  "prefill_plan_exact": True,
                  "kimi_speculative_route": self._speculative_route()}
        if self.tree_route:
            from .qwen38_27b import _tile_node_budgets

            config.update(batch_size_route=TREE_ROUTE, dynamic_tree_overflow="chain",
                          tree_node_budget_by_lanes=_tile_node_budgets(
                              self.external_policy, 4, None, self.default_ast_tile_rows),
                          ast_target_executor=self.external_policy.get("ast_target_executor", "packed"))
        return config

    def create_external_batch(self, **kwargs):
        if self.tree_route:
            from ..runtime.hybrid_tree_executor import HybridTreeExecutor
            from ..runtime.kimi_tree_executor import KimiTreeExecutor
            from .qwen38_27b import _tile_node_budgets

            serial = self.external_policy.get("ast_target_executor", "packed") == "serial"
            kwargs.setdefault("dynamic_singleton_tree", True)
            kwargs.setdefault("dynamic_tree_max_width", 4)
            kwargs.setdefault("dynamic_tree_overflow", "chain")
            kwargs.setdefault("target_tree_executor_provider",
                              (HybridTreeExecutor if serial else KimiTreeExecutor).create())
            kwargs.setdefault("tree_node_budget_by_lanes", _tile_node_budgets(
                self.external_policy, 4, None, self.default_ast_tile_rows))
            for key in ("ast_topology", "ast_adaptive_rows"):
                if key in self.external_policy:
                    kwargs.setdefault(key, self.external_policy[key])
        return super().create_external_batch(**kwargs)

    def bind_serving_lane_matmul(self, receipt):
        """TensorFold/lane verdict for this BF16 checkpoint (fail closed).

        Raw TensorFold (``--lane-matmul tensorfold``) tiles affine-quantized
        ``QuantizedLinear`` weights only; this artifact has none, so it is
        refused as not applicable.  mlx2's own lane law (``exact`` or
        ``crossover``) covers the dense BF16 ``nn.Linear`` projections but
        not the 180-expert ``gather_mm`` experts, the absorbed MLA latent
        matmuls, the FP32 router or the KDA kernel.  No complete
        row-invariant projection law exists for this model, so a lane
        install is a performance option only and never makes verify rows
        bit-equal to one-row decode; the packed tree does not require it.
        """
        receipt = receipt or {}
        raw = str(receipt.get("law_id", "")).startswith("tensorfold-lane-qmm")
        if raw and getattr(self, "artifact_layout", "hub_bf16") == "hub_bf16":
            raise ValueError("TensorFold lane_qmm is not applicable to the BF16 Kimi artifact "
                             "(no affine-quantized projections); use --lane-matmul exact, crossover or off")
        # Raw TensorFold re-tiles packed banks in place; kv_b (absorbed MLA)
        # and the router are read through runtime.models.kimi_linear
        # .dense_weight, which untiles them (degenerate text on every prompt
        # before that fix, q8-lane-tf 2026-11-02). mlx2's lane law reads the q8/g64 QuantizedLinear projections
        # (vocabulary head included); the stacked experts (QuantizedSwitchLinear
        # gather_qmm) stay on stock kernels, so no row-invariant target law.
        self.lane_projection_law = {
            "law_id": receipt.get("law_id", "stock"),
            "covered": receipt.get("covered", {}),
            "uncovered": receipt.get("uncovered"),
            "row_invariant_target": False,
            "uncovered_by_construction": ["moe_switch_experts_gather", "mla_latent_absorbed_matmul",
                                          "router", "kda_vector_kernel"],
        }

    def _chat_template(self, request, *, tokenize):
        from ..output import callable_tools
        from .kimi_linear_output import normalize_call_ids
        from .ordinary_text import _template_history

        # tool_choice "none" withholds the declarations, as the call parser
        # then has no tools to bind.
        tools = None if request.get("tool_choice") == "none" else callable_tools(request) or None
        return self.tokenizer.apply_chat_template(
            normalize_call_ids(_template_history(request["messages"])), tools=tools,
            add_generation_prompt=True, tokenize=tokenize)

    def prompt_tokens(self, request):
        if "messages" in request:
            return self._chat_template(request, tokenize=True)
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request):
        return self._chat_template(request, tokenize=False) if "messages" in request else request["prompt"]

    def output_parser(self, request):
        from .kimi_linear_output import make_output_parser
        return make_output_parser(request)

    def tool_constraint(self, request):
        from .kimi_linear_output import tool_constraint
        return tool_constraint(request)

    @staticmethod
    def planning_cache_budget(config, *, mtp):
        from .kimi_linear_memory import KimiLinearCacheBudget
        return KimiLinearCacheBudget.from_config(config, mtp=mtp)

    def cache_budget(self, *, mtp):
        # An external drafter owns additional draft KV (5 layers x 2 x 2304
        # BF16 per committed token); keep the existing admission path for it.
        if getattr(self, "draft_model", None) is not None:
            return None
        from .kimi_linear_memory import KimiLinearCacheBudget
        return KimiLinearCacheBudget.from_config(self.config, mtp=mtp, mla=self.policy["mla"])

    def capability_receipts(self):
        """Adapter-local route states; serving and speculative receipts own selection."""
        status = self.model.mechanism_status()
        kda, mla = status["kda"], status["mla"]
        route = self._speculative_route()
        external = route is not None
        tree_calls = int(kda.get("tree_calls", 0)) + int(mla.get("tree_calls", 0))
        # The no-flag route as resolve_route_selection derives it: an
        # "ordinary" adapter default becomes prompt lookup when the
        # descriptor declares it and the class opts in.
        descriptor = getattr(self, "descriptor", DESCRIPTOR)
        no_flag = type(self).default_route
        if (no_flag == "ordinary" and Capability.PROMPT_LOOKUP in descriptor.capabilities
                and getattr(type(self), "prompt_lookup_default", True)):
            no_flag = "prompt_lookup"
        lane = getattr(self, "lane_projection_law", None)
        if getattr(self, "artifact_layout", "hub_bf16") == "hub_bf16":
            lane_scope = "generic mlx2 lane law on dense BF16 nn.Linear only"
            raw_tensorfold = "not_applicable_bf16_no_quantized_linear"
        else:
            lane_scope = ("mlx2 lane law or raw TensorFold on q8/g64 QuantizedLinear projections; "
                          "switch experts, absorbed MLA latent, router and KDA stay stock")
            raw_tensorfold = ("selected_q8_projections_experts_stock"
                              if lane and str(lane.get("law_id", "")).startswith("tensorfold-lane-qmm")
                              else "available_not_selected")
        return {
            "ordinary_reference": {"implemented": True, "qualified": False, "default": no_flag == "ordinary",
                                   "selected": None if not external else False,
                                   "selection_authority": "serving_route_receipt"},
            "prompt_lookup": {"implemented": True, "qualified": False, "default": no_flag == "prompt_lookup",
                              "selected": None if not external else False,
                              "selection_authority": "serving_route_receipt",
                              "observed_used": (not external and kda.get("rollback_records", 0) > 0) or None,
                              "rollback": "KDA conv closed form + FP32 state one-token replay; MLA kv_trim"},
            "external_dflash_chain": {"implemented": True, "qualified": False,
                                      "selected": route == "external_kimi_dflash_chain",
                                      "observed_used": route == "external_kimi_dflash_chain"
                                      and kda.get("rollback_records", 0) > 0},
            "ast_tree": {"implemented": True, "qualified": False,
                         "selected": bool(external and self.tree_route),
                         "observed_used_packed_tree_calls": tree_calls,
                         "drafters": ["prompt_lookup", "kimi_dflash"],
                         "executors": ["kimi_linear_layerwise_packed_tree_v1",
                                       "hybrid_tree_serial_exact_v1"]},
            "tensorfold_lane": {"implemented": lane_scope,
                                "qualified": False,
                                "selected": lane,
                                "raw_tensorfold": raw_tensorfold},
        }

    def diagnostics(self):
        from ..runtime.models import moe_nax_gather

        layout = LAYOUTS[self.artifact_layout]
        external = getattr(self, "draft_model", None) is not None
        return {"route": "external_draft" if external else "ordinary",
                "speculative_route": self._speculative_route(),
                "qualification": "unqualified", "policy": self.policy,
                "artifact": {"layout": self.artifact_layout, "variant": layout["variant"],
                             "checkpoint": layout["repo"]+"@"+layout["revision"]},
                "max_context": self.max_context, "counters": self.model.route_status(),
                **self.model.mechanism_status(), "moe": self.model.moe_status(),
                "moe_nax_gather": moe_nax_gather.status(),
                **({"draft": self.draft_model.receipt_settings,
                    "draft_artifact": getattr(self, "draft_record_receipt", None),
                    "external_policy": {k: v for k, v in self.external_policy.items()
                                        if k != "draft_model"}}
                   if external else {}),
                "capability_receipts": self.capability_receipts(),
                **({"tensorfold_qmv": {**self.tensorfold_qmv, "counters": _tensorfold_qmv_counters()}}
                   if getattr(self, "tensorfold_qmv", None) else {}),
                "nax": {"custom_quantized_gather": ("policy_" + self.policy["moe_nax"]
                                                    if self.artifact_layout == "mlx_affine"
                                                    else "not_applicable_original_bf16"),
                        "dense_and_gather_mm": "installed MLX hardware dispatch; not observed by this receipt"}}
