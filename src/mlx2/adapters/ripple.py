# SPDX-License-Identifier: Apache-2.0
"""Ripple Preview: strict source inspection and explicit W4A16 ordinary route.

Provenance: provenance/ripple-65b.json. Importing this module is GPU-free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .external_draft_policy import ExternalDraftAdapterMixin
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter, _template_history

REPO = "ricursiveAI/Ripple-65B-Preview-NVFP4"
REVISION = "99eea4477a001a41b86914e41feb4b2ed86105e5"
BF16_REPO = "ricursiveAI/Ripple-65B-Preview"
BF16_REVISION = "240d5966cbe00f0627c699e59ef58560b60303cf"
CACHE_LAYOUT = "ripple-perinorm-valres-w4a16-swa1024-full-kv-v1"
DESCRIPTOR = ModelDescriptor(
    model_type="ricursive", family="ripple", variant="65b-preview-nvfp4-w4a16",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING, Capability.PREFIX_REUSE,
                            # Batched rotating/full KV with per-lane RoPE
                            # offsets; greedy-exact against per-request
                            # decode on CPU (tests/test_ripple_batch_cpu.py).
                            Capability.CONTINUOUS_BATCH,
                            Capability.APC_V2, Capability.LAYERED_CACHE, Capability.REASONING,
                            # Attention-only (rotating + full KV with exact
                            # speculative rollback): the generic PLD verify
                            # path (tests/test_ripple_pld_cpu.py). Explicit
                            # --prompt-lookup only; see prompt_lookup_default.
                            Capability.PROMPT_LOOKUP}),
    cache_layout=CACHE_LAYOUT,
    metadata={"execution": "mlx2.adapters.ripple.RippleAdapter", "implemented": True,
              "qualification": "unqualified", "qualified": False, "selected": False,
              "observed_used": False, "ordinary_reference": True,
              "source_activation_precision": "nvfp4", "activation_precision": "bf16",
              "explicit_execution_policy_required": {"activation_precision": "bf16"},
              "scope": "ordinary W4A16; source W4A4, tools, batching, prompt lookup and long context unqualified"},
)


def storage_format(config):
    quant = config.get("quantization_config")
    if isinstance(quant, dict) and quant.get("quant_algo") == "NVFP4":
        return "nvfp4"
    if config.get("quantization"):
        return "affine"
    if quant:
        raise ValueError("unsupported Ripple quantization")
    return "bf16"


def affine_specs(config):
    """Resolve the complete per-module affine law; reject unknown overrides."""
    quant = config["quantization"]
    if quant != config.get("quantization_config", quant):
        raise ValueError("Ripple quantization aliases disagree")
    dense = tensor_schema({k: v for k, v in config.items()
                           if k not in {"quantization", "quantization_config"}})
    modules = {name.removesuffix(".weight"): shape for name, (_, shape) in dense.items()
               if name.endswith(".weight") and len(shape) == 2}
    base = {k: quant.get(k) for k in ("bits", "group_size", "mode")}
    if set(quant) - set(base) - set(modules):
        raise ValueError("unknown Ripple affine module override")
    result = {}
    for name, (_, cols) in modules.items():
        spec = quant.get(name, base)
        if (not isinstance(spec, dict) or set(spec) != set(base)
                or type(spec["bits"]) is not int or spec["bits"] not in {2, 3, 4, 5, 6, 8}
                or type(spec["group_size"]) is not int or spec["group_size"] not in {32, 64, 128}
                or spec["mode"] != "affine" or cols % spec["group_size"]
                or cols * spec["bits"] % 32):
            raise ValueError(f"unsupported Ripple affine recipe: {name}")
        result[name] = dict(spec)
    return result


def validate_config(config):
    expected = {
        "model_type": "ricursive", "architectures": ["RicursiveForCausalLM"],
        "hidden_size": 5120, "intermediate_size": 40960, "num_hidden_layers": 80,
        "num_attention_heads": 80, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 100352, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "sliding_window": 1024, "use_valres": True, "use_gated_attention": True,
        "gate_full_precision": True, "tie_word_embeddings": False,
        "embed_scale": 71.55417527999327, "ffn_clamp_threshold": 20.0,
        "hidden_act": "silu", "is_hybrid_swa": True,
        "max_position_embeddings": 2097152, "bos_token_id": 100257,
        "eos_token_id": [100265, 100257], "pad_token_id": 100277,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Ripple topology or numerical settings differ from the supported export")
    pattern = [1, 1, 1, 1, 0] * 16
    if (config.get("hybrid_layer_pattern") != pattern or config.get("layer_types") !=
            ["sliding_attention" if x else "full_attention" for x in pattern]):
        raise ValueError("Ripple requires the source four-local/one-global layer pattern")
    allowed = set(expected) | {"auto_map", "transformers_version", "use_cache",
                              "hybrid_layer_pattern", "layer_types", "quantization_config",
                              "quantization", "dtype", "torch_dtype"}
    if set(config) - allowed:
        raise ValueError(f"unsupported Ripple config fields: {sorted(set(config) - allowed)}")
    if any(config.get(k, "bfloat16") != "bfloat16" for k in ("dtype", "torch_dtype")):
        raise ValueError("Ripple requires BF16 floating tensors")
    kind = storage_format(config)
    if kind == "affine":
        affine_specs(config)
        return
    if kind == "bf16":
        if "quantization" in config or "quantization_config" in config:
            raise ValueError("ambiguous empty Ripple quantization recipe")
        return
    if "quantization" in config:
        raise ValueError("ambiguous Ripple NVFP4 and affine recipes")
    quant = config["quantization_config"]
    recipe = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
    if (quant.get("quant_algo") != "NVFP4" or quant.get("quant_method") != "modelopt"
            or quant.get("config_groups") != {"group_0": {
                "input_activations": recipe, "weights": recipe, "targets": ["Linear"]}}
            or set(quant.get("ignore", [])) != {"lm_head", "model.embed_tokens", *(
                f"model.layers.{i}.self_attn*" for i in range(80))}
            or set(quant) - {"quant_algo", "quant_method", "config_groups", "ignore", "producer"}):
        raise ValueError("Ripple requires the published ModelOpt NVFP4 MLP recipe")


def tensor_schema(config):
    """Complete source tensor schema, also usable with tiny CPU fixtures."""
    h, f, d = config["hidden_size"], config["intermediate_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    schema = {"model.embed_tokens.weight": ("BF16", [config["vocab_size"], h]),
              "model.norm.weight": ("BF16", [h]),
              "lm_head.weight": ("BF16", [config["vocab_size"], h])}
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            schema[p + norm + ".weight"] = ("BF16", [h])
        for proj, shape in {"q_proj": [q, h], "k_proj": [kv, h], "v_proj": [kv, h],
                            "o_proj": [h, q], "g_proj": [q, h]}.items():
            schema[p + "self_attn." + proj + ".weight"] = ("BF16", shape)
        for norm in ("q_norm", "k_norm"):
            schema[p + "self_attn." + norm + ".weight"] = ("BF16", [d])
        schema[p + "self_attn.value_res_lambda"] = ("BF16", [1])
        for proj, (rows, cols) in {"gate_proj": (f, h), "up_proj": (f, h), "down_proj": (h, f)}.items():
            prefix = p + "mlp." + proj
            if storage_format(config) == "nvfp4":
                schema[prefix + ".weight"] = ("U8", [rows, cols // 2])
                schema[prefix + ".weight_scale"] = ("F8_E4M3", [rows, cols // 16])
                schema[prefix + ".weight_scale_2"] = ("F32", [])
                schema[prefix + ".input_scale"] = ("F32", [])
            else:
                schema[prefix + ".weight"] = ("BF16", [rows, cols])
    if storage_format(config) == "affine":
        for name, spec in affine_specs(config).items():
            _, (rows, cols) = schema[name + ".weight"]
            schema[name + ".weight"] = ("U32", [rows, cols * spec["bits"] // 32])
            for suffix in (".scales", ".biases"):
                schema[name + suffix] = ("BF16", [rows, cols // spec["group_size"]])
    return schema


def validate_headers(path, weight_map, schema):
    from ..conversion.nvfp4 import read_header

    if set(weight_map) != set(schema):
        raise ValueError("Ripple index differs from the complete tensor topology")
    seen = set()
    headers = {}
    for shard in sorted(set(weight_map.values())):
        header, payload = read_header(Path(path) / shard)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name not in schema or weight_map[name] != shard or name in seen:
                raise ValueError(f"Ripple index/header mismatch: {name}")
            if (entry["dtype"], entry["shape"]) != schema[name]:
                raise ValueError(f"Ripple tensor dtype/shape differs: {name}")
            seen.add(name)
        headers[shard] = (header, payload)
    if seen != set(schema):
        raise ValueError("Ripple shard headers are incomplete")
    return headers


def inspect_artifact(model_path):
    artifact = inspect_indexed_artifact(model_path)
    validate_config(artifact["config"])
    path = Path(artifact["identity"]["path"])
    headers = validate_headers(path, artifact["weight_map"], tensor_schema(artifact["config"]))
    # Header content, tokenizer special-token semantics and numerical profile
    # supplement the standard metadata/stat identity. No remote code executes.
    digest = hashlib.sha256(artifact["identity"]["fingerprint"].encode())
    digest.update(json.dumps(headers, sort_keys=True).encode())
    digest.update((path / "special_tokens_map.json").read_bytes())
    kind = storage_format(artifact["config"])
    layout = CACHE_LAYOUT if kind == "nvfp4" else CACHE_LAYOUT.replace("w4a16", kind)
    digest.update(layout.encode())
    artifact["identity"]["fingerprint"] = digest.hexdigest()
    artifact.update(qualification="unqualified", storage_format=kind, cache_layout=layout,
                    source_activation_precision="nvfp4" if kind == "nvfp4" else "bf16",
                    activation_precision="bf16", nvfp4_modules=240 if kind == "nvfp4" else 0)
    return artifact


def artifact_descriptor(artifact, *, selected=False):
    kind = artifact["storage_format"]
    return replace(DESCRIPTOR, variant=f"65b-preview-{kind}-bf16",
        cache_layout=artifact["cache_layout"], metadata={**DESCRIPTOR.metadata,
            "selected": selected, "source_activation_precision": artifact["source_activation_precision"],
            "explicit_execution_policy_required": {"activation_precision": "bf16"} if kind == "nvfp4" else None,
            "scope": "ordinary BF16 activations; model serving unqualified"})


def source_weights(artifact):
    """Stream one tensor at a time; keep FP8 byte codes and packed FP4 exact."""
    import mlx.core as mx
    import numpy as np

    from ..conversion.nvfp4 import decode_e4m3, read_header
    from ..runtime.ubc_evict import ubc_evict

    path = Path(artifact["identity"]["path"])
    identities = {name: (size, mtime) for name, size, mtime in artifact["identity"].get("files", [])}
    for shard in sorted(set(artifact["weight_map"].values())):
        initial = (path / shard).stat()
        observed = (initial.st_size, initial.st_mtime_ns)
        if identities and identities.get(shard) != observed:
            raise ValueError(f"Ripple shard changed after inspection: {shard}")
        header, payload = read_header(path / shard)
        try:
            with (path / shard).open("rb") as stream:
                for name, entry in header.items():
                    if name == "__metadata__":
                        continue
                    start, end = entry["data_offsets"]
                    stream.seek(payload + start)
                    raw = stream.read(end - start)
                    if len(raw) != end - start:
                        raise ValueError(f"Ripple tensor truncated during load: {name}")
                    if entry["dtype"] == "F32":
                        scalar = np.frombuffer(raw, "<f4")
                        if scalar.size != 1 or not np.isfinite(scalar).all() or scalar.item() <= 0:
                            raise ValueError(f"Ripple NVFP4 scale must be finite and positive: {name}")
                        if name.endswith(".input_scale"):
                            # Validated but unused by the explicitly selected W4A16 law.
                            continue
                        name = name.removesuffix(".weight_scale_2") + ".global_scale"
                        value = mx.array(scalar.reshape(()))
                    elif entry["dtype"] == "F8_E4M3":
                        array = np.frombuffer(raw, np.uint8).reshape(entry["shape"])
                        decode_e4m3(array)  # Reject negative/NaN scale codes.
                        value = mx.array(array)
                        name = name.removesuffix(".weight_scale") + ".scales"
                    elif entry["dtype"] == "U8":
                        shape = [entry["shape"][0], entry["shape"][1] // 4]
                        value = mx.array(np.frombuffer(raw, "<u4").reshape(shape))
                    elif entry["dtype"] == "BF16":
                        value = mx.array(np.frombuffer(raw, "<u2").reshape(entry["shape"])).view(mx.bfloat16)
                    elif entry["dtype"] == "U32":
                        value = mx.array(np.frombuffer(raw, "<u4").reshape(entry["shape"]))
                    else:
                        raise ValueError(f"unsupported Ripple tensor dtype: {entry['dtype']}")
                    mx.eval(value)
                    del raw
                    yield name, value
            final = (path / shard).stat()
            if (final.st_size, final.st_mtime_ns) != observed:
                raise ValueError(f"Ripple shard changed during loading: {shard}")
        finally:
            ubc_evict(str(path / shard))


# Adapter-owned execution options (default off; see docs/ripple-65b.md):
#   fused_elementwise  per-layer elementwise chains in compiled regions.
#                      Kept OFF: bit-identical to eager on CPU and Metal
#                      (real widths, 1 and 512 rows) and in serving, but no
#                      measurable gain: decode 10.72 vs 10.67 t/s eager on
#                      oQ4e (qualification/runs/ripple-perf-ab-20261104/run3).
#   gqa_folded_attention  decode/verify rows (<= 16) run GQA-folded SDPA
#                      (one KV read per head); not bit-identical, default
#                      off pending a near-tie-classified GPU A/B.
#   metal_rmsnorm      one-launch RMSNorm kernel (source rounding law, FP32
#                      sum in a different order); default off pending GPU
#                      exactness, A/B and near-tie classification.
OPTIONS = {"fused_elementwise": bool, "gqa_folded_attention": bool, "metal_rmsnorm": bool,
           "metal_glue": bool, "window_blocked_prefill": bool}

# Default-on options per storage format (Pierre, 2026-11-04; selected
# unqualified). Measured only on the affine oQ4e checkpoint:
#   metal_rmsnorm         plain decode +1.8/+3.1%, DFlash2 n7 prose +7.1%
#                         (qualification/runs/ripple-perf-ab-20261104/rmsnorm)
#   gqa_folded_attention  DFlash2 n7 prose +4.9% interleaved (window7)
# Token differences fall only at previously measured near-tie positions;
# their two-arm log-prob windows are still owed. NVFP4 and BF16 keep them
# opt-in until measured. An explicit False in the policy opts out.
#   metal_glue            (Pierre, 2026-11-05) QK-norm+RoPE, SwiGLU and value
#                         blend as one-launch kernels;
#                         bit-identical to the metal_rmsnorm law (gate: 0
#                         mismatches), DFlash2 n7 prose +1.7%, prefill +8.4%
#                         (qualification/runs/ripple-glue-20261105/run2).
#   window_blocked_prefill stays OFF: -49% sliding attention per layer at
#                         2048-row chunks, but only -11% at the default 512-row
#                         schedule, where attention is ~7% of prefill; serving
#                         prefill 263.7 vs 266.6/258.0 rows/s (no gain).
DEFAULT_OPTIONS = {"affine": {"metal_rmsnorm": True, "gqa_folded_attention": True, "metal_glue": True}}


def resolved_options(kind, options):
    """Explicit options over the storage format's defaults."""
    return {**DEFAULT_OPTIONS.get(kind, {}), **options}


NUMERICS_KEYS = ("activation_precision",)


def split_policy(policy):
    """``(numerics, adapter options, external-draft policy)``.

    Every key that is neither numerics nor a Ripple option goes to the
    external-draft parser, which refuses unknown keys and any external key
    without a ``draft_model``.
    """
    numerics, options = split_options(policy)
    external = {k: v for k, v in numerics.items() if k not in NUMERICS_KEYS}
    numerics = {k: v for k, v in numerics.items() if k in NUMERICS_KEYS}
    return numerics, options, external


def split_options(policy):
    """``(numerics policy, adapter options)``; unknown keys fail closed."""
    policy = dict(policy or {})
    options = {}
    for key, kind in OPTIONS.items():
        if key in policy:
            value = policy.pop(key)
            if type(value) is not kind:
                raise ValueError(f"Ripple option {key} must be a {kind.__name__}")
            options[key] = value
    return policy, options


def parse_policy(policy):
    if policy != {"activation_precision": "bf16"}:
        raise ValueError(
            "Ripple source declares W4A4. This adapter implements unqualified W4A16 only; "
            "select execution_policy={'activation_precision': 'bf16'} explicitly"
        )
    return dict(policy)


TREE_TARGET = "attention_tree_v1"
TREE_ROUTE_TAG = "external-ripple-dflash2-ast-tiles-v1"
# Keys ``--ast`` writes for the tree route (server.apply_ast_invocation) plus
# the tile knobs an operator may pass by hand; validated in ``tree_policy``.
TREE_POLICY_KEYS = frozenset({
    "batch_size_route", "ast_topology", "ast_tiles", "ast_tile_rows",
    "ast_multilane_tile_rows", "tree_node_budget_by_lanes", "ast_adaptive_rows",
    "ast_tree_calibration", "ast_batch_row_sharing", "ast_graft_prompt_lookup",
})


def tree_policy(policy):
    """Validate the explicit AST tree-tile keys of an external policy.

    Returns the tree keys present (empty for the plain DFlash2 chain).  Tile
    budgets follow the shared ``ast_tiles``/``ast_tile_rows`` law; the
    prompt-lookup graft has no Ripple implementation and is refused.
    """
    from .qwen38_27b import _TREE_BATCH_ROUTES, _tile_node_budgets

    tree = {key: policy[key] for key in TREE_POLICY_KEYS if key in policy}
    if not tree:
        return tree
    route = tree.get("batch_size_route")
    if route not in _TREE_BATCH_ROUTES:
        raise ValueError("Ripple tree tiles need batch_size_route tree15_b1_chain_b2plus_v1 "
                         "or tree15_b1_b4_chain_b5plus_v1 (--ast selects it)")
    if tree.get("ast_graft_prompt_lookup"):
        raise ValueError("Ripple tree tiles do not graft prompt lookup onto the DFlash2 lattice")
    adaptive = tree.get("ast_adaptive_rows")
    if adaptive is not None and type(adaptive) is not bool:
        raise ValueError("ast_adaptive_rows must be a boolean")
    if "ast_topology" in tree:
        from ..runtime.ast_tree_proposals import parse_topology

        parse_topology(tree["ast_topology"])
    _tile_node_budgets(tree, _TREE_BATCH_ROUTES[route], None, 16)
    return tree


class RippleAdapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    # Vendor DFlash2 drafter (ricursiveAI/Ripple-65B-Preview-DFlash2): taps
    # target blocks 1/20/39/58/77 before the final norm. Depth unmeasured on
    # Apple silicon; 7 is a starting point for the A/B, not a tuned default.
    EXTERNAL_DEFAULT_NUM_DRAFT = 7
    EXTERNAL_ROUTE_TAG = "external-ripple-dflash2-v1"
    EXTERNAL_PROFILE = "ripple-apcv2-dflash2"
    # Identity pin an auto-paired default writes (best_route_defaults); the
    # adapter re-checks it below so a drafter changed after pairing fails.
    EXTERNAL_POLICY_EXTENSIONS = frozenset({"draft_revision"}) | TREE_POLICY_KEYS
    # AST tree tiles (``--ast --ast-drafter <DFlash2 path>``): the DFlash2
    # lattice is verified as one parent-closed tile per round on the generic
    # attention tree target (runtime/attention_tree_executor.py), with the
    # sliding layers rotated at ``offset + depth`` and the global layers
    # NoPE (Model.attention_tree_position_scope).  CPU-exact against ordinary
    # greedy (tests/test_ripple_ast_cpu.py).  The oQ4e no-flag route is this
    # tree route at 8 tile rows (best_route_defaults.json, Pierre 2026-11-05;
    # selected unqualified): prose 20.3 vs chain n7 17.5/16.8 t/s.
    ast_tree_tiles = True
    attention_tree_tiles = True
    # Generic attention tiles keep the model dtype; the TensorFold GDN
    # BF16-only gate does not apply.
    ast_tree_tile_dtype_gate = False
    # ``--ast-drafter pld`` keeps Ripple's ordinary prompt-lookup contract
    # (Capability.PROMPT_LOOKUP); the tree drafter here is DFlash2.
    ast_prompt_lookup_tiles = False
    default_ast_tile_rows = 16
    profile = "ripple-nvfp4-w4a16-apcv2-ordinary"
    # Wider lanes are admitted (--max-lanes); the no-flag width stays one
    # until a GPU A/B selects a default.
    default_max_lanes = 1
    # PLD is CPU-exact across window wraps but has no GPU A/B on a Ripple
    # artifact yet, so it is not the no-flag route. Flip after an owned A/B
    # and the user's approval (AGENTS.md default-change rule).
    prompt_lookup_default = False

    def execution_config(self, *, max_lanes, prefill_step):
        if getattr(self, "draft_model", None) is not None:
            return self._external_execution_config(max_lanes=max_lanes, prefill_step=prefill_step)
        return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def __init__(self, model_path, *, execution_policy=None):
        from .process_globals import guarded_construction

        self.policy = execution_policy
        guarded_construction(self, lambda: self._load(model_path))

    def _load(self, model_path):
        # All artifact and policy checks precede tensor allocation.
        artifact = inspect_artifact(model_path)
        self.config, self.identity = artifact["config"], artifact["identity"]
        self.kind = artifact["storage_format"]
        numerics, explicit, external = split_policy(self.policy)
        self.options = resolved_options(self.kind, explicit)
        self.option_sources = {key: "explicit" if key in explicit else "storage_format_default"
                               for key in self.options}
        draft_record = None
        self.tree = {}
        if self._parse_external_policy(external, family="Ripple"):
            # Tree-tile keys fail closed here, before any tensor loads; the
            # plain chain carries none.
            self.tree = tree_policy(self.external_policy)
            # Header-only drafter inspection before any target tensor loads.
            from .dflash2 import inspect_drafter

            draft_record = inspect_drafter(self.external_policy["draft_model"], artifact["identity"]["path"])
            pin = self.external_policy.pop("draft_revision", None)
            if pin is not None:
                from .dflash2 import content_revision

                if content_revision(draft_record) != pin:
                    raise ValueError("Ripple drafter content differs from its pinned draft_revision")
            self._check_num_draft(draft_record)
        if self.kind == "nvfp4":
            self.policy = parse_policy(numerics)
        elif numerics not in ({}, {"activation_precision": "bf16"}):
            raise ValueError("Ripple BF16/affine accepts only BF16 activations")
        else:
            self.policy = {"activation_precision": "bf16"}
        # Options that change arithmetic (not bit-identical to eager) get
        # their own APCv2 namespace, so a prefix cached under one law is
        # never warm-hit under another.
        # metal_glue is bit-identical to the metal_rmsnorm law, so it shares
        # that namespace; the others change arithmetic.
        numerics_options = sorted(k for k in ("metal_rmsnorm", "gqa_folded_attention", "window_blocked_prefill")
                                  if self.options.get(k))
        self.layout = artifact["cache_layout"] + ("+" + "+".join(numerics_options) if numerics_options else "")
        self.descriptor = replace(artifact_descriptor(artifact, selected=True), cache_layout=self.layout)
        self.profile = f"ripple-{self.kind}-apcv2-ordinary"

        import mlx.core as mx
        from transformers import PreTrainedTokenizerFast

        from ..process_env import PROCESS_NUMERICS, require_process_numerics
        from ..runtime.models.ricursive import Model, ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

        require_process_numerics("Ripple W4A16 ordinary")
        self.environment = dict(PROCESS_NUMERICS)
        self.model = Model(ModelArgs.from_dict(self.config), nvfp4=self.kind == "nvfp4")
        if self.kind == "affine":
            from mlx import nn

            specs = affine_specs(self.config)
            nn.quantize(self.model, class_predicate=lambda name, module: specs.get(name, False))
        self.model.set_fused_elementwise(self.options.get("fused_elementwise", False))
        self.model.set_gqa_folded_attention(16 if self.options.get("gqa_folded_attention") else 0)
        self.model.set_metal_rmsnorm(self.options.get("metal_rmsnorm", False))
        self.model.set_metal_glue(self.options.get("metal_glue", False))
        self.model.set_window_blocked_prefill(self.options.get("window_blocked_prefill", False))
        weights = list(source_weights(artifact))
        self.model.load_weights(weights, strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        path = Path(self.identity["path"])
        tokenizer = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
        tokenizer.chat_template = (path / "chat_template.jinja").read_text()
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=self.config["eos_token_id"])
        self.max_context = self.config["max_position_embeddings"]
        if draft_record is not None:
            from .dflash2 import load_drafter

            if self.tree:
                # Distinct identity digest, APCv2 layout suffix and route
                # receipt for tree verification of the same drafter.
                self.EXTERNAL_ROUTE_TAG = TREE_ROUTE_TAG
            self._bind_external_drafter(draft_record, load_drafter, self.descriptor)

    @staticmethod
    def _check_request(request):
        from ..output import callable_tools

        if callable_tools(request):
            raise ValueError("Ripple tool calling is not implemented in this initial adapter")
        # The flag only has template meaning for chat requests; raw prompts
        # (e.g. classifier scoring) set it without asking for a control.
        if "messages" in request and request.get("enable_thinking") is False:
            raise ValueError("Ripple's source template does not implement a thinking-off control")

    def render_prompt(self, request):
        self._check_request(request)
        if "messages" not in request:
            return request["prompt"]
        return self.tokenizer.apply_chat_template(_template_history(request["messages"]),
                                                  add_generation_prompt=True, tokenize=False)

    def prompt_tokens(self, request):
        return self.tokenizer.encode(self.render_prompt(request), add_special_tokens=False)

    def output_parser(self, request):
        from ..output import OutputParser

        self._check_request(request)
        # The source prompt does not pre-open <think>. The shared parser
        # detects a leading generated marker and separates reasoning then.
        return OutputParser(chat="messages" in request, thinking=False, tools=None,
                            stops=request.get("stop", ()))

    def diagnostics(self):
        from ..runtime.models.modelopt_nvfp4 import nvfp4_model_status

        status = nvfp4_model_status(self.model)
        return {"route": "ordinary", "qualification": "unqualified", "policy": self.policy,
                "options": dict(self.options),
                "option_sources": dict(getattr(self, "option_sources", {})),
                "fused_elementwise": bool(getattr(self.model, "fused_elementwise", False)),
                "metal_rmsnorm": self._metal_rmsnorm_status(),
                "metal_glue": self._metal_glue_status(),
                "window_blocked_prefill": {"selected": bool(self.options.get("window_blocked_prefill")),
                                           "calls": self.model.window_blocked_calls() if self.model else 0},
                "gqa_folded_attention": {"selected": bool(self.options.get("gqa_folded_attention")),
                                         "calls": self.model.gqa_fold_calls() if self.model else 0},
                "implemented": True, "qualified": False, "selected": True,
                "speculation": ("external-ripple-dflash2-implemented-unqualified"
                                if getattr(self, "draft_model", None) is not None else "none"),
                "ast_tree_tiles": self._tree_receipt(),
                "observed_used": getattr(self.model, "forward_calls", 0) > 0,
                "forward_calls": getattr(self.model, "forward_calls", 0),
                "storage_format": self.kind,
                "source_activation_precision": "nvfp4" if self.kind == "nvfp4" else "bf16",
                "activation_precision": "bf16",
                "source_w4a4_parity": "not_implemented" if self.kind == "nvfp4" else "not_applicable",
                "sliding_layers": 64,
                "global_layers": 16, "sliding_window": 1024, "cache_layout": self.layout,
                "nvfp4": status}

    def _metal_glue_status(self):
        from ..runtime.models.ricursive_glue import STATS

        return {"selected": bool(self.options.get("metal_glue")), **STATS}

    def _tree_receipt(self):
        """Adapter-local tree-tile state; the speculative route receipt owns
        observations (external_tree_rounds, the provider's dispatches)."""
        tree = dict(getattr(self, "tree", {}) or {})
        return {"implemented": True, "qualified": False, "selected": bool(tree),
                "observed_used": False, "observation_authority": "speculative_route_receipt",
                "tree_target": TREE_TARGET, "drafters": ["dflash2_lattice"],
                "policy": tree,
                "refusals": {"quantized_kv": "attention tree tiles verify exact KVCache/"
                             "RotatingKVCache planes only",
                             "ast_graft_prompt_lookup": "not implemented"}}

    def _metal_rmsnorm_status(self):
        from ..runtime.models.ricursive_rmsnorm import STATS

        return {"selected": bool(self.options.get("metal_rmsnorm")), **STATS}

    @staticmethod
    def planning_cache_budget(config, *, mtp):
        from .ripple_memory import RippleCacheBudget

        return RippleCacheBudget.from_config(config, mtp=mtp)

    def cache_budget(self, *, mtp):
        return self.planning_cache_budget(self.config, mtp=mtp)

    def close(self):
        super().close()
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.close()
        self.model = None
        self.tokenizer = None
