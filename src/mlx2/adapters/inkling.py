"""Inkling Small mixed-2bit artifact gate and native text-serving adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
from itertools import pairwise
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .external_draft_policy import ExternalDraftAdapterMixin
from .ordinary_artifact import inspect_indexed_artifact

REPOSITORY = "Sawfwair/Inkling-Small-MLX-Mixed-2bit"
REVISION = "22c95fc2e30400a893dc9ac9fe4df17f306362ae"
CACHE_LAYOUT = "inkling-kv-four-shortconv-v1"
DESCRIPTOR = ModelDescriptor(
    model_type="inkling",
    family="inkling-small",
    variant="mixed-2bit-text",
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
            Capability.STREAMING,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.PROMPT_LOOKUP,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.inkling.InklingAdapter",
        "qualification": "pending",
        "scope": "native text; four causal shortconv states per layer; mixed affine q2/BF16",
        "ordinary_reference": "implemented",
        "native_mtp": "no MTP tensors in conversion",
    },
)


def text_tensor_specs():
    """Exact released text tensor shapes/dtypes, independent of MLX imports."""
    specs = {
        "language_model.lm_head.weight": ("BF16", [201024, 4096]),
        "language_model.model.embed_tokens.weight": ("BF16", [201024, 4096]),
        "language_model.model.embed_norm.weight": ("BF16", [4096]),
        "language_model.model.norm.weight": ("BF16", [4096]),
    }
    for i in range(42):
        base = f"language_model.model.layers.{i}."
        local = (i + 1) % 6 != 0
        shape = {
            "attn_sconv.conv.weight": [4096, 4, 1],
            "mlp_sconv.conv.weight": [4096, 4, 1],
            "input_layernorm.weight": [4096],
            "post_attention_layernorm.weight": [4096],
            "self_attn.k_norm.weight": [128],
            "self_attn.q_norm.weight": [128],
            "self_attn.k_sconv.conv.weight": [1024, 4, 1],
            "self_attn.v_sconv.conv.weight": [1024, 4, 1],
            "self_attn.rel_proj": [16, 512 if local else 1024],
            **{
                f"self_attn.{p}_proj.weight": [width, 4096]
                for p, width in (
                    ("q", 4096),
                    ("k", 1024),
                    ("v", 1024),
                    ("r", 512),
                    ("o", 4096),
                )
            },
            "mlp.global_scale": [1],
        }
        if i < 2:
            shape.update(
                {
                    f"mlp.{p}_proj.weight": dims
                    for p, dims in (
                        ("gate", [16384, 4096]),
                        ("up", [16384, 4096]),
                        ("down", [4096, 16384]),
                    )
                }
            )
        else:
            shape.update(
                {
                    "mlp.gate_weight": [258, 4096],
                    "mlp.e_score_correction_bias": [256],
                    "mlp.shared_experts.gate_proj.weight": [2, 2048, 4096],
                    "mlp.shared_experts.up_proj.weight": [2, 2048, 4096],
                    "mlp.shared_experts.down_proj.weight": [2, 4096, 2048],
                }
            )
            for p in ("gate", "up", "down"):
                out_dim, in_dim = (4096, 2048) if p == "down" else (2048, 4096)
                for leaf, divisor in (("weight", 16), ("scales", 128), ("biases", 128)):
                    specs[base + f"mlp.switch_mlp.{p}_proj.{leaf}"] = (
                        "U32" if leaf == "weight" else "BF16",
                        [256, out_dim, in_dim // divisor],
                    )
        for name, dims in shape.items():
            dtype = (
                "F32"
                if (
                    i >= 2
                    and name in ("mlp.e_score_correction_bias", "mlp.global_scale")
                )
                else "BF16"
            )
            specs[base + name] = (dtype, dims)
    return specs


def inspect_artifact(model_path):
    artifact = inspect_indexed_artifact(model_path)
    config, weights = artifact["config"], artifact["weight_map"]
    expected = {
        "hidden_size": 4096,
        "num_hidden_layers": 42,
        "vocab_size": 201024,
        "unpadded_vocab_size": 200058,
        "moe_intermediate_size": 2048,
        "dense_intermediate_size": 16384,
        "dense_mlp_idx": 2,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_shared_experts": 2,
        "sconv_kernel_size": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "swa_num_attention_heads": 32,
        "swa_num_key_value_heads": 8,
        "swa_head_dim": 128,
        "sliding_window_size": 512,
        "d_rel": 16,
        "rel_extent": 1024,
        "gate_activation": "sigmoid",
        "route_scale": 8.0,
        "shared_expert_sink": True,
        "norm_after_topk": True,
        "use_embed_norm": True,
        "use_sconv": True,
        "use_gate_bias": True,
        "use_global_scale": True,
        "q_bias": False,
        "o_bias": False,
        "rms_norm_eps": 1e-6,
        "logits_mup_width_multiplier": 16.0,
        "log_scaling_alpha": 0.1,
        "log_scaling_n_floor": 128000,
        "model_max_length": 1048576,
        "final_logit_softcapping": None,
        "local_layer_ids": [i for i in range(42) if (i + 1) % 6],
    }
    text = config.get("text_config", {})
    if config.get("model_type") != "inkling" or config.get("architectures") != [
        "InklingForConditionalGeneration"
    ]:
        raise ValueError("not an Inkling conditional-generation artifact")
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Inkling Small topology differs from the supported mixed-2bit checkpoint"
        )
    for key in ("layer_types", "mlp_layer_types"):
        if key in text:
            raise ValueError("Inkling layer-order overrides are not supported")
    quant = config.get("quantization")
    law = {"bits": 2, "group_size": 128, "mode": "affine"}
    modules = {
        f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}"
        for i in range(2, 42)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    if not isinstance(quant, dict) or any(
        quant.get(key) != value for key, value in law.items()
    ):
        raise ValueError("Inkling requires affine 2-bit/group-128 routed experts")
    if (
        set(quant) != set(law) | {"scope"} | modules
        or quant.get("scope") != "routed_experts"
    ):
        raise ValueError(
            "Inkling quantization must cover exactly the 120 routed projections"
        )
    if any(quant[name] != law for name in modules):
        raise ValueError("Inkling mixed-precision expert override differs")
    if config.get("quantization_config", quant) != quant:
        raise ValueError("Inkling quantization aliases disagree")
    specs = text_tensor_specs()
    actual = {name for name in weights if name.startswith("language_model.")}
    if actual != set(specs):
        raise ValueError("Inkling text tensor roles are incomplete or unexpected")
    packed = {name[: -len(".scales")] for name in weights if name.endswith(".scales")}
    if packed != modules or any(
        name + suffix not in weights
        for name in modules
        for suffix in (".weight", ".scales", ".biases")
    ):
        raise ValueError(
            "Inkling packed tensor roles differ from quantization metadata"
        )
    if any("mtp" in name for name in weights):
        raise ValueError("Inkling mixed-2bit route does not load MTP tensors")
    # Validate every index/header binding without materializing 84 GB of tensors.
    seen = set()
    path = Path(artifact["identity"]["path"])
    for name in sorted(set(weights.values())):
        with (path / name).open("rb") as handle:
            raw = handle.read(8)
            size = struct.unpack("<Q", raw)[0]
            if size > 32 << 20:
                raise ValueError("Inkling safetensors header is unreasonably large")
            header = json.loads(handle.read(size))
        intervals = []
        data_bytes = (path / name).stat().st_size - 8 - size
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if key in seen or weights.get(key) != name:
                raise ValueError("Inkling index/header closure differs")
            seen.add(key)
            shape, dtype, offsets = (
                entry.get("shape"),
                entry.get("dtype"),
                entry.get("data_offsets"),
            )
            sizes = {"BF16": 2, "F32": 4, "U32": 4}
            if (
                not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or dtype not in sizes
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= data_bytes
                or offsets[1] - offsets[0] != math.prod(shape) * sizes[dtype]
            ):
                raise ValueError(f"Inkling tensor payload bounds differ: {key}")
            intervals.append(tuple(offsets))
            if key in specs:
                dtype, shape = specs[key]
                if entry.get("shape") != shape or entry.get("dtype") != dtype:
                    raise ValueError(f"Inkling tensor shape/dtype differs: {key}")
        intervals.sort()
        if any(a[1] > b[0] for a, b in pairwise(intervals)):
            raise ValueError("Inkling tensor payloads overlap")
    if seen != set(weights):
        raise ValueError("Inkling indexed tensors are missing from shard headers")
    artifact.update(text_config=text, has_mtp=False, qualification="pending")
    receipt_path = path / ".hf-download-complete.json"
    attested = False
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        attested = (
            receipt.get("repo") == REPOSITORY and receipt.get("revision") == REVISION
        )
    artifact["identity"].update(
        source_repository=REPOSITORY,
        source_revision=REVISION if attested else "unattested-local-copy",
    )
    digest = hashlib.sha256(artifact["identity"]["fingerprint"].encode())
    extra = path / "special_tokens_map.json"
    if extra.is_file():
        digest.update(extra.name.encode())
        digest.update(extra.read_bytes())
    artifact["identity"]["fingerprint"] = digest.hexdigest()
    return artifact


def validate_policy(value):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("Inkling execution policy must be an object")
    value = dict(value)
    executor = value.pop("inkling_tree_executor", "serial")
    if executor not in ("serial", "packed"):
        raise ValueError("Inkling tree executor must be serial or packed")
    kernels = {
        name: value.pop(name, False)
        for name in ("inkling_fused", "inkling_q2_down_combine")
    }
    if any(type(v) is not bool for v in kernels.values()):
        raise TypeError("Inkling kernel options must be booleans")
    mode = value.pop("inkling_moe_nax", None)
    if mode not in (None, "off", "gather", "fused"):
        raise ValueError("Inkling MoE NAX mode must be off, gather or fused")
    kernels["inkling_moe_nax"] = mode
    kernels["inkling_tree_executor"] = executor
    if not value:
        if executor != "serial":
            raise ValueError(
                "packed Inkling executor requires a prompt-lookup AST route"
            )
        return kernels, {}
    supported = {
        "draft_model",
        "num_draft",
        "ast_topology",
        "ast_tiles",
        "ast_tile_rows",
        "ast_multilane_tile_rows",
        "pairwise_selection",
        "tensorfold_cohort_limit",
        "ast_adaptive_rows",
        "batch_size_route",
        "cost_per_commit_handoff",
    }
    if set(value) - supported:
        raise ValueError(
            f"unsupported Inkling AST controls: {sorted(set(value) - supported)}"
        )
    if value.get("ast_tiles", 1) != 1 or not 4 <= value.get("ast_tile_rows", 16) <= 16:
        raise ValueError("Inkling AST admits one tile of 4..16 rows")
    if (
        value.get("batch_size_route", "tree15_b1_b4_chain_b5plus_v1")
        != "tree15_b1_b4_chain_b5plus_v1"
    ):
        raise ValueError("Inkling AST supports its bounded B1..B4 route only")
    from .qwen38_27b import PROMPT_LOOKUP_DRAFT_NAMES, _native_mtp_tree_policy

    if value.get("draft_model") not in PROMPT_LOOKUP_DRAFT_NAMES:
        raise ValueError(
            "Inkling AST currently requires the model-free prompt-lookup drafter"
        )
    return kernels, _native_mtp_tree_policy(value)


class InklingAdapter(ExternalDraftAdapterMixin):
    descriptor = DESCRIPTOR
    default_route = "ordinary"
    prompt_lookup_default = False
    reasoning_effort_semantics = "reasoning_strength_0_0.99"
    ast_tree_tiles = True
    ast_tree_tile_dtype_gate = False  # adapter owns mixed FP32 convolution-state math
    EXTERNAL_DEFAULT_NUM_DRAFT = 15
    EXTERNAL_ROUTE_TAG = "inkling-pld-ast-v1"
    EXTERNAL_PROFILE = "inkling-apcv2-pld-ast"

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        from .process_globals import guarded_construction

        try:
            guarded_construction(
                self,
                lambda: self._initialize(
                    model_path, execution_policy, weight_streaming
                ),
            )
        except BaseException:
            self.close()
            raise

    def _initialize(self, model_path, execution_policy, weight_streaming):
        if weight_streaming is not None:
            raise ValueError("Inkling has no qualified expert-streaming loader")
        self.kernel_policy, external = validate_policy(execution_policy)
        artifact = inspect_artifact(model_path)
        self.config, self.identity, self.layout = (
            artifact["config"],
            artifact["identity"],
            CACHE_LAYOUT,
        )
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            require_process_numerics,
        )

        require_process_numerics("Inkling native profile")
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("Inkling native profile")
        from ..runtime.models import moe_nax_gather, switch_layers
        from .process_globals import MOE_NAX_GATHER, MOE_RHS_PAD_POLICY, claim

        mode = self.kernel_policy.get("inkling_moe_nax")
        mode = os.environ.get("MLX2_MOE_NAX_GATHER", "off") if mode is None else mode
        bits = moe_nax_gather.affine_bits_from_env()
        if mode != "off" and not os.environ.get("MLX2_MOE_NAX_GATHER_BITS"):
            bits = tuple(sorted({*bits, 2}))
        self.kernel_policy.update(inkling_moe_nax=mode, moe_affine_bits=list(bits))
        self._process_claim = claim(
            self,
            "Inkling native adapter",
            {
                MOE_NAX_GATHER: (mode, moe_nax_gather.set_mode),
                MOE_RHS_PAD_POLICY: ("floor", switch_layers.set_pad_policy),
                "moe_nax_affine_bits": (bits, moe_nax_gather.set_affine_bits),
            },
        )
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.inkling import Model, ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(artifact["text_config"]))
        path = Path(self.identity["path"])
        weights = self.model.sanitize(
            load_shards_evicting(
                [path / name for name in sorted(set(artifact["weight_map"].values()))]
            )
        )
        quant = self.config["quantization"]

        def predicate(name, module):
            override = quant.get("language_model." + name)
            return override if name + ".scales" in weights else False

        nn.quantize(
            self.model, group_size=128, bits=2, mode="affine", class_predicate=predicate
        )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        self.model.configure_kernels(
            fused=self.kernel_policy.get("inkling_fused", False),
            down_combine=self.kernel_policy.get("inkling_q2_down_combine", False),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        # End-message separates thought/tool/text messages; only end-sampling ends the turn.
        terminal = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
        if terminal != self.config["eos_token_id"]:
            raise ValueError("Inkling end-sampling token differs from checkpoint EOS")
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[terminal],
        )
        self.max_context = int(artifact["text_config"]["model_max_length"])
        self.draft_model = None
        if external:
            from ..runtime.drafters.prompt_lookup_draft import (
                load_prompt_lookup_drafter,
            )
            from .qwen38_27b import _prompt_lookup_record

            self.external_policy = external
            record = _prompt_lookup_record(external, self.identity["fingerprint"])
            self.identity = {
                **self.identity,
                "draft_revision": record["draft_revision"],
                "target_revision": record["target_revision"],
            }
            self._bind_external_drafter(
                record, load_prompt_lookup_drafter, self.descriptor
            )

    def close(self):
        if getattr(self, "_process_claim", None) is not None:
            from .process_globals import release

            release(self)

    def execution_numerics_contract(self):
        return {
            "operation": "inkling-native-v1",
            "kernel_policy": self.kernel_policy,
            "source_sha256": hashlib.sha256(
                (
                    Path(__file__).parents[1] / "runtime/models/inkling/language.py"
                ).read_bytes()
            ).hexdigest(),
            "qualification": "unqualified",
        }

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Inkling conversion contains no MTP head")
        return (
            self.EXTERNAL_PROFILE
            if self.draft_model is not None
            else "inkling-apcv2-ordinary"
        )

    def execution_config(self, *, max_lanes, prefill_step):
        if self.draft_model is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        return {
            "persistent": True,
            "num_draft": 0,
            "backend": "ordinary",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    @staticmethod
    def prefill_step_default():
        return 128

    def cache_budget(self, *, mtp):
        from .inkling_memory import InklingCacheBudget

        return InklingCacheBudget.from_config(self.config, mtp=mtp)

    def create_external_batch(self, **kwargs):
        from ..runtime.hybrid_tree_executor import HybridTreeExecutor
        from ..runtime.inkling_tree_executor import InklingTreeExecutor
        from .qwen38_27b import _tile_node_budgets

        provider = (
            InklingTreeExecutor
            if self.kernel_policy.get("inkling_tree_executor") == "packed"
            else HybridTreeExecutor
        )
        kwargs.setdefault("dynamic_singleton_tree", True)
        kwargs.setdefault("dynamic_tree_max_width", 4)
        kwargs.setdefault("dynamic_tree_overflow", "cohort")
        kwargs.setdefault(
            "tree_node_budget_by_lanes", _tile_node_budgets(self.external_policy, 4)
        )
        kwargs.setdefault(
            "tensorfold_cohort_limit",
            self.external_policy.get("tensorfold_cohort_limit", 4),
        )
        kwargs.setdefault("target_tree_executor_provider", provider.create())
        for key in ("ast_topology", "ast_adaptive_rows", "pairwise_selection"):
            if key in self.external_policy:
                kwargs.setdefault(key, self.external_policy[key])
        return super().create_external_batch(**kwargs)

    def bind_serving_lane_matmul(self, receipt):
        if self.kernel_policy.get("inkling_tree_executor") != "packed":
            return
        from mlx import nn

        from ..runtime.lane import installed_lane

        missing = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if (
                installed_lane(module) is None
                or getattr(module, "_lane_min_rows", None) != 1
                or getattr(module, "_lane_max_rows", 0) < 16
            ):
                missing.append(name)
        if not receipt or missing:
            raise ValueError(
                "Packed Inkling AST needs --lane-matmul exact covering all dense projections; "
                f"uncovered: {missing[:4]}"
            )
        self.packed_projection_law = receipt.get("law_id")

    def prompt_tokens(self, request):
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        return self.tokenizer.apply_chat_template(
            self._messages(request),
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort=request.get("reasoning_effort", "high"),
        )

    def render_prompt(self, request):
        if "messages" not in request:
            return request["prompt"]
        return self.tokenizer.apply_chat_template(
            self._messages(request),
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=request.get("reasoning_effort", "high"),
        )

    @staticmethod
    def _messages(request):
        if request.get("tools"):
            raise ValueError("Inkling tool routing is not implemented")
        messages = copy.deepcopy(request["messages"])
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(
                part.get("type") not in ("text", "input_text")
                for part in content
                if isinstance(part, dict)
            ):
                raise ValueError("Inkling native text adapter does not accept media")
        return messages

    @staticmethod
    def thinking_enabled(request):
        return False  # generated content markers select the channel

    def output_parser(self, request):
        from .inkling_output import InklingOutputParser

        return InklingOutputParser(
            chat="messages" in request, stops=request.get("stop", ())
        )

    def diagnostics(self):
        from ..runtime.models import moe_nax_gather

        return {
            "moe_nax_gather": moe_nax_gather.status(),
            "family": "inkling-small",
            "qualification": "unqualified",
            "adapter_default_route": "ordinary",
            "ordinary_reference_preserved": True,
            "cache_layout": self.layout,
            "kernel_policy": self.kernel_policy,
            "kernel_counters": self.model.kernel_receipt(),
            "target_forward_observed": self.model.completed_forwards > 0,
            "ast": {
                "implemented": True,
                "provider": self.kernel_policy.get("inkling_tree_executor", "serial"),
                "qualified": False,
            },
            "tensorfold": {
                "implemented": True,
                "selection_authority": "server_lane_receipt",
                "qualified": False,
            },
            "media": "weights downloaded; native text route refuses media",
        }
