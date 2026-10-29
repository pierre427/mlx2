# SPDX-License-Identifier: MIT
"""Native packed V4.1 ordinary serving, APCv2 and opt-in adaptive copy draft."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, StatePlane
from ..runtime.models.deepseek_v41_cache import CACHE_LAYOUT
from ..sampling_defaults import SamplingDefaults, VendorSampling
from .deepseek_v4_gguf import DESCRIPTOR as V4_DESCRIPTOR
from .deepseek_v4_gguf import DeepSeekV4GGUFAdapter
from .deepseek_v41_artifact import (
    FILENAME,
    MODEL_TYPE,
    REFERENCE_FILE,
    inspect_artifact,
    tokenizer_metadata,
)
from .deepseek_v41_execution import POLICY_CHOICES as CSA_POLICY_CHOICES
from .deepseek_v41_execution import (
    bind_cache_layout,
    execution_receipt,
    model_options,
    validate_csa_policy,
)

DESCRIPTOR = replace(
    V4_DESCRIPTOR,
    model_type=MODEL_TYPE,
    family="deepseek_v41",
    variant="flash-1017-maya-xs-native-packed-text",
    cache_layout=CACHE_LAYOUT,
    capabilities=V4_DESCRIPTOR.capabilities | {Capability.TOOLS, Capability.GRAMMAR},
    state_planes=V4_DESCRIPTOR.state_planes | {StatePlane.RECURRENT},
    metadata={
        "execution": "mlx2.adapters.deepseek_v41.DeepSeekV41Adapter",
        "qualification": "unqualified",
        "scope": "B1 text, source-shared CSA2, exact ngram state; optional adaptive copy-chain; learned DSpark/vision unavailable",
    },
)
SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        source="official V4.1 inference default temperature1; neutral filters",
    ),
    model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
)
POLICY_CHOICES = {
    **CSA_POLICY_CHOICES,
    "deepseek_hc": ("reference", "fused_fp32"),
    "deepseek_qat": ("reference", "fused_fp32"),
    "deepseek_packed": ("reference", "cooperative"),
    "deepseek_dense_prefill": ("reference", "nax_fp16"),
    "deepseek_expert_prefill": ("reference", "segmented_nax_fp16"),
    "deepseek_expert_decode": ("reference", "row_buffers"),
    "deepseek_expert_buffers": ("stack", "row_groups"),
    "deepseek_verify_commit": ("replay", "local_frontier"),
    "deepseek_expert_small": ("reference", "row_exact"),
    "deepseek_expert_reuse": ("one", "two", "four"),
    "deepseek_expert_iq1": ("scalar", "broadcast"),
    "deepseek_expert_down": ("serial", "parallel"),
    "deepseek_expert_binding": ("twelve", "six"),
    "deepseek_expert_io": ("blocking", "overlap"),
    "deepseek_expert_reads": ("reference", "coalesced"),
    "deepseek_expert_retention": ("lru", "adaptive"),
    "deepseek_expert_atlas": ("off", "history"),
    "deepseek_engram_cache": ("fp32", "bf16_lut"),
    "deepseek_verify_rows": ("serial", "layerwise"),
    "deepseek_dense_epilogue": ("reference", "fused"),
    "deepseek_dense_pair": ("reference", "fused"),
    "deepseek_pointwise": ("reference", "fused"),
    "deepseek_expert_groups": ("six", "ten"),
    "deepseek_expert_simd": ("one", "four"),
    "deepseek_shared": ("reference", "fused"),
    "deepseek_residency": ("fixed", "elastic"),
    "deepseek_attention_epilogue": ("reference", "fused"),
    "deepseek_hc_epilogue": ("reference", "fused"),
    "deepseek_dense_storage": ("resident", "paged"),
}


def execution_policy(value):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - (
        POLICY_CHOICES.keys() | {"ast_prompt_lookup", "num_draft", "draft_model", "deepseek_dense_cache_mib", "deepseek_context_limit"}
    ):
        raise ValueError(
            "unsupported V4.1 execution policy; foreign V4 DSpark/KV/attention policies are not compatible"
        )
    result = {k: value.get(k, v[0]) for k, v in POLICY_CHOICES.items()}
    if "deepseek_context_limit" in value:
        result["deepseek_context_limit"] = value["deepseek_context_limit"]
    if any(
        type(result[k]) is not str or result[k] not in v
        for k, v in POLICY_CHOICES.items()
    ):
        raise ValueError("unsupported V4.1 kernel policy")
    dense_cache = value.get("deepseek_dense_cache_mib", 1024)
    if type(dense_cache) is not int or not 256 <= dense_cache <= 16384:
        raise ValueError("V4.1 dense cache must be integer MiB in256..16384")
    if "deepseek_dense_cache_mib" in value and result["deepseek_dense_storage"] != "paged":
        raise ValueError("V4.1 dense cache size requires paged dense storage")
    if result["deepseek_dense_storage"] == "paged":
        result["deepseek_dense_cache_mib"] = dense_cache
    draft = value.get("draft_model")
    if draft not in (None, "prompt_lookup"):
        raise ValueError(
            "V4.1 requires draft_model=prompt_lookup; a matching learned DSpark artifact is unavailable"
        )
    result["ast_prompt_lookup"] = value.get(
        "ast_prompt_lookup", draft == "prompt_lookup"
    )
    result["num_draft"] = value.get("num_draft", 5)
    if (
        type(result["ast_prompt_lookup"]) is not bool
        or type(result["num_draft"]) is not int
        or not 1 <= result["num_draft"] <= 5
    ):
        raise ValueError(
            "V4.1 adaptive copy draft requires boolean ast_prompt_lookup and integer num_draft1..5"
        )
    if result["ast_prompt_lookup"] != (draft == "prompt_lookup"):
        raise ValueError("adaptive copy-chain requires draft_model=prompt_lookup")
    if "num_draft" in value and not result["ast_prompt_lookup"]:
        raise ValueError("num_draft requires ast_prompt_lookup")
    validate_csa_policy(result)
    return result


def optimized_policy(*, copy_chain=False):
    """Explicit unqualified candidate set; never an artifact/serving default."""
    policy = {
        "deepseek_hc": "fused_fp32",
        "deepseek_qat": "fused_fp32",
        "deepseek_packed": "cooperative",
        "deepseek_dense_prefill": "nax_fp16",
        "deepseek_expert_prefill": "segmented_nax_fp16",
        "deepseek_expert_decode": "row_buffers",
        "deepseek_expert_buffers": "row_groups",
        "deepseek_verify_commit": "local_frontier",
        "deepseek_expert_small": "row_exact",
        "deepseek_expert_reuse": "four",
        "deepseek_expert_iq1": "scalar",
        "deepseek_expert_down": "serial",
        "deepseek_expert_binding": "twelve",
        "deepseek_expert_io": "blocking",
        "deepseek_expert_reads": "coalesced",
        "deepseek_expert_retention": "lru",
        "deepseek_expert_atlas": "history",
        "deepseek_engram_cache": "bf16_lut",
        "deepseek_verify_rows": "layerwise",
        "deepseek_dense_epilogue": "fused",
        "deepseek_dense_pair": "fused",
        "deepseek_pointwise": "fused",
        "deepseek_expert_groups": "ten",
        "deepseek_expert_simd": "one",
        "deepseek_shared": "fused",
        "deepseek_attention_epilogue": "fused",
        "deepseek_hc_epilogue": "fused",
    }
    if copy_chain:
        policy.update(draft_model="prompt_lookup", num_draft=5)
    return policy


class DeepSeekV41Adapter(DeepSeekV4GGUFAdapter):
    weight_streaming_modes = frozenset({"moe_experts"})
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    sampling_defaults = SAMPLING
    profile = "deepseek-v41-1017-maya-xs-native-gguf-ordinary-b1"
    ast_prompt_lookup_tiles = False
    reasoning_effort_semantics = "numeric_strength_1_100"
    anthropic_tool_schema_history = True
    tool_call_open_marker = "<｜DSML｜ calls>"

    @staticmethod
    def lower_ast_chain(drafter, artifact):
        if drafter == "pld":
            return "external_draft", {"draft_model": "prompt_lookup", "ast_prompt_lookup": True}
        if drafter == "embedded_dspark":
            return "external_draft", {"draft_model": "embedded_dspark"}
        return None

    @staticmethod
    def effective_route_capabilities(capabilities, *, route, max_lanes):
        selected = set(capabilities)
        if route != "ordinary" or max_lanes != 1:
            selected -= {Capability.TOOLS, Capability.GRAMMAR}
        return frozenset(selected)

    def normalize_request(self, request):
        from .deepseek_v41_protocol import normalize_request

        result = normalize_request(request)
        if getattr(self, "draft_model", None) is not None and (
            (result.get("tools") and result.get("tool_choice", "auto") != "none")
            or result.get("grammar") is not None
            or result.get("response_format") not in (None, {"type": "text"})
        ):
            raise ValueError("V4.1 tools and structured output require the ordinary B1 route")
        return result

    @staticmethod
    def thinking_enabled(request):
        from .deepseek_v41_protocol import normalize_request

        return normalize_request(request).get("enable_thinking", False)

    def thinking_close_token_ids(self):
        ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        return tuple(ids) or None

    def tool_constraint(self, request):
        from ..output import callable_tools
        from .deepseek_v41_protocol import constrained_tool_grammar

        request = self.normalize_request(request)
        tools = callable_tools(request)
        return constrained_tool_grammar(
            tools, request.get("tool_choice", "auto"),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        ) if tools else None

    def output_parser(self, request):
        from ..output import callable_tools, constrained_tool_choice
        from .deepseek_v41_protocol import DeepSeekV41OutputParser

        request = self.normalize_request(request)
        self._validate_external_request(request)
        return DeepSeekV41OutputParser(
            chat="messages" in request, thinking=self.thinking_enabled(request),
            tools=callable_tools(request), tokenizer=self.tokenizer,
            stops=request.get("stop", ()),
            constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    adapter_owned_default_policy_keys = frozenset(POLICY_CHOICES) | {
        "deepseek_context_limit",
        "ast_prompt_lookup",
        "num_draft",
        "draft_model",
        "deepseek_dense_cache_mib",
    }

    @staticmethod
    def resident_weight_bytes(model_path):
        artifact = inspect_artifact(model_path)
        return sum(row["packed_bytes"] for kind, row in artifact["inventory"]["formats"].items() if kind != "I8")

    @staticmethod
    def weight_streaming_external_draft(policy):
        return policy.get("draft_model") == "prompt_lookup" and "draft_models" not in policy

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        from ..runtime.streamed_load import require_declared

        require_declared(type(self), weight_streaming)
        self.weight_streaming_request = weight_streaming
        self.weight_stream = None
        self.kernel_policy = globals()["execution_policy"](execution_policy)
        self.external_policy = {}
        self.draft_model = self.draft_weights = None
        self.engram_tables = {}
        self.preload_residency = None
        self.model_residency = None
        self.external_record = self.external_source = None
        self.profile += (
            ":"
            + hashlib.sha256(
                json.dumps(self.kernel_policy, sort_keys=True).encode()
            ).hexdigest()[:16]
        )
        from .process_globals import guarded_construction

        try:
            guarded_construction(self, lambda: self._load(model_path))
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:  # noqa: BLE001 - preserve load failure
                error.add_note(f"V4.1 resource cleanup also failed: {cleanup_error!r}")
            raise

    def _load(self, model_path):
        import os

        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        require_process_numerics("native DeepSeek V4.1")
        clear_inherited_profile(())
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("native DeepSeek V4.1")
        artifact = inspect_artifact(model_path)
        self.identity, self.metadata, self.config = (
            artifact["identity"],
            artifact["metadata"],
            dict(artifact["config"]),
        )
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        from ..runtime.models.deepseek_v41 import DeepSeekV41Model
        from ..runtime.models.deepseek_v41_config import DeepSeekV41Config
        from ..runtime.models.deepseek_v41_engram import EngramHasher, EngramRows
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from .deepseek_v4_gguf_tokenizer import align_tokenizer

        args = DeepSeekV41Config.from_gguf(self.metadata)
        path = Path(self.identity["path"])
        aligned, self.tokenizer_receipt = align_tokenizer(
            json.loads((path / REFERENCE_FILE).read_text()),
            tokenizer_metadata(self.metadata),
        )
        tokens = self.metadata["tokenizer.ggml.tokens"]
        backend = Tokenizer.from_str(json.dumps(aligned, ensure_ascii=False))
        if backend.get_vocab() != {t: i for i, t in enumerate(tokens)}:
            raise ValueError("V4.1 tokenizer IDs changed")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token=tokens[0],
            eos_token=tokens[1],
            pad_token=tokens[2],
            model_max_length=args.max_position_embeddings,
            clean_up_tokenization_spaces=False,
        )
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[1]
        )
        from ..runtime.gguf.v41_metal import self_check
        from .deepseek_v41_weights import DeepSeekV41Weights

        self_check()
        if self.kernel_policy["deepseek_expert_decode"] == "row_buffers":
            from ..runtime.gguf.v41_metal import self_check_rows

            self_check_rows()
        # This artifact's resident payload nearly fills a 128GiB host. Admit
        # each allocation to the normal serving residency set as it is loaded,
        # instead of first swapping it out and wiring the complete model later.
        from ..runtime.weight_residency import wire_serving_weights

        self.preload_residency = wire_serving_weights()
        self.weights = DeepSeekV41Weights(
            path / FILENAME,
            expected_stat=artifact["weight_stat"],
            compute_policy=self.kernel_policy["deepseek_packed"],
            dense_prefill_policy=self.kernel_policy["deepseek_dense_prefill"],
            expert_prefill_policy=self.kernel_policy["deepseek_expert_prefill"],
        )
        self.weights.expert_decode_policy = self.kernel_policy["deepseek_expert_decode"]
        self.weights.expert_buffer_policy = self.kernel_policy["deepseek_expert_buffers"]
        for name in ("small", "reuse", "iq1", "down", "binding", "io", "groups", "simd"):
            setattr(self.weights, f"expert_{name}_policy", self.kernel_policy[f"deepseek_expert_{name}"])
        for name in ("epilogue", "pair"):
            setattr(self.weights, f"dense_{name}_policy", self.kernel_policy[f"deepseek_dense_{name}"])
        self.weights.shared_policy = self.kernel_policy["deepseek_shared"]
        self.weights.expert_buffer_counters = {}
        dense_plan = None
        if self.kernel_policy["deepseek_dense_storage"] == "paged":
            if self.weight_streaming_request is None:
                raise ValueError("V4.1 dense paging requires declared native expert streaming")
            dense_plan = self.weights.dense_paging_plan(self.kernel_policy["deepseek_dense_cache_mib"] << 20)
        if self.weight_streaming_request is not None:
            from .deepseek_v41_expert_stream import PackedExpertStream

            request = self.weight_streaming_request
            if (request.mode != "moe_experts" or request.mtp_resident or request.collector is not None
                    or request.prefill_admission != "lru" or request.router_hint != "off"
                    or request.adaptive_residency.enabled):
                raise ValueError("V4.1 requires adapter-owned IQ1_S streaming policies")
            tables = [t for t in self.weights.tensors.values() if t.ggml_type == 19]
            table_bytes = sum(t.nbytes for t in tables)
            per_layer = table_bytes // args.num_hidden_layers
            direct_only = (self.kernel_policy["deepseek_expert_small"] == "row_exact"
                           and self.kernel_policy["deepseek_expert_decode"] == "row_buffers"
                           and self.kernel_policy["deepseek_expert_buffers"] == "row_groups"
                           and self.kernel_policy["deepseek_expert_prefill"] == "segmented_nax_fp16")
            atlas = None
            if self.kernel_policy["deepseek_expert_atlas"] == "history":
                from ..runtime.expert_stream_atlas import ExpertStreamAtlas

                digest = hashlib.sha256((self.identity["fingerprint"]+json.dumps(self.kernel_policy, sort_keys=True)).encode()).hexdigest()
                atlas = ExpertStreamAtlas(digest, layers=args.num_hidden_layers, experts=args.n_routed_experts,
                                          top_k=args.num_experts_per_tok,
                                          path=Path.home()/".cache"/"mlx2"/"expert-stream-atlas"/(digest+".json"))
            transient = (1 if direct_only else 2)*per_layer + 64*(per_layer//args.n_routed_experts) + ((256 << 20) if atlas else 0)
            remainder = (dense_plan["resident_remainder_bound_bytes"] if dense_plan is not None else
                         sum(t.nbytes for t in self.weights.tensors.values() if t.ggml_type != 19 and ".engram_embd." not in t.name)) + (256 << 20)
            ceiling = request.budget_bytes
            if request.budget_mode == "auto":
                ceiling = min(70 << 30, request.admission_limit_bytes-remainder-max(5 << 30, request.headroom_bytes)-transient)
            self.weight_stream = PackedExpertStream(
                path / FILENAME, tensors=self.weights.tensors, data_offset=self.weights.file.data_offset,
                expected_stat=artifact["weight_stat"], ceiling_bytes=ceiling, workers=request.read_workers,
                coalesced=self.kernel_policy["deepseek_expert_reads"] == "coalesced",
                adaptive=self.kernel_policy["deepseek_expert_retention"] == "adaptive", atlas=atlas,
                direct_only=direct_only)
            self.weights.expert_stream = self.weight_stream
            self.weight_stream.plan.resident_remainder_bytes = remainder
            self.weight_stream.plan.ceiling_source = request.budget_mode
            if dense_plan is not None:
                self.weights.enable_dense_paging(dense_plan["cache_bytes"], binding=self.identity["fingerprint"])
                self.weight_stream.bind_dense_provider(self.weights)
        hasher = EngramHasher(self.metadata)
        for layer, rows in zip(args.engram_layer_ids, args.engram_num_embeddings):
            tensor = self.weights.tensors[f"blk.{layer}.engram_embd.weight"]
            self.engram_tables[layer] = EngramRows(
                path / FILENAME,
                offset=self.weights.file.data_offset + tensor.offset,
                rows=rows,
                expected_stat=artifact["weight_stat"],
                cache_format=self.kernel_policy["deepseek_engram_cache"],
            )
        if self.weight_stream is not None:
            self.weight_stream.bind_ple_providers(self.engram_tables)
        binding = (
            self.identity["fingerprint"]
            + ":"
            + hashlib.sha256(
                json.dumps(self.kernel_policy, sort_keys=True).encode()
            ).hexdigest()
        )
        self.model = DeepSeekV41Model(
            args,
            self.weights,
            identity=binding,
            hasher=hasher,
            engram_tables=self.engram_tables,
            hc_policy=self.kernel_policy["deepseek_hc"],
            qat_policy=self.kernel_policy["deepseek_qat"],
            **model_options(self.kernel_policy),
        )
        self.model.preload().eval()
        self.model.verification_commit_policy = self.kernel_policy["deepseek_verify_commit"]
        self.model.verify_rows_policy = self.kernel_policy["deepseek_verify_rows"]
        self.model.pointwise_policy = self.kernel_policy["deepseek_pointwise"]
        self.model.attention_epilogue_policy = self.kernel_policy["deepseek_attention_epilogue"]
        self.model.hc_epilogue_policy = self.kernel_policy["deepseek_hc_epilogue"]
        self.model.verified_commit_stats = {}
        if self.weight_stream is not None and (self.kernel_policy["deepseek_residency"] == "elastic" or self.weight_streaming_request.elastic):
            self.weight_stream.enable_elastic(self.engram_tables)
        if self.weight_stream is not None:
            import mlx.core as mx

            limit = self.weight_streaming_request.admission_limit_bytes
            if limit is not None and mx.get_active_memory()+self.weight_stream.unfilled_reserve_bytes()+max(5 << 30, self.weight_streaming_request.headroom_bytes) > limit:
                raise ValueError("V4.1 retained weights plus expert reserve do not fit admission")
        self.layout = CACHE_LAYOUT
        self.max_context = self.model.context_limit
        self.descriptor = DESCRIPTOR
        bind_cache_layout(self)
        if self.kernel_policy["ast_prompt_lookup"]:
            from ..runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel

            if self.kernel_policy["deepseek_verify_commit"] == "local_frontier":
                from .deepseek_v41_external import local_frontier_drafter

                self.draft_model = local_frontier_drafter(self.model, num_draft=self.kernel_policy["num_draft"])
            else:
                self.draft_model = PromptLookupDraftModel(self.model, num_draft=self.kernel_policy["num_draft"])
            from ..sampling_contract import GREEDY_REFUSAL_SAMPLING_CONTRACT

            self.default_route = "external_draft"
            self.descriptor = replace(
                self.descriptor,
                capabilities=self.descriptor.capabilities | {Capability.EXTERNAL_DRAFT},
                metadata={
                    **DESCRIPTOR.metadata,
                    "speculation": "adaptive copy chain, ordinary width-one; commit="+self.kernel_policy["deepseek_verify_commit"]+"; unqualified",
                    "sampling_contract": GREEDY_REFUSAL_SAMPLING_CONTRACT,
                },
            )
            self.sampling_defaults = VendorSampling.single(
                SamplingDefaults(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    source="V4.1 unqualified adaptive copy-chain greedy contract",
                ),
                model="DeepSeek-V4.1-Flash-1017+copy-chain",
            )
        from .deepseek_v41_residency import DeepSeekV41Residency

        self.model_residency = DeepSeekV41Residency(self)

    def _validate_external_request(self, request):
        if self.draft_model is not None and (
            type(request.get("temperature", 0.0)) not in (int, float)
            or request.get("temperature", 0.0) != 0.0
        ):
            raise ValueError(
                "V4.1 adaptive copy-chain supports greedy temperature0 only"
            )

    def execution_config(self, *, max_lanes, prefill_step):
        width = getattr(self, "physical_batch_width", 1)
        if type(max_lanes) is not int or not 1 <= max_lanes <= width or not 1 <= prefill_step <= 1024:
            raise ValueError(f"V4.1 requires physical width1..{width} and prefill chunks1..1024")
        from .ordinary_text import OrdinaryTextAdapter

        result = OrdinaryTextAdapter.execution_config(
            self, max_lanes=max_lanes, prefill_step=prefill_step
        )
        if self.draft_model is not None:
            result.update(
                backend="external_draft", num_draft=self.kernel_policy["num_draft"]
            )
        return result

    def create_external_batch(self, **kwargs):
        if getattr(self.draft_model, "kind", None) == "deepseek_v41_dspark":
            from .deepseek_v41_dspark_external import create_dspark_batch

            kwargs.setdefault("num_draft", self.external_policy["num_draft"])
            kwargs.setdefault("cost_per_commit_handoff", False)
            return create_dspark_batch(self.model, self.draft_model, **kwargs)
        if self.draft_model is None:
            raise ValueError("V4.1 copy-chain route is not selected")
        from .deepseek_v41_external import create_batch

        if (
            kwargs.get("num_draft", self.kernel_policy["num_draft"])
            != self.kernel_policy["num_draft"]
        ):
            raise ValueError("copy draft depth differs from bound V4.1 profile")
        kwargs.setdefault("num_draft", self.kernel_policy["num_draft"])
        kwargs.setdefault("cost_per_commit_handoff", False)
        return create_batch(
            self.model, self.draft_model, binding=self.model.identity, **kwargs
        )

    def cache_budget(self, *, mtp):
        if mtp:
            raise ValueError("V4.1 DSpark uses the explicit Source external-draft route, not native MTP")
        from .deepseek_v41_memory import DeepSeekV41CacheBudget

        return DeepSeekV41CacheBudget.from_config(self.model.args, kv_storage_policy=getattr(self.model, "kv_storage_policy", "reference"))

    def _template(self, request, *, tokenize):
        request = self.normalize_request(request)
        self._validate_external_request(request)
        if "messages" in request:
            from ..runtime.models.deepseek_v41_encoding import encode_messages
            from .deepseek_v41_protocol import messages_for_encoding

            text = encode_messages(
                messages_for_encoding(request),
                thinking_mode="thinking" if self.thinking_enabled(request) else "chat",
                reasoning_effort=request.get("reasoning_effort"),
                drop_thinking=not request.get("preserve_thinking", False),
            )
        else:
            text = request["prompt"]
        return self.tokenizer.encode(text, add_special_tokens=False) if tokenize else text

    @staticmethod
    def chat_template_options():
        return {"preserve_thinking": {"implemented": True, "default": False}}

    def diagnostics(self):
        from ..runtime.models.deepseek_v41_quant import fused_status

        return {
            "architecture": "deepseek-v41-staggered-hc-csa2-engram",
            "cache_layout": self.layout,
            "route": "external_draft" if self.draft_model is not None else "ordinary",
            "qualification": "unqualified",
            "physical_width": 1,
            "supported_context": self.max_context,
            "weights": "IQ1_S/Q8_0 native blocks",
            "kv_storage": "physical FP32; BF16/FP8/FP4-rounded values",
            "execution_policy": dict(self.kernel_policy),
            "preload_residency": self.preload_residency,
            "expert_stream": self.weight_stream.receipt() if getattr(self, "weight_stream", None) is not None else None,
            "dense_storage": getattr(self.weights, "dense_receipt", lambda: None)(),
            "model_residency": self.model_residency.receipt() if getattr(self, "model_residency", None) is not None else None,
            "engram": {str(k): v.diagnostics() for k, v in self.engram_tables.items()},
            "scheduled_expert_calls": dict(self.weights.scheduled_expert_calls),
            "scheduled_expert_buffer_operations": dict(getattr(self.weights, "expert_buffer_counters", {})),
            "verified_prefix_commits": dict(getattr(self.model, "verified_commit_stats", {})),
            "scheduled_dense_calls": dict(self.weights.scheduled_dense_calls),
            "scheduled_grouped_calls": dict(self.weights.scheduled_grouped_calls),
            "qat_runtime": fused_status(),
            "speculation": {
                "kind": "adaptive_copy_chain",
                "verification": ("layerwise_row_exact_" if self.kernel_policy["deepseek_verify_rows"] == "layerwise" else "ordinary_width_one_")+(
                    "local_frontier" if self.kernel_policy["deepseek_verify_commit"] == "local_frontier" else "snapshot_replay"),
                "packed_tree_ast": False,
                "native_dspark": "unavailable; no matching artifact",
                "qualified": False,
            }
            if self.draft_model is not None
            else "unavailable",
            "vision": "unavailable",
            "scheduled_call_observation": "Python scheduling only; not GPU completion evidence",
            **execution_receipt(self.model),
        }

    def close(self):
        error = None
        self.model_residency = None
        stream = getattr(self, "weight_stream", None)
        self.weight_stream = None
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:  # noqa: BLE001 - retire every independent resource
                error = exc
        for reader in getattr(self, "engram_tables", {}).values():
            try:
                reader.close()
            except Exception as exc:  # noqa: BLE001 - finish closing every independent owner
                error = error or exc
        self.engram_tables = {}
        try:
            super().close()
        finally:
            residency = getattr(self, "preload_residency", None)
            if residency and residency.get("wired"):
                import mlx.core as mx

                mx.set_wired_limit(residency["previous_wired_limit_bytes"])
                self.preload_residency = None
            if error is not None:
                raise error
