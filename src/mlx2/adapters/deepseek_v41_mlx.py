# SPDX-License-Identifier: MIT
"""Standalone native affine V4.1 serving with bounded global expert storage.

Full target math/cache semantics are inherited; native QMM arithmetic has its
own unqualified profile. See provenance/deepseek-v41-mlx-native.json.
"""

import hashlib
import json
import os
from dataclasses import replace

from ..contracts import Capability
from ..sampling_defaults import SamplingDefaults, VendorSampling
from .deepseek_v41 import DESCRIPTOR as V41_DESCRIPTOR
from .deepseek_v41 import SAMPLING, DeepSeekV41Adapter
from .deepseek_v41_artifact import REFERENCE_FILE, tokenizer_metadata
from .deepseek_v41_execution import POLICY_CHOICES as CSA_POLICY_CHOICES
from .deepseek_v41_execution import (
    bind_cache_layout,
    execution_receipt,
    model_options,
    validate_csa_policy,
)
from .deepseek_v41_mlx_artifact import (
    FORMAT,
    MODEL_TYPE,
    NUMERIC_PROFILE,
    _sealed_stat,
    asset_path,
    file_stamp,
    inspect_artifact,
)

NUMERICS_REVISION = "mlx2-v41-native-f32-qmm-v1"
DEFAULT_EXPERT_CACHE_BYTES = 64 << 30
MAX_EXPERT_CACHE_BYTES = 70 << 30
DESCRIPTOR = replace(
    V41_DESCRIPTOR,
    model_type=MODEL_TYPE,
    variant="flash-1017-maya-xs-native-affine-text",
    metadata={
        "execution": "mlx2.adapters.deepseek_v41_mlx.DeepSeekV41MLXAdapter",
        "qualification": "unqualified",
        "numeric_profile": NUMERIC_PROFILE,
        "scope": "standalone native affine B1 text; full-depth V4.1; exact APCv2 within native profile; optional ordinary width-one copy verification",
    },
)
POLICY_CHOICES = {
    **CSA_POLICY_CHOICES,
    "deepseek_hc": ("reference", "fused_fp32"),
    "deepseek_qat": ("reference", "fused_fp32"),
    "deepseek_engram_cache": ("fp32", "bf16_lut"),
    "deepseek_pointwise": ("reference", "fused"),
    "deepseek_attention_epilogue": ("reference", "fused"),
    "deepseek_hc_epilogue": ("reference", "fused"),
    # Wider native QMM has its own reduction geometry. Verify one row at a
    # time against this artifact's ordinary decode instead of assuming parity.
    "deepseek_verify_rows": ("serial",),
    "deepseek_verify_commit": ("replay", "local_frontier"),
}


def execution_policy(value=None):
    value = {} if value is None else value
    allowed = set(POLICY_CHOICES) | {"draft_model", "ast_prompt_lookup", "num_draft", "deepseek_context_limit"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(
            "unsupported native V4.1 execution policy; GGUF-only kernels are unavailable"
        )
    result = {
        key: value.get(key, choices[0]) for key, choices in POLICY_CHOICES.items()
    }
    if "deepseek_context_limit" in value:
        result["deepseek_context_limit"] = value["deepseek_context_limit"]
    if any(
        type(result[key]) is not str or result[key] not in choices
        for key, choices in POLICY_CHOICES.items()
    ):
        raise ValueError("unsupported native V4.1 numerical policy")
    draft = value.get("draft_model")
    if draft not in (None, "prompt_lookup"):
        raise ValueError(
            "native V4.1 supports only the ordinary width-one prompt-lookup verifier"
        )
    enabled = value.get("ast_prompt_lookup", draft == "prompt_lookup")
    count = value.get("num_draft", 5)
    if (
        type(enabled) is not bool
        or enabled != (draft == "prompt_lookup")
        or type(count) is not int
        or not 1 <= count <= 5
        or ("num_draft" in value and not enabled)
    ):
        raise ValueError(
            "native V4.1 copy-chain requires draft_model=prompt_lookup and num_draft1..5"
        )
    result.update(ast_prompt_lookup=enabled, num_draft=count)
    validate_csa_policy(result)
    return result


def optimized_policy(*, copy_chain=False):
    """Explicit pointwise/QAT candidates; QMM and verification law unchanged."""
    value = {
        "deepseek_hc": "fused_fp32",
        "deepseek_qat": "fused_fp32",
        "deepseek_engram_cache": "bf16_lut",
        "deepseek_pointwise": "fused",
        "deepseek_attention_epilogue": "fused",
        "deepseek_hc_epilogue": "fused",
    }
    if copy_chain:
        value.update(draft_model="prompt_lookup", num_draft=5)
    return value


def validate_streaming_request(request):
    """Validate shared provider ownership without sizing a different profile."""
    if (
        request.mode != "moe_experts"
        or request.mtp_resident
        or request.collector is not None
        or request.prefill_admission != "lru"
        or request.router_hint != "off"
        or request.adaptive_residency.enabled
        or request.elastic
        or request.expert_overlap
    ):
        raise ValueError("native V4.1 requires synchronous bounded expert streaming")
    if (
        type(request.admission_limit_bytes) is not int
        or request.admission_limit_bytes <= 0
    ):
        raise MemoryError(
            "native V4.1 requires a positive measured early admission limit"
        )


def admission_plan(artifact, request, *, planner=None):
    """Resolve the bounded expert cache before any native weight allocation."""
    from .deepseek_v41_mlx_weights import native_weight_plan

    planner = native_weight_plan if planner is None else planner
    ceiling = (
        DEFAULT_EXPERT_CACHE_BYTES
        if request.budget_mode == "auto"
        else request.budget_bytes
    )
    if not 0 < ceiling <= MAX_EXPERT_CACHE_BYTES:
        raise ValueError("native V4.1 expert cache must be positive and at most 70 GiB")
    validate_streaming_request(request)
    headroom = max(5 << 30, request.headroom_bytes)
    ple_cache = (8 << 20) * len(getattr(artifact["args"], "engram_layer_ids", ()))
    plan = planner(artifact, ceiling)
    metadata_reserve = max(256 << 20, plan.get("metadata_bound_bytes", 0))
    if request.budget_mode == "auto":
        overhead = (
            plan["total_reservation_bytes"]
            - ceiling
            + headroom
            + metadata_reserve
            + ple_cache
        )
        ceiling = min(ceiling, request.admission_limit_bytes - overhead)
        if ceiling <= 0:
            raise MemoryError(
                "native V4.1 dense weights and execution reserve exceed admission"
            )
        plan = planner(artifact, ceiling)
    required = plan["total_reservation_bytes"] + headroom + metadata_reserve + ple_cache
    if (
        request.admission_limit_bytes is not None
        and required > request.admission_limit_bytes
    ):
        raise MemoryError(
            "native V4.1 dense/expert/staging reservation exceeds early admission"
        )
    return {
        **plan,
        "expert_cache_bytes": ceiling,
        "headroom_bytes": headroom,
        "metadata_reserve_bytes": metadata_reserve,
        "ple_cache_bytes": ple_cache,
        "required_bytes": required,
        "admission_limit_bytes": request.admission_limit_bytes,
        "ceiling_source": request.budget_mode,
        "phase": "pre_materialization",
    }


class DeepSeekV41MLXAdapter(DeepSeekV41Adapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    sampling_defaults = SAMPLING
    profile = "deepseek-v41-1017-maya-xs-native-f32-qmm-ordinary-b1"
    numeric_profile = NUMERIC_PROFILE
    numerics_revision = NUMERICS_REVISION
    weight_format = FORMAT
    policy_parser = staticmethod(execution_policy)
    weight_streaming_modes = frozenset({"moe_experts"})

    @staticmethod
    def weight_streaming_external_draft(policy):
        if 'draft_models' in policy or policy.get('draft_model') != 'prompt_lookup':
            return False
        execution_policy(policy)
        return True

    adapter_owned_default_policy_keys = frozenset(POLICY_CHOICES) | {
        "deepseek_context_limit",
        "draft_model",
        "ast_prompt_lookup",
        "num_draft",
    }

    @classmethod
    def resident_weight_bytes(cls, model_path):
        # Full native dense+expert size triggers the existing automatic early
        # streaming selection. PLE is always SSD-backed and not uploaded.
        totals = cls.artifact_inspector(model_path)["inventory"]["storage_bytes"]
        return totals["dense"] + totals["expert"]

    def _inspect_model_artifact(self, model_path):
        return self.artifact_inspector(model_path)

    def _weight_admission_plan(self, artifact, request):
        return admission_plan(artifact, request)

    def _construct_weights(self, artifact, request):
        from .deepseek_v41_mlx_weights import NativeV41Weights

        return NativeV41Weights(
            artifact,
            expert_cache_bytes=self.native_admission["expert_cache_bytes"],
            expert_read_workers=request.read_workers,
        )

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        from ..runtime.streamed_load import require_declared
        from .process_globals import guarded_construction

        require_declared(type(self), weight_streaming)
        self.kernel_policy = self.policy_parser(execution_policy)
        self.weight_streaming_request = weight_streaming
        self.weights = self.weight_stream = self.model = None
        self.draft_model = self.draft_weights = None
        self.external_record = self.external_source = None
        self.external_policy = {}
        self.engram_tables = {}
        self.preload_residency = self.model_residency = None
        self.native_admission = None
        self.profile += (
            ":"
            + hashlib.sha256(
                json.dumps(self.kernel_policy, sort_keys=True).encode()
            ).hexdigest()[:16]
        )
        try:
            guarded_construction(self, lambda: self._load(model_path))
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:  # noqa: BLE001 - preserve construction failure
                error.add_note(f"native V4.1 cleanup also failed: {cleanup_error!r}")
            raise

    def _load(self, model_path):
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        require_process_numerics("standalone native DeepSeek V4.1")
        clear_inherited_profile(())
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("standalone native DeepSeek V4.1")
        artifact = self._inspect_model_artifact(model_path)
        self.identity, self.metadata, self.config = (
            artifact["identity"],
            artifact["metadata"],
            dict(artifact["config"]),
        )
        args, path = artifact["args"], artifact["root"]
        if self.weight_streaming_request is None:
            from ..runtime.streamed_load import WeightStreamRequest
            from ..serving import preload_admission_limit_bytes

            limit = preload_admission_limit_bytes()
            if limit is None:
                raise MemoryError(
                    "native V4.1 default streaming requires measurable early admission"
                )
            self.weight_streaming_request = WeightStreamRequest(
                mode="moe_experts",
                budget_bytes=DEFAULT_EXPERT_CACHE_BYTES,
                admission_limit_bytes=limit,
                read_workers=2,
                bypass_page_cache=True,
            )
        request = self.weight_streaming_request
        self.native_admission = self._weight_admission_plan(artifact, request)

        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from .deepseek_v4_gguf_tokenizer import align_tokenizer

        aligned, self.tokenizer_receipt = align_tokenizer(
            json.loads(asset_path(path, REFERENCE_FILE).read_text()),
            tokenizer_metadata(self.metadata),
        )
        tokens = self.metadata["tokenizer.ggml.tokens"]
        backend = Tokenizer.from_str(json.dumps(aligned, ensure_ascii=False))
        if backend.get_vocab() != {token: index for index, token in enumerate(tokens)}:
            raise ValueError("native V4.1 tokenizer IDs changed")
        self.tokenizer = TokenizerWrapper(
            PreTrainedTokenizerFast(
                tokenizer_object=backend,
                bos_token=tokens[0],
                eos_token=tokens[1],
                pad_token=tokens[2],
                model_max_length=args.max_position_embeddings,
                clean_up_tokenization_spaces=False,
            ),
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[1],
        )

        from ..runtime.models.deepseek_v41 import DeepSeekV41Model
        from ..runtime.models.deepseek_v41_engram import EngramHasher, EngramRows
        from ..runtime.models.deepseek_v41_engram import file_stamp as ple_stamp
        from ..runtime.weight_residency import wire_serving_weights

        self.preload_residency = wire_serving_weights()
        self.weights = self._construct_weights(artifact, request)
        self.weight_stream = self.weights.expert_stream
        self.weight_stream.plan.resident_remainder_bytes = (
            self.native_admission["dense_payload_bytes"]
            + self.native_admission["dense_prepared_bound_bytes"]
            + self.native_admission["metadata_reserve_bytes"]
        )
        self.weight_stream.plan.ceiling_source = request.budget_mode
        for layer, rows in zip(args.engram_layer_ids, args.engram_num_embeddings):
            storage = artifact["tensors"][f"blk.{layer}.engram_embd.weight"]["storage"]
            table_path = asset_path(path, storage["file"])
            stamp = table_path.stat()
            if file_stamp(stamp) != _sealed_stat(artifact["files"][storage["file"]]):
                raise ValueError("native Engram file changed before opening")
            self.engram_tables[layer] = EngramRows(
                table_path,
                offset=0,
                rows=rows,
                expected_stat=ple_stamp(stamp),
                workers=min(request.read_workers, 32),
                cache_format=self.kernel_policy["deepseek_engram_cache"],
            )
        self.weight_stream.bind_ple_providers(self.engram_tables)
        binding = (
            self.identity["fingerprint"]
            + ":"
            + self.numerics_revision
            + ":"
            + hashlib.sha256(
                json.dumps(self.kernel_policy, sort_keys=True).encode()
            ).hexdigest()
        )
        self.model = DeepSeekV41Model(
            args,
            self.weights,
            identity=binding,
            hasher=EngramHasher(self.metadata),
            engram_tables=self.engram_tables,
            hc_policy=self.kernel_policy["deepseek_hc"],
            qat_policy=self.kernel_policy["deepseek_qat"],
            **model_options(self.kernel_policy),
        )
        self.model.verification_commit_policy = self.kernel_policy[
            "deepseek_verify_commit"
        ]
        self.model.verify_rows_policy = self.kernel_policy["deepseek_verify_rows"]
        self.model.pointwise_policy = self.kernel_policy["deepseek_pointwise"]
        self.model.attention_epilogue_policy = self.kernel_policy[
            "deepseek_attention_epilogue"
        ]
        self.model.hc_epilogue_policy = self.kernel_policy["deepseek_hc_epilogue"]
        self.model.verified_commit_stats = {}
        self.model.preload().eval()
        self.weights.verify_source()
        import mlx.core as mx

        limit = request.admission_limit_bytes
        if (
            limit is not None
            and mx.get_active_memory()
            + self.weight_stream.unfilled_reserve_bytes()
            + self.native_admission["headroom_bytes"]
            + self.native_admission["metadata_reserve_bytes"]
            > limit
        ):
            raise MemoryError(
                "native V4.1 retained tensors plus expert reserve exceed admission"
            )
        self.layout = type(self).descriptor.cache_layout
        self.max_context = self.model.context_limit
        self.descriptor = type(self).descriptor
        bind_cache_layout(self)
        self._configure_copy_chain()

    def _configure_copy_chain(self):
        if not self.kernel_policy["ast_prompt_lookup"]:
            return
        from ..runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
        from ..sampling_contract import GREEDY_REFUSAL_SAMPLING_CONTRACT

        if self.kernel_policy["deepseek_verify_commit"] == "local_frontier":
            from .deepseek_v41_external import local_frontier_drafter
            self.draft_model = local_frontier_drafter(self.model, num_draft=self.kernel_policy["num_draft"])
        else:
            self.draft_model = PromptLookupDraftModel(
                self.model, num_draft=self.kernel_policy["num_draft"]
            )
        self.default_route = "external_draft"
        self.descriptor = replace(
            self.descriptor,
            capabilities=self.descriptor.capabilities
            | {Capability.EXTERNAL_DRAFT},
            metadata={
                **type(self).descriptor.metadata,
                "speculation": "ordinary width-one copy verification; unqualified",
                "sampling_contract": GREEDY_REFUSAL_SAMPLING_CONTRACT,
            },
        )
        self.sampling_defaults = VendorSampling.single(
            SamplingDefaults(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                source="native V4.1 unqualified copy-chain greedy contract",
            ),
            model="DeepSeek-V4.1-Flash-1017-native-MLX+copy-chain",
        )

    def diagnostics(self):
        return {
            "architecture": "deepseek-v41-staggered-hc-csa2-engram",
            "cache_layout": self.layout,
            "route": "external_draft" if self.draft_model is not None else "ordinary",
            "qualification": "unqualified",
            "physical_width": 1,
            "supported_context": self.max_context,
            "weights": self.weight_format,
            "numeric_profile": self.numeric_profile,
            "numerics_revision": self.numerics_revision,
            "original_gguf_required": False,
            "full_depth_prefill": True,
            "ced": "unavailable",
            "execution_policy": dict(self.kernel_policy),
            "early_admission": self.native_admission,
            "preload_residency": self.preload_residency,
            "expert_stream": self.weight_stream.receipt(),
            "dense_storage": self.weights.dense_receipt(),
            "model_residency": None,
            "engram": {str(k): v.diagnostics() for k, v in self.engram_tables.items()},
            "scheduled_expert_calls": dict(self.weights.scheduled_expert_calls),
            "scheduled_dense_calls": dict(self.weights.scheduled_dense_calls),
            "scheduled_grouped_calls": dict(self.weights.scheduled_grouped_calls),
            "kv_storage": "physical FP32; existing BF16/FP8/FP4-rounded activation law",
            "speculation": {
                "kind": "adaptive_copy_chain",
                "verification": "ordinary_width_one_" + self.kernel_policy["deepseek_verify_commit"],
                "qualified": False,
                "native_dspark": False,
                "packed_tree_ast": False,
            }
            if self.draft_model is not None
            else "unavailable",
            "vision": "unavailable",
            "scheduled_call_observation": "Python scheduling only; not GPU completion evidence",
            **execution_receipt(self.model),
        }

    def close(self):
        """Keep providers reachable when a fence/read owner cannot retire."""
        from .process_globals import release

        error = None
        weights = getattr(self, "weights", None)
        if weights is not None:
            try:
                weights.close()
            except BaseException as failure:  # noqa: BLE001 - retain every uncertain owner
                error = failure
            else:
                self.weights = self.weight_stream = None
                self.model = self.draft_model = None
        remaining = {}
        for layer, reader in getattr(self, "engram_tables", {}).items():
            try:
                reader.close()
            except BaseException as failure:  # noqa: BLE001 - finish independent retirement
                remaining[layer] = reader
                error = error or failure
        self.engram_tables = remaining
        if error is not None:
            raise error
        residency = getattr(self, "preload_residency", None)
        if residency and residency.get("wired"):
            import mlx.core as mx

            mx.set_wired_limit(residency["previous_wired_limit_bytes"])
        self.preload_residency = self.model_residency = None
        release(self)
