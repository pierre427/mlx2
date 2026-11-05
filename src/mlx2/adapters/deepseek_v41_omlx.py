# SPDX-License-Identifier: MIT
"""Ordinary B1 text serving for one pinned omlx-layout V4.1 MLX checkpoint.

Target math, cache layout, APCv2 and the copy-chain option are inherited from
the native affine adapter; weights come from deepseek_v41_omlx_weights and
Engram rows from the three-plane affine reader. The embedded DSpark head and
vision tower are inventoried but not served by this route. Unqualified; see
provenance/deepseek-v41-omlx.json.
"""
import hashlib
import json
import os
from dataclasses import replace

import numpy as np

from .deepseek_v41_execution import bind_cache_layout, model_options
from .deepseek_v41_mlx import DESCRIPTOR as MLX_DESCRIPTOR
from .deepseek_v41_mlx import DeepSeekV41MLXAdapter, validate_streaming_request
from .deepseek_v41_mlx import execution_policy as mlx_execution_policy
from .deepseek_v41_omlx_artifact import FORMAT, inspect_artifact
from .deepseek_v41_omlx_weights import (
    NUMERIC_PROFILE,
    layer_bank_bytes,
    omlx_weight_plan,
)

NUMERICS_REVISION = "mlx2.v41-omlx-affine-act-fp8-f32-qmm.v1"
# Expert-cache ceiling: everything that fits under early admission, at most
# the whole 79.1 GiB routed-expert table of this checkpoint.
MAX_EXPERT_CACHE_BYTES = 80 << 30
DESCRIPTOR = replace(
    MLX_DESCRIPTOR,
    model_type="deepseek_v41",
    variant="flash-1017-gsq-rco-coder-omlx-affine-text",
    metadata={
        "execution": "mlx2.adapters.deepseek_v41_omlx.DeepSeekV41OmlxAdapter",
        "qualification": "unqualified",
        "numeric_profile": NUMERIC_PROFILE,
        "scope": "omlx-layout affine B1 text; full-depth V4.1; 192 routed experts; exact APCv2 within this profile; DSpark and vision not served",
    },
)


# bank_gather_v1 stays opt-in (2026-11-04, M5 Max, MLX 0.32.4, checkpoint
# 6d10f797): stock gather_qmm for 2-bit g64 experts is 2x slower than the
# per-expert qmm at one decode row (2.96 vs 1.53 ms per layer, bit-exact) and
# only faster at 8+ rows (3.9 vs 10.1 ms; 34.9 vs 53.6 ms at 128), where it is
# not bit-exact. End to end it lost: 4.2 vs 7.1 tok/s warm decode, 49 vs 80
# tok/s 3.3K prefill, with 83 GiB resident pushing idle pages to swap. Evidence:
# qualification/runs/ds41-omlx-ab-20261104/evidence-r2 and evidence-micro.
# hoisted_v1: same per-expert QMM, with the FP8 input, SwiGLU and middle FP8
# each run once over the layer's expert-sorted rows; bit-exact by construction.
# First choice is the default.
EXPERT_POLICIES = ("hoisted_v1", "reference", "bank_gather_v1")


# Format-agnostic model options the Source route already exposes; the model
# constructor validates their combinations. First choice is the default.
# Defaults (Pierre, 2026-11-04): the bit-exact set, token-identical to the
# all-reference route over a 1300-char greedy replay on checkpoint 6d10f797,
# warm decode 7.29 -> 13.48 tok/s with hoisted_v1 (exploratory, no thermal
# holds; qualification/runs/ds41-omlx-ab-20261104/evidence-r3). Unqualified.
# The Source HC coefficients exclude deepseek_hc=fused_fp32: pass
# deepseek_source_hc_coefficients=reference to use that one instead.
MODEL_POLICY_CHOICES = {
    "deepseek_source_dense_qat": ("fused_fp32", "reference"),
    # async_v1: end each layer with mx.async_eval and no expert-window fence,
    # so graph building overlaps GPU work; arithmetic unchanged.
    # mxfp8_gemv_v1: wo_a as exact MXFP8 codes (bitwise round trip proven at
    # load, else the load fails) read by the Source MXFP8 GEMV; halves its bytes.
    # Both defaults (Pierre, 2026-11-04): token-identical to the prior defaults
    # over a 1300-char greedy replay; warm decode 13.02 -> 15.27 tok/s, 3.3K
    # prefill 93.8 -> 99.2 (exploratory; ds41-omlx-ab-20261104/evidence-r4).
    "deepseek_omlx_layer_eval": ("async_v1", "sync"),
    "deepseek_source_woa_storage": ("mxfp8_gemv_v1", "reference"),
    "deepseek_source_hc_coefficients": ("fused_fp32", "reference"),
    "deepseek_source_hc_mega": ("mega_v1", "reference"),
    "deepseek_source_hc_boundary": ("reference", "expand_collapse_seq"),
    "deepseek_index_policy": ("finite_all_retained", "reference"),
    "deepseek_selected_kv_policy": ("decode_reuse", "reference"),
    "deepseek_source_attention_math": ("all_retained_broadcast", "reference", "omlx_nax_bf16_online_v1",
                                       "source_short_decode_joint_f32_v1", "source_short_decode_grouped_f32_v1"),
}


def execution_policy(value=None):
    """Native affine policy surface plus this route's expert path and model options."""
    value = {} if value is None else dict(value)
    experts = value.pop("deepseek_omlx_experts", EXPERT_POLICIES[0])
    if experts not in EXPERT_POLICIES:
        raise ValueError("unsupported omlx V4.1 expert path")
    model = {key: value.pop(key, choices[0]) for key, choices in MODEL_POLICY_CHOICES.items()}
    if any(type(model[key]) is not str or model[key] not in choices for key, choices in MODEL_POLICY_CHOICES.items()):
        raise ValueError("unsupported omlx V4.1 model policy")
    return {**mlx_execution_policy(value), "deepseek_omlx_experts": experts, **model}


def model_policy_options(policy):
    # deepseek_source_dense_qat and deepseek_omlx_layer_eval are weights-provider
    # choices, not model options.
    return {"hc_coeff_policy": "source_fused_fp32" if policy["deepseek_source_hc_coefficients"] == "fused_fp32"
            else "reference",
            "source_hc_mega_policy": policy["deepseek_source_hc_mega"],
            "source_hc_boundary_policy": policy["deepseek_source_hc_boundary"],
            "index_policy": policy["deepseek_index_policy"],
            "selected_kv_policy": policy["deepseek_selected_kv_policy"],
            "source_attention_math_policy": policy["deepseek_source_attention_math"]}


def admission_plan(artifact, request, *, experts="reference", planner=omlx_weight_plan):
    """Size the expert LRU (and resident banks) from measured early admission.

    bank_gather_v1 keeps whole layers' stacked experts resident for device-side
    gather_qmm, as many as fit under admission; any remaining layers use the
    bounded per-expert LRU with whatever is left.
    """
    validate_streaming_request(request)
    args = artifact["args"]
    table = artifact["inventory"]["storage_bytes"]["expert"]
    item = artifact["inventory"]["expert_item_bytes"]
    headroom = max(5 << 30, request.headroom_bytes)
    ple_cache = (8 << 20) * len(args.engram_layer_ids)
    bank_layers = ()
    if experts == "bank_gather_v1":
        if request.budget_mode != "auto":
            raise ValueError("omlx V4.1 expert banks are sized from automatic admission only")
        layer = layer_bank_bytes(artifact)
        for count in range(args.num_hidden_layers, -1, -1):
            trial = tuple(range(count))
            plan = planner(artifact, item, trial)
            reserve = max(256 << 20, plan["metadata_bound_bytes"])
            spare = request.admission_limit_bytes - (plan["total_reservation_bytes"] + headroom + reserve + ple_cache)
            if spare >= 0:
                bank_layers = trial
                break
        requested = min(max(item, spare + item), table - layer*len(bank_layers) or item, MAX_EXPERT_CACHE_BYTES)
    else:
        requested = min(table, MAX_EXPERT_CACHE_BYTES) if request.budget_mode == "auto" else request.budget_bytes
    if type(requested) is not int or not item <= requested <= MAX_EXPERT_CACHE_BYTES:
        raise ValueError("omlx V4.1 expert cache must hold one expert and be at most 80 GiB")
    plan = planner(artifact, requested, bank_layers)
    reserve = max(256 << 20, plan["metadata_bound_bytes"])
    overhead = plan["total_reservation_bytes"] - requested + headroom + reserve + ple_cache
    ceiling = requested
    if request.budget_mode == "auto":
        ceiling = min(requested, request.admission_limit_bytes - overhead)
        if ceiling < item:
            raise MemoryError("omlx V4.1 dense weights and execution reserve exceed admission")
        plan = planner(artifact, ceiling, bank_layers)
    required = plan["total_reservation_bytes"] + headroom + reserve + ple_cache
    if required > request.admission_limit_bytes:
        raise MemoryError("omlx V4.1 dense/expert/staging reservation exceeds early admission")
    streamed = table - plan["expert_bank_bytes"]
    return {**plan, "expert_cache_bytes": ceiling, "headroom_bytes": headroom,
            "metadata_reserve_bytes": reserve, "ple_cache_bytes": ple_cache, "required_bytes": required,
            "admission_limit_bytes": request.admission_limit_bytes, "ceiling_source": request.budget_mode,
            "expert_table_bytes": table, "expert_path": experts,
            "expert_resident_fraction": round(min(table, plan["expert_bank_bytes"] + ceiling)/table, 4),
            "streamed_layer_resident_fraction": round(min(1.0, ceiling/streamed), 4) if streamed else 1.0,
            "phase": "pre_materialization"}


class DeepSeekV41OmlxAdapter(DeepSeekV41MLXAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "deepseek-v41-1017-gsq-rco-coder-omlx-affine-ordinary-b1"
    numeric_profile = NUMERIC_PROFILE
    numerics_revision = NUMERICS_REVISION
    weight_format = FORMAT
    policy_parser = staticmethod(execution_policy)
    adapter_owned_default_policy_keys = DeepSeekV41MLXAdapter.adapter_owned_default_policy_keys | {
        "deepseek_omlx_experts", *MODEL_POLICY_CHOICES}

    @staticmethod
    def weight_streaming_external_draft(policy):
        if "draft_models" in policy or policy.get("draft_model") != "prompt_lookup":
            return False
        execution_policy(policy)
        return True

    def _weight_admission_plan(self, artifact, request):
        return admission_plan(artifact, request, experts=self.kernel_policy["deepseek_omlx_experts"])

    def _construct_weights(self, artifact, request):
        from .deepseek_v41_omlx_weights import OmlxV41Weights
        return OmlxV41Weights(artifact, expert_cache_bytes=self.native_admission["expert_cache_bytes"],
                              expert_read_workers=request.read_workers, bounds=self.native_admission,
                              bank_layers=self.native_admission["expert_bank_layers"],
                              expert_math="hoisted_v1" if self.kernel_policy["deepseek_omlx_experts"] == "hoisted_v1"
                              else "reference", dense_qat=self.kernel_policy["deepseek_source_dense_qat"],
                              layer_eval=self.kernel_policy["deepseek_omlx_layer_eval"],
                              woa_storage=self.kernel_policy["deepseek_source_woa_storage"])

    def _load(self, model_path):
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )
        require_process_numerics("omlx-layout DeepSeek V4.1")
        clear_inherited_profile(())
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        os.environ.update(self.environment)
        assert_process_profile_applied("omlx-layout DeepSeek V4.1")
        artifact = self._inspect_model_artifact(model_path)
        self.omlx_artifact = artifact
        self.identity, self.config = artifact["identity"], dict(artifact["config"])
        args, root = artifact["args"], artifact["root"]
        if self.weight_streaming_request is None:
            from ..runtime.streamed_load import WeightStreamRequest
            from ..serving import preload_admission_limit_bytes
            limit = preload_admission_limit_bytes()
            if limit is None:
                raise MemoryError("omlx V4.1 default streaming requires measurable early admission")
            # Auto: as many experts as fit under measured admission stay resident.
            self.weight_streaming_request = WeightStreamRequest(
                mode="moe_experts", budget_bytes=0, budget_mode="auto",
                admission_limit_bytes=limit, read_workers=2, bypass_page_cache=True)
        request = self.weight_streaming_request
        self.native_admission = self._weight_admission_plan(artifact, request)

        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from .deepseek_v41_source import hasher_metadata
        # tokenizer.json is pinned and byte-identical to the official release.
        backend = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.metadata = hasher_metadata(backend, args)
        self.tokenizer = TokenizerWrapper(PreTrainedTokenizerFast(tokenizer_object=backend,
            bos_token=backend.id_to_token(0), eos_token=backend.id_to_token(1), pad_token=backend.id_to_token(2),
            model_max_length=args.max_position_embeddings, clean_up_tokenization_spaces=False),
            detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[1])
        self.tokenizer_receipt = {"source": "pinned tokenizer.json (official bytes)",
            "token_map_sha256": hashlib.sha256(np.array(self.metadata["deepseek41.engram.token_map"],
                                                        dtype="<i8").tobytes()).hexdigest()}

        from ..runtime.models.deepseek_v41 import DeepSeekV41Model
        from ..runtime.models.deepseek_v41_engram import (
            EngramHasher,
            EngramRows,
            file_stamp,
        )
        from ..runtime.weight_residency import wire_serving_weights
        self.preload_residency = wire_serving_weights()
        self.weights = self._construct_weights(artifact, request)
        self.weight_stream = self.weights.expert_stream
        self.weight_stream.plan.resident_remainder_bytes = (
            self.native_admission["dense_payload_bytes"] + self.native_admission["metadata_reserve_bytes"])
        self.weight_stream.plan.ceiling_source = request.budget_mode
        from .deepseek_v41_source_artifact import checked_file
        for layer, table in artifact["engram"].items():
            # Seal check in the seal's own stamp format; EngramRows then
            # re-verifies its descriptor against its own stamp on every read.
            path = checked_file(root, table["file"], artifact["seals"][table["file"]])
            self.engram_tables[layer] = EngramRows(path, offset=table["offset"], scale_offset=table["scale_offset"],
                bias_offset=table["bias_offset"], rows=table["rows"], expected_stat=file_stamp(path.stat()),
                workers=min(8, request.read_workers), cache_format=self.kernel_policy["deepseek_engram_cache"])
        self.weight_stream.bind_ple_providers(self.engram_tables)
        binding = (self.identity["fingerprint"] + ":" + self.numerics_revision + ":"
                   + hashlib.sha256(json.dumps(self.kernel_policy, sort_keys=True).encode()).hexdigest())
        self.model = DeepSeekV41Model(args, self.weights, identity=binding, hasher=EngramHasher(self.metadata),
            engram_tables=self.engram_tables, hc_policy=self.kernel_policy["deepseek_hc"],
            qat_policy=self.kernel_policy["deepseek_qat"], **model_policy_options(self.kernel_policy),
            **model_options(self.kernel_policy))
        self.model.verification_commit_policy = self.kernel_policy["deepseek_verify_commit"]
        self.model.verify_rows_policy = self.kernel_policy["deepseek_verify_rows"]
        self.model.pointwise_policy = self.kernel_policy["deepseek_pointwise"]
        self.model.attention_epilogue_policy = self.kernel_policy["deepseek_attention_epilogue"]
        self.model.hc_epilogue_policy = self.kernel_policy["deepseek_hc_epilogue"]
        self.model.verified_commit_stats = {}
        self.model.preload().eval()
        self.weights.verify_source()
        import mlx.core as mx
        limit = request.admission_limit_bytes
        if (limit is not None and mx.get_active_memory() + self.weight_stream.unfilled_reserve_bytes()
                + self.native_admission["headroom_bytes"] + self.native_admission["metadata_reserve_bytes"] > limit):
            raise MemoryError("omlx V4.1 retained tensors plus expert reserve exceed admission")
        self.layout = type(self).descriptor.cache_layout
        self.max_context = self.model.context_limit
        self.descriptor = type(self).descriptor
        bind_cache_layout(self)
        self._configure_copy_chain()

    def diagnostics(self):
        result = super().diagnostics()
        storage = self.omlx_artifact["inventory"]["storage_bytes"]
        result.update(
            weights=self.weight_format, numeric_profile=self.numeric_profile,
            numerics_revision=self.numerics_revision, original_gguf_required=False,
            checkpoint=f"{self.omlx_artifact['identity']['repository']}@{self.identity['fingerprint']}",
            checkpoint_revision=self.omlx_artifact["identity"]["revision"],
            routed_experts=self.omlx_artifact["args"].n_routed_experts,
            scheduled_hc_calls=dict(getattr(self.model, "scheduled_hc_calls", {})),
            embedded_dspark={"present_bytes": storage["mtp"], "served": False,
                             "reason": "omlx MXFP8/MXFP4 draft binding not implemented on this route"},
            vision={"present_bytes": storage["vision"], "served": False,
                    "reason": "artifact-provided vision reader not implemented on this route"})
        return result
