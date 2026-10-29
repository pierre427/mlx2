"""Native packed DeepSeek V4 text serving; optional routes are explicit gates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import SamplingDefaults, VendorSampling
from .deepseek_v4_gguf_artifact import (
    FILENAME,
    MODEL_TYPE,
    REFERENCE_FILE,
    inspect_artifact,
)
from .ordinary_text import OrdinaryTextAdapter

CACHE_LAYOUT = "deepseek-v4-local-overlap-pooled-b1-v1"
SUPPORTED_CONTEXT = 65536
DESCRIPTOR = ModelDescriptor(
    model_type=MODEL_TYPE,
    family="deepseek_v4",
    variant="vision-exp-packed-text",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.SPARSE_INDEX,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.REASONING,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.deepseek_v4_gguf.DeepSeekV4GGUFAdapter",
        "qualification": "unqualified",
        "scope": "native packed ordinary B1 text; exact frontier cache snapshots; "
        "vision and speculative routes remain unavailable until implemented",
    },
)
SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        source="GGUF general.sampling.temp/top_p; neutral top_k",
    ),
    model="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
)
EXTERNAL_POLICY_KEYS = frozenset(
    {"draft_model", "num_draft", "deepseek_verifier", "deepseek_draft_confidence"}
)
EXTERNAL_SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        source="unqualified DS4 DSpark sequential route supports greedy only",
    ),
    model="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp+DSpark",
)

POLICY_CHOICES = {
    "deepseek_attention": (
        "reference",
        "visible_rows_fp32",
        "packed_visible_rows_fp32",
    ),
    "deepseek_kv": ("fp32", "fp8_mixed"),
    "deepseek_embedding": ("resident", "file_rows"),
    "deepseek_packed": ("reference", "cooperative"),
    "deepseek_hc": ("reference", "fused_fp32"),
    "deepseek_hc_projection": ("reference", "fixed_fp32"),
    "deepseek_rope_qat": ("reference", "fused_fp32"),
    "deepseek_expert_prefill": ("reference", "segmented_nax_fp16"),
    "deepseek_dense_prefill": ("reference", "nax_fp16"),
}


def split_execution_policy(value):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - (
        POLICY_CHOICES.keys() | EXTERNAL_POLICY_KEYS
    ):
        raise ValueError("unsupported DeepSeek packed execution policy")
    result = {
        key: value.get(key, choices[0]) for key, choices in POLICY_CHOICES.items()
    }
    if any(
        type(result[key]) is not str or result[key] not in choices
        for key, choices in POLICY_CHOICES.items()
    ):
        raise ValueError("unsupported DeepSeek packed execution policy value")
    if (
        result["deepseek_attention"] == "packed_visible_rows_fp32"
        and result["deepseek_kv"] != "fp8_mixed"
    ):
        raise ValueError(
            "direct packed attention requires explicit deepseek_kv=fp8_mixed"
        )
    external = {key: value[key] for key in EXTERNAL_POLICY_KEYS if key in value}
    if external:
        path = external.get("draft_model")
        count = external.get("num_draft", 5)
        if (
            type(path) is not str
            or not path
            or type(count) is not int
            or not 1 <= count <= 5
        ):
            raise ValueError(
                "DSpark requires exact draft_model GGUF path and num_draft1..5"
            )
        verifier = external.get("deepseek_verifier", "sequential")
        if verifier not in ("sequential", "causal_slab"):
            raise ValueError("unsupported DeepSeek external verifier")
        confidence = external.get("deepseek_draft_confidence", 0.6)
        if (
            type(confidence) not in (int, float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("deepseek_draft_confidence must be finite in [0,1]")
        external["deepseek_draft_confidence"] = float(confidence)
        external["num_draft"] = count
        external["deepseek_verifier"] = verifier
    return result, external


def validate_execution_policy(value):
    # Retain the existing public kernel-policy helper and defaults.
    return split_execution_policy(value)[0]


class DeepSeekV4GGUFAdapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    sampling_defaults = SAMPLING
    profile = "deepseek-v4-vision-exp-native-gguf-ordinary-b1"
    default_route = "ordinary"
    prompt_lookup_default = False
    adapter_owned_default_policy_keys = frozenset(POLICY_CHOICES)

    def __init__(self, model_path: str, *, execution_policy=None):
        self.kernel_policy, self.external_policy = split_execution_policy(
            execution_policy
        )
        self.draft_model = self.draft_weights = None
        self.external_record = self.external_source = None
        if execution_policy:
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
            except Exception as cleanup_error:  # noqa: BLE001 - retain the original load failure
                error.add_note(
                    f"DeepSeek weight cleanup also failed: {cleanup_error!r}"
                )
            raise

    def _load(self, model_path):
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        require_process_numerics("the native DeepSeek V4 packed profile")
        clear_inherited_profile(())
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("the native DeepSeek V4 packed profile")
        artifact = inspect_artifact(model_path)
        if self.external_policy:
            from .deepseek_v4_dspark_artifact import (
                inspect_sidecar,
                prepare_sidecar,
                seal_path,
                verified_source_identity,
            )

            sidecar_path = Path(self.external_policy["draft_model"])
            # Missing preparation is a first-use full SHA check, never a
            # download-receipt shortcut. A stale existing seal fails closed.
            self.external_record = (
                inspect_sidecar(sidecar_path)
                if seal_path(sidecar_path).exists()
                else prepare_sidecar(sidecar_path)
            )
            self.external_source = verified_source_identity()
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT
        if self.kernel_policy["deepseek_kv"] == "fp8_mixed":
            from ..runtime.models.deepseek_v4_packed_cache import (
                CACHE_LAYOUT as packed_layout,
            )

            self.layout = packed_layout
        self.descriptor = replace(DESCRIPTOR, cache_layout=self.layout)
        self.metadata = artifact["metadata"]
        self.config = dict(artifact["config"])
        from ..runtime.gguf.weights import GGUFWeights
        from ..runtime.models.deepseek_v4 import (
            DeepSeekV4Config,
            DeepSeekV4Model,
            validate_weight_inventory,
        )
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from .deepseek_v4_gguf_tokenizer import load_tokenizer

        args = DeepSeekV4Config.from_gguf(self.metadata)
        if self.external_policy:
            self._validate_external_target(args)

        path = Path(self.identity["path"])
        # Validate the vocabulary/template before uploading eighty GiB of
        # packed weights, so tokenizer refusal cannot strand model storage.
        raw = json.loads((path / REFERENCE_FILE).read_text())
        tokenizer, self.tokenizer_receipt = load_tokenizer(raw, self.metadata)
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[int(self.metadata["tokenizer.ggml.eos_token_id"])],
        )
        self.weights = GGUFWeights(
            path / FILENAME,
            embedding_policy=self.kernel_policy["deepseek_embedding"],
            expected_stat=artifact["weight_stat"],
            compute_policy=self.kernel_policy["deepseek_packed"],
            expert_prefill_policy=self.kernel_policy["deepseek_expert_prefill"],
            dense_prefill_policy=self.kernel_policy["deepseek_dense_prefill"],
        )
        validate_weight_inventory(args, self.weights.tensors)
        from ..runtime.models.deepseek_v4_identity import NATIVE_NUMERICS_REVISION

        self.model = DeepSeekV4Model(
            args,
            self.weights,
            identity=self.identity["fingerprint"]
            + ":" + NATIVE_NUMERICS_REVISION + ":compute="
            + self.kernel_policy["deepseek_packed"],
            attention_policy=self.kernel_policy["deepseek_attention"],
            cache_policy=self.kernel_policy["deepseek_kv"],
            hc_policy=self.kernel_policy["deepseek_hc"],
            hc_projection_policy=self.kernel_policy["deepseek_hc_projection"],
            rope_qat_policy=self.kernel_policy["deepseek_rope_qat"],
            expert_prefill_policy=self.kernel_policy["deepseek_expert_prefill"],
            dense_prefill_policy=self.kernel_policy["deepseek_dense_prefill"],
        )
        # Admission observes real resident weight storage, never a lazy model
        # whose first request unexpectedly uploads another eighty GiB.
        self.model.preload().eval()
        self.max_context = min(args.max_position_embeddings, SUPPORTED_CONTEXT)
        if self.external_policy:
            self._load_external()

    def _validate_external_target(self, args):
        # Sealed CPU metadata must reject incompatible targets before any
        # eighty-GiB target or six-GiB sidecar device allocation.
        from ..runtime.drafters.deepseek_v4_dspark import DeepSeekV4DSparkConfig

        fields = dict(self.external_record["config"])
        fields["target_layers"] = tuple(fields["target_layers"])
        DeepSeekV4DSparkConfig(**fields).validate_target(args)

    def _load_external(self):
        import mlx.core as mx

        from ..runtime.drafters.deepseek_v4_dspark import (
            DeepSeekV4DSparkBackbone,
            DeepSeekV4DSparkConfig,
        )
        from ..runtime.gguf.weights import GGUFWeights
        from .deepseek_v4_external import DeepSeekV4DSparkDraftModel

        record = self.external_record
        from .deepseek_v4_dspark_artifact import file_stat

        if file_stat(Path(record["path"])) != record["stat"]:
            raise ValueError("sidecar changed before packed preload")
        self.draft_weights = GGUFWeights(
            record["path"],
            expected_stat=record["stat"],
            compute_policy=self.kernel_policy["deepseek_packed"],
            expert_prefill_policy=self.kernel_policy["deepseek_expert_prefill"],
        )
        config = DeepSeekV4DSparkConfig.from_index(
            self.draft_weights.metadata, self.draft_weights.tensors
        )
        config.validate_target(self.model.args)
        # Admission must see the whole6GB sidecar resident, not a lazy request.
        for name, tensor in self.draft_weights.tensors.items():
            if tensor.ggml_type in (0, 1):
                self.draft_weights.array(name)
            else:
                self.draft_weights.packed(name)
        mx.eval(list(self.draft_weights._dense.values()))
        if file_stat(Path(record["path"])) != record["stat"]:
            raise ValueError("sidecar changed during packed preload")
        backbone = DeepSeekV4DSparkBackbone(
            config,
            self.model.args,
            self.draft_weights,
            self.weights,
            target_sha256=self.identity["weight_sha256"],
            sidecar_sha256=record["sha256"],
            target_metadata=self.metadata,
            source_snapshot=self.external_source["source_snapshot_sha256"],
        )
        # Constructor creates HC staging and rotary arrays after raw uploads.
        # Account/materialize every support plane before serving is published.
        backbone.preload()
        self.draft_model = DeepSeekV4DSparkDraftModel(
            backbone,
            verification_mode=self.external_policy["deepseek_verifier"],
            confidence_threshold=self.external_policy.get(
                "deepseek_draft_confidence", 0.6
            ),
        ).bind(self.model)
        self.target_identity = dict(self.identity)
        from ..runtime.models.deepseek_v4_identity import dspark_artifact_fingerprint

        binding = dspark_artifact_fingerprint(
            self.draft_model.execution_binding, self.external_policy["num_draft"]
        )
        self.identity = {
            **self.identity,
            "target_fingerprint": self.identity["fingerprint"],
            "draft_fingerprint": record["sha256"],
            "fingerprint": binding,
            "external_source_snapshot": self.external_source["source_snapshot_sha256"],
        }
        self.layout += ":dspark-chain-v1:" + binding
        self.profile = (
            "deepseek-v4-vision-exp-dspark-greedy-"
            + self.external_policy["deepseek_verifier"]
            + "-b1:"
            + binding
        )
        self.default_route = "external_draft"
        self.sampling_defaults = EXTERNAL_SAMPLING
        self.descriptor = replace(
            self.descriptor,
            capabilities=self.descriptor.capabilities | {Capability.EXTERNAL_DRAFT},
            state_planes=self.descriptor.state_planes | {StatePlane.DRAFT},
            cache_layout=self.layout,
            metadata={
                **self.descriptor.metadata,
                "scope": "unqualified B1 greedy DSpark chain; explicit verifier="
                + self.external_policy["deepseek_verifier"],
                "qualification": "unqualified",
                "sampling_contract": {
                    "schema": "mlx2.route-sampling-contract.v1",
                    "modes": ["greedy"],
                    "unsupported": "refuse",
                },
                "packed_ast": False,
                "external_binding": binding,
                "sidecar": record,
                "source": self.external_source,
            },
        )

    def create_external_batch(self, **kwargs):
        if self.draft_model is None:
            raise ValueError("no DeepSeek DSpark sidecar bound")
        from .deepseek_v4_external import DeepSeekV4ExternalBatchGenerator

        if (
            kwargs.get("num_draft", self.external_policy["num_draft"])
            != self.external_policy["num_draft"]
        ):
            raise ValueError("external draft depth differs from bound adapter profile")
        kwargs.setdefault("num_draft", self.external_policy["num_draft"])
        kwargs.setdefault("cost_per_commit_handoff", False)
        return DeepSeekV4ExternalBatchGenerator(
            self.model,
            draft_model=self.draft_model,
            target_artifact_fingerprint=self.target_identity["fingerprint"],
            **kwargs,
        )

    def _validate_external_request(self, request):
        if getattr(self, "draft_model", None) is not None:
            value = request.get("temperature", 0.0)
            if type(value) not in (int, float) or value != 0:
                raise ValueError(
                    "DeepSeek DSpark sequential route supports greedy temperature0 only"
                )

    def execution_config(self, *, max_lanes, prefill_step):
        if max_lanes != 1:
            raise ValueError("native DeepSeek V4 currently implements physical B1 only")
        result = super().execution_config(
            max_lanes=max_lanes, prefill_step=prefill_step
        )
        if getattr(self, "draft_model", None) is not None:
            result.update(
                backend="external_draft", num_draft=self.external_policy["num_draft"]
            )
        return result

    def prefill_step_default(self):
        return 512

    def cache_budget(self, *, mtp):
        if mtp:
            raise ValueError("native DeepSeek V4 MTP route is not implemented yet")
        from .deepseek_v4_memory import DeepSeekV4CacheBudget

        if self.kernel_policy["deepseek_kv"] == "fp8_mixed":
            from .deepseek_v4_packed_memory import DeepSeekV4PackedCacheBudget

            if self.kernel_policy["deepseek_attention"] == "packed_visible_rows_fp32":
                from .deepseek_v4_packed_memory import DeepSeekV4DirectPackedCacheBudget

                return DeepSeekV4DirectPackedCacheBudget.from_config(self.model.args)
            return DeepSeekV4PackedCacheBudget.from_config(self.model.args)
        return DeepSeekV4CacheBudget.from_config(self.model.args)

    @staticmethod
    def thinking_enabled(request):
        value = request.get("enable_thinking", False)
        if type(value) is not bool:
            raise ValueError("enable_thinking must be boolean")
        return value

    def _template(self, request, *, tokenize):
        self._validate_external_request(request)
        if request.get("tools") and request.get("tool_choice") != "none":
            raise ValueError("DeepSeek DSML tool execution has not been implemented")
        if "messages" not in request:
            value = request["prompt"]
            return (
                self.tokenizer.encode(value, add_special_tokens=False)
                if tokenize
                else value
            )
        return self.tokenizer.apply_chat_template(
            copy.deepcopy(request["messages"]),
            tokenize=tokenize,
            add_generation_prompt=True,
            enable_thinking=self.thinking_enabled(request),
        )

    def prompt_tokens(self, request):
        return self._template(request, tokenize=True)

    def render_prompt(self, request):
        return self._template(request, tokenize=False)

    def thinking_close_token_ids(self):
        ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        return tuple(ids) if len(ids) == 1 else None

    def output_parser(self, request):
        self._validate_external_request(request)
        from ..output import OutputParser, callable_tools

        if callable_tools(request):
            raise ValueError("DeepSeek DSML tool execution is unavailable")
        return OutputParser(
            chat="messages" in request,
            thinking=self.thinking_enabled(request),
            tools=None,
            stops=request.get("stop", ()),
        )

    def diagnostics(self):
        projection_status = None
        if self.kernel_policy["deepseek_hc_projection"] == "fixed_fp32":
            from ..runtime.models.deepseek_v4_hc_projection import route_status

            projection_status = route_status()
        direct_status = None
        if self.kernel_policy["deepseek_attention"] == "packed_visible_rows_fp32":
            from ..runtime.models.deepseek_v4_packed_visible_attention import (
                route_status,
            )

            direct_status = route_status()
        return {
            "packed_visible_attention": direct_status,
            "hc_projection_status": projection_status,
            "architecture": "deepseek-v4-hc-hisa-packed",
            "cache_layout": self.layout,
            "route": "external_draft" if self.draft_model is not None else "ordinary",
            "qualification": "unqualified",
            "weights": "IQ2_XXS/Q2_K/Q8_0 native blocks",
            "physical_width": 1,
            "supported_context": self.max_context,
            "kv_storage": self.kernel_policy["deepseek_kv"],
            "execution_policy": dict(self.kernel_policy),
            "embedding": self.kernel_policy["deepseek_embedding"],
            "ple": "not applicable; checkpoint has no PLE tables",
            "scheduled_expert_calls": dict(self.weights.scheduled_expert_calls),
            "scheduled_grouped_calls": dict(self.weights.scheduled_grouped_calls),
            "scheduled_dense_calls": dict(self.weights.scheduled_dense_calls),
            "scheduled_call_observation": "Python scheduling only; does not establish completed GPU execution",
            "tensorfold": "not selected",
            "nax_prefill": self.kernel_policy["deepseek_expert_prefill"],
            "dense_nax_prefill": self.kernel_policy["deepseek_dense_prefill"],
            "hc_projection": self.kernel_policy["deepseek_hc_projection"],
            "vision": "unavailable",
            "speculation": self.draft_model.receipt_settings
            if self.draft_model is not None
            else "unavailable",
            "sidecar": self.external_record,
            "source": self.external_source,
        }

    def close(self):
        from .process_globals import release

        self.model = self.draft_model = None
        draft_weights = getattr(self, "draft_weights", None)
        self.draft_weights = None
        weights = getattr(self, "weights", None)
        self.weights = None
        try:
            if draft_weights is not None:
                draft_weights.close()
        finally:
            try:
                if weights is not None:
                    weights.close()
            finally:
                release(self)
