"""Owned Nemotron 3.5 Super VL adapter and source-bound mixed affine loader.

Artifact inspection is CPU-only. All serving routes remain unqualified until
tested against the completed converted artifact. See provenance/.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..conversion.nemotron35_plan import make_plan
from ..conversion.nvfp4 import read_header, sha256
from ..sampling_defaults import MODEL_CARD, SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin
from .nemotron3_super import Nemotron3SuperAdapter, configure_environment
from .nemotron35_super_config import (
    SOURCE_CONFIG_SHA256,
    SOURCE_REVISION,
    language_config,
    vision_config,
)

CACHE_LAYOUT = "nemotron35-super-vl-hybrid-fp32-router-v1"
EXECUTION_LIMITS = {
    "active_sessions": "1_default_or_2_or_4_exact_progressive_external",
    "draft_depth": "head_or_runtime_bound",
    "verification_rows": "request_private_per_session",
}
DESCRIPTOR = ModelDescriptor(
    model_type="nemotron_h_omni",
    family="nemotron35-super-vl-120b-a12b",
    variant="mixed-affine",
    cache_layout=CACHE_LAYOUT,
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
            StatePlane.DRAFT,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.VISION,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.GRAMMAR,
            Capability.MTP,
        }
    ),
    metadata={
        "execution": "mlx2.adapters.nemotron35_super.Nemotron35SuperAdapter",
        "qualification": "unqualified",
        "ordinary_reference": "implemented",
        "vision": "image-prefill-b1-candidate",
        "video": "encoder-only-no-serving-route",
        "native_mtp": "candidate",
        "external_draft": "requires-bound-head",
        "execution_limits": EXECUTION_LIMITS,
    },
)
SAMPLING = VendorSampling(
    {
        "text": SamplingDefaults(temperature=1.0, top_p=0.95, source=MODEL_CARD),
        "vision": SamplingDefaults(temperature=0.6, top_p=0.95, source=MODEL_CARD),
    },
    general="text",
    model="NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
)


def _split_super_execution_policy(execution_policy):
    if execution_policy is not None and not isinstance(execution_policy, dict):
        raise ValueError("Super VL execution policy must be an object")
    policy = dict(execution_policy or {})
    expert_sort = policy.pop("expert_gather_sort", "auto")
    mamba_conv = policy.pop("mamba_conv", "auto")
    mamba_norm = policy.pop("mamba_norm", "auto")
    mamba_ssm = policy.pop("mamba_ssm", "auto")
    mamba_projection = policy.pop("mamba_projection", "auto")
    mamba_segment = policy.pop("mamba_segment", "auto")
    relu2_down = policy.pop("moe_relu2_down", "auto")
    if not isinstance(expert_sort, str) or expert_sort not in {
        "auto",
        "unsorted_decode",
    }:
        raise ValueError("Super VL expert_gather_sort must be auto or unsorted_decode")
    if not isinstance(mamba_conv, str) or mamba_conv not in {
        "auto",
        "fused_b1",
        "fused_b1_compact",
        "fused_short",
    }:
        raise ValueError(
            "Super VL mamba_conv must be auto, fused_b1, fused_b1_compact, "
            "or fused_short"
        )
    if not isinstance(mamba_norm, str) or mamba_norm not in {
        "auto",
        "fused_b1",
        "fused_short",
    }:
        raise ValueError("Super VL mamba_norm must be auto, fused_b1, or fused_short")
    if not isinstance(mamba_ssm, str) or mamba_ssm not in {
        "auto",
        "fused_short",
        "fused_short_gate",
    }:
        raise ValueError(
            "Super VL mamba_ssm must be auto, fused_short, or fused_short_gate"
        )
    if not isinstance(relu2_down, str) or relu2_down not in {"auto", "fused_b1"}:
        raise ValueError("Super VL moe_relu2_down must be auto or fused_b1")
    if not isinstance(mamba_segment, str) or mamba_segment not in {
        "auto",
        "layerwise_short_exact",
        "layerwise_short_exact_routed_batch",
    }:
        raise ValueError(
            "Super VL mamba_segment must be auto, layerwise_short_exact, or "
            "layerwise_short_exact_routed_batch"
        )
    if not isinstance(mamba_projection, str) or mamba_projection not in {
        "auto",
        "compact_padded",
        "compact_row_exact",
        "adaptive_row_exact",
        "short_row_exact",
        "native_qmm_residual_b1",
    }:
        raise ValueError(
            "Super VL mamba_projection must be auto, compact_padded, or "
            "compact_row_exact, adaptive_row_exact, short_row_exact, or "
            "native_qmm_residual_b1"
        )
    return (
        expert_sort,
        mamba_conv,
        mamba_norm,
        mamba_ssm,
        mamba_projection,
        mamba_segment,
        relu2_down,
        policy,
    )


MOE_NAX_GATHER_MODES = ("off", "gather")


def _split_moe_nax_gather(execution_policy):
    """``(policy without "moe_nax_gather", mode)``; the mode defaults to off.

    "gather" runs prefill expert gathers through the segmented NAX sorted
    kernel (``runtime/models/moe_nax_gather.py``), bit-identical to the stock
    sorted kernel; "fused" needs a fused gate/up module Nemotron's relu2
    experts do not have.
    """
    if not isinstance(execution_policy, dict) or "moe_nax_gather" not in execution_policy:
        return execution_policy, "off"
    policy = dict(execution_policy)
    mode = policy.pop("moe_nax_gather")
    if mode not in MOE_NAX_GATHER_MODES:
        raise ValueError("Super VL moe_nax_gather must be off or gather")
    return policy, mode


def _moe_nax_gather_report(mode):
    from ..runtime.models import moe_nax_gather as _moe_nax

    # The qualifier reads engaged calls and fallbacks from the status keys
    # (qualification.moe_nax_gather_engagement), so they sit at top level.
    return {
        **(_moe_nax.status() if mode != "off" else {"mode": "off"}),
        "schema": "mlx2.nemotron35-super-moe-nax-gather.v1",
        "implemented": True,
        "qualified": False,
        "selected": mode != "off",
        "policy": mode,
    }


NEMOTRON_INT8_PREFILL_SCOPES = ("mlp", "all")


def nemotron_int8_prefill_select(scope):
    """Trunk projections that may take the int8 (W8A8) prefill route.

    ``mlp``: the 8-bit shared experts (``mixer.shared_experts.*``).  ``all``
    adds the Mamba ``in_proj``/``out_proj`` and attention ``q/k/v/o``.  The
    routed experts (sorted gathers), the bf16 router and latent projections,
    the LM head, the MTP head and any media tower stay stock.  Approximate:
    the Mamba projections feed 40 recurrent layers, so this needs a
    long-context KL check before it is ever selected by default.
    """

    def select(path, module):
        segments = path.split(".")
        if segments[:2] != ["backbone", "layers"] or len(segments) < 5:
            return False
        if segments[3] != "mixer" or "switch_mlp" in segments:
            return False
        if "shared_experts" in segments:
            return segments[-1] in {"up_proj", "down_proj"}
        if scope != "all":
            return False
        return segments[-1] in {"in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"}

    return select


def _super_expert_gathers(model):
    layers = list(model.layers)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        layers.extend(mtp.layers)
    return [
        layer.mixer.switch_mlp
        for layer in layers
        if getattr(layer, "block_type", None) == "E"
    ]


def _super_mamba_mixers(model):
    layers = list(model.layers)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        layers.extend(mtp.layers)
    return [
        layer.mixer for layer in layers if getattr(layer, "block_type", None) == "M"
    ]


def _super_moe_mixers(model):
    layers = list(model.layers)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        layers.extend(mtp.layers)
    return [
        layer.mixer for layer in layers if getattr(layer, "block_type", None) == "E"
    ]


def native_draft_depth(value):
    """Preserve the runtime's bounded depth range without imposing a three-row cap."""
    from .mtp_depth_cap import MAX_SELF_MTP_DEPTH_CAP, validate_self_mtp_num_draft

    return validate_self_mtp_num_draft(
        value, {"MLX2_MTP_DEPTH_CAP": str(MAX_SELF_MTP_DEPTH_CAP)}
    )


def inspect_artifact(model_path):
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    language_config(config)
    vision_config(config)
    receipt = json.loads((path / "mlx2-conversion.json").read_text())
    if (
        receipt.get("schema") != "mlx2.nemotron35-super-vl-conversion.v1"
        or receipt.get("source_revision") != SOURCE_REVISION
        or receipt.get("source_config_sha256") != SOURCE_CONFIG_SHA256
        or receipt.get("output_config_sha256") != sha256(path / "config.json")
        or receipt.get("output_index_sha256")
        != sha256(path / "model.safetensors.index.json")
    ):
        raise ValueError("Super VL requires a source-bound completed mixed conversion")
    plan = make_plan(config, expert_bits=receipt.get("expert_bits"))
    expected_quant = {
        "bits": plan["expert_bits"],
        "group_size": 64,
        "mode": "affine",
        **plan["modules"],
    }
    if (
        config.get("quantization") != expected_quant
        or receipt.get("recipe") != plan["recipe"]
        or receipt.get("tensor_bytes") != plan["tensor_bytes"]
    ):
        raise ValueError("Super VL mixed precision policy differs from its recipe")
    mapping = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if set(mapping) != set(plan["schema"]):
        raise ValueError("Super VL index has missing or unexpected tensors")
    digest, records = hashlib.sha256(), []
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "mlx2-conversion.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"Missing or escaped Super VL metadata: {name}")
        digest.update(name.encode())
        digest.update(item.read_bytes())
    output_records = {item["file"]: item for item in receipt["output_shards"]}
    if set(output_records) != set(mapping.values()):
        raise ValueError("Super VL conversion receipt shard closure differs")
    for name in sorted(output_records):
        if Path(name).name != name or "\\" in name or not name.endswith(".safetensors"):
            raise ValueError("Super VL shard path escapes artifact")
        item = path / name
        if (
            item.is_symlink()
            or not item.is_file()
            or item.stat().st_size != output_records[name]["bytes"]
        ):
            raise ValueError("Super VL shard is absent, escaped or truncated")
        header, _ = read_header(item)
        tensors = {key: entry for key, entry in header.items() if key != "__metadata__"}
        if set(tensors) != {key for key, shard in mapping.items() if shard == name}:
            raise ValueError("Super VL shard/index differs")
        for key, entry in tensors.items():
            if {field: entry[field] for field in ("shape", "dtype")} != plan["schema"][
                key
            ]:
                raise ValueError(f"Super VL tensor geometry differs: {key}")
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
        digest.update(json.dumps(header, sort_keys=True).encode())
    return {
        "config": config,
        "weight_map": mapping,
        "conversion": receipt,
        "has_mtp": True,
        "quantization": plan["modules"],
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
    }


class Nemotron35SuperAdapter(ExternalDraftAdapterMixin, Nemotron3SuperAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    # Locked in by Pierre on 2026-10-18 after the feature sweep: native MTP
    # depth 2 is the default text route.  Greedy output was identical to
    # ordinary on every probe (feature sweep 33235e9f1, mtp-d2 unit), row-exact
    # canaries 0 failed, decode about 1.3-2x ordinary at 4K-32K (exploratory
    # single-repetition ladders on snapshot 6f843160).  Images are served on
    # the ordinary route only (``multimodal_routes``), so a multimodal server
    # passes --ordinary.  Not a qualification: the route stays unqualified.
    default_route = "native_mtp"
    multimodal_routes = frozenset({"ordinary"})
    default_mtp_ordinary_handoff_max_width = None
    # The exact kernels of that route: the fused short Mamba conv and gated
    # SSM, short row-exact projections, the layerwise short-exact routed-batch
    # segment, and the bit-identical NAX sorted MoE gather for prefill.  An
    # operator policy that names any of these keys wins.
    adapter_owned_default_policy_keys = frozenset({
        "mamba_conv", "mamba_projection", "mamba_segment", "mamba_ssm",
        "moe_nax_gather", "num_draft",
    })
    default_route_execution_policy = {
        "native_mtp": {
            "mamba_conv": "fused_short",
            "mamba_projection": "short_row_exact",
            "mamba_segment": "layerwise_short_exact_routed_batch",
            "mamba_ssm": "fused_short_gate",
            "moe_nax_gather": "gather",
            "num_draft": 2,
        },
    }
    think_close_separator = ""
    EXTERNAL_DEFAULT_NUM_DRAFT = 7
    EXTERNAL_ROUTE_TAG = "super-vl-external-v1"
    EXTERNAL_PROFILE = "nemotron35-super-vl-apcv2-external"
    # int8 (W8A8) prefill: declared, never selected without an explicit
    # --int8-prefill or a matching qualification record (int8_prefill auto).
    int8_prefill_supported = staticmethod(lambda: NEMOTRON_INT8_PREFILL_SCOPES)
    int8_prefill_select = staticmethod(nemotron_int8_prefill_select)
    EXTERNAL_POLICY_EXTENSIONS = frozenset({"draft_weight_quantization"})
    incremental_tokenizer_renderer_revision = "nemotron35-super-vl-renderer-v1"

    def __init__(self, model_path, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self, lambda: self._init_super(model_path, execution_policy)
        )

    def _init_super(self, model_path, execution_policy):
        from .process_globals import claim_moe, claim_stock_moe

        execution_policy, self.moe_nax_gather = _split_moe_nax_gather(execution_policy)
        (
            expert_sort,
            mamba_conv,
            mamba_norm,
            mamba_ssm,
            mamba_projection,
            mamba_segment,
            relu2_down,
            policy,
        ) = _split_super_execution_policy(execution_policy)
        external = bool(policy.get("draft_model"))
        if external:
            self._parse_external_policy(policy, family="Nemotron 3.5 Super VL")
        elif policy and set(policy) - {"num_draft"}:
            raise ValueError("Super VL native policy supports num_draft only")
        else:
            self._parse_external_policy(None, family="Nemotron 3.5 Super VL")
        from .nemotron35_draft_quantization import normalize_runtime_quantization

        draft_weight_quantization = normalize_runtime_quantization(
            self.external_policy.get("draft_weight_quantization")
        )
        self.expert_gather_sort = expert_sort
        self.mamba_conv = mamba_conv
        self.mamba_norm = mamba_norm
        self.mamba_ssm = mamba_ssm
        self.mamba_projection = mamba_projection
        self.mamba_segment = mamba_segment
        self.moe_relu2_down = relu2_down
        self._num_draft = native_draft_depth(
            3 if external else policy.get("num_draft", 3)
        )
        artifact = inspect_artifact(model_path)
        self.identity, self.layout = artifact["identity"], CACHE_LAYOUT
        self.source_config, self.conversion = artifact["config"], artifact["conversion"]
        draft_record = None
        draft_loader = None
        if external:
            draft_config = json.loads(
                (Path(self.external_policy["draft_model"]) / "config.json").read_text()
            )
            if draft_config.get("architectures") == ["DSparkDraftModel"]:
                from .nemotron35_dspark import inspect_drafter, load_drafter

                draft_record = inspect_drafter(
                    self.external_policy["draft_model"], model_path
                )

                def draft_loader(record, target_model):
                    return load_drafter(
                        record,
                        target_model,
                        runtime_quantization=draft_weight_quantization,
                    )
            else:
                from .lilicorr_nemotron import inspect_drafter, load_drafter

                draft_record = inspect_drafter(
                    self.external_policy["draft_model"], target_artifact=artifact
                )
                contract = self.nemotron_lilicorr_target_contract(draft_record)

                def draft_loader(record, target_model):
                    return load_drafter(
                        record,
                        target_model,
                        target_contract=contract,
                        runtime_quantization=draft_weight_quantization,
                    )
            self.external_policy.setdefault(
                "num_draft", draft_record["args"].block_size - 1
            )
            self._check_num_draft(draft_record)
        self.environment = configure_environment()
        self.environment["MLX_LM_MTP_BOUNDARY_COW"] = "1"
        os.environ["MLX_LM_MTP_BOUNDARY_COW"] = "1"
        if self.moe_nax_gather == "off":
            claim_stock_moe(self, "the Nemotron 3.5 Super VL adapter")
        else:
            claim_moe(self, "the Nemotron 3.5 Super VL adapter", self.moe_nax_gather)
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.nemotron35_super import Model
        from ..runtime.models.nemotron_h import ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        path = Path(self.identity["path"])
        self.model = Model(ModelArgs.from_dict(language_config(self.source_config)))
        weights = self.model.sanitize(
            load_shards_evicting(
                path / name for name in sorted(set(artifact["weight_map"].values()))
            )
        )
        modules = {
            name.removeprefix("language_model."): quant
            for name, quant in artifact["quantization"].items()
        }
        nn.quantize(
            self.model,
            class_predicate=lambda name, module: (
                dict(modules[name]) if name in modules else False
            ),
        )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        self._super_expert_gathers = _super_expert_gathers(self.model)
        self._super_mamba_mixers = _super_mamba_mixers(self.model)
        self._super_moe_mixers = _super_moe_mixers(self.model)
        if self.expert_gather_sort != "auto":
            for switch in self._super_expert_gathers:
                switch.set_sort_policy(self.expert_gather_sort)
            self.layout += ":expert-gather-unsorted-decode-v1"
        if self.mamba_conv != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.set_conv_policy(self.mamba_conv)
            self.layout += (
                ":mamba-conv4-silu-short-checkpoints-v1"
                if self.mamba_conv == "fused_short"
                else (
                    ":mamba-conv4-silu-b1-compact-state-v2"
                    if self.mamba_conv == "fused_b1_compact"
                    else ":mamba-conv4-silu-b1-v1"
                )
            )
        if self.mamba_norm != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.norm.set_policy(self.mamba_norm)
            self.layout += (
                ":mamba-gate-group-rms-short-v1"
                if self.mamba_norm == "fused_short"
                else ":mamba-gate-group-rms-b1-v1"
            )
        if self.mamba_ssm != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.set_ssm_policy(self.mamba_ssm)
            self.layout += (
                ":mamba-ssm-short-dt-gate-v1"
                if self.mamba_ssm == "fused_short_gate"
                else ":mamba-ssm-short-dt-v1"
            )
        if self.mamba_projection != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.set_projection_policy(self.mamba_projection)
            projection_layout = {
                "compact_padded": ":mamba-varlen-q8-projection-v1",
                "compact_row_exact": ":mamba-varlen-row-exact-q8-projection-v1",
                "adaptive_row_exact": ":mamba-varlen-adaptive-row-exact-q8-projection-v1",
                "short_row_exact": ":mamba-short-row-exact-q8-projection-v1",
                "native_qmm_residual_b1": ":mamba-native-qmm-residual-b1-v1",
            }
            self.layout += projection_layout[self.mamba_projection]
        if self.mamba_segment != "auto":
            self.model.set_short_segment_policy(self.mamba_segment)
            self.layout += (
                ":mamba-layerwise-short-exact-routed-batch-v1"
                if self.mamba_segment == "layerwise_short_exact_routed_batch"
                else ":mamba-layerwise-short-exact-v1"
            )
        if self.moe_relu2_down != "auto":
            for mixer in self._super_moe_mixers:
                mixer.set_relu2_down_policy(self.moe_relu2_down)
            self.layout += ":relu2-down-fp32-top22-b1-v1"
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        if tokenizer.convert_tokens_to_ids("<image>") != 18:
            raise ValueError("Super VL tokenizer image token differs")
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[11]
        )
        self.max_context = self.model.args.max_position_embeddings
        if draft_record is not None:
            self._bind_external_drafter(draft_record, draft_loader, DESCRIPTOR)

    def prompt_tokens(self, request):
        if "_mlx2_prompt_tokens" in request:
            return list(request["_mlx2_prompt_tokens"])
        return super().prompt_tokens(request)

    def prepare_multimodal_request(self, request, *, file_loader=None):
        from ..multimodal import media_fingerprint, resolve_media
        from .mlx_vlm import _plain_messages, _source
        from .nemotron35_media import PROCESSOR_REVISION, preprocess_image

        media, replacements, images, counts = [], [], [], []
        for message in request["messages"]:
            content = message.get("content")
            # Check string and text-list content before injecting placeholders.
            texts = (
                [content]
                if isinstance(content, str)
                else [
                    part.get("text", "")
                    for part in content or []
                    if part.get("type") == "text"
                ]
            )
            if any(
                marker in text
                for text in texts
                for marker in ("<image>", "<img>", "</img>")
            ):
                raise ValueError("Super VL text must not contain media placeholders")
            for part in content if isinstance(content, list) else []:
                if part["type"] == "text":
                    continue
                if part["type"] not in {"image_url", "input_image"}:
                    raise ValueError("Super VL serving currently accepts images only")
                if self.draft_model is not None:
                    raise ValueError(
                        "Super VL external draft with media prefill is not implemented"
                    )
                value = resolve_media(
                    _source(part, "image"), kind="image", file_loader=file_loader
                )
                pixels, count = preprocess_image(
                    value.value,
                    mean=self.source_config["norm_mean"],
                    std=self.source_config["norm_std"],
                )
                media.append(value)
                images.append(pixels)
                counts.append(count)
                replacements.append("<img>" + "<image>" * count + "</img>")
        messages = _plain_messages(request["messages"], replacements)
        if not media:
            return {**request, "messages": messages}
        tokens = super().prompt_tokens({**request, "messages": messages})
        positions = [index for index, token in enumerate(tokens) if token == 18]
        if len(positions) != sum(counts):
            raise ValueError("Super VL rendered image tokens/features differ")
        return {
            **request,
            "messages": messages,
            "_mlx2_prompt_tokens": tokens,
            "sampling_profile": request.get("sampling_profile", "vision"),
            "_mlx2_prefill_inputs": {"image_pixels": images},
            "_mlx2_media_token_end": positions[-1] + 1,
            "_mlx2_media_fingerprint": media_fingerprint(
                media,
                policy={
                    "processor": PROCESSOR_REVISION,
                    "artifact": self.identity["fingerprint"],
                },
            ),
        }

    def validate_prefill_inputs(self, request, remaining_tokens, prefill_input):
        expected = sum(token == 18 for token in request["_mlx2_prompt_tokens"])
        observed = sum(token == 18 for token in remaining_tokens)
        if observed == 0:
            return {}
        if observed != expected:
            raise ValueError("Super VL cannot resume in the middle of an image span")
        return prefill_input

    def profile_name(self, mtp):
        return f"nemotron35-super-vl-affine{self.conversion['expert_bits']}-apcv2-" + (
            f"mtp{self._num_draft}" if mtp else "ordinary"
        )

    def _validate_session_width(self, width):
        if type(width) is not int or width not in {1, 2, 4}:
            raise ValueError(
                "Super VL requires one session, or B=2/B=4 for exact progressive external verification"
            )
        if width in {2, 4} and not (
            self.draft_model is not None
            and self.external_policy.get("progressive_verification_tile") is not None
            and self.external_policy.get("progressive_verification_fidelity", "exact")
            == "exact"
        ):
            raise ValueError(
                "Super VL B=2/B=4 requires exact progressive external verification"
            )
        return width

    def execution_config(self, *, max_lanes, prefill_step):
        self._validate_session_width(max_lanes)
        if self.draft_model is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def create_external_batch(self, **kwargs):
        self._validate_session_width(kwargs.get("completion_batch_size", 1))
        return super().create_external_batch(**kwargs)

    def execution_numerics_contract(self):
        return {
            "router_compute": "float32",
            "recurrent_state": "float32",
            "time_step_limit": [0.0, "infinity"],
            "source_revision": SOURCE_REVISION,
            "quantization_recipe": self.conversion["recipe"],
            "qualification": "unqualified",
        }

    def nemotron_lilicorr_target_contract(self, draft_record):
        """Bind an inspected owned-head companion to exact Super target taps."""
        taps = tuple(draft_record["args"].target_layer_ids)
        text = draft_record["target"]["text_config"]
        if draft_record["target_fingerprint"] != self.identity["fingerprint"]:
            raise ValueError("Nemotron LiLiCorr target revision changed")
        return {
            "schema": "mlx2.nemotron-lilicorr-target.v1",
            "target_model_type": "nemotron_h_omni",
            "target_fingerprint": self.identity["fingerprint"],
            "target_layer_ids": taps,
            "target_feature_width": len(taps) * int(text["hidden_size"]),
            "capture": "post-layer-pre-final-norm",
            "ordinary_reference_preserved": True,
            "speculative_rollback": "exact",
            "owned_lm_head": True,
            "owned_final_norm": True,
            "qualified": False,
        }

    def diagnostics(self):
        gather_counts = {}
        for switch in getattr(self, "_super_expert_gathers", ()):
            for name, count in switch.sort_status()["counts"].items():
                gather_counts[name] = gather_counts.get(name, 0) + int(count)
        conv_counts = {}
        for mixer in getattr(self, "_super_mamba_mixers", ()):
            for name, count in mixer.conv_status()["counts"].items():
                conv_counts[name] = conv_counts.get(name, 0) + int(count)
        norm_counts = {}
        for mixer in getattr(self, "_super_mamba_mixers", ()):
            for name, count in mixer.norm.status()["counts"].items():
                norm_counts[name] = norm_counts.get(name, 0) + int(count)
        ssm_counts = {}
        ssm_refusal_reasons = {}
        for mixer in getattr(self, "_super_mamba_mixers", ()):
            status = mixer.ssm_status()
            for name, count in status["counts"].items():
                ssm_counts[name] = ssm_counts.get(name, 0) + int(count)
            for name, count in status["refusal_reasons"].items():
                ssm_refusal_reasons[name] = ssm_refusal_reasons.get(name, 0) + int(
                    count
                )
        projection_counts = {}
        for mixer in getattr(self, "_super_mamba_mixers", ()):
            for name, count in mixer.projection_status()["counts"].items():
                projection_counts[name] = projection_counts.get(name, 0) + int(count)
        relu2_down_counts = {}
        for mixer in getattr(self, "_super_moe_mixers", ()):
            for name, count in mixer.relu2_down_status()["counts"].items():
                relu2_down_counts[name] = relu2_down_counts.get(name, 0) + int(count)
        return {
            "architecture": "nemotron-h-mamba-moe-radio",
            "layout": self.layout,
            "mtp_head_present": True,
            "qualification": "unqualified",
            "conversion": {
                key: self.conversion[key]
                for key in ("recipe", "expert_bits", "activation_aware")
            },
            "external_draft": self.draft_model is not None,
            "moe_nax_gather": _moe_nax_gather_report(
                getattr(self, "moe_nax_gather", "off")
            ),
            "row_exact": (
                self.model.row_exact_status()
                if hasattr(self.model, "row_exact_status")
                else None
            ),
            "expert_gather_sort": {
                "schema": "mlx2.nemotron35-super-expert-gather-sort.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.expert_gather_sort != "auto",
                "observed_used": gather_counts.get("unsorted_decode", 0) > 0,
                "policy": self.expert_gather_sort,
                "modules": len(getattr(self, "_super_expert_gathers", ())),
                "counts": gather_counts,
            },
            "mamba_conv": {
                "schema": "mlx2.nemotron35-super-mamba-conv.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_conv != "auto",
                "observed_used": any(
                    conv_counts.get(name, 0) > 0
                    for name in ("fused_b1", "fused_b1_compact", "fused_short")
                ),
                "policy": self.mamba_conv,
                "modules": len(getattr(self, "_super_mamba_mixers", ())),
                "counts": conv_counts,
            },
            "mamba_norm": {
                "schema": "mlx2.nemotron35-super-mamba-gate-group-rms.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_norm != "auto",
                "observed_used": any(
                    norm_counts.get(name, 0) > 0 for name in ("fused_b1", "fused_short")
                ),
                "policy": self.mamba_norm,
                "modules": len(getattr(self, "_super_mamba_mixers", ())),
                "counts": norm_counts,
            },
            "mamba_ssm": {
                "schema": "mlx2.nemotron35-super-mamba-ssm-short-dt.v2",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_ssm != "auto",
                "observed_used": any(
                    ssm_counts.get(name, 0) > 0
                    for name in ("fused_short", "fused_short_gate")
                ),
                "policy": self.mamba_ssm,
                "modules": len(getattr(self, "_super_mamba_mixers", ())),
                "counts": ssm_counts,
                "refusal_reasons": ssm_refusal_reasons,
            },
            "mamba_projection": {
                "schema": "mlx2.nemotron35-super-mamba-varlen-projection.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_projection != "auto",
                "observed_used": any(
                    projection_counts.get(name, 0) > 0
                    for name in (
                        "in_proj_compaction_calls",
                        "out_proj_compaction_calls",
                        "in_proj_short_kernel",
                        "out_proj_short_kernel",
                    )
                ),
                "policy": self.mamba_projection,
                "modules": len(getattr(self, "_super_mamba_mixers", ())),
                "counts": projection_counts,
            },
            "mamba_segment": {
                "schema": "mlx2.nemotron35-super-mamba-segment.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_segment != "auto",
                "observed_used": any(
                    self.model.short_segment_status()["counts"].get(name, 0) > 0
                    for name in (
                        "layerwise_short_exact",
                        "layerwise_short_exact_routed_batch",
                    )
                ),
                **self.model.short_segment_status(),
            },
            "moe_relu2_down": {
                "schema": "mlx2.nemotron35-super-relu2-down-fp32-top22.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.moe_relu2_down != "auto",
                "observed_used": relu2_down_counts.get("fused_b1", 0) > 0,
                "policy": self.moe_relu2_down,
                "modules": len(getattr(self, "_super_moe_mixers", ())),
                "counts": relu2_down_counts,
            },
            "execution_limits": dict(EXECUTION_LIMITS),
            "image_prefill": "b1-candidate",
            "video": "encoder-only",
        }
