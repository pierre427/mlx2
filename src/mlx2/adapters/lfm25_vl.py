"""LFM2.5-VL candidate with separate, unqualified DSpark draft artifact.

The tensor and processor implementations remain in the pinned local mlx-vlm
source. This module supplies mlx2 artifact gates and image/frame preparation.
The serving cache uses mlx2's checkpoint-aware KV and convolution state.
See provenance/lfm25-vl.json.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import secrets
import struct
from collections.abc import Mapping
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..multimodal import media_fingerprint, resolve_media
from ..output import OutputParser, callable_tools
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .mlx_vlm import MediaFeatureCache, _LogitsModel, _ids_and_kwargs, _plain_messages, _source

# Not the project pin (mlx_vlm_pin.MLX_VLM_REVISION, which pyproject installs
# and which does not descend from this revision): this adapter loads only when
# a clean checkout of SOURCE_REVISION is first on PYTHONPATH, and fails closed
# naming both revisions otherwise (sweep 2026-10-06 G2-04/G5-05).
SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
SOURCE_PATHS = ("mlx_vlm/models/lfm2_vl", "mlx_vlm/models/lfm2/language.py",
                "mlx_vlm/models/lfm2/speculative_verifier.py",
                "mlx_vlm/models/base.py", "mlx_vlm/models/cache.py",
                "mlx_vlm/speculative/drafters/dspark",
                "mlx_vlm/speculative/drafters/__init__.py",
                "mlx_vlm/speculative/dflash.py", "mlx_vlm/speculative/utils.py",
                "mlx_vlm/generate/ar.py", "mlx_vlm/generate/dispatch.py",
                "mlx_vlm/prompt_utils.py")
CACHE_LAYOUT = "lfm25-vl-hybrid-image-apcv2-v2"
VIDEO_SAMPLE_FPS = 1
VIDEO_MAX_FRAMES = 16
LFM25_VL = ModelDescriptor(
    model_type="lfm2_vl", family="lfm2.5-vl", variant="ordinary-image-frame-candidate",
    state_planes=frozenset({
        StatePlane.ATTENTION_KV, StatePlane.RECURRENT,
        StatePlane.RNG, StatePlane.TRANSCRIPT,
    }),
    capabilities=frozenset({
        Capability.TEXT, Capability.VISION, Capability.VIDEO, Capability.STREAMING,
        Capability.PREFIX_REUSE, Capability.APC_V2,
    }),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.lfm25_vl.LFM25VLAdapter",
        "qualification": "pending", "draft": "DSpark auxiliary unqualified",
        "source_revision": SOURCE_REVISION,
        "required_qualification_checks": (
            "text_ordinary", "image_placeholder_alignment", "image_apcv2_reuse",
            "video_frame_order_alignment", "video_apcv2_reuse",
            "hybrid_cache_replay", "batching_if_requested",
        ),
    },
)


def _image_spans(ids, image_id, start_id, end_id) -> list[tuple[int, int]] | None:
    """Return each ``<|image_start|>``..``<|image_end|>`` span in ``ids``.

    The pinned processor expands every image or video frame into one span of
    64+ ``<image>`` tokens plus tile and thumbnail markers, so placeholders
    are counted as spans.  A nested, unclosed or empty span, or an
    ``<image>`` outside a span, returns None.
    """
    spans, start, filled = [], None, False
    for index, token in enumerate(ids):
        if token == start_id:
            if start is not None:
                return None
            start, filled = index, False
        elif token == end_id:
            if start is None or not filled:
                return None
            spans.append((start, index))
            start = None
        elif token == image_id:
            if start is None:
                return None
            filled = True
    return None if start is not None else spans


def _object(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            result[key] = value
        return result
    result = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return result


def _header(path: Path) -> tuple[dict, str]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("truncated safetensors file")
        length = struct.unpack("<Q", raw_length)[0]
        if not 0 < length <= min(64 << 20, size - 8):
            raise ValueError("invalid safetensors header length")
        raw = stream.read(length)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate safetensors tensor {key!r}")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("invalid safetensors header")
    for name, entry in value.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
            raise ValueError(f"invalid safetensors tensor {name!r}")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(n) is int for n in offsets) or not 0 <= offsets[0] <= offsets[1] <= size - 8 - length:
            raise ValueError(f"invalid tensor offsets for {name!r}")
    return value, hashlib.sha256(raw).hexdigest()


def _identity(path: Path, config: dict) -> tuple[dict, dict]:
    weight = path / "model.safetensors"
    if not weight.is_file():
        raise ValueError("missing LFM safetensors weights")
    header, header_hash = _header(weight)
    digest = hashlib.sha256()
    for name in (
        "config.json", "processor_config.json", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
    ):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    stat = weight.stat()
    digest.update(header_hash.encode())
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return {
        "path": str(path), "fingerprint": digest.hexdigest(),
        "files": [(weight.name, stat.st_size, stat.st_mtime_ns)],
        "header_sha256": header_hash, "tensor_count": len(header),
        "config": config, "qualification": "pending",
    }, header


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = _object(path / "config.json")
    text, vision = config.get("text_config"), config.get("vision_config")
    if config.get("model_type") != "lfm2_vl" or config.get("architectures") != ["Lfm2VlForConditionalGeneration"]:
        raise ValueError("expected LFM2.5-VL target")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ValueError("missing LFM text or vision configuration")
    if any(text.get(k) != v for k, v in {
        "num_hidden_layers": 30, "hidden_size": 2048,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "vocab_size": 128000, "conv_L_cache": 3,
    }.items()):
        raise ValueError("unsupported LFM text topology")
    layers = text.get("layer_types")
    if not isinstance(layers, list) or len(layers) != 30 or layers.count("full_attention") != 8 or any(x not in {"conv", "full_attention"} for x in layers):
        raise ValueError("unsupported LFM hybrid layer layout")
    if any(vision.get(k) != v for k, v in {
        "num_hidden_layers": 27, "hidden_size": 1152,
        "num_attention_heads": 16, "patch_size": 16,
    }.items()):
        raise ValueError("unsupported LFM vision topology")
    if config.get("image_token_id") != 124907 or config.get("downsample_factor") != 2 or config.get("projector_hidden_size") != 2048:
        raise ValueError("unsupported LFM image projection layout")
    identity, header = _identity(path, config)
    required = {
        "model.language_model.embed_tokens.weight": [128000, 2048],
        "model.multi_modal_projector.linear_1.weight": [2048, 4608],
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight": [1152, 768],
    }
    for i, kind in enumerate(layers):
        required[f"model.language_model.layers.{i}.{'self_attn.q_proj.weight' if kind == 'full_attention' else 'conv.conv.weight'}"] = None
    for name, shape in required.items():
        record = header.get(name)
        if record is None or (shape is not None and record.get("shape") != shape):
            raise ValueError(f"missing or mismatched LFM tensor: {name}")
    return {**identity, "max_context": int(text["max_position_embeddings"]), "image_token_id": 124907, "supports_self_mtp": False}


def inspect_dspark_artifact(model_path: str | Path, *, target: dict | None = None) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = _object(path / "config.json")
    dflash = config.get("dflash_config")
    if config.get("architectures") != ["Lfm2DSparkDraftModel"] or config.get("model_type") != "qwen3" or not isinstance(dflash, dict):
        raise ValueError("expected LFM2.5 DSpark auxiliary artifact")
    if any(config.get(k) != v for k, v in {
        "hidden_size": 2048, "num_hidden_layers": 4,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 64, "vocab_size": 128000,
        "block_size": 9, "markov_rank": 256,
        "enable_confidence_head": True,
    }.items()):
        raise ValueError("unsupported DSpark topology")
    if dflash.get("target_layer_ids") != [2, 9, 17, 21, 27] or dflash.get("num_target_layers") != 30 or dflash.get("mask_token_id") != 125017:
        raise ValueError("unsupported DSpark target binding")
    identity, header = _identity(path, config)
    required = {
        "fc.weight": [2048, 10240],
        "markov_head.markov_w1.weight": [128000, 256],
        "markov_head.markov_w2.weight": [128000, 256],
        "confidence_head.proj.weight": [1, 2304],
    }
    for i in range(4):
        required[f"layers.{i}.self_attn.q_proj.weight"] = None
    for name, shape in required.items():
        record = header.get(name)
        if record is None or (shape is not None and record.get("shape") != shape):
            raise ValueError(f"missing or mismatched DSpark tensor: {name}")
    if target is not None:
        text = target.get("config", {}).get("text_config", {})
        if text.get("hidden_size") != 2048 or text.get("vocab_size") != 128000 or text.get("num_hidden_layers") != 30:
            raise ValueError("DSpark target model is incompatible")
        identity["target_fingerprint"] = target["fingerprint"]
    return {**identity, "auxiliary": True, "selected": False, "qualified": False, "proposal_length": 9, "verification_width": 10}


def _require_source_revision():
    from ._direct_mlx_vlm import load_backend
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or spec.origin is None:
        raise RuntimeError("LFM2.5-VL requires a pinned local mlx-vlm checkout")
    source_root = Path(spec.origin).resolve().parents[1]
    load_backend(source_root, SOURCE_REVISION, SOURCE_PATHS)
    return {"revision": SOURCE_REVISION, "source": str(source_root)}


def load_candidate_dspark(model_path: str | Path, *, target_artifact: dict, target_model):
    """Load a bound draft for offline evaluation; serving never selects it.

    DSpark's confidence calibration and target verification need independent
    evidence before any route can declare a speculative capability.
    """
    record = inspect_dspark_artifact(model_path, target=target_artifact)
    _require_source_revision()
    from .lfm25_dspark_compat import (
        install_offline_dspark_bridge, offline_dspark_target_transaction,
        require_offline_dspark_target,
    )
    from mlx_vlm.models.base import LanguageModelOutput
    from mlx_vlm.models.cache import ArraysCache, KVCache
    from mlx_vlm.models.lfm2.language import (
        LanguageModel as TextLanguageModel, _EXACT_SPECULATIVE_VERIFIER,
    )
    from mlx_vlm.models.lfm2_vl.language import LanguageModel as VLLanguageModel

    with offline_dspark_target_transaction(target_model):
        install_offline_dspark_bridge(
            target_model, vl_language_class=VLLanguageModel,
            text_language_class=TextLanguageModel,
            verifier=_EXACT_SPECULATIVE_VERIFIER,
            cache_classes=(ArraysCache, KVCache), output_class=LanguageModelOutput,
        )
        require_offline_dspark_target(target_model, record["config"])
        import mlx.core as mx
        from mlx_vlm.speculative.drafters.dspark import DSparkDraftModel, ModelConfig

        config = ModelConfig.from_dict(record["config"])
        model = DSparkDraftModel(config)
        weights = model.sanitize(mx.load(str(Path(record["path"]) / "model.safetensors")))
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        mx.eval(model.parameters())
        model.bind(target_model)
    return model, record


def generate_candidate_dspark(target_path: str | Path, draft_path: str | Path,
                              prompt: str, *, image: str | Path | None = None,
                              max_tokens: int = 128,
                              verification_width: int = 10,
                              collect_speculative_stats: bool = False) -> dict:
    """Run an explicit offline DSpark candidate with target verification.

    This is separate from mlx2's serving route; acceptance and speed need
    source-bound evaluation before speculative selection can be considered.

    ``verification_width`` is the anchor plus at most width - 1 proposals per
    round; the pinned checkpoint was trained for nine, so 2..10 is accepted.
    Decoding is greedy (temperature 0). ``evaluation_contract`` records only
    the REQUESTED configuration: it is not the observed effective width,
    acceptance or engagement, confidence-head use, state parity or speed, and
    the final round may verify fewer tokens when max_tokens runs out.

    ``collect_speculative_stats=True`` adds ``speculative_stats``: the change
    in the pinned source's lifetime drafter counters (rounds, accepted and
    drafted proposals) across this request, read by host getattr from the
    freshly loaded private drafter before and after generation.  The source
    records a round before EOS or output-budget truncation, so counts are
    target-verified draft proposals and may exceed emitted tokens.  Malformed,
    partial or inconsistent counters make the diagnostic unavailable with
    None counts.  ``engaged`` needs at least one round and one proposal.
    Totals do not show effective-width distribution, confidence-head use,
    per-round accept/reject, rollback or state correctness, and never affect
    decoding or selection.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("DSpark candidate prompt must be nonempty")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
        raise ValueError("DSpark candidate max_tokens must be 1..512")
    if type(verification_width) is not int or not 2 <= verification_width <= 10:
        raise ValueError("DSpark candidate verification_width must be 2..10")
    if type(collect_speculative_stats) is not bool:
        raise ValueError("DSpark candidate collect_speculative_stats must be a bool")
    if image is not None and not Path(image).expanduser().is_file():
        raise ValueError("DSpark candidate image file is missing")
    target = inspect_artifact(target_path)
    inspect_dspark_artifact(draft_path, target=target)
    _require_source_revision()
    from mlx_vlm import load
    from mlx_vlm.generate import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(target["path"], lazy=False, strict=True,
                            trust_remote_code=False)
    draft, record = load_candidate_dspark(draft_path, target_artifact=target,
                                          target_model=model)
    formatted = apply_chat_template(processor, model.config, prompt,
                                    num_images=int(image is not None))

    def read_counters(phase: str) -> tuple[tuple[int, int, int] | None, str]:
        # Host reads only; the source recorder starts absent counters at zero.
        absent = object()
        try:
            raw = [getattr(draft, name, absent) for name in (
                "speculative_total_rounds", "speculative_total_accepted",
                "speculative_total_drafted")]
        except Exception:
            return None, f"{phase}_counter_read_error"
        if all(value is absent for value in raw) and phase == "baseline":
            return (0, 0, 0), "uninitialized"
        if any(value is absent for value in raw):
            return None, f"{phase}_counters_missing"
        rounds, accepted, drafted = raw
        if type(rounds) is not int or type(drafted) is not int or min(rounds, drafted) < 0:
            return None, f"{phase}_counter_invalid"
        if type(accepted) is float:
            # is_integer() is False for NaN/Inf; 2**53 bounds exact floats.
            if not (accepted.is_integer() and 0 <= accepted < 2 ** 53):
                return None, f"{phase}_counter_invalid"
            accepted = int(accepted)
        elif type(accepted) is not int or accepted < 0:
            return None, f"{phase}_counter_invalid"
        if accepted > drafted:
            return None, f"{phase}_accepted_exceeds_drafted"
        return (rounds, accepted, drafted), "lifetime"

    if collect_speculative_stats:
        before, baseline_kind = read_counters("baseline")
    result = generate(model, processor, formatted,
                      image=str(image) if image is not None else None,
                      max_tokens=max_tokens, verbose=False, draft_model=draft,
                      draft_kind="dflash", draft_block_size=verification_width,
                      temperature=0.0)
    if collect_speculative_stats:
        delta = None
        reason = None if before is not None else baseline_kind
        if before is not None:
            after, after_kind = read_counters("post")
            if after is None:
                reason = after_kind
            else:
                rounds, accepted, drafted = (a - b for a, b in zip(after, before))
                if min(rounds, accepted, drafted) < 0:
                    reason = "counter_regression"
                elif rounds == 0 and (accepted or drafted):
                    reason = "zero_round_nonzero_delta"
                elif accepted > drafted:
                    reason = "delta_accepted_exceeds_drafted"
                elif drafted > rounds * (verification_width - 1):
                    reason = "requested_ceiling_exceeded"
                else:
                    delta = (rounds, accepted, drafted)
    payload = {"text": result.text, "finish_reason": result.finish_reason,
               "target_fingerprint": target["fingerprint"],
               "draft_fingerprint": record["fingerprint"],
               "evaluation_contract": {
                   "scope": "requested", "source_revision": SOURCE_REVISION,
                   "requested_verification_width": verification_width,
                   "maximum_proposals_per_round": verification_width - 1,
                   "temperature": 0.0, "max_tokens": max_tokens,
               },
               "qualified": False, "selected": False}
    if collect_speculative_stats:
        payload["speculative_stats"] = {
            "scope": "request", "attribution": "b1-private-drafter",
            "source_revision": SOURCE_REVISION,
            "unit": "target-verified draft proposals, not emitted tokens",
            "available": delta is not None,
            "engaged": delta is not None and delta[0] > 0 and delta[2] > 0,
            "unavailable_reason": reason,
            "baseline_kind": baseline_kind if delta is not None else None,
            "rounds": None if delta is None else delta[0],
            "accepted_proposals": None if delta is None else delta[1],
            "drafted_proposals": None if delta is None else delta[2],
            "qualified": False, "selected": False,
        }
    return payload


class _LFMLogitsModel(_LogitsModel):
    apc_v2_layout = CACHE_LAYOUT

    def make_cache(self):
        # The pinned mlx-vlm ArraysCache retains only the latest convolution
        # state and cannot restore an earlier prefix.  The source ShortConv
        # reads/writes cache[0] and calls advance(), which mlx2's cache also
        # implements.  Its state_checkpoint/trim_to_position and COW copy
        # semantics make exact APCv2 branches possible.
        from ..runtime.models.cache import ArraysCache, KVCache

        layers = self._model.language_model.layers
        if len(layers) != 30:
            raise ValueError("LFM cache layout no longer matches the pinned model")
        expected = self._model.config.text_config.layer_types
        if any(bool(layer.is_attention_layer) != (kind == "full_attention")
               for layer, kind in zip(layers, expected, strict=True)):
            raise ValueError("LFM loaded layer topology disagrees with its config")
        return [KVCache() if layer.is_attention_layer else ArraysCache(size=1)
                for layer in layers]

    def __call__(self, inputs, cache=None, **kwargs):
        output = self._model(
            inputs, kwargs.pop("pixel_values", None), None,
            cache=cache, **kwargs,
        )
        return getattr(output, "logits", output)


# LiquidAI/LFM2.5-VL-3B generation_config.json and model card (text):
# temperature 0.2, top_k 50, repetition_penalty 1.0.
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=0.2, top_k=50, repetition_penalty=1.0,
                     source=GENERATION_CONFIG,
                     note="do_sample=true; also the model card's text profile"),
    model="LiquidAI/LFM2.5-VL-3B",
)


class LFM25VLAdapter:
    default_route = "ordinary"
    descriptor = LFM25_VL
    sampling_defaults = SAMPLING

    def prefill_step_default(self):
        """Decline a family override so the generic prompt schedule owns it."""
        return None

    def __init__(self, model_path: str, *, execution_policy=None):
        # Refuse an explicitly enabled TF32 switch like the standard decoder
        # (process_env.require_process_numerics); MLX latches it.
        from ..process_env import require_process_numerics

        require_process_numerics("the LFM2.5-VL adapter")
        self.media_checkpoint_enabled = execution_policy == {
            "lfm_media_checkpoint": "candidate_v1"
        }
        if execution_policy not in (None, {}) and not self.media_checkpoint_enabled:
            raise ValueError("LFM2.5-VL has no qualified execution policy")
        self._media_proof_key = secrets.token_bytes(32)
        self.identity = inspect_artifact(model_path)
        self.environment = {"mlx_vlm_revision": SOURCE_REVISION}
        self.mlx_vlm_runtime = _require_source_revision()
        from mlx_vlm import load
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        self.model, self.processor = load(str(self.identity["path"]), lazy=False, strict=True, trust_remote_code=False)
        from ..runtime.chat_templates import secure_model_chat_templates
        secure_model_chat_templates(self.processor)
        from .lfm25_fused_shortconv import (
            ShortConvCounters, enabled as fused_shortconv_enabled,
            install as install_shortconv,
        )

        self.fused_shortconv_counters = ShortConvCounters()
        self.fused_shortconv_opt_in_requested = fused_shortconv_enabled()
        install_shortconv(self.model, self.fused_shortconv_counters)
        self.model = _LFMLogitsModel(self.model)
        self.media_feature_cache = MediaFeatureCache()
        self.layout = CACHE_LAYOUT
        self.max_context = self.identity["max_context"]
        self.tokenizer = TokenizerWrapper(
            self.processor.tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[int(self.identity["config"]["eos_token_id"])],
        )

    def execution_config(self, *, max_lanes, prefill_step):
        if max_lanes != 1:
            raise ValueError("LFM2.5-VL candidate requires max_lanes=1 until mixed-length batch state is qualified")
        counters = getattr(self, "fused_shortconv_counters", None)
        shortconv_mode = (
            "fused-metal-candidate" if counters is not None and counters.installed
            else "source"
        )
        return {
            "persistent": True, "num_draft": 0, "backend": "ordinary",
            "rate_gate": False, "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False, "segment_aware_cohort_size": 1,
            "lfm_shortconv": shortconv_mode,
            "lfm_media_checkpoint": (
                "candidate_v1"
                if getattr(self, "media_checkpoint_enabled", False)
                else "off"
            ),
        }

    def exact_prefix_cascade_contract(self):
        """Declare the exact, language-only continuation-cascade boundary.

        LFM's eight attention and 22 ShortConv planes can reuse an exact
        request-private ordinary checkpoint.  Vision/frame prefill is outside
        the cascade and must already be bound below that checkpoint.  DSpark
        has a separate state machine and is never admitted by this contract.
        """
        return {
            "execution_domain": "autoregressive_language_only",
            "verification_order": "longest_first",
            "invalid_sibling_pruning": True,
            "accepted_prefix_state": "exact_ordinary_hybrid_checkpoint",
            "shared_prefix_reuse": "suffix_only_after_exact_geometry_gate",
            "cache_layout": CACHE_LAYOUT,
            "attention_layers": 8,
            "recurrent_layers": 22,
            "max_lanes": 1,
            "media_prefill": "bound_below_language_checkpoint",
            "dspark_state": "excluded",
            "request_private_only": True,
            "apcv2_publication": False,
            "ordinary_reference_preserved": True,
            "implemented": True,
            "implementation_scope": "planner_and_adapter_state_gate",
            "qualified": False,
            "selected": False,
            "observed_used": False,
        }

    def _require_exact_prefix_state_binding(self, binding, *, prefix_tokens):
        if not isinstance(binding, Mapping):
            raise ValueError("LFM exact-prefix reuse requires a state binding")
        expected_fingerprint = getattr(self, "identity", {}).get("fingerprint")
        required_planes = frozenset(plane.value for plane in self.descriptor.state_planes)
        raw_planes = binding.get("state_planes")
        if not isinstance(raw_planes, (tuple, list, set, frozenset)):
            raise ValueError("LFM exact-prefix state planes are missing")
        planes = frozenset(raw_planes)
        start = binding.get("prefix_start_position")
        checkpoint = binding.get("checkpoint_position")
        media_end = binding.get("media_token_end", 0)
        media_fingerprint = binding.get("media_fingerprint")
        media_tokens = binding.get("media_prompt_tokens")
        media_proof = binding.get("media_proof")
        failures = []
        if binding.get("execution_domain") != "autoregressive_language_only":
            failures.append("execution domain")
        if binding.get("cache_layout") != CACHE_LAYOUT:
            failures.append("cache layout")
        if expected_fingerprint is None or binding.get("artifact_fingerprint") != expected_fingerprint:
            failures.append("artifact fingerprint")
        if binding.get("source_revision") != SOURCE_REVISION:
            failures.append("source revision")
        if binding.get("checkpoint_kind") != "ordinary_exact":
            failures.append("checkpoint kind")
        if type(start) is not int or start < 0:
            failures.append("prefix start")
        if type(checkpoint) is not int or type(start) is not int or checkpoint != start + prefix_tokens:
            failures.append("checkpoint position")
        if planes != required_planes:
            failures.append("state planes")
        if binding.get("attention_layers") != 8 or binding.get("recurrent_layers") != 22:
            failures.append("hybrid layer geometry")
        if binding.get("batch_size") != 1:
            failures.append("batch size")
        if binding.get("dspark_state") != "absent":
            failures.append("DSpark state")
        if type(media_end) is not int or media_end < 0 or type(start) is not int or media_end > start:
            failures.append("media boundary")
        if media_end:
            if not isinstance(media_fingerprint, str) or not media_fingerprint:
                failures.append("media fingerprint")
            if (
                not isinstance(media_tokens, list)
                or not media_tokens
                or any(type(token) is not int or token < 0 for token in media_tokens)
                or media_end > len(media_tokens) - 1
                or not isinstance(media_proof, str)
                or not hmac.compare_digest(
                    media_proof,
                    self._media_checkpoint_proof(
                        media_tokens, media_fingerprint, media_end
                    ),
                )
            ):
                failures.append("media proof")
        elif any(
            value is not None
            for value in (media_fingerprint, media_tokens, media_proof)
        ):
            failures.append("unexpected media binding")
        if failures:
            raise ValueError(
                "LFM exact-prefix state binding differs: " + ", ".join(failures)
            )

    def plan_exact_prefix_cascade(
        self, paths, accepted_prefix=(), *, attempted=(), state_binding=None
    ):
        """Plan a suffix-only AR stage after validating reusable hybrid state."""
        from ..runtime.exact_prefix_cascade import next_cascade_stage

        stage = next_cascade_stage(paths, accepted_prefix, attempted=attempted)
        if stage is not None and stage.accepted_prefix:
            self._require_exact_prefix_state_binding(
                state_binding, prefix_tokens=len(stage.accepted_prefix)
            )
        return stage

    def _media_checkpoint_proof(self, tokens, fingerprint, media_end):
        payload = json.dumps(
            [tokens, fingerprint, media_end], separators=(",", ":")
        ).encode()
        return hmac.new(self._media_proof_key, payload, hashlib.sha256).hexdigest()

    def apc_media_checkpoint_position(self, request, tokens, *, cached_tokens):
        """Declare one exact post-media prefix for the generic APCv2 planner.

        The prepared request binds this position to the token stream and
        resolved media.  A warm request already beyond the media span needs
        no extra checkpoint.  The scheduler owns chunking, budget, and publish.
        """
        if not self.media_checkpoint_enabled:
            return None
        media_end = request.get("_mlx2_media_token_end")
        if media_end is None:
            return None
        prepared = request.get("_mlx2_prompt_tokens")
        fingerprint = request.get("_mlx2_media_fingerprint")
        proof = request.get("_mlx2_media_proof")
        inputs = request.get("_mlx2_prefill_inputs")
        if (
            type(media_end) is not int or not 0 < media_end <= len(tokens) - 1
            or type(cached_tokens) is not int or cached_tokens < 0
            or not isinstance(prepared, list) or prepared != list(tokens)
            or not isinstance(fingerprint, str) or not fingerprint
            or not isinstance(proof, str)
            or not isinstance(inputs, dict) or inputs.get("pixel_values") is None
            or not hmac.compare_digest(
                proof, self._media_checkpoint_proof(prepared, fingerprint, media_end)
            )
        ):
            raise ValueError(
                "LFM post-media APCv2 checkpoint is not bound to prepared media"
            )
        return media_end if cached_tokens < media_end < len(tokens) - 1 else None

    def cache_budget(self, *, mtp):
        from .lfm25_memory import LFM25CacheBudget

        return LFM25CacheBudget.from_config(self.identity["config"]["text_config"], mtp=mtp)

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("LFM2.5-VL has no qualified speculative route")
        return "lfm25-vl-ordinary-candidate"

    def prompt_tokens(self, request):
        if "_mlx2_prompt_tokens" in request:
            return list(request["_mlx2_prompt_tokens"])
        return self.tokenizer.encode(self.render_prompt(request), add_special_tokens=False)

    def render_prompt(self, request):
        if "_mlx2_prompt_tokens" in request:
            raise ValueError("prepared image prompts have no plain text rendering")
        if "messages" in request:
            return self._apply_chat_template(request["messages"])
        return request["prompt"]

    def _apply_chat_template(self, messages):
        # The pinned LFM processor has no processor-level chat template; its
        # tokenizer owns the checkpoint's chat_template.jinja.
        owner = self.processor if getattr(self.processor, "chat_template", None) else self.processor.tokenizer
        return owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def output_parser(self, request):
        if callable_tools(request):
            raise ValueError("LFM2.5-VL tool calling is unqualified")
        return OutputParser(chat="messages" in request, stops=request.get("stop", ()))

    def prepare_multimodal_request(self, request, *, file_loader=None):
        import mlx.core as mx
        import numpy as np
        media, replacements, images = [], [], []
        for message in request["messages"]:
            parts = message.get("content")
            for part in parts if isinstance(parts, list) else ():
                kind = part.get("type")
                if kind == "text":
                    continue
                if kind in {"image_url", "input_image"}:
                    value = resolve_media(_source(part, "image"), kind="image", file_loader=file_loader)
                    images.append(value.value)
                    replacements.append(self.processor.image_token)
                elif kind == "input_video":
                    value = resolve_media(
                        _source(part, "video"), kind="video", file_loader=file_loader,
                        fps=VIDEO_SAMPLE_FPS, max_frames=VIDEO_MAX_FRAMES,
                    )
                    frames = list(value.value)
                    if not frames:
                        raise ValueError("LFM2.5-VL video has no sampled frames")
                    shapes = {tuple(np.asarray(frame).shape) for frame in frames}
                    if len(shapes) != 1 or len(next(iter(shapes))) != 3:
                        raise ValueError("LFM2.5-VL video frames must have one HWC shape")
                    times = tuple(value.metadata.get("timestamps_seconds", ()))
                    if len(times) != len(frames):
                        raise ValueError("LFM2.5-VL video timestamps do not match frames")
                    images.extend(frames)
                    replacements.append("\n".join(
                        f"[video frame {i + 1}/{len(frames)} at {float(t):.3f}s] {self.processor.image_token}"
                        for i, t in enumerate(times)
                    ))
                else:
                    raise ValueError(f"LFM2.5-VL input part {kind!r} is unsupported")
                media.append(value)
        if not media:
            return request
        messages = _plain_messages(request["messages"], replacements)
        prompt = self._apply_chat_template(messages)
        processed = self.processor(text=prompt, images=images, return_tensors="np")
        ids, kwargs = _ids_and_kwargs(processed)
        image_id = self.identity["image_token_id"]
        convert = self.processor.tokenizer.convert_tokens_to_ids
        spans = _image_spans(
            ids, image_id, convert(self.processor.image_start_token),
            convert(self.processor.image_end_token),
        )
        if not spans or len(spans) != len(images):
            raise ValueError("LFM processor image placeholders do not match resolved media")
        positions = [i for i, token in enumerate(ids) if token == image_id]
        if positions[-1] >= len(ids) - 1:
            # PromptBatch.generate holds the final prompt token as its decode
            # anchor.  It cannot pass image features to that separate forward.
            raise ValueError("LFM image placeholder must precede the final prompt token")
        for name in ("image_rows", "image_cols", "image_sizes"):
            kwargs.pop(name, None)
        kwargs = {k: mx.array(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        fingerprint = media_fingerprint(media, policy={
            "family": "lfm25-vl", "processor_revision": SOURCE_REVISION,
            "video_fps": VIDEO_SAMPLE_FPS, "video_max_frames": VIDEO_MAX_FRAMES,
        })
        media_end = positions[-1] + 1
        return {
            **request, "messages": messages,
            "_mlx2_prompt_tokens": ids, "_mlx2_prefill_inputs": kwargs,
            "_mlx2_media_token_end": media_end,
            "_mlx2_media_fingerprint": fingerprint,
            "_mlx2_media_proof": self._media_checkpoint_proof(ids, fingerprint, media_end),
        }

    def diagnostics(self):
        counters = self.fused_shortconv_counters
        return {
            "architecture": "lfm2_vl", "layout": CACHE_LAYOUT,
            "mlx_vlm": self.mlx_vlm_runtime, "qualification": "pending",
            "dspark_selected": False,
            "post_media_checkpoint": (
                "candidate_v1" if getattr(self, "media_checkpoint_enabled", False) else "off"
            ),
            "shortconv_fused_candidate": {
                "qualification": "pending",
                "opt_in_requested": self.fused_shortconv_opt_in_requested,
                "installed_layers": counters.installed,
                "engaged": counters.engaged,
                "refused": counters.refused,
                "build_failed": counters.build_failed,
            },
        }

    def close(self):
        self.media_feature_cache.clear()
        self.model = None
        self.processor = None
        self.tokenizer = None
