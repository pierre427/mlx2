"""Inkling 2 Mini (``inkling2_loop``) native text adapter.

Original mlx2 adapter; model math in ``runtime/models/inkling2`` (mined from
Blaizzy/mlx-vlm@111ac668, see provenance/inkling2-mini.json).  Unqualified.

The checkpoint is served as published: BF16 dense paths, FP32 attention
residual queries/routing bias, NVFP4 routed experts (layers 0-19, ModelOpt
``amax / outer_scale_denominator`` per-expert global scale) and MXFP8 routed
experts (layers 20-23).  Activations stay floating point (W4A16 / W8A16); the
vendor runtime's activation quantization is not reproduced, so numerical
parity with the reference runtime is not claimed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter

REPOSITORY = "thinkingmachines/Inkling-2-Mini"
CACHE_LAYOUT = "inkling2-kv-four-shortconv-v1"
NUMERICS = "inkling2-native-w4a16-v1"
# generation_config.json of thinkingmachines/Inkling-2-Mini: do_sample, T 1.0.
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, top_p=1.0, source=GENERATION_CONFIG,
                     note="do_sample=true; no top_p/top_k in generation_config"),
    model=REPOSITORY,
)
DESCRIPTOR = ModelDescriptor(
    model_type="inkling2",
    family="inkling2",
    variant="mini-loop-nvfp4-mxfp8-text",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RECURRENT, StatePlane.RNG, StatePlane.TRANSCRIPT}
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
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.inkling2.Inkling2Adapter",
        "qualification": "pending",
        "scope": "native text; 24 layers / 48 visits; four shortconv states per visit",
        "ordinary_reference": "implemented",
        "dspark": "bundled mtp.safetensors; opt-in external draft (implemented, unqualified)",
        "vision": "weights present; native text route refuses media",
    },
)

# Topology this port implements; anything else refuses before loading.
_EXPECTED = {
    "model_type": "inkling2_loop_text",
    "hidden_size": 2304,
    "num_hidden_layers": 24,
    "vocab_size": 201024,
    "unpadded_vocab_size": 200058,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "swa_num_attention_heads": 24,
    "swa_num_key_value_heads": 8,
    "swa_head_dim": 128,
    "sliding_window_size": 1024,
    "d_rel": 16,
    "rel_extent": 1024,
    "sconv_kernel_size": 4,
    "n_routed_experts": 128,
    "num_experts_per_tok": 8,
    "n_shared_experts": 0,
    "intermediate_size": 1152,
    "use_gate_bias": True,
    "q_bias": False,
    "o_bias": False,
    "use_value_norm_no_gamma": True,
    "attn_res_normalize_values": True,
    "n_layers_per_attn_res_block": 6,
}
_LOOP = {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False}
NVFP4, MXFP8 = "nvfp4_nearest", "mxfp8_nearest"


def _quant_groups(path: Path) -> dict[str, str]:
    """Expert tensor -> weight transform, from hf_quant_config.json."""
    config = json.loads((path / "hf_quant_config.json").read_text())
    if config.get("quant_method") != "inkling2":
        raise ValueError("Inkling 2 hf_quant_config.json quant_method must be inkling2")
    transforms = {}
    for group in config["model"]["parameter_groups"]:
        kind = group.get("weight_transform")
        if kind == "cast":
            continue
        if kind not in (NVFP4, MXFP8):
            raise ValueError(f"Inkling 2 weight transform {kind!r} is not implemented")
        for name in group["tensors"]:
            transforms[name] = kind
    return transforms


def inspect_artifact(model_path) -> dict:
    """Fail-closed topology/tensor check with no MLX import."""
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    if config.get("model_type") != "inkling2_loop":
        raise ValueError("artifact is not Inkling 2 (inkling2_loop)")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Inkling 2 config has no text_config")
    for key, value in _EXPECTED.items():
        if text.get(key) != value:
            raise ValueError(f"Inkling 2 {key}={text.get(key)!r} is not implemented")
    if text.get("loop_config") != _LOOP:
        raise ValueError(f"Inkling 2 loop_config {text.get('loop_config')!r} is not implemented")
    local = text.get("local_layer_ids")
    if not isinstance(local, list) or sorted(local) != [
        i for i in range(24) if (i + 1) % 6
    ]:
        raise ValueError("Inkling 2 local/global layer pattern is not implemented")
    path = Path(artifact["identity"]["path"])
    transforms = _quant_groups(path)
    weights = artifact["weight_map"]
    required = {"model.llm.embed.weight", "model.llm.unembed.weight", "model.llm.norm.weight",
                "model.llm.attn_res.output_query", "model.llm.middle_cycle_cross_pass_queries",
                "model.llm.middle_cycle_cross_pass_kv_queries"}
    expert_kinds = {}
    for i in range(24):
        p = f"model.llm.layers.{i}."
        required |= {p + "attn.wq_du.weight", p + "attn.attn_sink_proj.weight",
                     p + "attn.rel_logits_proj.proj", p + "mlp.gate.weight", p + "mlp.gate.bias",
                     p + "attn_sconv.weight", p + "mlp_sconv.weight"}
        kinds = {transforms.get(p + "mlp.experts." + n) for n in ("w13_weight", "w2_weight")}
        if len(kinds) != 1 or None in kinds:
            raise ValueError(f"Inkling 2 layer {i} experts have no single declared format")
        kind = kinds.pop()
        expert_kinds[i] = kind
        for n in ("w13_weight", "w2_weight"):
            required |= {p + f"mlp.experts.{n}", p + f"mlp.experts.{n}_scale"}
            if kind == NVFP4:
                required |= {p + f"mlp.experts.{n}_amax",
                             p + f"mlp.experts.{n}_outer_scale_denominator"}
        if 6 <= i < 18:
            required.add(p + "controllers.attn.down_weight")
    missing = sorted(required - weights.keys())
    if missing:
        raise ValueError(f"Inkling 2 tensors are incomplete: {missing[0]}")
    artifact.update(text_config=text, expert_kinds=expert_kinds,
                    has_mtp=(path / "mtp.safetensors").is_file())
    return artifact


# Opt-in execution switches; every one defaults off and is unqualified.
KERNEL_OPTIONS = {
    # Row-stacked q/r/gate/sink(/k/v) input projection, one GEMV when a
    # visit's q/k/v inputs coincide (implemented 2026-11-20; A/B pending).
    "inkling2_fused_projections": "fused_projections",
    # BF16-weight FP32-accumulate head GEMV instead of the cached FP32 head
    # copy (1.85 GB resident, re-read every token); summation order differs.
    "inkling2_head_kernel": "head_kernel",
    # One-token decode passes the per-query sink to SDPA ``sinks`` (one SDPA
    # per row) instead of concatenating a zero K/V column, which copied the
    # whole K/V on all 48 visits every token (kapitsa/llama.cpp/Rapid-MLX do
    # the same; implemented 2026-11-21, A/B pending). GPU reduction
    # partition differs (S vs S+1 keys).
    "inkling2_sink_sdpa": "sink_sdpa",
}
# Varloop (Pierre 2026-11-21): run the middle
# block N times instead of the trained 3. The checkpoint has is_varloop
# false, so every N != 3 is out of distribution; it is a research knob, not a
# serving option. Incompatible with DSpark (its taps assume 48 visits).
LOOP_PASSES_OPTION = "inkling2_loop_passes"
# Hard no-thinking (Pierre 2026-11-21): with
# this policy on, a request that turns thinking off (enable_thinking false,
# reasoning_effort "none", or thinking_budget 0) gets the assistant header and
# text marker appended to the prompt, so the model starts in the text channel
# instead of at the header where it may open <|content_thinking|>. Default
# off: without it "none" stays the template's soft hint.
HARD_NO_THINKING_OPTION = "inkling2_hard_no_thinking"
TEXT_PREFILL = ("<|message_model|>", "<|content_text|>")
# Prefill mask budget (MiB; default 64). Long-context prefill builds the
# relative-bias mask in row chunks of this many bytes and runs one SDPA per
# chunk over the whole K/V: at 128K a 64 MiB budget is ~5 query rows per call,
# so a 512-token chunk re-reads K/V ~100 times per global visit. A larger
# budget trades transient memory for fewer passes; rows per call change, so
# SDPA kernel choice may too (near-exact). Opt-in pending an A/B.
PREFILL_MASK_OPTION = "inkling2_prefill_mask_mb"


def validate_policy(value):
    """``(kernel switches, external-draft keys)`` from an execution policy."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("Inkling 2 execution policy must be an object")
    value = dict(value)
    kernels = {}
    for key, name in KERNEL_OPTIONS.items():
        flag = value.pop(key, False)
        if type(flag) is not bool:
            raise TypeError(f"Inkling 2 {key} must be a boolean")
        kernels[name] = flag
    hard = value.pop(HARD_NO_THINKING_OPTION, False)
    if type(hard) is not bool:
        raise TypeError(f"Inkling 2 {HARD_NO_THINKING_OPTION} must be a boolean")
    kernels["hard_no_thinking"] = hard
    mask_mb = value.pop(PREFILL_MASK_OPTION, None)
    if mask_mb is not None and (type(mask_mb) is not int or not 16 <= mask_mb <= 4096):
        raise TypeError(f"Inkling 2 {PREFILL_MASK_OPTION} must be an integer 16..4096")
    kernels["prefill_mask_mb"] = mask_mb
    passes = value.pop(LOOP_PASSES_OPTION, None)
    if passes is not None and (type(passes) is not int or not 1 <= passes <= 8):
        raise TypeError(f"Inkling 2 {LOOP_PASSES_OPTION} must be an integer 1..8")
    kernels["loop_passes"] = passes
    return kernels, value


def _deinterleave(v):
    """Fused ``[..., 2*inter, in]`` gate/up rows (gate, up alternating) -> halves."""
    import mlx.core as mx

    *lead, rows, cols = v.shape
    w = v.reshape(*lead, rows // 2, 2, cols)
    return mx.contiguous(w[..., 0, :]), mx.contiguous(w[..., 1, :])


def install_experts(model, experts: dict, kinds: dict) -> dict:
    """Replace each layer's SwitchGLU projections with its checkpoint format.

    NVFP4: ``ScaledNVFP4SwitchLinear`` (E2M1 x E4M3/16 x FP32 per-expert
    ``amax / outer_scale_denominator``).  MXFP8: ``QuantizedSwitchLinear``
    mode ``mxfp8`` (E4M3 x E8M0/32).  Gate/up arrive row-interleaved.
    """
    import mlx.core as mx

    from ..runtime.models.switch_layers import QuantizedSwitchLinear
    from ..runtime.nvfp4_linear import ScaledNVFP4SwitchLinear

    counts = {NVFP4: 0, MXFP8: 0}
    cfg = model.config
    hidden, inter, num = cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts
    for layer, tensors in sorted(experts.items()):
        kind = kinds[layer]
        mlp = model.model.layers[layer].mlp.switch_mlp
        for name, projections, (in_dims, out_dims) in (
            ("w13_weight", ("gate_proj", "up_proj"), (hidden, inter)),
            ("w2_weight", ("down_proj",), (inter, hidden)),
        ):
            weight = tensors.pop(name).view(mx.uint32)
            scales = tensors.pop(name + "_scale")
            parts = (
                list(zip(_deinterleave(weight), _deinterleave(scales)))
                if len(projections) == 2
                else [(weight, scales)]
            )
            global_scale = None
            if kind == NVFP4:
                amax = tensors.pop(name + "_amax").astype(mx.float32)
                denominator = tensors.pop(name + "_outer_scale_denominator").astype(mx.float32)
                global_scale = (amax / denominator).reshape(num)
            for projection, (w, s) in zip(projections, parts):
                if kind == NVFP4:
                    module = ScaledNVFP4SwitchLinear(in_dims, out_dims, num)
                    module.update({"weight": w, "scales": s, "weight_scale_2": global_scale})
                else:
                    module = QuantizedSwitchLinear(
                        in_dims, out_dims, num, bias=False, group_size=32, bits=8, mode="mxfp8"
                    )
                    module.weight, module.scales, module.biases = w, s, None
                    module.freeze()
                if module.weight.shape != (num, out_dims, in_dims * (4 if kind == NVFP4 else 8) // 32):
                    raise ValueError(f"Inkling 2 layer {layer} {projection} weight shape {module.weight.shape}")
                setattr(mlp, projection, module)
            counts[kind] += 1
        if tensors:
            raise ValueError(f"Inkling 2 layer {layer} has unused expert tensors {sorted(tensors)}")
    return {"nvfp4_layers": counts[NVFP4] // 2, "mxfp8_layers": counts[MXFP8] // 2}


def load_model(path, artifact):
    import mlx.core as mx

    from ..runtime.models.inkling2 import Model, TextConfig
    from ..runtime.ubc_evict import load_shards_evicting

    path = Path(path)
    model = Model(TextConfig.from_dict(artifact["text_config"]))

    def text_only(shard):
        return {k: v for k, v in shard.items() if k.startswith("model.llm.")}

    shards = [path / name for name in sorted(set(artifact["weight_map"].values()))]
    raw = load_shards_evicting(shards, sanitize=text_only)
    weights, experts = model.sanitize(raw)
    raw.clear()
    receipt = install_experts(model, experts, artifact["expert_kinds"])
    from mlx.utils import tree_flatten

    # Installed experts carry their own arrays; strict load checks the rest.
    weights.update((k, v) for k, v in tree_flatten(model.parameters()) if ".switch_mlp." in k)
    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    model.eval()
    mx.eval(model.parameters())
    mx.clear_cache()
    return model, receipt


DSPARK_DRAFT_NAMES = frozenset({"embedded_dspark"})


def dspark_record(path, artifact):
    """External-draft record for the bundled DSpark (header-only inspection)."""
    from types import SimpleNamespace

    from .inkling2_dspark import inspect_drafter

    inner = inspect_drafter(path, artifact["config"])
    return {
        "fingerprint": inner["draft_revision"],
        "draft_revision": inner["draft_revision"],
        "target_revision": artifact["identity"]["fingerprint"],
        "config": {"architectures": ["Inkling2DSpark"]},
        "args": SimpleNamespace(block_size=inner["block_size"], minimum_proposal_length=1),
        "dspark": inner,
    }



def _zero_thinking_budget(request):
    budget = request.get("thinking_budget")
    return type(budget) is int and budget == 0

class Inkling2Adapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    EXTERNAL_DEFAULT_NUM_DRAFT = 7
    EXTERNAL_ROUTE_TAG = "inkling2-dspark-v1"
    EXTERNAL_PROFILE = "inkling2-apcv2-dspark"
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "inkling2-apcv2-ordinary"
    reasoning_effort_semantics = "reasoning_strength_0_0.99"

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        from .process_globals import guarded_construction

        try:
            guarded_construction(
                self, lambda: self._initialize(model_path, execution_policy, weight_streaming)
            )
        except BaseException:
            self.close()
            raise

    def _initialize(self, model_path, execution_policy, weight_streaming):
        if weight_streaming is not None:
            raise ValueError("Inkling 2 declares no weight streaming mode")
        self.kernel_policy, external = validate_policy(execution_policy)
        self._parse_external_policy(external, family="Inkling 2")
        if external and self.kernel_policy["loop_passes"] not in (None, _LOOP["num_passes"]):
            raise ValueError("Inkling 2 DSpark taps assume the trained loop count")
        if external and external.get("draft_model") not in DSPARK_DRAFT_NAMES:
            raise ValueError(f"Inkling 2 draft_model must be one of {sorted(DSPARK_DRAFT_NAMES)}")
        artifact = inspect_artifact(model_path)
        record = None
        if external:
            if not artifact["has_mtp"]:
                raise ValueError("Inkling 2 artifact carries no mtp.safetensors")
            # Fail closed on the drafter before the target is loaded.
            record = dspark_record(artifact["identity"]["path"], artifact)
            self._check_num_draft(record)
        self.config, self.identity, self.layout = (
            artifact["config"], artifact["identity"], CACHE_LAYOUT,
        )
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        require_process_numerics("Inkling 2 native profile")
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        clear_inherited_profile(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_"))
        os.environ.update(self.environment)
        assert_process_profile_applied("Inkling 2 native profile")
        from transformers import AutoTokenizer

        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

        path = Path(self.identity["path"])
        self.model, self.expert_receipt = load_model(path, artifact)
        self.hard_no_thinking = self.kernel_policy.pop("hard_no_thinking")
        self.model.configure_kernels(**self.kernel_policy)
        self.layout = self.model.apc_v2_layout
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        # <|end_message|> separates thought/text messages; only end-sampling ends the turn.
        terminal = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
        if terminal != self.config["eos_token_id"]:
            raise ValueError("Inkling 2 end-sampling token differs from checkpoint EOS")
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[terminal]
        )
        self.max_context = int(artifact["text_config"]["model_max_length"])
        self.has_mtp = artifact["has_mtp"]
        if record is not None:
            from functools import partial

            from ..runtime.drafters.inkling2_dspark_external import load_external_drafter

            self.identity = {**self.identity, "draft_revision": record["draft_revision"],
                             "target_revision": record["target_revision"]}
            self._bind_external_drafter(
                record,
                lambda rec, model: load_external_drafter(
                    rec["dspark"], model, num_draft=self._external_num_draft()),
                self.descriptor,
            )

    @property
    def checkpoint(self):
        return f"{Path(self.identity['path']).name}@{self.identity['fingerprint']}"

    def execution_numerics_contract(self):
        root = Path(__file__).parents[1] / "runtime/models"
        names = ("inkling2/language.py", "inkling2/config.py", "inkling/cache.py",
                 "inkling/language.py", "switch_layers.py", "../nvfp4_linear.py")
        return {
            "operation": NUMERICS,
            "kernel_policy": dict(getattr(self, "kernel_policy", {})),
            "source_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
            },
            "qualification": "unqualified",
        }

    def execution_config(self, *, max_lanes, prefill_step):
        if getattr(self, "draft_model", None) is not None:
            if max_lanes != 1:
                # Verify over sliding-window rings beside recurrent state is
                # B1 only (HybridVerifyRows); fail at startup, not mid-round.
                raise ValueError("Inkling 2 DSpark serving is B1 only: start with --max-lanes 1")
            return self._external_execution_config(max_lanes=max_lanes, prefill_step=prefill_step)
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
        return 512

    # ----- prompts and output -----

    @staticmethod
    def _reasoning_effort(request):
        # The template's lowest level is "none" (0.0), which the card says does
        # not fully disable reasoning; unset renders the template default 0.9.
        effort = request.get("reasoning_effort")
        if _zero_thinking_budget(request):
            # A zero budget is the template's lowest level, not a hard cap:
            # the model may still open a thinking block (kapitsa #1380).
            if effort not in (None, "none") or request.get("enable_thinking") is True:
                raise ValueError(
                    "Inkling 2 thinking_budget 0 conflicts with reasoning_effort/enable_thinking"
                )
            return "none"
        if effort is None:
            return "none" if request.get("enable_thinking") is False else None
        return effort

    @staticmethod
    def _messages(request):
        if request.get("tools") and request.get("tool_choice", "auto") != "none":
            raise ValueError("Inkling 2 tool routing is not implemented")
        if request.get("thinking_budget") is not None and not _zero_thinking_budget(request):
            raise ValueError("Inkling 2 thinking budgets other than 0 are not implemented")
        if request.get("thinking_steer_alpha"):
            raise ValueError("Inkling 2 thinking steering is not implemented")
        messages = copy.deepcopy(request["messages"])
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(
                part.get("type") not in ("text", "input_text")
                for part in content
                if isinstance(part, dict)
            ):
                raise ValueError("Inkling 2 native text adapter does not accept media")
        return messages

    def _text_prefill(self, request):
        """Whether this request starts generation inside the text channel."""
        if not getattr(self, "hard_no_thinking", False) or "messages" not in request:
            return False
        return (request.get("enable_thinking") is False
                or self._reasoning_effort(request) == "none")

    def thinking_control(self, request):
        """Per-request receipt: how thinking was (not) suppressed."""
        effort = self._reasoning_effort(request) if "messages" in request else None
        return {
            "mode": "hard_text_prefill" if self._text_prefill(request) else "template_effort",
            "reasoning_effort": effort,
            "policy_hard_no_thinking": bool(getattr(self, "hard_no_thinking", False)),
        }

    def _render(self, request, tokenize):
        kwargs = {}
        effort = self._reasoning_effort(request)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        rendered = self.tokenizer.apply_chat_template(
            self._messages(request), tokenize=tokenize, add_generation_prompt=True, **kwargs
        )
        if not self._text_prefill(request):
            return rendered
        if not tokenize:
            return rendered + "".join(TEXT_PREFILL)
        ids = [self.tokenizer.convert_tokens_to_ids(marker) for marker in TEXT_PREFILL]
        if any(not isinstance(i, int) or i < 0 for i in ids):
            raise ValueError("Inkling 2 tokenizer lacks the text-channel markers")
        return list(rendered) + ids

    def prompt_tokens(self, request):
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        return self._render(request, True)

    def render_prompt(self, request):
        if "messages" not in request:
            return request["prompt"]
        return self._render(request, False)

    @staticmethod
    def thinking_enabled(request):
        return False  # generated content markers select the channel

    def output_parser(self, request):
        from .inkling_output import INKLING2_TEXT_MARKERS, InklingOutputParser

        parser = InklingOutputParser(
            chat="messages" in request,
            stops=request.get("stop", ()),
            text_markers=INKLING2_TEXT_MARKERS,
            header_names=True,
        )
        if self._text_prefill(request):
            parser.channel = "content"  # the prompt already opened the text channel
        return parser

    def tool_constraint(self, request):
        return None

    def diagnostics(self):
        return {
            "family": "inkling2",
            "route": "ordinary",
            "qualification": "unqualified",
            "checkpoint": self.checkpoint if getattr(self, "identity", None) else None,
            "adapter_default_route": "ordinary",
            "ordinary_reference_preserved": True,
            "cache_layout": self.layout,
            "numerics": NUMERICS,
            "kernel_policy": dict(getattr(self, "kernel_policy", {})),
            "kernel_counters": (self.model.kernel_receipt() if getattr(self, "model", None) else None),
            "loop": (self.model.loop_receipt() if getattr(self, "model", None) else None),
            "hard_no_thinking": bool(getattr(self, "hard_no_thinking", False)),
            "experts": getattr(self, "expert_receipt", None),
            "target_forward_observed": bool(getattr(getattr(self, "model", None), "completed_forwards", 0)),
            "dspark": {
                "implemented": True,
                "selected": getattr(self, "draft_model", None) is not None,
                "qualified": False,
                "weights_present": bool(getattr(self, "has_mtp", False)),
                "num_draft": self._external_num_draft() if getattr(self, "draft_model", None) is not None else None,
                "stats": _drafter_stats(getattr(self, "draft_model", None)),
            },
            "media": "weights present; native text route refuses media",
        }


def _drafter_stats(draft):
    while draft is not None and hasattr(draft, "backend"):
        draft = draft.backend
    return dict(getattr(draft, "stats", {}) or {}) if draft is not None else None
