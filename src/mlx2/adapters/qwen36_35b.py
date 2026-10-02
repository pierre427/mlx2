"""GPU-free inspection plus the Qwen3.6 35B-A3B serving adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import PROCESS_NUMERICS, require_process_numerics
from .mtp_depth_cap import validate_self_mtp_num_draft
from .qwen38_27b import (
    EAGER_DISPATCH_POLICY_KEYS,
    Qwen3827BAdapter,
    eager_dispatch_environment,
    eager_dispatch_policy,
    resolve_eos_token_ids,
)

CACHE_LAYOUT = "qwen36-35b-a3b-hybrid-layer-segments-v1"
# Default on at width 3 (width 4 until 2026-10-02; Qwen3.8-27B and Xing stay
# at 4).  Explicit rather than inherited through Qwen3.8.
#
# Width 3: a 4-lane cohort kept on MTP at width 4 loses to ordinary decode
# (260.2 vs 284.0 tok/s, -8%); width 3 hands it off and measured 298.6 vs
# 260.2 tok/s at 4 lanes (+14.0/+21.6/+12.5/+15.1% paired reps), equal at 3
# and 8 lanes (qualification/runs/port-park-memory-20261002,
# qwen36-width4-vs-width3.json).  Width 4 passed the earlier Qwen3.6
# 131K/16-GiB handoff campaign for explicitly selected native MTP.
DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH = 3


def descriptor_for(*, has_mtp: bool) -> ModelDescriptor:
    capabilities = {
        Capability.TEXT,
        Capability.STREAMING,
        Capability.TOOLS,
        Capability.REASONING,
        Capability.CONTINUOUS_BATCH,
        Capability.PREFIX_REUSE,
        Capability.APC_V2,
        Capability.LAYERED_CACHE,
        Capability.GRAMMAR,
        Capability.PROMPT_LOOKUP,
    }
    planes = {
        StatePlane.ATTENTION_KV,
        StatePlane.RECURRENT,
        StatePlane.RNG,
        StatePlane.TRANSCRIPT,
        StatePlane.GRAMMAR,
    }
    if has_mtp:
        capabilities.update({Capability.MTP, Capability.SEGMENTED_MTP})
        planes.add(StatePlane.DRAFT)
    return ModelDescriptor(
        model_type="qwen3_5_moe",
        family="qwen3.6-35b-a3b",
        variant="native-mtp" if has_mtp else "ordinary",
        state_planes=frozenset(planes),
        capabilities=frozenset(capabilities),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.qwen36_35b.Qwen3635BA3BAdapter",
            "qualification": "pending",
            "scope": "text-only",
            "compiled_decode": "implemented-upstream-not-selected",
            "moe_kernels": "implemented-shared-not-selected",
        },
    )


QWEN36_35B = descriptor_for(has_mtp=False)


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError("Qwen3.6 35B-A3B requires qwen3_5_moe")
    expected = {
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match Qwen3.6 35B-A3B")
    if text.get("mtp_num_hidden_layers", 0) not in (0, 1):
        raise ValueError("only the single-layer Qwen3.6 MTP head is implemented")
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("artifact has no indexed weights")
    shards = sorted(set(weight_map.values()))
    for name in shards:
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("weight shard paths must stay within the artifact")
        if not (path / name).is_file():
            raise ValueError(f"missing weight shard: {name}")
    mtp_keys = [
        key for key in weight_map if key.startswith(("language_model.mtp.", "mtp."))
    ]
    has_mtp = bool(mtp_keys)
    if has_mtp and text.get("mtp_num_hidden_layers") != 1:
        raise ValueError("MTP tensors and configured head count disagree")
    if has_mtp:
        normalized = {key.removeprefix("language_model.") for key in mtp_keys}
        required = {
            "mtp.fc.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.gate.weight",
        }
        converted = "mtp.layers.0.mlp.switch_mlp.down_proj.weight" in normalized and (
            "mtp.layers.0.mlp.switch_mlp.gate_up_proj.weight" in normalized
            or {
                "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
                "mtp.layers.0.mlp.switch_mlp.up_proj.weight",
            }
            <= normalized
        )
        # The official checkpoint keeps the experts fused in the hub layout
        # (``experts.gate_up_proj`` [E, 2I, H], ``experts.down_proj``
        # [E, H, I]).  ``Model.sanitize`` converts both and the strict load
        # checks every shape (MTPLX#574 lost these keys to a lenient load).
        fused = {
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
        } <= normalized
        if not required <= normalized or not (converted or fused):
            raise ValueError("embedded MTP head is incomplete")
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    records = []
    for name in shards:
        stat = (path / name).stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": weight_map,
        "has_mtp": has_mtp,
        "mtp_tensor_count": len(mtp_keys),
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
    }


# Kernel switches the execution policy may select for a GPU A/B.  Each
# defaults to the qualified stock profile, so a policy that omits them keeps
# the environment -- and the qualification identity -- byte-identical.
#   fused_gdn_decode: bit-exact B1 decode kernel, 1.077x/1.107x warm decode
#     (docs/ports/QWEN36-35B-A3B.md); batched and speculative-verify cells
#     are unmeasured.  Falls back (counted) wherever admission declines.
#   moe_fused_gate_up: load-time gate/up concatenation, on in Flash-Next,
#     pinned off here with no recorded reason.
#   gdn_core: MLX's native gated_delta_update for 17-256 row prefill chunks;
#     parity on this head geometry is not established.
#   moe_routed_candidate: omlx #4113 top-8 routed decode over the split
#     gate/up tables (qwen4_routed_decode candidate).  UNQUALIFIED, default
#     off, explicit execution policy only (never inherited from the
#     environment); loading refuses it when no MoE layer admits it.
# New decode slices below are explicit-only, default-off and unqualified.
# The Flash-Next tile4 down remains ineligible; Qwen3.6 keeps stock down numerics.
KERNEL_POLICY_ENV = {
    "fused_gdn_decode": "MLX_QWEN36_FUSED_GDN_DECODE",
    "moe_fused_gate_up": "MLX_QWEN4_MOE_FUSED_GATE_UP",
    "gdn_core": "MLX_GDN_CORE",
    "moe_routed_candidate": "MLX_QWEN36_MOE_ROUTED_CANDIDATE",
    "moe_routed_decode": "MLX_QWEN4_MOE_ROUTED_DECODE",
    "moe_topk_fold": "MLX_QWEN4_MOE_TOPK_FOLD",
    "moe_window": "MLX_QWEN36_MOE_WINDOW",
    "fused_gdn_batch_decode": "MLX_QWEN36_FUSED_GDN_BATCH_DECODE",
    "fused_gdn_verify": "MLX_QWEN36_FUSED_GDN_VERIFY",
    "fused_gdn_batch_verify": "MLX_QWEN36_FUSED_GDN_BATCH_VERIFY",
}
NEW_DECODE_KERNELS = frozenset(
    {
        "moe_routed_decode",
        "moe_topk_fold",
        "moe_window",
        "fused_gdn_batch_decode",
        "fused_gdn_verify",
        "fused_gdn_batch_verify",
    }
)
ENUM_KERNELS = {
    "moe_routed_decode": (
        "off",
        "gate_up",
        "gate_up_down",
        "gate_up_down_shared",
        "two_launch",
    ),
    "moe_topk_fold": ("off", "launch"),
}
EXPLICIT_ONLY_KERNELS = NEW_DECODE_KERNELS | {"moe_routed_candidate"}
# Profile variables ``_select_decode_wins`` re-applies to every module that
# reads them, through live setters, after each load; a value a module latched
# under an earlier profile in this process is overwritten.  Exempt from the
# import-order guard so a second Qwen3.6 load (A/B arms, harnesses) may change
# them.  ``MLX_QWEN36_DECODE_WINS`` is receipt identity only (nothing reads
# it).  The routed-decode and top-k modes are latched by qwen3_next at import
# but set on every Qwen36SparseMoeBlock (decoder and MTP layers alike) by
# set_moe_routed_decode_mode / set_moe_topk_mode, and qwen3_next reads no other
# state from them.  Every other latch (MLX_GDN_*, the other MLX_QWEN4_MOE_*,
# QSDPA) must match.
LIVE_APPLIED_ENV = frozenset(
    {
        "MLX_QWEN4_MOE_ROUTED_DECODE",
        "MLX_QWEN4_MOE_TOPK_FOLD",
        "MLX_QWEN36_FUSED_GDN_DECODE",
        "MLX_QWEN36_FUSED_GDN_BATCH_DECODE",
        "MLX_QWEN36_FUSED_GDN_VERIFY",
        "MLX_QWEN36_FUSED_GDN_BATCH_VERIFY",
        "MLX_QWEN36_MOE_WINDOW",
        "MLX_QWEN36_DECODE_WINS",
    }
)


def validate_kernel_choice(key, value):
    if key in ENUM_KERNELS:
        if value not in ENUM_KERNELS[key]:
            raise ValueError(f"Qwen3.6 {key} must be one of {ENUM_KERNELS[key]}")
    elif type(value) is not bool:
        raise ValueError(f"Qwen3.6 {key} must be boolean")
    return value


def choice_selected(key, value):
    return value != "off" if key in ENUM_KERNELS else bool(value)


def configure_environment(kernels=None) -> dict[str, str]:
    require_process_numerics("the Qwen3.6 profile")
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **PROCESS_NUMERICS,
        "MLX_GDN_PACKED": "1",
        "MLX_GDN_CORE": "0",
        "MLX_QWEN36_FUSED_GDN_DECODE": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_QWEN4_MOE_FUSED_GATE_UP": "0",
        "MLX_QWEN4_MOE_ROUTER_KERNEL": "0",
        "MLX_QWEN4_FUSED_EXPERT_KERNEL": "stock",
        "MLX_LM_SEGMENTED_SELF_MTP": "1",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "1",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
        # Committed MTP-boundary COW snapshots: default-on gate, pinned so the
        # receipt records it.
        "MLX_LM_MTP_BOUNDARY_COW": "1",
    }
    # These three switches are supported Qwen3.6 A/B choices.  Preserve an
    # operator's explicit environment value unless the request's execution
    # policy selects that switch.  Other inherited MLX kernel experiments are
    # still cleared below; they are not qualified for this model's geometry.
    for key, name in KERNEL_POLICY_ENV.items():
        if name in os.environ and key not in EXPLICIT_ONLY_KERNELS:
            profile[name] = os.environ[name]
    for key, enabled in (kernels or {}).items():
        validate_kernel_choice(key, enabled)
        if key in EXPLICIT_ONLY_KERNELS and not choice_selected(key, enabled):
            # Absent unless selected, so stock receipts keep matching; an
            # inherited value is cleared with the MLX_QWEN prefix below.
            continue
        profile[KERNEL_POLICY_ENV[key]] = (
            enabled if key in ENUM_KERNELS else ("1" if enabled else "0")
        )
    if any(
        choice_selected(key, value)
        for key, value in (kernels or {}).items()
        if key in NEW_DECODE_KERNELS
    ):
        profile["MLX_QWEN36_DECODE_WINS"] = "1"
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


class Qwen3635BA3BAdapter(Qwen3827BAdapter):
    # Native MTP, restored 2026-09-20 once the wide-cohort ordinary handoff
    # removed the reason it was demoted.  The 2026-09-19 demotion to ordinary
    # was correct on its own evidence: native MTP matched ordinary
    # single-stream (102.0 vs 102.7 tok/s) and lost 25-34% batched (B8 175 vs
    # 235, B16 212 vs 283), because a cohort locks its compute width after the
    # first true-batched cycle and defers late arrivals.  With the handoff (at
    # ``max_mtp_width`` 4 for these numbers; 3 since 2026-10-02) the cohort migrates to the ordinary batcher at a
    # closed boundary and the batched loss inverts into a gain: on GPU at
    # mlx2 994123b, ordinary / fixed MTP / MTP+handoff was 96.6 / 97.0 / 98.3
    # single-stream, 234.5 / 163.0 / 243.8 at B8 and 282.6 / 212.1 / 305.0 at
    # B16.  Fixed MTP without the handoff still loses, so this default is only
    # sound while ``mtp_ordinary_handoff`` below stays enabled.
    default_route = "native_mtp"
    # Required by the route above, not merely available to it: see the note on
    # ``default_route``.  Also applies when native MTP is selected explicitly.
    default_mtp_ordinary_handoff_max_width = DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH
    # Interior checkpoints ``"auto"``: measured on the two sibling hybrids that
    # share this GDN cache and capture path (Flash-Next and Qwen3.8 27B native
    # MTP, zero output differences; qualification/runs/interior-ckpt-20260919).
    # The capture is exact state, so the sibling evidence covers correctness;
    # the 35B TTFT gain itself is unmeasured.
    default_route_execution_policy = {
        "native_mtp": {"apc_interior_checkpoints": "auto"},
    }
    descriptor = QWEN36_35B
    # Per-layer eager dispatch (MTPLX #579, the Flash-Next mechanism), default
    # stride 2 (Pierre, 2026-10-01).  Bit-exact here, and this 3B-active MoE is
    # host-bound enough to gain at the forward (3-row verify -21%), for
    # ordinary B4 (+7%, every round) and native-MTP B1 (median +2.9%, 4 of 6
    # rounds ahead); ordinary B1 was noise (qualification/runs/
    # recon-20261001/l7-decode-perf).  ``{"eager_dispatch_stride": 0}`` turns
    # it off.  It is route identity, so the 35B route needs requalification.
    default_eager_dispatch_stride = 2
    # Unqualified candidates: the server accepts these policy keys only in
    # qualification mode (ServingEngine); direct adapter harnesses may opt in.
    qualification_mode_only_policy = frozenset({"moe_routed_candidate"})
    # Vendor sampling defaults: Qwen/Qwen3.6-35B-A3B model card and the
    # artifact's generation_config.json (see ``adapters/qwen.py``).
    from .qwen import QWEN36_35B_SAMPLING as sampling_defaults

    # Early-load MoE expert streaming (runtime/streamed_load.py).  Read from
    # this class's own __dict__: subclasses must declare it themselves.
    weight_streaming_modes = frozenset({"moe_experts"})

    def __init__(
        self,
        model_path: str,
        *,
        require_mtp: bool = False,
        execution_policy=None,
        weight_streaming=None,
    ):
        from ..runtime.streamed_load import require_declared

        stream_request = require_declared(type(self), weight_streaming)
        if execution_policy is not None and not isinstance(execution_policy, dict):
            raise ValueError("execution policy must be a JSON object")
        policy = {} if execution_policy is None else dict(execution_policy)
        if set(policy) - {
            "num_draft",
            "gdn_state_dtype",
            *KERNEL_POLICY_ENV,
            *EAGER_DISPATCH_POLICY_KEYS,
        }:
            raise ValueError(
                "Qwen3.6 execution policy supports only num_draft, gdn_state_dtype, "
                "the eager-dispatch keys and the kernel switches "
                + ", ".join(sorted(KERNEL_POLICY_ENV))
            )
        eager_dispatch = eager_dispatch_policy(
            policy, self.default_eager_dispatch_stride
        )
        from .flash_next_policy import FlashNextPolicy

        # GDN recurrent-state storage class (runtime/models/gdn_state.py).
        gdn_state_dtype = FlashNextPolicy(
            gdn_state_dtype=policy.pop("gdn_state_dtype", "float32")
        ).gdn_state_dtype
        self._num_draft = validate_self_mtp_num_draft(policy.get("num_draft", 2))
        self._kernels = {}
        for key in KERNEL_POLICY_ENV:
            if key in policy:
                self._kernels[key] = validate_kernel_choice(key, policy[key])
        if self._kernels.get("moe_routed_candidate") and (
            self._kernels.get("moe_routed_decode", "off") != "off"
            or self._kernels.get("moe_window")
        ):
            raise ValueError(
                "historical routed candidate and new routed/window slices are exclusive"
            )
        if stream_request is not None and (
            self._kernels.get("moe_routed_candidate")
            or any(
                choice_selected(k, v)
                for k, v in self._kernels.items()
                if k in {"moe_routed_decode", "moe_window"}
            )
        ):
            raise ValueError(
                "moe_routed_candidate reads resident expert tables and cannot "
                "run with weight streaming"
            )
        self.weight_stream = None
        artifact = inspect_artifact(model_path)
        if require_mtp and not artifact["has_mtp"]:
            raise ValueError("requested MTP requires embedded head weights")
        self.identity = artifact["identity"]
        self.descriptor = descriptor_for(has_mtp=artifact["has_mtp"])
        self.environment = eager_dispatch_environment(
            configure_environment(self._kernels)
            if self._kernels
            else configure_environment(),
            eager_dispatch,
        )
        from ..runtime.models.import_env import assert_profile_applied

        # Import-time latches (GDN core/packed, MoE gate/up, QSDPA) must match
        # this profile; the decode switches are re-applied live below.
        assert_profile_applied("the Qwen3.6 35B adapter", live=LIVE_APPLIED_ENV)
        self.layout = CACHE_LAYOUT
        self._tables = []
        path = Path(self.identity["path"])
        config = dict(artifact["config"])
        config["text_config"] = dict(config.get("text_config", config))
        if not artifact["has_mtp"]:
            config["text_config"]["mtp_num_hidden_layers"] = 0

        from transformers import AutoTokenizer

        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        try:
            self._load_weights(
                path,
                artifact,
                config,
                stream_request,
                load_shards_evicting,
                eager_dispatch,
            )
            self._record_load_dtype()
            self._select_gdn_state(gdn_state_dtype)
            self._select_routed_candidate()
            self._select_decode_wins()
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            # transformers' Qwen2Tokenizer drops the declared combining-mark split rule.
            from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

            self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
            # The official config names only <|endoftext|>; the tokenizer's chat
            # EOS <|im_end|> ends an assistant turn and must stop generation too.
            eos = resolve_eos_token_ids(config, tokenizer)
            self.tokenizer = TokenizerWrapper(
                tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=eos
            )
            self.max_context = int(config["text_config"]["max_position_embeddings"])
            if self.weight_stream is not None:
                # The dtype probe above paged experts in as load evidence only.
                self.weight_stream.begin_serving()
        except BaseException:
            self.close()
            raise

    def _load_weights(
        self,
        path,
        artifact,
        config,
        stream_request,
        load_shards_evicting,
        eager_dispatch,
    ):
        """Build ``self.model`` and load its weights, ordinary or streamed.

        The ordinary branch is the reference: shard-eager load, sanitize,
        quantize, strict load, evaluate.  The streamed branch replaces the
        routed expert tables before anything is evaluated.
        """
        import mlx.core as mx
        import mlx.nn as nn

        from ..runtime.models.qwen36_35b import Model, ModelArgs
        from .norm_repair import norm_means

        self.model = Model(ModelArgs.from_dict(config))
        names = sorted(set(artifact["weight_map"].values()))
        files = [path / name for name in names]
        quant = config.get("quantization", config.get("quantization_config"))

        def quantize(weights):
            if not quant:
                return

            def predicate(name, module):
                if name in quant:
                    return quant[name]
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights

            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant.get("mode", "affine"),
                class_predicate=predicate,
            )

        if stream_request is None:
            weights = self.model.sanitize(
                load_shards_evicting(files, sanitize=self.model.shard_prune)
            )
        else:
            weights = self._load_streamed(
                path, names, artifact, config, stream_request, quantize
            )
        # ``sanitize`` decided the norm fold per group and repaired MTP norms
        # a converter left unshifted (oQ's mean<0.5 rule skips four of seven
        # on this head; see norm_repair). Record which ones.
        report = getattr(
            getattr(self.model, "language_model", None), "norm_convention", None
        )
        self.norm_convention = report
        self.mtp_norm_repairs = [] if report is None else report.repaired_head_keys
        self.mtp_norm_means = norm_means(weights, "mtp.")
        if stream_request is None:
            quantize(weights)
            self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        if eager_dispatch[0]:
            self.model.model.set_eager_dispatch(*eager_dispatch)
        weights.clear()
        mx.clear_cache()

    def _load_streamed(self, path, names, artifact, config, stream_request, quantize):
        """Early-load expert streaming; the manager joins ``self._tables``."""
        from ..runtime.streamed_load import force_stock_expert_arithmetic, load_streamed

        if self.environment.get("MLX_LM_COMPILED_DECODE", "0") != "0":
            raise ValueError("weight streaming cannot run under compiled decode")
        top_k = config["text_config"].get("num_experts_per_tok")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("weight streaming needs the routed num_experts_per_tok")

        def installed(manager):
            # Explicit stock expert arithmetic, recorded in the receipt.
            manager.forced_stock = force_stock_expert_arithmetic(self.model)

        loaded = load_streamed(
            self.model,
            path,
            names,
            request=stream_request,
            sanitize=self.model.sanitize,
            quantize=quantize,
            records=self.identity["files"],
            weight_map=artifact["weight_map"],
            shard_prune=self.model.shard_prune,
            top_k=top_k,
            on_installed=installed,
        )
        self.weight_stream = loaded.manager
        self._tables.append(loaded.manager)
        return loaded.weights

    def profile_name(self, mtp):
        if mtp and Capability.MTP not in self.descriptor.capabilities:
            raise ValueError("requested MTP requires embedded head weights")
        return f"qwen36-35b-a3b-apcv2-{'mtp' + str(self._num_draft) if mtp else 'ordinary'}"

    def cache_budget(self, *, mtp):
        """Dense-27B cache topology, but this model's own verify transient.

        Qwen3.6-35B-A3B shares ``Qwen38CacheBudget``'s cache arithmetic with
        the dense Qwen3.8-27B, and inherited its 3.1 GiB/lane forward
        workspace with it.  3.1 is a dense-27B measurement: this model is a
        3B-active MoE and measures 0.015-0.023 GiB/lane at k=2 across
        1K/4K/16K x 1/2/4 lanes on an M3 Pro (2026-09-19; raw numbers in
        provenance/lane-transient-moe.json).  Charging 3.1 on a 36 GiB host
        exceeded the whole ~2.71 GiB lane budget, so every self-MTP request
        fell to the k=0 depth floor and the route ran zero draft cycles.
        """
        from dataclasses import replace

        from ..runtime.memory_policy import SelfMTPLaneAdmissionController

        return replace(
            super().cache_budget(mtp=mtp),
            transient_gib_per_lane=(
                SelfMTPLaneAdmissionController.MOE_TRANSIENT_GIB_PER_LANE
            ),
        )

    def _select_routed_candidate(self):
        """Apply an explicitly selected candidate, or refuse the load.

        Fail closed: every MoE layer must admit the candidate structurally
        (tables only, nothing evaluated); per-call runtime declines (Metal,
        served-SiLU form) run the reference body and are counted.
        """
        if self.environment.get("MLX_QWEN36_MOE_ROUTED_CANDIDATE") != "1":
            return
        import mlx.core as mx

        from ..runtime.models import qwen4_routed_decode as routed
        from ..runtime.models.qwen3_next import Qwen3NextSparseMoeBlock

        blocks = [
            m
            for _, m in self.model.named_modules()
            if isinstance(m, Qwen3NextSparseMoeBlock)
        ]
        if not blocks:
            raise ValueError("moe_routed_candidate: no MoE layer")
        for block in blocks:
            switch = block.switch_mlp
            hidden = switch.down_proj["weight"].shape[1]
            probe = mx.zeros((1, 1, 1, 1, hidden), dtype=mx.bfloat16)
            inds = mx.zeros((1, 1, block.top_k), dtype=mx.uint32)
            scores = mx.zeros((1, 1, block.top_k), dtype=mx.bfloat16)
            admission = routed.admit_routed_candidate(
                probe,
                inds,
                scores,
                switch.get("gate_proj"),
                switch.get("up_proj"),
                switch.down_proj,
            )
            if not admission.accepted:
                raise ValueError(f"moe_routed_candidate refused: {admission.reason}")
            block.set_moe_routed_candidate_mode("two_launch")

    def _select_decode_wins(self):
        # Live setters make the selection independent of import-time latches.
        from ..runtime.models.qwen36_35b import GatedDeltaNet
        from ..runtime.models.qwen36_moe_decode import Qwen36SparseMoeBlock

        fused_decode = (
            "fused"
            if self.environment.get("MLX_QWEN36_FUSED_GDN_DECODE") == "1"
            else "stock"
        )
        for _, module in self.model.named_modules():
            if isinstance(module, GatedDeltaNet):
                # The profile value, not ``_kernels``: an operator's explicit
                # environment value is preserved when the policy omits it.
                module.set_fused_gdn_decode_mode(fused_decode)
                for choice in ("batch_decode", "verify", "batch_verify"):
                    key = "fused_gdn_" + choice
                    getattr(module, "set_" + key + "_mode")(
                        "row_exact" if self._kernels.get(key, False) else "off"
                    )
            if isinstance(module, Qwen36SparseMoeBlock):
                module.set_moe_routed_decode_mode(
                    self._kernels.get("moe_routed_decode", "off")
                )
                module.set_moe_topk_mode(self._kernels.get("moe_topk_fold", "off"))
                module.set_moe_window_consumers(
                    ("batch_decode", "verify", "row_exact")
                    if self._kernels.get("moe_window")
                    else ()
                )

    def diagnostics(self):
        result = super().diagnostics()
        result["architecture"] = "sparse-moe-hybrid-gdn-gqa"
        result["layout"] = self.layout
        result["optimized_moe_selected"] = any(
            choice_selected(k, v)
            for k, v in (getattr(self, "_kernels", None) or {}).items()
            if k
            in {
                "moe_routed_decode",
                "moe_topk_fold",
                "moe_window",
                "moe_routed_candidate",
            }
        )
        result["compiled_decode_selected"] = False
        result["mtp_norm_repairs"] = list(getattr(self, "mtp_norm_repairs", ()))
        result["mtp_norm_means"] = dict(getattr(self, "mtp_norm_means", {}))
        if (getattr(self, "_kernels", None) or {}).get("fused_gdn_decode"):
            # The A/B reads fused calls and fallback reasons here.
            from ..runtime.models.qwen36_35b import qwen36_fused_gdn_stats

            result["fused_gdn_decode"] = qwen36_fused_gdn_stats(self.model)
        if (getattr(self, "_kernels", None) or {}).get("moe_routed_candidate"):
            from ..runtime.models.qwen3_next import routed_candidate_stats

            result["moe_routed_candidate"] = {
                "qualification": "unqualified",
                **routed_candidate_stats(self.model),
            }
        if any(
            choice_selected(k, v)
            for k, v in (getattr(self, "_kernels", None) or {}).items()
            if k in NEW_DECODE_KERNELS
        ):
            from ..runtime.models.qwen36_35b import qwen36_decode_wins_stats

            result["decode_wins"] = {
                "qualification": "unqualified",
                **qwen36_decode_wins_stats(self.model),
            }
        return result
