"""CPU-safe artifact inspection and the dense Qwen3.8 27B serving adapter.

Import/inspect performs no tensor imports or model loads. Instantiating the
adapter loads weights and is reserved for a separately authorized GPU window.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .external_draft_policy import ExternalDraftAdapterMixin
from .flash_next import FlashNextAdapter, gdn_state_diagnostics
from .mtp_depth_cap import validate_self_mtp_num_draft
from ..process_env import PROCESS_NUMERICS, require_process_numerics

CACHE_LAYOUT = "qwen38-27b-hybrid-layer-segments-v1"
# Adapter-owned rather than inherited from Flash-Next: threshold four passed
# the Qwen3.8 131K/16-GiB handoff campaign and is the intended post-qualification
# default.
DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH = 4


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
        Capability.PROMPT_LOOKUP,
        Capability.GRAMMAR,
    }
    planes = {
        StatePlane.ATTENTION_KV,
        StatePlane.RECURRENT,
        StatePlane.RNG,
        StatePlane.TRANSCRIPT,
    }
    if has_mtp:
        capabilities.update({Capability.MTP, Capability.SEGMENTED_MTP})
        planes.add(StatePlane.DRAFT)
    return ModelDescriptor(
        model_type="qwen3_5",
        family="qwen3.8-27b",
        variant="27b-mtp" if has_mtp else "27b-ordinary",
        state_planes=frozenset(planes),
        capabilities=frozenset(capabilities),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.qwen38_27b.Qwen3827BAdapter",
            "qualification": "pending",
            "scope": "text-only",
            "true_batched_segmented_mtp": "implemented-cpu-oracle-gpu-unqualified",
        },
    )


QWEN38_27B = descriptor_for(has_mtp=True)
QWEN38_27B_ORDINARY = descriptor_for(has_mtp=False)


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate local metadata without importing MLX or opening tensor payloads."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    if config.get("model_type") != "qwen3_5" or text.get("num_experts", 0):
        raise ValueError("Qwen3.8 27B requires the dense qwen3_5 artifact layout")
    expected = {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    if any(text.get(k) != v for k, v in expected.items()):
        raise ValueError("artifact topology does not match Qwen3.8 27B")
    if text.get("layer_types") not in (
        None,
        [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(64)
        ],
    ):
        raise ValueError("artifact layer order does not match Qwen3.8 27B")
    if text.get("mtp_num_hidden_layers", 0) not in (0, 1):
        raise ValueError("only the single-layer Qwen3.8 MTP head is implemented")
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if not isinstance(index, dict) or not index:
        raise ValueError("artifact has no indexed weights")
    names = sorted(set(index.values()))
    for name in names:
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("weight shard paths must stay within the artifact")
        if not (path / name).is_file():
            raise ValueError(f"missing weight shard: {name}")
    mtp_keys = [k for k in index if k.startswith(("language_model.mtp.", "mtp."))]
    has_mtp = bool(mtp_keys)
    if has_mtp and text.get("mtp_num_hidden_layers") != 1:
        raise ValueError("MTP tensors and configured head count disagree")
    if has_mtp:
        normalized = {k.removeprefix("language_model.") for k in mtp_keys}
        required = {
            "mtp.fc.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
        }
        if not required <= normalized:
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
    for name in names:
        stat = (path / name).stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": index,
        "has_mtp": has_mtp,
        "mtp_tensor_count": len(mtp_keys),
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
    }


def content_revision(model_path: str | Path) -> str:
    """Content pin of the target: config plus weight index (mlx2).

    ``identity["fingerprint"]`` also binds shard sizes and mtimes; this pin
    is what an external-draft policy names so a re-downloaded identical
    revision still matches while any config or tensor-map change fails.
    """
    path = Path(model_path).expanduser().resolve()
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        digest.update(name.encode())
        digest.update((path / name).read_bytes())
    return digest.hexdigest()


# External DFlash2 policy keys.  The two revision pins are mandatory: the
# drafter was trained against one target's hidden taps, so a draft or target
# that is not the pinned pair must fail before any tensor loads.
EXTERNAL_POLICY_KEYS = frozenset(
    {
        "draft_model",
        "num_draft",
        "pairwise_selection",
        "adaptive_verification",
        "proposal_composition",
        "continuation_pool",
        "lilicorr_feedback",
        "draft_revision",
        "target_revision",
        "draft_quantization",
        "tensorfold_prefill",
        "tensorfold_prefill_backend",
        "gdn_prefill_chunk",
        "gdn_prefill_segment_rows",
    }
)


def inspect_external_policy(policy: dict, model_path: str | Path) -> dict:
    """Header-only drafter inspection and revision check; no tensor loads."""
    from .dflash2 import content_revision as draft_content_revision
    from .dflash2 import inspect_drafter, validate_runtime_quantization

    unknown = set(policy) - EXTERNAL_POLICY_KEYS
    if unknown:
        raise ValueError(
            f"Qwen3.8 27B external draft policy has unknown keys: {sorted(unknown)}"
        )
    from .flash_next_policy import FlashNextPolicy

    FlashNextPolicy.from_mapping(
        {
            key: policy[key]
            for key in (
                "tensorfold_prefill",
                "tensorfold_prefill_backend",
                "gdn_prefill_chunk",
                "gdn_prefill_segment_rows",
            )
            if key in policy
        }
    )
    for key in ("draft_revision", "target_revision"):
        if not isinstance(policy.get(key), str) or len(policy[key]) != 64:
            raise ValueError(f"Qwen3.8 27B external draft policy must pin {key}")
    if policy.get("pairwise_selection", "host") not in ("host", "batched"):
        raise ValueError("pairwise_selection must be 'host' or 'batched'")
    target_revision = content_revision(model_path)
    if target_revision != policy["target_revision"]:
        raise ValueError(
            "DFlash2 target revision mismatch: policy pins "
            f"{policy['target_revision'][:12]}, artifact is {target_revision[:12]}"
        )
    record = inspect_drafter(policy["draft_model"], model_path)
    draft_revision = draft_content_revision(record)
    if draft_revision != policy["draft_revision"]:
        raise ValueError(
            "DFlash2 draft revision mismatch: policy pins "
            f"{policy['draft_revision'][:12]}, artifact is {draft_revision[:12]}"
        )
    args = record["args"]
    count = policy.get("num_draft", Qwen3827BAdapter.EXTERNAL_DEFAULT_NUM_DRAFT)
    if type(count) is not int or not 1 <= count < args.block_size:
        raise ValueError("num_draft must be a positive integer below the draft block size")
    adaptive = policy.get("adaptive_verification")
    if adaptive is not None:
        from ..runtime.acceptance_estimator import AdaptiveVerificationPolicy

        AdaptiveVerificationPolicy.from_value(adaptive, count)
    quantization = validate_runtime_quantization(policy.get("draft_quantization"))
    if quantization is not None:
        # Numerics differ from the bf16 drafter: a distinct cache identity.
        record = {
            **record,
            "fingerprint": hashlib.sha256(
                (record["fingerprint"] + json.dumps(quantization, sort_keys=True)).encode()
            ).hexdigest(),
        }
    return {
        **record,
        "draft_revision": draft_revision,
        "target_revision": target_revision,
        "runtime_quantization": quantization,
    }


def configure_environment() -> dict[str, str]:
    """Candidate dense profile; flags confer no qualification by themselves."""
    require_process_numerics("the Qwen3.8 27B profile")
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **PROCESS_NUMERICS, "MLX_GDN_PACKED": "1",
        "MLX_GDN_CORE": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "1",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "1",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
        # Committed MTP-boundary COW snapshots: default-on gate, pinned so the
        # receipt records it.
        "MLX_LM_MTP_BOUNDARY_COW": "1",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


def fused_gdn_policy(policy: dict) -> bool:
    """27B decode switch in the existing execution policy; false kills it."""
    enabled = policy.get("fused_gdn", False)
    if type(enabled) is not bool:
        raise ValueError("fused_gdn must be boolean")
    return enabled


EAGER_DISPATCH_POLICY_KEYS = ("eager_dispatch_stride", "eager_dispatch_max_rows")
# Route identity of a selected stride: recorded in the adapter environment
# (and so in qualification settings) only when the lever is on, so receipts
# of routes that leave it off stay byte-identical.  Nothing reads them back.
EAGER_DISPATCH_ENV = ("MLX2_EAGER_DISPATCH_STRIDE", "MLX2_EAGER_DISPATCH_MAX_ROWS")


def eager_dispatch_policy(policy: dict, default_stride: int = 0) -> tuple[int, int]:
    """Validate the per-layer eager-dispatch policy keys; stride 0 = off.

    The adapter's ``default_eager_dispatch_stride`` applies when the policy
    omits the key; an explicit 0 turns it off.
    """
    stride = policy.get("eager_dispatch_stride", default_stride)
    max_rows = policy.get("eager_dispatch_max_rows", 64)
    if type(stride) is not int or stride < 0:
        raise ValueError("eager_dispatch_stride must be a non-negative integer")
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError("eager_dispatch_max_rows must be a positive integer")
    return stride, max_rows


def eager_dispatch_environment(environment: dict, eager_dispatch) -> dict:
    """``environment`` plus the selected eager-dispatch identity, if any."""
    environment = {k: v for k, v in environment.items() if k not in EAGER_DISPATCH_ENV}
    for name in EAGER_DISPATCH_ENV:
        os.environ.pop(name, None)
    stride, max_rows = eager_dispatch
    if stride:
        selected = dict(zip(EAGER_DISPATCH_ENV, (str(stride), str(max_rows))))
        environment.update(selected)
        os.environ.update(selected)
    return environment


def eager_dispatch_diagnostics(adapter) -> dict:
    trunk = getattr(getattr(adapter, "model", None), "model", None)
    if not getattr(trunk, "eager_dispatch_stride", 0):
        return {}
    from ..runtime.round_levers import counters

    levers = counters()
    return {
        "eager_dispatch": {
            "stride": trunk.eager_dispatch_stride,
            "max_rows": trunk.eager_dispatch_max_rows,
            "forwards": int(levers["eager_dispatch_forwards"]),
            "row_declines": int(levers["eager_dispatch_row_declines"]),
            "async_evals": int(levers["eager_async_evals"]),
        },
        # The serving qualifier reads eager_async_evals here (observed use).
        "round_levers": levers,
    }


def resolve_eos_token_ids(config: dict, tokenizer) -> list[int]:
    """Combine artifact and tokenizer EOS ids without trusting either alone."""
    text = config.get("text_config", config)
    configured = config.get("eos_token_id", text.get("eos_token_id"))
    # A copy: appending to the config's own list mutated the artifact config.
    values = list(configured) if isinstance(configured, list) else [configured]
    values.append(getattr(tokenizer, "eos_token_id", None))
    result = []
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            if value not in result:
                result.append(value)
    if not result:
        raise ValueError("Qwen tokenizer and config declare no EOS token")
    return result


class Qwen3827BAdapter(ExternalDraftAdapterMixin, FlashNextAdapter):
    default_route = "native_mtp"
    # Explicit because Flash-Next deliberately defaults its handoff off.
    default_mtp_ordinary_handoff_max_width = (
        DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH
    )
    # Interior checkpoints ``"auto"``: GPU-qualified on this model, native MTP,
    # zero output differences, gate go (TTFT shared system 7.72 -> 0.29 s, RAG
    # 20.82 -> 0.42 s; qualification/runs/interior-ckpt-20260919/
    # qwen38-27b-shared-rag.json).
    #
    # Copy drafts (single-lane default, batched_max_span 0): GO at B1 on this
    # model, native MTP d2, full pre-registered criterion.  Code 1.204x (t0)
    # and 1.261x (t0.7); prose 0.999x (worst rep 0.962); B4 1.009x; peak
    # memory +/-0.02 GiB (qualification/runs/copy-mtp-20260919/
    # STATUS-rm01-gpu.md, ab-27b-t0-v2.json).  Above the handoff width the
    # cohort runs ordinary and copies are inert.  Not declared on Qwen3.6:
    # its prose dispersion did not clear.
    default_route_execution_policy = {
        "native_mtp": {
            "apc_interior_checkpoints": "auto",
            "self_mtp_copy_draft": {"enabled": True},
        },
    }
    """Dense text adapter using shared chat parsing and modern runtime state."""

    descriptor = QWEN38_27B
    # Candidate external route: Inco's DFlash2 block drafter
    # (incoai/Qwen3.8-27B-DFlash2, block 8, taps 5/19/33/47/61) verified on
    # the hybrid target through ``runtime/hybrid_verify_rows``.  Opt-in via
    # ``--external-draft`` and a pinned policy
    # (qualification/policies/qwen38-27b-dflash2.json); implemented, not
    # qualified.  The default stays self-MTP K=2.
    EXTERNAL_DEFAULT_NUM_DRAFT = 7
    EXTERNAL_ROUTE_TAG = "external-dflash2-qwen38-v1"
    EXTERNAL_PROFILE = "qwen38-27b-apcv2-dflash2"
    # Vendor sampling defaults: Qwen/Qwen3.8-27B model card and the artifact's
    # generation_config.json (see ``adapters/qwen.py``).
    from .qwen import QWEN38_27B_SAMPLING as sampling_defaults
    artifact_inspector = staticmethod(inspect_artifact)
    descriptor_builder = staticmethod(descriptor_for)
    environment_configurator = staticmethod(configure_environment)
    # Dense trunk-MLP paging (runtime/streamed_load.py).  Read from this
    # class's own __dict__: the Qwen3.5 9B and Qwen3.6 subclasses do not
    # inherit it.
    weight_streaming_modes = frozenset({"dense_mlp"})

    # Per-layer eager dispatch stays off on the dense 27B: bit-exact, but
    # neutral end to end (native MTP B1 1.001x, ordinary B1 0.996x, B4
    # 0.995x; qualification/runs/recon-20261001/l7-decode-perf).
    default_eager_dispatch_stride = 0

    def __init__(
        self, model_path: str, *, require_mtp: bool = False, execution_policy=None,
        weight_streaming=None,
    ):
        from ..runtime.streamed_load import require_declared

        stream_request = require_declared(type(self), weight_streaming)
        if execution_policy is not None and not isinstance(execution_policy, dict):
            raise ValueError("execution policy must be a JSON object")
        policy = {} if execution_policy is None else dict(execution_policy)
        self.fused_gdn = fused_gdn_policy(policy)
        policy.pop("fused_gdn", None)
        if stream_request is not None:
            if "draft_model" in policy:
                raise ValueError(
                    "dense weight streaming refuses the external draft route in "
                    "this slice"
                )
            if policy.get("tensorfold_prefill"):
                raise ValueError(
                    "TensorFold prefill repacks MLP projections and cannot run "
                    "with dense weight streaming"
                )
        self.weight_stream = None
        from .flash_next_policy import FlashNextPolicy

        # GDN recurrent-state storage class (runtime/models/gdn_state.py);
        # applies to the target on every route, external draft included.
        gdn_state_dtype = FlashNextPolicy(
            gdn_state_dtype=policy.pop("gdn_state_dtype", "float32")
        ).gdn_state_dtype
        prefill_policy = FlashNextPolicy.from_mapping(
            {
                key: policy[key]
                for key in (
                    "tensorfold_prefill",
                    "tensorfold_prefill_backend",
                    "gdn_prefill_chunk",
                    "gdn_prefill_segment_rows",
                    "gdn_core",
                )
                if key in policy
            }
        )
        self.external_policy = {}
        self.draft_model = None
        draft_record = None
        if "draft_model" in policy:
            if require_mtp:
                raise ValueError("External draft is not native MTP")
            # Revision pins and drafter headers are checked before the
            # target's tensors load; a mismatch fails closed here.
            draft_record = inspect_external_policy(policy, model_path)
            self.external_policy = policy
            policy = {}
        if set(policy) - {
            "num_draft",
            "gdn_core",
            "fp32_head_logits",
            "tensorfold_prefill",
            "tensorfold_prefill_backend",
            "gdn_prefill_chunk",
            "gdn_prefill_segment_rows",
            *EAGER_DISPATCH_POLICY_KEYS,
        }:
            raise ValueError(
                "Qwen3.8 27B execution policy supports only num_draft, gdn_core, "
                "fp32_head_logits, tensorfold_prefill, tensorfold_prefill_backend, "
                "gdn_prefill_chunk, gdn_prefill_segment_rows, "
                "eager_dispatch_stride and eager_dispatch_max_rows"
            )
        eager_dispatch = eager_dispatch_policy(policy, self.default_eager_dispatch_stride)
        # Opt-in: the quantized lm_head stores fp32 logits instead of rounding
        # them to bf16 (runtime/fp32_head.py).  Absent keeps receipts as-is.
        fp32_head = policy.get("fp32_head_logits", False)
        if type(fp32_head) is not bool:
            raise ValueError("fp32_head_logits must be boolean")
        self._num_draft = validate_self_mtp_num_draft(policy.get("num_draft", 2))
        # A/B switch for MLX's native gated_delta_update on 17-256 row prefill
        # chunks (MLX_GDN_CORE).  Absent keeps the pinned "0" profile and its
        # qualification identity; parity on this geometry is unestablished.
        gdn_core = policy.get("gdn_core")
        if gdn_core is not None and type(gdn_core) is not bool:
            raise ValueError("gdn_core must be boolean")
        artifact = self.artifact_inspector(model_path)
        if require_mtp and not artifact["has_mtp"]:
            raise ValueError("requested MTP requires embedded head weights")
        self.identity = artifact["identity"]
        self.descriptor = self.descriptor_builder(has_mtp=artifact["has_mtp"])
        self.environment = self.environment_configurator()
        if gdn_core is not None:
            self.environment = {
                **self.environment, "MLX_GDN_CORE": "1" if gdn_core else "0"
            }
            os.environ["MLX_GDN_CORE"] = self.environment["MLX_GDN_CORE"]
        self.environment = eager_dispatch_environment(self.environment, eager_dispatch)
        if self.fused_gdn:
            # Receipt identity only; no runtime module reads this variable.
            self.environment = {**self.environment, "MLX2_QWEN38_FUSED_GDN": "1"}
        from ..runtime.models.import_env import assert_profile_applied

        # Model modules read GDN/QSDPA selections at import: one imported
        # under another profile would run a route this receipt does not name.
        assert_profile_applied(f"the {type(self).__name__} adapter")
        self.layout = self.descriptor.cache_layout
        self._tables = []
        path = Path(self.identity["path"])
        config = artifact["config"]
        import mlx.core as mx
        import mlx.nn as nn
        from transformers import AutoTokenizer
        from ..runtime.models.qwen38_27b import Model, ModelArgs
        from ..runtime.tokenizer_utils import TokenizerWrapper, BPEStreamingDetokenizer
        from ..runtime.ubc_evict import load_shards_evicting

        # Conversion configs may advertise a head that was stripped from weights.
        config = dict(config)
        config["text_config"] = dict(config.get("text_config", config))
        if not artifact["has_mtp"] or draft_record is not None:
            # The external route never runs the embedded head: do not load it.
            config["text_config"]["mtp_num_hidden_layers"] = 0
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
            weights = self.model.sanitize(load_shards_evicting(files))
            self.norm_convention = getattr(
                getattr(self.model, "language_model", None), "norm_convention", None
            )
            quantize(weights)
            self.model.load_weights(list(weights.items()), strict=True)
        else:
            if self.environment.get("MLX_LM_COMPILED_DECODE", "0") != "0":
                raise ValueError("weight streaming cannot run under compiled decode")
            from ..runtime.streamed_load import load_streamed, trunk_mlp_targets

            try:
                loaded = load_streamed(
                    self.model,
                    path,
                    names,
                    request=stream_request,
                    sanitize=self.model.sanitize,
                    quantize=quantize,
                    records=self.identity["files"],
                    weight_map=artifact["weight_map"],
                    dense_targets=lambda model: trunk_mlp_targets(
                        model, layers_prefix="language_model.model.layers."
                    ),
                )
            except BaseException:
                self.close()
                raise
            weights = loaded.weights
            self.weight_stream = loaded.manager
            self._tables.append(loaded.manager)
            self.norm_convention = getattr(
                getattr(self.model, "language_model", None), "norm_convention", None
            )
        try:
            self._finish_load(
                weights, prefill_policy, fp32_head, path, config,
                AutoTokenizer, TokenizerWrapper, BPEStreamingDetokenizer,
                eager_dispatch, gdn_state_dtype,
            )
        except BaseException:
            if self.weight_stream is not None:
                self.close()
            raise
        if draft_record is not None:
            from .dflash2 import load_drafter

            self.identity = {
                **self.identity,
                "draft_revision": draft_record["draft_revision"],
                "target_revision": draft_record["target_revision"],
            }
            base = self.descriptor_builder(has_mtp=False)
            self._bind_external_drafter(
                draft_record,
                lambda record, target: load_drafter(
                    record, target,
                    runtime_quantization=record["runtime_quantization"],
                ),
                base,
            )
            mx.clear_cache()

    def _finish_load(
        self, weights, prefill_policy, fp32_head, path, config,
        AutoTokenizer, TokenizerWrapper, BPEStreamingDetokenizer, eager_dispatch,
        gdn_state_dtype,
    ):
        """Everything after the weights load: installs, probe, tokenizer.

        Shared by the ordinary and the dense-streamed load; with streaming the
        dtype probe's page-ins are load evidence, and serving counters start
        only at :meth:`begin_serving` below.
        """
        import mlx.core as mx

        self.model.eval()
        from ..runtime.models.qwen38_fused_gdn import configure as configure_fused_gdn

        configure_fused_gdn(self.model, self.fused_gdn)
        mx.eval(self.model.parameters())
        if eager_dispatch[0]:
            self.model.model.set_eager_dispatch(*eager_dispatch)
        self.tensorfold_prefill = None
        if prefill_policy.tensorfold_prefill:
            weights.clear()
            from ..runtime.models.tensorfold_prefill import install as install_prefill

            self.tensorfold_prefill = install_prefill(
                self.model, backend=prefill_policy.tensorfold_prefill_backend
            )
        self.gdn_prefill_scan = None
        if prefill_policy.gdn_prefill_chunk:
            from ..runtime.models.gated_delta import install_prefill_scan

            self.gdn_prefill_scan = install_prefill_scan(
                self.model,
                prefill_policy.gdn_prefill_chunk,
                prefill_policy.gdn_prefill_segment_rows,
            )
        from ..runtime.prefill_plan import execution_identity

        self.prefill_execution_identity = execution_identity(
            self.tensorfold_prefill, self.gdn_prefill_scan
        )
        self.fp32_head = None
        if fp32_head:
            from ..runtime.fp32_head import enable_fp32_head_logits

            self.fp32_head = enable_fp32_head_logits(self.model.language_model)
        weights.clear()
        mx.clear_cache()
        self._record_load_dtype()
        self._select_gdn_state(gdn_state_dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        # transformers' Qwen2Tokenizer drops the declared combining-mark split rule.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        eos = resolve_eos_token_ids(config, tokenizer)
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=eos
        )
        self.max_context = int(config["text_config"]["max_position_embeddings"])
        if getattr(self, "weight_stream", None) is not None:
            self.weight_stream.begin_serving()

    def create_external_batch(self, **kwargs):
        """Candidate DFlash2 draft/verify batch; implemented, not qualified."""
        if getattr(self, "draft_model", None) is None:
            raise ValueError("No external draft model bound")
        from ..runtime.external_speculative import ExternalDraftBatchGenerator

        self._initialize_external_feedback()
        if hasattr(self.draft_model, "last_continuation_selections"):
            kwargs.setdefault("continuation_pool", self.draft_model.policy)

        adaptive = self.external_policy.get("adaptive_verification")
        if adaptive is not None:
            kwargs.setdefault("adaptive_verification", adaptive)
        return ExternalDraftBatchGenerator(
            self.model,
            draft_model=self.draft_model,
            binding=self.identity["fingerprint"],
            num_draft=self._external_num_draft(),
            pairwise_selection=self.external_policy.get("pairwise_selection", "host"),
            # Keep B>1 lanes in lockstep (see ExternalDraftBatchGenerator).
            ready_drain="all",
            **kwargs,
        )

    def profile_name(self, mtp):
        if mtp and Capability.MTP not in self.descriptor.capabilities:
            raise ValueError("requested MTP requires embedded head weights")
        return (
            f"qwen38-27b-apcv2-mtp{getattr(self, '_num_draft', 2)}"
            if mtp
            else "qwen38-27b-apcv2-ordinary"
        )

    def execution_config(self, *, max_lanes, prefill_step):
        if getattr(self, "draft_model", None) is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        config = {
            "persistent": True,
            "num_draft": getattr(self, "_num_draft", 2)
            if Capability.MTP in self.descriptor.capabilities
            else 0,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": True,
            "segment_aware_cohort_size": max_lanes,
        }
        if getattr(self, "fp32_head", None):
            config["fp32_head_logits"] = True
        return config

    def approximate_kv_operations(self):
        """KV quantization for the ordinary route (implemented, unqualified).

        Every full-attention layer allocates a plain ``KVCache`` (supports
        ``to_quantized`` and batched merge) and attends through
        ``scaled_dot_product_attention``, which dispatches quantized SDPA.
        Gated-delta layers hold ``ArraysCache`` recurrent state, which has no
        ``to_quantized`` and stays exact.
        """
        from ..runtime.approximate_kv import standard_kv_quantization_operations

        return standard_kv_quantization_operations(group_size=64)

    def cache_budget(self, *, mtp):
        from .flash_next import gdn_state_bytes
        from .qwen38_memory import Qwen38CacheBudget

        return Qwen38CacheBudget.from_config(
            self.model.args.text_config,
            mtp=mtp,
            recurrent_state_bytes=gdn_state_bytes(self),
        )

    def _select_gdn_state(self, value):
        """Bind the GDN state storage class; fp16 also gets its own APCv2 layout.

        ``float32`` installs nothing and keeps the layout, so default
        receipts and cache identities are unchanged.
        """
        from ..runtime.models.gdn_state import (
            install_state_dtype,
            layout_with_state_dtype,
        )

        self.gdn_state = (
            install_state_dtype(self.model, value) if value != "float32" else None
        )
        self.layout = layout_with_state_dtype(self.layout, value)

    def _record_load_dtype(self):
        """Record the float32-norm cast receipt and the load dtype check."""
        from ..runtime.models.dtype_normalize import check_compute_dtype

        text = getattr(self.model, "language_model", None)
        self.dtype_normalized = getattr(text, "dtype_normalization", None)
        self.load_dtype_check = (
            None
            if text is None
            else check_compute_dtype(text, getattr(text, "compute_dtype", None))
        )

    def dtype_diagnostics(self):
        return {
            "dtype_normalized": getattr(self, "dtype_normalized", None),
            "load_dtype_check": getattr(self, "load_dtype_check", None),
        }

    def diagnostics(self):
        from ..runtime.segmented_self_mtp import segmented_self_mtp_stats

        return {
            "architecture": "dense-hybrid-gdn-gqa",
            "layout": self.layout,
            "mtp_head_present": Capability.MTP in self.descriptor.capabilities,
            "speculation": (
                "external-dflash2-implemented-unqualified"
                if getattr(self, "draft_model", None) is not None
                else "self-mtp"
                if Capability.MTP in self.descriptor.capabilities
                else "ordinary"
            ),
            "segmented_mtp": segmented_self_mtp_stats(),
            "norm_convention": (
                None
                if getattr(self, "norm_convention", None) is None
                else self.norm_convention.summary()
            ),
            **self.dtype_diagnostics(),
            **(
                {
                    "tensorfold_prefill": {
                        **self.tensorfold_prefill,
                        "counters": dict(self.tensorfold_prefill["counters"]),
                    }
                }
                if getattr(self, "tensorfold_prefill", None)
                else {}
            ),
            **(
                {
                    "gdn_prefill_scan": {
                        **self.gdn_prefill_scan,
                        "counters": dict(self.gdn_prefill_scan["counters"]),
                    }
                }
                if getattr(self, "gdn_prefill_scan", None)
                else {}
            ),
            **(
                {"fp32_head_logits": self.fp32_head}
                if getattr(self, "fp32_head", None)
                else {}
            ),
            **eager_dispatch_diagnostics(self),
            **gdn_state_diagnostics(self),
            **(
                {"fused_gdn": self._fused_gdn_diagnostics()}
                if getattr(self, "fused_gdn", False) else {}
            ),
        }

    def execution_numerics_contract(self):
        """Selected target math for APCv2 and external learning identities."""
        import mlx.core as mx
        from mlx import nn

        from ..runtime.models.gdn_state import is_gdn_layer
        from ..runtime.models.qwen38_fused_gdn import GatedDeltaNet

        modules = list(self.model.named_modules())
        layers = [
            module
            for _, module in modules
            if isinstance(module, GatedDeltaNet)
        ]
        enabled = bool(getattr(self, "fused_gdn", False))
        if (enabled and not layers) or any(
            module.fused_gdn_enabled is not enabled for module in layers
        ):
            raise ValueError("selected fused_gdn policy disagrees with live target layers")
        contract = {}
        if enabled:
            contract["fused_gdn"] = {
                "algorithm": "qwen35-served-silu-decode-v1",
                "scope": "initialized-nonspeculating-single-token",
                "verify_prefill": "reference",
            }
        state_receipt = getattr(self, "gdn_state", None)
        state_layers = [module for _, module in modules if is_gdn_layer(module)]
        state_selected = state_receipt is not None
        expected = mx.float16 if state_selected else mx.float32
        if (state_selected and (
                not isinstance(state_receipt, dict)
                or state_receipt.get("state_dtype") != "float16"
                or not state_layers
                or state_receipt.get("layers") != len(state_layers))) or any(
            (getattr(layer, "_gdn_state_dtype", None) or mx.float32) != expected
            for layer in state_layers
        ):
            raise ValueError("selected GDN state dtype disagrees with live target layers")
        if state_selected:
            contract["gdn_state"] = {
                "algorithm": "gdn-state-fp16-v1",
                "storage_dtype": "float16",
                "compute_dtype": "float32",
                "rounding": "per-token-round-to-nearest-even",
            }
        head_receipt = getattr(self, "fp32_head", None)
        if head_receipt is not None:
            head = getattr(getattr(self.model, "language_model", None), "lm_head", None)
            if (not isinstance(head_receipt, dict)
                    or head_receipt.get("enabled") is not True
                    or not isinstance(head, nn.QuantizedLinear)
                    or "bias" in head
                    or head.mode != "affine"
                    or head.scales.dtype != mx.float32
                    or (head.get("biases") is not None and head.biases.dtype != mx.float32)
                    or any(head_receipt.get(key) != getattr(head, key)
                           for key in ("bits", "group_size", "mode"))):
                raise ValueError("selected fp32 head policy disagrees with live target head")
            contract["fp32_head_logits"] = {
                "algorithm": "affine-head-fp32-scales-v1",
                "bits": int(head.bits),
                "group_size": int(head.group_size),
            }
        if not contract:
            return None
        # Retain the existing fused-only schema spelling and identity; selected
        # storage/head precision extends it without changing the default.
        return {"schema": "mlx2.qwen38-fused-gdn-numerics.v1", **contract}

    def _fused_gdn_diagnostics(self):
        from ..runtime.models.qwen38_fused_gdn import stats

        return stats(self.model)
