"""CPU-safe artifact inspection and the dense Qwen3.8 27B serving adapter.

Import/inspect performs no tensor imports or model loads. Instantiating the
adapter loads weights and is reserved for a separately authorized GPU window.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .external_draft_policy import ExternalDraftAdapterMixin
from .flash_next import FlashNextAdapter, gdn_state_diagnostics
from .mtp_depth_cap import validate_self_mtp_num_draft
from ..process_env import (
    PROCESS_NUMERICS,
    clear_inherited_profile,
    require_process_numerics,
)

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


def _source_binding(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _read_bytes_with_binding(path: Path):
    before = _source_binding(path)
    content = path.read_bytes()
    after = _source_binding(path)
    if before != after:
        raise ValueError(f"target source changed while inspecting: {path.name}")
    return content, before


def _file_sha256_with_binding(path: Path):
    before = _source_binding(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    after = _source_binding(path)
    if before != after:
        raise ValueError(f"target source changed while hashing: {path.name}")
    return digest.hexdigest(), before


def _verify_target_source_bindings(root: Path, records) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("DFlash2 target has no inspected source bindings")
    for record in records:
        if (
            not isinstance(record, (list, tuple))
            or len(record) != 6
            or not isinstance(record[0], str)
        ):
            raise ValueError("DFlash2 target has an invalid source binding")
        try:
            current = _source_binding(root / record[0])
        except OSError as exc:
            raise ValueError(
                f"DFlash2 target source is unavailable: {record[0]}"
            ) from exc
        if current != tuple(record[1:]):
            raise ValueError(f"DFlash2 target source binding changed: {record[0]}")


def _load_target_weights(files, *, external_draft: bool, sanitize):
    """Materialize target shards, omitting an unreachable embedded MTP head.

    External DFlash2 owns proposal generation and cannot execute the target's
    embedded head. Prune those tensors while they are still lazy so expanded
    GGUF conversions do not read or retain them. Artifact inspection and the
    payload pin still cover every indexed shard before this optimization runs.
    """
    from ..runtime.ubc_evict import load_shards_evicting

    options = {}
    if external_draft:
        options = {
            "keep_lazy": lambda name: name.startswith(
                ("language_model.mtp.", "model.language_model.mtp.", "mtp.")
            ),
            "prune_lazy": True,
        }
    return sanitize(load_shards_evicting(files, **options))


def _target_revision_inputs(model_path: str | Path):
    path = Path(model_path).expanduser().resolve()
    config_raw, config_binding = _read_bytes_with_binding(path / "config.json")
    index_raw, index_binding = _read_bytes_with_binding(
        path / "model.safetensors.index.json"
    )
    try:
        mapping = json.loads(index_raw)["weight_map"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("invalid target weight index") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("target weight index must name at least one tensor")
    values = list(mapping.values())
    if any(not isinstance(name, str) for name in values):
        raise ValueError("target weight index contains an invalid shard path")
    names = sorted(set(values))
    if any(
        not name.endswith(".safetensors")
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        or not (path / name).is_file()
        for name in names
    ):
        raise ValueError("target weight index contains an invalid shard path")
    source_bindings = [
        ("config.json", *config_binding),
        ("model.safetensors.index.json", *index_binding),
    ]
    return path, config_raw, index_raw, names, source_bindings


def _legacy_content_revision(model_path: str | Path) -> str:
    """The former metadata-only pin, retained only to diagnose stale policy."""
    _, config_raw, index_raw, _, _ = _target_revision_inputs(model_path)
    digest = hashlib.sha256()
    for name, raw in (
        ("config.json", config_raw),
        ("model.safetensors.index.json", index_raw),
    ):
        digest.update(name.encode())
        digest.update(raw)
    return digest.hexdigest()


def _inspect_target_content(model_path: str | Path) -> dict:
    path, config_raw, index_raw, names, source_bindings = _target_revision_inputs(
        model_path
    )
    weights = []
    for name in names:
        digest, binding = _file_sha256_with_binding(path / name)
        weights.append([name, binding[2], digest])
        source_bindings.append((name, *binding))
    _verify_target_source_bindings(path, source_bindings)
    record = {
        "schema": "mlx2.qwen38-target-content.v2",
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "weights": weights,
    }
    revision = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "path": str(path),
        "revision": revision,
        "source_bindings": source_bindings,
        **record,
    }


def content_revision(model_path: str | Path) -> str:
    """Payload-bound target pin, stable across byte-identical downloads."""
    return _inspect_target_content(model_path)["revision"]


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
        "continuation_strategy",
        "lilicorr_feedback",
        "draft_revision",
        "target_revision",
        "draft_quantization",
        "tensorfold_prefill",
        "tensorfold_prefill_backend",
        "varlen_dense_mlp",
        "external_varlen_prefill",
        "external_prefill_coalesce_ms",
        "external_prefill_coalesce_min_tokens",
        "gdn_prefill_chunk",
        "gdn_prefill_segment_rows",
        "batch_size_route",
        "tree_node_budget_by_lanes",
        "tensorfold_cohort_limit",
        "minimum_draft_proposals",
        "exact_verification",
    }
)
# Target-wide keys the adapter consumes before an external policy is split
# off; they are valid beside an external draft but not part of its receipt.
TARGET_POLICY_KEYS = frozenset({"fused_gdn", "fused_gdn_prefill", "gdn_state_dtype"})
_TREE_BATCH_ROUTES = {
    "tree15_b1_chain_b2plus_v1": 1,
    "tree15_b1_b4_chain_b5plus_v1": 4,
}


def _normalized_adaptive_verification(value, depth):
    """Return the effective JSON settings, not a caller's shorthand."""
    if value is None or value is False:
        return None
    from ..runtime.acceptance_estimator import AdaptiveVerificationPolicy

    policy = AdaptiveVerificationPolicy.from_value(value, depth)
    if policy is None:
        return None
    settings = asdict(policy)
    # These fields accept integers, but their execution law is floating point.
    settings["draft_cost"] = float(policy.draft_cost)
    settings["min_gain"] = float(policy.min_gain)
    # Detach tuples (including nested cost tables) into their receipt form.
    return json.loads(json.dumps(settings, sort_keys=True, allow_nan=False))


def _tree_node_budgets(value, width):
    if value is None:
        return {lane_width: 15 for lane_width in range(1, width + 1)}
    if not isinstance(value, dict):
        raise ValueError("tree_node_budget_by_lanes must be a mapping")
    normalized = {}
    for key, nodes in value.items():
        try:
            lane_width = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("tree node budget widths must be integers") from exc
        if type(nodes) is not int or not 3 <= nodes <= 15:
            raise ValueError("tree node budgets must be integers from 3 to 15")
        normalized[lane_width] = nodes
    if set(normalized) != set(range(1, width + 1)):
        raise ValueError("tree node budgets must declare every enabled tree width")
    ordered = [normalized[lane_width] for lane_width in range(1, width + 1)]
    if any(left < right for left, right in zip(ordered, ordered[1:])):
        raise ValueError("tree node budgets must not increase as lanes fill")
    return normalized


def inspect_external_policy(
    policy: dict,
    model_path: str | Path,
    *,
    allow_continuation_strategy: bool = False,
) -> dict:
    """Static drafter inspection and payload-bound revision check; no tensor loads."""
    from .dflash2 import _legacy_content_revision as draft_legacy_revision
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
    strategy = policy.get("continuation_strategy")
    if strategy is not None:
        if not allow_continuation_strategy:
            raise ValueError(
                "continuation_strategy requires an adapter-declared exact cache geometry"
            )
        if strategy != "longest_first_exact_prefix":
            raise ValueError("unsupported continuation_strategy")
        if "continuation_pool" not in policy:
            raise ValueError("continuation_strategy requires continuation_pool")
    if "external_varlen_prefill" in policy and type(
        policy["external_varlen_prefill"]
    ) is not bool:
        raise ValueError("external_varlen_prefill must be boolean")
    from ..runtime.models.varlen_dense_mlp import VarlenDenseMLPPolicy

    varlen_dense_mlp = VarlenDenseMLPPolicy.from_value(
        policy.get("varlen_dense_mlp", False)
    )
    if policy.get("external_varlen_prefill") and not varlen_dense_mlp.enabled:
        raise ValueError(
            "external_varlen_prefill requires varlen_dense_mlp selection"
        )
    coalesce_ms = policy.get("external_prefill_coalesce_ms", 0)
    if type(coalesce_ms) is not int or not 0 <= coalesce_ms <= 1000:
        raise ValueError(
            "external_prefill_coalesce_ms must be an integer from 0 to 1000"
        )
    if coalesce_ms and not policy.get("external_varlen_prefill"):
        raise ValueError(
            "external_prefill_coalesce_ms requires external_varlen_prefill"
        )
    coalesce_min_tokens = policy.get("external_prefill_coalesce_min_tokens", 1)
    if type(coalesce_min_tokens) is not int or coalesce_min_tokens < 1:
        raise ValueError(
            "external_prefill_coalesce_min_tokens must be a positive integer"
        )
    batch_route = policy.get("batch_size_route")
    if batch_route is None and os.environ.get("MLX2_DFLASH_TOPOLOGY") == "tree15":
        raise ValueError("tree15 requires an explicit batch_size_route policy")
    exact_verification = policy.get("exact_verification", "token")
    if exact_verification not in ("token", "block"):
        raise ValueError("exact_verification must be 'token' or 'block'")
    if exact_verification == "block" and (
        batch_route is not None
        or any(key in policy for key in ("adaptive_verification", "continuation_pool"))
    ):
        # Block verification decides over one linear chain; the pinned
        # pair's default tree route is left with "batch_size_route": null.
        raise ValueError(
            "block verification requires the linear chain route without tree "
            "batch_size_route, adaptive_verification or continuation_pool"
        )
    if batch_route is not None:
        if type(batch_route) is not str or batch_route not in _TREE_BATCH_ROUTES:
            raise ValueError("unsupported Qwen3.8 external batch_size_route")
        # An explicit ``"proposal_composition": false`` opts out; it is not
        # a chain-only policy.
        if any(policy.get(key) not in (None, False) for key in (
            "adaptive_verification", "proposal_composition", "continuation_pool",
        )):
            raise ValueError("tree15 bounded route conflicts with chain-only proposal policy")
        if any(os.environ.get(name) is not None for name in (
            "MLX2_DFLASH_TOPOLOGY", "MLX2_QWEN_TARGET_EXECUTION",
            "MLX2_TENSORFOLD_COHORT_LIMIT",
        )):
            raise ValueError("tree15 bounded route conflicts with explicit topology override")
        from .qwen38_tensorfold_source import validate_source

        validate_source()
        _tree_node_budgets(
            policy.get("tree_node_budget_by_lanes"),
            _TREE_BATCH_ROUTES[batch_route],
        )
        cohort_limit = policy.get("tensorfold_cohort_limit")
        if cohort_limit is not None and (
            type(cohort_limit) is not int
            or not 1 <= cohort_limit <= _TREE_BATCH_ROUTES[batch_route]
        ):
            raise ValueError(
                "tensorfold_cohort_limit must be an integer from 1 to the "
                "selected batch route width"
            )
    elif "tree_node_budget_by_lanes" in policy:
        raise ValueError("tree node budgets require a batch_size_route")
    elif "tensorfold_cohort_limit" in policy:
        raise ValueError("tensorfold_cohort_limit requires a batch_size_route")
    if policy["target_revision"] == _legacy_content_revision(model_path):
        raise ValueError(
            "DFlash2 target revision mismatch: policy uses a legacy "
            "metadata-only pin; regenerate a payload-bound policy"
        )
    target_content = _inspect_target_content(model_path)
    target_revision = target_content["revision"]
    if target_revision != policy["target_revision"]:
        raise ValueError(
            "DFlash2 target revision mismatch: policy pins "
            f"{policy['target_revision'][:12]}, artifact is {target_revision[:12]}"
        )
    record = inspect_drafter(policy["draft_model"], model_path)
    draft_revision = draft_content_revision(record)
    if draft_revision != policy["draft_revision"]:
        if policy["draft_revision"] == draft_legacy_revision(record):
            raise ValueError(
                "DFlash2 draft revision mismatch: policy uses a legacy "
                "metadata-only pin; regenerate a payload-bound policy"
            )
        raise ValueError(
            "DFlash2 draft revision mismatch: policy pins "
            f"{policy['draft_revision'][:12]}, artifact is {draft_revision[:12]}"
        )
    args = record["args"]
    count = policy.get("num_draft", Qwen3827BAdapter.EXTERNAL_DEFAULT_NUM_DRAFT)
    if type(count) is not int or not 1 <= count < args.block_size:
        raise ValueError("num_draft must be a positive integer below the draft block size")
    floor = policy.get("minimum_draft_proposals", min(3, count))
    if type(floor) is not int or not min(2, count) <= floor <= count:
        raise ValueError(
            "minimum_draft_proposals must be an integer from min(2, num_draft) "
            "to num_draft"
        )
    adaptive = policy.get("adaptive_verification")
    if adaptive is not None:
        _normalized_adaptive_verification(adaptive, count)
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
        "target_path": target_content["path"],
        "target_source_bindings": target_content["source_bindings"],
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
    clear_inherited_profile(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_"))
    os.environ.update(profile)
    return profile


def fused_gdn_policy(policy: dict, default: bool = False) -> bool:
    """27B decode switch in the existing execution policy; false kills it.

    ``default`` is the adapter's own preference (``default_fused_gdn``),
    used when the policy does not name the key.
    """
    enabled = policy.get("fused_gdn", default)
    if type(enabled) is not bool:
        raise ValueError("fused_gdn must be boolean")
    return enabled


def fused_gdn_prefill_policy(policy: dict, default: bool = False) -> bool:
    """27B fused GDN prefill switch (qwen38_fused_gdn); false kills it.

    Independent of ``fused_gdn`` (decode/verify).  ``default`` is the
    adapter's own preference (``default_fused_gdn_prefill``).
    """
    enabled = policy.get("fused_gdn_prefill", default)
    if type(enabled) is not bool:
        raise ValueError("fused_gdn_prefill must be boolean")
    return enabled


def _validate_tree_gdn_state_dtype(external_policy: dict, value: str) -> None:
    """The owned topology kernel currently preserves exact fp32 recurrence."""

    if external_policy.get("batch_size_route") is not None and value != "float32":
        raise ValueError(
            "TensorFold tree GDN requires float32 recurrent state; "
            "gdn_state_dtype=float16 is unsupported"
        )


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
    # Adapter-owned rather than inherited: this family selects an eight-token
    # copy ceiling, hence a nine-row target verify/rollback window.
    # Native self-MTP only. The external TensorFold tree independently owns a
    # 15-proposal/16-row target verification geometry.
    max_exact_self_mtp_verification_rows = 9
    max_exact_self_mtp_rollback_rows = 9
    default_route = "native_mtp"
    # Explicit because Qwen3.8 retains its independently measured threshold
    # instead of inheriting Flash-Next's width-three default.
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
    #
    # Decode-first publication in "order" mode on both routes since
    # 2026-10-02 (Pierre): decode-to-client lag max 670-765 -> 57-101 ms,
    # TTFT, gaps and lane tok/s unchanged (options-sweep-27b-20261002, item
    # 3).  MLX2_DECODE_FIRST=0 is the kill switch; an explicit "decode_first"
    # in the execution policy (false included) wins.
    default_route_execution_policy = {
        "ordinary": {
            "decode_first": {"enabled": True, "shared_prefill_budget": False},
        },
        "native_mtp": {
            "apc_interior_checkpoints": "auto",
            "self_mtp_copy_draft": {"enabled": True},
            "decode_first": {"enabled": True, "shared_prefill_budget": False},
        },
        # Artifact-bound external defaults for the exact target/DFlash2 pair
        # declared below.  A different pair receives no tree defaults.  This
        # selects the measured topology while keeping the route candidate and
        # explicitly selected by --external-draft.
        "external_draft": {
            "pairwise_selection": "host",
            "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
            "tree_node_budget_by_lanes": {"1": 15, "2": 7, "3": 4, "4": 3},
            "external_varlen_prefill": False,
            "varlen_dense_mlp": False,
            "draft_quantization": {"bits": 4, "group_size": 64},
        },
    }
    default_external_route_binding = {
        "target_revision": "e59471c5c6fa8c6819b81cb5957bcab10736020db8bacb47fbb9089813fa93f8",
        "draft_revision": "34ec93d71399f3dd6db9646194f1ad4db345715c61258d72318325e0d8095c90",
    }
    """Dense text adapter using shared chat parsing and modern runtime state."""

    descriptor = QWEN38_27B
    packed_prefill_dense_mlp_semantics = {
        "schema": "mlx2.dense-glu-semantics.v1",
        "gate_projection": "gate_proj",
        "up_projection": "up_proj",
        "down_projection": "down_proj",
        "activation": "silu",
        "combination": "activated_gate_times_up",
    }

    def native_cohort_backend(self, kind):
        from .native_hybrid import HybridNativeCohort
        return HybridNativeCohort(kind)

    def create_native_paged_hybrid_b2(self, requests, *, profile,
                                       permit_candidate=False, cancelled=lambda: False):
        """Explicit default-off adapter capability; ordinary route remains available."""
        from ..runtime.qwen35_paged_graph_factory import create_shared_hybrid_graph_pack
        return create_shared_hybrid_graph_pack(self, requests, profile=profile,
            permit_candidate=permit_candidate, cancelled=cancelled)
    def create_native_packed_prefill_b2(self, requests, *, profile, live_identity,
                                         permit_candidate=False, cancelled=lambda: False):
        """Explicit source-bound short/long research capability; real-row math stays adapter-owned."""
        from ..runtime.hybrid_packed_prefill import create_cold_packed_hybrid
        language=getattr(self.model,"language_model",self.model);args=language.args
        if (args.num_hidden_layers,args.num_attention_heads,args.num_key_value_heads,args.head_dim)!=(64,24,4,256):
            raise ValueError("packed-prefill serving adapter geometry unavailable")
        return create_cold_packed_hybrid(self,requests,profile=profile,live_identity=live_identity,
            permit_candidate=permit_candidate,cancelled=cancelled)

    def native_cohort_memory_admission(self,requests,*,profile_path,manifest_path,mlx_wheel_path,permit_candidate=False):
        """Explicit adapter-owned shared cost; scheduler does not infer model math."""
        from ..runtime.hybrid_packed_prefill_n import estimate_cold_cohort_memory_n
        return estimate_cold_cohort_memory_n(self,requests,profile_path=profile_path,manifest_path=manifest_path,mlx_wheel_path=mlx_wheel_path,permit_candidate=permit_candidate)

    def create_native_packed_prefill_n(self,requests,*,profile,live_identity,source_input_ids,permit_candidate=False,cancelled=lambda:False,phase_boundary=None):
        """Distinct source-bound N1..20 private native candidate, default off."""
        from ..runtime.hybrid_packed_prefill_n import create_cold_packed_hybrid_n
        language=getattr(self.model,"language_model",self.model);args=language.args
        if (args.num_hidden_layers,args.num_attention_heads,args.num_key_value_heads,args.head_dim)!=(64,24,4,256):raise ValueError("nativeN20 adapter geometry unavailable")
        return create_cold_packed_hybrid_n(self,requests,profile=profile,live_identity=live_identity,source_input_ids=source_input_ids,permit_candidate=permit_candidate,cancelled=cancelled,phase_boundary=phase_boundary)

    # Candidate external route: Inco's DFlash2 block drafter
    # (incoai/Qwen3.8-27B-DFlash2, block 8, taps 5/19/33/47/61) verified on
    # the hybrid target through ``runtime/hybrid_verify_rows``.  Opt-in via
    # ``--external-draft`` and a pinned policy
    # (qualification/policies/qwen38-27b-dflash2.json); implemented, not
    # qualified.  The native default is self-MTP K=3; K=2 remains an explicit
    # historical comparator.
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
    # Fused GDN decode (qwen38_fused_gdn) on every route since 2026-10-02
    # (Pierre): bit-identical everywhere (264/264 lanes), ordinary B1-B16
    # +1.5..+3.2%, MTP neutral (qualification/runs/options-sweep-27b-20261002).
    # The 2026-10-06 corrected bounded B1 multi-token port is independently
    # validation-gated; unsupported shapes retain the reference path.
    # {"fused_gdn": false} in the execution policy is the kill switch.  Read
    # from this class's own __dict__: the Qwen3.5 9B and Qwen3.6 subclasses
    # carry no measurement and keep it off.
    default_fused_gdn = True
    # Fused GDN prefill prework + swish norm-gate (omlx #3903 kernels in their
    # Qwen3.5-numerics variant, provenance/qwen38-fused-gdn-prefill.json).
    # Opt-in through {"fused_gdn_prefill": true} until its GPU gate and A/B
    # say otherwise.  Own __dict__ only, like default_fused_gdn.
    default_fused_gdn_prefill = False
    fused_gdn_architecture = "qwen38"
    # Self-MTP draft depth on the native-MTP route when the policy names no
    # num_draft: 3 since 2026-10-02 (Pierre): in-process mtp:2 +30.7%, mtp:3
    # +14.0%, mtp:4 +9.0%; served B2 +15%, B4 +11% (interior off), B1 within
    # noise; B1 tokens identical (options-sweep-27b-20261002).  Own __dict__
    # only, like default_fused_gdn: subclasses keep 2.
    default_num_draft = 3

    def __init__(
        self, model_path: str, *, require_mtp: bool = False, execution_policy=None,
        weight_streaming=None,
    ):
        from .process_globals import guarded_construction

        # The profile edits os.environ before tensors load; a failed load
        # (import-order conflict, weights, tokenizer, drafter) must not leave
        # it behind.  Qwen3.6 27B and dense Qwen3.5 inherit this constructor.
        guarded_construction(
            self,
            lambda: self._init_qwen38(
                model_path,
                require_mtp=require_mtp,
                execution_policy=execution_policy,
                weight_streaming=weight_streaming,
            ),
        )

    def _init_qwen38(
        self, model_path: str, *, require_mtp: bool = False, execution_policy=None,
        weight_streaming=None,
    ):
        from ..runtime.streamed_load import require_declared

        stream_request = require_declared(type(self), weight_streaming)
        if execution_policy is not None and not isinstance(execution_policy, dict):
            raise ValueError("execution policy must be a JSON object")
        policy = {} if execution_policy is None else dict(execution_policy)
        self.fused_gdn = fused_gdn_policy(
            policy, vars(type(self)).get("default_fused_gdn", False)
        )
        policy.pop("fused_gdn", None)
        self.fused_gdn_prefill = fused_gdn_prefill_policy(
            policy, vars(type(self)).get("default_fused_gdn_prefill", False)
        )
        policy.pop("fused_gdn_prefill", None)
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
                    "invariant_prefill",
                )
                if key in policy
            }
        )
        from ..runtime.models.varlen_dense_mlp import VarlenDenseMLPPolicy

        # Target-side arithmetic remains adapter-owned when an external
        # drafter is selected.  Keep the setting in the source-bound external
        # policy receipt, but parse it before the external-only fields are
        # separated from the target's construction policy.
        varlen_dense_mlp_policy = VarlenDenseMLPPolicy.from_value(
            policy.get("varlen_dense_mlp", False)
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
        else:
            policy.pop("varlen_dense_mlp", None)
        _validate_tree_gdn_state_dtype(self.external_policy, gdn_state_dtype)
        if set(policy) - {
            "num_draft",
            "gdn_core",
            "fp32_head_logits",
            "tensorfold_prefill",
            "tensorfold_prefill_backend",
            "gdn_prefill_chunk",
            "gdn_prefill_segment_rows",
            "invariant_prefill",
            *EAGER_DISPATCH_POLICY_KEYS,
        }:
            raise ValueError(
                "Qwen3.8 27B execution policy supports only num_draft, gdn_core, "
                "fp32_head_logits, tensorfold_prefill, tensorfold_prefill_backend, "
                "gdn_prefill_chunk, gdn_prefill_segment_rows, invariant_prefill, "
                "varlen_dense_mlp, "
                "eager_dispatch_stride and eager_dispatch_max_rows"
            )
        eager_dispatch = eager_dispatch_policy(policy, self.default_eager_dispatch_stride)
        # Opt-in: the quantized lm_head stores fp32 logits instead of rounding
        # them to bf16 (runtime/fp32_head.py).  Absent keeps receipts as-is.
        fp32_head = policy.get("fp32_head_logits", False)
        if type(fp32_head) is not bool:
            raise ValueError("fp32_head_logits must be boolean")
        self._num_draft = validate_self_mtp_num_draft(
            policy.get("num_draft", vars(type(self)).get("default_num_draft", 2))
        )
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
        if artifact.get("identity_evidence") is not None:
            # The inspector says which evidence admitted the family (Qwen3.6:
            # config revision or chat template); keep it on the descriptor.
            self.descriptor = replace(
                self.descriptor,
                metadata={
                    **self.descriptor.metadata,
                    "identity_evidence": artifact["identity_evidence"],
                },
            )
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
        if self.fused_gdn_prefill:
            # Receipt identity only; no runtime module reads this variable.
            self.environment = {
                **self.environment, "MLX2_QWEN38_FUSED_GDN_PREFILL": "1"
            }
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

        if draft_record is not None:
            _verify_target_source_bindings(
                path, draft_record["target_source_bindings"]
            )
        if stream_request is None:
            weights = _load_target_weights(
                files,
                external_draft=draft_record is not None,
                sanitize=self.model.sanitize,
            )
            self.norm_convention = getattr(
                getattr(self.model, "language_model", None), "norm_convention", None
            )
            quantize(weights)
            self.model.load_weights(list(weights.items()), strict=True)
            if draft_record is not None:
                _verify_target_source_bindings(
                    path, draft_record["target_source_bindings"]
                )
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
                eager_dispatch, gdn_state_dtype, varlen_dense_mlp_policy,
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
        gdn_state_dtype, varlen_dense_mlp_policy,
    ):
        """Everything after the weights load: installs, probe, tokenizer.

        Shared by the ordinary and the dense-streamed load; with streaming the
        dtype probe's page-ins are load evidence, and serving counters start
        only at :meth:`begin_serving` below.
        """
        import mlx.core as mx

        self.model.eval()
        from ..runtime.models.varlen_dense_mlp import (
            install as install_varlen_dense_mlp,
        )

        self.varlen_dense_mlp = install_varlen_dense_mlp(
            self.model, varlen_dense_mlp_policy
        )
        from ..runtime.models.qwen38_fused_gdn import configure as configure_fused_gdn

        configure_fused_gdn(
            self.model, self.fused_gdn, architecture=self.fused_gdn_architecture,
            prefill=self.fused_gdn_prefill,
        )
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
        self.invariant_prefill = None
        if prefill_policy.invariant_prefill:
            from ..runtime.models.invariant_prefill import install as install_invariant

            handle = install_invariant(self.model.language_model.model)
            if not handle.installed:
                raise ValueError(f"invariant_prefill refused: {handle.refusal}")
            self.invariant_prefill = handle
        from ..runtime.models.varlen_dense_mlp import identity as varlen_identity
        from ..runtime.prefill_plan import (
            EXTERNAL_VARLEN_PREFILL_IDENTITY,
            execution_identity,
        )

        external_varlen_prefill = None
        if self.external_policy.get("external_varlen_prefill"):
            if self.varlen_dense_mlp is None:
                raise ValueError(
                    "external_varlen_prefill requires an installed varlen dense MLP"
                )
            external_varlen_prefill = EXTERNAL_VARLEN_PREFILL_IDENTITY

        self.prefill_execution_identity = execution_identity(
            self.tensorfold_prefill,
            self.gdn_prefill_scan,
            None if self.invariant_prefill is None else self.invariant_prefill.identity(),
            varlen_identity(self.varlen_dense_mlp),
            external_varlen_prefill=external_varlen_prefill,
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
        self._external_execution_started = True
        if hasattr(self.draft_model, "last_continuation_selections"):
            kwargs.setdefault("continuation_pool", self.draft_model.policy)
            strategy = getattr(self, "continuation_verification_strategy", None)
            if callable(strategy):
                selected_strategy = strategy()
                if selected_strategy is not None:
                    kwargs.setdefault(
                        "continuation_verification_strategy", selected_strategy
                    )

        adaptive = self.external_policy.get("adaptive_verification")
        if adaptive is not None:
            kwargs.setdefault("adaptive_verification", adaptive)
        if self.external_policy.get("exact_verification", "token") != "token":
            kwargs.setdefault(
                "exact_verification", self.external_policy["exact_verification"]
            )
        route = self.external_policy.get("batch_size_route")
        tree_width = _TREE_BATCH_ROUTES.get(route) if type(route) is str else None
        if route is not None and tree_width is None:
            raise ValueError("unsupported Qwen3.8 external batch_size_route")
        if tree_width is not None and (
            kwargs.get("dynamic_singleton_tree", True) is not True
            or kwargs.get("dynamic_tree_max_width", tree_width) != tree_width
        ):
            raise ValueError("tree15 bounded route cannot be overridden at batch creation")
        if tree_width is None and (
            kwargs.get("dynamic_singleton_tree") is True
            or kwargs.get("dynamic_tree_max_width", 1) != 1
            or os.environ.get("MLX2_DFLASH_TOPOLOGY") == "tree15"
        ):
            raise ValueError("tree15 requires an explicit batch_size_route policy")
        kwargs.setdefault("dynamic_singleton_tree", tree_width is not None)
        if tree_width is not None:
            kwargs.setdefault("dynamic_tree_max_width", tree_width)
            kwargs.setdefault(
                "tensorfold_cohort_limit",
                self.external_policy.get("tensorfold_cohort_limit"),
            )
            kwargs.setdefault(
                "tree_node_budget_by_lanes",
                _tree_node_budgets(
                    self.external_policy.get("tree_node_budget_by_lanes"),
                    tree_width,
                ),
            )
        return ExternalDraftBatchGenerator(
            self.model,
            draft_model=self.draft_model,
            binding=self.identity["fingerprint"],
            num_draft=self._external_num_draft(),
            minimum_draft_proposals=self.external_policy.get(
                "minimum_draft_proposals", min(3, self._external_num_draft())
            ),
            pairwise_selection=self.external_policy.get("pairwise_selection", "host"),
            # Keep B>1 lanes in lockstep (see ExternalDraftBatchGenerator).
            ready_drain="all",
            external_varlen_prefill=bool(
                self.external_policy.get("external_varlen_prefill", False)
            ),
            # Formation is an ingress concern: concurrent tokenization has
            # finished and no physical executor step has started yet.
            external_prefill_coalesce_ms=0,
            external_prefill_coalesce_min_tokens=self.external_policy.get(
                "external_prefill_coalesce_min_tokens", 1
            ),
            **kwargs,
        )

    def lane_policy_defaults(self):
        """Declare the row-stable projection geometry for packed varlen/tree."""

        # Subclasses without an external-draft route (Qwen3.5 122B) never set
        # external_policy; the serving lane hook must not crash on them.
        policy = getattr(self, "external_policy", None) or {}
        if not (
            policy.get("external_varlen_prefill")
            and policy.get("batch_size_route")
        ):
            return None
        return {"max_rows": 128, "chunk_above_max": True}

    @staticmethod
    def int8_prefill_supported():
        # Dense MLP and (with "all") attention / GDN input and output
        # projections.  Heads, the MTP layer, the tiny GDN a/b projections
        # (ineligible shapes) and any vision tower stay stock.  Declared so the
        # 8-bit checkpoint can run the in-place Q8 W8A8 prefill (omlx #4350
        # port); default off, approximate, qualification-gated like any
        # int8 prefill route.  Dense subclasses share the module layout.
        return ("mlp", "all")

    def make_recurrent_depth_caches(self, passes: int):
        """Allocate one independent Qwen cache stack per recurrent pass."""
        if (
            isinstance(passes, bool)
            or not isinstance(passes, int)
            or not 1 <= passes <= 8
        ):
            raise ValueError("recurrent-depth passes must be in 1..8")
        return tuple(self.model.make_cache() for _ in range(passes))

    def recurrent_depth_hidden(
        self, inputs, *, cache, input_embeddings=None, **model_kwargs
    ):
        """Expose the adapter-owned embedding-to-final-hidden Qwen seam."""
        return self.model.model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            **model_kwargs,
        )

    def recurrent_depth_logits(self, hidden):
        """Project an adapter-owned final hidden state through the LM head."""
        return self.model.logits(hidden)

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
            config = self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
            if self.external_policy.get("pairwise_selection", "host") == "batched":
                config["pairwise_selection"] = "batched"
            config["minimum_draft_proposals"] = self.external_policy.get(
                "minimum_draft_proposals", min(3, self._external_num_draft())
            )
            if self.external_policy.get("external_varlen_prefill", False):
                from ..runtime.prefill_plan import EXTERNAL_VARLEN_PREFILL_IDENTITY

                config["external_varlen_prefill"] = {
                    "enabled": True,
                    **EXTERNAL_VARLEN_PREFILL_IDENTITY,
                }
            coalesce_ms = self.external_policy.get(
                "external_prefill_coalesce_ms", 0
            )
            if coalesce_ms:
                config["ingress_cohort"] = {
                    "enabled": True,
                    "mechanism": "external_varlen_prefill",
                    "maximum_wait_ms": coalesce_ms,
                    "minimum_prompt_tokens": self.external_policy.get(
                        "external_prefill_coalesce_min_tokens", 1
                    ),
                    "target_lanes": max_lanes,
                }
            adaptive = _normalized_adaptive_verification(
                self.external_policy.get("adaptive_verification"),
                self._external_num_draft(),
            )
            if adaptive is not None:
                config["adaptive_verification"] = adaptive
            strategy = getattr(self, "continuation_verification_strategy", None)
            if callable(strategy):
                selected_strategy = strategy()
                if selected_strategy is not None:
                    config["continuation_verification_strategy"] = (
                        selected_strategy.as_dict()
                        if hasattr(selected_strategy, "as_dict")
                        else selected_strategy
                    )
            if self.external_policy.get("batch_size_route"):
                config["batch_size_route"] = self.external_policy["batch_size_route"]
                config["tensorfold_cohort_limit"] = self.external_policy.get(
                    "tensorfold_cohort_limit",
                    _TREE_BATCH_ROUTES[self.external_policy["batch_size_route"]],
                )
                config["tree_node_budget_by_lanes"] = _tree_node_budgets(
                    self.external_policy.get("tree_node_budget_by_lanes"),
                    _TREE_BATCH_ROUTES[self.external_policy["batch_size_route"]],
                )
            return config
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

    def _external_execution_numerics(self):
        """Import-free external target laws used by direct and served users."""
        contract = {}
        if self.external_policy.get("external_varlen_prefill"):
            from ..runtime.prefill_plan import EXTERNAL_VARLEN_PREFILL_IDENTITY

            if getattr(self, "varlen_dense_mlp", None) is None:
                raise ValueError(
                    "selected external varlen prefill is not installed on the target"
                )
            contract["external_varlen_prefill"] = dict(
                EXTERNAL_VARLEN_PREFILL_IDENTITY
            )
        adaptive = _normalized_adaptive_verification(
            self.external_policy.get("adaptive_verification"),
            self._external_num_draft(),
        )
        if adaptive is not None:
            contract["external_adaptive_verification"] = {
                "algorithm": "exact-chain-adaptive-target-verify-width-v1",
                "policy": adaptive,
            }
        if self.external_policy.get("exact_verification", "token") == "block":
            # Same target law, another RNG schedule: seeded outputs differ.
            contract["external_exact_verification"] = {
                "algorithm": "block-verification-sun-2024-alg2-v1",
            }
        return contract

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
        from ..runtime.models.varlen_dense_mlp import status as varlen_dense_mlp_status
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
                {"invariant_prefill": self.invariant_prefill.status()}
                if getattr(self, "invariant_prefill", None) is not None
                else {}
            ),
            **(
                {"varlen_dense_mlp": varlen_dense_mlp_status(self.varlen_dense_mlp)}
                if getattr(self, "varlen_dense_mlp", None) is not None
                else {}
            ),
            **(
                {"fused_gdn": self._fused_gdn_diagnostics()}
                if getattr(self, "fused_gdn", False)
                or getattr(self, "fused_gdn_prefill", False) else {}
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
        prefill = bool(getattr(self, "fused_gdn_prefill", False))
        if (prefill and not layers) or any(
            module.fused_gdn_prefill_enabled is not prefill for module in layers
        ):
            raise ValueError(
                "selected fused_gdn_prefill policy disagrees with live target layers"
            )
        contract = self._external_execution_numerics()
        if enabled:
            architecture = getattr(self, "fused_gdn_architecture", "qwen38")
            contract["fused_gdn"] = {
                "algorithm": f"{architecture}-corrected-served-silu-v2",
                "architecture": architecture,
                "scope": "initialized-single-token-or-b1-unmasked-width-2-to-17",
                "verify_prefill": "fused-when-admitted-reference-otherwise",
                "speculative_rollback": "exact-snapshots",
            }
        if prefill:
            contract["fused_gdn_prefill"] = {
                "algorithm": "omlx3903-prework-norm-gate-qwen35-numerics-v1",
                "architecture": getattr(self, "fused_gdn_architecture", "qwen38"),
                "scope": "b1-unmasked-non-speculative-rows-at-least-64",
                "recurrence": "reference-gated-delta-update",
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
        if self.external_policy.get("batch_size_route"):
            from ..runtime.qwen38_tensorfold import EXPECTED_REVISION

            contract["external_batch_size_route"] = {
                "algorithm": self.external_policy["batch_size_route"].replace("_", "-"),
                "tensorfold_revision": EXPECTED_REVISION,
            }
        if not contract:
            return None
        # Retain the existing fused-only schema spelling and identity; selected
        # storage/head precision extends it without changing the default.
        return {"schema": "mlx2.qwen38-fused-gdn-numerics.v1", **contract}

    def _fused_gdn_diagnostics(self):
        from ..runtime.models.qwen38_fused_gdn import stats
        from ..runtime.models.qwen38_tree_gdn import stats as tree_stats

        # TensorFold tree verify (external DFlash2) runs every GDN layer
        # through the owned tree kernel and never reaches the step kernel
        # above, so its launches are reported beside the step counters.  The
        # tree counters are process-wide (one served model per process).
        tree = tree_stats()
        return {
            "architecture": self.fused_gdn_architecture,
            **stats(self.model),
            "tree_calls": int(tree["tree_calls"]),
            "tree_rows": int(tree["tree_rows"]),
        }
