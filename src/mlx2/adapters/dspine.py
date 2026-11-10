"""DSpine draft artifacts: strict config, tensor schema, inspector and loader.

DSpine (Tencent, arXiv 2609.36173) is a DFlash-format block drafter
(``architectures: ["DFlashDraftModel"]``, ``model_type: qwen3``) with a code
bridge after every draft layer that writes a predicted predecessor feature
into each block row, and a serial "exact sender" refinement of the last write
over the target head's top-K candidates.  Reference implementation:
``Tencent/DSpine`` ``dspine_mlx/`` and ``sglang_dspine/`` (Apache-2.0);
provenance/qwen38-dspine-drafter.json.  Runtime math lives in
``runtime/drafters/dspine.py``.

The artifact declares the same architecture as a plain DFlash head, so it is
recognized by its ``dflash_config.selected_prefix_code_bridge`` block
(:func:`is_dspine_config`), never by a model name.  The format is reachable only
through an explicit ``--external-draft`` policy naming a DSpine artifact
(``external_drafter_formats`` dispatches on the bridge block); nothing
selects it by default.

Only the inference contract the reference serves is accepted: serial decode,
independent writes on every layer, terminal write disabled, causal block,
``serial_table`` second pass redoing the last write only, and one refinement
width ``K`` (a power of two in [2, 64]).  Sliding layers are accepted only in
the reference's native SWA shape (every layer ``sliding_attention``, one int
window).  ``kappa``, the teacher schedule and checkpointing flags are training
only and do not enter the forward.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..runtime.drafters.base_config import DFlashConfig
from .dflash2 import _decode_unique_json

ARCHITECTURES = ("DFlashDraftModel", "DSpineDraftModel")
VERSION_SCALARS = (
    "draft_block_causal_version",
    "last_write_refine_version",
    "selected_prefix_bridge.independent_write_version",
    "selected_prefix_bridge.terminal_disabled_version",
)
_TRAINING_ONLY = frozenset({
    "context_fusion_checkpointing", "decoder_checkpointing",
    "selected_prefix_feature_gt_training", "selected_prefix_feature_teacher_layers",
    "selected_prefix_feature_teacher_schedule",
})
_INFERENCE_KEYS = frozenset({
    "draft_block_causal", "mask_token_id", "target_layer_ids",
    "selected_prefix_code_bridge", "selected_prefix_decode_mode",
    "selected_prefix_feature_layers", "selected_prefix_feature_write_mode",
    "selected_prefix_feature_writes", "selected_prefix_last_write_refine",
    "selected_prefix_terminal",
})


def _require(condition, message):
    if not condition:
        raise ValueError(f"DSpine: {message}")


def is_dspine_config(config) -> bool:
    """True for a DFlash-format artifact carrying the DSpine code bridge."""
    if not isinstance(config, dict):
        return False
    archs = config.get("architectures")
    dflash = config.get("dflash_config")
    return (
        isinstance(archs, list) and len(archs) == 1 and archs[0] in ARCHITECTURES
        and isinstance(dflash, dict) and "selected_prefix_code_bridge" in dflash
    )


@dataclass
class DSpineConfig(DFlashConfig):
    is_causal: bool = True
    code_size: int = 1024
    bridge_rank: int = 512
    bridge_eps: float = 1e-6
    head_topk: int = 16
    feature_layers: list = field(default_factory=list)  # 0-based draft layers

    @classmethod
    def from_dict(cls, raw: dict) -> "DSpineConfig":
        _require(is_dspine_config(raw), "not a DSpine (code-bridge DFlash) config")
        _require(raw.get("model_type") == "qwen3", f"model_type {raw.get('model_type')!r}")
        _require(raw.get("hidden_act", "silu") == "silu", "hidden_act must be silu")
        _require(not raw.get("attention_bias", False), "attention_bias must be false")
        _require(not raw.get("tie_word_embeddings", False), "tie_word_embeddings must be false")
        dflash = raw["dflash_config"]
        unknown = set(dflash) - _INFERENCE_KEYS - _TRAINING_ONLY
        _require(not unknown, f"unknown dflash_config keys {sorted(unknown)}")
        _require(dflash.get("draft_block_causal") is True, "draft_block_causal must be true")
        _require(raw.get("is_causal", True) is True, "is_causal must be true")
        _require(dflash.get("selected_prefix_terminal") is False, "terminal write must be disabled")
        _require(dflash.get("selected_prefix_decode_mode") == "serial", "decode mode must be serial")
        _require(dflash.get("selected_prefix_feature_writes") is True, "feature writes required")
        _require(dflash.get("selected_prefix_feature_write_mode") == "independent",
                 "write mode must be independent")
        bridge = dflash["selected_prefix_code_bridge"]
        _require(isinstance(bridge, dict) and set(bridge) <= {"code_size", "rank", "eps", "kappa"},
                 "bad selected_prefix_code_bridge")
        layers = int(raw["num_hidden_layers"])
        features = dflash.get("selected_prefix_feature_layers")
        _require(isinstance(features, list) and [int(x) - 1 for x in features] == list(range(layers)),
                 "every draft layer must carry a bridge write (1-based feature layers)")
        widths = []
        refine = dflash.get("selected_prefix_last_write_refine")
        if refine is not None:
            _require(isinstance(refine, dict) and set(refine) == {"kind", "version", "head_topk", "weight"},
                     "bad selected_prefix_last_write_refine")
            _require(refine["kind"] == "exact_sender" and int(refine["version"]) == 1
                     and float(refine["weight"]) >= 0, "refine must be exact_sender v1")
            widths.append(int(refine["head_topk"]))
        second = raw.get("code_bridge_second_pass")
        if second is not None:
            _require(isinstance(second, dict) and set(second) <= {"mode", "first_exact", "head_topk"},
                     "bad code_bridge_second_pass")
            _require(second.get("mode") == "serial_table", "second pass must be serial_table")
            _require(int(second.get("first_exact", -1)) == layers - 1,
                     "serial_table redoes the last write only")
            widths.append(int(second["head_topk"]))
        _require(widths and len(set(widths)) == 1, f"refinement widths missing or disagree: {widths}")
        k = widths[0]
        _require(k & (k - 1) == 0 and 2 <= k <= 64, f"head_topk {k} not a power of two in [2,64]")
        types = raw.get("layer_types") or ["full_attention"] * layers
        _require(len(types) == layers, "layer_types length")
        window = raw.get("sliding_window")
        if "sliding_attention" in types:
            _require(types == ["sliding_attention"] * layers and type(window) is int and window >= 2
                     and raw.get("use_sliding_window", True) is True,
                     "sliding layers must all slide with one int window")
        else:
            _require(types == ["full_attention"] * layers and not raw.get("use_sliding_window"),
                     "full-attention drafts declare no window")
        rope = raw.get("rope_parameters") or {}
        _require(rope.get("rope_type", "default") == "default" and raw.get("rope_scaling") is None,
                 "default RoPE only")
        _require(int(raw.get("draft_vocab_size", raw["vocab_size"])) == int(raw["vocab_size"]),
                 "draft_vocab_size != vocab_size")
        flat = {k2: v for k2, v in raw.items() if k2 != "dflash_config"}
        flat.update(
            mask_token_id=int(dflash["mask_token_id"]),
            target_layer_ids=[int(x) for x in dflash["target_layer_ids"]],
            rope_theta=float(raw.get("rope_theta", rope.get("rope_theta"))),
            layer_types=list(types),
            sliding_window=window if "sliding_attention" in types else None,
            code_size=int(bridge["code_size"]), bridge_rank=int(bridge["rank"]),
            bridge_eps=float(bridge["eps"]), head_topk=k,
            feature_layers=list(range(layers)), is_causal=True,
        )
        names = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in flat.items() if key in names})
        _require(0 <= config.mask_token_id < config.vocab_size, "mask token outside vocab")
        _require(all(0 <= t < config.num_target_layers for t in config.target_layer_ids)
                 and len(set(config.target_layer_ids)) == len(config.target_layer_ids),
                 "target_layer_ids out of range or duplicated")
        _require(config.num_attention_heads % config.num_key_value_heads == 0, "GQA geometry")
        _require(config.block_size >= 2, "block_size must be >= 2")
        return config

    from_hf_dict = from_dict


def expected_weight_shapes(args: DSpineConfig) -> dict:
    """Raw checkpoint schema (71 tensors for the Qwen3.8-27B b16 artifact)."""
    hidden, layers = args.hidden_size, args.num_hidden_layers
    heads = args.num_attention_heads * args.head_dim
    kv = args.num_key_value_heads * args.head_dim
    code, rank = args.code_size, args.bridge_rank
    bridge = "selected_prefix_bridge."
    shapes = {
        "fc.weight": [hidden, len(args.target_layer_ids) * hidden],
        "hidden_norm.weight": [hidden],
        "norm.weight": [hidden],
        bridge + "codebook": [args.vocab_size, code],
        bridge + "read_weight": [code, hidden],
        bridge + "early_read_weights": [layers - 1, code, hidden],
        bridge + "feature_write_weights": [layers, hidden, rank],
        bridge + "prev.weight": [rank, code],
        bridge + "gate.weight": [rank, code],
        bridge + "gate.bias": [rank],
        bridge + "predecessor_gate.weight": [rank, code],
        bridge + "out.weight": [code, rank],
    }
    shapes.update({name: [] for name in VERSION_SCALARS})
    for index in range(layers):
        p = f"layers.{index}."
        shapes.update({
            p + "input_layernorm.weight": [hidden],
            p + "post_attention_layernorm.weight": [hidden],
            p + "mlp.gate_proj.weight": [args.intermediate_size, hidden],
            p + "mlp.up_proj.weight": [args.intermediate_size, hidden],
            p + "mlp.down_proj.weight": [hidden, args.intermediate_size],
            p + "self_attn.q_proj.weight": [heads, hidden],
            p + "self_attn.k_proj.weight": [kv, hidden],
            p + "self_attn.v_proj.weight": [kv, hidden],
            p + "self_attn.o_proj.weight": [hidden, heads],
            p + "self_attn.q_norm.weight": [args.head_dim],
            p + "self_attn.k_norm.weight": [args.head_dim],
        })
    return shapes


def inspect_drafter(path, target):
    """Bind config, header schema and payload hashes; no tensor is loaded."""
    from .draft_artifacts import inspect_files

    path = Path(path).expanduser().resolve()
    config_raw = (path / "config.json").read_bytes()
    config = _decode_unique_json(config_raw, "DSpine config.json")
    args = DSpineConfig.from_dict(config)
    target_config = _decode_unique_json((Path(target) / "config.json").read_bytes(),
                                        "target config.json")
    text = target_config.get("text_config", target_config)
    for name in ("hidden_size", "vocab_size"):
        _require(text[name] == getattr(args, name), f"target {name} mismatch")
    _require(text["num_hidden_layers"] == args.num_target_layers, "target layer count mismatch")
    record = inspect_files(
        path, config_raw, expected=expected_weight_shapes(args), dtype=config.get("dtype"),
        label="DSpine", dtype_overrides={name: "I64" for name in VERSION_SCALARS},
    )
    from .dflash2 import _file_sha256_with_binding

    # Streamed (8 MiB chunks): never hold the 4 GB payload in memory.
    record["weight_sha256"] = [
        _file_sha256_with_binding(path / name)[0] for name, _, _ in record["files"]
    ]
    record.update(config=config, args=args,
                  config_sha256=hashlib.sha256(config_raw).hexdigest())
    return record


def content_revision(record):
    """Payload-bound content pin, stable across identical downloads."""
    return hashlib.sha256(json.dumps(
        ["mlx2.dspine-content.v1", record["config"], record["header_sha256"],
         record["weight_sha256"]], sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def legacy_content_revision(record):
    """Metadata-only pin, used only to diagnose a stale policy pin."""
    return hashlib.sha256(json.dumps(
        ["mlx2.dspine-metadata.v1", record["config"], record["header_sha256"]],
        sort_keys=True).encode()).hexdigest()


def load_drafter(record, target_model, *, runtime_quantization=None, draft_kernel=None):
    """Strict load, version/terminal checks, bind to the target, build tables."""
    import mlx.core as mx
    import mlx.nn as nn

    from ..runtime.drafters.dspine import DSpineDraftModel

    root = Path(record["path"])
    model = DSpineDraftModel(record["args"])
    weights = {}
    for name, _, _ in record["files"]:
        weights.update(mx.load(str(root / name)))
    for key in VERSION_SCALARS:
        value = weights.pop(key)
        _require(int(value.item()) == 1, f"unsupported {key}={int(value.item())}")
    _require(not bool(mx.any(weights["selected_prefix_bridge.out.weight"] != 0).item()),
             "terminal-disabled checkpoint carries a nonzero terminal projection")
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    if runtime_quantization is not None:
        from .dflash2 import validate_runtime_quantization

        runtime_quantization = validate_runtime_quantization(runtime_quantization)
        # Backbone projections only; the bridge stays BF16 (reference
        # quantize_draft: the refinement depends on its exact rounding).
        nn.quantize(model, group_size=runtime_quantization["group_size"],
                    bits=runtime_quantization["bits"],
                    class_predicate=lambda name, module: isinstance(module, nn.Linear)
                    and not name.startswith("selected_prefix_bridge"))
    object.__setattr__(model, "runtime_quantization", runtime_quantization)
    model.eval()
    mx.eval(model.parameters())
    if draft_kernel is not None and draft_kernel != "stock":
        if draft_kernel != "lane":
            raise ValueError('DSpine supports draft_kernel "stock" or "lane"')
        from ..runtime.drafters.dflash2 import configure_draft_kernel

        # Backbone projections on the lane matmul (DFlash2's law); the bridge
        # stays on stock kernels: its weights are read directly (packed
        # [G_q;G_p;A], the serial tables), not through the Linear modules.
        receipt = configure_draft_kernel(model, "lane", drafter="dspine")
        object.__setattr__(model, "draft_kernel_receipt", receipt)
        mx.clear_cache()
    return model.bind(target_model)


__all__ = [
    "DSpineConfig", "VERSION_SCALARS", "content_revision", "expected_weight_shapes",
    "inspect_drafter", "is_dspine_config", "legacy_content_revision", "load_drafter",
]
