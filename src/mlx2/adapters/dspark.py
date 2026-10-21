"""Model-independent DSpark configuration, tensor schema and DFlash-format loader.

``DSparkConfig`` covers both DSpark proposal-attention variants (see
``runtime/drafters/dspark.py``).  The speculators/RedHat Super VL binding
stays in ``nemotron35_dspark.py``; this module also binds DFlash-format
Qwen3-backbone DSpark heads (``architectures: ["Qwen3DSparkModel"]``, e.g.
DimInfer/Qwen3.8-27B-Dspark-v1) to any target whose hidden width, vocabulary
and layer count match.  Provenance: provenance/qwen38-dspark-drafter.json.

Block contract of a DFlash-format head (llama.cpp ``draft-dspark``; no
``dflash.sample_from_anchor``/``dflash.attention.causal`` keys means anchor
sampling and non-causal attention): the block is the anchor plus
``block_size - 1`` mask tokens, every row proposes (row 0 predicts the token
after the anchor), so ``block_size`` proposals.  ``target_layer_ids`` are
post-layer indices, which is mlx2's tap convention.  Embedding and LM head
are the target's.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..runtime.drafters.base_config import DFlashConfig
from .xpress import XPressConfig
from .xpress import expected_weight_shapes as xpress_shapes

QWEN_DSPARK_ARCHITECTURE = "Qwen3DSparkModel"
QWEN_TAP_CONVENTION = "dflash-target-layer-ids-are-post-layer-v1"
QWEN_BLOCK_CONTRACT = "dflash-format-dspark-anchor-sampled-noncausal-v1"
PROPOSAL_ATTENTION = ("causal", "bidirectional")


@dataclass
class DSparkConfig(DFlashConfig):
    markov_rank: int = 256
    sample_from_anchor: bool = True
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    # ``causal``: sliding prefix window, causal proposal triangle
    # (speculators).  ``bidirectional``: stock DFlash full attention.
    proposal_attention: str = "causal"

    def __post_init__(self):
        shared = {key: getattr(self, key) for key in DFlashConfig.__dataclass_fields__}
        shared.update(layer_types=["full_attention"] * self.num_hidden_layers, sliding_window=None)
        XPressConfig(**shared, xpress_num_passes=1)
        if (self.sample_from_anchor is not True or type(self.markov_rank) is not int
                or self.markov_rank <= 0 or self.enable_confidence_head is not True
                or self.confidence_head_with_markov is not True
                or self.proposal_attention not in PROPOSAL_ATTENTION):
            raise ValueError("Unsupported DSpark Markov/confidence/attention geometry")
        if self.proposal_attention == "causal":
            valid = (self.layer_types == ["sliding_attention"] * self.num_hidden_layers
                     and type(self.sliding_window) is int and self.sliding_window >= 2)
        else:
            valid = (self.layer_types == ["full_attention"] * self.num_hidden_layers
                     and self.sliding_window is None)
        if not valid:
            raise ValueError("Unsupported DSpark Markov/confidence/attention geometry")


def expected_weight_shapes(args):
    shared = {key: getattr(args, key) for key in DFlashConfig.__dataclass_fields__}
    shared.update(layer_types=["full_attention"] * args.num_hidden_layers, sliding_window=None)
    shapes = {key: shape for key, shape in xpress_shapes(XPressConfig(**shared, xpress_num_passes=1)).items()
              if not key.startswith("xpress_head.")}
    shapes.update({"markov_head.markov_w1.weight": [args.vocab_size, args.markov_rank],
                   "markov_head.markov_w2.weight": [args.vocab_size, args.markov_rank],
                   "confidence_head.proj.weight": [1, args.hidden_size + args.markov_rank],
                   "confidence_head.proj.bias": [1]})
    return shapes


def qwen_dspark_config(config):
    """Strict DSparkConfig of a DFlash-format Qwen3-backbone DSpark head."""
    rope = config.get("rope_parameters") or {}
    layers = config.get("num_hidden_layers")
    factors = {config.get("partial_rotary_factor", 1.0), rope.get("partial_rotary_factor", 1.0)}
    if (config.get("architectures") != [QWEN_DSPARK_ARCHITECTURE]
            or config.get("markov_head_type") != "vanilla"
            or config.get("enable_confidence_head") is not True
            or config.get("confidence_head_with_markov") is not True
            # Absent keys are the llama.cpp defaults this binding implements.
            or config.get("sample_from_anchor", True) is not True
            or config.get("is_causal", False) is not False
            or config.get("sliding_window") is not None
            or config.get("layer_types") != ["full_attention"] * (layers or 0)
            or config.get("hidden_act") != "silu" or config.get("attention_bias") is not False
            or config.get("attention_dropout", 0.0) != 0.0
            or rope.get("rope_type", "default") != "default" or factors != {1.0}
            or "rope_scaling" in config and config["rope_scaling"] is not None
            or config.get("quantization") or config.get("quantization_config")
            or type(config.get("block_size")) is not int or config["block_size"] < 1):
        raise ValueError("Unsupported DSpark source contract")
    theta = rope.get("rope_theta", config.get("rope_theta"))
    if type(theta) not in (int, float) or not theta > 0:
        raise ValueError("Unsupported DSpark source contract")
    fields = {key: config[key] for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "rms_norm_eps", "vocab_size",
        "max_position_embeddings", "mask_token_id", "num_target_layers", "markov_rank",
    )}
    fields.update(
        # mlx2 counts the anchor in block_size: rows = block_size - 1.
        block_size=config["block_size"] + 1,
        target_layer_ids=list(config["target_layer_ids"]),
        rope_theta=float(theta), layer_types=["full_attention"] * layers,
        sliding_window=None, tie_word_embeddings=False,
        enable_confidence_head=True, confidence_head_with_markov=True,
        sample_from_anchor=True, proposal_attention="bidirectional",
    )
    return DSparkConfig(**fields)


def _target_geometry(target):
    from .dflash2 import _decode_unique_json

    config = _decode_unique_json((Path(target) / "config.json").read_bytes(), "target config.json")
    return config.get("text_config", config)


def inspect_drafter(path, target):
    """Bind a DFlash-format DSpark artifact's bytes and schema; no tensor loads."""
    from .dflash2 import (
        _decode_unique_json,
        _file_sha256_with_binding,
        _read_bytes_with_binding,
        _read_safetensors_header,
        _source_binding,
        _verify_source_bindings,
    )

    path = Path(path).expanduser().resolve()
    config_raw, config_binding = _read_bytes_with_binding(path / "config.json")
    config = _decode_unique_json(config_raw, "draft config.json")
    args = qwen_dspark_config(config)
    text = _target_geometry(target)
    for name in ("hidden_size", "vocab_size"):
        if text[name] != getattr(args, name):
            raise ValueError(f"DSpark target {name} mismatch")
    if text["num_hidden_layers"] != args.num_target_layers:
        raise ValueError("DSpark target layer count mismatch")
    if not 0 <= args.mask_token_id < args.vocab_size:
        raise ValueError("DSpark mask token outside the target vocabulary")
    name = "model.safetensors"
    # HF snapshots symlink this lexical name into the blob store, as the other
    # external-drafter inspectors accept; the weight bytes are hashed below.
    file = path / name
    if not file.is_file() or (path / "model.safetensors.index.json").exists():
        raise ValueError("DSpark expects one model.safetensors file")
    binding = _source_binding(file)
    raw_header, header, _ = _read_safetensors_header(file)
    expected = expected_weight_shapes(args)
    actual = {key: entry for key, entry in header.items() if key != "__metadata__"}
    if (set(actual) != set(expected)
            or any(not isinstance(entry, dict) or entry.get("shape") != expected[key]
                   or entry.get("dtype") != "BF16" for key, entry in actual.items())):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"DSpark tensor schema differs (missing={missing}, extra={extra})")
    weight_hash, _ = _file_sha256_with_binding(file, binding)
    source_bindings = [("config.json", *config_binding), (name, *binding)]
    _verify_source_bindings(path, source_bindings, label="DSpark artifact")
    header_hash = hashlib.sha256(raw_header).hexdigest()
    digest = hashlib.sha256(config_raw)
    digest.update(header_hash.encode())
    digest.update(weight_hash.encode())
    digest.update((QWEN_TAP_CONVENTION + QWEN_BLOCK_CONTRACT).encode())
    return {
        "path": str(path), "fingerprint": digest.hexdigest(),
        "files": [(name, binding[2], binding[3])],
        "header_sha256": [header_hash], "weight_sha256": [weight_hash],
        "config_sha256": hashlib.sha256(config_raw).hexdigest(), "index_sha256": None,
        "source_bindings": source_bindings, "config": config, "args": args,
        "tap_convention": QWEN_TAP_CONVENTION, "block_contract": QWEN_BLOCK_CONTRACT,
    }


def content_revision(record):
    """Payload-bound DSpark content pin, stable across identical downloads."""
    weights = record.get("weight_sha256")
    if (not isinstance(weights, list) or len(weights) != 1
            or any(not isinstance(value, str) or len(value) != 64 for value in weights)):
        raise ValueError("DSpark content revision requires the full weight payload hash")
    return hashlib.sha256(json.dumps(
        ["mlx2.dspark-content.v1", record["config"], record["header_sha256"], weights,
         QWEN_TAP_CONVENTION, QWEN_BLOCK_CONTRACT],
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def legacy_content_revision(record):
    """DSpark never had a metadata-only pin; nothing can match it."""
    return


def apply_runtime_quantization(model, value):
    """Affine weight-only quantization of the backbone projections (mlx2).

    The Markov and confidence heads stay BF16: the tree lattice reads their
    raw rows, and their proposal law is part of the trained contract.
    """
    from mlx import nn

    if value is None:
        model.runtime_weight_quantization = None
        return None
    modules = []

    def predicate(name, module):
        accepted = (isinstance(module, nn.Linear) and not name.startswith(("markov_head", "confidence_head"))
                    and int(module.weight.shape[-1]) % value["group_size"] == 0)
        if accepted:
            modules.append(name)
        return accepted

    nn.quantize(model, group_size=value["group_size"], bits=value["bits"], class_predicate=predicate)
    model.runtime_weight_quantization = {
        "schema": "mlx2.dspark-draft-runtime-quantization.v1", "mode": "affine",
        "bits": value["bits"], "group_size": value["group_size"],
        "module_count": len(modules), "heads": "bf16",
        "source_artifact_unchanged": True, "qualified": False,
    }
    return model.runtime_weight_quantization


def load_drafter(record, target_model, *, runtime_quantization=None):
    """Load an inspected DFlash-format DSpark head and bind the target's embedding/head."""
    from .dflash2 import _verify_source_bindings, validate_runtime_quantization

    root = Path(record["path"])
    runtime_quantization = validate_runtime_quantization(runtime_quantization)

    def verify_sources():
        _verify_source_bindings(root, record.get("source_bindings"), label="DSpark artifact")
        if "target_source_bindings" in record:
            _verify_source_bindings(Path(record["target_path"]), record["target_source_bindings"],
                                    label="DSpark target artifact")

    verify_sources()
    if qwen_dspark_config(record["config"]) != record["args"]:
        raise ValueError("DSpark config changed since inspection")
    import mlx.core as mx

    from ..runtime.drafters.dspark import dspark_draft_model

    model = dspark_draft_model(record["args"])
    weights = mx.load(str(root / record["files"][0][0]))
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    apply_runtime_quantization(model, runtime_quantization)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    mx.clear_cache()
    bound = model.bind(target_model)
    verify_sources()
    return bound


__all__ = [
    "QWEN_DSPARK_ARCHITECTURE",
    "DSparkConfig",
    "apply_runtime_quantization",
    "content_revision",
    "expected_weight_shapes",
    "inspect_drafter",
    "legacy_content_revision",
    "load_drafter",
    "qwen_dspark_config",
]
