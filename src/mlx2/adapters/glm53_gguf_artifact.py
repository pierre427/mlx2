"""CPU-only exact checkpoint admission for the REAP50 GLM-5.3 GGUF."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..runtime.gguf.mlx_storage import atomic_json, sha256, stamp
from ..runtime.gguf.reader import GGUFFile

FILENAME = "GLM-5.3-Flash-REAP50-Q3_K_M.gguf"
WEIGHT_SHA256 = "5a865918aea87a362ff8ae419581a0447929bfd68534a44d1125b394c272762d"
WEIGHT_BYTES = 78776759712
MODEL_TYPE = "glm53_gguf"
SEAL = "mlx2-glm53-integrity.json"
REFERENCE = "official-tokenizer.json"


def config_from_metadata(md):
    root = "glm5-next."
    expected = {
        "general.architecture": "glm5-next",
        root + "block_count": 46,
        root + "embedding_length": 4096,
        root + "feed_forward_length": 12288,
        root + "attention.head_count": 64,
        root + "expert_count": 144,
        root + "expert_used_count": 8,
        root + "expert_shared_count": 1,
        root + "expert_feed_forward_length": 2048,
        root + "leading_dense_block_count": 3,
        root + "attention.q_lora_rank": 1536,
        root + "attention.kv_lora_rank": 512,
        root + "attention.key_length_mla": 256,
        root + "attention.value_length_mla": 256,
        root + "kda.head_dim": 128,
        root + "ssm.conv_kernel": 4,
        root + "ssm.gate_lower_bound": -5.0,
        root + "rope.dimension_count": 0,
        root + "expert_gating_func": 2,
        root + "expert_weights_norm": True,
        root + "expert_weights_scale": 2.5,
        root + "swiglu_limit": 10.0,
        root + "attention.indexer.head_count": 32,
        root + "attention.indexer.key_length": 128,
        root + "attention.indexer.top_k": 2048,
        root + "attention.hc.mult": 4,
        root + "attention.hc.sinkhorn_iters": 20,
        root + "nextn_predict_layers": 1,
        root + "vocab_size": 154880,
    }
    for key, value in expected.items():
        if md.get(key) != value:
            raise ValueError(f"GLM-5.3 GGUF topology mismatch: {key}")
    kinds = md.get(root + "attention.head_count_kv")
    wanted = [int((i + 1) % 4 == 0 or i == 45) for i in range(46)]
    if not isinstance(kinds, (list, tuple)) or list(kinds) != wanted:
        raise ValueError("GLM-5.3 requires the exact per-block KDA/MLA schedule")
    eps = md.get(root + "attention.layer_norm_rms_epsilon")
    if not isinstance(eps, (int, float)) or not math.isclose(eps, 1e-5, rel_tol=1e-6):
        raise ValueError("GLM-5.3 RMS epsilon mismatch")
    return {
        "model_type": MODEL_TYPE,
        "architectures": ["Glm53GGUFForCausalLM"],
        "hidden_size": 4096,
        "vocab_size": 154880,
        "num_hidden_layers": 45,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "n_routed_experts": 144,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "max_position_embeddings": 65536,
        "rms_norm_eps": float(eps),
        "layer_types": [
            "linear_attention" if k == 0 else "deepseek_sparse_attention"
            for k in kinds[:45]
        ],
        "weight_file": FILENAME,
        "weight_sha256": WEIGHT_SHA256,
        "qualification": "unqualified",
        "gguf_a_convention": "positive_exp",
        "embedded_mtp": "present-not-selected",
        "vision": "not-present",
    }


def required_shapes(config):
    result = {
        "token_embd.weight": (4096, 154880),
        "output.weight": (4096, 154880),
        "output_norm.weight": (4096,),
    }
    for i, kind in enumerate(config["layer_types"]):
        p = f"blk.{i}."
        for half in ("attn", "ffn"):
            result.update(
                {
                    p + f"hc_{half}_fn": (16384, 24),
                    p + f"hc_{half}_base": (24,),
                    p + f"hc_{half}_scale": (3,),
                }
            )
        result.update({p + "attn_norm.weight": (4096,), p + "ffn_norm.weight": (4096,)})
        if kind == "linear_attention":
            for proj in ("q", "k", "v"):
                result[p + f"attn_{proj}.weight"] = (4096, 8192)
                result[p + f"ssm_conv1d_{proj}.weight"] = (4, 1, 8192)
            result.update(
                {
                    p + "attn_output.weight": (8192, 4096),
                    p + "ssm_a": (64,),
                    p + "ssm_dt.bias": (8192,),
                    p + "ssm_beta.weight": (4096, 64),
                    p + "ssm_norm.weight": (128,),
                }
            )
            for letter in ("f", "g"):
                result[p + f"ssm_{letter}_a.weight"] = (4096, 128)
                result[p + f"ssm_{letter}_b.weight"] = (128, 8192)
        else:
            result.update(
                {
                    p + "attn_q_a.weight": (4096, 1536),
                    p + "attn_q_a_norm.weight": (1536,),
                    p + "attn_q_b.weight": (1536, 16384),
                    p + "attn_kv_a_mqa.weight": (4096, 512),
                    p + "attn_kv_a_norm.weight": (512,),
                    p + "attn_k_b.weight": (256, 512, 64),
                    p + "attn_v_b.weight": (512, 256, 64),
                    p + "attn_output.weight": (16384, 4096),
                    p + "indexer.attn_k.weight": (4096, 128),
                    p + "indexer.attn_q_b.weight": (1536, 4096),
                    p + "indexer.k_norm.weight": (128,),
                    p + "indexer.k_norm.bias": (128,),
                    p + "indexer.proj.weight": (4096, 32),
                    p + "indexer.kpool_ape": (128, 4),
                    p + "indexer.kpool_gate": (4096, 128),
                }
            )
        if i < 3:
            result.update(
                {
                    p + "ffn_gate.weight": (4096, 12288),
                    p + "ffn_up.weight": (4096, 12288),
                    p + "ffn_down.weight": (12288, 4096),
                }
            )
        else:
            result.update(
                {p + "ffn_gate_inp.weight": (4096, 144), p + "exp_probs_b.bias": (144,)}
            )
            for proj in ("gate", "up", "down"):
                shape = (2048, 4096) if proj == "down" else (4096, 2048)
                result[p + f"ffn_{proj}_shexp.weight"] = shape
                result[p + f"ffn_{proj}_exps.weight"] = (*shape, 144)
    return result


def validate_inventory(config, tensors):
    shapes = required_shapes(config)
    for name, shape in shapes.items():
        t = tensors.get(name)
        if t is None or t.shape != shape:
            raise ValueError("GLM-5.3 tensor missing or wrong shape: " + name)
        if t.ggml_type not in (0, 2, 11, 12, 13, 14):
            raise ValueError("GLM-5.3 tensor format not implemented: " + name)
    for i in range(3, 45):
        if (
            tensors[f"blk.{i}.ffn_gate_exps.weight"].ggml_type
            != tensors[f"blk.{i}.ffn_up_exps.weight"].ggml_type
        ):
            raise ValueError("GLM-5.3 fused experts require matching gate/up formats")
    return shapes


def aligned_tokenizer(reference, md):
    """Retain official pre-tokenization; bind all GGUF IDs including padded rows."""
    import copy

    data = copy.deepcopy(reference)
    tokens = md["tokenizer.ggml.tokens"]
    merges = md["tokenizer.ggml.merges"]
    model = data["model"]
    actual = [" ".join(m) if isinstance(m, list) else m for m in model["merges"]]
    if (
        model.get("type") != "BPE"
        or actual != list(merges)
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("GLM-5.3 GGUF/reference tokenizer BPE mismatch")
    known = dict(model["vocab"])
    known.update({t["content"]: t["id"] for t in data["added_tokens"]})
    if any(i >= len(tokens) or tokens[i] != s for s, i in known.items()):
        raise ValueError("GLM-5.3 GGUF/reference tokenizer ID mismatch")
    known_ids = set(known.values())
    if any(s != f"[PAD{i}]" for i, s in enumerate(tokens) if i not in known_ids):
        raise ValueError("unexpected GLM-5.3 vocabulary extension")
    model["vocab"] = {s: i for i, s in enumerate(tokens)}
    return data


def prepare_artifact(root):
    import numpy as np

    root = Path(root).resolve()
    weight = root / FILENAME
    before = stamp(weight)
    if weight.is_symlink() or before["size"] != WEIGHT_BYTES:
        raise ValueError("GLM-5.3 source must be the exact regular GGUF")
    digest = sha256(weight)
    if digest != WEIGHT_SHA256 or stamp(weight) != before:
        raise ValueError("GLM-5.3 GGUF SHA256/stat mismatch")
    with GGUFFile(weight) as file:
        config = config_from_metadata(file.metadata)
        shapes = validate_inventory(config, file.tensors)
        aligned_tokenizer(json.loads((root / REFERENCE).read_text()), file.metadata)
        for i, kind in enumerate(config["layer_types"]):
            if kind == "linear_attention":
                view = file.packed_view(f"blk.{i}.ssm_a")
                try:
                    a = np.frombuffer(view, dtype="<f4")
                    valid = bool(np.all(np.isfinite(a) & (a > 0)))
                    del a
                finally:
                    view.release()
                    file.release_mapping()
                if not valid:
                    raise ValueError("checkpoint does not use positive exp(A_log)")
        payload = sum(file.tensors[n].nbytes for n in shapes)
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("refusing a conflicting GLM-5.3 config")
    if not config_path.exists():
        atomic_json(config_path, config)
    seal = {
        "schema": "mlx2.glm53-gguf.v1",
        "weight_sha256": digest,
        "weight_stat": before,
        "config_sha256": sha256(config_path),
        "reference_sha256": sha256(root / REFERENCE),
        "ordinary_payload_bytes": payload,
        "qualified": False,
    }
    if stamp(weight) != before:
        raise ValueError("GLM-5.3 GGUF changed during preparation")
    atomic_json(root / SEAL, seal)
    return seal


def inspect_artifact(root):
    root = Path(root).resolve()
    seal = json.loads((root / SEAL).read_text())
    if (
        seal.get("schema") != "mlx2.glm53-gguf.v1"
        or seal.get("weight_sha256") != WEIGHT_SHA256
        or stamp(root / FILENAME) != seal.get("weight_stat")
        or sha256(root / "config.json") != seal.get("config_sha256")
        or sha256(root / REFERENCE) != seal.get("reference_sha256")
    ):
        raise ValueError("GLM-5.3 artifact differs from CPU integrity preparation")
    with GGUFFile(root / FILENAME) as file:
        config = config_from_metadata(file.metadata)
        validate_inventory(config, file.tensors)
        metadata = file.metadata
    if json.loads((root / "config.json").read_text()) != config:
        raise ValueError("GLM-5.3 config differs from GGUF")
    return {
        "config": config,
        "metadata": metadata,
        "weight_stat": seal["weight_stat"],
        "identity": {
            "path": str(root),
            "fingerprint": WEIGHT_SHA256,
            "weight_sha256": WEIGHT_SHA256,
        },
        "ordinary_payload_bytes": seal["ordinary_payload_bytes"],
        "has_mtp": True,
        "qualified": False,
    }
