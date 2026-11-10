# SPDX-License-Identifier: MIT
"""GPU-free admission of the exact Maya-XS V4.1 artifact; no V4 impersonation."""

import hashlib
import json
import math
import os
from pathlib import Path

from ..runtime.gguf.reader import GGUFFile
from ..runtime.models.deepseek_v41_config import DeepSeekV41Config
from ..runtime.models.deepseek_v41_engram import EngramHasher
from .deepseek_v4_gguf_artifact import _digest, _regular_asset, _stamp
from .deepseek_v4_gguf_tokenizer import align_tokenizer
from ..runtime.immutable_weight_pages import same_stamp

MODEL_TYPE = "deepseek41_gguf"
FILENAME = "DeepSeek-V4.1-Flash-1017-Maya-XS-Abliterated.gguf"
WEIGHT_SHA256 = "f414995b3797f77009634fac995f50834e70393771f9423fca984c892503b42a"
WEIGHT_BYTES = 365713686528
TOKENIZER_SHA256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
INTEGRITY_FILE = "mlx2-v41-integrity.json"
REFERENCE_FILE = "tokenizer.reference.json"
_SCHEMA = "mlx2.deepseek-v41-gguf-integrity.v1"
_CONFIG = {
    "model_type": MODEL_TYPE,
    "mlx2_weight_format": "gguf-packed-v1",
    "gguf_file": FILENAME,
    "weight_sha256": WEIGHT_SHA256,
    "tokenizer_reference": REFERENCE_FILE,
}


def tokenizer_metadata(metadata):
    # The GGUF retains exact vocabulary/merges, but chat encoding is Python in
    # the official release. This marker identifies that explicit encoder.
    return {
        **metadata,
        "tokenizer.ggml.padding_token_id": 2,
        "tokenizer.chat_template": "mlx2-deepseek-v41-encoding-v1",
    }


def validate_header(metadata, tensors):
    c = DeepSeekV41Config.from_gguf(metadata)
    expected = DeepSeekV41Config()
    for name in expected.__dataclass_fields__:
        if name in ("rope_scaling",):
            continue
        a, b = getattr(c, name), getattr(expected, name)
        if (isinstance(b, float) and not math.isclose(a, b, rel_tol=1e-6)) or (
            not isinstance(b, float) and a != b
        ):
            raise ValueError("unsupported V4.1 checkpoint topology: " + name)
    for name, value in (
        ("factor", 16.0),
        ("beta_fast", 32.0),
        ("beta_slow", 1.0),
        ("original_max_position_embeddings", 65536.0),
    ):
        if c.rope_scaling.get(name) != value:
            raise ValueError("unsupported V4.1 YaRN geometry: " + name)
    if (
        metadata.get("deepseek41.hidden_act") != "silu"
        or metadata.get("deepseek41.topk_method") != "noaux_tc"
    ):
        raise ValueError("unsupported V4.1 activation or routing method")
    if metadata.get("deepseek41.engram.encoding") != "e4m3_e8m0_32_row264":
        raise ValueError("unsupported V4.1 Engram encoding")
    EngramHasher(metadata)
    d, h, rank, o, m, e = (
        c.hidden_size,
        c.head_dim,
        c.q_lora_rank,
        c.o_groups,
        c.moe_intermediate_size,
        c.n_routed_experts,
    )
    shapes = {
        "token_embd.weight": ((d, c.vocab_size), 8),
        "output.weight": ((d, c.vocab_size), 8),
        "output_norm.weight": ((d,), 0),
    }
    for i in range(c.num_hidden_layers):
        p = f"blk.{i}."

        def put(name, shape, kind=0, prefix=p):
            shapes[prefix + name] = (shape, kind)

        for kind in ("attn", "ffn"):
            put(f"hc_{kind}_fn.weight", (c.hc_mult * d, (2 + c.hc_mult) * c.hc_mult), 1)
            put(f"hc_{kind}_base.weight", ((2 + c.hc_mult) * c.hc_mult,))
            put(f"hc_{kind}_scale.weight", (3,))
            put(f"{kind}_norm.weight", (d,))
        for name, shape in (
            ("attn_q_a", (d, rank)),
            ("attn_q_b", (rank, c.num_attention_heads * h)),
            ("attn_kv", (d, h)),
            ("attn_output_a", (c.num_attention_heads * h // o, o * c.o_lora_rank)),
            ("attn_output_b", (o * c.o_lora_rank, d)),
        ):
            put(name + ".weight", shape, 8)
        for name, width in (
            ("attn_q_a_norm", rank),
            ("attn_kv_a_norm", h),
            ("attn_sinks", c.num_attention_heads),
        ):
            put(name + ".weight", (width,))
        put("ffn_gate_inp.weight", (d, e))
        put("exp_probs_b.bias", (e,))
        put("exp_probs_b_vl.bias", (e,))
        for kind in ("gate", "up", "down"):
            shape = (m, d) if kind == "down" else (d, m)
            put(f"ffn_{kind}_shexp.weight", shape, 8)
            put(f"ffn_{kind}_exps.weight", (*shape, e), 19)
        if i in c.kv_source_layer_ids:
            put("attn_compressor_kv.weight", (d, h), 8)
            put("attn_compressor_norm.weight", (h,))
            put("indexer.attn_k.weight", (h, c.index_head_dim), 1)
            put("indexer.k_norm.weight", (c.index_head_dim,))
            if c.compress_ratios[i] > 1:
                put("attn_compressor_gate.weight", (d, h), 8)
        if i in c.index_source_layer_ids:
            put(
                "indexer.attn_q_b.weight", (rank, c.index_n_heads * c.index_head_dim), 8
            )
            put("indexer.proj.weight", (d, c.index_n_heads), 1)
        if i in c.engram_layer_ids:
            put("engram_q_norm.weight", (d, c.hc_mult))
            put("engram_k_norm.weight", (d, c.hc_mult))
            put("engram_kv.weight", (24 * 256, (c.hc_mult + 1) * d), 8)
            put(
                "engram_embd.weight",
                (264, c.engram_num_embeddings[c.engram_layer_ids.index(i)]),
                24,
            )
    if set(tensors) != set(shapes):
        raise ValueError("V4.1 exact tensor inventory changed")
    for name, (shape, kind) in shapes.items():
        t = tensors[name]
        if t.shape != shape or t.ggml_type != kind:
            raise ValueError("V4.1 tensor geometry/format mismatch: " + name)
    return c


def _same_seal(existing, seal):
    # The persisted weight_stat may carry a pre-reboot device id (same_stamp).
    return (isinstance(existing, dict) and same_stamp(existing.get("weight_stat"), seal["weight_stat"])
            and {k: v for k, v in existing.items() if k != "weight_stat"}
            == {k: v for k, v in seal.items() if k != "weight_stat"})


def prepare_artifact(model_path, reference_tokenizer):
    """Full payload SHA check precedes publishing stat-bound serving assets."""
    path = Path(model_path).expanduser().resolve()
    weight = _regular_asset(path / FILENAME)
    before = _stamp(weight)
    if before["size"] != WEIGHT_BYTES:
        raise ValueError("complete pinned V4.1 weight is absent")
    reference_path = _regular_asset(Path(reference_tokenizer))
    if _digest(reference_path) != TOKENIZER_SHA256:
        raise ValueError("V4.1 tokenizer does not match pinned official source")
    reference = json.loads(reference_path.read_text())
    with GGUFFile(weight) as file:
        validate_header(file.metadata, file.tensors)
        _, binding = align_tokenizer(reference, tokenizer_metadata(file.metadata))
    if _digest(weight) != WEIGHT_SHA256 or _stamp(weight) != before:
        raise ValueError("V4.1 payload changed or differs from pinned SHA256")
    payloads = {
        "config.json": json.dumps(_CONFIG, indent=2) + "\n",
        REFERENCE_FILE: reference_path.read_text(),
    }
    for name, data in payloads.items():
        dest = path / name
        if dest.is_symlink() or (dest.exists() and dest.read_text() != data):
            raise ValueError("refusing to overwrite different V4.1 asset: " + name)
    for name, data in payloads.items():
        dest = path / name
        if not dest.exists():
            tmp = path / (name + ".tmp")
            with tmp.open("x") as stream:
                stream.write(data)
            os.replace(tmp, dest)
    seal = {
        "schema": _SCHEMA,
        "filename": FILENAME,
        "sha256": WEIGHT_SHA256,
        "weight_stat": before,
        "assets": {k: _digest(path / k) for k in payloads},
        "tokenizer_binding": binding,
    }
    if _stamp(weight) != before:
        raise ValueError("V4.1 weight changed while sealing")
    target = path / INTEGRITY_FILE
    if target.is_symlink() or (
        target.exists() and not _same_seal(json.loads(target.read_text()), seal)
    ):
        raise ValueError("different V4.1 integrity seal already exists")
    if not target.exists():
        tmp = target.with_suffix(".tmp")
        with tmp.open("x") as stream:
            json.dump(seal, stream, indent=2)
        os.replace(tmp, target)
    return inspect_artifact(path)


def inspect_artifact(model_path):
    path = Path(model_path).expanduser().resolve()
    if json.loads(_regular_asset(path / "config.json").read_text()) != _CONFIG:
        raise ValueError("V4.1 adapter configuration changed")
    seal = json.loads(_regular_asset(path / INTEGRITY_FILE).read_text())
    if (
        seal.get("schema") != _SCHEMA
        or seal.get("filename") != FILENAME
        or seal.get("sha256") != WEIGHT_SHA256
    ):
        raise ValueError("invalid V4.1 integrity seal")
    weight = _regular_asset(path / FILENAME)
    if (
        not same_stamp(_stamp(weight), seal.get("weight_stat"))
        or weight.stat().st_size != WEIGHT_BYTES
    ):
        raise ValueError("V4.1 payload no longer matches verified seal")
    assets = {
        name: _digest(_regular_asset(path / name))
        for name in ("config.json", REFERENCE_FILE)
    }
    if assets != seal.get("assets") or assets[REFERENCE_FILE] != TOKENIZER_SHA256:
        raise ValueError("V4.1 sealed tokenizer/config changed")
    with GGUFFile(weight) as file:
        validate_header(file.metadata, file.tensors)
        inventory, metadata = file.inventory(), file.metadata
    fingerprint = hashlib.sha256(
        json.dumps(
            {"weight": WEIGHT_SHA256, "assets": assets},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "identity": {
            "path": str(path),
            "fingerprint": fingerprint,
            "weight_sha256": WEIGHT_SHA256,
            "files": [(FILENAME, WEIGHT_BYTES, weight.stat().st_mtime_ns)],
        },
        "config": dict(_CONFIG),
        "metadata": metadata,
        "inventory": inventory,
        "weight_stat": seal["weight_stat"],
        "tokenizer_binding": seal["tokenizer_binding"],
        "qualification": "unqualified",
        "weight_format": "native-packed-GGUF",
    }
