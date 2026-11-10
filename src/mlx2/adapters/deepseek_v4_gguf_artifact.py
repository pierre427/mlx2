"""GPU-free, exact-artifact admission for the packed Vision-Exp checkpoint.

This identity gate does not load a model, grant vision/speculative capability,
or turn a partial transfer into a usable artifact. Original integration code;
tokenizer and tensor-format provenance is recorded in their own manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..runtime.gguf import GGUFFile
from ..runtime.immutable_weight_pages import same_stamp
from .deepseek_v4_gguf_tokenizer import align_tokenizer

MODEL_TYPE = "deepseek4_gguf"
FILENAME = (
    "DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"
)
WEIGHT_SHA256 = "66e47437ce7201546fa870a23ccdc094282660f1e52f3403ed5fb59604b2d894"
WEIGHT_BYTES = 86720111776
INTEGRITY_FILE = "mlx2-gguf-integrity.json"
REFERENCE_FILE = "tokenizer.reference.json"
_ASSETS = ("config.json", REFERENCE_FILE)
_SCHEMA = "mlx2.deepseek-v4-gguf-integrity.v1"


def _regular_asset(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("adapter asset must be a regular local file: " + path.name)
    return path


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stamp(path):
    st = path.stat()
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "inode": st.st_ino,
        "device": st.st_dev,
    }


def validate_header(metadata, tensors):
    expected = {
        "general.architecture": "deepseek4",
        "deepseek4.checkpoint_variant": "vision-exp",
        "deepseek4.block_count": 43,
        "deepseek4.embedding_length": 4096,
        "deepseek4.vocab_size": 129280,
        "deepseek4.expert_count": 256,
        "deepseek4.expert_used_count": 6,
        "deepseek4.hash_layer_count": 3,
        "deepseek4.attention.head_count": 64,
        "deepseek4.attention.key_length": 512,
        "deepseek4.attention.value_length": 512,
        "deepseek4.attention.sliding_window": 128,
        "deepseek4.hyper_connection.count": 4,
        "deepseek4.vision.sidecar_required": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value or type(metadata[key]) is not type(value):
            raise ValueError("unsupported packed Vision-Exp topology: " + key)
    for layer in range(43):
        prefix = f"blk.{layer}."
        for name, shape, fmt in (
            ("ffn_gate_exps.weight", (4096, 2048, 256), "IQ2_XXS"),
            ("ffn_up_exps.weight", (4096, 2048, 256), "IQ2_XXS"),
            ("ffn_down_exps.weight", (2048, 4096, 256), "Q2_K"),
        ):
            tensor = tensors.get(prefix + name)
            if tensor is None or tensor.shape != shape or tensor.format != fmt:
                raise ValueError("packed expert layout mismatch: " + prefix + name)
    if len(tensors) != 1328:
        raise ValueError("Vision-Exp target tensor inventory changed")


def prepare_artifact(model_path, reference_tokenizer):
    """Seal a completed pinned download and tokenizer without GPU allocation.

    Verify the payload again here: the acquisition receipt alone cannot prove
    that the file has not changed since download. The resulting stat-bound seal
    avoids rehashing 81 GiB on every status/adapter resolution.
    """
    path = Path(model_path).expanduser().resolve()
    weight = path / FILENAME
    if (
        not weight.is_file()
        or weight.is_symlink()
        or weight.stat().st_size != WEIGHT_BYTES
    ):
        raise ValueError(
            "complete pinned GGUF is absent; partial downloads cannot serve"
        )
    download = json.loads(
        _regular_asset(path / (FILENAME + ".receipt.json")).read_text()
    )
    spec = download.get("spec", {})
    if (
        download.get("status") != "verified"
        or download.get("verified_sha256") != WEIGHT_SHA256
        or spec.get("sha256") != WEIGHT_SHA256
        or spec.get("size") != WEIGHT_BYTES
        or spec.get("filename") != FILENAME
    ):
        raise ValueError("missing verified pinned GGUF acquisition receipt")
    before = _stamp(weight)
    if _digest(weight) != WEIGHT_SHA256 or _stamp(weight) != before:
        raise ValueError("GGUF payload changed or does not match the pinned SHA256")
    with GGUFFile(weight) as file:
        validate_header(file.metadata, file.tensors)
        reference = json.loads(Path(reference_tokenizer).read_text())
        _, tokenizer_receipt = align_tokenizer(reference, file.metadata)
    config = {
        "model_type": MODEL_TYPE,
        "mlx2_weight_format": "gguf-packed-v1",
        "gguf_file": FILENAME,
        "weight_sha256": WEIGHT_SHA256,
        "tokenizer_reference": REFERENCE_FILE,
    }
    values = {"config.json": config, REFERENCE_FILE: reference}
    for name, value in values.items():
        dest = path / name
        if dest.is_symlink():
            raise ValueError("refusing symlinked adapter asset: " + name)
        if dest.exists() and json.loads(dest.read_text()) != value:
            raise ValueError("refusing to overwrite different adapter asset: " + name)
        if not dest.exists():
            dest.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    seal = {
        "schema": _SCHEMA,
        "filename": FILENAME,
        "sha256": WEIGHT_SHA256,
        "weight_stat": before,
        "assets": {name: _digest(path / name) for name in _ASSETS},
        "tokenizer_binding": tokenizer_receipt,
    }
    target = path / INTEGRITY_FILE
    if _stamp(weight) != before:
        raise ValueError("GGUF payload changed while preparing adapter assets")
    if target.is_symlink():
        raise ValueError("refusing symlinked GGUF integrity seal")
    if target.exists():
        if json.loads(target.read_text()) != seal:
            raise ValueError("existing GGUF integrity seal differs")
    else:
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(seal, indent=2) + "\n")
        os.replace(tmp, target)
    return inspect_artifact(path)


def inspect_artifact(model_path):
    path = Path(model_path).expanduser().resolve()
    config = json.loads(_regular_asset(path / "config.json").read_text())
    if isinstance(config, dict) and config.get("mlx2_weight_format") == "mlx-packed-v1":
        from .deepseek_v4_mlx_artifact import inspect_artifact as inspect_mlx

        return inspect_mlx(path)
    if config != {
        "model_type": MODEL_TYPE,
        "mlx2_weight_format": "gguf-packed-v1",
        "gguf_file": FILENAME,
        "weight_sha256": WEIGHT_SHA256,
        "tokenizer_reference": REFERENCE_FILE,
    }:
        raise ValueError("DeepSeek packed adapter configuration changed")
    seal = json.loads(_regular_asset(path / INTEGRITY_FILE).read_text())
    if seal.get("schema") != _SCHEMA or seal.get("filename") != FILENAME:
        raise ValueError("unsupported GGUF integrity seal schema or filename")
    weight = path / FILENAME
    if (
        weight.is_symlink()
        or not weight.is_file()
        or seal.get("sha256") != WEIGHT_SHA256
        or not same_stamp(_stamp(weight), seal.get("weight_stat"))
        or weight.stat().st_size != WEIGHT_BYTES
    ):
        raise ValueError("GGUF no longer matches its verified integrity seal")
    assets = {name: _digest(_regular_asset(path / name)) for name in _ASSETS}
    if seal.get("assets") != assets:
        raise ValueError("GGUF adapter/tokenizer assets changed after sealing")
    with GGUFFile(weight) as file:
        validate_header(file.metadata, file.tensors)
        inventory = file.inventory()
        metadata = file.metadata
    identity_material = {"weights": WEIGHT_SHA256, "assets": assets}
    fingerprint = hashlib.sha256(
        json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "identity": {
            "path": str(path),
            "fingerprint": fingerprint,
            "weight_sha256": WEIGHT_SHA256,
            "files": [(FILENAME, WEIGHT_BYTES, weight.stat().st_mtime_ns)],
        },
        "config": config,
        "metadata": metadata,
        "inventory": inventory,
        "tokenizer_binding": seal["tokenizer_binding"],
        "weight_stat": seal["weight_stat"],
        "weight_format": "native-packed-GGUF",
        "qualification": "unqualified",
    }
