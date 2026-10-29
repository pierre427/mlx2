"""Exact Vision-Exp lossless packed MLX artifact admission and conversion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..runtime.gguf.mlx_storage import (
    MANIFEST, SEAL, MLXPackedFile, atomic_json, convert_gguf, inspect_storage,
    local_file, sha256, stamp,
)

FORMAT = "mlx-packed-v1"
ARTIFACT_SEAL = "mlx2-artifact-integrity.json"


def _assets(path):
    return {n:sha256(local_file(path,n)) for n in ("config.json", "tokenizer.reference.json", MANIFEST)}


def inspect_artifact(model_path):
    from .deepseek_v4_gguf_artifact import MODEL_TYPE, WEIGHT_SHA256, validate_header

    path = Path(model_path).expanduser().resolve()
    config = json.loads(local_file(path,"config.json").read_text())
    if not isinstance(config,dict) or config.get("model_type") not in (MODEL_TYPE, "deepseek4_mlx") or config != dict(model_type=config["model_type"], mlx2_weight_format=FORMAT, source_weight_sha256=WEIGHT_SHA256,
                      packed_manifest=MANIFEST, tokenizer_reference="tokenizer.reference.json"):
        raise ValueError("unsupported packed MLX target configuration")
    manifest, _, _ = inspect_storage(path)
    if manifest["source_sha256"] != WEIGHT_SHA256:
        raise ValueError("packed MLX target source differs")
    seal = json.loads(local_file(path,ARTIFACT_SEAL).read_text())
    assets = _assets(path)
    if seal.get("schema") != "mlx2.deepseek-v4-mlx-integrity.v1" or seal.get("assets") != assets:
        raise ValueError("packed MLX adapter assets differ from seal")
    # CPU index construction only; no MLX arrays are allocated here.
    file = MLXPackedFile(path)
    try:
        validate_header(file.metadata,file.tensors)
        formats = {}
        for t in file.tensors.values():
            entry = formats.setdefault(t.format,dict(tensors=0,packed_bytes=0))
            entry["tensors"] += 1;entry["packed_bytes"] += t.nbytes
    finally:
        file.close()
    material = dict(format=FORMAT,source_weight_sha256=WEIGHT_SHA256,assets=assets)
    fingerprint = hashlib.sha256(json.dumps(material,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    return dict(identity=dict(path=str(path),fingerprint=fingerprint,weight_sha256=WEIGHT_SHA256,
                              files=[(n,r["bytes"],(path/n).stat().st_mtime_ns) for n,r in manifest["shards"].items()]),
                config=config,metadata=manifest["metadata"],inventory=dict(tensor_count=len(manifest["tensors"]),formats=formats),
                weight_stat=None,weight_format="MLX-safetensors-native-packed",qualification="unqualified",
                tokenizer_binding=seal["tokenizer_binding"],storage_manifest_sha256=assets[MANIFEST])


def inspect_sidecar(path):
    from ..runtime.drafters.deepseek_v4_dspark import SIDECAR_SHA256, DeepSeekV4DSparkConfig
    from dataclasses import asdict

    path = Path(path).expanduser().resolve()
    manifest, _, _ = inspect_storage(path)
    if manifest["source_sha256"] != SIDECAR_SHA256:
        raise ValueError("packed MLX DSpark source differs")
    file = MLXPackedFile(path)
    try:
        config = DeepSeekV4DSparkConfig.from_index(file.metadata,file.tensors)
    finally:
        file.close()
    return dict(path=str(path),sha256=SIDECAR_SHA256,config=json.loads(json.dumps(asdict(config))),
                tensor_count=len(manifest["tensors"]),stat=stamp(path/MANIFEST),
                seal_path=str(path/SEAL),seal_sha256=sha256(path/SEAL),verification="lossless_packed_mlx_verified_shards",
                weight_format=FORMAT)


def convert_artifact(source, destination, *, sidecar=None, progress=None):
    """Convert the sealed exact target, optional matching DSpark, then admit it."""
    import shutil
    from .deepseek_v4_gguf_artifact import inspect_artifact as inspect_gguf, FILENAME, MODEL_TYPE, REFERENCE_FILE
    from ..runtime.models.deepseek_v4 import DeepSeekV4Config, required_weights

    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    original = inspect_gguf(source)
    args = DeepSeekV4Config.from_gguf(original["metadata"])
    linears = [n for n,k in required_weights(args).items() if k == "linear"]
    manifest = convert_gguf(source/FILENAME,destination,source_sha256=original["identity"]["weight_sha256"],
                            linear_names=linears,progress=progress)
    shutil.copyfile(source/REFERENCE_FILE,destination/REFERENCE_FILE)
    config = dict(model_type="deepseek4_mlx",mlx2_weight_format=FORMAT,source_weight_sha256=manifest["source_sha256"],
                  packed_manifest=MANIFEST,tokenizer_reference=REFERENCE_FILE)
    atomic_json(destination/"config.json",config)
    if sidecar is not None:
        from .deepseek_v4_dspark_artifact import inspect_sidecar as inspect_gguf_sidecar
        record = inspect_gguf_sidecar(sidecar)
        # All native rank-two sidecar matrices are bounded; embedding is target-owned.
        from ..runtime.gguf.reader import GGUFFile
        with GGUFFile(sidecar) as file:
            names = [n for n,t in file.tensors.items() if len(t.shape)==2 and t.ggml_type in (1,30)]
        convert_gguf(sidecar,destination/"dspark",source_sha256=record["sha256"],linear_names=names,
                     progress=(lambda p:progress(dict(sidecar=True,**p))) if progress else None)
        inspect_sidecar(destination/"dspark")
    atomic_json(destination/ARTIFACT_SEAL,dict(schema="mlx2.deepseek-v4-mlx-integrity.v1",assets=_assets(destination),
                                             tokenizer_binding=original["tokenizer_binding"]))
    return inspect_artifact(destination)


def migrate_registration(model_path):
    """Reseal only config/asset identity for the dedicated adapter; no shard edits.

    Source-era containers remain inspectable for receipts. Migration validates
    the existing seal first and restores both assets if publication fails.
    """
    path=Path(model_path).expanduser().resolve()
    previous=inspect_artifact(path)
    config_path=path/"config.json";seal_path=path/ARTIFACT_SEAL
    old_config,old_seal=config_path.read_bytes(),seal_path.read_bytes()
    config=json.loads(old_config)
    if config["model_type"]=="deepseek4_mlx":return previous
    config["model_type"]="deepseek4_mlx"
    seal=json.loads(old_seal)
    try:
        atomic_json(config_path,config)
        seal["assets"]=_assets(path)
        atomic_json(seal_path,seal)
        return inspect_artifact(path)
    except BaseException:
        config_path.write_bytes(old_config);seal_path.write_bytes(old_seal)
        raise
