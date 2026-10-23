"""Pure one-use suspension epoch arithmetic for exact APCv2 manifests."""
from __future__ import annotations

import copy
import math


class SuspensionRestoreError(RuntimeError):
    """A suspension epoch cannot safely be consumed; keep its files intact."""


def consume_epoch(value, now):
    marker = value.get("suspension")
    if marker is None:
        return value, 0.0
    if (not isinstance(marker, dict) or set(marker) != {"id", "frozen_at"}
            or not isinstance(marker["id"], str) or not 1 <= len(marker["id"]) <= 128):
        raise SuspensionRestoreError("Malformed APCv2 suspension epoch")
    frozen = marker["frozen_at"]
    if (type(frozen) not in (int, float) or not math.isfinite(frozen)
            or not math.isfinite(now) or frozen > now):
        raise SuspensionRestoreError("APCv2 suspension clock moved backwards or is invalid")
    delta = now - frozen
    result = copy.deepcopy(value)
    result.pop("suspension", None)
    for name in ("created_at", "last_access_at"):
        stamp = result.get(name)
        if type(stamp) not in (int, float) or not math.isfinite(stamp):
            raise SuspensionRestoreError("Invalid APCv2 retention timestamp")
        result[name] = stamp + delta
    for name in ("disk_pins", "resident_pins"):
        for pin in result.get(name, []):
            expiry = pin.get("expires_at")
            if type(expiry) not in (int, float) or not math.isfinite(expiry):
                raise SuspensionRestoreError("Invalid APCv2 suspended pin expiry")
            pin["expires_at"] = expiry + delta
    return result, delta


def lora_file_receipt(path):
    """Hash an adapter's source bytes without importing MLX or materializing tensors."""
    import hashlib
    from pathlib import Path

    from .lora import read_lora_config

    directory = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Suspended LoRA requires a real adapter directory")
    config_path = directory / "adapter_config.json"
    weights_path = directory / "adapters.safetensors"
    if config_path.is_symlink() or weights_path.is_symlink() or config_path.stat().st_size > 1 << 20:
        raise ValueError("Suspended LoRA files must be bounded, regular source files")
    resolved, config = read_lora_config(directory)
    fingerprint = hashlib.sha256(repr((config["rank"], config["scale"], tuple(sorted(config["keys"])))).encode())
    weights_digest = hashlib.sha256()
    with weights_path.open("rb") as stream:
        while block := stream.read(1 << 20):
            fingerprint.update(block)
            weights_digest.update(block)
    return {"path": str(resolved), "fingerprint": fingerprint.hexdigest(),
            "files": {"adapter_config.json": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                      "adapters.safetensors": weights_digest.hexdigest()}}


def verify_lora_receipt(receipt, root):
    from pathlib import Path

    if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str) or not isinstance(root, str):
        raise TypeError("Suspended LoRA identity is unavailable")
    directory, allowed = Path(receipt["path"]), Path(root).resolve(strict=True)
    if not directory.resolve(strict=True).is_relative_to(allowed):
        raise ValueError("Suspended LoRA left its allowlisted directory")
    current = lora_file_receipt(directory)
    if any(current[key] != receipt.get(key) for key in ("path", "fingerprint", "files")):
        raise ValueError("LoRA source files changed since suspension")
    return current
