"""CPU-only indexed artifact checks shared by new ordinary text adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def inspect_indexed_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("artifact has no indexed weights")
    names = sorted(set(weights.values()))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("weight shard paths must stay within the artifact")
        item = (path / name).resolve()
        if not item.is_relative_to(path) or not item.is_file():
            raise ValueError(f"missing or escaped weight shard: {name}")
        stat = item.stat()
        if stat.st_size < 8:
            raise ValueError(f"empty weight shard: {name}")
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": weights,
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
    }
