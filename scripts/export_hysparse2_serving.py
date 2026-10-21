#!/usr/bin/env python3
"""Export a HySparse2 research checkpoint as an mlx2 serving artifact.

The export directory hard-links the immutable training outputs
(``model.safetensors``, ``semantic-ple.safetensors``, ``state.json``), copies
the training tokenizer, and writes a generated ``config.json`` plus a
manifest binding every source hash.  The source checkpoint is never written.
Hard links need the same filesystem; ``--copy`` copies instead.

    python scripts/export_hysparse2_serving.py \\
        --checkpoint ~/mlx-training/hysparse2/<run>/step-NNNNNNNN \\
        --tokenizer ~/mlx-training/hysparse2/tokens-20260929-v2/tokenizer.json \\
        --output ~/mlx-models/HySparse2-toy-<name>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlx2.adapters.hysparse2 import (  # noqa: E402
    ARCHITECTURE,
    MANIFEST,
    MANIFEST_SCHEMA,
    PLE_FILE,
    TOKENIZER_FILE,
    inspect_artifact,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(checkpoint: Path, tokenizer: Path, output: Path, *, copy: bool = False) -> dict:
    checkpoint, tokenizer = checkpoint.expanduser().resolve(), tokenizer.expanduser().resolve()
    output = output.expanduser()
    state = json.loads((checkpoint / "state.json").read_text())
    config = dict(state["config"])
    names = ["model.safetensors", "state.json"]
    if config.get("semantic_ple_rows"):
        names.append(PLE_FILE)
    for name in names:
        if not (checkpoint / name).is_file():
            raise SystemExit(f"checkpoint is missing {name}")
    tokenizer_sha = sha256(tokenizer)
    pinned = (state.get("run") or {}).get("tokenizer_sha256")
    if pinned is not None and pinned != tokenizer_sha:
        raise SystemExit(
            f"tokenizer {tokenizer_sha} differs from the checkpoint's {pinned}"
        )
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in names:
        source, target = checkpoint / name, output / name
        if copy:
            shutil.copy2(source, target)
        else:
            os.link(source, target)
        hashes[name] = sha256(target)
    shutil.copy2(tokenizer, output / TOKENIZER_FILE)
    hashes[TOKENIZER_FILE] = tokenizer_sha
    serving_config = {
        **config,
        "architectures": [ARCHITECTURE],
        "eos_token": "<eos>",
    }
    (output / "config.json").write_text(json.dumps(serving_config, indent=2) + "\n")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_checkpoint": str(checkpoint),
        "source_tokenizer": str(tokenizer),
        "step": state.get("step"),
        "materialization": "copy" if copy else "hardlink",
        "sha256": hashes,
        "chat_format": "training corpus messages_text: 'role: content' turns joined by blank lines",
        "status": {
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
        },
    }
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    artifact = inspect_artifact(output)
    return {"output": str(output), "fingerprint": artifact["identity"]["fingerprint"], **manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="copy instead of hard-linking")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.checkpoint, args.tokenizer, args.output, copy=args.copy), indent=2))


if __name__ == "__main__":
    main()
