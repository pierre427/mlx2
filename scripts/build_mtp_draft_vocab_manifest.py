#!/usr/bin/env python3
"""Bind a reviewed MTP draft-vocabulary id list to one local model artifact.

This is a CPU/filesystem preparation step.  It never imports MLX, loads model
weights, starts a server, or claims a GPU.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = "mlx2.mtp-draft-vocab.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ids(path: Path, vocab_size: int) -> list[int]:
    values = set()
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            lines = stream.readlines()
    else:
        lines = path.read_text().splitlines()
    for line_number, raw in enumerate(lines, 1):
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        try:
            token_id = int(value, 10)
        except ValueError:
            raise SystemExit(f"{path}:{line_number}: expected an integer") from None
        if not 0 <= token_id < vocab_size:
            raise SystemExit(f"{path}:{line_number}: id {token_id} outside vocabulary")
        values.add(token_id)
    ids = sorted(values)
    if len(ids) < 4096 or len(ids) % 64:
        raise SystemExit("id count must be at least 4096 and a multiple of 64")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--license-file", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--corpus-profile", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to --model-path; use a staging directory for review",
    )
    args = parser.parse_args()
    model = args.model_path.expanduser().resolve()
    output = (args.output_dir or model).expanduser().resolve()
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    vocab_size = int(text_config["vocab_size"])
    ids = parse_ids(args.ids.expanduser().resolve(), vocab_size)
    output.mkdir(parents=True, exist_ok=True)
    ids_output = output / "mtp_draft_vocab.ids"
    ids_output.write_text("".join(f"{value}\n" for value in ids))
    license_output = output / "mtp_draft_vocab.ids.LICENSE"
    license_output.write_bytes(args.license_file.expanduser().resolve().read_bytes())
    manifest = {
        "schema": SCHEMA,
        "vocab_size": vocab_size,
        "token_count": len(ids),
        "ids_sha256": sha256(ids_output),
        "license_sha256": sha256(license_output),
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(model / "model.safetensors.index.json"),
        "tokenizer_sha256": sha256(model / "tokenizer.json"),
        "source_repository": args.source_repository,
        "source_revision": args.source_revision,
        "corpus_profile": args.corpus_profile,
    }
    manifest_output = output / "mtp_draft_vocab.json"
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_output), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
