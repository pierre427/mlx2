"""CPU-only, pinned-artifact check of request-scoped compact MTP proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mlx2.runtime.mtp_draft_vocab import RequestCompactGreedyHead


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    path = args.artifact.expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    mapping = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "language_model.lm_head."
    filenames = {mapping[prefix + field] for field in ("weight", "scales", "biases")}
    if len(filenames) != 1:
        raise ValueError("target head parameters span shards")
    shard_name = filenames.pop()
    tensors = mx.load(path / shard_name, stream=mx.cpu)
    weight = tensors[prefix + "weight"]
    scales = tensors[prefix + "scales"]
    biases = tensors[prefix + "biases"]
    group_size = int(config["quantization"]["group_size"])
    bits = int(config["quantization"]["bits"])
    head = nn.QuantizedLinear(
        scales.shape[1] * group_size, weight.shape[0], bias=False,
        group_size=group_size, bits=bits,
    )
    head.weight, head.scales, head.biases = weight, scales, biases
    request = RequestCompactGreedyHead.from_bound_artifact(path, head)
    hidden = mx.random.normal((1, 1, scales.shape[1] * group_size),
                              key=mx.random.key(427)).astype(mx.float16)
    full = head(hidden)
    proposed, compact = request.propose(hidden, greedy=True)
    selected = mx.take(full, request.head.token_ids, axis=-1)
    mx.eval(full, proposed, compact, selected)
    error = float(mx.max(mx.abs(compact.astype(mx.float32) -
                                selected.astype(mx.float32))).item())
    print(json.dumps({
        "artifact": str(path), "device": "CPU", "selected_for_serving": False,
        "config_sha256": _sha(path / "config.json"),
        "index_sha256": _sha(path / "model.safetensors.index.json"),
        "head_shard": shard_name, "head_shard_sha256": _sha(path / shard_name),
        "manifest_sha256": _sha(path / "mtp_draft_vocab.json"),
        "proposal_rows": request.head.token_ids.size,
        "full_rows": weight.shape[0], "hidden_size": hidden.shape[-1],
        "selected_row_max_abs_logit_difference": error,
        "compact_token": int(proposed[0, 0].item()),
        "full_target_token": int(mx.argmax(full[0, 0]).item()),
        "target_head_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
