"""Save revision-bound independent-window logits using an offline adapter.

Run in a dedicated worker under scripts/run_with_gpu_locks.py. Input JSON is
a list of token-ID windows; no windows are concatenated or padded together.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--order", choices=("layer_major", "window_major"), default="layer_major"
    )
    parser.add_argument("--activation-budget-mib", type=int, default=512)
    parser.add_argument("--dense-stream-budget-mib", type=int, default=0)
    args = parser.parse_args()
    owners = [
        json.loads(Path(p).read_text())
        for p in (
            "/tmp/gpu.lock/owner.json",
            "/Users/Shared/mlxuag/gpu.lock/owner.json",
        )
    ]
    if owners[0] != owners[1]:
        raise RuntimeError("teacher capture requires matching owned GPU locks")
    if owners[0]["pid"] != os.getppid():
        raise RuntimeError("teacher capture must be a child of the GPU lock owner")
    os.kill(owners[0]["pid"], 0)
    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.streamed_load import WeightStreamRequest
    import mlx.core as mx

    windows = json.loads(args.tokens_json.read_text())
    # Exclusive directory creation keeps incomplete/old capture evidence from
    # being overwritten. A failure leaves no final manifest.
    args.out.mkdir(parents=True, exist_ok=False)
    request = (
        None
        if args.dense_stream_budget_mib == 0
        else WeightStreamRequest(
            mode="dense_mlp", budget_bytes=args.dense_stream_budget_mib << 20
        )
    )
    adapter = resolve_adapter(args.model)(args.model, weight_streaming=request)
    try:
        capture = getattr(adapter, "capture_teacher_windows", None)
        if capture is None:
            raise ValueError(
                "adapter does not implement independent-window teacher capture"
            )
        files = []

        def sink(index, tokens, hidden, cache):
            logits = adapter.model.logits(hidden)
            mx.eval(logits)
            path = args.out / f"window-{index:04d}.safetensors"
            mx.save_safetensors(str(path), {"logits": logits})
            files.append(
                {
                    "file": path.name,
                    "tokens": list(tokens),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

        receipt = capture(
            windows,
            sink,
            order=args.order,
            activation_budget_bytes=args.activation_budget_mib << 20,
        )
        receipt.update(
            {
                "artifact_identity": adapter.identity,
                "windows_files": files,
                "gpu_owner": owners[0],
                "streaming": None
                if adapter.weight_stream is None
                else {
                    "receipt": adapter.weight_stream.receipt(),
                    "counters": adapter.weight_stream.counters(),
                },
            }
        )
        (args.out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
