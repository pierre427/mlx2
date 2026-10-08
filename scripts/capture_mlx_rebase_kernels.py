#!/usr/bin/env python3
"""Capture one #4640 thin-N and one #4641 qmm split-K Metal dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx


def tree_digest(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    size = 0
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix().encode()
        data = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
        files += 1
        size += len(data)
    return digest.hexdigest(), files, size


def capture(path: Path, build_output) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite capture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.synchronize()
    mx.metal.start_capture(str(path))
    try:
        output = build_output()
        mx.eval(output)
        mx.synchronize()
    finally:
        mx.metal.stop_capture()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("refusing Metal capture without --i-own-the-gpu")

    mx.random.seed(4640)
    a = mx.random.normal((512, 10240)).astype(mx.bfloat16)
    b = mx.random.normal((4, 10240)).astype(mx.bfloat16)
    mx.eval(a, b)
    thin_path = args.capture_dir / "thin-nax-m512-n4-k10240.gputrace"
    capture(thin_path, lambda: a @ b.T)

    mx.random.seed(4641)
    x = mx.random.normal((33, 2560)).astype(mx.bfloat16)
    w = mx.random.normal((1024, 2560)).astype(mx.bfloat16)
    wq = mx.quantize(w, group_size=64, bits=4, mode="affine")
    mx.eval(x, *wq)
    qmm_path = args.capture_dir / "qmm-splitk-m33-n1024-k2560.gputrace"
    capture(
        qmm_path,
        lambda: mx.quantized_matmul(
            x, *wq, transpose=True, group_size=64, bits=4, mode="affine"
        ),
    )

    captures = {}
    for name, path in (("thin_n", thin_path), ("qmm_splitk", qmm_path)):
        sha256, files, size = tree_digest(path)
        captures[name] = {
            "path": str(path),
            "tree_sha256": sha256,
            "files": files,
            "bytes": size,
        }
    receipt = {
        "schema": "mlx2.mlx-rebase-metal-captures.v1",
        "mlx": mx.__version__,
        "device": mx.device_info(mx.gpu),
        "captures": captures,
        "claim_boundary": "dispatch evidence only; not qualification or performance evidence",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
