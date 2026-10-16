"""Build the pinned MiniLM Core ML (CPU_AND_NE) encoder bundle and a build receipt.

Never uses the GPU: MLX runs on CPU for the reference, and Core ML is loaded
only with CPU_ONLY or CPU_AND_NE. Prints a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="pinned all-MiniLM-L6-v2 checkpoint directory")
    parser.add_argument("--output", required=True, help="new bundle directory (must not exist)")
    parser.add_argument("--buckets", default="32,64,128,256")
    parser.add_argument("--layout", default=None, help="sequence_major or channels_first")
    parser.add_argument("--receipt", default=None,
                        help="build receipt path (default: <output parent>/build-receipt.json)")
    parser.add_argument("--compare-layouts", action="store_true",
                        help="also build every layout in scratch and record placement/parity")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args(argv)

    import mlx.core as mx

    mx.set_default_device(mx.cpu)  # before any MLX op; the GPU is never used
    from mlx2.runtime import coreml_encoder_build as build

    buckets = build.validate_buckets(tuple(int(item) for item in args.buckets.split(",")))
    output = Path(args.output).expanduser().absolute()
    receipt_path = Path(args.receipt) if args.receipt else output.parent / "build-receipt.json"
    if receipt_path.exists():
        raise SystemExit(f"refusing to overwrite receipt {receipt_path}")
    details = {}
    started = time.perf_counter()
    manifest = build.build_minilm_bundle(
        args.model, output, buckets=buckets,
        layout=args.layout or build.DEFAULT_LAYOUT, details=details,
    )
    receipt = {
        "schema": "mlx2.coreml-encoder.build-receipt.v1",
        "bundle": str(output),
        "bundle_bytes": build.bundle_bytes(output),
        "manifest_sha256": build._sha256_bytes((output / build.MANIFEST_NAME).read_bytes()),
        "build": details,
        "note": "exploratory build evidence; latency is not a performance claim",
    }
    if args.compare_layouts:
        tick = time.perf_counter()
        receipt["layout_comparison"] = build.compare_layouts(args.model, buckets=buckets)
        receipt["layout_comparison_s"] = round(time.perf_counter() - tick, 3)
    lock = build.gpu_lock_state()
    receipt["gpu_lock_at_latency"] = lock
    if args.skip_latency or not lock["latency_allowed"]:
        receipt["latency"] = None
        receipt["latency_skipped"] = "requested" if args.skip_latency else "gpu lock held by bench-class run"
    else:
        receipt["latency"] = build.measure_bundle_latency(
            output, manifest, text=build.DEFAULT_PARITY_TEXTS[-1],
        )
    receipt["wall_s"] = round(time.perf_counter() - started, 3)
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    summary = {
        "bundle": str(output), "bundle_bytes": receipt["bundle_bytes"], "receipt": str(receipt_path),
        "layout": details["layout"], "build_parity": manifest["build_parity"],
        "compute_plan": manifest["compute_plan"], "latency": receipt["latency"],
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
