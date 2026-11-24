#!/usr/bin/env python3
"""GPU self-test of the staged native step-replay module (run under the GPU
lease; never on a CPU-only preflight).

Loads build/step_replay/stage/_step_replay<ext> in this process (right after
``import mlx.core``, before MLX compiles any kernel, so every pipeline is
reflected), installs the hooks and checks what the module implements today:

  abi      the nanobind probe accepts an mlx.core.array; exactly one
           libmlx.dylib image; version constants match the running MLX;
  census   the selectors of the concurrent compute encoder class, and the
           methods MLX actually calls while recording small graphs
           (elementwise, matmul, reduction, SDPA, a metal_kernel, an
           in-place slice update);
  record   each graph recorded twice: supported (no texture, ICB, two
           encoders, attribute stride, use_resource ...), every kernel
           reflected, the two recordings structurally equal;
  inkling2 (``--model PATH``) the fixed-capacity Inkling 2 decode step on a
           synthetic state, recorded twice: dispatch/barrier/buffer counts,
           unsupported reason, stability.

Sealing and replay are not implemented in the module yet, so this receipt
never authorizes ``build_step_replay.py --install`` ("passed" stays false
with reason ``replay_unimplemented``; R1 needs replay).

    $PY scripts/step_replay_selftest.py --out receipt.json [--model $MODEL]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import mlx.core as mx  # first: the module shares its nanobind registry

ROOT = Path(__file__).resolve().parents[1]
STAGED = ROOT / "build" / "step_replay" / "stage" / (
    "_step_replay" + sysconfig.get_config_var("EXT_SUFFIX"))


def load_staged(path):
    loader = importlib.machinery.ExtensionFileLoader("mlx2_step_replay._step_replay", str(path))
    spec = importlib.util.spec_from_file_location("mlx2_step_replay._step_replay", str(path),
                                                  loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def graphs():
    key = mx.random.key(0)
    a = mx.random.normal((64, 128), key=key)
    b = mx.random.normal((128, 96), key=mx.random.split(key)[0])
    q = mx.random.normal((1, 8, 1, 64)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, 300, 64)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, 300, 64)).astype(mx.bfloat16)
    bank = mx.zeros((1, 2, 512, 64), mx.bfloat16)
    row = mx.ones((1, 2, 1, 64), mx.bfloat16)
    at = mx.array([7], mx.uint32)
    kernel = mx.fast.metal_kernel(
        name="selftest_axpy", input_names=["x", "y"], output_names=["out"],
        source="uint i = thread_position_in_grid.x; out[i] = 2.0f * x[i] + y[i];")
    mx.eval(a, b, q, k, v, bank, row, at)
    return {
        "elementwise": lambda: [mx.exp(a) * 3 + a],
        "matmul": lambda: [a @ b],
        "reduction": lambda: [mx.sum(a, axis=1), mx.max(a)],
        "sdpa": lambda: [mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)],
        "metal_kernel": lambda: kernel(inputs=[a.reshape(-1), a.reshape(-1)],
                                       grid=(a.size, 1, 1), threadgroup=(256, 1, 1),
                                       output_shapes=[(a.size,)], output_dtypes=[mx.float32]),
        "slice_update": lambda: [mx.slice_update(bank, row, at, axes=(2,))],
    }


def record(module, fn):
    mx.synchronize()  # inputs' producers finish before the window opens
    module.record_begin()
    try:
        outputs = fn()
        mx.eval(outputs)
    finally:
        rec = module.record_end()
    return rec


def record_twice(module, fn):
    fn()  # warm: trace and compile outside the recording
    mx.eval(fn())
    first, second = record(module, fn), record(module, fn)
    a, b = first.summary(), second.summary()
    return {"first": a, "second": b, "compare": module.compare_recordings(first, second),
            "supported": not a["unsupported"] and not b["unsupported"],
            "reflected": a["unreflected_kernels"] == 0}


def inkling2(module, model_path, pos):
    from mlx2.adapters.inkling2 import (inspect_artifact, load_model, model_kernel_options,
                                        validate_policy)
    from mlx2.runtime.models.inkling2 import capacity_state as cs

    artifact = inspect_artifact(model_path)
    model, _ = load_model(Path(artifact["identity"]["path"]), artifact)
    policy = model_kernel_options(validate_policy({"inkling2_native_replay": True})[0])
    policy["moe_nax_gather"] = False  # prefill-only; the step is decode
    model.configure_kernels(**policy)
    cache = cs.synthetic_cache(model, pos)
    kinds, reason = cs.plan(model, cache)
    if reason is not None:
        return {"declined": reason}
    cs.normalize(model, cache, kinds, {})
    arrays, _ = cs.round_inputs(model, mx.array([[11]], mx.int32), cache, kinds)
    step = cs.make_step(model, kinds)
    out = record_twice(module, lambda: step(arrays))
    out["checkpoint"] = f"{Path(model_path).name}@{artifact['identity'].get('fingerprint')}"
    out["pos"] = pos
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", default=str(STAGED))
    parser.add_argument("--out", required=True)
    parser.add_argument("--model")
    parser.add_argument("--pos", type=int, default=1500)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise SystemExit("Metal required")
    path = Path(args.module)
    receipt = {"module": str(path), "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "mlx": mx.__version__, "started": time.time(), "passed": False,
               "reason": "replay_unimplemented"}
    try:
        receipt["source_revision"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True,
            check=True).stdout.strip()
    except subprocess.CalledProcessError:
        receipt["source_revision"] = None
    module = load_staged(path)
    receipt["abi"] = {"probe": module.abi_probe(mx.zeros((1,))),
                      "libmlx_images": module.libmlx_image_count(),
                      "mlx_version": module.MLX_VERSION, "runtime": module.mlx_runtime_version(),
                      "nanobind": module.NANOBIND_VERSION, "api": module.API_VERSION,
                      "capabilities": list(module.CAPABILITIES)}
    receipt["install"] = module.install_hooks()
    receipt["census"] = {"encoder_selectors": module.class_selectors("encoder")}
    receipt["graphs"] = {name: record_twice(module, fn) for name, fn in graphs().items()}
    if args.model:
        receipt["inkling2"] = inkling2(module, args.model, args.pos)
    probe = mx.arange(6).reshape(2, 3)[:, 1:]
    mx.eval(probe)  # array_layout refuses unevaluated arrays (layout depends on it)
    receipt["layout_probe"] = module.array_layout(probe)
    receipt["pipelines_filed"] = module.pipeline_count()
    receipt["finished"] = time.time()
    ok = (receipt["abi"]["probe"] is True and receipt["abi"]["libmlx_images"] == 1
          and receipt["install"]["installed"] and not receipt["install"]["error"]
          and all(g["supported"] and g["compare"] == "equal" for g in receipt["graphs"].values()))
    receipt["hooks_and_recording_ok"] = ok
    Path(args.out).write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    print(json.dumps({k: receipt[k] for k in ("hooks_and_recording_ok", "passed", "reason")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
