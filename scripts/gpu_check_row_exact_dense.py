#!/usr/bin/env python3
"""Metal gate: dense (bf16 ``nn.Linear``) projections inside row-exact verify windows.

Mixed-precision Flash-Next artifacts (the uncensored MLX2 4-bit MTP build)
keep the HC ``block_inject_weight`` ([4, 10240]) and the MoE
``shared_expert_gate`` ([1, 2560]) as plain bf16 ``nn.Linear``.  MLX's dense
matmul is ``gemv`` at one row and ``gemv_wide``/GEMM tiling at more, so a
verify window that runs them at M=R does not carry the one-token step's bits.
Real weights, raw-byte comparisons:

  dense  per layer, the inject (attn and mlp HC) and the shared gate at R
         rows: the stock M=R call and the route's ``_RowExactLinear`` (inside
         a ``row_exact_verify.window``) vs R one-token calls.
  hc     per HC module, the whole ``GatedResidual`` at R rows inside a window
         with the route's class swaps, for HC kernels off, on (row-exact
         window kernels on) and on with window kernels off, vs R one-token
         calls under the same setting; plus the pre-fix swap (quantized
         projections only) to show what the window used to claim.

  scratchpad/gpuq.sh dense-gate env PYTHONPATH=src MLX_ENABLE_TF32=0 \\
      .venv/bin/python scripts/gpu_check_row_exact_dense.py --i-own-the-gpu \\
      --out dense.json
"""

import argparse
import json
import struct
from pathlib import Path

DEFAULT_MODEL = Path("~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP").expanduser()
SCALES = (0.05, 1.0, 8.0, 40.0)


class Tensors:
    """Read single tensors from safetensors shards by header offset (no
    shard-wide load)."""

    def __init__(self, root: Path):
        self.root = root
        self.index = {}
        for path in sorted(root.glob("*.safetensors")):
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            for key, meta in header.items():
                if key != "__metadata__":
                    self.index[key] = (path, 8 + n, meta)

    def has(self, key):
        return key in self.index

    def get(self, key):
        import mlx.core as mx
        import numpy as np

        path, base, meta = self.index[key]
        start, end = meta["data_offsets"]
        with open(path, "rb") as fh:
            fh.seek(base + start)
            raw = fh.read(end - start)
        kind = {"BF16": np.uint16, "U32": np.uint32, "F32": np.float32}[meta["dtype"]]
        arr = mx.array(np.frombuffer(raw, dtype=kind).reshape(meta["shape"]))
        return arr.view(mx.bfloat16) if meta["dtype"] == "BF16" else arr

    def prefix(self, prefix):
        return {k[len(prefix) + 1:]: self.get(k) for k in self.index if k.startswith(prefix + ".")}


def _same(a, b) -> bool:
    import mlx.core as mx

    mx.eval(a, b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (mx.bfloat16, mx.float16):
        a, b = a.view(mx.uint16), b.view(mx.uint16)
    return bool(mx.array_equal(a, b).item())


def _rows_equal(got, ref, rows):
    return sum(_same(got[:, j : j + 1], ref[:, j : j + 1]) for j in range(rows))


def _maxdiff(a, b) -> float:
    import mlx.core as mx

    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def check_dense(a, store):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime import row_exact_verify as REV
    from mlx2.runtime.models import qwen4_row_exact as RE

    swapped = RE._subclass(RE._RowExactLinear, nn.Linear)
    cells = []
    for layer in a.layers:
        for name in ("attn_hyper_connection.block_inject_weight",
                     "mlp_hyper_connection.block_inject_weight",
                     "mlp.shared_expert_gate"):
            key = f"language_model.model.layers.{layer}.{name}.weight"
            if not store.has(key) or store.has(key.replace(".weight", ".scales")):
                cells.append({"layer": layer, "module": name, "skipped": "not dense in this artifact"})
                continue
            weight = store.get(key)
            lin = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            lin.weight = weight
            mx.eval(lin.parameters())
            for r in a.rows:
                for case in range(a.cases):
                    x = (mx.random.normal((1, r, weight.shape[1]), key=mx.random.key(97 * r + case))
                         * SCALES[case % 4]).astype(mx.bfloat16)
                    mx.eval(x)
                    ref = mx.concatenate([lin(x[:, j : j + 1]) for j in range(r)], axis=1)
                    stock = lin(x)
                    lin.__class__ = swapped
                    record = REV.Window(r)
                    with REV.window(record):
                        got = lin(x)
                    mx.eval(got)
                    lin.__class__ = nn.Linear
                    cells.append({
                        "layer": layer, "module": name, "rows": r, "case": case,
                        "stock_rows_equal": _rows_equal(stock, ref, r),
                        "stock_max_abs_diff": _maxdiff(stock, ref),
                        "route_rows_equal": _rows_equal(got, ref, r),
                        "route_identical": _same(got, ref),
                        "stages": record.stages, "window_exact": record.exact,
                    })
            print("dense", layer, name, flush=True)
    return cells


def check_hc(a, store):
    import mlx.core as mx
    import mlx.nn as nn
    from types import SimpleNamespace

    from mlx2.runtime import row_exact_verify as REV
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import qwen4_hc_decode as HCD
    from mlx2.runtime.models import qwen4_row_exact as RE

    config = json.loads((a.model / "config.json").read_text())
    text = config.get("text_config", config)
    args = SimpleNamespace(hc_count=text["hc_count"], hidden_size=text["hidden_size"],
                           hc_lowrank=text["hc_lowrank"], rms_norm_eps=text["rms_norm_eps"])
    width = args.hc_count * args.hidden_size
    children = ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")

    def build(prefix):
        tensors = store.prefix(prefix)
        module = Q.GatedResidual(args, use_combine=True)
        for child in children:
            if f"{child}.scales" in tensors:
                packed = tensors[f"{child}.weight"].shape[1]
                bits = packed * 32 // getattr(module, child).weight.shape[1]
                module[child] = nn.QuantizedLinear.from_linear(module[child], group_size=64, bits=bits)
        module.load_weights(list(tensors.items()), strict=True)
        module.eval()
        mx.eval(module.parameters())
        return module

    def swap(module, dense):
        for child in children:
            cls = type(module[child])
            if cls is nn.QuantizedLinear:
                module[child].__class__ = RE._subclass(RE._RowExactQuantizedLinear, cls)
            elif cls is nn.Linear and dense:
                module[child].__class__ = RE._subclass(RE._RowExactLinear, cls)

    def unswap(module):
        for child in children:
            module[child].__class__ = getattr(type(module[child]), "_row_exact_base", type(module[child]))

    def one_token(module, x, r):
        parts = [module(x[:, j : j + 1]) for j in range(r)]
        return tuple(mx.concatenate([p[i] for p in parts], axis=1) for i in (0, 2))

    def window(module, x, r, dense):
        swap(module, dense)
        before = HCD.hc_decode_status()["row_exact_calls"]
        record = REV.Window(r)
        with REV.window(record), Q._declared_width(1):
            out = module(x)
        got = (out[0], out[2])
        mx.eval(*got)
        unswap(module)
        return got, record, HCD.hc_decode_status()["row_exact_calls"] - before

    # (arm, hc kernels, row-exact window kernels, dense swap)
    arms = (
        ("hc_off", False, True, True),
        ("hc_on", True, True, True),
        ("hc_on_window_kernels_off", True, False, True),
        ("hc_off_prefix_swap_only", False, True, False),
    )
    results = {}
    HCD.reset_for_tests()
    for layer in a.layers:
        for part in ("attn", "mlp"):
            name = f"language_model.model.layers.{layer}.{part}_hyper_connection"
            module = build(name)
            kinds = {c: type(module[c]).__name__ for c in children}
            cells = []
            for r in a.rows:
                for case in range(a.cases):
                    x = (mx.random.normal((1, r, width), key=mx.random.key(1000 * r + case))
                         * SCALES[case % 4]).astype(mx.bfloat16)
                    mx.eval(x)
                    refs = {}
                    for hc_on in (False, True):
                        HCD.set_hc_decode_enabled(hc_on)
                        refs[hc_on] = one_token(module, x, r)
                        mx.eval(*refs[hc_on])
                    for arm, hc_on, kernels, dense in arms:
                        HCD.set_hc_decode_enabled(hc_on)
                        HCD.set_hc_row_exact_enabled(kernels)
                        got, record, served = window(module, x, r, dense)
                        ref = refs[hc_on]
                        mixed_eq = _rows_equal(got[0], ref[0], r)
                        inject_eq = _rows_equal(got[1], ref[1], r)
                        cells.append({
                            "arm": arm, "rows": r, "case": case,
                            "mixed_rows_equal": mixed_eq, "inject_rows_equal": inject_eq,
                            "identical": mixed_eq == r and inject_eq == r,
                            "window_exact": record.exact, "served_by_kernel": served == 1,
                            "stages": record.stages,
                            "inject_max_abs_diff": _maxdiff(got[1], ref[1]),
                        })
            HCD.set_hc_decode_enabled(False)
            HCD.set_hc_row_exact_enabled(True)
            results[name] = {"classes": kinds, "cells": cells}
            by_arm = {}
            for c in cells:
                s = by_arm.setdefault(c["arm"], [0, 0, 0])
                s[0] += c["identical"]
                s[1] += 1
                s[2] += c["window_exact"] and not c["identical"]
            print("hc", name, kinds["block_inject_weight"],
                  {k: f"{v[0]}/{v[1]} identical, {v[2]} mislabelled" for k, v in by_arm.items()},
                  flush=True)
            del module
            mx.clear_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 3, 23, 47])
    ap.add_argument("--rows", type=int, nargs="+", default=[2, 3, 8, 17])
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.flash_next import configure_environment

    configure_environment(a.model)
    import mlx.core as mx

    assert mx.default_device() == mx.gpu and mx.metal.is_available()
    mx.set_cache_limit(4 << 30)
    store = Tensors(a.model)
    report = {"model": str(a.model), "mlx": mx.__version__,
              "args": {k: str(v) for k, v in vars(a).items()}}
    out = Path(a.out)
    summary = {}
    if "dense" not in a.skip:
        cells = [c for c in check_dense(a, store) if "skipped" not in c]
        report["dense"] = cells
        summary["dense_cases"] = len(cells)
        summary["dense_route_identical"] = sum(c["route_identical"] for c in cells)
        summary["dense_stock_identical"] = sum(c["stock_rows_equal"] == c["rows"] for c in cells)
        summary["dense_stock_rows_differing"] = sum(c["rows"] - c["stock_rows_equal"] for c in cells)
        out.write_text(json.dumps(report, indent=1))
    if "hc" not in a.skip:
        report["hc"] = check_hc(a, store)
        arms = {}
        for mod in report["hc"].values():
            for c in mod["cells"]:
                s = arms.setdefault(c["arm"], {"cases": 0, "identical": 0, "exact_label": 0,
                                               "mislabelled": 0, "kernel": 0})
                s["cases"] += 1
                s["identical"] += c["identical"]
                s["exact_label"] += c["window_exact"]
                s["mislabelled"] += c["window_exact"] and not c["identical"]
                s["kernel"] += c["served_by_kernel"]
        summary["hc"] = arms
    report["summary"] = summary
    out.write_text(json.dumps(report, indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
