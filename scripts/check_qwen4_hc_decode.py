"""Metal gate for the two-launch Flash-Next HC decode (omlx #4038 port).

Loads REAL hyper-connection weights (a few trunk layers, the trunk mixer and
the MTP head's HC modules) from the served artifact into stand-alone
``GatedResidual`` modules under the Flash-Next environment profile, then:

  exact   for each module and many inputs (one row and 2..8 folded rows),
          compares the composed path
          (kernel off) with the two launches bit for bit: ``mixed`` and
          ``inject``, plus the kernel's normed row and SiLU activations against
          the composed intermediates (localises any mismatch).
  launches  op trace of one call per arm: the primitives in the lazy graph that
          dispatch a kernel (views such as Reshape/Broadcast are not counted).
  rowexact  (``--row-exact-cases``) inside a row-exact verify window, the
          one-row law per row against each row's own one-token composed call.
  micro   a 96-call decode-shaped chain (x -> HC -> apply_inject(branch=mixed)),
          one eval per chain, arms alternated per rep after a warm-up.

  PYTHONPATH=src .venv/bin/python scripts/check_qwen4_hc_decode.py --i-own-the-gpu \
      --out /tmp/hc-exact.json

Projection formats follow the artifact: a projection with ``scales`` is
quantized at the width its packed weight implies (4 or 8 bits, group 64), one
without stays a dense ``nn.Linear`` (the uncensored artifact's bf16 inject).
"""

import argparse
import json
import os
import re
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

DEFAULT_MODEL = Path("~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
# (batch, seq) shapes: one-row decode (law 0), verify windows and multi-lane
# decode (law 1, MLX qmv_wide).
SHAPES = [(1, 1), (1, 1), (1, 3), (1, 2), (4, 1), (1, 5), (1, 8), (2, 3)]
NO_LAUNCH = {"Reshape", "Broadcast", "Squeeze", "ExpandDims", "Transpose", "AsStrided",
             "StopGradient", "Depends"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 3, 23, 46, 47])
    ap.add_argument("--cases", type=int, default=48)
    ap.add_argument("--chain", type=int, default=96)
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--row-exact-cases", type=int, default=0,
                    help="cases per module inside a row-exact window (0: skip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.flash_next import configure_environment

    configure_environment(a.model)
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import qwen4_hc_decode as HCD

    assert mx.default_device() == mx.gpu and mx.metal.is_available()
    # The gate covers both projection laws: lift the serving default that
    # keeps 2..8-row calls composed.
    HCD.set_hc_multi_row_enabled(True)
    mx.set_cache_limit(4 << 30)
    config = json.loads((a.model / "config.json").read_text())
    text = config.get("text_config", config)
    args = SimpleNamespace(
        hc_count=text["hc_count"], hidden_size=text["hidden_size"],
        hc_lowrank=text["hc_lowrank"], rms_norm_eps=text["rms_norm_eps"],
    )
    index = json.loads((a.model / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {}

    def tensors(prefix):
        out = {}
        for full in (prefix, "language_model." + prefix):
            for key, shard in index.items():
                if key.startswith(full + "."):
                    if shard not in shards:
                        shards[shard] = mx.load(str(a.model / shard))
                    out[key[len(full) + 1:]] = shards[shard][key]
            if out:
                return out
        mtp_file = a.model / "model-mtp-q4.safetensors"
        if mtp_file.exists():
            mtp = mx.load(str(mtp_file))
            out = {k[len(prefix) + 1:]: v for k, v in mtp.items() if k.startswith(prefix + ".")}
        return out

    def build(prefix, combine):
        weights = tensors(prefix)
        assert weights, prefix
        module = Q.GatedResidual(args, use_combine=combine)

        def fmt(path, layer):
            if f"{path}.scales" not in weights:
                return False  # dense projection in the artifact
            packed = weights[f"{path}.weight"].shape[1]
            return {"group_size": 64, "bits": packed * 32 // layer.weight.shape[1]}

        nn.quantize(module, class_predicate=fmt)
        module.load_weights(list(weights.items()), strict=True)
        module.eval()
        mx.eval(module.parameters())
        return module

    modules = {}
    for layer in a.layers:
        for part in ("attn", "mlp"):
            name = f"language_model.model.layers.{layer}.{part}_hyper_connection"
            modules[name] = build(name, True)
    for name, combine in (("language_model.model.hyper_connection_mixer", False),
                          ("mtp.layers.0.attn_hyper_connection", True),
                          ("mtp.layers.0.mlp_hyper_connection", True),
                          ("mtp.hyper_connection_mixer", False)):
        modules[name] = build(name, combine)
    first = next(iter(modules.values()))

    def describe(layer):
        return f"q{layer.bits}" if isinstance(layer, nn.QuantizedLinear) else f"dense {layer.weight.dtype}"

    formats = sorted({
        " / ".join(f"{n} {describe(m[n])}" for n in
                   ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight") if n in m)
        for m in modules.values()})
    print("loaded", len(modules), "modules;", "norm dtype", first.hc_norm.weight.dtype,
          "formats", formats, flush=True)
    for name, module in modules.items():
        reason = HCD.static_admission(module)
        assert reason is None, (name, reason)

    width = args.hc_count * args.hidden_size

    def composed(module, x):
        HCD.set_hc_decode_enabled(False)
        return module(x)

    def fused(module, x):
        HCD.set_hc_decode_enabled(True)
        out = module(x)
        HCD.set_hc_decode_enabled(False)
        return out

    def bits_equal(p, q):
        return p.shape == q.shape and p.dtype == q.dtype and bool(
            mx.array_equal(p.view(mx.uint16), q.view(mx.uint16)).item())

    # ---- elementwise table ---------------------------------------------------
    # Every bf16 input through the kernels' epilogue helpers vs the eager ops.
    import numpy as np

    allv = mx.array(np.arange(65536, dtype=np.uint16)).view(mx.bfloat16)
    finite = np.isfinite(np.array(allv.astype(mx.float32)))
    table = mx.fast.metal_kernel(
        name="mlx2_qwen4_hc_decode_table", input_names=["inp"],
        output_names=["silu", "sig", "inj"], header=HCD.HEADER,
        source="""
          uint i = thread_position_in_grid.x;
          T x = inp[i];
          T a = x / T(4);
          silu[i] = a * hcd_sigmoid_jit<T>(a);
          sig[i] = hcd_sigmoid_unary<T>(x);
          inj[i] = T(2) * hcd_sigmoid_unary<T>(a);
        """)
    k_silu, k_sig, k_inj = table(
        inputs=[allv], template=[("T", mx.bfloat16)], grid=(65536, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(65536,)] * 3,
        output_dtypes=[mx.bfloat16] * 3)
    e_silu = nn.silu(allv / 4)
    e_sig = mx.sigmoid(allv)
    e_inj = 2 * mx.sigmoid(allv / 4)
    mx.eval(k_silu, k_sig, k_inj, e_silu, e_sig, e_inj)

    def table_diff(p, q):
        return int(((np.array(p.view(mx.uint16)) != np.array(q.view(mx.uint16))) & finite).sum())

    elementwise = {"silu_div_hc": table_diff(k_silu, e_silu), "sigmoid": table_diff(k_sig, e_sig),
                   "inject_gate": table_diff(k_inj, e_inj), "finite_inputs": int(finite.sum())}
    print("TABLE", elementwise, flush=True)

    # ---- exact -------------------------------------------------------------
    HCD.reset_for_tests()
    report = {"model": str(a.model), "mlx": mx.__version__, "modules": {},
              "formats": formats, "elementwise_table_mismatches": elementwise}
    scales = [0.02, 0.3, 1.0, 4.0, 30.0]
    total_mismatch = 0
    for name, module in modules.items():
        stats = {"cases": 0, "mixed_equal": 0, "inject_equal": 0, "normed_equal": 0,
                 "act_equal": 0, "max_abs_mixed": 0.0, "max_abs_inject": 0.0, "by_rows": {}}
        for case in range(a.cases):
            key = mx.random.key(7919 * case + len(name))
            k1, k2 = mx.random.split(key)
            scale = scales[case % len(scales)]
            shape = SHAPES[case % len(SHAPES)]
            x = mx.random.normal((*shape, width), key=k1) * scale
            if case % 3 == 2:  # heavy-tailed residual streams
                x = x * mx.exp(mx.random.normal((*shape, width), key=k2))
            x = x.astype(mx.bfloat16)
            mx.eval(x)
            ref = composed(module, x)
            got = fused(module, x)
            ref = ref if isinstance(ref, tuple) else (ref,)
            got = got if isinstance(got, tuple) else (got,)
            mx.eval(ref, got)
            stats["cases"] += 1
            ok_mixed = bits_equal(ref[0], got[0])
            stats["mixed_equal"] += ok_mixed
            stats["max_abs_mixed"] = max(stats["max_abs_mixed"], float(
                mx.abs(ref[0].astype(mx.float32) - got[0].astype(mx.float32)).max().item()))
            assert got[1] is x if len(got) == 3 else True
            if len(ref) == 3:
                ok_inj = bits_equal(ref[2], got[2])
                stats["inject_equal"] += ok_inj
                stats["max_abs_inject"] = max(stats["max_abs_inject"], float(
                    mx.abs(ref[2].astype(mx.float32) - got[2].astype(mx.float32)).max().item()))
            else:
                ok_inj = True
                stats["inject_equal"] += 1
            # intermediates
            normed = module.hc_norm(x)
            act = nn.silu(module.input_mix_weight_down(normed) / module.hc_count)
            rows = shape[0] * shape[1]
            _m, _i, xn, kact = HCD.hc_decode_launch(module, x.reshape(rows, width), debug=True)
            mx.eval(normed, act, xn, kact)
            stats["by_rows"][rows] = stats["by_rows"].get(rows, 0) + 1
            stats["normed_equal"] += bits_equal(normed.reshape(rows, width), xn)
            stats["act_equal"] += bits_equal(act.reshape(kact.shape), kact)
            if not (ok_mixed and ok_inj):
                total_mismatch += 1
        report["modules"][name] = stats
        print(name, json.dumps(stats), flush=True)
    status = HCD.hc_decode_status()
    report["exact"] = {"all_bit_identical": total_mismatch == 0,
                       "mismatched_cases": total_mismatch, "status": status}
    print("EXACT", total_mismatch == 0, "mismatched", total_mismatch, "status", status, flush=True)

    # ---- row-exact windows -----------------------------------------------
    if a.row_exact_cases:
        from mlx2.runtime import row_exact_verify as REV

        HCD.set_hc_row_exact_enabled(True)
        re_stats = {"cases": 0, "rows_equal": 0, "rows": 0, "stages": {}}
        for name, module in modules.items():
            for case in range(a.row_exact_cases):
                rows = (2, 3, 5, 8, 17)[case % 5]
                x = (mx.random.normal((1, rows, width), key=mx.random.key(104729 + case))
                     * scales[case % len(scales)]).astype(mx.bfloat16)
                mx.eval(x)
                with Q._declared_width(1):
                    want = [composed(module, x[:, r:r + 1]) for r in range(rows)]
                    record = REV.Window(rows)
                    with REV.window(record):
                        got = fused(module, x)
                got = got if isinstance(got, tuple) else (got,)
                want = [w if isinstance(w, tuple) else (w,) for w in want]
                mx.eval(got, want)
                re_stats["cases"] += 1
                for r in range(rows):
                    ok = bits_equal(got[0][:, r:r + 1], want[r][0])
                    if len(got) == 3:
                        ok = ok and bits_equal(got[2][:, r:r + 1], want[r][2])
                    re_stats["rows"] += 1
                    re_stats["rows_equal"] += ok
                for stage, routes in record.stages.items():
                    for route, n in routes.items():
                        key = f"{stage}:{route}"
                        re_stats["stages"][key] = re_stats["stages"].get(key, 0) + n
        HCD.set_hc_row_exact_enabled(False)
        re_stats["all_rows_equal"] = re_stats["rows_equal"] == re_stats["rows"]
        report["row_exact_window"] = re_stats
        print("ROWEXACT", json.dumps(re_stats), flush=True)

    # ---- launches ----------------------------------------------------------
    def launches(fn, module, seq=1):
        x = mx.random.normal((1, seq, width)).astype(mx.bfloat16)
        mx.eval(x)
        out = fn(module, x)
        outs = list(out) if isinstance(out, tuple) else [out]
        outs = [o for o in outs if o is not x]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "g.dot")
            mx.export_to_dot(path, *outs)
            labels = re.findall(r'label ="([^"]+)"', open(path).read())
        counted = [l for l in labels if l not in NO_LAUNCH]
        return {"launches": len(counted), "ops": sorted(counted)}

    trunk = modules[f"language_model.model.layers.{a.layers[0]}.attn_hyper_connection"]
    mixer = modules["language_model.model.hyper_connection_mixer"]
    report["launches_per_call"] = {
        "composed_combine": launches(composed, trunk),
        "fused_combine": launches(fused, trunk),
        "composed_mixer": launches(composed, mixer),
        "fused_mixer": launches(fused, mixer),
        "composed_combine_verify3": launches(composed, trunk, 3),
        "fused_combine_verify3": launches(fused, trunk, 3),
    }
    for k, v in report["launches_per_call"].items():
        print("LAUNCHES", k, v["launches"], v["ops"], flush=True)

    # ---- micro -------------------------------------------------------------
    chain_modules = [m for n, m in modules.items() if "layers." in n and "mtp" not in n]

    def chain(on, x):
        HCD.set_hc_decode_enabled(on)
        for i in range(a.chain):
            m = chain_modules[i % len(chain_modules)]
            mixed, residual, inject = m(x)
            x = Q._apply_inject(residual, mixed, inject)
        HCD.set_hc_decode_enabled(False)
        return x

    x0 = (mx.random.normal((1, 1, width)) * 0.5).astype(mx.bfloat16)
    mx.eval(x0)
    ref_final = chain(False, x0)
    got_final = chain(True, x0)
    mx.eval(ref_final, got_final)
    report["chain_bit_identical"] = bits_equal(ref_final, got_final)
    print("CHAIN identical", report["chain_bit_identical"], flush=True)
    times = {"composed": [], "fused": []}
    arms = ["composed", "fused"]
    for rep in range(a.reps + 2):
        order = arms if rep % 2 == 0 else arms[::-1]
        for arm in order:
            mx.synchronize()
            t0 = time.perf_counter()
            y = chain(arm == "fused", x0)
            mx.eval(y)
            dt = time.perf_counter() - t0
            if rep >= 2:
                times[arm].append(1e6 * dt / a.chain)
    report["micro_us_per_call"] = {
        arm: {"median": statistics.median(v), "min": min(v), "max": max(v)} for arm, v in times.items()}
    report["micro_speedup"] = (report["micro_us_per_call"]["composed"]["median"]
                               / report["micro_us_per_call"]["fused"]["median"])
    print("MICRO", json.dumps(report["micro_us_per_call"]), "speedup", report["micro_speedup"], flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
