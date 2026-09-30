"""Metal bit-exactness gate for the batched one-token fused GDN decode.

Loads real Flash-Next GatedDeltaNet layers (projections, conv, A_log,
dt_bias, norm) from the artifact safetensors, gives every row its own history
of a different length (ragged conv/recurrent states), then compares with
``mx.array_equal`` on the GPU:

* kernel:  ``qwen4_fused_gdn_batch_decode`` row r vs the B=1
  ``qwen4_fused_gdn_decode`` launch on row r's identical inputs (output, next
  conv state, next recurrent state);
* layer:   the full layer forward at B rows with the batched route vs
  (a) each row alone through the B=1 fused route, (b) the stock batched chain,
  (c) each row alone through the stock chain;
* churn:   a batch that loses a lane (``filter``) and gains one (``extend``)
  mid-decode, per-row against each lane's own B=1 fused trajectory.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python \\
      scripts/check_qwen4_gdn_batch_decode.py --i-own-the-gpu \\
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --out gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 44])
    ap.add_argument("--rows", type=int, nargs="+", default=[2, 3, 4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lane-exact", action="store_true",
                    help="also rerun every check with lane-exact (row-invariant) projections")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    args = ap.parse_args()
    if not args.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        ap.error("set MLX_ENABLE_TF32=0 (the adapter pins it)")

    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.models.cache import ArraysCache
    from mlx2.runtime.models.qwen4_exp import GatedDeltaNet, TextModelArgs
    from mlx2.runtime.models.qwen4_fused_gdn import (
        probe_qwen4_fused_gdn_decode,
        qwen4_fused_gdn_batch_decode,
        qwen4_fused_gdn_decode,
    )

    mx.set_cache_limit(4 << 30)
    model_path = Path(args.model).expanduser()
    config = json.loads((model_path / "config.json").read_text())
    text = TextModelArgs.from_dict(config["text_config"])
    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    hidden = int(config["text_config"]["hidden_size"])
    tg_y = probe_qwen4_fused_gdn_decode(mx.bfloat16)
    assert tg_y is not None, "B=1 fused GDN probe declined on this device"

    def load_layer(index):
        prefix = f"language_model.model.layers.{index}.linear_attn."
        shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
        weights = {}
        for shard in shards:
            loaded = mx.load(str(model_path / shard))
            weights.update(
                {k[len(prefix):]: v for k, v in loaded.items() if k.startswith(prefix)}
            )
            del loaded
        layer = GatedDeltaNet(text)
        quant = config["quantization"]
        nn.quantize(
            layer,
            group_size=quant["group_size"],
            bits=quant["bits"],
            class_predicate=lambda path, module: hasattr(module, "to_quantized")
            and f"{path}.scales" in weights,
        )
        layer.load_weights(list(weights.items()), strict=True)
        layer.eval()
        mx.eval(layer.parameters())
        return layer

    def set_route(layer, route):
        # route: batch | stock_batch | b1 | stock
        layer.set_fused_gdn_decode_mode("stock" if route == "stock" else "fused")
        layer.set_fused_gdn_batch_decode_mode("row_exact" if route == "batch" else "off")

    def fresh_cache(conv=None, state=None):
        cache = ArraysCache(size=2)
        cache[0], cache[1] = conv, state
        return cache

    def same(x, y):
        return bool(x.shape == y.shape and x.dtype == y.dtype and mx.array_equal(x, y).item())

    def history(layer, steps, key):
        """One lane's own state after ``steps`` B=1 fused decode tokens."""
        set_route(layer, "b1")
        cache = fresh_cache()
        xs = (mx.random.normal((steps, 1, 1, hidden), key=key) * 0.8).astype(mx.bfloat16)
        for t in range(steps):
            out = layer(xs[t], cache=cache)
            mx.eval(out, cache[0], cache[1])
        return cache[0], cache[1]

    def run_law(layer, lanes, keys, law):
        """All checks for one projection law (stock qmv, or lane exact)."""
        entry = {"rows": {}, "churn": None}
        for rows in args.rows:
            conv = mx.concatenate([lanes[r][0] for r in range(rows)])
            state = mx.concatenate([lanes[r][1] for r in range(rows)])
            xs = (mx.random.normal((args.steps, rows, 1, hidden),
                                   key=mx.random.split(keys[-1], rows + 1)[rows]) * 0.8).astype(mx.bfloat16)
            # Kernel level: identical inputs, batched launch vs B=1 launches.
            qkv, z, b, a = layer._input_projections(xs[0])
            batch_out = qwen4_fused_gdn_batch_decode(
                qkv, z, b, a, conv, layer.conv1d.weight, layer.A_log, layer.dt_bias,
                state, layer.norm.weight, layer.norm.eps, threadgroup_y=tg_y)
            mx.eval(*batch_out)
            kernel_equal = True
            projections_equal = True
            for r in range(rows):
                one = qwen4_fused_gdn_decode(
                    qkv[r:r + 1], z[r:r + 1], b[r:r + 1], a[r:r + 1], conv[r:r + 1],
                    layer.conv1d.weight, layer.A_log, layer.dt_bias, state[r:r + 1],
                    layer.norm.weight, layer.norm.eps, threadgroup_y=tg_y)
                mx.eval(*one)
                kernel_equal &= all(same(x[r:r + 1], y) for x, y in zip(batch_out, one))
                solo = layer._input_projections(xs[0][r:r + 1])
                projections_equal &= all(same(x[r:r + 1], y) for x, y in zip((qkv, z, b, a), solo))
                projections_equal &= same(layer.out_proj(batch_out[0])[r:r + 1],
                                          layer.out_proj(batch_out[0][r:r + 1]))
            # Layer level over several steps.
            trajectories = {}
            for route in ("batch", "stock_batch", "b1", "stock"):
                set_route(layer, route)
                calls0 = layer.fused_gdn_batch_decode_calls
                if route in ("batch", "stock_batch"):
                    cache = fresh_cache(conv, state)
                    outs = []
                    for t in range(args.steps):
                        o = layer(xs[t], cache=cache)
                        mx.eval(o, cache[0], cache[1])
                        outs.append(o)
                    trajectories[route] = (mx.concatenate(outs, axis=1), cache[0], cache[1])
                else:
                    per_row = []
                    for r in range(rows):
                        cache = fresh_cache(conv[r:r + 1], state[r:r + 1])
                        outs = []
                        for t in range(args.steps):
                            o = layer(xs[t][r:r + 1], cache=cache)
                            mx.eval(o, cache[0], cache[1])
                            outs.append(o)
                        per_row.append((mx.concatenate(outs, axis=1), cache[0], cache[1]))
                    trajectories[route] = tuple(
                        mx.concatenate([p[i] for p in per_row]) for i in range(3))
                if route == "batch":
                    engaged = layer.fused_gdn_batch_decode_calls - calls0
                    assert engaged == args.steps, (engaged, layer.fused_gdn_batch_decode_last_fallback)

            def cmp(x, y):
                return {name: same(p, q) for name, p, q in zip(
                    ("output", "conv_state", "recurrent_state"), trajectories[x], trajectories[y])}

            def max_abs(x, y):
                return float(mx.max(mx.abs(trajectories[x][0].astype(mx.float32)
                                           - trajectories[y][0].astype(mx.float32))).item())

            row_entry = {
                "kernel_batch_vs_b1_launch": kernel_equal,
                "projections_row_invariant": projections_equal,
                "layer_batch_vs_b1_fused": cmp("batch", "b1"),
                "layer_batch_vs_stock_batch": cmp("batch", "stock_batch"),
                "layer_stock_batch_vs_stock_b1": cmp("stock_batch", "stock"),
                "layer_b1_fused_vs_stock_b1": cmp("b1", "stock"),
                "max_abs_output_batch_vs_b1_fused": max_abs("batch", "b1"),
                "max_abs_output_stock_batch_vs_stock_b1": max_abs("stock_batch", "stock"),
            }
            entry["rows"][rows] = row_entry
            print(f"{law} B={rows}", json.dumps(row_entry), flush=True)
            mx.clear_cache()

        # Churn: 4 lanes decode, lane 1 finishes, a fifth lane joins.  The
        # batched route is compared per row with the stock batched chain on
        # the same membership schedule and with each lane's own B=1 fused
        # trajectory (equal only under row-invariant projections).
        order = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3], [0, 2, 3, 4], [0, 2, 3, 4], [0, 2, 3, 4]]
        xs = (mx.random.normal((len(order), 6, 1, hidden), key=keys[-2]) * 0.8).astype(mx.bfloat16)

        def churn(route):
            set_route(layer, route)
            calls0 = layer.fused_gdn_batch_decode_calls
            live = [0, 1, 2, 3]
            cache = fresh_cache(mx.concatenate([lanes[r][0] for r in live]),
                                mx.concatenate([lanes[r][1] for r in live]))
            rows_out = {r: [] for r in range(5)}
            for step, members in enumerate(order):
                if members != live:
                    keep = [live.index(r) for r in members if r in live]
                    if len(keep) != len(live):
                        cache.filter(keep)
                    live = [r for r in live if r in members]
                    for r in members:
                        if r not in live:
                            cache.extend(fresh_cache(lanes[r][0], lanes[r][1]))
                            live.append(r)
                    assert live == members
                x = mx.concatenate([xs[step, r][None] for r in live])
                o = layer(x, cache=cache)
                mx.eval(o, cache[0], cache[1])
                for i, r in enumerate(live):
                    rows_out[r].append(o[i:i + 1])
            final = {r: (cache[0][i:i + 1], cache[1][i:i + 1]) for i, r in enumerate(live)}
            return rows_out, final, layer.fused_gdn_batch_decode_calls - calls0

        fused_rows, fused_final, engaged = churn("batch")
        stock_rows, stock_final, _ = churn("stock_batch")
        vs_stock = all(same(p, q) for r in fused_rows for p, q in zip(fused_rows[r], stock_rows[r]))
        vs_stock &= all(same(fused_final[r][i], stock_final[r][i]) for r in fused_final for i in (0, 1))
        set_route(layer, "b1")
        vs_b1 = True
        for r in range(5):
            steps = [s for s, m in enumerate(order) if r in m]
            cache = fresh_cache(lanes[r][0], lanes[r][1])
            for k, s in enumerate(steps):
                o = layer(xs[s, r][None], cache=cache)
                mx.eval(o, cache[0], cache[1])
                vs_b1 &= same(o, fused_rows[r][k])
            if r in fused_final:
                vs_b1 &= same(cache[0], fused_final[r][0]) and same(cache[1], fused_final[r][1])
        entry["churn"] = {"batch_vs_stock_batch": vs_stock, "batch_vs_b1_fused": vs_b1,
                          "batched_calls": engaged, "steps": len(order)}
        print(f"{law} churn", json.dumps(entry["churn"]), flush=True)
        return entry

    report = {"model": str(model_path), "threadgroup_y": tg_y, "mlx": mx.__version__,
              "rows": args.rows, "steps": args.steps, "laws": {}, "all_equal": {}}
    laws = ["stock"] + (["lane_exact"] if args.lane_exact else [])
    for law in laws:
        report["laws"][law] = {}
    for layer_index in args.layers:
        t0 = time.time()
        layer = load_layer(layer_index)
        max_rows = max(args.rows + [6])
        keys = mx.random.split(mx.random.key(args.seed * 1000 + layer_index), max_rows + 2)
        lanes = [history(layer, 1 + (7 * r) % 23, keys[r]) for r in range(max_rows)]
        for law in laws:
            if law == "lane_exact":
                from mlx2.runtime.lane import installer as lane_installer
                from mlx2.runtime.lane.matmul import MAX_ROWS as LANE_MAX_ROWS

                receipt = lane_installer.install(layer, min_rows=1, max_rows=LANE_MAX_ROWS)
                print("lane install", json.dumps(receipt, default=str)[:400], flush=True)
            entry = run_law(layer, lanes, keys, f"layer {layer_index} {law}")
            entry["seconds"] = round(time.time() - t0, 1)
            report["laws"][law][layer_index] = entry
        del layer
        mx.clear_cache()

    for law, layers in report["laws"].items():
        summary = {}
        for key in ("kernel_batch_vs_b1_launch", "projections_row_invariant",
                    "layer_batch_vs_b1_fused", "layer_batch_vs_stock_batch",
                    "layer_stock_batch_vs_stock_b1", "layer_b1_fused_vs_stock_b1"):
            values = []
            for entry in layers.values():
                for row in entry["rows"].values():
                    value = row[key]
                    values.append(all(value.values()) if isinstance(value, dict) else value)
            summary[key] = all(values)
        for key in ("batch_vs_stock_batch", "batch_vs_b1_fused"):
            summary["churn_" + key] = all(e["churn"][key] for e in layers.values())
        report["all_equal"][law] = summary
    report["peak_gib"] = mx.get_peak_memory() / 2**30
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["all_equal"], indent=1))


if __name__ == "__main__":
    main()
