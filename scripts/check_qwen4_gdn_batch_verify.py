"""Metal bit-exactness gate for the batched fused GDN verify (B lanes x S rows).

Loads real Flash-Next GatedDeltaNet layers from the artifact safetensors (conv,
A_log, dt_bias, norm and the real input/output projections), gives every lane
its own history (conv window and recurrent state from B=1 fused decode steps of
different lengths), and compares with ``mx.array_equal`` on the GPU:

* kernel:  ``qwen4_fused_gdn_batch_verify`` / ``..._batch_replay_verify`` lane r
  (``row_steps[r]`` valid rows of a ragged ``(B, S)`` block) vs the B=1
  ``qwen4_fused_gdn_verify`` / ``..._replay_verify`` launch at width
  ``row_steps[r]`` on lane r's inputs: output rows, next conv window, next
  recurrent state, every snapshot / tape step; a one-row lane vs the B=1
  one-token fused decode kernel; padded rows are zeros;
* rollback: per-lane partial accepts (a different m per lane) rebuilt in one
  batched dynamic-count reconstruct vs that lane's B=1 restore (host-int
  template reconstruct and the B=1 dynamic form);
* decode chain: for one geometry, every lane's rows vs that lane's tokens run
  one at a time through the B=1 one-token fused decode kernel;
* layer:   the GatedDeltaNet layer with the batched route on a merged
  ``ArraysCache`` and on the served ``SegmentedBatchArraysCache`` over B1 row
  caches: ragged verify -> per-lane partial accept (``trim_ragged``) -> second
  ragged verify -> (segmented) lane churn (a lane leaves, a new one joins) ->
  third verify, every lane's GDN output rows, conv window and recurrent state
  vs the same lane alone on a B=1 cache through the B=1 fused verify (one-row
  lanes: the B=1 one-token step), in compact-tape and snapshot rollback modes.
  The layer's projections are replaced by one-token calls so the comparison is
  of the GDN core (stock projections are not row-invariant; that is outside
  this route).

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python \\
      scripts/check_qwen4_gdn_batch_verify.py --i-own-the-gpu \\
      --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP --out gate.json
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
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 44])
    ap.add_argument("--rows", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--steps", type=int, nargs="+", default=[3, 9, 17])
    ap.add_argument("--layer-rows", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--layer-steps", type=int, nargs="+", default=[3, 9, 17])
    ap.add_argument("--st16", action="store_true", help="also gate the fp16 state class")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    args = ap.parse_args()
    if not args.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        ap.error("set MLX_ENABLE_TF32=0 (the adapter pins it)")

    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.models import qwen4_fused_gdn_verify as V
    from mlx2.runtime.models.cache import ArraysCache
    from mlx2.runtime.models.qwen4_exp import GatedDeltaNet, TextModelArgs
    from mlx2.runtime.models.qwen4_fused_gdn import (
        probe_qwen4_fused_gdn_decode,
        qwen4_fused_gdn_decode,
    )
    from mlx2.runtime.segmented_batch_cache import SegmentedBatchArraysCache

    mx.set_cache_limit(4 << 30)
    V.set_verify_max_steps(V.MAX_VERIFY_WIDTH_PROVEN)
    model_path = Path(args.model).expanduser()
    config = json.loads((model_path / "config.json").read_text())
    text = TextModelArgs.from_dict(config["text_config"])
    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    hidden = int(config["text_config"]["hidden_size"])
    rng = mx.random.key(args.seed)

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

        def predicate(path, module):
            if not hasattr(module, "to_quantized") or f"{path}.scales" not in weights:
                return False
            per = quant.get(f"language_model.model.layers.{index}.linear_attn.{path}")
            if isinstance(per, dict):
                return {"group_size": per["group_size"], "bits": per["bits"]}
            return True

        nn.quantize(layer, group_size=quant["group_size"], bits=quant["bits"],
                    class_predicate=predicate)
        layer.load_weights(list(weights.items()), strict=True)
        layer.eval()
        mx.eval(layer.parameters())
        return layer

    def same(x, y):
        return bool(x.shape == y.shape and x.dtype == y.dtype and mx.array_equal(x, y).item())

    def key():
        nonlocal rng
        rng, sub = mx.random.split(rng)
        return sub

    failures = []

    def check(name, ok):
        if not ok:
            failures.append(name)
        return ok

    # ---- per-lane histories ------------------------------------------------
    def history(layer, steps, state_dtype):
        layer.set_fused_gdn_decode_mode("fused")
        layer._gdn_state_dtype = state_dtype
        cache = ArraysCache(size=2)
        xs = (mx.random.normal((steps, 1, 1, hidden), key=key()) * 0.8).astype(mx.bfloat16)
        for t in range(steps):
            out = layer(xs[t], cache=cache)
            mx.eval(out, cache[0], cache[1])
        layer._gdn_state_dtype = None
        return cache[0], cache[1]

    def spans_for(rows, steps, pattern):
        if pattern == "full":
            return [steps] * rows
        # ragged: lane 0 at the full width, the rest spread over 1..steps
        values = [steps]
        for r in range(1, rows):
            values.append(1 + (5 * r + 3 * steps) % steps)
        return values

    # ---- kernel level ------------------------------------------------------
    def kernel_checks(layer, lanes, rows, steps, pattern, st16):
        tg_compact = V.probe_qwen4_fused_gdn_batch_verify(
            mx.bfloat16, steps, compact=True, **({"state_dtype": mx.float16} if st16 else {}))
        tg_snap = V.probe_qwen4_fused_gdn_batch_verify(
            mx.bfloat16, steps, compact=False, **({"state_dtype": mx.float16} if st16 else {}))
        assert tg_compact is not None and tg_snap is not None, "batched verify probe declined"
        tg_dec = probe_qwen4_fused_gdn_decode(
            mx.bfloat16, **({"state_dtype": mx.float16} if st16 else {}))
        spans = spans_for(rows, steps, pattern)
        conv = mx.concatenate([lanes[r][0] for r in range(rows)])
        state = mx.concatenate([lanes[r][1] for r in range(rows)])
        xs = (mx.random.normal((rows, steps, hidden), key=key()) * 0.8).astype(mx.bfloat16)
        qkv, z, b, a = layer._input_projections(xs)
        mx.eval(qkv, z, b, a)
        common = (layer.conv1d.weight, layer.A_log, layer.dt_bias)
        snap = V.qwen4_fused_gdn_batch_verify(
            qkv, z, b, a, conv, *common, state, layer.norm.weight, layer.norm.eps,
            spans, threadgroup_y=tg_snap)
        tape = V.qwen4_fused_gdn_batch_replay_verify(
            qkv, z, b, a, conv, *common, state, layer.norm.weight, layer.norm.eps,
            spans, threadgroup_y=tg_compact)
        mx.eval(*snap, *tape)
        result = {"spans": spans, "lanes": {}}
        accepts = []
        for r, span in enumerate(spans):
            lane = {}
            sl = (slice(r, r + 1), slice(0, span))
            if span == 1:
                one = qwen4_fused_gdn_decode(
                    qkv[sl], z[sl], b[sl], a[sl], conv[r:r + 1], *common, state[r:r + 1],
                    layer.norm.weight, layer.norm.eps, threadgroup_y=tg_dec)
                mx.eval(*one)
                for name, batched in (("snapshot", snap), ("compact", tape)):
                    lane[f"{name}_vs_decode_kernel"] = all(
                        same(x, y) for x, y in zip(
                            (batched[0][sl], batched[1][r:r + 1], batched[2][r:r + 1]), one))
                accepts.append(0)
            else:
                ref = V.qwen4_fused_gdn_verify(
                    qkv[sl], z[sl], b[sl], a[sl], conv[r:r + 1], *common, state[r:r + 1],
                    layer.norm.weight, layer.norm.eps, threadgroup_y=tg_snap)
                ref_tape = V.qwen4_fused_gdn_replay_verify(
                    qkv[sl], z[sl], b[sl], a[sl], conv[r:r + 1], *common, state[r:r + 1],
                    layer.norm.weight, layer.norm.eps, threadgroup_y=tg_compact)
                mx.eval(*ref, *ref_tape)
                lane["snapshot_vs_b1"] = {
                    "output": same(snap[0][sl], ref[0]),
                    "conv": same(snap[1][r:r + 1], ref[1]),
                    "state": same(snap[2][r:r + 1], ref[2]),
                    "state_snapshots": same(snap[3][r:r + 1, : span - 1], ref[3]),
                    "conv_snapshots": same(snap[4][r:r + 1, : span - 1], ref[4]),
                }
                lane["compact_vs_b1"] = {
                    "output": same(tape[0][sl], ref_tape[0]),
                    "conv": same(tape[1][r:r + 1], ref_tape[1]),
                    "state": same(tape[2][r:r + 1], ref_tape[2]),
                    "keys": same(tape[3][r:r + 1, : span - 1], ref_tape[3]),
                    "corrections": same(tape[4][r:r + 1, : span - 1], ref_tape[4]),
                    "decay": same(tape[5][r:r + 1, : span - 1], ref_tape[5]),
                }
                m = 1 + (3 * r + steps) % (span - 1)
                accepts.append(m)
                b1_template = V.qwen4_fused_gdn_reconstruct(
                    state[r:r + 1], ref_tape[3], ref_tape[4], ref_tape[5], m,
                    threadgroup_y=tg_compact)
                b1_dynamic = V.qwen4_fused_gdn_reconstruct(
                    state[r:r + 1], ref_tape[3], ref_tape[4], ref_tape[5],
                    mx.array([m], dtype=mx.int32), threadgroup_y=tg_compact)
                lane["accept"] = m
                lane["b1_restore"] = (b1_template, b1_dynamic, ref[3][:, m - 1])
            lane["padded_rows_zero"] = all(
                bool(mx.all(t[0][r, span:] == 0).item()) for t in (snap, tape))
            result["lanes"][r] = lane
        # Falsifier: one ulp-scale change to lane 0's state must be visible
        # to the same comparison, or "equal" above would mean nothing.
        # (2^-20 relative is below fp16 resolution: the fp16 class nudges by 2^-9.)
        eps = 2.0**-9 if state.dtype == mx.float16 else 2.0**-20
        nudged = mx.concatenate([(state[:1] * (1 + eps)).astype(state.dtype), state[1:]])
        probe = V.qwen4_fused_gdn_batch_verify(
            qkv, z, b, a, conv, *common, nudged, layer.norm.weight, layer.norm.eps,
            spans, threadgroup_y=tg_snap)
        mx.eval(*probe)
        result["falsifier_nudged_state_detected"] = not same(probe[2][:1], snap[2][:1])
        result["falsifier_other_lanes_unchanged"] = all(
            same(probe[i][1:], snap[i][1:]) for i in range(3))
        batched_restore = V.qwen4_fused_gdn_reconstruct(
            state, tape[3], tape[4], tape[5], mx.array(accepts, dtype=mx.int32),
            threadgroup_y=tg_compact)
        mx.eval(batched_restore)
        for r, lane in result["lanes"].items():
            if "b1_restore" in lane:
                t1, d1, s1 = lane.pop("b1_restore")
                mx.eval(t1, d1)
                lane["restore_vs_b1_template"] = same(batched_restore[r:r + 1], t1)
                lane["restore_vs_b1_dynamic"] = same(batched_restore[r:r + 1], d1)
                lane["restore_vs_snapshot"] = same(batched_restore[r:r + 1], s1)
            else:
                lane["restore_count0_keeps_checkpoint"] = same(
                    batched_restore[r:r + 1], state[r:r + 1])
        ok = (result["falsifier_nudged_state_detected"]
              and result["falsifier_other_lanes_unchanged"])
        for lane in result["lanes"].values():
            for value in lane.values():
                if isinstance(value, dict):
                    ok &= all(value.values())
                elif isinstance(value, bool):
                    ok &= value
        result["all_equal"] = ok
        return result, (qkv, z, b, a, conv, state, spans, snap, tg_dec)

    def decode_chain_check(layer, bundle):
        """Every lane's rows vs its tokens through the one-token fused decode."""
        qkv, z, b, a, conv, state, spans, snap, tg_dec = bundle
        common = (layer.conv1d.weight, layer.A_log, layer.dt_bias)
        ok = True
        for r, span in enumerate(spans):
            c, s = conv[r:r + 1], state[r:r + 1]
            for t in range(span):
                sl = (slice(r, r + 1), slice(t, t + 1))
                o, c, s = qwen4_fused_gdn_decode(
                    qkv[sl], z[sl], b[sl], a[sl], c, *common, s,
                    layer.norm.weight, layer.norm.eps, threadgroup_y=tg_dec)
                mx.eval(o, c, s)
                ok &= same(o, snap[0][sl])
                if t < span - 1:
                    ok &= same(s, snap[3][r:r + 1, t])
            ok &= same(c, snap[1][r:r + 1]) and same(s, snap[2][r:r + 1])
        return ok

    # ---- layer level -------------------------------------------------------
    class Capture(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.seen = []

        def __call__(self, x):
            self.seen.append(x)
            return self.inner(x)

    def per_token_projections(layer):
        original = GatedDeltaNet._input_projections

        def project(inputs):
            rows = []
            for r in range(inputs.shape[0]):
                tokens = [original(layer, inputs[r:r + 1, t:t + 1]) for t in range(inputs.shape[1])]
                rows.append(tuple(mx.concatenate([tok[i] for tok in tokens], axis=1) for i in range(4)))
            return tuple(mx.concatenate([row[i] for row in rows], axis=0) for i in range(4))

        return project

    def layer_checks(layer, lanes, rows, steps, rollback_mode):
        """Three ragged rounds through the layer, merged and segmented caches."""
        layer.set_fused_gdn_verify_mode("fused")
        layer.set_fused_gdn_decode_mode("fused")
        layer.set_fused_gdn_replay_rollback_mode(rollback_mode)
        capture = Capture(layer.out_proj)
        original_out = layer.out_proj
        layer.out_proj = capture
        layer._input_projections = per_token_projections(layer)
        n_lanes = rows + 1  # one spare lane joins in round 3
        x_rounds = [
            (mx.random.normal((n_lanes, steps, hidden), key=key()) * 0.8).astype(mx.bfloat16)
            for _ in range(3)
        ]
        members = [list(range(rows)), list(range(rows)),
                   [r for r in range(rows) if r != 1] + [rows]]
        spans_r = [
            spans_for(rows, steps, "ragged"),
            [max(1, steps - (r % 3)) for r in range(rows)],
            [1 + (r * 7) % steps for r in range(len(members[2]))],
        ]
        spans_r[2][0] = steps
        report = {}

        def lane_alone(lane_id):
            """The lane's own B=1 trajectory (B=1 fused verify per round)."""
            layer.set_fused_gdn_batch_verify_mode("off")
            cache = ArraysCache(size=2)
            cache[0], cache[1] = lanes[lane_id]
            cache.start_speculation()
            outs, states = [], []
            for rnd in range(3):
                if lane_id not in members[rnd]:
                    outs.append(None)
                    states.append(None)
                    continue
                i = members[rnd].index(lane_id)
                span = spans_r[rnd][i]
                capture.seen.clear()
                before = layer.fused_gdn_verify_calls
                layer(x_rounds[rnd][lane_id:lane_id + 1, :span], cache=cache)
                if span >= 2:
                    assert layer.fused_gdn_verify_calls == before + 1, layer.fused_gdn_verify_last_fallback
                out = capture.seen[-1]
                accept = accepts[rnd][i]
                if accept < span:
                    cache.trim(span - accept)
                mx.eval(out, cache[0], cache[1])
                outs.append(out)
                states.append((cache[0], cache[1]))
            cache.stop_speculation()
            return outs, states

        accepts = [[1 + (r * 5 + rnd) % span for r, span in enumerate(spans)]
                   for rnd, spans in enumerate(spans_r)]
        alone = {lane_id: lane_alone(lane_id) for lane_id in range(n_lanes)}

        def batched(kind, mode="row_exact"):
            layer.set_fused_gdn_batch_verify_mode(mode)
            calls0 = layer.fused_gdn_batch_verify_calls
            rows_cache = {}
            for lane_id in range(n_lanes):
                c = ArraysCache(size=2)
                c[0], c[1] = lanes[lane_id]
                c.start_speculation()
                rows_cache[lane_id] = c
            merged = None
            results = {lane_id: ([], []) for lane_id in range(n_lanes)}
            last_rnd = 3 if kind == "segmented" else 2
            for rnd in range(last_rnd):
                ids = members[rnd]
                spans = spans_r[rnd]
                width = max(spans)
                if kind == "segmented":
                    view = SegmentedBatchArraysCache([rows_cache[i] for i in ids])
                else:
                    if merged is None:
                        merged = ArraysCache(size=2)
                        merged[0] = mx.concatenate([lanes[i][0] for i in ids])
                        merged[1] = mx.concatenate([lanes[i][1] for i in ids])
                        merged.start_speculation()
                    view = merged
                view.prepare(lengths=spans)
                x = mx.concatenate([x_rounds[rnd][lane_id:lane_id + 1, :width] for lane_id in ids])
                capture.seen.clear()
                mask = view.make_mask(width)
                layer(x, mask=mask, cache=view)
                out = capture.seen[-1]
                view.finalize()
                drops = [span - acc for span, acc in zip(spans, accepts[rnd])]
                view.trim_ragged(drops)
                mx.eval(out, view[0], view[1])
                for i, lane_id in enumerate(ids):
                    results[lane_id][0].append((rnd, out[i:i + 1, : spans[i]]))
                    results[lane_id][1].append((rnd, (view[0][i:i + 1], view[1][i:i + 1])))
            engaged = layer.fused_gdn_batch_verify_calls - calls0
            return results, engaged, last_rnd

        for kind in ("merged", "segmented"):
            results, engaged, n_rounds = batched(kind)
            ok_out = ok_state = True
            for lane_id, (outs, states) in results.items():
                for rnd, out in outs:
                    ok_out &= same(out, alone[lane_id][0][rnd])
                for rnd, (c, s) in states:
                    ref = alone[lane_id][1][rnd]
                    ok_state &= same(c, ref[0]) and same(s, ref[1])
            entry = {
                "outputs_vs_lane_alone": ok_out,
                "states_after_partial_accept_vs_lane_alone": ok_state,
                "batched_calls": engaged,
                "rounds": n_rounds,
                "spans": spans_r[:n_rounds],
                "accepts": accepts[:n_rounds],
            }
            check(f"layer {kind} {rows}x{steps} {rollback_mode}",
                  ok_out and ok_state and engaged == n_rounds)
            report[kind] = entry
        # Informational: the stock multi-row chain (mode off) on the same
        # merged schedule with the same one-token projections, i.e. whether
        # the stock GDN core is itself per-lane exact (not a gate).
        results, _, _ = batched("merged", mode="off")
        report["stock_chain_same_schedule"] = {
            "outputs_vs_lane_alone": all(
                same(out, alone[lane_id][0][rnd])
                for lane_id, (outs, _) in results.items() for rnd, out in outs),
            "states_vs_lane_alone": all(
                same(c, alone[lane_id][1][rnd][0]) and same(s, alone[lane_id][1][rnd][1])
                for lane_id, (_, states) in results.items() for rnd, (c, s) in states),
        }
        layer.out_proj = original_out
        del layer._input_projections
        layer.set_fused_gdn_batch_verify_mode("off")
        return report

    report = {"model": str(model_path), "mlx": mx.__version__, "layers": {},
              "rows": args.rows, "steps": args.steps}
    t_start = time.time()
    for layer_index in args.layers:
        t0 = time.time()
        layer = load_layer(layer_index)
        entry = {"kernel": {}, "decode_chain": None, "layer": {}}
        classes = [None] + ([mx.float16] if args.st16 else [])
        for state_dtype in classes:
            tag = "fp16" if state_dtype is not None else "fp32"
            max_rows = max(args.rows + args.layer_rows) + 1
            lanes = [history(layer, 1 + (7 * r) % 23, state_dtype) for r in range(max_rows)]
            for rows in args.rows:
                for steps in args.steps:
                    for pattern in ("full", "ragged"):
                        if state_dtype is not None and (rows, pattern) != (8, "ragged"):
                            continue
                        res, bundle = kernel_checks(layer, lanes, rows, steps, pattern,
                                                    state_dtype is not None)
                        name = f"{tag} B={rows} S={steps} {pattern}"
                        check(f"layer {layer_index} kernel {name}", res["all_equal"])
                        entry["kernel"][name] = res
                        print(layer_index, name, res["all_equal"], res["spans"], flush=True)
                        if (entry["decode_chain"] is None and state_dtype is None
                                and rows == 4 and steps == 9 and pattern == "ragged"):
                            entry["decode_chain"] = decode_chain_check(layer, bundle)
                            check(f"layer {layer_index} decode chain", entry["decode_chain"])
                            print(layer_index, "decode chain", entry["decode_chain"], flush=True)
                        mx.clear_cache()
            if state_dtype is None:
                for rows in args.layer_rows:
                    for steps in args.layer_steps:
                        for mode in ("compact", "snapshots"):
                            res = layer_checks(layer, lanes, rows, steps, mode)
                            name = f"B={rows} S={steps} {mode}"
                            entry["layer"][name] = res
                            print(layer_index, "layer", name,
                                  {k: (v["outputs_vs_lane_alone"],
                                       v.get("states_after_partial_accept_vs_lane_alone",
                                             v.get("states_vs_lane_alone")),
                                       v.get("batched_calls")) for k, v in res.items()},
                                  flush=True)
                            mx.clear_cache()
        entry["seconds"] = round(time.time() - t0, 1)
        report["layers"][layer_index] = entry
        del layer
        mx.clear_cache()
    report["failures"] = failures
    report["all_equal"] = not failures
    report["seconds"] = round(time.time() - t_start, 1)
    report["peak_gib"] = mx.get_peak_memory() / 2**30
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"all_equal": report["all_equal"], "failures": failures[:20]}, indent=1))


if __name__ == "__main__":
    main()
