"""Metal gate for routed decode on the served Flash-Next tables (split gate/up).

The served artifact loads split gate/up expert tables
(``MLX_QWEN4_MOE_FUSED_GATE_UP=0``), so the fused-table #3912 kernel never
engages there. This checks the split-table routed path against the block the
Flash-Next profile actually runs, using REAL layer weights read from the
artifact's safetensors (routed tables, router gate, shared expert, shared
gate) for a few layers, including the MTP layer:

  block        Qwen3NextSparseMoeBlock with routed decode ``off`` (reference:
               gather_qmm gate, gather_qmm up, swiglu, tile4 fused down)
  gate_up      split gate+up/SwiGLU kernel, then the block's tile4 down
  gate_up_down split gate+up kernel + served_down (tile4 arithmetic, 2 or 4
               rows per threadgroup, one-expert views on or off)
  gate_up_down_shared  the same with the shared expert, its 8-bit gate and
               the combine folded into the two launches (block-level; on the
               switch-level synthetic cases it runs as gate_up_down)

Per case it also compares the isolated pieces: the SwiGLU hidden against
``swiglu(gather_qmm gate, gather_qmm up)`` and served_down against
``qwen4_fused_down(variant="tile4")`` on the same hidden.

Cases: natural routing (the block's own router on random activations) and
synthetic skewed routings fed to the switch directly: high expert ids (the
one-expert views must index past the view), one dominant score, near-uniform
scores, and consecutive ids.

Verdict ``pass`` only if every comparison is bit-identical (uint16 views) and
every routed arm engaged on every case; otherwise ``fail`` with max_abs_diff.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/check_fn_split_routed_decode.py \
      --i-own-the-gpu --out split-routed-exact.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MLX_ENABLE_TF32", "0")
# The served Flash-Next profile (adapters/flash_next.py) for the MoE block.
for name in tuple(os.environ):
    if name.startswith(("MLX_QWEN", "MLX_LM_")):
        del os.environ[name]
os.environ["MLX_QWEN4_MOE_FUSED_GATE_UP"] = "0"
os.environ["MLX_QWEN4_FUSED_EXPERT_KERNEL"] = "auto"

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

MODEL = Path.home() / "mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
E, H, I, TOPK = 512, 2560, 640, 10


def load_layer(prefix, index, cache):
    """{relative key: array} for one MoE block from the artifact."""
    wanted = {k: f for k, f in index.items() if k.startswith(prefix + ".mlp.")}
    out = {}
    for key, fname in wanted.items():
        if fname not in cache:
            cache[fname] = mx.load(str(MODEL / fname))
        out[key[len(prefix) + len(".mlp."):]] = cache[fname][key]
    return out


def qsl(w, s, b):
    from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear

    m = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
    nn.Module.__init__(m)
    m.weight, m.scales, m.biases = w, s, b
    m.group_size, m.bits, m.mode = 64, 4, "affine"
    m.freeze()
    return m


def ql(tensors, name, bits):
    w = tensors[f"{name}.weight"]
    out_dims, packed = w.shape
    m = nn.QuantizedLinear(packed * 32 // bits, out_dims, bias=False, group_size=64, bits=bits)
    m.weight, m.scales, m.biases = w, tensors[f"{name}.scales"], tensors[f"{name}.biases"]
    m.freeze()
    return m


def build_block(tensors):
    from mlx2.runtime.models import qwen3_next as QN

    args = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
        norm_topk_prob=True, num_experts=E, num_experts_per_tok=TOPK,
    )
    block = QN.Qwen3NextSparseMoeBlock(args)  # random switch tables stay lazy
    assert not block.fused_gate_up and block.fused_expert_kernel_mode == "auto"
    sw = block.switch_mlp
    for proj in ("gate_proj", "up_proj", "down_proj"):
        setattr(sw, proj, qsl(*(tensors[f"switch_mlp.{proj}.{p}"] for p in ("weight", "scales", "biases"))))
    block.gate = ql(tensors, "gate", 8)
    block.shared_expert_gate = ql(tensors, "shared_expert_gate", 8)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        setattr(block.shared_expert, proj, ql(tensors, f"shared_expert.{proj}", 4))
    block.eval()
    mx.eval(block.parameters())
    return block


def bits_equal(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def maxdiff(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item())


def synthetic_routes(seed):
    """Skewed one-token routings: (name, indices uint32 [1,1,10], scores bf16)."""
    key = mx.random.key(seed)
    k1, k2, k3 = mx.random.split(key, 3)
    routes = []
    high = mx.array([[list(range(E - 1, E - 1 - TOPK, -1))]], mx.uint32)
    routes.append(("high_ids", high, mx.softmax(mx.random.normal((1, 1, TOPK), key=k1), -1)))
    dom = mx.array([0.91] + [0.01] * 9)[None, None]
    rand_ids = mx.random.permutation(E, key=k2)[:TOPK].astype(mx.uint32)[None, None]
    routes.append(("dominant", rand_ids, dom))
    # contiguous copies: the router hands over forward-strided rows. (A
    # reversed [..., ::-1] view is misread by the tile4 reference itself; see
    # tile4_strided_diagnostic.)
    routes.append(("uniform", mx.contiguous(rand_ids[..., ::-1]), mx.full((1, 1, TOPK), 0.1)))
    start = int(mx.random.randint(0, E - TOPK, (), key=k3).item())
    routes.append(("consecutive", mx.arange(start, start + TOPK, dtype=mx.uint32)[None, None],
                   mx.softmax(mx.random.normal((1, 1, TOPK), key=k3) * 3, -1)))
    return [(n, i, (s / s.sum(-1, keepdims=True)).astype(mx.bfloat16)) for n, i, s in routes]


def chain_bench(a, index, cache):
    """Launch-chain microbench: ``--chain`` full MoE blocks (router, shared
    expert, combine) cycling over ``--bench-layers`` distinct real layers, one
    eval per chain (the decode dependency shape), arms rotated per rep after
    a discarded warm-up. Synthetic evidence only; the model A/B decides."""
    import statistics
    import time

    from mlx2.runtime.models import qwen4_routed_decode as RD

    blocks = []
    for i in range(a.bench_layers):
        layer = (47 * i) // max(1, a.bench_layers - 1)
        blocks.append(build_block(load_layer(f"language_model.model.layers.{layer}", index, cache)))
    xs = [(mx.random.normal((1, 1, H), key=mx.random.key(20000 + i))).astype(mx.bfloat16)
          for i in range(a.chain)]
    mx.eval(xs)
    arms = {
        "off": ("off", 2, True),
        "gate_up": ("gate_up", 2, True),
        "gate_up_noviews": ("gate_up", 2, False),
        "gate_up_down": ("gate_up_down", 2, True),
        "gate_up_down_rows4": ("gate_up_down", 4, True),
        "gate_up_down_noviews": ("gate_up_down", 2, False),
        "gate_up_down_shared": ("gate_up_down_shared", 2, True),
    }

    def chain(arm):
        mode, rows, views = arms[arm]
        RD.set_served_down_rows(rows)
        RD.set_expert_views(views)
        for b in blocks:
            b.set_moe_routed_decode_mode(mode)
        y = xs[0]
        for i in range(a.chain):
            y = blocks[i % len(blocks)]((y + xs[i]).astype(mx.bfloat16))
        t0 = time.perf_counter()
        mx.eval(y)
        return time.perf_counter() - t0, y

    names = list(arms)
    outs = {}
    for arm in names:
        _, outs[arm] = chain(arm)
    same = {arm: bits_equal(outs[arm], outs["off"]) for arm in names}
    times = {arm: [] for arm in names}
    for rep in range(a.reps):
        order = names[rep % len(names):] + names[: rep % len(names)]
        if rep % 2:
            order = order[::-1]
        for arm in order:
            times[arm].append(chain(arm)[0] * 1e3)
    for b in blocks:
        b.set_moe_routed_decode_mode("off")
    RD.set_served_down_rows(2)
    RD.set_expert_views(True)
    base = statistics.median(times["off"])
    summary = {arm: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v),
                     "us_per_block": 1e3 * statistics.median(v) / a.chain,
                     "delta_vs_off_pct": 100 * (statistics.median(v) / base - 1),
                     "chain_output_bit_identical_to_off": same[arm]}
               for arm, v in times.items()}
    print(json.dumps(summary, indent=1), flush=True)
    return {"layers": a.bench_layers, "chain": a.chain, "reps": a.reps, "chain_ms": times,
            "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--layers", nargs="+", default=[
        "language_model.model.layers.0", "language_model.model.layers.23",
        "language_model.model.layers.47", "mtp.layers.0"])
    ap.add_argument("--cases", type=int, default=16, help="natural-routing cases per layer")
    ap.add_argument("--synthetic-seeds", type=int, default=4)
    ap.add_argument("--bench-layers", type=int, default=0,
                    help="after the gate: chain-bench this many distinct real layers")
    ap.add_argument("--chain", type=int, default=48)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if not mx.metal.is_available():
        sys.exit("Metal unavailable")
    mx.set_cache_limit(4 << 30)

    from mlx2.runtime.models import qwen4_routed_decode as RD
    from mlx2.runtime.models.qwen4_fused_moe import qwen4_fused_down

    index = json.load(open(MODEL / "model.safetensors.index.json"))["weight_map"]
    cache = {}
    arms = [("gate_up", 2, True), ("gate_up_down", 2, True), ("gate_up_down", 4, True),
            ("gate_up_down", 2, False), ("gate_up_down", 4, False),
            ("gate_up_down_shared", 2, True), ("gate_up_down_shared", 4, True),
            ("gate_up_down_shared", 2, False)]
    counts = {}
    diffs = {}
    engaged_ok = True
    per_layer = {}

    def tally(name, same, diff):
        counts.setdefault(name, [0, 0])
        counts[name][0] += int(same)
        counts[name][1] += 1
        diffs[name] = max(diffs.get(name, 0.0), diff)

    for prefix in a.layers:
        tensors = load_layer(prefix, index, cache)
        block = build_block(tensors)
        sw = block.switch_mlp
        views_seen = set()
        n_cases = 0

        def run_arm(mode, rows, views, fn, natural):
            RD.set_served_down_rows(rows)
            RD.set_expert_views(views)
            block.set_moe_routed_decode_mode(mode)
            before = (sw.routed_decode_calls, sw.routed_down_calls, block.shared_fold_calls)
            out = fn()
            mx.eval(out)
            after = (sw.routed_decode_calls, sw.routed_down_calls, block.shared_fold_calls)
            want_down = 1 if mode.startswith("gate_up_down") else 0
            # the fold is block-level: natural cases only
            want_fold = 1 if mode == "gate_up_down_shared" and natural else 0
            ok = (after[0] - before[0] == 1 and after[1] - before[1] == want_down
                  and after[2] - before[2] == want_fold)
            if not ok:
                print("NOT ENGAGED", mode, rows, views, natural, before, after,
                      block.shared_fold_last_fallback, sw.routed_down_last_fallback, flush=True)
            block.set_moe_routed_decode_mode("off")
            return out, ok

        cases = []
        for c in range(a.cases):
            scale = [0.5, 1.0, 2.0, 4.0][c % 4]
            x = (mx.random.normal((1, 1, H), key=mx.random.key(7000 + c)) * scale).astype(mx.bfloat16)
            cases.append(("natural", x, None, None))
        for seed in range(a.synthetic_seeds):
            x = (mx.random.normal((1, 1, H), key=mx.random.key(9100 + seed))).astype(mx.bfloat16)
            for name, inds, scores in synthetic_routes(seed):
                cases.append((name, x, inds, scores))

        for kind, x, inds, scores in cases:
            n_cases += 1
            if kind == "natural":
                block.set_moe_routed_decode_mode("off")
                ref = block(x)
                mx.eval(ref)
                call = lambda: block(x)  # noqa: E731
                # the block's own routing, for the piecewise checks
                gates = mx.softmax(block.gate(x), axis=-1, precise=True)
                inds = mx.argpartition(gates, kth=-TOPK, axis=-1)[..., -TOPK:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                scores = scores / scores.sum(axis=-1, keepdims=True)
            else:
                block.set_moe_routed_decode_mode("off")
                ref = sw(x, inds, scores=scores, variant="auto")
                mx.eval(ref)
                call = lambda: sw(x, inds, scores=scores, variant="auto")  # noqa: E731
            for mode, rows, views in arms:
                out, ok = run_arm(mode, rows, views, call, kind == "natural")
                engaged_ok &= ok
                name = f"{mode}/rows{rows}/views{int(views)}=={'block' if kind == 'natural' else 'switch'}"
                tally(name, bits_equal(out, ref), maxdiff(out, ref))
            # piecewise: hidden and down on the same inputs
            xe = mx.expand_dims(x, (-2, -3))
            ref_h = sw.activation(sw.up_proj(xe, inds), sw.gate_proj(xe, inds)).reshape(TOPK, I)
            for views in (True, False):
                RD.set_expert_views(views)
                ker_h = RD.split_gate_up_swiglu(xe, inds, sw.gate_proj, sw.up_proj)
                mx.eval(ker_h)
                tally(f"hidden/views{int(views)}", bits_equal(ker_h, ref_h), maxdiff(ker_h, ref_h))
                tile4 = qwen4_fused_down(
                    ref_h.reshape(1, 1, TOPK, I), inds.reshape(1, 1, TOPK), scores.reshape(1, 1, TOPK),
                    sw.down_proj["weight"], sw.down_proj["scales"], sw.down_proj["biases"], variant="tile4",
                ).reshape(H)
                for rows in (2, 4):
                    got = RD.served_down(ref_h, inds, scores.astype(mx.bfloat16), sw.down_proj, rows=rows)
                    mx.eval(got, tile4)
                    tally(f"served_down/rows{rows}/views{int(views)}==tile4", bits_equal(got, tile4), maxdiff(got, tile4))
            views_seen.add(tuple(RD.expert_view_count(p) for p in (sw.gate_proj, sw.up_proj, sw.down_proj)))
        per_layer[prefix] = {"cases": n_cases, "view_counts": sorted(views_seen),
                             "switch_types": [type(getattr(sw, p)).__name__ for p in ("gate_proj", "up_proj", "down_proj")]}
        print(prefix, json.dumps(per_layer[prefix]), flush=True)
        print(json.dumps({k: f"{v[0]}/{v[1]}" for k, v in counts.items()}), flush=True)
        del block, sw, tensors
        cache.clear()
        mx.clear_cache()

    RD.set_expert_views(True)
    RD.set_served_down_rows(2)
    # Diagnostic only (not in the verdict): the existing tile4 kernel reads
    # indices through a 32-bit elem_to_loc; a reversed view has a negative
    # stride. Compare tile4 on a reversed view vs its contiguous copy.
    tensors = load_layer(a.layers[0], index, cache)
    sw = build_block(tensors).switch_mlp
    rev = mx.random.permutation(E, key=mx.random.key(5))[:TOPK].astype(mx.uint32)[None, None][..., ::-1]
    sc = mx.softmax(mx.random.normal((1, 1, TOPK), key=mx.random.key(6)), -1).astype(mx.bfloat16)
    xe = mx.expand_dims(mx.random.normal((1, 1, H), key=mx.random.key(7)).astype(mx.bfloat16), (-2, -3))
    hh = sw.activation(sw.up_proj(xe, mx.contiguous(rev)), sw.gate_proj(xe, mx.contiguous(rev))).squeeze(-2)
    d = sw.down_proj
    t_view = qwen4_fused_down(hh, rev, sc, d["weight"], d["scales"], d["biases"], variant="tile4")
    t_copy = qwen4_fused_down(hh, mx.contiguous(rev), sc, d["weight"], d["scales"], d["biases"], variant="tile4")
    mx.eval(t_view, t_copy)
    tile4_diag = {"reversed_view_equals_contiguous": bits_equal(t_view, t_copy),
                  "max_abs_diff": maxdiff(t_view, t_copy)}
    del sw, tensors
    cache.clear()
    all_exact = all(v[0] == v[1] for v in counts.values())
    rec = {
        "verdict": "pass" if all_exact and engaged_ok else "fail",
        "all_bit_identical": all_exact,
        "every_arm_engaged": engaged_ok,
        "bit_identical_counts": {k: f"{v[0]}/{v[1]}" for k, v in counts.items()},
        "max_abs_diff": diffs,
        "layers": per_layer,
        "tile4_strided_diagnostic": tile4_diag,
        "mlx": mx.__version__,
        "device": mx.device_info().get("device_name"),
        "model": str(MODEL),
    }
    if a.bench_layers:
        rec["chain_bench"] = chain_bench(a, index, cache)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("verdict", "bit_identical_counts", "max_abs_diff")}, indent=1))
    sys.exit(0 if rec["verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
