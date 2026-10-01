"""Metal gate for the routed MoE row window and the router top-k launch/fold.

Real Flash-Next layer weights (``check_fn_split_routed_decode`` loads one MoE
block from the artifact's safetensors: routed tables, 8-bit router, shared
expert, 8-bit shared gate) for layers 0, 23, 47 and ``mtp.layers.0``.

Reference: the served one-token block (routed decode ``off``: gather_qmm
gate/up, swiglu, tile4 fused down, eager shared expert and combine, stock
routing), called on each row alone as a [1, 1, hidden] input.

Checks (all bit-identity on uint16/uint32 views):

window       the block on R rows (R = 2, 3, 4, 8, 16, 17) through each
             consumer (batch_decode [R, 1, H], verify [1, R, H] inside the
             verify scope, row_exact inside a row-exact window), for top-k
             off / launch / fold and the shared fold on / off, vs the R
             one-token reference calls. Natural rows (random activations at
             scales 0.5..4, plus windows of repeated rows).
kernels      ``routed_rows`` / ``shared_rows`` on skewed routings (high ids,
             one dominant score, uniform scores, consecutive ids, every row
             the same experts) vs the one-token kernels per row
             (``split_gate_up_swiglu`` + ``served_down``, ``shared_fold_decode``).
routing      ``router_topk`` and the fold's stored (indices, scores) vs the
             stock softmax/argpartition/normalize, per row, on the router's
             natural logits and on synthetic logits with heavy ties; and the
             stock routing of R rows vs the same rows one at a time.
b1           one-token top-k ``launch`` (routed off / gate_up_down /
             gate_up_down_shared) and ``fold`` (gate_up_down /
             gate_up_down_shared) vs the reference block.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/check_fn_moe_window.py \
      --i-own-the-gpu --out moe-window-exact.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_fn_split_routed_decode as base  # noqa: E402  (sets the served env)

import mlx.core as mx  # noqa: E402

H, TOPK, E = base.H, base.TOPK, base.E


def u16(a):
    return a.astype(mx.bfloat16).view(mx.uint16) if a.dtype != mx.uint32 else a


def same(a, b):
    a, b = mx.contiguous(a), mx.contiguous(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(u16(a), u16(b)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--layers", nargs="+", default=[
        "language_model.model.layers.0", "language_model.model.layers.23",
        "language_model.model.layers.47", "mtp.layers.0"])
    ap.add_argument("--rows", type=int, nargs="+", default=[2, 3, 4, 8, 16, 17])
    ap.add_argument("--cases", type=int, default=3, help="natural windows per (layer, R)")
    ap.add_argument("--b1-cases", type=int, default=12)
    ap.add_argument("--routing-rows", type=int, default=4096, help="synthetic routing rows per layer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if not mx.metal.is_available():
        sys.exit("Metal unavailable")
    mx.set_cache_limit(4 << 30)

    from mlx2.runtime import row_exact_verify as REV
    from mlx2.runtime import verify_scope
    from mlx2.runtime.models import qwen3_next as QN
    from mlx2.runtime.models import qwen4_moe_window as W
    from mlx2.runtime.models import qwen4_routed_decode as RD

    index = json.load(open(base.MODEL / "model.safetensors.index.json"))["weight_map"]
    cache = {}
    counts, notes = {}, []

    def tally(name, ok):
        c = counts.setdefault(name, [0, 0])
        c[0] += int(ok)
        c[1] += 1
        if not ok and len(notes) < 40:
            notes.append(name)
            print("MISMATCH", name, flush=True)

    def reference(block, rows_x):
        block.set_moe_window_consumers(())
        block.set_moe_topk_mode("off")
        block.set_moe_routed_decode_mode("off")
        outs = [block(rows_x[r].reshape(1, 1, H)) for r in range(rows_x.shape[0])]
        out = mx.concatenate([o.reshape(1, H) for o in outs], axis=0)
        mx.eval(out)
        return out

    def stock_routing(block, gates):
        return QN._stock_routing(block, gates)

    for prefix in a.layers:
        block = base.build_block(base.load_layer(prefix, index, cache))
        sw = block.switch_mlp
        short = prefix.replace("language_model.model.", "")
        # ---------------------------------------------------------------- window
        for R in a.rows:
            windows = []
            for c in range(a.cases):
                scales = mx.array([[0.5, 1.0, 2.0, 4.0][(c + r) % 4] for r in range(R)])[:, None]
                x = (mx.random.normal((R, H), key=mx.random.key(100 * R + c)) * scales).astype(mx.bfloat16)
                windows.append(("natural", x))
            one = (mx.random.normal((1, H), key=mx.random.key(999 + R))).astype(mx.bfloat16)
            windows.append(("repeated", mx.repeat(one, R, axis=0)))
            for kind, x in windows:
                ref = reference(block, x)
                for consumer in ("batch_decode", "verify", "row_exact"):
                    for topk in W.TOPK_MODES:
                        for shared in (True, False):
                            W.set_window_shared(shared)
                            block.set_moe_window_consumers({consumer})
                            block.set_moe_topk_mode(topk)
                            before = (dict(block.moe_window_calls), block.moe_window_shared_calls,
                                      dict(block.moe_topk_calls))
                            if consumer == "batch_decode":
                                out = block(x.reshape(R, 1, H))
                            elif consumer == "verify":
                                with verify_scope.verify_forward():
                                    out = block(x.reshape(1, R, H))
                            else:
                                with REV.window(REV.Window(R)):
                                    out = block(x.reshape(1, R, H))
                            mx.eval(out)
                            engaged = (block.moe_window_calls[consumer] == before[0][consumer] + 1
                                       and block.moe_window_shared_calls == before[1] + int(shared)
                                       and (topk == "off" or block.moe_topk_calls[topk] == before[2][topk] + 1))
                            if not engaged:
                                print("NOT ENGAGED", short, R, consumer, topk, shared,
                                      block.moe_window_last_fallback, flush=True)
                            tally(f"window/R{R}/{consumer}/topk_{topk}/shared{int(shared)}/engaged", engaged)
                            tally(f"window/R{R}/{consumer}/topk_{topk}/shared{int(shared)}=={kind}_one_token",
                                  same(out.reshape(R, H), ref))
                # negative control: the comparison must see a one-ulp change
                bumped = (out.reshape(R, H).view(mx.uint16) ^ mx.array(1, mx.uint16)).view(mx.bfloat16)
                tally(f"control/R{R}/bit_flip_detected", not same(bumped, ref))
                W.set_window_shared(True)
                block.set_moe_window_consumers(())
                block.set_moe_topk_mode("off")
                # routing of R rows: stock batched vs one row at a time; launch and fold vs stock
                gates = block.gate(x)
                bi, bs = stock_routing(block, gates)
                rows_i, rows_s = zip(*[stock_routing(block, gates[r:r + 1]) for r in range(R)])
                tally(f"routing/R{R}/stock_rows==stock_one_row",
                      same(bi, mx.concatenate(rows_i)) and same(bs, mx.concatenate(rows_s)))
                li, ls = W.router_topk(gates)
                tally(f"routing/R{R}/launch==stock", same(li, bi) and same(ls, bs))
                _, fi, fs = W.routed_rows(x, None, None, sw.gate_proj, sw.up_proj, sw.down_proj, logits=gates)
                tally(f"routing/R{R}/fold_stored==stock", same(fi, bi) and same(fs, bs))
                _, gi, gs = W.shared_rows(x, None, None, sw.gate_proj, sw.up_proj, sw.down_proj,
                                          block.shared_expert, block.shared_expert_gate(x), logits=gates)
                tally(f"routing/R{R}/shared_fold_stored==stock", same(gi, bi) and same(gs, bs))
            # ---------------------------------------------------------- kernels
            x = (mx.random.normal((R, H), key=mx.random.key(5000 + R))).astype(mx.bfloat16)
            routes = [base.synthetic_routes(R * 10 + r)[r % 4] for r in range(R)]
            if R > 2:
                routes[-1] = routes[0]  # two rows on the same experts
            inds = mx.concatenate([i.reshape(1, TOPK) for _, i, _ in routes]).astype(mx.uint32)
            scores = mx.concatenate([s.reshape(1, TOPK) for _, _, s in routes])
            # one-row gate logits: the stock multi-row 8-bit matmul is not
            # row-invariant from 16 rows on (the window computes them with the
            # row-exact qmv kernel)
            logit = mx.concatenate([block.shared_expert_gate(x[r:r + 1]) for r in range(R)])
            got_r = W.routed_rows(x, inds, scores, sw.gate_proj, sw.up_proj, sw.down_proj)
            got_s = W.shared_rows(x, inds, scores, sw.gate_proj, sw.up_proj, sw.down_proj,
                                  block.shared_expert, logit)
            ref_r, ref_s = [], []
            for r in range(R):
                xe = x[r].reshape(1, 1, 1, 1, H)
                h = RD.split_gate_up_swiglu(xe, inds[r:r + 1], sw.gate_proj, sw.up_proj)
                ref_r.append(RD.served_down(h, inds[r], scores[r], sw.down_proj).reshape(1, H))
                ref_s.append(RD.shared_fold_decode(
                    x[r].reshape(1, 1, H), inds[r], scores[r], sw.gate_proj, sw.up_proj, sw.down_proj,
                    block.shared_expert, block.shared_expert_gate(x[r].reshape(1, 1, H))).reshape(1, H))
            tally(f"kernels/R{R}/routed_rows==one_token_kernels", same(got_r, mx.concatenate(ref_r)))
            tally(f"kernels/R{R}/shared_rows==one_token_fold", same(got_s, mx.concatenate(ref_s)))
            for rps in RD.DOWN_ROWS_SERVED_CHOICES:
                got = W.routed_rows(x, inds, scores, sw.gate_proj, sw.up_proj, sw.down_proj, rows_per_tg=rps)
                tally(f"kernels/R{R}/routed_rows_rps{rps}==one_token_kernels", same(got, mx.concatenate(ref_r)))
            print(short, "R", R, json.dumps({k: f"{v[0]}/{v[1]}" for k, v in counts.items()
                                               if f"/R{R}/" in k and not k.endswith("engaged")}), flush=True)
        # ---------------------------------------------------------- b1 top-k
        for c in range(a.b1_cases):
            scale = [0.5, 1.0, 2.0, 4.0][c % 4]
            x = (mx.random.normal((1, 1, H), key=mx.random.key(31000 + c)) * scale).astype(mx.bfloat16)
            ref = reference(block, x.reshape(1, H))
            for topk, routed in (("launch", "off"), ("launch", "gate_up_down"), ("launch", "gate_up_down_shared"),
                                 ("fold", "gate_up_down"), ("fold", "gate_up_down_shared")):
                block.set_moe_routed_decode_mode(routed)
                block.set_moe_topk_mode(topk)
                before = block.moe_topk_calls[topk]
                out = block(x)
                mx.eval(out)
                tally(f"b1/topk_{topk}/routed_{routed}/engaged", block.moe_topk_calls[topk] == before + 1)
                tally(f"b1/topk_{topk}/routed_{routed}==reference", same(out.reshape(1, H), ref))
            block.set_moe_routed_decode_mode("off")
            block.set_moe_topk_mode("off")
        # ------------------------------------------------- synthetic routing
        n = a.routing_rows
        keys = mx.random.split(mx.random.key(77), 3)
        synth = {
            "normal": mx.random.normal((n, E), key=keys[0]) * 2,
            "coarse_ties": mx.round(mx.random.normal((n, E), key=keys[1]) * 4) / 4,
            "few_values": mx.random.randint(0, 3, (n, E), key=keys[2]).astype(mx.float32),
        }
        nat_x = (mx.random.normal((n, H), key=mx.random.key(4242)) *
                 mx.array([0.5, 1.0, 2.0, 4.0])[mx.arange(n) % 4][:, None]).astype(mx.bfloat16)
        synth["natural"] = block.gate(nat_x)
        for name, logits in synth.items():
            logits = logits.astype(mx.bfloat16)
            si, ss = stock_routing(block, logits[:, None, :])
            si, ss = si.reshape(n, TOPK), ss.reshape(n, TOPK)
            for start in range(0, n, 17):
                chunk = logits[start:start + 17]
                li, ls = W.router_topk(chunk)
                tally(f"routing_synth/{name}/launch==stock", same(li, si[start:start + 17]) and same(ls, ss[start:start + 17]))
            for start in range(0, min(n, 17 * 24), 17):
                chunk = logits[start:start + 17]
                xx = nat_x[start:start + chunk.shape[0]]
                _, fi, fs = W.routed_rows(xx, None, None, sw.gate_proj, sw.up_proj, sw.down_proj, logits=chunk)
                tally(f"routing_synth/{name}/fold==stock",
                      same(fi, si[start:start + chunk.shape[0]]) and same(fs, ss[start:start + chunk.shape[0]]))
        print(short, json.dumps({k: f"{v[0]}/{v[1]}" for k, v in counts.items() if k.startswith(("b1", "routing_synth"))}), flush=True)
        block = sw = None
        cache.clear()
        mx.clear_cache()

    all_ok = all(v[0] == v[1] for v in counts.values())
    rec = {
        "verdict": "pass" if all_ok else "fail",
        "counts": {k: f"{v[0]}/{v[1]}" for k, v in counts.items()},
        "first_mismatches": notes,
        "layers": a.layers,
        "rows": a.rows,
        "mlx": mx.__version__,
        "device": mx.device_info().get("device_name"),
        "model": str(base.MODEL),
    }
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({"verdict": rec["verdict"], "failing": [k for k, v in counts.items() if v[0] != v[1]]}, indent=1))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
