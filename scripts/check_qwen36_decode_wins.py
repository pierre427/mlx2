"""Coordinator-owned real-weight per-call gate; never runs without ownership.

Exit 0 requires bit-identical finite MoE rows and GDN output/state/rollback
with positive engagement. JSON is written on either success or failure.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 10, 20, 39])
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument(
        "--components",
        nargs="+",
        choices=("moe", "gdn", "router"),
        default=["moe", "gdn", "router"],
    )
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if a.cases < 1:
        ap.error("--cases must be positive")
    import mlx.core as mx

    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

    adapter = Qwen3635BA3BAdapter(
        a.model, execution_policy={"eager_dispatch_stride": 0}
    )
    from mlx2.runtime import verify_scope
    from mlx2.runtime.models import qwen3_next as N
    from mlx2.runtime.models import qwen4_moe_window as W
    from mlx2.runtime.models import qwen4_routed_decode as RD
    from mlx2.runtime.models import qwen36_35b as Q
    from mlx2.runtime.models import qwen36_moe_decode as M
    from mlx2.runtime.models.cache import ArraysCache
    from mlx2.runtime.models.precise_ops import gate_sigmoid

    mx.set_default_device(mx.gpu)
    record = {
        "artifact": adapter.identity,
        "source_sha256": {},
        "checks": [],
        "qualification": "unqualified",
        "components": a.components,
    }
    root = Path(__file__).resolve().parents[1]
    for path in [
        *root.glob("src/mlx2/runtime/models/qwen36*.py"),
        root / "src/mlx2/runtime/models/qwen4_fused_gdn_verify.py",
        root / "src/mlx2/runtime/models/qwen4_moe_window.py",
        root / "src/mlx2/runtime/models/qwen4_routed_decode.py",
    ]:
        record["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    def equal(label, want, got):
        mx.eval(want, got)
        finite = bool(
            mx.all(mx.isfinite(want)).item() and mx.all(mx.isfinite(got)).item()
        )
        dtype = mx.uint16 if want.dtype in (mx.bfloat16, mx.float16) else mx.uint32
        exact = (
            want.shape == got.shape
            and want.dtype == got.dtype
            and bool(mx.array_equal(want.view(dtype), got.view(dtype)).item())
        )
        record["checks"].append({"case": label, "finite": finite, "bit_equal": exact})
        if not (finite and exact):
            raise AssertionError(label)

    try:
        blocks = [
            m
            for _, m in adapter.model.named_modules()
            if isinstance(m, M.Qwen36SparseMoeBlock)
        ]
        gdns = [
            m
            for _, m in adapter.model.named_modules()
            if isinstance(m, Q.GatedDeltaNet)
        ]
        if not blocks or not gdns:
            raise AssertionError("no model layers")
        selected = sorted(set(a.layers))
        if any(i < 0 or i >= len(blocks) for i in selected):
            raise ValueError("invalid MoE layer index")
        routed_checks = 0
        for li in selected if any(c in a.components for c in ("moe", "router")) else []:
            block = blocks[li]
            sw = block.switch_mlp
            for rows in [1, 3, 4, 16]:
                for seed in range(a.cases):
                    x = (
                        mx.random.normal(
                            (rows, 1, 2048), key=mx.random.key(1000 + seed)
                        ).astype(mx.bfloat16)
                        * 0.1
                    )
                    flat = x.reshape(rows, 2048)
                    logits = mx.concatenate(
                        [block.gate(x[r : r + 1]).reshape(1, -1) for r in range(rows)]
                    )
                    inds, scores = N._stock_routing(block, logits)
                    launch = W.router_topk(logits, top_k=8)
                    equal(
                        f"router/{li}/{rows}/{seed}/indices",
                        inds.astype(mx.uint32),
                        launch[0],
                    )
                    equal(f"router/{li}/{rows}/{seed}/scores", scores, launch[1])
                    skew_logits = mx.broadcast_to(
                        mx.where(
                            mx.arange(256) < 8,
                            mx.array(8, mx.bfloat16),
                            mx.array(-8, mx.bfloat16),
                        )[None, :],
                        (rows, 256),
                    )
                    skew_ref = N._stock_routing(block, skew_logits)
                    skew_launch = W.router_topk(skew_logits, top_k=8)
                    equal(
                        f"router-skew/{li}/{rows}/{seed}/indices",
                        skew_ref[0].astype(mx.uint32),
                        skew_launch[0],
                    )
                    equal(
                        f"router-skew/{li}/{rows}/{seed}/scores",
                        skew_ref[1],
                        skew_launch[1],
                    )
                    if "moe" not in a.components:
                        continue
                    refusal = RD.admit_split_routed_decode(
                        x[:1],
                        inds[:1],
                        scores[:1],
                        sw.gate_proj,
                        sw.up_proj,
                        sw.down_proj,
                    )
                    if not refusal.accepted:
                        # Formats like UD-Q8 are negative gates, not silent success.
                        record.setdefault("refusals", []).append(
                            {"layer": li, "reason": refusal.reason}
                        )
                        continue
                    for routing in ["natural", "skewed"]:
                        ii = (
                            inds
                            if routing == "natural"
                            else mx.broadcast_to(
                                mx.arange(8, dtype=mx.uint32)[None, :], (rows, 8)
                            )
                        )
                        for shared in [False, True]:
                            reason = (
                                M.shared_admission(block.shared_expert, 2048, 512)
                                if shared
                                else None
                            )
                            if reason:
                                record.setdefault("shared_refusals", []).append(
                                    {"layer": li, "reason": reason}
                                )
                                continue
                            want = []
                            sg = mx.concatenate(
                                [
                                    gate_sigmoid(
                                        block.shared_expert_gate(x[r : r + 1])
                                    ).reshape(1, 1)
                                    for r in range(rows)
                                ]
                            )
                            for r in range(rows):
                                h = N.SwiGLU()(
                                    sw.up_proj(
                                        mx.expand_dims(x[r : r + 1], (-2, -3)),
                                        ii[r : r + 1],
                                    ),
                                    sw.gate_proj(
                                        mx.expand_dims(x[r : r + 1], (-2, -3)),
                                        ii[r : r + 1],
                                    ),
                                )
                                yy = sw.down_proj(h, ii[r : r + 1]).squeeze(-2)
                                yy = (yy * scores[r : r + 1, :, None]).sum(-2)
                                if shared:
                                    yy = yy + sg[r : r + 1] * block.shared_expert(
                                        x[r : r + 1]
                                    ).reshape(1, -1)
                                want.append(yy.reshape(1, -1))
                            got = M.routed_rows(
                                flat,
                                ii,
                                scores,
                                sw.gate_proj,
                                sw.up_proj,
                                sw.down_proj,
                                shared=block.shared_expert if shared else None,
                                shared_gate=sg if shared else None,
                            )
                            equal(
                                f"moe/{li}/{rows}/{seed}/{routing}/shared={shared}",
                                mx.concatenate(want),
                                got,
                            )
                            routed_checks += 1
                    # End-to-end adapter layer plumbing, natural routing.
                    block.set_moe_routed_decode_mode("off")
                    block.set_moe_window_consumers(())
                    block.set_moe_topk_mode("off")
                    want = mx.concatenate(
                        [block(x[r : r + 1]).reshape(1, -1) for r in range(rows)]
                    )
                    block.set_moe_routed_decode_mode("gate_up_down_shared")
                    block.set_moe_window_consumers(
                        ("batch_decode", "verify", "row_exact")
                    )
                    block.set_moe_topk_mode("launch")
                    got = block(x).reshape(rows, -1)
                    equal(f"block/{li}/{rows}/{seed}", want, got)
                    if block.qwen36_decode_calls < 1:
                        raise AssertionError("MoE did not engage")
                    if rows > 1:
                        with verify_scope.verify_forward():
                            got = block(x.reshape(1, rows, 2048)).reshape(rows, -1)
                        equal(f"verify-window/{li}/{rows}/{seed}", want, got)
        if "moe" in a.components and not routed_checks:
            raise AssertionError(
                "no admitted routed real-weight cases; artifact format unsupported"
            )

        # Real GDN projections and weights; stock per-row reference, every
        # acceptance length including zero/full plus continuation.
        for li in (
            sorted(set(min(i, len(gdns) - 1) for i in selected))
            if "gdn" in a.components
            else []
        ):
            layer = gdns[li]
            for rows in [1, 4, 16]:
                for steps in [1, 3]:
                    for seed in range(a.cases):
                        x = (
                            mx.random.normal(
                                (rows, steps, 2048), key=mx.random.key(2000 + seed)
                            ).astype(mx.bfloat16)
                            * 0.1
                        )
                        initial = [
                            mx.random.normal(
                                (rows, 3, 8192), key=mx.random.key(3000 + seed)
                            ).astype(mx.bfloat16)
                            * 0.01,
                            mx.random.normal(
                                (rows, 32, 128, 128), key=mx.random.key(4000 + seed)
                            )
                            * 0.01,
                        ]

                        def cache(initial=initial, steps=steps):
                            c = ArraysCache(size=2)
                            c[0], c[1] = initial
                            if steps > 1:
                                c.start_speculation()
                            return c

                        layer.set_fused_gdn_decode_mode("stock")
                        layer.set_fused_gdn_batch_decode_mode("off")
                        layer.set_fused_gdn_verify_mode("off")
                        layer.set_fused_gdn_batch_verify_mode("off")
                        reference = cache()
                        # Full-batch stock reference checks lifecycle and mask
                        # semantics; the row oracle separately checks arithmetic.
                        want = layer(x, cache=reference)
                        mx.eval(want, *reference.state)
                        layer.set_fused_gdn_decode_mode("fused")
                        layer.set_fused_gdn_batch_decode_mode("row_exact")
                        layer.set_fused_gdn_verify_mode("row_exact")
                        layer.set_fused_gdn_batch_verify_mode("row_exact")
                        fused = cache()
                        got = layer(x, cache=fused)
                        equal(f"gdn/{li}/{rows}/{steps}/{seed}/out", want, got)
                        for j in [0, 1]:
                            equal(
                                f"gdn/{li}/{rows}/{steps}/{seed}/state{j}",
                                reference[j],
                                fused[j],
                            )
                        choice = (
                            "batch_verify"
                            if rows > 1 and steps > 1
                            else (
                                "verify"
                                if steps > 1
                                else ("batch_decode" if rows > 1 else "decode")
                            )
                        )
                        if getattr(layer, "fused_gdn_" + choice + "_calls") < 1:
                            raise AssertionError("GDN did not engage")
                        if steps > 1:
                            for accepted in range(steps + 1):
                                # Fresh transactions: rollback itself consumes records.
                                layer.set_fused_gdn_verify_mode("off")
                                layer.set_fused_gdn_batch_verify_mode("off")
                                rc = cache()
                                ro = layer(x, cache=rc)
                                mx.eval(ro, *rc.state)
                                layer.set_fused_gdn_verify_mode("row_exact")
                                layer.set_fused_gdn_batch_verify_mode("row_exact")
                                fc = cache()
                                fo = layer(x, cache=fc)
                                mx.eval(fo, *fc.state)
                                rc.trim(steps - accepted)
                                fc.trim(steps - accepted)
                                rc.stop_speculation()
                                fc.stop_speculation()
                                for j in [0, 1]:
                                    equal(
                                        f"rollback/{li}/{rows}/{seed}/{accepted}/{j}",
                                        rc[j],
                                        fc[j],
                                    )
                                token = x[:, :1]
                                equal(
                                    f"continuation/{li}/{rows}/{seed}/{accepted}",
                                    layer(token, cache=rc),
                                    layer(token, cache=fc),
                                )
        record["counters"] = Q.qwen36_decode_wins_stats(adapter.model)
        record["verdict"] = "pass"
    except Exception as exc:
        record["verdict"] = "fail"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        adapter.close()
        Path(a.out).write_text(json.dumps(record, indent=2) + "\n")
    print(record["verdict"])
    return 0 if record["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
