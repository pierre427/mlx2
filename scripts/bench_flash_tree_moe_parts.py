"""Flash-Next tree-tile MoE cost split into its parts, on recorded real inputs.

Diagnostic only. One real native-MTP 16-row tile is run with the full
row-invariant stack while every MoE layer's input rows are recorded. Each
part of the device MoE path is then timed alone over all 48 layers in one
eval, from evaluated inputs: router + shared gates, routing kernel, sort
glue, NAX gate/up(+SwiGLU), NAX down, unsort + combine, shared expert.
No qualification or performance claim. GPU lease only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

PROMPT = "Explain how a database index speeds up queries. Use one short paragraph."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip()
    if head != args.source:
        raise RuntimeError(f"source HEAD {head} != {args.source}")
    if args.out.exists():
        raise RuntimeError("refusing an existing evidence directory")
    args.out.mkdir(parents=True)
    report = {"schema": "mlx2.flash-tree-moe-parts-bench.v1", "source_revision": head,
              "checkpoint": f"{args.model.name}@sha256:{args.fingerprint}", "status": "running",
              "loadavg_before": subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip(),
              "parts_ms": {}}
    save = lambda: (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    adapter = None
    try:
        from mlx2.adapters.registry import inspect_model
        if inspect_model(args.model).artifact["fingerprint"] != args.fingerprint:
            raise RuntimeError("checkpoint fingerprint changed")
        import mlx.core as mx
        mx.set_memory_limit(80 << 30)
        from mlx2.adapters.flash_next import FlashNextAdapter
        policy = {"draft_model": str(args.model), "num_draft": 5, "row_exact_verify": False,
                  "progressive_verification_tile": 2, "progressive_alternative_branches": 3,
                  "progressive_alternative_strategy": "confidence_tree", "cost_per_commit_handoff": False,
                  "continuation_pool": {"sources": ["external"], "limit": 3, "ast_planner": True,
                                        "packed_pattern": [5, 5, 5]},
                  "ast_target_tree_oracle": str(args.out / "unused-oracle.json")}
        adapter = FlashNextAdapter(str(args.model), execution_policy=policy)
        model = adapter.model
        from mlx2.adapters import flash_next_tree as T
        from mlx2.runtime.models import flash_tree_mma_law as L, moe_nax_gather as NAX
        from mlx2.runtime.models.activations import swiglu
        from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
        if os.environ.get("MLX_QWEN4_PLE_NVME") != str(args.model / "ple_rows.bin"):
            raise RuntimeError("PLE offload is not active")
        L.FUSED_GLUE = L.CACHED_PADDING = True
        trunk = model.language_model.model
        layers = (len(trunk.layers) - 1,)
        parts = ("projections", "experts_device", "qsa_chains", "router_gates", "router_kernel", "hc_matrix")
        provider = T.FlashNextTargetTreeExecutor(model, tile_law="row_invariant_mma", mma_parts=parts)
        draft = NativeMTPDraftModel(model, num_draft=5, revision=adapter.identity["fingerprint"])
        prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": PROMPT}],
                                             "enable_thinking": False}))
        cache = model.make_cache()
        logits = model(mx.array([prompt]), cache=cache)
        greedy = [int(mx.argmax(logits[0, -1]).item())]
        while len(greedy) < 20:
            logits = model(mx.array([[greedy[-1]]]), cache=cache)
            greedy.append(int(mx.argmax(logits[0, -1]).item()))
        context, anchor = prompt + greedy[:16], greedy[16]
        base = model.make_cache()
        features = model.prefill_body(mx.array([context]), base, layers)
        mx.eval(features)
        slate = draft.draft_packed_paths([anchor], features, draft.make_cache(), (5, 5, 5),
                                         context_tokens=[context[1:] + [anchor]])[0]
        tokens, parents = [anchor], [-1]
        for path in slate:
            parent = 0
            for token in path.tokens:
                parents.append(parent)
                parent = len(tokens)
                tokens.append(token)
        recorded = []
        original = T._moe

        def recorder(block, x, **kw):
            recorded.append((block, x, kw))
            return original(block, x, **kw)

        T._moe = recorder
        try:
            out = provider.forward(model, tokens, parents, base, layers)
            mx.eval([x for _, x, _ in recorded] + [out[0], out[1]])
            out[2].abort()
        finally:
            T._moe = original
        report["moe_calls"] = len(recorded)

        def timed(fn, reps=7):
            samples = []
            for _ in range(reps + 1):
                mx.synchronize(); t0 = time.perf_counter()
                result = fn(); mx.eval(result); mx.synchronize()
                samples.append((time.perf_counter() - t0) * 1000)
            return round(statistics.median(samples[1:]), 2)

        # Inputs of each part, evaluated once.
        stage = []
        for block, x, kw in recorded:
            rows = x.reshape(-1, x.shape[-1])
            gates = L.gate_project(block.gate, x)
            gate = L.gate_project(block.shared_expert_gate, x)
            inds, scores = L.route(block, gates)
            flat = inds.reshape(-1).astype(mx.uint32)
            order = mx.argsort(flat).astype(mx.uint32)
            sorted_experts = mx.take(flat, order)
            row_map = (order // int(inds.shape[-1])).astype(mx.uint32)
            sw = block.switch_mlp
            table = lambda linear: (linear["weight"], linear["scales"], linear.get("biases"))
            fmt = dict(group_size=int(sw.gate_proj.group_size), bits=int(sw.gate_proj.bits), mode="affine")
            h = NAX.sorted_gather_qmm_swiglu_split(rows[:, None, :], table(sw.gate_proj), table(sw.up_proj),
                                                   sorted_experts, row_map=row_map, **fmt)
            d = sw.down_proj
            y = NAX.sorted_gather_qmm(h, d["weight"], d["scales"], d.get("biases"), sorted_experts,
                                      group_size=int(d.group_size), bits=int(d.bits), mode="affine")
            sh = block.shared_expert
            s = L._linear(sh.down_proj, swiglu(L._linear(sh.gate_proj, rows), L._linear(sh.up_proj, rows)))
            stage.append(dict(block=block, x=x, rows=rows, gates=gates, gate=gate, inds=inds, scores=scores,
                              order=order, sorted=sorted_experts, row_map=row_map, h=h, y=y, s=s, fmt=fmt))
            mx.eval([v for v in stage[-1].values() if isinstance(v, mx.array)])
        report["distinct_experts_mean"] = statistics.mean(
            len(set(map(int, st["sorted"].tolist()))) for st in stage)

        def each(fn):
            return lambda: [fn(st) for st in stage]

        def gate_up(st):
            sw = st["block"].switch_mlp
            table = lambda linear: (linear["weight"], linear["scales"], linear.get("biases"))
            return NAX.sorted_gather_qmm_swiglu_split(st["rows"][:, None, :], table(sw.gate_proj), table(sw.up_proj),
                                                      st["sorted"], row_map=st["row_map"], **st["fmt"])

        def down(st):
            d = st["block"].switch_mlp.down_proj
            return NAX.sorted_gather_qmm(st["h"], d["weight"], d["scales"], d.get("biases"), st["sorted"],
                                         group_size=int(d.group_size), bits=int(d.bits), mode="affine")

        def sort_glue(st):
            flat = st["inds"].reshape(-1).astype(mx.uint32)
            order = mx.argsort(flat).astype(mx.uint32)
            return [mx.take(flat, order), (order // int(st["inds"].shape[-1])).astype(mx.uint32)]

        def combine(st):
            count, top_k = st["scores"].shape[0], st["scores"].shape[-1]
            inverse = mx.argsort(st["order"])
            picked = mx.take(st["y"].reshape(-1, st["rows"].shape[-1]), inverse, axis=0).reshape(count, top_k, -1)
            return L._combine_compiled(picked, st["scores"].reshape(count, top_k).astype(mx.float32), st["s"],
                                       st["gate"].reshape(count, 1))

        def shared(st):
            sh = st["block"].shared_expert
            return L._linear(sh.down_proj, swiglu(L._linear(sh.gate_proj, st["rows"]), L._linear(sh.up_proj, st["rows"])))

        def scan_only(st):
            scan = NAX._get_kernel("scan")
            M, E = int(st["sorted"].shape[0]), int(st["block"].switch_mlp.gate_proj["weight"].shape[0])
            bm = 64
            max_tiles = (M + bm - 1) // bm + min(E, M)
            return list(scan(inputs=[st["sorted"], mx.array([M, E, max_tiles], dtype=mx.int32)],
                             template=[("BM", bm), ("MAXE", NAX._MAX_EXPERTS)],
                             grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
                             output_shapes=[(max_tiles * 4,), (1,)], output_dtypes=[mx.uint32, mx.uint32]))

        def stock_sorted(st):
            sw = st["block"].switch_mlp
            xs = mx.take(st["rows"], st["row_map"], axis=0)[:, None, :]
            g = mx.gather_qmm(xs, sw.gate_proj["weight"], sw.gate_proj["scales"], sw.gate_proj.get("biases"),
                              rhs_indices=st["sorted"], transpose=True, sorted_indices=True, **st["fmt"])
            u = mx.gather_qmm(xs, sw.up_proj["weight"], sw.up_proj["scales"], sw.up_proj.get("biases"),
                              rhs_indices=st["sorted"], transpose=True, sorted_indices=True, **st["fmt"])
            return swiglu(g, u)

        from mlx2.runtime.models import qwen36_moe_grouped as G

        def lane(st):
            sw = st["block"].switch_mlp
            count, top_k = st["scores"].shape[0], st["scores"].shape[-1]
            return G.lane_grouped_rows(st["rows"], st["inds"].reshape(count, top_k), sw.gate_proj, sw.up_proj,
                                       sw.down_proj)

        st0 = stage[0]
        sw0 = st0["block"].switch_mlp
        report["lane_refusal"] = (G.admission_refusal(st0["rows"], st0["inds"], sw0.gate_proj, sw0.up_proj, sw0.down_proj)
                                  or G.lane_refusal(sw0.gate_proj, sw0.up_proj, sw0.down_proj))
        if report["lane_refusal"] is None:
            diffs = []
            for st in stage:
                count, top_k = st["scores"].shape[0], st["scores"].shape[-1]
                inverse = mx.argsort(st["order"])
                nax = mx.take(st["y"].reshape(-1, st["rows"].shape[-1]), inverse, axis=0).reshape(count, top_k, -1)
                ln = lane(st).reshape(count, top_k, -1)
                diffs.append(float(mx.max(mx.abs(nax.astype(mx.float32) - ln.astype(mx.float32))).item()))
            report["lane_vs_nax_max_abs_diff"] = max(diffs)

        law = frozenset(parts)
        variants = {
            "full_moe": each(lambda st: T._moe(st["block"], st["x"], law=law)),
            "gates": each(lambda st: [L.gate_project(st["block"].gate, st["x"]),
                                      L.gate_project(st["block"].shared_expert_gate, st["x"])]),
            "routing": each(lambda st: list(L.route(st["block"], st["gates"]))),
            "sort_glue": each(sort_glue),
            "nax_gate_up_swiglu": each(gate_up),
            "nax_down": each(down),
            "unsort_combine": each(combine),
            "shared_expert": each(shared),
            **({"lane_grouped_experts": each(lane),
                "full_moe_lane": each(lambda st: T._moe(st["block"], st["x"],
                                                        law=frozenset(set(parts) - {"experts_device"} | {"experts_lane"})))}
               if report["lane_refusal"] is None else {}),
            "nax_scan_only_x2": lambda: [scan_only(st) for st in stage for _ in range(2)],
            "stock_gather_qmv_gate_up_swiglu": each(stock_sorted),
        }
        for name, fn in variants.items():
            L.begin_forward()
            report["parts_ms"][name] = timed(fn)
            print(name, report["parts_ms"][name], flush=True)
            save()
        report["status"] = "passed"
    except BaseException as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        if adapter is not None:
            adapter.close()
        save()


if __name__ == "__main__":
    main()
