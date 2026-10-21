#!/usr/bin/env python3
# ruff: noqa: B023 -- every closure here is called within its own loop iteration.
"""Qwen3.6 35B verify-forward cost by row count, split MoE / mixers / glue.

Coordinator-owned real-weight probe; refuses Metal without --i-own-the-gpu.

One model load.  After a prefill of realistic prose, the model's own greedy
continuation (with its second choices as siblings) gives the draft rows.  For
each row count R it times, interleaved, median of --reps (``mx.eval`` +
``mx.synchronize`` boundaries, cache restored between forwards):

* ``plain``     the model's own forward over a chain of R tokens;
* ``tile_chain`` / ``tile_tree``  the vendored TensorFold lane forward over a
  chain / a best-first-shaped tree of R rows;

each ``full``, with the MoE blocks replaced by their captured outputs
(``no_moe``), and with MoE and attention/GDN mixers replaced (``glue``).
``grouped`` reruns ``full`` with the expert-grouped rows selected.  It then
times the MoE blocks alone on the captured tree-tile inputs (stock vs
grouped) and checks, per layer and R, that the grouped per-slot expert
outputs and block outputs equal the stock ones bit for bit and that a row's
output does not depend on its tile neighbours.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _swapouts() -> int:
    for line in subprocess.check_output(["vm_stat"], text=True).splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return -1


def _therm() -> str:
    return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


PROSE = [
    "docs/QUALIFICATION.md",
    "AGENTS.md",
    "README.md",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rows", default="1,2,4,8,16")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--lane-policy", default=None,
                    help="JSON lane policy override; adds lane-matmul forward modes (dense projections)")
    ap.add_argument("--lane-split-k", default="[]", help="JSON list of K-slice counts to time for lane experts")
    ap.add_argument("--default-geometry", default="{}", help="JSON merged into the grouped DEFAULT_GEOMETRY")
    ap.add_argument("--geometries", default="[{}]",
                    help="JSON list of grouped launch geometries to time (routed experts alone)")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if a.out.exists():
        ap.error(f"{a.out} exists; refusing to overwrite evidence")
    row_counts = [int(r) for r in a.rows.split(",")]

    import mlx.core as mx

    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

    swap0, therm0 = _swapouts(), _therm()
    adapter = Qwen3635BA3BAdapter(a.model)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(4 << 30)
    from mlx2.runtime import qwen38_tensorfold as TF
    from mlx2.runtime.models import qwen36_moe_grouped as G

    G.DEFAULT_GEOMETRY.update(json.loads(a.default_geometry))
    from mlx2.runtime.models.cache import KVCache

    TF._modules(None, cached=True)
    lane_receipt = None
    if a.lane_policy:
        # Lane matmul on the dense projections (experts are SwitchLinear and
        # never wrapped); toggled per measurement with set_enabled.
        from mlx2.runtime.lane import installer as lane_installer
        from mlx2.runtime.lane import policy as lane_policy

        resolved = lane_policy.resolve(lane_policy.detect(adapter.model), overrides=a.lane_policy)
        lane_receipt = lane_installer.apply_policy(adapter.model, resolved)
        lane_installer.set_enabled(False)
    from mlx2.runtime.tensorfold_qwen38 import lane_multi

    model = adapter.model
    core = model.model
    layers = list(core.layers)
    blocks = [getattr(layer, "_layer", layer).mlp for layer in layers]
    tok = adapter.tokenizer
    checkpoint = f"{Path(a.model).name}@{adapter.identity['fingerprint'][:8]}"

    text = "\n\n".join((ROOT / p).read_text() for p in PROSE if (ROOT / p).exists())
    ids = tok.encode(text)[: a.context]
    cache = model.make_cache()
    for i in range(0, len(ids), 512):
        logits = model(mx.array([ids[i:i + 512]]), cache=cache)
        mx.eval(logits, [c.state for c in cache])

    def snap():
        return [c.offset if isinstance(c, KVCache) else list(c.cache) for c in cache]

    def restore(s):
        for c, v in zip(cache, s):
            if isinstance(c, KVCache):
                c.offset = v
            else:
                c.cache = list(v)

    base = snap()
    start = len(ids)
    # Greedy continuation and second choices (tree siblings).
    chain, alts = [], []
    last = logits[0, -1]
    for _ in range(max(row_counts)):
        order = mx.argsort(-last.astype(mx.float32))[:2].tolist()
        chain.append(int(order[0]))
        alts.append(int(order[1]))
        last = model(mx.array([[chain[-1]]]), cache=cache)[0, -1]
    restore(base)
    mx.eval([c.state for c in cache if not isinstance(c, KVCache)])

    def tree_of(rows):
        """Best-first-like tree: a spine of greedy tokens, each spine node's
        second choice as a sibling while rows last."""
        tokens, parents, spine = [chain[0]], [-1], [0]
        depth = 1
        while len(tokens) < rows:
            parent = spine[-1]
            tokens.append(chain[depth])
            parents.append(parent)
            spine.append(len(tokens) - 1)
            if len(tokens) < rows:
                tokens.append(alts[depth])
                parents.append(parent)
            depth += 1
        return tokens, parents

    # --- capture / stub plumbing -------------------------------------------
    real_attention, real_gdn = lane_multi._attention, lane_multi._gdn
    mixer_names = ["linear_attn" if getattr(getattr(l, "_layer", l), "is_linear", False) else "self_attn"
                   for l in layers]

    class Rec:
        def __init__(self, fn, store):
            self.fn, self.store = fn, store

        def __call__(self, *args, **kwargs):
            out = self.fn(*args, **kwargs)
            self.store.append(out)
            return out

    class Replay:
        def __init__(self, values):
            self.values, self.i = values, 0

        def __call__(self, *args, **kwargs):
            out = self.values[self.i % len(self.values)]
            self.i += 1
            return out

    def run(kind, rows):
        if kind == "plain":
            return model(mx.array([chain[:rows]]), cache=cache)
        if kind == "tile_chain":
            toks, pars = chain[:rows], list(range(-1, rows - 1))
        else:
            toks, pars = tree_of(rows)
        logits, _records, _ = lane_multi.multi_tree_forward(core, model.logits, [toks], [pars], [cache], [start])
        return logits

    def capture(kind, rows):
        """Per-layer mlp inputs/outputs and mixer outputs of one full forward."""
        mlp_in, mlp_out, mix_out = [], [], []
        for li, (layer, block) in enumerate(zip(layers, blocks)):
            inner = getattr(layer, "_layer", layer)

            def rec(x, block=block):
                mlp_in.append(x)
                y = block(x)
                mlp_out.append(y)
                return y
            object.__setattr__(inner, "mlp", rec)
            if kind == "plain":
                name = mixer_names[li]
                object.__setattr__(inner, name, Rec(getattr(inner, name), mix_out))
        if kind != "plain":
            lane_multi._attention = Rec(real_attention, mix_out)
            lane_multi._gdn = Rec(real_gdn, mix_out)
        try:
            restore(base)
            out = run(kind, rows)
            mx.eval(out, mlp_in, mlp_out, mix_out)
        finally:
            unstub()
            restore(base)
        return mlp_in, mlp_out, mix_out

    def unstub():
        for layer in layers:
            inner = getattr(layer, "_layer", layer)
            for name in ("mlp", "linear_attn", "self_attn"):
                if name in inner.__dict__:
                    object.__delattr__(inner, name)
        lane_multi._attention, lane_multi._gdn = real_attention, real_gdn

    def stub(kind, mlp_out, mix_out, mixers):
        for layer, y in zip(layers, mlp_out):
            object.__setattr__(getattr(layer, "_layer", layer), "mlp", Replay([y]))
        if mixers:
            if kind == "plain":
                for layer, name, y in zip(layers, mixer_names, mix_out):
                    object.__setattr__(getattr(layer, "_layer", layer), name, Replay([y]))
            else:
                replay = Replay(mix_out)
                lane_multi._attention = replay
                lane_multi._gdn = replay

    def set_grouped(on):
        for block in blocks:
            block.set_grouped_rows(on)

    def once(fn):
        restore(base)
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        dt = time.perf_counter() - t0
        restore(base)
        return dt

    record = {
        "schema": "mlx2.qwen36_tree_moe_rows.v1",
        "checkpoint": checkpoint,
        "artifact_identity": adapter.identity,
        "mlx2_commit": _git_head(),
        "mlx": mx.__version__,
        "context_tokens": start,
        "rows": row_counts,
        "reps": a.reps,
        "chain": chain,
        "alts": alts,
        "forward_ms": {},
        "moe_only_ms": {},
        "unique_experts": {},
        "bit_checks": [],
        "grouped_stats": None,
        "lane_policy": a.lane_policy,
        "lane_receipt": None if lane_receipt is None else {
            k: lane_receipt.get(k) for k in ("law_id", "covered", "refused", "groups")},
    }

    # Captures (also warm every kernel once).
    caps = {}
    kinds = ["plain", "tile_chain", "tile_tree"]
    for kind in kinds:
        for rows in row_counts:
            caps[(kind, rows)] = capture(kind, rows)
            once(lambda: run(kind, rows))

    if not a.skip_forward:
        samples: dict = {}
        modes = ["full", "no_moe", "glue", "grouped", "moe_lane"]
        if lane_receipt is not None:
            modes += ["lane", "lane_no_moe", "lane_grouped", "lane_moe_lane"]
        for rep in range(a.reps):
            for kind in kinds:
                for rows in row_counts:
                    for mode in modes:
                        _mi, mo, xo = caps[(kind, rows)]
                        if mode in ("no_moe", "glue", "lane_no_moe"):
                            stub(kind, mo, xo, mode == "glue")
                        if mode in ("grouped", "lane_grouped"):
                            set_grouped(True)
                        if mode in ("moe_lane", "lane_moe_lane"):
                            set_grouped("lane")
                        if mode.startswith("lane"):
                            lane_installer.set_enabled(True)
                        try:
                            dt = once(lambda: run(kind, rows))
                        finally:
                            unstub()
                            set_grouped(False)
                            if lane_receipt is not None:
                                lane_installer.set_enabled(False)
                        samples.setdefault(f"{kind}/{rows}/{mode}", []).append(dt)
        record["forward_ms"] = {k: 1e3 * statistics.median(v) for k, v in samples.items()}
        record["forward_samples_ms"] = {k: [1e3 * s for s in v] for k, v in samples.items()}

    # MoE blocks alone on the captured tree-tile inputs.
    from mlx2.runtime.models import qwen3_next as QN

    moe_samples: dict = {}
    for rep in range(a.reps):
        for rows in row_counts:
            mlp_in = caps[("tile_tree", rows)][0]
            for mode in ("stock", "grouped", "lane"):
                set_grouped({"stock": False, "grouped": True, "lane": "lane"}[mode])
                try:
                    dt = once(lambda: [b(x) for b, x in zip(blocks, mlp_in)])
                finally:
                    set_grouped(False)
                moe_samples.setdefault(f"{rows}/{mode}", []).append(dt)
    record["moe_only_ms"] = {k: 1e3 * statistics.median(v) for k, v in moe_samples.items()}

    # Routed experts alone (per-slot outputs, no router/shared/combine):
    # stock gathers vs the grouped kernels at each launch geometry.
    geometries = json.loads(a.geometries)
    routed = {}
    for rows in row_counts:
        if rows < 2:
            continue
        mlp_in = caps[("tile_tree", rows)][0]
        routed[rows] = [(b.switch_mlp, x, QN._stock_routing(b, b.gate(x))[0]) for b, x in zip(blocks, mlp_in)]
        mx.eval([r[2] for r in routed[rows]])
    sw_samples: dict = {}
    sw_equal: dict = {}
    for rep in range(a.reps):
        for rows, items in routed.items():
            set_grouped(False)
            sw_samples.setdefault(f"{rows}/stock", []).append(once(lambda: [s(x, i) for s, x, i in items]))
            for gi, geo in enumerate(geometries):
                fn = lambda: [G.grouped_rows(x, i, s.gate_proj, s.up_proj, s.down_proj, geometry=geo)  # noqa: E731
                              for s, x, i in items]
                sw_samples.setdefault(f"{rows}/g{gi}", []).append(once(fn))
                if rep == 0:
                    ref = [s(x, i) for s, x, i in items]
                    got = fn()
                    mx.eval(ref, got)
                    sw_equal[f"{rows}/g{gi}"] = sum(bool(mx.array_equal(r, g).item()) for r, g in zip(ref, got))
    # Combined routed output (router-weighted top-k sum): stock gathers +
    # combine vs the existing per-(row, slot) window kernels (no dedup,
    # combine fused; called directly, bypassing their 4/8-bit-only admission).
    from mlx2.runtime.models import qwen36_moe_decode as MD

    combined = {}
    for rows, items in routed.items():
        combined[rows] = []
        for (s, x, i), b in zip(items, blocks):
            scores = QN._stock_routing(b, b.gate(x))[1]
            combined[rows].append((s, x, i, scores))
        mx.eval([c[3] for c in combined[rows]])
    for rep in range(a.reps):
        for rows, items in combined.items():
            stock = lambda: [(s(x, i) * sc[..., None]).sum(axis=-2) for s, x, i, sc in items]  # noqa: E731
            window = lambda: [MD.routed_rows(x, i, sc, s.gate_proj, s.up_proj, s.down_proj)  # noqa: E731
                              for s, x, i, sc in items]
            sw_samples.setdefault(f"{rows}/combined_stock", []).append(once(stock))
            sw_samples.setdefault(f"{rows}/combined_window", []).append(once(window))
            if rep == 0:
                ref, got = stock(), window()
                mx.eval(ref, got)
                sw_equal[f"{rows}/combined_window"] = sum(
                    bool(mx.array_equal(r.reshape(-1), g.reshape(-1)).item()) for r, g in zip(ref, got))
    # Tensor-unit law: timing, row invariance and deviation from the stock gathers.
    lane_checks = {}
    for rep in range(a.reps):
        for rows, items in routed.items():
            fn = lambda: [G.lane_grouped_rows(x, i, s.gate_proj, s.up_proj, s.down_proj)  # noqa: E731
                          for s, x, i in items]
            sw_samples.setdefault(f"{rows}/lane", []).append(once(fn))
            for sk in json.loads(a.lane_split_k):
                fsk = lambda: [G.lane_grouped_rows(x, i, s.gate_proj, s.up_proj, s.down_proj,  # noqa: E731
                                                   split_k=sk) for s, x, i in items]
                sw_samples.setdefault(f"{rows}/lane_sk{sk}", []).append(once(fsk))
            if rep == 0:
                ref = [s(x, i) for s, x, i in items]
                got = fn()
                rev = [G.lane_grouped_rows(x[:, ::-1], i[:, ::-1], s.gate_proj, s.up_proj, s.down_proj)[:, ::-1]
                       for s, x, i in items]
                alone = [mx.concatenate([G.lane_grouped_rows(x[:, r:r + 1], i[:, r:r + 1], s.gate_proj,
                                                             s.up_proj, s.down_proj) for r in range(rows)], axis=1)
                         for s, x, i in items]
                mx.eval(ref, got, rev, alone)
                diffs = [mx.abs(r.astype(mx.float32) - g.astype(mx.float32)) for r, g in zip(ref, got)]
                lane_checks[str(rows)] = {
                    "layers": len(items),
                    "reversed_tile_equal": sum(bool(mx.array_equal(r, g).item()) for r, g in zip(rev, got)),
                    "rows_alone_equal": sum(bool(mx.array_equal(r, g).item()) for r, g in zip(alone, got)),
                    "stock_equal_layers": sum(bool(mx.array_equal(r, g).item()) for r, g in zip(ref, got)),
                    "stock_max_abs_diff": max(float(mx.max(d).item()) for d in diffs),
                    "stock_frac_elements_differ": statistics.mean(float(mx.mean(d > 0).item()) for d in diffs),
                    "stock_rel_rms": statistics.mean(
                        float((mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(r.astype(mx.float32) ** 2))).item())
                        for d, r in zip(diffs, ref)),
                }
    record["lane_law_checks"] = lane_checks
    record["routed_only_ms"] = {k: 1e3 * statistics.median(v) for k, v in sw_samples.items()}
    record["routed_only_equal_layers"] = sw_equal
    record["geometries"] = geometries
    record["default_geometry"] = dict(G.DEFAULT_GEOMETRY)

    # Bit checks and unique-expert counts.
    for rows in row_counts:
        mlp_in = caps[("tile_tree", rows)][0]
        uniq = []
        for li, (block, x) in enumerate(zip(blocks, mlp_in)):
            gates = block.gate(x)
            inds, _scores = QN._stock_routing(block, gates)
            flat = inds.reshape(-1)
            uniq.append(len(set(flat.tolist())))
            if rows < 2:
                continue
            sw = block.switch_mlp
            set_grouped(False)
            stock_slots = sw(x, inds)
            stock_block = block(x)
            set_grouped(True)
            grouped_slots = sw(x, inds)
            grouped_block = block(x)
            set_grouped(False)
            # Row invariance: each row alone through the stock module, and the
            # rows reversed through the grouped kernels.
            alone = mx.concatenate([sw(x[:, r:r + 1], inds[:, r:r + 1]) for r in range(rows)], axis=1)
            set_grouped(True)
            rev = sw(x[:, ::-1], inds[:, ::-1])[:, ::-1]
            set_grouped(False)
            mx.eval(stock_slots, stock_block, grouped_slots, grouped_block, alone, rev)
            record["bit_checks"].append({
                "rows": rows,
                "layer": li,
                "bits": int(sw.gate_proj.bits),
                "slots_equal": bool(mx.array_equal(stock_slots, grouped_slots).item()),
                "block_equal": bool(mx.array_equal(stock_block, grouped_block).item()),
                "rows_alone_equal": bool(mx.array_equal(alone, grouped_slots).item()),
                "reversed_tile_equal": bool(mx.array_equal(rev, grouped_slots).item()),
                "finite": bool(mx.all(mx.isfinite(grouped_block.astype(mx.float32))).item()),
                "max_abs_diff": float(mx.max(mx.abs(stock_slots.astype(mx.float32)
                                                    - grouped_slots.astype(mx.float32))).item()),
            })
        record["unique_experts"][str(rows)] = {
            "mean": statistics.mean(uniq), "min": min(uniq), "max": max(uniq),
            "assignments": rows * 8,
        }
    record["grouped_stats"] = blocks[0].grouped_rows_stats()
    checks = record["bit_checks"]
    record["bit_summary"] = {
        key: f"{sum(c[key] for c in checks)}/{len(checks)}"
        for key in ("slots_equal", "block_equal", "rows_alone_equal", "reversed_tile_equal", "finite")
    }
    record["swapouts_delta"] = _swapouts() - swap0
    record["thermal_before"], record["thermal_after"] = therm0, _therm()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(a.out)
    print(json.dumps({k: record[k] for k in ("checkpoint", "mlx2_commit", "forward_ms", "moe_only_ms",
                                             "routed_only_ms", "routed_only_equal_layers", "lane_law_checks",
                                             "unique_experts", "bit_summary", "swapouts_delta")}, indent=1))
    return 0 if all(c["slots_equal"] and c["block_equal"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
