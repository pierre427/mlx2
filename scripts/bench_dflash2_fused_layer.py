#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Coordinator-only q4 g64 lane/fused microbench, under matching GPU locks.

No server, no target decode, no qualification/default/acceptance claim. B1 is
16 proposal rows, B4 is 4x4, both after 1024 committed context rows. Each
profile reports median of 15 fenced samples after 4 warmups. Graph build is
separate from fenced eval (submission + device completion); these wall-clock
spans are NOT hardware-only kernel timestamps. Per-call measures the full
start_tree pending lattice, excluding host finish_tree search.
With --stages, layer 0 at B1 also gets 15 samples of every fenced stage;
post-attention norm is included in the fused finish. Prefix context is 1024
rows and the incoming context tail is six rows, as in the whole-layer profile.
Fenced stage sums include added synchronization and are diagnostic only.

Requires a fresh --output directory. The lock wrapper owns cleanup/release;
this process never acquires or releases another session's locks.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

TARGET = "~/.cache/huggingface/hub/models--rapid-mlx--Qwen3.8-27B-4bit-MTP-MLX/snapshots/aa985c29ff5b334cbfdcbbc787d47e66e9d9e456"
DRAFTER = "~/mlx-models/Qwen3.8-27B-DFlash2"


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def paired_owner():
    paths = (Path("/Users/Shared/mlxuag/gpu.lock/owner.json"), Path("/tmp/gpu.lock/owner.json"))
    try:
        owners = [json.loads(p.read_text()) for p in paths]
    except (OSError, ValueError) as exc:
        raise ValueError("run under the shared paired-lock wrapper") from exc
    if owners[0] != owners[1] or not owners[0].get("lease_id"):
        raise ValueError("GPU paired-lock owner receipts do not match")
    ancestor, ancestors = os.getppid(), set()
    while ancestor > 1 and ancestor not in ancestors:
        ancestors.add(ancestor)
        ancestor = int(command("ps", "-p", str(ancestor), "-o", "ppid="))
    if owners[0].get("pid") not in ancestors:
        raise ValueError("paired GPU locks belong to another process tree")
    return owners[0]


def host_state():
    return {"thermal": command("pmset", "-g", "therm"),
            "swap": command("sysctl", "vm.swapusage"), "time": time.time()}


def atomic(path, value):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".receipt-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def stage_forward(layer, x, x_ctx, rope, cache, *, fused_path, measure):
    """Instrument one B1 layer with the same grouped inputs and cache law.

    measure(name, callable) returns the callable's value after fencing it.
    Fused post-attention norm is included in attention_conv_finish_residual_norm;
    no zero-cost post-norm measurement is fabricated. Cross-layer fusion is
    intentionally absent.
    """
    import mlx.core as mx
    from mlx2.runtime.drafters import dflash2_fused as fused
    from mlx2.runtime.drafters.dflash_base import swiglu

    if x.shape[0] != 1:
        raise ValueError("stage breakdown supports B1 only")
    attn = layer.self_attn
    b, length, _ = x.shape
    span = x_ctx.shape[1]
    entry = cache.rows[0] if hasattr(cache, "rows") else cache
    offset = entry.offset

    def dynamic(module, normalized):
        groups = normalized.shape[-1] // module.group_size
        return module.kernel_projection(normalized).reshape(
            *normalized.shape[:-1], 2, module.kernel_size, groups
        )

    def prepare(module, normalized, weights):
        if fused_path:
            return fused.conv(normalized, weights[..., 0, :, :], module.base_kernel[0], module.group_size)
        from mlx2.runtime.drafters.dflash2 import _grouped_dynamic_convolve
        return _grouped_dynamic_convolve(normalized, weights[..., 0, :, :], module.base_kernel[0], module.group_size)

    residual = x
    normalized = measure("input_norm", lambda: layer.input_layernorm(x))
    weights = measure("attention_kernel_projection", lambda: dynamic(layer.attention_conv, normalized))
    x = measure("attention_conv_prepare", lambda: prepare(layer.attention_conv, normalized, weights))
    q, ck, cv, pk, pv = measure("qkv_projections", lambda: attn._projections(x, x_ctx))

    def norm_rope():
        qr = q.reshape(b, length, attn.n_heads, attn.head_dim)
        cr = ck.reshape(b, span, attn.n_kv_heads, attn.head_dim)
        pr = pk.reshape(b, length, attn.n_kv_heads, attn.head_dim)
        if fused_path:
            return fused.qk_rope(qr, cr, pr, attn.q_norm.weight, attn.k_norm.weight,
                                 [offset], rope, attn.q_norm.eps, attn.k_norm.eps)
        return (rope(attn.q_norm(qr).transpose(0, 2, 1, 3), offset=offset + span),
                rope(attn.k_norm(cr).transpose(0, 2, 1, 3), offset=offset),
                rope(attn.k_norm(pr).transpose(0, 2, 1, 3), offset=offset + span))

    q, ck, pk = measure("qk_norm_rope", norm_rope)

    def sdpa():
        values = cv.reshape(b, span, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        proposals = pv.reshape(b, length, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        return fused.attend_cached(attn, q, ck, values, pk, proposals, cache)

    o = measure("sdpa_cache_append", sdpa)
    y = measure("o_proj", lambda: attn.o_proj(o.transpose(0, 2, 1, 3).reshape(b, length, -1)))
    kernel = weights[..., 1, :, :]
    if fused_path:
        norm = layer.post_attention_layernorm
        x, normalized = measure("attention_conv_finish_residual_norm", lambda: fused.conv_finish_norm(
            y, kernel, layer.attention_conv.base_kernel[1], layer.attention_conv.group_size,
            residual, norm.weight, norm.eps,
        ))
    else:
        x = measure("attention_conv_finish_residual", lambda: residual + layer.attention_conv.finish(y, kernel))
        normalized = measure("post_attention_norm", lambda: layer.post_attention_layernorm(x))
    residual = x
    weights = measure("mlp_kernel_projection", lambda: dynamic(layer.mlp_conv, normalized))
    x = measure("mlp_conv_prepare", lambda: prepare(layer.mlp_conv, normalized, weights))
    gate, up = measure("mlp_gate_up", lambda: (layer.mlp.gate_proj(x), layer.mlp.up_proj(x)))
    hidden = measure("swiglu", lambda: swiglu(gate, up))
    y = measure("down_proj", lambda: layer.mlp.down_proj(hidden))
    kernel = weights[..., 1, :, :]
    if fused_path:
        return measure("mlp_conv_finish_residual", lambda: fused.conv(
            y, kernel, layer.mlp_conv.base_kernel[1], layer.mlp_conv.group_size, residual,
        ))
    return measure("mlp_conv_finish_residual", lambda: residual + layer.mlp_conv.finish(y, kernel))


def stage_breakdown(layer, x, x_ctx, rope, cache_factory, *, fused_path):
    """Median of 15 fully fenced stage samples after four warmups."""
    import mlx.core as mx

    samples = {}
    for rep in range(19):
        cache = cache_factory()
        mx.synchronize()

        def measure(name, fn):
            start = time.perf_counter()
            value = fn()
            built = time.perf_counter()
            mx.eval(value)
            mx.synchronize()
            end = time.perf_counter()
            if rep >= 4:
                samples.setdefault(name, []).append({
                    "build_ms": (built-start)*1000,
                    "fenced_eval_ms": (end-built)*1000,
                    "total_ms": (end-start)*1000,
                })
            return value

        stage_forward(layer, x, x_ctx, rope, cache, fused_path=fused_path, measure=measure)
    result = {name: {"median": {key: statistics.median(s[key] for s in rows) for key in rows[0]},
                     "samples": rows} for name, rows in samples.items()}
    if fused_path:
        result["post_attention_norm"] = {"included_in": "attention_conv_finish_residual_norm"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--drafter", default=DRAFTER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", action="store_true", help="also fence each stage of layer 0 at B1 after 1024 context rows")
    args = parser.parse_args()
    owner = paired_owner()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("refusing a nonempty evidence directory; use a fresh --output")
    repo = Path(__file__).resolve().parents[1]
    # Import/instantiate the target adapter before model modules: it pins the
    # process numerical environment before the drafter/model implementation.
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, content_revision
    from mlx2.adapters.dflash2 import inspect_drafter, content_revision as draft_revision

    record = inspect_drafter(Path(args.drafter).expanduser(), args.target)
    diff_hash = hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--", "src", "scripts", "tests"]
    ))
    for relative in command("git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "--", "src", "scripts", "tests").splitlines():
        diff_hash.update(relative.encode() + b"\0" + (repo / relative).read_bytes())
    evidence = {
        "source_revision": command("git", "-C", str(repo), "rev-parse", "HEAD"),
        "worktree_diff_sha256": diff_hash.hexdigest(),
        "target": f"{Path(args.target).parent.parent.name}@{content_revision(args.target)}",
        "drafter": f"{Path(args.drafter).expanduser().name}@{draft_revision(record)}",
        "paired_lock_owner": owner, "before": host_state(),
        "status": "running", "qualification": "unqualified", "results": [],
        "timing": "median of 15; 4 warmups; graph build and fenced eval (submission plus device completion) separated",
    }
    atomic(args.output / "receipt.json", evidence)
    adapter = draft = None
    try:
        adapter = Qwen3827BAdapter(args.target)
        import mlx.core as mx
        from mlx2.adapters.dflash2 import load_drafter
        from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator as G
        from mlx2.runtime.lane import apply_policy
        from mlx2.runtime.lane.policy import detect, resolve

        mx.set_default_device(mx.gpu)
        target_config = json.loads((Path(args.target) / "config.json").read_text())
        target_policy = resolve(
            detect(adapter.model, target_config), family=adapter.descriptor.family,
            adapter=adapter.lane_policy_defaults(), mode="auto",
        )
        evidence["target_lane_receipt"] = apply_policy(adapter.model, target_policy)
        atomic(args.output / "receipt.json", evidence)

        def rnd(shape, seed):
            return (mx.random.normal(shape, key=mx.random.key(seed)) * 0.3).astype(mx.bfloat16)

        def timed(factory):
            samples = []
            for rep in range(19):
                # Fresh immutable-cache snapshots and evaluated inputs lie
                # outside both timing spans; no growing context across reps.
                fn = factory()
                mx.synchronize()
                start = time.perf_counter()
                outputs = fn()
                built = time.perf_counter()
                mx.eval(outputs)
                mx.synchronize()
                end = time.perf_counter()
                if rep >= 4:
                    samples.append({"build_ms": (built-start)*1000,
                                    "fenced_eval_ms": (end-built)*1000,
                                    "total_ms": (end-start)*1000})
            return {"median": {key: statistics.median(s[key] for s in samples) for key in samples[0]},
                    "samples": samples}

        for kernel in ("lane", "fused"):
            draft = load_drafter(record, adapter.model, runtime_quantization={"bits": 4, "group_size": 64}, draft_kernel=kernel)
            try:
                for name, batch, rows, tail_rows in (("B1", 1, 16, 6), ("B4", 4, 4, 4)):
                    width = draft.config.hidden_size
                    taps = len(draft.config.target_layer_ids)
                    context = rnd((1, 1024, width * taps), 1)
                    tail = rnd((batch, tail_rows, width * taps), 2)
                    x = rnd((batch, rows, width), 3)
                    hctx = draft.hidden_norm(draft.fc(tail))
                    mx.eval(context, tail, x, hctx)
                    cache_rows = []
                    for _ in range(batch):
                        caches = draft.make_cache()
                        draft.append_context(context, caches)
                        mx.eval([c.state for c in caches])
                        cache_rows.append(caches)

                    def snapshots():
                        copies = [G._snapshot_draft_state(c, tail[i:i+1])[0] for i, c in enumerate(cache_rows)]
                        mx.eval([entry.state for row in copies for entry in row])
                        return draft.batch_caches(copies)

                    layers = []
                    for idx, layer in enumerate(draft.layers):
                        def factory(idx=idx, layer=layer):
                            entry = snapshots()[idx]
                            return lambda: layer(x, hctx, draft.rope, entry)
                        layers.append(timed(factory))

                    def call_factory():
                        caches = snapshots()
                        return lambda: draft.start_tree(
                            list(range(17, 17 + batch)), tail, caches, rows - 1,
                            codebooks=draft.tree_codebooks()[:2], submit=False,
                        )["pending"]

                    # Warm codebooks outside timed graph construction.
                    draft.tree_codebooks()
                    result = {"kernel": kernel, "profile": name, "batch": batch,
                              "proposal_rows_per_lane": rows, "context_rows": 1024,
                              "tail_rows_per_lane": tail_rows,
                              "route_receipt": draft.draft_kernel_receipt,
                              "layers": layers, "call": timed(call_factory), "host": host_state()}
                    if args.stages and batch == 1:
                        result["stages"] = stage_breakdown(
                            draft.layers[0], x, hctx, draft.rope, lambda: snapshots()[0],
                            fused_path=kernel == "fused",
                        )
                    evidence["results"].append(result)
                    atomic(args.output / "receipt.json", evidence)
                    print(json.dumps({**{k: v for k, v in result.items() if k not in ("layers", "call", "stages")},
                                      "layer_medians": [v["median"] for v in layers],
                                      "call_median": result["call"]["median"],
                                      **({"stage_medians": {k: v.get("median", v) for k, v in result["stages"].items()}}
                                         if "stages" in result else {})}), flush=True)
                    del cache_rows, context, tail, x, hctx
            finally:
                draft = layer = factory = call_factory = None
                gc.collect()
                mx.clear_cache()
        evidence["status"] = "completed"
    except BaseException as exc:
        evidence["status"] = "failed"
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        draft = adapter = None
        gc.collect()
        evidence["after"] = host_state()
        atomic(args.output / "receipt.json", evidence)


if __name__ == "__main__":
    main()
