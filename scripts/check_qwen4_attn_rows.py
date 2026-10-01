"""Metal bit-exactness gate for the fused Qwen4 attention rows (omlx #4052 port).

Loads a few real Flash-Next attention layers from the artifact safetensors and
compares, byte for byte on the GPU:

  proj   one concatenated q|k|v|index_qk quantized_matmul vs the four stock
         calls, M = 1..17 (admitted: M <= PROJECTION_MAX_ROWS);
  prep   prep_qk vs q_norm/k_norm + nn.RoPE (int and array offsets, R = 1..17);
  sdpa   sdpa_gate vs mx.fast.scaled_dot_product_attention + transpose +
         sigmoid-gate multiply, R = 1, 2, contexts across every plan switch,
         no mask / all-true / QSA-like sparse masks, cache-view K/V;
  layer  Attention.__call__ with the fused rows on vs off on identical caches
         (QSAKVCache and the batched cache), decode and verify widths.

  PYTHONPATH=src MLX_ENABLE_TF32=0 python scripts/check_qwen4_attn_rows.py \
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
      --out attn_rows_check.json
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def load_layer(model: Path, layer):
    """A trunk attention layer by index, or the MTP head's ("mtp")."""
    from mlx2.runtime.models import qwen4_exp as Q

    config = json.loads((model / "config.json").read_text())
    args = Q.TextModelArgs.from_dict(config["text_config"])
    if layer == "mtp":
        attn = Q.Attention(args, 0, summary_layer_id="mtp:0")
        prefix = "mtp.layers.0.self_attn."
        files = ["model-mtp-q4.safetensors"]
    else:
        layer = int(layer)
        attn = Q.Attention(args, layer)
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = f"language_model.model.layers.{layer}.self_attn."
        files = sorted({f for k, f in index.items() if k.startswith(prefix)})
    nn.quantize(attn, group_size=64, bits=4, class_predicate=lambda _p, m: isinstance(m, nn.Linear))
    weights = {}
    for name in files:
        for key, value in mx.load(str(model / name)).items():
            if key.startswith(prefix):
                weights[key[len(prefix):]] = value
    attn.load_weights(list(weights.items()), strict=True)
    attn.eval()
    mx.eval(attn.parameters())
    return args, attn


def same(a, b) -> bool:
    mx.eval(a, b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(mx.array_equal(a.view(mx.uint16) if a.dtype in (mx.bfloat16, mx.float16) else a,
                               b.view(mx.uint16) if b.dtype in (mx.bfloat16, mx.float16) else b).item())


def maxdiff(a, b) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def check_proj(attn, rng, report):
    from mlx2.runtime.models import qwen4_exp as Q

    table = attn._fused_projection_table()
    widths = [attn.num_heads * attn.head_dim * 2, attn.num_kv_heads * attn.head_dim,
              attn.num_kv_heads * attn.head_dim]
    rows = []
    for m in range(1, 18):
        x = mx.array(rng.standard_normal((1, m, attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits)).astype(np.float32)).astype(mx.bfloat16)
        ref = [attn.q_proj(x), attn.k_proj(x), attn.v_proj(x), attn.indexer.index_qk_proj(x)]
        got = mx.split(Q._table_matmul(table, x), [widths[0], widths[0] + widths[1], sum(widths)], axis=-1)
        ok = all(same(mx.contiguous(g), r) for g, r in zip(got, ref))
        rows.append({"m": m, "identical": ok,
                     "max_abs_diff": max(maxdiff(g, r) for g, r in zip(got, ref))})
    report["proj"] = rows
    return rows


def check_prep(attn, rng, report):
    from mlx2.runtime.models import qwen4_attn_rows as R

    rows = []
    for r in (1, 2, 3, 4, 8, 9, 17):
        for offset in (0, 1, 1023, 4095, 32767, 65536, 131071, 262000):
            for kind in ("int", "array"):
                qg = mx.array(rng.standard_normal((1, r, attn.num_heads * 2 * attn.head_dim)).astype(np.float32) * 2).astype(mx.bfloat16)
                kf = mx.array(rng.standard_normal((1, r, attn.num_kv_heads * attn.head_dim)).astype(np.float32) * 2).astype(mx.bfloat16)
                off = offset if kind == "int" else mx.array([offset], dtype=mx.int32)
                q, _gate = mx.split(qg.reshape(1, r, attn.num_heads, -1), 2, axis=-1)
                k = kf.reshape(1, r, attn.num_kv_heads, attn.head_dim)
                q_ref = attn.rope(attn.q_norm(q).transpose(0, 2, 1, 3), offset=off)
                k_ref = attn.rope(attn.k_norm(k).transpose(0, 2, 1, 3), offset=off)
                q_got, k_got = R.prep_qk(qg, kf, attn.q_norm.weight, attn.k_norm.weight, attn.q_norm.eps, off,
                                         heads=attn.num_heads, kv_heads=attn.num_kv_heads, rope=attn.rope)
                rows.append({"rows": r, "offset": offset, "offset_kind": kind,
                             "q_identical": same(q_got, q_ref), "k_identical": same(k_got, k_ref),
                             "max_abs_diff": max(maxdiff(q_got, q_ref), maxdiff(k_got, k_ref))})
    report["prep"] = rows
    return rows


def check_index_q(attn, rng, report):
    from mlx2.runtime.models import qwen4_exp as Q

    ix = attn.indexer
    width = (ix.n_heads + 1) * ix.head_dim
    rows = []
    for r in (1, 2, 3, 8, 17):
        for offset in (0, 5, 4093, 32767, 65536, 131071, 262000):
            for batched in (False, True):
                qk = mx.array(rng.standard_normal((1, r, width)).astype(np.float32) * 3).astype(mx.bfloat16)
                if batched:
                    q_pos = mx.array([offset])[:, None] + mx.arange(r)[None, :]
                else:
                    q_pos = mx.arange(offset, offset + r)[None, :]
                q, _raw = mx.split(qk, [ix.n_heads * ix.head_dim], axis=-1)
                ref = ix.q_layernorm(q.reshape(1, r, ix.n_heads, ix.head_dim))
                ref = Q._apply_rope_positions(ref, q_pos[..., None], ix.rotary_dim, ix.rope_theta, ix.rope_scaling)
                got = ix._fused_query(qk, q_pos)
                rows.append({"rows": r, "offset": offset, "batched_positions": batched,
                             "identical": same(got, ref), "max_abs_diff": maxdiff(got, ref)})
    report["index_q"] = rows
    return rows


def _sparse_mask(rng, rows, n, budget_blocks=512, block=4):
    """QSA-shaped: a random set of complete blocks plus the causal tail per row."""
    mask = np.zeros((1, 1, rows, n), dtype=bool)
    for r in range(rows):
        q_pos = n - rows + r
        complete = (q_pos + 1) // block
        chosen = rng.choice(complete, size=min(budget_blocks, complete), replace=False) if complete else []
        for b in chosen:
            mask[0, 0, r, b * block:(b + 1) * block] = True
        mask[0, 0, r, complete * block:q_pos + 1] = True
    return mx.array(mask)


def check_sdpa(attn, rng, report, contexts):
    from mlx2.runtime.models import qwen4_attn_rows as R

    heads, kvh, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
    rows_out = []
    cap = max(contexts) + 512
    # K/V as the model produces them: projections of random hidden states,
    # k normed + roped, inside a larger cache buffer (views, like the cache).
    hid = attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits
    keys_buf = mx.zeros((1, kvh, cap, d), dtype=mx.bfloat16)
    vals_buf = mx.zeros((1, kvh, cap, d), dtype=mx.bfloat16)
    chunk = 8192
    for start in range(0, cap, chunk):
        n = min(chunk, cap - start)
        x = mx.array(rng.standard_normal((1, n, hid)).astype(np.float32)).astype(mx.bfloat16)
        k = attn.k_norm(attn.k_proj(x).reshape(1, n, kvh, d)).transpose(0, 2, 1, 3)
        k = attn.rope(k, offset=start)
        v = attn.v_proj(x).reshape(1, n, kvh, d).transpose(0, 2, 1, 3)
        keys_buf[..., start:start + n, :] = k
        vals_buf[..., start:start + n, :] = v
        mx.eval(keys_buf, vals_buf)
    for n in contexts:
        for r in (1, 2):
            if r > n:
                continue
            keys = keys_buf[..., :n, :]
            vals = vals_buf[..., :n, :]
            x = mx.array(rng.standard_normal((1, r, hid)).astype(np.float32)).astype(mx.bfloat16)
            qg = attn.q_proj(x)
            q, gate = mx.split(qg.reshape(1, r, heads, -1), 2, axis=-1)
            q = attn.rope(attn.q_norm(q).transpose(0, 2, 1, 3), offset=n - r)
            masks = {"none": None,
                     "all_true": mx.ones((1, 1, r, n), dtype=mx.bool_),
                     "sparse": _sparse_mask(rng, r, n)}
            if r == 1:
                masks["all_true_bcast"] = mx.ones((1, 1, 1, n), dtype=mx.bool_)
            for kind, mask in masks.items():
                if r == 2 and kind == "none":
                    continue  # MLX would not be causal; the model always passes a mask
                ref = mx.fast.scaled_dot_product_attention(q, keys, vals, scale=attn.scale, mask=mask)
                ref_pre = ref.transpose(0, 2, 1, 3).reshape(1, r, -1)
                ref_gated = ref_pre * mx.sigmoid(gate.reshape(1, r, -1))
                reason = R.sdpa_supported(q, keys, vals, mask)
                if reason is not None:
                    rows_out.append({"n": n, "rows": r, "mask": kind, "refused": reason})
                    continue
                got_gated = R.sdpa_gate(q, keys, vals, attn.scale, mask=mask, gate=mx.sigmoid(gate))
                got_pre = R.sdpa_gate(q, keys, vals, attn.scale, mask=mask)
                plan = R.sdpa_plan(n, r, heads, kvh, d)
                rows_out.append({"n": n, "rows": r, "mask": kind, "plan": list(plan),
                                 "gated_identical": same(got_gated, ref_gated),
                                 "pre_identical": same(got_pre, ref_pre),
                                 "max_abs_diff": max(maxdiff(got_gated, ref_gated), maxdiff(got_pre, ref_pre))})
        mx.clear_cache()
    report["sdpa"] = rows_out
    return rows_out


def _prefill(attn, rng_seed, n, batched):
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models.base import create_attention_mask

    rng = np.random.default_rng(rng_seed)
    hid = attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits
    cache = Q.QSAKVCache(attn.indexer.summary_identity)
    step = 4096
    for start in range(0, n, step):
        m = min(step, n - start)
        x = mx.array(rng.standard_normal((1, m, hid)).astype(np.float32)).astype(mx.bfloat16)
        mask = create_attention_mask(x, cache, return_array=True)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None]
        out = attn(x, mask, cache)
        mx.eval(out, cache.keys, cache.values, cache.index_keys)
    if batched:
        cache = Q.QSAKVCache.merge([cache])
    return cache


def check_layer(attn, report, contexts, steps, widths):
    from mlx2.runtime.models import qwen4_attn_rows as R
    from mlx2.runtime.models.base import create_attention_mask

    hid = attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits
    rows_out = []
    for n in contexts:
        for batched in (False, True):
            for width in widths:
                caches = {arm: _prefill(attn, 1234 + n, n, batched) for arm in ("stock", "fused")}
                rng = np.random.default_rng(99 + n + width)
                identical = True
                worst = 0.0
                for step in range(steps):
                    x = mx.array(rng.standard_normal((1, width, hid)).astype(np.float32)).astype(mx.bfloat16)
                    outs = {}
                    for arm in ("stock", "fused"):
                        R.set_enabled(arm == "fused")
                        cache = caches[arm]
                        mask = create_attention_mask(x, cache, return_array=True)
                        if mask is not None and mask.ndim == 2:
                            mask = mask[None, None]
                        outs[arm] = attn(x, mask, cache)
                        mx.eval(outs[arm])
                    R.set_enabled(False)
                    ok = same(outs["stock"], outs["fused"])
                    identical &= ok
                    worst = max(worst, maxdiff(outs["stock"], outs["fused"]))
                    kv_ok = same(caches["stock"].keys[..., : caches["stock"]._idx if batched else caches["stock"].offset, :],
                                 caches["fused"].keys[..., : caches["fused"]._idx if batched else caches["fused"].offset, :])
                    identical &= kv_ok
                rows_out.append({"n": n, "batched": batched, "width": width, "steps": steps,
                                 "identical": bool(identical), "max_abs_diff": worst})
                print("layer", rows_out[-1], flush=True)
                del caches
                mx.clear_cache()
    report["layer"] = rows_out
    return rows_out


# Primitives that only re-describe a buffer (no Metal dispatch).  Launches
# hidden inside a primitive are added by HIDDEN below.
VIEWS = {"Reshape", "Split", "Transpose", "Slice", "Broadcast", "ExpandDims",
         "Squeeze", "AsStrided", "StopGradient", "Depends", "Contiguous"}


def count_launches(dot_path: str, rows: int, n_keys: int, strided_q_norm: bool) -> dict:
    import collections
    import re

    text = open(dot_path).read()
    prims = collections.Counter(re.findall(r'label ="([^"]+)"', text))
    launches = sum(n for p, n in prims.items() if p not in VIEWS)
    hidden = 0
    # RoPE with dims < head_dim copies its input first; MLX's two-pass vector
    # SDPA is two dispatches; RMSNorm copies a strided input (the query half
    # of the q|gate projection); the fallback composed SDPA is counted by node.
    hidden += prims.get("RoPE", 0)
    hidden += int(strided_q_norm)
    sdpa = prims.get("ScaledDotProductAttention", 0)
    if sdpa and n_keys >= 1024:
        hidden += sdpa
    return {"nodes": sum(prims.values()), "dispatch_nodes": launches,
            "hidden_dispatches": hidden, "primitives": dict(prims)}


def check_count(attn, report, dot_dir, contexts, widths):
    from mlx2.runtime.models import qwen4_attn_rows as R
    from mlx2.runtime.models.base import create_attention_mask

    hid = attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits
    rows_out = []
    for n in contexts:
        for width in widths:
            for arm in ("stock", "fused"):
                cache = _prefill(attn, 7 + n, n, batched=True)
                # one warm step so the pooled-key cache is current
                x = mx.ones((1, width, hid), dtype=mx.bfloat16)
                R.set_enabled(arm == "fused")
                for step in range(2):
                    mask = create_attention_mask(x, cache, return_array=True)
                    if mask is not None and mask.ndim == 2:
                        mask = mask[None, None]
                    mx.eval(x, mask, cache.keys, cache.values, cache.index_keys, cache.offset)
                    if cache._qsa_pooled_keys is not None:
                        mx.eval(cache._qsa_pooled_keys)
                    out = attn(x, mask, cache)
                    if step == 1:
                        path = f"{dot_dir}/layer_{arm}_n{n}_r{width}.dot"
                        mx.export_to_dot(path, out)
                        entry = {"n": n, "width": width, "arm": arm,
                                 **count_launches(path, width, n, strided_q_norm=arm == "stock")}
                        rows_out.append(entry)
                        print("count", n, width, arm, entry["dispatch_nodes"], "+",
                              entry["hidden_dispatches"], flush=True)
                    mx.eval(out)
                R.set_enabled(False)
                del cache
    report["count"] = rows_out
    return rows_out


def check_time(attn, report, contexts, widths, steps, arms):
    """Wall time of one attention-layer call (eval'd alone), arms rotating
    step by step on twin caches; medians.  A layer in isolation, so this is
    the attention cost the full model pays twelve times per token, without
    the whole-model noise.  An arm is ``stock``/``fused`` with an optional
    ``@<tokens>`` one-token indexed threshold (``@inf``: masked arm always,
    ``@0``: indexed from the indexer budget)."""
    import statistics

    from mlx2.runtime.models import qwen4_attn_rows as R
    from mlx2.runtime.models import qwen4_qsa_indexed as QI
    from mlx2.runtime.models.base import create_attention_mask

    default_m1 = QI._AUTO_MIN_CONTEXT_M1

    def configure(arm):
        base, _, threshold = arm.partition("@")
        R.set_enabled(base == "fused")
        QI._AUTO_MIN_CONTEXT_M1 = (
            default_m1 if not threshold else 2**31 - 1 if threshold == "inf" else int(threshold)
        )

    hid = attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits
    rows_out = []
    for n in contexts:
        for width in widths:
            caches = {arm: _prefill(attn, 5 + n, n, batched=True) for arm in arms}
            times = {arm: [] for arm in arms}
            x = mx.ones((1, width, hid), dtype=mx.bfloat16)
            mx.eval(x)
            for step in range(steps + 4):
                order = arms[step % len(arms):] + arms[: step % len(arms)]
                for arm in order:
                    configure(arm)
                    cache = caches[arm]
                    mask = create_attention_mask(x, cache, return_array=True)
                    if mask is not None and mask.ndim == 2:
                        mask = mask[None, None]
                    mx.eval(mask)
                    mx.synchronize()
                    t0 = time.perf_counter()
                    out = attn(x, mask, cache)
                    mx.eval(out)
                    mx.synchronize()
                    if step >= 4:
                        times[arm].append(1e6 * (time.perf_counter() - t0))
                configure("stock")
            entry = {"n": n, "width": width}
            for arm in arms:
                ts = sorted(times[arm])
                entry[arm] = {"median_us": statistics.median(ts),
                              "iqr_us": [ts[len(ts) // 4], ts[3 * len(ts) // 4]]}
            base = entry[arms[0]]["median_us"]
            entry["delta_pct_vs_first"] = {
                arm: 100 * (entry[arm]["median_us"] / base - 1) for arm in arms[1:]}
            rows_out.append(entry)
            print("time", n, width, {arm: round(entry[arm]["median_us"], 1) for arm in arms}, flush=True)
            del caches
            mx.clear_cache()
    configure("stock")
    report["time"] = rows_out
    return rows_out


def check_mask(attn, report, contexts, widths):
    """qsa_mask vs QSASelection.dense_mask() on selections the real indexer
    makes over prefilled caches (plain and batched), plus a broadcast
    shared-top-k selection."""
    from mlx2.runtime.models import qwen4_attn_rows as R
    from mlx2.runtime.models.base import create_attention_mask

    hid = attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits
    rows_out = []
    for n in contexts:
        for batched in (False, True):
            cache = _prefill(attn, 3 + n, n, batched)
            rng = np.random.default_rng(n)
            for width in widths:
                x = mx.array(rng.standard_normal((1, width, hid)).astype(np.float32)).astype(mx.bfloat16)
                mask = create_attention_mask(x, cache, return_array=True)
                if mask is not None and mask.ndim == 2:
                    mask = mask[None, None]
                sel = attn.indexer(x, mask, cache)
                entry = {"n": n, "batched": batched, "width": width, "kind": sel.kind}
                reason = R.qsa_mask_supported(sel)
                if reason is None:
                    ref = sel.dense_mask()
                    got = R.qsa_mask(sel)
                    entry["identical"] = bool(ref.shape == got.shape and mx.array_equal(ref, got).item())
                    entry["true_cells"] = int(got.astype(mx.int32).sum().item())
                else:
                    entry["refused"] = reason
                rows_out.append(entry)
                print("mask", entry, flush=True)
            del cache
            mx.clear_cache()
    report["mask"] = rows_out
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", nargs="+", default=["3", "23", "47", "mtp"])
    ap.add_argument("--stages", nargs="+", default=["proj", "prep", "index_q", "sdpa", "mask", "layer", "count", "time"])
    ap.add_argument("--sdpa-contexts", type=int, nargs="+",
                    default=[1, 2, 17, 511, 1000, 1023, 1024, 1025, 2048, 4096, 8192, 8193,
                             16384, 32768, 32769, 49152, 65536, 65537, 98304, 131072])
    ap.add_argument("--layer-contexts", type=int, nargs="+", default=[700, 1500, 2600, 20000, 70000])
    ap.add_argument("--layer-widths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--layer-steps", type=int, nargs="+", default=[6])
    ap.add_argument("--count-contexts", type=int, nargs="+", default=[1024, 1500, 32768])
    ap.add_argument("--dot-dir", default="attn_rows_dots")
    ap.add_argument("--time-contexts", type=int, nargs="+",
                    default=[512, 1500, 4096, 16384, 32768, 65536, 98304, 131072])
    ap.add_argument("--time-widths", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--time-arms", nargs="+", default=["stock", "fused", "fused@inf", "fused@0"])
    ap.add_argument("--time-steps", type=int, default=30)
    ap.add_argument("--mask-contexts", type=int, nargs="+", default=[2600, 9000, 40000, 70001])
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    mx.set_cache_limit(4 << 30)
    # The served Flash-Next environment (QSA arms, thresholds) before any
    # tensor module is imported; the fused rows themselves stay off here and
    # are switched per arm.
    from mlx2.adapters.flash_next import configure_environment

    configure_environment(Path(a.model).expanduser())
    from mlx2.runtime.models import qwen4_attn_rows as R

    report = {"mlx": mx.__version__, "device": mx.device_info(), "layers": {}}
    t0 = time.time()
    for layer in a.layers:
        _args, attn = load_layer(Path(a.model).expanduser(), layer)
        rng = np.random.default_rng(sum(map(ord, str(layer))))
        entry = {}
        if "proj" in a.stages:
            rows = check_proj(attn, rng, entry)
            print("proj", layer, [(r["m"], r["identical"]) for r in rows], flush=True)
        if "prep" in a.stages:
            rows = check_prep(attn, rng, entry)
            print("prep", layer, "all identical:", all(r["q_identical"] and r["k_identical"] for r in rows),
                  "max diff", max(r["max_abs_diff"] for r in rows), flush=True)
        if "index_q" in a.stages:
            rows = check_index_q(attn, rng, entry)
            print("index_q", layer, "all identical:", all(r["identical"] for r in rows),
                  "max diff", max(r["max_abs_diff"] for r in rows), flush=True)
        if "sdpa" in a.stages:
            rows = check_sdpa(attn, rng, entry, a.sdpa_contexts)
            bad = [r for r in rows if "refused" not in r and not (r["gated_identical"] and r["pre_identical"])]
            print("sdpa", layer, len(rows), "cases,", len(bad), "not identical;",
                  "refused:", sorted({r["refused"] for r in rows if "refused" in r}), flush=True)
            for r in bad[:10]:
                print("  ", r, flush=True)
        if "layer" in a.stages:
            check_layer(attn, entry, a.layer_contexts, a.layer_steps[0], a.layer_widths)
        if "mask" in a.stages:
            check_mask(attn, entry, a.mask_contexts, a.layer_widths)
        if "time" in a.stages and layer == a.layers[0]:
            check_time(attn, entry, a.time_contexts, a.time_widths, a.time_steps, a.time_arms)
        if "count" in a.stages and layer == a.layers[0]:
            Path(a.dot_dir).mkdir(parents=True, exist_ok=True)
            check_count(attn, entry, a.dot_dir, a.count_contexts, a.layer_widths)
        entry["status"] = R.status(reset=True)
        report["layers"][str(layer)] = entry
        del attn
        mx.clear_cache()
    report["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("wrote", a.out, f"{report['seconds']:.0f}s")


if __name__ == "__main__":
    main()
