"""Bounded DS4.1 KV components; caller owns and validates the GPU lease.

Importing this module performs no MLX work. run() requires GPU, receives an
atomic write() closure, and never acquires ownership or loads model weights.
Synthetic activations and projection outputs cannot establish model quality.
"""
import hashlib
import time
from types import SimpleNamespace


def run(mx, np, report, write):
    if mx.default_device() != mx.gpu:
        raise RuntimeError("DS4.1 KV components require the caller-owned GPU; no CPU fallback")
    if "kv_components" in report:
        raise ValueError("Refusing to overwrite KV component evidence")
    from mlx2.runtime.models import deepseek_v41 as v41
    from mlx2.runtime.models.deepseek_v41_config import DeepSeekV41Config
    from mlx2.runtime.models.deepseek_v41_quant import quantize, fused_quantize
    from mlx2.runtime.models import deepseek_v41_fused as epilogue
    from mlx2.runtime.models.deepseek_v4 import Rotary
    from mlx2.runtime.models.deepseek_v4_hc import expand as hc_expand, split_sinkhorn

    section = report["kv_components"] = {
        "scope": "synthetic finite Q/K, real geometry; index linear outputs supplied; HC projection uses synthetic weights",
        "model_geometry": "40layers/64heads*512/32indexheads*128/top512/window128",
        "checkpoint": "synthetic activations; no checkpoint loaded by this component",
        "device": str(mx.default_device()), "requires_gpu": True,
        "qualified": False, "quality_established": False,
        "timing_scope": "interleaved3reps after raw parity and warmup; graph-build and eval-completion wall time separated",
        "cases": [], "skipped_unphysical_index_cells": [],
    }
    write()
    rounded = lambda x: x.astype(mx.bfloat16).astype(mx.float32)
    rng = np.random.default_rng(4120261025)
    c = DeepSeekV41Config()

    def arrays(result):
        return list(result.values())

    def host(result):
        mx.eval(arrays(result))
        return {key: np.array(value) for key, value in result.items()}

    def compare(name, functions, *, require_exact=True, **meta):
        row = {"name": name, **meta, "variants": {}, "exact": True,
               "requires_exact": require_exact}
        section["cases"].append(row)
        baseline = None
        for variant, function in functions.items():
            value = host(function())
            raw = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "bytes": v.nbytes,
                       "sha256": hashlib.sha256(v.tobytes()).hexdigest()}
                   for k, v in value.items()}
            exact = baseline is None or (
                baseline.keys() == value.keys()
                and all(baseline[k].shape == v.shape and baseline[k].dtype == v.dtype
                        and baseline[k].tobytes() == v.tobytes() for k, v in value.items()))
            row["variants"][variant] = {"raw": raw, "exact": exact, "samples": []}
            if baseline is None:
                baseline = value
            if not exact:
                row["exact"] = False
                row["first_mismatch"] = next((k for k in value if k not in baseline
                    or baseline[k].shape != value[k].shape or baseline[k].dtype != value[k].dtype
                    or baseline[k].tobytes() != value[k].tobytes()), "array inventory")
                differences = {}
                for key, actual in value.items():
                    expected = baseline.get(key)
                    if expected is None or expected.shape != actual.shape or expected.dtype != actual.dtype:
                        differences[key] = {"geometry_or_dtype_mismatch": True}
                        continue
                    changed = np.any(expected.view(np.uint8).reshape(expected.size, expected.itemsize)
                                     != actual.view(np.uint8).reshape(actual.size, actual.itemsize), axis=-1)
                    if np.any(changed):
                        i = int(np.flatnonzero(changed)[0])
                        delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
                        differences[key] = {"changed_elements": int(changed.sum()),
                            "max_abs": float(np.max(delta)), "first_flat_index": i,
                            "reference_value": float(expected.flat[i]), "candidate_value": float(actual.flat[i])}
                row["variants"][variant]["differences"] = differences
                row["classification"] = ("required_raw_mismatch" if require_exact
                                          else "unqualified_numerical_profile_difference")
                write()
                if require_exact:
                    raise AssertionError("KV raw-array parity failed: " + name + "/" + variant)
        write()
        # Warm all paths before the interleaved timing samples. No host arrays
        # or validation conversions are included in these timing intervals.
        for function in functions.values():
            mx.eval(arrays(function()))
        names = list(functions)
        for repeat in range(3):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for variant in order:
                start = time.perf_counter_ns()
                result = functions[variant]()
                built = time.perf_counter_ns()
                mx.eval(arrays(result))
                done = time.perf_counter_ns()
                row["variants"][variant]["samples"].append({
                    "repeat": repeat, "graph_build_ms": (built-start)/1e6,
                    "completion_ms": (done-built)/1e6, "total_ms": (done-start)/1e6})
        for value in row["variants"].values():
            value["median_ms"] = {key: float(np.median([s[key] for s in value["samples"]]))
                                  for key in ("graph_build_ms", "completion_ms", "total_ms")}
        write()
        return row

    # Real query/key geometry and reference RoPE/QAT. Linear projection
    # outputs are synthetic constants, so these timings exclude dense GEMMs.
    for layer, ratio in ((2, 2), (20, 1), (24, 1)):
        for count in (0, 1, 50, 511, 512, 513):
            for rows in (1, 7, 99):
                # Ratio-two cells end on an odd incomplete-group frontier.
                end = 2*count+1 if ratio == 2 else count
                start = end-rows
                if start < 0:
                    section["skipped_unphysical_index_cells"].append({
                        "layer": layer, "ratio": ratio, "pool_rows": count,
                        "query_rows": rows, "reason": "negative start would violate source frontier"})
                    continue
                qlinear = rounded(mx.array(rng.normal(0, .2, (1, rows, c.index_n_heads*c.index_head_dim)).astype(np.float32)))
                weights = rounded(mx.array(rng.normal(0, .1, (1, rows, c.index_n_heads)).astype(np.float32)))
                key = mx.array(rng.normal(0, .2, (1, count, c.index_head_dim)).astype(np.float32))
                key = rounded(quantize(key, "index_fp4")) if count else key
                source = SimpleNamespace(index_keys=key)
                x, qr = mx.zeros((1, rows, c.hidden_size), mx.float32), mx.zeros((1, rows, c.q_lora_rank), mx.float32)
                lens = ((mx.arange(start, end)+1)//ratio)[None, :, None]
                prior = (mx.arange(count)[None, None]//c.candidate_block_size*c.candidate_block_size) < lens
                mx.eval(qlinear, weights, key, x, qr, prior)
                selectors, calls = {}, {}
                for policy in ("reference", "finite_all_retained"):
                    m = SimpleNamespace(index_policy=policy, scheduled_kv_calls={}, pointwise_policy="reference",
                        attention_epilogue_policy="reference", qat_policy="reference", round=rounded,
                        qat=lambda value, kind: rounded(quantize(value, kind)))
                    a = object.__new__(v41.Attention)
                    a.c, a.model, a.layer, a.ratio = c, m, layer, ratio
                    a.rope = Rotary(c.qk_rope_head_dim, c.compress_rope_theta, c.rope_scaling)
                    a.linear = lambda value, name, q=qlinear, w=weights: q if name == "indexer.attn_q_b" else w
                    selectors[policy] = a
                def select(policy):
                    shared = {"candidates": prior} if layer > c.candidate_source_layer_id else {}
                    a = selectors[policy]
                    a.model.scheduled_kv_calls = {}
                    a.select(x, qr, source, start, shared)
                    calls[policy] = dict(a.model.scheduled_kv_calls)
                    return {k: shared[k] for k in ("indices", "selected_mask", "candidates") if k in shared}
                row = compare("index_selection", {
                    "reference": lambda: select("reference"),
                    "finite_all_retained": lambda: select("finite_all_retained")},
                    layer=layer, ratio=ratio, pool_rows=count, query_rows=rows,
                    start=start, end=end, odd_partial=bool(ratio == 2 and end % 2))
                row["scheduled_kv_calls"] = dict(calls)
                if count > 512 and calls["finite_all_retained"].get("all_retained_index", 0):
                    raise AssertionError("Pool513 must retain ordinary top-k scoring")
                write()

    def owner(pool, offset=None):
        return SimpleNamespace(_pooled=pool, pooled=pool, _pool_count=pool.shape[1],
                               offset=pool.shape[1] if offset is None else offset)

    # Width-one raw attention: exactly the same stock equations, with only
    # gather ownership changed. Six queries model a CSA2 reuse group.
    for count in (1, 50, 512):
        pool = rounded(mx.array(rng.normal(0, .25, (1, count, c.head_dim)).astype(np.float32)))
        source = owner(pool)
        indices = mx.arange(count, dtype=mx.uint32)[None, None]
        valid = mx.ones((1, 1, count), mx.bool_)
        local = rounded(mx.array(rng.normal(0, .25, (1, 1, 128, c.head_dim)).astype(np.float32)))
        queries = [rounded(mx.array(rng.normal(0, .25, (1, c.num_attention_heads, 1, c.head_dim)).astype(np.float32))) for _ in range(6)]
        sinks = mx.array(rng.normal(0, .1, c.num_attention_heads).astype(np.float32))
        mask = mx.ones((1, 1, 1, 128), mx.bool_)
        mx.eval(pool, indices, valid, local, queries, sinks, mask)
        counters = {}
        def attention(reuse, layers):
            shared = {"_index_generation": 1}
            counters.clear()
            result = {}
            for i in range(layers):
                gathered = v41.decode_selected_kv(source, indices, valid, shared, counters=counters) if reuse else None
                result["layer_"+str(i)] = v41.selected_attention(queries[i], local, mask, pool,
                    indices, valid, sinks, gathered=gathered)
            return result
        for layers in (1, 6):
            row = compare("selected_attention", {
                "reference": lambda: attention(False, layers),
                "decode_reuse": lambda: attention(True, layers)}, pool_rows=count, query_rows=1, layers=layers)
            row["scheduled_kv_calls"] = dict(counters)
            if counters != {"decode_gather_build": 1, **({"decode_gather_hit": layers-1} if layers > 1 else {})}:
                raise AssertionError("Decode reuse control counts do not match layer group")
            write()

    # Exercise invalidation with actual GPU arrays at the maximum selected
    # width, including changed masks, index generation, owner and pool growth.
    pool = mx.arange(512*512, dtype=mx.float32).reshape(1, 512, 512)*2**-18
    source = owner(pool)
    indices = mx.arange(512, dtype=mx.uint32)[None, None]
    valid = mx.ones(indices.shape, mx.bool_)
    shared, counters = {"_index_generation": 1}, {}
    value = v41.decode_selected_kv(source, indices, valid, shared, counters=counters)
    same = v41.decode_selected_kv(source, indices, valid, shared, counters=counters)
    if same is not value:
        raise AssertionError("Identical selection did not reuse the gathered array")
    for mutation in ("generation", "mask", "growth", "owner"):
        old = value
        if mutation == "generation":
            shared["_index_generation"] += 1
        elif mutation == "mask":
            valid = mx.concatenate((mx.zeros((1, 1, 1), mx.bool_), valid[:, :, 1:]), 2)
        elif mutation == "growth":
            source._pooled = source.pooled = mx.concatenate((source.pooled, mx.full((1, 1, 512), .75, mx.float32)), 1)
            source._pool_count = source.offset = 513
            indices = mx.arange(1, 513, dtype=mx.uint32)[None, None]
        else:
            source = owner(source.pooled)
        value = v41.decode_selected_kv(source, indices, valid, shared, counters=counters)
        expected = v41._selected_rows(source.pooled, indices)
        actual_host, expected_host = host({"gather": value}), host({"gather": expected})
        exact = actual_host["gather"].tobytes() == expected_host["gather"].tobytes()
        section["cases"].append({"name": "gather_invalidation", "mutation": mutation,
            "pool_rows": source._pool_count, "selected_rows": 512, "query_rows": 1,
            "different_object": value is not old, "exact": exact,
            "sha256": hashlib.sha256(actual_host["gather"].tobytes()).hexdigest(),
            "scheduled_kv_calls": dict(counters)})
        write()
        if value is old or not exact:
            raise AssertionError("Decode gather invalidation failed: " + mutation)
    if counters != {"decode_gather_build": 5, "decode_gather_hit": 1}:
        raise AssertionError("Invalidation did not rebuild the expected bounded gathers")

    # Existing exact store/epilogue mechanisms, independently compared as raw
    # outputs at real dimensions. These are not new attention math profiles.
    for rows in (1, 7, 99):
        x = mx.array(rng.normal(0, 1, (1, c.num_attention_heads, rows, c.head_dim)).astype(np.float32))
        rope = Rotary(c.qk_rope_head_dim, c.compress_rope_theta, c.rope_scaling)
        mx.eval(x)
        compare("pointwise_bf16_store", {"reference": lambda: {"output": rounded(x)},
            "fused": lambda: {"output": epilogue.rounded(x)}}, query_rows=rows)
        for inverse in (False, True):
            compare("attention_rope_qat_epilogue", {
                "reference": lambda: {"output": rounded(fused_quantize(rounded(rope(x, 127, inverse=inverse)), "fp8"))},
                "fused": lambda: {"output": epilogue.rotate_qat(x, rope, "fp8", 127, inverse=inverse)}},
                query_rows=rows, inverse=inverse)
        h = mx.array(rng.normal(0, .2, (1, rows, c.hc_mult, c.hidden_size)).astype(np.float32))
        b = mx.array(rng.normal(0, .2, (1, rows, c.hidden_size)).astype(np.float32))
        post = mx.array(rng.normal(0, .2, (1, rows, c.hc_mult)).astype(np.float32))
        comb = mx.array(rng.normal(0, .2, (1, rows, c.hc_mult, c.hc_mult)).astype(np.float32))
        mx.eval(h, b, post, comb)
        compare("hc_fused_bf16_store", {
            "reference": lambda: {"output": rounded(hc_expand(b, h, post, comb))},
            "fused": lambda: {"output": epilogue.expand(b, h, post, comb)}}, query_rows=rows)
        # Source Block.mixes does NOT round its projection to BF16. Retain
        # the exact F32 projection/RMS chain, comparing raw coefficients too;
        # a fused-HC profile may differ before any BF16 residual store.
        state = rounded(mx.array(rng.normal(0, .2, (1, rows, c.hc_mult, c.hidden_size)).astype(np.float32)))
        fn = rounded(mx.array(rng.normal(0, .02, (24, c.hc_mult*c.hidden_size)).astype(np.float32)))
        scales = mx.array([.1, .2, .3], mx.float32)
        base = mx.array(rng.normal(0, .1, 24).astype(np.float32))
        mx.eval(state, fn, scales, base)
        def coefficients(fused):
            flat = state.reshape(1, rows, c.hc_mult*c.hidden_size).astype(mx.float32)
            projection = flat @ fn.T
            mixes = projection * mx.rsqrt(mx.mean(mx.square(flat), -1, keepdims=True)+c.rms_norm_eps)
            pre, post, comb = (split_sinkhorn(mixes, scales, base,
                iterations=c.hc_sinkhorn_iters, eps=c.hc_eps) if fused else
                v41._sinkhorn_mix_fp32(mixes, scales, base, c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps))
            return {"projection": projection, "mixes": mixes, "pre": pre, "post": post, "comb": comb}
        compare("hc_mix_projection_and_coefficients", {
            "reference": lambda: coefficients(False), "fused_fp32": lambda: coefficients(True)},
            require_exact=False, query_rows=rows, iterations=c.hc_sinkhorn_iters, eps=c.hc_eps,
            projection_rounding="rawF32; BF16-grid hidden and synthetic fn; Source Block.mixes chain",
            policy="explicit candidate profile; any nonexact coefficients exclude it from exact subset")
        from mlx2.runtime.models import deepseek_v41_source_hc as source_hc
        def source_coefficients(reduction,softmax_exp):
            flat=state.reshape(1,rows,c.hc_mult*c.hidden_size).astype(mx.float32)
            projection=flat @ fn.T
            mixes=projection * mx.rsqrt(mx.mean(mx.square(flat),-1,keepdims=True)+c.rms_norm_eps)
            pre,post,comb=source_hc.split(mixes,scales,base,iterations=c.hc_sinkhorn_iters,
                eps=c.hc_eps,reduction=reduction,softmax_exp=softmax_exp)
            return dict(projection=projection,mixes=mixes,pre=pre,post=post,comb=comb)
        functions={'reference':lambda:coefficients(False)}
        for reduction in source_hc.REDUCTIONS:
            for exponential in source_hc.SOFTMAX_EXPS:
                key=reduction+'|'+exponential
                functions[key]=lambda r=reduction,e=exponential:source_coefficients(r,e)
        compare('source_hc_coefficients',functions,require_exact=False,query_rows=rows,
            iterations=c.hc_sinkhorn_iters,eps=c.hc_eps,
            policy='Source-specific candidate strategies; only exact common strategy may enter exact profile')
    section["complete"] = True
    section["all_raw_exact"] = all(row["exact"] for row in section["cases"])
    section["required_raw_exact"] = all(row["exact"] for row in section["cases"] if row.get("requires_exact", True))
    section["candidate_profile_differences"] = [
        {"name": row["name"], "query_rows": row.get("query_rows"), "classification": row["classification"]}
        for row in section["cases"] if not row["exact"] and not row.get("requires_exact", True)]
    source_cases=[r for r in section['cases'] if r['name']=='source_hc_coefficients']
    section['source_hc_exact_strategies']=sorted(set.intersection(*[
        {name for name,value in row['variants'].items() if name!='reference' and value['exact']}
        for row in source_cases])) if source_cases else []
    write()
