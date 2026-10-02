"""Metal check of the batched one-token sparse QSA arm at served geometry.

One Flash-Next QSA attention layer (served config, random bf16 weights),
two rows prefilled as B1 to different lengths and merged into a left-padded
BatchQSAKVCache, then ``--steps`` one-token decode steps per arm from
identical caches: masked (the default), gather, indexed.  Reports, per step,
max |diff| vs masked, max |masked|, and the fraction of bit-equal elements,
plus whether the K/V written is identical.  Rounding-level differences (a
few bf16 ulps) mean the arms attend the same keys; a semantic difference
would show as O(|out|).

  MLX_ENABLE_TF32=0 PYTHONPATH=src python scripts/check_qsa_batch_decode_sparse.py \\
      --i-own-the-gpu --config ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP/config.json
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[32768, 30001])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--min-context", type=int, default=16384)
    ap.add_argument("--cpu", action="store_true", help="smoke the script on the CPU")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not (a.i_own_the_gpu or a.cpu):
        ap.error("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx

    if a.cpu:
        mx.set_default_device(mx.cpu)

    from mlx2.runtime.models import qwen4_exp as Q

    config = json.load(open(a.config))
    args = Q.TextModelArgs.from_dict(config.get("text_config", config))
    mx.random.seed(0)
    layer = args.layer_types.index("full_attention") if hasattr(args, "layer_types") else 3
    attention = Q.Attention(args, layer_idx=layer)
    attention.set_dtype(mx.bfloat16)
    attention.eval()
    mx.eval(attention.parameters())
    hidden_size = args.hidden_size

    def batched():
        rows = []
        for length in a.lengths:
            mx.random.seed(1000 + length)
            row = Q.QSAKVCache(attention.indexer.summary_identity)
            for start in range(0, length, 4096):
                n = min(4096, length - start)
                x = (mx.random.normal((1, n, hidden_size)) * 0.5).astype(mx.bfloat16)
                mx.eval(attention(x, row.make_mask(n, return_array=True, window_size=None), row), row.state)
            rows.append(row)
        cache = Q.BatchQSAKVCache.merge(rows)
        mx.eval(cache.state)
        return cache

    outputs, states, counters = {}, {}, {}
    for arm in ("off", "gather", "indexed"):
        Q.set_qsa_batch_decode_sparse(arm, min_context=a.min_context)
        Q.qsa_batch_decode_sparse_status(reset=True)
        cache = batched()
        mx.random.seed(99)
        outs = []
        for _ in range(a.steps):
            x = (mx.random.normal((len(a.lengths), 1, hidden_size)) * 0.5).astype(mx.bfloat16)
            out = attention(x, cache.make_mask(1, return_array=True), cache)
            mx.eval(out, cache.state)
            outs.append(out)
        outputs[arm], states[arm] = outs, cache.state
        counters[arm] = Q.qsa_batch_decode_sparse_status()["counts"]
        del cache
        mx.clear_cache()
    report = {"lengths": a.lengths, "layer": layer, "counters": counters, "arms": {}}
    for arm in ("gather", "indexed"):
        steps = []
        for ref, got in zip(outputs["off"], outputs[arm]):
            r, g = ref.astype(mx.float32), got.astype(mx.float32)
            steps.append({
                "max_abs_diff": float(mx.max(mx.abs(r - g))),
                "max_abs_ref": float(mx.max(mx.abs(r))),
                "bit_equal_fraction": float(mx.mean((ref == got).astype(mx.float32))),
            })
        report["arms"][arm] = {
            "steps": steps,
            "cache_state_identical": all(
                bool(mx.array_equal(x, y)) for x, y in zip(states["off"], states[arm])),
        }
    for x, y in zip(outputs["gather"], outputs["indexed"]):
        report.setdefault("gather_vs_indexed_bit_equal", []).append(bool(mx.array_equal(x, y)))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
