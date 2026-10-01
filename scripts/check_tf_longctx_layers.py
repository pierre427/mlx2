"""Metal bit-exactness of the TensorFold 0.6.1 intake on real Flash-Next layers.

Loads the artifact through the Flash-Next adapter (the served environment and
quantization), prefills real attention layers (trunk layers and the MTP
head's) to 16K/32K/64K tokens, then for decode (1 row) and MTP verify widths
(2, 3 rows) runs ``Attention.__call__`` with the fused QSA block scores off
and on from the same cache state and compares, as raw bytes: the selected
block ids (``QSASelection.raw_block_ids``), the attention output, and the
K/V written.  Caches: ``QSAKVCache`` (int offset), the batched cache over one
row (array offset) and over two rows of different lengths (left padding).

  MLX_ENABLE_TF32=0 PYTHONPATH=src python scripts/check_tf_longctx_layers.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP --out r.json
"""

import argparse
import json

import mlx.core as mx
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs="+", default=["3", "23", "47", "mtp"])
    ap.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768, 65536])
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.flash_next import FlashNextAdapter

    adapter = FlashNextAdapter(a.model)
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import qwen4_qsa_scores as S
    from mlx2.runtime.models.base import create_attention_mask

    model = adapter.model
    mx.set_cache_limit(4 << 30)
    trunk = model.language_model.model.layers

    def attention(name):
        if name == "mtp":
            return model.mtp.layers[0].self_attn
        return trunk[int(name)].self_attn

    captured = []
    stock_call = Q.QSAIndexer.__call__

    def capture(self, *args, **kw):
        selection = stock_call(self, *args, **kw)
        captured.append(selection)
        return selection

    Q.QSAIndexer.__call__ = capture

    def mask_for(x, cache):
        mask = create_attention_mask(x, cache, return_array=True)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None]
        return mask

    def prefill(attn, n, seed):
        rng = np.random.default_rng(seed)
        hid = model.language_model.args.hidden_size
        cache = Q.QSAKVCache(attn.indexer.summary_identity)
        for start in range(0, n, 4096):
            m = min(4096, n - start)
            x = mx.array(rng.standard_normal((1, m, hid)).astype(np.float32) * 0.5).astype(mx.bfloat16)
            mx.eval(attn(x, mask_for(x, cache), cache), cache.keys, cache.values, cache.index_keys)
        return cache

    def raw(x):
        return np.array(x.view(mx.uint16) if x.dtype in (mx.bfloat16, mx.float16) else x)

    def same(x, y):
        mx.eval(x, y)
        return x.shape == y.shape and x.dtype == y.dtype and raw(x).tobytes() == raw(y).tobytes()

    def kv_view(cache):
        width = cache._idx if isinstance(cache, Q.BatchQSAKVCache) else cache.offset
        return cache.keys[..., :width, :], cache.values[..., :width, :]

    rows = []
    mismatches = 0
    hid = model.language_model.args.hidden_size
    for name in a.layers:
        attn = attention(name)
        for n in a.contexts:
            base = prefill(attn, n, 1000 + n)
            other = prefill(attn, n - 37, 2000 + n)
            forms = {"plain": base, "batched1": Q.QSAKVCache.merge([base]),
                     "batched2_leftpad": Q.QSAKVCache.merge([base, other])}
            for form, cache in forms.items():
                batch = 2 if form == "batched2_leftpad" else 1
                for width in a.widths:
                    rng = np.random.default_rng(n * 10 + width + len(form))
                    ok = True
                    engaged = 0
                    for step in range(a.steps):
                        x = mx.array(rng.standard_normal((batch, width, hid)).astype(np.float32) * 0.5
                                     ).astype(mx.bfloat16)
                        outs = {}
                        for arm in ("stock", "fused"):
                            S.set_enabled(arm == "fused")
                            S.status(reset=True)
                            captured.clear()
                            out = attn(x, mask_for(x, cache), cache)
                            keys, values = kv_view(cache)
                            selection = captured[-1]
                            ids = selection.raw_block_ids
                            mx.eval(out, keys, values, *([] if ids is None else [ids]))
                            outs[arm] = (out, keys, values, ids, selection.kind)
                            if arm == "fused":
                                engaged += S.status()["counts"].get("engaged", 0)
                            cache.trim(width)
                        S.set_enabled(False)
                        (o0, k0, v0, i0, kind0), (o1, k1, v1, i1, kind1) = outs["stock"], outs["fused"]
                        step_ok = (kind0 == kind1 and same(o0, o1) and same(k0, k1) and same(v0, v1)
                                   and ((i0 is None and i1 is None) or same(i0, i1)))
                        ok &= step_ok
                        # advance the shared state by one stock step
                        mx.eval(attn(x, mask_for(x, cache), cache))
                    if not ok:
                        mismatches += 1
                    rec = {"layer": name, "context": n, "cache": form, "width": width, "steps": a.steps,
                           "identical": bool(ok), "fused_engaged_calls": engaged}
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
            del forms, base, other
            mx.clear_cache()
    Q.QSAIndexer.__call__ = stock_call
    report = {"mlx": mx.__version__, "rows": rows, "mismatching": mismatches,
              "policy": adapter.policy.as_dict()}
    json.dump(report, open(a.out, "w"), indent=1)
    print("MISMATCHING", mismatches, "of", len(rows), flush=True)
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
