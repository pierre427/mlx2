"""Metal gate for the row-exact attention window and HC row-exact mode (W2-B).

Real Flash-Next weights (trunk attention layers and the MTP head's; trunk,
mixer and MTP hyper-connection modules), served environment profile, all
comparisons as raw bytes:

  attn   per attention layer and starting context, verify windows of R rows
         through the row-exact route (``_RowExactAttention`` inside a
         ``row_exact_verify.window``, fused attention rows on) vs R separate
         one-token forwards on a twin cache (the bits MTP-off decode gives
         each row): the window output, and the cache's K, V and raw index
         keys afterwards.  Windows are run back to back so they cross the
         1,024-key plan switch, the 2,048-token indexer budget and the
         8,192 / 32,768 two-pass partition switches.  Each window records
         which form ran (dense window, deferred SDPA window, per row).
  hc     per HC module, R rows inside a window (law 0 grid rows, class-
         swapped projections) vs R one-row calls with the kernels on and with
         them off.
  time   per attention layer, one window call: the window forms vs the
         per-row forwards (MLX_QWEN4_ROW_EXACT_ATTN_WINDOW off), arms
         alternated, medians.

  scratchpad/gpuq.sh w2b-window env PYTHONPATH=src MLX_ENABLE_TF32=0 \\
      .venv/bin/python scripts/check_row_exact_window.py --i-own-the-gpu \\
      --out window.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

DEFAULT_MODEL = Path("~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--layers", nargs="+", default=["3", "23", "47", "mtp"])
    ap.add_argument("--contexts", type=int, nargs="+", default=[1010, 2030, 8180, 32750])
    ap.add_argument("--rows", type=int, nargs="+", default=[2, 3, 4, 8, 16, 17])
    ap.add_argument("--caches", nargs="+", default=["plain", "batched"],
                    help="QSAKVCache and/or the B=1 BatchQSAKVCache of the ordinary route")
    ap.add_argument("--hc-cases", type=int, default=4)
    ap.add_argument("--time-contexts", type=int, nargs="*", default=[1000, 8000, 32000])
    ap.add_argument("--time-rows", type=int, nargs="*", default=[3, 8, 17])
    ap.add_argument("--time-steps", type=int, default=12)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.flash_next import configure_environment

    configure_environment(a.model)
    import mlx.core as mx

    assert mx.default_device() == mx.gpu and mx.metal.is_available()
    mx.set_cache_limit(4 << 30)
    report = {"model": str(a.model), "mlx": mx.__version__, "args": {k: str(v) for k, v in vars(a).items()}}
    out = Path(a.out)
    if "attn" not in a.skip:
        report["attn"] = check_attention(a)
        out.write_text(json.dumps(report, indent=1))
    if "hc" not in a.skip:
        report["hc"] = check_hc(a)
        out.write_text(json.dumps(report, indent=1))
    if "time" not in a.skip and a.time_contexts:
        report["time"] = time_attention(a)
    out.write_text(json.dumps(report, indent=1))
    summary = {}
    if "attn" in report:
        cells = [c for layer in report["attn"].values() for c in layer]
        summary["attn_windows"] = len(cells)
        summary["attn_identical"] = sum(c["identical"] for c in cells)
        forms = {}
        for c in cells:
            forms[c["form"]] = forms.get(c["form"], 0) + 1
        summary["attn_forms"] = forms
    if "hc" in report:
        cells = [c for mod in report["hc"].values() for c in mod]
        summary["hc_cases"] = len(cells)
        summary["hc_identical"] = sum(c["identical"] for c in cells)
    report["summary"] = summary
    out.write_text(json.dumps(report, indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _same(a, b) -> bool:
    import mlx.core as mx

    mx.eval(a, b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (mx.bfloat16, mx.float16):
        a, b = a.view(mx.uint16), b.view(mx.uint16)
    return bool(mx.array_equal(a, b).item())


def _maxdiff(a, b) -> float:
    import mlx.core as mx

    if a.shape != b.shape:
        return float("inf")
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _swap_row_exact(attn):
    """Class-swap the attention module and its projections as the route does."""
    import mlx.nn as nn

    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import qwen4_row_exact as RE

    for _, module in attn.named_modules():
        cls = type(module)
        if cls is nn.QuantizedLinear:
            module.__class__ = RE._subclass(RE._RowExactQuantizedLinear, cls)
        elif cls is Q.Attention:
            module.__class__ = RE._subclass(RE._RowExactAttention, cls)


def _hidden(attn):
    return attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits


def _prefill(attn, seed, n, batched=False):
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import qwen4_attn_rows as AR
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models.base import create_attention_mask

    rng = np.random.default_rng(seed)
    was = AR.enabled()
    AR.set_enabled(False)
    cache = Q.QSAKVCache(attn.indexer.summary_identity)
    step = 4096
    for start in range(0, n, step):
        m = min(step, n - start)
        x = mx.array(rng.standard_normal((1, m, _hidden(attn))).astype(np.float32)).astype(mx.bfloat16)
        mask = create_attention_mask(x, cache, return_array=True)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None]
        y = attn(x, mask, cache)
        mx.eval(y, cache.keys, cache.values, cache.index_keys)
    AR.set_enabled(was)
    if batched:
        cache = Q.QSAKVCache.merge([cache])
    return cache


def _one_token_rows(attn, x, cache):
    """R one-token forwards (the MTP-off bits of each row)."""
    import mlx.core as mx

    from mlx2.runtime.models.qwen4_row_exact import _ordinary_b1_mask

    outs = []
    for j in range(x.shape[1]):
        outs.append(attn(x[:, j : j + 1], _ordinary_b1_mask(cache), cache))
    return mx.concatenate(outs, axis=1)


def _window_call(attn, x, cache):
    from mlx2.runtime import row_exact_verify as REV
    from mlx2.runtime.models import qwen4_exp as Q

    record = REV.Window(int(x.shape[1]))
    with REV.window(record), Q._declared_width(1):
        y = attn(x, None, cache)
    return y, record


def _form(record):
    stages = record.stages
    if "window_dense" in stages.get("attention", {}):
        return "dense_window"
    if "window_deferred" in stages.get("attention_sdpa", {}):
        return "deferred_window"
    return "per_row"


def _cache_view(cache):
    n = cache._idx if hasattr(cache, "_idx") else cache.offset
    return cache.keys[..., :n, :], cache.values[..., :n, :], cache.index_keys[:, :n]


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def check_attention(a):
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import qwen4_attn_rows as AR

    from check_qwen4_attn_rows import load_layer

    from mlx2.runtime.models import qwen4_attn_window as AW

    AW.set_enabled(True)
    results = {}
    for layer in a.layers:
        _, attn = load_layer(a.model, layer)
        _swap_row_exact(attn)
        rows_out = []
        for n, kind in [(n, kind) for n in a.contexts for kind in a.caches]:
            caches = {arm: _prefill(attn, 1234 + n, n, kind == "batched") for arm in ("ref", "win")}
            rng = np.random.default_rng(77 + n)
            AR.set_enabled(True)
            for r in a.rows:
                x = mx.array(rng.standard_normal((1, r, _hidden(attn))).astype(np.float32)).astype(mx.bfloat16)
                start = _cache_view(caches["ref"])[0].shape[2]
                ref = _one_token_rows(attn, x, caches["ref"])
                got, record = _window_call(attn, x, caches["win"])
                mx.eval(ref, got)
                rows_equal = [_same(got[:, j], ref[:, j]) for j in range(r)]
                kv_ok = all(_same(p, q) for p, q in zip(_cache_view(caches["win"]), _cache_view(caches["ref"])))
                cell = {
                    "layer": layer, "cache": kind, "start_keys": start, "rows": r, "form": _form(record),
                    "rows_equal": sum(rows_equal), "kv_index_equal": kv_ok,
                    "identical": all(rows_equal) and kv_ok, "max_abs_diff": _maxdiff(got, ref),
                    "stages": {k: v for k, v in record.stages.items() if k.startswith("attention")},
                    "failures": record.failures,
                }
                rows_out.append(cell)
                print("attn", json.dumps(cell), flush=True)
            AR.set_enabled(False)
            del caches
            mx.clear_cache()
        results[layer] = rows_out
        del attn
        mx.clear_cache()
    return results


def time_attention(a):
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import qwen4_attn_rows as AR
    from mlx2.runtime.models import qwen4_attn_window as AW

    from check_qwen4_attn_rows import load_layer

    _, attn = load_layer(a.model, a.layers[0])
    _swap_row_exact(attn)
    arms = ["per_row", "window"]
    rows_out = []
    AR.set_enabled(True)
    for n in a.time_contexts:
        for r in a.time_rows:
            caches = {arm: _prefill(attn, 5 + n, n) for arm in arms}
            times = {arm: [] for arm in arms}
            rng = np.random.default_rng(n + r)
            x = mx.array(rng.standard_normal((1, r, _hidden(attn))).astype(np.float32)).astype(mx.bfloat16)
            mx.eval(x)
            forms = set()
            for step in range(a.time_steps + 2):
                order = arms if step % 2 == 0 else arms[::-1]
                for arm in order:
                    AW.set_enabled(arm == "window")
                    cache = caches[arm]
                    mx.synchronize()
                    t0 = time.perf_counter()
                    y, record = _window_call(attn, x, cache)
                    mx.eval(y)
                    mx.synchronize()
                    if step >= 2:
                        times[arm].append(1e6 * (time.perf_counter() - t0))
                    if arm == "window":
                        forms.add(_form(record))
                    # Keep the context fixed: drop the window's rows again.
                    cache.trim(r)
            AW.set_enabled(True)
            entry = {"start_keys": n, "rows": r, "window_forms": sorted(forms)}
            for arm in arms:
                ts = sorted(times[arm])
                entry[arm] = {"median_us": statistics.median(ts), "min_us": ts[0], "max_us": ts[-1]}
            entry["window_over_per_row"] = entry["window"]["median_us"] / entry["per_row"]["median_us"]
            rows_out.append(entry)
            print("time", json.dumps(entry), flush=True)
            del caches
            mx.clear_cache()
    AR.set_enabled(False)
    return rows_out


# ---------------------------------------------------------------------------
# HC
# ---------------------------------------------------------------------------


def check_hc(a):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime import row_exact_verify as REV
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import qwen4_hc_decode as HCD
    from mlx2.runtime.models import qwen4_row_exact as RE
    from types import SimpleNamespace

    config = json.loads((a.model / "config.json").read_text())
    text = config.get("text_config", config)
    args = SimpleNamespace(hc_count=text["hc_count"], hidden_size=text["hidden_size"],
                           hc_lowrank=text["hc_lowrank"], rms_norm_eps=text["rms_norm_eps"])
    index = json.loads((a.model / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {}

    def tensors(prefix):
        found = {}
        for key, shard in index.items():
            if key.startswith(prefix + "."):
                if shard not in shards:
                    shards[shard] = mx.load(str(a.model / shard))
                found[key[len(prefix) + 1:]] = shards[shard][key]
        if not found:
            mtp = mx.load(str(a.model / "model-mtp-q4.safetensors"))
            found = {k[len(prefix) + 1:]: v for k, v in mtp.items() if k.startswith(prefix + ".")}
        return found

    def build(prefix, combine):
        module = Q.GatedResidual(args, use_combine=combine)
        nn.quantize(module, group_size=64, bits=4)
        module.load_weights(list(tensors(prefix).items()), strict=True)
        module.eval()
        mx.eval(module.parameters())
        return module

    names = []
    for layer in (0, 3, 23, 47):
        for part in ("attn", "mlp"):
            names.append((f"language_model.model.layers.{layer}.{part}_hyper_connection", True))
    names += [("language_model.model.hyper_connection_mixer", False),
              ("mtp.layers.0.attn_hyper_connection", True),
              ("mtp.layers.0.mlp_hyper_connection", True),
              ("mtp.hyper_connection_mixer", False)]
    width = args.hc_count * args.hidden_size
    results = {}
    HCD.reset_for_tests()
    HCD.set_hc_row_exact_enabled(True)
    for name, combine in names:
        module = build(name, combine)
        cells = []
        for r in a.rows:
            for case in range(a.hc_cases):
                key = mx.random.key(1000 * r + case)
                scale = (0.05, 1.0, 8.0, 40.0)[case % 4]
                x = (mx.random.normal((1, r, width), key=key) * scale).astype(mx.bfloat16)
                mx.eval(x)
                refs = {}
                for arm, on in (("one_row_kernel", True), ("one_row_composed", False)):
                    HCD.set_hc_decode_enabled(on)
                    parts = [module(x[:, j : j + 1]) for j in range(r)]
                    if combine:
                        refs[arm] = (mx.concatenate([p[0] for p in parts], axis=1),
                                     mx.concatenate([p[2] for p in parts], axis=1))
                    else:
                        refs[arm] = (mx.concatenate(parts, axis=1),)
                # the route's class swap on the projections
                for child in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
                    if child in module and type(module[child]) is nn.QuantizedLinear:
                        module[child].__class__ = RE._subclass(RE._RowExactQuantizedLinear, nn.QuantizedLinear)
                HCD.set_hc_decode_enabled(True)
                before = HCD.hc_decode_status()["row_exact_calls"]
                record = REV.Window(r)
                with REV.window(record), Q._declared_width(1):
                    got = module(x)
                got = (got[0], got[2]) if combine else (got,)
                served = HCD.hc_decode_status()["row_exact_calls"] - before
                for child in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
                    if child in module:
                        module[child].__class__ = nn.QuantizedLinear
                HCD.set_hc_decode_enabled(False)
                ok_kernel = all(_same(g, w) for g, w in zip(got, refs["one_row_kernel"]))
                ok_comp = all(_same(g, w) for g, w in zip(got, refs["one_row_composed"]))
                cells.append({"rows": r, "case": case, "served_by_kernel": served == 1,
                              "equal_one_row_kernel": ok_kernel, "equal_one_row_composed": ok_comp,
                              "identical": ok_kernel and ok_comp and served == 1,
                              "max_abs_diff": max(_maxdiff(g, w) for g, w in zip(got, refs["one_row_kernel"]))})
        results[name] = cells
        print("hc", name, sum(c["identical"] for c in cells), "/", len(cells), flush=True)
        del module
    shards.clear()
    mx.clear_cache()
    return results


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
