"""A/B decode levers on Flash-Next self-MTP (triage 2026-09-25).

Arms: --sort-threshold N (MoE sorted-gather cut), --fused-down-m3 (admit the
M=3 fused down candidate), --legacy-trim (restore pre-bcbb0012 segmented
rollback: per-row full replay + unconditional re-join), and the
MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER env vars (recorded).

Derived from bench_kv_page_align.py (probes-20260924).

MLX writes a ``slice_update`` in place only when the destination's buffer is
at most one 16 KiB page larger than the array (``is_donatable``), and its
buffer cache can hand back a recycled buffer up to one page larger than the
page-rounded request.  ``KVCache.update_and_fetch`` grows to ``prev + k*step``
where ``prev`` is arbitrary after a prefill or MTP trim, so a bank whose byte
size is not page aligned can land in a buffer that never donates; every
verify write then copies the whole bank (MTPLX v2.12.0, commit 5e938c75).

Arms (one per process; alternate runs):
  --align            grow to a whole multiple of ``step`` (page aligned)
  MLX_MAX_MB_PER_BUFFER=<mb> in the environment for the command-buffer cap

GPU only, single request, self-MTP, greedy:
  PYTHONPATH=src .venv/bin/python scripts/bench_kv_page_align.py --i-own-the-gpu \
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --prompt-file big.txt \
      --context 57000 --gen 1024 [--align] --out run.json
"""

import argparse
import json
import os
import sys
import time

import mlx.core as mx

PAGE = 16384


def aligned_update_and_fetch(self, keys, values):
    prev = self.offset
    if self.keys is None or prev + keys.shape[2] > self.keys.shape[2]:
        (B, n_kv_heads, _, k_head_dim) = keys.shape
        v_head_dim = values.shape[3]
        total = (prev + keys.shape[2] + self.step - 1) // self.step * self.step
        grow = total - prev
        new_k = mx.zeros((B, n_kv_heads, grow, k_head_dim), keys.dtype)
        new_v = mx.zeros((B, n_kv_heads, grow, v_head_dim), values.dtype)
        if self.keys is not None:
            self.keys = mx.concatenate([self.keys[..., :prev, :], new_k], axis=2)
            self.values = mx.concatenate([self.values[..., :prev, :], new_v], axis=2)
        else:
            (self.keys, self.values) = (new_k, new_v)
    self.offset += keys.shape[2]
    self.keys[..., prev : self.offset, :] = keys
    self.values[..., prev : self.offset, :] = values
    return self.keys_and_values()


def fused_counters(model):
    totals = {}
    for _, module in model.named_modules():
        for name in ("fused_gdn_decode_calls", "fused_gdn_decode_fallbacks",
                     "fused_gdn_verify_calls", "fused_gdn_verify_fallbacks"):
            totals[name] = totals.get(name, 0) + int(getattr(module, name, 0) or 0)
        for reason, n in (getattr(module, "fused_gdn_verify_fallback_reasons", None) or {}).items():
            key = f"verify_fallback:{reason}"
            totals[key] = totals.get(key, 0) + int(n)
    return totals


def fused_counters_prefill(model):
    totals = {}
    for _, module in model.named_modules():
        for name, value in vars(module).items():
            if (name.startswith("fused_gdn_prefill") or "weighted_sum" in name) and isinstance(value, int):
                totals[name] = totals.get(name, 0) + value
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--context", type=int, default=57000)
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--num-draft", type=int, default=2)
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--batch", type=int, default=1, help="concurrent requests (lanes)")
    ap.add_argument("--moe-omlx", default=None,
                    help="path to omlx moe_verify_gather.py; route 2-8 row MoE calls to it")
    ap.add_argument("--sort-threshold", type=int, default=None)
    ap.add_argument("--fused-down-m3", action="store_true")
    ap.add_argument("--legacy-trim", action="store_true")
    ap.add_argument("--post-env", action="append", default=[],
                    help="K=V set after the adapter configures the environment")
    ap.add_argument("--gdn-prefill-fused", action="store_true")
    ap.add_argument("--moe-weighted-sum", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime import generate as G
    from mlx2.runtime.models import cache as C
    from mlx2.runtime.sample_utils import LaneRNG

    growth = {"events": 0, "misaligned": 0}
    base = aligned_update_and_fetch if a.align else C.KVCache.update_and_fetch

    def counted(self, keys, values):
        before = None if self.keys is None else self.keys.shape[2]
        out = base(self, keys, values)
        if self.keys.shape[2] != before:
            growth["events"] += 1
            growth["misaligned"] += self.keys.nbytes % PAGE != 0
        return out

    C.KVCache.update_and_fetch = counted

    moe = {"omlx": 0, "stock": 0}
    if a.moe_omlx:
        import importlib.util
        from mlx2.runtime.models import qwen3_next as QN

        spec = importlib.util.spec_from_file_location("omlx_moe_verify_gather", a.moe_omlx)
        og = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(og)
        original = QN.FusedGateUpSwitchGLU.__call__

        def routed(self, x, indices, scores=None, variant="scalar"):
            rows = x.size // x.shape[-1]
            top_k = indices.shape[-1]
            if (2 <= rows <= og.MAX_ROWS and og.supported(self.gate_up_proj, x.dtype)
                    and og.supported(self.down_proj, x.dtype)):
                moe["omlx"] += 1
                pairs = indices.reshape(rows * top_k)
                gu = og.gather_qmv(self.gate_up_proj, x.reshape(rows, -1), pairs, top_k)
                half = self.hidden_dims
                hidden = self.activation(gu[..., half:], gu[..., :half])
                y = og.gather_qmv(self.down_proj, hidden, pairs, 1)
                y = y.reshape(*indices.shape, -1)
                object.__setattr__(self, "_last_fused_variant", None)
                if scores is not None:
                    return (y * scores[..., None]).sum(axis=-2)
                return y
            moe["stock"] += 1
            return original(self, x, indices, scores=scores, variant=variant)

        QN.FusedGateUpSwitchGLU.__call__ = routed

    from mlx2.runtime.models import switch_layers as SL
    if a.sort_threshold is not None:
        SL._GATHER_SORT_MIN_ASSIGNMENTS = a.sort_threshold
    if a.fused_down_m3:
        from mlx2.runtime.models import qwen4_fused_moe as FM
        FM.QUALIFIED_TOKEN_WIDTHS = (1, 3)
    if a.legacy_trim:
        from mlx2.runtime import segmented_batch_cache as SBC
        cls = SBC.SegmentedBatchArraysCache
        cls._shared_replay = staticmethod(lambda fn: fn)
        _orig_trim = cls.trim_ragged

        def legacy_trim(self, counts, *, validate=True):
            self._written_slots = set()
            return _orig_trim(self, counts, validate=validate)

        cls.trim_ragged = legacy_trim
    from mlx2.runtime.models import qwen3_next as QN
    fused_seen = {}
    _orig_try = QN._try_qwen4_fused_down

    def counted_try(hidden, indices, *args, **kw):
        out = _orig_try(hidden, indices, *args, **kw)
        key = f"{'fused' if out is not None else 'stock'}:M={indices.size // indices.shape[-1]}"
        fused_seen[key] = fused_seen.get(key, 0) + 1
        return out

    QN._try_qwen4_fused_down = counted_try
    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    for item in a.post_env:
        key, _, value = item.partition("=")
        os.environ[key] = value
    toggled = {"gdn": 0, "moe": 0}
    for _, module in adapter.model.named_modules():
        if a.gdn_prefill_fused and hasattr(module, "set_fused_gdn_prefill_mode"):
            module.set_fused_gdn_prefill_mode("fused"); toggled["gdn"] += 1
        if a.moe_weighted_sum and hasattr(module, "set_moe_weighted_sum"):
            module.set_moe_weighted_sum(True); toggled["moe"] += 1
    print("TOGGLED", toggled, flush=True)
    mx.eval(adapter.model.parameters())
    print("LOADED", f"active={mx.get_active_memory()/2**30:.1f}GiB", flush=True)
    ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))
    if len(ids) < a.context:
        ap.error(f"prompt file has {len(ids)} tokens < --context {a.context}")
    if len(ids) < a.context * a.batch:
        ap.error(f"prompt file has {len(ids)} tokens < --context x --batch")
    prompts = [ids[i * a.context : (i + 1) * a.context] for i in range(a.batch)]
    gen = G.BatchGenerator(
        adapter.model, completion_batch_size=a.batch, prefill_batch_size=1,
        prefill_step_size=a.prefill_step,
        self_mtp={"num_draft": a.num_draft, "persistent": True, "rate_gate": False,
                  "prefill_step_size": a.prefill_step},
    )
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    uids = gen.insert(prompts, max_tokens=[a.gen] * a.batch,
                      lane_rngs=[LaneRNG(1 + i) for i in range(a.batch)],
                      self_mtp_configs=[{"sampling_temp": 0.0}] * a.batch)
    emitted, t_first, step_times = 0, None, []
    tokens = {}
    done = set()
    started = set()
    total = a.gen * a.batch
    last = None
    try:
        while emitted < total and len(done) < a.batch:
            _p, responses = gen.next()
            now = time.perf_counter()
            started.update(r.uid for r in responses)
            if responses and t_first is None and started >= set(uids):
                # Every lane is decoding: the steady-state window starts here.
                t_first = now
                prefill_peak = mx.get_peak_memory()
            elif responses and last is not None and t_first is not None:
                step_times.append((now - last, len(responses)))
            if responses:
                last = now
            for r in responses:
                tokens.setdefault(r.uid, []).append(int(r.token))
                if r.finish_reason:
                    done.add(r.uid)
            if t_first is not None:
                emitted += len(responses)
        t_end = time.perf_counter()
    finally:
        gen.close()
    decode_s = t_end - t_first
    rec = {
        "label": a.label, "align": a.align,
        "max_mb_per_buffer": os.environ.get("MLX_MAX_MB_PER_BUFFER"),
        "max_ops_per_buffer": os.environ.get("MLX_MAX_OPS_PER_BUFFER"),
        "sort_threshold": SL._GATHER_SORT_MIN_ASSIGNMENTS,
        "fused_down_m3": a.fused_down_m3, "legacy_trim": a.legacy_trim,
        "moe_down_dispatch": fused_seen,
        "post_env": a.post_env,
        "gdn_prefill": {k: v for k, v in fused_counters_prefill(adapter.model).items() if v},
        "segmented": {k: v for k, v in __import__("mlx2.runtime.segmented_self_mtp", fromlist=["x"]).segmented_self_mtp_stats().items() if isinstance(v, int) and v},
        "step_ms_mean": 1e3 * sum(t for t, _ in step_times) / max(1, len(step_times)),
        "tokens_per_step": sum(n for _, n in step_times) / max(1, len(step_times)),
        "context": a.context, "emitted": emitted,
        "prefill_s": t_first - t0, "decode_s": decode_s,
        "batch": a.batch,
        "decode_tps": emitted / decode_s,
        "fused_gdn": fused_counters(adapter.model),
        "steps": len(step_times),
        "peak_gib": mx.get_peak_memory() / 2**30,
        "prefill_peak_gib": prefill_peak / 2**30,
        "growth": growth,
        "moe": moe,
        "step_ms_p50": sorted(t for t, _ in step_times)[len(step_times) // 2] * 1e3,
        "tokens_sha": [
            __import__("hashlib").sha256(json.dumps(tokens.get(u, [])).encode()).hexdigest()[:16]
            for u in uids
        ],
        "mlx": mx.__version__,
    }
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
