#!/usr/bin/env python3
"""Where does Flash-Next prefill time actually go?

Nothing in this repo breaks a Flash-Next prefill down by mechanism.  The
context ladder in ``qualification/runs/flash-next-peer-pr-ab-20260917`` gives
total TTFT; rm15 gives total prefill versus chunk size on the 27B; neither
says how much of a prefill is GDN, how much is QSA, how much is MoE.  Without
that split there is no ceiling estimate for any proposed prefill kernel, so a
proposal cannot pre-register a pass criterion.

The full artifact is 97 GB of resident weights on a 128 GB host, so this
harness does NOT load it.  Instead it instantiates **one real DecoderLayer of
each kind** from the production model module, with the production shapes,
production dtypes and production quantization (group_size 64, bits 4, affine),
randomly initialised.  Tensor-op cost is set by shape and dtype, not by weight
values, so a per-layer measurement scales: multiply by the layer counts read
from the artifact config (36 linear/GDN, 12 full/QSA, 48 MoE, PLE on the
config's ``ple_layer_ids`` only).

Per layer kind it reports:
  * the uninstrumented whole-layer time (the ground truth), and
  * a seam-by-seam breakdown -- hyper-connection, GDN or QSA branch, MoE --
    obtained by calling the same submodules the real ``DecoderLayer.__call__``
    calls, with ``mx.eval`` at each seam.

The seam pass forces synchronization and therefore costs more than the
uninstrumented pass; both totals are reported so the overhead is visible rather
than hidden, and shares are computed from the seam pass.

The GDN recurrence is timed separately against the whole GDN branch, which is
the number that decides whether a faster chunked GDN can move end-to-end TTFT.

Caveats, stated rather than buried:
  * ``cache=None`` and ``ple_layer_ids=[]`` -- PLE and cache management are
    excluded, so this is a lower bound on total prefill and PLE share is 0 by
    construction.  Only layer 2 carries PLE in this artifact.
  * QSA path selection depends on runtime gates (NAX needs batch==1 and
    physical width >= 16384).  With no cache the physical width equals T, so
    which QSA kernel engages here may differ from production at the same T.
  * Random weights route the MoE differently than real weights, but the number
    of activated experts is fixed by ``num_experts_per_tok``, so the work is
    the same.

Metal: run under the GPU queue only.  --dry-run prints the plan and exits.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "mlx2.prefill-component-profile.v1"


def pin_production_environment(model_path):
    """Install the adapter's serving profile BEFORE any tensor module imports.

    Every MLX_QWEN4_* / MLX_GDN_* gate in qwen4_exp is read into a module-level
    constant at import time, so profiling without this measures different code
    paths than production runs.  ``configure_environment`` only sets os.environ
    and returns the profile; it loads no weights.
    """
    from mlx2.adapters.flash_next import configure_environment

    return configure_environment(Path(model_path))


def load_args(model_path):
    """TextModelArgs from the artifact config -- metadata only, no weights."""
    from mlx2.runtime.models.qwen4_exp import TextModelArgs

    config = json.loads((Path(model_path) / "config.json").read_text())
    text_config = config.get("text_config", config)
    args = TextModelArgs.from_dict(text_config)
    quant = config.get("quantization") or {}
    return args, quant, config


def build_layer(args, kind, quant, dtype):
    """One real DecoderLayer of `kind`, in the checkpoint's dtype, quantized
    like production.

    The dtype cast is load-bearing.  MLX initialises parameters as float32 and
    this model never casts them -- production gets bfloat16 purely because
    ``load_weights`` copies bfloat16 tensors out of the checkpoint.  A
    synthetic layer left at float32 moves twice the bytes on every weight read
    and activation, and measures roughly 2.4x slower than production.  That was
    the whole absolute-total bias in the first version of this harness.
    """
    import mlx.core as mx
    from mlx.utils import tree_map

    from mlx2.runtime.models import qwen4_exp

    one = replace(args, num_hidden_layers=1, layer_types=[kind], ple_layer_ids=[])
    layer = qwen4_exp.DecoderLayer(one, 0)
    # MLX has no Module.to(); cast the parameter tree in place.  Must happen
    # BEFORE quantize so the packed scales/biases come out bf16 as they are in
    # the checkpoint.
    layer.update(tree_map(lambda p: p.astype(dtype), layer.parameters()))
    qwen4_exp.nn.quantize(
        layer,
        group_size=quant.get("group_size", 64),
        bits=quant.get("bits", 4),
        mode=quant.get("mode", "affine"),
    )
    layer.eval()
    mx.eval(layer.parameters())
    return layer


def dtype_audit(mx, layer):
    """{dtype: MB} over the layer's parameters, so a dtype regression is visible."""
    flat = {}

    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            elif isinstance(v, mx.array):
                flat[pre + k] = v

    walk(layer.parameters())
    totals = {}
    for arr in flat.values():
        totals[str(arr.dtype)] = totals.get(str(arr.dtype), 0.0) + arr.nbytes / 1e6
    return {k: round(v, 1) for k, v in sorted(totals.items())}, round(
        sum(totals.values()), 1
    )


def timeit(fn, mx, warmup, reps):
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - t0)
    return samples


def causal_mask(mx, T):
    """Boolean causal mask, True = attend.  QSASelection.dense_mask() does
    ``causal_mask & sparse``, so this must be bool, not an additive float mask."""
    return mx.tril(mx.ones((T, T), dtype=mx.bool_))[None, None]


def chain_cost(mx, layer, kind, T, chain, warmup, reps, args):
    """Per-layer cost when K layers are queued before a single eval.

    A one-layer measurement exposes CPU dispatch time that production hides: in
    a real forward all 48 layers are queued back to back, so dispatch of layer
    N+1 overlaps GPU execution of layer N.  Chaining K calls of the same layer
    into one graph and evaluating once measures how much of the isolated
    per-layer cost was exposed dispatch rather than device work.

    DecoderLayer maps the hyper-connection state to itself (hc_count*hidden
    wide), so the output feeds straight back in.

    Caveat, and it biases the result DOWN: reusing one layer keeps its weights
    cache-warm across the chain, whereas production walks 48 distinct layers
    with 48 distinct expert tables.  So this measures dispatch hiding cleanly
    but understates any bandwidth-bound component.
    """
    hyper_width = args.hc_count * args.hidden_size
    x = mx.random.normal((1, T, hyper_width)).astype(mx.bfloat16)
    input_ids = mx.zeros((1, T), dtype=mx.uint32)
    mask = None if kind == "linear_attention" else causal_mask(mx, T)
    mx.eval(x)

    def run():
        h = x
        for _ in range(chain):
            h = layer(h, input_ids, mask=mask, cache=None)
        return h

    samples = timeit(run, mx, warmup, reps)
    return {
        "chain": chain,
        "chain_total_ms": statistics.median(samples) * 1e3,
        "chain_ms_per_layer": statistics.median(samples) * 1e3 / chain,
        "chain_min_ms": min(samples) * 1e3,
        "chain_max_ms": max(samples) * 1e3,
        "chain_n_samples": len(samples),
    }


def profile_layer(mx, layer, kind, T, warmup, reps, args):
    """Uninstrumented total, then a seam-by-seam pass."""
    from mlx2.runtime.models.qwen4_exp import _apply_inject

    B = 1
    # A DecoderLayer carries the full hyper-connection state, hc_count streams
    # of hidden_size; GatedResidual returns mixed [B,T,hidden] plus the
    # untouched [B,T,hc_count*hidden] residual and an inject gate.
    hyper_width = args.hc_count * args.hidden_size
    x = mx.random.normal((B, T, hyper_width)).astype(mx.bfloat16)
    input_ids = mx.zeros((B, T), dtype=mx.uint32)
    mx.eval(x)
    mask = None if kind == "linear_attention" else causal_mask(mx, T)

    # ---- uninstrumented whole layer (what production pays per layer)
    total = timeit(lambda: layer(x, input_ids, mask=mask, cache=None),
                   mx, warmup, reps)

    # ---- seam pass: same submodules, same order, eval at each boundary
    seams = {}

    def run_seams():
        h = x
        (mixed, residual, inject) = layer.attn_hyper_connection(h)
        mx.eval(mixed, residual, inject)
        t0 = time.perf_counter()
        if layer.is_linear:
            branch = layer.linear_attn(mixed, None, None)
        else:
            branch = layer.self_attn(mixed, mask, None)
        mx.eval(branch)
        seams.setdefault("branch", []).append(time.perf_counter() - t0)
        h = _apply_inject(residual, branch, inject)
        (mixed2, residual2, inject2) = layer.mlp_hyper_connection(h)
        mx.eval(mixed2, residual2, inject2)
        t1 = time.perf_counter()
        out = layer.mlp(mixed2)
        mx.eval(out)
        seams.setdefault("mlp", []).append(time.perf_counter() - t1)
        return _apply_inject(residual2, out, inject2)

    seams.clear()
    whole = timeit(run_seams, mx, warmup, reps)

    # GDN recurrence in isolation, only for linear layers
    recurrence = None
    if kind == "linear_attention":
        gdn = layer.linear_attn
        orig = gdn._gated_delta_update

        def timed_update(*a, **kw):
            # Materialize this call's own inputs first.  mx.eval() with no
            # arguments is a no-op, so without this the sync point below absorbs
            # the in_proj/conv1d/gate backlog and over-reports the recurrence
            # roughly tenfold.
            mx.eval(*[t for t in a if isinstance(t, mx.array)])
            t0 = time.perf_counter()
            r = orig(*a, **kw)
            mx.eval(r[0], r[1])
            recurrence["t"].append(time.perf_counter() - t0)
            return r

        recurrence = {"t": []}
        gdn._gated_delta_update = timed_update
        try:
            timeit(lambda: layer(x, input_ids, mask=None, cache=None),
                   mx, warmup, reps)
        finally:
            gdn._gated_delta_update = orig

    def med(xs):
        return statistics.median(xs) * 1e3

    out = {
        "kind": kind,
        "T": T,
        "layer_total_ms": med(total),
        "layer_total_min_ms": min(total) * 1e3,
        "layer_total_max_ms": max(total) * 1e3,
        "seam_pass_total_ms": med(whole),
        "seams": {k: med(v) for k, v in seams.items()},
        "n_samples": len(total),
    }
    out["hyper_connection_ms"] = out["seam_pass_total_ms"] - sum(out["seams"].values())
    if recurrence and recurrence["t"]:
        out["gdn_recurrence_ms"] = med(recurrence["t"])
        out["gdn_branch_ms"] = out["seams"].get("branch")
        if out["gdn_branch_ms"]:
            out["gdn_recurrence_share_of_branch"] = (
                out["gdn_recurrence_ms"] / out["gdn_branch_ms"]
            )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP",
                   help="artifact to read config.json from (metadata only; weights are NOT loaded)")
    p.add_argument("--tokens", default="1024,4096,16384")
    p.add_argument("--rounds", type=int, default=3, help="repeated profiles; medians per round")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--kinds", default="linear_attention,full_attention")
    p.add_argument("--chain", default="1,4,8,16",
                   help="queue K layer calls before one eval, to separate "
                        "exposed CPU dispatch from device work; empty to skip")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()

    tokens = [int(x) for x in a.tokens.split(",") if x.strip()]
    kinds = [x.strip() for x in a.kinds.split(",") if x.strip()]
    chains = [int(x) for x in a.chain.split(",") if x.strip()]
    env_profile = pin_production_environment(a.model)
    args, quant, config = load_args(a.model)

    text_config = config.get("text_config", config)
    layer_types = text_config.get("layer_types") or [
        "linear_attention" if (i + 1) % text_config["full_attention_interval"] else "full_attention"
        for i in range(text_config["num_hidden_layers"])
    ]
    counts = {
        "linear_attention": sum(1 for t in layer_types if t == "linear_attention"),
        "full_attention": sum(1 for t in layer_types if t != "linear_attention"),
        "total_layers": len(layer_types),
        "ple_layers": len(text_config.get("ple_layer_ids") or []),
    }

    plan = {
        "schema": SCHEMA,
        "model_config_only": a.model,
        "weights_loaded": False,
        "layer_counts": counts,
        "quantization": {"group_size": quant.get("group_size"), "bits": quant.get("bits"),
                         "mode": quant.get("mode", "affine")},
        "architecture": {k: text_config.get(k) for k in (
            "hidden_size", "num_hidden_layers", "full_attention_interval", "num_experts",
            "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim", "hc_count", "hc_lowrank",
            "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
            "linear_value_head_dim", "indexer_budget", "indexer_compress_ratio",
            "max_position_embeddings", "ple_layer_ids")},
        "tokens": tokens, "kinds": kinds,
        "chains": chains,
        "rounds": a.rounds, "warmup": a.warmup, "reps": a.reps,
        "method": "one real DecoderLayer per kind, random init, production quantization",
        "production_environment": env_profile,
    }
    if a.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")

    import mlx.core as mx

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("this harness measures Metal; no GPU available")

    report = dict(plan)
    report["mlx_version"] = mx.__version__
    report["device"] = str(mx.default_device())
    report["host_started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report["profiles"] = []

    dtype = getattr(mx, str(text_config.get("dtype") or "bfloat16"))
    report["param_dtype"] = str(dtype)
    report["checkpoint_dtype"] = text_config.get("dtype")
    layers = {}
    for k in kinds:
        layers[k] = build_layer(args, k, quant, dtype)
        by_dtype, total_mb = dtype_audit(mx, layers[k])
        report.setdefault("param_mb_by_dtype", {})[k] = {
            "by_dtype": by_dtype, "total_mb": total_mb
        }
        print(f"built {k} layer: {total_mb} MB {by_dtype}", flush=True)
    for T in tokens:
        for kind in kinds:
            best = None
            for _ in range(a.rounds):
                r = profile_layer(mx, layers[kind], kind, T, a.warmup, a.reps, args)
                if best is None or r["layer_total_ms"] < best["layer_total_ms"]:
                    best = r
            best["peak_mem_gb"] = mx.get_peak_memory() / 1e9
            mx.reset_peak_memory()
            if chains:
                best["chain_runs"] = [
                    chain_cost(mx, layers[kind], kind, T, c, a.warmup, a.reps, args)
                    for c in chains
                ]
                iso = best["layer_total_ms"]
                for c in best["chain_runs"]:
                    c["vs_isolated"] = iso / c["chain_ms_per_layer"]
                mx.clear_cache()
                best["chain_peak_mem_gb"] = mx.get_peak_memory() / 1e9
                mx.reset_peak_memory()
            report["profiles"].append(best)
            print(json.dumps(best, indent=2), flush=True)
            mx.clear_cache()

    # ---- scale per-layer costs to a whole-prefill estimate
    report["prefill_estimate"] = []
    for T in tokens:
        row = {"T": T}
        by_kind = {p["kind"]: p for p in report["profiles"] if p["T"] == T}
        lin, full = by_kind.get("linear_attention"), by_kind.get("full_attention")
        total_ms = 0.0
        if lin:
            row["gdn_layers_ms"] = lin["layer_total_ms"] * counts["linear_attention"]
            total_ms += row["gdn_layers_ms"]
            if "gdn_recurrence_ms" in lin:
                row["gdn_recurrence_all_layers_ms"] = (
                    lin["gdn_recurrence_ms"] * counts["linear_attention"])
                row["gdn_recurrence_share_of_prefill"] = (
                    row["gdn_recurrence_all_layers_ms"] / row["gdn_layers_ms"])
        if full:
            row["qsa_layers_ms"] = full["layer_total_ms"] * counts["full_attention"]
            total_ms += row["qsa_layers_ms"]
        mlp_ms = sum(p["seams"].get("mlp", 0.0) * (
            counts["linear_attention"] if p["kind"] == "linear_attention"
            else counts["full_attention"]) for p in by_kind.values())
        row["moe_all_layers_ms"] = mlp_ms
        row["estimated_total_ms"] = total_ms
        row["moe_share_of_estimated_total"] = (mlp_ms / total_ms) if total_ms else None
        if row.get("gdn_recurrence_all_layers_ms") and total_ms:
            row["gdn_recurrence_share_of_estimated_total"] = (
                row["gdn_recurrence_all_layers_ms"] / total_ms)
        row["note"] = ("excludes PLE, embedding, lm_head, cache management and "
                       "scheduler overhead; a lower bound on real prefill")
        report["prefill_estimate"].append(row)
        print(json.dumps(row, indent=2), flush=True)

    report["host_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
