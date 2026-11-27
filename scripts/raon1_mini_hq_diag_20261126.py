#!/usr/bin/env python3
"""Raon-1-mini mixed-affine diagnostics before recipe v4 (Pierre 2026-11-26: "another round ... in a v4").

Two questions the v3 A/B left open:

* ``reference`` + ``eval``: a held-out KL measurement that can separate mixes the 83-token vendor
  oracle cannot (v1 and v3 tie there).  Reference = the exact mxfp8e source; texts = ``rows`` rows of
  the recipe's calibration generator *after* the calibration rows and the v3 probe rows, so neither
  GPTQ nor the KL probe has seen them.  Every model runs the recipe's reference laws (no merged
  projections, ops mHC, no reasoning-effort steering).
* ``floor``: why the v3 probe charged 8-bit experts nearly as much as 4-bit ones in early layers
  (layer 1: a8g64 0.0044 vs a4g64 0.0055).  For a few layers, the same 4 probe rows v3 used and the
  same calibration, each variant swaps that layer's three routed expert banks and scores KL:
  ``mxfp8_roundtrip`` (dequantize + requantize the source codes: must be exactly 0, a control of
  the substitution), ``bf16_dense`` (the dequantized bank on the dense kernel), ``a8g64_rtn``,
  ``a8g64_gptq``, ``a4g64_rtn``, ``a4g64_gptq`` (the GPTQ variants are the recipe's planes).

Original mlx2 code.  Sanity numbers, single pass, no thermal hold.  Run under the GPU lease.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

POLICY = {"merged_projections": False, "mhc": "ops"}
VARIANTS = ("mxfp8_roundtrip", "bf16_dense", "a8g64_rtn", "a8g64_gptq", "a4g64_rtn", "a4g64_gptq")


def load(path: Path):
    import mlx.core as mx

    from mlx2.adapters import raon1_mini as A
    from mlx2.runtime.models import raon1_mini as R
    from mlx2.runtime.ubc_evict import load_shards_evicting

    artifact = A.inspect_artifact(path)
    model = R.Model(R.ModelArgs.from_dict(artifact["config"]))
    R.install_for_layout(model, artifact["layout"], artifact["config"], artifact["weight_map"].keys())
    root = Path(artifact["identity"]["path"])
    weights = model.sanitize(load_shards_evicting([root / n for n in sorted(set(artifact["weight_map"].values()))]))
    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    model.eval()
    model.configure(A.parse_policy(POLICY))
    model.set_reasoning_effort(None)
    mx.eval(model.parameters())
    return model, artifact


def generator_rows(source: Path, settings, start: int, count: int):
    """Rows ``start .. start+count-1`` after the calibration rows (0-based past calibration)."""
    from transformers import AutoTokenizer

    from mlx2.conversion import raon1_mini_hq as HQ

    tok = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    n_calib = -(-settings.tokens // settings.seq)
    rows = HQ.calibration_sequences(tok, settings.tokens + (start + count) * settings.seq, settings.seq)
    calib, picked = rows[:n_calib], rows[n_calib + start:n_calib + start + count]
    assert len(picked) == count
    return calib, picked


def cmd_reference(args):
    from mlx2 import layer_kl
    from mlx2.conversion import raon1_mini_hq as HQ

    settings = HQ.HQSettings()
    _, rows = generator_rows(args.source, settings, settings.kl_rows, args.rows)
    texts = [{"id": f"heldout-{i}", "split": "held", "ids": list(map(int, ids))} for i, ids in enumerate(rows)]
    model, artifact = load(args.source)
    meta = layer_kl.write_reference(args.out, texts, layer_kl.reference_from_model(model), skip=args.skip,
                                    checkpoint=artifact["checkpoint"], dtype="float16",
                                    source={"rows_after_calibration": [settings.kl_rows, settings.kl_rows + args.rows],
                                            "generator": "raon1_mini_hq.calibration_sequences", "policy": POLICY})
    print(json.dumps({k: meta[k] for k in ("checkpoint", "positions", "texts_sha256", "skip")}))


def cmd_eval(args):
    import mlx.core as mx

    from mlx2 import layer_kl

    meta, texts = layer_kl.load_reference(args.reference)
    out = {}
    for path in args.models:
        started = time.time()
        model, artifact = load(path)
        got = layer_kl.score(model, args.reference, meta, texts, splits=("held",))
        out[artifact["checkpoint"]] = {"path": str(path), **got.as_dict(), "seconds": round(time.time() - started, 1)}
        print(json.dumps({"checkpoint": artifact["checkpoint"], "kl_mean": got.kl_mean, "top1": got.top1}), flush=True)
        del model
        gc.collect()
        mx.clear_cache()
    Path(args.out).write_text(json.dumps({"reference": meta["checkpoint"], "texts_sha256": meta["texts_sha256"],
                                          "positions": meta["positions"], "policy": POLICY, "models": out},
                                         indent=1) + "\n")


def _variant_module(variant, module, H, *, damp):
    """A replacement bank for ``variant`` built from ``module``'s exact (dequantized) weights."""
    import mlx.core as mx

    from mlx2.conversion import raon1_mini_hq as HQ
    from mlx2.runtime.models.switch_layers import SwitchLinear

    dense = HQ._dense_bank(module)
    E, O, I = dense.shape
    base = SwitchLinear(I, O, E, bias=False)
    base.weight = dense
    if variant == "bf16_dense":
        return base
    if variant == "mxfp8_roundtrip":
        if getattr(module, "mode", None) != "mxfp8":
            raise ValueError("mxfp8_roundtrip needs an mxfp8 source bank")
        return base.to_quantized(group_size=module.group_size, bits=module.bits, mode="mxfp8")
    fmt, method = variant.split("_")
    spec = HQ.fmt_spec(fmt)
    if spec.mode != "affine":
        raise ValueError(f"{variant}: the floor variants are affine only (got {spec.mode})")
    bits, group = spec.bits, spec.group
    q = base.to_quantized(group_size=group, bits=bits, mode="affine")
    if method == "gptq":
        (w, sc, bi), _ = HQ.gptq_bank(dense, H, bits, group, damp=damp)
        q["weight"], q["scales"], q["biases"] = w, sc, bi
    mx.eval(q.parameters())
    return q


def cmd_floor(args):
    import mlx.core as mx

    from mlx2 import layer_kl
    from mlx2.conversion import raon1_mini_hq as HQ

    settings = HQ.HQSettings()
    calib, probe = generator_rows(args.source, settings, 0, settings.kl_rows)
    model, artifact = load(args.source)
    started = time.time()
    hessians, _rows, _counts = HQ.calibrate(model, calib, log=lambda m: print(m, flush=True))
    print(f"calibrated {time.time() - started:.0f}s", flush=True)
    texts = [{"id": f"probe-{i}", "split": "dev", "ids": list(map(int, ids))} for i, ids in enumerate(probe)]
    meta = layer_kl.write_reference(args.work / "probe-reference", texts, layer_kl.reference_from_model(model),
                                    skip=settings.kl_skip, checkpoint=artifact["checkpoint"], dtype="float16",
                                    source={"recipe_probe_rows": settings.kl_rows})
    base = layer_kl.score(model, args.work / "probe-reference", meta, texts)
    rows = {"baseline": {"kl_mean": base.kl_mean, "top1": base.top1}, "layers": {}}
    for i in args.layers:
        layer = model.layers[i]
        if not layer.is_moe:
            raise SystemExit(f"layer {i} is not an MoE layer")
        glu = layer.moe.switch_mlp
        rows["layers"][str(i)] = {}
        for variant in args.variants:
            originals = {}
            try:
                for name, key in HQ._BANKS:
                    originals[name] = glu[name]
                    setattr(glu, name, _variant_module(variant, glu[name], hessians[f"{i}.{key}"], damp=settings.damp))
                got = layer_kl.score(model, args.work / "probe-reference", meta, texts)
            finally:
                for name, module in originals.items():
                    setattr(glu, name, module)
                mx.clear_cache()
            rows["layers"][str(i)][variant] = {"kl_mean": got.kl_mean, "kl_minus_baseline": got.kl_mean - base.kl_mean,
                                               "top1": got.top1}
            print(f"layer {i} {variant}: KL-base {got.kl_mean - base.kl_mean:.4g} top1 {got.top1:.4f} "
                  f"({time.time() - started:.0f}s)", flush=True)
    again = layer_kl.score(model, args.work / "probe-reference", meta, texts)
    rows["restore_check"] = "bit-equal" if (again.kl_mean == base.kl_mean and again.top1 == base.top1) else "FAILED"
    rows.update({"checkpoint": artifact["checkpoint"], "positions": meta["positions"],
                 "texts_sha256": meta["texts_sha256"], "policy": POLICY, "seconds": round(time.time() - started, 1)})
    Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")
    if rows["restore_check"] != "bit-equal":
        raise SystemExit("restore check failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--rows", type=int, default=8)
    r.add_argument("--skip", type=int, default=1024)
    e = sub.add_parser("eval")
    e.add_argument("--reference", type=Path, required=True)
    e.add_argument("--models", type=Path, nargs="+", required=True)
    e.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("floor")
    f.add_argument("--source", type=Path, required=True)
    f.add_argument("--work", type=Path, required=True)
    f.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 12, 24])
    f.add_argument("--variants", type=lambda s: s.split(","), default=list(VARIANTS))
    f.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    {"reference": cmd_reference, "eval": cmd_eval, "floor": cmd_floor}[args.cmd](args)


if __name__ == "__main__":
    main()
