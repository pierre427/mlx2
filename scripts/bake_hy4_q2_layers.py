#!/usr/bin/env python3
"""Bake a Hy4-flash MLX pack with some layers' routed experts re-quantized.

The derived artifact is the source pack with:

* the chosen layers' ``model-experts-L<k>.safetensors`` shards rewritten.
  Each routed-expert projection (``switch_mlp.{gate,up,down}_proj``) is
  dequantized from the pack's planes and re-quantized round-to-nearest to
  ``--bits/--group-size`` (MLX affine), exactly as ``scripts/layer_kl.py
  scan`` measured it (``mlx2.layer_kl._rtn_planes``, fp32 quantize, scales
  and biases cast back to the pack's dtype);
* per-module quantization overrides for those projections in
  ``config.json`` (``quantization`` and ``quantization_config``) and an
  ``mlx2_derivation`` record (source checkpoint, layers, law, mlx2 commit);
* every other file hardlinked (copied when a hardlink is impossible).

The output is written to ``<out>.partial`` and renamed when complete; a
nonempty ``<out>`` is refused.  The derived checkpoint is unqualified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def requantize_shard(src: Path, dst: Path, layer: int, quant: dict, *, bits: int, group_size: int,
                     mode: str = "affine") -> dict:
    """Rewrite one experts shard with the layer's switch_mlp banks at ``bits``."""
    import mlx.core as mx

    from mlx2.layer_kl import _rtn_planes

    tensors = mx.load(str(src))
    out = dict(tensors)
    receipt = {}
    for proj in PROJECTIONS:
        stem = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
        if f"{stem}.weight" not in tensors or f"{stem}.scales" not in tensors:
            raise ValueError(f"{src.name} has no quantized {stem}")
        old = quant.get(stem) if isinstance(quant.get(stem), dict) else quant
        if old.get("mode", "affine") != "affine":
            raise ValueError(f"{stem} is {old.get('mode')!r}; only affine banks are re-quantized")
        scales = tensors[f"{stem}.scales"]
        dense = mx.dequantize(tensors[f"{stem}.weight"], scales, tensors.get(f"{stem}.biases"),
                              group_size=int(old["group_size"]), bits=int(old["bits"]), mode="affine")
        planes = _rtn_planes(mx, dense, group_size=group_size, bits=bits, mode=mode)
        out[f"{stem}.weight"] = planes[0]
        out[f"{stem}.scales"] = planes[1].astype(scales.dtype)
        out[f"{stem}.biases"] = planes[2].astype(scales.dtype)
        receipt[proj] = {"from": {"bits": int(old["bits"]), "group_size": int(old["group_size"])},
                         "to": {"bits": bits, "group_size": group_size, "mode": mode},
                         "shape": list(dense.shape)}
        del dense, planes
        mx.clear_cache()
    mx.eval(list(out.values()))
    mx.save_safetensors(str(dst), out, metadata={"format": "mlx"})
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pack", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layers", required=True, help="comma list of decoder layers")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="compute on the CPU device")
    a = parser.parse_args(argv)
    import mlx.core as mx

    if a.cpu:
        mx.set_default_device(mx.cpu)
    pack, out = a.pack.expanduser().resolve(), a.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    layers = sorted({int(x) for x in a.layers.split(",") if x.strip()})
    config = json.loads((pack / "config.json").read_text())
    quant = config.get("quantization") or {}
    index = json.loads((pack / "model.safetensors.index.json").read_text())
    shards = {}
    for layer in layers:
        files = {index["weight_map"].get(f"model.layers.{layer}.mlp.switch_mlp.{p}.weight")
                 for p in PROJECTIONS}
        if len(files) != 1 or None in files:
            raise SystemExit(f"layer {layer}: routed experts are not in one shard ({files})")
        name = files.pop()
        others = [k for k, v in index["weight_map"].items()
                  if v == name and not k.startswith(f"model.layers.{layer}.mlp.switch_mlp.")]
        if others:
            raise SystemExit(f"layer {layer}: shard {name} also holds {len(others)} other tensors")
        shards[layer] = name
    from mlx2.adapters.hy_v4_flash import inspect_artifact

    source_fp = inspect_artifact(pack)["identity"]["fingerprint"]
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        rewritten = set(shards.values())
        methods = {}
        for src in sorted(pack.rglob("*")):
            rel = src.relative_to(pack)
            if src.is_dir() or rel.parts[0].startswith(".") or str(rel) in rewritten or str(rel) == "config.json":
                continue
            methods[str(rel)] = _link_or_copy(src, tmp / rel)
        receipts = {}
        for layer, name in shards.items():
            print(f"[bake] layer {layer} -> {a.bits}-bit g{a.group_size} ({name})", flush=True)
            receipts[str(layer)] = requantize_shard(pack / name, tmp / name, layer, quant,
                                                    bits=a.bits, group_size=a.group_size)
        new_quant = dict(quant)
        for layer in layers:
            for proj in PROJECTIONS:
                new_quant[f"model.layers.{layer}.mlp.switch_mlp.{proj}"] = {
                    "group_size": a.group_size, "bits": a.bits, "mode": "affine"}
        config["quantization"] = new_quant
        if "quantization_config" in config:
            config["quantization_config"] = new_quant
        config["mlx2_derivation"] = {
            "kind": "routed_expert_requantization",
            "source_checkpoint": f"{pack.name}@sha256:{source_fp}",
            "layers": layers, "bits": a.bits, "group_size": a.group_size, "mode": "affine",
            "law": "round-to-nearest from the pack's dequantized planes (mlx2.layer_kl._rtn_planes)",
            "mlx2_commit": _commit(), "layers_receipt": receipts, "qualified": False,
        }
        (tmp / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        os.replace(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    fp = inspect_artifact(out)["identity"]["fingerprint"]
    print(json.dumps({"out": str(out), "checkpoint": f"{out.name}@sha256:{fp}",
                      "source": f"{pack.name}@sha256:{source_fp}", "layers": layers,
                      "linked": sum(v == "hardlink" for v in methods.values()),
                      "copied": sum(v == "copy" for v in methods.values())}), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
