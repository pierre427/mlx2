"""Load a dense checkpoint into its mlx2 model class and export a planned quantization.

Original mlx2 code.  This is the mlx-lm convert shape (load dense weights,
``nn.quantize`` with a per-module class predicate, save), driven by a
:mod:`mlx2.conversion.knapsack` plan.  It reuses MLX's own round-to-nearest
quantizers and the model's own ``sanitize`` and ``quant_predicate``; no
kernel is added.  The output is an unqualified candidate artifact.
"""
from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Optional

from . import knapsack as K

COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json",
              "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json",
              "processor_config.json", "added_tokens.json", "tokenizer.model", "tekken.json",
              "LICENSE", "LICENSE.txt", "NOTICE")
SHARD_BYTES = 4 << 30


def model_module(config: dict, name: Optional[str] = None):
    """The runtime model module (``Model``/``ModelArgs``) for a config's ``model_type``."""
    name = name or f"mlx2.runtime.models.{config['model_type']}"
    module = importlib.import_module(name)
    if not hasattr(module, "Model") or not hasattr(module, "ModelArgs"):
        raise ValueError(f"{name} has no Model/ModelArgs")
    return module


def build_model(source: Path, *, module_name: Optional[str] = None):
    """An unloaded model (lazy parameters, no weights read): enough for :func:`knapsack.inventory`."""
    config = json.loads((Path(source) / "config.json").read_text())
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("plan from a dense checkpoint (this one is already quantized)")
    module = model_module(config, module_name)
    return module.Model(module.ModelArgs.from_dict(config)), config


def load_dense(source: Path, *, module_name: Optional[str] = None):
    """The model with its dense checkpoint weights loaded through its own ``sanitize``."""
    import mlx.core as mx

    model, config = build_model(source, module_name=module_name)
    shards = sorted(p for p in Path(source).glob("*.safetensors") if not p.name.startswith("."))
    if not shards:
        raise ValueError(f"no safetensors shards in {source}")
    weights = {}
    for shard in shards:
        weights.update(mx.load(str(shard)))
    sanitize = getattr(model, "sanitize", None)
    if callable(sanitize):
        weights = sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    return model, config


def export(source: Path, destination: Path, plan: dict, *, module_name: Optional[str] = None,
           log=print, validate=None) -> dict:
    """Quantize ``source`` per ``plan`` into ``destination`` (must not exist).

    Everything is written to ``<destination>.partial`` first; ``validate(path)``
    (e.g. the serving adapter's artifact inspection) runs on it, and only then
    is it renamed, so a failed or interrupted export never leaves a directory
    that looks complete.  Its return value is recorded as ``validated``.
    """
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten

    source, destination = Path(source), Path(destination)
    final, destination = destination, destination.with_name(destination.name + ".partial")
    for path in (final, destination):
        if path.exists():
            raise FileExistsError(f"{path} exists")
    K.validate_plan(plan)
    model, config = load_dense(source, module_name=module_name)
    units = K.inventory(model, layer_prefix=plan["layer_prefix"])
    predicate = K.plan_predicate(plan, units)
    quant = K.quantization_config(plan, units)
    nn.quantize(model, class_predicate=predicate)
    # Every unit must now be quantized exactly as planned.
    modules = dict(model.named_modules())
    for unit in units:
        want = predicate(unit.path, None)
        got = modules[unit.path]
        have = (getattr(got, "group_size", None), getattr(got, "bits", None), getattr(got, "mode", "affine"))
        if want and have != (want["group_size"], want["bits"], want["mode"]):
            raise RuntimeError(f"{unit.path} quantized as {have}, plan says {want}")
    # Format.nbytes assumes 2-byte affine scales/biases (a bf16/fp16 source);
    # an fp32 source would store fp32 scales and silently exceed the budget.
    planned = sum(int(modules[u.path][k].nbytes) for u in units if u.role == "planned"
                  for k in ("weight", "scales", "biases") if k in modules[u.path])
    if planned != plan["solution"]["planned_bytes"]:
        raise RuntimeError(f"planned modules store {planned} bytes, plan says "
                           f"{plan['solution']['planned_bytes']} (fp32 source weights?)")
    destination.mkdir(parents=True)
    flat = tree_flatten(model.parameters())
    shards, current, size = [], {}, 0
    for key, value in flat:
        nbytes = int(value.nbytes)
        if current and size + nbytes > SHARD_BYTES:
            shards.append(current)
            current, size = {}, 0
        current[key] = value
        size += nbytes
    if current:
        shards.append(current)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.eval(list(shard.values()))
        mx.save_safetensors(str(destination / name), shard, metadata={"format": "mlx"})
        total += sum(int(v.nbytes) for v in shard.values())
        weight_map.update({k: name for k in shard})
        log(f"wrote {name} ({len(shard)} tensors)")
        shard.clear()
        mx.clear_cache()
    out_config = dict(config)
    out_config.pop("quantization_config", None)
    out_config["quantization"] = quant
    out_config["knapsack_plan"] = K.model_config(plan)
    (destination / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")
    for name in COPY_FILES:
        path = source / name
        if path.is_file():
            shutil.copyfile(path, destination / name)
    (destination / "knapsack-plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    validated = validate(destination) if validate is not None else None
    receipt = {"destination": str(final), "tensors": len(flat), "shards": len(shards), "bytes": total,
               "plan_sha256": plan["plan_sha256"], "validated": validated}
    # The receipt is in place before the rename, so a final directory never lacks one.
    (destination / "knapsack-export-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    destination.rename(final)
    return receipt
