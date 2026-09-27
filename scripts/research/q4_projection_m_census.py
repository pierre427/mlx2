#!/usr/bin/env python3
"""Research-only call-time Q4 projection shape census for the pinned 27B model.

This observes host shapes before QuantizedLinear calls in a direct model
process. It does not measure serving traffic, throughput, or GPU kernel choice.
Default mode is CPU/static; --run requires an external CPG GPU lease and the
owned service-quiesce wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = Path("~/mlx-models/Qwen3.8-27B-oQ4e-mtp")
SOURCE_PATHS = (
    "src/mlx2/runtime/models/qwen38_27b.py",
    "src/mlx2/runtime/models/qwen3_5.py",
    "src/mlx2/runtime/models/cache.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/adapters/registry.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_kind(name: str) -> str:
    for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                   "up_proj", "down_proj", "mtp.fc", "lm_head"):
        if name.endswith(suffix):
            return suffix
    return "other"


class Census:
    def __init__(self) -> None:
        self.active_phase: str | None = None
        self.counts: dict[tuple, int] = {}
        self.examples: dict[tuple, str] = {}

    @contextmanager
    def phase(self, label: str):
        if self.active_phase is not None:
            raise RuntimeError("nested phase is not admitted")
        self.active_phase = label
        try:
            yield
        finally:
            self.active_phase = None

    def record(self, name: str, module, x) -> None:
        if self.active_phase is None:
            raise RuntimeError("unattributed QuantizedLinear call")
        shape = tuple(int(v) for v in x.shape)
        if len(shape) < 2:
            raise RuntimeError(f"unexpected QuantizedLinear input: {shape}")
        b = shape[0] if len(shape) == 3 else 1
        s = shape[1] if len(shape) == 3 else shape[0]
        m = b * s
        k = shape[-1]
        n = int(module.weight.shape[0])
        key = (self.active_phase, module_kind(name), b, s, m, k, n,
               int(module.bits), int(module.group_size))
        self.counts[key] = self.counts.get(key, 0) + 1
        self.examples.setdefault(key, name)

    def rows(self) -> list[dict]:
        labels = ("phase", "kind", "batch", "sequence", "M", "K", "N",
                  "bits", "group_size")
        return [{**dict(zip(labels, key)), "calls": count,
                 "example_module": self.examples[key]}
                for key, count in sorted(self.counts.items())]


def install(model, census: Census):
    """Observe eligible Q4 modules by class swap, restoring all classes later."""
    from mlx import nn

    handle = []
    subclass = {}
    for name, module in model.named_modules():
        if type(module) is not nn.QuantizedLinear:
            continue
        if int(module.bits) != 4 or int(module.group_size) != 64:
            continue
        original = type(module)
        if original not in subclass:
            class TraceQuantized(original):
                def __call__(self, x):
                    census.record(names[id(self)], self, x)
                    return super().__call__(x)
            subclass[original] = TraceQuantized
        names[id(module)] = name
        handle.append((module, original))
        module.__class__ = subclass[original]
    return handle


names: dict[int, str] = {}


def remove(handle) -> None:
    for module, original in handle:
        module.__class__ = original
        names.pop(id(module), None)


def _cache_positions(cache) -> list:
    out = []
    for entry in cache:
        if hasattr(entry, "_rollback_position"):
            value = getattr(entry, "_rollback_positions", None)
            if value is None:
                out.append(int(entry._rollback_position))
            else:
                positions = tuple(int(x) for x in value)
                out.append(positions[0] if len(positions) == 1 else positions)
        else:
            out.append(int(entry.size()))
    return out


def run(model_path: Path, prompt: str, *, include_batch: bool) -> tuple[list[dict], dict]:
    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models.cache import make_prompt_cache

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("Metal GPU is required")
    adapter = resolve_adapter(str(model_path))(str(model_path))
    model, tok = adapter.model, adapter.tokenizer
    ids = list(tok.encode(prompt * 4, add_special_tokens=False))[:32]
    if len(ids) < 8:
        raise RuntimeError("need at least eight prompt tokens")
    census = Census()
    handle = install(model, census)
    if not handle:
        raise RuntimeError("no 4-bit group-64 QuantizedLinear modules found")
    phase_shapes = {}
    try:
        cache = list(make_prompt_cache(model))
        with census.phase("b1_prefill"):
            result = model(mx.array([ids], dtype=mx.uint32), cache=cache)
            mx.eval(result)
        phase_shapes["b1_prefill"] = list(result.shape)
        with census.phase("b1_ordinary_decode"):
            result = model(mx.array([[ids[-1]]], dtype=mx.uint32), cache=cache)
            mx.eval(result)
        phase_shapes["b1_ordinary_decode"] = list(result.shape)

        started = []
        try:
            for entry in cache:
                entry.start_speculation()
                started.append(entry)
            before = _cache_positions(cache)
            with census.phase("b1_target_verify_m3"):
                result = model(mx.array([ids[-3:]], dtype=mx.uint32), cache=cache)
                mx.eval(result)
            phase_shapes["b1_target_verify_m3"] = list(result.shape)
            trims = [entry.trim(3) for entry in cache]
            after = _cache_positions(cache)
            if any(n != 3 for n in trims) or after != before:
                raise RuntimeError(
                    f"target verify rollback failed: trims={trims}, "
                    f"before={before}, after={after}")
        finally:
            for entry in reversed(started):
                entry.stop_speculation()

        if getattr(model, "mtp_step", None) is not None:
            hidden_size = int(model.language_model.args.hidden_size)
            mtp_cache = model.language_model.make_mtp_cache()
            with census.phase("b1_mtp_draft_synthetic_hidden"):
                result, _ = model.mtp_step(
                    mx.zeros((1, 1, hidden_size), dtype=mx.bfloat16),
                    mx.array([[ids[-1]]], dtype=mx.uint32), mtp_cache)
                mx.eval(result)
            phase_shapes["b1_mtp_draft_synthetic_hidden"] = list(result.shape)

        if include_batch:
            for b in (4, 8):
                cache = list(make_prompt_cache(model))
                with census.phase(f"b{b}_equal_prefix_prefill"):
                    result = model(mx.array([ids] * b, dtype=mx.uint32), cache=cache)
                    mx.eval(result)
                phase_shapes[f"b{b}_equal_prefix_prefill"] = list(result.shape)
                with census.phase(f"b{b}_equal_prefix_decode"):
                    result = model(mx.array([[ids[-1]]] * b, dtype=mx.uint32), cache=cache)
                    mx.eval(result)
                phase_shapes[f"b{b}_equal_prefix_decode"] = list(result.shape)
    finally:
        remove(handle)
    return census.rows(), {"adapter_identity": str(adapter.identity),
                           "quantized_modules": len(handle),
                           "prompt_token_count": len(ids),
                           "prompt_token_sha256": hashlib.sha256(
                               json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                           "phase_output_shapes": phase_shapes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ARTIFACT)
    parser.add_argument("--prompt", default="Explain the cache contract briefly. ")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--include-equal-prefix-batches", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"missing artifact: {args.model}")
    source_hashes = {p: sha256(ROOT / p) for p in SOURCE_PATHS}
    receipt = {"schema": "mlx2.q4-projection-m-census.v1",
               "scope": "direct_model_call_shapes_not_live_serving_frequency",
               "repo_head": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "model": str(args.model.resolve()),
               "model_config_sha256": sha256(args.model / "config.json"),
               "model_index_sha256": sha256(args.model / "model.safetensors.index.json"),
               "harness_sha256": sha256(Path(__file__)),
               "source_sha256": source_hashes,
               "planned_phases": ["b1_prefill", "b1_ordinary_decode",
                                  "b1_target_verify_m3", "b1_mtp_draft_synthetic_hidden"]
                                 + (["b4_equal_prefix_prefill", "b4_equal_prefix_decode",
                                     "b8_equal_prefix_prefill", "b8_equal_prefix_decode"]
                                    if args.include_equal_prefix_batches else [])}
    if args.run:
        if not args.i_own_the_gpu:
            raise SystemExit("refusing GPU without --i-own-the-gpu")
        sys.path.insert(0, str(ROOT / "src"))
        rows, metadata = run(args.model, args.prompt,
                             include_batch=args.include_equal_prefix_batches)
        if {p: sha256(ROOT / p) for p in SOURCE_PATHS} != source_hashes:
            raise RuntimeError("controlling source changed during census")
        receipt.update({"rows": rows, **metadata})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
