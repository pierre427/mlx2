"""Bounded architecture ablation; no learned-quality claim."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import _load_model_state, file_hash, loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("use a fresh receipt")
    state = json.loads((args.checkpoint / "state.json").read_text())
    c = Config(**state["config"])
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx import nn
        from mlx.utils import tree_flatten

        from mlx2.experimental.hysparse2.model import Model

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(20 << 30)
        mx.set_cache_limit(256 << 20)
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        removed = Model(replace(c, diffusion_layers=0))
        removed.load_weights(
            [
                (name, value)
                for name, value in tree_flatten(model.parameters())
                if not name.startswith("diffusion_student.")
            ],
            strict=True,
        )
        mx.eval(removed.parameters())
        tokens = (mx.arange(128).reshape(2, 64) * 17 + 1) % c.vocab_size

        def objective(m):
            return loss(m, tokens)

        fn = nn.value_and_grad(model, objective)
        no_diff_fn = nn.value_and_grad(removed, objective)
        # Warm both graphs once; measurements below are one repetition each.
        mx.eval(fn(model), no_diff_fn(removed))
        mx.synchronize()
        start = time.perf_counter()
        with_loss, with_grad = fn(model)
        mx.eval(with_loss, with_grad)
        mx.synchronize()
        with_seconds = time.perf_counter() - start
        start = time.perf_counter()
        without_loss, without_grad = no_diff_fn(removed)
        mx.eval(without_loss, without_grad)
        mx.synchronize()
        without_seconds = time.perf_counter() - start
        a, b = dict(tree_flatten(with_grad)), dict(tree_flatten(without_grad))
        deltas = {
            name: float(mx.max(mx.abs(value - b[name])).item())
            for name, value in a.items()
            if name in b
        }
        trunk = {
            name: delta
            for name, delta in deltas.items()
            if name.startswith(
                ("self_decoder.", "cross_decoder.", "semantic_ple.", "mtp_head.")
            )
        }
        # Inference bypass is exact for the same teacher weights.
        model.eval()
        removed.eval()
        with_logits = model(tokens)[0]
        without_logits = removed(tokens)[0]
        inference_error = float(mx.max(mx.abs(with_logits - without_logits)).item())
        report = {
            "schema": "mlx2.hysparse2-diffusion-ablation.v1",
            "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
            "model_parameters": c.capacity()["parameters"],
            "diffusion_parameters": c.capacity()["parameters"]
            - removed.config.capacity()["parameters"],
            "batch": 2,
            "sequence": 64,
            "dtype": "float32",
            "warmed_repetitions": 1,
            "with_diffusion_loss": float(with_loss.item()),
            "without_diffusion_loss": float(without_loss.item()),
            "diffusion_weighted_auxiliary_loss": float(
                (with_loss - without_loss).item()
            ),
            "with_diffusion_forward_backward_seconds": with_seconds,
            "without_diffusion_forward_backward_seconds": without_seconds,
            "shared_embedding_gradient_max_abs_delta": deltas["embedding.weight"],
            "teacher_trunk_gradient_max_abs_delta": max(trunk.values()),
            "diffusion_gradient_absolute_sum": sum(
                float(mx.sum(mx.abs(value)).item())
                for name, value in a.items()
                if name.startswith("diffusion_student.")
            ),
            "inference_logits_max_abs_error": inference_error,
            "peak_memory_bytes": mx.get_peak_memory(),
            "quality_benefit_qualified": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
