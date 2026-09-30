"""Bounded GPU mechanics qualification; does not qualify a serving route."""

import argparse
import json
import os
import time
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard


def main():
    os.environ["MLX_ENABLE_TF32"] = "0"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequence", type=int, default=20)
    args = p.parse_args()
    if args.output.exists() or not 4 <= args.sequence <= 512:
        p.error("use a fresh output directory and a sequence from 4 to 512")
    args.output.mkdir(parents=True)
    c = Config(**json.loads(args.config.read_text()))
    report = {
        "schema": "mlx2.hysparse2-component-exercise.v1",
        "config": c.as_dict(),
        "parameters": c.capacity()["parameters"],
        "serving_route_qualified": False,
        "checks": {},
        "completed": False,
    }

    def save():
        path = args.output / "receipt.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(path)

    save()
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx import nn, optimizers
        from mlx.utils import tree_flatten

        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import (
            load_checkpoint,
            loss,
            save_checkpoint,
        )

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(20 << 30)
        mx.set_cache_limit(256 << 20)
        mx.random.seed(42)
        model = Model(c)
        model.checkpoint_layers = True
        tokens = (
            mx.arange(2 * args.sequence).reshape(2, args.sequence) * 17 + 1
        ) % c.vocab_size
        watched = [
            "self_decoder.0.attention.q.weight",
            "self_decoder.12.attention.q.weight",
            "self_decoder.0.moe.router.weight",
            "self_decoder.0.moe.shared.0.up.weight",
            "self_decoder.0.attention_hc.mix.weight",
            "cross_decoder.0.attention.k.weight",
            "cross_decoder.1.attention.q.weight",
            "semantic_ple.embedding.weight",
            "mtp_head.projection.weight",
            "diffusion_student.layers.0.mlp.up.weight",
            "diffusion_student.layers.1.mlp.up.weight",
        ]
        for stack in ("self_decoder", "cross_decoder"):
            for i, _ in enumerate(getattr(model, stack)):
                watched.extend(
                    f"{stack}.{i}.{suffix}"
                    for suffix in (
                        "attention.q.weight",
                        "moe.router.weight",
                        "moe.up",
                        "moe.shared.0.up.weight",
                        "attention_hc.mix.weight",
                        "ffn_hc.mix.weight",
                    )
                )
        watched = list(dict.fromkeys(watched))
        # Inspect the actual module tree so renamed optional probes cannot create
        # a false successful run.
        parameters = dict(tree_flatten(model.parameters()))
        missing = set(watched) - set(parameters)
        if missing:
            raise AssertionError(f"missing required probes: {sorted(missing)}")
        before = {name: parameters[name] for name in watched}
        mx.eval(before)
        start = time.perf_counter()
        value, gradients = nn.value_and_grad(model, loss)(model, tokens)
        mx.eval(value, gradients)
        flat_gradients = dict(tree_flatten(gradients))
        gradient_checks = {}
        for name in watched:
            grad = flat_gradients[name]
            finite = bool(mx.all(mx.isfinite(grad)).item())
            magnitude = float(mx.sum(mx.abs(grad)).item())
            gradient_checks[name] = {"finite": finite, "absolute_sum": magnitude}
            if not finite or magnitude <= 0:
                raise AssertionError(f"inactive gradient: {name}")
        optimizer = optimizers.AdamW(learning_rate=1e-4)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        after = dict(tree_flatten(model.parameters()))
        for name in watched:
            delta = float(mx.max(mx.abs(after[name] - before[name])).item())
            gradient_checks[name]["max_parameter_delta"] = delta
            if delta <= 0:
                raise AssertionError(f"inactive parameter update: {name}")
        report["checks"]["optimizer"] = {
            "loss": float(value.item()),
            "seconds": time.perf_counter() - start,
            "modules": gradient_checks,
        }
        save()
        del gradients, flat_gradients, before, after, parameters
        checkpoint = save_checkpoint(args.output, model, optimizer, 1, {"seed": 42})
        model.eval()
        reference = model(tokens)[0]
        split = args.sequence - 2
        cached, cache = model.prefill(tokens[:, :split])
        errors = [
            float(mx.max(mx.abs(cached - reference[:, split - 1 : split])).item())
        ]
        for i in range(split, args.sequence):
            got = model.decode(tokens[:, i : i + 1], cache)
            errors.append(float(mx.max(mx.abs(got - reference[:, i : i + 1])).item()))
        if max(errors) > 1e-3:
            raise AssertionError(f"cached parity failed: {errors}")
        report["checks"]["batch2_cached_parity"] = {"max_abs_errors": errors}
        baseline = reference
        weight = model.semantic_ple.value.weight
        model.semantic_ple.value.weight = mx.zeros_like(weight)
        ablated = model(tokens)[0]
        ple_effect = float(mx.max(mx.abs(baseline - ablated)).item())
        model.semantic_ple.value.weight = weight
        if ple_effect <= 0:
            raise AssertionError("PLE ablation did not change output")
        report["checks"]["ple_ablation"] = {"max_logit_delta": ple_effect}
        restored = Model(c)
        restored_optimizer = optimizers.AdamW(learning_rate=1e-4)
        load_checkpoint(checkpoint, restored, restored_optimizer, {"seed": 42})
        restored.eval()
        recovered = restored(tokens)[0]
        restored_error = float(mx.max(mx.abs(baseline - recovered)).item())
        if restored_error != 0:
            raise AssertionError(f"checkpoint differs: {restored_error}")
        report["checks"]["checkpoint"] = {
            "max_abs_error": restored_error,
            "ple_digest": restored.ple_sidecar_digest,
            "optimizer_step": int(restored_optimizer.step.item()),
        }
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["completed"] = True
        save()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
