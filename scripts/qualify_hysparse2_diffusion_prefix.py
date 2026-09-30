"""M3 prefix diffusion training/proposal mechanics; not generation quality."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original_config = Config(
        **json.loads((args.checkpoint / "state.json").read_text())["config"]
    )
    c = replace(
        original_config,
        diffusion_conditioning="prefix",
        diffusion_trunk_gradient_scale=1.0,
    )
    report = {
        "schema": "mlx2.hysparse2-prefix-diffusion.v1",
        "completed": False,
        "serving_route_qualified": False,
        "proposal_quality_qualified": False,
        "target_verified": False,
        "config": c.as_dict(),
        "source_checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
        "new_training_lineage": True,
        "optimizer_restored": False,
    }
    with gpu_guard(wait_seconds=0):
        import time

        import mlx.core as mx
        from mlx import nn, optimizers
        from mlx.utils import tree_flatten

        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import (
            _load_model_state,
            load_checkpoint,
            loss,
            save_checkpoint,
        )

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(20 << 30)
        mx.set_cache_limit(256 << 20)
        original = Model(original_config)
        _load_model_state(args.checkpoint, original)
        model = Model(c)
        model.load_weights(tree_flatten(original.parameters()), strict=True)
        del original
        model.checkpoint_layers = True
        tokens = (mx.arange(2 * 144).reshape(2, 144) * 17 + 1) % c.vocab_size
        value, gradients = nn.value_and_grad(model, loss)(model, tokens)
        mx.eval(value, gradients)
        flat = dict(tree_flatten(gradients))
        watched = [
            "self_decoder.0.attention.q.weight",
            "semantic_ple.value.weight",
            "diffusion_student.condition.weight",
            "diffusion_student.layers.0.mlp.up.weight",
            "diffusion_student.layers.1.mlp.up.weight",
        ]
        before = dict(tree_flatten(model.parameters()))
        mx.eval([before[k] for k in watched])
        optimizer = optimizers.Adam(1e-4)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        after = dict(tree_flatten(model.parameters()))
        report["loss"] = float(value.item())
        report["updates"] = {
            k: {
                "gradient_abs_sum": float(mx.sum(mx.abs(flat[k])).item()),
                "max_parameter_delta": float(
                    mx.max(mx.abs(after[k] - before[k])).item()
                ),
            }
            for k in watched
        }
        assert all(
            v["gradient_abs_sum"] > 0 and v["max_parameter_delta"] > 0
            for v in report["updates"].values()
        )
        del flat, gradients, before, after
        path = save_checkpoint(
            args.output / "checkpoints", model, optimizer, 1, {"prefix_diffusion": True}
        )
        model.eval()
        _, cache = model.prefill(tokens[:, :72])
        before_state = (cache.length, cache.self_layer_calls, cache.cross_layer_calls)
        start = time.perf_counter()
        proposals, receipt = model.diffusion_propose(cache, count=8, steps=3)
        mx.eval(proposals)
        report["proposal_seconds"] = time.perf_counter() - start
        report["proposal_receipt"] = receipt
        report["proposals"] = proposals.tolist()
        assert before_state == (
            cache.length,
            cache.self_layer_calls,
            cache.cross_layer_calls,
        )
        report["target_cache_unchanged"] = True
        report["fixture_future_token_match_rate"] = float(
            mx.mean((proposals == tokens[:, 72:80]).astype(mx.float32)).item()
        )
        teacher = mx.random.normal((2, 144, c.hidden_size))
        a, _ = model.diffusion_student(tokens, model.embedding, teacher)
        altered = mx.concatenate([teacher[:, :72], teacher[:, 72:] + 100], axis=1)
        b, _ = model.diffusion_student(tokens, model.embedding, altered)
        report["future_teacher_perturbation_error"] = float(
            mx.max(mx.abs(a - b)).item()
        )
        assert report["future_teacher_perturbation_error"] == 0
        restored = Model(c)
        load_checkpoint(
            path, restored, optimizers.Adam(1e-4), {"prefix_diffusion": True}
        )
        restored.eval()
        _, restored_cache = restored.prefill(tokens[:, :72])
        recovered, _ = restored.diffusion_propose(restored_cache, count=8, steps=3)
        report["checkpoint_proposals_equal"] = bool(
            mx.all(proposals == recovered).item()
        )
        assert report["checkpoint_proposals_equal"]
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["source_hashes"] = {
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path("src/mlx2/experimental/hysparse2/model.py"),
                Path("src/mlx2/experimental/hysparse2/config.py"),
                Path("src/mlx2/experimental/hysparse2/train.py"),
            ]
        }
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
