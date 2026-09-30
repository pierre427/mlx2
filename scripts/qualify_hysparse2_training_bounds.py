"""All-layer paired loss/gradient probe for exact attention visibility bounds."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("attention_reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    report = {
        "schema": "mlx2.hysparse2-training-bounds.v1",
        "completed": False,
        "serving_route_qualified": False,
        "optimizer_update_performed": False,
        "one_repetition_no_thermal_control": True,
        "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
        "reference_sha256": file_hash(args.reference),
        "batch": 2,
        "sequence": 256,
    }
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx import nn
        from mlx.utils import tree_flatten

        from mlx2.experimental.hysparse2 import attention
        from mlx2.experimental.hysparse2 import model as model_module
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.train import _load_model_state, loss

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(20 << 30)
        mx.set_cache_limit(256 << 20)
        c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model = model_module.Model(c)
        _load_model_state(args.checkpoint, model)
        model.train()
        model.checkpoint_layers = True
        tokens = (mx.arange(512).reshape(2, 256) * 17 + 1) % c.vocab_size
        mx.eval(model.parameters(), tokens)
        report["parameters"] = c.capacity()["parameters"]
        report["attention_source_sha256"] = file_hash(Path(attention.__file__))
        report["model_source_sha256"] = file_hash(Path(model_module.__file__))
        runs = []
        try:
            for label, function in (
                ("reference", reference.attention),
                ("bounded", attention.attention),
            ):
                model_module.attention = function
                compute = nn.value_and_grad(model, loss)
                warm_value, warm_gradients = compute(model, tokens)
                mx.eval(warm_value, warm_gradients)
                del warm_value, warm_gradients
                start = time.perf_counter()
                value, gradients = compute(model, tokens)
                mx.eval(value, gradients)
                elapsed = (time.perf_counter() - start) * 1000
                runs.append((float(value.item()), dict(tree_flatten(gradients))))
                report[label] = {"loss": runs[-1][0], "forward_backward_ms": elapsed}
            left, right = runs[0][1], runs[1][1]
            assert left.keys() == right.keys()
            errors = {
                key: float(mx.max(mx.abs(left[key] - right[key])).item())
                for key in left
            }
            assert all(bool(mx.all(mx.isfinite(g)).item()) for g in right.values())
            report["gradient_tensors_compared"] = len(errors)
            report["gradient_max_abs_error"] = max(errors.values())
            report["worst_gradient_tensor"] = max(errors, key=errors.get)
            report["loss_abs_error"] = abs(runs[0][0] - runs[1][0])
            probes = [
                f"{stack}.{i}.attention.q.weight"
                for stack in ("self_decoder", "cross_decoder")
                for i in range(len(getattr(model, stack)))
            ] + [
                "semantic_ple.value.weight",
                "mtp_head.projection.weight",
                "diffusion_student.layers.0.mlp.up.weight",
                "diffusion_student.layers.1.mlp.up.weight",
            ]
            report["active_gradient_probes"] = {
                key: float(mx.sum(mx.abs(right[key])).item()) for key in probes
            }
            assert all(value > 0 for value in report["active_gradient_probes"].values())
            assert report["loss_abs_error"] < 1e-4
            assert report["gradient_max_abs_error"] < 1e-4
            report["peak_memory_bytes"] = mx.get_peak_memory()
            report["completed"] = True
        finally:
            model_module.attention = attention.attention
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
