"""Adam resume payload preflight against a trained all-layer checkpoint."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.checkpoint / "state.json").read_text())
    report = {"schema": "mlx2.hysparse2-optimizer-state.v1", "completed": False,
              "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
              "optimizer_sha256": file_hash(args.checkpoint / "optimizer.safetensors"),
              "serving_route_qualified": False, "training_performed": False, "rejections": []}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx import optimizers
        from mlx.utils import tree_flatten
        from mlx2.experimental.hysparse2 import train as training
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.model import Model

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(10 << 30)
        mx.set_cache_limit(256 << 20)
        model = Model(Config(**metadata["config"]))
        training._load_model_state(args.checkpoint, model)
        model.eval()
        original_load = mx.load
        payload = original_load(str(args.checkpoint / "optimizer.safetensors"))
        mx.eval(payload)
        moment = next(key for key in payload if key.endswith(".m"))
        optimizer = optimizers.Adam(1e-4)
        before = dict(tree_flatten(model.parameters()))
        owner, epoch, digest = model._cache_owner, model._parameter_epoch, model.ple_sidecar_digest
        state = optimizer.state
        for damage in ("missing_moment", "shape", "nonfinite", "step", "negative_variance"):
            changed = dict(payload)
            if damage == "missing_moment":
                del changed[moment]
            elif damage == "shape":
                changed[moment] = changed[moment].reshape(-1)[:1]
            elif damage == "nonfinite":
                changed[moment] = mx.full(changed[moment].shape, float("nan"))
            elif damage == "negative_variance":
                variance = moment[:-1] + "v"
                changed[variance] = mx.full(changed[variance].shape, -1.0)
            else:
                changed["step"] = changed["step"] + 1
            def injected(path, *a, **kw):
                return changed if Path(path).name == "optimizer.safetensors" else original_load(path, *a, **kw)
            with patch.object(mx, "load", injected):
                try:
                    training.load_checkpoint(args.checkpoint, model, optimizer, metadata["run"])
                    raise AssertionError("invalid optimizer payload accepted")
                except ValueError as exc:
                    assert "optimizer state" in str(exc)
            assert model._cache_owner is owner and model._parameter_epoch is epoch
            assert model.ple_sidecar_digest == digest and optimizer.state is state
            assert all(dict(tree_flatten(model.parameters()))[key] is value for key, value in before.items())
            report["rejections"].append({"damage": damage, "live_state_unchanged": True})
        report["restored_step"] = training.load_checkpoint(args.checkpoint, model, optimizer, metadata["run"])
        restored = dict(tree_flatten(optimizer.state))
        assert restored.keys() == payload.keys()
        report["optimizer_tensors_compared"] = len(payload)
        report["optimizer_max_error"] = max(float(mx.max(mx.abs(restored[key] - value)).item()) for key, value in payload.items())
        assert report["optimizer_max_error"] == 0 and report["restored_step"] == metadata["step"]
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["source_hashes"] = {str(path): file_hash(path) for path in (Path(__file__), Path(training.__file__))}
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
