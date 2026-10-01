"""Bounded checkpoint failure isolation; no training or serving qualification."""

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
    report = {"schema": "mlx2.hysparse2-checkpoint-failure.v1", "completed": False,
              "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
              "serving_route_qualified": False, "training_performed": False}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx import optimizers
        from mlx2.experimental.hysparse2 import train as training
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.model import Model

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(10 << 30)
        mx.set_cache_limit(256 << 20)
        config = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model = Model(config)
        training._load_model_state(args.checkpoint, model)
        # Changed PLE content ensures a successful save would change its digest.
        model.update({"semantic_ple": {"value": {"weight": model.semantic_ple.value.weight + 0.001}}})
        model.eval()
        optimizer = optimizers.Adam(1e-4)
        tokens = (mx.arange(72)[None] * 17 + 1) % config.vocab_size
        next_token = mx.array([[3]])
        _, reference_cache = model.prefill(tokens)
        expected = model.decode(next_token, reference_cache)
        mx.eval(expected)
        report["failures"] = []
        for failure in ("optimizer", "metadata", "rename"):
            root = args.output / failure
            unrelated = root / ".writing-unrelated"
            unrelated.mkdir(parents=True)
            (unrelated / "keep").write_text("preserve")
            _, cache = model.prefill(tokens)
            owner, digest = model._cache_owner, model.ple_sidecar_digest
            def fail(*a, **kw):
                raise OSError("injected checkpoint failure")
            if failure == "optimizer":
                original = mx.save_safetensors
                def injected(path, *a, **kw):
                    return fail() if Path(path).name == "optimizer.safetensors" else original(path, *a, **kw)
                context = patch.object(mx, "save_safetensors", injected)
            elif failure == "metadata":
                original = Path.write_text
                def injected(path, *a, **kw):
                    return fail() if path.name == "state.json" else original(path, *a, **kw)
                context = patch.object(Path, "write_text", injected)
            else:
                context = patch.object(Path, "rename", fail)
            with context:
                try:
                    training.save_checkpoint(root, model, optimizer, 0, {}, mode="full")
                    raise AssertionError("injected failure accepted")
                except OSError as exc:
                    assert "injected checkpoint failure" in str(exc)
            assert model._cache_owner is owner and model.ple_sidecar_digest == digest
            assert sorted(path.name for path in root.iterdir()) == [".writing-unrelated"]
            assert (unrelated / "keep").read_text() == "preserve"
            actual = model.decode(next_token, cache)
            error = float(mx.max(mx.abs(actual - expected)).item())
            assert error == 0
            report["failures"].append({"stage": failure, "decode_logits_error": error,
                                       "identity_preserved": True, "own_staging_removed": True,
                                       "unrelated_staging_preserved": True})
        old_digest = model.ple_sidecar_digest
        saved = training.save_checkpoint(args.output / "retry", model, optimizer, 0, {}, mode="model")
        assert model.ple_sidecar_digest != old_digest
        fresh = Model(config)
        training._load_model_state(saved, fresh)
        fresh.eval()
        actual, _ = fresh.prefill(tokens)
        expected, _ = model.prefill(tokens)
        report["retry_reload_logits_error"] = float(mx.max(mx.abs(actual - expected)).item())
        assert report["retry_reload_logits_error"] == 0
        report["saved_checkpoint_sha256"] = file_hash(saved / "model.safetensors")
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["source_hashes"] = {str(path): file_hash(path) for path in (Path(__file__), Path(training.__file__))}
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
