"""Bounded PLE mirror disagreement and load-rollback checks on M3."""

import argparse
import json
import os
import shutil
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < (2 << 30):
        raise OSError("PLE fixture needs two GiB disk headroom")
    report = {"schema": "mlx2.hysparse2-ple-binding.v1", "completed": False,
              "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
              "serving_route_qualified": False, "training_performed": False, "mismatches": []}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx2.experimental.hysparse2 import train as training
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.model import Model

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(10 << 30)
        mx.set_cache_limit(256 << 20)
        original = json.loads((args.checkpoint / "state.json").read_text())
        config = Config(**original["config"])
        model = Model(config)
        training._load_model_state(args.checkpoint, model)
        model.eval()
        tokens = (mx.arange(72)[None] * 17 + 1) % config.vocab_size
        following = mx.array([[3]])
        _, reference = model.prefill(tokens)
        expected = model.decode(following, reference)
        mx.eval(expected)
        key = "semantic_ple.value.weight"
        for mismatch in ("values", "coverage", "shape", "dtype"):
            root = args.output / mismatch
            root.mkdir()
            metadata = json.loads(json.dumps(original))
            sidecar = metadata["permanent_sidecar"]
            source_sidecar = args.checkpoint / sidecar["file"]
            if mismatch == "values":
                weights = mx.load(str(args.checkpoint / "model.safetensors"))
                weights[key] = weights[key] + 0.01
                mx.eval(weights)
                mx.save_safetensors(str(root / "model.safetensors"), weights)
                del weights
                shutil.copyfile(source_sidecar, root / sidecar["file"])
            else:
                # Read-only hard link; this fixture never writes this model file.
                os.link(args.checkpoint / "model.safetensors", root / "model.safetensors")
                weights = mx.load(str(source_sidecar))
                if mismatch == "coverage":
                    del weights[key]
                elif mismatch == "shape":
                    weights[key] = weights[key][:1]
                else:
                    weights[key] = weights[key].astype(mx.float16)
                mx.eval(weights)
                mx.save_safetensors(str(root / sidecar["file"]), weights)
                del weights
                sidecar["sha256"] = file_hash(root / sidecar["file"])
                sidecar["apcv2_identity"] = config.apcv2_identity(ple_sidecar_digest=sidecar["sha256"])
            (root / "state.json").write_text(json.dumps(metadata))
            before = dict(tree_flatten(model.parameters()))
            owner, epoch, digest = model._cache_owner, model._parameter_epoch, model.ple_sidecar_digest
            _, cache = model.prefill(tokens)
            try:
                training._load_model_state(root, model)
                raise AssertionError("PLE mismatch accepted")
            except ValueError as exc:
                assert "PLE sidecar tensors" in str(exc)
            restored = dict(tree_flatten(model.parameters()))
            assert all(restored[name] is value for name, value in before.items())
            assert model._cache_owner is owner and model._parameter_epoch is epoch
            assert model.ple_sidecar_digest == digest
            error = float(mx.max(mx.abs(model.decode(following, cache) - expected)).item())
            assert error == 0
            report["mismatches"].append({"type": mismatch, "rejected": True,
                                         "exact_state_preserved": True, "decode_logits_error": error})
        training._load_model_state(args.checkpoint, model)
        _, cache = model.prefill(tokens)
        report["valid_reload_decode_error"] = float(mx.max(mx.abs(model.decode(following, cache) - expected)).item())
        assert report["valid_reload_decode_error"] == 0
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["source_hashes"] = {str(path): file_hash(path) for path in (Path(__file__), Path(training.__file__))}
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
