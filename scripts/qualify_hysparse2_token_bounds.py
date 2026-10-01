"""Bounded model token preflight and cached-state preservation checks."""

import argparse
import json
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "mlx2.hysparse2-token-bounds.v1", "completed": False,
              "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
              "serving_route_qualified": False, "training_performed": False, "rejections": []}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx2.experimental.hysparse2 import model as module
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.train import _load_model_state

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(10 << 30)
        mx.set_cache_limit(256 << 20)
        config = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model = module.Model(config)
        _load_model_state(args.checkpoint, model)
        model.eval()
        prompt = (mx.arange(144)[None] * 17 + 1) % config.vocab_size
        _, cache = model.prefill(prompt)
        _, reference = model.prefill(prompt)
        expected = model.decode(mx.array([[3]]), reference)
        mx.eval(expected)
        arrays = cache.arrays()
        before = (cache.length, cache.self_layer_calls, cache.cross_layer_calls, cache.boundary, cache.ple_history)
        suffix = (mx.arange(config.prefill_chunk)[None] * 17 + 1) % config.vocab_size
        for value in (-1, config.vocab_size):
            for operation in ("prefill", "decode", "teacher", "mtp_teacher"):
                try:
                    if operation == "prefill":
                        model.prefill(mx.concatenate((suffix, mx.array([[value]])), axis=1), cache)
                    elif operation == "decode":
                        model.decode(mx.array([[value]]), cache)
                    elif operation == "teacher":
                        model(mx.array([[value]]))
                    else:
                        model(mx.array([[1]]), next_tokens=mx.array([[value]]))
                    raise AssertionError("invalid token accepted")
                except ValueError as exc:
                    assert "vocabulary" in str(exc)
                after = (cache.length, cache.self_layer_calls, cache.cross_layer_calls, cache.boundary, cache.ple_history)
                assert before[:3] == after[:3] and before[3] is after[3] and before[4] is after[4]
                assert all(a is b for a, b in zip(arrays, cache.arrays(), strict=True))
                report["rejections"].append({"operation": operation, "token": value, "cache_unchanged": True})
        actual = model.decode(mx.array([[3]]), cache)
        report["valid_decode_error"] = float(mx.max(mx.abs(actual - expected)).item())
        assert report["valid_decode_error"] == 0
        report["prefill_suffix_tokens"] = config.prefill_chunk + 1
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["source_hashes"] = {str(path): file_hash(path) for path in (Path(__file__), Path(module.__file__))}
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
