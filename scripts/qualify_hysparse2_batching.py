"""M3 ragged request cohorts and serialized concurrent caller mechanics."""

import argparse
import json
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
    c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
    report = {
        "schema": "mlx2.hysparse2-request-cohorts-exercise.v1",
        "completed": False,
        "serving_route_qualified": False,
        "mixed_memory_batching_qualified": False,
        "parameters": c.capacity()["parameters"],
        "prompt_lengths": [144, 65, 144, 65, 65],
    }
    with gpu_guard(wait_seconds=0):
        from concurrent.futures import ThreadPoolExecutor

        import mlx.core as mx

        from mlx2.experimental.hysparse2.batching import ResearchBatcher
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(20 << 30)
        mx.set_cache_limit(256 << 20)
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.eval()
        mx.eval(model.parameters())
        batcher = ResearchBatcher(model, max_lanes=2)
        prompts = [
            [(j * 17 + i * 31 + 1) % c.vocab_size for j in range(n)]
            for i, n in enumerate(report["prompt_lengths"])
        ]
        reference = [model.prefill(mx.array([row])) for row in prompts]
        logits, caches, receipt = batcher.prefill(prompts)
        errors = [
            float(mx.max(mx.abs(got - expected[0])).item())
            for got, expected in zip(logits, reference)
        ]
        assert max(errors) < 1e-3
        report["prefill_errors"] = errors
        report["prefill_receipt"] = receipt
        report["decode_errors"] = []
        report["decode_receipts"] = []
        for step in range(3):
            rows = [[100 + step + i] for i in range(len(prompts))]
            lengths = [cache.length for cache in caches]
            got, updated, receipt = batcher.decode(rows, caches)
            errors = [
                float(
                    mx.max(
                        mx.abs(
                            got[i] - model.decode(mx.array([rows[i]]), reference[i][1])
                        )
                    ).item()
                )
                for i in range(len(rows))
            ]
            assert max(errors) < 1e-3
            assert [cache.length for cache in caches] == lengths
            assert [cache.length for cache in updated] == [n + 1 for n in lengths]
            caches = updated
            report["decode_errors"].append(errors)
            report["decode_receipts"].append(receipt)
        report["request_state_unchanged_until_commit"] = True
        # Caller threads are serialized by one batcher; GPU kernels are not
        # claimed to run concurrently. Outputs are consumed on the owner thread.
        with ThreadPoolExecutor(max_workers=2) as pool:
            threaded = list(pool.map(lambda row: batcher.prefill([row]), prompts[:2]))
        thread_errors = []
        for row, result in zip(prompts[:2], threaded):
            expected, _ = model.prefill(mx.array([row]))
            thread_errors.append(float(mx.max(mx.abs(result[0][0] - expected)).item()))
        assert max(thread_errors) < 1e-3
        report["serialized_caller_errors"] = thread_errors
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["source_hashes"] = {
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path("src/mlx2/experimental/hysparse2/batching.py"),
                Path("src/mlx2/experimental/hysparse2/model.py"),
            ]
        }
        report["checkpoint_sha256"] = file_hash(args.checkpoint / "model.safetensors")
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
