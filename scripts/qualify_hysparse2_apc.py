"""Bounded full-checkpoint APCv2 endpoint exercise on Apple GPU."""

import argparse
import json
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.checkpoint / "state.json").read_text())
    config = Config(**metadata["config"])
    revision = file_hash(args.checkpoint / "model.safetensors")
    report = {
        "schema": "mlx2.hysparse2-apcv2-endpoint.v1",
        "completed": False,
        "serving_route_qualified": False,
        "checkpoint_revision": revision,
        "parameters": config.capacity()["parameters"],
        "context": 144,
        "dtype": "bfloat16",
    }
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        import numpy as np

        from mlx2.experimental.hysparse2.apc import EndpointAPC
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state
        from mlx2.runtime.apc_v2 import APCv2

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(24 << 30)
        mx.set_cache_limit(256 << 20)
        model = Model(config)
        _load_model_state(args.checkpoint, model)
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        values = np.load(args.tokens, allow_pickle=False)
        tokens = [int(t) for t in values[:144]]
        if len(tokens) != 144:
            raise ValueError("need 144 prompt tokens")
        now = [0.0]
        engine = APCv2(
            max_size=4,
            layout_name="hysparse2-endpoint-v1",
            idle_disk_seconds=1,
            idle_disk_dir=str(args.output / "spill"),
            now_fn=lambda: now[0],
        )
        leases = []
        try:
            bridge = EndpointAPC(
                model,
                engine,
                checkpoint_revision=revision,
                tokenizer_fingerprint=metadata["run"].get(
                    "tokenizer_sha256", "checkpoint-tokenizer:" + file_hash(args.tokens)
                ),
            )
            _, cache = model.prefill(mx.array([tokens]))
            published = bridge.publish(tokens, cache)
            report["capability"] = {
                "topology": published.topology,
                "exact_prefix": published.exact_prefix,
                "arbitrary_branch": published.arbitrary_branch,
                "stored": published.stored,
            }
            continuation = int(values[144])
            reference = model.decode(mx.array([[continuation]]), cache)
            mx.eval(reference)
            a, hit = bridge.restore(tokens + [continuation])
            leases.append(hit.cache)
            actual = model.decode(mx.array([[continuation]]), a)
            report["resident_restore_error"] = float(
                mx.max(mx.abs(actual - reference)).item()
            )
            assert report["resident_restore_error"] == 0
            for lease in leases:
                if hasattr(lease, "close"):
                    lease.close()
            leases.clear()
            now[0] = 2.0
            report["spilled_entries"] = engine.spill_idle_entries(now=2.0)
            assert report["spilled_entries"] == 1
            b, disk_hit = bridge.restore(tokens + [continuation])
            leases.append(disk_hit.cache)
            report["disk_restore_error"] = float(
                mx.max(
                    mx.abs(model.decode(mx.array([[continuation]]), b) - reference)
                ).item()
            )
            assert report["disk_restore_error"] == 0
            branch, miss = bridge.restore(
                tokens[:-2] + [(tokens[-2] + 1) % config.vocab_size]
            )
            assert branch is None and not miss.hit
            report["interior_branch_rejected"] = True
            changed = EndpointAPC(
                model,
                engine,
                checkpoint_revision=revision + ":changed",
                tokenizer_fingerprint=bridge.tokenizer,
            )
            assert changed.restore(tokens + [continuation])[0] is None
            report["checkpoint_revision_miss"] = True
            report["cached_tokens"] = disk_hit.cached_tokens
            report["apcv2_stats"] = engine.apc_stats
            report["peak_memory_bytes"] = mx.get_peak_memory()
            report["completed"] = True
        finally:
            for lease in leases:
                if hasattr(lease, "close"):
                    lease.close()
            engine.close()
        report["source_hashes"] = {
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path("src/mlx2/experimental/hysparse2/apc.py"),
                Path("src/mlx2/experimental/hysparse2/model.py"),
            ]
        }
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "apcv2_stats"}), flush=True
    )


if __name__ == "__main__":
    main()
