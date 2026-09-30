"""Full-checkpoint M3 greedy MTP verification and controlled re-entry probe."""

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
    c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
    report = {
        "schema": "mlx2.hysparse2-mtp-reference-exercise.v1",
        "completed": False,
        "serving_route_qualified": False,
        "accelerated_verification_qualified": False,
        "parameters": c.capacity()["parameters"],
        "dtype": "bfloat16",
        "lanes": 1,
        "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
    }
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        import numpy as np

        from mlx2.experimental.hysparse2.batching import ResearchBatcher
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.speculation import GreedyMTPReference
        from mlx2.experimental.hysparse2.train import _load_model_state
        from mlx2.runtime.adaptive_policy import CohortAdaptiveMTPDepth

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(24 << 30)
        mx.set_cache_limit(256 << 20)
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        prompt = [int(t) for t in np.load(args.tokens, allow_pickle=False)[:144]]
        logits, reference = model.prefill(mx.array([prompt]))
        expected = []
        for _ in range(8):
            token = int(mx.argmax(logits[0, -1]).item())
            expected.append(token)
            logits = model.decode(mx.array([[token]]), reference)
        natural, cache, receipt = GreedyMTPReference(model).generate(
            prompt, max_tokens=8
        )
        assert natural == expected and cache.length == len(prompt) + 8
        final = model._cross(cache.boundary, cache, cache.length - 1)
        report["natural_final_logit_error"] = float(
            mx.max(mx.abs(final - logits)).item()
        )
        assert report["natural_final_logit_error"] == 0
        report["natural"] = receipt
        report["natural_tokens"] = natural
        report["natural_proposed"] = sum(r["depth"] for r in receipt["rounds"])
        report["natural_accepted"] = sum(r["accepted"] for r in receipt["rounds"])
        attempts = [0]

        def injected(m, first, state):
            attempts[0] += 1
            clone = ResearchBatcher(m)._merge([state])
            next_logits = m.decode(mx.array([[first]]), clone)
            correct = int(mx.argmax(next_logits[0, -1]).item())
            return (correct + 1) % m.config.vocab_size if attempts[0] == 1 else correct

        control = CohortAdaptiveMTPDepth(
            max_depth=1,
            adaptive_single_lane=True,
            ewma_alpha=1,
            loss_rounds=1,
            gain_rounds=1,
            park_rounds=2,
            probe_interval=128,
            min_samples_per_depth=128,
        )
        forced, forced_cache, controlled = GreedyMTPReference(
            model, policy=control, proposal_fn=injected
        ).generate(prompt, max_tokens=8)
        assert forced == expected and forced_cache.length == len(prompt) + 8
        assert [r["depth"] for r in controlled["rounds"][:4]] == [1, 0, 0, 1]
        assert controlled["rounds"][3]["accepted"]
        report["controlled_test"] = controlled
        report["controlled_outcomes_are_injected"] = True
        report["controlled_parks"] = controlled["controller_counters"]["parks"]
        report["controlled_reentries"] = controlled["controller_counters"]["reentries"]
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["source_hashes"] = {
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path("src/mlx2/experimental/hysparse2/speculation.py"),
                Path("src/mlx2/experimental/hysparse2/model.py"),
                Path("src/mlx2/runtime/adaptive_policy.py"),
            ]
        }
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("natural", "controlled_test")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
