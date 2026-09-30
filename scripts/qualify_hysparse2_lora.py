"""M3 bounded LoRA mechanics, not tool-use quality or serving qualification."""

import argparse
import json
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "mlx2.hysparse2-lora-mechanics.v1",
        "completed": False,
        "trained": False,
        "selected": False,
        "tool_use_quality_qualified": False,
        "serving_route_qualified": False,
    }
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from mlx2.experimental.hysparse2.lora import LoRAEpisode
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import file_hash, loss

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(20 << 30)
        mx.set_cache_limit(256 << 20)
        mx.random.seed(42)
        c = Config(**json.loads(args.config.read_text()))
        model = Model(c)
        model.checkpoint_layers = True
        model.eval()
        tokens = (mx.arange(2 * 144).reshape(2, 144) * 17 + 1) % c.vocab_size
        baseline = model(tokens)[0]
        mx.eval(baseline)
        keys = [
            "self_decoder.0.attention.q",
            "cross_decoder.0.attention.q",
            "semantic_ple.value",
            "diffusion_student.condition",
            "mtp_head.projection",
        ]
        base = {k: dict(model.named_modules())[k].weight for k in keys}
        revision = "random-seed42-config:" + file_hash(args.config)
        _, cache = model.prefill(tokens[:, :140])
        episode = LoRAEpisode(model, keys, base_revision=revision, max_steps=3)
        try:
            losses = [episode.step(tokens, loss) for _ in range(3)]
            report["losses"] = losses
            report["adapter_b_magnitudes"] = {
                k: float(mx.sum(mx.abs(episode.weights()[k + ".lora_b"])).item())
                for k in keys
            }
            assert min(report["adapter_b_magnitudes"].values()) > 0
            report["adapter_parameters"] = sum(
                v.size for v in episode.weights().values()
            )
            report["trainable_keys"] = list(
                dict(tree_flatten(model.trainable_parameters()))
            )
            report["base_weight_errors"] = {
                k: float(mx.max(mx.abs(episode.originals[k].weight - base[k])).item())
                for k in keys
            }
            assert max(report["base_weight_errors"].values()) == 0
            model.eval()
            adapted = model(tokens)[0]
            mx.eval(adapted)
            report["adapted_logit_delta"] = float(
                mx.max(mx.abs(adapted - baseline)).item()
            )
            assert report["adapted_logit_delta"] > 0
            try:
                model.decode(tokens[:, 140:141], cache)
            except ValueError:
                report["old_cache_rejected"] = True
            else:
                raise AssertionError("old cache accepted")
            heldout = (tokens + 31) % c.vocab_size
            report["evaluation"] = episode.evaluate(
                heldout, (tokens + 73) % c.vocab_size, loss
            )
            # Synthetic mechanics are not grounds to select a tool-use adapter.
            episode.promoted = False
            episode.export(args.output / "candidate")
        finally:
            episode.close()
        restored = model(tokens)[0]
        report["rollback_error"] = float(mx.max(mx.abs(restored - baseline)).item())
        assert report["rollback_error"] == 0
        candidate = LoRAEpisode.load_candidate(
            model, args.output / "candidate", base_revision=revision
        )
        try:
            model.eval()
            report["reload_error"] = float(
                mx.max(mx.abs(model(tokens)[0] - adapted)).item()
            )
            assert report["reload_error"] == 0
        finally:
            candidate.close()
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["trained"] = True
        report["completed"] = True
        report["source_hashes"] = {
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path("src/mlx2/experimental/hysparse2/lora.py"),
                Path("src/mlx2/experimental/hysparse2/model.py"),
            ]
        }
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
