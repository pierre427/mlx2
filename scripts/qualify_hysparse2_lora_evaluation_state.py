"""M3 baseline/candidate evaluation state isolation, including evaluator failure."""

import argparse
import json
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "mlx2.hysparse2-lora-evaluation-state.v1",
        "completed": False,
        "selected": False,
        "serving_route_qualified": False,
        "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
    }
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx

        from mlx2.experimental.hysparse2 import lora
        from mlx2.experimental.hysparse2.batching import ResearchBatcher
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state, loss

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(20 << 30)
        mx.set_cache_limit(256 << 20)
        c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.eval()
        model.checkpoint_layers = True
        tokens = (mx.arange(144).reshape(1, 144) * 17 + 1) % c.vocab_size
        baseline, _ = model.prefill(tokens[:, :140])
        mx.eval(baseline)
        episode = lora.LoRAEpisode(
            model,
            ["self_decoder.0.attention.q", "semantic_ple.value"],
            base_revision=report["checkpoint_sha256"],
            max_steps=1,
        )
        try:
            report["step_loss"] = episode.step(tokens, loss)
            revision = model.adapter_revision
            cases = []
            for fail in (False, True):
                captured = []

                def evaluate(m, t, fail=fail, captured=captured):
                    value, cache = m.prefill(t[:, :140])
                    mx.eval(value, cache.arrays())
                    is_base = not hasattr(m.self_decoder[0].attention.q, "lora_a")
                    captured.append((is_base, m.adapter_revision, cache))
                    assert (
                        cache.apcv2_identity.get("adapter_revision")
                        == m.adapter_revision
                    )
                    if is_base:
                        assert float(mx.max(mx.abs(value - baseline)).item()) == 0
                        if fail:
                            raise RuntimeError("intentional evaluator failure")
                    return mx.array(1.0)

                try:
                    gate = episode.evaluate(tokens, tokens, evaluate)
                    assert not fail and not gate["promoted"]
                except RuntimeError as error:
                    assert fail and str(error) == "intentional evaluator failure"
                assert model.adapter_revision == revision
                assert hasattr(model.self_decoder[0].attention.q, "lora_a")
                for is_base, observed_revision, cache in captured:
                    assert observed_revision == (None if is_base else revision)
                    for decode in (
                        lambda cache: model.decode(tokens[:, 140:141], cache),
                        lambda cache: ResearchBatcher(model).decode(
                            [[int(tokens[0, 140].item())]], [cache]
                        ),
                    ):
                        try:
                            decode(cache)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError("evaluation cache escaped")
                    assert cache.length == 140
                _, valid = model.prefill(tokens[:, :140])
                model.decode(tokens[:, 140:141], valid)
                assert valid.length == 141
                cases.append(
                    {"evaluator_raised": fail, "stale_caches_rejected": len(captured)}
                )
            report["cases"] = cases
            report["source_sha256"] = file_hash(Path(lora.__file__))
            report["peak_memory_bytes"] = mx.get_peak_memory()
        finally:
            episode.close()
        assert model.adapter_revision is None
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
