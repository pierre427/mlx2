"""Parent-bound LoRA identity mechanics, not learned behavior qualification."""

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
    revision = file_hash(args.checkpoint / "model.safetensors")
    report = {"schema": "mlx2.hysparse2-lora-parent.v1", "completed": False,
              "checkpoint_sha256": revision, "serving_route_qualified": False,
              "learned_behavior_qualified": False, "parent_objective": "synthetic sum of adapter B; identity fixture only"}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten
        from mlx2.experimental.hysparse2 import lora as lora_module
        from mlx2.experimental.hysparse2 import model as model_module
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.train import _load_model_state
        from mlx2.experimental.hysparse2.apc import EndpointAPC
        from mlx2.runtime.apc_v2 import APCv2

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(10 << 30)
        mx.set_cache_limit(256 << 20)
        c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model, plain = model_module.Model(c), model_module.Model(c)
        _load_model_state(args.checkpoint, model)
        plain.load_weights(tree_flatten(model.parameters()), strict=True)
        model.eval()
        plain.eval()
        parent = lora_module.LoRAEpisode(model, ["self_decoder.0.attention.q"], base_revision=revision, rank=2, max_steps=1)
        child = unrelated = loaded = None
        engine = APCv2(max_size=4, layout_name="hysparse2-endpoint-v1")
        try:
            def diagnostic(m, _):
                layer = m.self_decoder[0].attention.q
                return mx.sum(layer.lora_b) if hasattr(layer, "lora_b") else mx.array(0.)

            parent.step(None, diagnostic)
            assert parent.evaluate(None, None, diagnostic)["promoted"]
            parent.close()
            report["parent_adapter_revision"] = model.adapter_revision
            mx.random.seed(73)
            child = lora_module.LoRAEpisode(model, ["semantic_ple.value"], base_revision=revision, rank=2, max_steps=1)
            mx.random.seed(73)
            unrelated = lora_module.LoRAEpisode(plain, ["semantic_ple.value"], base_revision=revision, rank=2, max_steps=1)
            assert model.adapter_revision != plain.adapter_revision
            report["child_revisions_distinct"] = True
            report["unmanaged_updates_rejected"] = []
            for key in ("embedding.weight", "semantic_ple.value.lora_b"):
                before = dict(tree_flatten(model.parameters()))[key]
                owner, adapter_revision = model._cache_owner, model.adapter_revision
                try:
                    model.update(tree_unflatten([(key, before + 0.1)]))
                    raise AssertionError("unmanaged parameter update accepted")
                except ValueError as exc:
                    assert "owns parameter updates" in str(exc)
                assert dict(tree_flatten(model.parameters()))[key] is before
                assert model._cache_owner is owner and model.adapter_revision == adapter_revision
                report["unmanaged_updates_rejected"].append(key)
            a = EndpointAPC(model, engine, checkpoint_revision=revision, tokenizer_fingerprint="synthetic")
            b = EndpointAPC(plain, engine, checkpoint_revision=revision, tokenizer_fingerprint="synthetic")
            assert a.key() != b.key()
            report["apc_keys_distinct"] = True
            child.export(args.output / "child")
            model.eval()
            tokens = (mx.arange(72)[None] * 17 + 1) % c.vocab_size
            expected, _ = model.prefill(tokens)
            child_revision = model.adapter_revision
            child.close()
            unrelated.close()
            owner = plain._cache_owner
            try:
                lora_module.LoRAEpisode.load_candidate(plain, args.output / "child", base_revision=revision)
                raise AssertionError("wrong parent accepted")
            except ValueError as exc:
                assert "parent adapter revision" in str(exc)
            assert plain._cache_owner is owner and plain.adapter_revision is None
            report["wrong_parent_rejected_without_mutation"] = True
            loaded = lora_module.LoRAEpisode.load_candidate(model, args.output / "child", base_revision=revision)
            assert model.adapter_revision == child_revision
            model.eval()
            actual, _ = model.prefill(tokens)
            report["correct_parent_reload_logits_error"] = float(mx.max(mx.abs(expected - actual)).item())
            assert report["correct_parent_reload_logits_error"] == 0
            base_weight = model.embedding.weight
            def broken(m, _):
                m.update({"embedding": {"weight": m.embedding.weight + 0.1}})
                return mx.array(0.)
            try:
                loaded.step(None, broken)
                raise AssertionError("base mutation inside step accepted")
            except ValueError as exc:
                assert "base parameters" in str(exc)
            report["base_weight_error_after_rejected_step"] = float(mx.max(mx.abs(model.embedding.weight - base_weight)).item())
            assert report["base_weight_error_after_rejected_step"] == 0 and not loaded._allow_parameter_update
            assert loaded.steps == 0
            report["base_update_rejected_inside_managed_step"] = True
            report["peak_memory_bytes"] = mx.get_peak_memory()
            report["source_hashes"] = {str(path): file_hash(path) for path in (Path(__file__), Path(lora_module.__file__), Path(model_module.__file__))}
            report["completed"] = True
        finally:
            for episode in (loaded, unrelated, child):
                if episode is not None:
                    episode.close()
            parent.rollback()
            engine.close()
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
