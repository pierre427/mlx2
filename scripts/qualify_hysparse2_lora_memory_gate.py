"""Capsule changes must not qualify LoRA improvements; synthetic contract fixture."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_memory import SEMANTIC_SCHEMA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-smoke", action="store_true")
    p.add_argument("--step-binding-check", action="store_true", help="Exercise transactional objective/optimizer memory changes")
    args = p.parse_args()
    if not args.cpu_smoke and args.checkpoint is None:
        p.error("GPU exercise requires the scaled checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "mlx2.hysparse2-lora-memory-gate-exercise.v1", "completed": False,
              "serving_route_qualified": False, "learned_behavior_qualified": False,
              "device": "cpu" if args.cpu_smoke else "gpu", "cases": []}
    with nullcontext() if args.cpu_smoke else gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx2.experimental.hysparse2.capsule_memory import CapsuleMemory
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.lora import LoRAEpisode
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state, loss

        mx.set_default_device(mx.cpu if args.cpu_smoke else mx.gpu)
        if not args.cpu_smoke:
            mx.set_memory_limit(8 << 30)
            mx.set_cache_limit(128 << 20)
        mx.random.seed(42)
        c = (replace(Config.smoke(), diffusion_conditioning="prefix") if args.cpu_smoke
             else Config(**json.loads((args.checkpoint / "state.json").read_text())["config"]))
        model = Model(c)
        if not args.cpu_smoke:
            _load_model_state(args.checkpoint, model)
        model.eval()
        model.checkpoint_layers = True
        revision = "cpu-smoke-seed42" if args.cpu_smoke else file_hash(args.checkpoint / "model.safetensors")
        store = CapsuleStore(args.output / "capsules")
        memories = []
        bindings = {"model_binding": revision, "tokenizer_binding": "synthetic-character-v1",
                    "runtime_binding": "lora-memory-gate-v1"}
        for text in ("validated arguments", "revised validated arguments"):
            graph = {"schema": SEMANTIC_SCHEMA, "concepts": {
                "a": {"label": "tool call"}, "b": {"label": text}}, "proposals": [],
                "edges": [{"subject": "a", "relation": "has_property", "object": "b",
                           "authority": "committed",
                           "evidence_digest": hashlib.sha256(text.encode()).hexdigest()}]}
            capsule = store.put(kind="semantic_base", data=graph, **bindings,
                               provenance={"source": "original synthetic gate fixture"})
            memories.append(CapsuleMemory.from_store(
                store, capsule.digest, **bindings, vocab_size=c.vocab_size,
                encode=lambda s: [ord(x) % c.vocab_size for x in s]))
        tokens = (mx.arange(144)[None] * 17 + 1) % c.vocab_size
        keys = ["semantic_ple.value", "diffusion_student.condition"]
        if args.step_binding_check:
            for phase in ("objective", "optimizer"):
                model.attach_semantic_capsules(memories[0])
                model.eval()
                baseline = model(tokens)[0]
                mx.eval(baseline)
                episode = LoRAEpisode(model, keys, base_revision=revision, max_steps=2)
                update = episode.optimizer.update
                try:
                    value = episode.step(tokens, loss)
                    model.eval()
                    weights = episode.weights()
                    state = dict(tree_flatten(episode.optimizer.state))
                    mx.eval(weights, state)
                    adapter_revision = episode.live_revision

                    def objective(m, ids):
                        result = loss(m, ids)
                        if phase == "objective":
                            m.attach_semantic_capsules(memories[1])
                        return result

                    def changed_update(m, gradients):
                        update(m, gradients)
                        if phase == "optimizer":
                            m.attach_semantic_capsules(memories[1])

                    episode.optimizer.update = changed_update
                    try:
                        episode.step(tokens, objective)
                    except ValueError as exc:
                        assert "capsule binding" in str(exc)
                    else:
                        raise AssertionError("mixed memory step accepted")
                    assert episode.steps == 1 and episode.live_revision == adapter_revision
                    assert model.adapter_revision == adapter_revision and not model.training
                    assert model.capsule_binding == memories[1].binding()
                    assert not episode._allow_parameter_update
                    weight_error = max(float(mx.max(mx.abs(v - weights[k])).item())
                                       for k, v in episode.weights().items())
                    restored = dict(tree_flatten(episode.optimizer.state))
                    assert restored.keys() == state.keys()
                    state_error = max(float(mx.max(mx.abs(restored[k] - v)).item())
                                      for k, v in state.items())
                    assert weight_error == state_error == 0
                    episode.optimizer.update = update
                    model.attach_semantic_capsules(memories[0])
                    retry_value = episode.step(tokens, loss)
                    assert episode.steps == 2
                finally:
                    episode.optimizer.update = update
                    episode.close()
                model.eval()
                error = float(mx.max(mx.abs(model(tokens)[0] - baseline)).item())
                assert error == 0
                report["cases"].append({"change_during": phase, "rejected": True,
                    "training_loss": value, "retry_training_loss": retry_value,
                    "adapter_restore_error": weight_error, "optimizer_restore_error": state_error,
                    "caller_memory_change_preserved": True, "retry_steps": 2,
                    "rollback_logit_error": error})
        for change_at in (() if args.step_binding_check else (1, 3)):
            model.attach_semantic_capsules(memories[0])
            baseline = model(tokens)[0]
            mx.eval(baseline)
            episode = LoRAEpisode(model, keys, base_revision=revision, max_steps=1)
            try:
                value = episode.step(tokens, loss)
                magnitudes = {key: float(mx.sum(mx.abs(episode.weights()[key + ".lora_b"])).item()) for key in keys}
                assert min(magnitudes.values()) > 0
                calls = 0

                def changed(m, _):
                    nonlocal calls
                    calls += 1
                    score = 0.5 if m.capsule_binding == memories[0].binding() else 1.0
                    if calls == change_at:
                        m.attach_semantic_capsules(memories[1])
                    return mx.array(score)

                try:
                    episode.evaluate(None, None, changed)
                except ValueError as exc:
                    assert "capsule binding" in str(exc)
                else:
                    raise AssertionError("mixed memory gate accepted")
                assert not episode.promoted and model.adapter_revision == episode.live_revision
                assert all(hasattr(dict(model.named_modules())[key], "lora_b") for key in keys)
                model.attach_semantic_capsules(memories[0])
                retry = episode.evaluate(None, None, lambda *_: mx.array(1.0))
                assert not retry["promoted"] and retry["evaluation_capsule_binding"] == memories[0].binding()
            finally:
                episode.promoted = False
                episode.close()
            error = float(mx.max(mx.abs(model(tokens)[0] - baseline)).item())
            assert error == 0
            report["cases"].append({"change_at_score": change_at, "rejected": True,
                                    "training_loss": value, "adapter_b_magnitudes": magnitudes,
                                    "retry_unpromoted": True, "rollback_logit_error": error})
        report["parameters"] = c.capacity()["parameters"]
        report["adapter_steps_per_episode"] = 2 if args.step_binding_check else 1
        if not args.cpu_smoke:
            report["peak_gib"] = mx.get_peak_memory() / (1 << 30)
    root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {str(path.relative_to(root)): file_hash(path)
        for path in (Path(__file__).resolve(), root / "src/mlx2/experimental/hysparse2/lora.py")}
    report["base_revision"] = revision
    report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
