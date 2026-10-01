"""M3 revision changes during diffusion proposal publication; synthetic memory."""

import argparse
import hashlib
import json
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_memory import SEMANTIC_SCHEMA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "mlx2.hysparse2-diffusion-revision.v1", "completed": False,
              "serving_route_qualified": False, "optimizer_updates": 0, "cases": []}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.capsule_memory import CapsuleMemory
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(8 << 30)
        mx.set_cache_limit(128 << 20)
        c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.eval()
        mx.eval(model.parameters())
        store = CapsuleStore(args.output / "capsules")
        bindings = {"model_binding": file_hash(args.checkpoint / "model.safetensors"),
                    "tokenizer_binding": "synthetic-character-v1",
                    "runtime_binding": "diffusion-revision-fixture-v1"}
        memories = []
        for text in ("validated arguments", "revised validated arguments"):
            graph = {"schema": SEMANTIC_SCHEMA,
                     "concepts": {"a": {"label": "tool call"}, "b": {"label": text}},
                     "edges": [{"subject": "a", "relation": "has_property", "object": "b",
                                "authority": "committed",
                                "evidence_digest": hashlib.sha256(text.encode()).hexdigest()}],
                     "proposals": []}
            capsule = store.put(kind="semantic_base", data=graph, **bindings,
                                provenance={"source": "original synthetic revision fixture"})
            memories.append(CapsuleMemory.from_store(
                store, capsule.digest, **bindings, vocab_size=c.vocab_size,
                encode=lambda s: [ord(x) % c.vocab_size for x in s]))
        tokens = (mx.arange(288).reshape(2, 144) * 17 + 1) % c.vocab_size

        def state(cache):
            return (cache.owner, cache.length, cache.self_layer_calls, cache.cross_layer_calls,
                    id(cache.boundary), id(cache.ple_history), tuple(id(x) for x in cache.arrays()))

        for revision in ("parameters", "capsule"):
            model.attach_semantic_capsules(memories[0])
            _, cache = model.prefill(tokens)
            before = state(cache)
            baseline, _ = model.diffusion_propose(cache, count=8, steps=3)
            original = model.diffusion_student.denoise

            def changed(*values):
                if revision == "parameters":
                    model.update({"embedding": {"weight": model.embedding.weight}})
                else:
                    model.attach_semantic_capsules(memories[1])
                return original(*values)

            model.diffusion_student.denoise = changed
            try:
                model.diffusion_propose(cache, count=8, steps=3)
            except ValueError as exc:
                assert "revision" in str(exc)
                rejection = str(exc)
            else:
                raise AssertionError("stale proposal returned")
            finally:
                model.diffusion_student.denoise = original
            assert state(cache) == before
            model.attach_semantic_capsules(memories[0])
            _, fresh = model.prefill(tokens)
            proposals, receipt = model.diffusion_propose(fresh, count=8, steps=3)
            assert bool(mx.all(proposals == baseline).item())
            assert not receipt["kv_committed"] and not receipt["target_verified"]
            report["cases"].append({"revision": revision, "rejection": rejection,
                                    "source_cache_unchanged": True,
                                    "original_binding_fresh_retry_proposals_exact": True})
        report["parameters"] = c.capacity()["parameters"]
        report["peak_gib"] = mx.get_peak_memory() / (1 << 30)
        report["capsule_bindings"] = [memory.binding() for memory in memories]
    root = Path(__file__).resolve().parents[1]
    report["checkpoint_sha256"] = file_hash(args.checkpoint / "model.safetensors")
    report["source_sha256"] = {str(path.relative_to(root)): file_hash(path)
        for path in (Path(__file__).resolve(), root / "src/mlx2/experimental/hysparse2/model.py")}
    report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
