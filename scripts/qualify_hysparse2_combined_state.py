"""M3 capsule/batching/APCv2/diffusion/LoRA composition; no learned tool claim."""

import argparse
import hashlib
import json
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_memory import SEMANTIC_SCHEMA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    revision = file_hash(args.checkpoint / "model.safetensors")
    report = {
        "schema": "mlx2.hysparse2-combined-state.v1",
        "completed": False,
        "checkpoint_sha256": revision,
        "serving_route_qualified": False,
        "learned_tool_memory_behavior_qualified": False,
        "mixed_memory_batching": False,
    }
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx

        from mlx2.experimental.hysparse2.apc import EndpointAPC
        from mlx2.experimental.hysparse2.batching import ResearchBatcher
        from mlx2.experimental.hysparse2.capsule_memory import CapsuleMemory
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.lora import LoRAEpisode
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state, loss
        from mlx2.runtime.apc_v2 import APCv2

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(12 << 30)
        mx.set_cache_limit(256 << 20)
        c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.eval()
        model.checkpoint_layers = True
        store = CapsuleStore(args.output / "capsules")
        bindings = {
            "model_binding": revision,
            "tokenizer_binding": "character-fixture",
            "runtime_binding": "hysparse2-capsule-v1",
        }

        def snapshot(text):
            graph = {
                "schema": SEMANTIC_SCHEMA,
                "concepts": {"a": {"label": "tool call"}, "b": {"label": text}},
                "edges": [
                    {
                        "subject": "a",
                        "relation": "has_property",
                        "object": "b",
                        "authority": "committed",
                        "evidence_digest": hashlib.sha256(text.encode()).hexdigest(),
                    }
                ],
                "proposals": [],
            }
            capsule = store.put(
                kind="semantic_base",
                data=graph,
                **bindings,
                provenance={"source": "original synthetic composition fixture"},
            )
            return CapsuleMemory.from_store(
                store,
                capsule.digest,
                **bindings,
                encode=lambda s: [ord(x) % c.vocab_size for x in s],
                vocab_size=c.vocab_size,
            )

        a, b = snapshot("valid arguments"), snapshot("revised arguments")
        model.attach_semantic_capsules(a)
        prompts = [
            [(j * 17 + i + 1) % c.vocab_size for j in range(n)]
            for i, n in enumerate((144, 65, 144))
        ]
        batcher = ResearchBatcher(model, max_lanes=2)
        logits, caches, receipt = batcher.prefill(prompts)
        references = [model.prefill(mx.array([p])) for p in prompts]
        errors = [
            float(mx.max(mx.abs(value - ref[0])).item())
            for value, ref in zip(logits, references, strict=True)
        ]
        rows = [[31 + i] for i in range(3)]
        decoded, updated, _ = batcher.decode(rows, caches)
        errors += [
            float(mx.max(mx.abs(value - model.decode(mx.array([row]), ref[1]))).item())
            for value, row, ref in zip(decoded, rows, references, strict=True)
        ]
        assert max(errors) < 1e-3
        report["same_memory_ragged_parity_max_error"] = max(errors)
        report["batch_receipt"] = receipt
        engine = APCv2(max_size=8, layout_name="hysparse2-endpoint-v1")
        leases = []
        episode = None
        try:
            bridge = EndpointAPC(
                model,
                engine,
                checkpoint_revision=revision,
                tokenizer_fingerprint="character-fixture",
            )
            bridge.publish(prompts[0], caches[0])
            # APC exactness compares the same row geometry, while cohort-vs-B1
            # parity above permits normal floating-point reduction differences.
            endpoint_reference = model.decode(mx.array([rows[0]]), caches[0])
            mx.eval(endpoint_reference)

            def restore():
                restored, hit = bridge.restore(prompts[0] + rows[0])
                assert hit.hit
                leases.append(hit.cache)
                return restored

            restored = restore()
            proposal, proposal_receipt = model.diffusion_propose(
                restored, count=8, steps=3
            )
            assert restored.length == 144
            got, _, _ = batcher.decode([rows[0]], [restored])
            assert float(mx.max(mx.abs(got[0] - endpoint_reference)).item()) == 0
            report["apc_to_batch_decode_error"] = 0.0
            report["diffusion_proposal"] = proposal.tolist()
            report["diffusion_receipt"] = proposal_receipt
            model.attach_semantic_capsules(b)
            for stale in (caches[0], updated[0], restored):
                try:
                    batcher.decode([rows[0]], [stale])
                except ValueError:
                    pass
                else:
                    raise AssertionError("stale memory request survived")
            assert bridge.restore(prompts[0] + rows[0])[0] is None
            report["revised_memory_rejects_old_state_and_apc"] = True
            model.attach_semantic_capsules(a)
            restored = restore()
            episode = LoRAEpisode(
                model,
                ["semantic_ple.value", "diffusion_student.condition"],
                base_revision=revision,
                max_steps=1,
            )
            report["lora_step_loss"] = episode.step(mx.array([prompts[0]]), loss)
            model.eval()
            assert bridge.restore(prompts[0] + rows[0])[0] is None
            try:
                batcher.decode([rows[0]], [restored])
            except ValueError:
                pass
            else:
                raise AssertionError("stale adapter request survived")
            report["adapter_update_rejects_old_state_and_apc"] = True
            episode.rollback()
            recovered = restore()
            value = model.decode(mx.array([rows[0]]), recovered)
            error = float(mx.max(mx.abs(value - endpoint_reference)).item())
            assert error == 0
            report["rollback_apc_decode_error"] = error
            report["peak_memory_bytes"] = mx.get_peak_memory()
            report["source_hashes"] = {
                str(p): file_hash(p)
                for p in (
                    Path(__file__),
                    *[
                        Path("src/mlx2/experimental/hysparse2") / (name + ".py")
                        for name in (
                            "model",
                            "batching",
                            "apc",
                            "lora",
                            "capsule_memory",
                        )
                    ],
                )
            }
            report["completed"] = True
        finally:
            if episode is not None:
                episode.close()
            for lease in leases:
                if hasattr(lease, "close"):
                    lease.close()
            engine.close()
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
