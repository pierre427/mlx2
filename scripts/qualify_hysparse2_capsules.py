"""M3 capsule tensor/gradient/persistence mechanics; no learned retrieval claim."""

import argparse
import json
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.runtime.hyper_directory import DirectoryContext, HyperDirectory
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_memory import SemanticMemory, SemanticProposal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    c = Config(**json.loads(args.config.read_text()))
    store = CapsuleStore(args.output / "capsules")
    directory = HyperDirectory(args.output / "directory", store)
    bindings = {
        "model_binding": "synthetic-seed42",
        "tokenizer_binding": "character-fixture",
        "runtime_binding": "hysparse2-capsule-v1",
    }
    memory = SemanticMemory(capsules=store, directory=directory, **bindings)
    context = DirectoryContext(
        model="hysparse2", tenant="fixture", session="tool-cycle"
    )
    proposals = [
        SemanticProposal(
            "tool call",
            "has_property",
            "valid arguments",
            0.99,
            0.9,
            "Original synthetic fixture: tool calls require valid arguments.",
        ),
        SemanticProposal(
            "lookup result",
            "supports",
            "cited evidence",
            0.99,
            0.9,
            "Original synthetic fixture: lookup results provide cited evidence.",
        ),
        SemanticProposal(
            "memory update",
            "has_property",
            "revision check",
            0.99,
            0.9,
            "Original synthetic fixture: memory updates check revision.",
        ),
    ]
    commit = memory.commit_after_delivery(
        context,
        proposals,
        response_delivered=True,
        authenticated_tenant=True,
        expected_revision=0,
    )
    assert commit["committed"]
    _, digest, revision = memory.load(context)
    retrieved = memory.retrieve(context, "tool call", limit=8)
    assert retrieved.edges
    report = {
        "schema": "mlx2.hysparse2-capsule-mechanics.v1",
        "completed": False,
        "serving_route_qualified": False,
        "learned_memory_behavior_qualified": False,
        "commit": commit,
        "retrieved_edges": len(retrieved.edges),
        "revision": revision,
    }
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx import nn, optimizers
        from mlx.utils import tree_flatten

        from mlx2.experimental.hysparse2.capsule_memory import CapsuleMemory
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import (
            file_hash,
            load_checkpoint,
            loss,
            save_checkpoint,
        )

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(20 << 30)
        mx.set_cache_limit(256 << 20)
        mx.random.seed(42)
        model = Model(c)
        model.checkpoint_layers = True
        tokens = (mx.arange(2 * 144).reshape(2, 144) * 17 + 1) % c.vocab_size
        model.eval()
        baseline = model(tokens)[0]
        mx.eval(baseline)
        snapshot = CapsuleMemory.from_store(
            store,
            digest,
            encode=lambda s: [ord(x) % c.vocab_size for x in s],
            vocab_size=c.vocab_size,
            top_k=2,
            **bindings,
        )
        model.attach_semantic_capsules(snapshot)
        full = model(tokens)[0]
        mx.eval(full)
        report["memory_logit_delta"] = float(mx.max(mx.abs(full - baseline)).item())
        assert report["memory_logit_delta"] > 0
        got, cache = model.prefill(tokens[:, :140])
        errors = [float(mx.max(mx.abs(got - full[:, 139:140])).item())]
        for i in range(140, 144):
            got = model.decode(tokens[:, i : i + 1], cache)
            errors.append(float(mx.max(mx.abs(got - full[:, i : i + 1])).item()))
        report["cached_errors"] = errors
        assert max(errors) < 1e-3
        model.train()
        mx.random.seed(7)
        value, gradients = nn.value_and_grad(model, loss)(model, tokens)
        mx.eval(value, gradients)
        with_memory = dict(tree_flatten(gradients))
        model.attach_semantic_capsules(None)
        mx.random.seed(7)
        _, detached = nn.value_and_grad(model, loss)(model, tokens)
        mx.eval(detached)
        without = dict(tree_flatten(detached))
        watched = [
            "semantic_ple.embedding.weight",
            "semantic_ple.key.weight",
            "semantic_ple.value.weight",
            "self_decoder.0.attention.q.weight",
            "diffusion_student.condition.weight",
        ]
        report["gradient_changes"] = {
            k: float(mx.max(mx.abs(with_memory[k] - without[k])).item())
            for k in watched
        }
        assert min(report["gradient_changes"].values()) > 0
        assert all(
            bool(mx.all(mx.isfinite(g)).item()) for _, g in tree_flatten(gradients)
        )
        model.attach_semantic_capsules(snapshot)
        optimizer = optimizers.Adam(1e-4)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        del detached, without, with_memory, gradients
        path = save_checkpoint(
            args.output / "checkpoints", model, optimizer, 1, {"fixture": True}
        )
        model.eval()
        reference = model(tokens)[0]
        mx.eval(reference)
        restored = Model(c)
        try:
            load_checkpoint(path, restored, optimizers.Adam(1e-4), {"fixture": True})
        except ValueError as exc:
            assert "capsule read binding" in str(exc)
            report["missing_capsule_restore_rejected"] = True
        else:
            raise AssertionError("unbound checkpoint restored")
        restored.attach_semantic_capsules(snapshot)
        load_checkpoint(path, restored, optimizers.Adam(1e-4), {"fixture": True})
        restored.eval()
        report["checkpoint_error"] = float(
            mx.max(mx.abs(restored(tokens)[0] - reference)).item()
        )
        assert report["checkpoint_error"] == 0
        update = SemanticProposal(
            "tool call",
            "has_property",
            "validated arguments",
            0.99,
            0.9,
            "Original revised synthetic fixture.",
        )
        memory.commit_after_delivery(
            context,
            [update],
            response_delivered=True,
            authenticated_tenant=True,
            expected_revision=revision,
        )
        _, new_digest, _ = memory.load(context)
        assert new_digest != digest
        revised = CapsuleMemory.from_store(
            store,
            new_digest,
            encode=lambda s: [ord(x) % c.vocab_size for x in s],
            vocab_size=c.vocab_size,
            top_k=2,
            **bindings,
        )
        restored.attach_semantic_capsules(revised)
        try:
            load_checkpoint(path, restored, optimizers.Adam(1e-4), {"fixture": True})
        except ValueError as exc:
            assert "capsule read binding" in str(exc)
            report["revised_capsule_restore_rejected"] = True
        else:
            raise AssertionError("revised memory checkpoint restored")
        report["capsule_binding"] = snapshot.binding()
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["source_hashes"] = {
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path("src/mlx2/experimental/hysparse2/capsule_memory.py"),
                Path("src/mlx2/experimental/hysparse2/model.py"),
                Path("src/mlx2/experimental/hysparse2/train.py"),
            ]
        }
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
