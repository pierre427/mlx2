"""Bounded capsule capacity and diffusion grids; synthetic mechanics only."""

import argparse
import hashlib
import json
import time
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
    revision = file_hash(args.checkpoint / "model.safetensors")
    report = {"schema": "mlx2.hysparse2-memory-diffusion-bounds.v1", "completed": False,
              "checkpoint_sha256": revision, "optimizer_update_performed": False,
              "serving_route_qualified": False, "learned_tool_memory_behavior_qualified": False}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx import nn
        from mlx.utils import tree_flatten
        from mlx2.experimental.hysparse2 import model as model_module
        from mlx2.experimental.hysparse2 import capsule_memory as memory_module
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.train import _load_model_state, loss

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(12 << 30)
        mx.set_cache_limit(256 << 20)
        c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model = model_module.Model(c)
        _load_model_state(args.checkpoint, model)
        store = CapsuleStore(args.output / "capsules")
        concepts, edges = {}, []
        for i in range(32):
            subject, obj = f"tool{i}", f"record{i}"
            label = "validated synthetic argument " + str(i) + "x" * (i * 13)
            concepts[subject] = {"label": subject}
            concepts[obj] = {"label": label}
            edges.append({"subject": subject, "relation": "has_property", "object": obj,
                          "authority": "committed", "evidence_digest": hashlib.sha256(label.encode()).hexdigest()})
        bindings = {"model_binding": revision, "tokenizer_binding": "synthetic-character-v1", "runtime_binding": "hysparse2-bounds-v1"}
        capsule = store.put(kind="semantic_base", data={"schema": SEMANTIC_SCHEMA, "concepts": concepts, "edges": edges, "proposals": []},
                            provenance={"source": "original synthetic capacity fixture; no observed tool traces"}, **bindings)
        memory = memory_module.CapsuleMemory.from_store(store, capsule.digest, **bindings, encode=lambda s: [ord(x) % c.vocab_size for x in s], vocab_size=c.vocab_size, top_k=4)
        report["memory"] = {"records": len(memory.tokens), "top_k": memory.top_k, "max_record_tokens": max(map(len, memory.tokens)), "binding": memory.binding()}
        assert len(memory.tokens) == 32 and max(map(len, memory.tokens)) <= 512
        tokens = (mx.arange(288).reshape(2, 144) * 17 + 1) % c.vocab_size
        model.train()
        model.checkpoint_layers = True
        mx.random.seed(17)
        _, baseline = nn.value_and_grad(model, loss)(model, tokens)
        mx.eval(baseline)
        model.attach_semantic_capsules(memory)
        mx.random.seed(17)
        value, gradients = nn.value_and_grad(model, loss)(model, tokens)
        mx.eval(value, gradients)
        left, right = dict(tree_flatten(baseline)), dict(tree_flatten(gradients))
        probes = ("self_decoder.0.attention.q.weight", "semantic_ple.value.weight", "diffusion_student.layers.0.mlp.up.weight", "diffusion_student.layers.1.mlp.up.weight")
        report["memory_gradient_differences"] = {key: float(mx.max(mx.abs(left[key] - right[key])).item()) for key in probes}
        assert all(x > 0 for x in report["memory_gradient_differences"].values())
        assert all(bool(mx.all(mx.isfinite(x)).item()) for x in right.values())
        del baseline, gradients, left, right
        model.eval()
        _, cache = model.prefill(tokens)
        before = (cache.length, cache.self_layer_calls, cache.cross_layer_calls, cache.owner, tuple(id(x) for x in cache.arrays()), id(cache.boundary), id(cache.ple_history))
        report["cells"] = []
        for count in (1, 8, 64):
            for steps in (1, 3, 16):
                started = time.perf_counter()
                proposals, receipt = model.diffusion_propose(cache, count=count, steps=steps)
                mx.eval(proposals)
                assert proposals.shape == (2, count) and bool(mx.all((proposals >= 0) & (proposals < c.vocab_size)).item())
                after = (cache.length, cache.self_layer_calls, cache.cross_layer_calls, cache.owner, tuple(id(x) for x in cache.arrays()), id(cache.boundary), id(cache.ple_history))
                assert after == before and not receipt["kv_committed"] and not receipt["target_verified"]
                report["cells"].append({"count": count, "steps": steps, "seconds": time.perf_counter() - started, "target_kv_unchanged": True})
        for count, steps in ((0, 1), (65, 1), (1, 0), (1, 17)):
            try:
                model.diffusion_propose(cache, count=count, steps=steps)
                raise AssertionError("unbounded proposal accepted")
            except ValueError as exc:
                assert "bounded limits" in str(exc)
        report["invalid_budget_cases_rejected"] = 4
        report["invalid_prefix_cases_rejected"] = []
        for damage in ("missing_layer", "offset", "boundary", "history"):
            _, bad = model.prefill(tokens)
            if damage == "missing_layer":
                del bad.self_kv[0]
            elif damage == "offset":
                k, v, offset = bad.cross_kv[0][0]
                bad.cross_kv[0][0] = k, v, offset + 1
            elif damage == "boundary":
                bad.boundary = bad.boundary[:, :, :1]
            else:
                bad.ple_history = None
            before = (bad.length, bad.self_layer_calls, bad.cross_layer_calls)
            try:
                model.diffusion_propose(bad, count=8, steps=3)
                raise AssertionError("incomplete diffusion prefix accepted")
            except ValueError as exc:
                assert "endpoint state" in str(exc)
            assert before == (bad.length, bad.self_layer_calls, bad.cross_layer_calls)
            report["invalid_prefix_cases_rejected"].append(damage)
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["source_hashes"] = {str(path): file_hash(path) for path in (Path(__file__), Path(model_module.__file__), Path(memory_module.__file__))}
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
