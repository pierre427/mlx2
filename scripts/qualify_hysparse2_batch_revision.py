"""Bounded M3 batching revision transition rejection; no serving qualification."""

import argparse
import json
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
    report = {"schema": "mlx2.hysparse2-batch-revision.v1", "completed": False,
              "serving_route_qualified": False, "cases": []}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx2.experimental.hysparse2.batching import ResearchBatcher
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(8 << 30)
        mx.set_cache_limit(128 << 20)
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.eval()
        mx.eval(model.parameters())
        prompts = [[1, 2, 3], [4, 5, 6]]
        for phase in ("merge", "decode_split", "prefill_split"):
            batcher = ResearchBatcher(model, max_lanes=2)
            _, caches, _ = batcher.prefill(prompts)
            before = [list(cache.arrays()) for cache in caches]
            owner = model._cache_owner
            method = "_merge" if phase == "merge" else "_split"
            original = getattr(batcher, method)

            def changed(*values):
                model.update({"embedding": {"weight": model.embedding.weight}})
                return original(*values)

            setattr(batcher, method, changed)
            try:
                if phase == "prefill_split":
                    batcher.prefill(prompts)
                else:
                    batcher.decode([[7], [8]], caches)
            except ValueError as exc:
                assert "revision" in str(exc) or "owner" in str(exc)
                message = str(exc)
            else:
                raise AssertionError("revision transition accepted")
            assert all(cache.owner is owner and cache.length == 3 for cache in caches)
            assert all(all(a is b for a, b in zip(old, cache.arrays()))
                       for old, cache in zip(before, caches))
            setattr(batcher, method, original)
            _, fresh, _ = batcher.prefill(prompts)
            got, updated, _ = batcher.decode([[7], [8]], fresh)
            errors = []
            for i, row in enumerate(prompts):
                _, reference = model.prefill(mx.array([row]))
                expected = model.decode(mx.array([[7 + i]]), reference)
                errors.append(float(mx.max(mx.abs(got[i] - expected)).item()))
            assert max(errors) < 1e-3
            assert all(cache.length == 4 for cache in updated)
            report["cases"].append({"phase": phase, "rejection": message,
                                    "source_state_unchanged": True,
                                    "fresh_retry_errors": errors})
        original_prefill = model.prefill
        calls = 0

        def changed_prefill(*values, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                model.update({"embedding": {"weight": model.embedding.weight}})
            return original_prefill(*values, **kwargs)

        model.prefill = changed_prefill
        try:
            ResearchBatcher(model).prefill([[1, 2, 3], [4, 5]])
        except ValueError as exc:
            assert "revision" in str(exc) or "owner" in str(exc)
            report["cross_cohort_revision_rejected"] = True
        else:
            raise AssertionError("mixed revisions returned")
        finally:
            model.prefill = original_prefill
        report["peak_gib"] = mx.get_peak_memory() / (1 << 30)
    root = Path(__file__).resolve().parents[1]
    report["parameters"] = c.capacity()["parameters"]
    report["checkpoint_sha256"] = file_hash(args.checkpoint / "model.safetensors")
    report["source_sha256"] = {str(path.relative_to(root)): file_hash(path)
        for path in (Path(__file__).resolve(),
                     root / "src/mlx2/experimental/hysparse2/batching.py")}
    report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
