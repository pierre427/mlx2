"""Bounded full BF16 APCv2 endpoint revision-race exercise on M3."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "mlx2.hysparse2-apc-revision.v1", "completed": False,
              "serving_route_qualified": False, "optimizer_updates": 0, "cases": []}

    class Lease(list):
        closed = False

        def close(self):
            self.closed = True

    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx2.experimental.hysparse2.apc import EndpointAPC
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state
        from mlx2.runtime.apc_v2 import APCv2

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(12 << 30)
        mx.set_cache_limit(128 << 20)
        c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        revision = file_hash(args.checkpoint / "model.safetensors")
        tokens = [(i * 17 + 1) % c.vocab_size for i in range(144)]
        for phase in ("publish_validation", "restore_validation", "restore_allocation"):
            engine = APCv2(max_size=4, layout_name="hysparse2-endpoint-v1")
            raw = None
            try:
                bridge = EndpointAPC(model, engine, checkpoint_revision=revision,
                                     tokenizer_fingerprint="synthetic-token-fixture-v1")
                _, cache = model.prefill(mx.array([tokens]))
                original = bridge._validate_endpoint

                def changed(value):
                    original(value)
                    if phase == "publish_validation":
                        model.attach_semantic_capsules(None)
                    else:
                        model.update({"embedding": {"weight": model.embedding.weight}})

                if phase == "publish_validation":
                    with patch.object(bridge, "_validate_endpoint", changed):
                        try:
                            bridge.publish(tokens, cache)
                        except ValueError as exc:
                            assert "revision" in str(exc) or "owner" in str(exc)
                        else:
                            raise AssertionError("stale endpoint published")
                    lease_closed = None
                else:
                    bridge.publish(tokens, cache)
                    raw = engine.lookup(bridge.key(), tokens + [5])
                    assert raw.hit
                    lease = Lease(raw.cache)
                    with patch.object(engine, "lookup", lambda *_: SimpleNamespace(
                            hit=True, cache=lease, cached_tokens=len(tokens))):
                        if phase == "restore_validation":
                            injection = patch.object(bridge, "_validate_endpoint", changed)
                        else:
                            allocate = model.new_cache
                            calls = 0

                            def changed_allocation(*values, **kwargs):
                                nonlocal calls
                                calls += 1
                                if calls == 3:
                                    model.update({"embedding": {"weight": model.embedding.weight}})
                                return allocate(*values, **kwargs)

                            injection = patch.object(model, "new_cache", changed_allocation)
                        with injection:
                            try:
                                bridge.restore(tokens + [5])
                            except ValueError as exc:
                                assert any(word in str(exc) for word in ("revision", "owner", "parameter update"))
                            else:
                                raise AssertionError("stale endpoint restored")
                    assert lease.closed
                    lease_closed = True
                rebound = EndpointAPC(model, engine, checkpoint_revision=revision,
                                      tokenizer_fingerprint="synthetic-token-fixture-v1")
                _, fresh = model.prefill(mx.array([tokens]))
                rebound.publish(tokens, fresh)
                restored, hit = rebound.restore(tokens + [5])
                try:
                    expected = model.decode(mx.array([[5]]), fresh)
                    actual = model.decode(mx.array([[5]]), restored)
                    error = float(mx.max(mx.abs(expected - actual)).item())
                    assert error == 0
                finally:
                    if hasattr(hit.cache, "close"):
                        hit.cache.close()
                report["cases"].append({"phase": phase, "rejected": True,
                                        "failed_restore_lease_closed": lease_closed,
                                        "fresh_bound_retry_decode_error": error})
            finally:
                if raw is not None and hasattr(raw.cache, "close"):
                    raw.cache.close()
                engine.close()
        report["parameters"] = c.capacity()["parameters"]
        report["dtype"] = "bfloat16"
        report["peak_gib"] = mx.get_peak_memory() / (1 << 30)
    root = Path(__file__).resolve().parents[1]
    report["checkpoint_sha256"] = file_hash(args.checkpoint / "model.safetensors")
    report["source_sha256"] = {str(path.relative_to(root)): file_hash(path)
        for path in (Path(__file__).resolve(), root / "src/mlx2/experimental/hysparse2/apc.py")}
    report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
