"""Bounded full-weight update: invalidate KV/APC, then bind exact saved weights."""

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
    report = {"schema": "mlx2.hysparse2-parameter-state.v1", "completed": False,
              "serving_route_qualified": False, "optimizer_steps": 1,
              "parent_checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors")}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx import nn, optimizers
        from mlx2.experimental.hysparse2 import model as model_module
        from mlx2.experimental.hysparse2 import apc as apc_module
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.train import _load_model_state, loss, save_checkpoint
        from mlx2.runtime.apc_v2 import APCv2

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(12 << 30)
        mx.set_cache_limit(256 << 20)
        c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        model = model_module.Model(c)
        _load_model_state(args.checkpoint, model)
        model.eval()
        tokens = (mx.arange(144)[None] * 17 + 1) % c.vocab_size
        prefix = tokens[:, :72].tolist()[0]
        _, old = model.prefill(tokens[:, :72])
        engine = APCv2(max_size=4, layout_name="hysparse2-endpoint-v1")
        try:
            bridge = apc_module.EndpointAPC(model, engine, checkpoint_revision=report["parent_checkpoint_sha256"], tokenizer_fingerprint="synthetic-arithmetic-v1")
            bridge.publish(prefix, old)
            before = model.self_decoder[0].attention.q.weight
            model.train()
            model.checkpoint_layers = True
            value, gradients = nn.value_and_grad(model, loss)(model, tokens)
            mx.eval(value, gradients)
            optimizer = optimizers.Adam(1e-4)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            report["loss"] = float(value.item())
            report["q_update_max_abs"] = float(mx.max(mx.abs(model.self_decoder[0].attention.q.weight - before)).item())
            assert bool(mx.isfinite(value).item()) and report["q_update_max_abs"] > 0
            model.eval()
            length = old.length
            try:
                model.decode(tokens[:, 72:73], old)
                raise AssertionError("old KV accepted")
            except ValueError as exc:
                assert "another model" in str(exc)
            assert old.length == length
            report["old_kv_rejected_before_mutation"] = True
            for operation in (lambda: bridge.restore(prefix + [int(tokens[0, 72].item())]), lambda: bridge.publish(prefix, model.prefill(tokens[:, :72])[1])):
                try:
                    operation()
                    raise AssertionError("old checkpoint bridge accepted")
                except ValueError as exc:
                    assert "parameter update" in str(exc)
            report["old_bridge_restore_and_publish_rejected"] = True
            path = save_checkpoint(args.output / "checkpoints", model, optimizer, 1, {"qualification": "parameter-state"}, mode="model")
            revision = file_hash(path / "model.safetensors")
            rebound = apc_module.EndpointAPC(model, engine, checkpoint_revision=revision, tokenizer_fingerprint="synthetic-arithmetic-v1")
            _, miss = rebound.restore(prefix + [int(tokens[0, 72].item())])
            assert not miss.hit
            _, fresh = model.prefill(tokens[:, :72])
            rebound.publish(prefix, fresh)
            restored, hit = rebound.restore(prefix + [int(tokens[0, 72].item())])
            assert hit.hit
            try:
                expected = model.decode(tokens[:, 72:73], fresh)
                actual = model.decode(tokens[:, 72:73], restored)
                report["rebound_decode_max_error"] = float(mx.max(mx.abs(expected - actual)).item())
                assert report["rebound_decode_max_error"] == 0
            finally:
                if hasattr(hit.cache, "close"):
                    hit.cache.close()
            report["updated_checkpoint_sha256"] = revision
            report["peak_memory_bytes"] = mx.get_peak_memory()
            report["source_hashes"] = {str(path): file_hash(path) for path in (Path(__file__), Path(model_module.__file__), Path(apc_module.__file__))}
            report["completed"] = True
        finally:
            engine.close()
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
