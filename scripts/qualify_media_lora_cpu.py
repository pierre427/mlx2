#!/usr/bin/env python3
"""Reproduce tiny real-backend LoRA effect/training contracts on Apple CPU.

Use mlx2's venv for Qwen/Music3 and the pinned LTX venv for LTX. These checks
produce synthetic CPU receipts, never production route qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        required=True,
        choices=["qwen-image-2.1", "ltx-2.5", "minimax-music3"],
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    import numpy as np

    from mlx2.runtime.media_lora import install_media_lora, write_media_lora
    from mlx2.runtime.media_lora_training import (
        image_flow_example,
        music_flow_example,
        train_media_lora,
        video_flow_example,
    )

    if args.family == "qwen-image-2.1":
        from mlx2.adapters.generative_media import (
            QWEN_BACKEND_REVISION,
            _verify_qwen_backend_revision,
        )

        _verify_qwen_backend_revision()
        from mlx_vlm.models.qwen_image.transformer import QwenImageTransformer

        revision = QWEN_BACKEND_REVISION
        config = {
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 1,
            "num_attention_heads": 1,
            "attention_head_dim": 8,
            "context_in_dim": 8,
            "axes_dims_rope": (2, 2, 4),
        }
        model = QwenImageTransformer(**config)
        key = "transformer_blocks.0.attn.to_v"

        def make(scale, base):
            clean = mx.full((1, 4, 4), scale)
            return image_flow_example(
                clean,
                mx.zeros_like(clean),
                mx.ones((1, 2, 8)),
                time=0.4,
                img_shape=(1, 2, 2),
                base_fingerprint=base,
            )
    elif args.family == "minimax-music3":
        if args.runtime_root is None:
            parser.error("--runtime-root required for Music3")
        from mlx2.adapters.music3 import MUSIC3_RUNTIME_REVISION
        from mlx2.adapters.music3_pin import SOURCE_SHA256
        from mlx2.runtime.media_lora import _digest

        root = args.runtime_root.resolve()
        for name, expected in SOURCE_SHA256.items():
            if _digest(root / name) != expected:
                raise ValueError("Music3 source pin differs: " + name)
        sys.path.insert(0, str(root))
        from minimax_music3_mlx.dit import DiT

        revision = MUSIC3_RUNTIME_REVISION
        config = {
            "num_layers": 1,
            "num_attention_heads": 1,
            "attention_head_dim": 8,
            "ff_inner_dim": 16,
            "rotary_dim": 4,
            "fourier_embedding_dim": 8,
        }
        model = DiT(**config)
        key = "blocks.0.attn.to_qkv"

        def make(scale, base):
            clean = mx.full((1, 128, 2), scale)
            return music_flow_example(
                clean,
                mx.zeros_like(clean),
                mx.ones((1, 2, 2048)),
                time=0.4,
                base_fingerprint=base,
            )
    else:
        import subprocess

        from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

        from mlx2.adapters.generative_media import LTX_RUNTIME_REVISION

        if args.runtime_root is None:
            parser.error("--runtime-root required for LTX")
        if (
            subprocess.check_output(
                ["git", "-C", str(args.runtime_root), "rev-parse", "HEAD"], text=True
            ).strip()
            != LTX_RUNTIME_REVISION
        ):
            raise ValueError("LTX source pin differs")
        revision = LTX_RUNTIME_REVISION
        config = {
            "num_layers": 1,
            "video_dim": 8,
            "audio_dim": 8,
            "video_num_heads": 1,
            "audio_num_heads": 1,
            "video_head_dim": 8,
            "audio_head_dim": 8,
            "av_cross_num_heads": 1,
            "av_cross_head_dim": 8,
            "video_patch_channels": 4,
            "audio_patch_channels": 4,
            "ff_mult": 2,
            "timestep_embedding_dim": 8,
            "model_version": "2.5",
        }
        model = LTXModel(LTXModelConfig(**config))
        key = "transformer_blocks.0.attn1.to_v"

        def make(scale, base):
            v, a = mx.full((1, 4, 4), scale), mx.full((1, 2, 4), scale)
            return video_flow_example(
                v,
                mx.zeros_like(v),
                a,
                mx.zeros_like(a),
                time=0.4,
                video_condition=mx.ones((1, 2, 8)),
                audio_condition=mx.ones((1, 2, 8)),
                video_positions=mx.zeros((1, 4, 3)),
                audio_positions=mx.zeros((1, 2, 1)),
                base_fingerprint=base,
            )

    fingerprint = hashlib.sha256(
        json.dumps(
            {"family": args.family, "tiny_config": config}, sort_keys=True
        ).encode()
    ).hexdigest()
    train, held = make(1.0, fingerprint), make(0.9, fingerprint)

    def outputs():
        value = model(**held.inputs)
        return value if isinstance(value, tuple) else (value,)

    reference = outputs()
    mx.eval(reference)
    linear = dict(model.named_modules())[key]
    out_dim, in_dim = linear.weight.shape
    results = {}
    with tempfile.TemporaryDirectory(prefix="media-lora-cpu-") as temporary:
        root = Path(temporary)
        for label, b in [("zero", 0.0), ("nonzero", 0.2)]:
            rng = np.random.default_rng(7)
            tensors = {
                key + ".lora_a": (rng.standard_normal((in_dim, 2)) * 0.1).astype(
                    np.float32
                ),
                key + ".lora_b": (rng.standard_normal((2, out_dim)) * b).astype(
                    np.float32
                ),
            }
            adapter = write_media_lora(
                root / label,
                tensors,
                family=args.family,
                base_fingerprint=fingerprint,
                backend_revision=revision,
            )
            session = install_media_lora(model, adapter)
            actual = outputs()
            differences = [
                float(mx.max(mx.abs(v - r)).item()) for v, r in zip(actual, reference)
            ]
            if (label == "zero" and any(d != 0 for d in differences)) or (
                label == "nonzero" and not any(d > 0 for d in differences)
            ):
                raise RuntimeError("real-backend adapter effect check failed")
            results[label + "_max_abs_difference"] = differences
            session.restore()
            if args.family == "ltx-2.5":
                from ltx_pipelines_mlx.utils._orchestration import fuse_pending_loras
                from mlx.utils import tree_flatten

                from mlx2.runtime.media_lora import export_ltx_native

                native = root / (label + "-native.safetensors")
                export_ltx_native(adapter, native)
                base_weights = dict(tree_flatten(model.parameters()))
                fused_weights = fuse_pending_loras(base_weights, [(str(native), 1.0)])
                fused_model = LTXModel(LTXModelConfig(**config))
                fused_model.load_weights(list(fused_weights.items()), strict=True)
                fused_outputs = fused_model(**held.inputs)
                native_difference = [
                    float(mx.max(mx.abs(a - b)).item())
                    for a, b in zip(fused_outputs, actual)
                ]
                if any(
                    not mx.allclose(a, b, atol=1e-5).item()
                    for a, b in zip(fused_outputs, actual)
                ):
                    raise RuntimeError("native LTX fusion differs from reversible LoRA")
                results[label + "_native_fusion_vs_wrapper_max_abs_difference"] = (
                    native_difference
                )
            if any(
                not mx.array_equal(v, r).item() for v, r in zip(outputs(), reference)
            ):
                raise RuntimeError("base restoration failed")
        trained = train_media_lora(
            model,
            family=args.family,
            base_fingerprint=fingerprint,
            backend_revision=revision,
            keys=[key],
            examples=[train],
            validation_examples=[held],
            output=root / "trained",
            steps=3,
            learning_rate=0.001,
        )
        results["training"] = trained.config["training"]
        loaded = install_media_lora(model, trained)
        changed = outputs()
        mx.eval(changed)
        saved = loaded.export(root / "saved", training=trained.config["training"])
        loaded.restore()
        loaded = install_media_lora(model, saved)
        if any(not mx.array_equal(a, b).item() for a, b in zip(outputs(), changed)):
            raise RuntimeError("trained save/reload parity failed")
        loaded.restore()
    if any(not mx.array_equal(v, r).item() for v, r in zip(outputs(), reference)):
        raise RuntimeError("training restoration failed")
    receipt = {
        "family": args.family,
        "backend_revision": revision,
        "device": str(mx.default_device()),
        "synthetic_base_fingerprint": fingerprint,
        "tiny_config": config,
        "results": results,
        "base_restored": True,
        "trained_reload_exact": True,
        "qualified": False,
        "production_trained": False,
        "production_selected": False,
        "production_observed_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
