"""Local-only native MLX pipeline for the pinned MOSS-Image-SFT artifact."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from PIL import Image

from .mlx_vlm_pin import require_pinned_mlx_vlm
from .moss_image_autoencoder import MOSSImageAutoEncoder, convert_vae_weights
from .moss_image_transformer import MOSSImageTransformer, flow_schedule

Q8 = {"group_size": 64, "bits": 8, "mode": "affine"}


def _load_indexed(root, index_name, *, include_prefix=None):
    index = json.loads((root / index_name).read_text())["weight_map"]
    weights = {}
    for shard in sorted(set(index.values())):
        loaded = mx.load(str(root / shard))
        weights.update(
            {
                name: value
                for name, value in loaded.items()
                if include_prefix is None or name.startswith(include_prefix)
            }
        )
    return weights


def _quantize_if_declared(model, config):
    quantization = config.get("quantization")
    if quantization is None:
        return
    if quantization != Q8 or config.get("mlx_format") is not True:
        raise ValueError("unsupported MOSS-Image MLX quantization")
    # The conversion deliberately preserves token embeddings.  MLX quantizes
    # both Linear and Embedding modules by default, so keep the runtime module
    # selection exactly aligned with the packed artifact.
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda _path, module: isinstance(module, nn.Linear),
    )


class TextEncoder:
    def __init__(self, root, tokenizer_root):
        self.runtime = require_pinned_mlx_vlm()
        from mlx_vlm.models.qwen3_5.config import TextConfig
        from mlx_vlm.models.qwen3_5.language import Qwen3_5Model
        from transformers import AutoTokenizer

        config = json.loads((root / "config.json").read_text())
        self.model = Qwen3_5Model(TextConfig.from_dict(config["text_config"]))
        _quantize_if_declared(self.model, config)
        weights = _load_indexed(
            root,
            "model.safetensors.index.json",
            include_prefix="language_model.",
        )
        sanitized = {}
        norm_suffixes = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for name, value in weights.items():
            name = name.removeprefix("language_model.")
            if "conv1d.weight" in name and value.ndim == 3 and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if name.endswith(norm_suffixes) and value.ndim == 1:
                value = value + 1
            sanitized[name] = value
        self.model.load_weights(list(sanitized.items()), strict=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_root, local_files_only=True, trust_remote_code=False
        )

    def encode(self, prompt, max_length=4096):
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        values = self.tokenizer(
            formatted, truncation=True, max_length=max_length, return_tensors="np"
        )["input_ids"]
        inputs = mx.array(values, dtype=mx.int32)
        positions = mx.broadcast_to(
            mx.arange(inputs.shape[1], dtype=mx.int32)[None, None],
            (3, 1, inputs.shape[1]),
        )
        result = self.model(inputs, position_ids=positions)[0].astype(mx.bfloat16)
        mx.eval(result)
        return result


class MOSSImagePipeline:
    def __init__(self, root):
        root = Path(root)
        transformer_config = json.loads((root / "transformer/config.json").read_text())
        self.transformer = MOSSImageTransformer(transformer_config)
        _quantize_if_declared(self.transformer, transformer_config)
        transformer_weights = _load_indexed(
            root / "transformer", "diffusion_pytorch_model.safetensors.index.json"
        )
        bands = transformer_weights.pop("rope.bands", None)
        expected_bands = 1 / (
            float(transformer_config.get("rope_base", 10000.0))
            ** (
                mx.arange(
                    int(transformer_config["dim"])
                    // int(transformer_config["num_heads"])
                    // 8,
                    dtype=mx.float32,
                )
                / (
                    int(transformer_config["dim"])
                    // int(transformer_config["num_heads"])
                    // 8
                )
            )
        )
        if (
            bands is None
            or bands.shape != expected_bands.shape
            or not mx.allclose(
                bands.astype(mx.float32), expected_bands, rtol=1e-6, atol=0
            ).item()
        ):
            raise ValueError("MOSS-Image derived rotary bands differ")
        self.transformer.load_weights(list(transformer_weights.items()), strict=True)
        self.transformer.eval()

        vae_config = json.loads((root / "vae/config.json").read_text())
        self.vae = MOSSImageAutoEncoder(vae_config)
        vae_weights = mx.load(str(root / "vae/diffusion_pytorch_model.safetensors"))
        self.vae.load_weights(
            list(convert_vae_weights(vae_weights).items()), strict=True
        )
        self.vae.eval()
        self.text_encoder = TextEncoder(root / "text_encoder", root / "tokenizer")
        mx.eval(
            self.transformer.parameters(),
            self.vae.parameters(),
            self.text_encoder.model.parameters(),
        )

    def generate(
        self,
        prompt,
        *,
        width,
        height,
        steps,
        seed,
        guidance_scale=3.0,
        negative_prompt="",
        uncond_skip_layers=(22,),
    ):
        positive = self.text_encoder.encode(prompt)
        negative = (
            self.text_encoder.encode(negative_prompt) if guidance_scale != 1.0 else None
        )
        latent_height, latent_width = height // 16, width // 16
        # Preserve the publisher's CPU Torch fp32 RNG boundary.
        import torch

        noise = (
            torch.randn(
                (1, 128, latent_height, latent_width),
                generator=torch.Generator(device="cpu").manual_seed(seed),
                dtype=torch.float32,
            )
            .permute(0, 2, 3, 1)
            .numpy()
        )
        latents = mx.array(noise, dtype=mx.float32)
        schedule = flow_schedule(steps)
        values = schedule.tolist()
        for timestep, following in pairwise(values):
            conditional = self.transformer(latents, timestep * 1000, positive).astype(
                mx.float32
            )
            if negative is None:
                prediction = conditional
            else:
                unconditional = self.transformer(
                    latents,
                    timestep * 1000,
                    negative,
                    skip_layers=uncond_skip_layers,
                ).astype(mx.float32)
                prediction = unconditional + guidance_scale * (
                    conditional - unconditional
                )
            latents = latents + (following - timestep) * prediction
            mx.eval(latents)
        if not mx.all(mx.isfinite(latents)).item():
            raise FloatingPointError("nonfinite MOSS-Image denoising state")
        image = self.vae.decode(latents)
        if not mx.all(mx.isfinite(image)).item():
            raise FloatingPointError("nonfinite MOSS-Image decoded image")
        pixels = ((mx.clip(image[0], -1, 1) + 1) * 127.5).astype(mx.uint8)
        mx.eval(pixels)
        return Image.fromarray(np.array(pixels))
