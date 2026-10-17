"""Local-only native MLX pipeline for the pinned distilled FLUX 3 package."""

from __future__ import annotations

import json
import math

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from .flux3_autoencoder import Flux3AutoEncoder, convert_vae_weights
from .flux3_transformer import Flux3Transformer, latent_ids, timesteps
from .mlx_vlm_pin import require_pinned_mlx_vlm

OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


class TextEncoder:
    def __init__(self, root):
        self.runtime = require_pinned_mlx_vlm()
        from mlx_vlm.models.qwen3_vl.config import TextConfig
        from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel
        from transformers import AutoProcessor

        config = json.loads((root / "config.json").read_text())["text_config"]
        self.model = Qwen3VLModel(TextConfig.from_dict(config))
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        weights = {}
        for shard in sorted(set(index.values())):
            weights.update(
                {
                    k.removeprefix("model.language_model."): v
                    for k, v in mx.load(str(root / shard)).items()
                    if k.startswith("model.language_model.")
                }
            )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
        self.processor.tokenizer.padding_side = "right"

    def encode(self, prompt):
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenizer = self.processor.tokenizer
        real = len(tokenizer(text, truncation=True, max_length=4096)["input_ids"])
        length = min(math.ceil(real / 80) * 80, 4096)
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=length,
            return_tensors="np",
        )
        valid = mx.array(inputs["attention_mask"], dtype=mx.bool_)
        positions = mx.where(valid, mx.cumsum(valid.astype(mx.int32), axis=-1) - 1, 1)
        positions = mx.broadcast_to(positions[None], (3, 1, length))
        causal = mx.arange(length)[:, None] >= mx.arange(length)[None, :]
        mask = causal[None, None] & valid[:, None, None, :]
        h = self.model.embed_tokens(mx.array(inputs["input_ids"], dtype=mx.int32))
        states = []
        for i, layer in enumerate(self.model.layers[: max(OUTPUT_LAYERS)], 1):
            h = layer(h, mask=mask, position_ids=positions)
            if i in OUTPUT_LAYERS:
                states.append(h)
        tokens = mx.stack(states, axis=2).reshape(length, -1).astype(mx.bfloat16)
        ids = mx.stack((mx.zeros((length,)),) * 3 + (mx.arange(length),), axis=-1)
        mx.eval(tokens)
        return tokens, ids


def prepare_reference(image, *, multiple):
    """Prepare one reference as the pinned BFL pipeline does.

    A single reference is fitted under 4 MP and padded, centred, onto its
    16px-grid canvas. Each reference of a multi-reference edit is fitted
    under 1 MP with floored sides and centre-cropped to the 16px grid.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    if multiple:
        width, height = image.size
        if width * height > 1024 * 1024:
            scale = math.sqrt(1024 * 1024 / (width * height))
            width, height = int(width * scale), int(height * scale)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        w, h = width // 16 * 16, height // 16 * 16
        if not w or not h:
            raise ValueError("multi-reference image is smaller than 16px on a side")
        left, top = (width - w) // 2, (height - h) // 2
        return image.crop((left, top, left + w, top + h))
    limit = 4 * 1024 * 1024
    scale = min(
        1, math.sqrt(limit / (image.width * image.height)), 4096 / max(image.size)
    )
    while True:
        fitted = tuple(max(1, round(v * scale)) for v in image.size)
        canvas = tuple(min(4096, max(64, math.ceil(v / 16) * 16)) for v in fitted)
        if canvas[0] * canvas[1] <= limit:
            break
        scale *= 0.995
    if fitted != image.size:
        image = image.resize(fitted, Image.Resampling.LANCZOS)
    left, top = (canvas[0] - image.width) // 2, (canvas[1] - image.height) // 2
    padding = (
        (top, canvas[1] - image.height - top),
        (left, canvas[0] - image.width - left),
        (0, 0),
    )
    # Reflect needs a pad shorter than the side; the reference falls back to
    # symmetric for images 16px or smaller.
    mode = "reflect" if min(image.size) > 16 else "symmetric"
    return Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))


class Flux3Pipeline:
    def __init__(self, root):
        config = json.loads((root / "config.json").read_text())
        self.transformer = Flux3Transformer(config["dit"])
        self.transformer.load_weights(
            list(mx.load(str(root / "model.safetensors")).items()), strict=True
        )
        self.transformer.eval()
        self.vae = Flux3AutoEncoder()
        self.vae.load_weights(
            list(convert_vae_weights(mx.load(str(root / "ae.safetensors"))).items()),
            strict=True,
        )
        self.vae.eval()
        self.text_encoder = TextEncoder(root / "text_encoder")
        mx.eval(
            self.transformer.parameters(),
            self.vae.parameters(),
            self.text_encoder.model.parameters(),
        )

    def generate(self, prompt, references, *, width, height, steps, seed):
        text, text_ids = self.text_encoder.encode(prompt)
        tokens, reference_ids = [], []
        for i, reference in enumerate(references):
            image = prepare_reference(reference, multiple=len(references) > 1)
            values = mx.array(np.asarray(image), dtype=mx.float32) / 127.5 - 1
            t, ids = self.vae.encode(values)
            ids = ids * mx.array([1, 0.5, 0.5, 1])
            ids = ids.at[:, 3].add(-(1 + i))
            tokens.append(t.astype(mx.bfloat16))
            reference_ids.append(ids)
        h, w = height // 16, width // 16
        # Match the reference CPU Torch RNG, including its BF16-rounded initial state.
        # This is a CPU-only optional dependency; all model math executes in MLX.
        import torch

        noise = torch.randn(
            (h * w, 128), generator=torch.Generator(device="cpu").manual_seed(seed)
        )
        noise = mx.array(noise.to(torch.bfloat16).float().numpy())
        data = {
            "image": noise,
            "text": text,
            "image_cond": mx.concatenate(tokens)
            if tokens
            else mx.zeros((0, 128), dtype=mx.bfloat16),
        }
        ids = {
            "image": latent_ids(h, w) * mx.array([1, 0.5, 0.5, 1]),
            "text": text_ids,
            "image_cond": mx.concatenate(reference_ids)
            if reference_ids
            else mx.zeros((0, 4)),
        }
        schedule = timesteps(steps)
        deltas = (schedule[1:] - schedule[:-1]).tolist()
        schedule = schedule.tolist()
        for current, target, delta in zip(schedule, schedule[1:], deltas):
            velocity = self.transformer(data, ids, current, target)
            # The reference rounds the velocity update to BF16 before adding
            # it into the persistent FP32 state.
            data["image"] = data["image"] + (delta * velocity).astype(mx.float32)
            mx.eval(data["image"])
        if not mx.all(mx.isfinite(data["image"])).item():
            raise FloatingPointError("nonfinite FLUX 3 denoising state")
        image = self.vae.decode(data["image"], h, w)
        if not mx.all(mx.isfinite(image)).item():
            raise FloatingPointError("nonfinite FLUX 3 decoded image")
        image = ((mx.clip(image, -1, 1) + 1) * 127.5).astype(mx.uint8)
        mx.eval(image)
        return Image.fromarray(np.array(image))
