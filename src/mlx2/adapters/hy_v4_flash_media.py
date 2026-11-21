# SPDX-License-Identifier: Apache-2.0
# Ported from transformers PR 50903 ``image_processing_hy_v4_flash.py`` and
# ``processing_hy_v4_flash.py``.  See provenance/hy-v4-flash-vision.json.
"""Hy4-flash image preprocessing and placeholder expansion (no MLX import).

* ``smart_resize``: both sides a multiple of ``patch * merge`` (32), area in
  ``[512*512, 2048*2048]``, aspect ratio kept (``image_processing_*``).
* Resampling is PIL BICUBIC: the reference processor resizes with BICUBIC
  whatever the config says, after the original HunyuanOCR implementation,
  which called ``PIL.Image.resize`` with its RGB default.
* Rescale by 1/255, normalize with the artifact processor's ``image_mean`` /
  ``image_std`` (tencent/Hy4-flash ``processor_config.json``: 0.5 / 0.5); the
  OpenAI CLIP mean/std are only the processor class defaults.
* ``patchify``: patches in row-major grid order, each flattened as
  ``(channel, row, col)``; ``grid_thw`` = ``(1, H/16, W/16)``.
* Each image placeholder becomes ``(H/32) * (W/32 + 1)`` image tokens (one
  newline token per merged row), between the template's start/end tokens.
"""

from __future__ import annotations

import math

import numpy as np

PROCESSOR_REVISION = "transformers-c2ec8505-hy_v4_flash-image-v2-pil-bicubic-config-norm"
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MIN_PIXELS = 512 * 512
MAX_PIXELS = 2048 * 2048


def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Hy4-flash image aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def patchify(image: np.ndarray, patch: int) -> tuple[np.ndarray, int, int]:
    """``[C, H, W]`` -> ``([H/p * W/p, C*p*p], H/p, W/p)``, row-major patches."""
    C, H, W = image.shape
    gh, gw = H // patch, W // patch
    patches = image.reshape(C, gh, patch, gw, patch).transpose(1, 3, 0, 2, 4)
    return patches.reshape(gh * gw, C * patch * patch), gh, gw


def normalize(rgb: np.ndarray, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD) -> np.ndarray:
    """``uint8 [H, W, 3]`` -> normalized ``float32 [3, H, W]``."""
    x = rgb.astype(np.float32) * np.float32(1.0 / 255.0)
    mean = np.asarray(mean, np.float32)
    std = np.asarray(std, np.float32)
    return ((x - mean) / std).transpose(2, 0, 1)


def preprocess_image(image, *, patch_size: int = 16, merge_size: int = 2,
                     min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS,
                     image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD):
    """A PIL RGB image -> ``(pixel_values float32 [n, 3*p*p], (1, gh, gw))``."""
    from PIL import Image

    width, height = image.size
    rh, rw = smart_resize(height, width, factor=patch_size * merge_size,
                          min_pixels=min_pixels, max_pixels=max_pixels)
    resized = image.convert("RGB").resize((rw, rh), resample=Image.BICUBIC)
    pixels, gh, gw = patchify(normalize(np.asarray(resized), image_mean, image_std), patch_size)
    return np.ascontiguousarray(pixels, dtype=np.float32), (1, gh, gw)


def image_token_count(grid, merge_size: int = 2) -> int:
    t, gh, gw = (int(v) for v in grid)
    return t * (gh // merge_size) * (gw // merge_size + 1)


def expand_image_placeholders(ids, counts, *, image_token_id: int, start_token_id: int,
                              end_token_id: int) -> tuple[list[int], int]:
    """Expand each rendered single image placeholder to its token count.

    Returns ``(ids, media_end)`` with ``media_end`` one past the last image
    token.  Refuses unless the rendered prompt has exactly one framed
    placeholder per image.
    """
    ids = [int(t) for t in ids]
    where = [i for i, t in enumerate(ids) if t == image_token_id]
    if len(where) != len(counts):
        raise ValueError("Hy4-flash text must not contain an image placeholder")
    for i in where:
        if i == 0 or i + 1 >= len(ids) or ids[i - 1] != start_token_id or ids[i + 1] != end_token_id:
            raise ValueError("Hy4-flash image placeholder is not framed by its start/end tokens")
    out, cursor, end = [], 0, 0
    for i, count in zip(where, counts):
        out.extend(ids[cursor:i])
        out.extend([image_token_id] * int(count))
        end = len(out)
        cursor = i + 1
    out.extend(ids[cursor:])
    return out, end
