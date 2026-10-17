"""Pinned Super VL image preprocessing, without executing HF remote code.

The source uses CPU torch antialiased bicubic, not PIL's resize kernel.
Torch is imported only when a request actually contains an image.
Provenance: provenance/nemotron35-super-vl.json.
"""
import math

import numpy as np

PROCESSOR_REVISION = "super-vl-image-torch-bicubic-aa-alignfalse-v1"


def target_grid(width, height, *, budget=13312, minimum=1024):
    if type(width) is not int or type(height) is not int or min(width, height) < 3:
        raise ValueError("Invalid Super VL image dimensions")
    closest_h, closest_w = round(height / 16 + .5), round(width / 16 + .5)
    factor = min(math.sqrt(budget / (closest_h * closest_w)), 1.)
    h, w = math.floor(factor * closest_h), math.floor(factor * closest_w)
    # Extreme aspect ratios can round one dimension to zero in the source.
    # Fail at the boundary instead of dividing by zero or changing its grid.
    if min(h, w) < 1:
        raise ValueError("Super VL image aspect ratio cannot fit its patch budget")
    if budget > minimum and h * w < minimum:
        up = math.sqrt(minimum / (h * w))
        h, w = math.ceil(up * h), math.ceil(up * w)
    if h % 2:
        h = h + 1 if (h + 1) * w <= budget else max(2, h - 1)
    if w % 2:
        w = w + 1 if h * (w + 1) <= budget else max(2, w - 1)
    if min(h, w) < 2 or h * w > budget:
        raise ValueError("Super VL image grid exceeds its patch budget")
    return h, w


def preprocess_image(image, *, mean, std):
    import torch
    h, w = target_grid(*image.size)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    if tensor.shape[-2:] != (h * 16, w * 16):
        tensor = torch.nn.functional.interpolate(tensor, size=(h * 16, w * 16),
                                                mode="bicubic", align_corners=False, antialias=True)
    mean, std = (torch.tensor(value).view(1, 3, 1, 1) for value in (mean, std))
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("Invalid Super VL pixel normalization")
    pixels = ((tensor / 255. - mean) / std).numpy()
    return pixels, h * w // 4
