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


# -- video (pinned processing/image_processing_nemotron_h_omni.py) ----------
# The source's video path: every sampled frame is one tile sized by the
# aspect-preserving ``video_target_num_patches`` rule (1024 patches, pixel
# shuffle 2), resized and normalized exactly as images; frames pair into
# temporal tubelets of ``video_temporal_patch_dim`` (2), the last padded with
# its final frame by the model; each tubelet is one ``<img>...</img>`` chunk
# labelled with its frames' sample times.  EVS token pruning
# (``video_pruning_rate``) is not implemented: every video token is kept, as
# the export validated (conversion_info validation_settings rate 0).
VIDEO_PROCESSOR_REVISION = "super-vl-video-aspect1024-tubelet2-labels-noevs-v1"
VIDEO_TARGET_PATCHES = 1024
VIDEO_TEMPORAL_PATCH = 2
VIDEO_SAMPLE_FPS = 2.0
VIDEO_MAX_FRAMES = 64


def video_grid(width, height, *, target=VIDEO_TARGET_PATCHES, divisor=2):
    """``(h_patches, w_patches)`` of the source's ``_compute_target_patches_video``."""
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Invalid Super VL video frame dimensions")
    aspect = width / max(height, 1)
    ph = max(round(math.sqrt(target / aspect)), 1)
    pw = max(round(math.sqrt(target * aspect)), 1)
    rem_h, rem_w = ph % divisor, pw % divisor
    ph_up = ph + (divisor - rem_h if rem_h else 0)
    pw_up = pw + (divisor - rem_w if rem_w else 0)
    if ph_up * pw_up <= target:
        ph, pw = ph_up, pw_up
    else:
        ph, pw = max(divisor, ph - rem_h), max(divisor, pw - rem_w)
    return ph, pw


def preprocess_video(frames, *, mean, std):
    """Normalized ``[N, 3, H, W]`` float32 frames and tokens per tubelet."""
    import torch

    if not frames:
        raise ValueError("Super VL video has no sampled frames")
    shapes = {tuple(np.asarray(frame).shape) for frame in frames}
    if len(shapes) != 1 or len(next(iter(shapes))) != 3 or next(iter(shapes))[2] != 3:
        raise ValueError("Super VL video frames must share one HWC RGB shape")
    height, width, _ = next(iter(shapes))
    h, w = video_grid(int(width), int(height))
    mean, std = (torch.tensor(value).view(1, 3, 1, 1) for value in (mean, std))
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("Invalid Super VL pixel normalization")
    out = []
    for frame in frames:
        array = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        if tensor.shape[-2:] != (h * 16, w * 16):
            tensor = torch.nn.functional.interpolate(
                tensor, size=(h * 16, w * 16), mode="bicubic",
                align_corners=False, antialias=True,
            )
        out.append(((tensor / 255. - mean) / std).numpy()[0])
    return np.stack(out), h * w // 4


def video_prompt(frames, tokens_per_tubelet, *, source_fps=None, indices=None,
                 temporal=VIDEO_TEMPORAL_PATCH):
    """The source processor's expansion of one ``<video>``: per tubelet,
    ``Frame i sampled at T seconds and frame j ...: <img><image>*n</img>``."""
    groups = (frames + temporal - 1) // temporal
    duration_ms = int(1000.0 / source_fps) if source_fps else None
    chunk = "<img>" + "<image>" * tokens_per_tubelet + "</img>"
    labels = []
    for group in range(groups):
        parts = []
        for offset in range(temporal):
            index = group * temporal + offset
            if index >= frames:
                break
            prefix = "Frame" if offset == 0 else "frame"
            if source_fps and indices is not None and index < len(indices):
                seconds = int(indices[index]) * duration_ms / 1000.0
                parts.append(f"{prefix} {index + 1} sampled at {seconds:.2f} seconds")
            elif source_fps:
                parts.append(f"{prefix} {index + 1} sampled at {index / source_fps:.2f} seconds")
            else:
                parts.append(f"{prefix} {index + 1}")
        labels.append(" and ".join(parts) + ": ")
    return "\n".join(label + chunk for label in labels)
