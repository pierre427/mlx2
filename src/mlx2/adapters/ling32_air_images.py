# SPDX-License-Identifier: MIT
# Adapted from mlx-vlm; see provenance/ling32-air-vision.json and .NOTICE.
"""Ling-3.2-air CPU image processor (Qwen2-VL law, BailingMM2ImageProcessor).

smart_resize with factor patch*merge, PIL BICUBIC resize, rescale 1/255 in
float64 then float32, per-channel normalize in float32, temporal repeat to
``temporal_patch_size``, and the Qwen2-VL patch flatten order (merge blocks,
then channel, temporal, patch row, patch column).
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

PROCESSOR_VERSION = 'ling32-air-pil-bicubic-qwen2vl-patches-v1'
PROCESSOR_TYPE = 'BailingMM2ImageProcessor'
RESCALE_FACTOR = 1 / 255
EXPECTED = {
    'min_pixels': 4096,
    'max_pixels': 4194304,
    'patch_size': 16,
    'temporal_patch_size': 2,
    'merge_size': 2,
    'image_mean': [0.48145466, 0.4578275, 0.40821073],
    'image_std': [0.26862954, 0.26130258, 0.27577711],
}
# Keys that, if present, may only hold the processor's default law.
DEFAULT_LAW = {
    'do_resize': True, 'do_rescale': True, 'do_normalize': True, 'do_convert_rgb': True,
    'resample': 3, 'rescale_factor': RESCALE_FACTOR,
}


def _refuse(detail):
    raise ValueError(f'Unsupported Ling-3.2-air image preprocessing law: {detail}')


def policy(model_path):
    path = Path(model_path)
    raw = (path / 'preprocessor_config.json').read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict):
        _refuse('preprocessor_config.json is not a mapping')
    for key, value in EXPECTED.items():
        if config.get(key) != value:
            _refuse(f'{key}={config.get(key)!r}')
    if config.get('image_processor_type') != PROCESSOR_TYPE:
        _refuse(f'image_processor_type={config.get("image_processor_type")!r}')
    if 'size' in config:
        size = config['size']
        if not isinstance(size, dict) or size.get('shortest_edge') != EXPECTED['min_pixels'] \
                or size.get('longest_edge') != EXPECTED['max_pixels']:
            _refuse(f'size={size!r}')
    for key, value in DEFAULT_LAW.items():
        if key in config:
            got = config[key]
            if isinstance(value, bool):
                ok = got is value
            elif key == 'resample':
                ok = not isinstance(got, bool) and got == value
            else:
                ok = not isinstance(got, bool) and math.isclose(got, value, rel_tol=0, abs_tol=1e-12)
            if not ok:
                _refuse(f'{key}={got!r}')
    code = Path(__file__).read_bytes()
    vision = Path(__file__).resolve().parents[1] / 'runtime/models/ling32_vision.py'
    revision = hashlib.sha256(raw + code + vision.read_bytes()
                              + pillow_version.encode() + np.__version__.encode()).hexdigest()
    settings = {**EXPECTED, 'rescale_factor': RESCALE_FACTOR, 'resample': 'bicubic',
                'version': PROCESSOR_VERSION}
    return {'version': PROCESSOR_VERSION, 'revision': revision, 'pillow': pillow_version,
            'numpy': np.__version__, 'settings': settings, **settings}


def smart_resize(height, width, factor=32, min_pixels=4096, max_pixels=4194304):
    """Vendor smart_resize: both sides multiples of factor, area in bounds."""
    if min(height, width) < 1:
        raise ValueError('Ling-3.2-air image has an empty side')
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f'absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}')
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


def _check_settings(settings):
    if not isinstance(settings, dict) or settings.get('version') != PROCESSOR_VERSION:
        _refuse('settings were not produced by this processor policy')
    for key, value in EXPECTED.items():
        if settings.get(key) != value:
            _refuse(f'settings {key}={settings.get(key)!r}')


def process_images(images, settings, *, max_patches=None):
    _check_settings(settings)
    patch, merge = settings['patch_size'], settings['merge_size']
    temporal = settings['temporal_patch_size']
    mean = np.array(settings['image_mean'], dtype=np.float32)[:, None, None]
    std = np.array(settings['image_std'], dtype=np.float32)[:, None, None]
    sized_images = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError('Ling-3.2-air image processing accepts PIL images only')
        size = smart_resize(image.height, image.width, patch * merge,
                            settings['min_pixels'], settings['max_pixels'])
        sized_images.append((image, size))
    if not sized_images:
        raise ValueError('Ling-3.2-air image processing needs at least one image')
    total = sum((height // patch) * (width // patch) for _, (height, width) in sized_images)
    if max_patches is not None and total > max_patches:
        raise ValueError(f'Ling-3.2-air vision mode accepts at most {max_patches} image patches '
                         f'per request (got {total}); send smaller images')

    pixels, grids = [], []
    for image, (height, width) in sized_images:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
        array = np.asarray(image).transpose(2, 0, 1)
        array = (array.astype(np.float64) * settings['rescale_factor']).astype(np.float32)
        array = (array - mean) / std
        gh, gw = height // patch, width // patch
        blocks = array.reshape(3, gh // merge, merge, patch, gw // merge, merge, patch)
        blocks = blocks.transpose(1, 4, 2, 5, 0, 3, 6)
        blocks = np.broadcast_to(blocks[:, :, :, :, :, None],
                                 (*blocks.shape[:5], temporal, *blocks.shape[5:]))
        pixels.append(blocks.reshape(gh * gw, 3 * temporal * patch * patch))
        grids.append([1, gh, gw])
    return np.ascontiguousarray(np.concatenate(pixels), dtype=np.float32), grids
