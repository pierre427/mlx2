# SPDX-License-Identifier: MIT
# Adapted from mlx-vlm; see provenance/muse-vision-serving.json and .NOTICE.
"""Muse's bounded CPU image processor and ordered request/media identity."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

PROCESSOR_VERSION = 'muse-pil-lanczos-temporal-channel-patches-v1'
IMAGE_MARKER = '<|patch|>'


def policy(model_path):
    path = Path(model_path)
    raw = (path / 'processor_config.json').read_bytes()
    config = json.loads(raw).get('image_processor', {})
    expected = {'patch_size': 14, 'temporal_patch_size': 2, 'merge_size': 2,
                'max_image_tokens': 4096, 'image_mean': [.5, .5, .5],
                'image_std': [.5, .5, .5], 'rescale_factor': 1 / 255}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Unsupported Muse image preprocessing law')
    code = Path(__file__).read_bytes()
    vision = Path(__file__).resolve().parents[1] / 'runtime/models/muse_vision.py'
    multimodal = vision.with_name('muse_multimodal.py')
    revision = hashlib.sha256(raw + code + vision.read_bytes() + multimodal.read_bytes()
                              + pillow_version.encode() + np.__version__.encode()).hexdigest()
    return {'version': PROCESSOR_VERSION, 'revision': revision,
            'pillow': pillow_version, 'numpy': np.__version__, **expected}


def smart_resize(height, width, patch_size, max_tokens):
    ideal_h, ideal_w = height / patch_size, width / patch_size
    ratio = ideal_w / ideal_h
    if ideal_h * ideal_w > max_tokens:
        ideal_h = math.sqrt(max_tokens / ratio)
        ideal_w = ideal_h * ratio
    candidates = set(itertools.product((math.floor(ideal_h), math.ceil(ideal_h)),
                                       (math.floor(ideal_w), math.ceil(ideal_w))))
    candidates = [(h, w) for h, w in candidates if h >= 1 and w >= 1 and h * w <= max_tokens]
    if not candidates:
        candidates = [(max(1, round(ideal_h)), max(1, round(ideal_w)))]
    h, w = min(candidates, key=lambda pair: abs(pair[0] / pair[1] - height / width))
    if h * w > max_tokens:
        # The reference's fallback rounds a sub-patch side up to one; for
        # extreme aspect ratios that can exceed its advertised token cap.
        # Refuse that unsupported geometry before resize or allocation.
        raise ValueError('Muse image aspect ratio cannot fit the processor token bound')
    return h * patch_size, w * patch_size


def process_images(images, settings):
    pixels, grids = [], []
    patch, merge = settings['patch_size'], settings['merge_size']
    for image in images:
        height, width = smart_resize(image.height, image.width, patch * merge, settings['max_image_tokens'])
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        array = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
        array *= settings['rescale_factor']
        array = (array - np.array(settings['image_mean'], np.float32)[:, None, None]) / np.array(settings['image_std'], np.float32)[:, None, None]
        gh, gw = height // patch, width // patch
        patches = array.reshape(3, gh, patch, gw, patch).transpose(1, 3, 0, 2, 4)
        patches = np.repeat(patches[:, :, None], settings['temporal_patch_size'], axis=2)
        pixels.append(patches.reshape(gh * gw, settings['temporal_patch_size'] * 3 * patch * patch))
        grids.append([1, gh, gw])
    return np.concatenate(pixels), grids


def prepare(adapter, request, *, file_loader=None):
    from ..multimodal import resolve_media, media_fingerprint
    from .mlx_vlm import _plain_messages, _require_media_markers, _source
    from .muse_glimmer import render_prompt_text

    media = []
    for message in request['messages']:
        content = message.get('content')
        for part in content if isinstance(content, list) else ():
            kind = part.get('type')
            if kind == 'text':
                continue
            if kind not in {'image_url', 'input_image'}:
                raise ValueError('Muse image serving accepts still images only')
            media.append(resolve_media(_source(part, 'image'), kind='image', file_loader=file_loader))
    if not media:
        return request
    if len(media) > 8:
        raise ValueError('Muse image serving accepts at most eight images per request')
    messages = _plain_messages(request['messages'], [IMAGE_MARKER] * len(media))
    prompt = render_prompt_text(adapter.tokenizer, {**request, 'messages': messages})
    _require_media_markers(prompt, 'Muse', ((IMAGE_MARKER, len(media)),))
    pixels, grids = process_images([m.value for m in media], adapter.image_policy)
    # Split exactly once per original marker: expanded patch tokens are never
    # mistaken for the next image. Preserve all message and image ordering.
    parts = prompt.split(IMAGE_MARKER)
    expanded = parts[0]
    for grid, suffix in zip(grids, parts[1:]):
        count = math.prod(grid) // adapter.image_policy['merge_size'] ** 2
        expanded += '<|image_start|>' + IMAGE_MARKER * count + '<|image_end|>' + suffix
    ids = adapter.tokenizer.encode(expanded, add_special_tokens=False)
    token = adapter._config['image_token_id']
    end = max(i + 1 for i, value in enumerate(ids) if value == token)
    identity = media_fingerprint(media, policy=adapter.image_policy)
    return {**request, 'messages': messages, '_mlx2_prompt_tokens': ids,
            '_mlx2_prefill_inputs': {'pixel_values': pixels, 'image_grid_thw': grids},
            '_mlx2_media_token_end': end, '_mlx2_media_fingerprint': identity}
