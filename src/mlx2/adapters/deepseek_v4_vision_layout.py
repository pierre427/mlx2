"""CPU DeepSeek V4 image prefill layout from the pinned official reference.

This builds the checkpoint's exact image sentinel and patch ordering contract.
The ViT, aligner, image visibility routing, and decoder integration are separate
unfinished MLX work; callers must not treat this as image generation support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4
PATCH = 14
DOWNSAMPLE = 3
MAX_IMAGE_TOKENS = 384
MIN_PIXELS = 147456
MAX_WH_RATIO = 8


@dataclass(frozen=True)
class ImagePrefill:
    start: int
    patches: np.ndarray  # [n_vit_h * n_vit_w, 3, 14, 14], F32 [-1, 1]
    n_vit_h: int
    n_vit_w: int
    types: np.ndarray  # N-layout sentinels in final token order
    perm: np.ndarray  # Aligner-row indices for image sentinel positions


def grid_tokens(height: int, width: int) -> tuple[int, int, int]:
    if height <= 0 or width <= 0 or height % PATCH or width % PATCH:
        raise ValueError("image grid must be positive and patch-aligned")
    n_h = math.ceil((height // PATCH) / DOWNSAMPLE)
    n_w = math.ceil((width // PATCH) / DOWNSAMPLE)
    count = n_h * (n_w + 1) + 2
    if n_h % 2:
        count += n_w + 1
    count += (n_h + 1) // 2 * (n_w + 1) % 2 * 2
    return n_h, n_w, count


def _solve_resize_ratio(height: int, width: int, budget: int) -> tuple[int, int, int]:
    ratio = height / width
    max_w_float = math.sqrt((budget - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1:
        max_w = 1
        max_h = (budget - 2) // (max_w + 1)
        if max_h % 2:
            max_h -= 1
        best_width = max_w * PATCH * DOWNSAMPLE
        best_height = max_h * PATCH * DOWNSAMPLE
    elif max_h_float < 2:
        max_h = 2
        max_w = (budget - 2) // max_h - 1
        if max_w <= 1:
            raise ValueError("image token budget cannot fit this aspect ratio")
        best_width = max_w * PATCH * DOWNSAMPLE
        best_height = max_h * PATCH * DOWNSAMPLE
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2:
            max_h -= 1
        beta = min(max_w * PATCH * DOWNSAMPLE / width,
                   max_h * PATCH * DOWNSAMPLE / height)
        best_width = math.floor(width * beta / PATCH) * PATCH
        best_height = math.floor(height * beta / PATCH) * PATCH
    _, _, tokens = grid_tokens(best_height, best_width)
    return best_height, best_width, tokens


def image_size(height: int, width: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    effective_width = min(width, height * MAX_WH_RATIO)
    if effective_width * height < MIN_PIXELS:
        ratio = math.sqrt(MIN_PIXELS / (effective_width * height))
        effective_width = int(effective_width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(effective_width / PATCH) * PATCH
    best_height = math.ceil(height / PATCH) * PATCH
    budget = MAX_IMAGE_TOKENS - (COMPRESS_PAD_TO - 1)
    _, _, count = grid_tokens(best_height, best_width)
    while count > budget:
        best_height, best_width, count = _solve_resize_ratio(height, effective_width, budget)
        budget -= 1
        if budget < 8:
            raise ValueError("unable to fit image into token budget")
    return best_height, best_width


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int) -> tuple[np.ndarray, np.ndarray]:
    if n_llm_h < 1 or n_llm_w < 1 or start_pos < 0:
        raise ValueError("invalid image block dimensions or position")
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    rows = n_llm_h + n_llm_h % 2
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = np.array(([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
                     + [IMAGE_PAD] * (row_len * (n_llm_h % 2)), dtype=np.int64)
    order = np.arange(rows * row_len).reshape(rows // 2, 2, row_len).transpose(0, 2, 1).reshape(-1)
    image_idx = np.full((rows, row_len), -1, dtype=np.int64)
    image_idx[:n_llm_h, :n_llm_w] = np.arange(n_llm_h * n_llm_w).reshape(n_llm_h, n_llm_w)
    perm = image_idx.reshape(-1)[order]
    perm = perm[perm >= 0]
    types = np.concatenate((np.full(compress_pad, IMAGE_PAD, dtype=np.int64),
                            np.array([IMAGE_START], dtype=np.int64), types[order],
                            np.full(pad_last, IMAGE_PAD, dtype=np.int64),
                            np.array([IMAGE_END], dtype=np.int64)))
    assert len(perm) == n_llm_h * n_llm_w
    return types, perm


def prepare_local_image(path: str | Path, *, start_pos: int) -> ImagePrefill:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("image must be a local file")
    with Image.open(path) as source:
        image = source.convert("RGB")
    best_height, best_width = image_size(image.height, image.width)
    if image.width >= MAX_WH_RATIO * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    n_vit_h, n_vit_w = best_height // PATCH, best_width // PATCH
    data = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    data = (data - 0.5) / 0.5
    patches = data.reshape(3, n_vit_h, PATCH, n_vit_w, PATCH).transpose(1, 3, 0, 2, 4)
    patches = patches.reshape(n_vit_h * n_vit_w, 3, PATCH, PATCH)
    n_llm_h, n_llm_w, _ = grid_tokens(best_height, best_width)
    types, perm = build_image_block(n_llm_h, n_llm_w, start_pos)
    if len(types) > MAX_IMAGE_TOKENS:
        raise ValueError("image prefill exceeds checkpoint budget")
    return ImagePrefill(start_pos, patches, n_vit_h, n_vit_w, types, perm)


def expand_image_tokens(prompt_tokens: list[int], image_token_id: int,
                        images: list[str | Path], vocab_size: int) -> tuple[list[int], list[ImagePrefill]]:
    """Replace each placeholder with the exact N-layout sentinel block."""
    if prompt_tokens.count(image_token_id) != len(images):
        raise ValueError("image placeholder count differs from supplied images")
    if vocab_size <= image_token_id or image_token_id < 0:
        raise ValueError("invalid image placeholder token")
    output: list[int] = []
    prepared: list[ImagePrefill] = []
    image_iter = iter(images)
    for token in prompt_tokens:
        if token != image_token_id:
            output.append(token)
            continue
        item = prepare_local_image(next(image_iter), start_pos=len(output))
        output.extend((vocab_size + item.types).tolist())
        prepared.append(item)
    return output, prepared


def merge_image_embeddings(token_embeddings: np.ndarray,
                           prepared: list[ImagePrefill],
                           aligned: list[np.ndarray],
                           sentinels: dict[int, np.ndarray]) -> np.ndarray:
    """Insert aligned ViT outputs before hyper-connection expansion (B1)."""
    hidden = np.asarray(token_embeddings)
    if hidden.ndim != 2 or hidden.shape[1] <= 0 or len(prepared) != len(aligned):
        raise ValueError("invalid vision prefill embedding shape")
    if set(sentinels) != {IMAGE_START, IMAGE_PAD, IMAGE_NEW_LINE, IMAGE_END}:
        raise ValueError("all image sentinel embeddings are required")
    dim = hidden.shape[1]
    for vector in sentinels.values():
        if np.asarray(vector).shape != (dim,):
            raise ValueError("image sentinel width mismatch")
    output = hidden.copy()
    occupied = np.zeros(hidden.shape[0], dtype=bool)
    for image, features in zip(prepared, aligned):
        features = np.asarray(features)
        n_image = int((image.types == IMAGE).sum())
        if features.shape != (n_image, dim) or len(image.perm) != n_image:
            raise ValueError("aligner output shape differs from N-layout image slots")
        end = image.start + len(image.types)
        if image.start < 0 or end > len(output) or occupied[image.start:end].any():
            raise ValueError("image prefill spans overlap or exceed prompt")
        block = np.stack([sentinels.get(int(kind), sentinels[IMAGE_PAD])
                          for kind in image.types]).astype(hidden.dtype)
        block[image.types == IMAGE] = features[image.perm].astype(hidden.dtype)
        output[image.start:end] = block
        occupied[image.start:end] = True
    return output


def image_visible(input_ids: list[int] | np.ndarray, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Official per-token left/right image span visibility for sparse attention."""
    ids = np.asarray(input_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("image visibility expects one token sequence")
    left = np.zeros(len(ids), dtype=np.int32)
    right = np.zeros(len(ids), dtype=np.int32)
    start = None
    for pos, token in enumerate(ids):
        if token == vocab_size + IMAGE_START:
            if start is not None:
                raise ValueError("nested image spans are invalid")
            start = pos
        elif token == vocab_size + IMAGE_END:
            if start is None:
                raise ValueError("image end lacks start")
            positions = np.arange(start, pos + 1)
            left[start:pos + 1] = np.minimum(positions - start, MAX_IMAGE_TOKENS - 1)
            right[start:pos + 1] = np.minimum(pos - positions, MAX_IMAGE_TOKENS)
            start = None
    if start is not None:
        raise ValueError("image start lacks end")
    return left, right


def visible_window_indices(seq_len: int, window_size: int,
                           left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Sparse window indices with image-right visibility, matching reference."""
    if seq_len <= 0 or window_size <= 0 or left.shape != (seq_len,) or right.shape != (seq_len,):
        raise ValueError("invalid image-visible window contract")
    width = min(seq_len, window_size + MAX_IMAGE_TOKENS)
    idx = np.arange(seq_len, dtype=np.int64)
    left_add = np.maximum(left.astype(np.int64) - (window_size - 1), 0)
    starts = np.maximum(idx - (window_size - 1) - left_add, 0)
    matrix = starts[:, None] + np.arange(width, dtype=np.int64)
    matrix = np.where(matrix > (idx + right)[:, None], -1, matrix)
    return matrix.astype(np.int32)
