# SPDX-License-Identifier: Apache-2.0
"""NumPy reference for the GVR-style exact self-sampling QSA selector.

The selection idea (sample the current row, guess several thresholds at once,
verify with one counting pass, gather a bounded candidate set, then select
exactly among the candidates) follows NVIDIA TensorRT-LLM PRs #18702/#19076 and
the "GVR V2: Self-Sampling and Multi-Thresholding for Faster Exact Top-K"
article (Apache-2.0).  No source was copied; see docs/PROVENANCE.md.

This module mirrors the Metal kernel ``_gvr_kernel`` step for step (sample
positions, threshold ranks, the threshold choice rule and the in-kernel
direct8 fallback), so its per-row diagnostics must equal the kernel's.  The
output contract is the existing selector law, implemented independently in
:func:`selector_law`:

* only the first ``valid_count = min(blocks, (q_position + 1) // ratio)``
  columns are eligible;
* the ``min(topk, valid_count)`` largest composite keys
  ``(float_order_key(score) << 32) | block_id`` are selected, so larger block
  IDs win exact score ties and NaN/-0.0 are ordered by their bit patterns;
* selected IDs are emitted in ascending order, followed by the invalid fill
  ``blocks - invalid_count + i``.

It imports no MLX and runs on any host.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PATH_EMPTY = 0
PATH_DENSE = 1
PATH_SAMPLED = 2
PATH_FALLBACK = 3
PATH_NAMES = {
    PATH_EMPTY: "empty",
    PATH_DENSE: "dense",
    PATH_SAMPLED: "sampled",
    PATH_FALLBACK: "fallback_direct8",
}
NUM_THRESHOLDS = 4
FRACTION_SCALE = 256
_LOW_FRACTION = 1.15
_HIGH_CAPACITY_SHARE = 0.85


@dataclass(frozen=True)
class GvrConfig:
    """Compile-time constants shared by the reference and the Metal kernel."""

    topk: int
    ratio: int
    width: int = 1024
    capacity: int = 2048

    def __post_init__(self) -> None:
        if self.width not in (256, 512, 1024):
            raise ValueError(f"GVR width {self.width} is unsupported")
        if self.capacity % self.width or not (
            self.width <= self.capacity <= 4 * self.width
        ):
            raise ValueError(f"GVR capacity {self.capacity} is unsupported")
        if not 1 <= self.topk <= min(self.capacity, 1024):
            raise ValueError(f"GVR top-k {self.topk} is unsupported")
        if self.ratio <= 0:
            raise ValueError("GVR compress ratio must be positive")

    @property
    def samples(self) -> int:
        return self.width

    @property
    def fractions256(self) -> tuple[int, ...]:
        return threshold_fractions256(self.topk, self.capacity)


def threshold_fractions256(topk: int, capacity: int) -> tuple[int, ...]:
    """Target candidate counts per threshold, as ``topk * f / 256``.

    Four geometric targets between 1.15*k and 0.85*capacity.  The verify pass
    picks the highest threshold whose real count reaches k, so a target that
    lands under k costs nothing but a wasted comparison.
    """
    low = _LOW_FRACTION
    high = max(low, _HIGH_CAPACITY_SHARE * capacity / max(1, topk))
    values = []
    for index in range(NUM_THRESHOLDS):
        fraction = low * (high / low) ** (index / (NUM_THRESHOLDS - 1))
        values.append(max(FRACTION_SCALE, round(fraction * FRACTION_SCALE)))
    return tuple(values)


def float_order_keys(scores: np.ndarray) -> np.ndarray:
    """Monotone uint32 keys of FP32 bit patterns (NaN ordered by bits)."""
    bits = np.ascontiguousarray(scores, dtype=np.float32).view(np.uint32)
    negative = (bits & np.uint32(0x80000000)) != 0
    return np.where(negative, ~bits, bits ^ np.uint32(0x80000000)).astype(np.uint32)


def valid_counts(q_positions: np.ndarray, blocks: int, ratio: int) -> np.ndarray:
    """Mirror the kernel's C++ truncating ``(qpos + 1) / ratio`` clamp."""
    qpos = np.asarray(q_positions, dtype=np.int64).reshape(-1)
    complete = np.trunc((qpos + 1) / int(ratio)).astype(np.int64)
    complete = np.where(complete > 0, complete, 0)
    return np.minimum(int(blocks), complete)


def _finish(blocks: int, topk: int, selected_ids: np.ndarray) -> np.ndarray:
    selected = np.sort(np.asarray(selected_ids, dtype=np.int64))
    invalid = topk - selected.size
    fill = np.arange(invalid, dtype=np.int64) + (blocks - invalid)
    return np.concatenate([selected, fill]).astype(np.uint32)


def selector_law(
    scores: np.ndarray, q_positions: np.ndarray, *, topk: int, compress_ratio: int
) -> np.ndarray:
    """The selector law shared by radix_exact, direct8 and direct4."""
    scores = np.asarray(scores, dtype=np.float32)
    rows, blocks = scores.shape
    counts = valid_counts(q_positions, blocks, compress_ratio)
    out = np.empty((rows, topk), dtype=np.uint32)
    for row in range(rows):
        valid = int(counts[row])
        keys = float_order_keys(scores[row, :valid]).astype(np.uint64)
        composite = (keys << np.uint64(32)) | np.arange(valid, dtype=np.uint64)
        take = min(topk, valid)
        chosen = np.argsort(composite)[::-1][:take]
        out[row] = _finish(blocks, topk, chosen)
    return out


def threshold_ranks(selected_valid: int, valid_count: int, config: GvrConfig):
    """Descending sample ranks of the candidate thresholds (kernel integer math)."""
    ranks = []
    for fraction in config.fractions256:
        target = (selected_valid * fraction + FRACTION_SCALE - 1) // FRACTION_SCALE
        rank = (target * config.samples) // valid_count
        ranks.append(min(max(rank, 1), config.samples) - 1)
    return ranks


def sample_positions(valid_count: int, config: GvrConfig) -> np.ndarray:
    lanes = np.arange(config.samples, dtype=np.int64)
    return (lanes * int(valid_count)) // config.samples


def _select_candidates(
    cand_ids: np.ndarray, cand_keys: np.ndarray, selected_valid: int
) -> np.ndarray:
    """Exact selection among ascending-ID candidates (kernel compaction law)."""
    if cand_keys.size == selected_valid:
        return cand_ids
    tf = np.sort(cand_keys)[::-1][selected_valid - 1]
    greater = cand_keys > tf
    tied = np.flatnonzero(cand_keys == tf)
    need = selected_valid - int(np.count_nonzero(greater))
    keep = greater.copy()
    keep[tied[tied.size - need :]] = True
    return cand_ids[keep]


def gvr_reference_row(
    row_scores: np.ndarray, q_position: int, *, blocks: int, config: GvrConfig
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return (block_ids, (path, candidate_count, threshold_index)) for one row."""
    valid = int(valid_counts(np.array([q_position]), blocks, config.ratio)[0])
    selected_valid = min(config.topk, valid)
    if selected_valid == 0:
        return _finish(blocks, config.topk, np.empty(0)), (PATH_EMPTY, 0, 0)
    keys = float_order_keys(row_scores[:valid])
    ids = np.arange(valid, dtype=np.int64)
    if valid <= config.capacity:
        chosen = _select_candidates(ids, keys, selected_valid)
        return _finish(blocks, config.topk, chosen), (PATH_DENSE, valid, 0)

    samples = keys[sample_positions(valid, config)]
    ordered = np.sort(samples)[::-1]
    thresholds = [
        int(ordered[rank]) for rank in threshold_ranks(selected_valid, valid, config)
    ]
    counts = [int(np.count_nonzero(keys >= threshold)) for threshold in thresholds]
    picked = next(
        (index for index, count in enumerate(counts) if count >= selected_valid),
        None,
    )
    if picked is None or counts[picked] > config.capacity:
        composite = (keys.astype(np.uint64) << np.uint64(32)) | ids.astype(np.uint64)
        chosen = np.argsort(composite)[::-1][:selected_valid]
        index = NUM_THRESHOLDS if picked is None else picked
        return _finish(blocks, config.topk, chosen), (PATH_FALLBACK, 0, index)
    mask = keys >= thresholds[picked]
    chosen = _select_candidates(ids[mask], keys[mask], selected_valid)
    return _finish(blocks, config.topk, chosen), (PATH_SAMPLED, counts[picked], picked)


def gvr_reference_select(
    scores: np.ndarray,
    q_positions: np.ndarray,
    *,
    topk: int,
    compress_ratio: int,
    width: int = 1024,
    capacity: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(block_ids [rows, topk], diagnostics [rows, 3])``."""
    config = GvrConfig(int(topk), int(compress_ratio), int(width), int(capacity))
    scores = np.asarray(scores, dtype=np.float32)
    rows, blocks = scores.shape
    positions = np.asarray(q_positions).reshape(-1)
    ids = np.empty((rows, config.topk), dtype=np.uint32)
    diag = np.empty((rows, 3), dtype=np.uint32)
    for row in range(rows):
        ids[row], diag[row] = gvr_reference_row(
            scores[row], int(positions[row]), blocks=blocks, config=config
        )
    return ids, diag
