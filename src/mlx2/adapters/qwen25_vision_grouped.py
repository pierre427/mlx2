"""Opt-in Qwen2.5-VL vision window grouping candidate.

The pinned vision source dispatches one SDPA per window and materializes the
same boundary tensors in every block. This adapter batches only adjacent,
equal-length windows. Its installation is explicit; the pinned implementation
remains the ordinary reference and default route.
"""

from __future__ import annotations

from contextvars import ContextVar
from importlib import import_module


_MAX_GROUP_WINDOWS = 16
_MAX_GROUP_TOKENS = 4096
_MAX_WINDOWS = 4096
_forward_boundaries: ContextVar[dict | None] = ContextVar(
    "qwen25_vision_forward_boundaries", default=None
)
_counters = {
    "grouped_calls": 0,
    "grouped_windows": 0,
    "boundary_materializations": 0,
    "reference_fallbacks": 0,
}


class _LocalCounters:
    def __init__(self):
        self.values = {name: 0 for name in _counters}


def _bump(name: str, amount: int = 1, local=None) -> None:
    _counters[name] += amount
    if local is not None:
        local.values[name] += amount


def grouped_vision_counters(model=None) -> dict[str, int]:
    """Return cheap path counters; no device synchronization or timing."""
    if model is not None:
        vision = getattr(model, "vision_tower", model)
        local = getattr(vision, "_mlx2_grouped_counters", None)
        if local is not None:
            return dict(local.values)
    return dict(_counters)


def plan_equal_windows(boundaries, token_count: int):
    """Plan bounded adjacent groups, or return None for ineligible geometry.

    Each tuple is (token_start, token_end, window_length, window_count).
    A count of one retains the source's per-window SDPA semantics.
    """
    if not isinstance(token_count, int) or token_count < 1:
        return None
    if not isinstance(boundaries, (list, tuple)) or not 3 <= len(boundaries) <= _MAX_WINDOWS + 1:
        return None
    if any(type(value) is not int for value in boundaries):
        return None
    if boundaries[0] != 0 or boundaries[-1] != token_count:
        return None
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if any(length < 0 for length in lengths):
        return None
    # The pinned vision tower emits repeated cumulative boundaries for padded
    # spatial windows. Source SDPA receives an empty slice at each repetition;
    # removing only those zero-length intervals leaves every token and its
    # nonempty attention window in the original order.
    if any(length == 0 for length in lengths):
        boundaries = [boundaries[0], *(
            end for start, end in zip(boundaries, boundaries[1:]) if end > start
        )]
        lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    groups = []
    index = 0
    while index < len(lengths):
        length = lengths[index]
        count = 1
        while (
            index + count < len(lengths)
            and lengths[index + count] == length
            and count < _MAX_GROUP_WINDOWS
            and (count + 1) * length <= _MAX_GROUP_TOKENS
        ):
            count += 1
        groups.append((boundaries[index], boundaries[index + count], length, count))
        index += count
    return groups if any(group[3] > 1 for group in groups) else None


def _grouped_sdpa(mx, q, k, v, groups, scale, local=None):
    """Apply independent window attention using the batch axis for groups."""
    heads, width = q.shape[1], q.shape[3]
    outputs = []
    for start, end, length, count in groups:
        slices = [tensor[:, :, start:end, :] for tensor in (q, k, v)]
        if count > 1:
            slices = [
                tensor.reshape(heads, count, length, width).transpose(1, 0, 2, 3)
                for tensor in slices
            ]
        output = mx.fast.scaled_dot_product_attention(
            *slices, scale=scale, mask=None
        )
        if count > 1:
            output = output.transpose(1, 0, 2, 3).reshape(1, heads, end - start, width)
            _bump("grouped_windows", count, local)
        outputs.append(output)
    return mx.concatenate(outputs, axis=2)


def install_grouped_vision_attention(model, *, enable: bool = False) -> int:
    """Explicitly install the candidate on a pinned Qwen2.5-VL vision model.

    Returns the number of patched blocks. A disabled call is a no-op. Only
    exact pinned source classes are accepted, and installation rolls back on
    failure. The forward scope owns boundary reuse for one vision call.
    """
    if not enable:
        return 0
    source = import_module("mlx_vlm.models.qwen2_5_vl.vision")
    mx = import_module("mlx.core")
    vision = getattr(model, "vision_tower", model)
    if type(vision) is not source.VisionModel:
        raise TypeError("grouped vision requires the pinned Qwen2.5-VL VisionModel")
    blocks = tuple(vision.blocks)
    if not blocks or any(type(block.attn) is not source.Attention for block in blocks):
        raise TypeError("grouped vision requires unmodified pinned attention blocks")
    local = _LocalCounters()

    class ScopedVision(source.VisionModel):
        __slots__ = ()

        def __call__(self, *args, **kwargs):
            token = _forward_boundaries.set({})
            try:
                return super().__call__(*args, **kwargs)
            finally:
                _forward_boundaries.reset(token)

    class GroupedAttention(source.Attention):
        __slots__ = ()

        def __call__(self, x, cu_seqlens, rotary_pos_emb=None):
            scope = _forward_boundaries.get()
            if scope is None or len(x.shape) != 2:
                _bump("reference_fallbacks", local=local)
                return super().__call__(x, cu_seqlens, rotary_pos_emb)
            key = id(cu_seqlens)
            entry = scope.get(key)
            if entry is None or entry[0] is not cu_seqlens:
                boundaries = cu_seqlens.tolist()
                _bump("boundary_materializations", local=local)
                entry = (cu_seqlens, boundaries)
                scope[key] = entry
            groups = plan_equal_windows(entry[1], x.shape[0])
            if groups is None:
                _bump("reference_fallbacks", local=local)
                return super().__call__(x, cu_seqlens, rotary_pos_emb)

            seq_length = x.shape[0]
            qkv = self.qkv(x).reshape(
                seq_length, 3, self.num_heads, -1
            ).transpose(1, 0, 2, 3)
            q, k, v = mx.split(qkv, 3)
            rotate = source.apply_rotary_pos_emb_vision
            q = rotate(mx.expand_dims(q, 0), rotary_pos_emb)[0].transpose(0, 2, 1, 3)
            k = rotate(mx.expand_dims(k, 0), rotary_pos_emb)[0].transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            output = _grouped_sdpa(mx, q, k, v, groups, self.scale, local)
            _bump("grouped_calls", local=local)
            return self.proj(output.transpose(0, 2, 1, 3).reshape(seq_length, -1))

    changed = []
    changed_vision = False
    try:
        for block in blocks:
            block.attn.__class__ = GroupedAttention
            changed.append(block.attn)
        vision.__class__ = ScopedVision
        changed_vision = True
        vision._mlx2_grouped_counters = local
    except Exception:
        if changed_vision:
            vision.__class__ = source.VisionModel
        for attention in changed:
            attention.__class__ = source.Attention
        raise
    return len(blocks)
