# SPDX-License-Identifier: MIT
"""Explicit ordinary-B1 prefill law for exact Source prefix reuse.

The outer scheduler may cut anywhere, including an APC interior checkpoint.
Every token still uses the ordinary one-row program.  This deliberately trades
prefill throughput for a partition-independent numerical profile; it does not
change expert memory admission or qualify a serving route.

``position_grid_v1`` is the multi-row alternative: forwards are cut at
absolute multiples of a fixed grid, the live row cap is floored to whole
cells, and only grid-aligned state whose every forward started on the grid is
published. Off-grid state (decode, a cap below one cell, an off-grid cut)
still runs with ordinary semantics but is tainted and never donated.
"""
import hashlib
from pathlib import Path

POLICY = 'canonical_m1_v1'
ADAPTIVE = 'adaptive_v1'
# Absolute-position grid law: every multi-row forward is cut at multiples of
# ``grid`` tokens from sequence start, so a grid-aligned prefix state depends
# only on its tokens, never on the outer chunking or the live row cap.
GRID = 'position_grid_v1'
GRID_ROWS = (128, 256, 512, 1024)
GRID_DEFAULT_ROWS = 1024
POLICIES = (ADAPTIVE, POLICY, GRID)


def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def partition_receipt(policy, grid_rows=None):
    if policy not in POLICIES:
        raise ValueError('Unsupported Source prefill partition policy')
    canonical = policy == POLICY
    if policy == GRID:
        grid = _grid(grid_rows)
        return dict(policy=policy, arithmetic_rows=f'absolute grid cells of {grid}',
                    grid_rows=grid, outer_chunk_changes_numerics=False,
                    row_cap_changes_numerics='only below the grid; such state is never published',
                    final_token_law='ordinary_B1', adaptive_expert_residency=True,
                    apcv2_reuse_enabled=True,
                    apcv2_publication=f'grid-aligned untainted prefill states only (multiples of {grid})',
                    qualification='unqualified',
                    prefill_performance='adaptive chunks floored to grid multiples')
    return dict(policy=policy, arithmetic_rows=1 if canonical else 'adaptive',
                outer_chunk_changes_numerics=not canonical,
                final_token_law='ordinary_B1', adaptive_expert_residency=True,
                apcv2_reuse_enabled=canonical, qualification='unqualified',
                prefill_performance='ordinary B1 per token' if canonical else 'adaptive chunks')


def _grid(grid):
    """Grid sizes are powers of two within the ordinary 1..1024 row bound."""
    if type(grid) is not int or not 1 <= grid <= 1024 or grid & (grid - 1):
        raise ValueError('Source prefill grid rows must be a power of two in 1..1024')
    return grid


def grid_pieces(offset, rows, grid):
    """Split ``rows`` tokens starting at absolute ``offset`` at grid multiples.

    Returns ``(first_row, length)`` pairs relative to the call. Pieces never
    straddle a multiple of ``grid``; a call that starts on the grid and spans
    whole cells therefore computes each cell exactly as any other call would.
    """
    grid = _grid(grid)
    if type(offset) is not int or offset < 0 or type(rows) is not int or rows < 1:
        raise ValueError('Source grid pieces require nonnegative offset and positive rows')
    pieces, first = [], 0
    while first < rows:
        position = offset + first
        length = min(rows - first, grid - position % grid)
        pieces.append((first, length))
        first += length
    return tuple(pieces)


def aligned_rows(rows, grid):
    """Floor a live row cap to whole grid cells; below one cell it is kept.

    This composes over whatever cap is in force (memory advice, depth bounds,
    the configured step). A cap below one cell still runs, but the resulting
    state is tainted for publication instead of silently changing a law.
    """
    grid = _grid(grid)
    if type(rows) is not int or rows < 1:
        raise ValueError('Source grid row cap must be positive')
    return rows if rows < grid else rows - rows % grid


def grid_row_cap(recommend, grid):
    """Wrap a route's ``recommended_prefill_rows`` without widening it."""
    grid = _grid(grid)
    if not callable(recommend):
        raise ValueError('Source grid row cap requires a callable recommendation')

    def recommended(maximum, *, context_tokens=0):
        return aligned_rows(recommend(maximum, context_tokens=context_tokens), grid)

    recommended.source_prefill_grid_rows = grid
    return recommended


def aligned_checkpoint(prompt_tokens, grid, *, cached_tokens=0, cuts=()):
    """The deepest publishable grid boundary of a cold prefill, or None.

    The prompt boundary ``P - 1`` is published only when it is itself on the
    grid. Otherwise this is the largest grid multiple not beyond ``P - 1`` nor
    beyond the first off-grid cut (which taints every later state). It is a
    pure function of the prompt length and the cold plan's cuts, so cold and
    warm prefills of one prompt agree on it.
    """
    grid = _grid(grid)
    if type(prompt_tokens) is not int or type(cached_tokens) is not int or cached_tokens < 0:
        raise ValueError('Source grid checkpoint requires integer token counts')
    boundary = prompt_tokens - 1
    bound = min([boundary, *(int(c) for c in cuts if int(c) % grid and 0 < int(c) < boundary)])
    position = bound - bound % grid
    if position == boundary or not cached_tokens < position < boundary or position <= 0:
        return None
    return position


def grid_publication_refusal(rows, grid=None):
    """Host-only exactness gate for state produced under the grid law.

    ``grid`` is the route's bound grid; rows without provenance (an APCv2
    restore that has not run a forward) inherit it, because only untainted
    grid-aligned state is ever published under that identity. Without a
    route grid, rows that carry no provenance are outside this law.
    """
    rows = list(rows)
    if not rows:
        return 'Source grid partition state is empty'
    grids = {getattr(row, 'source_partition_grid', None) or grid for row in rows}
    if grids == {None}:
        return None
    if len(grids) != 1 or None in grids or (grid is not None and grids != {grid}):
        return 'Source grid partition provenance is incomplete or foreign'
    grid = _grid(grids.pop())
    if any(getattr(row, 'source_partition_tainted', False) for row in rows):
        return 'Source grid partition state followed an off-grid forward'
    offsets = {getattr(row, 'offset', None) for row in rows}
    if len(offsets) != 1:
        return 'Source grid partition offsets disagree'
    offset = offsets.pop()
    if type(offset) is not int or offset % grid:
        return 'Source grid partition state is not on a grid boundary'
    return None


def canonical_forward(model, forward, inputs, cache=None, *, logits_to_keep=None,
                      capture_layers=(), body_only=False):
    """Run the original B1 forward in token order, retaining original ownership.

    Validate the complete input before the first mutation. Each ordinary call
    owns its existing completion/memory/bank scope. Failure leaves its existing
    incomplete-state guard in force; there is no approximate publication or
    silent switch back to a multi-row arithmetic law.
    """
    import mlx.core as mx
    from ..runtime.prefill_checkpoint_rows import select_checkpoint_rows
    if (model.identity != model._source_canonical_identity
            or getattr(model.weights, 'source_prefill_partition_policy', None) != POLICY):
        raise ValueError('Source canonical prefill owner/profile changed')
    manager = getattr(model, '_source_ast_bank_manager', None)
    if manager is not None:
        manager.before_ordinary()
    cache, _ = model.validate_forward(inputs, cache, logits_to_keep)
    rows = inputs.shape[1]
    if rows == 1:
        return forward(inputs, cache, logits_to_keep=logits_to_keep,
                       capture_layers=capture_layers, body_only=body_only)
    logits, features = [], []
    keep = rows if logits_to_keep in (None, 0) else min(rows, logits_to_keep)
    try:
        for row in range(rows):
            wanted = row >= rows - keep
            out, taps = forward(inputs[:, row:row + 1], cache, logits_to_keep=0,
                                capture_layers=capture_layers,
                                body_only=body_only or not wanted)
            if out is not None:
                logits.append(out)
            if taps is not None:
                features.append(taps)
    except BaseException:
        # A partially completed outer call must never become an APC entry,
        # even if the failing B1 call failed before setting its own marker.
        for state in cache:
            state._in_forward = True
        raise
    result = mx.concatenate(logits, axis=1) if logits else None
    # The existing prompt guard may request just its last row. Apply that
    # request after serial B1 heads, preserving each head's ordinary row law.
    if result is not None and logits_to_keep is None:
        result = select_checkpoint_rows(result)
    stats = model.scheduled_kv_calls
    stats['source_canonical_m1_prefill_tokens'] = stats.get('source_canonical_m1_prefill_tokens', 0) + rows
    stats['source_canonical_m1_prefill_outer_calls'] = stats.get('source_canonical_m1_prefill_outer_calls', 0) + 1
    return result, mx.concatenate(features, axis=1) if features else None


def canonical_model_class():
    """Keep the model's own layer/self references; no proxy or global patch."""
    from ..runtime.models.deepseek_v41 import DeepSeekV41Model

    class CanonicalM1SourceModel(DeepSeekV41Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._source_canonical_identity = self.identity

        def _forward(self, inputs, cache=None, *, logits_to_keep=None,
                     capture_layers=(), body_only=False):
            return canonical_forward(self, super()._forward, inputs, cache,
                logits_to_keep=logits_to_keep, capture_layers=capture_layers, body_only=body_only)

    return CanonicalM1SourceModel


def _stamp(cache, grid, tainted):
    for state in cache:
        state.source_partition_grid = grid
        state.source_partition_tainted = bool(tainted)


def grid_forward(model, forward, inputs, cache=None, *, logits_to_keep=None,
                 capture_layers=(), body_only=False):
    """Run the ordinary forward once per absolute grid cell, in token order.

    A call that starts on the grid computes exactly the cells any other call
    covering them computes, so grid-boundary state is partition invariant.
    A call that starts off the grid (decode, an off-grid cut, a row cap below
    one cell) still runs with ordinary semantics, but taints the state so it
    can never donate an APCv2 prefix. Validation precedes any mutation; a
    failure after the first piece poisons the outer call like canonical M1.
    """
    import mlx.core as mx
    from ..runtime.prefill_checkpoint_rows import _REQUEST
    grid = getattr(model.weights, 'source_prefill_grid_rows', None)
    if (model.identity != getattr(model, '_source_canonical_identity', None)
            or getattr(model.weights, 'source_prefill_partition_policy', None) != GRID):
        raise ValueError('Source grid prefill owner/profile changed')
    grid = _grid(grid)
    manager = getattr(model, '_source_ast_bank_manager', None)
    if manager is not None:
        manager.before_ordinary()
    cache, _ = model.validate_forward(inputs, cache, logits_to_keep)
    rows, offset = inputs.shape[1], cache[0].offset
    tainted = (any(getattr(state, 'source_partition_tainted', False) for state in cache)
               or offset % grid != 0)
    stats = model.scheduled_kv_calls
    if tainted:
        stats['source_grid_tainted_calls'] = stats.get('source_grid_tainted_calls', 0) + 1
    pieces = grid_pieces(offset, rows, grid)
    if len(pieces) == 1:
        try:
            result = forward(inputs, cache, logits_to_keep=logits_to_keep,
                             capture_layers=capture_layers, body_only=body_only)
        except BaseException:
            # The ordinary forward retains its own incomplete-state guard.
            _stamp(cache, grid, True)
            raise
        _stamp(cache, grid, tainted)
        return result
    keep = rows if logits_to_keep in (None, 0) else min(rows, logits_to_keep)
    request = _REQUEST.get()
    # The prompt-chunk guard only needs this call's last row. Narrow the head
    # to it here; per-piece widths never match the request's width.
    narrowed = (logits_to_keep is None and request is not None and not request.applied
                and len(request.rows) == 1 and request.width == rows
                and request.rows[0] == rows - 1)
    if narrowed:
        keep = 1
    logits, features = [], []
    try:
        for first, length in pieces:
            wanted = max(0, min(length, first + length - (rows - keep)))
            out, taps = forward(inputs[:, first:first + length], cache,
                                logits_to_keep=wanted if wanted else 0,
                                capture_layers=capture_layers,
                                body_only=body_only or not wanted)
            if out is not None and wanted:
                logits.append(out)
            if taps is not None:
                features.append(taps)
    except BaseException:
        for state in cache:
            state._in_forward = True
        _stamp(cache, grid, True)
        raise
    _stamp(cache, grid, tainted)
    result = mx.concatenate(logits, axis=1) if logits else None
    if narrowed and result is not None:
        request.applied += 1
    elif result is not None and logits_to_keep is None:
        from ..runtime.prefill_checkpoint_rows import select_checkpoint_rows
        result = select_checkpoint_rows(result)
    stats['source_grid_split_forwards'] = stats.get('source_grid_split_forwards', 0) + len(pieces)
    stats['source_grid_outer_calls'] = stats.get('source_grid_outer_calls', 0) + 1
    return result, mx.concatenate(features, axis=1) if features else None


def grid_model_class():
    """Ordinary Source model whose multi-row forwards follow the grid law."""
    from ..runtime.models.deepseek_v41 import DeepSeekV41Model

    class PositionGridSourceModel(DeepSeekV41Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._source_canonical_identity = self.identity

        def make_cache(self):
            cache = super().make_cache()
            _stamp(cache, _grid(getattr(self.weights, 'source_prefill_grid_rows', None)), False)
            return cache

        def _forward(self, inputs, cache=None, *, logits_to_keep=None,
                     capture_layers=(), body_only=False):
            return grid_forward(self, super()._forward, inputs, cache,
                logits_to_keep=logits_to_keep, capture_layers=capture_layers, body_only=body_only)

    return PositionGridSourceModel
