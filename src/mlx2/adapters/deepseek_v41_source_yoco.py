# SPDX-License-Identifier: MIT
"""APPROXIMATE YOCO-style long-prefill decoder skip (``yoco_v1``), opt-in.

What the skipped layers normally produce
----------------------------------------
DS4.1 Source has 40 layers. Layers 0-1 are uncompressed, 2-19 are ratio-2
compressed consumers, 20-39 are ratio-1 consumers. Global KV pools (and their
index keys) are published only by the KV-source layers 2, 8, 14 and 20; the
ratio-1 decoder layers 21-39 *consume* layer 20's pool. Layer 20 is also the
candidate source (per-query block candidates); layers 24/28/32/36 are index
sources that only compute transient per-query top-k selections over that
complete pool. Engram lives in layers 1 and 14. Therefore, for a prompt of T
rows, an exact prefill leaves exactly two kinds of decoder (21-39) output:

* persistent: each decoder layer's local sliding-window KV, i.e. the
  FP8-rounded, rotated latent of the last ``sliding_window`` (128) positions
  at that layer -- the only cache state a ratio-1 non-source layer owns;
* transient: decoder hidden rows for all T positions, which matter only
  where they feed a later layer's local window or the final logits.

The exact dependency cone of those last 128 local rows grows by W-1 = 127
rows per decoder layer (``runtime/models/deepseek_v41_ced.py``), about 2.5K
rows over 19 layers.

What this approximation leaves in their place
---------------------------------------------
For a prompt wave of at least ``min_rows`` tokens, layers ``0..first-1``
(default 0-20) run over every row in ordinary chunk-major order, so every
global pool, index key, Engram row, ratio-2 partial group and the Engram hash
tail are produced exactly as by ordinary prefill. Layers ``first..last``
(default 21-39, which must end at the final layer) run only over the last
``keep_rows`` (default 140, >= the 128-row window) positions. Their weights
are not read for any earlier chunk. Each decoder layer's local KV is reset at
the tail start, so a tail row within W-1 of the cut attends a truncated local
window (keys before the cut are absent, not zero-filled); its global
(layer-20 pool) attention is complete. The published per-layer local KV (the
final 128 rows) is therefore computed from those approximate tail hidden
states. Nothing else differs. Decode, verification, all-row logits and
images always run full depth.

This reimplements the idea of streaming-llama-stack's ``yoco_keep`` (credited
there to DeepSeek's DeepInfer); no code is copied. The mechanism reuses this
tree's exact CED dependency-trim primitive (``trim_shared`` and the
local-KV reset of ``_forward_impl``) with a fixed tail instead of the cone.

Execution: under this policy the route admits one prefill call per uncached
span (``recommended_prefill_rows`` returns the caller maximum); the model
splits it into ordinary <= 1024-row sub-chunks, aligning the final sub-chunk
to hold the whole tail. Calls below ``min_rows`` (or otherwise not admitted)
run the ordinary exact chunk sequence. Isolation, identity salt and APCv2
refusal: see :mod:`deepseek_v41_source_approx`.
"""
import hashlib
import json
from contextlib import nullcontext

import numpy as np

from .deepseek_v41_source_approx import (
    YOCO_KEEP,
    YOCO_LAYERS,
    YOCO_MIN_ROWS,
    stamp_cache,
)

CHUNK_ROWS = 1024
REVISION = 'source_yoco_v1'


def plan_layers(args, layers, keep):
    """Config-bound admission of the skipped range; returns (first, last)."""
    first, last = layers
    total = args.num_hidden_layers
    if last != total - 1:
        raise ValueError('YOCO skipped range must end at the final layer: rows outside the tail have no later hidden state')
    if not 1 <= first <= last:
        raise ValueError('YOCO skipped range must leave at least one full-depth layer')
    if args.kv_source_layer_ids and first <= max(args.kv_source_layer_ids):
        raise ValueError('YOCO skipped range contains a global KV pool owner')
    if args.candidate_source_layer_id >= 0 and first <= args.candidate_source_layer_id:
        raise ValueError('YOCO skipped range contains the candidate source')
    if args.engram_layer_ids and first <= max(args.engram_layer_ids):
        raise ValueError('YOCO skipped range contains an Engram layer')
    if keep < args.sliding_window:
        raise ValueError('YOCO keep rows must cover the decoder sliding window')
    return first, last


def configure_yoco(model, policy):
    first, last = plan_layers(model.args, policy[YOCO_LAYERS], policy[YOCO_KEEP])
    model.source_prefill_approx_plan = dict(first=first, last=last, keep=policy[YOCO_KEEP],
                                            min_rows=policy[YOCO_MIN_ROWS], revision=REVISION)
    model.source_prefill_approx_last = None
    base = model.recommended_prefill_rows

    def advice(maximum, *, context_tokens=0):
        # One call per uncached span; sub-chunks keep live chunk advice.
        if type(maximum) is int and maximum > CHUNK_ROWS:
            return maximum
        return base(maximum, context_tokens=context_tokens)

    advice.source_prefill_approx_base = base
    model.recommended_prefill_rows = advice


def partition(rows, chunk, keep):
    """Ordinary sub-chunks; the final one holds the whole retained tail."""
    if not 1 <= chunk <= CHUNK_ROWS or not 1 <= keep <= CHUNK_ROWS or rows <= keep:
        raise ValueError('YOCO partition requires rows > keep and chunk/keep within one ordinary chunk')
    remainder = rows % chunk or chunk
    final = remainder if remainder >= keep else min(rows, keep)
    bounds = [(start, min(rows - final, start + chunk)) for start in range(0, rows - final, chunk)]
    bounds.append((rows - final, rows))
    return bounds, final - keep


def _chunk_rows(model, context):
    advice = getattr(model.recommended_prefill_rows, 'source_prefill_approx_base', None)
    advice = advice or getattr(model.weights, 'recommended_prefill_rows', None)
    chunk = CHUNK_ROWS if not callable(advice) else advice(CHUNK_ROWS, context_tokens=context)
    if type(chunk) is not int or chunk < 1:
        raise ValueError('Source chunk advice must be a positive integer')
    return min(chunk, CHUNK_ROWS)


def decision(model, inputs, cache, *, logits_to_keep=None, capture_layers=(), body_only=False, options=None):
    """Return None when the call is admitted, else the refusal reason."""
    plan = getattr(model, 'source_prefill_approx_plan', None)
    weights = model.weights
    if plan is None:
        return 'not_configured'
    if capture_layers or body_only or any(v not in (None, False) for v in (options or {}).values()):
        return 'taps_body_or_extra_forward_options_require_full_depth'
    if logits_to_keep not in (None, 1):
        return 'all_row_logits_require_full_depth'
    if getattr(inputs, 'ndim', None) != 2 or inputs.shape[0] != 1:
        return 'B1_prompt_required'
    if inputs.shape[1] < plan['min_rows']:
        return 'prompt_below_minimum_rows'
    if (getattr(model, '_mlx2_ds41_ast_active', None) is not None or getattr(model, '_mlx2_ds41_ast_poisoned', False)
            or getattr(weights, '_source_current_frame', None) is not None
            or getattr(weights, 'source_resident_bank_session', None) is not None
            or callable(getattr(weights, 'evaluate_block', None))
            or getattr(weights, 'source_row_packet_policy', 'reference') != 'reference'
            or getattr(weights, 'source_frame_reuse_policy', 'reference') != 'reference'
            or getattr(model, 'kv_storage_policy', 'reference') != 'reference'
            or getattr(model, 'ced_policy', 'reference') != 'reference'):
        return 'owner_not_idle_ordinary_reference_cache'
    return None


def _prefill_contexts(weights, rows):
    from contextlib import ExitStack

    from ..runtime.gguf.prefill_scope import dense_prefill_context
    from ..runtime.gguf.segmented_prefill import expert_prefill_context
    stack = ExitStack()
    stack.enter_context(dense_prefill_context(getattr(weights, 'dense_prefill_policy', 'reference'), tokens=rows))
    stack.enter_context(expert_prefill_context(getattr(weights, 'expert_prefill_policy', 'reference'), tokens=rows))
    return stack


def _final_head(model, hidden, wave_rows, cache, shared=()):
    from ..runtime.prefill_checkpoint_rows import (
        select_checkpoint_rows_from_final_suffix,
    )
    with model._weight_scope():
        hidden = select_checkpoint_rows_from_final_suffix(hidden, full_width=wave_rows)
        logits = model.linear(hidden, 'output.weight', rounded=False)
        model._weight_eval(logits, hidden, [v.state for v in cache], shared)
    return logits


def _exact_chunks(model, ordinary, inputs, cache):
    """Ordinary exact chunk sequence for a wide call that YOCO did not admit."""
    rows = inputs.shape[1]
    base = 0 if cache is None else int(cache[0].offset)
    chunk = _chunk_rows(model, base + rows)
    hidden = None
    for start in range(0, rows, chunk):
        end = min(rows, start + chunk)
        with _prefill_contexts(model.weights, end - start):
            hidden, _ = ordinary(inputs[:, start:end], cache, logits_to_keep=1, head_hidden_only=True)
    return _final_head(model, hidden, rows, cache), None


def _partial_forward(model, inputs, cache, *, first_skipped, decoder_first):
    """Layers < first_skipped on every row; later layers on rows >= decoder_first.

    ``decoder_first is None`` skips decoder layers entirely (no weight reads).
    Mirrors ``DeepSeekV41Model._forward_impl`` for the admitted ordinary
    text configuration; unsupported executors were refused by ``decision``.
    """
    import mlx.core as mx

    from ..runtime.models.deepseek_v41_ced import trim_shared
    from ..runtime.models.deepseek_v41_frame_geometry import frame_geometry
    c, weights = model.args, model.weights
    rows = inputs.shape[1]
    ids = np.asarray(inputs)
    base = int(cache[0].offset)
    stream = getattr(weights, 'expert_stream', None)
    if stream is not None and stream.atlas is not None:
        from ..runtime.expert_stream_atlas import ExpertStreamAtlas
        stream.begin_context(ExpertStreamAtlas.keys(model.hasher.token_map[ids[0]],
                                                    np.asarray(cache[0].hash_tail)[::-1]))
    hashes = tail = None
    if model.hasher is not None:
        hashes, tail = model.hasher(ids[0], np.asarray(cache[0].hash_tail))
        for index, layer in enumerate(c.engram_layer_ids):
            model.engram_tables[layer].prefetch(hashes[:, index])
    with model._weight_scope():
        x = model.round(weights.embedding(inputs.astype(mx.uint32)))
        h = mx.broadcast_to(x[:, :, None], (1, rows, c.hc_mult, c.hidden_size))
        pre = mx.broadcast_to(mx.array([1.0] + [0.0] * (c.hc_mult - 1)), h.shape[:-1])
        model._weight_eval(h, pre)
    shared = {}
    if frame_geometry(weights, rows, model.scheduled_kv_calls) is not None:
        raise ValueError('YOCO does not compose with Source frame geometry')
    gpu = mx.default_device() == mx.gpu
    for i, (block, state) in enumerate(zip(model.layers, cache)):
        if i >= first_skipped:
            if decoder_first is None:
                break
            if i == first_skipped and decoder_first:
                h, pre = h[:, decoder_first:], pre[:, decoder_first:]
                trim_shared(shared, decoder_first)
            # Decoder local KV restarts at the tail; earlier keys are absent.
            state.keys = None
            state.offset = base + decoder_first
        with model._weight_scope():
            if i in c.engram_layer_ids:
                h = model.engram(h, i, hashes[:, c.engram_layer_ids.index(i)])
            h, pre = block(h, pre, state, cache, shared)
            if model.eval_interval and gpu:
                mx.async_eval(h, pre)
            model._weight_eval(h, pre, [v.state for v in cache], shared)
    if tail is not None:
        cache[0].hash_tail = mx.array(tail, mx.int32)
    if stream is not None:
        stream.end_context()
    return h, pre, shared


def _sub_forward(model, inputs, cache, **kwargs):
    weights = model.weights
    rows = inputs.shape[1]
    rows_execution = getattr(weights, 'execution_scope_for_rows', None)
    execution = getattr(weights, 'execution_scope', None)
    scope = (rows_execution(rows) if callable(rows_execution)
             else execution() if execution is not None else nullcontext())
    try:
        with _prefill_contexts(weights, rows), scope:
            return _partial_forward(model, inputs, cache, **kwargs)
    finally:
        end_frame = getattr(weights, 'end_source_frame', None)
        if end_frame is not None:
            end_frame()


def _counter(model, name, amount):
    model.scheduled_kv_calls[name] = model.scheduled_kv_calls.get(name, 0) + amount


def yoco_forward(model, ordinary, inputs, cache=None, *, logits_to_keep=None, capture_layers=(),
                 body_only=False, **options):
    """Admitted waves run the approximate skip; everything else is exact."""
    import mlx.core as mx
    plan = model.source_prefill_approx_plan
    reason = decision(model, inputs, cache, logits_to_keep=logits_to_keep, capture_layers=capture_layers,
                      body_only=body_only, options=options)
    rows = int(inputs.shape[1]) if getattr(inputs, 'ndim', 0) == 2 else 0
    if reason is not None:
        if rows > CHUNK_ROWS:
            model.source_prefill_approx_last = dict(admitted=False, reason=reason, rows=rows, revision=REVISION)
        if rows > CHUNK_ROWS and reason == 'prompt_below_minimum_rows' and logits_to_keep in (None, 1):
            _counter(model, 'source_yoco_exact_fallback_waves', 1)
            cache = model.make_cache() if cache is None else cache
            return _exact_chunks(model, ordinary, inputs, cache)
        return ordinary(inputs, cache, logits_to_keep=logits_to_keep, capture_layers=capture_layers,
                        body_only=body_only, **options)
    if inputs.dtype not in (mx.int32, mx.uint32, mx.int64):
        raise ValueError('YOCO wave requires integer token IDs')
    keep, first_skipped = plan['keep'], plan['first']
    context = (0 if cache is None else int(cache[0].offset)) + rows
    chunk = _chunk_rows(model, context)
    bounds, decoder_first = partition(rows, chunk, keep)
    # Validate the whole wave before any mutation.
    cache, _ = model.validate_forward(inputs[:, :bounds[0][1]], cache, 0)
    if cache[0].offset + rows > model.context_limit:
        raise ValueError('V4.1 cache prefix or context mismatch')
    ids = np.asarray(inputs)
    if np.any(ids < 0) or np.any(ids >= model.args.vocab_size):
        raise ValueError('V4.1 token ID outside vocabulary')
    start_offset = int(cache[0].offset)
    model._weight_executions += 1
    try:
        for state in cache:
            state._in_forward = True
        h = pre = shared = None
        for k, (start, end) in enumerate(bounds):
            final = k == len(bounds) - 1
            h, pre, shared = _sub_forward(model, inputs[:, start:end], cache, first_skipped=first_skipped,
                                          decoder_first=decoder_first if final else None)
        with model._weight_scope():
            hidden = model.norm(model.collapse(h, pre), model.norm_weight)
        logits = _final_head(model, hidden, rows, cache, shared)
        for state in cache:
            state.validate_state()
        if len({state.offset for state in cache}) != 1 or cache[0].offset != start_offset + rows:
            raise ValueError('YOCO wave ended with inconsistent layer frontiers')
        stamp_cache(cache, model.source_approx_private_profile)
        for state in cache:
            state._in_forward = False
    except BaseException as error:
        discard = getattr(model, 'discard_engram_prefetch', None)
        if discard is not None:
            discard(error)
        # Cache rows stay pending (_in_forward): validate_forward refuses them.
        raise
    finally:
        model._weight_executions -= 1
    skipped_layers = model.args.num_hidden_layers - first_skipped
    skipped_rows = (rows - keep) * skipped_layers
    model.source_prefill_approx_last = dict(
        admitted=True, revision=REVISION, rows=rows, start_offset=start_offset, keep_rows=keep,
        skipped_layers=[first_skipped, model.args.num_hidden_layers - 1], sub_chunks=len(bounds),
        chunk_rows=chunk, final_sub_chunk_rows=bounds[-1][1] - bounds[-1][0],
        skipped_decoder_layer_rows=skipped_rows,
        full_depth_layer_rows=rows * first_skipped + keep * skipped_layers,
        approximate=True, qualification='unqualified', private_cache_profile=model.source_approx_private_profile)
    _counter(model, 'source_yoco_waves', 1)
    _counter(model, 'source_yoco_rows', rows)
    _counter(model, 'source_yoco_skipped_decoder_layer_rows', skipped_rows)
    return logits, None


def exact_reference_prefill(model, inputs, cache):
    """GPU-script control: the ordinary exact chunk sequence on this model.

    Uses the same salted model/identity and chunking advice; returns final-row
    logits. Never called by serving.
    """
    ordinary = super(type(model), model)._forward
    return _exact_chunks(model, ordinary, inputs, cache)[0]


def plan_receipt(model):
    plan = getattr(model, 'source_prefill_approx_plan', None)
    if plan is None:
        return None
    value = dict(plan)
    value['identity'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value
