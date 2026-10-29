# SPDX-License-Identifier: MIT
"""Opt-in layer-major Source prefill waves: one expert pass per layer per wave.

The ordinary adaptive prefill runs each outer chunk (at most 1024 rows) through
all 40 layers before the next chunk. With 384 experts per layer, every chunk
touches almost the whole routed bank, and the Source+Q1 bank does not fit beside
the hot set, so each chunk re-reads most cold experts from SSD.

``layer_major_v1`` accepts a larger outer *wave* (a whole multiple of the live
chunk size, bounded by measured memory) and executes it layer-major: for layer
``i`` every sub-chunk runs the unchanged ordinary ``Block`` in token order before
layer ``i+1`` starts. Consecutive ``moe()`` calls of one layer reuse the experts
the previous sub-chunk just made resident, so each wave reads a layer's experts
roughly once instead of once per chunk.

Arithmetic: every sub-chunk keeps the ordinary row partition, operator shapes
and per-layer cache progression. A compressed consumer sees its source pool at
exactly the extent the chunk-major program would expose (the same bounded copy
used by layerwise verification). Only execution order changes. The profile is
nevertheless salted with a distinct identity and remains unqualified; APCv2 stays
disabled exactly as under ``adaptive_v1``.

Fail closed: CED, vision, DSpark/AST, resident banks, canonical partitions,
native packed KV, frame/row-packet reuse, physical batching and drafts refuse.
Provenance: provenance/deepseek-v41-source-layer-major-prefill.json.
"""
import hashlib
from copy import copy
from pathlib import Path

import numpy as np

KEY = 'deepseek_source_prefill_order'
POLICY = 'layer_major_v1'
REFERENCE = 'reference'
CHUNK_ROWS = 1024
MAX_WAVE_ROWS = 16384
# Retained activations per wave row across sub-chunks: HC residual stream
# (hc_mult x hidden x f32 = 80 KiB for production geometry), collapse weights,
# per-chunk shared index/selection/candidate tensors and slack. Conservative.
WAVE_ROW_BYTES = 256 << 10
WAVE_SAFETY_BYTES = 4 << 30

_REFUSED = (
    ('deepseek_ced', 'reference'),
    ('deepseek_vision', 'reference'),
    ('deepseek_kv_storage', 'reference'),
    ('deepseek_physical_batch', 'b1'),
    ('deepseek_source_prefill_partition', 'adaptive_v1'),
    ('deepseek_source_dspark_ast', 'off'),
    ('deepseek_source_dspark_prepare', 'off'),
    ('deepseek_source_dspark_target_forward', 'reference'),
    ('deepseek_source_ast_kv_preparation', 'reference'),
    ('deepseek_source_resident_gather', 'reference'),
    ('deepseek_source_resident_prefill', 'reference'),
    ('deepseek_source_frame_reuse', 'reference'),
    ('deepseek_source_row_packet', 'reference'),
)


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    names = ('adapters/deepseek_v41_source_layer_major.py', 'runtime/models/deepseek_v41.py',
             'runtime/models/deepseek_v41_cache.py')
    return hashlib.sha256(''.join(name + ':' + hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in names).encode()).hexdigest()


def validate_policy(policy):
    """Refuse every combination whose forward or state law was not composed."""
    selected = policy.get(KEY, REFERENCE)
    if selected not in (REFERENCE, POLICY):
        raise ValueError('invalid Source policy: ' + KEY)
    if selected == REFERENCE:
        return selected
    for name, reference in _REFUSED:
        if policy.get(name, reference) != reference:
            raise ValueError('Source layer-major prefill cannot combine with ' + name)
    if policy.get('draft_model') not in (None, 'none') or policy.get('ast_prompt_lookup', False):
        raise ValueError('Source layer-major prefill requires ordinary decoding without drafts')
    return selected


def receipt(policy, weights=None):
    if policy not in (REFERENCE, POLICY):
        raise ValueError('Unsupported Source prefill order policy')
    value = dict(policy=policy, qualification='unqualified')
    if policy == POLICY:
        value.update(order='layer-major waves of ordinary chunk-major sub-chunks',
                     sub_chunk_rows_max=CHUNK_ROWS, wave_rows_max=MAX_WAVE_ROWS,
                     wave_row_bytes=WAVE_ROW_BYTES, apcv2_reuse_enabled=False,
                     numerical_identity='distinct salt; arithmetic partition unchanged')
        if weights is not None:
            value['last_wave_advice'] = getattr(weights, 'source_layer_major_advice', None)
    return value


def wave_rows(weights, maximum, chunk, available):
    """Return a memory-bounded wave, a whole multiple of the live chunk size.

    The wave only adds retained activations; expert cache targets stay adaptive.
    Unknown availability, reduced chunks or insufficient room return ``chunk``.
    """
    if type(maximum) is not int or type(chunk) is not int or not 1 <= chunk <= maximum:
        raise ValueError('Source layer-major wave requires chunk1..maximum')
    advice = getattr(weights, 'source_memory_advice', None) or {}
    workspace = int(advice.get('workspace_bytes', 0) or 0)
    reason = None
    rows = chunk
    if chunk < CHUNK_ROWS:
        reason = 'chunk_reduced_by_memory_advice'
    elif type(available) is not int or available < 0:
        reason = 'availability_unknown'
    else:
        room = available - WAVE_SAFETY_BYTES - workspace
        limit = max(0, room) // WAVE_ROW_BYTES
        rows = min(maximum, MAX_WAVE_ROWS, limit) // chunk * chunk
        if rows <= chunk:
            rows, reason = chunk, 'insufficient_live_memory'
    weights.source_layer_major_advice = dict(policy=POLICY, requested_rows=maximum, chunk_rows=chunk,
        wave_rows=rows, available_bytes=available, workspace_bytes=workspace,
        retained_activation_bytes=rows * WAVE_ROW_BYTES, fallback=reason)
    return rows


def _chunk_rows(model, rows, context):
    limit = int(getattr(model, 'source_layer_major_rows', CHUNK_ROWS))
    advise = getattr(model.weights, 'source_layer_major_chunk_rows', None)
    chunk = limit if advise is None else advise(context_tokens=context)
    if type(chunk) is not int or not 1 <= chunk:
        raise ValueError('Source layer-major chunk advice must be positive')
    return min(chunk, limit, rows)


def layer_major_forward(model, ordinary, inputs, cache=None, *, logits_to_keep=None, capture_layers=(),
                        body_only=False, **options):
    """Dispatch a wave larger than one ordinary chunk; otherwise run ordinary."""
    import mlx.core as mx
    limit = int(getattr(model, 'source_layer_major_rows', CHUNK_ROWS))
    if inputs.ndim != 2 or inputs.shape[1] <= limit:
        return ordinary(inputs, cache, logits_to_keep=logits_to_keep, capture_layers=capture_layers,
                        body_only=body_only, **options)
    weights = model.weights
    if getattr(weights, 'source_prefill_order_policy', REFERENCE) != POLICY:
        raise ValueError('Source layer-major owner/profile changed')
    if capture_layers or body_only or any(value not in (None, False) for value in options.values()):
        raise ValueError('Source layer-major waves require ordinary text prefill without taps, CED or images')
    rows = inputs.shape[1]
    if inputs.shape[0] != 1 or rows > MAX_WAVE_ROWS or inputs.dtype not in (mx.int32, mx.uint32, mx.int64):
        raise ValueError('Source layer-major wave requires integer B1 tokens, rows1..%d' % MAX_WAVE_ROWS)
    if logits_to_keep is not None and (type(logits_to_keep) is not int or logits_to_keep < 0):
        raise ValueError('invalid V4.1 logits_to_keep')
    if (getattr(model, '_mlx2_ds41_ast_active', None) is not None or getattr(model, '_mlx2_ds41_ast_poisoned', False)
            or getattr(weights, '_source_current_frame', None) is not None
            or getattr(weights, 'source_resident_bank_session', None) is not None
            or getattr(weights, 'evaluate_block', None) is not None
            or getattr(model, 'kv_storage_policy', 'reference') != 'reference'
            or getattr(model, 'ced_policy', 'reference') != 'reference'):
        raise ValueError('Source layer-major wave requires an idle ordinary reference-cache owner')
    context = (0 if cache is None else int(cache[0].offset)) + rows
    chunk = _chunk_rows(model, rows, context)
    bounds = [(start, min(rows, start + chunk)) for start in range(0, rows, chunk)]
    # Validate the whole wave before any mutation: cache topology and the
    # first chunk through the ordinary validator, then context and all ids.
    cache, _ = model.validate_forward(inputs[:, :bounds[0][1]], cache, 0)
    ids = np.asarray(inputs)
    if cache[0].offset + rows > model.context_limit:
        raise ValueError('V4.1 cache prefix or context mismatch')
    if np.any(ids < 0) or np.any(ids >= model.args.vocab_size):
        raise ValueError('V4.1 token ID outside vocabulary')
    model._weight_executions += 1
    try:
        rows_execution = getattr(weights, 'execution_scope_for_rows', None)
        execution = getattr(weights, 'execution_scope', None)
        from contextlib import nullcontext
        scope = (rows_execution(chunk) if callable(rows_execution)
                 else execution() if execution is not None else nullcontext())
        with scope:
            return _wave(model, inputs, ids, cache, bounds, logits_to_keep)
    except BaseException as error:
        discard = getattr(model, 'discard_engram_prefetch', None)
        if discard is not None:
            discard(error)
        raise
    finally:
        end_frame = getattr(weights, 'end_source_frame', None)
        if end_frame is not None:
            end_frame()
        model._weight_executions -= 1


def _wave(model, inputs, ids, cache, bounds, logits_to_keep):
    import mlx.core as mx
    from ..runtime.prefill_checkpoint_rows import select_checkpoint_rows
    c, weights = model.args, model.weights
    base = cache[0].offset
    stats = model.scheduled_kv_calls
    stream = getattr(weights, 'expert_stream', None)
    before = dict(getattr(stream, 'stats', {}) or {})
    if stream is not None and stream.atlas is not None:
        from ..runtime.expert_stream_atlas import ExpertStreamAtlas
        stream.begin_context(ExpertStreamAtlas.keys(model.hasher.token_map[ids[0]],
                                                    np.asarray(cache[0].hash_tail)[::-1]))
    hashes, tail = [], None
    if model.hasher is not None:
        # Thread the tail exactly as successive ordinary chunks would.
        tail = np.asarray(cache[0].hash_tail)
        for start, end in bounds:
            value, tail = model.hasher(ids[0][start:end], tail)
            hashes.append(value)
            tail = np.asarray(mx.array(tail, mx.int32))
        for index, layer in enumerate(c.engram_layer_ids):
            model.engram_tables[layer].prefetch(hashes[0][:, index])
    for state in cache:
        state._in_forward = True
    hidden, pre = [], []
    with model._weight_scope():
        for start, end in bounds:
            x = model.round(weights.embedding(inputs[:, start:end].astype(mx.uint32)))
            hidden.append(mx.broadcast_to(x[:, :, None], (1, end - start, c.hc_mult, c.hidden_size)))
            pre.append(mx.broadcast_to(mx.array([1.0] + [0.0] * (c.hc_mult - 1)), hidden[-1].shape[:-1]))
        model._weight_eval(hidden, pre)
    shared = [{} for _ in bounds]
    gpu = mx.default_device() == mx.gpu
    for layer, (block, state) in enumerate(zip(model.layers, cache)):
        owner_index = c.kv_source_for(layer) if state.ratio and not state.source else None
        engram = c.engram_layer_ids.index(layer) if layer in c.engram_layer_ids else None
        for k, (start, end) in enumerate(bounds):
            if state.offset != base + start:
                raise ValueError('Source layer-major layer frontier mismatch')
            with model._weight_scope():
                if engram is not None:
                    table = model.engram_tables[layer]
                    values = mx.array(table.read(hashes[k][:, engram]))
                    if k + 1 < len(bounds):
                        table.prefetch(hashes[k + 1][:, engram])
                    hidden[k] = model.engram(hidden[k], layer, None, rows=values)
                visible = cache
                if owner_index is not None:
                    # Expose the source pool at its chunk-major extent.
                    owner = copy(cache[owner_index])
                    owner.offset = base + end
                    owner._pool_count = owner.offset // owner.ratio
                    visible = list(cache)
                    visible[owner_index] = owner
                hidden[k], pre[k] = block(hidden[k], pre[k], state, visible, shared[k])
                if model.eval_interval and gpu:
                    mx.async_eval(hidden[k], pre[k])
                model._weight_eval(hidden[k], pre[k], [v.state for v in cache], shared[k])
    if tail is not None:
        cache[0].hash_tail = mx.array(tail, mx.int32)
    with model._weight_scope():
        final = [model.norm(model.collapse(h, p), model.norm_weight) for h, p in zip(hidden, pre)]
        whole = mx.concatenate(final, 1)
        if logits_to_keep is None:
            selected = select_checkpoint_rows(whole)
            pieces = [selected] if selected.shape[1] != whole.shape[1] else final
        elif logits_to_keep:
            pieces = [whole[:, -logits_to_keep:]]
        else:
            pieces = final
        # Unnarrowed heads run per sub-chunk, matching ordinary row counts.
        logits = mx.concatenate([model.linear(piece, 'output.weight', rounded=False) for piece in pieces], 1)
        model._weight_eval(logits, hidden, [v.state for v in cache], shared)
    for state in cache:
        state.validate_state()
    for state in cache:
        state._in_forward = False
    if stream is not None:
        stream.end_context()
    rows = bounds[-1][1]
    stats['source_layer_major_waves'] = stats.get('source_layer_major_waves', 0) + 1
    stats['source_layer_major_chunks'] = stats.get('source_layer_major_chunks', 0) + len(bounds)
    stats['source_layer_major_rows'] = stats.get('source_layer_major_rows', 0) + rows
    after = getattr(stream, 'stats', None) or {}
    for name in ('bytes_read', 'cache_misses', 'cache_hits'):
        if name in after:
            key = 'source_layer_major_expert_' + name
            stats[key] = stats.get(key, 0) + int(after[name]) - int(before.get(name, 0))
    return logits, None


def layer_major_model_class():
    """Adapter-owned subclass; layers keep genuine model/self references."""
    from ..runtime.models.deepseek_v41 import DeepSeekV41Model

    class LayerMajorSourceModel(DeepSeekV41Model):
        source_layer_major_rows = CHUNK_ROWS

        def _forward(self, inputs, cache=None, *, logits_to_keep=None, capture_layers=(),
                     body_only=False, **options):
            return layer_major_forward(self, super()._forward, inputs, cache, logits_to_keep=logits_to_keep,
                                       capture_layers=capture_layers, body_only=body_only, **options)

    return LayerMajorSourceModel
