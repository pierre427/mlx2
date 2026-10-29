# SPDX-License-Identifier: MIT
"""Chronological count classes for explicit Source AST index-score candidates.

Index columns are part of the ordinary numerical law. An ancestry mask over a
larger physical table does not preserve the same matrix shape or top-k call.
The explicit batched-M1 route selects a grouped law only after its owning
actual-input probe. Other model profiles and the native route are unchanged.
"""
import hashlib
from pathlib import Path

import mlx.core as mx

from ..runtime.models.deepseek_v41 import candidate_blocks
from ..runtime.models.deepseek_v4 import stable_pool_topk

PROFILE = 'source_ast_chronological_index_classes_candidate_v1'
INDEX_GPU_IMPLEMENTATION = 'native_batch_v1'


def count_classes(logical_rows):
    if len(logical_rows) != 16:
        raise ValueError('Source AST index classes require16 query rows')
    groups = {}
    for node, rows in enumerate(logical_rows):
        if any(type(v) is not int or v < 0 for v in rows) or len(set(rows)) != len(rows):
            raise ValueError('Source AST index row inventory is invalid')
        groups.setdefault(len(rows), []).append(node)
    return tuple((count, tuple(nodes)) for count, nodes in sorted(groups.items()))


def matrix_scores(query, keys, *, check_live=None, implementation=None):
    """Return head scores[G,H,N] for equal ordinary chronological counts.

    Query is[G,H,K], keys is[G,N,K]. A singleton uses the actual owning B1
    einsum with reshape singleton strides, not a None-axis control surrogate.
    """
    if (query.ndim != 3 or keys.ndim != 3 or query.shape[0] != keys.shape[0]
            or query.shape[-1] != keys.shape[-1]):
        raise ValueError('Source AST chronological index matrix geometry mismatch')
    batches, heads, dim = query.shape
    count = keys.shape[1]
    if mx.default_device() == mx.gpu:
        if not callable(check_live) or check_live() is not True:
            raise ValueError('Source AST index scores require live caller ownership')
        if (heads, dim) != (32, 128):
            raise ValueError('Source AST exact index-score GPU head geometry is not admitted')
        if batches > 1:
            from ..runtime.gguf.source_ast_batched_index_matmul import private_matmul
            if implementation is None:
                raise ValueError('Source AST exact grouped index-score GPU law is not admitted')
            return private_matmul(query.astype(mx.float32),
                mx.swapaxes(keys.astype(mx.float32), -1, -2),
                check_live=check_live, implementation=implementation)
    if batches == 1:
        return mx.einsum('bhsd,btd->bsht',
            mx.contiguous(query).reshape(1, heads, 1, dim),
            mx.contiguous(keys).reshape(1, count, dim))[:, 0]
    # CPU-only algebraic reference. Actual GPU dispatch is never inferred here.
    return mx.matmul(query.astype(mx.float32), mx.swapaxes(keys.astype(mx.float32), -1, -2))


def weighted_scores(head_scores, weights):
    """Preserve the owning[batch,query,heads,count] head-reduction axes."""
    if weights.shape != head_scores.shape[:2]:
        raise ValueError('Source AST index head weight geometry mismatch')
    groups, heads, count = head_scores.shape
    return mx.sum(mx.maximum(head_scores.reshape(groups, 1, heads, count), 0)
                  * weights.reshape(groups, 1, heads, 1), axis=2)


def select_tile(attn, x, qr, owner, positions, shared, *, check_live=None):
    """Publish per-node ordinary-width indices/candidate masks, never padded.

    Only count classes loop in Python. Index IDs and selections stay on device.
    No full KV copies are made; chronological index-vector gathers are bounded
    separately and remain an explicit preparation cost for future panel reuse.
    """
    from .deepseek_v41_ast_batched_dense_exact import linear_tile
    model, config = attn.model, attn.c
    if len(shared) != 16 or len(positions) != 16:
        raise ValueError('Source AST index tile query inventory mismatch')
    owner.assert_current()
    classes = count_classes(owner.logical_rows)
    finite = model.index_policy == 'finite_all_retained'
    def bypass(count):
        all_blocks = (count+config.candidate_block_size-1)//config.candidate_block_size <= config.candidate_topk_blocks
        return finite and count <= config.index_topk and (
            attn.layer != config.candidate_source_layer_id or all_blocks)
    needs_score = any(count and not bypass(count) for count, _ in classes)
    if mx.default_device() == mx.gpu and needs_score:
        if not callable(check_live) or check_live() is not True:
            raise ValueError('Source AST index tile requires live caller ownership')
        if any(count and not bypass(count) and len(nodes) > 1 for count, nodes in classes):
            if INDEX_GPU_IMPLEMENTATION is None:
                raise ValueError('Source AST exact grouped index-score GPU law is not admitted')
    for row in shared:
        row['_index_generation'] = row.get('_index_generation', 0)+1
        for key in ('indices', 'selected_mask', '_decode_selected_kv', '_all_retained_chronological'):
            row.pop(key, None)
    query = weight = None
    if needs_score:
        query = linear_tile(model, qr, attn.prefix+'indexer.attn_q_b.weight')
        query = query.reshape(1, 16, config.index_n_heads, config.index_head_dim).transpose(0, 2, 1, 3)
        query = attn.rotate_quantized(query, 'index_fp4', positions=positions)
        weight = linear_tile(model, x, attn.prefix+'indexer.proj.weight') * (
            config.index_head_dim**-.5 * config.index_n_heads**-.5)
    for count, nodes in classes:
        groups = len(nodes)
        lenses = mx.array([(positions[node]+1)//attn.ratio for node in nodes], mx.int32).reshape(groups, 1, 1)
        candidates = None
        if not count:
            indices = mx.zeros((groups, 1, 0), mx.int32)
            valid = mx.zeros((groups, 1, 0), mx.bool_)
            if attn.layer == config.candidate_source_layer_id:
                candidates = valid
        elif bypass(count):
            indices = mx.broadcast_to(mx.arange(count, dtype=mx.uint32).reshape(1, 1, count), (groups, 1, count))
            valid = indices < lenses
            if attn.layer == config.candidate_source_layer_id:
                candidates = indices//config.candidate_block_size*config.candidate_block_size < lenses
        else:
            ids = mx.array(nodes, mx.int32)
            local_query = mx.take(query, ids, axis=2)[0].transpose(1, 0, 2)
            rows = mx.array([owner.logical_rows[node] for node in nodes], mx.int32)
            keys = owner.index_table[0, rows]
            scores = matrix_scores(local_query, keys, check_live=check_live,
                implementation=INDEX_GPU_IMPLEMENTATION)
            scores = weighted_scores(scores, mx.take(weight[0], ids, axis=0))
            scores = mx.where(mx.arange(count).reshape(1, 1, count) < lenses, scores, -mx.inf)
            if attn.layer == config.candidate_source_layer_id:
                candidates = candidate_blocks(scores, lenses, config.candidate_topk_blocks,
                    config.candidate_block_size, finite_all_retained=finite)
            if attn.layer > config.candidate_source_layer_id >= 0:
                prior = _prior(shared, nodes, count)
                scores = mx.where(prior, scores, -mx.inf)
            indices = stable_pool_topk(scores, min(config.index_topk, count))
            valid = mx.take_along_axis(scores, indices, axis=-1) > -mx.inf
        if count and bypass(count) and attn.layer > config.candidate_source_layer_id >= 0:
            valid = valid & _prior(shared, nodes, count)
        for slot, node in enumerate(nodes):
            shared[node]['indices'] = indices[slot:slot+1]
            shared[node]['selected_mask'] = valid[slot:slot+1]
            if candidates is not None:
                shared[node]['candidates'] = candidates[slot:slot+1]
    owner._exact_index_count_classes = classes
    owner._exact_index_gather_bytes = sum(count*len(nodes)*config.index_head_dim*4
        for count, nodes in classes if count and not bypass(count))


def _prior(shared, nodes, count):
    rows = []
    for node in nodes:
        value = shared[node].get('candidates')
        if value is None or value.shape != (1, 1, count):
            raise ValueError('Source AST prior candidate mask chronological width changed')
        rows.append(value)
    return mx.concatenate(rows, axis=0)


def implementation_identity():
    from ..runtime.gguf.source_ast_batched_index_matmul import implementation_identity as kernel_identity
    return hashlib.sha256(Path(__file__).read_bytes()+kernel_identity().encode()).hexdigest()
