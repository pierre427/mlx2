"""Shared-layer GLM AST traversal with ordinary row arithmetic.

See provenance/glm53-gguf-forward.json. Native packed kernels put the token on
grid z and keep their one-row dot/reduction law. Dense native projections,
routing, HC and stateful attention stay one row at a time. No expert union
may reorder a query's top-k axis. A layer barrier bounds temporary storage.

Two opt-in MLA tree policies (docs/design/glm53-sparse-dsa.md 7.2, 7.3); the
defaults (per_row, clone) are the pre-policy loop bit for bit:

``glm53_tree_rows_attention=batched_v1``  every row runs ``_mla_prepare`` at
    its own L=1 shapes in tree order (a child forks only after its parent
    appended), then the selected rows attend in one ``rows_attention`` call
    (law V1: each row equals its one-row decode bit for bit). Rows without
    ids (T <= 2048) take the ordinary dense consumer one at a time.
``glm53_tree_frontier=tails_v1``  an MLA row's state is a Glm53MLAFrontier
    (committed base + private tail) instead of a descriptor clone of its
    parent's cache, so no row copies the whole latent; the transaction
    materializes the accepted path at commit. KDA rows keep private
    Glm53KDACache clones (they refuse trims and carry inherent state).
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from ..runtime.models import glm53_sparse_attention as sparse_attention
from ..runtime.models.glm53_gguf import INCREMENTAL_POOLS, Q8_REFUSAL, hc_collapse, hc_expand, rms
from ..runtime.models.glm53_gguf_indexer import Glm53PoolCache
from ..runtime.models.precise_ops import gate_sigmoid

# Tree policy values this module implements; the adapter refuses any other
# (glm53_gguf._check_sparse_combination). First value = today's loop.
SUPPORTED_TREE_ROWS_ATTENTION = ("per_row", "batched_v1")
SUPPORTED_TREE_FRONTIER = ("clone", "tails_v1")


def tree_policy(model):
    """(batched rows, frontier tails) for ``model``; refuses an undefined law.

    The adapter already refuses these combinations; a directly built model
    fails closed here instead of running a law nobody declared.
    """
    rows = getattr(model, "tree_rows_attention", "per_row")
    frontier = getattr(model, "tree_frontier", "clone")
    if rows not in SUPPORTED_TREE_ROWS_ATTENTION or frontier not in SUPPORTED_TREE_FRONTIER:
        raise ValueError("unsupported GLM-5.3 layered tree policy")
    attention = getattr(model, "sparse_attention", "mask_reference")
    if rows == "batched_v1" and attention == "mask_reference":
        raise ValueError("GLM-5.3 batched tree rows require a gathered sparse attention policy")
    if frontier == "tails_v1" and (
        getattr(model, "indexer_pools", "recompute") not in INCREMENTAL_POOLS
        or getattr(model, "indexer_select", "argsort") == "argsort"
        or attention == "mask_reference"
    ):
        raise ValueError(
            "GLM-5.3 tree frontier tails require incremental pools, topk_v1 selection "
            "and a gathered sparse attention policy"
        )
    return rows == "batched_v1", frontier == "tails_v1"


def fork_state(state, clone, *, tails=False, base_tokens=None):
    """A row's private successor of its parent's (or the base's) layer state.

    A frontier forks (aliased tails, cloned pools); under ``tails`` a base
    Glm53MLACache becomes a new frontier over it (the base is never written,
    its pools are cloned, not attached); anything else is a descriptor clone.
    """
    if isinstance(state, sparse_attention.Glm53MLAFrontier):
        return state.fork()
    if tails:
        pools = Glm53PoolCache() if getattr(state, "pools", None) is None else state.pools.clone()
        return sparse_attention.Glm53MLAFrontier(state, base_tokens, pools=pools)
    return clone([state])[0]


def mla_rows(model, prepared, prefix):
    """Batched verify rows (law V1): prepared [(q (1, 64, 1, 512), ids, source)].

    ``prepared`` holds each row's ``_mla_prepare`` result, computed in tree
    order at L=1 (selection, pools and the one append are already the
    one-row decode's). Rows with ids attend in one ``rows_attention`` call
    under the model's gathered policy; rows without ids (T <= 2048) take the
    ordinary dense consumer per row. Returns each row's (1, 1, 16384)
    pre-output core.
    """
    outs = [None] * len(prepared)
    sparse = []
    for row, (q, ids, source) in enumerate(prepared):
        if q.shape[:3] != (1, 64, 1):
            raise ValueError("GLM-5.3 batched tree rows need one query row each")
        if ids is None:
            outs[row] = model._mla_attend(q, None, source)
        elif isinstance(source, tuple):
            raise ValueError(Q8_REFUSAL)
        else:
            sparse.append(row)
    if sparse:
        out = sparse_attention.rows_attention(
            mx.concatenate([prepared[row][0] for row in sparse], axis=0),
            [(prepared[row][2], prepared[row][1]) for row in sparse],
            policy=model.sparse_attention,
            counters=model.calls,
        )
        for i, row in enumerate(sparse):
            outs[row] = out[i : i + 1]
    return [model._mla_finish(out, prefix, _return_pre_output=True) for out in outs]


def _attention_cores(model, layer, index, prefix, row_inputs, projected, parents, base,
                     successors, clone, offset, depth, policy):
    """Each row's pre-output attention core from its own private state."""
    batched, tails = policy
    cores = []
    prepared = []
    for row, parent in enumerate(parents):
        state = fork_state(
            base[index] if parent < 0 else successors[parent][index],
            clone,
            tails=tails and not layer.is_linear,
            base_tokens=offset,
        )
        x = row_inputs[row]
        # Integer pre-append frontier of this row (a descendant starts from
        # its parent's advanced state, so the base offset alone is wrong).
        frontier = offset + depth[row] - 1
        if layer.is_linear:
            local = {name: value[:, row : row + 1] for name, value in projected.items()}
            cores.append(
                model._kda(x, state, prefix, _projected=local, _return_pre_output=True)
            )
        elif batched:
            # Prepare now, in tree order: a child forks from this state below.
            prepared.append(model._mla_prepare(x, state, prefix, frontier=frontier))
        else:
            cores.append(
                model._mla(x, state, prefix, frontier=frontier, _return_pre_output=True)
            )
        successors[row].append(state)
    return mla_rows(model, prepared, prefix) if prepared else cores


def project_rows(model, x, name):
    project = model.weights.linear(name)
    if model.weights.tensors[name].ggml_type != 0:
        return project(x)
    # A native F32 matrix may select another reduction at M>1. Keep the
    # ordinary M1 law rather than relying on mathematical equivalence.
    return mx.concatenate([project(x[:, i : i + 1]) for i in range(x.shape[1])], axis=1)


def kda_projection_rows(model, x, prefix):
    def project(value, suffix):
        return project_rows(model, value, prefix + "." + suffix)

    return {
        "qkv": mx.concatenate(
            [project(x, "attn_" + name + ".weight") for name in ("q", "k", "v")],
            axis=-1,
        ),
        "a": project(project(x, "ssm_f_a.weight"), "ssm_f_b.weight"),
        "beta": project(x, "ssm_beta.weight"),
        "gate": project(project(x, "ssm_g_a.weight"), "ssm_g_b.weight"),
    }


def mlp_rows(model, x, prefix, index):
    def shared(suffix):
        gate = mx.minimum(
            project_rows(model, x, prefix + ".ffn_gate" + suffix + ".weight"), 10
        )
        up = mx.clip(
            project_rows(model, x, prefix + ".ffn_up" + suffix + ".weight"), -10, 10
        )
        return project_rows(
            model, nn.silu(gate) * up, prefix + ".ffn_down" + suffix + ".weight"
        )

    if index < 3:
        return shared("")
    ids = []
    scores = []
    for row in range(x.shape[1]):
        probability = gate_sigmoid(
            model.weights.linear(prefix + ".ffn_gate_inp.weight")(x[:, row : row + 1])
        )
        biased = probability + model.weights.array(prefix + ".exp_probs_b.bias")
        chosen = mx.argsort(-biased, axis=-1)[..., :8].astype(mx.uint32)
        weight = mx.take_along_axis(probability, chosen, axis=-1)
        weight = weight / weight.sum(-1, keepdims=True) * 2.5
        ids.append(chosen)
        scores.append(weight)
    ids = mx.concatenate(ids, axis=1)
    scores = mx.concatenate(scores, axis=1)
    routed = model.weights.moe(
        x.reshape(-1, x.shape[-1]), ids.reshape(-1, 8), scores.reshape(-1, 8), prefix
    ).reshape(x.shape)
    return routed + shared("_shexp")


def collapse_rows(model, h, prefix, eps):
    """Fused HC collapse of all tree rows at once, (1, W, 4, D) -> (1, W, D).

    The fused Sinkhorn, weighted-sum and expand kernels compute each row
    independently, and rms_norm reduces within a row, so batching rows keeps
    every row's values. The native F32 mixing projection keeps its M1 law.
    """
    from ..runtime.models.deepseek_v4_hc import split_sinkhorn, weighted_sum

    z = rms(h.reshape(*h.shape[:-2], -1), None, eps)
    mixes = project_rows(model, z, prefix + "_fn")
    scale, base = model.weights.array(prefix + "_scale"), model.weights.array(prefix + "_base")
    pre, post, comb = split_sinkhorn(mixes, scale, base, iterations=20, eps=1e-6)
    return weighted_sum(h, pre), post, comb


def _batched_layer(model, layer, index, prefix, hidden, parents, base, successors, eps, clone,
                   offset, depth, policy):
    """One layer with HC/norm work over all rows at once (fused HC only)."""
    from ..runtime.models.deepseek_v4_hc import expand

    rows = len(hidden)
    h = mx.concatenate(hidden, axis=1)
    x, post, comb = collapse_rows(model, h, prefix + ".hc_attn", eps)
    attention_inputs = rms(x, model.weights.array(prefix + ".attn_norm.weight"), eps)
    projected = kda_projection_rows(model, attention_inputs, prefix) if layer.is_linear else None
    cores = _attention_cores(
        model, layer, index, prefix,
        [attention_inputs[:, row : row + 1] for row in range(rows)],
        projected, parents, base, successors, clone, offset, depth, policy,
    )
    branches = project_rows(model, mx.concatenate(cores, axis=1), prefix + ".attn_output.weight")
    h = expand(branches, h, post, comb)
    x, post, comb = collapse_rows(model, h, prefix + ".hc_ffn", eps)
    mixed = rms(x, model.weights.array(prefix + ".ffn_norm.weight"), eps)
    h = expand(mlp_rows(model, mixed, prefix, index), h, post, comb)
    return [h[:, row : row + 1] for row in range(rows)]


def forward(model, inputs, parents, base, capture_layers, clone):
    rows = len(inputs)
    layers = tuple(capture_layers)
    policy = tree_policy(model)
    if (
        len(base) != len(model.layers)
        or not layers
        or tuple(sorted(set(layers))) != layers
        or layers[0] < 0
        or layers[-1] >= len(model.layers)
    ):
        raise ValueError("GLM layered tree requires exact cache and capture topology")
    # Python-int frontier (KDA offsets); array-valued MLA offsets would force
    # a GPU sync per truth test (see Glm53GGUFModel.__call__).
    offset = next(state.offset for state, layer in zip(base, model.layers) if layer.is_linear)
    depth = []
    for parent in parents:
        depth.append(1 if parent < 0 else depth[parent] + 1)
    if offset + max(depth) > model.args.max_position_embeddings:
        raise ValueError("GLM layered tree exceeds context capacity")
    for state, layer in zip(base, model.layers, strict=True):
        if layer.is_linear:
            if state.identity != model.identity or state.offset != offset:
                raise ValueError("GLM layered tree recurrent frontier mismatch")
        elif len(state.caches) != 2 or any(
                isinstance(c.offset, int) and c.offset != offset for c in state.caches):
            raise ValueError("GLM layered tree MLA/indexer frontier mismatch")
    embedded = model.weights.embedding(mx.array([inputs], mx.uint32))
    hidden = [
        mx.contiguous(
            mx.broadcast_to(embedded[:, i : i + 1, None, :], (1, 1, 4, embedded.shape[-1]))
        )
        for i in range(rows)
    ]
    successors = [[] for _ in inputs]
    taps = {}
    eps = model.args.rms_norm_eps
    batched_hc = bool(getattr(model, "hc_fused", False))
    for index, layer in enumerate(model.layers):
        prefix = f"blk.{index}"
        if batched_hc:
            hidden = _batched_layer(
                model, layer, index, prefix, hidden, parents, base, successors, eps, clone,
                offset, depth, policy,
            )
            # Research knob: barrier every N layers (1 = every layer, as before).
            every = getattr(model, "tree_eval_every", 1)
            if (index + 1) % every == 0 or index + 1 == len(model.layers):
                mx.eval(hidden, [s[i].state for s in successors for i in range(index - (index % every), index + 1)])
            if index in layers:
                taps[index] = [h.mean(-2) for h in hidden]
            continue
        attention_inputs = []
        attention_posts = []
        attention_combs = []
        for h in hidden:
            x, post, comb = hc_collapse(
                h, model.weights, prefix + ".hc_attn", eps, fused=model.hc_fused
            )
            attention_inputs.append(
                rms(x, model.weights.array(prefix + ".attn_norm.weight"), eps)
            )
            attention_posts.append(post)
            attention_combs.append(comb)
        projected = (
            kda_projection_rows(model, mx.concatenate(attention_inputs, axis=1), prefix)
            if layer.is_linear
            else None
        )
        attention_cores = _attention_cores(
            model, layer, index, prefix, attention_inputs, projected, parents, base,
            successors, clone, offset, depth, policy,
        )
        branches = project_rows(
            model,
            mx.concatenate(attention_cores, axis=1),
            prefix + ".attn_output.weight",
        )
        mixed = []
        residual = []
        posts = []
        combs = []
        for row in range(rows):
            h = hc_expand(
                branches[:, row : row + 1],
                hidden[row],
                attention_posts[row],
                attention_combs[row],
                fused=model.hc_fused,
            )
            x, post, comb = hc_collapse(
                h, model.weights, prefix + ".hc_ffn", eps, fused=model.hc_fused
            )
            mixed.append(rms(x, model.weights.array(prefix + ".ffn_norm.weight"), eps))
            residual.append(h)
            posts.append(post)
            combs.append(comb)
        branch = mlp_rows(model, mx.concatenate(mixed, axis=1), prefix, index)
        hidden = [
            hc_expand(
                branch[:, row : row + 1],
                residual[row],
                posts[row],
                combs[row],
                fused=model.hc_fused,
            )
            for row in range(rows)
        ]
        mx.eval(hidden, [state[index].state for state in successors])
        if index in layers:
            taps[index] = [h.mean(-2) for h in hidden]
    normed = [
        rms(h.mean(-2), model.weights.array("output_norm.weight"), eps) for h in hidden
    ]
    if model.external_feature_source == "mtp_seed":
        if layers != (len(model.layers) - 1,):
            raise ValueError("MTP tree requires the post-norm final seed only")
        features = mx.concatenate(normed, axis=1)
    else:
        features = mx.concatenate(
            [
                mx.concatenate([taps[layer][row] for layer in layers], axis=-1)
                for row in range(rows)
            ],
            axis=1,
        )
    logits = project_rows(model, mx.concatenate(normed, axis=1), "output.weight")
    mx.eval(
        logits, features, [state.state for frontier in successors for state in frontier]
    )
    return logits, features, successors
