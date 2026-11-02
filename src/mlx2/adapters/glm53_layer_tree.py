"""Shared-layer GLM AST traversal with ordinary row arithmetic.

See provenance/glm53-gguf-forward.json. Native packed kernels put the token on
grid z and keep their one-row dot/reduction law. Dense native projections,
routing, HC and stateful attention stay one row at a time. No expert union
may reorder a query's top-k axis. A layer barrier bounds temporary storage.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from ..runtime.models.glm53_gguf import hc_collapse, hc_expand, rms
from ..runtime.models.precise_ops import gate_sigmoid


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


def forward(model, inputs, parents, base, capture_layers, clone):
    rows = len(inputs)
    layers = tuple(capture_layers)
    if (
        len(base) != len(model.layers)
        or not layers
        or tuple(sorted(set(layers))) != layers
        or layers[0] < 0
        or layers[-1] >= len(model.layers)
    ):
        raise ValueError("GLM layered tree requires exact cache and capture topology")
    offset = base[3][0].offset
    depth = []
    for parent in parents:
        depth.append(1 if parent < 0 else depth[parent] + 1)
    if offset + max(depth) > model.args.max_position_embeddings:
        raise ValueError("GLM layered tree exceeds context capacity")
    for state, layer in zip(base, model.layers, strict=True):
        if layer.is_linear:
            if state.identity != model.identity or state.offset != offset:
                raise ValueError("GLM layered tree recurrent frontier mismatch")
        elif len(state.caches) != 2 or any(c.offset != offset for c in state.caches):
            raise ValueError("GLM layered tree MLA/indexer frontier mismatch")
    embedded = model.weights.embedding(mx.array([inputs], mx.uint32))
    hidden = [
        mx.contiguous(mx.broadcast_to(embedded[:, i : i + 1, None, :], (1, 1, 4, 4096)))
        for i in range(rows)
    ]
    successors = [[] for _ in inputs]
    taps = {}
    eps = model.args.rms_norm_eps
    for index, layer in enumerate(model.layers):
        prefix = f"blk.{index}"
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
        attention_cores = []
        for row, parent in enumerate(parents):
            state = clone([base[index] if parent < 0 else successors[parent][index]])[0]
            x = attention_inputs[row]
            if layer.is_linear:
                local = {
                    name: value[:, row : row + 1] for name, value in projected.items()
                }
                branch = model._kda(
                    x, state, prefix, _projected=local, _return_pre_output=True
                )
            else:
                branch = model._mla(x, state, prefix, _return_pre_output=True)
            attention_cores.append(branch)
            successors[row].append(state)
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
