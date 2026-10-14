"""Offline capture math for the dense/MoE Qwen3.5 hybrid trunk contracts."""

from contextlib import nullcontext


class Qwen35CaptureDriver:
    def __init__(self, model, weight_stream=None):
        import mlx.core as mx
        from ..runtime.models.base import create_attention_mask, create_ssm_mask

        self.mx = mx
        self.model = model
        self.trunk = model.model
        if self.trunk.pipeline_size != 1:
            raise ValueError("offline layer capture requires a single pipeline worker")
        self.layer_count = len(model.layers)
        self.weight_stream = weight_stream
        self.attention_mask = create_attention_mask
        self.ssm_mask = create_ssm_mask

    def start(self, tokens):
        if any(token >= self.trunk.embed_tokens.weight.shape[0] for token in tokens):
            raise ValueError("capture token ID outside the model vocabulary")
        tokens = self.mx.array([tokens], self.mx.uint32)
        hidden = self.trunk.embed_tokens(tokens)
        cache = self.model.make_cache()
        fa = (
            None
            if self.trunk.fa_idx is None
            else self.attention_mask(hidden, cache[self.trunk.fa_idx])
        )
        ssm = (
            None
            if self.trunk.ssm_idx is None
            else self.ssm_mask(hidden, cache[self.trunk.ssm_idx])
        )
        return {
            "tokens": tokens,
            "hidden": hidden,
            "cache": cache,
            "fa": fa,
            "ssm": ssm,
        }

    def step(self, state, index):
        layer = self.model.layers[index]
        # Run each layer in the scope the trunk's own forward opens (e.g. the
        # NAX MoE gather's prefill phase), so capture executes the mechanisms
        # its execution binding names (sweep 2026-10-12 L7-1).
        scope = getattr(self.trunk, "forward_scope", None)
        with (
            scope(state["tokens"], state["cache"]) if callable(scope) else nullcontext()
        ):
            state["hidden"] = layer(
                state["hidden"],
                mask=state["ssm"] if layer.is_linear else state["fa"],
                cache=state["cache"][index],
            )
        return state

    def arrays(self, state):
        from mlx.utils import tree_flatten

        arrays = [state["tokens"], state["hidden"], state["fa"], state["ssm"]]
        for cache in state["cache"]:
            arrays.extend(value for _, value in tree_flatten(cache.state))
        return [value for value in arrays if isinstance(value, self.mx.array)]

    def state_bytes(self, state):
        return sum(v.nbytes for v in {id(v): v for v in self.arrays(state)}.values())

    def hidden_bytes(self, state):
        return int(state["hidden"].nbytes)

    def transition_bytes(self, state, layer):
        from mlx.utils import tree_flatten

        old = [state["hidden"]] + [
            v
            for _, v in tree_flatten(state["cache"][layer].state)
            if isinstance(v, self.mx.array)
        ]
        return sum(v.nbytes for v in {id(v): v for v in old}.values())

    def evaluate(self, state):
        self.mx.eval(*self.arrays(state))

    def weight_scope(self, index):
        manager = self.weight_stream
        if manager is not None and manager.mode == "dense_mlp":
            return manager.capture_layer(f"language_model.model.layers.{index}")
        return nullcontext()

    def finish(self, index, tokens, state, sink):
        hidden = self.trunk.norm(state["hidden"])
        self.mx.eval(hidden)
        # Cache arrays are evaluated per layer; the sink may inspect them to
        # validate against ordinary prefill, but cannot publish them to APCv2.
        sink(index, tokens, hidden, state["cache"])
