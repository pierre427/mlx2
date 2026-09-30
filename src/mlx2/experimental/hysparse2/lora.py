"""Bounded research LoRA episodes. No autonomous data generation or serving route."""

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn, optimizers
from mlx.utils import tree_flatten, tree_unflatten


class EpisodeLinear(nn.Module):
    def __init__(self, base, rank, scale):
        super().__init__()
        self.linear = base
        self.scale = scale
        out_dim, in_dim = base.weight.shape
        self.lora_a = mx.random.normal((in_dim, rank)) * 0.01
        self.lora_b = mx.zeros((rank, out_dim), dtype=base.weight.dtype)

    def __call__(self, x):
        return self.linear(x) + (self.scale * ((x @ self.lora_a) @ self.lora_b)).astype(
            x.dtype
        )


class LoRAEpisode:
    """Freeze the base, train explicit linear keys, then evaluate or roll back.

    Call close in a finally block. Promotion requires held-out objective improvement
    and bounded preservation loss; a promotion is not a serving qualification.
    """

    def __init__(
        self,
        model,
        keys,
        *,
        base_revision,
        rank=8,
        scale=1.0,
        learning_rate=1e-3,
        max_steps=32,
    ):
        keys = tuple(keys)
        if type(max_steps) is not int or not 1 <= max_steps <= 1024:
            raise ValueError("episode budget must be from 1 to 1024 steps")
        self.max_steps = max_steps
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("explicit unique module keys required")
        if type(rank) is not int or not 1 <= rank <= 1024:
            raise ValueError("invalid rank")
        if not base_revision or not all(
            math.isfinite(v) and v > 0 for v in (scale, learning_rate)
        ):
            raise ValueError("revision and finite positive scales required")
        modules = dict(model.named_modules())
        if any(not isinstance(modules.get(key), nn.Linear) for key in keys):
            raise ValueError("episode keys must select ordinary Linear modules")
        if getattr(model, "_lora_episode", None) is not None:
            raise ValueError("an episode is already active")
        self.model, self.keys, self.base_revision = model, keys, base_revision
        self.rank, self.scale = rank, scale
        self.originals = {key: modules[key] for key in keys}
        self.frozen = [
            (module, set(module._no_grad)) for _, module in model.named_modules()
        ]
        self.was_training = model.training
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.steps, self.promoted, self.closed = 0, False, False
        self.previous_revision = model.adapter_revision
        replacements = [(key, EpisodeLinear(modules[key], rank, scale)) for key in keys]
        model.update_modules(tree_unflatten(replacements))
        model.freeze()
        for key in keys:
            dict(model.named_modules())[key].unfreeze(
                recurse=False, keys=["lora_a", "lora_b"]
            )
        model._lora_episode = self
        self._invalidate()

    def _invalidate(self):
        weights = self.weights()
        mx.eval(weights)
        digest = hashlib.sha256(
            json.dumps(
                {"keys": self.keys, "rank": self.rank, "scale": self.scale},
                sort_keys=True,
            ).encode()
        )
        for key, value in sorted(weights.items()):
            digest.update(key.encode())
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(np.array(value).tobytes())
        self.model.adapter_revision = self.base_revision + ":" + digest.hexdigest()
        self.live_revision = self.model.adapter_revision
        self.model._cache_owner = object()

    def weights(self):
        parameters = dict(tree_flatten(self.model.parameters()))
        return {
            f"{key}.{leaf}": parameters[f"{key}.{leaf}"]
            for key in self.keys
            for leaf in ("lora_a", "lora_b")
        }

    def step(self, tokens, objective):
        if self.closed or self.promoted or self.steps >= self.max_steps:
            raise ValueError("episode is no longer trainable")
        self.model.train()
        value, gradients = nn.value_and_grad(self.model, objective)(self.model, tokens)
        mx.eval(value, gradients)
        if not math.isfinite(value.item()) or any(
            not bool(mx.all(mx.isfinite(g)).item()) for _, g in tree_flatten(gradients)
        ):
            raise ValueError("nonfinite episode loss or gradient")
        self.optimizer.update(self.model, gradients)
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.steps += 1
        self._invalidate()
        return float(value.item())

    def evaluate(
        self, heldout, preservation, objective, *, max_preservation_regression=0.0
    ):
        if self.closed or not self.steps or self.promoted:
            raise ValueError("evaluation requires an active trained episode")
        if (
            not math.isfinite(max_preservation_regression)
            or max_preservation_regression < 0
        ):
            raise ValueError("invalid preservation budget")
        self.model.eval()
        adapted = [
            float(objective(self.model, x).item()) for x in (heldout, preservation)
        ]
        wrapped = {key: dict(self.model.named_modules())[key] for key in self.keys}
        try:
            self.model.update_modules(tree_unflatten(list(self.originals.items())))
            baseline = [
                float(objective(self.model, x).item()) for x in (heldout, preservation)
            ]
        finally:
            self.model.update_modules(tree_unflatten(list(wrapped.items())))
        self.promoted = (
            all(math.isfinite(v) for v in adapted + baseline)
            and adapted[0] < baseline[0]
            and adapted[1] <= baseline[1] + max_preservation_regression
        )
        return {
            "baseline": baseline,
            "adapted": adapted,
            "promoted": self.promoted,
            "steps": self.steps,
            "serving_route_qualified": False,
        }

    def export(self, path):
        if self.closed:
            raise ValueError("closed episode")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        mx.save_safetensors(str(path / "adapters.safetensors"), self.weights())
        config = {
            "fine_tune_type": "lora",
            "lora_parameters": {
                "rank": self.rank,
                "scale": self.scale,
                "dropout": 0,
                "keys": list(self.keys),
            },
            "base_revision": self.base_revision,
            "adapter_revision": self.model.adapter_revision,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "promoted": self.promoted,
        }
        (path / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")

    @classmethod
    def load_candidate(cls, model, path, *, base_revision):
        path = Path(path)
        config = json.loads((path / "adapter_config.json").read_text())
        if config["base_revision"] != base_revision:
            raise ValueError("adapter base revision differs")
        params = config["lora_parameters"]
        episode = cls(
            model,
            params["keys"],
            base_revision=base_revision,
            rank=params["rank"],
            scale=params["scale"],
            max_steps=config.get("max_steps", 32),
        )
        try:
            weights = mx.load(str(path / "adapters.safetensors"))
            expected = episode.weights()
            if set(weights) != set(expected) or any(
                weights[k].shape != expected[k].shape
                or not bool(mx.all(mx.isfinite(weights[k])).item())
                for k in expected
            ):
                raise ValueError("adapter tensor coverage or values differ")
            model.load_weights(list(weights.items()), strict=False)
            episode._invalidate()
            if model.adapter_revision != config["adapter_revision"]:
                raise ValueError("adapter content revision differs")
            steps = config["steps"]
            if type(steps) is not int or not 0 <= steps <= episode.max_steps:
                raise ValueError("invalid saved episode step count")
            episode.steps = steps
            return episode
        except BaseException:
            episode.close()
            raise

    def rollback(self):
        if self.model.adapter_revision != self.live_revision or getattr(
            self.model, "_lora_episode", None
        ) not in (None, self):
            raise ValueError("model no longer has this episode revision")
        self.promoted = False
        self.closed = False
        self.close()

    def close(self):
        if self.closed:
            return
        if not self.promoted:
            self.model.update_modules(tree_unflatten(list(self.originals.items())))
            self.model.adapter_revision = self.previous_revision
        for module, frozen in self.frozen:
            module._no_grad = frozen
        if self.promoted:
            for key in self.keys:
                dict(self.model.named_modules())[key].freeze()
        self.model.train(self.was_training)
        self.model._cache_owner = object()
        self.model._lora_episode = None
        self.closed = True
