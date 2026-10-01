# SPDX-License-Identifier: Apache-2.0
"""Original MLX M-DFlash/DPara candidate from arXiv:2609.27396v1.

See provenance/dpara.json. Synthetic tensor validation establishes the branch
algorithm, not a trained model, a selected serving route, or hardware overlap.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from ..dpara import DParaBinding, DParaPrepared
from ..speculative_sampling import RequestRNG, probability, softmax
from .dflash_base import DFlashAttention, Qwen3MLP
from .dpara_config import DParaConfig


def branch_layout(draft_length, context_length=0):
    """Visibility and repeated positions for one shared-spine tree.

    Spine has d+1 tokens. Each boundary r has d bidirectional masks following
    r+1 featureless anchors. Cached featured tokens precede the tree. Position
    IDs restart after each boundary, never after the previous flattened branch.
    """
    if type(draft_length) is not int or draft_length < 1:
        raise ValueError("DPara draft length must be positive")
    if type(context_length) is not int or context_length < 0:
        raise ValueError("DPara context length must be nonnegative")
    spine = draft_length + 1
    total = spine + spine * draft_length
    allowed = np.zeros((total, context_length + total), dtype=bool)
    allowed[:, :context_length] = True
    positions = list(range(context_length, context_length + spine))
    for i in range(spine):
        allowed[i, context_length : context_length + i + 1] = True
    for r in range(spine):
        start = spine + r * draft_length
        stop = start + draft_length
        allowed[start:stop, context_length : context_length + r + 1] = True
        allowed[start:stop, context_length + start : context_length + stop] = True
        positions.extend(
            range(context_length + r + 1, context_length + r + 1 + draft_length)
        )
    return allowed, np.asarray(positions, dtype=np.int32)


def _position_rope(x, positions, config):
    batch, heads, length, dims = x.shape
    # MLX accepts one offset per batch item. Treat each tree token as a batch
    # item with T=1 so repeated, nonmonotonic position IDs remain exact.
    flattened = x.transpose(0, 2, 1, 3).reshape(batch * length, heads, 1, dims)
    offsets = mx.broadcast_to(positions[None], (batch, length)).reshape(-1)
    rotated = mx.fast.rope(
        flattened,
        config.head_dim,
        traditional=False,
        base=config.rope_theta,
        scale=1.0,
        offset=offsets,
    )
    return rotated.reshape(batch, length, heads, dims).transpose(0, 2, 1, 3)


@dataclass(frozen=True)
class DParaContext:
    """Only committed target-feature KV; precompute cannot append to it."""

    binding: DParaBinding
    length: int
    layers: tuple


@dataclass(frozen=True)
class DParaBranches:
    binding: DParaBinding
    spine: tuple[int, ...]
    context: DParaContext
    hidden: mx.array  # [d+1, d, H], for deferred vocabulary projection
    logits: mx.array  # [d+1, d, V]


@dataclass(frozen=True)
class DParaContinuation:
    binding: DParaBinding
    accepted_tokens: tuple[int, ...]
    bonus: int
    draft_tokens: tuple[int, ...]
    proposal_probabilities: tuple[np.ndarray, ...]
    context: DParaContext

    @property
    def next_spine(self):
        return (self.bonus, *self.draft_tokens)

    def receipt(self):
        return {
            "kind": "dpara_candidate",
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "candidate_invoked": True,
            "request_id": self.binding.request_id,
            "target_revision": self.binding.target_revision,
            "draft_revision": self.binding.draft_revision,
            "generation": self.binding.generation,
            "draft_length": len(self.draft_tokens),
            "accepted_length": len(self.accepted_tokens),
            "overlap": "caller_scheduled_unmeasured",
        }


class DParaAttention(DFlashAttention):
    """DFlash projections with tree visibility and immutable context KV."""

    def project_context(self, features, positions, config):
        batch, length, _ = features.shape
        keys = self.k_norm(
            self.k_proj(features).reshape(
                batch,
                length,
                self.n_kv_heads,
                self.head_dim,
            )
        ).transpose(0, 2, 1, 3)
        values = (
            self.v_proj(features)
            .reshape(
                batch,
                length,
                self.n_kv_heads,
                self.head_dim,
            )
            .transpose(0, 2, 1, 3)
        )
        return _position_rope(keys, positions, config), values

    def __call__(self, x, context_kv, positions, allowed, config):
        batch, length, _ = x.shape
        queries = self.q_norm(
            self.q_proj(x).reshape(
                batch,
                length,
                self.n_heads,
                self.head_dim,
            )
        ).transpose(0, 2, 1, 3)
        keys, values = self.project_context(x, positions, config)
        queries = _position_rope(queries, positions, config)
        ctx_keys, ctx_values = context_kv
        keys = mx.concatenate((ctx_keys, keys), axis=2)
        values = mx.concatenate((ctx_values, values), axis=2)
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=allowed,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class DParaLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = DParaAttention(config, index)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, context_kv, positions, allowed, config):
        x = x + self.self_attn(
            self.input_layernorm(x), context_kv, positions, allowed, config
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class DParaMarkovHead(nn.Module):
    """Paper's WE(previous token), using an independent low-rank embedding."""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.markov_rank)
        self.projection = nn.Linear(config.markov_rank, config.vocab_size, bias=False)

    def __call__(self, previous):
        return self.projection(self.embedding(previous))


class MDFlashDParaDraftModel(nn.Module):
    """Single-request research implementation, intentionally outside serving.

    No model-name branches, no legacy cache engine, and no target-state mutation.
    ``context`` holds already-projected accepted target features for all layers.
    """

    def __init__(self, config: DParaConfig):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DParaLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.markov_head = DParaMarkovHead(config)

    def _check_binding(self, binding):
        if (
            binding.draft_revision != self.config.backbone_revision
            or binding.target_revision != self.config.target_revision
        ):
            raise ValueError("DPara artifact revision mismatch")

    def _check_features(self, features):
        expected = len(self.config.target_layer_ids) * self.config.hidden_size
        if (
            features.ndim != 3
            or features.shape[0] != 1
            or features.shape[2] != expected
        ):
            raise ValueError("DPara target feature geometry mismatch")

    def _project_features(self, features, offset):
        self._check_features(features)
        if features.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("DPara target features must be floating point")
        if not bool(mx.all(mx.isfinite(features)).item()):
            raise ValueError("DPara committed target features must be finite")
        hidden = self.hidden_norm(self.fc(features))
        positions = mx.arange(offset, offset + features.shape[1], dtype=mx.int32)
        return tuple(
            layer.self_attn.project_context(hidden, positions, self.config)
            for layer in self.layers
        )

    def context(self, target_features, binding):
        self._check_binding(binding)
        layers = self._project_features(target_features, 0)
        return DParaContext(binding, target_features.shape[1], layers)

    def _check_context(self, context):
        self._check_binding(context.binding)
        if (
            type(context.length) is not int
            or context.length < 0
            or len(context.layers) != len(self.layers)
        ):
            raise ValueError("DPara context geometry mismatch")
        expected = (
            1,
            self.config.num_key_value_heads,
            context.length,
            self.config.head_dim,
        )
        if any(
            key.shape != expected or value.shape != expected
            for key, value in context.layers
        ):
            raise ValueError("DPara context KV geometry mismatch")

    def _logits(self, hidden):
        logits = self.lm_head(hidden)
        cap = self.config.final_logit_softcapping
        return mx.tanh(logits / cap) * cap if cap is not None else logits

    def _forward_tree(self, token_ids, positions, allowed, context):
        x = self.embed_tokens(mx.array([token_ids], dtype=mx.int32))
        positions = mx.array(positions, dtype=mx.int32)
        allowed = mx.array(allowed, dtype=mx.bool_)
        for layer, kv in zip(self.layers, context.layers, strict=True):
            x = layer(x, kv, positions, allowed, self.config)
        return self.norm(x)

    def prepare(self, spine, context):
        """Compute every branch once, before acceptance and bonus are known."""
        self._check_context(context)
        spine = tuple(spine)
        if len(spine) != self.config.block_size:
            raise ValueError("DPara spine must match configured block size")
        if any(
            type(token) is not int or not 0 <= token < self.config.vocab_size
            for token in spine
        ):
            raise ValueError("DPara spine token is outside vocabulary")
        d = len(spine) - 1
        allowed, positions = branch_layout(d, context.length)
        ids = (*spine, *((self.config.mask_token_id,) * ((d + 1) * d)))
        x = self._forward_tree(ids, positions, allowed, context)
        hidden = x[0, d + 1 :].reshape(d + 1, d, self.config.hidden_size)
        logits = self._logits(hidden)
        # Lazy graphs alone are not precomputation. Caller chooses the
        # device/stream; force work before the future completes.
        mx.eval(hidden, logits)
        return DParaPrepared(
            context.binding,
            spine,
            context,
            DParaBranches(context.binding, spine, context, hidden, logits),
        )

    def resolve(
        self,
        prepared,
        verification,
        *,
        rng=None,
        temperature=0.0,
        probability_transform=None,
    ):
        """Select r, condition on actual bonus, export exact sequential q.

        The optional transform receives (scores, position, preceding_tokens)
        and must return the probability law actually used for sampling. Returned
        features advance only the accepted prefix; no speculative KV is copied.
        """
        if prepared.binding != prepared.context.binding:
            prepared.discard()
            raise ValueError("DPara prepared handle/context binding mismatch")
        self._check_context(prepared.context)
        if temperature < 0 or not np.isfinite(temperature):
            raise ValueError("DPara temperature must be finite and nonnegative")
        rng = rng or RequestRNG()

        def finalize(branches, outcome):
            if (
                not isinstance(branches, DParaBranches)
                or branches.binding != prepared.binding
                or branches.spine != prepared.spine
                or branches.context is not prepared.context
            ):
                raise ValueError("DPara prepared branch binding/spine/context mismatch")
            if (
                type(outcome.bonus) is not int
                or not 0 <= outcome.bonus < self.config.vocab_size
            ):
                raise ValueError("DPara bonus token is outside vocabulary")
            self._check_features(outcome.target_features)
            if outcome.target_features.shape[1] != len(prepared.spine):
                raise ValueError("DPara verification feature count must match spine")
            # Build prospective committed context before any request RNG draws.
            count = outcome.accepted + 1
            projected = self._project_features(
                outcome.target_features[:, :count], prepared.context.length
            )
            layers = tuple(
                (
                    mx.concatenate((old[0], new[0]), axis=2),
                    mx.concatenate((old[1], new[1]), axis=2),
                )
                for old, new in zip(prepared.context.layers, projected, strict=True)
            )
            next_binding = prepared.binding.next()
            next_context = DParaContext(
                next_binding, prepared.context.length + count, layers
            )
            selected = branches.logits[outcome.accepted]
            previous = outcome.bonus
            proposals, laws = [], []
            snapshot = rng.snapshot()
            try:
                for index in range(selected.shape[0]):
                    corrected = selected[index] + self.markov_head(
                        mx.array(previous, dtype=mx.int32)
                    )
                    mx.eval(corrected)
                    scores = np.asarray(corrected, dtype=np.float64)
                    law = (
                        softmax(scores, temperature=temperature)
                        if probability_transform is None
                        else probability(
                            probability_transform(
                                scores, index, (outcome.bonus, *proposals)
                            )
                        )
                    )
                    if law.shape != (self.config.vocab_size,):
                        raise ValueError(
                            "DPara transformed proposal vocabulary mismatch"
                        )
                    # Greedy mode uses a point mass; transformed laws are sampled
                    # exactly rather than silently changed by the temperature.
                    previous = rng.sample(law)
                    proposals.append(previous)
                    law = law.copy()
                    law.setflags(write=False)
                    laws.append(law)
                mx.eval(*[value for kv in layers for value in kv])
            except BaseException:
                restored = RequestRNG(state=snapshot)
                rng.generator = restored.generator
                rng.draws = restored.draws
                raise
            return DParaContinuation(
                next_binding,
                prepared.spine[1 : outcome.accepted + 1],
                outcome.bonus,
                tuple(proposals),
                tuple(laws),
                next_context,
            )

        return prepared.resolve(verification, finalize)


def load_dpara_artifact(
    path, *, expected_target_revision=None, expected_draft_revision=None
):
    """Load only explicitly converted M-DFlash artifacts with pinned bytes.

    Format is this module's parameter names plus dpara_config compatibility
    marker. This is deliberately not a DSpark checkpoint conversion shortcut.
    The caller still owns model qualification and device selection.
    """
    root = Path(path)
    config_path = root / "config.json"
    config = DParaConfig.from_dict(json.loads(config_path.read_text()))
    if (
        expected_target_revision is not None
        and config.target_revision != expected_target_revision
    ):
        raise ValueError("DPara artifact expected target revision mismatch")
    if (
        expected_draft_revision is not None
        and config.backbone_revision != expected_draft_revision
    ):
        raise ValueError("DPara artifact expected draft revision mismatch")
    manifest = json.loads((root / "dpara-manifest.json").read_text())
    if (
        manifest.get("backbone_revision") != config.backbone_revision
        or manifest.get("target_revision") != config.target_revision
        or set(manifest.get("sha256", {})) != {"config.json", "model.safetensors"}
    ):
        raise ValueError("DPara artifact manifest mismatch")
    for name, expected in manifest["sha256"].items():
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"DPara artifact hash mismatch: {name}")
    model = MDFlashDParaDraftModel(config)
    weights = mx.load(str(root / "model.safetensors"))
    if any(
        weight.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        for weight in weights.values()
    ):
        raise ValueError("DPara artifact weights must be floating point")
    model.load_weights(list(weights.items()), strict=True)
    return model
