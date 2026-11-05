"""Frozen-base LoRA with explicit gradient-safe Kimi suffix execution.

See provenance/kimi-linear-lora.json. No loader, scheduler, tokenizer, training
loop, or model-weight writer lives here. Artifacts are immutable weight programs.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from ...runtime.models.kimi_linear import KimiDeltaAttention, Model
from ...runtime.models import moe_nax_gather
from ..weight_space.programs import MAX_TARGETS


SAFE_MODULES = frozenset({
    'attention.q_proj', 'attention.k_proj', 'attention.v_proj', 'attention.o_proj',
    'attention.kv_a_proj_with_mqa', 'attention.f_a_proj', 'attention.f_b_proj',
    'attention.g_a_proj', 'attention.g_b_proj', 'attention.b_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
    'mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj', 'mlp.shared_experts.down_proj',
})
TARGET = re.compile(r'model\.layers\.(\d+)\.(.+)\Z')


def _validate_target_capacity(keys):
    if len(keys) > MAX_TARGETS:
        raise ValueError(
            f"Weight programs support at most {MAX_TARGETS} LoRA targets; got {len(keys)}"
        )


def select_targets(model, *, last_n_layers=1, modules=('attention.o_proj', 'attention.q_proj')):
    if type(last_n_layers) is not int or not 1 <= last_n_layers <= len(model.layers):
        raise ValueError('last_n_layers must select a nonempty model suffix')
    if not modules or set(modules) - SAFE_MODULES or len(set(modules)) != len(modules):
        raise ValueError('Select unique supported callable projection suffixes')
    named = dict(model.named_modules())
    keys = tuple(f'model.layers.{i}.{suffix}' for i in range(len(model.layers) - last_n_layers, len(model.layers))
                 for suffix in modules if f'model.layers.{i}.{suffix}' in named)
    if not keys:
        raise ValueError('Requested module suffixes do not occur in selected layers')
    _validate_target_capacity(keys)
    return keys


class TrainingLoRALinear(nn.Module):
    """Match serving weight-program arithmetic; scale is a direct multiplier."""

    def __init__(self, base, rank, scale):
        super().__init__()
        self.linear = base
        out_dim, in_dim = base.weight.shape
        if isinstance(base, nn.QuantizedLinear):
            in_dim = in_dim * 32 // base.bits
        bound = 1 / math.sqrt(in_dim)
        self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(in_dim, rank), dtype=mx.float32)
        self.lora_b = mx.zeros((rank, out_dim), dtype=mx.float32)
        self.scale = scale

    def __call__(self, x):
        y = self.linear(x)
        update = (x.astype(mx.float32) @ self.lora_a) @ self.lora_b
        return (y.astype(mx.float32) + self.scale * update).astype(y.dtype)


def masked_cross_entropy(logits, targets, loss_mask):
    """Targets and mask are already shifted once by the caller."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or loss_mask.shape != targets.shape:
        raise ValueError('Expected logits[B,T,V], targets[B,T], loss_mask[B,T]')
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction='none')
    mask = loss_mask.astype(mx.float32)
    # The host-side data validator must reject empty masks before tracing.
    return (losses * mask).sum() / mx.maximum(mask.sum(), 1)


def chunked_head_cross_entropy(head, hidden, targets, loss_mask, *, chunk_size):
    """Checkpoint frozen-head CE numerators; normalize once over the full mask.

    Only the vocabulary projection is split. Every hidden row was computed
    with its complete causal context. Checkpointing allows the vocabulary-sized
    intermediates to be recomputed during backward instead of retained, but
    actual allocator scheduling and peak savings require a device measurement.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('projection chunk size must be a positive integer')
    if (hidden.ndim != 3 or not hidden.shape[1] or targets.shape != hidden.shape[:2]
            or loss_mask.shape != targets.shape):
        raise ValueError('Expected hidden[B,T,D], targets[B,T], loss_mask[B,T]')
    if tree_flatten(head.trainable_parameters()):
        raise ValueError('Chunked projection requires a frozen language-model head')
    mask = loss_mask.astype(mx.float32)

    def numerator(value, labels, weights):
        # checkpoint traces each argument for its VJP; integer target indices
        # are observations and must never request a gather-index derivative.
        losses = nn.losses.cross_entropy(head(value).astype(mx.float32),
                                        mx.stop_gradient(labels), reduction='none')
        return (losses * weights).sum()

    projected_loss = mx.checkpoint(numerator)
    total = mx.array(0, dtype=mx.float32)
    for start in range(0, hidden.shape[1], chunk_size):
        stop = min(start + chunk_size, hidden.shape[1])
        total = total + projected_loss(hidden[:, start:stop], targets[:, start:stop], mask[:, start:stop])
    # Match the dense loss's zero-mask law. Training data validation separately
    # refuses empty assistant masks before tracing a real update.
    return total / mx.maximum(mask.sum(), 1)


class LoRATrainingSession:
    def __init__(self, model, keys, *, rank=4, scale=1.0, base_checkpoint, source_revision, base_paths,
                 kda_policy='reference', mla_policy='expanded', projection_chunk_size=None):
        if not isinstance(model, Model):
            raise ValueError('This training entry requires the mlx2 Kimi Linear model')
        if (type(rank) is not int or not 1 <= rank <= 1024 or isinstance(scale, bool)
                or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
            raise ValueError('A positive rank and finite direct LoRA scale are required')
        if not base_checkpoint or not source_revision or not base_paths:
            raise ValueError('Exact checkpoint, source revision and protected base paths are required')
        if kda_policy not in {'reference', 'chunk_wy'}:
            raise ValueError('Training KDA policy must be reference or chunk_wy')
        if not isinstance(mla_policy, str) or mla_policy not in {'expanded', 'padded_sdpa'}:
            raise ValueError('Training MLA policy must be expanded or padded_sdpa')
        if projection_chunk_size is not None and (
                type(projection_chunk_size) is not int or projection_chunk_size < 1):
            raise ValueError('projection chunk size must be a positive integer or None')
        if getattr(model, '_kimi_lora_training_session', None) is not None:
            raise ValueError('A Kimi LoRA training session is already active')
        if moe_nax_gather.status()['mode'] != 'off':
            raise ValueError('Training requires moe_nax=off; custom gather has no verified backward')
        keys = tuple(keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError('Explicit unique projection keys are required')
        _validate_target_capacity(keys)
        named = dict(model.named_modules())
        indices = []
        for key in keys:
            match = TARGET.fullmatch(key)
            if not match or match[2] not in SAFE_MODULES or not isinstance(named.get(key), (nn.Linear, nn.QuantizedLinear)):
                raise ValueError(f'Unsupported training/serving projection: {key}')
            indices.append(int(match[1]))
        self.model, self.keys, self.rank, self.scale = model, keys, rank, float(scale)
        self.kda_policy = kda_policy
        self.mla_policy, self.projection_chunk_size = mla_policy, projection_chunk_size
        self.base_checkpoint, self.source_revision = base_checkpoint, source_revision
        self.base_paths = tuple(Path(p).expanduser().resolve() for p in base_paths)
        self.first_layer = min(indices)
        for key, module in named.items():
            match = TARGET.fullmatch(key)
            if key == 'lm_head' or (match and int(match[1]) >= self.first_layer):
                if isinstance(module, nn.QuantizedLinear) and (
                        type(module) is not nn.QuantizedLinear or getattr(module, '_lane_tiled', False)):
                    raise ValueError(f'Training requires stock, untiled quantized projections: {key}')
        self.originals = {key: named[key] for key in keys}
        self.module_state = [(module, set(module._no_grad), module.training) for _, module in model.named_modules()]
        self.policies = []
        self.closed = False
        replacements = [(key, TrainingLoRALinear(named[key], rank, self.scale)) for key in keys]
        try:
            model.update_modules(tree_unflatten(replacements))
            model.freeze()
            model.train()
            for index, layer in enumerate(model.layers):
                if index < self.first_layer:
                    layer.eval()
                    continue
                attn = layer.attention
                settings = ({'reference': True, 'chunk32': False, 'chunk_wy': kda_policy == 'chunk_wy',
                             'fused_projections': False, 'fused_norm': False}
                            if isinstance(attn, KimiDeltaAttention) else {
                                'policy': 'expanded', 'flash': 'off', 'training_mla_policy': mla_policy})
                self.policies.append((attn, {k: getattr(attn, k) for k in settings}))
                for key, value in settings.items():
                    setattr(attn, key, value)
                gate = getattr(layer.mlp, 'gate', None)
                if gate is not None:
                    self.policies.append((gate, {'training_indices': gate.training_indices}))
                    gate.training_indices = True
            named = dict(model.named_modules())
            for key in keys:
                named[key].unfreeze(recurse=False, keys=['lora_a', 'lora_b'])
            expected = {f'{key}.{leaf}' for key in keys for leaf in ('lora_a', 'lora_b')}
            # MLX freeze() excludes array leaves, but preserves dictionary
            # skeletons, including integer-valued route counters. New counters
            # would then change the gradient tree after Adam initializes.
            # Admit only ancestors of the explicit factors, freezing every
            # other module member as a whole (including telemetry dictionaries).
            # close() restores each original module's previous freeze set.
            for prefix, module in named.items():
                start = prefix + '.' if prefix else ''
                allowed = {key[len(start):].split('.', 1)[0] for key in expected
                           if key.startswith(start)}
                module._no_grad.update(set(module) - allowed)
            if set(dict(tree_flatten(model.trainable_parameters()))) != expected:
                raise ValueError('Training parameter coverage differs from explicit adapter factors')
            object.__setattr__(model, '_kimi_lora_training_session', self)
        except BaseException:
            self.close()
            raise

    def forward_hidden(self, input_ids, *, checkpoint_layers=True):
        """Return final normalized hidden states for every causal input row."""
        if self.closed or getattr(self.model, '_kimi_lora_training_session', None) is not self:
            raise ValueError('Training session is not active')
        if input_ids.ndim != 2 or not input_ids.shape[1]:
            raise ValueError('Training requires nonempty input_ids[B,T]')
        model = self.model
        with moe_nax_gather.forward_scope(input_ids, None):
            hidden = model.model.word_embeddings(input_ids)
            mask = model.model.attention_mask(hidden, None)
            for index, layer in enumerate(model.layers):
                if index == self.first_layer:
                    # No trainable factor precedes this boundary. The frozen
                    # prefix may use inference kernels without their VJPs.
                    hidden = mx.stop_gradient(hidden)
                layer_mask = mask if layer.is_global else None
                if checkpoint_layers and index >= self.first_layer:
                    def apply(params, value, block=layer, attention_mask=layer_mask):
                        block.update(params)
                        return block(value, mask=attention_mask, cache=None)
                    hidden = mx.checkpoint(apply)(layer.trainable_parameters(), hidden)
                else:
                    hidden = layer(hidden, mask=layer_mask, cache=None)
            return model.model.norm(hidden)

    def forward(self, input_ids, *, checkpoint_layers=True):
        # Bypass serving's ambient prefill checkpoint-row selection: every
        # causal token receives a logit row for the explicitly masked loss.
        return self.model.lm_head(self.forward_hidden(input_ids, checkpoint_layers=checkpoint_layers))

    def loss(self, input_ids, targets, loss_mask, *, checkpoint_layers=True):
        """Aligned input rows and already-shifted targets/mask; no internal shift."""
        if self.projection_chunk_size is None:
            return masked_cross_entropy(self.forward(input_ids, checkpoint_layers=checkpoint_layers),
                                        targets, loss_mask)
        hidden = self.forward_hidden(input_ids, checkpoint_layers=checkpoint_layers)
        return chunked_head_cross_entropy(self.model.lm_head, hidden, targets, loss_mask,
                                         chunk_size=self.projection_chunk_size)

    def weights(self):
        if self.closed:
            raise ValueError('Training session is closed')
        parameters = dict(tree_flatten(self.model.trainable_parameters()))
        return {f'{key}.{leaf}': parameters[f'{key}.{leaf}'] for key in self.keys for leaf in ('lora_a', 'lora_b')}

    def restore_weights(self, weights):
        expected = self.weights()
        if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in weights):
            raise ValueError('Adapter snapshot coverage or shape mismatch')
        self.model.update(tree_unflatten(list(weights.items())))

    def export_program(self, output_dir, provenance):
        import numpy as np
        from ..weight_space.programs import create_program, export_program

        path = Path(output_dir).expanduser().resolve()
        if any(path == base or path.is_relative_to(base) or base.is_relative_to(path) for base in self.base_paths):
            raise ValueError('Adapter export may not write into or replace a protected base path')
        weights = self.weights()
        mx.eval(weights)
        factors = {key: (np.array(weights[f'{key}.lora_a']), np.array(weights[f'{key}.lora_b']) * self.scale)
                   for key in self.keys}
        program = create_program(factors, checkpoint=self.base_checkpoint, source_revision=self.source_revision,
                                 construction='kimi_linear_supervised_lora',
                                 provenance={**provenance, 'training_direct_scale': self.scale,
                                             'scale_export': 'folded_into_b', 'frozen_prefix_layers': self.first_layer,
                                             'training_kda_policy': self.kda_policy,
                                             'training_mla_policy': self.mla_policy,
                                             'projection_chunk_size': self.projection_chunk_size,
                                             'training_arithmetic': 'fp32_delta_round_once_v1'})
        export_program(program, path)
        return program

    def close(self):
        if self.closed:
            return
        self.model.update_modules(tree_unflatten(list(self.originals.items())))
        for module, settings in self.policies:
            for key, value in settings.items():
                setattr(module, key, value)
        for module, no_grad, training in self.module_state:
            module._no_grad = set(no_grad)
            module.train(training)
        if getattr(self.model, '_kimi_lora_training_session', None) is self:
            object.__delattr__(self.model, '_kimi_lora_training_session')
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
