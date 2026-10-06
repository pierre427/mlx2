# SPDX-License-Identifier: MIT
"""Default-off 27B routing to the shared Qwen3.5-numerics Metal kernels.

See provenance/qwen38-fused-gdn.json and
provenance/qwen38-fused-gdn-multitoken.json. Unsupported multi-token shapes
remain on the reference path.
"""
from __future__ import annotations

import mlx.core as mx
from .qwen3_5 import GatedDeltaNet as ReferenceGatedDeltaNet
from .qwen3_next import Qwen3NextRMSNormGated
from .gdn_state import check_state
from . import qwen4_fused_gdn as kernels
from . import qwen4_fused_gdn_verify as verify_kernels
from .served_exp import is_device_fault

VERIFY_REFUSAL = 'verify shape outside corrected fused path'
PREFILL_REFUSAL = 'prefill shape outside corrected fused path'


class GatedDeltaNet(ReferenceGatedDeltaNet):
    """Keep projections, ordinary math, cache and mixed-forward contracts intact."""

    def __init__(self, args):
        super().__init__(args)
        self.fused_gdn_enabled = False
        self.fused_gdn_architecture = 'qwen38'
        object.__setattr__(self, 'fused_gdn_counters', {
            'decode_calls': 0, 'batch_decode_calls': 0, 'batch_decode_rows': 0,
            'verify_calls': 0, 'verify_tokens': 0, 'verify_rollbacks': 0,
            'prefill_calls': 0, 'prefill_tokens': 0,
            'fallbacks': 0, 'last_fallback': None, 'reasons': {},
        })

    def set_fused_gdn_enabled(self, enabled: bool):
        if type(enabled) is not bool:
            raise ValueError('fused_gdn must be boolean')
        self.fused_gdn_enabled = enabled

    def set_fused_gdn_architecture(self, architecture: str):
        if architecture not in ('qwen35', 'qwen38'):
            raise ValueError('unsupported fused_gdn architecture')
        self.fused_gdn_architecture = architecture

    def _fallback(self, reason):
        counts = self.fused_gdn_counters
        counts['fallbacks'] += 1
        counts['last_fallback'] = reason
        if reason not in counts['reasons'] and len(counts['reasons']) >= 32:
            reason = 'other'
        counts['reasons'][reason] = counts['reasons'].get(reason, 0) + 1
        return None

    def _try_fused_decode(self, qkv, z, b, a, mask, cache):
        if not self.fused_gdn_enabled:
            return None
        if qkv.shape[1] != 1:
            return self._try_fused_multitoken(qkv, z, b, a, mask, cache)
        if cache is None or cache[0] is None or cache[1] is None:
            return self._fallback('uninitialized cache')
        check_state(cache[1], getattr(self, '_gdn_state_dtype', None))
        if type(self.norm) is not Qwen3NextRMSNormGated:
            return self._fallback('unsupported gated norm')
        describe = getattr(cache, 'rollback_spans', None)
        spans = describe(1, mask) if callable(describe) else None
        rows = int(qkv.shape[0])
        admit = (kernels.admit_qwen4_fused_gdn_decode if rows == 1
                 else kernels.admit_qwen4_fused_gdn_batch_decode)
        geometry = dict(architecture=self.fused_gdn_architecture, num_key_heads=self.num_k_heads,
            num_value_heads=self.num_v_heads, key_head_dim=self.head_k_dim,
            value_head_dim=self.head_v_dim, conv_kernel=self.conv_kernel_size)
        admission = admit(qkv=qkv, z=z, b=b, a=a, conv_state=cache[0],
            recurrent_state=cache[1], conv_weight=self.conv1d.weight,
            A_log=self.A_log, dt_bias=self.dt_bias, norm_weight=self.norm.weight,
            mask=mask, spans=spans, speculating=bool(getattr(cache, 'speculating', False)),
            training=bool(self.training), sharded=self.sharding_group is not None,
            gate_activation='swish', **geometry)
        if not admission.accepted:
            return self._fallback(admission.reason)
        if not kernels.fused_gdn_runtime_supported():
            return self._fallback('Metal runtime unavailable')
        refusal = kernels.served_silu_refusal()
        if refusal is not None:
            return self._fallback(refusal)
        try:
            ty = kernels.probe_qwen4_fused_gdn_decode(qkv.dtype,
                **({'state_dtype': mx.float16} if cache[1].dtype == mx.float16 else {}))
            if ty is None:
                return self._fallback('Metal kernel probe declined')
            build = (kernels.qwen4_fused_gdn_decode if rows == 1
                     else kernels.qwen4_fused_gdn_batch_decode)
            out, conv, state = build(qkv, z, b, a, cache[0], self.conv1d.weight,
                self.A_log, self.dt_bias, cache[1], self.norm.weight, self.norm.eps,
                threadgroup_y=ty, **geometry)
            output = self.out_proj(out)
        except Exception as exc:
            if is_device_fault(exc):
                raise
            return self._fallback(f'Metal kernel dispatch failed: {type(exc).__name__}')
        cache[0], cache[1] = conv, state
        cache.advance(1)
        counts = self.fused_gdn_counters
        counts['decode_calls' if rows == 1 else 'batch_decode_calls'] += 1
        if rows > 1:
            counts['batch_decode_rows'] += rows
        counts['last_fallback'] = None
        return output

    def _try_fused_multitoken(self, qkv, z, b, a, mask, cache):
        """Fuse an exact B=1 block; speculative blocks retain rollback points."""
        speculating = bool(getattr(cache, 'speculating', False))
        refusal = VERIFY_REFUSAL if speculating else PREFILL_REFUSAL
        if cache is None or cache[0] is None or cache[1] is None:
            return self._fallback('uninitialized cache')
        check_state(cache[1], getattr(self, '_gdn_state_dtype', None))
        if type(self.norm) is not Qwen3NextRMSNormGated:
            return self._fallback('unsupported gated norm')
        describe = getattr(cache, 'rollback_spans', None)
        record = getattr(cache, 'record_rollback', None)
        if not callable(describe) or (speculating and not callable(record)):
            return self._fallback('cache lacks rollback geometry')
        steps = int(qkv.shape[1])
        spans = describe(steps, mask)
        geometry = dict(
            architecture=self.fused_gdn_architecture,
            num_key_heads=self.num_k_heads,
            num_value_heads=self.num_v_heads,
            key_head_dim=self.head_k_dim,
            value_head_dim=self.head_v_dim,
            conv_kernel=self.conv_kernel_size,
        )
        admission = verify_kernels.admit_qwen4_fused_gdn_verify(
            qkv=qkv, z=z, b=b, a=a, conv_state=cache[0],
            recurrent_state=cache[1], conv_weight=self.conv1d.weight,
            A_log=self.A_log, dt_bias=self.dt_bias, norm_weight=self.norm.weight,
            mask=mask, spans=spans, speculating=speculating,
            catchup=not speculating, training=bool(self.training),
            sharded=self.sharding_group is not None, gate_activation='swish',
            **geometry,
        )
        if not admission.accepted:
            return self._fallback(admission.reason or refusal)
        if not kernels.fused_gdn_runtime_supported():
            return self._fallback('Metal runtime unavailable')
        silu_refusal = kernels.served_silu_refusal()
        if silu_refusal is not None:
            return self._fallback(silu_refusal)
        state_kw = ({'state_dtype': mx.float16}
                    if cache[1].dtype == mx.float16 else {})
        probe = (verify_kernels.probe_qwen4_fused_gdn_verify if speculating
                 else verify_kernels.probe_qwen4_fused_gdn_catchup)
        build = (verify_kernels.qwen4_fused_gdn_verify if speculating
                 else verify_kernels.qwen4_fused_gdn_catchup)
        try:
            ty = probe(qkv.dtype, steps, architecture=self.fused_gdn_architecture,
                       num_value_heads=self.num_v_heads, **state_kw)
            if ty is None:
                return self._fallback('Metal kernel probe declined')
            outputs = build(
                qkv, z, b, a, cache[0], self.conv1d.weight, self.A_log,
                self.dt_bias, cache[1], self.norm.weight, self.norm.eps,
                threadgroup_y=ty, architecture=self.fused_gdn_architecture,
                num_value_heads=self.num_v_heads,
            )
            out, conv, state = outputs[:3]
        except Exception as exc:
            if is_device_fault(exc):
                raise
            return self._fallback(f'Metal kernel dispatch failed: {type(exc).__name__}')
        if speculating:
            state_snapshots, conv_snapshots = outputs[3:]

            def rollback(m, conv=conv_snapshots, state=state_snapshots):
                self.fused_gdn_counters['verify_rollbacks'] += 1
                return [mx.contiguous(conv[:, m - 1]), state[:, m - 1]]

            record(steps, rollback, [cache[0], cache[1]])
        cache[0], cache[1] = conv, state
        cache.advance(steps)
        counts = self.fused_gdn_counters
        kind = 'verify' if speculating else 'prefill'
        counts[f'{kind}_calls'] += 1
        counts[f'{kind}_tokens'] += steps
        counts['last_fallback'] = None
        return self.out_proj(out)


def configure(model, enabled, *, architecture='qwen38'):
    """Adapter-owned switch; explicit false is also the immediate kill switch."""
    if enabled:
        # Prompt lookup's eight drafts produce a nine-token target block. The
        # shared kernel is parity-gated through 17; select that full bound for
        # this adapter-owned route instead of its historical Qwen4 default 8.
        verify_kernels.set_verify_max_steps(
            verify_kernels.MAX_VERIFY_WIDTH_PROVEN
        )
    for _, module in model.named_modules():
        if isinstance(module, GatedDeltaNet):
            module.set_fused_gdn_architecture(architecture)
            module.set_fused_gdn_enabled(enabled)


def stats(model):
    report = {'enabled': False, 'qualified': False, 'layers': 0,
              'decode_calls': 0, 'batch_decode_calls': 0, 'batch_decode_rows': 0,
              'verify_calls': 0, 'verify_tokens': 0, 'verify_rollbacks': 0,
              'prefill_calls': 0, 'prefill_tokens': 0,
              'fallbacks': 0, 'reasons': {}}
    for _, module in model.named_modules():
        if isinstance(module, GatedDeltaNet):
            report['layers'] += 1
            report['enabled'] |= module.fused_gdn_enabled
            counts = module.fused_gdn_counters
            for name in ('decode_calls', 'batch_decode_calls', 'batch_decode_rows',
                         'verify_calls', 'verify_tokens', 'verify_rollbacks',
                         'prefill_calls', 'prefill_tokens', 'fallbacks'):
                report[name] += counts[name]
            for reason, count in counts['reasons'].items():
                report['reasons'][reason] = report['reasons'].get(reason, 0) + count
    return report
