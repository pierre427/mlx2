# SPDX-License-Identifier: MIT
"""Default-off 27B routing to the shared Qwen3.5-numerics decode kernels.

See provenance/qwen38-fused-gdn.json. Verify/prefill are intentionally left
on the reference: Flash-Next's kernels have different rounding and gating.
"""
from __future__ import annotations

import mlx.core as mx
from .qwen3_5 import GatedDeltaNet as ReferenceGatedDeltaNet
from .qwen3_next import Qwen3NextRMSNormGated
from .gdn_state import check_state
from . import qwen4_fused_gdn as kernels
from .served_exp import is_device_fault

VERIFY_REFUSAL = 'verify arithmetic incompatible: q/k RMS rounding, fp32 beta, SiLU gate'
PREFILL_REFUSAL = 'prefill arithmetic incompatible: q/k RMS rounding, fp32 beta, SiLU gate'


class GatedDeltaNet(ReferenceGatedDeltaNet):
    """Keep projections, ordinary math, cache and mixed-forward contracts intact."""

    def __init__(self, args):
        super().__init__(args)
        self.fused_gdn_enabled = False
        object.__setattr__(self, 'fused_gdn_counters', {
            'decode_calls': 0, 'batch_decode_calls': 0, 'batch_decode_rows': 0,
            'fallbacks': 0, 'last_fallback': None, 'reasons': {},
        })

    def set_fused_gdn_enabled(self, enabled: bool):
        if type(enabled) is not bool:
            raise ValueError('fused_gdn must be boolean')
        self.fused_gdn_enabled = enabled

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
            return self._fallback(VERIFY_REFUSAL if bool(getattr(cache, 'speculating', False))
                                  else PREFILL_REFUSAL)
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
        geometry = dict(architecture='qwen38', num_key_heads=self.num_k_heads,
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


def configure(model, enabled):
    """Adapter-owned switch; explicit false is also the immediate kill switch."""
    for _, module in model.named_modules():
        if isinstance(module, GatedDeltaNet):
            module.set_fused_gdn_enabled(enabled)


def stats(model):
    report = {'enabled': False, 'qualified': False, 'layers': 0,
              'decode_calls': 0, 'batch_decode_calls': 0, 'batch_decode_rows': 0,
              'fallbacks': 0, 'reasons': {}}
    for _, module in model.named_modules():
        if isinstance(module, GatedDeltaNet):
            report['layers'] += 1
            report['enabled'] |= module.fused_gdn_enabled
            counts = module.fused_gdn_counters
            for name in ('decode_calls', 'batch_decode_calls', 'batch_decode_rows', 'fallbacks'):
                report[name] += counts[name]
            for reason, count in counts['reasons'].items():
                report['reasons'][reason] = report['reasons'].get(reason, 0) + count
    return report
