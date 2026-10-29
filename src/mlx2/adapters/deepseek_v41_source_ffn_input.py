# SPDX-License-Identifier: MIT
"""Frame-owned Source FFN input-QAT operand; no global tensor cache or math.

The existing routed site creates the tensor lazily. Shared gate/up borrows a
reshape of that exact tensor. See source-ffn-input-reuse provenance/audit.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import re
import threading

POLICY = 'shared_routed_qat_reuse'


@lru_cache(maxsize=1)
def implementation_identity():
    own = Path(__file__)
    quant = own.parent.parent / 'runtime/models/deepseek_v41_quant.py'
    return hashlib.sha256((POLICY + ':' + hashlib.sha256(own.read_bytes()).hexdigest()
                           + ':' + hashlib.sha256(quant.read_bytes()).hexdigest()).encode()).hexdigest()


def configuration_supported(args):
    return (getattr(args, 'hidden_size', None) == 5120
            and getattr(args, 'activation_dtype', None) == 'bfloat16'
            and getattr(args, 'simulate_quantization', None) is True)


def _policies(owner):
    return (getattr(owner, 'source_ffn_input_policy', 'reference'),
            getattr(owner, 'expert_math_policy', 'reference'),
            getattr(owner, 'source_dense_qat_policy', 'reference'),
            getattr(owner, 'expert_io_policy', 'blocking'),
            getattr(owner, 'shared_policy', 'reference'))


def _aliases(prefix):
    if type(prefix) is not str or re.fullmatch(r'blk\.(0|[1-9][0-9]?)', prefix) is None:
        return None
    if int(prefix.split('.')[1]) >= 40:
        return None
    return tuple(prefix + '.' + name + '.weight' for name in ('ffn_gate_shexp', 'ffn_up_shexp'))


def _pair_seal(owner, aliases):
    import mlx.core as mx
    result = []
    for name in aliases:
        pair = getattr(owner, '_quant', {}).get(name)
        if not isinstance(pair, tuple) or len(pair) != 2:
            return None
        q, s = pair
        if (not isinstance(q, mx.array) or not isinstance(s, mx.array)
                or q.dtype != mx.uint32 or q.ndim != 2 or q.shape[1] != 1280
                or q.shape[0] <= 0 or s.dtype != mx.uint8 or s.shape != (q.shape[0], 160)):
            return None
        result.append((id(q), id(s), q.shape, s.shape))
    if result[0][2] != result[1][2]:
        return None
    return tuple(result)


def create(owner, raw, prefix):
    """Unsupported factory requests retain ordinary call paths unchanged."""
    import mlx.core as mx
    if getattr(owner, 'source_ffn_input_policy', 'reference') == 'reference':
        return None
    if _policies(owner) != (POLICY, 'routed_fused_fp32', 'fused_fp32', 'blocking', 'reference'):
        raise ValueError('Source FFN input selected configuration changed')
    if not configuration_supported(getattr(owner, 'artifact', {}).get('args')):
        raise ValueError('Source FFN input requires canonical BF16 Source configuration')
    if (not isinstance(raw, mx.array) or raw.dtype != mx.float32 or raw.ndim != 3
            or raw.shape[0] != 1 or raw.shape[1] not in (1, 99) or raw.shape[2] != 5120):
        counts=owner.scheduled_dense_calls
        counts['source_ffn_input_reference_fallback']=counts.get('source_ffn_input_reference_fallback',0)+1
        return None
    aliases = _aliases(prefix)
    if aliases is None:
        counts=owner.scheduled_dense_calls
        counts['source_ffn_input_reference_fallback']=counts.get('source_ffn_input_reference_fallback',0)+1
        return None
    pair_seal = _pair_seal(owner, aliases)
    if pair_seal is None:
        counts=owner.scheduled_dense_calls
        counts['source_ffn_input_reference_fallback']=counts.get('source_ffn_input_reference_fallback',0)+1
        return None
    return FfnInputToken(owner, raw, prefix, aliases, pair_seal)


class FfnInputToken:
    """Unserializable one-frame operand with explicit canonical identities."""
    __slots__ = ('_owner', '_raw', '_flat', '_prefix', '_aliases', '_pair',
                 '_policies', '_thread', '_prepared', '_routed', '_shared', '_closed')

    def __init__(self, owner, raw, prefix, aliases, pair_seal):
        self._owner, self._raw, self._flat = owner, raw, raw.reshape(-1, 5120)
        self._prefix, self._aliases, self._pair = prefix, aliases, pair_seal
        self._policies, self._thread = _policies(owner), threading.get_ident()
        self._prepared = None
        self._routed = self._shared = self._closed = False

    @property
    def raw_input(self):return self._raw

    @property
    def flat_input(self):return self._flat

    @property
    def prefix(self):return self._prefix

    @property
    def aliases(self):return self._aliases

    def _validate(self, owner):
        if self._closed:
            raise ValueError('Source FFN input token is closed')
        if owner is not self._owner or threading.get_ident() != self._thread:
            raise ValueError('Source FFN input token owner/thread mismatch')
        if (_policies(owner) != self._policies or _pair_seal(owner, self.aliases) != self._pair
                or not configuration_supported(getattr(owner, 'artifact', {}).get('args'))
                or getattr(owner, '_closed', False) or getattr(owner, '_parked', False)):
            raise ValueError('Source FFN input token policy/weights/lifetime changed')

    def validate_routed_request(self, owner, raw_flat, prefix):
        self._validate(owner)
        if raw_flat is not self.flat_input or prefix != self.prefix or self._routed or self._shared:
            raise ValueError('Source FFN input routed operand/role mismatch')

    def routed(self, owner, raw_flat, prefix, prepare):
        import mlx.core as mx
        self.validate_routed_request(owner,raw_flat,prefix)
        value = prepare()
        if not isinstance(value, mx.array) or value.dtype != mx.float32 or value.shape != self.flat_input.shape:
            raise ValueError('Source FFN input prepared geometry/dtype mismatch')
        self._prepared, self._routed = value, True
        counts = owner.scheduled_dense_calls
        counts['source_ffn_input_qat_materialized'] = counts.get('source_ffn_input_qat_materialized', 0) + 1
        return value

    def shared(self, owner, raw, gate_name, up_name):
        self._validate(owner)
        if (raw is not self.raw_input or (gate_name, up_name) != self.aliases
                or not self._routed or self._shared or self._prepared is None):
            raise ValueError('Source FFN input shared operand/role mismatch')
        value = self._prepared.reshape(self.raw_input.shape)
        self._shared = True
        counts = owner.scheduled_dense_calls
        counts['source_ffn_input_qat_calls_avoided'] = counts.get('source_ffn_input_qat_calls_avoided', 0) + 1
        return value

    def close(self):
        self._closed = True
        self._owner = self._raw = self._flat = self._prepared = None
