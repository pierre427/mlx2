# SPDX-License-Identifier: MIT
"""Bounded, request-owned hybrid prefill snapshots within one projection pass.

Only the explicitly supported ordinary Qwen3.8 cache law is admitted. Each
GDN core ends at the same requested recurrent boundaries; input/output and
other dense projections remain wide. KV descriptors are trimmed to the exact
prefix. This changes projection geometry relative to split forwards and owns
a separate APCv2 numerical namespace. See provenance/apple-four-gaps.json.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import mlx.core as mx

from .models.cache import ArraysCache, KVCache

ACTIVE = ContextVar("mlx2_inpass_checkpoints", default=None)
MAX_CAPTURES = 4
MAX_RETAINED_BYTES = 512 << 20


class CaptureRefused(ValueError):
    """Pre-mutation refusal; caller may retain the split-boundary path."""


class CaptureTape:
    def __init__(self, trunk, cache, width, positions):
        positions = tuple(positions)
        if (not positions or len(positions) > MAX_CAPTURES
                or any(type(p) is not int or not 0 < p < width for p in positions)
                or positions != tuple(sorted(set(positions)))):
            raise CaptureRefused("in-pass capture requires 1..4 sorted interior positions")
        if trunk.pipeline_size != 1 or len(cache) != len(trunk.layers):
            raise CaptureRefused("in-pass capture requires a complete unsharded trunk")
        if len({id(c) for c in cache}) != len(cache):
            raise CaptureRefused('in-pass cache layers must have distinct owners')
        offsets={c.offset for c in cache if type(c) is KVCache}
        if len(offsets)>1:
            raise CaptureRefused('in-pass KV frontiers are not aligned')
        estimate = 0
        for layer, c in zip(trunk.layers, cache):
            expected = ArraysCache if layer.is_linear else KVCache
            if (type(c) is not expected or getattr(c,"speculating",False)
                    or getattr(c,"lengths",None) is not None
                    or getattr(c,"left_padding",None) is not None):
                raise CaptureRefused("in-pass capture requires ordinary unpadded cache planes")
            if c.committed_descriptor_clone() is None:
                raise CaptureRefused('in-pass cache has an open transaction or unknown descriptor state')
            if layer.is_linear:
                g = layer.linear_attn
                if g.sharding_group is not None or hasattr(g,"_prefill_counts"):
                    raise CaptureRefused("in-pass capture requires ordinary GDN projections")
                estimate += max(c.nbytes, 4*g.num_v_heads*g.head_v_dim*g.head_k_dim
                                + 4*(g.conv_kernel_size-1)*g.conv_dim)
            else:
                attn = layer.self_attn
                estimate += c.nbytes + 2*width*attn.num_key_value_heads*attn.head_dim*4
        if estimate * len(positions) > MAX_RETAINED_BYTES:
            raise CaptureRefused("in-pass checkpoint retained-byte limit exceeded")
        self.positions, self.width = positions, width
        self.by_id = {id(c): i for i,c in enumerate(cache)}
        self.planes = {p: [None]*len(cache) for p in positions}
        self.estimated_retained_bytes = estimate * len(positions)

    @contextmanager
    def scope(self):
        if ACTIVE.get() is not None: raise CaptureRefused("nested in-pass capture")
        token = ACTIVE.set(self)
        try: yield self
        finally: ACTIVE.reset(token)

    def recurrent(self, layer, qkv, z, b, a, mask, cache, dtype):
        if mask is not None: raise ValueError("in-pass recurrent mask changed after admission")
        outputs, start = [], 0
        for end in (*self.positions, self.width):
            span = slice(start,end)
            out = layer._recurrent_core(qkv[:,span],z[:,span],b[:,span],a[:,span],None,cache,dtype=dtype)
            outputs.append(out)
            if end in self.planes: self.record(cache,end)
            start=end
        return mx.concatenate(outputs,axis=1)

    def record(self, cache, position):
        clone = cache.committed_descriptor_clone()
        if clone is None: raise RuntimeError("admitted in-pass cache refused committed snapshot")
        self.planes[position][self.by_id[id(cache)]] = clone

    def record_kv(self, cache):
        for p in self.positions:
            self.record(cache,p)
            saved = self.planes[p][self.by_id[id(cache)]]
            if saved.trim(self.width-p) != self.width-p:
                raise RuntimeError("in-pass KV prefix trim failed")
            # Trim descriptor payloads as well as the offset. Future K/V rows
            # from this wide pass must never enter a prefix checkpoint blob.
            saved.keys=mx.contiguous(saved.keys[...,:saved.offset,:])
            saved.values=mx.contiguous(saved.values[...,:saved.offset,:])

    def finish(self):
        if any(c is None for group in self.planes.values() for c in group):
            raise RuntimeError("in-pass checkpoint is missing a layer")
        # Nothing becomes publishable until every layer completes and all
        # retained tensors evaluate successfully.
        mx.eval([c.state for group in self.planes.values() for c in group])
        return self.planes


def recurrent_capture(layer,qkv,z,b,a,mask,cache,*,dtype):
    tape=ACTIVE.get()
    if tape is None or id(cache) not in tape.by_id: return None
    return tape.recurrent(layer,qkv,z,b,a,mask,cache,dtype)


def record_kv(cache):
    tape=ACTIVE.get()
    if tape is not None and id(cache) in tape.by_id: tape.record_kv(cache)
