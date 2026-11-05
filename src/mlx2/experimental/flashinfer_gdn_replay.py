# SPDX-License-Identifier: Apache-2.0
"""Output-only GDN verify and accepted-prefix fold; CPU and native candidates.

The NumPy executor is an executable scalar FP32 law, not a claim of equality
to MLX GDN, CuTe WY or TF32. The checkpoint is frozen during verify. Tape holds
only keys, post-beta corrections and decays, not per-token state snapshots or
a final recurrent-state output. Native staging is lazy and opt-in; no import
of MLX, Torch or device initialization occurs when importing this module.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np

CPU_LAW = "gdn-output-only-cpu-f32-separate-v1"
NATIVE_LAW = "gdn-output-only-metal-f32-no-contract-v1"


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        owner = getattr(self, "_owner", self)
        with owner._lock:
            return method(self, *args, **kwargs)

    return call


def _epoch(value):
    if type(value) is not int or not 0 <= value < (1 << 63):
        raise ValueError("epoch requires a nonnegative signed 64-bit integer")
    return value


def _array(value, dtype, name):
    if type(value) is not np.ndarray or value.dtype != dtype:
        raise ValueError(f"{name} requires CPU NumPy {np.dtype(dtype)}")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _owned(value):
    value = np.array(value, copy=True, order="C")
    value.flags.writeable = False
    return value


@dataclass(frozen=True)
class ReplayBinding:
    checkpoint: str
    source_revision: str
    contract_revision: str

    def __post_init__(self):
        if not isinstance(self.checkpoint, str) or not re.fullmatch(
            r"[^@]+@[0-9a-f]{64}", self.checkpoint
        ):
            raise ValueError("checkpoint requires artifact@SHA256")
        if not isinstance(self.source_revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", self.source_revision
        ):
            raise ValueError("source_revision requires full commit")
        if not isinstance(self.contract_revision, str) or not self.contract_revision:
            raise ValueError("contract_revision is required")


def state_element_offset(shape, layer, slot, head, value, key, *, layout="value_key"):
    """64-bit host counterpart of native addressing; no pool allocation."""
    if (
        not isinstance(shape, tuple)
        or len(shape) != 5
        or any(type(x) is not int or x <= 0 for x in shape)
    ):
        raise ValueError(
            "state shape must be positive [layers, slots, heads, value, key]"
        )
    for index, limit in zip((layer, slot, head, value, key), shape):
        if type(index) is not int or not 0 <= index < limit:
            raise ValueError("state coordinate out of range")
    _layers, slots, heads, values, keys = shape
    base = ((layer * slots + slot) * heads + head) * values * keys
    if layout == "value_key":
        offset = base + value * keys + key
    elif layout == "key_value":
        offset = base + key * values + value
    else:
        raise ValueError("unsupported recurrent layout")
    if offset > (1 << 63) - 1 or offset * 4 > (1 << 63) - 1:
        raise ValueError("state address exceeds signed 64-bit byte range")
    return offset


def _dot(left, right):
    result = np.float32(0)
    for index in range(len(left)):
        result = np.float32(result + np.float32(left[index] * right[index]))
    return result


def _step(state, key, value, decay, beta):
    """One separate-multiply/add update, returning state and post-beta delta."""
    result = np.empty_like(state)
    delta = np.empty(len(value), dtype=np.float32)
    for row in range(len(value)):
        decayed = np.asarray(state[row] * decay, dtype=np.float32)
        delta[row] = np.float32(np.float32(value[row] - _dot(decayed, key)) * beta)
        result[row] = np.asarray(
            decayed + np.asarray(key * delta[row], dtype=np.float32), dtype=np.float32
        )
    return result, delta


class OutputOnlyReplay:
    """One model-owned pool and at most one bounded verification transaction.

    The CPU pool owns its FP32 arrays. Optional transposed physical layout only
    changes address mapping; it does not reassociate recurrence arithmetic.
    There is no convolution, attention-KV or APCv2 publication in this API.
    A serving adapter must commit those planes in its existing transaction.
    """

    def __init__(
        self,
        state,
        binding: ReplayBinding,
        *,
        enabled=False,
        layout="value_key",
        max_steps=17,
        max_tape_bytes=64 << 20,
    ):
        if enabled is not True:
            raise ValueError("output-only replay requires explicit enabled=True")
        if not isinstance(binding, ReplayBinding):
            raise TypeError("binding requires ReplayBinding")
        state = _array(state, np.dtype("float32"), "state")
        if state.ndim != 5 or not all(state.shape):
            raise ValueError("state must be [layers, slots, heads, value, key]")
        if type(max_steps) is not int or not 1 <= max_steps <= 32:
            raise ValueError("max_steps must be 1..32")
        if type(max_tape_bytes) is not int or max_tape_bytes <= 0:
            raise ValueError("max_tape_bytes must be positive")
        state_element_offset(tuple(state.shape), 0, 0, 0, 0, 0, layout=layout)
        self.binding, self.layout = binding, layout
        self.shape = tuple(state.shape)
        self.max_steps, self.max_tape_bytes = max_steps, max_tape_bytes
        self._state = np.array(
            state if layout == "value_key" else state.swapaxes(-1, -2),
            copy=True,
            order="C",
        )
        self.epoch = 0
        self._pending = None
        self._lock = threading.RLock()

    @_serialized
    def state(self):
        return np.array(
            self._state if self.layout == "value_key" else self._state.swapaxes(-1, -2),
            copy=True,
        )

    @_serialized
    def invalidate(self):
        """Invalidate outstanding work when the owning model changes epoch."""
        self.epoch += 1

    @_serialized
    def verify(
        self, query, keys, values, decay, beta, slots, lengths, *, binding, epoch
    ):
        _epoch(epoch)
        if binding != self.binding or epoch != self.epoch:
            raise ValueError("replay identity or epoch is stale")
        if self._pending is not None:
            raise ValueError("previous replay transaction is still owned")
        for name, array in (
            ("query", query),
            ("keys", keys),
            ("values", values),
            ("decay", decay),
            ("beta", beta),
        ):
            _array(array, np.dtype("float32"), name)
        slots, lengths = (
            _owned(_array(slots, np.dtype("int64"), "slots")),
            _owned(_array(lengths, np.dtype("int64"), "lengths")),
        )
        layers, pool_slots, heads, dv, dk = self.shape
        if query.ndim != 5:
            raise ValueError("query must be [layers,batch,steps,key_heads,key_dim]")
        l, batch, steps, key_heads, dim = query.shape
        if (
            l != layers
            or dim != dk
            or not 1 <= steps <= self.max_steps
            or not key_heads
            or heads % key_heads
            or keys.shape != query.shape
            or values.shape != (layers, batch, steps, heads, dv)
            or decay.shape != (layers, batch, steps, heads)
            or beta.shape != decay.shape
            or slots.shape != (batch,)
            or lengths.shape != (batch,)
        ):
            raise ValueError("replay input geometry is inconsistent")
        if (
            np.any(slots < -1)
            or np.any(slots >= pool_slots)
            or np.any(lengths < 0)
            or np.any(lengths > steps)
        ):
            raise ValueError("state slot or valid length out of range")
        live = slots[slots >= 0]
        if len(set(map(int, live))) != len(live):
            raise ValueError("live state slots must be unique")
        if (
            np.any(decay < 0)
            or np.any(decay > 1)
            or np.any(beta < 0)
            or np.any(beta > 1)
        ):
            raise ValueError("decay and beta must be probabilities")
        tape_bytes = (
            keys.nbytes + values.nbytes + decay.nbytes + slots.nbytes + lengths.nbytes
        )
        if tape_bytes > self.max_tape_bytes:
            raise ValueError("replay tape exceeds bound")
        owned_keys, owned_decay = _owned(keys), _owned(decay)
        corrections, outputs = np.zeros_like(values), np.zeros_like(values)
        canonical = (
            self._state if self.layout == "value_key" else self._state.swapaxes(-1, -2)
        )
        for layer in range(layers):
            for row, slot in enumerate(slots):
                if slot < 0:
                    continue
                work = np.array(canonical[layer, slot], copy=True)
                for token in range(int(lengths[row])):
                    for head in range(heads):
                        kh = head // (heads // key_heads)
                        work[head], corrections[layer, row, token, head] = _step(
                            work[head],
                            owned_keys[layer, row, token, kh],
                            values[layer, row, token, head],
                            owned_decay[layer, row, token, head],
                            beta[layer, row, token, head],
                        )
                        for value_index in range(dv):
                            outputs[layer, row, token, head, value_index] = _dot(
                                work[head, value_index], query[layer, row, token, kh]
                            )
        if not np.isfinite(corrections).all() or not np.isfinite(outputs).all():
            raise ValueError("output-only verify overflow")
        transaction = ReplayTransaction(
            self,
            self.epoch,
            owned_keys,
            _owned(corrections),
            owned_decay,
            _owned(slots),
            _owned(lengths),
            _owned(outputs),
            tape_bytes,
        )
        self._pending = transaction
        return transaction


class ReplayTransaction:
    def __init__(
        self,
        owner,
        epoch,
        keys,
        corrections,
        decay,
        slots,
        lengths,
        outputs,
        tape_bytes,
    ):
        self._owner, self.epoch = owner, epoch
        self._keys, self._corrections, self._decay = keys, corrections, decay
        self._slots, self._lengths = slots, lengths
        self.outputs, self.tape_bytes, self.closed = outputs, tape_bytes, False

    def receipt(self):
        return {
            "schema": "mlx2.output-only-gdn.cpu.v1",
            "law": CPU_LAW,
            "checkpoint": self._owner.binding.checkpoint,
            "source_revision": self._owner.binding.source_revision,
            "contract_revision": self._owner.binding.contract_revision,
            "epoch": self.epoch,
            "implemented": True,
            "selected": True,
            "qualified": False,
            "observed_used": True,
            "device": "cpu",
            "native_observed_used": False,
            "native_validated": False,
            "state_written_during_verify": False,
            "final_state_output": False,
            "tape_bytes": self.tape_bytes,
            "apcv2_publication": False,
            "layout": self._owner.layout,
            "closed": self.closed,
        }

    @_serialized
    def cancel(self):
        if self._owner._pending is self:
            self._owner._pending = None
        self.closed = True

    @_serialized
    def commit(self, accepted, *, binding, epoch):
        _epoch(epoch)
        owner = self._owner
        if (
            self.closed
            or owner._pending is not self
            or binding != owner.binding
            or epoch != self.epoch
            or epoch != owner.epoch
        ):
            raise ValueError("replay transaction identity or epoch is stale")
        accepted = _owned(_array(accepted, np.dtype("int64"), "accepted"))
        if (
            accepted.shape != self._lengths.shape
            or np.any(accepted < 0)
            or np.any(accepted > self._lengths)
        ):
            raise ValueError("accepted prefix exceeds valid tape")
        canonical = (
            owner._state
            if owner.layout == "value_key"
            else owner._state.swapaxes(-1, -2)
        )
        # Stage only live slots. Every validation/reconstruction completes before
        # the owner publishes any slot; null/zero-accept slots are not touched.
        staged = []
        _, _, heads, dv, _ = owner.shape
        key_heads = self._keys.shape[3]
        for row, slot in enumerate(self._slots):
            if slot < 0 or accepted[row] == 0:
                continue
            work = np.array(canonical[:, slot], copy=True)
            for layer in range(owner.shape[0]):
                for token in range(int(accepted[row])):
                    for head in range(heads):
                        key = self._keys[
                            layer, row, token, head // (heads // key_heads)
                        ]
                        g = self._decay[layer, row, token, head]
                        correction = self._corrections[layer, row, token, head]
                        for value_index in range(dv):
                            work[layer, head, value_index] = np.asarray(
                                np.asarray(
                                    work[layer, head, value_index] * g, dtype=np.float32
                                )
                                + np.asarray(
                                    key * correction[value_index], dtype=np.float32
                                ),
                                dtype=np.float32,
                            )
            if not np.isfinite(work).all():
                raise ValueError("accepted-prefix fold overflow")
            staged.append((int(slot), work))
        for slot, work in staged:
            canonical[:, slot] = work
        owner.epoch += 1
        self.cancel()
        return owner.state()


def metal_source():
    return Path(__file__).with_suffix(".metal").read_text()


def metal_source_sha256():
    return hashlib.sha256(metal_source().encode()).hexdigest()


def _metal_kernel(mx, *, fold=False):
    source = metal_source()
    header = source.split("// HEADER_BEGIN\n")[1].split("// HEADER_END")[0]
    tag = "FOLD" if fold else "VERIFY"
    body = source.split(f"// {tag}_BODY_BEGIN\n")[1].split(f"// {tag}_BODY_END")[0]
    inputs = (
        ["checkpoint", "keys", "corrections", "decay", "slots", "accepted", "dims"]
        if fold
        else [
            "checkpoint",
            "query",
            "keys",
            "values",
            "decay",
            "beta",
            "slots",
            "lengths",
            "dims",
        ]
    )
    return mx.fast.metal_kernel(
        name=f"mlx2_gdn_output_only_{tag.lower()}_{metal_source_sha256()[:12]}",
        input_names=inputs,
        output_names=["successor"] if fold else ["outputs", "corrections"],
        header="#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n"
        + header,
        source=body,
        compile_options={"math_mode": "safe"},
    )


@dataclass(frozen=True)
class NativeFoldSuccessor:
    states: object
    slots: np.ndarray
    publish_mask: np.ndarray
    binding: ReplayBinding
    epoch: int
    law: str = NATIVE_LAW


class NativeReplayCandidate:
    """Lazy native graph staging; the adapter still owns terminal publication.

    No dispatch is observed merely by building a lazy graph. Caller-supplied
    binding/epoch must be read from the live adapter again before fold. The
    returned compact successor is a separate array; input checkpoint is immutable.
    This candidate has not been run or qualified on a GPU.
    """

    def __init__(
        self,
        mx,
        checkpoint,
        keys,
        decay,
        slots,
        lengths,
        dims,
        outputs,
        corrections,
        binding,
        epoch,
        layout,
        max_successor_bytes,
    ):
        self._mx, self._checkpoint = mx, checkpoint
        self._keys, self._decay = keys, decay
        self._slots, self._lengths, self._dims = slots, lengths, dims
        self.outputs, self._corrections = outputs, corrections
        self.binding, self.epoch, self.layout = binding, epoch, layout
        self._source_hash = metal_source_sha256()
        self._max_successor_bytes = max_successor_bytes
        self.closed = False
        self._lock = threading.RLock()

    def receipt(self):
        return {
            "schema": "mlx2.output-only-gdn.native.v1",
            "law": NATIVE_LAW,
            "checkpoint": self.binding.checkpoint,
            "source_revision": self.binding.source_revision,
            "contract_revision": self.binding.contract_revision,
            "epoch": self.epoch,
            "selected": True,
            "implemented": True,
            "qualified": False,
            "observed_used": False,
            "native_observed_used": False,
            "implementation_scope": "lazy_native_staging",
            "state_written_during_verify": False,
            "final_state_output": False,
            "publication": "adapter_must_validate_staged_successor",
            "metal_source_sha256": self._source_hash,
        }

    @_serialized
    def cancel(self):
        self.closed = True

    @_serialized
    def fold(self, accepted, *, current_binding, current_epoch):
        _epoch(current_epoch)
        accepted = _owned(_array(accepted, np.dtype("int64"), "accepted"))
        if (
            self.closed
            or current_binding != self.binding
            or current_epoch != self.epoch
            or metal_source_sha256() != self._source_hash
        ):
            raise ValueError("native replay identity, epoch or source is stale")
        if (
            accepted.shape != self._lengths.shape
            or np.any(accepted < 0)
            or np.any(accepted > self._lengths)
        ):
            raise ValueError("accepted prefix exceeds native tape")
        mx = self._mx
        dims = mx.array(self._dims, dtype=mx.uint64)
        l, _, h, dv, dk, b, _, _, _ = self._dims
        output_shape = (l, b, h, dv, dk)
        if int(np.prod(output_shape, dtype=object)) * 4 > self._max_successor_bytes:
            raise ValueError("native successor exceeds bound")
        rows = l * b * h * dv
        result = _metal_kernel(mx, fold=True)(
            inputs=[
                self._checkpoint,
                self._keys,
                self._corrections,
                self._decay,
                mx.array(self._slots, dtype=mx.int64),
                mx.array(accepted, dtype=mx.int64),
                dims,
            ],
            grid=(int(rows), 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[output_shape],
            output_dtypes=[mx.float32],
        )[0]
        self.closed = True
        return NativeFoldSuccessor(
            result,
            _owned(self._slots),
            _owned((self._slots >= 0) & (accepted > 0)),
            self.binding,
            self.epoch,
        )


def build_native_candidate(
    checkpoint,
    query,
    keys,
    values,
    decay,
    beta,
    slots,
    lengths,
    *,
    binding,
    epoch,
    enabled=False,
    layout="value_key",
    max_tape_bytes=64 << 20,
    max_successor_bytes=256 << 20,
):
    """Build output/correction graphs through MLX only after explicit opt-in.

    Native operands are contiguous FP32 arrays of already normalized GDN inputs;
    this intentionally does not inherit the BF16-WY/TF32 arithmetic law. Slot
    vectors are CPU int64 metadata so validation needs no device synchronization.
    """
    if enabled is not True:
        raise ValueError("native output-only replay requires explicit enabled=True")
    if not isinstance(binding, ReplayBinding):
        raise TypeError("binding requires ReplayBinding")
    _epoch(epoch)
    slots = _owned(_array(slots, np.dtype("int64"), "slots"))
    lengths = _owned(_array(lengths, np.dtype("int64"), "lengths"))
    if any(type(x) is not int or x <= 0 for x in (max_tape_bytes, max_successor_bytes)):
        raise ValueError("native byte budgets must be positive integers")
    if layout not in ("value_key", "key_value"):
        raise ValueError("unsupported native recurrent layout")
    # Importing the backend and building device graphs happen only here.
    import mlx.core as mx

    operands = (checkpoint, query, keys, values, decay, beta)
    if any(not isinstance(x, mx.array) or x.dtype != mx.float32 for x in operands):
        raise ValueError("native operands require MLX FP32 arrays")
    if len(checkpoint.shape) != 5 or len(query.shape) != 5:
        raise ValueError("native state and query must be rank five")
    l, s, h, a, z = map(int, checkpoint.shape)
    dv, dk = (a, z) if layout == "value_key" else (z, a)
    ql, b, t, hk, qk = map(int, query.shape)
    if (
        min(l, s, h, dv, dk, b, hk) <= 0
        or ql != l
        or qk != dk
        or h % hk
        or not 1 <= t <= 32
        or dk > 128
        or dv > 128
        or dk % 4
        or dv % 4
        or keys.shape != query.shape
        or tuple(values.shape) != (l, b, t, h, dv)
        or tuple(decay.shape) != (l, b, t, h)
        or beta.shape != decay.shape
        or slots.shape != (b,)
        or lengths.shape != (b,)
    ):
        raise ValueError("native replay geometry/alignment is unsupported")
    if (
        np.any(slots < -1)
        or np.any(slots >= s)
        or np.any(lengths < 0)
        or np.any(lengths > t)
        or len(set(map(int, slots[slots >= 0]))) != len(slots[slots >= 0])
    ):
        raise ValueError("native live slots/lengths invalid")
    if (
        l * b * t * (hk * dk + h * dv + h) * 4 + slots.nbytes + lengths.nbytes
    ) > max_tape_bytes:
        raise ValueError("native tape exceeds bound")
    if l * b * h * dv * dk * 4 > max_successor_bytes or l * b * h * dv > 0xFFFFFFFF:
        raise ValueError("native state/grid exceeds bound")
    state_element_offset(
        (l, s, h, dv, dk), l - 1, s - 1, h - 1, dv - 1, dk - 1, layout=layout
    )
    checkpoint, query, keys, values, decay, beta = (mx.contiguous(x) for x in operands)
    dims = (l, s, h, dv, dk, b, t, hk, 0 if layout == "value_key" else 1)
    outputs, corrections = _metal_kernel(mx)(
        inputs=[
            checkpoint,
            query,
            keys,
            values,
            decay,
            beta,
            mx.array(slots, dtype=mx.int64),
            mx.array(lengths, dtype=mx.int64),
            mx.array(dims, dtype=mx.uint64),
        ],
        grid=(l * b * h * dv, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(l, b, t, h, dv)] * 2,
        output_dtypes=[mx.float32] * 2,
    )
    return NativeReplayCandidate(
        mx,
        checkpoint,
        keys,
        decay,
        slots,
        lengths,
        dims,
        outputs,
        corrections,
        binding,
        epoch,
        layout,
        max_successor_bytes,
    )
