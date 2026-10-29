# SPDX-License-Identifier: MIT
"""One immutable CPU owner and one upload for a Source prefill route frame.

There is no MLX import or device work on import. This metadata optimization
does not merge independent experts into a shared-weight matrix or change a
job's virtual M/reduction. See deepseek-v41-source-row-packet provenance.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .deepseek_v41_source_prefill import Plan, Job, Batch

POLICY = 'row_packet_v1'
SCHEMA = 1
MAX_WORDS = 1 << 20
_U32 = np.iinfo(np.uint32).max
_U64 = np.iinfo(np.uint64).max
_SEAL = object()


@lru_cache(maxsize=1)
def implementation_identity():
    """Source-bound profile identity; no arrays or device availability probes."""
    root = Path(__file__).resolve().parents[1]
    paths = ('adapters/deepseek_v41_source_row_packet.py',
             'adapters/deepseek_v41_source_prefill.py',
             'adapters/deepseek_v41_source_math.py',
             'adapters/deepseek_v41_source_weights.py',
             'runtime/gguf/source_gate_up_prefill.py',
             'runtime/models/deepseek_v41.py')
    records = '\n'.join(path + ':' + hashlib.sha256((root / path).read_bytes()).hexdigest()
                        for path in paths)
    return hashlib.sha256((POLICY + '\n' + records).encode()).hexdigest()


def _integer(value, name, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('Source row packet invalid '+name)
    return value


def _validate_plan(plan):
    if not isinstance(plan, Plan):
        raise ValueError('Source row packet requires an authoritative frame Plan')
    values, order = plan.values, plan.order
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint32
            or values.ndim != 2 or values.shape[1] != 6 or not 2 <= values.shape[0] <= 1024
            or np.any(values >= 384) or values.flags.writeable
            or not isinstance(order, np.ndarray) or order.dtype != np.uint32
            or order.shape != (values.size,) or order.flags.writeable):
        raise ValueError('Source row packet requires readonly reused-plan routes')
    # Validate independently of argsort: each job must equal the original
    # occurrence list and the global order is their ascending-ID concatenation.
    expected_ids = sorted(set(int(v) for v in values.ravel()))
    if (type(plan.jobs) is not tuple or any(not isinstance(j, Job) for j in plan.jobs)
            or [j.expert_id for j in plan.jobs] != expected_ids):
        raise ValueError('Source row packet expert identity/order mismatch')
    cursor = 0
    for job in plan.jobs:
        expected = np.flatnonzero(values.ravel() == job.expert_id).astype(np.uint32)
        if (job.kind not in ('original', 'q1') or type(job.count) is not int
                or job.count != expected.size or not isinstance(job.locations, np.ndarray)
                or not isinstance(job.rows, np.ndarray) or job.locations.dtype != np.uint32
                or job.rows.dtype != np.uint32 or job.locations.flags.writeable
                or job.rows.flags.writeable or not np.array_equal(job.locations, expected)
                or not np.array_equal(job.rows, expected // 6)
                or not np.array_equal(order[cursor:cursor + job.count], expected)):
            raise ValueError('Source row packet job occurrence/whole-M mismatch')
        cursor += job.count
    if cursor != values.size:
        raise ValueError('Source row packet incomplete occurrence coverage')
    inverse = np.empty(values.size, np.uint32)
    inverse[order] = np.arange(values.size, dtype=np.uint32)
    if (not isinstance(plan.inverse, np.ndarray) or plan.inverse.dtype != np.uint32
            or not np.array_equal(plan.inverse, inverse)
            or not isinstance(plan.combine_offsets, np.ndarray)
            or plan.combine_offsets.dtype != np.uint32
            or not np.array_equal(plan.combine_offsets, np.sort(inverse.reshape(values.shape), axis=1))):
        raise ValueError('Source row packet inverse/ordered-fold mismatch')
    seen = []
    for batch in plan.batches:
        if not isinstance(batch, Batch):
            raise ValueError('Source row packet invalid batch descriptor')
        indices = batch.job_indices
        if (not 1 <= len(indices) <= 15 or any(type(i) is not int or not 0 <= i < len(plan.jobs) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Source row packet invalid batch jobs')
        jobs = [plan.jobs[i] for i in indices]
        counts = tuple(j.count for j in jobs)
        offsets = tuple(np.cumsum((0, *counts)).tolist())
        if (any(j.kind != batch.kind for j in jobs) or batch.row_counts != counts
                or batch.offsets != offsets or not np.array_equal(batch.rows, np.concatenate([j.rows for j in jobs]))
                or not np.array_equal(batch.locations, np.concatenate([j.locations for j in jobs]))):
            raise ValueError('Source row packet batch row identity mismatch')
        if ((batch.kind == 'original' and (len(set(counts)) != 1 or counts[0] > 12))
                or (batch.kind == 'q1' and any(count > 1024 for count in counts))):
            raise ValueError('Source row packet batch changes an admitted row law')
        seen.extend(indices)
    fallback = tuple(plan.fallback_job_indices)
    if (any(type(i) is not int or not 0 <= i < len(plan.jobs) for i in fallback)
            or sorted(seen + list(fallback)) != list(range(len(plan.jobs)))):
        raise ValueError('Source row packet duplicate/missing job ownership')
    return values, order


@dataclass(frozen=True)
class Field:
    offset: int
    words: int
    shape: tuple


class RowPacket:
    """Factory-issued CPU metadata; all arrays borrow one immutable bytes owner."""
    __slots__ = ('plan', 'backing', 'fields', 'frame_key', 'generation', 'layer', 'digest', '_seal')

    def __setattr__(self, name, value):
        if getattr(self, '_seal', None) is _SEAL:
            raise AttributeError('Source row packet is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, *, _seal, plan, backing, fields, frame_key, generation, layer):
        if _seal is not _SEAL:
            raise ValueError('Source row packets must be prepared from an admitted plan')
        self.plan, self.backing = plan, backing
        self.fields = MappingProxyType(fields)
        self.frame_key, self.generation, self.layer = frame_key, generation, layer
        self.digest = None  # Receipt hashing is outside the normal upload path.
        self._seal = _seal

    def view(self, name):
        field = self.fields[name]
        return self.backing[field.offset:field.offset + field.words].reshape(field.shape)

    def validate(self, plan, *, frame_key, generation, layer, contents=True):
        if type(contents) is not bool:
            raise ValueError('Source row packet content-validation flag must be boolean')
        if (self._seal is not _SEAL or self.plan is not plan or self.frame_key != frame_key
                or self.generation != generation or self.layer != layer):
            raise ValueError('Source row packet stale frame/generation/owner')
        if contents:
            self.validate_contents()

    def validate_contents(self):
        """Prove borrowed CPU metadata still matches the immutable snapshot.

        This is a bounded CPU comparison, not a device readback. Keep its cost
        inside stage timing; it is intended at ownership/consumer boundaries.
        """
        if self._seal is not _SEAL or self.backing.flags.writeable:
            raise ValueError('Source row packet has invalid factory ownership')
        for name in ('values', 'order', 'inverse', 'combine_offsets'):
            current = getattr(self.plan, name)
            if (not isinstance(current, np.ndarray) or current.dtype != np.uint32
                    or current.flags.writeable or not np.array_equal(current, self.view(name))):
                raise ValueError('Source row packet plan metadata changed after preparation')
        for index, batch in enumerate(self.plan.batches):
            for name in ('rows', 'locations', 'job_indices', 'offsets', 'row_counts'):
                if not np.array_equal(getattr(batch, name), self.view(f'batch{index}.{name}')):
                    raise ValueError('Source row packet batch metadata changed after preparation')

    def upload(self, backend):
        """Caller selects the backend/device explicitly; performs exactly one array upload."""
        root = backend.array(self.backing, dtype=backend.uint32)
        if (getattr(root, 'shape', None) != self.backing.shape or getattr(root, 'ndim', None) != 1
                or getattr(root, 'dtype', None) != backend.uint32):
            raise ValueError('Source row packet backend upload geometry/dtype mismatch')
        return DeviceRowPacket(self, root, _seal=_SEAL)

    def receipt(self):
        digest = self.digest or hashlib.sha256(self.backing).hexdigest()
        return {'policy': POLICY, 'schema': SCHEMA, 'frame_key': self.frame_key,
                'generation': self.generation, 'layer': self.layer, 'words': int(self.backing.size),
                'metadata_bytes': int(self.backing.nbytes), 'metadata_sha256': digest,
                'coefficient_repack_bytes': 0, 'activation_repack_bytes': 0,
                'whole_expert_M_preserved': True, 'qualification': 'unqualified'}


class DeviceRowPacket:
    """Retain the upload and independent consumers until an explicit completion.

    This owns metadata only. The caller's existing provider/completion scope
    continues to own expert coefficients. A failed completion retains roots.
    """
    __slots__ = ('_packet', '_root', '_outputs', '_retired', '_seal', '_views')

    def __init__(self, packet, root, *, _seal=None):
        if _seal is not _SEAL or not isinstance(packet, RowPacket) or packet._seal is not _SEAL:
            raise ValueError('Source device row packets must be factory uploaded')
        self._packet, self._root = packet, root
        self._outputs, self._retired = {}, False
        self._seal = _seal
        self._views = {}

    @property
    def packet(self):
        return self._packet

    @property
    def root(self):
        return self._root

    def _live(self):
        if self._seal is not _SEAL or self._retired:
            raise ValueError('Source device row packet already retired')

    def view(self, name):
        self._live()
        if name not in self._views:
            field = self.packet.fields[name]
            self._views[name] = self.root[field.offset:field.offset + field.words].reshape(field.shape)
        return self._views[name]

    def batch(self, index):
        self._live()
        _integer(index, 'batch index', len(self.packet.plan.batches) - 1)
        return self.view(f'batch{index}.rows'), self.view(f'batch{index}.locations')

    def job(self, index):
        self._live()
        _integer(index, 'job index', len(self.packet.plan.jobs) - 1)
        start = sum(j.count for j in self.packet.plan.jobs[:index])
        count = self.packet.plan.jobs[index].count
        return self.view('rows')[start:start + count], self.view('order')[start:start + count]

    def retain(self, *outputs):
        self._live()
        for value in outputs:
            if value is None:
                raise ValueError('Source row packet cannot retain an absent output')
            self._outputs[id(value)] = value

    def completion_roots(self):
        """Roots for an existing collector; this method never evaluates them.

        Serving graphs consume immutable metadata views through ordinary MLX
        dependency ownership. Standalone probes can register independent
        outputs and pass these roots to their existing completion coordinator.
        """
        self._live()
        return (self.root, *self._outputs.values())

    def complete(self, fence):
        self._live()
        if not self._outputs:
            raise ValueError('Source row packet has no submitted consumer')
        # Keep every independent output, including the upload itself. Only
        # discard ownership after the caller's fence succeeds.
        fence(self.root, *self._outputs.values())
        self._outputs.clear()
        self._views.clear()
        self._root = None
        self._retired = True

    def close_unused(self):
        self._live()
        if self._outputs:
            raise ValueError('Source row packet has pending consumers')
        self._views.clear()
        self._root = None
        self._retired = True


def prepare_row_packet(plan, *, frame_key, generation, layer, position_start=0, max_words=MAX_WORDS):
    """Pack one already authoritative plan without changing any job or kernel.

    Positions are u64 split into low/high u32 words. This B1 packet has one
    request lane, but keeps logical token position and route slot explicit.
    """
    values, order = _validate_plan(plan)
    if type(frame_key) is not str or not frame_key or len(frame_key) > 256:
        raise ValueError('Source row packet requires a bounded frame key')
    _integer(generation, 'generation', int(_U64))
    _integer(layer, 'layer', 39)
    _integer(position_start, 'position start', int(_U64) - values.shape[0] + 1)
    if type(max_words) is not int or not 1 <= max_words <= MAX_WORDS:
        raise ValueError('Source row packet invalid metadata bound')
    arrays, fields, cursor = [], {}, 0

    def add(name, data):
        nonlocal cursor
        a = np.asarray(data)
        if a.dtype.kind not in 'iu' or (a.size and (np.any(a < 0) or np.any(a > _U32))):
            raise ValueError('Source row packet field is not lossless u32: '+name)
        size = int(a.size)
        if cursor + size > max_words:
            raise ValueError('Source row packet metadata capacity exceeded')
        fields[name] = Field(cursor, size, a.shape)
        arrays.append(a.astype(np.uint32, copy=False).reshape(-1))
        cursor += size

    add('header', [SCHEMA, layer, generation & int(_U32), generation >> 32,
                   values.shape[0], values.shape[1], len(plan.jobs), len(plan.batches)])
    for name, array in (('values', values), ('order', order), ('inverse', plan.inverse),
                        ('combine_offsets', plan.combine_offsets), ('rows', order // 6)):
        add(name, array)
    positions = np.arange(values.shape[0], dtype=np.uint64) + np.uint64(position_start)
    add('position_low', (positions & np.uint64(_U32)).astype(np.uint32))
    add('position_high', (positions >> np.uint64(32)).astype(np.uint32))
    add('lane_ids', np.zeros(values.shape[0], np.uint32))
    add('route_slots', order % 6)
    cursor_job, descriptors = 0, []
    for job in plan.jobs:
        descriptors.append((job.expert_id, int(job.kind == 'q1'), cursor_job, job.count, job.count))
        cursor_job += job.count
    add('jobs', np.asarray(descriptors, np.uint32))
    for index, batch in enumerate(plan.batches):
        for name, array in (('rows', batch.rows), ('locations', batch.locations),
                            ('job_indices', batch.job_indices), ('offsets', batch.offsets),
                            ('row_counts', batch.row_counts)):
            add(f'batch{index}.{name}', array)
    # Empty fallback is represented as a zero-word typed view, not padding.
    add('fallback_jobs', np.asarray(plan.fallback_job_indices, np.uint32))
    backing = np.frombuffer(np.concatenate(arrays).tobytes(), dtype=np.uint32)
    return RowPacket(_seal=_SEAL, plan=plan, backing=backing, fields=fields,
                     frame_key=frame_key, generation=generation, layer=layer)
