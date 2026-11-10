# SPDX-License-Identifier: MIT
"""Opt-in APPROXIMATE Source routed experts from an EXL3 bank (trellis-coded).

Policy (inside ``deepseek_source_experts``, ``mode: original_lfu``)::

    "exl3": {"root": "/abs/DeepSeek-V4.1-Flash-1017-Source-Expert-EXL3-3bpw",
             "scope": "all",                 # the only admitted scope
             "pool_bytes": 40000000000,      # carved from budget_bytes (default 0 = uncached)
             "layers": [0, 42],              # optional; default every layer
             "fingerprint": "<64 hex>",      # optional pin of the bank manifest fingerprint
             "math": "reference",            # or "kernel" (see below)
             "missing_path": "/abs/missing.json"}  # optional, rewritten after forwards

Unknown fields fail closed. The bank is the converter's output
(scripts/research/ds41_exl3/convert_bank.py): ``manifest.json`` with
``format`` exactly :data:`MANIFEST_FORMAT`, ``bits_per_weight``, ``source``
``<name>@<Source fingerprint>``, ``files`` {relpath: sha256} and
``fingerprint`` = sha256(json.dumps(files, sort_keys=True)). Only files listed
in the manifest are admitted (a bank still being written is read as of its
last manifest); each is stat-sealed at load and content-hashed once on first
read. A file holds ``layers.L.ffn.experts.E.{w1,w3,w2}.{trellis,suh,svh,mcg}``
(upstream exllamav3 names: w1 gate, w3 up, w2 down; trellis I16
[in/16, out/16, words], suh F16 [in], svh F16 [out], mcg I32 marker).

Serving rule (prefill AND decode, every row width): a routed (layer, expert)
present in the bank and inside ``layers`` is computed from EXL3; every other
expert is the ordinary original MXFP4 expert (fallback). The decision is
static per key, so it never depends on residency or cache history.

Math law: the same fp8 QAT input, the same middle (limited SwiGLU x score,
bf16 rounding, or the fused middle under fused expert math), the same bf16
rounding of each expert's down output and the same combine order as the
original path. Only the projection changes: ``y = x @ W`` with ``W`` the
decoded EXL3 weight ``diag(suh) H_in inner H_out diag(svh)``
(exl3_reference.effective_weight, [in, out]).

* ``math: reference`` (default): float32 throughout. ``inner`` is decoded per
  call (Metal RECONSTRUCT kernel on GPU, the host decoder on CPU; both exact
  fp16 codebook values), then H128(x*suh) @ inner in float32, H128, * svh.
* ``math: kernel``: the Qwen EXL3 kernels with their fp16 boundaries.
  Rows < KERNEL_WIDE_ROWS: PREPARE (fp16) -> ROWS_GEMV (fp32 accum) -> FINISH
  (fp16). Wider rows: PREPARE -> RECONSTRUCT -> fp16 mx.matmul -> FINISH.
  FUSED_GATE_UP is not used: its middle is plain SwiGLU, not the Source law.

A call holding any EXL3 expert takes the per-expert path of ``_moe_routes``;
the fused B1 down-combine and the batched prefill gate/up kernels consume
original packed objects only and are bypassed for that call (counted).
Multi-row calls evaluate each EXL3 expert before the next (bounds the
per-call decoded weights). The pool is carved from ``budget_bytes`` and,
under adaptive residency, scales with the original share like the
reduced-tier pools. The measured default kernel bundle is not selected under
``exl3`` (approximate profile); its keys may be named explicitly.
``math: kernel`` runs only on Metal; on CPU the reference law runs
(``math_effective`` in the receipt).

Approximate contract (as reduced tiers): identity salt over the bank
fingerprint, settings and implementation; numerics_revision/profile/
numeric_profile salted; forwards that served EXL3 stamp cache rows with a
private profile; APCv2 reuse and publication refused; receipts say
unqualified.

Receipts: per-layer acquires served EXL3 vs original fallback, routed slots
per tier, bytes read per tier, pool hits/misses, and the missing set (every
fallback (layer, expert) with its acquire count). ``missing_path`` is written
in the converter's ``--experts`` format ({"<layer>": [experts]}).

Status: implemented_untested_unqualified. Provenance:
provenance/deepseek-v41-source-exl3-experts.json.
"""
from collections import Counter, OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

KEY = 'exl3'
NAME = 'exl3_bank_v1'
FIELDS = frozenset(('root', 'scope', 'pool_bytes', 'layers', 'fingerprint', 'math', 'missing_path'))
SCOPES = ('all',)
MATH_LAWS = ('reference', 'kernel')
MANIFEST_FORMAT = 'exl3 (upstream exllamav3 tensor layout, mcg codebook)'
QUALIFICATION = 'unqualified'
STATUS = 'implemented_untested_unqualified'
PUBLICATION_QUALIFIED = False
APC_REASON = 'approximate_source_operation_unqualified:' + NAME
PRIVATE_PROFILE_PREFIX = 'mlx2.ds41.source-exl3.v1:'
PARTS = {'gate': 'w1', 'up': 'w3', 'down': 'w2'}
GEOMETRY = {'gate': (5120, 2304), 'up': (5120, 2304), 'down': (2304, 5120)}   # (in, out)
KERNEL_WIDE_ROWS = 64
# Metal sources the two laws load at run time (bound into the identity salt).
KERNEL_FILES = ('header_mcg_w16.metal', 'PREPARE_SOURCE.metal', 'FINISH_SOURCE.metal',
                'ROWS_GEMV_SOURCE.metal', 'RECONSTRUCT_SOURCE.metal')
FILE_RE = re.compile(r'^w(\d+)/layers\.(\d+)\.ffn\.experts\.(\d+)\.safetensors$')

# Route/state laws this substitution is not composed with (key, reference).
REFUSALS = (
    ('deepseek_physical_batch', 'b1'), ('deepseek_vision', 'reference'),
    ('deepseek_source_dspark_ast', 'off'), ('deepseek_source_dspark_prepare', 'off'),
    ('deepseek_source_dspark_target_forward', 'reference'),
    ('deepseek_source_ast_kv_preparation', 'reference'),
    # Resident bank sessions compute from attached original banks without acquire().
    ('deepseek_source_resident_gather', 'reference'), ('deepseek_source_resident_prefill', 'reference'),
    # Overlap / read-ahead read original bytes for routes ahead of acquire().
    ('deepseek_source_expert_io', 'reference'), ('deepseek_source_prefill_readahead', 'reference'),
    # Row packets belong to the batched prefill kernels, which EXL3 calls bypass.
    ('deepseek_source_row_packet', 'reference'),
    ('deepseek_source_q2_tier_math', 'reference'),
    ('deepseek_source_prefill_approx', 'off'), ('deepseek_source_route_bias', 'off'),
    ('deepseek_source_route_drop', 'off'),
)
# Expert-policy keys refused next to ``exl3`` (they read or rank originals
# ahead of the static EXL3 rule, or are another reduced-precision engine).
EXPERT_REFUSALS = ('q1_on_miss', 'tiers', 'atlas_path', 'preheat', 'lookahead_depth', 'lookahead_heads',
                   'loader', 'hot_bytes', 'hotset_path')


# ---- Policy ------------------------------------------------------------------

def _abs(value):
    return isinstance(value, str) and Path(value).is_absolute()


def parse(value):
    if not isinstance(value, dict) or set(value) - FIELDS or 'root' not in value:
        raise ValueError('exl3 must be {"root": abs path, "scope": "all", "pool_bytes": int, ...}; keys: '
                         + ', '.join(sorted(FIELDS)))
    if not _abs(value['root']):
        raise ValueError('exl3.root must be absolute')
    scope = value.get('scope', 'all')
    if scope not in SCOPES:
        raise ValueError('exl3.scope must be one of ' + ', '.join(SCOPES))
    pool = value.get('pool_bytes', 0)
    if type(pool) is not int or pool < 0:
        raise ValueError('exl3.pool_bytes must be a non-negative integer byte count')
    layers = value.get('layers', [0, 1023])
    if (not isinstance(layers, (list, tuple)) or len(layers) != 2 or not all(type(v) is int for v in layers)
            or not 0 <= layers[0] <= layers[1] <= 1023):
        raise ValueError('exl3.layers must be two integer layer IDs 0 <= first <= last')
    fingerprint = value.get('fingerprint')
    if fingerprint is not None and (not isinstance(fingerprint, str) or not re.fullmatch(r'[0-9a-f]{64}', fingerprint)):
        raise ValueError('exl3.fingerprint must be 64 lowercase hex characters')
    law = value.get('math', 'reference')
    if law not in MATH_LAWS:
        raise ValueError('exl3.math must be one of ' + ', '.join(MATH_LAWS))
    if 'missing_path' in value and not _abs(value['missing_path']):
        raise ValueError('exl3.missing_path must be absolute')
    result = dict(root=value['root'], scope=scope, pool_bytes=pool, layers=[layers[0], layers[1]], math=law)
    if fingerprint is not None:
        result['fingerprint'] = fingerprint
    if 'missing_path' in value:
        result['missing_path'] = value['missing_path']
    return result


def validate_expert_policy(result):
    """In-place check inside a parsed ``deepseek_source_experts`` policy."""
    if KEY not in result:
        return result
    if result.get('mode') != 'original_lfu':
        raise ValueError('exl3 requires mode original_lfu')
    blocked = [k for k in EXPERT_REFUSALS if k in result]
    if blocked:
        raise ValueError('exl3 cannot combine with deepseek_source_experts keys: ' + ', '.join(blocked))
    settings = result[KEY] = parse(result[KEY])
    from .deepseek_v41_source_lfu import ORIGINAL_BYTES
    if settings['pool_bytes'] > result['budget_bytes'] - 2 * ORIGINAL_BYTES:
        raise ValueError('exl3.pool_bytes must leave budget_bytes room for one original expert plus staging')
    return result


def mode_of(policy):
    expert = policy.get('deepseek_source_experts') if isinstance(policy, dict) else None
    return KEY if isinstance(expert, dict) and KEY in expert else None


def enabled(policy):
    return mode_of(policy) is not None


def validate_combination(policy):
    if not enabled(policy):
        return
    blocked = [name for name, reference in REFUSALS if policy.get(name, reference) != reference]
    if policy.get('draft_model') not in (None, 'none'):
        blocked.append('draft_model')
    if policy.get('ast_prompt_lookup', False):
        blocked.append('ast_prompt_lookup')
    if policy.get('deepseek_verify_rows', 'serial') not in ('serial', 'reference'):
        blocked.append('deepseek_verify_rows')
    if policy.get('deepseek_verify_moe_rows', 'reference') != 'reference':
        blocked.append('deepseek_verify_moe_rows')
    if (policy.get('deepseek_source_experts') or {}).get('residency') == 'dynamic_precision_v1':
        blocked.append('deepseek_source_experts.residency=dynamic_precision_v1')
    if blocked:
        raise ValueError('Source exl3 experts cannot combine with: ' + ', '.join(blocked))


# ---- Bank ----------------------------------------------------------------------

def load_manifest(root, source_binding, *, expected_fingerprint=None):
    """(root, records {(layer, expert): record}, fingerprint, words, meta).

    Admits only files the manifest lists, with their sha256; stat-seals each.
    """
    from .deepseek_v41_source_artifact import checked_file
    from .deepseek_v41_mlx_weights import _path, _stamp
    from stat import S_ISREG
    root = Path(root).resolve()
    manifest = json.loads(checked_file(root, 'manifest.json').read_text())
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('EXL3 manifest lists no files')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if manifest.get('fingerprint') != digest:
        raise ValueError('EXL3 manifest fingerprint mismatch')
    if expected_fingerprint is not None and digest != expected_fingerprint:
        raise ValueError('EXL3 bank fingerprint differs from the policy pin')
    if manifest.get('format') != MANIFEST_FORMAT:
        raise ValueError('EXL3 bank format is not admitted: ' + repr(manifest.get('format')))
    source = manifest.get('source')
    if not isinstance(source, str) or source.rpartition('@')[2] != source_binding:
        raise ValueError('EXL3 bank does not bind this Source checkpoint')
    bpw = manifest.get('bits_per_weight')
    if type(bpw) not in (int, float) or not math.isfinite(bpw) or bpw * 16 != int(bpw * 16):
        raise ValueError('EXL3 bits_per_weight must name whole tile words')
    words = int(bpw * 16)
    if words % 2 or not 16 <= words <= 128:
        raise ValueError('EXL3 tile words must be even in 16..128')
    records = {}
    for relative, sha in files.items():
        match = FILE_RE.match(relative) if isinstance(relative, str) else None
        if match is None or int(match.group(1)) != words:
            raise ValueError('EXL3 manifest file name is not an admitted expert path: ' + repr(relative))
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
            raise ValueError('EXL3 manifest sha256 is invalid for ' + relative)
        key = (int(match.group(2)), int(match.group(3)))
        if key in records or not 0 <= key[1] < 384:
            raise ValueError('invalid EXL3 expert inventory')
        path = _path(root, relative)
        current = path.stat(follow_symlinks=False)
        if not S_ISREG(current.st_mode):
            raise ValueError('EXL3 bank file must be a regular file: ' + relative)
        stat = _stamp(current)
        records[key] = {'file': relative, 'sha256': sha, 'size': stat['size'], 'stat': stat}
    meta = {k: manifest.get(k) for k in ('source', 'format', 'bits_per_weight', 'selection')}
    return root, records, digest, words, meta


def payload_bytes(words):
    """Uploaded bytes of one expert (trellis + suh + svh of three projections)."""
    total = 0
    for i, o in GEOMETRY.values():
        total += (i // 16) * (o // 16) * words * 2 + 2 * i + 2 * o
    return total


def exl3_read(root, record, key, words):
    """Verified host tensors {kind.trellis (u16 [1,I/16,O/16,W]), kind.suh [1,I], kind.svh [1,O]}."""
    from .deepseek_v41_mlx_weights import read_sealed_bank_file
    from ..runtime.trellis.upstream import CODEBOOKS
    raw = read_sealed_bank_file(root, record['file'], record)
    layer, expert = key
    base = f'layers.{layer}.ffn.experts.{expert}.'
    expected = {base + p + '.' + f for p in PARTS.values() for f in ('trellis', 'suh', 'svh', 'mcg')}
    if set(raw) != expected:
        raise ValueError('EXL3 tensor inventory mismatch for ' + record['file'])
    marker = np.array([CODEBOOKS['mcg']], np.uint32).view(np.int32)
    result = {}
    for kind, part in PARTS.items():
        i, o = GEOMETRY[kind]
        t, suh, svh, mcg = (raw[base + part + '.' + f] for f in ('trellis', 'suh', 'svh', 'mcg'))
        if (t.dtype != np.dtype('<i2') or t.shape != (i // 16, o // 16, words)
                or suh.dtype != np.dtype('<f2') or suh.shape != (i,)
                or svh.dtype != np.dtype('<f2') or svh.shape != (o,)
                or mcg.dtype != np.dtype('<i4') or mcg.shape != (1,) or not np.array_equal(mcg, marker)):
            raise ValueError('EXL3 tensor geometry/codebook mismatch for ' + record['file'])
        if not (np.isfinite(suh).all() and np.isfinite(svh).all()):
            raise ValueError('EXL3 scale vectors must be finite')
        result[kind + '.trellis'] = t.view(np.uint16)[None]
        result[kind + '.suh'] = suh[None]
        result[kind + '.svh'] = svh[None]
    return result


class Exl3Expert(dict):
    """EXL3 arrays under distinct names: code unaware of EXL3 fails closed (KeyError)."""
    source_tier = 'exl3'
    words = None
    law = 'reference'

    def linear(self, kind):
        """Exl3Linear view of one projection; built once per object (its constructor syncs)."""
        cache = self.__dict__.setdefault('_linears', {})
        value = cache.get(kind)
        if value is None:
            from ..runtime.models.exl3 import Exl3Linear
            value = cache[kind] = Exl3Linear(self[kind + '.trellis'], self[kind + '.suh'], self[kind + '.svh'])
        return value


def is_exl3(data):
    return getattr(data, 'source_tier', None) == 'exl3'


def upload(raw, words, law):
    import mlx.core as mx
    arrays = Exl3Expert({name: mx.array(value) for name, value in raw.items()})
    arrays.words, arrays.law = words, law
    mx.eval(list(arrays.values()))
    if sum(v.nbytes for v in arrays.values()) != payload_bytes(words):
        raise ValueError('EXL3 upload exceeded admitted bytes')
    if mx.default_device() != mx.cpu:
        for kind in PARTS:
            arrays.linear(kind)   # once at upload, never inside graph building
    return arrays


# ---- Math ------------------------------------------------------------------------

def _decoded_inner(data, kind):
    """Decoded fp16 inner weight [I, O] (exact codebook values)."""
    import mlx.core as mx
    if mx.default_device() == mx.cpu:
        from ..runtime.models.exl3_reference import decode_inner
        return mx.array(decode_inner(np.asarray(data[kind + '.trellis'][0])))
    return data.linear(kind).reconstruct(mx.zeros((1,), mx.uint32))[0]


def _h128(x):
    from ..runtime.models.exl3 import _h128 as h
    return h(x)


def effective_law(law):
    """The law that actually runs: the kernels exist on Metal only."""
    import mlx.core as mx
    return 'reference' if law == 'kernel' and mx.default_device() == mx.cpu else law


def project(data, kind, activation):
    """float32 ``activation @ W`` for one EXL3 projection; rows x I -> rows x O.

    ``kernel`` keeps the Qwen kernels' fp16 PREPARE/FINISH boundaries (values
    beyond fp16 range would overflow there; the original path has no fp16
    step); on CPU it runs the reference law (``effective_law``).
    """
    import mlx.core as mx
    # Geometry is the expert's own (bank files are checked against GEOMETRY at read).
    if not is_exl3(data) or activation.ndim != 2 or activation.shape[1] != data[kind + '.suh'].shape[-1]:
        raise ValueError('EXL3 projection geometry mismatch')
    if data.law == 'kernel' and mx.default_device() != mx.cpu:
        bank = data.linear(kind)
        rows = activation.shape[0]
        ids = mx.zeros((rows,), mx.uint32)
        prepared = bank.prepare(activation, ids)
        if rows < KERNEL_WIDE_ROWS:
            from ..runtime.models.exl3 import row_plan
            inner = bank._rows(prepared, row_plan(ids, 1))
        else:
            inner = mx.matmul(prepared, bank.reconstruct(mx.zeros((1,), mx.uint32))[0])
        return bank.finish(inner, ids).astype(mx.float32)
    suh = data[kind + '.suh'][0].astype(mx.float32)
    svh = data[kind + '.svh'][0].astype(mx.float32)
    prepared = _h128(activation.astype(mx.float32) * suh)
    inner = mx.matmul(prepared, _decoded_inner(data, kind).astype(mx.float32))
    return _h128(inner) * svh


# ---- LRU pool ----------------------------------------------------------------------

class Exl3Pool:
    """One global LRU over EXL3 experts with a byte capacity carved from the total."""

    def __init__(self, capacity):
        self.capacity = self.target = capacity
        self.objects = OrderedDict()
        self.live = 0
        self.stats = dict(hits=0, misses=0, insertions=0, evictions=0, uncached=0, pressure_evictions=0)

    def _evict_to(self, limit, protected, *, pressure=False):
        evicted = 0
        for victim in list(self.objects):
            if self.live <= limit:
                break
            if protected.get(victim) is self.objects[victim]:
                continue
            gone = self.objects.pop(victim)
            amount = sum(v.nbytes for v in gone.values())
            self.live -= amount; evicted += amount
            self.stats['evictions'] += 1
            if pressure:
                self.stats['pressure_evictions'] += 1
        return evicted

    def scale(self, fraction, protected):
        """Adaptive residency: the pool target follows the original share (never above capacity)."""
        self.target = min(self.capacity, int(self.capacity * max(0.0, fraction)))
        return self._evict_to(self.target, protected, pressure=True)

    def lookup(self, key):
        arrays = self.objects.get(key)
        if arrays is None:
            self.stats['misses'] += 1
            return None
        self.objects.move_to_end(key)
        self.stats['hits'] += 1
        return arrays

    def insert(self, key, arrays, size, protected):
        if size > self.target:
            self.stats['uncached'] += 1
            return False
        self._evict_to(self.target - size, protected)
        if self.live + size > self.target:
            self.stats['uncached'] += 1
            return False
        self.objects[key] = arrays
        self.live += size
        self.stats['insertions'] += 1
        return True

    def clear(self):
        self.objects.clear()
        self.live = 0

    def receipt(self):
        return dict(eviction='global_lru', capacity_bytes=self.capacity, target_bytes=self.target, live_bytes=self.live,
                    objects=len(self.objects), **self.stats)


# ---- Rule ----------------------------------------------------------------------------

def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    names = ('adapters/deepseek_v41_source_exl3.py', 'adapters/deepseek_v41_source_lfu.py',
             'adapters/deepseek_v41_source_weights.py', 'runtime/models/exl3.py',
             'runtime/models/exl3_reference.py', 'runtime/trellis/upstream.py',
             'runtime/gguf/source_middle_fusion.py',
             *('runtime/models/exl3_kernels/' + k for k in KERNEL_FILES))
    return hashlib.sha256(''.join(name + ':' + hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in names).encode()).hexdigest()


def identity_salt(settings, fingerprint, words, source_binding):
    # missing_path names a receipt file only; it never changes numerics.
    numeric = {k: v for k, v in settings.items() if k not in ('missing_path', 'pool_bytes')}
    binding = dict(operation=NAME, settings=numeric, bank_fingerprint=fingerprint, words=words,
                   source=source_binding, implementation=implementation_identity(), qualification=QUALIFICATION)
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def private_profile(salt):
    return PRIVATE_PROFILE_PREFIX + salt[:32]


class Exl3Rule:
    """Per-store state: bank records, pool, counters, missing set."""

    def __init__(self, settings, *, root, records, fingerprint, words, meta, layers, source_binding):
        first, last = settings['layers']
        if first >= layers:
            raise ValueError('exl3.layers outside this checkpoint')
        self.first, self.last = first, min(last, layers - 1)
        self.settings = dict(settings, layers=[self.first, self.last])
        self.root, self.records, self.fingerprint, self.words, self.meta = root, records, fingerprint, words, meta
        if any(not 0 <= k[0] < layers for k in records):
            raise ValueError('EXL3 bank names a layer outside this checkpoint')
        self.law = settings['math']
        self.size = payload_bytes(words)
        self.pool = Exl3Pool(settings['pool_bytes']) if settings['pool_bytes'] else None
        self.salt = identity_salt(self.settings, fingerprint, words, source_binding)
        self.source_binding = source_binding
        self.missing_path = Path(settings['missing_path']) if 'missing_path' in settings else None
        self.layers = layers
        self.served = [0] * layers
        self.fallback = [0] * layers
        self.slots = {'exl3': [0] * layers, 'original': [0] * layers}
        self.missing = Counter()
        self.missing_written = 0
        self.stats = dict(exl3_acquires=0, fallback_acquires=0, out_of_range_acquires=0, exl3_reads=0, exl3_bytes_read=0,
                          exl3_read_ns=0, exl3_upload_ns=0, exl3_calls=0, mixed_calls=0,
                          stamped_forwards=0, missing_file_writes=0,
                          missing_file_errors=0)
        self.bank_layers = sorted({k[0] for k in records})

    @classmethod
    def from_policy(cls, policy, *, source_binding, layers):
        settings = policy[KEY]
        root, records, fingerprint, words, meta = load_manifest(
            settings['root'], source_binding, expected_fingerprint=settings.get('fingerprint'))
        return cls(settings, root=root, records=records, fingerprint=fingerprint, words=words, meta=meta,
                   layers=layers, source_binding=source_binding)

    def serves(self, key):
        return self.first <= key[0] <= self.last and key in self.records

    def route(self, layer, flat):
        """Record routed slots by tier; returns the EXL3 expert ids of this call."""
        ids = [int(e) for e in flat]
        exl3 = {e for e in set(ids) if self.serves((layer, e))}
        n = sum(e in exl3 for e in ids)
        self.slots['exl3'][layer] += n
        self.slots['original'][layer] += len(ids) - n
        if exl3:
            self.stats['exl3_calls'] += 1
            if len(exl3) != len(set(ids)):
                self.stats['mixed_calls'] += 1
        return exl3

    def record_fallback(self, key):
        """An original served for ``key``; only experts absent from the bank enter the missing set."""
        self.stats['fallback_acquires'] += 1
        self.fallback[key[0]] += 1
        if key in self.records:
            self.stats['out_of_range_acquires'] += 1   # in the bank, outside ``layers``
        else:
            self.missing[key] += 1

    def record_served(self, key):
        self.stats['exl3_acquires'] += 1
        self.served[key[0]] += 1

    def missing_experts(self):
        """{"<layer>": [experts]}: the converter's --experts input for exactly what fell back."""
        result = {}
        for (layer, expert) in sorted(self.missing):
            result.setdefault(str(layer), []).append(expert)
        return result

    def write_missing(self):
        if self.missing_path is None or len(self.missing) == self.missing_written:
            return False
        try:
            tmp = self.missing_path.with_name(self.missing_path.name + '.tmp')
            tmp.write_text(json.dumps(self.missing_experts()))
            os.replace(tmp, self.missing_path)
            self.missing_written = len(self.missing)
            self.stats['missing_file_writes'] += 1
            return True
        except OSError:
            self.stats['missing_file_errors'] += 1
            return False

    def receipt(self):
        bounded = sorted(self.missing.items(), key=lambda kv: (-kv[1], kv[0]))[:256]
        total_slots = sum(self.slots['exl3']) + sum(self.slots['original'])
        return dict(selected=True, policy=NAME, root=str(self.root), bank_fingerprint=self.fingerprint,
                    bank_meta=self.meta, bank_experts=len(self.records), bank_layers=self.bank_layers,
                    bits_per_weight=self.words / 16, tile_words=self.words, expert_payload_bytes=self.size,
                    math=self.law, math_effective=effective_law(self.law), layers=[self.first, self.last], scope=self.settings['scope'],
                    pool=None if self.pool is None else self.pool.receipt(),
                    identity_salt=self.salt, private_cache_profile=private_profile(self.salt),
                    approximate=True, qualification=QUALIFICATION, status=STATUS, default_selected=False,
                    publication_qualified=PUBLICATION_QUALIFIED,
                    serving='every routed expert in the bank computed from EXL3 in prefill and decode; '
                            'others original MXFP4',
                    counters=dict(self.stats),
                    bytes_read={'exl3': self.stats['exl3_bytes_read']},
                    per_layer_exl3_acquires=list(self.served), per_layer_fallback_acquires=list(self.fallback),
                    per_layer_slots={k: list(v) for k, v in self.slots.items()},
                    exl3_slot_fraction=(sum(self.slots['exl3']) / total_slots if total_slots else 0.0),
                    missing_count=len(self.missing),
                    missing_top=[[k[0], k[1], c] for k, c in bounded],
                    missing_experts=self.missing_experts(),
                    missing_path=None if self.missing_path is None else str(self.missing_path))


# ---- Adapter glue ------------------------------------------------------------------------

def store_rule(adapter):
    return getattr(getattr(adapter, 'weight_stream', None), 'exl3', None)


def bind_identity(adapter):
    if not enabled(adapter.kernel_policy):
        return None
    rule = store_rule(adapter)
    if rule is None:
        raise ValueError('exl3 selected but the expert store did not bind it')
    adapter.numerics_revision += ':source_exl3=' + NAME + ':salt=' + rule.salt
    adapter.profile += ':exl3-' + rule.salt[:16]
    adapter.numeric_profile += ':' + NAME + '_approximate_unqualified'
    return rule.salt


def apc_refusal_reason(policy):
    if not enabled(policy) or PUBLICATION_QUALIFIED:
        return None
    return APC_REASON


def model_class(base, policy):
    """Stamp cache rows of forwards that served EXL3; write the missing set."""
    if not enabled(policy):
        return base

    class Exl3SourceModel(base):
        source_exl3_enabled = True

        def _forward(self, inputs, cache=None, **options):
            rule = getattr(self, 'source_exl3', None)
            before = rule.stats['exl3_acquires'] if rule is not None else 0
            try:
                result = super()._forward(inputs, cache, **options)
            finally:
                if rule is not None:
                    rule.write_missing()
            if rule is not None and rule.stats['exl3_acquires'] != before:
                rule.stats['stamped_forwards'] += 1
                if cache is not None:
                    from .deepseek_v41_source_approx import stamp_cache
                    stamp_cache(cache, private_profile(rule.salt))
            return result

    return Exl3SourceModel


def configure_model(adapter):
    if enabled(adapter.kernel_policy):
        adapter.model.source_exl3 = store_rule(adapter)


def diagnostics(adapter):
    if not enabled(adapter.kernel_policy):
        return {'selected': False}
    rule = store_rule(adapter)
    if rule is None:
        value = {'selected': True, 'bound': False}
    else:
        lock = getattr(getattr(adapter, 'weight_stream', None), '_lock', None)
        if lock is None:
            value = rule.receipt()
        else:
            with lock:
                value = rule.receipt()
    value.update(apcv2_publication='refused: ' + APC_REASON)
    return value
