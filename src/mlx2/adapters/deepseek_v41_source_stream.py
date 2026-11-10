# SPDX-License-Identifier: MIT
"""Opt-in Source routed-expert streaming policies (all lossless, timing only).

Ported from the Kimi-K3.1-Pro expert-drafting experiment (2026-11-07) onto
the DS4.1 Source LFU store. See provenance/deepseek-v41-source-expert-lookahead.json.

* ``CycleAwareIndex`` (retention ``cycle_aware_v1``): evict the sampled
  resident expert with the lowest reuse-EMA / layer-distance-to-next-use.
  K3.1/Kimi Linear sim: 3-25% fewer misses than LRU when the cache is smaller
  than one token's working set; no gain at 50% cache. LFU lost to LRU there.
* Storage tiers: RAM -> internal NVMe -> (future remote-layer tier) ->
  external (T7). Accounting only; reads are unchanged. A remote tier slots in
  by rank through ``TierChain.insert``.
* ``RidgeLookaheadHeads``: per-(layer, depth) residual linear heads added to
  layer L's router scores applied to x_{L-d}. K3.1 held-out recall@16 trained
  vs untrained: d1 .79/.72, d2 .72/.62, d4 .65/.54.
* ``recommend_placement``: which shards earn the internal NVMe tier.

Predictions only choose optional early host reads; the real router selects
every expert and every miss is read exactly, so outputs cannot change.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np

from .deepseek_v41_source_lfu import LFUIndex

# ---------------------------------------------------------------- retention

CYCLE_AWARE = 'cycle_aware_v1'
CYCLE_ALPHA_DEFAULT = 0.1   # K3.1/Kimi Linear sweep optimum (ed_cache.py)
CYCLE_SAMPLE_DEFAULT = 64   # sampled victim set; 0 scans every resident object


class CycleAwareIndex(LFUIndex):
    """Reuse EMA per expert divided by layers until its layer next runs.

    Decode visits layers 0..L-1 cyclically, so an expert of the layer that just
    ran is not needed for ~L layers while one of the next layer is needed now.
    The EMA decays per observation of the expert's own layer (lazily, like
    DecayedLFUIndex); lifetime counts stay an honest atlas for persistence.
    """

    def __init__(self, layers, *, alpha=CYCLE_ALPHA_DEFAULT, sample=CYCLE_SAMPLE_DEFAULT, seed=0):
        if type(layers) is not int or layers < 1:
            raise ValueError('cycle-aware retention requires a positive layer count')
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha <= 1:
            raise ValueError('cycle-aware retention alpha must be in (0, 1]')
        if type(sample) is not int or not 0 <= sample <= 65536:
            raise ValueError('cycle-aware retention sample must be 0..65536')
        super().__init__()
        self.layers, self.alpha, self.sample = layers, float(alpha), sample
        self.epochs, self.ema = Counter(), {}
        self.position = layers - 1  # nothing observed: layer 0 runs next
        self._rng = random.Random(seed)
        self.counters = dict(cycle_victim_scans=0, cycle_victim_candidates=0)

    def reuse(self, key):
        value, stamp = self.ema.get(key, (0.0, self.epochs[key[0]]))
        return value * (1.0 - self.alpha) ** (self.epochs[key[0]] - stamp)

    def distance(self, key):
        """Layers until key's layer runs again; the current layer is furthest.

        Experts selected by the current observation are about to be acquired
        one by one (not leased yet); ``victim`` ranks them last instead.
        """
        gap = (key[0] - self.position) % self.layers
        return self.layers if gap == 0 else gap

    def score(self, key):
        return self.reuse(key) / self.distance(key)

    def observe_counts(self, layer, counts):
        counts = list(counts)
        if type(layer) is not int or not 0 <= layer < self.layers:
            raise ValueError('cycle-aware observation outside admitted layers')
        self.epochs[layer] += 1
        for expert, count in counts:
            key = (layer, expert)
            # Indicator update: selected in this observation of its layer.
            self.ema[key] = (self.reuse(key) * (1.0 - self.alpha) + self.alpha, self.epochs[layer])
            self.counts[key] += count
        self.position = layer
        self.current = frozenset((layer, expert) for expert, _ in counts)

    current = frozenset()

    def victim(self, resident, leased, eligible):
        pool = [k for k in resident if k not in leased and eligible(k)]
        if not pool:
            raise MemoryError('expert budget exhausted by protected/active objects')
        if self.sample and len(pool) > self.sample:
            pool = self._rng.sample(pool, self.sample)
        self.counters['cycle_victim_scans'] += 1
        self.counters['cycle_victim_candidates'] += len(pool)
        current = self.current
        return min(pool, key=lambda k: (k in current, self.score(k), self.last.get(k, 0), k))


# ------------------------------------------------------------ storage tiers

TIER_RAM = 'ram'
TIER_INTERNAL = 'internal_nvme'
TIER_REMOTE = 'remote_layer'
TIER_EXTERNAL = 'external'
# Lower rank = closer. A remote-layer tier (another Mac over TB/LAN holding
# whole layers; K3.1 split prototype 1.085x, expert-parallel lost at 0.52x)
# belongs between internal NVMe and the T7.
TIER_RANKS = {TIER_RAM: 0, TIER_INTERNAL: 10, TIER_REMOTE: 20, TIER_EXTERNAL: 30}
_INTERNAL_MOUNTS = ('/', '/System/Volumes/Data')


def mount_point(path):
    path = os.path.realpath(path)
    device = os.stat(path).st_dev
    while path != os.path.dirname(path):
        parent = os.path.dirname(path)
        if os.stat(parent).st_dev != device:
            return path
        path = parent
    return path


def classify_path(path, declared=None):
    """Tier of a storage path after following declared-tier symlinks.

    ``declared`` maps a file name or resolved path prefix to a tier name and
    wins over the mount heuristic (internal system/data volume vs /Volumes/*).
    """
    real = os.path.realpath(path)
    for key, tier in (declared or {}).items():
        if tier not in TIER_RANKS:
            raise ValueError('unknown storage tier: '+str(tier))
        if key == os.path.basename(path) or real == key or real.startswith(key.rstrip('/')+'/'):
            return tier
    mount = mount_point(real)
    if mount in _INTERNAL_MOUNTS:
        return TIER_INTERNAL
    return TIER_EXTERNAL if mount.startswith('/Volumes/') else TIER_INTERNAL


class StorageTier:
    """One below-RAM tier. ``holds(key)`` decides membership for accounting.

    ``read_job(key)`` may later return an alternative exact read (a remote
    tier); local file tiers return None and the store's own read is used.
    The store currently consults read_job only for accounting and refuses a
    tier that supplies one, until a remote tier is implemented and qualified.
    """

    def __init__(self, name, holds, rank=None):
        if name not in TIER_RANKS and rank is None:
            raise ValueError('custom storage tier needs an explicit rank')
        self.name, self.rank, self.holds = name, TIER_RANKS.get(name, rank) if rank is None else rank, holds

    def read_job(self, key):
        return None


class TierChain:
    def __init__(self, tiers=()):
        self.tiers = []
        for tier in tiers:
            self.insert(tier)

    def insert(self, tier):
        if any(t.name == tier.name for t in self.tiers):
            raise ValueError('storage tier already present: '+tier.name)
        self.tiers.append(tier)
        self.tiers.sort(key=lambda t: t.rank)

    def locate(self, key):
        for tier in self.tiers:
            if tier.holds(key):
                if tier.read_job(key) is not None:
                    raise ValueError('alternative tier reads are not admitted yet: '+tier.name)
                return tier.name
        return 'unknown'

    @classmethod
    def for_source(cls, artifact, shard_of, declared=None):
        """File tiers classified once per participating shard (lazily).

        ``shard_of(key)`` returns every shard holding the object; an object
        split across tiers is charged to its slowest one.
        """
        cache = {}

        def tier_of_shard(shard):
            if shard not in cache:
                cache[shard] = classify_path(Path(artifact['root'])/shard, declared)
            return cache[shard]

        def tier_of_key(key):
            return max((tier_of_shard(s) for s in shard_of(key)), key=TIER_RANKS.__getitem__)
        return cls([StorageTier(TIER_INTERNAL, lambda k: tier_of_key(k) == TIER_INTERNAL),
                    StorageTier(TIER_EXTERNAL, lambda k: tier_of_key(k) == TIER_EXTERNAL)])


class TierAccounting:
    """Misses, bytes and wall time by tier, plus RAM hits. Integer counters."""

    def __init__(self, chain):
        self.chain = chain
        self.counters = {}

    def _add(self, name, value):
        self.counters[name] = self.counters.get(name, 0) + value

    def hit(self):
        self._add('tier_hits_'+TIER_RAM, 1)

    def miss(self, key, nbytes, elapsed_ns=0):
        tier = self.chain.locate(key)
        self._add('tier_misses_'+tier, 1)
        self._add('tier_bytes_'+tier, int(nbytes))
        self._add('tier_read_ns_'+tier, int(elapsed_ns))
        return tier

    def receipt(self):
        return dict(order=[t.name for t in self.chain.tiers], counters=dict(sorted(self.counters.items())))


# ------------------------------------------------------- lookahead predictor

LOOKAHEAD_MAX_DEPTH = 4
HEADS_SCHEMA = 'mlx2.v41-source-ridge-lookahead-heads.v1'


def fit_ridge_head(hidden, base_scores, target_scores, lam=100.0):
    """Closed-form residual ridge head: target ~ base + hidden @ W.T + b.

    ``hidden`` (n, d) is x_{L-depth}; ``base_scores`` are layer L's router
    scores on that hidden (the untrained lookahead); ``target_scores`` are the
    true layer-L scores (or a routing-weight target). K3.1 optimum lam ~100-300.
    """
    x = np.asarray(hidden, np.float64)
    r = np.asarray(target_scores, np.float64) - np.asarray(base_scores, np.float64)
    if x.ndim != 2 or r.ndim != 2 or x.shape[0] != r.shape[0] or x.shape[0] < 2:
        raise ValueError('ridge head needs (n, d) hidden and (n, experts) scores')
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError('ridge lambda must be positive')
    mean_x, mean_r = x.mean(0), r.mean(0)
    xc, rc = x - mean_x, r - mean_r
    gram = xc.T @ xc + lam*np.eye(x.shape[1])
    weight = np.linalg.solve(gram, xc.T @ rc).T            # (experts, d)
    bias = mean_r - weight @ mean_x
    return weight.astype(np.float32), bias.astype(np.float32)


def recall_at(predicted, actual):
    """Fraction of actually routed experts present in the predicted set."""
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    hits = sum(len(set(map(int, p)) & set(map(int, a))) for p, a in zip(predicted, actual))
    return hits / max(1, actual.size)


def save_heads(path, heads, *, binding, hidden_size, experts, lam):
    """heads: {(layer, depth): (weight, bias)}; written atomically with a manifest."""
    path = Path(path)
    arrays = {}
    for (layer, depth), (weight, bias) in sorted(heads.items()):
        if (np.asarray(weight).shape != (experts, hidden_size) or np.asarray(bias).shape != (experts,)
                or not 1 <= depth <= LOOKAHEAD_MAX_DEPTH):
            raise ValueError('ridge head geometry mismatch')
        arrays[f'w_{layer}_{depth}'] = np.asarray(weight, np.float16)
        arrays[f'b_{layer}_{depth}'] = np.asarray(bias, np.float32)
    manifest = dict(schema=HEADS_SCHEMA, binding=binding, hidden_size=hidden_size, experts=experts,
                    lam=float(lam), heads=sorted([list(k) for k in heads]))
    arrays['manifest'] = np.frombuffer(json.dumps(manifest, sort_keys=True).encode(), np.uint8)
    tmp = path.with_name(path.name+'.tmp')
    with open(tmp, 'wb') as stream:
        np.savez(stream, **arrays)
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RidgeLookaheadHeads:
    """Loaded heads bound to one checkpoint fingerprint and geometry."""

    def __init__(self, path, *, binding, hidden_size, experts):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(bytes(data['manifest']).decode())
            if (manifest.get('schema') != HEADS_SCHEMA or manifest.get('binding') != binding
                    or manifest.get('hidden_size') != hidden_size or manifest.get('experts') != experts):
                raise ValueError('ridge lookahead heads do not bind this Source checkpoint')
            self.heads = {}
            for layer, depth in manifest['heads']:
                weight, bias = data[f'w_{layer}_{depth}'], data[f'b_{layer}_{depth}']
                if weight.shape != (experts, hidden_size) or bias.shape != (experts,):
                    raise ValueError('ridge head geometry mismatch')
                self.heads[(int(layer), int(depth))] = (weight, bias)
        self.lam = manifest['lam']
        self._device = {}

    def get(self, layer, depth):
        return self.heads.get((layer, depth))

    def device(self, layer, depth):
        """Lazy MLX copies of one head (fp16 weight, fp32 bias)."""
        key = (layer, depth)
        if key not in self.heads:
            return None
        if key not in self._device:
            import mlx.core as mx
            weight, bias = self.heads[key]
            self._device[key] = (mx.array(weight), mx.array(bias))
        return self._device[key]

    def receipt(self):
        return dict(sha256=self.sha256, heads=len(self.heads), lam=self.lam)


def parse_lookahead(policy):
    """Validated (depth, heads_path, loader) from an LFU expert policy dict."""
    depth = policy.get('lookahead_depth', 1)
    heads = policy.get('lookahead_heads')
    loader = policy.get('loader', 'fifo')
    if type(depth) is not int or not 1 <= depth <= LOOKAHEAD_MAX_DEPTH:
        raise ValueError('Source lookahead depth must be 1..4')
    if heads is not None and (not isinstance(heads, str) or not Path(heads).is_absolute()):
        raise ValueError('Source lookahead heads path must be absolute')
    if loader not in ('fifo', 'priority_v1'):
        raise ValueError('Source expert loader must be fifo or priority_v1')
    if depth > 1 and loader != 'priority_v1':
        # K3.1 sim: a FIFO channel queues demand behind deeper predictions.
        raise ValueError('Source lookahead depth > 1 requires loader priority_v1')
    return depth, heads, loader


# ---------------------------------------------------------- shard placement

# Advisory cost-model defaults (bytes/s; seconds per random row read at the
# reader's queue depth). T7 bandwidth is the K3.1 measurement range (35-235
# MB/s, exFAT); NVMe is the M5 internal 6.8 GB/s. Random-row latencies are
# assumptions until a measured pread probe replaces them.
PLACEMENT_DEFAULTS = dict(t7_bandwidth=150e6, nvme_bandwidth=6.8e9, t7_row_s=0.5e-3/8, nvme_row_s=0.08e-3/8,
                          engram_rows_per_token=24, ram_expert_bytes=75_000_000_000, top_k=6)


def recommend_placement(artifact, internal_bytes, *, expert_frequency=None, **overrides):
    """Rank shards for the internal NVMe tier by decode seconds saved per byte.

    * Expert shard: experts assumed RAM-resident (the most frequent that fit
      ``ram_expert_bytes``) cost nothing; every other selection is charged a
      full miss, expert_bytes x (1/T7 - 1/NVMe). Frequencies are selections
      per token from an atlas, or uniform top_k/experts without one.
    * Engram shard: ``engram_rows_per_token`` random row reads per table per
      token at the (T7 - NVMe) row latency. Prefill multiplies this by chunk
      rows, so prefill is the stronger argument for Engram on the fast tier.
    * Dense/backbone shard: read once at load; no per-token cost.
    Returns (internal, external, scores). Advisory: it moves no file.
    """
    unknown = set(overrides) - set(PLACEMENT_DEFAULTS)
    if unknown:
        raise ValueError('unknown placement parameter: '+', '.join(sorted(unknown)))
    p = dict(PLACEMENT_DEFAULTS, **overrides)
    sizes = {file: seal['size'] for file, seal in artifact.get('seals', {}).items()}
    shards, experts = {}, {}
    for name, row in artifact['tensors'].items():
        info = shards.setdefault(row['file'], dict(bytes=0, engram_tables=set(), saving=0.0))
        info['bytes'] += row['size']
        if '.engram.embed.' in name:
            info['engram_tables'].add(name.split('.')[1])
        elif '.ffn.experts.' in name:
            parts = name.split('.')
            record = experts.setdefault((int(parts[1]), int(parts[4])), dict(bytes=0, files=Counter()))
            record['bytes'] += row['size']
            record['files'][row['file']] += row['size']
    per_layer = Counter(layer for layer, _ in experts)
    frequency = {key: (expert_frequency.get(key, 0.0) if expert_frequency else p['top_k']/per_layer[key[0]])
                 for key in experts}
    budget = p['ram_expert_bytes']
    for key in sorted(experts, key=lambda k: (-frequency[k], k)):
        record = experts[key]
        if budget >= record['bytes']:
            budget -= record['bytes']
            continue
        penalty = frequency[key]*record['bytes']*(1/p['t7_bandwidth'] - 1/p['nvme_bandwidth'])
        for file, part in record['files'].items():
            shards[file]['saving'] += penalty*part/record['bytes']
    for info in shards.values():
        info['saving'] += len(info['engram_tables'])*p['engram_rows_per_token']*(p['t7_row_s'] - p['nvme_row_s'])
    scores = {name: info['saving']/max(1, sizes.get(name, info['bytes'])) for name, info in shards.items()}
    internal, external, used = [], [], 0
    for name in sorted(shards, key=lambda n: (-scores[n], n)):
        size = sizes.get(name, shards[name]['bytes'])
        if used + size <= internal_bytes:
            internal.append(name)
            used += size
        else:
            external.append(name)
    return internal, external, scores


# ----------------------------------------------------------- Engram row I/O

ENGRAM_IO_CHOICES = ('reference', 'tiered_v1')
# tiered_v1 rationale: the reference Source Engram reader keeps an 8 MiB LRU
# per table (16K bf16 rows; one decode token needs 24 rows per table, a
# 2048-row prefill chunk ~49K) and at most 8 pread workers. A 256 MiB LRU
# (512K rows) keeps a long context's repeated n-grams resident and 16 workers
# deepen the random-read queue, which internal NVMe and USB SSDs both reward.
# Rows are exact source rows either way: lossless, timing only.
ENGRAM_TIERED_CACHE_BYTES = 256 << 20
ENGRAM_TIERED_WORKERS = 16


def engram_io_settings(policy, read_workers):
    if policy not in ENGRAM_IO_CHOICES:
        raise ValueError('invalid Source Engram I/O policy')
    if policy == 'tiered_v1':
        # At least 16 workers; a larger configured read_workers is honoured up
        # to EngramRows' 32-worker bound.
        return dict(workers=max(ENGRAM_TIERED_WORKERS, min(32, read_workers)), cache_bytes=ENGRAM_TIERED_CACHE_BYTES)
    return dict(workers=min(8, read_workers), cache_bytes=None)


def engram_io_receipt(policy, tables, declared=None):
    result = dict(policy=policy, status='implemented_untested_unqualified', tables={})
    for layer, table in tables.items():
        try:
            tier = classify_path(table.path, declared)
        except OSError as error:
            tier = 'unavailable: '+type(error).__name__
        stats = getattr(table, 'stats', {})
        result['tables'][str(layer)] = dict(storage_tier=tier, workers=table.workers,
            cache_budget_bytes=table.cache_rows*table.row_bytes, cache_hits=stats.get('cache_hits', 0),
            cache_misses=stats.get('cache_misses', 0), bytes_read=stats.get('bytes_read', 0))
    return result
