# SPDX-License-Identifier: MIT
"""Scoped completion-boundary diagnosis of an already-resident exact tree.

Import does not run MLX work. Timings include Python, evaluation and fences;
they are an intervention, never a natural GPU stage budget or serving rate.
No implementation, coefficient, numerical law or publication policy changes.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import time

_QUARANTINE = []


def _roots(value, array_type, seen=None):
    """Only arrays, containers and explicit segment evaluation roots."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, array_type):
        return [value]
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, (tuple, list)):
        items = value
    elif callable(getattr(type(value), 'evaluation_roots', None)):
        items = value.evaluation_roots()
    else:
        return []
    return [root for item in items for root in _roots(item, array_type, seen)]


class StageObserver:
    """Single-thread diagnostic observer; nested intervals stay explicit."""
    def __init__(self, xp, *, clock=time.perf_counter):
        self.xp, self.clock = xp, clock
        self.events, self.stack, self.restores = [], [], []

    def drain(self, value):
        roots = _roots(value, self.xp.array)
        if roots:
            self.xp.eval(*roots)
        self.xp.synchronize()

    def wrap(self, original, label, *, layer=None, extra=None):
        if getattr(original, '_ds41_stage_observer', False):
            raise ValueError('Stage observer cannot nest another observer')

        @wraps(original)
        def observed(*args, **kwargs):
            # Materialize only actual argument roots; this wait is outside the
            # child's interval but inside its parent (when present).
            self.drain((args, kwargs))
            event = dict(stage=label, layer=layer,
                         depth=len(self.stack), parent=self.stack[-1] if self.stack else None,
                         start_seconds=self.clock(), completed=False)
            index = len(self.events)
            self.events.append(event)
            self.stack.append(index)
            try:
                result = original(*args, **kwargs)
                self.drain((result, extra(args, kwargs, result) if extra else None))
                event['completed'] = True
                return result
            finally:
                event['stop_seconds'] = self.clock()
                event['seconds'] = event['stop_seconds'] - event['start_seconds']
                assert self.stack.pop() == index
        observed._ds41_stage_observer = True
        return observed

    def patch(self, owner, name, label, **kwargs):
        original = getattr(owner, name)
        had_own = name in vars(owner)
        wrapped = self.wrap(original, label, **kwargs)
        setattr(owner, name, wrapped)
        self.restores.append((owner, name, original, wrapped, had_own))

    def restore(self):
        conflicts = []
        for owner, name, original, wrapped, had_own in reversed(self.restores):
            if getattr(owner, name) is not wrapped:
                conflicts.append(name)
            # Restore even on an injected error, including inherited methods.
            if had_own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.restores.clear()
        if conflicts:
            raise RuntimeError('Stage observer patch changed concurrently: ' + ','.join(conflicts))

    def summary(self):
        result = {}
        for index, row in enumerate(self.events):
            children = sum(v['seconds'] for v in self.events if v['parent'] == index)
            row['exclusive_seconds'] = row['seconds'] - children
            value = result.setdefault(row['stage'], dict(calls=0, seconds=0., exclusive_seconds=0., depth=row['depth']))
            value['calls'] += 1
            value['seconds'] += row['seconds']
            value['exclusive_seconds'] += row['exclusive_seconds']
        return result


@contextmanager
def observe(model, xp):
    from mlx2.adapters import deepseek_v41_ast_batched_attention_exact as attention
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
    from mlx2.adapters import deepseek_v41_ast_batched_moe as moe
    if getattr(model, '_mlx2_ds41_ast_active', None) is not None or model._weight_executions:
        raise ValueError('Stage observer requires an idle target')
    observer = StageObserver(xp)
    try:
        for name, label in [('engram_tile', 'engram'), ('router_tile', 'router'),
                            ('shared_expert_tile', 'shared_expert'), ('head_tile', 'head')]:
            observer.patch(dense, name, label)
        observer.patch(attention, 'attention_half_tile', 'attention_half')
        observer.patch(attention, 'attention_core', 'attention_core')
        observer.patch(attention, '_publish_source', 'cache_compression')
        observer.patch(attention, '_select', 'index_selection', extra=lambda args, kwargs, result: args[5])
        observer.patch(moe, 'moe_tile', 'routed_experts')
        for index, block in enumerate(model.layers):
            observer.patch(block, 'finish', 'ffn_finish', layer=index)
        yield observer
    finally:
        observer.restore()


def run(adapter, unit, check_live, prefix, anchor, paths, discovery, discovery_states):
    import json
    import mlx.core as mx
    if mx.default_device() != mx.gpu or not callable(check_live):
        raise ValueError('Exact stage profile requires a parent-owned GPU')
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree, RESIDENT_EXPERT_LAW
    from ds41_resident_prefill_model_gate import _array_compare, _cache_compare, _counters, _checkpoint
    from ds41_ast_batched_model_probe import _materialize, _close, _delta, _write
    if check_live() is False:
        raise RuntimeError('Exact stage profile ownership lost')
    unit = Path(unit)
    path = unit / 'ast-exact-stage-profile.json'
    if path.exists() or path.with_name(path.name + '.tmp').exists():
        raise FileExistsError('Exact stage profile evidence already exists')
    model = adapter.model
    if getattr(adapter.weights, 'source_resident_bank_session', None) is None:
        raise ValueError('Exact stage profile requires an attached resident bank')
    if len(discovery_states) != 16:
        raise ValueError('Exact stage profile needs all16 raw reference states')
    spec = json.loads((unit / 'spec.json').read_text())
    report = dict(schema='mlx2.ds41-ast-exact-stage-profile.v1', checkpoint=_checkpoint(adapter),
                  source_revision=spec['source_revision'], source_snapshot_sha256=spec['source_snapshot_sha256'],
                  status='running', qualified=False, rate_eligible=False, APCv2_publication=False,
                  timing_scope='Synchronized completion boundaries; Python+evaluation+fences; observer changes scheduling',
                  natural_GPU_stage_budget=False, nested_intervals_additive=False, arms=[])
    _write(path, report)
    prefix_states = [list(row.state) for row in prefix]
    prefix_meta = [tuple(row.meta_state) for row in prefix]
    try:
        for instrumented in (False, True):
            tile = states = observer = None
            drained = True
            try:
                if check_live() is False:
                    raise RuntimeError('Exact stage profile ownership lost')
                mx.synchronize()
                before = _counters(adapter)
                drained = False
                started = time.perf_counter()
                if instrumented:
                    with observe(model, mx) as observer:
                        tile = execute_batched_tree(model, prefix, anchor, paths,
                            expert_law=RESIDENT_EXPERT_LAW, forward_law='fully_batched_m1_v1')
                        mx.eval(tile.logits, tile.features)
                        mx.synchronize()
                else:
                    tile = execute_batched_tree(model, prefix, anchor, paths,
                        expert_law=RESIDENT_EXPERT_LAW, forward_law='fully_batched_m1_v1')
                    mx.eval(tile.logits, tile.features)
                    mx.synchronize()
                seconds = time.perf_counter() - started
                drained = True
                delta = _delta(_counters(adapter), before)
                if check_live() is False:
                    raise RuntimeError('Exact stage profile ownership lost')
                drained = False
                states = _materialize(tile)
                mx.eval([[row.state for row in cache] for cache in states])
                drained = True
                nodes = []
                for node in range(16):
                    checks = dict(logits=_array_compare(discovery.logits[:, node:node+1], tile.logits[:, node:node+1], mx),
                                  features=_array_compare(discovery.features[:, node:node+1], tile.features[:, node:node+1], mx),
                                  cache=_cache_compare(discovery_states[node], states[node], mx))
                    nodes.append(dict(node=node, checks=checks, exact=all(v['equal'] for v in checks.values())))
                prefix_exact = all(tuple(row.meta_state) == prefix_meta[i] and
                    len(row.state) == len(prefix_states[i]) and
                    all(_array_compare(a, b, mx)['equal'] for a, b in zip(prefix_states[i], row.state))
                    for i, row in enumerate(prefix))
                zero = all(delta.get(k, 0) == 0 for k in ('expert_bytes_read', 'expert_uploads',
                    'engram_bytes_read', 'host_route_materializations', 'ordinary_replays'))
                arm = dict(instrumented=instrumented, seconds=seconds, counter_delta=delta,
                           zero_io_host_routes_or_replay=zero, nodes=nodes, prefix_unchanged=prefix_exact,
                           all_nodes_raw_exact=all(n['exact'] for n in nodes))
                if observer is not None:
                    arm.update(stages=observer.summary(), events=observer.events,
                               top_level_stage_seconds=sum(e['seconds'] for e in observer.events if e['depth'] == 0))
                    arm['outside_observed_stage_seconds'] = seconds - arm['top_level_stage_seconds']
                report['arms'].append(arm)
                _write(path, report)
                if not (zero and prefix_exact and arm['all_nodes_raw_exact']):
                    raise RuntimeError('Exact stage profile raw/residency/prefix gate failed')
            finally:
                if not drained:
                    try:
                        mx.synchronize()
                        drained = True
                    except BaseException:
                        _QUARANTINE.append((adapter, prefix, discovery, discovery_states, tile, states))
                if drained:
                    if tile is not None:
                        tile.transaction.abort()
                    if states is not None:
                        for cache in states:
                            _close(cache)
        report.update(status='passed', passed=True)
        _write(path, report)
        return report
    except BaseException as error:
        report.update(status='failed', passed=False, error=dict(type=type(error).__name__, message=str(error)))
        _write(path, report)
        raise
