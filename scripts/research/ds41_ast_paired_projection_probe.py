# SPDX-License-Identifier: MIT
"""Resident paired-output projection experiment; private trees only.

No GPU work on import. The caller supplies its live bank, ordinary-exact
reference, source-bound unit and lease guard. Joining coefficients is setup;
timings include the complete uninstrumented tree and its normal completion.
"""
from contextlib import nullcontext, contextmanager
from pathlib import Path
import json
import time

_QUARANTINE = []


def order():
    return ('reference', 'joined', 'joined', 'reference')


@contextmanager
def capture_inputs():
    """Borrow actual pair inputs without evaluating or modifying results."""
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
    original = dense.paired_linear_tile
    captured = {}
    def call(model, x, a, b, rounded=True):
        if a.endswith(('.attn_q_a.weight', '.ffn_gate_shexp.weight')):
            key = (a, b)
            if key in captured:
                raise ValueError('Projection capture pair repeated within one tree')
            captured[key] = (model, x)
        return original(model, x, a, b, rounded=rounded)
    dense.paired_linear_tile = call
    try:
        yield captured
    finally:
        conflict = dense.paired_linear_tile is not call
        dense.paired_linear_tile = original
        if conflict:
            raise RuntimeError('Projection capture entry changed concurrently')


def validate_pairs(prepared, captured, live, xp):
    if len(captured) != 80:
        raise ValueError('Projection capture requires all80 actual eligible pairs')
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
    from mlx2.adapters.deepseek_v41_ast_paired_projection import installed
    from ds41_resident_prefill_model_gate import _array_compare
    original = dense.paired_linear_tile
    cases = []
    for (a, b), (model, x) in captured.items():
        if live() is False:
            raise RuntimeError('Projection component ownership lost')
        reference = original(model, x, a, b, rounded=False)
        with installed(prepared) as metrics:
            joined = dense.paired_linear_tile(model, x, a, b, rounded=False)
            xp.eval(reference, joined)
            xp.synchronize()
            counts = dict(metrics)
        comparisons = [_array_compare(left, right, xp) for left, right in zip(reference, joined)]
        row = dict(names=[a, b], input_shape=list(x.shape), comparisons=comparisons,
                   joined_calls=counts.get('joined_calls'), exact=len(comparisons) == 2 and all(v['equal'] for v in comparisons))
        cases.append(row)
        if not row['exact'] or row['joined_calls'] != 1:
            raise RuntimeError('Actual paired projection raw component differs: ' + a)
    return dict(cases=cases, pairs=len(cases), all_raw_exact=True, rounded=False,
                scope='Actual ordinary-tree inputs and immutable original coefficients; outside full-tile timers')


def run(adapter, unit, live, prefix, anchor, paths, discovery, discovery_states):
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree, RESIDENT_EXPERT_LAW
    if mx.default_device() != mx.gpu or not callable(live):
        raise ValueError('Paired projection probe requires parent-owned GPU')
    if live() is False:
        raise RuntimeError('Paired projection probe ownership lost')
    from mlx2.adapters.deepseek_v41_ast_paired_projection import prepare, installed
    from ds41_resident_prefill_model_gate import _checkpoint, _array_compare, _cache_compare, _counters
    from ds41_ast_batched_model_probe import _delta, _materialize, _close, _write
    model = adapter.model
    if (getattr(adapter.weights, 'source_resident_bank_session', None) is None
            or getattr(model, '_mlx2_ds41_ast_active', None) is not None):
        raise ValueError('Paired projection probe needs idle target and resident bank')
    unit = Path(unit)
    path = unit / 'ast-paired-projection-probe.json'
    if path.exists() or path.with_name(path.name + '.tmp').exists():
        raise FileExistsError('Paired projection evidence exists')
    spec = json.loads((unit / 'spec.json').read_text())
    report = dict(schema='mlx2.ds41-ast-paired-projection-probe.v1', checkpoint=_checkpoint(adapter),
                  source_revision=spec['source_revision'], source_snapshot_sha256=spec['source_snapshot_sha256'],
                  status='preparing', prefix_tokens=prefix[0].offset, qualified=False,
                  rate_eligible=False, APCv2_publication=False, controlled_performance=False,
                  numerical_law='Same private stock M1 batch-axis dot; output rows joined; actual raw gate required',
                  order=list(order()), timed_samples_per_arm=2, independent_ABBA_repetitions=1,
                  warmups=[], arms=[])
    _write(path, report)
    prepared = None
    prefix_meta = [tuple(row.meta_state) for row in prefix]
    prefix_arrays = [list(row.state) for row in prefix]
    try:
        started = time.perf_counter()
        prepared = prepare(model)
        # The prepared object owns all joined arrays through final drain. No
        # coefficient packing/allocation is included in the measured arms.
        mx.eval(prepared.evaluation_roots())
        mx.synchronize()
        report['preparation'] = dict(seconds=time.perf_counter() - started,
                                     receipt=prepared.receipt(), inside_timer=False)
        if live() is False:
            raise RuntimeError('Paired projection probe ownership lost')
        report['status'] = 'warming'
        _write(path, report)
        for sequence, (policy, timed) in enumerate([(p, False) for p in ('reference', 'joined')]
                                                  + [(p, True) for p in order()]):
            tile = states = None
            drained = True
            try:
                if live() is False:
                    raise RuntimeError('Paired projection probe ownership lost')
                mx.synchronize()
                before = _counters(adapter)
                with (capture_inputs() if sequence == 0 else nullcontext(None)) as captured, \
                        (installed(prepared) if policy == 'joined' else nullcontext({})) as metrics:
                    drained = False
                    started = time.perf_counter()
                    tile = execute_batched_tree(model, prefix, anchor, paths,
                        expert_law=RESIDENT_EXPERT_LAW, forward_law='fully_batched_m1_v1')
                    mx.eval(tile.logits, tile.features)
                    mx.synchronize()
                    seconds = time.perf_counter() - started
                    drained = True
                    actual_metrics = dict(metrics)
                delta = _delta(_counters(adapter), before)
                zero = all(delta.get(k, 0) == 0 for k in ('expert_bytes_read', 'expert_uploads',
                    'engram_bytes_read', 'host_route_materializations', 'ordinary_replays'))
                if live() is False:
                    raise RuntimeError('Paired projection probe ownership lost')
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
                    len(row.state) == len(prefix_arrays[i]) and
                    all(_array_compare(a, b, mx)['equal'] for a, b in zip(prefix_arrays[i], row.state))
                    for i, row in enumerate(prefix))
                row = dict(sequence=sequence, policy=policy, timed=timed, seconds=seconds,
                           counter_delta=delta, zero_io_host_routes_or_replay=zero, metrics=actual_metrics,
                           nodes=nodes, all_nodes_raw_exact=all(n['exact'] for n in nodes), prefix_unchanged=prefix_exact,
                           timing_scope='One full tree with normal completion; all checks and coefficient preparation outside timer')
                report['arms' if timed else 'warmups'].append(row)
                _write(path, report)
                if not (zero and prefix_exact and row['all_nodes_raw_exact']):
                    raise RuntimeError('Paired projection raw/residency/prefix gate failed')
                if sequence == 0:
                    tile.transaction.abort()
                    component_started = time.perf_counter()
                    report['components'] = validate_pairs(prepared, captured, live, mx)
                    report['components']['seconds'] = time.perf_counter() - component_started
                    _write(path, report)
                if policy == 'joined' and actual_metrics.get('joined_calls', 0) != 80:
                    raise RuntimeError('Paired projection did not execute all80 expected pairs')
            finally:
                if not drained:
                    try:
                        mx.synchronize()
                        drained = True
                    except BaseException:
                        _QUARANTINE.append((adapter, prefix, prepared, discovery, discovery_states, tile, states))
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
        # A failed coefficient upload or helper call can leave device roots in
        # flight even before an arm owns a transaction. Retain on failed drain.
        try:
            mx.synchronize()
        except BaseException:
            _QUARANTINE.append((adapter, prefix, prepared, discovery, discovery_states))
        report.update(status='failed', passed=False, error=dict(type=type(error).__name__, message=str(error)))
        _write(path, report)
        raise
