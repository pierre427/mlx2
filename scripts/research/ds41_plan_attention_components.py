# SPDX-License-Identifier: MIT
"""Prepared bounded short-attention suite; never starts its own GPU owner.

Call ``run(mx, check_live)`` only inside an authorized ownership/thermal unit.
The callback must return True. Import, inventory and CPU oracles perform no
MLX import, shader creation, model load or device work. No CLI runs this suite.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import hashlib
import json
from pathlib import Path
import statistics
import time
import re

SEED = 4102611
REPEATS = 7
LOCAL_ROWS = (1, 127, 128)
SELECTED_ROWS = (0, 1, 511, 512)
SCOPES = ('fresh_selected_gather', 'reused_selected_gather')
VARIANTS = ('ordinary_reference', 'joint_f32', 'native_hg1', 'native_hg2', 'native_hg4')
ARRAY_BUDGET_BYTES = 32 << 20
RTOL, ATOL = 2e-4, 2e-5


class ComponentSuiteError(RuntimeError):
    """An execution error preserving completed raw samples for the controller."""
    def __init__(self, message, partial_result):
        super().__init__(message)
        self.partial_result = partial_result


def source_binding(root):
    """A frozen component runs without .git; its controller seals the revision."""
    root=Path(root)
    frozen=root/'desktop-source.json'
    if frozen.exists():
        if frozen.is_symlink() or not frozen.is_file() or frozen.stat().st_size>1<<20:
            raise ValueError('Invalid frozen attention source metadata')
        value=json.loads(frozen.read_text())
        revision=value.get('source_revision')
        digest=value.get('source_sha256')
        if (type(revision) is not str or re.fullmatch('[0-9a-f]{40}',revision) is None
                or type(digest) is not str or re.fullmatch('[0-9a-f]{64}',digest) is None):
            raise ValueError('Invalid frozen attention source identities')
        return revision,digest
    import subprocess
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if re.fullmatch('[0-9a-f]{40}',revision) is None:
        raise ValueError('Invalid attention source commit')
    return revision,None


def plan():
    cells = [dict(name=f'L{local}_P{selected}', local_rows=local,
                  selected_rows=selected, control='finite')
             for local in LOCAL_ROWS for selected in SELECTED_ROWS]
    cells += [dict(name='masked_tail', local_rows=128, selected_rows=512, control='masked_tail'),
              dict(name='sink_only', local_rows=1, selected_rows=0, control='sink_only'),
              dict(name='masked_nan', local_rows=128, selected_rows=512, control='masked_nan')]
    # One cell at a time; include host/device original, gathered/sanitized
    # operands, joined KV, scores and all five output owners conservatively.
    maximum = 2*(64*512+128*512+514*512+512+128+64)*4
    maximum += (4*640*512+6*64*512+4*64*641)*4
    if maximum > ARRAY_BUDGET_BYTES:
        raise ValueError('Short-attention component inventory exceeds its bound')
    return dict(schema='mlx2.ds41-plan-attention-components-plan.v1', seed=SEED,
        repeats=REPEATS, cells=cells, scopes=list(SCOPES), variants=list(VARIANTS),
        case_count=len(cells)*len(SCOPES)*len(VARIANTS),
        heads=64, head_dim=512, query_rows=1, physical_width=1,
        maximum_local_rows=128, maximum_selected_rows=512,
        conservative_operand_and_scratch_bound_bytes=maximum,
        array_budget_bytes=ARRAY_BUDGET_BYTES,
        reference_masked_nan='observational diagnostic only; ordinary PV may propagate masked storage NaNs',
        numerical_acceptance=dict(finite=True, rtol=RTOL, atol=ATOL,
            exact_reduction_equality_claimed=False),
        clock='fresh graph build plus mx.eval(output); warm first-use separate; diagnostics/live checks outside clock',
        checkpoint='synthetic-short-attention@seed4102611',
        target_checkpoint_not_run='DeepSeek-V4.1-Flash-1017-Source-Hot60GB-Q1g64@927325be8884e00a7bf60b9aa7d540b167a5efec569b9c7c30b1d34c2793d000',
        source_revision='caller must bind current mlx2 commit and sealed source snapshot',
        qualification=False, performance_qualified=False, model_rate_claim=False)


def _live(callback):
    if callback() is not True:
        raise RuntimeError('Short-attention ownership/live callback did not return True')


def _fixture(cell, *, seed=SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    local_rows, count = cell['local_rows'], cell['selected_rows']
    q = rng.normal(0, .2, (1, 64, 1, 512)).astype(np.float32)
    local = rng.normal(0, .25, (1, 1, local_rows, 512)).astype(np.float32)
    mask = np.ones((1, 1, 1, local_rows), bool)
    sinks = rng.normal(0, .3, (64,)).astype(np.float32)
    if count:
        pooled = rng.normal(0, .25, (1, count+2, 512)).astype(np.float32)
        indices = np.arange(count, dtype=np.uint32).reshape(1, 1, count)
        valid = np.ones(indices.shape, bool)
    else:
        pooled = indices = valid = None
    if cell['control'] in ('masked_tail', 'masked_nan'):
        mask[..., -3:] = False
        valid[..., -3:] = False
        poison = np.nan if cell['control'] == 'masked_nan' else np.float32(2048)
        local[:, :, -3:] = poison
        pooled[:, count-3:count] = poison
    elif cell['control'] == 'sink_only':
        mask[:] = False
        sinks[:] = np.float32(-1000)
    return q, local, mask, pooled, indices, valid, sinks


def numpy_oracle(operands):
    """Independent F64 mathematical oracle, never a raw F32 parity oracle."""
    import numpy as np
    q, local, mask, pooled, indices, valid, sinks = operands
    vectors = [local[0, 0, i].astype(np.float64)
               for i in range(local.shape[2]) if mask[0, 0, 0, i]]
    if pooled is not None:
        vectors += [pooled[0, int(indices[0, 0, i])].astype(np.float64)
                    for i in range(indices.shape[-1]) if valid[0, 0, i]]
    if not vectors:
        return np.zeros(q.shape, np.float32)
    values = np.stack(vectors)
    queries = q[0, :, 0].astype(np.float64)*512**-0.5
    logits = queries @ values.T
    top = np.maximum(np.max(logits, axis=-1), sinks.astype(np.float64))
    mass = np.exp(logits-top[:, None])
    denom = mass.sum(axis=-1)+np.exp(sinks.astype(np.float64)-top)
    return ((mass @ values)/denom[:, None])[None, :, None].astype(np.float32)


def _diagnostic(value, expected):
    import numpy as np
    actual = np.array(value, dtype=np.float32, copy=True)
    shape_valid = actual.shape == expected.shape
    finite = shape_valid and bool(np.isfinite(actual).all())
    delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64)) if shape_valid else None
    within = finite and bool(np.all(delta <= ATOL+RTOL*np.abs(expected)))
    window = []
    if finite:
        for index in np.argsort(delta.reshape(-1))[-8:][::-1]:
            window.append(dict(flat_index=int(index), actual=float(actual.reshape(-1)[index]),
                               expected=float(expected.reshape(-1)[index]), absolute_error=float(delta.reshape(-1)[index])))
    return dict(shape=list(actual.shape), finite=finite, tolerance_passed=within,
        maximum_absolute_error=float(delta.max()) if finite else None,
        output_f32_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        full_array_compared=True, largest_error_window=window,
        classification='finite_tolerance_valid' if within else 'numerical_validity_failed',
        exact_reduction_equality_claimed=False)


def _measure(mx, function):
    begin = time.perf_counter_ns()
    value = function()
    submitted = time.perf_counter_ns()
    mx.eval(value)
    done = time.perf_counter_ns()
    return value, dict(host_graph_submit_seconds=(submitted-begin)/1e9,
        completion_envelope_seconds=(done-submitted)/1e9,
        total_seconds=(done-begin)/1e9)


def run(mx, check_live=None):
    """Run bounded components inside the caller's authorized GPU window.

    A missing/non-callable callback or CPU backend refuses before any import
    of model/kernel helpers or shader creation. The outer controller owns
    leases, thermal gating, source seals, atomic receipts and final cleanup.
    This function neither acquires ownership nor waives a thermal gate.
    """
    if not callable(check_live):
        raise ValueError('Short-attention suite requires an ownership/live callback')
    _live(check_live)
    if mx.default_device() != mx.gpu:
        raise ValueError('Short-attention suite requires the caller-owned GPU backend')
    from mlx2.runtime.models import deepseek_v41 as reference_module
    from mlx2.runtime.models.deepseek_v41 import selected_attention
    from mlx2.runtime.models import deepseek_v41_short_attention as candidate
    inventory = plan()
    root = Path(__file__).resolve().parents[2]
    source_revision,source_snapshot=source_binding(root)
    inventory['source_revision'] = source_revision
    report = dict(schema='mlx2.ds41-plan-attention-components.v1', plan=inventory,
        complete=False, cases=[], rotated_orders=[], numerical_validity_passed=False,
        raw_timing_samples_retained=True, exact_reduction_equality_claimed=False,
        gpu_kernel_timestamp_claim=False, qualification=False,
        performance_qualified=False, model_rate_claim=False,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        attention_implementation_sha256=candidate.implementation_identity(),
        reference_implementation_sha256=hashlib.sha256(Path(reference_module.__file__).read_bytes()).hexdigest(),
        source_revision=source_revision,
        source_snapshot_sha256=source_snapshot,
        checkpoint=inventory['checkpoint'],
        timing_scope='normal complete graph submission/completion; inputs resident; fresh or shared gathered scope explicit; warm first-use and diagnostics excluded')
    try:
        for cell_index, cell in enumerate(inventory['cells']):
            _live(check_live)
            host = _fixture(cell, seed=SEED+cell_index)
            expected = numpy_oracle(host)
            device = tuple(None if value is None else mx.array(value) for value in host)
            mx.eval(*(value for value in device if value is not None))
            q, local, mask, pooled, indices, valid, sinks = device
            reused = None if pooled is None else pooled[0, indices[0]][None]
            if reused is not None:
                mx.eval(reused)
            for scope in SCOPES:
                gathered = reused if scope == 'reused_selected_gather' else None
                functions = {
                    'ordinary_reference': lambda: selected_attention(*device, gathered=gathered),
                    'joint_f32': lambda: candidate.joint_attention(*device, gathered=gathered,
                        numerical_profile=candidate.JOINT_PROFILE),
                }
                for heads in (1, 2, 4):
                    functions[f'native_hg{heads}'] = lambda heads=heads: candidate.native_attention(
                        *device, gathered=gathered, head_group=heads,
                        numerical_profile=candidate.NATIVE_PROFILE)
                rows = []
                for variant in VARIANTS:
                    _live(check_live)
                    row = dict(cell=cell['name'], geometry=cell, scope=scope, variant=variant,
                        numerical_profile=('ordinary_reference' if variant == 'ordinary_reference' else
                            candidate.JOINT_PROFILE if variant == 'joint_f32' else candidate.NATIVE_PROFILE),
                        head_group=int(variant[-1]) if variant.startswith('native_hg') else None,
                        samples=[], sample_numerical_diagnostics=[],
                        warm_first_use_scope='includes first graph compilation, encoding and execution; excluded from steady samples')
                    report['cases'].append(row)
                    value, warm = _measure(mx, functions[variant])
                    row['warm_first_use'] = warm
                    row['warm_diagnostic'] = _diagnostic(value, expected)
                    observational = variant == 'ordinary_reference' and cell['control'] == 'masked_nan'
                    row['observational_reference_only'] = observational
                    row['eligible_for_timing'] = not observational and row['warm_diagnostic']['tolerance_passed']
                    if observational:
                        row['diagnostic_scope'] = 'Actual ordinary reference on poisoned masked storage; no rate or quality verdict'
                    rows.append(row)
                for repeat in range(REPEATS):
                    rotation = repeat % len(VARIANTS)
                    order = VARIANTS[rotation:]+VARIANTS[:rotation]
                    report['rotated_orders'].append(dict(cell=cell['name'], scope=scope,
                        repeat=repeat, variants=list(order)))
                    for variant in order:
                        row = next(row for row in rows if row['variant'] == variant)
                        if not row['eligible_for_timing']:
                            continue
                        _live(check_live)
                        value, timing = _measure(mx, functions[variant])
                        row['samples'].append(timing)
                        diagnostic = _diagnostic(value, expected)
                        row['sample_numerical_diagnostics'].append(diagnostic)
                        if not diagnostic['tolerance_passed']:
                            row['eligible_for_timing'] = False
                for row in rows:
                    if row['samples']:
                        row['medians'] = {name: statistics.median(s[name] for s in row['samples'])
                                          for name in row['samples'][0]}
                # Per-cell owners are replaced next iteration. Each timed or
                # warm graph has completed before any operand can be released.
        checks = [row for row in report['cases'] if not row['observational_reference_only']]
        report['numerical_validity_passed'] = all(row['warm_diagnostic']['tolerance_passed']
            and len(row['samples']) == REPEATS
            and all(d['tolerance_passed'] for d in row['sample_numerical_diagnostics']) for row in checks)
        report['complete'] = True
        return report
    except Exception as error:
        report['execution_error'] = dict(type=type(error).__name__, message=str(error))
        raise ComponentSuiteError('Short-attention component execution failed; retain partial_result', report) from error
