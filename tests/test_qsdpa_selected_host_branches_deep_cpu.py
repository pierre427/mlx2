"""A3 deeper host controls: real fork plus NumPy-backed source dispatch.

Checkpoint no_model@not_applicable. No real MLX import or device work.
Run with --noconftest; the imported fixture installs inert MLX stand-ins.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import signal
import subprocess
import sys
import textwrap
from concurrent.futures import CancelledError

import numpy as np
import pytest
from test_qsdpa_selected_host_branches_cpu import (
    BRANCHES,
    MODELS,
    function_node,
)
from test_qsdpa_selected_host_branches_cpu import (
    host as host,  # noqa: PLC0414 - re-export pytest fixture
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_real_registered_atfork_callback_with_parent_lock_held(record_property):
    script = textwrap.dedent(r"""
        import json, os, select, signal, sys, time, types
        from pathlib import Path
        # The subprocess starts without MLX. The production module's only
        # MLX import resolves to a deliberately inert stand-in.
        assert 'mlx.core' not in sys.modules
        mlx=types.ModuleType('mlx'); mlx.__path__=[]
        mx=types.ModuleType('mlx.core'); mlx.core=mx
        sys.modules['mlx']=mlx; sys.modules['mlx.core']=mx
        qvm=types.ModuleType('_real_fork_counter_probe')
        path=Path(sys.argv[1])
        exec(compile(path.read_text(), str(path), 'exec'), qvm.__dict__)
        qvm.note_selected_host_branch('composed')
        parent=qvm.selected_host_branch_snapshot()
        read_fd,write_fd=os.pipe()
        child=None; reaped=False
        lock=qvm._SELECTED_HOST_BRANCH_LOCK
        lock.acquire()
        try:
            child=os.fork()
            if child==0:
                os.close(read_fd)
                code=0
                try:
                    initial=qvm.selected_host_branch_snapshot()
                    assert initial['process_id']==os.getpid()!=parent['process_id']
                    assert initial['epoch']==parent['epoch']+1
                    assert all(v==0 for v in initial['counts'].values())
                    try:
                        qvm.selected_host_branch_delta(parent)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('foreign parent baseline admitted')
                    qvm.note_selected_host_branch('direct_decode')
                    delta=qvm.selected_host_branch_delta(initial)
                    assert delta['counts']['direct_decode']==1
                    qvm.reset_selected_host_branches()
                    final=qvm.selected_host_branch_snapshot()
                    assert all(v==0 for v in final['counts'].values())
                    result={'child_pid':os.getpid(),'initial':initial,'delta':delta,'final':final}
                except BaseException as error:
                    result={'error_type':type(error).__name__,'error':str(error)}
                    code=2
                try:
                    os.write(write_fd,json.dumps(result).encode())
                finally:
                    os.close(write_fd)
                    os._exit(code)
            os.close(write_fd); write_fd=None
            # Keep the parent's old mutex held until the child has used the
            # replacement lock. A missing/broken callback would hang here.
            ready,_,_=select.select([read_fd],[],[],3.0)
            assert ready,'child counter callback/snapshot timed out'
            data=json.loads(os.read(read_fd,65536))
            deadline=time.monotonic()+2.0
            while time.monotonic()<deadline:
                got,status=os.waitpid(child,os.WNOHANG)
                if got:
                    reaped=True
                    break
                time.sleep(0.01)
            assert reaped,'child did not exit after counter receipt'
            assert os.waitstatus_to_exitcode(status)==0,data
            assert 'error_type' not in data,data
        finally:
            if child and not reaped:
                try: os.kill(child,signal.SIGKILL)
                except ProcessLookupError: pass
                os.waitpid(child,0)
            if write_fd is not None: os.close(write_fd)
            os.close(read_fd)
            lock.release()
        # Child reset is copy-on-write and cannot alter the parent's counts.
        assert qvm.selected_host_branch_snapshot()==parent
        qvm.note_selected_host_branch('direct_verify')
        parent_delta=qvm.selected_host_branch_delta(parent)
        assert parent_delta['counts']['direct_verify']==1
        assert parent_delta['counts']['direct_decode']==0
        print(json.dumps({'parent_pid':os.getpid(),'parent':parent,'child':data,
                          'parent_after_delta':parent_delta,'child_reaped':reaped}))
    """)
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(MODELS / "qsdpa_verify_metal.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        # The outer deadline kills the whole group, including a fork child,
        # even if the probe's own explicit kill/wait cleanup is broken.
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=3)
        pytest.fail(f"fork probe process group timed out: {stdout}\n{stderr}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
    assert process.returncode == 0, stdout + stderr
    result = json.loads(stdout)
    assert result["child_reaped"] is True
    assert result["child"]["child_pid"] != result["parent_pid"]
    record_property("real_fork_receipt", json.dumps(result, sort_keys=True))


class StripObservation(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "qvm"
            and call.func.attr == "note_selected_host_branch"
        ):
            return None
        return node


def compile_dispatch(env, *, observed):
    node = function_node()
    if not observed:
        node = StripObservation().visit(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    scope = dict(env)
    exec(  # noqa: S102 - trusted source against inert NumPy-backed operations
        compile(ast.fix_missing_locations(module), str(MODELS / "base.py"), "exec"),
        scope,
    )
    return scope["quantized_scaled_dot_product_attention"]


def dense_attention(q, k, v, scale, mask):
    repeat = q.shape[1] // k.shape[1]
    k = np.repeat(k, repeat, axis=1)
    v = np.repeat(v, repeat, axis=1)
    scores = (q @ k.swapaxes(-2, -1)) * scale
    if mask is not None:
        scores = (
            np.where(mask, scores, -np.inf) if mask.dtype == np.bool_ else scores + mask
        )
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (scores / scores.sum(axis=-1, keepdims=True)) @ v


def numpy_case(kind):
    rng = np.random.default_rng(20261027)
    query = rng.normal(size=(2, 4, 12, 4))
    keys = rng.normal(size=(2, 2, 17, 4))
    values = rng.normal(size=(2, 2, 17, 3))
    causal = np.arange(5, 17)[:, None] >= np.arange(17)[None, :]
    if kind == "causal":
        return query, keys, values, "causal", causal
    if kind == "none":
        return query, keys, values, None, None
    if kind == "bool2d":
        return query, keys, values, causal.copy(), causal
    mask = np.broadcast_to(causal, (2, 1, 12, 17)).copy()
    mask[1, :, :, 0] = False
    if kind == "float4d":
        additive = np.where(mask, rng.uniform(-0.5, 0, size=mask.shape), -np.inf)
        return query, keys, values, additive, additive
    if kind == "broadcast":
        mask = np.ones((2, 1, 1, 17), dtype=bool)
        mask[1, :, :, 0] = False
    return query, keys, values, mask, mask


def run_numpy_dispatch(host, kind, *, observed, fail_at=None, error_type=RuntimeError):
    query, keys, values, mask, full_mask = numpy_case(kind)
    host.qvm.reset_selected_host_branches()
    host.qvm.STATS.clear()
    host.qvm.STATS.update(
        verify_kernel_calls=0, verify_kernel_left_padded=0, composed_calls=0
    )
    host.qvm.STATS_BY_ROWS.clear()
    records, evaluated = [], []
    host.mx.arange = np.arange
    host.mx.concatenate = np.concatenate
    host.mx.eval = lambda output: evaluated.append(output.copy())

    def composed(q, qk, qv, scale, tile_mask, *_format):
        records.append(
            {"query": q.copy(), "mask": None if tile_mask is None else tile_mask.copy()}
        )
        if len(records) == fail_at:
            raise error_type("injected tile construction interruption")
        return dense_attention(q, qk[0], qv[0], scale, tile_mask)

    env = dict(
        host.env, _scores_tile_rows=lambda *args: [5, 3, 4], _composed_qsdpa=composed
    )
    call = compile_dispatch(env, observed=observed)
    baseline = host.qvm.selected_host_branch_snapshot()
    output = error = None
    try:
        output = call(
            query, (keys, None, None), (values, None, None), scale=0.25, mask=mask
        )
    except (RuntimeError, CancelledError) as caught:
        error = caught
    return {
        "output": output,
        "error": error,
        "records": records,
        "evaluated": evaluated,
        "legacy": copy.deepcopy(host.qvm.STATS),
        "by_rows": copy.deepcopy(host.qvm.STATS_BY_ROWS),
        "delta": host.qvm.selected_host_branch_delta(baseline),
        "query": query,
        "keys": keys,
        "values": values,
        "full_mask": full_mask,
    }


def assert_same_records(actual, baseline):
    assert len(actual["records"]) == len(baseline["records"])
    for got, want in zip(actual["records"], baseline["records"]):
        np.testing.assert_array_equal(got["query"], want["query"])
        if want["mask"] is None:
            assert got["mask"] is None
        else:
            np.testing.assert_array_equal(got["mask"], want["mask"])
    assert len(actual["evaluated"]) == len(baseline["evaluated"])
    for got, want in zip(actual["evaluated"], baseline["evaluated"]):
        np.testing.assert_array_equal(got, want)
    assert actual["legacy"] == baseline["legacy"]
    assert actual["by_rows"] == baseline["by_rows"]


@pytest.mark.parametrize(
    "kind", ["causal", "bool2d", "bool4d", "float4d", "broadcast", "none"]
)
def test_masked_tiled_source_dispatch_matches_uninstrumented_baseline(host, kind):
    baseline = run_numpy_dispatch(host, kind, observed=False)
    actual = run_numpy_dispatch(host, kind, observed=True)
    assert baseline["error"] is actual["error"] is None
    assert_same_records(actual, baseline)
    np.testing.assert_array_equal(actual["output"], baseline["output"])
    expected = dense_attention(
        actual["query"], actual["keys"], actual["values"], 0.25, actual["full_mask"]
    )
    np.testing.assert_allclose(actual["output"], expected, atol=1e-13, rtol=1e-13)
    start = 0
    for row_count, record in zip((5, 3, 4), actual["records"]):
        stop = start + row_count
        np.testing.assert_array_equal(
            record["query"], actual["query"][:, :, start:stop]
        )
        full = actual["full_mask"]
        if full is None:
            assert record["mask"] is None
        else:
            wanted = full[..., start:stop, :] if full.shape[-2] == 12 else full
            np.testing.assert_array_equal(record["mask"], wanted)
        start = stop
    assert actual["delta"]["counts"] == {
        branch: int(branch == "composed_tiled") for branch in BRANCHES
    }
    assert all(value == 0 for value in baseline["delta"]["counts"].values())
    assert actual["legacy"]["composed_tiled_calls"] == 1
    assert actual["legacy"]["composed_tile_calls"] == 3
    assert actual["by_rows"] == {
        "composed_tiled_L12": 1,
        "composed_tile_L5": 1,
        "composed_tile_L3": 1,
        "composed_tile_L4": 1,
    }


@pytest.mark.parametrize("error_type", [RuntimeError, CancelledError])
def test_tiled_failure_or_cancel_keeps_selection_and_legacy_frontier(host, error_type):
    baseline = run_numpy_dispatch(
        host, "causal", observed=False, fail_at=2, error_type=error_type
    )
    actual = run_numpy_dispatch(
        host, "causal", observed=True, fail_at=2, error_type=error_type
    )
    assert isinstance(actual["error"], error_type) and isinstance(
        baseline["error"], error_type
    )
    assert actual["output"] is baseline["output"] is None
    assert_same_records(actual, baseline)
    assert len(actual["records"]) == 2 and len(actual["evaluated"]) == 1
    assert actual["delta"]["counts"] == {
        branch: int(branch == "composed_tiled") for branch in BRANCHES
    }
    assert actual["legacy"]["composed_tile_calls"] == 1
