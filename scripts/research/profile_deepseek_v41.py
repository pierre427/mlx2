"""Temporary evaluated-boundary diagnostic; timings are interventions, not throughput.

Reuses the local DS4 boundary profiler. All hooks are process-local and restored.
"""

from contextlib import contextmanager
import threading
import time

import numpy as np

import probe_deepseek_v4_layer_cost as base
from probe_deepseek_v4_ordinary import arrays_numpy


def arrays_equal(left, right):
    """Logical array bytes, including signed zeros and exact dtype/shape."""
    if isinstance(left, (tuple,list)):
        return isinstance(right,(tuple,list)) and len(left)==len(right) and all(arrays_equal(a,b) for a,b in zip(left,right))
    if left is None or right is None:
        return left is right
    a,b = np.asarray(left),np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def patch_dense_pages(pages, profiler, patch):
    """Instance callbacks only; all timing stays on the evaluation owner."""
    if pages is None:
        return
    for owner, name, category in (
        (pages, "acquire", "dense_page_lookup"),
        (pages.pool, "call", "dense_page_read"),
        (pages, "materialize", "dense_page_materialize"),
        (pages, "evict", "dense_page_evict"),
        (pages, "synchronize", "dense_page_fence"),
    ):
        original = getattr(owner, name)
        def measured(*args, _original=original, _category=category, **kwargs):
            return profiler.call(_category, "immutable_tensor_page", _original, args, kwargs)
        patch(owner, name, measured)


@contextmanager
def instrument(model, profiler):
    from mlx2.adapters import deepseek_v41_expert_stream as stream
    from mlx2.runtime.gguf import v41_metal, v41_prefill, v41_row_prefill, v41_row_exact
    from mlx2.runtime.gguf.packed_metal import PackedMatrix
    from mlx2.runtime.models import deepseek_v41 as module
    from mlx2.runtime.models.deepseek_v41_engram import EngramRows

    patches, io_rows = [], []
    io_lock = threading.Lock()
    depth = [0]

    def patch(owner, name, value):
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def method(cls, name, category, label):
        original = getattr(cls, name)

        def wrapped(obj, *args, **kwargs):
            return profiler.call(category, label(obj), lambda *a, **k: original(obj, *a, **k), args, kwargs)

        patch(cls, name, wrapped)

    original_leaves = base.leaves

    def leaves(value, mx):
        if isinstance(value, PackedMatrix):
            return [value.data]
        return original_leaves(value, mx)

    try:
        patch(base, "leaves", leaves)
        # Callback instances were captured when the pager was built.
        patch_dense_pages(getattr(model.weights, "dense_pages", None), profiler, patch)
        original_block = module.Block.__call__

        def block(obj, *args, **kwargs):
            return profiler.call("block_other", obj.prefix, lambda *a, **k: original_block(obj, *a, **k),
                                 args, kwargs, layer=obj.layer)

        patch(module.Block, "__call__", block)
        method(module.Attention, "__call__", "attention_other", lambda obj: obj.prefix)
        method(module.Block, "mixes", "hc", lambda obj: obj.prefix)
        method(module.Block, "shared_expert", "shared_expert", lambda obj: obj.prefix)
        method(module.Block, "finish", "hc", lambda obj: obj.prefix+"ffn_expand")
        method(type(model), "expand", "hc", lambda obj: "attn_expand")
        method(type(model.weights), "gate_up", "shared_expert", lambda obj: "gate_up")
        method(type(model.weights), "grouped", "dense", lambda obj: "grouped")
        method(type(model.weights), "embedding", "embedding", lambda obj: "embedding")
        method(type(model.weights), "fused_project", "dense", lambda obj: "q8_round_epilogue")
        method(type(model.weights), "paired_project", "dense", lambda obj: "q8_pair")
        method(EngramRows, "read", "engram_read", lambda obj: "rows")
        linear = type(model.weights).linear

        def project(obj, name):
            fn = linear(obj, name)
            return lambda *a, **k: profiler.call("shared_expert" if "_shexp." in name else "dense", name, fn, a, k)

        patch(type(model.weights), "linear", project)
        original_prepare = stream.PackedExpertStream.prepare

        def prepare(obj, prefix, ids):
            def execute():
                # Isolate the required routing readback before the original
                # implementation's repeated (now ready) asarray call.
                profiler.call("routing_readback", prefix, np.asarray, (ids,), {})
                return original_prepare(obj, prefix, ids)

            # Deliberately no input barrier here: it would consume readback.
            return profiler.call("expert_gather_other", prefix, execute, (), {})

        patch(stream.PackedExpertStream, "prepare", prepare)
        original_resolve = stream.PackedExpertStream.resolve

        def resolve(obj, ticket, **kwargs):
            def execute():
                depth[0] += 1
                try:
                    return original_resolve(obj, ticket, **kwargs)
                finally:
                    depth[0] -= 1
            return profiler.call("expert_gather_other", ticket.prefix, execute, (), {})

        patch(stream.PackedExpertStream, "resolve", resolve)
        concatenate = profiler.mx.concatenate

        def concat(*args, **kwargs):
            if not depth[0]:
                return concatenate(*args, **kwargs)
            value = profiler.call("selected_stack_copy", "concatenate", concatenate, args, kwargs)
            profiler.copy_bytes += value.nbytes
            return value

        patch(profiler.mx, "concatenate", concat)
        read_rows = stream.read_rows

        def read(fd, entries):
            start = time.perf_counter()
            result = read_rows(fd, entries)
            with io_lock:
                io_rows.append({"seconds": time.perf_counter()-start,
                                "bytes": sum(len(p) for _, planes in result for p in planes)})
            return result

        patch(stream, "read_rows", read)
        for owner in (v41_metal, v41_prefill, v41_row_prefill, v41_row_exact):
            original = owner.moe

            def compute(*args, _fn=original, **kwargs):
                return profiler.call("expert_compute", _fn.__module__, _fn, args, kwargs)

            patch(owner, "moe", compute)
        original_rows = v41_metal.moe_rows

        def rows(*args, **kwargs):
            return profiler.call("expert_compute", "row_buffers", original_rows, args, kwargs)

        patch(v41_metal, "moe_rows", rows)
        yield io_rows
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def profile_case(model, cache, inputs, *, prefill, mx):
    snapshots = [s.snapshot() for s in cache]
    mx.eval([s.arrays for s in snapshots])
    forward = model.prefill_forward if prefill else model

    def restore():
        for state, saved in zip(cache, snapshots):
            state.restore(saved)
        mx.eval([s.state for s in cache])

    def run():
        value = forward(inputs, cache)
        mx.eval(value, [s.state for s in cache])
        return value

    try:
        start = time.perf_counter()
        selected = model.weights.expert_decode_policy
        selected_buffers = model.weights.expert_buffer_policy
        model.weights.expert_decode_policy = "reference"
        model.weights.expert_buffer_policy = "stack"
        try:
            expected = run()
        finally:
            model.weights.expert_decode_policy = selected
            model.weights.expert_buffer_policy = selected_buffers
        reference_seconds = time.perf_counter()-start
        expected_logits = np.array(expected)
        expected_states = [arrays_numpy(s.state) for s in cache]
        expected_meta = [s.meta_state for s in cache]
        restore()
        # Compile/warm the selected shape before the instrumented observation.
        warmup = run()
        warmup_equal = bool(arrays_equal(np.array(warmup), expected_logits))
        restore()
        profiler = base.BoundaryProfiler(mx)
        profiler.copy_bytes = 0
        before = model.weights.expert_stream.counters()
        buffers_before = dict(model.weights.expert_buffer_counters)
        start = time.perf_counter()
        with instrument(model, profiler) as reads:
            actual = run()
        wall = time.perf_counter()-start
        counters = model.weights.expert_stream.counters()
        result = {"rows": inputs.shape[1], "frontier": cache[0].offset-inputs.shape[1],
                  "preceding_reference_seconds": reference_seconds, "reference_decode_policy": "reference",
                  "profile_decode_policy": selected, "profiled_seconds": wall,
                  "reference_buffer_policy": "stack", "profile_buffer_policy": selected_buffers,
                  "selected_warmup_bitwise_equal": warmup_equal,
                  "selected_stack_copy_bytes": profiler.copy_bytes,
                  "expert_counter_delta": {k: counters[k]-before[k] for k in before},
                  "expert_buffer_operation_delta": {k: v-buffers_before.get(k,0) for k,v in model.weights.expert_buffer_counters.items()},
                  "read_tasks": reads, "read_task_time_is_overlapping": True,
                  "logits_bitwise_equal": bool(arrays_equal(np.array(actual), expected_logits)),
                  "finite_logits": bool(np.isfinite(expected_logits).all() and np.isfinite(np.array(actual)).all()),
                  "state_arrays_bitwise_equal": arrays_equal(expected_states, [arrays_numpy(s.state) for s in cache]),
                  "state_metadata_equal": expected_meta == [s.meta_state for s in cache],
                  **profiler.summary()}
        result["unattributed_wall_seconds"] = wall-result["exclusive_total_seconds"]
        restore()
        start = time.perf_counter()
        control = run()
        result["following_uninstrumented_seconds"] = time.perf_counter()-start
        result["following_control_bitwise_equal"] = bool(arrays_equal(np.array(control), expected_logits))
        result["passed"] = all(result[k] for k in ("logits_bitwise_equal", "state_arrays_bitwise_equal",
                                                   "state_metadata_equal", "following_control_bitwise_equal",
                                                   "selected_warmup_bitwise_equal", "finite_logits"))
        return result
    finally:
        restore()


def commit_probe(model, ids, mx):
    from mlx2.adapters.deepseek_v4_external import DeepSeekV4VerifyRows
    from mlx2.adapters.deepseek_v41_external import DeepSeekV41VerifyRows

    cache = model.make_cache()
    initial = model.prefill_forward(mx.array([ids], mx.uint32), cache)
    mx.eval(initial, [s.state for s in cache])
    base_states = [s.snapshot() for s in cache]
    mx.eval([s.arrays for s in base_states])
    inputs = mx.array([ids[-6:]], mx.uint32)
    cases = []

    def restore():
        for state, snapshot in zip(cache,base_states):
            state.restore(snapshot)

    def call_count():
        return sum(model.weights.scheduled_expert_calls.values())//len(model.layers)

    try:
        for accepted in range(7):
            restore()
            before = call_count()
            tx = DeepSeekV4VerifyRows(model,[cache],(len(cache)-1,)).begin([6])
            expected, expected_hidden = tx.forward(inputs)
            mx.eval(expected, expected_hidden)
            tx.commit([accepted])
            mx.eval([s.state for s in cache])
            reference_calls = call_count()-before
            expected_states = [arrays_numpy(s.state) for s in cache]
            expected_meta = [s.meta_state for s in cache]
            restore()
            before = call_count()
            tx = DeepSeekV41VerifyRows(model,[cache],(len(cache)-1,)).begin([6])
            actual, hidden = tx.forward(inputs)
            retained = tx.retained_nbytes
            mx.eval(actual, hidden)
            tx.commit([accepted])
            mx.eval([s.state for s in cache])
            row = {"accepted_inputs": accepted, "reference_target_forwards": reference_calls,
                   "candidate_target_forwards": call_count()-before, "retained_bytes_before_commit": retained,
                   "logits_bitwise_equal": bool(arrays_equal(np.array(expected),np.array(actual))),
                   "finite_logits": bool(np.isfinite(np.array(expected)).all() and np.isfinite(np.array(actual)).all()),
                   "features_bitwise_equal": bool(arrays_equal(np.array(expected_hidden),np.array(hidden))),
                   "state_arrays_bitwise_equal": arrays_equal(expected_states,[arrays_numpy(s.state) for s in cache]),
                   "state_metadata_equal": expected_meta == [s.meta_state for s in cache]}
            row["passed"] = all(row[k] for k in ("logits_bitwise_equal","features_bitwise_equal",
                                                "state_arrays_bitwise_equal","state_metadata_equal","finite_logits"))
            cases.append(row)
        return {"prefix_tokens": len(ids), "verification_rows":6, "cases": cases,
                "passed": all(c["passed"] for c in cases)}
    finally:
        restore()


def profile_model(adapter, cache, token, mx):
    model = adapter.model
    decode = profile_case(model, cache, mx.array([[token]], mx.uint32), prefill=False, mx=mx)
    request = {"messages": [{"role": "user", "content":
        "Reference notes: " + "prefix caching saves exact model state. "*160 + " Reply only Hello."}],
        "enable_thinking": False}
    ids = list(adapter.prompt_tokens(request))[:512]
    prefill = profile_case(model, model.make_cache(), mx.array([ids], mx.uint32), prefill=True, mx=mx)
    commits = commit_probe(model, ids, mx)
    return {"scope": "single evaluated-boundary intervention per shape, warmed by preceding reference; no throughput or qualification claim",
            "policy": adapter.kernel_policy, "decode": decode, "prefill": prefill, "commits":commits}
