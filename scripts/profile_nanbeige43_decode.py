"""Bounded Nanbeige diagnostic; call run_profile inside an owned GPU unit.

No CLI, lease acquisition or GPU execution occurs on import. The caller owns
ordinary-reference smoke, thermal/process admission, paired locks and CPG.
Component fences alter execution and cannot explain fractions of plain latency.
"""

from __future__ import annotations

import copy
import hashlib
import statistics
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def _median(rows, name):
    return statistics.median(row[name] for row in rows)


def _arrays(value):
    import mlx.core as mx

    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _arrays(item)


@contextmanager
def _components(model):
    """Restore every class/function hook, including exceptional forwards."""
    import mlx.core as mx

    from mlx2.adapters import nanbeige43_fused
    from mlx2.runtime.models import standard_decoder

    categories = {}
    for name, module in model.named_modules():
        if name.endswith("lm_head"):
            category = "lm_head"
        elif name.endswith(
            ("input_layernorm", "post_attention_layernorm", "model.norm")
        ):
            category = "norms"
        elif name.endswith(("q_proj", "k_proj", "v_proj", "qkv")):
            category = "qkv"
        elif name.endswith("o_proj"):
            category = "attention_output_projection"
        elif name.endswith(("gate_proj", "up_proj", "gate_up")):
            category = "mlp_gate_up"
        elif name.endswith("down_proj"):
            category = "mlp_down"
        elif name.endswith("self_attn"):
            category = "attention_preparation_other"
        elif name.endswith("mlp"):
            category = "mlp_activation_other"
        elif name.endswith("embed_tokens"):
            category = "embedding"
        else:
            continue
        categories[id(module)] = category
    modules = [
        module for _, module in model.named_modules() if id(module) in categories
    ]
    # Collect original functions before installing any wrapper. Guard reentry
    # where a derived projection calls a patched base through super().
    classes = {type(module) for module in modules}
    originals = {cls: ("__call__" in cls.__dict__, cls.__call__) for cls in classes}
    totals, calls, fences, stack, active = (
        defaultdict(float),
        defaultdict(int),
        defaultdict(int),
        [],
        set(),
    )

    def timed(name, function, positional, keywords):
        incoming = list(_arrays((positional, keywords)))
        if incoming:
            mx.eval(incoming)
        stack.append(0.0)
        started = time.perf_counter()
        try:
            output = function(*positional, **keywords)
            mx.eval(list(_arrays(output)))
        except BaseException:
            stack.pop()
            raise
        elapsed = time.perf_counter() - started
        child = stack.pop()
        totals[name] += elapsed - child
        calls[name] += 1
        fences[name] += 1 + bool(incoming)
        if stack:
            stack[-1] += elapsed
        return output

    def wrapper(original):
        def call(self, *args, **kwargs):
            identity = id(self)
            category = categories.get(identity)
            if category is None or identity in active:
                return original(self, *args, **kwargs)
            active.add(identity)
            try:
                return timed(category, original, (self, *args), kwargs)
            finally:
                active.remove(identity)

        return call

    saved_functions = []
    try:
        for cls, (_, original) in originals.items():
            cls.__call__ = wrapper(original)
        for owner, attribute, category in (
            (standard_decoder, "scaled_dot_product_attention", "attention"),
            (nanbeige43_fused, "scaled_dot_product_attention", "attention"),
            (nanbeige43_fused, "residual_rms", "residual_norm"),
        ):
            original = getattr(owner, attribute)
            saved_functions.append((owner, attribute, original))

            def call(*args, _name=category, _fn=original, **kwargs):
                return timed(_name, _fn, args, kwargs)

            setattr(owner, attribute, call)
        yield totals, calls, fences
    finally:
        for owner, attribute, original in reversed(saved_functions):
            setattr(owner, attribute, original)
        for cls, (owned, original) in originals.items():
            if owned:
                cls.__call__ = original
            else:
                delattr(cls, "__call__")


def run_profile(adapter, context=256, reps=3):
    """Profile one already-loaded Nanbeige model in the caller's GPU window."""
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.nanbeige43 import matches_config

    if type(context) is not int or not 32 <= context <= 2048:
        raise ValueError("context must be 32..2048")
    if type(reps) is not int or not 1 <= reps <= 5:
        raise ValueError("reps must be 1..5")
    if mx.default_device() != mx.gpu or not matches_config(adapter.config):
        raise ValueError("profile requires the owned GPU and exact Nanbeige topology")
    model = adapter.model
    prompt = (
        "Profile single-request decode without conflating kernel and host timing. "
        * 512
    )
    ids = adapter.tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) < context + 1:
        raise ValueError("profile text did not fill requested context")
    ids = ids[: context + 1]
    cache = model.make_cache()
    prefill_step = 128
    for start in range(0, context, prefill_step):
        mx.eval(
            model(mx.array([ids[start : min(context, start + prefill_step)]]), cache)
        )
    mx.eval([item.state for item in cache])
    anchor = mx.array([[ids[-1]]], mx.uint32)

    def decode():
        local = copy.deepcopy(cache)
        mx.eval([item.state for item in local])
        token = anchor
        pending = None
        host = []
        build, dispatch, wait = 0.0, 0.0, 0.0
        started = time.perf_counter()
        for _ in range(16):
            before = time.perf_counter()
            logits = model(token, local)[:, -1, :].astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            current = mx.argmax(logprobs, axis=-1)
            token = current.reshape(1, 1)
            build += time.perf_counter() - before
            before = time.perf_counter()
            mx.async_eval(current, logprobs)
            dispatch += time.perf_counter() - before
            if pending is not None:
                before = time.perf_counter()
                mx.eval(pending)
                host.append(int(pending.item()))
                wait += time.perf_counter() - before
            pending = current
        before = time.perf_counter()
        mx.eval(pending, [item.state for item in local])
        host.append(int(pending.item()))
        wait += time.perf_counter() - before
        elapsed = time.perf_counter() - started
        return {
            "wall_seconds": elapsed,
            "graph_build_seconds": build,
            "async_dispatch_seconds": dispatch,
            "host_wait_seconds": wait,
            "output_tokens": 16,
            "tokens_per_second": 16 / elapsed,
            "token_ids": host,
            "cache_offsets": [int(item.offset) for item in local],
        }

    decode()  # Warm graph/device paths outside retained plain samples.
    plain = [decode() for _ in range(reps)]
    single_cache = copy.deepcopy(cache)
    reference = model(anchor, single_cache)
    mx.eval(reference)
    z = mx.zeros((1,))
    mx.eval(z)
    sync = []
    for _ in range(64):
        before = time.perf_counter()
        mx.eval(z + 1)
        sync.append(time.perf_counter() - before)
    sync_floor = statistics.median(sync)
    rows = []
    for _ in range(reps):
        local = copy.deepcopy(cache)
        mx.eval([item.state for item in local])
        with _components(model) as (totals, calls, fences):
            before = time.perf_counter()
            output = model(anchor, local)
            mx.eval(output)
            elapsed = time.perf_counter() - before
        delta = np.asarray(output.astype(mx.float32)) - np.asarray(
            reference.astype(mx.float32)
        )
        rows.append(
            {
                "wall_seconds": elapsed,
                "exclusive_seconds": dict(totals),
                "calls": dict(calls),
                "fences": dict(fences),
                "unwrapped_seconds": elapsed - sum(totals.values()),
                "max_abs_logit_error_vs_uninstrumented": float(np.max(np.abs(delta))),
                "logits_bitwise_equal": bool(
                    np.array_equal(delta, np.zeros_like(delta))
                ),
            }
        )
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        root / "src/mlx2/adapters/nanbeige43.py",
        root / "src/mlx2/adapters/nanbeige43_fused.py",
        root / "src/mlx2/adapters/nanbeige43_kernels.py",
        root / "src/mlx2/runtime/models/standard_decoder.py",
    ]
    identity = adapter.identity
    artifact = Path(identity["path"])
    categories = sorted({key for row in rows for key in row["exclusive_seconds"]})
    return {
        "schema": "mlx2.nanbeige43-decode-profile.v1",
        "checkpoint": artifact.name + "@" + identity["fingerprint"],
        "model_path": str(artifact),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "artifact_metadata_sha256": {
            name: hashlib.sha256((artifact / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")
            if (artifact / name).is_file()
        },
        "context_tokens": context,
        "shape": {"batch": 1, "decode_steps": 16},
        "qualification": False,
        "plain": {
            "samples": plain,
            "median_tokens_per_second": _median(plain, "tokens_per_second"),
            "median_graph_build_seconds": _median(plain, "graph_build_seconds"),
            "median_wall_seconds": _median(plain, "wall_seconds"),
        },
        "sync_floor_seconds": sync_floor,
        "components": {
            "samples": rows,
            "median_synchronized_one_token_seconds": _median(rows, "wall_seconds"),
            "exclusive_medians_seconds": {
                key: statistics.median(
                    row["exclusive_seconds"].get(key, 0.0) for row in rows
                )
                for key in categories
            },
        },
        "limitations": [
            "Direct fixed 16-step greedy mechanics; excludes HTTP, tokenizer, scheduler and stop handling.",
            "Component fences change execution and inflate absolute time; no fractions of plain decode latency are inferred.",
            "Sync floor is reported as a diagnostic, not subtracted as a calibrated correction.",
            "No hardware-counter, bandwidth-saturation or hardware-roofline claim.",
        ],
    }
