"""Owned-window compact-IQ1 versus affine expert diagnostic.

Actual layer2 operands come from the ordinary functional probe. Repeated
prefill rows are explicitly a synthetic workload, not whole-model throughput.
No weight representation is expanded or changed by this harness.
"""
from contextlib import contextmanager
from pathlib import Path
import statistics
import time

import numpy as np

from mlx2.desktop_campaign_worker import atomic


def block_oracle(mx):
    """Device unpack only, including half subnormals; no matmul reductions."""
    from mlx2.runtime.gguf.blocks import decode_blocks
    from mlx2.runtime.gguf.packed_metal import HEADER

    raw = np.random.default_rng(410125).integers(0, 256, (16, 50), np.uint8)
    words = np.array([0, 0x8000, 1, 0x8001, 0x3ff, 0x83ff, 0x400, 0x8400,
                      0x1400, 0x9400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x3555, 0xb555], dtype="<u2")
    raw[:, :2] = words.view(np.uint8).reshape(-1, 2)
    expected = decode_blocks(raw.tobytes(), 19).reshape(16, 256)
    kernel = mx.fast.metal_kernel(name="mlx2_v41_compact_iq1_unpack_oracle_v1",
        input_names=["packed"], output_names=["values"], header=HEADER,
        source="uint i=thread_position_in_grid.x; if(i<4096) values[i]=gguf_value(packed+(i/256)*50,i%256,19);")
    value = kernel(inputs=[mx.array(raw.reshape(-1))], grid=(4096, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(16, 256)], output_dtypes=[mx.float32])[0]
    mx.eval(value)
    actual = np.array(value)
    check = delta(expected, actual)
    check.update(blocks=16, values=4096, scope="synthetic packed IQ1_S device unpack; zero, signedzero, subnormal, normal and maxfinite half scales")
    if not check["finite"] or not np.array_equal(expected, actual):
        raise RuntimeError("packed IQ1_S GPU unpack differs from independent CPU oracle: "+str(check))
    return check


@contextmanager
def capture(weights, mx):
    frames = []
    original = weights.moe

    def observe(x, ids, scores, prefix, **kwargs):
        if prefix == "blk.2":
            mx.eval(x, ids, scores)
            frames.append((x, ids, scores))
        return original(x, ids, scores, prefix, **kwargs)

    weights.moe = observe
    try:
        yield frames
    finally:
        weights.moe = original


def delta(a, b):
    a, b = np.asarray(a).astype(np.float64), np.asarray(b).astype(np.float64)
    d = b-a
    rms = float(np.sqrt(np.mean(a*a)))
    return dict(finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                max_abs=float(np.max(np.abs(d))), rmse=float(np.sqrt(np.mean(d*d))),
                relative_rms=float(np.sqrt(np.mean(d*d)))/max(rms, 1e-30),
                bitwise_equal=np.asarray(a, np.float32).tobytes() == np.asarray(b, np.float32).tobytes())


def run(adapter, affine_path, frames, mx, out, phase, guard):
    from mlx2.adapters.deepseek_v41_mlx_artifact import inspect_artifact
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
    from mlx2.runtime.gguf.segmented_prefill import expert_prefill_context

    out = Path(out)
    if not frames or frames[0][0].shape[0] != 16 or frames[-1][0].shape[0] != 1:
        raise ValueError("compact diagnostic requires actual16-row prompt and one-row decode captures")
    affine = inspect_artifact(affine_path)
    if affine["manifest"]["derived_identity_sha256"] != adapter.weights.artifact["manifest"]["source_affine"]["derived_identity_sha256"]:
        raise ValueError("compact/affine source binding differs")
    reference = NativeV41Weights(affine, expert_cache_bytes=8 << 30)
    x, ids, scores = frames[0]
    cases = [("actual_decode1", frames[-1]), ("actual_prompt16", frames[0]),
             ("synthetic_repeat_prompt512", (mx.tile(x, (32, 1)), mx.tile(ids, (32, 1)), mx.tile(scores, (32, 1))))]
    report = dict(schema="mlx2.v41.iq1-affine-expert-ab.v1", passed=False,
                  compact_checkpoint=adapter.identity["fingerprint"], affine_checkpoint=affine["identity"]["fingerprint"],
                  layer=2, repetitions=3, scope="hot layer2 expert-only wall timings; actual ordinary operands plus explicit repeated-row synthetic prefill; no controlled performance or quality qualification",
                  cases=[])

    def evaluate(provider, operands):
        count = operands[0].shape[0]
        with expert_prefill_context(provider.expert_prefill_policy, tokens=count):
            result = provider.moe(*operands, "blk.2")
            mx.eval(result)
        return result

    try:
        for name, operands in cases:
            guard()
            providers = {"affine_f32_qmm": reference, "compact_iq1": adapter.weights}
            warm = {}; values = {}
            for key, provider in providers.items():
                before = provider.expert_stream.counters()
                values[key] = evaluate(provider, operands)
                after = provider.expert_stream.counters()
                warm[key] = {k: after[k]-before.get(k, 0) for k in after}
            numerical = delta(np.array(values["affine_f32_qmm"]), np.array(values["compact_iq1"]))
            if not numerical["finite"]:
                raise RuntimeError("compact expert A/B produced nonfinite output")
            samples = {k: [] for k in providers}; counters = {k: [] for k in providers}
            for repetition in range(3):
                for key in list(providers) if repetition % 2 == 0 else list(providers)[::-1]:
                    guard(); provider = providers[key]
                    before = provider.expert_stream.counters(); started = time.perf_counter()
                    value = evaluate(provider, operands)
                    samples[key].append(time.perf_counter()-started)
                    after = provider.expert_stream.counters()
                    change = {k: after[k]-before.get(k, 0) for k in after}; counters[key].append(change)
                    if change.get("v41_expert_bytes_read", 0):
                        raise RuntimeError("timed expert A/B was not resident")
                    del value
            row = dict(name=name, rows=operands[0].shape[0], numerical=numerical, samples_seconds=samples,
                       median_seconds={k: statistics.median(v) for k, v in samples.items()},
                       timed_counters=counters, warmup_counters=warm)
            report["cases"].append(row); atomic(out/"compact-iq1-ab.json", report)
            phase("compact_iq1_expert_case", case_name=name, medians=row["median_seconds"], numerical=numerical)
        report["passed"] = True
        return report
    finally:
        reference.close()
        atomic(out/"compact-iq1-ab.json", report)
