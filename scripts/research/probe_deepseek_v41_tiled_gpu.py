"""Bounded actual-expert T64 layout/routing shader gates and hot A/B.

No complete model is admitted by this component probe. The ordinary model and
APCv2 lifecycle are validated separately with probe_deepseek_v41_gpu.py.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts/research")]
from mlx2.desktop_campaign_worker import atomic, check_ownership

MODEL = Path(str(Path.home() / 'mlx-models/DeepSeek-V4.1-Flash-1017-Maya-XS-Abliterated-MLX-IQ1S'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def operands(width):
    rng = np.random.default_rng(410126)
    def bf16(value):
        bits = np.asarray(value, np.float32).view(np.uint32)
        return ((bits+np.uint32(0x7fff)+((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    x = bf16(rng.normal(0, .25, (512, width)))
    ids = (np.arange(512, dtype=np.uint32)[:, None]+np.arange(6, dtype=np.uint32)) % 6
    s = rng.uniform(.05, 1, (512, 6)).astype(np.float32)
    return x, ids, bf16(s/s.sum(axis=1, keepdims=True))


def execute(mx, spec, out, report):
    from mlx2.adapters.deepseek_v41_iq1_artifact import inspect_artifact
    from mlx2.adapters.deepseek_v41_mlx_weights import read_native_file
    from mlx2.runtime.gguf.reader import GGUFTensor
    from mlx2.runtime.gguf.packed_metal import PackedMatrix
    from mlx2.runtime.gguf import v41_iq1_tiled as tiled, v41_row_exact, v41_row_prefill
    from probe_deepseek_v41_iq1_ab import delta

    artifact = inspect_artifact(spec["model"])
    if artifact["identity"]["fingerprint"] != spec["fingerprint"]:
        raise ValueError("component artifact changed")
    args = artifact["args"]
    if spec.get("column"):
        from mlx2.runtime.gguf import v41_iq1_column as candidate_module
    else:
        candidate_module = tiled
    previous_column = None
    if spec.get("reference_column"):
        module_spec = importlib.util.spec_from_file_location(
            "mlx2.runtime.gguf._column_capacity_reference", spec["reference_column"])
        previous_column = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(previous_column)
    raw_planes = [[], [], []]; host_tiles = [[] for _ in range(9)]
    read_bytes = 0
    for expert in range(6):
        relative = f"experts/layer-000/expert-{expert:03d}.safetensors"
        record = artifact["files"][relative]
        raw = read_native_file(artifact["root"], relative, record)
        read_bytes += record["size"]
        for i, kind in enumerate(("gate", "up", "down")):
            row = artifact["tensors"][f"blk.0.ffn_{kind}_exps.weight"]
            shape = tuple(row["numpy_shape"][1:])
            data = mx.array(raw[kind+".data"])
            raw_planes[i].append(PackedMatrix(data, GGUFTensor(kind, (*row["shape"][:2], 1), 19, 0, data.nbytes)))
            prepared = tiled.pack(raw[kind+".data"], shape)
            if tiled.unpack(prepared, shape).tobytes() != raw[kind+".data"].tobytes():
                raise RuntimeError("actual expert pack round trip differs")
            for j, value in enumerate(prepared):
                host_tiles[3*i+j].append(value)
        del raw
    banks = tuple(mx.array(np.concatenate(parts)) for parts in host_tiles)
    raw_planes = tuple(tuple(p) for p in raw_planes)
    del host_tiles
    mapping = np.full(384, np.uint32(0xffffffff), np.uint32)
    mapping[:6] = np.arange(6, dtype=np.uint32)
    slot_map = mx.array(mapping)
    mx.eval(banks, slot_map, [w.data for p in raw_planes for w in p])
    x, ids, scores = operands(args.hidden_size)
    np.savez(out/"operands.npz", x=x, ids=ids, scores=scores)
    report.update(weight_bytes_read=read_bytes, timed_weight_bytes_read=0,
                  raw_payload_bytes=sum(w.data.nbytes for p in raw_planes for w in p),
                  tiled_payload_bytes=sum(p.nbytes for p in banks), cases=[])
    for count in ((1, 16, 64) if spec.get("column") else (1, 16, 64, 512)):
        for prefill in ((False,) if count == 1 or spec.get("column") else (False, True)):
            check_ownership(spec)
            operands_mx = mx.array(x[:count]), mx.array(ids[:count]), mx.array(scores[:count])
            mx.eval(operands_mx)
            def original():
                if prefill:
                    return v41_row_prefill.moe(operands_mx[0], ids[:count], operands_mx[2], *raw_planes)
                return v41_row_exact.moe(operands_mx[0], ids[:count], operands_mx[2], *raw_planes,
                    reuse=1 if count == 1 else 4, broadcast=False, group_experts=10, simd_groups=4)
            def candidate():
                return candidate_module.moe(*operands_mx, banks, slot_map, capacity=6,
                    hidden_size=args.hidden_size, intermediate_size=args.moe_intermediate_size, prefill=prefill)
            a, b = original(), candidate(); mx.eval(a, b)
            numerical = delta(np.array(a), np.array(b))
            if previous_column is not None:
                old = previous_column.moe(*operands_mx, banks, slot_map, capacity=6,
                    hidden_size=args.hidden_size, intermediate_size=args.moe_intermediate_size)
                mx.eval(old)
                previous_delta = delta(np.array(old), np.array(b))
                if not previous_delta["bitwise_equal"]:
                    raise RuntimeError(f"capacity repair changed column output: {previous_delta}")
                report.setdefault("previous_column_parity", []).append(
                    dict(rows=count, **previous_delta))
            if not numerical["finite"] or (not spec.get("column") and not numerical["bitwise_equal"]):
                raise RuntimeError(f"tiled shader parity failed rows={count}, prefill={prefill}: {numerical}")
            missing = tiled.missing_routes(operands_mx[1], slot_map, 6)
            if bool(mx.any(missing).item()):
                raise RuntimeError("resident six-expert map reported a miss")
            samples = {"row_major": [], "tiled64": []}
            functions = {"row_major": original, "tiled64": candidate}
            for repetition in range(spec["repetitions"]):
                for key in tuple(functions) if repetition % 2 == 0 else tuple(functions)[::-1]:
                    check_ownership(spec); started = time.perf_counter()
                    value = functions[key](); mx.eval(value)
                    samples[key].append(time.perf_counter()-started); del value
            case = dict(rows=count, prefill=prefill, numerical=numerical, samples_seconds=samples,
                        median_seconds={k: statistics.median(v) for k, v in samples.items()} if spec["repetitions"] else {})
            report["cases"].append(case); atomic(out/"results.json", report)
            print(json.dumps(case), flush=True)
            del a, b, missing
    # Sparse global expert labels exercise slot maps beyond the old256 limit.
    labels = np.array([0, 1, 127, 255, 256, 383], np.uint32)
    sparse = np.full(384, np.uint32(0xffffffff), np.uint32)
    sparse[labels] = np.arange(6, dtype=np.uint32)
    sparse_map = mx.array(sparse)
    local = mx.array(ids[:16]); global_ids = mx.array(labels[ids[:16]])
    for prefill in ((False,) if spec.get("column") else (False, True)):
        a = candidate_module.moe(mx.array(x[:16]), local, mx.array(scores[:16]), banks, slot_map,
            capacity=6, hidden_size=args.hidden_size, intermediate_size=args.moe_intermediate_size, prefill=prefill)
        b = candidate_module.moe(mx.array(x[:16]), global_ids, mx.array(scores[:16]), banks, sparse_map,
            capacity=6, hidden_size=args.hidden_size, intermediate_size=args.moe_intermediate_size, prefill=prefill)
        mx.eval(a, b)
        if not np.array_equal(np.array(a).view(np.uint32), np.array(b).view(np.uint32)):
            raise RuntimeError("sparse global slot mapping changed output")
    bad_ids = mx.array([[6, 384, 0xffffffff, 1, 2, 3]], mx.uint32)
    expected = np.array([[True, True, True, False, False, False]])
    if not np.array_equal(np.array(tiled.missing_routes(bad_ids, slot_map, 6)), expected):
        raise RuntimeError("missing/out-of-range route guard differs")
    empty = tuple(mx.array([], mx.uint8) for _ in range(9))
    value = candidate_module.moe(mx.array(x[:1]), mx.array(ids[:1]), mx.array(scores[:1]), empty,
        mx.full((384,), 0xffffffff, mx.uint32), capacity=0,
        hidden_size=args.hidden_size, intermediate_size=args.moe_intermediate_size)
    if not bool(mx.all(value == 0).item()):
        raise RuntimeError("empty bank output is unsafe")
    report["sparse_and_missing_guards"] = True


def unit(out):
    from mlx2.process_env import apply_process_numerics
    apply_process_numerics()
    spec = json.loads((out/"spec.json").read_text())
    report = dict(spec=spec, passed=False, full_model=False, qualification=False,
                  performance_claim=False, thermal_controlled=False)
    try:
        report["ownership"] = check_ownership(spec)
        for name, expected in spec["source_hashes"].items():
            if sha(ROOT/name) != expected:
                raise RuntimeError("owning source changed: "+name)
        from probe_deepseek_v41_gpu import environment
        report["before"] = environment()
        import mlx.core as mx
        mx.set_default_device(mx.gpu); mx.set_memory_limit(2 << 30); mx.set_cache_limit(128 << 20)
        execute(mx, spec, out, report); report["passed"] = True
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        if "mx" in locals():
            mx.synchronize(); gc.collect(); mx.clear_cache()
            report["cleanup"] = dict(active_bytes=mx.get_active_memory(), cache_bytes=mx.get_cache_memory())
        report["finished_at"] = time.time(); atomic(out/"results.json", report)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("owned", "unit")); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--column", action="store_true")
    parser.add_argument("--reference-column", type=Path)
    args = parser.parse_args(); out = args.output.resolve()
    if args.mode == "unit": return unit(out)
    if args.repetitions not in (0, 3, 5): raise ValueError("bounded repetitions are0/3/5")
    if out.exists() and any(out.iterdir()): raise ValueError("refusing nonempty evidence")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.adapters.deepseek_v41_iq1_artifact import inspect_artifact
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    artifact = inspect_artifact(MODEL)
    names = ["scripts/research/probe_deepseek_v41_tiled_gpu.py", "src/mlx2/runtime/gguf/v41_iq1_tiled.py",
             "src/mlx2/runtime/gguf/v41_iq1_splash.py", "src/mlx2/runtime/gguf/v41_row_exact.py",
             "src/mlx2/runtime/gguf/v41_row_prefill.py", "src/mlx2/runtime/gguf/v41_prefill.py",
             "src/mlx2/runtime/gguf/segmented_prefill.py"]
    sources = {name: sha(ROOT/name) for name in names}
    if args.reference_column:
        if not args.column:
            raise ValueError("reference column requires --column")
        name = str(args.reference_column.resolve().relative_to(ROOT))
        names.append(name); sources[name] = sha(ROOT/name)
    if args.column:
        name = "src/mlx2/runtime/gguf/v41_iq1_column.py"
        names.append(name); sources[name] = sha(ROOT/name)
    for name in names:
        target = out/"source"/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((ROOT/name).read_bytes())
    spec = dict(model=str(MODEL), fingerprint=artifact["identity"]["fingerprint"],
        checkpoint=MODEL.name+"@"+artifact["identity"]["fingerprint"], source_hashes=sources,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        driver_sha256=sha(__file__), repetitions=args.repetitions, column=args.column,
        reference_column=str(args.reference_column.resolve()) if args.reference_column else None,
        out=str(out), job_id=out.name,
        mcp_url="http://127.0.0.1:8766/mcp", scope="six actual layer0 experts; exact and selectedNAX; no model inference")
    atomic(out/"spec.json", spec)
    with DesktopCPGLease(out, out.name, checkpoint=spec["checkpoint"], source_revision=spec["source_revision"]) as lease:
        cmd = [sys.executable, str(ROOT/"scripts/run_with_gpu_locks.py"), "--session", lease.owner["cpg_session"],
               "--label", out.name, "--desktop-cpg-lease", str(lease.path), "--receipt", str(out/"paired-locks.json"),
               "--", sys.executable, __file__, "unit", "--output", str(out)]
        child = spawn(cmd, start_new_session=True); deadline = time.monotonic()+600
        try:
            while child.poll() is None:
                lease.check()
                if time.monotonic() > deadline: raise TimeoutError("bounded tiled component unit")
                time.sleep(.2)
            lease.success = child.returncode == 0
            return child.returncode
        finally: child.stop(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
