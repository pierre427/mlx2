#!/usr/bin/env python3
"""F8 kernel probe: output-only GDN verify and accepted-prefix fold on Metal.

Feature 8 (7af38665c, ``experimental/flashinfer_gdn_replay.py``) verifies a
speculative strip without writing recurrent state (outputs + per-token
corrections only) and later folds the accepted prefix into a compact
successor. No serving adapter calls it, so this probe is kernel level:

1. law parity: native verify outputs/corrections and fold successor against
   the module's CPU oracle (small geometry, null slot, partial acceptance);
2. full Qwen3.8-27B GDN geometry (48 layers, 48 value / 16 key heads,
   128x128, T=17): the same law re-expressed in vectorized NumPy with the
   kernel's sequential accumulation order (expected bit-exact);
3. ordinary reference: mlx2's production ``gated_delta_kernel`` on the same
   operands -- outputs for every token and the state after each slot's
   accepted prefix (expected close, not equal: different association);
4. cost: device time of native verify, fold, and the production kernel
   (strip pass + accepted-prefix replay); tape bytes vs per-step snapshots.

Synthetic seeded operands (L2-normalized q/k, decay in (0.85,1), beta in
(0.05,0.95)). Component numbers, no thermal hold. Original mlx2 code.

Usage: ``probe_gdn_output_only.py owned OUT``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FULL = {"L": 48, "H": 48, "HK": 16, "V": 128, "K": 128, "T": 17}
# Synthetic operands are named by a hash of their generator (geometry + seeds).
SYNTHETIC = "synthetic-qwen38-27b-gdn@" + __import__("hashlib").sha256(
    json.dumps({"full": FULL, "seeds": [1105, 2001, 2004], "operands": "v1"},
               sort_keys=True).encode()).hexdigest()


def checkout_env() -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if p)
    return environment


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def source() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"),
            "dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no"))}


def host_sample() -> dict:
    return {"thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True),
            "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()}


def operands(rng, L, S, B, H, HK, V, K, T):
    def unit(shape):
        x = rng.standard_normal(shape).astype(np.float32)
        return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)

    return {
        "state": (rng.standard_normal((L, S, H, V, K)) * 0.1).astype(np.float32),
        "query": (unit((L, B, T, HK, K)) / np.float32(np.sqrt(K))).astype(np.float32),
        "keys": unit((L, B, T, HK, K)),
        "values": rng.standard_normal((L, B, T, H, V)).astype(np.float32),
        "decay": rng.uniform(0.85, 1.0, (L, B, T, H)).astype(np.float32),
        "beta": rng.uniform(0.05, 0.95, (L, B, T, H)).astype(np.float32),
    }


def law_numpy(op, slots, lengths, accepted):
    """Vectorized kernel law: per (l,b,h,v) row, sequential FP32 sums over K."""
    state, query, keys, values = op["state"], op["query"], op["keys"], op["values"]
    decay, beta = op["decay"], op["beta"]
    L, _, H, V, K = state.shape
    B, T, HK = query.shape[1], query.shape[2], query.shape[3]
    rep = H // HK
    outputs = np.zeros((L, B, T, H, V), np.float32)
    corrections = np.zeros_like(outputs)
    successor = np.zeros((L, B, H, V, K), np.float32)
    for b in range(B):
        if slots[b] < 0:
            continue
        s = state[:, slots[b]].copy()                       # [L,H,V,K]
        folded = s.copy() if accepted[b] == 0 else None
        for t in range(int(lengths[b])):
            k = np.repeat(keys[:, b, t], rep, axis=1)       # [L,H,K]
            q = np.repeat(query[:, b, t], rep, axis=1)
            g = decay[:, b, t][..., None]                   # [L,H,1]
            pred = np.zeros((L, H, V), np.float32)
            for j in range(K):
                s[..., j] = s[..., j] * g
                pred = pred + s[..., j] * k[:, :, None, j]
            delta = (values[:, b, t] - pred) * beta[:, b, t][..., None]
            out = np.zeros((L, H, V), np.float32)
            for j in range(K):
                s[..., j] = s[..., j] + k[:, :, None, j] * delta
                out = out + s[..., j] * q[:, :, None, j]
            corrections[:, b, t], outputs[:, b, t] = delta, out
            if t + 1 == accepted[b]:
                folded = s.copy()
        if folded is not None:
            successor[:, b] = folded
    return outputs, corrections, successor


def unit(out: Path) -> int:
    import mlx.core as mx

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.experimental import flashinfer_gdn_replay as f8
    from mlx2.runtime.models.gated_delta import gated_delta_kernel

    receipt = {"schema": "mlx2.f8-gdn-output-only-probe.v1", "source": source(),
               "checkpoint": SYNTHETIC, "device": mx.device_info(),
               "mlx": mx.__version__, "metal_source_sha256": f8.metal_source_sha256(),
               "host_before": host_sample(), "qualified": False}
    binding = f8.ReplayBinding(SYNTHETIC, source()["commit"],
                               "qwen38-gdn-output-only-candidate-v1")

    def native(op, slots, lengths, accepted, max_bytes):
        cand = f8.build_native_candidate(
            *(mx.array(op[k]) for k in ("state", "query", "keys", "values", "decay", "beta")),
            slots, lengths, binding=binding, epoch=0, enabled=True,
            max_tape_bytes=max_bytes, max_successor_bytes=max_bytes)
        outputs, corrections = cand.outputs, cand._corrections
        mx.eval(outputs, corrections)
        succ = cand.fold(accepted, current_binding=binding, current_epoch=0).states
        mx.eval(succ)
        return cand, np.array(outputs), np.array(corrections), np.array(succ)

    try:
        # 1. Law parity against the module CPU oracle (small geometry).
        rng = np.random.default_rng(1105)
        L, S, B, H, HK, V, K, T = 2, 4, 3, 4, 2, 16, 16, 5
        op = operands(rng, L, S, B, H, HK, V, K, T)
        slots = np.array([2, -1, 0], np.int64)
        lengths = np.array([5, 5, 3], np.int64)
        accepted = np.array([3, 0, 3], np.int64)
        cpu = f8.OutputOnlyReplay(op["state"], binding, enabled=True)
        tx = cpu.verify(op["query"], op["keys"], op["values"], op["decay"], op["beta"], slots, lengths,
                        binding=binding, epoch=0)
        cpu_outputs, cpu_corr = tx.outputs.copy(), tx._corrections.copy()
        cpu_state = tx.commit(accepted, binding=binding, epoch=0)
        _, n_out, n_corr, n_succ = native(op, slots, lengths, accepted, 1 << 30)
        live = [b for b in range(B) if slots[b] >= 0 and accepted[b] > 0]
        receipt["law_parity_small"] = {
            "geometry": {"L": L, "S": S, "B": B, "H": H, "HK": HK, "V": V, "K": K, "T": T},
            "outputs_exact": bool(np.array_equal(n_out, cpu_outputs)),
            "corrections_exact": bool(np.array_equal(n_corr, cpu_corr)),
            "successor_exact": all(bool(np.array_equal(n_succ[:, b], cpu_state[:, slots[b]])) for b in live),
            "null_and_zero_accept_rows_zero": bool(np.all(n_succ[:, 1] == 0)),
            "outputs_max_abs": float(np.max(np.abs(n_out - cpu_outputs))),
        }
        print(json.dumps({"law_parity_small": receipt["law_parity_small"]}), flush=True)

        # 2-4. Full 27B GDN geometry.
        def full_cell(B):
            g = FULL
            rng = np.random.default_rng(2000 + B)
            op = operands(rng, g["L"], B, B, g["H"], g["HK"], g["V"], g["K"], g["T"])
            slots = np.arange(B, dtype=np.int64)
            lengths = np.full(B, g["T"], np.int64)
            accepted = np.array([g["T"], 0, 5, 11][:B], np.int64)
            budget = 2 << 30
            cand, n_out, n_corr, n_succ = native(op, slots, lengths, accepted, budget)
            l_out, l_corr, l_succ = law_numpy(op, slots, lengths, accepted)
            live = [b for b in range(B) if accepted[b] > 0]
            cell = {"B": B, "accepted": accepted.tolist(),
                    "law_outputs_exact": bool(np.array_equal(n_out, l_out)),
                    "law_corrections_exact": bool(np.array_equal(n_corr, l_corr)),
                    "law_successor_exact": all(bool(np.array_equal(n_succ[:, b], l_succ[:, b])) for b in live),
                    "law_outputs_max_abs": float(np.max(np.abs(n_out - l_out)))}
            # Ordinary reference: production kernel, layers folded into batch.
            LB = g["L"] * B

            def flat(x):
                return mx.array(x.reshape(LB, *x.shape[2:]))

            q, k, v = flat(op["query"]), flat(op["keys"]), flat(op["values"])
            gd, bt = flat(op["decay"]), flat(op["beta"])
            st = mx.array(op["state"][:, slots].reshape(LB, g["H"], g["V"], g["K"]))
            y_ref, _ = gated_delta_kernel(q, k, v, gd, bt, st, None)
            mx.eval(y_ref)
            y_ref = np.array(y_ref).reshape(g["L"], B, g["T"], g["H"], g["V"])
            peak = float(np.max(np.abs(y_ref)))
            cell["ordinary_outputs_max_abs"] = float(np.max(np.abs(n_out - y_ref)))
            cell["ordinary_outputs_max_rel_to_peak"] = cell["ordinary_outputs_max_abs"] / peak
            state_diffs = []
            for b in live:
                a = int(accepted[b])
                sl = slice(b, b + 1)

                def pick(x, a=a, sl=sl):
                    return mx.array(x[:, sl, :a].reshape(g["L"], a, *x.shape[3:]))

                _, s_ref = gated_delta_kernel(
                    pick(op["query"]), pick(op["keys"]), pick(op["values"]), pick(op["decay"]),
                    pick(op["beta"]), mx.array(op["state"][:, int(slots[b])]), None)
                mx.eval(s_ref)
                s_ref = np.array(s_ref)
                state_diffs.append(float(np.max(np.abs(n_succ[:, b] - s_ref)) / np.max(np.abs(s_ref))))
            cell["ordinary_successor_max_rel_to_peak"] = max(state_diffs) if state_diffs else None

            # Cost: median of 7 device timings each.
            def timed(fn):
                mx.eval(fn())
                times = []
                for _ in range(7):
                    mx.synchronize()
                    start = time.perf_counter()
                    mx.eval(fn())
                    mx.synchronize()
                    times.append(time.perf_counter() - start)
                return sorted(times)[3] * 1e3

            arrays = {name: mx.array(op[name]) for name in op}

            def native_verify():
                c = f8.build_native_candidate(
                    *(arrays[n] for n in ("state", "query", "keys", "values", "decay", "beta")),
                    slots, lengths, binding=binding, epoch=0, enabled=True,
                    max_tape_bytes=budget, max_successor_bytes=budget)
                return [c.outputs, c._corrections]

            def native_fold():
                c = f8.build_native_candidate(
                    *(arrays[n] for n in ("state", "query", "keys", "values", "decay", "beta")),
                    slots, lengths, binding=binding, epoch=0, enabled=True,
                    max_tape_bytes=budget, max_successor_bytes=budget)
                mx.eval(c.outputs, c._corrections)
                return c.fold(accepted, current_binding=binding, current_epoch=0).states

            def ordinary_strip():
                return list(gated_delta_kernel(q, k, v, gd, bt, st, None))

            cell["ms_native_verify"] = timed(native_verify)
            cell["ms_native_verify_plus_fold"] = timed(native_fold)
            cell["ms_ordinary_strip_with_final_state"] = timed(ordinary_strip)
            state_bytes = g["L"] * B * g["H"] * g["V"] * g["K"] * 4
            cell["bytes_state_per_snapshot"] = state_bytes
            cell["bytes_per_step_snapshots"] = state_bytes * g["T"]
            cell["bytes_tape"] = g["L"] * B * g["T"] * (g["HK"] * g["K"] + g["H"] * g["V"] + g["H"]) * 4
            del cand
            return cell

        for B in (1, 4):
            cell = full_cell(B)
            receipt[f"full_B{B}"] = cell
            atomic_json(out / "receipt.json", receipt)
            print(json.dumps(cell), flush=True)
        receipt["passed"] = True
    except BaseException as exc:
        receipt.update(passed=False, error=repr(exc))
        raise
    finally:
        receipt["host_after"] = host_sample()
        atomic_json(out / "receipt.json", receipt)
    return 0


def owned(out: Path, job_id: str) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out, job_id, checkpoint=SYNTHETIC.split("@")[0],
                         source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out)]
        child = subprocess.Popen(command, cwd=ROOT, env=checkout_env(), start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--job-id", default="f8-gdn-output-only-20261105")
    args = parser.parse_args()
    out = args.out.resolve()
    return unit(out) if args.mode == "unit" else owned(out, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
