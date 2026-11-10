#!/usr/bin/env python3
"""Selection-swap probe: oracle vs candidate blocks vs N-LSA on one checkpoint.

Teacher-forced forwards of held-out windows under each sparse-selection arm,
same weights. Reports cross-entropy over all positions and over the sparse
region (positions past local + global, where selection actually drops keys),
and the cross anchors' attention-mass recall: the share of each anchor's
mean-over-heads attention probability that the selected support covers, for
the last ``--recall-queries`` rows. The oracle arm is the recall ceiling for
its budget. A checkpoint trained with one selector measures the others
training-free; this is not N-LSA's native-training result.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ARMS = {
    "oracle": {"candidate_block_size": 0, "candidate_blocks": 0},
    "candidate": {},  # the checkpoint's own selector settings
    "nlsa128": {
        "candidate_block_size": 0,
        "candidate_blocks": 0,
        "sparse_selection": "nlsa",
        "nlsa_cluster_block": 128,
    },
    "nlsa512": {
        "candidate_block_size": 0,
        "candidate_blocks": 0,
        "sparse_selection": "nlsa",
        "nlsa_cluster_block": 512,
    },
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokens-dir", type=Path, required=True)
    p.add_argument("--sources", nargs="+", default=["cosmopedia", "mlx2-source", "codex-sessions"])
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--recall-queries", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument(
        "--owned",
        action="store_true",
        help="take a CPG lease and the paired GPU locks, then rerun under them",
    )
    args = p.parse_args(argv)
    if args.owned:
        return owned(args, argv if argv is not None else sys.argv[1:])
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite nonempty {args.output}")

    import mlx.core as mx
    import numpy as np

    from mlx2.experimental.hysparse2 import model as model_module
    from mlx2.experimental.hysparse2.config import Config
    from mlx2.experimental.hysparse2.model import Model
    from mlx2.experimental.hysparse2.train import initialize_from_checkpoint

    state = json.loads((args.checkpoint / "state.json").read_text())
    base = Config(**state["config"])
    receipt_tokens = json.loads((args.tokens_dir / "receipt.json").read_text())
    if receipt_tokens["tokenizer_sha256"] != state["run"]["tokenizer_sha256"]:
        raise SystemExit("token files use a different tokenizer than the checkpoint")
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    mx.set_memory_limit(48 << 30)
    mx.set_cache_limit(1 << 30)
    model = Model(base)
    identity = initialize_from_checkpoint(args.checkpoint, model)
    model.eval()
    if args.dtype == "bfloat16":
        model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    def use(config):
        # Config objects are plain attributes, outside the module's array dict.
        swapped = 0
        for module in model.modules():
            for key, value in list(vars(module).items()):
                if isinstance(value, Config):
                    object.__setattr__(module, key, config)
                    swapped += 1
        if not swapped:
            raise RuntimeError("no Config attribute found to swap")

    captured = []
    original = model_module.attention

    def recording(q, blocks, **kw):
        out, selected = original(q, blocks, **kw)
        if kw.get("select") is not None:
            captured.append((q, blocks, selected, kw.get("nlsa") is not None))
        return out, selected

    def recall(q, blocks, selected, rows):
        k = mx.concatenate([b[0] for b in blocks], axis=2)
        total = q.shape[2]
        start = total - rows
        query = q[:, :, start:].astype(mx.float32)
        scores = (query @ k.astype(mx.float32).swapaxes(-1, -2)) * q.shape[-1] ** -0.5
        qp = mx.arange(start, total)
        valid = mx.arange(k.shape[2])[None, :] <= qp[:, None]
        probs = mx.softmax(mx.where(valid[None, None], scores, -1e30), axis=-1)
        mass = mx.mean(probs, axis=1)[0]  # rows, keys
        positions = selected[2][0, start:]
        ok = positions < k.shape[2]
        picked = mx.take_along_axis(mass, mx.where(ok, positions, 0), axis=-1)
        return mx.mean(mx.sum(mx.where(ok, picked, 0.0), axis=-1)).item()

    sparse_from = base.local_window + base.global_tokens
    results = []
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    receipt = {
        "schema": "mlx2.hysparse2-nlsa-probe.v1",
        "checkpoint": f"{args.checkpoint.parent.name}/{args.checkpoint.name}@{identity['model_sha256'][:12]}",
        "checkpoint_identity": identity,
        "source_revision": commit,
        "source_dirty": dirty,
        "host": platform.node(),
        "dtype": args.dtype,
        "device": args.device,
        "base_config": asdict(base),
        "arms": {name: ARMS[name] for name in args.arms},
        "sparse_region_starts_at": sparse_from,
        "results": results,
        "claims": "mechanism probe on an undertrained checkpoint; no quality qualification",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_module.attention = recording
    try:
        for source in args.sources:
            values = np.load(args.tokens_dir / f"valid-{source}.npy", mmap_mode="r")
            for length in args.lengths:
                if len(values) < length + 1:
                    continue
                tokens = mx.array(np.asarray(values[: length + 1], dtype=np.uint32)[None])
                for arm in args.arms:
                    use(replace(base, **ARMS[arm]))
                    captured.clear()
                    started = time.perf_counter()
                    logits = model(tokens[:, :-1])[0]
                    ce = (
                        mx.logsumexp(logits.astype(mx.float32), axis=-1)
                        - mx.take_along_axis(
                            logits.astype(mx.float32), tokens[:, 1:, None].astype(mx.int32), axis=-1
                        )[..., 0]
                    )[0]
                    mx.eval(ce)
                    seconds = time.perf_counter() - started
                    engaged = [flag for *_, flag in captured]
                    expected = ARMS[arm].get("sparse_selection") == "nlsa"
                    if not engaged or any(flag != expected for flag in engaged):
                        raise RuntimeError(f"{arm}: selector route did not engage as configured")
                    recalls = [recall(q, b, s, args.recall_queries) for q, b, s, _ in captured]
                    entry = {
                        "source": source,
                        "length": length,
                        "arm": arm,
                        "ce_all": ce.mean().item(),
                        "ce_sparse_region": ce[sparse_from:].mean().item(),
                        "anchor_mass_recall": recalls,
                        "anchor_mass_recall_mean": sum(recalls) / len(recalls),
                        # Anchor 0 sees identical inputs in every arm (nothing
                        # sparse precedes it), so only its recall is paired.
                        "anchor0_mass_recall_paired": recalls[0],
                        "nlsa_route_engaged": expected,
                        "forward_seconds_single_run": seconds,
                    }
                    results.append(entry)
                    print(json.dumps(entry), flush=True)
                    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
                    del logits, ce
                    captured.clear()
                    mx.clear_cache()
    finally:
        model_module.attention = original
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


def owned(args, argv):
    """One bounded window: CPG lease, paired host locks, then the probe."""
    import signal

    root = Path(__file__).resolve().parents[1]
    out = args.output.parent
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing nonempty evidence directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job = "hysparse2-nlsa-probe-" + time.strftime("%Y%m%dT%H%M%S")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    child_args = [a for a in argv if a != "--owned"]
    with DesktopCPGLease(
        out, job, checkpoint=args.checkpoint.name, source_revision=commit
    ) as lease:
        command = [
            sys.executable,
            str(root / "scripts/run_with_gpu_locks.py"),
            "--session",
            lease.owner["cpg_session"],
            "--label",
            job,
            "--desktop-cpg-lease",
            str(lease.path),
            "--receipt",
            str(out / "paired-locks.json"),
            "--",
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            *child_args,
        ]
        env = dict(os.environ, PYTHONPATH=str(root / "src"))
        child = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=60)


if __name__ == "__main__":
    sys.exit(main())
