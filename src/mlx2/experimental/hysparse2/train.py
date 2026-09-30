"""MLX-only training driver with bounded smoke, resume, and atomic checkpoints.

No dataset downloads or training occur on import. The default dry-run command
reports capacity without importing MLX. Training consumes prepared .npy tokens.
"""

import argparse
import fcntl
import hashlib
import json
import math
import platform
import re
import subprocess
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from .config import Config


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(root, model, optimizer, step, run):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step-{step:08d}"
    if target.exists():
        raise FileExistsError(target)
    temporary = root / f".writing-{uuid.uuid4().hex}"
    temporary.mkdir()
    model.save_weights(str(temporary / "model.safetensors"))
    state = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(str(temporary / "optimizer.safetensors"), state)
    metadata = {
        "schema": "mlx2.hysparse2-checkpoint.v1",
        "step": step,
        "config": asdict(model.config),
        "run": run,
    }
    (temporary / "state.json").write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.rename(target)
    return target


def load_checkpoint(path, model, optimizer, run):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    path = Path(path)
    metadata = json.loads((path / "state.json").read_text())
    if metadata["schema"] != "mlx2.hysparse2-checkpoint.v1" or metadata[
        "config"
    ] != asdict(model.config):
        raise ValueError("checkpoint configuration differs")
    if metadata["run"] != run:
        raise ValueError("resume data, tokenizer or training settings differ")
    model.load_weights(str(path / "model.safetensors"), strict=True)
    optimizer.state = tree_unflatten(
        list(mx.load(str(path / "optimizer.safetensors")).items())
    )
    mx.eval(model.parameters(), optimizer.state)
    return metadata["step"]


def loss(model, tokens, mtp_weight=0.1, router_weight=0.01):
    import mlx.core as mx
    from mlx import nn

    logits, aux, mtp = model(
        tokens[:, :-2], next_tokens=tokens[:, 1:-1] if model.config.mtp else None
    )
    value = mx.mean(nn.losses.cross_entropy(logits.astype(mx.float32), tokens[:, 1:-1]))
    if mtp is not None:
        value = value + mtp_weight * mx.mean(
            nn.losses.cross_entropy(mtp.astype(mx.float32), tokens[:, 2:])
        )
    return value + router_weight * aux


def train(args, c):
    import mlx.core as mx
    import numpy as np
    from mlx import nn, optimizers
    from mlx.utils import tree_map

    from .model import Model

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    mx.random.seed(args.seed)
    # FP32 parameters and optimizer state are deliberate for initial research
    # training. Inference can cast a saved model to BF16 separately.
    values = None
    if not args.smoke:
        values = np.load(args.tokens, mmap_mode="r", allow_pickle=False)
        if (
            values.ndim != 1
            or values.dtype != np.uint32
            or len(values) < args.sequence + 2
        ):
            raise ValueError(
                "tokens must be a one-dimensional uint32 NPY with enough tokens"
            )
        metadata_path = args.tokens.parent / "receipt.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("schema") == "mlx2.hysparse2-tokens.v1" and (
                metadata["tokenizer_sha256"] != args.tokenizer_sha256
                or metadata["vocab_size"] > c.vocab_size
            ):
                raise ValueError("prepared tokenizer hash or vocabulary differs")
    validation = None
    if args.validation_tokens is not None:
        validation = np.load(args.validation_tokens, mmap_mode="r", allow_pickle=False)
        if (
            validation.ndim != 1
            or validation.dtype != np.uint32
            or len(validation) < args.sequence + 1
        ):
            raise ValueError("validation must be a one-dimensional uint32 token NPY")
        if (
            args.tokens is not None
            and args.tokens.resolve() == args.validation_tokens.resolve()
        ):
            raise ValueError("validation and training inputs must differ")
    model = Model(c)
    model.train()
    model.checkpoint_layers = not args.no_checkpoint
    optimizer = optimizers.AdamW(
        learning_rate=args.learning_rate, betas=[0.9, 0.95], weight_decay=0.1
    )
    run = {
        "tokens_sha256": None if args.smoke else file_hash(args.tokens),
        "tokenizer_sha256": None if args.smoke else args.tokenizer_sha256,
        "validation_sha256": None
        if validation is None
        else file_hash(args.validation_tokens),
        "sequence": args.sequence,
        "batch": args.batch,
        "accumulation": args.accumulation,
        "seed": args.seed,
        "optimizer": "adamw",
        "learning_rate": args.learning_rate,
        "mtp_weight": args.mtp_weight,
        "router_weight": args.router_weight,
        "dtype": "float32",
        "checkpoint_layers": model.checkpoint_layers,
    }
    step = load_checkpoint(args.resume, model, optimizer, run) if args.resume else 0
    fn = nn.value_and_grad(
        model, lambda m, b: loss(m, b, args.mtp_weight, args.router_weight)
    )
    print(
        json.dumps(
            {
                "event": "training_start",
                "config": asdict(c),
                "capacity": c.capacity(bytes_per_element=4),
                "run": run,
            }
        ),
        flush=True,
    )
    for current in range(step, step + args.steps):
        started = time.perf_counter()
        gradients = None
        mean_loss = 0.0
        for micro in range(args.accumulation):
            rng = np.random.default_rng(args.seed + current * args.accumulation + micro)
            if args.smoke:
                batch = rng.integers(
                    0,
                    c.vocab_size,
                    size=(args.batch, args.sequence + 2),
                    dtype=np.uint32,
                )
            else:
                starts = rng.integers(
                    0, len(values) - args.sequence - 1, size=args.batch
                )
                batch = np.stack([values[s : s + args.sequence + 2] for s in starts])
                if np.any(batch >= c.vocab_size):
                    raise ValueError("token exceeds configured vocabulary")
            objective, g = fn(model, mx.array(batch))
            gradients = (
                g if gradients is None else tree_map(lambda a, b: a + b, gradients, g)
            )
            mx.eval(objective, gradients)
            mean_loss += float(objective.item()) / args.accumulation
        gradients = tree_map(lambda g: g / args.accumulation, gradients)
        gradients, norm = optimizers.clip_grad_norm(gradients, 1.0)
        mx.eval(norm)
        if not np.isfinite(mean_loss) or not bool(mx.isfinite(norm).item()):
            raise FloatingPointError(
                "nonfinite training loss or gradients; step not applied"
            )
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        print(
            json.dumps(
                {
                    "step": current + 1,
                    "loss": mean_loss,
                    "grad_norm": float(norm.item()),
                    "tokens": args.batch * args.sequence * args.accumulation,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        if validation is not None and (
            (current + 1) % args.eval_every == 0 or current == step + args.steps - 1
        ):
            rng = np.random.default_rng(args.seed)
            starts = rng.integers(0, len(validation) - args.sequence, size=args.batch)
            data = np.stack([validation[s : s + args.sequence + 1] for s in starts])
            if np.any(data >= c.vocab_size):
                raise ValueError("validation token exceeds configured vocabulary")
            model.eval()
            logits = model(mx.array(data[:, :-1]))[0]
            ce = mx.mean(
                nn.losses.cross_entropy(
                    logits.astype(mx.float32), mx.array(data[:, 1:])
                )
            )
            mx.eval(ce)
            print(
                json.dumps(
                    {
                        "step": current + 1,
                        "validation_ce": float(ce.item()),
                        "validation_tokens": args.batch * args.sequence,
                    }
                ),
                flush=True,
            )
            model.train()
        if (current + 1) % args.save_every == 0 or current == step + args.steps - 1:
            print(
                "checkpoint",
                save_checkpoint(args.output, model, optimizer, current + 1, run),
                flush=True,
            )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny test geometry and generated tokens; never measures model quality",
    )
    p.add_argument("--tokens", type=Path)
    p.add_argument("--tokenizer-sha256")
    p.add_argument("--validation-tokens", type=Path)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--sequence", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--mtp-weight", type=float, default=0.1)
    p.add_argument("--router-weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--no-checkpoint", action="store_true")
    args = p.parse_args(argv)
    if args.smoke and args.config:
        p.error("--smoke and --config are mutually exclusive")
    c = (
        Config.smoke()
        if args.smoke
        else Config(**json.loads(args.config.read_text()))
        if args.config
        else Config()
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": asdict(c),
                    "bf16_inference": c.capacity(),
                    "fp32_training_tensors": c.capacity(bytes_per_element=4),
                },
                indent=2,
            )
        )
        return 0
    if args.output is None or (
        not args.smoke and (args.tokens is None or not args.tokenizer_sha256)
    ):
        p.error(
            "training requires --output and, outside smoke, --tokens and --tokenizer-sha256"
        )
    if any(
        getattr(args, n) < 1
        for n in (
            "steps",
            "sequence",
            "batch",
            "accumulation",
            "save_every",
            "eval_every",
        )
    ):
        p.error(
            "step, sequence, batch, accumulation and checkpoint counts must be positive"
        )
    if args.sequence + 2 > c.max_context or args.sequence > 8192:
        p.error(
            "reference training supports up to 8192 tokens/step; 2M is the inference target, not a proven full-backprop training length"
        )
    if not args.smoke and not re.fullmatch(r"[0-9a-f]{64}", args.tokenizer_sha256):
        p.error("tokenizer SHA256 must be 64 lowercase hexadecimal characters")
    if (
        not all(
            math.isfinite(v)
            for v in (args.learning_rate, args.mtp_weight, args.router_weight)
        )
        or args.learning_rate <= 0
        or args.mtp_weight < 0
        or args.router_weight < 0
    ):
        p.error("invalid optimizer/loss settings")
    with ExitStack() as stack:
        if args.device == "gpu":
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                p.error("GPU training requires Apple Silicon")
            for lock in ("/Users/Shared/mlxuag/gpu.lock", "/tmp/gpu.lock"):
                f = stack.enter_context(open(lock, "a"))
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            processes = subprocess.check_output(["ps", "-axo", "command"], text=True)
            if any(
                "-m mlx2.server" in row
                or "tensorfold serve" in row
                or "run_perf.py" in row
                for row in processes.splitlines()
            ):
                raise RuntimeError("another inference/qualification process is running")
        return train(args, c)


if __name__ == "__main__":
    raise SystemExit(main())
