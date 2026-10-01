"""MLX-only training driver with bounded smoke, resume, and atomic checkpoints.

No dataset downloads or training occur on import. The default dry-run command
reports capacity without importing MLX. Training consumes prepared .npy tokens.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from .config import Config


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_base_checkpoint_topology(model):
    from .lora import EpisodeLinear

    if (
        getattr(model, "_lora_episode", None) is not None
        or model.adapter_revision is not None
        or any(isinstance(module, EpisodeLinear) for _, module in model.named_modules())
    ):
        raise ValueError("base checkpoints do not support LoRA overlays; use adapter export and restore the base first")


def save_checkpoint(root, model, optimizer, step, run, mode="full"):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if mode not in {"full", "model"}:
        raise ValueError("checkpoint mode must be full or model")
    _require_base_checkpoint_topology(model)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step-{step:08d}"
    if target.exists():
        raise FileExistsError(target)
    temporary = root / f".writing-{uuid.uuid4().hex}"
    # Adam moments plus weights can exceed 20 GB at the default geometry.
    # Refuse before opening any checkpoint payload when disk headroom is low.
    flat_parameters = tree_flatten(model.parameters())
    ple = {
        name: value
        for name, value in flat_parameters
        if name.startswith("semantic_ple.")
    }
    model_bytes = sum(x.nbytes for _, x in flat_parameters)
    state_bytes = (
        sum(x.nbytes for _, x in tree_flatten(optimizer.state)) if mode == "full" else 0
    )
    sidecar_bytes = sum(x.nbytes for x in ple.values())
    if shutil.disk_usage(root).free < int(
        (model_bytes + state_bytes + sidecar_bytes) * 1.05
    ) + (64 << 20):
        raise OSError("insufficient disk space for atomic checkpoint")
    temporary.mkdir()
    try:
        model.save_weights(str(temporary / "model.safetensors"))
        permanent_sidecar = None
        if ple:
            sidecar_path = temporary / "semantic-ple.safetensors"
            mx.save_safetensors(str(sidecar_path), ple)
            sidecar_sha256 = file_hash(sidecar_path)
            permanent_sidecar = {
                "schema": "mlx2.hysparse2-semantic-ple.v1",
                "file": sidecar_path.name,
                "sha256": sidecar_sha256,
                "rows": model.config.semantic_ple_rows,
                "dimension": model.config.semantic_ple_dim,
                "ngram": model.config.semantic_ngram,
                "apcv2_identity": model.config.apcv2_identity(
                    ple_sidecar_digest=sidecar_sha256
                ),
            }
        if mode == "full":
            state = dict(tree_flatten(optimizer.state))
            mx.save_safetensors(str(temporary / "optimizer.safetensors"), state)
        metadata = {
            "schema": (
                "mlx2.hysparse2-checkpoint.v1"
                if mode == "full"
                else "mlx2.hysparse2-model-checkpoint.v1"
            ),
            "step": step,
            "config": asdict(model.config),
            "run": run,
            "permanent_sidecar": permanent_sidecar,
            "capsule_binding": model.capsule_binding,
            "optimizer_state_saved": mode == "full",
            "exact_training_resume": mode == "full",
        }
        (temporary / "state.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.rename(target)
    except BaseException:
        # Remove only this attempt; other writers and prior artifacts are untouched.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if permanent_sidecar is not None:
        model.ple_sidecar_digest = permanent_sidecar["sha256"]
    return target


def _load_model_state(path, model):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    _require_base_checkpoint_topology(model)
    path = Path(path)
    metadata = json.loads((path / "state.json").read_text())
    try:
        saved_config = asdict(Config(**metadata["config"]))
    except (TypeError, ValueError):
        saved_config = None
    if metadata.get("schema") not in {
        "mlx2.hysparse2-checkpoint.v1",
        "mlx2.hysparse2-model-checkpoint.v1",
    }:
        raise ValueError("unsupported checkpoint schema")
    if saved_config != asdict(model.config):
        raise ValueError("checkpoint configuration differs")
    if metadata.get("capsule_binding") != model.capsule_binding:
        raise ValueError("checkpoint semantic capsule read binding differs")
    sidecar = metadata.get("permanent_sidecar")
    if model.config.semantic_ple_rows:
        if not isinstance(sidecar, dict):
            raise ValueError("checkpoint permanent semantic PLE sidecar differs")
        expected_identity = model.config.apcv2_identity(
            ple_sidecar_digest=sidecar.get("sha256")
        )
        legacy_identity = model.config.apcv2_identity()
        legacy_fingerprint = legacy_identity["semantic_fingerprint"]
        legacy_identity["semantic_fingerprint"] = (
            legacy_fingerprint[:2] + legacy_fingerprint[3:]
        )
        # Old checkpoints predate attention-math cache identity. Their exact
        # saved configuration is checked above; rebuild new cache identities
        # from it rather than rejecting otherwise valid saved weights.
        accepted_identities = [expected_identity, legacy_identity]
        for identity in list(accepted_identities):
            old_math = dict(identity)
            old_math["cache_layout_fingerprint"] = identity[
                "cache_layout_fingerprint"
            ].rsplit(":", 1)[0]
            accepted_identities.append(old_math)
        if sidecar.get("schema") != "mlx2.hysparse2-semantic-ple.v1" or sidecar.get(
            "apcv2_identity"
        ) not in [json.loads(json.dumps(identity)) for identity in accepted_identities]:
            raise ValueError("checkpoint permanent semantic PLE sidecar differs")
        sidecar_path = path / sidecar.get("file", "")
        if not sidecar_path.is_file() or file_hash(sidecar_path) != sidecar.get(
            "sha256"
        ):
            raise ValueError(
                "checkpoint permanent semantic PLE sidecar is missing or corrupt"
            )
    previous = tree_flatten(model.parameters())
    owner, epoch = model._cache_owner, model._parameter_epoch
    try:
        model.load_weights(str(path / "model.safetensors"), strict=True)
        mx.eval(model.parameters())
        if model.config.semantic_ple_rows:
            restored_ple = {
                name: value for name, value in tree_flatten(model.parameters())
                if name.startswith("semantic_ple.")
            }
            sidecar_ple = mx.load(str(sidecar_path))
            if set(restored_ple) != set(sidecar_ple) or any(
                restored_ple[name].shape != value.shape
                or restored_ple[name].dtype != value.dtype
                or not bool(mx.all(restored_ple[name] == value).item())
                for name, value in sidecar_ple.items()
            ):
                raise ValueError("checkpoint PLE sidecar tensors differ from model weights")
    except BaseException:
        model.update(tree_unflatten(previous))
        model._cache_owner, model._parameter_epoch = owner, epoch
        raise
    if model.config.semantic_ple_rows:
        model.ple_sidecar_digest = sidecar["sha256"]
    model._cache_owner = object()
    return metadata


def initialize_from_checkpoint(path, model):
    """Load model state while explicitly starting a new optimizer/run lineage."""
    path = Path(path)
    metadata = _load_model_state(path, model)
    sidecar = metadata.get("permanent_sidecar") or {}
    return {
        "schema": metadata["schema"],
        "step": metadata.get("step"),
        "model_sha256": file_hash(path / "model.safetensors"),
        "state_sha256": file_hash(path / "state.json"),
        "semantic_ple_sha256": sidecar.get("sha256"),
        "optimizer_state_restored": False,
        "exact_training_resume": False,
    }


def load_checkpoint(path, model, optimizer, run):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    _require_base_checkpoint_topology(model)
    path = Path(path)
    metadata = json.loads((path / "state.json").read_text())
    if metadata.get("schema") != "mlx2.hysparse2-checkpoint.v1":
        raise ValueError("checkpoint does not contain exact optimizer resume state")
    if metadata["run"] != run:
        raise ValueError("resume data, tokenizer or training settings differ")
    # Reject missing/corrupt optimizer payloads before replacing live weights.
    state = tree_unflatten(list(mx.load(str(path / "optimizer.safetensors")).items()))
    mx.eval(state)
    metadata = _load_model_state(path, model)
    optimizer.state = state
    mx.eval(model.parameters(), optimizer.state)
    return metadata["step"]


def loss(model, tokens, mtp_weight=0.1, router_weight=0.01, diffusion_weight=0.2):
    import mlx.core as mx
    from mlx import nn

    logits, aux, mtp, diffusion = model(
        tokens[:, :-2], next_tokens=tokens[:, 1:-1] if model.config.mtp else None
    )
    value = mx.mean(nn.losses.cross_entropy(logits.astype(mx.float32), tokens[:, 1:-1]))
    if mtp is not None:
        value = value + mtp_weight * mx.mean(
            nn.losses.cross_entropy(mtp.astype(mx.float32), tokens[:, 2:])
        )
    if diffusion is not None:
        diffusion_logits, mask = diffusion
        token_loss = nn.losses.cross_entropy(
            diffusion_logits.astype(mx.float32), tokens[:, :-2][:, -diffusion_logits.shape[1] :]
        )
        value = value + diffusion_weight * (
            mx.sum(mx.where(mask, token_loss, 0.0)) / mx.maximum(mx.sum(mask), 1)
        )
    return value + router_weight * aux


def train(args, c):
    import mlx.core as mx
    import numpy as np
    from mlx import nn, optimizers
    from mlx.utils import tree_map

    from .model import Model

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    if args.device == "gpu":
        mx.set_memory_limit(int(args.memory_limit_gib * 2**30))
        mx.set_cache_limit(1 << 30)
        mx.reset_peak_memory()
    mx.random.seed(args.seed)
    # FP32 parameters and optimizer state are deliberate for initial research
    # training. Inference can cast a saved model to BF16 separately.
    values = None
    if not args.smoke and args.mixture is None:
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
    mixture = None
    if args.mixture is not None:
        from .mixture import Mixture

        mixture = Mixture(
            args.mixture, tokenizer_sha256=args.tokenizer_sha256, sequence=args.sequence
        )
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
    if args.validation_tokens is not None:
        validation_metadata = args.validation_tokens.parent / "receipt.json"
        if validation_metadata.exists():
            meta = json.loads(validation_metadata.read_text())
            if (
                meta.get("schema") == "mlx2.hysparse2-tokens.v1"
                and meta["tokenizer_sha256"] != args.tokenizer_sha256
            ):
                raise ValueError("validation tokenizer differs")
    model = Model(c)
    model.train()
    model.checkpoint_layers = not args.no_checkpoint
    optimizer = optimizers.AdamW(
        learning_rate=args.learning_rate, betas=[0.9, 0.95], weight_decay=0.1
    )
    run = {
        "tokens_sha256": None
        if args.smoke or mixture is not None
        else file_hash(args.tokens),
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
    if c.diffusion_layers:
        run["diffusion_weight"] = args.diffusion_weight
    if mixture is not None:
        run["mixture"] = mixture.receipt
    if args.initialize_from is not None:
        run["initialized_from"] = initialize_from_checkpoint(
            args.initialize_from, model
        )
    step = load_checkpoint(args.resume, model, optimizer, run) if args.resume else 0
    fn = nn.value_and_grad(
        model,
        lambda m, b: loss(
            m, b, args.mtp_weight, args.router_weight, args.diffusion_weight
        ),
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
        sampled_sources = {}
        for micro in range(args.accumulation):
            rng = np.random.default_rng(args.seed + current * args.accumulation + micro)
            if args.smoke:
                batch = rng.integers(
                    0,
                    c.vocab_size,
                    size=(args.batch, args.sequence + 2),
                    dtype=np.uint32,
                )
            elif mixture is not None:
                batch, names = mixture.sample(rng, args.batch)
                for name in names:
                    sampled_sources[name] = sampled_sources.get(name, 0) + args.sequence
                if np.any(batch >= c.vocab_size):
                    raise ValueError("mixture token exceeds configured vocabulary")
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
                    "sampled_source_tokens": sampled_sources,
                    "mlx_peak_memory_bytes": mx.get_peak_memory()
                    if args.device == "gpu"
                    else None,
                    "mlx_active_memory_bytes": mx.get_active_memory()
                    if args.device == "gpu"
                    else None,
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
                save_checkpoint(
                    args.output,
                    model,
                    optimizer,
                    current + 1,
                    run,
                    mode=args.checkpoint_mode,
                ),
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
    p.add_argument("--mixture", type=Path)
    p.add_argument("--memory-limit-gib", type=float, default=64.0)
    p.add_argument(
        "--wait-for-gpu",
        type=float,
        default=0,
        help="Maximum seconds to wait without loading MLX weights",
    )
    p.add_argument("--tokenizer-sha256")
    p.add_argument("--validation-tokens", type=Path)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "Warm-start model and permanent PLE weights while starting a new "
            "optimizer/run lineage; this is never an exact resume"
        ),
    )
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--sequence", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--mtp-weight", type=float, default=0.1)
    p.add_argument("--router-weight", type=float, default=0.01)
    p.add_argument("--diffusion-weight", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument(
        "--checkpoint-mode",
        choices=["full", "model"],
        default="full",
        help=(
            "full saves exact optimizer resume state; model omits optimizer state "
            "and is an explicitly non-resumable low-disk artifact"
        ),
    )
    p.add_argument("--no-checkpoint", action="store_true")
    args = p.parse_args(argv)
    if not math.isfinite(args.wait_for_gpu) or args.wait_for_gpu < 0:
        p.error("GPU wait must be finite and nonnegative")
    if args.mixture is not None and (args.tokens is not None or args.smoke):
        p.error("--mixture is mutually exclusive with --tokens and --smoke")
    if args.resume is not None and args.initialize_from is not None:
        p.error("--resume and --initialize-from are mutually exclusive")
    if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0:
        p.error("memory limit must be finite and positive")
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
        not args.smoke
        and (
            (args.tokens is None and args.mixture is None) or not args.tokenizer_sha256
        )
    ):
        p.error(
            "training requires --output and, outside smoke, --tokens or --mixture, plus --tokenizer-sha256"
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
    from .resources import gpu_guard

    with (
        gpu_guard(wait_seconds=args.wait_for_gpu)
        if args.device == "gpu"
        else nullcontext()
    ):
        return train(args, c)


if __name__ == "__main__":
    raise SystemExit(main())
