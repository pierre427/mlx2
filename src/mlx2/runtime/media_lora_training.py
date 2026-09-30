"""Bounded MLX LoRA training on pre-encoded conditional flow examples.

This is latent/velocity training, not a raw image/video/audio dataset pipeline.
Qwen/LTX use noise-minus-data velocity; Music3 uses data-minus-noise velocity.
Encoder, VAE, audio alignment and patchification stay backend-owned.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from dataclasses import dataclass

from .media_lora import install_media_lora, target_key, write_media_lora


@dataclass(frozen=True)
class FlowExample:
    family: str
    base_fingerprint: str
    inputs: dict
    targets: tuple


def _array(value, *, ndim, label):
    import mlx.core as mx

    if value.ndim != ndim or value.shape[0] != 1 or any(d <= 0 for d in value.shape):
        raise ValueError(
            f"{label} must have rank {ndim}, batch one and nonempty dimensions"
        )
    if value.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError(f"{label} must be floating point")
    if value.size > 4_194_304 or not mx.all(mx.isfinite(value)).item():
        raise ValueError(f"{label} exceeds the finite tensor budget")


def _flow(clean, noise, time, *, reverse=False):
    import mlx.core as mx

    _array(clean, ndim=3, label="clean latent")
    _array(noise, ndim=3, label="noise")
    if (
        clean.shape != noise.shape
        or not isinstance(time, (int, float))
        or isinstance(time, bool)
        or not math.isfinite(time)
        or not 0 < time < 1
    ):
        raise ValueError("flow noise shape must match and time must be in (0,1)")
    if reverse:
        latent, velocity = (1 - time) * noise + time * clean, clean - noise
    else:
        latent, velocity = (1 - time) * clean + time * noise, noise - clean
    return latent, velocity, mx.array([time], dtype=clean.dtype)


def image_flow_example(
    clean, noise, condition, *, time, img_shape, base_fingerprint, condition_mask=None
):
    _array(condition, ndim=3, label="image text conditioning")
    latent, velocity, timestep = _flow(clean, noise, time)
    if (
        not isinstance(img_shape, tuple)
        or len(img_shape) != 3
        or any(
            isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in img_shape
        )
        or math.prod(img_shape) != clean.shape[1]
        or clean.shape[1] > 4096
    ):
        raise ValueError("Qwen img_shape must match bounded packed latent tokens")
    inputs = {
        "hidden_states": latent,
        "encoder_hidden_states": condition,
        "timestep": timestep,
        "img_shape": img_shape,
    }
    if condition_mask is not None:
        if tuple(condition_mask.shape) != tuple(condition.shape[:2]):
            raise ValueError("image text mask shape mismatch")
        inputs["encoder_hidden_states_mask"] = condition_mask
    return FlowExample("qwen-image-2.1", base_fingerprint, inputs, (velocity,))


def music_flow_example(clean, noise, condition, *, time, base_fingerprint):
    _array(condition, ndim=3, label="music aligned conditioning")
    if (
        clean.shape[1] != 128
        or condition.shape[1] != clean.shape[2]
        or condition.shape[2] != 2048
        or clean.shape[2] > 4096
    ):
        raise ValueError("Music3 requires [1,128,L] latents and [1,L,2048] alignment")
    latent, velocity, timestep = _flow(clean, noise, time, reverse=True)
    return FlowExample(
        "minimax-music3",
        base_fingerprint,
        {"hidden": latent, "timestep": timestep, "cond": condition},
        (velocity,),
    )


def video_flow_example(
    clean_video,
    noise_video,
    clean_audio,
    noise_audio,
    *,
    time,
    video_condition,
    audio_condition,
    video_positions,
    audio_positions,
    base_fingerprint,
):
    _array(video_condition, ndim=3, label="video text conditioning")
    _array(audio_condition, ndim=3, label="audio text conditioning")
    video, video_target, timestep = _flow(clean_video, noise_video, time)
    audio, audio_target, _ = _flow(clean_audio, noise_audio, time)
    for label, latent, positions in (
        ("video", video, video_positions),
        ("audio", audio, audio_positions),
    ):
        if (
            latent.shape[1] > 4096
            or positions.ndim != 3
            or positions.shape[:2] != latent.shape[:2]
        ):
            raise ValueError(f"LTX {label} positions must match bounded packed tokens")
    inputs = {
        "video_latent": video,
        "audio_latent": audio,
        "timestep": timestep,
        "video_text_embeds": video_condition,
        "audio_text_embeds": audio_condition,
        "video_positions": video_positions,
        "audio_positions": audio_positions,
    }
    return FlowExample(
        "ltx-2.5", base_fingerprint, inputs, (video_target, audio_target)
    )


def _loss(model, example):
    import mlx.core as mx

    predictions = model(**example.inputs)
    predictions = predictions if isinstance(predictions, tuple) else (predictions,)
    if len(predictions) != len(example.targets):
        raise ValueError("media training output modality count mismatch")
    losses = []
    for predicted, target in zip(predictions, example.targets):
        if predicted.shape != target.shape:
            raise ValueError("media training output shape mismatch")
        losses.append(
            mx.mean(mx.square(predicted.astype(mx.float32) - target.astype(mx.float32)))
        )
    return sum(losses) / len(losses)


def train_media_lora(
    model,
    *,
    family,
    base_fingerprint,
    backend_revision,
    keys,
    examples,
    validation_examples,
    output,
    rank=4,
    alpha=4.0,
    steps=10,
    learning_rate=1e-3,
    seed=0,
    resume=None,
):
    """Train only A/B, gate on held-out velocity loss, export, restore the base.

    Explicit steps (1..100), batch-one examples, max 16 cached examples, and
    4M values per tensor bound this experiment. GPU ownership is the caller's
    responsibility; CPU operation is supported. No route becomes qualified.
    """
    import mlx.core as mx
    import numpy as np
    from mlx import nn, optimizers
    from mlx.utils import tree_flatten

    from .media_lora import inspect_media_lora

    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("media training steps must be in 1..100")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 128:
        raise ValueError("training rank must be in 1..128")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
        or isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.1
    ):
        raise ValueError("invalid media training alpha or learning rate")
    if (
        not keys
        or len(set(keys)) != len(keys)
        or any(target_key(family, k) != k for k in keys)
    ):
        raise ValueError("training requires unique canonical targets")
    examples, validation_examples = tuple(examples), tuple(validation_examples)
    if not 1 <= len(examples) <= 16 or not 1 <= len(validation_examples) <= 16:
        raise ValueError("training requires 1..16 train and held-out examples each")
    digest = hashlib.sha256()
    for split, dataset in (("train", examples), ("validation", validation_examples)):
        digest.update(split.encode())
        for example in dataset:
            if example.family != family or example.base_fingerprint != base_fingerprint:
                raise ValueError("training example identity mismatch")
            for key, value in sorted(example.inputs.items()):
                digest.update(key.encode())
                if hasattr(value, "shape"):
                    if value.size > 4_194_304 or not mx.all(mx.isfinite(value)).item():
                        raise ValueError("invalid training tensor budget or values")
                    digest.update(str((value.shape, value.dtype)).encode())
                    digest.update(np.array(value.astype(mx.float32)).tobytes())
                else:
                    digest.update(str(value).encode())
            for value in example.targets:
                _array(value, ndim=3, label="velocity target")
                digest.update(np.array(value.astype(mx.float32)).tobytes())
    modules = dict(model.named_modules())
    tensors = {}
    rng = np.random.default_rng(seed)
    parameter_count = 0
    for key in keys:
        linear = modules.get(key)
        if not isinstance(linear, (nn.Linear, nn.QuantizedLinear)):
            raise ValueError(  # noqa: TRY004 - artifact incompatibility is a ValueError API
                f"unsupported training target {key}"
            )
        outputs, inputs = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            inputs = inputs * 32 // linear.bits
        parameter_count += (inputs + outputs) * rank
        if parameter_count > 16_777_216:
            raise ValueError("media training exceeds 16M LoRA parameter budget")
        tensors[key + ".lora_a"] = (rng.standard_normal((inputs, rank)) * 0.01).astype(
            np.float32
        )
        tensors[key + ".lora_b"] = np.zeros((rank, outputs), np.float32)

    # Preserve the exact prior freeze and train/eval flags, including on failure.
    flags = [
        (module, set(module._no_grad), module.training)
        for _, module in model.named_modules()
    ]
    session = None
    try:
        with tempfile.TemporaryDirectory(prefix="mlx2-media-train-") as temporary:
            if resume is None:
                artifact = write_media_lora(
                    temporary + "/initial",
                    tensors,
                    family=family,
                    base_fingerprint=base_fingerprint,
                    backend_revision=backend_revision,
                    scale=alpha / rank,
                )
            else:
                artifact = inspect_media_lora(
                    resume,
                    family=family,
                    base_fingerprint=base_fingerprint,
                    backend_revision=backend_revision,
                )
                if set(artifact.keys) != set(keys):
                    raise ValueError("resume targets differ")
                resumed_tensors = artifact.tensors()
                if (
                    any(resumed_tensors[k + ".lora_a"].shape[1] > 128 for k in keys)
                    or sum(v.size for v in resumed_tensors.values()) > 16_777_216
                ):
                    raise ValueError(
                        "resumed media training exceeds rank/parameter budget"
                    )
            session = install_media_lora(model, artifact)
            model.freeze()
            for wrapper in session.wrappers.values():
                wrapper.unfreeze(recurse=False, keys=["lora_a", "lora_b"], strict=True)
            expected = {k + "." + leaf for k in keys for leaf in ("lora_a", "lora_b")}
            if set(dict(tree_flatten(model.trainable_parameters()))) != expected:
                raise RuntimeError("media training base freeze coverage failed")
            model.eval()
            initial_loss = sum(
                float(_loss(model, e).item()) for e in validation_examples
            ) / len(validation_examples)
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            loss_and_grad = nn.value_and_grad(model, _loss)
            losses = []
            for step in range(steps):
                loss, grads = loss_and_grad(model, examples[step % len(examples)])
                mx.eval(loss, grads)
                if not math.isfinite(loss.item()) or any(
                    not mx.all(mx.isfinite(v)).item() for _, v in tree_flatten(grads)
                ):
                    raise ValueError("nonfinite media training loss/gradient")
                optimizer.update(model, grads)
                mx.eval(model.trainable_parameters(), optimizer.state)
                losses.append(float(loss.item()))
            final_loss = sum(
                float(_loss(model, e).item()) for e in validation_examples
            ) / len(validation_examples)
            if (
                not math.isfinite(initial_loss)
                or not math.isfinite(final_loss)
                or final_loss > initial_loss
            ):
                raise ValueError(
                    "media training held-out loss gate failed; base restored"
                )
            training = {
                "trained": True,
                "qualified": False,
                "objective": "conditional-flow-velocity-mse",
                "data_fingerprint": digest.hexdigest(),
                "steps": steps,
                "initial_validation_loss": initial_loss,
                "final_validation_loss": final_loss,
                "training_losses": losses,
                "seed": seed,
                "resumed_from": artifact.fingerprint if resume is not None else None,
            }
            return session.export(output, training=training)
    finally:
        if session is not None:
            session.restore()
        for module, frozen, training in flags:
            module._no_grad.clear()
            module._no_grad.update(frozen)
            module._training = training
