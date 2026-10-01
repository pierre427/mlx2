"""Strict, revision-bound LoRA artifacts for MLX media transformers.

Target layouts are researched in provenance/media-lora.json. This module owns
artifact conversion and reversible mutation, not route qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

FAMILIES = frozenset({"qwen-image-2.1", "ltx-2.5", "minimax-music3"})
_QWEN = re.compile(
    r"transformer_blocks\.(0|[1-9][0-9]*)\.(attn\.(to_q|to_k|to_v|to_out\.0)|img_mlp\.(proj|gate_layer|out))$"
)
_LTX = re.compile(
    r"transformer_blocks\.(0|[1-9][0-9]*)\.((attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.(to_q|to_k|to_v|to_out)|(ff|audio_ff)\.(proj_in|proj_out))$"
)
_MUSIC = re.compile(r"blocks\.(0|[1-9][0-9]*)\.(attn\.(to_qkv|to_out)|ff_in|ff_out)$")


def target_key(family: str, source: str) -> str:
    if family not in FAMILIES:
        raise ValueError("unsupported media LoRA family")
    key = source
    for prefix in ("base_model.model.", "diffusion_model.", "transformer."):
        key = key.removeprefix(prefix)
    if family == "ltx-2.5":
        key = (
            key.replace(".to_out.0", ".to_out")
            .replace(".ff.net.0.proj", ".ff.proj_in")
            .replace(".ff.net.2", ".ff.proj_out")
        )
        key = key.replace(".audio_ff.net.0.proj", ".audio_ff.proj_in").replace(
            ".audio_ff.net.2", ".audio_ff.proj_out"
        )
    if family == "minimax-music3":
        match = re.fullmatch(
            r"layers\.(0|[1-9][0-9]*)\.(self_attn\.(to_qkv|to_out)|ff\.ff\.(0\.proj|2))",
            key,
        )
        if match:
            projection = {
                "self_attn.to_qkv": "attn.to_qkv",
                "self_attn.to_out": "attn.to_out",
                "ff.ff.0.proj": "ff_in",
                "ff.ff.2": "ff_out",
            }[match[2]]
            key = f"blocks.{match[1]}.{projection}"
    pattern = {"qwen-image-2.1": _QWEN, "ltx-2.5": _LTX, "minimax-music3": _MUSIC}[
        family
    ]
    if not pattern.fullmatch(key):
        raise ValueError(f"unsupported {family} LoRA target: {source}")
    return key


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite numeric")
    return float(value)


def _identity(family, base_fingerprint, backend_revision):
    if family not in FAMILIES:
        raise ValueError("unsupported media LoRA family")
    if not isinstance(base_fingerprint, str) or not re.fullmatch(
        "[0-9a-f]{64}", base_fingerprint
    ):
        raise ValueError("base fingerprint must be a SHA-256 identity")
    if not isinstance(backend_revision, str) or not re.fullmatch(
        "[0-9a-f]{40}", backend_revision
    ):
        raise ValueError("backend revision must be a pinned Git SHA")


@dataclass(frozen=True)
class MediaLoRA:
    path: Path
    config: dict
    fingerprint: str

    @property
    def keys(self):
        return tuple(self.config["targets"])

    def tensors(self):
        from safetensors.numpy import load

        # Re-verify before every consumption: artifact identity binds bytes.
        current = json.loads((self.path / "adapter_config.json").read_text())
        if (
            hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
            != self.fingerprint
        ):
            raise ValueError("media LoRA manifest changed")
        if (
            hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
            != self.fingerprint
        ):
            raise ValueError("media LoRA in-memory identity changed")
        data = (self.path / "adapters.safetensors").read_bytes()
        if hashlib.sha256(data).hexdigest() != self.config["weights_sha256"]:
            raise ValueError("media LoRA weights changed")
        return load(data)


def inspect_media_lora(
    path, *, family, base_fingerprint, backend_revision
) -> MediaLoRA:
    _identity(family, base_fingerprint, backend_revision)
    root = Path(path).expanduser().resolve()
    config = json.loads((root / "adapter_config.json").read_text())
    required = {
        "format",
        "family",
        "base_fingerprint",
        "backend_revision",
        "scale",
        "targets",
        "weights_sha256",
        "training",
    }
    if (
        not isinstance(config, dict)
        or set(config) != required
        or config["format"] != "mlx2-media-lora-v1"
    ):
        raise ValueError("unsupported media LoRA manifest")
    if (config["family"], config["base_fingerprint"], config["backend_revision"]) != (
        family,
        base_fingerprint,
        backend_revision,
    ):
        raise ValueError("media LoRA base/backend identity mismatch")
    _number(config["scale"], "scale")
    targets = config["targets"]
    if (
        not isinstance(targets, list)
        or not targets
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("media LoRA targets must be nonempty and unique")
    if any(target_key(family, k) != k for k in targets):
        raise ValueError("media LoRA targets must be canonical")
    _validate_training(config["training"])
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    artifact = MediaLoRA(root, config, fingerprint)
    tensors = artifact.tensors()
    _validate_tensors(tensors, targets)
    return artifact


def _validate_training(training):
    if not isinstance(training, dict) or type(training.get("trained")) is not bool:
        raise ValueError("media LoRA requires explicit training state")
    if not training["trained"]:
        return
    if (
        training.get("objective") != "conditional-flow-velocity-mse"
        or not isinstance(training.get("data_fingerprint"), str)
        or not re.fullmatch("[0-9a-f]{64}", training["data_fingerprint"])
        or type(training.get("steps")) is not int
        or training["steps"] < 1
    ):
        raise ValueError("trained media LoRA requires training evidence")
    losses = training.get("training_losses")
    if not isinstance(losses, list) or len(losses) != training["steps"]:
        raise ValueError("trained media LoRA requires per-step losses")
    initial = _number(
        training.get("initial_validation_loss"), "initial validation loss"
    )
    final = _number(training.get("final_validation_loss"), "final validation loss")
    if (
        initial < 0
        or final < 0
        or final > initial
        or any(_number(v, "training loss") < 0 for v in losses)
    ):
        raise ValueError("trained media LoRA loss evidence failed")


def _source_tensors(data):
    """Deserialize validated safetensors bytes on CPU, including BF16 PEFT."""
    import numpy as np
    from safetensors import deserialize

    result = {}
    for name, tensor in deserialize(data):
        dtype = tensor["dtype"]
        if dtype == "BF16":
            value = (
                np.frombuffer(tensor["data"], dtype="<u2").astype("<u4") << 16
            ).view("<f4")
        elif dtype in ("F16", "F32", "F64"):
            value = np.frombuffer(
                tensor["data"], dtype={"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype]
            )
        else:
            raise ValueError("media LoRA source must be floating point")
        result[name] = value.reshape(tensor["shape"]).copy()
    return result


def _validate_tensors(tensors, targets):
    import numpy as np

    expected = {k + "." + leaf for k in targets for leaf in ("lora_a", "lora_b")}
    if set(tensors) != expected:
        raise ValueError("media LoRA tensor coverage mismatch")
    for key in targets:
        a, b = tensors[key + ".lora_a"], tensors[key + ".lora_b"]
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[1] != b.shape[0]
            or not 1 <= a.shape[1] <= 1024
            or min(*a.shape, *b.shape) <= 0
        ):
            raise ValueError(f"invalid media LoRA shape at {key}")
        for value in (a, b):
            if value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError("media LoRA tensors must be finite floating point")


def write_media_lora(
    output,
    tensors,
    *,
    family,
    base_fingerprint,
    backend_revision,
    scale=1.0,
    training=None,
):
    """Publish a new complete directory atomically; never replace an artifact."""
    import numpy as np
    from safetensors.numpy import save_file

    _identity(family, base_fingerprint, backend_revision)
    scale = _number(scale, "scale")
    targets = sorted({k.rsplit(".", 1)[0] for k in tensors})
    for key in targets:
        if target_key(family, key) != key:
            raise ValueError("export requires canonical targets")
    _validate_tensors(tensors, targets)
    if not targets:
        raise ValueError("empty media LoRA")
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(
        prefix=".media-lora-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        weights = staging / "adapters.safetensors"
        save_file(
            {k: np.ascontiguousarray(v.astype(np.float32)) for k, v in tensors.items()},
            str(weights),
        )
        config = {
            "format": "mlx2-media-lora-v1",
            "family": family,
            "base_fingerprint": base_fingerprint,
            "backend_revision": backend_revision,
            "scale": scale,
            "targets": targets,
            "weights_sha256": _digest(weights),
            "training": training or {"trained": False},
        }
        (staging / "adapter_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
        artifact = inspect_media_lora(
            staging,
            family=family,
            base_fingerprint=base_fingerprint,
            backend_revision=backend_revision,
        )
        # A directory with contents cannot overwrite a concurrently published artifact.
        os.rename(staging, output)
    return MediaLoRA(output, artifact.config, artifact.fingerprint)


def convert_media_lora(
    source, output, *, family, base_fingerprint, backend_revision, alpha, strength=1.0
):
    """Convert strict PEFT A/B (including .default) with explicit alpha/rank.

    Only declared block linear targets are accepted. DoRA, LoKr, convolutions,
    fused gate_up, biases, partial pairs and unknown keys fail before export.
    """
    alpha, strength = _number(alpha, "alpha"), _number(strength, "strength")
    source = Path(source).expanduser().resolve()
    source_data = source.read_bytes()
    source_hash = hashlib.sha256(source_data).hexdigest()
    peft = {}
    peft_config = source.parent / "adapter_config.json"
    if peft_config.is_file():
        peft = json.loads(peft_config.read_text())
        if (
            not isinstance(peft, dict)
            or any(
                peft.get(k)
                for k in ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern")
            )
            or peft.get("bias", "none") != "none"
            or peft.get("modules_to_save")
        ):
            raise ValueError("unsupported PEFT configuration")
    raw = _source_tensors(source_data)
    tensors, ranks = {}, set()
    for key, value in raw.items():
        match = re.fullmatch(r"(.+)\.lora_([AB])(?:\.default)?\.weight", key)
        if match is None:
            raise ValueError(f"unsupported media LoRA tensor: {key}")
        target = target_key(family, match[1]) + (
            ".lora_a" if match[2] == "A" else ".lora_b"
        )
        if target in tensors:
            raise ValueError("media LoRA mapping collision")
        if value.ndim != 2:
            raise ValueError("only linear LoRA tensors are supported")
        tensors[target] = value.T.copy()
        ranks.add(value.shape[0] if match[2] == "A" else value.shape[1])
    if len(ranks) != 1:
        raise ValueError("conversion requires one uniform rank")
    rank = ranks.pop()
    if rank <= 0:
        raise ValueError("invalid LoRA rank")
    if "r" in peft and (type(peft["r"]) is not int or peft["r"] != rank):
        raise ValueError("PEFT rank differs from tensor rank")
    if "lora_alpha" in peft and _number(peft["lora_alpha"], "PEFT alpha") != alpha:
        raise ValueError("PEFT alpha differs from supplied alpha")
    _check_embedded_peft_metadata(source_data, rank=rank, alpha=alpha)
    return write_media_lora(
        output,
        tensors,
        family=family,
        base_fingerprint=base_fingerprint,
        backend_revision=backend_revision,
        scale=strength * alpha / rank,
        training={
            "trained": False,
            "origin": "imported-unverified",
            "source_sha256": source_hash,
        },
    )


def _check_embedded_peft_metadata(data, *, rank, alpha):
    # Rust deserialize has already validated the file. Inspect its original
    # header as well so embedded alpha/rank cannot silently override the CLI.
    header = json.loads(data[8 : 8 + int.from_bytes(data[:8], "little")])
    metadata = header.get("__metadata__", {})

    def walk(value, field=""):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
        leaf = field.rsplit(".", 1)[-1]
        if isinstance(value, dict):
            if leaf in ("rank_pattern", "alpha_pattern"):
                if value:
                    raise ValueError("unsupported embedded PEFT configuration")
                return
            if "alpha" in field.lower():
                if leaf != "network_alphas":
                    raise ValueError("unsupported embedded PEFT alpha metadata")
                for item in value.values():
                    walk(item, "alpha")
                return
            for key, item in value.items():
                walk(item, key)
            return
        leaf = field.rsplit(".", 1)[-1]
        if (
            leaf
            in (
                "use_dora",
                "use_rslora",
                "rank_pattern",
                "alpha_pattern",
                "modules_to_save",
            )
            and value
        ):
            raise ValueError("unsupported embedded PEFT configuration")
        if leaf == "bias" and value != "none":
            raise ValueError("unsupported embedded PEFT configuration")
        if leaf in ("alpha", "lora_alpha"):
            if _number(value, "embedded PEFT alpha") != alpha:
                raise ValueError("embedded PEFT alpha differs from supplied alpha")
        elif leaf in ("r", "rank"):
            if type(value) is not int or value != rank:
                raise ValueError("embedded PEFT rank differs from tensor rank")
        elif "alpha" in field.lower():
            raise ValueError("unsupported embedded PEFT alpha metadata")

    walk(metadata)


def validate_model_targets(model, artifact):
    from mlx import nn

    tensors, modules = artifact.tensors(), dict(model.named_modules())
    for key in artifact.keys:
        linear = modules.get(key)
        if not isinstance(linear, (nn.Linear, nn.QuantizedLinear)):
            raise ValueError(  # noqa: TRY004 - artifact incompatibility is a ValueError API
                f"media LoRA target {key} is not an ordinary linear"
            )
        output, inputs = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            inputs = inputs * 32 // linear.bits
        if (
            tensors[key + ".lora_a"].shape[0] != inputs
            or tensors[key + ".lora_b"].shape[1] != output
        ):
            raise ValueError(f"media LoRA model shape mismatch at {key}")
    return tensors, modules


@dataclass
class MediaLoRASession:
    model: object
    artifact: MediaLoRA
    originals: dict
    wrappers: dict
    restored: bool = False

    def check_restore(self):
        if not self.restored:
            current = dict(self.model.named_modules())
            if any(current.get(k) is not v for k, v in self.wrappers.items()):
                raise RuntimeError("media LoRA model changed since installation")

    def restore(self):
        from mlx.utils import tree_unflatten

        if self.restored:
            return
        self.check_restore()
        self.model.update_modules(tree_unflatten(list(self.originals.items())))
        self.restored = True

    def export(self, output, *, training=None):
        import numpy as np

        tensors = {
            k + "." + leaf: np.array(getattr(v, leaf))
            for k, v in self.wrappers.items()
            for leaf in ("lora_a", "lora_b")
        }
        return write_media_lora(
            output,
            tensors,
            family=self.artifact.config["family"],
            base_fingerprint=self.artifact.config["base_fingerprint"],
            backend_revision=self.artifact.config["backend_revision"],
            scale=self.artifact.config["scale"],
            training=training,
        )


def install_media_lora(model, artifact):
    """Validate and materialize every delta before publishing any replacement."""
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_unflatten

    tensors, modules = validate_model_targets(model, artifact)

    class MediaLinear(nn.Module):
        def __init__(self, linear, a, b):
            super().__init__()
            self.linear = linear
            self.lora_a, self.lora_b = mx.array(a), mx.array(b)
            self.scale = artifact.config["scale"]

        def __call__(self, x):
            base = self.linear(x)
            return base + (self.scale * ((x @ self.lora_a) @ self.lora_b)).astype(
                base.dtype
            )

    originals = {key: modules[key] for key in artifact.keys}
    wrappers = {
        key: MediaLinear(
            originals[key], tensors[key + ".lora_a"], tensors[key + ".lora_b"]
        )
        for key in artifact.keys
    }
    mx.eval([v.parameters() for v in wrappers.values()])
    try:
        model.update_modules(tree_unflatten(list(wrappers.items())))
    except BaseException:
        model.update_modules(tree_unflatten(list(originals.items())))
        raise
    return MediaLoRASession(model, artifact, originals, wrappers)


def export_ltx_native(artifact, output):
    """Native LTX fusion uses B@A*strength: bake our full scale once in B."""
    import numpy as np
    from safetensors.numpy import save_file

    if artifact.config["family"] != "ltx-2.5":
        raise ValueError("LTX export requires LTX artifact")
    raw = artifact.tensors()
    tensors = {}
    for key in artifact.keys:
        tensors[f"diffusion_model.{key}.lora_A.weight"] = raw[key + ".lora_a"].T.copy()
        tensors[f"diffusion_model.{key}.lora_B.weight"] = np.ascontiguousarray(
            raw[key + ".lora_b"].T * artifact.config["scale"]
        )
    if any(not np.isfinite(value).all() for value in tensors.values()):
        raise ValueError("LTX scaled tensors overflowed")
    save_file(tensors, str(output))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert strict PEFT media LoRA to revision-bound mlx2 artifacts"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--base-fingerprint", required=True)
    parser.add_argument("--backend-revision", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--strength", type=float, default=1.0)
    args = vars(parser.parse_args())
    artifact = convert_media_lora(**args)
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "fingerprint": artifact.fingerprint,
                "targets": artifact.keys,
            }
        )
    )


if __name__ == "__main__":
    main()
