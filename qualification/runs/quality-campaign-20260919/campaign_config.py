"""Static campaign matrix and CPU-only preflight validation.

This module is deliberately import-safe: defining the campaign never imports
MLX or opens a model shard.  ``validate_cpu_preflight`` opts into MLX CPU,
inspects artifacts through the registry, and loads only tokenizer/processor
configuration.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

RUN = Path(__file__).resolve().parent
# Default to the source tree that contains this campaign; the coordinator may
# point at a different pinned clean worktree explicitly.
ROOT = Path(os.environ.get("MLX2_CAMPAIGN_ROOT", str(Path(__file__).resolve().parents[3]))).resolve()
PYTHON = Path("~/Desktop/mlx2/.venv/bin/python")
SDK_PYTHON = Path("~/evalplus-venv/bin/python")
PORT = 8297
BASE_URL = f"http://127.0.0.1:{PORT}"
MLX_VLM_ROOT = Path("/private/tmp/mlx-vlm-653f1f13").resolve()
MLX_VLM_REVISION = "653f1f13e238abb313fd45071bbd04b3de414635"


@dataclass(frozen=True)
class Route:
    name: str
    flags: tuple[str, ...] = ()
    policy: str | None = None
    opt_in_policy: str | None = None
    speculative: bool = False
    apc_interior: bool = False


@dataclass(frozen=True)
class Model:
    name: str
    label: str
    path: str
    max_context: int
    routes: tuple[Route, ...]
    default_route: str
    capabilities: frozenset[str]
    cache_gib: int = 8
    ladder_cache_gib: int = 64
    media: tuple[str, ...] = ()
    extra_flags: tuple[str, ...] = ()


TEXT = frozenset({
    "text", "streaming", "tools", "reasoning", "thinking-deferral",
    "grammar", "apc",
})
MUSE_TEXT = TEXT - {"thinking-deferral"}
MM_GEMMA = frozenset({"text", "streaming", "vision", "audio", "apc"})
MM_MINICPM = frozenset({"text", "streaming", "vision", "audio", "apc"})

MODELS = (
    Model(
        "qwen36", "Qwen3.6-35B-A3B",
        "~/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp",
        262144,
        (
            Route("ordinary", ("--ordinary",), "qwen36-ordinary.json", "qwen36-ordinary-opt-in.json", apc_interior=True),
            Route("mtp2", (), "qwen36-mtp2.json", "qwen36-mtp2-opt-in.json", True, True),
            Route("prompt-lookup", ("--prompt-lookup",), "prompt-lookup.json", "qwen36-pld-opt-in.json", True),
        ),
        "mtp2", TEXT, 8, 64,
    ),
    Model(
        "qwen38", "Qwen3.8-27B", "~/mlx-models/Qwen3.8-27B-oQ4e-mtp", 262144,
        (
            Route("ordinary", ("--ordinary",), None, "qwen38-ordinary-opt-in.json", apc_interior=True),
            Route("mtp2", (), "qwen38-mtp2.json", "qwen38-mtp2-opt-in.json", True, True),
        ),
        "mtp2", TEXT, 8, 64,
    ),
    Model(
        "flash-next", "Qwen3.8 Flash-Next", "~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP", 262144,
        (
            Route("ordinary", ("--ordinary",), "flash-next-mtp2.json", "flash-next-ordinary-opt-in.json", apc_interior=True),
            Route("mtp2", (), "flash-next-mtp2.json", "flash-next-mtp2-opt-in.json", True, True),
        ),
        "mtp2", TEXT, 16, 96,
    ),
    Model(
        "muse", "Muse-Glimmer-30B", "~/mlx-models/Muse-Glimmer-30B-mlx-4bit", 131072,
        (
            Route("ordinary", ("--ordinary",), None, "muse-ordinary-opt-in.json"),
            Route("dflash2", ("--external-draft",), "muse-dflash2.json", "muse-dflash2-opt-in.json", True),
            Route("prompt-lookup", ("--prompt-lookup",), "prompt-lookup.json", "muse-pld-opt-in.json", True),
        ),
        "ordinary", MUSE_TEXT, 8, 64,
    ),
    Model(
        "north", "North-Mini-Code", "~/mlx-models/North-Mini-Code-1.0-mlx-4bit", 500000,
        (
            Route("ordinary", ("--ordinary",), None, "north-ordinary-opt-in.json"),
            Route("prompt-lookup", ("--prompt-lookup",), "prompt-lookup.json", "north-pld-opt-in.json", True),
        ),
        "ordinary", TEXT, 16, 64, extra_flags=("--no-thinking-auto-calibration",),
    ),
    Model(
        "laguna", "Laguna XS 2.1",
        "~/.cache/huggingface/hub/models--AtomicChat--Laguna-XS-2.1-MLX-8bit/snapshots/ba635f7386219675b9b71acdc2812d12fa69aab2",
        262144, (Route("ordinary", ("--ordinary",), None, "laguna-ordinary-opt-in.json"),),
        "ordinary", TEXT, 8, 64,
    ),
    Model(
        "xing", "Xing4.0-29B-A4B", "~/mlx-models/Xing4.0-29B-A4B-mlx-6bit", 262144,
        (
            Route("ordinary", ("--ordinary",), None, "xing-ordinary-opt-in.json"),
            Route("mtp1", (), "xing-mtp1.json", "xing-mtp1-opt-in.json", True),
            Route("prompt-lookup", ("--prompt-lookup",), "prompt-lookup.json", "xing-pld-opt-in.json", True),
        ),
        "mtp1", TEXT, 8, 64,
    ),
    Model(
        "gemma3n", "Gemma 3n E2B",
        "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/5e092ebca197cdcd8d8b195040accf22693501bc",
        32768, (Route("ordinary", ("--ordinary",), None, "gemma3n-ordinary-opt-in.json"),),
        "ordinary", MM_GEMMA, 8, 16, ("image", "audio"),
    ),
    Model(
        "minicpmo", "MiniCPM-o 2.6",
        "~/.cache/huggingface/hub/models--openbmb--MiniCPM-o-2_6/snapshots/06849bfd36da94da1f5a88fa17e8ba63f08a4c4d",
        32768, (Route("ordinary", ("--ordinary",), None, "minicpmo-ordinary-opt-in.json"),),
        "ordinary", MM_MINICPM, 8, 16, ("image", "audio"),
    ),
)


def policy_path(name: str | None) -> Path | None:
    return RUN / "policies" / name if name else None


def requires_mlx_vlm(model: Model) -> bool:
    return bool(model.media)


def stage_pythonpath(model: Model) -> str:
    if requires_mlx_vlm(model):
        return os.pathsep.join((str(MLX_VLM_ROOT), "src"))
    return "src"


def _command_output(command, *, cwd=None, env=None, allow_failure=False):
    result = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode and not allow_failure:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"mlx-vlm runtime check failed: {detail or command}")
    return result


def verify_mlx_vlm_runtime(*, env=None) -> dict:
    """Verify the exact detached optional runtime without loading model weights."""
    if not MLX_VLM_ROOT.is_dir():
        raise RuntimeError(f"mlx-vlm runtime check failed: missing {MLX_VLM_ROOT}")
    top = Path(
        _command_output(
            ["git", "-C", MLX_VLM_ROOT, "rev-parse", "--show-toplevel"]
        ).stdout.strip()
    ).resolve()
    if top != MLX_VLM_ROOT:
        raise RuntimeError(
            f"mlx-vlm runtime check failed: checkout root is {top}, expected {MLX_VLM_ROOT}"
        )
    revision = _command_output(
        ["git", "-C", MLX_VLM_ROOT, "rev-parse", "HEAD"]
    ).stdout.strip()
    if revision != MLX_VLM_REVISION:
        raise RuntimeError(
            "mlx-vlm runtime check failed: revision "
            f"{revision or '<missing>'} != {MLX_VLM_REVISION}"
        )
    dirty = _command_output(
        ["git", "-C", MLX_VLM_ROOT, "status", "--porcelain=v1", "--untracked-files=all"]
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"mlx-vlm runtime check failed: checkout is dirty: {dirty}")
    symbolic = _command_output(
        ["git", "-C", MLX_VLM_ROOT, "symbolic-ref", "-q", "HEAD"],
        allow_failure=True,
    )
    if symbolic.returncode == 0:
        raise RuntimeError(
            "mlx-vlm runtime check failed: checkout is not detached: "
            f"{symbolic.stdout.strip()}"
        )
    import_env = dict(os.environ if env is None else env)
    import_env["PYTHONPATH"] = os.pathsep.join((str(MLX_VLM_ROOT), "src"))
    imported = _command_output(
        [
            PYTHON,
            "-c",
            (
                "from pathlib import Path; import mlx_vlm; "
                "print(Path(mlx_vlm.__file__).resolve())"
            ),
        ],
        cwd=ROOT,
        env=import_env,
    ).stdout.strip()
    origin = Path(imported).resolve()
    package_root = (MLX_VLM_ROOT / "mlx_vlm").resolve()
    if not origin.is_relative_to(package_root):
        raise RuntimeError(
            f"mlx-vlm runtime check failed: imported {origin}, expected under {package_root}"
        )
    return {
        "schema": "mlx2.quality-campaign-mlx-vlm-runtime.v1",
        "root": str(MLX_VLM_ROOT),
        "revision": revision,
        "short_revision": revision[:8],
        "clean": True,
        "detached": True,
        "import_origin": str(origin),
    }


def server_args(model: Model, route: Route, phase: str, *, opt_in: bool = False) -> list[str]:
    if phase == "sanity":
        context, lanes, inflight, cache = min(model.max_context, 32768), 20, 40, model.cache_gib
    elif phase == "ladder":
        context, lanes, inflight, cache = model.max_context, 4, 8, model.ladder_cache_gib
    else:
        context, lanes, inflight, cache = min(model.max_context, 32768), 4, 8, model.cache_gib
    args = [
        "--model", model.path, "--host", "127.0.0.1", "--port", str(PORT),
        "--max-context", str(context), "--max-lanes", str(lanes),
        "--max-inflight", str(inflight), "--cache-bytes", str(cache << 30),
        "--qualification-mode", *model.extra_flags, *route.flags,
    ]
    selected = route.opt_in_policy if opt_in else route.policy
    if selected:
        args += ["--execution-policy", str(policy_path(selected))]
    return args


def validate_server_arguments(args: list[str], *, parser=None):
    """Parse and validate one generated command without resolving its model."""
    from mlx2.server import (
        approximate_kv_mode,
        build_parser,
        native_mtp_mode,
        request_body_limit,
        serving_engine_kwargs,
    )
    from mlx2.serving import ServingEngine

    parsed = (parser or build_parser()).parse_args(args)
    policy = (
        json.loads(parsed.execution_policy.read_text())
        if parsed.execution_policy
        else None
    )
    native_mtp = native_mtp_mode(parsed, policy)
    approximate_kv = approximate_kv_mode(parsed, native_mtp)
    if not parsed.qualification_mode and not parsed.qualification:
        raise ValueError("provide --qualification or explicitly run --qualification-mode")
    kwargs = serving_engine_kwargs(
        parsed,
        policy,
        native_mtp=native_mtp,
        approximate_kv=approximate_kv,
        max_request_bytes=request_body_limit(
            parsed.max_context, parsed.max_request_bytes
        ),
    )
    ServingEngine.validate_arguments(parsed.model, **kwargs)
    return parsed


def _validate_policy(model: Model, route: Route, path: Path) -> dict:
    from mlx2.runtime.pld import PromptLookupBatchGenerator
    from mlx2.runtime.speculative_sampling import FLyVerificationPolicy
    from mlx2.serving import apc_interior_checkpoint_policy

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path}: policy must be an object")
    for name in ("constrained_tool_grammar", "tolerant_tool_markers"):
        if name in value and type(value[name]) is not bool:
            raise ValueError(f"{path}: {name} must be boolean")
    apc_interior_checkpoint_policy(value.get("apc_interior_checkpoints"))
    FLyVerificationPolicy.from_value(value.get("fly_verification"))
    if "prompt_lookup" in value:
        PromptLookupBatchGenerator.validate_policy(value["prompt_lookup"])
    adapter_value = {
        key: item for key, item in value.items()
        if key not in {"prompt_lookup", "fly_verification", "apc_interior_checkpoints", "constrained_tool_grammar", "tolerant_tool_markers"}
    }
    if model.name == "flash-next":
        from mlx2.adapters.flash_next_policy import FlashNextPolicy
        FlashNextPolicy.from_mapping(adapter_value or None)
    elif model.name in {"qwen36", "qwen38"}:
        if set(adapter_value) - {"num_draft"}:
            raise ValueError(f"{path}: unsupported Qwen policy fields")
        if adapter_value and (type(adapter_value.get("num_draft")) is not int or not 1 <= adapter_value["num_draft"] <= 3):
            raise ValueError(f"{path}: num_draft must be 1, 2, or 3")
    elif model.name == "xing":
        if set(adapter_value) - {"num_draft", "tokenizer_reference_fallback"}:
            raise ValueError(f"{path}: unsupported Xing policy fields")
        if "num_draft" in adapter_value and (type(adapter_value["num_draft"]) is not int or not 1 <= adapter_value["num_draft"] <= 3):
            raise ValueError(f"{path}: num_draft must be 1, 2, or 3")
    elif model.name == "muse":
        if set(adapter_value) - {"draft_model", "num_draft"}:
            raise ValueError(f"{path}: unsupported Muse policy fields")
        if adapter_value:
            from mlx2.adapters.dflash2 import inspect_drafter
            record = inspect_drafter(adapter_value["draft_model"], model.path)
            count = adapter_value.get("num_draft", 4)
            if type(count) is not int or not 1 <= count < record["args"].block_size:
                raise ValueError(f"{path}: invalid Muse num_draft")
    elif adapter_value:
        raise ValueError(f"{path}: {model.label} has no adapter policy overrides")
    return value


def _load_tokenizer_only(model: Model) -> dict:
    if model.name == "xing":
        from mlx2.adapters.xing_tokenizer import load_tokenizer
        tokenizer, receipt = load_tokenizer(model.path)
        return {"class": type(tokenizer).__name__, "vocab": len(tokenizer), **receipt}
    from transformers import AutoProcessor, AutoTokenizer
    if model.name == "gemma3n":
        processor = AutoProcessor.from_pretrained(model.path, local_files_only=True, trust_remote_code=True)
        tokenizer = processor.tokenizer
        return {"class": type(processor).__name__, "vocab": len(tokenizer)}
    tokenizer = AutoTokenizer.from_pretrained(
        model.path, local_files_only=True,
        trust_remote_code=model.name == "minicpmo",
        fix_mistral_regex=True,
    )
    return {"class": type(tokenizer).__name__, "vocab": len(tokenizer)}


def validate_cpu_preflight(models: Iterable[Model] = MODELS) -> dict:
    """Validate paths, registry resolution, tokenizers, North binding and CLI.

    This is the only preflight entry point that imports MLX.  It pins the
    default device to CPU before importing registry/adapter code and never
    constructs an adapter, so no weight shard is loaded.
    """
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx2.adapters.registry import inspect_model
    from mlx2.server import build_parser

    report = {"device": "cpu", "weights_loaded": False, "models": {}, "policies": {}}
    parser = build_parser()
    for model in models:
        path = Path(model.path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        resolved = inspect_model(path)
        token = _load_tokenizer_only(model)
        report["models"][model.name] = {
            "path": str(path), "adapter": resolved.adapter_type.__name__,
            "family": resolved.descriptor.family, "tokenizer": token,
        }
        for route in model.routes:
            for opt_in in (False, True):
                args = server_args(model, route, "smoke", opt_in=opt_in)
                parsed = validate_server_arguments(args, parser=parser)
                if parsed.execution_policy:
                    key = parsed.execution_policy.name
                    report["policies"][key] = _validate_policy(model, route, parsed.execution_policy)
    north = next(model for model in models if model.name == "north")
    from mlx2.thinking_calibration import artifact_identity, load_bound_direction
    north_path = Path(north.path)
    north_config = json.loads((north_path / "config.json").read_text())
    asset = ROOT / "src/mlx2/adapters/assets/north_mini_code_commit_direction.npz"
    identity = artifact_identity(north_path)
    direction = load_bound_direction(
        identity, [asset], hidden_size=north_config["hidden_size"],
        num_layers=north_config["num_hidden_layers"], layer=28,
    )
    if direction is None:
        raise RuntimeError("North calibrated commit direction is not bound to this artifact")
    report["north_commit_direction"] = {
        "artifact_identity": identity, "source": direction["source"], "layer": direction["layer"],
        "auto_calibration_disabled": True,
    }
    return report


def stage_names(phase: str) -> list[str]:
    if phase == "ladder":
        return [f"ladder-{model.name}-{model.default_route}" for model in MODELS]
    return [f"{phase}-{model.name}-{route.name}" for model in MODELS for route in model.routes]
