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
PYTHON = Path(os.environ.get("MLX2_SERIES_PYTHON", str(Path.home()) + "/Desktop/mlx2/.venv/bin/python"))
SDK_PYTHON = Path(os.environ.get("MLX2_SERIES_SDK_PYTHON", str(Path.home()) + "/evalplus-venv/bin/python"))
PORT = 8297
BASE_URL = f"http://127.0.0.1:{PORT}"
# Every mlx-vlm family runs the revision mlx2 pins (adapters/mlx_vlm_pin.py):
# Blaizzy main after 0.7.3.  Gemma 4, Gemma 3n and MiniCPM-o refuse any other
# revision, so a per-family override would fail closed at adapter load.
MLX_VLM_ROOT = Path("/private/tmp/mlx-vlm-67599f2e").resolve()
MLX_VLM_REVISION = "67599f2e8ec31bf35cbb7b02794114f20844f0bb"
VLM_RUNTIMES = {}


def vlm_runtime(model=None):
    fam = FAMILY_OF.get(model.name, model.name) if model is not None else None
    return VLM_RUNTIMES.get(fam, (MLX_VLM_ROOT, MLX_VLM_REVISION))


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
        str(Path.home()) + "/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp",
        262144,
        (
            Route("ordinary", ("--ordinary",), "qwen36-ordinary.json", "qwen36-ordinary-opt-in.json", apc_interior=True),
            Route("mtp2", (), "qwen36-mtp2.json", "qwen36-mtp2-opt-in.json", True, True),
            Route("prompt-lookup", ("--prompt-lookup",), "prompt-lookup.json", "qwen36-pld-opt-in.json", True),
        ),
        "mtp2", TEXT, 8, 64,
    ),
    Model(
        "qwen38", "Qwen3.8-27B", str(Path.home()) + "/mlx-models/Qwen3.8-27B-oQ4e-mtp", 262144,
        (
            Route("ordinary", ("--ordinary",), None, "qwen38-ordinary-opt-in.json", apc_interior=True),
            Route("mtp2", (), "qwen38-mtp2.json", "qwen38-mtp2-opt-in.json", True, True),
            Route("prompt-lookup", ("--prompt-lookup",), "prompt-lookup.json", "prompt-lookup.json", True),
        ),
        "mtp2", TEXT, 8, 64,
    ),
    Model(
        "flash-next", "Qwen3.8 Flash-Next", str(Path.home()) + "/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP", 262144,
        (
            Route("ordinary", ("--ordinary",), "flash-next-mtp2.json", "flash-next-ordinary-opt-in.json", apc_interior=True),
            Route("mtp2", (), "flash-next-mtp2.json", "flash-next-mtp2-opt-in.json", True, True),
        ),
        "mtp2", TEXT, 16, 24,
    ),
    Model(
        "muse", "Muse-Glimmer-30B", str(Path.home()) + "/mlx-models/Muse-Glimmer-30B-mlx-4bit", 131072,
        (
            Route("ordinary", ("--ordinary",), None, "muse-ordinary-opt-in.json"),
            Route("dflash2", ("--external-draft",), "muse-dflash2.json", "muse-dflash2-opt-in.json", True),
            Route("prompt-lookup", ("--prompt-lookup",), "prompt-lookup.json", "muse-pld-opt-in.json", True),
        ),
        "ordinary", MUSE_TEXT, 8, 64,
    ),
    Model(
        "north", "North-Mini-Code", str(Path.home()) + "/mlx-models/North-Mini-Code-1.0-mlx-4bit", 500000,
        (
            Route("ordinary", ("--ordinary",), None, "north-ordinary-opt-in.json"),
            Route("prompt-lookup", ("--prompt-lookup",), "prompt-lookup.json", "north-pld-opt-in.json", True),
        ),
        "ordinary", TEXT, 16, 64, extra_flags=("--no-thinking-auto-calibration",),
    ),
    Model(
        "laguna", "Laguna XS 2.1",
        str(Path.home()) + "/.cache/huggingface/hub/models--AtomicChat--Laguna-XS-2.1-MLX-8bit/snapshots/ba635f7386219675b9b71acdc2812d12fa69aab2",
        262144, (Route("ordinary", ("--ordinary",), None, "laguna-ordinary-opt-in.json"),),
        "ordinary", TEXT, 8, 64,
    ),
    Model(
        "xing", "Xing4.0-29B-A4B", str(Path.home()) + "/mlx-models/Xing4.0-29B-A4B-mlx-6bit", 262144,
        (
            Route("ordinary", ("--ordinary",), None, "xing-ordinary-opt-in.json"),
            Route("mtp1", (), "xing-mtp1.json", "xing-mtp1-opt-in.json", True),
            Route("prompt-lookup", ("--prompt-lookup",), "prompt-lookup.json", "xing-pld-opt-in.json", True),
        ),
        "mtp1", TEXT, 8, 64,
    ),
    Model(
        "gemma3n", "Gemma 3n E2B",
        str(Path.home()) + "/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/5e092ebca197cdcd8d8b195040accf22693501bc",
        32768, (Route("ordinary", ("--ordinary",), None, "gemma3n-ordinary-opt-in.json"),),
        "ordinary", MM_GEMMA, 8, 16, ("image", "audio"),
    ),
    Model(
        "minicpmo", "MiniCPM-o 2.6",
        str(Path.home()) + "/.cache/huggingface/hub/models--openbmb--MiniCPM-o-2_6/snapshots/06849bfd36da94da1f5a88fa17e8ba63f08a4c4d",
        32768, (Route("ordinary", ("--ordinary",), None, "minicpmo-ordinary-opt-in.json"),),
        "ordinary", MM_MINICPM, 8, 16, ("image", "audio"),
    ),
)


# --- series-20260924: every supported artifact, quants included ------------
# Variants share their family's routes and policies (policies are not bound
# to an artifact).  An artifact without an MTP head drops the MTP routes.  The
# ladder cache is sized so weights + cache stay under ~105 GB on the 128 GB
# host (the rest is the OS and the server's working set).
_BY_NAME = {model.name: model for model in MODELS}
# variant name -> family name; family-specific checks key on the family.
FAMILY_OF: dict[str, str] = {}


def family(model) -> str:
    return FAMILY_OF.get(model.name, model.name)


def _ladder_cache(weights_gb: int) -> int:
    return max(8, min(64, 105 - weights_gb))


def _variant(family, name, label, path, *, mtp, weights_gb, max_context=None, ladder_cache=None):
    base = _BY_NAME[family]
    FAMILY_OF[name] = family
    routes = tuple(r for r in base.routes if mtp or not r.name.startswith("mtp"))
    default = base.default_route if any(r.name == base.default_route for r in routes) else "ordinary"
    return Model(name, label, path, max_context or base.max_context, routes, default,
                 base.capabilities, base.cache_gib, ladder_cache or _ladder_cache(weights_gb), base.media, base.extra_flags)


_M = str(Path.home()) + "/mlx-models/"
_NEMOTRON = str(Path.home()) + "/Desktop/mlx-uag/models/Nemotron-3-Super-120B-A12B-5bit-MTP"
SERIES_VARIANTS = (
    _variant("qwen36", "qwen36-heretic-4bit", "Qwen3.6-35B-A3B Heretic 4-bit", _M + "Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit", mtp=False, weights_gb=22),
    _variant("qwen36", "qwen36-ud-q8", "Qwen3.6-35B-A3B UD-Q8_K_XL", _M + "Qwen3.6-35B-A3B-UD-Q8_K_XL-mlx", mtp=False, weights_gb=35),
    _variant("qwen38", "qwen38-uncensored-oq4e", "Qwen3.8-27B Uncensored oQ4e MTP", _M + "Qwen3.8-27B-Uncensored-oQ4e-fp16-mtp", mtp=True, weights_gb=16),
    _variant("qwen38", "thinkingcap-27b", "ThinkingCap Qwen3.8-27B bf16", _M + "ThinkingCap-Qwen3.8-27B", mtp=True, weights_gb=51),
    _variant("qwen38", "qwen38-mlx-4bit", "Qwen3.8-27B MLX 4-bit", _M + "Qwen3.8-27B-MLX-4bit", mtp=False, weights_gb=14),
    _variant("qwen38", "qwen38-mlx-6bit", "Qwen3.8-27B MLX 6-bit", _M + "Qwen3.8-27B-MLX-6bit", mtp=False, weights_gb=21),
    _variant("qwen38", "qwen38-mlx-8bit", "Qwen3.8-27B MLX 8-bit", _M + "Qwen3.8-27B-MLX-8bit", mtp=False, weights_gb=27),
    _variant("qwen38", "qwen38-crack-4bit", "Qwen3.8-27B CRACK 4-bit", _M + "Qwen3.8-27B-CRACK-MLX-4bit", mtp=False, weights_gb=16),
    _variant("qwen38", "qwen38-crack-8bit", "Qwen3.8-27B CRACK 8-bit", _M + "Qwen3.8-27B-CRACK-MLX-8bit", mtp=False, weights_gb=28),
    _variant("qwen38", "qwen38-crack-bf16", "Qwen3.8-27B CRACK bf16", _M + "Qwen3.8-27B-CRACK-MLX-bf16", mtp=False, weights_gb=50),
    _variant("qwen38", "qwen36-27b-heretic-4bit", "Qwen3.6-27B Heretic 4-bit", _M + "Qwen3.6-27B-Abliterated-Heretic-Uncensored-MLX-4bit", mtp=False, weights_gb=17),
    _variant("qwen38", "qwen36-27b-8bit", "Qwen3.6-27B MLX 8-bit", _M + "Qwen3.6-27B-MLX-8bit", mtp=False, weights_gb=27),
    _variant("flash-next", "flash-next-uncensored", "Flash-Next Uncensored MLX2 4-bit MTP", _M + "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP", mtp=True, weights_gb=99, ladder_cache=24),  # PLE rows stream from disk
    _variant("muse", "muse-8bit", "Muse-Glimmer-30B 8-bit", _M + "Muse-Glimmer-30B-mlx-8bit", mtp=False, weights_gb=32),
    _variant("muse", "muse-bf16", "Muse-Glimmer-30B MLX bf16", _M + "Muse-Glimmer-30B-mlx-bf16", mtp=False, weights_gb=55),
    _variant("muse", "muse-original", "Muse-Glimmer-30B (original bf16)", _M + "Muse-Glimmer-30B", mtp=False, weights_gb=55),
    _variant("muse", "muse-cyber-4bit", "Muse-Glimmer-30B cyber-clone 4-bit", _M + "Muse-Glimmer-30B-cyber-clone-4bit", mtp=False, weights_gb=19),
    _variant("muse", "muse-cyber-bf16", "Muse-Glimmer-30B cyber-clone bf16", _M + "Muse-Glimmer-30B-cyber-clone-bf16", mtp=False, weights_gb=55),
    _variant("north", "north-8bit", "North-Mini-Code 8-bit", _M + "North-Mini-Code-1.0-mlx-8bit", mtp=False, weights_gb=30),
    _variant("xing", "xing-bf16", "Xing4.0-29B-A4B bf16", _M + "Xing4.0-29B-A4B-mlx-bf16", mtp=True, weights_gb=56),
    Model(
        "nemotron", "Nemotron 3 Super 120B-A12B 5-bit", _NEMOTRON, 262144,
        (
            Route("ordinary", ("--ordinary",), None, None),
            Route("mtp3", ("--native-mtp",), None, None, True),
        ),
        "ordinary", TEXT, 8, _ladder_cache(79),
    ),
)
MM_GEMMA4 = frozenset({"text", "streaming", "vision", "apc"})
GEMMA4 = tuple(
    Model(name, label, _M + artifact, 262144, (Route("ordinary", ("--ordinary",), None, None),),
          "ordinary", MM_GEMMA4, 8, _ladder_cache(weights), ("image",))
    for name, label, artifact, weights in (
        ("gemma4-26b-q8", "Gemma 4 26B-A4B MLX 8-bit", "gemma-4-26B-A4B-MLX-8bit", 26),
        ("gemma4-26b-bf16", "Gemma 4 26B-A4B bf16", "gemma-4-26B-A4B", 48),
        ("gemma4-31b-q8", "Gemma 4 31B MLX 8-bit", "gemma-4-31B-MLX-8bit", 31),
        ("gemma4-31b-bf16", "Gemma 4 31B bf16", "gemma-4-31B", 58),
    )
)
FAMILY_OF.update({model.name: "gemma4" for model in GEMMA4})
MODELS = MODELS + SERIES_VARIANTS + GEMMA4
# Artifacts that are gone from disk are reported, not tested.
MISSING = tuple(model for model in MODELS if not Path(model.path).is_dir())
MODELS = tuple(model for model in MODELS if Path(model.path).is_dir())

def policy_path(name: str | None) -> Path | None:
    return RUN / "policies" / name if name else None


def requires_mlx_vlm(model: Model) -> bool:
    return bool(model.media)


def stage_pythonpath(model: Model) -> str:
    if requires_mlx_vlm(model):
        return os.pathsep.join((str(vlm_runtime(model)[0]), "src"))
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


def verify_mlx_vlm_runtime(*, env=None, model=None) -> dict:
    """Verify the exact detached optional runtime without loading model weights."""
    vlm_root, vlm_revision = vlm_runtime(model)
    if not vlm_root.is_dir():
        raise RuntimeError(f"mlx-vlm runtime check failed: missing {vlm_root}")
    top = Path(
        _command_output(
            ["git", "-C", vlm_root, "rev-parse", "--show-toplevel"]
        ).stdout.strip()
    ).resolve()
    if top != vlm_root:
        raise RuntimeError(
            f"mlx-vlm runtime check failed: checkout root is {top}, expected {vlm_root}"
        )
    revision = _command_output(
        ["git", "-C", vlm_root, "rev-parse", "HEAD"]
    ).stdout.strip()
    if revision != vlm_revision:
        raise RuntimeError(
            "mlx-vlm runtime check failed: revision "
            f"{revision or '<missing>'} != {vlm_revision}"
        )
    dirty = _command_output(
        ["git", "-C", vlm_root, "status", "--porcelain=v1", "--untracked-files=all"]
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"mlx-vlm runtime check failed: checkout is dirty: {dirty}")
    symbolic = _command_output(
        ["git", "-C", vlm_root, "symbolic-ref", "-q", "HEAD"],
        allow_failure=True,
    )
    if symbolic.returncode == 0:
        raise RuntimeError(
            "mlx-vlm runtime check failed: checkout is not detached: "
            f"{symbolic.stdout.strip()}"
        )
    import_env = dict(os.environ if env is None else env)
    import_env["PYTHONPATH"] = os.pathsep.join((str(vlm_root), "src"))
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
    package_root = (vlm_root / "mlx_vlm").resolve()
    if not origin.is_relative_to(package_root):
        raise RuntimeError(
            f"mlx-vlm runtime check failed: imported {origin}, expected under {package_root}"
        )
    return {
        "schema": "mlx2.quality-campaign-mlx-vlm-runtime.v1",
        "root": str(vlm_root),
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
    # A small host (the M3, 36 GB) caps the prefix cache so weights + cache
    # leave headroom; the orchestrator sets the cap for m3 jobs.
    cap = int(os.environ.get("MLX2_SERIES_CACHE_GIB_CAP", "0") or 0)
    if cap:
        cache = min(cache, cap)
    # Two private 32K copies of a warm prefix do not fit a 36 GB host
    # (admission correctly answers 429); m3 smoke runs at a smaller context.
    context_cap = int(os.environ.get("MLX2_SERIES_CONTEXT_CAP", "0") or 0)
    if context_cap:
        context = min(context, context_cap)
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
    if family(model) == "flash-next":
        from mlx2.adapters.flash_next_policy import FlashNextPolicy
        FlashNextPolicy.from_mapping(adapter_value or None)
    elif family(model) in {"qwen36", "qwen38"}:
        if set(adapter_value) - {"num_draft"}:
            raise ValueError(f"{path}: unsupported Qwen policy fields")
        if adapter_value and (type(adapter_value.get("num_draft")) is not int or not 1 <= adapter_value["num_draft"] <= 3):
            raise ValueError(f"{path}: num_draft must be 1, 2, or 3")
    elif family(model) == "xing":
        if set(adapter_value) - {"num_draft", "tokenizer_reference_fallback"}:
            raise ValueError(f"{path}: unsupported Xing policy fields")
        if "num_draft" in adapter_value and (type(adapter_value["num_draft"]) is not int or not 1 <= adapter_value["num_draft"] <= 3):
            raise ValueError(f"{path}: num_draft must be 1, 2, or 3")
    elif family(model) == "muse":
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
    if family(model) == "xing":
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
    if not any(model.name == "north" for model in models):
        return report
    # Since 2026-09-23 North ships no commit direction and steering is off by
    # default: on the corrected layer-0 RoPE body the recalibrated direction
    # lengthened reasoning (qualification/runs/north-rope-l0-20260923).  With
    # auto-calibration disabled the server must therefore serve guard-only.
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
    adapter = object.__new__(NorthMiniCodeAdapter)
    if adapter.thinking_guard_defaults().get("thinking_steer_alpha") != 0.0:
        raise RuntimeError("North steering default changed; revalidate the campaign's guard-only premise")
    if adapter.commit_direction_assets()["paths"]:
        raise RuntimeError("North ships a commit direction again; bind-check it here")
    report["north_commit_direction"] = {
        "shipped": False, "steering_default_alpha": 0.0, "auto_calibration_disabled": True,
    }
    return report


def stage_names(phase: str) -> list[str]:
    if phase == "ladder":
        return [f"ladder-{model.name}-{model.default_route}" for model in MODELS]
    return [f"{phase}-{model.name}-{route.name}" for model in MODELS for route in model.routes]
