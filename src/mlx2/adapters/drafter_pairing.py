"""GPU-free discovery of on-disk drafter artifacts and target pairing.

Shared by the model-matrix inventory (``scripts/model_matrix/inventory.py``)
and the server's per-artifact best-route defaults
(``mlx2/best_route_defaults.py``).  Discovery walks the local model roots for
``config.json`` and classifies each artifact by its declared architecture;
pairing asks the *target adapter's own* drafter inspector whether a drafter
binds (geometry, headers, payload-bound content revision), never a name
match.  The per-adapter inspector table lives here, in the adapter layer,
not in the scheduler, cache or request lifecycle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_ROOTS = (
    Path.home() / "mlx-models",
    Path("/Volumes/T7/models"),
    Path.home() / "models",
)
HF_HUB = Path.home() / ".cache/huggingface/hub"

# Non-text modalities and pipeline parts.  These are a separate lane.
MEDIA_MODEL_TYPES = frozenset({
    "llada", "diffusion_gemma", "AudioVideo", "t5", "clip_text_model",
    "whisper", "moonshine", "moonshine_streaming", "qwen3_asr", "qwen3_tts",
    "qwen3_tts_tokenizer_12hz", "nemotron_asr", "nemotron_asr_streaming",
    "melotts", "pocket_tts", "silero_vad", "sam", "sam_audio", "stt", "tts",
    "voxtral_realtime", "chatterbox_turbo", "wav2vec2-bert",
    "nemotron3_diarization", "style_text_to_speech_2",
})
PIPELINE_COMPONENTS = frozenset({"text_encoder", "text_encoder_2", "transformer", "vae"})
DRAFT_MODEL_TYPES = frozenset({
    "qwen3_5_mtp", "qwen4_exp_mtp", "muse_glimmer_assistant", "qwen3_5_text",
})
DRAFT_ARCH_MARKERS = ("Draft", "XPress", "DSpark", "Eagle", "LiLiCorr")


def shard_names(path: Path) -> tuple[list[str], bool]:
    """Weight shard names and whether every one is present."""
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            mapping = json.loads(index.read_text())["weight_map"]
            names = sorted(set(mapping.values()))
        except Exception:  # noqa: BLE001 - a broken index is an incomplete artifact
            return [], False
        return names, bool(names) and all((path / name).is_file() for name in names)
    names = sorted(p.name for p in path.glob("*.safetensors"))
    return names, bool(names) and all((path / name).is_file() for name in names)


def discover(roots, include_hf: bool) -> list[Path]:
    """Every directory holding a ``config.json`` under the roots (depth < 4)."""
    found = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for directory, dirs, files in os.walk(root, followlinks=True):
            depth = len(Path(directory).relative_to(root).parts)
            dirs[:] = [d for d in dirs if not d.startswith((".", "_"))] if depth < 4 else []
            if "config.json" in files:
                found.append(Path(directory))
            elif files and set(files) - {".DS_Store"} == {"mtp.safetensors"}:
                # A bare MTP head (no config): a drafter, never a target.
                found.append(Path(directory))
    if include_hf and HF_HUB.is_dir():
        for snapshot in sorted(HF_HUB.glob("models--*/snapshots/*")):
            for directory, dirs, files in os.walk(snapshot, followlinks=True):
                if "config.json" in files:
                    found.append(Path(directory))
    return found


def display_name(path: Path) -> str:
    parts = Path(path).parts
    if "snapshots" in parts:
        repo = parts[parts.index("snapshots") - 1]
        return repo.removeprefix("models--").split("--", 1)[-1]
    return Path(path).name


def classify(path: Path, config: dict) -> str | None:
    """``drafter``, a skipped media lane, or ``None`` for a target candidate."""
    if PIPELINE_COMPONENTS & set(Path(path).parts[-1:]):
        return "media_pipeline_component"
    model_type = config.get("model_type")
    if model_type in MEDIA_MODEL_TYPES:
        return "media_or_non_text_model"
    if not model_type and not config.get("architectures"):
        return "media_or_non_text_model"
    archs = config.get("architectures") or []
    if (
        "dflash_config" in config
        or model_type in DRAFT_MODEL_TYPES
        or any(marker in name for name in archs for marker in DRAFT_ARCH_MARKERS)
    ):
        return "drafter"
    return None


def drafter_row(path: Path) -> dict | None:
    """The inventory row of a complete drafter artifact, else ``None``."""
    path = Path(path)
    try:
        config = json.loads((path / "config.json").read_text())
    except Exception:  # noqa: BLE001 - unreadable configs are not drafters
        return None
    if not isinstance(config, dict) or classify(path, config) != "drafter":
        return None
    names, complete = shard_names(path)
    if not complete:
        return None
    return {
        "path": str(path),
        "name": display_name(path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures") or [],
        "weight_bytes": sum((path / name).stat().st_size for name in names),
    }


def discover_drafters(roots=None, *, include_hf: bool = True) -> list[dict]:
    """Complete drafter artifacts under the roots, de-duplicated by real path."""
    rows, seen = [], set()
    for path in discover(DEFAULT_ROOTS if roots is None else roots, include_hf):
        try:
            real = os.path.realpath(path)
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        row = drafter_row(path)
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda row: row["path"])


def pair_drafter(
    target: dict,
    draft: dict,
    cache: dict | None = None,
    *,
    target_revision: bool = True,
) -> dict:
    """Ask the target adapter's own inspector whether ``draft`` binds.

    ``target`` needs ``adapter`` (class name) and ``path``; ``draft`` needs
    ``path``, ``architectures`` and ``name``.  Raises ``ValueError`` (or the
    inspector's own error) when the pair is refused.  ``target_revision=False``
    skips the Qwen hybrid's payload-bound target hash; the adapter computes
    and checks it at load anyway.  Returns the pairing: identity
    ``draft_revision``/``identity``, the explicit-route policy and the
    identity pins (``pins``) an adapter policy accepts.
    """
    cache = {} if cache is None else cache
    adapter = target["adapter"]
    tpath, dpath = target["path"], draft["path"]
    archs = draft["architectures"]
    policy: dict = {"draft_model": dpath}
    pins: dict = {}
    record = None
    if adapter in {"Qwen3827BAdapter", "Qwen3627BAdapter", "Qwen3635BA3BAdapter",
                   "Qwen359BAdapter"}:
        if adapter == "Qwen359BAdapter":
            raise ValueError("Qwen3.5-9B adapter refuses external drafts")
        from . import qwen38_27b
        from .external_drafter_formats import drafter_format

        # The adapter's own dispatch: DFlash2 or DFlash-format DSpark by the
        # artifact's declared architecture (anything else fails DFlash2's
        # inspector with its precise reason).
        draft_format = drafter_format(dpath)
        record = draft_format.inspect_drafter(dpath, tpath)
        pins["draft_revision"] = draft_format.content_revision(record)
        if target_revision:
            key = ("qwen38_target", tpath)
            if key not in cache:
                cache[key] = qwen38_27b.content_revision(tpath)
            pins["target_revision"] = cache[key]
        policy.update(pins)
    elif adapter == "FlashNextAdapter":
        if archs != ["DFlash2DraftModel"]:
            raise ValueError(
                "Flash-Next external route accepts DFlash2DraftModel or its own MTP head; "
                "FlashNextAdapter attaches no separately shipped head"
            )
        from . import dflash2, flash_next

        record = dflash2.inspect_drafter(dpath, tpath)
        pins.update(
            draft_revision=dflash2.content_revision(record),
            target_revision=flash_next.artifact_identity(Path(tpath))["fingerprint"],
        )
        policy.update(pins, num_draft=3, row_exact_verify=True, progressive_verification_tile=2)
    elif adapter == "MuseGlimmerAdapter" and archs == ["MuseGlimmerAssistantModel"]:
        from . import muse_dflash

        record = muse_dflash.inspect_drafter(dpath, tpath)
        policy["num_draft"] = 4
    elif adapter == "MuseGlimmerAdapter":
        if archs != ["DFlash2DraftModel"]:
            raise ValueError(
                "Muse Glimmer external route accepts only DFlash2DraftModel "
                f"(artifact declares {archs})"
            )
        from . import dflash2

        record = dflash2.inspect_drafter(dpath, tpath)
        policy["num_draft"] = 4
    elif adapter == "NorthMiniCodeAdapter":
        from . import cohere_eagle

        record = cohere_eagle.inspect_drafter(dpath, target_path=tpath)
    elif adapter == "LagunaXS21Adapter":
        from . import laguna_dflash

        record = laguna_dflash.inspect_drafter(dpath, target_path=tpath)
    elif adapter in {"Nemotron35SuperAdapter", "Nemotron35SuperVLAdapter", "Nemotron35BoltAdapter"}:
        if archs == ["DSparkDraftModel"]:
            if adapter != "Nemotron35SuperAdapter":
                raise ValueError(f"{adapter} accepts only LiLiCoRR drafters")
            from . import nemotron35_dspark

            record = nemotron35_dspark.inspect_drafter(dpath, tpath)
        elif archs == ["LiLiCorrDraftModel"] and adapter == "Nemotron35BoltAdapter":
            from . import nemotron35_bolt
            from .registry import inspect_model

            record = nemotron35_bolt.inspect_lilicorr_drafter(
                dpath, target_artifact=inspect_model(tpath).artifact)
        elif archs == ["LiLiCorrDraftModel"]:
            from . import lilicorr_nemotron

            record = lilicorr_nemotron.inspect_drafter(dpath, target_path=tpath)
        else:
            raise ValueError(f"{adapter} accepts DSpark/LiLiCoRR drafters only")
    elif adapter == "StandardDecoderAdapter":
        if archs == ["Qwen3XPressModel"]:
            from . import xpress as module
        elif archs == ["LiLiCorrDraftModel"]:
            from . import lilicorr as module
        elif archs == ["Eagle3Speculator"]:
            from . import eagle3 as module
        else:
            raise ValueError("standard decoder accepts XPress/LiLiCoRR/EAGLE-3 drafters only")
        record = module.inspect_drafter(dpath, tpath)
        pins["draft_revision"] = module.content_revision(record)
        policy.update(pins)
    else:
        raise ValueError(f"{adapter} declares no external-draft route")
    identity = pins.get("draft_revision") or (record or {}).get("draft_revision") \
        or (record or {}).get("fingerprint")
    return {
        "draft": f"{draft['name']}@sha256:{identity}" if identity else draft.get("checkpoint"),
        "identity": identity,
        "draft_path": dpath,
        "draft_weight_bytes": draft.get("weight_bytes"),
        "draft_kind": archs[0] if archs else draft.get("model_type"),
        "route": "external_draft",
        "server_flags": ["--external-draft", "--execution-policy", "<policy.json>"],
        "execution_policy": policy,
        "pins": pins,
    }


def route_slug(draft: dict) -> str:
    kind = (draft["architectures"] or [draft.get("model_type") or "draft"])[0]
    kind = kind.replace("DraftModel", "").replace("ForCausalLM", "").replace("Model", "")
    name = draft["name"].lower().replace(".", "").replace("_", "-")
    return f"{kind.lower()}-{name}"[:80]
