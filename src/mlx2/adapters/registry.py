"""GPU-free artifact dispatch; model selection stays outside the scheduler.

Resolution validates the local artifact and requested implementation capability.
It does not qualify a route or load a model. Normal serving still requires an
artifact/runtime/settings-bound qualification receipt.
"""

from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane


@dataclass(frozen=True)
class AdapterResolution:
    adapter_type: type
    descriptor: ModelDescriptor
    artifact: dict

    @property
    def default_route(self) -> str:
        """Resolve the adapter preference against this artifact's capabilities."""
        route = getattr(self.adapter_type, "default_route", None)
        if route not in {"ordinary", "native_mtp"}:
            raise ValueError(
                f"{self.adapter_type.__name__} must declare default_route as "
                "'ordinary' or 'native_mtp'"
            )
        if route == "native_mtp" and Capability.MTP not in self.descriptor.capabilities:
            return "ordinary"
        return route

    @property
    def default_mtp_ordinary_handoff(self) -> dict | None:
        """Return the adapter-declared native-MTP handoff policy, if any."""
        if Capability.MTP not in self.descriptor.capabilities:
            return None
        width = getattr(
            self.adapter_type,
            "default_mtp_ordinary_handoff_max_width",
            None,
        )
        if width is None:
            return None
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError(
                f"{self.adapter_type.__name__} handoff width must be a "
                "positive integer"
            )
        return {"enabled": True, "max_mtp_width": width}

    def default_execution_policy(self, route: str) -> dict:
        """Return adapter-declared server-owned policy defaults for ``route``.

        Read from the adapter class's *own* namespace, never inherited: these
        are evidence-backed per-model performance defaults, and a subclass
        (Nemotron and Qwen3.5 9B subclass Flash-Next / Qwen3.8) must not
        acquire a measurement taken on its parent.  Only exact, server-owned
        mechanisms may be declared, and only for the two routes an adapter
        can default to; prompt-lookup and external-draft routes get nothing.
        """
        declared = vars(self.adapter_type).get("default_route_execution_policy")
        if declared is None:
            return {}
        name = self.adapter_type.__name__
        if not isinstance(declared, dict) or set(declared) - {
            "ordinary",
            "native_mtp",
        }:
            raise ValueError(
                f"{name} default_route_execution_policy must map 'ordinary' "
                "or 'native_mtp' to a policy object"
            )
        if route not in {"ordinary", "native_mtp"}:
            return {}
        if route == "native_mtp" and Capability.MTP not in self.descriptor.capabilities:
            return {}
        policy = declared.get(route) or {}
        if not isinstance(policy, dict):
            raise ValueError(f"{name} {route} default policy must be an object")
        unknown = set(policy) - ADAPTER_DEFAULT_POLICY_KEYS
        if unknown:
            raise ValueError(
                f"{name} declares non-default-able policy keys: {sorted(unknown)}"
            )
        return copy.deepcopy(policy)


# Server-owned execution-policy keys an adapter may default on.  The host
# signal changes admission timing, not token math; all keys fail closed at
# engine startup where their route or cache cannot support them.
ADAPTER_DEFAULT_POLICY_KEYS = frozenset(
    {
        "apc_interior_checkpoints",
        "apc_junction_checkpoints",
        "apc_rolling_checkpoints",
        "self_mtp_copy_draft",
        "prefill_scheduling",
        "host_memory_signals",
    }
)
# The subset that snapshots hybrid state; the engine refuses these on
# approximate-KV routes, so a default must not select them there.
STATE_CHECKPOINT_POLICY_KEYS = frozenset(
    {
        "apc_interior_checkpoints",
        "apc_junction_checkpoints",
        "apc_rolling_checkpoints",
    }
)


def _flash_next(path: Path, config: dict) -> AdapterResolution:
    if config.get("ngram_table") or not (path / "ple_rows.bin").is_file():
        raise ValueError(
            "Flash-Next requires the current unified artifact layout with ple_rows.bin"
        )
    module = importlib.import_module(".flash_next", __package__)
    weights = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Artifact must have a nonempty weight index")
    for name in weights.values():
        # A basename gate prevents traversal while keeping Hugging Face
        # snapshots, whose shards are links into the sibling blobs tree (the
        # rule standard_decoder already applies).
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Weight shard paths must stay within the artifact")
    has_mtp = any(name.startswith(("mtp.", "language_model.mtp.")) for name in weights)
    descriptor = module.FlashNextAdapter.descriptor
    if not has_mtp:
        descriptor = replace(
            descriptor,
            capabilities=descriptor.capabilities
            - {Capability.MTP, Capability.SEGMENTED_MTP},
            state_planes=descriptor.state_planes - {StatePlane.DRAFT},
        )
    artifact = module.artifact_identity(path)
    artifact["has_mtp"] = has_mtp
    return AdapterResolution(module.FlashNextAdapter, descriptor, artifact)


def _qwen3_5_dense(path: Path, config: dict) -> AdapterResolution:
    text = config.get("text_config", config)
    topology = (text.get("num_hidden_layers"), text.get("hidden_size"))
    if topology == (32, 2560):
        module = importlib.import_module(".qwen35_4b", __package__)
        artifact = module.inspect_artifact(path)
        return AdapterResolution(
            module.Qwen354BAdapter,
            module.descriptor_for(has_mtp=False),
            artifact,
        )
    if topology == (32, 4096):
        module = importlib.import_module(".qwen35_9b", __package__)
        artifact = module.inspect_artifact(path)
        return AdapterResolution(
            module.Qwen359BAdapter,
            module.descriptor_for(has_mtp=False),
            artifact,
        )
    if topology != (64, 5120):
        raise ValueError(f"No mlx2 dense qwen3_5 adapter for topology {topology!r}")
    module = importlib.import_module(".qwen38_27b", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(
        module.Qwen3827BAdapter,
        module.descriptor_for(has_mtp=artifact["has_mtp"]),
        artifact,
    )


def _qwen36_35b(path: Path, config: dict) -> AdapterResolution:
    text = config.get("text_config", config)
    if (text.get("num_hidden_layers"), text.get("hidden_size")) == (48, 3072):
        module = importlib.import_module(".qwen35_122b", __package__)
        artifact = module.inspect_artifact(path)
        return AdapterResolution(
            module.Qwen35122BA10BAdapter, module.QWEN35_122B, artifact
        )
    module = importlib.import_module(".qwen36_35b", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(
        module.Qwen3635BA3BAdapter,
        module.descriptor_for(has_mtp=artifact["has_mtp"]),
        artifact,
    )


def _muse_glimmer(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".muse_glimmer", __package__)
    return AdapterResolution(
        module.MuseGlimmerAdapter, module.MUSE_GLIMMER, module.inspect_artifact(path)
    )


def _north_mini_code(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".north_mini_code", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(
        module.NorthMiniCodeAdapter, module.NORTH_MINI_CODE, artifact
    )


def _laguna_xs21(path: Path, config: dict) -> AdapterResolution:
    if (config.get("num_hidden_layers"), config.get("hidden_size")) == (48, 3072):
        module = importlib.import_module(".laguna_s21", __package__)
        artifact = module.inspect_artifact(path)
        return AdapterResolution(
            module.LagunaS21Adapter, module.LAGUNA_S21, artifact
        )
    module = importlib.import_module(".laguna_xs21", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(
        module.LagunaXS21Adapter, module.LAGUNA_XS21, artifact
    )


def _xing4_0(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".xing", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(
        module.XingAdapter, module.descriptor_for(has_mtp=artifact["has_mtp"]), artifact
    )


def _nemotron_h(path: Path, config: dict) -> AdapterResolution:
    if (config.get("num_hidden_layers"), config.get("hidden_size")) == (52, 2688):
        module = importlib.import_module(".nemotron35_lightning", __package__)
        artifact = module.inspect_artifact(path)
        return AdapterResolution(
            module.Nemotron35LightningAdapter,
            module.descriptor_for(has_mtp=artifact["has_mtp"]),
            artifact,
        )
    module = importlib.import_module(".nemotron3_super", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(module.Nemotron3SuperAdapter, module.DESCRIPTOR, artifact)


def _gemma3n(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".mlx_vlm", __package__)
    artifact = module.inspect_artifact(path, expected="gemma3n")
    return AdapterResolution(module.Gemma3nAdapter, module.GEMMA3N, artifact)


def _minicpmo(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".mlx_vlm", __package__)
    artifact = module.inspect_artifact(path, expected="minicpmo")
    return AdapterResolution(module.MiniCPMOAdapter, module.MINICPMO, artifact)


def _gemma4(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".gemma4", __package__)
    artifact = module.inspect_gemma4_artifact(path)
    if artifact["variant"] == "26b-a4b":
        return AdapterResolution(module.Gemma4A4BAdapter, module.GEMMA4_A4B, artifact)
    return AdapterResolution(module.Gemma431BAdapter, module.GEMMA4_31B, artifact)


def _standard_decoder(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".standard_decoder", __package__)
    model_type = config["model_type"]
    artifact = module.inspect_artifact(path, expected=model_type)
    return AdapterResolution(
        module.StandardDecoderAdapter, module.descriptor_for(model_type), artifact
    )


def _agnes(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".agnes_3_flash", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(module.Agnes3FlashAdapter, module.DESCRIPTOR, artifact)


def _hy_v3(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".hy_v3", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(
        module.HYV3Adapter, module.descriptor_for(reap=artifact["reap"]), artifact
    )


def _gpt_oss(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".gpt_oss", __package__)
    artifact = module.inspect_artifact(path, expected=config["model_type"])
    if config["model_type"] == "gpt_oss_puzzle":
        return AdapterResolution(
            module.GptOssPuzzleAdapter, module.GPT_OSS_PUZZLE, artifact
        )
    return AdapterResolution(module.GptOssAdapter, module.GPT_OSS, artifact)


def _granite_swa(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".granite_swa", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(module.GraniteSWAAdapter, module.DESCRIPTOR, artifact)


def _lfm25_vl(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".lfm25_vl", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(module.LFM25VLAdapter, module.LFM25_VL, artifact)


def _smolvlm2(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".smolvlm2", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(module.SmolVLM2CandidateAdapter, module.DESCRIPTOR, artifact)


def _qwen25_vl(path: Path, config: dict) -> AdapterResolution:
    module = importlib.import_module(".qwen25_vl", __package__)
    artifact = module.inspect_artifact(path)
    return AdapterResolution(module.Qwen25VLCandidateAdapter, module.DESCRIPTOR, artifact)


_RESOLVERS: dict[str, Callable[[Path, dict], AdapterResolution]] = {
    "qwen4_exp": _flash_next,
    "qwen3_5": _qwen3_5_dense,
    "qwen3_5_moe": _qwen36_35b,
    "muse_glimmer": _muse_glimmer,
    "muse_glimmer_text": _muse_glimmer,
    "cohere2_moe": _north_mini_code,
    "laguna": _laguna_xs21,
    "xing4_0": _xing4_0,
    "nemotron_h": _nemotron_h,
    "gemma3n": _gemma3n,
    "gemma4": _gemma4,
    "minicpmo": _minicpmo,
    "qwen3": _standard_decoder,
    "qwen3_moe": _standard_decoder,
    "qwen2": _standard_decoder,
    "llama": _standard_decoder,
    "agnes": _agnes,
    "hy_v3": _hy_v3,
    "gpt_oss": _gpt_oss,
    "gpt_oss_puzzle": _gpt_oss,
    "granitemoe_swa": _granite_swa,
    "lfm2_vl": _lfm25_vl,
    "smolvlm": _smolvlm2,
    "qwen2_5_vl": _qwen25_vl,
}

# These source-backed multimodal bridges expose cache-safe candidate contracts,
# but have no model-path numerical qualification yet. Keep discovery available
# for qualification runs without silently selecting an unqualified server route.
_QUALIFICATION_GATED_TYPES = frozenset({"lfm2_vl", "smolvlm", "qwen2_5_vl"})


def inspect_model(model_path: str | Path) -> AdapterResolution:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict):
        raise TypeError("Model config must be a JSON object")
    if "dflash_config" in config or any(
        "Draft" in name for name in config.get("architectures", [])
    ):
        raise ValueError(
            "A speculative drafter cannot be served as a standalone target"
        )
    model_type = config.get("model_type")
    resolver = _RESOLVERS.get(model_type) if isinstance(model_type, str) else None
    if resolver is None:
        raise ValueError(f"No mlx2 adapter for model_type {model_type!r}")
    return resolver(path, config)


def inspect_audio_model(model_path: str | Path) -> AdapterResolution:
    """Resolve an audio classifier without admitting it to text serving."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") == "nemotron3_diarization":
        module = importlib.import_module(".nemotron3_diarization", __package__)
        return AdapterResolution(
            module.Nemotron3DiarizationAdapter,
            module.DIARIZATION_DESCRIPTOR,
            module.inspect_artifact(path),
        )
    raise ValueError(
        f"No mlx2 audio-classifier adapter for model_type {config.get('model_type')!r}"
    )


def resolve_adapter(
    model_path: str | Path, *, mtp: bool = False,
    qualification_mode: bool = False, qualification: str | Path | None = None,
) -> type:
    if type(mtp) is not bool:
        raise ValueError("mtp selection must be boolean")
    result = inspect_model(model_path)
    if mtp and Capability.MTP not in result.descriptor.capabilities:
        raise ValueError(
            f"{result.descriptor.family} artifact has no implemented native MTP route"
        )
    if (result.descriptor.model_type in _QUALIFICATION_GATED_TYPES
            and not qualification_mode and not qualification):
        raise ValueError(
            f"{result.descriptor.family} requires qualification mode or an "
            "artifact-bound qualification receipt before serving"
        )
    return result.adapter_type
