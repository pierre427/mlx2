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


# Server-owned execution-policy keys an adapter may default on.  Each is exact
# (it changes how fast tokens come, not which tokens) and each fails closed at
# engine startup where its route or cache cannot support it.
ADAPTER_DEFAULT_POLICY_KEYS = frozenset(
    {
        "apc_interior_checkpoints",
        "apc_junction_checkpoints",
        "apc_rolling_checkpoints",
        "self_mtp_copy_draft",
        "prefill_scheduling",
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
        if not isinstance(name, str) or not (path / name).resolve().is_relative_to(
            path
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
}


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


def resolve_adapter(model_path: str | Path, *, mtp: bool = False) -> type:
    if type(mtp) is not bool:
        raise ValueError("mtp selection must be boolean")
    result = inspect_model(model_path)
    if mtp and Capability.MTP not in result.descriptor.capabilities:
        raise ValueError(
            f"{result.descriptor.family} artifact has no implemented native MTP route"
        )
    return result.adapter_type
