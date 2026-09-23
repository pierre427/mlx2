"""Vendor-recommended sampling defaults, declared per adapter.

An adapter declares the sampling settings its model's vendor recommends as a
class attribute ``sampling_defaults = VendorSampling(...)``.  Defaults fill
only fields the request leaves unset: an explicit request value always wins,
including ``temperature: 0`` and a neutral penalty.  Nothing here is a model
name branch in the scheduler; the scheduler asks the adapter.

Profile selection (``VendorSampling.select``), highest precedence first:

1. ``sampling_profile`` in the request (an mlx2 extension field) names one of
   the adapter's profiles.  An unknown name fails the request closed.
2. When the request's thinking mode is known (a chat request; the adapter
   decides, see ``serving.thinking_enabled``) and the vendor gives a
   mode-specific profile, that profile is used.
3. Otherwise the vendor's general profile.

Fields that neither the request nor the selected profile set take the
neutral value (``NEUTRAL``: top_p 1, top_k and min_p disabled, penalties
off), matching vLLM, which applies only the vendor fields it is given.  An
adapter that declares no vendor defaults keeps mlx2's historical engine
fallback (``LEGACY_FALLBACK``) so existing behaviour is unchanged for it.

Penalty semantics (see ``runtime.sample_utils``):
``repetition_penalty`` is multiplicative (HF/CTRL, sign-aware) over prompt and
generated tokens, as vLLM and HF transformers apply it; ``presence_penalty``
and ``frequency_penalty`` are additive over generated tokens only, as vLLM
and the OpenAI API define them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional


SAMPLING_DEFAULTS_SCHEMA = "mlx2.sampling-defaults.v1"

# Request fields a vendor default may fill, in receipt order.
SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)

# Values used for a field neither the request nor the vendor profile sets.
NEUTRAL = MappingProxyType(
    {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
)

# mlx2's pre-vendor-defaults engine fallback, kept for adapters (and test
# doubles) that declare no vendor sampling.
LEGACY_FALLBACK = MappingProxyType(
    {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
)

# Positive temperatures below this sample greedily, as vLLM's
# ``_SAMPLING_EPS`` does.  The request validator only keeps 1/temperature
# finite; scaling a logprob by a reciprocal that large still overflows, and
# every route would then sample from an all-NaN law.
SAMPLING_EPS = 1e-5

GENERATION_CONFIG = "generation_config.json"
MODEL_CARD = "model card"


@dataclass(frozen=True)
class SamplingDefaults:
    """One vendor profile: only the fields the vendor recommends are set.

    ``source`` cites where the values come from (for example
    ``"generation_config.json"`` or ``"model card"``).  ``field_sources`` may
    refine that per field when a profile mixes sources.
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    source: str = ""
    field_sources: Mapping[str, str] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        if not self.source:
            raise ValueError("sampling defaults need a source citation")
        unknown = set(self.field_sources) - set(SAMPLING_FIELDS)
        if unknown:
            raise ValueError(f"unknown sampled fields: {sorted(unknown)}")
        object.__setattr__(
            self, "field_sources", MappingProxyType(dict(self.field_sources))
        )
        values = self.values()
        if not values:
            raise ValueError("sampling defaults must set at least one field")
        _validate_values(values)

    def values(self) -> dict:
        """The fields this profile sets, in ``SAMPLING_FIELDS`` order."""
        return {
            name: getattr(self, name)
            for name in SAMPLING_FIELDS
            if getattr(self, name) is not None
        }

    def source_of(self, name: str) -> str:
        return self.field_sources.get(name, self.source)

    def as_dict(self) -> dict:
        return {
            "values": self.values(),
            "sources": {name: self.source_of(name) for name in self.values()},
            **({"note": self.note} if self.note else {}),
        }


def _validate_values(values: Mapping) -> None:
    bounds = {
        "temperature": (0, 2),
        "top_p": (0, 1),
        "min_p": (0, 1),
        "repetition_penalty": (0.01, 10),
        "presence_penalty": (-2, 2),
        "frequency_penalty": (-2, 2),
    }
    for name, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"{name} default must be numeric")
        if name == "top_k":
            if not isinstance(value, int) or value < 0:
                raise ValueError("top_k default must be a nonnegative integer")
            continue
        lower, upper = bounds[name]
        if not isinstance(value, (int, float)) or not lower <= value <= upper:
            raise ValueError(f"{name} default must be between {lower} and {upper}")


@dataclass(frozen=True)
class VendorSampling:
    """A model's vendor sampling profiles and how a request selects one.

    ``profiles`` maps a profile name to its ``SamplingDefaults``.  ``general``
    names the profile used when the request's mode is unknown.  ``thinking``
    and ``non_thinking`` name mode-specific profiles (``None`` means the
    vendor gives none, and the general profile covers that mode too).
    ``model`` names the vendor model the values were read for.
    """

    profiles: Mapping[str, SamplingDefaults]
    general: str = "general"
    thinking: Optional[str] = None
    non_thinking: Optional[str] = None
    model: str = ""

    def __post_init__(self):
        if not self.profiles:
            raise ValueError("vendor sampling needs at least one profile")
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        for role, name in (
            ("general", self.general),
            ("thinking", self.thinking),
            ("non_thinking", self.non_thinking),
        ):
            if name is not None and name not in self.profiles:
                raise ValueError(f"{role} profile {name!r} is not declared")
            if role == "general" and name is None:
                raise ValueError("a general profile is required")

    @classmethod
    def single(cls, defaults: SamplingDefaults, *, model: str = "") -> "VendorSampling":
        return cls({"general": defaults}, model=model)

    def select(
        self, *, thinking: Optional[bool], requested: Optional[str] = None
    ) -> tuple[str, SamplingDefaults, str]:
        """Return ``(profile name, defaults, reason)`` for one request."""
        if requested is not None:
            if requested not in self.profiles:
                raise ValueError(
                    "unknown sampling_profile "
                    f"{requested!r}; this model declares "
                    + ", ".join(sorted(self.profiles))
                )
            return requested, self.profiles[requested], "requested"
        if thinking is True and self.thinking is not None:
            return self.thinking, self.profiles[self.thinking], "thinking"
        if thinking is False and self.non_thinking is not None:
            return self.non_thinking, self.profiles[self.non_thinking], "non_thinking"
        return self.general, self.profiles[self.general], "general"

    def as_dict(self) -> dict:
        return {
            "schema": SAMPLING_DEFAULTS_SCHEMA,
            "model": self.model,
            "general": self.general,
            "thinking": self.thinking,
            "non_thinking": self.non_thinking,
            "profiles": {
                name: profile.as_dict() for name, profile in self.profiles.items()
            },
        }


def vendor_sampling(adapter) -> Optional[VendorSampling]:
    """The adapter's declared ``VendorSampling``, or ``None``."""
    declared = getattr(adapter, "sampling_defaults", None)
    if declared is None:
        return None
    if callable(declared):
        declared = declared()
    if isinstance(declared, SamplingDefaults):
        declared = VendorSampling.single(declared)
    if not isinstance(declared, VendorSampling):
        raise TypeError("adapter sampling_defaults must be a VendorSampling")
    return declared


def validate_sampling_profile(value) -> None:
    """Shape check for the ``sampling_profile`` request extension."""
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise ValueError("sampling_profile must be a nonempty string")


def resolve_sampling(
    request: Mapping,
    vendor: Optional[VendorSampling],
    *,
    thinking: Optional[bool],
) -> tuple[dict, dict]:
    """Merge request sampling fields over vendor defaults.

    Returns ``(effective, record)``.  ``effective`` holds every
    ``SAMPLING_FIELDS`` value the samplers use.  ``record`` is the receipt
    entry: which fields the request set explicitly, which vendor defaults
    were applied (with their sources), and which fields took the neutral or
    legacy fallback.
    """
    requested = request.get("sampling_profile")
    if requested is not None:
        validate_sampling_profile(requested)
        if vendor is None:
            raise ValueError("this model declares no sampling profiles")
    explicit = [name for name in SAMPLING_FIELDS if name in request]
    effective = {name: request[name] for name in explicit}
    record = {
        "schema": SAMPLING_DEFAULTS_SCHEMA,
        "explicit": explicit,
        "applied": {},
        "sources": {},
    }
    if vendor is None:
        fallback = LEGACY_FALLBACK
        record.update(profile=None, profile_reason="no_vendor_defaults")
    else:
        name, profile, reason = vendor.select(thinking=thinking, requested=requested)
        fallback = NEUTRAL
        record.update(profile=name, profile_reason=reason, model=vendor.model)
        for field_name, value in profile.values().items():
            if field_name not in effective:
                effective[field_name] = value
                record["applied"][field_name] = value
                record["sources"][field_name] = profile.source_of(field_name)
    record["fallback"] = {
        name: fallback[name] for name in SAMPLING_FIELDS if name not in effective
    }
    for name, value in record["fallback"].items():
        effective[name] = value
    record["fallback_kind"] = "legacy" if vendor is None else "neutral"
    effective = {name: effective[name] for name in SAMPLING_FIELDS}
    temperature = effective["temperature"]
    if 0 < temperature < SAMPLING_EPS:
        # Every route reads the effective temperature, so this is the one
        # place a vanishing temperature becomes greedy decoding.
        effective["temperature"] = 0.0
        record["greedy_temperature"] = {"requested": temperature, "epsilon": SAMPLING_EPS}
    return effective, record


def generation_config_drift(model_path, vendor: Optional[VendorSampling]) -> dict:
    """Compare the artifact's ``generation_config.json`` with the declaration.

    Informational only (published in ``/v1/status``): a fine-tune may ship a
    different generation config than the vendor model the adapter cites.
    Declared values stay authoritative; mismatches are reported, not applied.
    """
    result = {"generation_config": None, "mismatches": {}}
    if vendor is None:
        return result
    path = Path(model_path).expanduser() / "generation_config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return result
    if not isinstance(config, dict):
        return result
    sampled = {
        name: config[name] for name in SAMPLING_FIELDS if name in config
    }
    result["generation_config"] = {
        "do_sample": config.get("do_sample"),
        **sampled,
    }
    general = vendor.profiles[vendor.general].values()
    for name, value in sampled.items():
        if name in general and general[name] != value:
            result["mismatches"][name] = {"declared": general[name], "artifact": value}
    if config.get("do_sample") is False and general.get("temperature", 0) > 0:
        result["mismatches"]["do_sample"] = {"declared": True, "artifact": False}
    return result


# ---------------------------------------------------------------------------
# Xing4.0 (XingChen-AGI/Xing4.0-29B-A4B).  Declared here so the Xing adapter
# (``adapters/xing.py``) can reuse it as ``sampling_defaults = XING4_SAMPLING``.
# generation_config.json: do_sample true, temperature 1.0, top_p 0.95,
# repetition_penalty 1.05.  Model card "Recommended Settings": complex
# reasoning / general tasks temperature 1.0, top_p 0.95,
# repetition_penalty 1.05; coding / agent tasks temperature 0.8, top_p 0.95,
# repetition_penalty 1.05.  The card gives no thinking-specific profile, so
# thinking on/off both use the general profile; coding/agent is selected per
# request with ``sampling_profile: "coding"`` (alias ``"agent"``).
# ---------------------------------------------------------------------------
_XING_CARD = "model card (XingChen-AGI/Xing4.0-29B-A4B, Recommended Settings)"
_XING_CODING = SamplingDefaults(
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.05,
    source=_XING_CARD + ": coding / agent tasks",
)
XING4_SAMPLING = VendorSampling(
    {
        "general": SamplingDefaults(
            temperature=1.0,
            top_p=0.95,
            repetition_penalty=1.05,
            source=GENERATION_CONFIG,
            note="also the model card's complex reasoning / general profile",
        ),
        "coding": _XING_CODING,
        "agent": _XING_CODING,
    },
    model="XingChen-AGI/Xing4.0-29B-A4B",
)


__all__ = [
    "GENERATION_CONFIG",
    "LEGACY_FALLBACK",
    "MODEL_CARD",
    "NEUTRAL",
    "SAMPLING_DEFAULTS_SCHEMA",
    "SAMPLING_FIELDS",
    "SamplingDefaults",
    "generation_config_drift",
    "VendorSampling",
    "XING4_SAMPLING",
    "resolve_sampling",
    "validate_sampling_profile",
    "vendor_sampling",
]
