"""Adapter-owned native profiles and factories; lifecycle remains engine-owned.

A backend returns a source-bound preparation before allocating tensors. The
engine still validates live jobs, installs atomically, publishes receipts and
retires failed owners. No profile grants qualification or selects a default.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NativeCohortPreparation:
    profile: Mapping[str, Any]
    identity: Mapping[str, Any]
    factory_profile: Mapping[str, Any]
    packed: bool = False
    neutral_filters_required: bool = False
    input_ids: tuple = ()
    options: Mapping[str, Any] | None = None


def backend_for(adapter, kind):
    hook = getattr(adapter, "native_cohort_backend", None)
    if not callable(hook):
        raise ValueError("adapter has no native cohort profile/factory capability")  # noqa: TRY004 - capability refusal
    backend = hook(kind)
    if not callable(getattr(backend, "prepare", None)) or not callable(
        getattr(backend, "allocate", None)
    ):
        raise ValueError("adapter returned an incomplete native cohort capability")  # noqa: TRY004 - capability refusal
    return backend


def live_identity(adapter, manifest_path, mlx_wheel_path):
    import _paged_kv_native

    from ..runtime.paged_price_identity import cached_live_price_identity

    return cached_live_price_identity(
        Path(manifest_path),
        Path(mlx_wheel_path),
        Path(_paged_kv_native.__file__).resolve(),
        adapter_artifact_root=Path(adapter.identity["path"]).resolve(),
    )
