"""Owned VLM load boundary, bound to route dependencies instead of Git HEAD.

The install pin remains a reproducible default. A different checkout is usable
only when every reachable source/resource byte matches the reviewed contract.
No import-path mutation or cross-revision package replacement occurs here.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ..source_dependencies import (
    UnresolvedDependency,
    add_resources,
    canonical_digest,
    source_closure,
)

_CONTRACT_FILES = {
    **dict.fromkeys(("gemma3n", "gemma4", "minicpmo", "qwen_image"), "67599f2e"),
    **dict.fromkeys(("lfm2_vl", "smolvlm", "qwen2_5_vl", "agnes"), "8a5e704e"),
}


def _package_root():
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or spec.origin is None:
        raise RuntimeError("multimodal adapters require the optional mlx-vlm runtime")
    root = Path(spec.origin).resolve().parent
    locations = tuple(
        Path(p).resolve() for p in (spec.submodule_search_locations or ())
    )
    if locations != (root,):
        raise RuntimeError("mlx-vlm must resolve to one source package")
    return root


def _contract(family):
    name = _CONTRACT_FILES.get(family)
    if name is None:
        raise ValueError(f"no owned mlx-vlm dependency contract for {family!r}")
    path = Path(__file__).with_name("vlm_contracts") / (name + ".json")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1:
        raise RuntimeError("unrecognized mlx-vlm source contract")
    return manifest, manifest["families"][family]


def verify_contract(family, package_root):
    """Hash imports without executing upstream code or loading model weights."""
    root = Path(package_root).resolve()
    manifest, contract = _contract(family)
    expected = {name: manifest["files"][name] for name in contract["files"]}
    try:
        files = source_closure(
            root, contract["roots"], reviewed_dynamic=contract["reviewed_dynamic"]
        )
        add_resources(root, files)
    except (UnresolvedDependency, OSError, ValueError) as exc:
        raise RuntimeError(
            f"mlx-vlm {family} source contract unavailable: {exc}"
        ) from exc
    if files != expected:
        changed = sorted(
            name
            for name in files.keys() | expected.keys()
            if files.get(name) != expected.get(name)
        )
        raise RuntimeError(
            f"mlx-vlm {family} dependency content differs from reviewed "
            f"{manifest['source_revision'][:8]}: {', '.join(changed[:6])}"
        )
    # Only code in this contract may already occupy a bound import name.
    for name, module in tuple(sys.modules.items()):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            origin = getattr(module, "__file__", None)
            if origin is not None and not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError("another mlx-vlm package is already imported")
    return {
        "schema": "mlx2.vlm-dependencies.v1",
        "family": family,
        "source_sha256": canonical_digest({"family": family, "files": files}),
        "dependency_files": len(files),
        "reference_revision": manifest["source_revision"],
    }


def validate_model_dispatch(family, model_path):
    config = json.loads((Path(model_path) / "config.json").read_text())
    model_type = str(config.get("model_type", "")).replace("-", "_")
    if model_type != family:
        raise ValueError("model artifact differs from bound VLM family")
    if (
        config.get("model_file")
        or config.get("dflash_config") is not None
        or config.get("speculators_model_type")
    ):
        raise ValueError(
            "custom model/drafter dispatch is outside the VLM load contract"
        )
    if set(config.get("architectures") or ()) & {
        "BoundaryExtractor",
        "DFlash2DraftModel",
        "Gemma4DSparkModel",
    }:
        raise ValueError("artifact architecture overrides the VLM family")


@dataclass(frozen=True)
class VLMBackend:
    family: str
    package_root: Path
    identity: dict
    provenance: dict | None

    def load(self, model_path, **kwargs):
        validate_model_dispatch(self.family, model_path)
        if set(kwargs) - {"lazy", "strict", "trust_remote_code"}:
            raise ValueError("loader overrides are outside the owned VLM contract")
        if kwargs.get("trust_remote_code", False) is not False:
            raise ValueError("owned VLM loader does not execute remote processor code")
        if kwargs.get("strict", True) is not True:
            raise ValueError("owned VLM loader requires strict weight matching")
        if _package_root() != self.package_root:
            raise RuntimeError("mlx-vlm import origin changed after binding")
        verify_contract(self.family, self.package_root)
        module = importlib.import_module("mlx_vlm")
        result = module.load(
            str(model_path), **dict(kwargs, trust_remote_code=False, strict=True)
        )
        # Catch source edits during a slow weight load before serving starts.
        verify_contract(self.family, self.package_root)
        model_module = type(result[0]).__module__
        expected_module = "mlx_vlm.models." + self.family
        if model_module != expected_module and not model_module.startswith(
            expected_module + "."
        ):
            raise RuntimeError("loaded model escaped the bound VLM family")
        return result


def bind_backend(family):
    root = _package_root()
    identity = verify_contract(family, root)
    from .mlx_vlm_pin import mlx_vlm_runtime

    return VLMBackend(family, root, identity, mlx_vlm_runtime())
