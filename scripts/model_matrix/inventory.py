#!/usr/bin/env python3
"""Inventory local text-target artifacts (>= 8B parameters) and their drafters.

GPU-free.  For every complete artifact under the model roots this resolves the
mlx2 adapter through ``mlx2.adapters.registry.inspect_model`` (the repo's
identity helper; the checkpoint identity is ``<name>@sha256:<fingerprint>``),
counts parameters from safetensors headers (packed quantized tensors are
expanded by their configured bit width), de-duplicates hard-linked or
byte-identical artifacts, and pairs each target with the external drafter
artifacts its adapter accepts, including the exact CLI and execution policy
needed to select them.  Drafter pairing is decided by the adapter's own
GPU-free drafter inspector, never by name matching.

Image/audio/video/diffusion models and pipeline components are listed as
skipped in a separate lane; drafter artifacts are never targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.adapters.drafter_pairing import (
    DEFAULT_ROOTS,
    DRAFT_ARCH_MARKERS,
    DRAFT_MODEL_TYPES,
    HF_HUB,
    MEDIA_MODEL_TYPES,
    PIPELINE_COMPONENTS,
    classify,
    discover,
    display_name,
    pair_drafter,
    shard_names,
)
from mlx2.adapters.drafter_pairing import (
    route_slug as _route_slug,
)

MIN_PARAMS = 8_000_000_000
SCHEMA = "mlx2.model-matrix-inventory.v1"

__all__ = [
    "DEFAULT_ROOTS", "DRAFT_ARCH_MARKERS", "DRAFT_MODEL_TYPES", "HF_HUB",
    "MEDIA_MODEL_TYPES", "PIPELINE_COMPONENTS", "classify", "discover",
    "display_name", "shard_names",
]


def _read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(length))


def _quant_bits(config: dict, module: str) -> int:
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if not isinstance(quant, dict):
        return 4
    candidates = [module]
    for prefix in ("language_model.", "model.language_model.", "model."):
        if module.startswith(prefix):
            candidates.append(module[len(prefix):])
        candidates.append(prefix + module)
    for name in candidates:
        override = quant.get(name)
        if isinstance(override, dict) and isinstance(override.get("bits"), int):
            return override["bits"]
    bits = quant.get("bits")
    return bits if isinstance(bits, int) and bits > 0 else 4


def parameter_count(path: Path, names: list[str], config: dict) -> int:
    headers = {}
    for name in names:
        for key, record in _read_header(path / name).items():
            if key != "__metadata__":
                headers[key] = record
    total = 0
    for key, record in headers.items():
        if key.endswith((".scales", ".biases")):
            continue
        numel = 1
        for value in record["shape"]:
            numel *= value
        if key.endswith("_scales") and key[: -len("_scales")] + "_blocks" in headers:
            continue  # HF MXFP4 block scales
        if key.endswith("_blocks") and record["dtype"] == "U8":
            total += numel * 2  # two FP4 values per byte
            continue
        if (
            key.endswith(".weight")
            and key[: -len(".weight")] + ".scales" in headers
            and record["dtype"] in {"U32", "U8", "I32"}
        ):
            width = {"U32": 32, "I32": 32, "U8": 8}[record["dtype"]]
            numel = numel * width // _quant_bits(config, key[: -len(".weight")])
        total += numel
    return total


def quant_label(config: dict) -> str:
    quant = config.get("quantization") or config.get("quantization_config")
    if isinstance(quant, dict):
        mode = quant.get("mode") or quant.get("quant_method")
        bits = quant.get("bits")
        group = quant.get("group_size")
        overrides = sum(1 for value in quant.values() if isinstance(value, dict))
        parts = [str(mode or "affine")]
        if bits:
            parts.append(f"{bits}bit")
        if group:
            parts.append(f"g{group}")
        if overrides:
            parts.append(f"mixed({overrides} overrides)")
        return "-".join(parts)
    text = config.get("text_config", config)
    return str(text.get("torch_dtype") or config.get("torch_dtype") or config.get("dtype") or "unquantized")


def physical_identity(path: Path, names: list[str]) -> tuple:
    rows = []
    for name in names:
        stat = (path / name).stat()
        rows.append((stat.st_dev, stat.st_ino))
    return tuple(sorted(rows))


# ---------------------------------------------------------------- pairing --

def _pair(target: dict, draft: dict, cache: dict) -> dict:
    """Ask the target adapter's own inspector whether ``draft`` binds.

    The pairing logic lives in ``mlx2.adapters.drafter_pairing`` and is
    shared with the server's best-route defaults.  A separately shipped MTP
    head binds through the adapter's own head hook instead.
    """
    if draft.get("mtp_head"):
        return _pair_mtp_head(target, draft)
    pairing = pair_drafter(target, draft, cache)
    pairing.pop("identity", None)
    pairing.pop("pins", None)
    return pairing


def _pair_mtp_head(target: dict, draft: dict) -> dict:
    """A separately shipped MTP head, bound by the adapter's own hook."""
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(target["path"])
    hook = getattr(resolution.adapter_type, "external_mtp_head_resolution", None)
    if not callable(hook):
        raise ValueError(f"{target['adapter']} does not attach external MTP heads")
    bound = hook(resolution, draft["path"])
    head = bound.artifact["identity"]["mtp_head"]
    return {
        "draft": f"{draft['name']}@sha256:{head['fingerprint']}",
        "draft_path": draft["path"],
        "draft_weight_bytes": draft.get("weight_bytes"),
        "draft_kind": "MTPHead",
        "route": "native_mtp",
        "server_flags": ["--native-mtp", "--mtp-head", draft["path"]],
        "execution_policy": None,
        "pair_fingerprint": bound.artifact["identity"]["fingerprint"],
        "head_quantization": head["quantization"],
    }


def plausible_family(target: dict, draft: dict) -> bool:
    """Name/geometry hint used only to decide whether a refusal is reportable."""
    tname, dname = target["name"].lower(), draft["name"].lower()
    for key in ("flash-next", "qwen3.8-27b", "qwen3.6-35b", "muse-glimmer", "north-mini",
                "laguna-xs", "nemotron-3.5-super", "nemotron-3.5-bolt",
                "nemotron-3.5-lightning", "qwen3-coder", "qwen3.6-27b"):
        if key in tname and key in dname:
            return True
    return False


def target_routes(target: dict) -> list[dict]:
    caps = set(target["capabilities"])
    routes = [{"route": "ordinary", "server_flags": ["--ordinary"]}]
    if "prompt_lookup" in caps:
        routes.append({"route": "pld_default", "server_flags": ["--prompt-lookup"]})
        routes.append({
            "route": "pld_advanced",
            "server_flags": ["--prompt-lookup", "--execution-policy", "<policy.json>"],
            "execution_policy": {"prompt_lookup": PLD_ADVANCED},
        })
    else:
        routes.append({
            "route": "pld_default",
            "server_flags": ["--prompt-lookup"],
            "expected": "fail_closed: adapter does not declare prompt_lookup",
        })
    if "mtp" in caps:
        routes.append({"route": "native_mtp", "server_flags": ["--native-mtp"]})
    if target.get("adapter") in AST_ROUTE_KIND:
        # --ast with the automatic drafter (MTP head, else model-free PLD tree);
        # families without tree tiles map it to their best chain route.
        routes.append({"route": "ast_auto", "server_flags": ["--ast"],
                       "expected_route_kind": AST_ROUTE_KIND[target["adapter"]]})
    return routes


# --ast families and the route their receipt must show.
AST_ROUTE_KIND = {
    "Qwen3827BAdapter": "external_draft", "Qwen3627BAdapter": "external_draft",
    "Qwen3635BA3BAdapter": "external_draft", "GptOssAdapter": "external_draft",
    "Qwen359BAdapter": "prompt_lookup",
}
AST_TILE_ADAPTERS = frozenset(k for k, v in AST_ROUTE_KIND.items() if v == "external_draft")


AST_DRAFT_KINDS = frozenset({"DFlash2DraftModel", "Qwen3DSparkModel", "MTPHead"})


def ast_drafter_route(target: dict, pairing: dict, slug: str) -> dict | None:
    if target.get("adapter") not in AST_TILE_ADAPTERS or pairing.get("draft_kind") not in AST_DRAFT_KINDS:
        return None
    return {"route": "ast_" + slug, "server_flags": ["--ast", "--ast-drafter", pairing["draft_path"]],
            "expected_route_kind": "external_draft", "draft": pairing["draft"],
            "draft_weight_bytes": pairing.get("draft_weight_bytes")}


PLD_ADVANCED = {
    "num_draft": 8,
    "ngram_min": 2,
    "ngram_max": 8,
    "lookback_ladder": [256, 1024, 4096, 16384],
    "deferred_admission": True,
    "cost_aware_admission": False,
    "cliff_aware_span": True,
    "recent_committed_segments": True,
    "recent_committed_max_responses": 4,
}


def build(roots: list[Path], include_hf: bool, host: str, revision_cache: dict) -> dict:
    from mlx2.adapters.registry import inspect_model

    started = time.time()
    raw = []
    from mlx2.adapters.qwen38_external_mtp_head import is_mtp_head_artifact

    for path in discover(roots, include_hf):
        if not (path / "config.json").is_file() and is_mtp_head_artifact(path):
            file = path / "mtp.safetensors"
            raw.append({
                "path": str(path), "name": display_name(path), "model_type": "qwen3_5_mtp",
                "architectures": [], "complete": True, "status": "drafter", "mtp_head": True,
                "weight_bytes": file.stat().st_size, "quant": "packed-per-module",
                "parameters": None,
            })
            continue
        try:
            config = json.loads((path / "config.json").read_text())
        except Exception as error:  # noqa: BLE001
            raw.append({"path": str(path), "status": "invalid_config", "error": str(error)})
            continue
        if not isinstance(config, dict):
            continue
        names, complete = shard_names(path)
        row = {
            "path": str(path),
            "name": display_name(path),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures") or [],
            "complete": complete,
        }
        if not complete:
            row["status"] = "incomplete_download"
            raw.append(row)
            continue
        row["weight_bytes"] = sum((path / name).stat().st_size for name in names)
        row["physical"] = physical_identity(path, names)
        lane = classify(path, config)
        row["quant"] = quant_label(config)
        if config.get("model_type") == "qwen3_5_mtp":
            row["mtp_head"] = True
        if lane == "drafter":
            row["status"] = "drafter"
        elif lane is not None:
            row["status"] = "skipped_" + lane
            raw.append(row)
            continue
        try:
            row["parameters"] = parameter_count(path, names, config)
        except Exception as error:  # noqa: BLE001
            row["parameters"] = None
            row["parameter_error"] = f"{type(error).__name__}: {error}"
        if lane == "drafter":
            raw.append(row)
            continue
        try:
            resolution = inspect_model(path)
            row["adapter"] = resolution.adapter_type.__name__
            row["fingerprint"] = resolution.artifact_fingerprint
            row["capabilities"] = sorted(c.value for c in resolution.descriptor.capabilities)
            row["default_route"] = resolution.default_route
            row["status"] = "target"
        except Exception as error:  # noqa: BLE001
            row["status"] = "adapter_refused"
            row["error"] = f"{type(error).__name__}: {str(error)[:400]}"
        raw.append(row)

    # De-duplicate by physical shard identity, then by artifact fingerprint.
    # Prefer the spelling earlier evidence already names (hard-linked copies
    # can carry different path-bound fingerprints).
    known = set()
    decisions = ROOT / "docs/research/model-feature-decisions.tsv"
    if decisions.is_file():
        for line in decisions.read_text().splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) > 1 and "@" in fields[1]:
                known.add(fields[1].split("@", 1)[0])
    preferred = sorted(
        raw,
        key=lambda r: (
            r.get("name") not in known,
            "snapshots" in r["path"],
            r["path"].startswith("/Volumes/"),
            Path(r["path"]).is_symlink(),
            r["path"],
        ),
    )
    seen_physical: dict = {}
    seen_fingerprint: dict = {}
    for row in preferred:
        physical = row.pop("physical", None)
        if physical:
            key = (row.get("status"), physical)
            if key in seen_physical:
                row["duplicate_of"] = seen_physical[key]
                row["duplicate_kind"] = "same_inodes"
                continue
            seen_physical[key] = row["path"]
        fingerprint = row.get("fingerprint")
        if fingerprint:
            key = (row["adapter"], fingerprint)
            if key in seen_fingerprint:
                row["duplicate_of"] = seen_fingerprint[key]
                row["duplicate_kind"] = "same_fingerprint"
                continue
            seen_fingerprint[key] = row["path"]

    def checkpoint(row: dict) -> str:
        fingerprint = row.get("fingerprint")
        if not fingerprint:
            digest = hashlib.sha256(row["path"].encode()).hexdigest()
            return f"{row['name']}@path-sha256:{digest[:16]}"
        return f"{row['name']}@sha256:{fingerprint}"

    for row in raw:
        if row.get("status") in {"target", "drafter", "adapter_refused"}:
            row["checkpoint"] = checkpoint(row)

    targets, drafters, skipped = [], [], []
    for row in raw:
        if row.get("duplicate_of"):
            skipped.append({**row, "reason": f"duplicate ({row['duplicate_kind']}) of {row['duplicate_of']}"})
        elif row.get("status") == "drafter":
            drafters.append(row)
        elif row.get("status") in {"target", "adapter_refused"}:
            params = row.get("parameters") or 0
            if params < MIN_PARAMS:
                skipped.append({**row, "reason": f"below 8B ({params / 1e9:.2f}B parameters)"})
            else:
                targets.append(row)
        else:
            skipped.append({**row, "reason": row.get("status")})

    pair_errors = []
    for target in targets:
        if target.get("status") != "target":
            target["routes"] = []
            continue
        target["routes"] = target_routes(target)
        target["drafters"] = []
        for draft in drafters:
            try:
                pairing = _pair(target, draft, revision_cache)
            except Exception as error:  # noqa: BLE001
                if plausible_family(target, draft):
                    message = f"{type(error).__name__}: {str(error)[:300]}"
                    target["drafters"].append({
                        "draft": draft["checkpoint"],
                        "draft_path": draft["path"],
                        "draft_kind": (draft["architectures"] or [draft.get("model_type")])[0],
                        "route": "external_draft",
                        "wirable": False,
                        "refusal": message,
                    })
                    pair_errors.append({"target": target["checkpoint"], "draft": draft["checkpoint"], "error": message})
                continue
            pairing["wirable"] = True
            target["drafters"].append(pairing)
            if pairing["route"] == "native_mtp":
                # An attached external MTP head serves the native-MTP route.
                target["routes"].append({
                    "route": "mtphead_" + _route_slug(draft),
                    "server_flags": pairing["server_flags"],
                    "expected_route_kind": "native_mtp",
                    "draft": pairing["draft"],
                    "draft_weight_bytes": pairing["draft_weight_bytes"],
                })
            else:
                target["routes"].append({
                    "route": "external_" + _route_slug(draft),
                    "server_flags": ["--external-draft", "--execution-policy", "<policy.json>"],
                    "execution_policy": pairing["execution_policy"],
                    "draft": pairing["draft"],
                    "draft_weight_bytes": pairing["draft_weight_bytes"],
                })
            ast_route = ast_drafter_route(target, pairing, _route_slug(draft))
            if ast_route:
                target["routes"].append(ast_route)
    orphan = [
        d["checkpoint"] for d in drafters
        if not any(
            p.get("draft_path") == d["path"] and p.get("wirable")
            for t in targets for p in t.get("drafters", ())
        )
    ]
    targets.sort(key=lambda r: (r.get("weight_bytes") or 0, r["path"]))
    return {
        "schema": SCHEMA,
        "host": host,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 2),
        "roots": [str(r) for r in roots] + ([str(HF_HUB)] if include_hf else []),
        "min_parameters": MIN_PARAMS,
        "targets": targets,
        "drafters": drafters,
        "unpaired_drafters": orphan,
        "pairing_refusals": pair_errors,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--host", default=socket.gethostname())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision-cache", type=Path,
                        help="JSON cache of payload-bound target revisions keyed by path/size/mtime")
    args = parser.parse_args(argv)
    roots = args.root or list(DEFAULT_ROOTS)
    cache_rows = {}
    if args.revision_cache and args.revision_cache.is_file():
        cache_rows = json.loads(args.revision_cache.read_text())
    cache = _RevisionCache(cache_rows)
    try:
        result = build(roots, not args.no_hf, args.host, cache)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if args.revision_cache:
            args.revision_cache.parent.mkdir(parents=True, exist_ok=True)
            args.revision_cache.write_text(json.dumps(cache.rows, indent=2, sort_keys=True) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    for row in result["targets"]:
        drafts = [d["draft_kind"] + ("" if d.get("wirable") else "(refused)") for d in row.get("drafters", ())]
        print(
            f"{(row.get('weight_bytes') or 0) / 1e9:7.1f}GB {(row.get('parameters') or 0) / 1e9:6.1f}B "
            f"{row.get('adapter', row.get('status')):28s} {row['checkpoint'][:90]} drafts={drafts}"
        )
    print(f"targets={len(result['targets'])} drafters={len(result['drafters'])} "
          f"skipped={len(result['skipped'])} unpaired={len(result['unpaired_drafters'])}")
    return 0


class _RevisionCache(dict):
    """Payload-hash cache invalidated by shard size/mtime."""

    def __init__(self, rows: dict):
        super().__init__()
        self.rows = dict(rows)

    @staticmethod
    def _stamp(path: str) -> str:
        names, _ = shard_names(Path(path))
        stats = [(n, (Path(path) / n).stat().st_size, (Path(path) / n).stat().st_mtime_ns) for n in names]
        return hashlib.sha256(json.dumps(stats).encode()).hexdigest()

    def __contains__(self, key) -> bool:
        row = self.rows.get(json.dumps(key))
        return bool(row) and row["stamp"] == self._stamp(key[1])

    def __getitem__(self, key):
        return self.rows[json.dumps(key)]["value"]

    def __setitem__(self, key, value) -> None:
        self.rows[json.dumps(key)] = {"stamp": self._stamp(key[1]), "value": value}


if __name__ == "__main__":
    raise SystemExit(main())
