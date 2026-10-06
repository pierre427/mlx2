"""Verify resident Nemotron artifacts and source-bound qualification state.

This preflight is deliberately host-only.  It reads JSON and safetensors
headers, hashes the small diarization checkpoint and the pinned Lightning MTP
sidecar, and refuses to import MLX.  Passing it means that a later native run
has the expected inputs; it does not qualify either model or route.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mlx2.adapters.nemotron3_diarization import (
    MODEL_REVISION,
    STREAMING_PROFILES,
)
from mlx2.adapters.nemotron3_diarization import (
    inspect_artifact as inspect_diarization,
)
from mlx2.adapters.nemotron3_super import DESCRIPTOR as SUPER_DESCRIPTOR
from mlx2.adapters.nemotron3_super import inspect_artifact as inspect_super
from mlx2.adapters.nemotron35_lightning import (
    TARGET_REVISION,
    descriptor_for,
)
from mlx2.adapters.nemotron35_lightning import (
    inspect_artifact as inspect_lightning,
)
from mlx2.diarization_qualification import (
    APPROVED_DIARIZATION_HARNESS,
    APPROVED_DIARIZATION_PLAN_SHA256S,
    APPROVED_NEMO_SCORER_REVISION,
    load_qualified_diarization_cli,
)
from mlx2.process_env import (
    PROCESS_NUMERICS,
    ProcessNumericsConflict,
    require_process_numerics,
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_identity() -> dict:
    source_tree = hashlib.sha256()
    for path in sorted((ROOT / "src/mlx2").rglob("*.py")):
        source_tree.update(str(path.relative_to(ROOT)).encode())
        source_tree.update(b"\0")
        source_tree.update(path.read_bytes())
        source_tree.update(b"\0")
    return {
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_dirty": bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
            ).strip()
        ),
        "source_tree_sha256": source_tree.hexdigest(),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                ROOT / "scripts/check_nemotron_cpu_preflight.py",
                ROOT / "src/mlx2/adapters/nemotron3_diarization.py",
                ROOT / "src/mlx2/adapters/nemotron35_lightning.py",
                ROOT / "src/mlx2/adapters/nemotron3_super.py",
                ROOT / "src/mlx2/diarize_cli.py",
                ROOT / "src/mlx2/diarization_qualification.py",
                ROOT / "src/mlx2/process_env.py",
                ROOT / "src/mlx2/runtime/nemotron_prefix_reuse.py",
                ROOT / "src/mlx2/runtime/models/nemotron_h.py",
                ROOT / "src/mlx2/runtime/models/ssm.py",
            )
        },
    }


def _execution_numerics() -> dict:
    observed = {name: os.environ.get(name) for name in PROCESS_NUMERICS}
    result = {
        "required": dict(PROCESS_NUMERICS),
        "observed": observed,
        "accepted": True,
    }
    try:
        require_process_numerics("Nemotron CPU preflight")
    except ProcessNumericsConflict as error:
        result.update(accepted=False, reason=str(error))
    return result


def qualification_differences(
    record: dict, artifact: dict, *, profile: str
) -> list[str]:
    """Explain why a retained diarization receipt is not current.

    The authoritative decision remains ``load_qualified_diarization_cli``.
    These stable labels make a refusal actionable without weakening that gate.
    """

    expected_settings = {
        "dtype": "float32",
        "attention": "flash",
        "threshold": 0.5,
        "min_frames": 0,
    }
    differences = []
    checks = {
        "schema": record.get("schema") == "mlx2.diarization-cli-qualification.v2",
        "status": record.get("status") == "qualified_cli",
        "qualification_harness": (
            record.get("qualification_harness") == APPROVED_DIARIZATION_HARNESS
        ),
        "plan_sha256": record.get("plan_sha256") in APPROVED_DIARIZATION_PLAN_SHA256S,
        "nemo_speech_revision": (
            record.get("nemo_speech_revision") == APPROVED_NEMO_SCORER_REVISION
        ),
        "model_revision": record.get("model_revision") == MODEL_REVISION,
        "artifact_sha256": record.get("artifact_sha256") == artifact["weight_sha256"],
        "adapter_source_sha256": (
            record.get("adapter_source_sha256") == artifact["source_sha256"]
        ),
        "settings": record.get("settings") == expected_settings,
        "runtime.mlx": (
            record.get("runtime", {}).get("mlx") == importlib.metadata.version("mlx")
        ),
        "runtime.machine": (
            record.get("runtime", {}).get("machine") == platform.machine()
        ),
        f"profiles.{profile}": profile in record.get("profiles", {}),
    }
    for name, matches in checks.items():
        if not matches:
            differences.append(name)
    return differences


def run_preflight(args: argparse.Namespace) -> tuple[dict, bool]:
    if "mlx" in sys.modules or "mlx.core" in sys.modules:
        raise RuntimeError("Nemotron CPU preflight must start before any MLX import")
    execution_numerics = _execution_numerics()
    report = {
        "schema": "mlx2.nemotron-cpu-preflight.v1",
        "semantics": "artifact and source readiness only; no native qualification",
        "source": _source_identity(),
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "mlx_distribution": importlib.metadata.version("mlx"),
        },
        "execution_numerics": execution_numerics,
        "artifacts": {},
    }
    ready = execution_numerics["accepted"]
    if args.diarization_model is not None:
        artifact = inspect_diarization(args.diarization_model, verify_hash=True)
        qualification = {
            "state": "not_provided",
            "accepted": False,
            "differences": [],
        }
        if args.diarization_receipt is not None:
            record = json.loads(args.diarization_receipt.read_text())
            qualification["differences"] = qualification_differences(
                record, artifact, profile=args.diarization_profile
            )
            try:
                route = load_qualified_diarization_cli(
                    args.diarization_receipt,
                    artifact=artifact,
                    profile=args.diarization_profile,
                    attention="flash",
                    dtype="float32",
                    threshold=0.5,
                    min_frames=0,
                )
            except ValueError as error:
                qualification["state"] = "refused_stale_or_incomplete"
                qualification["reason"] = str(error)
                if not qualification["differences"]:
                    qualification["differences"] = ["authoritative_gate_unclassified"]
            else:
                qualification.update(state="accepted", accepted=True, route=route)
        if args.require_diarization_qualified and not qualification["accepted"]:
            ready = False
        report["artifacts"]["diarization"] = {
            "path": artifact["path"],
            "revision": artifact["revision"],
            "weight_sha256": artifact["weight_sha256"],
            "adapter_source_sha256": artifact["source_sha256"],
            "profiles": list(STREAMING_PROFILES),
            "static_artifact_gate": "passed",
            "qualification": qualification,
        }
    if args.lightning_model is not None:
        artifact = inspect_lightning(args.lightning_model)
        descriptor = descriptor_for(has_mtp=artifact["has_mtp"])
        if descriptor.metadata.get("qualification") != "pending":
            raise AssertionError(
                "Lightning CPU preflight must not promote qualification"
            )
        report["artifacts"]["lightning"] = {
            "path": artifact["identity"]["path"],
            "target_revision": TARGET_REVISION,
            "fingerprint": artifact["identity"]["fingerprint"],
            "target_shards": len(artifact["identity"]["files"])
            - int(artifact["has_mtp"]),
            "has_verified_mtp": artifact["has_mtp"],
            "mtp_tensor_count": artifact["mtp_tensor_count"],
            "default_route": "ordinary",
            "qualification": descriptor.metadata["qualification"],
            "static_artifact_gate": "passed",
        }
    if args.super_model is not None:
        artifact = inspect_super(args.super_model)
        if SUPER_DESCRIPTOR.metadata.get("qualification") != "pending":
            raise AssertionError("Super CPU preflight must not promote qualification")
        report["artifacts"]["super"] = {
            "path": artifact["identity"]["path"],
            "fingerprint": artifact["identity"]["fingerprint"],
            "target_shards": len(artifact["identity"]["files"]),
            "has_embedded_mtp": artifact["has_mtp"],
            "mtp_tensor_count": artifact["mtp_tensor_count"],
            "default_route": "ordinary",
            "qualification": SUPER_DESCRIPTOR.metadata["qualification"],
            "static_artifact_gate": "passed",
        }
    if not report["artifacts"]:
        raise ValueError("select at least one Nemotron artifact")
    if "mlx" in sys.modules or "mlx.core" in sys.modules:
        raise RuntimeError("Nemotron CPU preflight imported MLX")
    report["mlx_imported"] = False
    report["status"] = "passed" if ready else "refused"
    return report, ready


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diarization-model", type=Path)
    parser.add_argument("--diarization-receipt", type=Path)
    parser.add_argument(
        "--diarization-profile", choices=tuple(STREAMING_PROFILES), default="offline"
    )
    parser.add_argument("--require-diarization-qualified", action="store_true")
    parser.add_argument("--lightning-model", type=Path)
    parser.add_argument("--super-model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.diarization_receipt is not None and args.diarization_model is None:
        parser.error("--diarization-receipt requires --diarization-model")
    if args.require_diarization_qualified and args.diarization_receipt is None:
        parser.error("--require-diarization-qualified requires --diarization-receipt")
    report, ready = run_preflight(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            parser.error(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
