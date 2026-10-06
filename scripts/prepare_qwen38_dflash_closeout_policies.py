"""Derive and host-validate source-bound DFlash closeout policies."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.abc
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (ROOT / "src").resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
DEFAULT_MODEL = Path.home() / "mlx-models" / "Qwen3.8-27B-MLX-4bit"
DEFAULT_TENSORFOLD = (
    Path.home() / ".codex/worktrees/tensorfold-upstream-parity-20260928"
)
DEFAULT_STAGE1_POLICY = (
    ROOT / "qualification/runs/dflash-varlen-context-ladder-20261005/policy.json"
)
DEFAULT_ABBA_CANDIDATE_POLICY = (
    ROOT / "qualification/policies/qwen38-27b-dflash2-varlen-tensorfold.json"
)
DEFAULT_ABBA_IDENTITY_TEMPLATE = (
    ROOT
    / "qualification/runs/dflash-varlen-tensorfold-20x20-r2-20261005"
    / "policy-candidate.json"
)
DEFAULT_PLD_CANDIDATE_POLICY = (
    ROOT / "qualification/runs/spec-proposal-arms-20261005/composed-policy.json"
)
DRAFT_PROVENANCE = ROOT / "provenance/qwen38-27b-dflash2.json"
ABBA_HARNESS = ROOT / "scripts/qualify_qwen38_dflash_fixed_cohort_abba.py"
EXPECTED_SOURCE_POLICY_SHA256 = {
    "stage0-b2": "b18415349e55a3cfa027e6f99d461c4922938b4d53fee5f491f2136cc4c64c6c",
    "stage1-parity": "907e71e3185c913a00af559ece065ddbe93e9b1b71871adf47841dc51fdfdf2c",
    "abba-candidate": "c64376a4f9f4ddfd37fc9d03a74a25ffc968e69526d0cdbf02f49607dafb1ea7",
    "abba-identity-template": "382dace320a29102308c5493d58fb0c69d82d5054e0d3a7092999c74eb82debe",
    "pld-candidate": "17fbdd846d1bd72663cb966fb25e6d50770c23948ad58d92575bc7f903c3884d",
}
OUTPUT_NAMES = {
    "stage0-b2": "stage0-b2-policy.json",
    "stage1-parity": "checkpoint-policy.json",
    "abba-control": "abba-control-policy.json",
    "abba-candidate": "abba-candidate-policy.json",
    "pld-control": "pld-control-policy.json",
    "pld-candidate": "pld-candidate-policy.json",
}
IDENTITY_FIELDS = ("draft_revision", "target_revision")
TOPOLOGY_OVERRIDES = (
    "MLX2_DFLASH_TOPOLOGY",
    "MLX2_QWEN_TARGET_EXECUTION",
    "MLX2_TENSORFOLD_COHORT_LIMIT",
)


class _BlockMlxImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise ImportError(
                f"MLX import blocked during host-only policy inspection: {fullname}"
            )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=path, text=True).strip()


def read_policy(raw: bytes, *, label: str) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid policy JSON: {label}") from error
    if not isinstance(value, dict):
        raise TypeError(f"policy must be a JSON object: {label}")
    return value


def stage0_source_bytes(stage1_raw: bytes) -> bytes:
    """Reproduce the exact B2 smoke input emitted by the historical harness."""

    policy = read_policy(stage1_raw, label="stage1 base for stage0")
    policy["tensorfold_cohort_limit"] = 2
    return json.dumps(policy, indent=2, sort_keys=True).encode() + b"\n"


def abba_control_source_bytes(candidate_raw: bytes) -> bytes:
    """Derive the no-compaction control from the corrected varlen candidate."""

    candidate = read_policy(candidate_raw, label="varlen ABBA candidate base")
    expected_varlen = {
        "enabled": True,
        "minimum_padding_fraction": 0.25,
        "minimum_padding_rows": 1,
    }
    if candidate.get("varlen_dense_mlp") != expected_varlen:
        raise RuntimeError("varlen ABBA candidate must use the corrected p25 policy")
    expected_budgets = {"1": 15, "2": 7, "3": 4, "4": 3}
    if candidate.get("tree_node_budget_by_lanes") != expected_budgets:
        raise RuntimeError(
            "varlen ABBA candidate must use corrected 15/7/4/3 lane budgets"
        )
    control = json.loads(json.dumps(candidate))
    control["varlen_dense_mlp"]["minimum_padding_fraction"] = 1.0
    expected_control = {
        **candidate,
        "varlen_dense_mlp": {
            **expected_varlen,
            "minimum_padding_fraction": 1.0,
        },
    }
    if control != expected_control:
        raise RuntimeError("varlen ABBA control changed fields besides the threshold")
    return json.dumps(control, indent=2, sort_keys=True).encode() + b"\n"


def abba_candidate_source_bytes(canonical_raw: bytes, identity_template_raw: bytes) -> bytes:
    """Transplant canonical semantics onto the required legacy identity pins."""

    canonical_policy = read_policy(canonical_raw, label="canonical varlen candidate")
    # Reuse the strict corrected-geometry validation before touching identity.
    abba_control_source_bytes(canonical_raw)
    identity_template = read_policy(
        identity_template_raw, label="varlen candidate identity template"
    )
    identity = {
        field: identity_template.get(field) for field in IDENTITY_FIELDS
    }
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in identity.values()
    ):
        raise RuntimeError("varlen identity template must contain exact legacy pins")
    transplanted = {**canonical_policy, **identity}
    if without_identity(transplanted) != without_identity(canonical_policy):
        raise RuntimeError("varlen candidate transplant changed canonical semantics")
    return json.dumps(transplanted, indent=2, sort_keys=True).encode() + b"\n"


def pld_control_source_bytes(candidate_raw: bytes) -> bytes:
    """Derive the exact-law DFlash control by removing only PLD arbitration."""

    policy = read_policy(candidate_raw, label="PLD/DFlash candidate base")
    composition = policy.pop("proposal_composition", None)
    if composition != {
        "prompt_lookup": True,
        "ngram_min": 3,
        "ngram_max": 6,
        "lookback": 4096,
        "native_mtp": False,
        "mtp_max_history": 4096,
    }:
        raise RuntimeError("PLD/DFlash candidate source has unexpected composition")
    return json.dumps(policy, indent=2, sort_keys=True).encode() + b"\n"


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise RuntimeError(f"missing literal assignment {name} in {path}")


def require_local_modules(names: tuple[str, ...]) -> dict[str, dict[str, str]]:
    identities = {}
    for name in names:
        module = sys.modules.get(name)
        module_path = Path(getattr(module, "__file__", "")).resolve()
        if not module_path.is_relative_to(SOURCE_ROOT):
            raise RuntimeError(
                f"{name} imported from {module_path}, not source-bound {SOURCE_ROOT}"
            )
        identities[name] = {
            "path": str(module_path),
            "sha256": sha256(module_path),
        }
    return identities


def without_identity(policy: dict) -> dict:
    return {key: value for key, value in policy.items() if key not in IDENTITY_FIELDS}


def _identity_span(raw: bytes, field: str) -> tuple[int, int]:
    pattern = re.compile(
        rb'("' + re.escape(field.encode()) + rb'"\s*:\s*")([0-9a-f]{64})(")'
    )
    matches = list(pattern.finditer(raw))
    if len(matches) != 1:
        raise ValueError(f"policy must contain exactly one 64-hex {field}")
    return matches[0].span(2)


def normalized_policy_bytes(raw: bytes) -> bytes:
    normalized = bytearray(raw)
    for field in IDENTITY_FIELDS:
        start, end = _identity_span(raw, field)
        normalized[start:end] = b"0" * (end - start)
    return bytes(normalized)


def derive_policy_bytes(
    source_raw: bytes,
    *,
    draft_revision: str,
    target_revision: str,
    label: str,
) -> tuple[bytes, dict]:
    revisions = {
        "draft_revision": draft_revision,
        "target_revision": target_revision,
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in revisions.values()):
        raise ValueError("payload revisions must be lowercase SHA-256 values")

    source = read_policy(source_raw, label=label)
    generated_raw = bytearray(source_raw)
    source_spans = {}
    for field in IDENTITY_FIELDS:
        start, end = _identity_span(source_raw, field)
        source_spans[field] = [start, end]
        generated_raw[start:end] = revisions[field].encode()
    generated_raw = bytes(generated_raw)
    generated = read_policy(generated_raw, label=f"generated {label}")

    expected = {**source, **revisions}
    if generated != expected:
        raise RuntimeError(f"{label}: generated policy changed unexpected values")
    changed_fields = [
        field for field in IDENTITY_FIELDS if source.get(field) != generated.get(field)
    ]
    if changed_fields != list(IDENTITY_FIELDS):
        raise RuntimeError(
            f"{label}: expected both stale identity fields to change, got {changed_fields}"
        )

    source_semantics = canonical(without_identity(source))
    generated_semantics = canonical(without_identity(generated))
    source_normalized = normalized_policy_bytes(source_raw)
    generated_normalized = normalized_policy_bytes(generated_raw)
    if source_semantics != generated_semantics:
        raise RuntimeError(f"{label}: canonical non-identity semantics changed")
    if source_normalized != generated_normalized:
        raise RuntimeError(f"{label}: non-identity policy bytes changed")

    diff = "".join(
        difflib.unified_diff(
            source_raw.decode().splitlines(keepends=True),
            generated_raw.decode().splitlines(keepends=True),
            fromfile=f"source/{label}.json",
            tofile=f"generated/{OUTPUT_NAMES[label]}",
        )
    )
    return generated_raw, {
        "changed_fields": changed_fields,
        "changes": {
            field: {"before": source[field], "after": generated[field]}
            for field in changed_fields
        },
        "identity_value_byte_spans": source_spans,
        "length_preserved": len(source_raw) == len(generated_raw),
        "canonical_non_identity_equal": True,
        "canonical_non_identity_sha256": sha256_bytes(source_semantics),
        "canonical_source_sha256": sha256_bytes(canonical(source)),
        "canonical_generated_sha256": sha256_bytes(canonical(generated)),
        "normalized_non_identity_bytes_equal": True,
        "normalized_non_identity_bytes_sha256": sha256_bytes(source_normalized),
        "unified_diff": diff,
    }


@contextmanager
def inspection_environment(tensorfold_source: Path):
    conflicts = {
        name: os.environ[name] for name in TOPOLOGY_OVERRIDES if name in os.environ
    }
    if conflicts:
        raise RuntimeError(f"explicit topology overrides are set: {conflicts}")
    previous = os.environ.get("MLX2_TENSORFOLD_SOURCE")
    resolved = str(tensorfold_source.resolve())
    if previous is not None and str(Path(previous).expanduser().resolve()) != resolved:
        raise RuntimeError(
            f"MLX2_TENSORFOLD_SOURCE={previous!r} differs from {resolved!r}"
        )
    os.environ["MLX2_TENSORFOLD_SOURCE"] = resolved
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MLX2_TENSORFOLD_SOURCE", None)
        else:
            os.environ["MLX2_TENSORFOLD_SOURCE"] = previous


def _mlx_modules() -> list[str]:
    return sorted(
        name for name in sys.modules if name == "mlx" or name.startswith("mlx.")
    )


def validation_summary(record: dict) -> dict:
    return {
        "target_revision": record["target_revision"],
        "draft_revision": record["draft_revision"],
        "runtime_draft_fingerprint": record["fingerprint"],
        "draft_config_sha256": record["config_sha256"],
        "draft_index_sha256": record.get("index_sha256"),
        "draft_weight_sha256": record["weight_sha256"],
        "target_source_binding_count": len(record["target_source_bindings"]),
        "draft_source_binding_count": len(record["source_bindings"]),
        "runtime_quantization": record["runtime_quantization"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tensorfold-source", type=Path, default=DEFAULT_TENSORFOLD)
    parser.add_argument("--stage1-policy", type=Path, default=DEFAULT_STAGE1_POLICY)
    parser.add_argument(
        "--abba-candidate-policy", type=Path, default=DEFAULT_ABBA_CANDIDATE_POLICY
    )
    parser.add_argument(
        "--abba-identity-template",
        type=Path,
        default=DEFAULT_ABBA_IDENTITY_TEMPLATE,
    )
    parser.add_argument(
        "--pld-candidate-policy", type=Path, default=DEFAULT_PLD_CANDIDATE_POLICY
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {output_dir}")
    source_commit = git(ROOT, "rev-parse", "HEAD")
    if source_commit != args.expected_source:
        raise RuntimeError(
            f"mlx2 source {source_commit} != expected {args.expected_source}"
        )
    tracked_diff = git(ROOT, "status", "--porcelain", "--untracked-files=no")
    if tracked_diff:
        raise RuntimeError(f"mlx2 tracked tree differs from HEAD:\n{tracked_diff}")
    importable_diff = git(
        ROOT,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src",
        "scripts",
    )
    if importable_diff:
        raise RuntimeError(
            f"mlx2 importable source differs from HEAD:\n{importable_diff}"
        )

    source_paths = {
        "stage1-parity": args.stage1_policy.resolve(),
        "abba-candidate": args.abba_candidate_policy.resolve(),
        "pld-candidate": args.pld_candidate_policy.resolve(),
    }
    sources = {}
    source_file_snapshots = {}
    for label, path in source_paths.items():
        raw = path.read_bytes()
        source_file_snapshots[path] = raw
        digest = sha256_bytes(raw)
        expected = EXPECTED_SOURCE_POLICY_SHA256[label]
        if digest != expected:
            raise RuntimeError(
                f"{label} source policy {digest} != historical pin {expected}"
            )
        sources[label] = {
            "path": path,
            "raw": raw,
            "policy": read_policy(raw, label=label),
            "source_derivation": None,
        }
    abba_identity_template_path = args.abba_identity_template.resolve()
    abba_identity_template_raw = abba_identity_template_path.read_bytes()
    source_file_snapshots[abba_identity_template_path] = abba_identity_template_raw
    identity_template_digest = sha256_bytes(abba_identity_template_raw)
    expected_identity_template = EXPECTED_SOURCE_POLICY_SHA256[
        "abba-identity-template"
    ]
    if identity_template_digest != expected_identity_template:
        raise RuntimeError(
            "abba identity template source policy "
            f"{identity_template_digest} != historical pin {expected_identity_template}"
        )
    stage1 = sources["stage1-parity"]
    stage0_raw = stage0_source_bytes(stage1["raw"])
    stage0_digest = sha256_bytes(stage0_raw)
    expected_stage0 = EXPECTED_SOURCE_POLICY_SHA256["stage0-b2"]
    if stage0_digest != expected_stage0:
        raise RuntimeError(
            f"reconstructed stage0 source {stage0_digest} != historical pin "
            f"{expected_stage0}"
        )
    sources = {
        "stage0-b2": {
            "path": stage1["path"],
            "raw": stage0_raw,
            "policy": read_policy(stage0_raw, label="stage0-b2"),
            "source_derivation": {
                "base_path": str(stage1["path"]),
                "base_sha256": sha256_bytes(stage1["raw"]),
                "operation": {"add": {"/tensorfold_cohort_limit": 2}},
                "historical_output_sha256": stage0_digest,
            },
        },
        **sources,
    }
    abba_canonical = sources["abba-candidate"]
    abba_candidate_raw = abba_candidate_source_bytes(
        abba_canonical["raw"], abba_identity_template_raw
    )
    sources["abba-candidate"] = {
        "path": abba_canonical["path"],
        "raw": abba_candidate_raw,
        "policy": read_policy(abba_candidate_raw, label="abba-candidate"),
        "source_derivation": {
            "base_path": str(abba_canonical["path"]),
            "base_sha256": sha256_bytes(abba_canonical["raw"]),
            "identity_template_path": str(abba_identity_template_path),
            "identity_template_sha256": identity_template_digest,
            "operation": {
                "replace": ["/draft_revision", "/target_revision"]
            },
            "derived_source_sha256": sha256_bytes(abba_candidate_raw),
        },
    }
    abba_candidate = sources["abba-candidate"]
    abba_control_raw = abba_control_source_bytes(abba_candidate["raw"])
    sources["abba-control"] = {
        "path": abba_candidate["path"],
        "raw": abba_control_raw,
        "policy": read_policy(abba_control_raw, label="abba-control"),
        "source_derivation": {
            "base_path": str(abba_candidate["path"]),
            "base_sha256": sha256_bytes(abba_candidate["raw"]),
            "operation": {
                "replace": {"/varlen_dense_mlp/minimum_padding_fraction": 1.0}
            },
            "derived_source_sha256": sha256_bytes(abba_control_raw),
        },
    }
    pld_candidate = sources["pld-candidate"]
    pld_control_raw = pld_control_source_bytes(pld_candidate["raw"])
    sources["pld-control"] = {
        "path": pld_candidate["path"],
        "raw": pld_control_raw,
        "policy": read_policy(pld_control_raw, label="pld-control"),
        "source_derivation": {
            "base_path": str(pld_candidate["path"]),
            "base_sha256": sha256_bytes(pld_candidate["raw"]),
            "operation": {"remove": ["/proposal_composition"]},
            "derived_source_sha256": sha256_bytes(pld_control_raw),
        },
    }

    draft_paths = {
        str(Path(record["policy"]["draft_model"]).expanduser().resolve())
        for record in sources.values()
    }
    if len(draft_paths) != 1:
        raise RuntimeError(f"source policies disagree on draft_model: {sorted(draft_paths)}")
    model = args.model.expanduser().resolve()
    draft = Path(draft_paths.pop())
    tensorfold = args.tensorfold_source.expanduser().resolve()

    if _mlx_modules():
        raise RuntimeError(f"MLX was imported before host inspection: {_mlx_modules()}")
    blocker = _BlockMlxImports()
    sys.meta_path.insert(0, blocker)
    try:
        from mlx2.adapters.dflash2 import (
            _legacy_content_revision as legacy_draft_revision,
        )
        from mlx2.adapters.dflash2 import content_revision as draft_revision
        from mlx2.adapters.dflash2 import inspect_drafter
        from mlx2.adapters.qwen38_27b import (
            _inspect_target_content,
            inspect_artifact,
            inspect_external_policy,
        )
        from mlx2.adapters.qwen38_27b import (
            _legacy_content_revision as legacy_target_revision,
        )
        from mlx2.adapters.qwen38_tensorfold_source import (
            qualification_source_identity,
        )

        module_identities = require_local_modules(
            (
                "mlx2.adapters.dflash2",
                "mlx2.adapters.qwen38_27b",
                "mlx2.adapters.qwen38_tensorfold_source",
            )
        )

        tensorfold_identity = qualification_source_identity(tensorfold)
        drafter = inspect_drafter(draft, model)
        target_content = _inspect_target_content(model)
        target_artifact = inspect_artifact(model)["identity"]
        expected_target_artifact = literal_assignment(
            ABBA_HARNESS, "EXPECTED_MODEL_ARTIFACT"
        )
        expected_target_config = literal_assignment(
            ABBA_HARNESS, "EXPECTED_MODEL_CONFIG"
        )
        if target_artifact["fingerprint"] != expected_target_artifact:
            raise RuntimeError(
                f"target artifact {target_artifact['fingerprint']} != campaign pin "
                f"{expected_target_artifact}"
            )
        if target_content["config_sha256"] != expected_target_config:
            raise RuntimeError(
                f"target config {target_content['config_sha256']} != campaign pin "
                f"{expected_target_config}"
            )
        draft_provenance = json.loads(DRAFT_PROVENANCE.read_text())
        draft_source = next(
            source
            for source in draft_provenance["sources"]
            if source.get("source_repository")
            == "huggingface.co/incoai/Qwen3.8-27B-DFlash2"
        )
        identity = {
            "target_revision": target_content["revision"],
            "draft_revision": draft_revision(drafter),
            "legacy_target_revision": legacy_target_revision(model),
            "legacy_draft_revision": legacy_draft_revision(drafter),
        }
        if draft_source.get("content_revision_pin") != identity["draft_revision"]:
            raise RuntimeError(
                "live draft payload revision differs from pinned draft provenance"
            )

        generated = {}
        manifest_policies = []
        with inspection_environment(tensorfold):
            for label, source in sources.items():
                policy = source["policy"]
                if policy.get("target_revision") != identity["legacy_target_revision"]:
                    raise RuntimeError(f"{label}: target pin is not the exact legacy pin")
                if policy.get("draft_revision") != identity["legacy_draft_revision"]:
                    raise RuntimeError(f"{label}: draft pin is not the exact legacy pin")
                raw, proof = derive_policy_bytes(
                    source["raw"],
                    draft_revision=identity["draft_revision"],
                    target_revision=identity["target_revision"],
                    label=label,
                )
                value = read_policy(raw, label=f"generated {label}")
                inspected = inspect_external_policy(value, model)
                summary = validation_summary(inspected)
                if summary["target_revision"] != identity["target_revision"]:
                    raise RuntimeError(f"{label}: adapter returned another target revision")
                if summary["draft_revision"] != identity["draft_revision"]:
                    raise RuntimeError(f"{label}: adapter returned another draft revision")
                generated[label] = raw
                manifest_policies.append(
                    {
                        "role": label,
                        "used_by": (
                            ["stage0-b2-smoke"]
                            if label == "stage0-b2"
                            else (
                                ["stage1-8k-parity", "stage1-16k-parity"]
                                if label == "stage1-parity"
                                else (
                                    ["performance-abba"]
                                    if label.startswith("abba-")
                                    else ["pld-dflash-qualification"]
                                )
                            )
                        ),
                        "source_path": str(source["path"]),
                        "source_sha256": sha256_bytes(source["raw"]),
                        "source_derivation": source["source_derivation"],
                        "generated_path": str(output_dir / OUTPUT_NAMES[label]),
                        "generated_sha256": sha256_bytes(raw),
                        "source_pin_classification": "legacy-metadata-only",
                        "proof": proof,
                        "adapter_inspection": summary,
                    }
                )
    finally:
        sys.meta_path.remove(blocker)

    if _mlx_modules():
        raise RuntimeError(f"host inspection imported MLX: {_mlx_modules()}")
    for path, original in source_file_snapshots.items():
        current = sha256(path)
        expected = sha256_bytes(original)
        if current != expected:
            raise RuntimeError(
                f"source policy changed during inspection: {path}: "
                f"{current} != {expected}"
            )

    manifest = {
        "schema": "mlx2.qwen38-dflash-closeout-policy-derivation.v1",
        "source": {
            "root": str(ROOT),
            "commit": source_commit,
            "tree": git(ROOT, "rev-parse", "HEAD^{tree}"),
            "tracked_diff": tracked_diff,
            "importable_diff": importable_diff,
            "modules": module_identities,
        },
        "target": {
            "path": str(model),
            "payload_revision": identity["target_revision"],
            "legacy_metadata_revision": identity["legacy_target_revision"],
            "config_sha256": target_content["config_sha256"],
            "index_sha256": target_content["index_sha256"],
            "weights": target_content["weights"],
            "artifact_fingerprint": target_artifact["fingerprint"],
            "artifact_files": target_artifact["files"],
            "identity_evidence": {
                "path": str(ABBA_HARNESS),
                "sha256": sha256(ABBA_HARNESS),
                "expected_artifact_fingerprint": expected_target_artifact,
                "expected_config_sha256": expected_target_config,
            },
        },
        "draft": {
            "path": str(draft),
            "payload_revision": identity["draft_revision"],
            "legacy_metadata_revision": identity["legacy_draft_revision"],
            "config_sha256": drafter["config_sha256"],
            "index_sha256": drafter.get("index_sha256"),
            "weight_sha256": drafter["weight_sha256"],
            "artifact_fingerprint": drafter["fingerprint"],
            "identity_evidence": {
                "path": str(DRAFT_PROVENANCE),
                "sha256": sha256(DRAFT_PROVENANCE),
                "pinned_payload_revision": draft_source["content_revision_pin"],
                "source_revision": draft_source["source_revision"],
            },
        },
        "tensorfold": tensorfold_identity,
        "identity_fields": list(IDENTITY_FIELDS),
        "all_source_policies_share_exact_legacy_pins": True,
        "real_adapter_inspection": {
            "call": "mlx2.adapters.qwen38_27b.inspect_external_policy",
            "mlx_import_blocked": True,
            "mlx_modules_after": [],
        },
        "served_artifact_binding": {
            "inputs_bound_here": [
                "target artifact fingerprint and payload revision",
                "runtime-quantized draft fingerprint and payload revision",
                "exact generated policy bytes",
                "source revision and adapter inspector module hashes",
            ],
            "final_fingerprint": (
                "derived only after model construction from these inputs plus the "
                "selected adapter and serving numerical namespaces"
            ),
            "qualification_rule": (
                "do not substitute a historical final fingerprint; require a valid "
                "runtime SHA-256 identity, stability across repeated same-policy arms, "
                "and separation across control/candidate policies"
            ),
        },
        "policies": manifest_policies,
    }
    output_dir.mkdir(parents=True)
    for label, raw in generated.items():
        (output_dir / OUTPUT_NAMES[label]).write_bytes(raw)
    manifest_path = output_dir / "policy-derivation-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "target_revision": identity["target_revision"],
        "draft_revision": identity["draft_revision"],
        "policies": {
            label: sha256(output_dir / OUTPUT_NAMES[label]) for label in generated
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
