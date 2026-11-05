"""Run a synthetic NumPy network through the real repair controller, CPU only.

The deployment backend is a toy in-memory implementation. Predictions come
from actual matrix products, including the exported/loaded analytical factors;
there are no predetermined passing model responses or real model executions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from .programs import analytical_directional_repair, load_program
from .self_repair import (
    RepairPolicy,
    RepairSession,
    RepairSuite,
    SelfRepairController,
    ToolCase,
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


class ToyDeploymentBackend:
    """A synthetic deployment whose tool-argument class is a NumPy prediction."""

    def __init__(self, *, weights, inputs, checkpoint):
        self.weights = np.asarray(weights, dtype=np.float32).copy()
        self.inputs = {key: np.asarray(value, dtype=np.float32).copy() for key, value in inputs.items()}
        self.checkpoint = checkpoint
        self.current = {"generation": 0, "default_adapter": None, "fingerprint": None, "program_id": None}
        self.programs = {}
        self.execution_records = []

    def state(self):
        return copy.deepcopy(self.current)

    def register(self, name, path, program):
        # Consume the same exported artifact a serving backend would receive.
        loaded = load_program(path)
        if loaded.program_id != program.program_id or loaded.checkpoint != self.checkpoint:
            raise ValueError("toy registration identity mismatch")
        fingerprint = _sha((loaded.program_id + ":synthetic_numpy_float32").encode())
        candidate = {"name": name, "program_id": loaded.program_id, "fingerprint": fingerprint}
        self.programs[name] = (loaded, candidate)
        return dict(candidate)

    def deployed_program(self, previous):
        program, candidate = self.programs[previous["default_adapter"]]
        if candidate["fingerprint"] != previous["fingerprint"]:
            raise ValueError("toy deployed fingerprint changed")
        return program

    def _infer(self, request, candidate=None, *, replayed=False, phase):
        case_id = request["messages"][-1]["content"]
        x = self.inputs[case_id]
        base = x @ self.weights.T
        delta = np.zeros_like(base)
        applications = 0
        if candidate is not None:
            program, registered = self.programs[candidate["name"]]
            if registered != candidate:
                raise ValueError("toy candidate changed")
            a, b = program.factor_map["tool_projection"]
            delta = (x @ a) @ b
            applications += 1
        logits = base + delta
        if not np.isfinite(logits).all():
            raise ValueError("toy projection became nonfinite")
        selected = int(np.argmax(logits))
        argument_key = ("filename", "path")[selected]
        arguments = {argument_key: "demo.txt"}
        observation = {
            "checkpoint": self.checkpoint,
            "program_id": candidate["program_id"] if candidate else None,
            "fingerprint": candidate["fingerprint"] if candidate else None,
            "tool_name": "read_file", "arguments": arguments,
            "protocol_valid": True, "replayed": replayed,
            "engaged": applications > 0,
            "observation_scope": "synthetic_numpy_network",
        }
        self.execution_records.append({
            "phase": phase, "case_id": case_id, "input": x.tolist(),
            "base_logits": base.tolist(), "delta_logits": delta.tolist(),
            "logits": logits.tolist(), "selected_class": selected,
            "adapter_delta_applications": applications,
            "delta_l2_norm": float(np.linalg.norm(delta.astype(np.float64))),
            "observation": copy.deepcopy(observation),
        })
        return observation

    def baseline(self, request, previous):
        candidate = None
        if previous["default_adapter"] is not None:
            candidate = self.programs[previous["default_adapter"]][1]
            if any(candidate[key] != previous[key] for key in ("program_id", "fingerprint")):
                raise ValueError("toy baseline identity mismatch")
        return self._infer(request, candidate, phase="controller_baseline")

    def run(self, request, candidate):
        return self._infer(request, candidate, replayed=True, phase="controller_candidate_fresh_replay")

    def run_default(self, request, *, phase):
        name = self.current["default_adapter"]
        candidate = self.programs[name][1] if name is not None else None
        return self._infer(request, candidate, phase=phase)

    def promote(self, candidate, generation, receipt):
        if generation != self.current["generation"] or receipt.get("passed") is not True:
            raise ValueError("toy promotion CAS or validation failed")
        if any(receipt.get(key) != candidate[key] for key in ("program_id", "fingerprint")):
            raise ValueError("toy promotion evidence identity mismatch")
        self.current = {"generation": generation + 1, "default_adapter": candidate["name"],
                        "fingerprint": candidate["fingerprint"], "program_id": candidate["program_id"]}
        return self.state()

    def rollback(self, generation, previous):
        if generation != self.current["generation"]:
            raise ValueError("toy rollback CAS failed")
        self.current = {**copy.deepcopy(previous), "generation": generation + 1}
        return self.state()


class _AuditedController(SelfRepairController):
    def __init__(self, **kwargs):
        self.transitions = []
        super().__init__(**kwargs)

    def _save(self, attempt):
        super()._save(attempt)
        if not self.transitions or self.transitions[-1] != attempt["state"]:
            self.transitions.append(attempt["state"])


def run_demo(output, *, source_revision, strength=2.0):
    """Write an exclusive fresh evidence directory; retain incomplete runs."""
    if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", source_revision):
        raise ValueError("source_revision must be an exact lowercase hexadecimal source SHA")
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("CPU demo refuses to overwrite any existing output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    w = np.array([[2, 0], [0, 2]], dtype=np.float32)
    inputs = {
        "active": np.array([1, 0], dtype=np.float32),
        "heldout": np.array([0.5, 0.2], dtype=np.float32),
        "regression": np.array([0, 1], dtype=np.float32),
    }
    config = {
        "architecture": "two-input-two-class-linear",
        "dtype": "float32", "weight_layout": "output_by_input",
        "classes": {"0": "filename", "1": "path"},
        "tool": "read_file", "argument_value": "demo.txt",
        "weight_sha256": _sha(w.tobytes()),
    }
    checkpoint = "synthetic-tool-projection@" + _sha(_json(config).encode())
    schema = {"type": "object", "properties": {"path": {"type": "string"}},
              "required": ["path"], "additionalProperties": False}
    schema_digest = _sha(_json(schema).encode())
    request = {"messages": [{"role": "user", "content": "active"}]}
    backend = ToyDeploymentBackend(weights=w, inputs=inputs, checkpoint=checkpoint)
    session = RepairSession("synthetic-chat", copy.deepcopy(request))
    expected = {"path": "demo.txt"}

    def oracle(name, arguments):
        return name == "read_file" and isinstance(arguments, dict) and arguments == expected

    def case(name):
        return ToolCase.create(name, [{"role": "user", "content": name}], "read_file", expected)

    controller = _AuditedController(
        checkpoint=checkpoint, source_revision=source_revision, root=output / "controller",
        policy=RepairPolicy(enabled=True, execution_enabled=True, automatic_promotion=True),
        backend=backend, verify_tool=oracle,
        suites=[RepairSuite("read_file", schema_digest, (case("heldout"),), (case("regression"),))],
        sessions=[session],
    )
    initial = backend.run_default(request, phase="before_repair")
    direction = np.array([1, -1], dtype=np.float32) / np.sqrt(np.float32(2))
    program = analytical_directional_repair(
        {"tool_projection": w}, {"tool_projection": direction}, {"tool_projection": [0]},
        strength=strength, checkpoint=checkpoint, source_revision=source_revision,
        provenance={"source": "synthetic analytical fixture", "model_checkpoint_run": None,
                    "goal": "repair incorrect tool argument key while preserving the second input"},
    )
    failure = controller.report_failure({
        "tool_name": "read_file", "tool_schema_digest": schema_digest,
        "failed_arguments": initial["arguments"], "corrected_arguments": expected,
        "error_kind": "arguments", "observed_error": "required path; unexpected filename",
    }, session_id=session.session_id)
    proposed = controller.propose(failure["id"], program)
    trial = controller.trial(failure["id"])
    promoted_state = backend.state()
    after_trial = backend.run_default(request, phase="after_trial_deployment_default")
    rollback = None
    if trial["state"] == "promoted":
        rollback = controller.rollback(failure["id"], "synthetic demonstration restores the original model")
    after_rollback = backend.run_default(request, phase="after_rollback")
    fixture_path = output / "fixture.npz"
    np.savez(fixture_path, weights=w, direction=direction, **inputs)
    source_files = [Path(__file__), Path(__file__).with_name("programs.py"), Path(__file__).with_name("self_repair.py")]
    snapshots = {str(path.resolve()): _sha(path.read_bytes()) for path in source_files}
    report = {
        "schema": "mlx2.weight-space-cpu-demo.v1",
        "scope": "synthetic_numpy_network_only",
        "model_checkpoint_run": None, "gpu": False,
        "mlx_imported": any(name == "mlx" or name.startswith("mlx.") for name in sys.modules),
        "real_model_qualified": False, "real_model_observed_used": False,
        "checkpoint": checkpoint, "source_revision": source_revision,
        "source_revision_origin": "caller_supplied_commit_with_current_source_snapshot_hashes",
        "source_snapshot_sha256": snapshots,
        "fixture": {**config, "npz_sha256": _sha(fixture_path.read_bytes()),
                    "input_sha256": {key: _sha(value.tobytes()) for key, value in inputs.items()}},
        "strength": float(strength), "attempt_id": failure["id"],
        "candidate_program_id": proposed["program_id"],
        "transition_sequence": controller.transitions,
        "before_repair": initial, "after_trial": after_trial, "after_rollback": after_rollback,
        "trial_state": trial["state"], "validation": trial["validation"],
        "promoted_deployment": promoted_state if trial["state"] == "promoted" else None,
        "final_deployment": backend.state(), "session_after": session.snapshot(),
        "rollback_completed": rollback is not None and rollback["state"] == "rolled_back",
        "executions": backend.execution_records,
        "limits": [
            "The deployment backend is synthetic and in memory, not a running mlx2 server.",
            "The fixed two-class network tests algebra and controller transitions, not language-model behavior.",
            "No real checkpoint, APCv2 state, GPU path or production deployment was exercised.",
        ],
    }
    _atomic_json(output / "receipt.json", report)
    descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--strength", type=float, default=2.0,
                        help="directional strength; a negative value provides a rejected-candidate control")
    args = parser.parse_args(argv)
    try:
        result = run_demo(args.output, source_revision=args.source_revision, strength=args.strength)
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"receipt": str(Path(args.output).expanduser().absolute() / "receipt.json"),
                      "trial_state": result["trial_state"],
                      "transition_sequence": result["transition_sequence"],
                      "scope": result["scope"], "gpu": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
