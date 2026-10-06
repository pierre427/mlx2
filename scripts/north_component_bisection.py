"""Host-safe component oracle for North B1-versus-batched diagnostics.

The harness owns evidence plumbing, not tensor math.  A native diagnostic gives
each stage two callables: one batched calculation and one concatenation of
independent one-row calculations.  The harness clones their common payload,
checks that the arms share no mutable array leaves, preserves the source
payload, and emits a fail-closed complete-stage receipt.

Importing this file does not import MLX.  A future Metal diagnostic may provide
an MLX-aware clone callback and return evaluated NumPy-compatible observations.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

SCHEMA = "mlx2.north-component-bisection.v1"
STAGES = (
    "q4_projection",
    "q8_router",
    "expert_gather_frozen",
    "sdpa",
    "tied_head",
)

_REQUIRED_EVIDENCE = {
    "q4_projection": {
        "shared": {"bits": 4, "group_size": 64},
    },
    "q8_router": {
        "shared": {"bits": 8, "group_size": 64},
    },
    "expert_gather_frozen": {
        "shared": {"routes_frozen": True},
    },
    "sdpa": {
        "shared": {"cache_inputs_frozen": True},
    },
    "tied_head": {
        "shared": {"tied": True, "bits": 4, "group_size": 64},
    },
}

for _requirements in _REQUIRED_EVIDENCE.values():
    _requirements["batched"] = {"batched_rows": 4, "rowwise_calls": 0}
    _requirements["rowwise"] = {"batched_rows": 0, "rowwise_calls": 4}

_MATCHING_EVIDENCE = {
    "expert_gather_frozen": ("route_ids_digest", "route_weights_digest"),
    "sdpa": ("preappend_cache_digest",),
}


@dataclasses.dataclass(frozen=True)
class Observation:
    """One arm's evaluated result and mechanism evidence."""

    output: Any
    state: Any = None
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return np.asarray(value)
        except (TypeError, ValueError):
            return None
    return None


def _walk(value: Any, path: str = "root"):
    array = _array(value)
    if array is not None:
        yield path, array
        return
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name), f"{path}.{field.name}")
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            yield from _walk(value[key], f"{path}.{key}")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
        return
    yield path, value


def tree_digest(value: Any) -> str:
    """Stable digest for diagnostic payloads, including exact array bytes."""

    digest = hashlib.sha256()
    for path, leaf in _walk(value):
        digest.update(path.encode())
        digest.update(b"\0")
        array = _array(leaf)
        if array is not None:
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(json.dumps(contiguous.shape).encode())
            digest.update(contiguous.view(np.uint8).tobytes())
        else:
            digest.update(
                json.dumps(
                    leaf,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=repr,
                ).encode()
            )
        digest.update(b"\n")
    return digest.hexdigest()


def _mutable_array_ids(value: Any) -> set[int]:
    array = _array(value)
    if array is not None:
        return {id(value)}
    if dataclasses.is_dataclass(value):
        return {
            item
            for field in dataclasses.fields(value)
            for item in _mutable_array_ids(getattr(value, field.name))
        }
    if isinstance(value, Mapping):
        return {item for child in value.values() for item in _mutable_array_ids(child)}
    if isinstance(value, (tuple, list)):
        return {item for child in value for item in _mutable_array_ids(child)}
    return set()


def _compare(left: Any, right: Any) -> dict[str, Any]:
    left_leaves = list(_walk(left))
    right_leaves = list(_walk(right))
    if [path for path, _ in left_leaves] != [path for path, _ in right_leaves]:
        return {
            "exact": False,
            "comparable": False,
            "reason": "tree_structure",
            "max_abs": None,
        }
    exact = True
    max_abs = 0.0
    numeric = False
    for (path, lhs), (_, rhs) in zip(left_leaves, right_leaves):
        lhs_array, rhs_array = _array(lhs), _array(rhs)
        if (lhs_array is None) != (rhs_array is None):
            return {
                "exact": False,
                "comparable": False,
                "reason": f"leaf_kind:{path}",
                "max_abs": None,
            }
        if lhs_array is None:
            exact &= lhs == rhs
            continue
        if lhs_array.shape != rhs_array.shape or lhs_array.dtype != rhs_array.dtype:
            return {
                "exact": False,
                "comparable": False,
                "reason": f"array_contract:{path}",
                "max_abs": None,
            }
        exact &= bool(np.array_equal(lhs_array, rhs_array, equal_nan=True))
        if np.issubdtype(lhs_array.dtype, np.number):
            numeric = True
            if lhs_array.size:
                delta = np.abs(
                    lhs_array.astype(np.float64) - rhs_array.astype(np.float64)
                )
                finite = delta[np.isfinite(delta)]
                if finite.size:
                    max_abs = max(max_abs, float(finite.max()))
                elif not np.array_equal(lhs_array, rhs_array, equal_nan=True):
                    max_abs = float("inf")
    return {
        "exact": bool(exact),
        "comparable": True,
        "reason": None,
        "max_abs": max_abs if numeric else None,
    }


def _validate_evidence(
    stage: str,
    batched: Mapping[str, Any],
    rowwise: Mapping[str, Any],
) -> list[str]:
    failures = []
    required = _REQUIRED_EVIDENCE[stage]
    for arm_name, evidence in (("batched", batched), ("rowwise", rowwise)):
        arm_required = {**required["shared"], **required[arm_name]}
        for key, expected in arm_required.items():
            if evidence.get(key) != expected:
                failures.append(f"{arm_name}.{key}")
    for key in _MATCHING_EVIDENCE.get(stage, ()):
        if not batched.get(key) or batched.get(key) != rowwise.get(key):
            failures.append(f"matching.{key}")
    return failures


class NorthComponentBisection:
    """Collect all required component observations before issuing a verdict."""

    def __init__(self, *, clone: Callable[[Any], Any] = copy.deepcopy):
        self._clone = clone
        self._stages: dict[str, dict[str, Any]] = {}

    def observe(
        self,
        stage: str,
        payload: Any,
        batched: Callable[[Any], Observation],
        rowwise: Callable[[Any], Observation],
    ) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown North component stage {stage!r}")
        if stage in self._stages:
            raise RuntimeError(f"North component stage {stage!r} already observed")

        source_digest = tree_digest(payload)
        batch_payload = self._clone(payload)
        row_payload = self._clone(payload)
        independent = not (
            _mutable_array_ids(batch_payload) & _mutable_array_ids(row_payload)
        )
        receipt: dict[str, Any] = {
            "executed": False,
            "complete": False,
            "exact": False,
            "input_unchanged": False,
            "independent_inputs": independent,
            "evidence_failures": [],
            "output": None,
            "state": None,
            "error": None,
        }
        try:
            batch_observation = batched(batch_payload)
            row_observation = rowwise(row_payload)
            if not isinstance(batch_observation, Observation) or not isinstance(
                row_observation, Observation
            ):
                raise TypeError("component arms must return Observation")
            receipt["executed"] = True
            receipt["input_unchanged"] = tree_digest(payload) == source_digest
            receipt["evidence_failures"] = _validate_evidence(
                stage,
                batch_observation.evidence,
                row_observation.evidence,
            )
            receipt["output"] = _compare(
                batch_observation.output, row_observation.output
            )
            if (batch_observation.state is None) != (row_observation.state is None):
                receipt["state"] = {
                    "exact": False,
                    "comparable": False,
                    "reason": "one_sided_state",
                    "max_abs": None,
                }
            elif batch_observation.state is None:
                receipt["state"] = None
            else:
                receipt["state"] = _compare(
                    batch_observation.state, row_observation.state
                )
            state_required = stage == "sdpa"
            state_complete = (
                receipt["state"] is not None
                and receipt["state"]["comparable"]
                if state_required
                else receipt["state"] is None or receipt["state"]["comparable"]
            )
            receipt["complete"] = bool(
                receipt["executed"]
                and receipt["independent_inputs"]
                and receipt["input_unchanged"]
                and not receipt["evidence_failures"]
                and receipt["output"]["comparable"]
                and state_complete
            )
            receipt["exact"] = bool(
                receipt["complete"]
                and receipt["output"]["exact"]
                and (receipt["state"] is None or receipt["state"]["exact"])
            )
            receipt["batched_evidence"] = dict(batch_observation.evidence)
            receipt["rowwise_evidence"] = dict(row_observation.evidence)
        except Exception as error:  # noqa: BLE001 - receipt must fail closed
            receipt["error"] = f"{type(error).__name__}: {error}"
        self._stages[stage] = receipt
        return copy.deepcopy(receipt)

    def receipt(self) -> dict[str, Any]:
        missing = [stage for stage in STAGES if stage not in self._stages]
        incomplete = [
            stage
            for stage in STAGES
            if stage in self._stages and not self._stages[stage]["complete"]
        ]
        complete = not missing and not incomplete
        divergent = [
            stage
            for stage in STAGES
            if stage in self._stages
            and self._stages[stage]["complete"]
            and not self._stages[stage]["exact"]
        ]
        return {
            "schema": SCHEMA,
            "semantics": (
                "component-local B1-versus-batched exactness diagnostic; "
                "not qualification, selection, or performance evidence"
            ),
            "required_stages": list(STAGES),
            "stages": copy.deepcopy(self._stages),
            "missing_stages": missing,
            "incomplete_stages": incomplete,
            "complete": complete,
            "width_sensitive_stages": divergent if complete else [],
            "first_width_sensitive_stage": divergent[0] if complete and divergent else None,
            "verdict": (
                "incomplete"
                if not complete
                else "width_sensitive"
                if divergent
                else "width_invariant"
            ),
        }


def describe() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "required_stages": list(STAGES),
        "required_evidence": copy.deepcopy(_REQUIRED_EVIDENCE),
        "matching_evidence": {
            stage: list(fields) for stage, fields in _MATCHING_EVIDENCE.items()
        },
        "sdpa_requires_state_comparison": True,
        "imports_mlx": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)
    if not args.describe:
        parser.error("this host-safe helper currently supports --describe only")
    print(json.dumps(describe(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
