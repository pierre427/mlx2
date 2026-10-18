"""Model-neutral, evidence-gated post-load profile selection.

This module validates and selects evidence.  It never mutates a live route:
construction-time choices are applied before load and only attested here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform as platform_module
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SCHEMA = "mlx2.adaptive-profile.v2"
LIVE_BINDING_SCHEMA = "mlx2.adaptive-profile-binding.v2"
OBJECTIVES = frozenset({"throughput", "efficiency", "power_budget"})
OUTCOMES = frozenset({"passed", "failed", "refused"})
_DIGEST_FIELDS = frozenset(
    {"qualification_receipt_sha256", "runtime_sha256", "settings_sha256"}
)


def _object(value, name):
    if not isinstance(value, dict):
        raise TypeError(f"adaptive profile {name} must be an object")
    return json.loads(json.dumps(value, sort_keys=True))


def _canonical_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"adaptive profile {name} must be a lowercase sha256")
    return value


def _positive(value, name, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"adaptive profile {name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"adaptive profile {name} must be finite and positive")
    return value


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"adaptive profile {name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"adaptive profile {name} must be finite and nonnegative")
    return value


def tuning_fingerprint(*, artifact, topology, platform, serving_shape, telemetry):
    payload = {
        "artifact": _object(artifact, "artifact"),
        "topology": _object(topology, "topology"),
        "platform": _object(platform, "platform"),
        "serving_shape": _object(serving_shape, "serving_shape"),
        "telemetry": _object(telemetry, "telemetry"),
    }
    return {"sha256": _canonical_sha256(payload), **payload}


@lru_cache(maxsize=1)
def _darwin_hardware_identity():
    """Stable performance topology, deliberately excluding the hostname."""
    if platform_module.system() != "Darwin":
        return {}
    result = {}
    for key in (
        "machdep.cpu.brand_string",
        "hw.model",
        "hw.physicalcpu",
        "hw.logicalcpu",
    ):
        try:
            value = subprocess.check_output(
                ["sysctl", "-n", key], text=True, timeout=2
            ).strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if value:
            result[key] = int(value) if value.isdigit() else value
    return result


def platform_fingerprint(*, memory_bytes=None, hardware=None):
    hardware = _darwin_hardware_identity() if hardware is None else hardware
    return {
        "system": platform_module.system(),
        "machine": platform_module.machine(),
        "os_release": platform_module.release(),
        "memory_bytes": memory_bytes,
        "logical_cpu_count": os.cpu_count(),
        "hardware": _object(hardware, "hardware"),
    }


def qualification_evidence(path, *, runtime, settings):
    return {
        "qualification_receipt_sha256": _file_sha256(path),
        "runtime_sha256": _canonical_sha256(_object(runtime, "runtime")),
        "settings_sha256": _canonical_sha256(_object(settings, "settings")),
    }


def _qualification(value):
    value = _object(value, "candidate qualification")
    if set(value) != _DIGEST_FIELDS:
        raise ValueError(
            "adaptive profile candidate qualification requires exact receipt, "
            "runtime, and settings digests"
        )
    return {name: _digest(value[name], name) for name in sorted(_DIGEST_FIELDS)}


@dataclass(frozen=True)
class ProfileCandidate:
    id: str
    policy: dict
    qualification: dict | None

    @property
    def qualified(self):
        return self.qualification is not None

    def as_dict(self):
        return {
            "id": self.id,
            "policy": self.policy,
            "qualification": self.qualification,
        }

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "id",
            "policy",
            "qualification",
        }:
            raise ValueError(
                "adaptive profile candidates require id, policy, and qualification"
            )
        identifier = value["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("adaptive profile candidate id must be nonempty")
        qualification = value["qualification"]
        if qualification is not None:
            qualification = _qualification(qualification)
        return cls(
            identifier,
            _object(value["policy"], "candidate policy"),
            qualification,
        )


def _telemetry_sample(value, *, max_sample_age_seconds):
    if value is None:
        return None, ["telemetry_missing"]
    value = _object(value, "observation telemetry")
    required = {
        "power_complete",
        "power_sample_age_seconds",
        "thermal_state",
        "thermal_sample_age_seconds",
        "gpu_dram_watts",
    }
    if set(value) != required:
        raise ValueError(
            "adaptive profile telemetry requires complete power, thermal, age, "
            "and GPU+DRAM watts fields"
        )
    if type(value["power_complete"]) is not bool:
        raise TypeError("adaptive profile power_complete must be boolean")
    power_age = _nonnegative(
        value["power_sample_age_seconds"], "power_sample_age_seconds"
    )
    thermal_age = _nonnegative(
        value["thermal_sample_age_seconds"], "thermal_sample_age_seconds"
    )
    watts = _positive(value["gpu_dram_watts"], "gpu_dram_watts")
    thermal = value["thermal_state"]
    if not isinstance(thermal, str):
        raise TypeError("adaptive profile thermal_state must be a string")
    reasons = []
    if not value["power_complete"]:
        reasons.append("power_incomplete")
    if power_age > max_sample_age_seconds:
        reasons.append("power_stale")
    if thermal_age > max_sample_age_seconds:
        reasons.append("thermal_stale")
    if thermal != "nominal":
        reasons.append("thermal_not_nominal")
    return {
        **value,
        "power_sample_age_seconds": power_age,
        "thermal_sample_age_seconds": thermal_age,
        "gpu_dram_watts": watts,
    }, reasons


class PostLoadAutotuner:
    def __init__(
        self,
        fingerprint,
        candidates,
        *,
        baseline_id,
        minimum_samples=1,
        max_throughput_regression=0.0,
        objective="throughput",
        power_budget_watts=None,
        max_sample_age_seconds=5.0,
    ):
        self.fingerprint = _object(fingerprint, "fingerprint")
        body = {
            key: value for key, value in self.fingerprint.items() if key != "sha256"
        }
        if self.fingerprint.get("sha256") != _canonical_sha256(body):
            raise ValueError("adaptive profile fingerprint digest differs")
        rows = tuple(ProfileCandidate.from_value(row) for row in candidates)
        if not rows or len({row.id for row in rows}) != len(rows):
            raise ValueError("adaptive profile candidate ids must be unique")
        self.candidates = {row.id: row for row in rows}
        if baseline_id not in self.candidates:
            raise ValueError("adaptive profile baseline is not a candidate")
        if not self.candidates[baseline_id].qualified:
            raise ValueError("adaptive profile baseline must be qualified")
        if type(minimum_samples) is not int or minimum_samples < 1:
            raise ValueError("adaptive profile minimum_samples must be positive")
        if (
            isinstance(max_throughput_regression, bool)
            or not isinstance(max_throughput_regression, (int, float))
            or not 0 <= float(max_throughput_regression) < 1
        ):
            raise ValueError("adaptive profile regression bound must be in [0, 1)")
        if objective not in OBJECTIVES:
            raise ValueError("adaptive profile objective is unsupported")
        if objective == "power_budget":
            power_budget_watts = _positive(power_budget_watts, "power_budget_watts")
        elif power_budget_watts is not None:
            raise ValueError("power_budget_watts requires the power_budget objective")
        self.max_sample_age_seconds = _positive(
            max_sample_age_seconds, "max_sample_age_seconds"
        )
        self.baseline_id = baseline_id
        self.minimum_samples = minimum_samples
        self.max_throughput_regression = float(max_throughput_regression)
        self.objective = objective
        self.power_budget_watts = power_budget_watts
        self.observations = []

    def observe(
        self,
        profile_id,
        *,
        outcome="passed",
        correct=None,
        tokens_per_second=None,
        joules_per_token=None,
        contaminated=False,
        telemetry=None,
        failure_reason=None,
        receipt=None,
    ):
        if profile_id not in self.candidates:
            raise ValueError("adaptive profile observation names an unknown candidate")
        if outcome not in OUTCOMES:
            raise ValueError("adaptive profile observation outcome is unsupported")
        if not isinstance(contaminated, bool):
            raise TypeError("adaptive profile contamination must be boolean")
        if outcome == "passed":
            if not isinstance(correct, bool):
                raise TypeError(
                    "a passed adaptive profile observation needs correctness"
                )
            throughput = _positive(tokens_per_second, "tokens_per_second")
            energy = _positive(joules_per_token, "joules_per_token", optional=True)
            if failure_reason is not None:
                raise ValueError("a passed observation cannot have a failure reason")
        else:
            if correct not in (None, False):
                raise ValueError("failed/refused observations cannot be correct")
            if not isinstance(failure_reason, str) or not failure_reason:
                raise ValueError("failed/refused observations need a failure reason")
            correct = False
            throughput = _positive(
                tokens_per_second, "tokens_per_second", optional=True
            )
            energy = _positive(joules_per_token, "joules_per_token", optional=True)
        telemetry, invalid = _telemetry_sample(
            telemetry, max_sample_age_seconds=self.max_sample_age_seconds
        )
        if outcome != "passed":
            invalid.append(outcome)
        if not correct:
            invalid.append("incorrect")
        if contaminated:
            invalid.append("contaminated")
        self.observations.append(
            {
                "profile_id": profile_id,
                "outcome": outcome,
                "correct": correct,
                "tokens_per_second": throughput,
                "joules_per_token": energy,
                "contaminated": contaminated,
                "telemetry": telemetry,
                "failure_reason": failure_reason,
                "valid_for_selection": not invalid,
                "invalid_reasons": sorted(set(invalid)),
                "receipt": _object(receipt or {}, "observation receipt"),
            }
        )

    def _summaries(self):
        result = {}
        for identifier, candidate in self.candidates.items():
            rows = [
                row
                for row in self.observations
                if row["profile_id"] == identifier and row["valid_for_selection"]
            ]
            rates = [row["tokens_per_second"] for row in rows]
            energy = [
                row["joules_per_token"]
                for row in rows
                if row["joules_per_token"] is not None
            ]
            watts = [row["telemetry"]["gpu_dram_watts"] for row in rows]
            result[identifier] = {
                "qualified": candidate.qualified,
                "qualification": candidate.qualification,
                "valid_samples": len(rows),
                "tokens_per_second_median": statistics.median(rates) if rates else None,
                "joules_per_token_median": statistics.median(energy)
                if energy
                else None,
                "gpu_dram_watts_median": statistics.median(watts) if watts else None,
            }
        return result

    def _selection(self, summaries):
        baseline = summaries[self.baseline_id]
        eligible = []
        if baseline["valid_samples"] >= self.minimum_samples:
            floor = baseline["tokens_per_second_median"] * (
                1 - self.max_throughput_regression
            )
            for identifier, summary in summaries.items():
                if (
                    summary["qualified"]
                    and summary["valid_samples"] >= self.minimum_samples
                    and summary["tokens_per_second_median"] >= floor
                    and (
                        self.objective != "efficiency"
                        or summary["joules_per_token_median"] is not None
                    )
                    and (
                        self.objective != "power_budget"
                        or summary["gpu_dram_watts_median"] <= self.power_budget_watts
                    )
                ):
                    eligible.append(identifier)
        if not eligible:
            return self.baseline_id, "baseline_fallback_no_eligible_profile", False
        if self.objective == "efficiency":
            key = lambda name: (
                summaries[name]["joules_per_token_median"],
                -summaries[name]["tokens_per_second_median"],
                name,
            )
            reason = "lowest_energy_no_regression_qualified_profile"
        elif self.objective == "power_budget":
            key = lambda name: (
                -summaries[name]["tokens_per_second_median"],
                summaries[name]["gpu_dram_watts_median"],
                name,
            )
            reason = "fastest_no_regression_profile_within_power_budget"
        else:
            key = lambda name: (
                -summaries[name]["tokens_per_second_median"],
                summaries[name]["joules_per_token_median"]
                if summaries[name]["joules_per_token_median"] is not None
                else math.inf,
                name,
            )
            reason = "fastest_no_regression_qualified_profile"
        return min(eligible, key=key), reason, True

    def finalize(self):
        summaries = self._summaries()
        selected, reason, valid = self._selection(summaries)
        body = {
            "schema": SCHEMA,
            "fingerprint": self.fingerprint,
            "candidates": [
                self.candidates[name].as_dict() for name in sorted(self.candidates)
            ],
            "baseline_id": self.baseline_id,
            "minimum_samples": self.minimum_samples,
            "max_throughput_regression": self.max_throughput_regression,
            "objective": self.objective,
            "power_budget_watts": self.power_budget_watts,
            "max_sample_age_seconds": self.max_sample_age_seconds,
            "selected_profile_id": selected,
            "selected_policy": dict(self.candidates[selected].policy),
            "selection_reason": reason,
            "selection_valid": valid,
            "summaries": summaries,
            "observations": list(self.observations),
            "thermally_controlled": valid
            and all(
                row["telemetry"].get("thermal_state") == "nominal"
                for row in self.observations
                if row["valid_for_selection"]
            ),
        }
        return {**body, "integrity_sha256": _canonical_sha256(body)}


def _recompute_receipt(value):
    tuner = PostLoadAutotuner(
        value["fingerprint"],
        value["candidates"],
        baseline_id=value["baseline_id"],
        minimum_samples=value["minimum_samples"],
        max_throughput_regression=value["max_throughput_regression"],
        objective=value["objective"],
        power_budget_watts=value.get("power_budget_watts"),
        max_sample_age_seconds=value["max_sample_age_seconds"],
    )
    for row in value["observations"]:
        tuner.observe(
            row["profile_id"],
            outcome=row["outcome"],
            correct=row["correct"],
            tokens_per_second=row["tokens_per_second"],
            joules_per_token=row["joules_per_token"],
            contaminated=row["contaminated"],
            telemetry=row["telemetry"],
            failure_reason=row["failure_reason"],
            receipt=row["receipt"],
        )
    return tuner.finalize()


def validate_receipt(receipt, fingerprint=None):
    value = _object(receipt, "receipt")
    if value.get("schema") != SCHEMA:
        raise ValueError("adaptive profile receipt schema differs")
    body = {key: item for key, item in value.items() if key != "integrity_sha256"}
    if value.get("integrity_sha256") != _canonical_sha256(body):
        raise ValueError("adaptive profile receipt integrity digest differs")
    if fingerprint is not None and value.get("fingerprint") != fingerprint:
        raise ValueError(
            "adaptive profile receipt fingerprint does not match this route"
        )
    try:
        recomputed = _recompute_receipt(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("adaptive profile receipt evidence is malformed") from error
    if recomputed != value:
        raise ValueError("adaptive profile receipt derived evidence differs")
    return value


def write_receipt(path, receipt):
    value = validate_receipt(receipt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def load_receipt(path, fingerprint):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return validate_receipt(json.loads(path.read_text()), fingerprint)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def live_tuning_fingerprint(
    *,
    adapter,
    runtime,
    settings,
    serving_shape,
    telemetry,
    memory_bytes=None,
    hardware=None,
):
    identity = getattr(adapter, "identity", None)
    artifact = identity.get("fingerprint") if isinstance(identity, dict) else None
    if artifact is None:
        raise ValueError("adaptive profile adapter has no artifact fingerprint")
    declared = getattr(adapter, "adaptive_profile_topology", None)
    declared = declared() if callable(declared) else {}
    descriptor = getattr(adapter, "descriptor", None)
    model = getattr(adapter, "model", None)
    topology = {
        "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "model": f"{type(model).__module__}.{type(model).__qualname__}",
        "layout": getattr(adapter, "layout", None),
        "capabilities": sorted(
            str(value) for value in getattr(descriptor, "capabilities", ())
        ),
        "declared": _object(declared, "adapter topology"),
        "route": settings.get("route"),
        "speculation": settings.get("speculation"),
        "process_env": _object(settings.get("process_env", {}), "process env"),
        "runtime": _object(runtime, "runtime"),
    }
    artifacts = {
        "route": artifact,
        "target": identity.get("target_fingerprint", artifact),
        "draft": identity.get("draft_fingerprint"),
        "sidecars": _object(identity.get("sidecar_fingerprints", {}), "sidecars"),
    }
    return tuning_fingerprint(
        artifact=artifacts,
        topology=topology,
        platform=platform_fingerprint(memory_bytes=memory_bytes, hardware=hardware),
        serving_shape=serving_shape,
        telemetry=telemetry,
    )


def live_telemetry_evidence(status, *, max_sample_age_seconds):
    if not isinstance(status, dict) or status.get("available") is not True:
        raise ValueError("adaptive profile requires available power telemetry")
    last, signal = status.get("last"), status.get("governor_signal")
    if not isinstance(last, dict) or not isinstance(signal, dict):
        raise TypeError("adaptive profile requires a completed telemetry sample")
    watts = last.get("watts")
    if not isinstance(watts, dict):
        raise TypeError("adaptive profile telemetry has no power domains")
    gpu, dram = watts.get("gpu"), watts.get("dram")
    if gpu is None or dram is None:
        raise ValueError("adaptive profile telemetry has incomplete GPU+DRAM power")
    evidence, reasons = _telemetry_sample(
        {
            "power_complete": signal.get("complete") is True,
            "power_sample_age_seconds": status.get("last_sample_age_seconds"),
            "thermal_state": last.get("system_thermal_state"),
            "thermal_sample_age_seconds": status.get("last_sample_age_seconds"),
            "gpu_dram_watts": float(gpu) + float(dram),
        },
        max_sample_age_seconds=max_sample_age_seconds,
    )
    if reasons:
        raise ValueError(
            "adaptive profile telemetry is not reusable: " + ", ".join(reasons)
        )
    return evidence


def bind_live_profile(
    path,
    fingerprint,
    *,
    active_execution_policy,
    qualification_state,
    qualification_binding,
    telemetry_status,
):
    if qualification_state != "qualified":
        raise ValueError(
            "adaptive profile reuse requires a matching route qualification"
        )
    receipt = load_receipt(path, fingerprint)
    if receipt is None:
        raise ValueError("adaptive profile receipt fingerprint or integrity differs")
    if receipt.get("selection_valid") is not True:
        raise ValueError("adaptive profile receipt has no valid selection")
    selected = receipt["selected_profile_id"]
    candidates = {row["id"]: row for row in receipt["candidates"]}
    if candidates[selected]["qualification"] != _qualification(qualification_binding):
        raise ValueError("adaptive profile candidate qualification binding differs")
    live_telemetry_evidence(
        telemetry_status, max_sample_age_seconds=receipt["max_sample_age_seconds"]
    )
    selected_policy = receipt["selected_policy"]
    if (
        not isinstance(selected_policy, dict)
        or set(selected_policy) != {"execution_policy"}
        or not isinstance(selected_policy["execution_policy"], dict)
    ):
        raise ValueError(
            "live adaptive profiles must select exactly one execution_policy object"
        )
    active = _object(active_execution_policy or {}, "active execution policy")
    if selected_policy["execution_policy"] != active:
        raise ValueError(
            "adaptive profile selected policy is not active; restart with the selected execution_policy before binding the receipt"
        )
    summary = receipt["summaries"][selected]
    return {
        "schema": LIVE_BINDING_SCHEMA,
        "state": "reused",
        "selected": True,
        "defaulted": False,
        "observed_used": False,
        "applied_before_bind": True,
        "runtime_mutation": False,
        "qualification": "qualified",
        "receipt_path": str(Path(path)),
        "fingerprint_sha256": fingerprint["sha256"],
        "selected_profile_id": selected,
        "selected_policy": selected_policy,
        "selection_reason": receipt["selection_reason"],
        "objective": receipt["objective"],
        "valid_samples": summary["valid_samples"],
        "thermally_controlled": True,
    }
