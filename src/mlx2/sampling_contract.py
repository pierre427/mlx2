"""Explicit source-owned route sampling scope; absent contracts preserve legacy."""

from __future__ import annotations

GREEDY_REFUSAL_SAMPLING_CONTRACT = {
    "schema": "mlx2.route-sampling-contract.v1",
    "modes": ["greedy"],
    "unsupported": "refuse",
}
GREEDY_SAMPLING_CHECKS = frozenset(
    {"greedy_sampling_controls", "sampling_refusal", "greedy_repeat"}
)


def normalize_sampling_contract(value):
    if value is None:
        return None
    if (
        type(value) is not dict
        or value != GREEDY_REFUSAL_SAMPLING_CONTRACT
        or set(value) != set(GREEDY_REFUSAL_SAMPLING_CONTRACT)
    ):
        raise ValueError("malformed or unsupported route sampling contract")
    return {
        "schema": value["schema"],
        "modes": list(value["modes"]),
        "unsupported": value["unsupported"],
    }


def descriptor_sampling_contract(descriptor):
    return normalize_sampling_contract(
        (getattr(descriptor, "metadata", None) or {}).get("sampling_contract")
    )


def sampling_coverage(contract, *, runtime, artifact):
    contract = normalize_sampling_contract(contract)
    if contract is None:
        return None
    return {
        "contract": contract,
        "supported": ["greedy"],
        "unsupported": ["stochastic"],
        "stochastic_tested": False,
        "unsupported_behavior": "http400_refusal",
        "binding": {"runtime": runtime, "artifact": artifact},
    }


def status_sampling_contract(status):
    contract = normalize_sampling_contract(
        (status.get("settings") or {}).get("sampling_contract")
    )
    if contract is not None:
        profiles = (status.get("sampling_defaults") or {}).get("profiles")
        if (
            not isinstance(profiles, dict)
            or not profiles
            or any(
                type(row) is not dict
                or type(row.get("values")) is not dict
                or type(row["values"].get("temperature")) not in (int, float)
                or row["values"]["temperature"] != 0
                for row in profiles.values()
            )
        ):
            raise ValueError(
                "greedy sampling contract conflicts with declared sampling defaults"
            )
    return contract


def validate_sampling_qualification(record, descriptor, *, settings, runtime, artifact):
    contract = descriptor_sampling_contract(descriptor)
    bound = normalize_sampling_contract(settings.get("sampling_contract"))
    if bound != contract:
        raise ValueError(
            "sampling contract does not match source-owned selected descriptor"
        )
    coverage = (record.get("coverage_scope") or {}).get("sampling")
    if contract is None:
        if coverage is not None:
            raise ValueError(
                "sampling coverage was narrowed without a descriptor contract"
            )
        return
    if coverage != sampling_coverage(contract, runtime=runtime, artifact=artifact):
        raise ValueError("greedy sampling coverage lacks exact source/artifact binding")
    checks = record.get("checks") or {}
    if any(
        (checks.get(name) or {}).get("passed") is not True
        for name in GREEDY_SAMPLING_CHECKS
    ):
        raise ValueError(
            "greedy sampling qualification requires controls, refusal and repeat evidence"
        )
    refusal = (checks["sampling_refusal"].get("evidence") or {}).get("refusal") or {}
    if (
        refusal.get("status") != 400
        or not isinstance(refusal.get("body_utf8"), str)
        or not refusal["body_utf8"].strip()
    ):
        raise ValueError("greedy sampling refusal lacks actual HTTP400 body")
    if "sampling_controls" in checks or "seed_repeat" in checks:
        raise ValueError("greedy-only receipt cannot claim stochastic sampling checks")
