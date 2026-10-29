"""Generic explicit greedy/refusal qualification scope; no inference or GPU."""

import dataclasses
import importlib.util
import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

from mlx2.adapters.qwen import QWEN4_FLASH_NEXT
from mlx2.contracts import Capability
from mlx2.qualification import (
    APPROVED_QUALIFICATION_HARNESS,
    load_qualified_route,
    required_generic_checks,
)
from mlx2.sampling_contract import GREEDY_REFUSAL_SAMPLING_CONTRACT as CONTRACT
from mlx2.sampling_contract import (
    GREEDY_SAMPLING_CHECKS,
    normalize_sampling_contract,
    sampling_coverage,
    status_sampling_contract,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "greedy_qualifier", ROOT / "scripts/qualify_serving.py"
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def descriptor():
    return dataclasses.replace(
        QWEN4_FLASH_NEXT,
        metadata={**QWEN4_FLASH_NEXT.metadata, "sampling_contract": dict(CONTRACT)},
        capabilities=frozenset(
            {
                Capability.TEXT,
                Capability.STREAMING,
                Capability.PREFIX_REUSE,
                Capability.APC_V2,
            }
        ),
    )


def test_absent_preserves_legacy_and_explicit_contract_changes_required_domain():
    assert normalize_sampling_contract(None) is None
    assert "sampling_controls" in required_generic_checks(QWEN4_FLASH_NEXT)
    needed = required_generic_checks(descriptor())
    assert GREEDY_SAMPLING_CHECKS <= needed and "sampling_controls" not in needed
    for bad in (
        {},
        False,
        {"schema": "wrong"},
        {**CONTRACT, "modes": ["greedy", "stochastic"]},
        {**CONTRACT, "extra": True},
        {**CONTRACT, "unsupported": "fallback"},
    ):
        with pytest.raises(ValueError):
            normalize_sampling_contract(bad)


def test_contract_requires_consistent_declared_defaults():
    status = {
        "settings": {"sampling_contract": CONTRACT},
        "sampling_defaults": {"profiles": {"default": {"values": {"temperature": 0}}}},
    }
    assert status_sampling_contract(status) == CONTRACT
    for profiles in ({}, {"x": {"values": {"temperature": 0.7}}}, {"x": {}}):
        status["sampling_defaults"]["profiles"] = profiles
        with pytest.raises(ValueError, match="defaults"):
            status_sampling_contract(status)
    assert status_sampling_contract({"settings": {}}) is None


def response(request):
    neutral = {
        k: request[k]
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        )
    }
    return {
        "choices": [{"message": {"content": "apple, pear, banana"}}],
        "mlx2": {"request_controls": {"effective_sampling": neutral}},
    }


def test_probe_actual_http400_body_and_greedy_repeat_are_separate_coverage():
    requests = []

    def post(request):
        requests.append(request)
        if request["temperature"] != 0:
            raise HTTPError(
                "http://loopback",
                400,
                "unsupported",
                {},
                io.BytesIO(b'{"error":{"message":"greedy only"}}'),
            )
        return response(request)

    rows = harness.run_greedy_sampling_probe(post)
    assert set(rows) == GREEDY_SAMPLING_CHECKS and all(
        v["passed"] for v in rows.values()
    )
    assert [r["temperature"] for r in requests] == [0, 0, 0.7]
    assert (
        rows["sampling_refusal"]["evidence"]["refusal"]["body_utf8"]
        == '{"error":{"message":"greedy only"}}'
    )
    assert "sampling_controls" not in rows and "seed_repeat" not in rows
    assert (
        sampling_coverage(CONTRACT, runtime={"source": "a"}, artifact="weights")[
            "stochastic_tested"
        ]
        is False
    )


@pytest.mark.parametrize(
    "behavior", ["accept", "wrongstatus", "emptybody", "wrongcontrols"]
)
def test_greedy_probe_fails_missing_refusal_or_wrong_controls(behavior):
    def post(request):
        if request["temperature"]:
            if behavior == "accept":
                return response(request)
            raise HTTPError(
                "http://loopback",
                429 if behavior == "wrongstatus" else 400,
                "unsupported",
                {},
                io.BytesIO(
                    b"" if behavior == "emptybody" else b'{"error":"unsupported"}'
                ),
            )
        r = response(request)
        if behavior == "wrongcontrols":
            r["mlx2"]["request_controls"]["effective_sampling"]["temperature"] = 0.7
        return r

    rows = harness.run_greedy_sampling_probe(post)
    assert not all(r["passed"] for r in rows.values())


def receipt():
    d = descriptor()
    settings = {
        "mtp": False,
        "max_context": 4096,
        "default_max_tokens": 64,
        "sampling_contract": CONTRACT,
    }
    runtime = {"source": "source"}
    artifact = "weights"
    record = {
        "passed": True,
        "runtime": runtime,
        "artifact": artifact,
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "coverage_scope": {
            "sampling": sampling_coverage(CONTRACT, runtime=runtime, artifact=artifact)
        },
        "checks": {name: {"passed": True} for name in required_generic_checks(d)},
    }
    record["checks"]["sampling_refusal"]["evidence"] = {
        "refusal": {"status": 400, "body_utf8": '{"error":"greedy only"}'}
    }
    return record, d


def test_loader_binds_explicit_scope_to_source_descriptor_and_actual_refusal(tmp_path):
    record, d = receipt()
    path = tmp_path / "q.json"
    path.write_text(json.dumps(record))
    args = {
        "runtime": record["runtime"],
        "artifact": record["artifact"],
        "settings": record["settings"],
        "descriptor": d,
        "name": "greedy",
    }
    assert "greedy" in load_qualified_route(path, **args).receipt
    original = json.loads(path.read_text())
    for mutate in (
        lambda r: r["coverage_scope"]["sampling"]["binding"].update(artifact="other"),
        lambda r: r["checks"].pop("sampling_refusal"),
        lambda r: r["checks"]["sampling_refusal"].update(
            evidence={"refusal": {"status": 400, "body_utf8": ""}}
        ),
        lambda r: r["checks"].update(sampling_controls={"passed": True}),
        lambda r: r["coverage_scope"]["sampling"].update(stochastic_tested=True),
    ):
        r = json.loads(json.dumps(original))
        mutate(r)
        path.write_text(json.dumps(r))
        with pytest.raises(ValueError, match="sampling|greedy"):
            load_qualified_route(path, **args)
    path.write_text(json.dumps(original))
    ordinary = dataclasses.replace(
        d, metadata={k: v for k, v in d.metadata.items() if k != "sampling_contract"}
    )
    with pytest.raises(ValueError, match="sampling contract"):
        load_qualified_route(path, **{**args, "descriptor": ordinary})


def test_approved_producer_pin_matches_final_harness_bytes():
    assert harness.qualification_harness_identity()==APPROVED_QUALIFICATION_HARNESS
