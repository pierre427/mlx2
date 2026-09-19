#!/usr/bin/env python3
"""Run the frozen serving qualifier with a one-shot pre-context thermal gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from urllib.request import Request


ROOT = Path("~/Desktop/mlx2")
QUALIFIER = ROOT / "scripts/qualify_serving.py"
MANIFEST = ROOT / "qualification/four-model-experiments.json"
EXPECTED_SOURCE = os.environ["EXPECTED_SOURCE_SHA"]
EXPECTED_PROFILE = os.environ["EXPECTED_PROFILE"]
SIDECAR = Path(os.environ["THERMAL_BARRIER_SIDECAR"])


def load_qualifier():
    spec = importlib.util.spec_from_file_location("mlx2_frozen_qualifier", QUALIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen qualifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qualifier = load_qualifier()
original_urlopen = qualifier.urlopen
gate_fired = False


def read_status(url: str) -> dict:
    with original_urlopen(url.rstrip("/") + "/v1/status", timeout=30) as response:
        status = json.load(response)
    assert status.get("healthy") is True and status.get("error") is None, status
    assert status.get("state") == "ready", status
    assert status.get("inflight") == 0 and status.get("queue_depth") == 0, status
    assert status["runtime"]["source_sha256"] == EXPECTED_SOURCE, status["runtime"]
    assert status.get("profile") == EXPECTED_PROFILE, status
    return status


def atomic_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def is_first_long_context(request: object) -> tuple[bool, dict | None]:
    if not isinstance(request, Request) or request.data is None:
        return False, None
    if not request.full_url.endswith("/v1/chat/completions"):
        return False, None
    try:
        body = json.loads(request.data)
    except (TypeError, ValueError, UnicodeDecodeError):
        return False, None
    messages = body.get("messages")
    if body.get("max_tokens") != qualifier.LONG_CONTEXT_COMPLETION_TOKENS:
        return False, body
    if not isinstance(messages, list) or len(messages) != 1:
        return False, body
    content = messages[0].get("content") if isinstance(messages[0], dict) else None
    match = (
        isinstance(content, str)
        and content.startswith("data ")
        and content.endswith(qualifier.LONG_CONTEXT_INSTRUCTION)
    )
    return match, body


def gated_urlopen(target, *args, **kwargs):
    global gate_fired
    matches, body = is_first_long_context(target)
    if matches and not gate_fired:
        base_url = target.full_url.split("/v1/chat/completions", 1)[0]
        status_before = read_status(base_url)
        expected_prompt = qualifier.long_context_prompt(status_before["max_context"])
        assert body["messages"][0]["content"] == expected_prompt
        from run_qualification_matrix import stabilize_thermal, thermally_stable

        policy = json.loads(MANIFEST.read_text())["thermal"]
        samples = stabilize_thermal(policy)
        consecutive = int(policy["consecutive_samples"])
        tail = samples[-consecutive:]
        assert len(tail) == consecutive == 3
        assert int(policy["required_thermal_state"]) == 0
        assert int(policy["sample_interval_seconds"]) == 15
        assert all(sample["thermal_state"] == 0 for sample in tail)
        assert all(thermally_stable(sample, policy) for sample in tail)
        temperatures = [sample["virtual_temperature_c"] for sample in tail]
        assert max(temperatures) - min(temperatures) <= float(
            policy["max_temperature_delta_c"]
        )
        status_after = read_status(base_url)
        assert status_after["runtime"] == status_before["runtime"]
        assert status_after["artifact"] == status_before["artifact"]
        assert status_after["settings"] == status_before["settings"]
        gate_fired = True
        atomic_json(
            SIDECAR,
            {
                "schema": "mlx2.qualifier-pre-long-thermal-barrier.v1",
                "gate_fired": 1,
                "expected_source_sha256": EXPECTED_SOURCE,
                "expected_profile": EXPECTED_PROFILE,
                "qualifier_sha256": hashlib.sha256(QUALIFIER.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                "request_sha256": hashlib.sha256(target.data).hexdigest(),
                "policy": policy,
                "samples": samples,
                "stable_tail": tail,
                "status_before": status_before,
                "status_after": status_after,
            },
        )
    return original_urlopen(target, *args, **kwargs)


if __name__ == "__main__":
    qualifier.urlopen = gated_urlopen
    qualifier.main()
