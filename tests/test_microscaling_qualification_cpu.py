"""CPU source-bound qualification contracts; no checkpoint execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from mlx2.adapters.qwen import QWEN4_FLASH_NEXT
from mlx2.contracts import Fidelity
from mlx2.qualification import (
    APPROVED_QUALIFICATION_HARNESS, REQUIRED_CHECKS, load_qualified_route,
    required_descriptor_checks, required_feature_checks, required_generic_checks,
)
from scripts.qualify_serving import feature_observations, microscaling_prefill_observation


def snapshot(*, dense=2, grouped=3, calls=0, rows=0):
    count = dense + grouped
    return {"microscaling_prefill": {
        "enabled": True, "active": True, "revision": "artifact-mask-activation-revision",
        "kernel_revision": "kernel-source-revision", "artifact_fingerprint": "checkpoint-fingerprint",
        "modules": count, "module_kinds": {"dense": dense, "grouped": grouped},
        "projection_paths": [f"projection.{i}" for i in range(count)],
        "counts": {"engaged_calls": calls, "engaged_rows": rows,
                   "dense_calls": calls if dense else 0, "grouped_calls": calls if grouped else 0},
    }}


def test_selected_microscaling_requires_its_own_engagement_check():
    assert "feature_microscaling_prefill" in required_feature_checks(
        {"microscaling_prefill": {"enabled": True}})
    baseline = required_feature_checks({})
    assert required_feature_checks({"microscaling_prefill": {"enabled": False}}) == baseline
    assert "microscaling_prefill" in feature_observations({})


def test_counter_growth_is_run_local_and_bound_to_actual_route():
    initial = snapshot(calls=10, rows=5120)
    final = snapshot(calls=12, rows=6144)
    assert microscaling_prefill_observation(final, initial) > 0
    assert feature_observations(final, initial=initial)["microscaling_prefill"] > 0
    assert microscaling_prefill_observation(final, None) == 0
    assert microscaling_prefill_observation(final, final) == 0
    assert microscaling_prefill_observation(initial, final) == 0
    for key, value in [("revision", "other-mask"), ("kernel_revision", "other-source"),
                       ("artifact_fingerprint", "other-checkpoint"), ("active", False),
                       ("enabled", False), ("modules", True), ("projection_paths", ["other"]),
                       ("module_kinds", {"dense": 0, "grouped": 5})]:
        mutated = deepcopy(final)
        mutated["microscaling_prefill"][key] = value
        assert microscaling_prefill_observation(mutated, initial) == 0, key


def test_declared_kind_requires_dispatch_not_just_total_counter():
    initial = snapshot()
    final = snapshot(calls=4, rows=2048)
    final["microscaling_prefill"]["counts"]["grouped_calls"] = 0
    assert microscaling_prefill_observation(final, initial) == 0
    # A true dense-only route has no grouped dispatch requirement.
    assert microscaling_prefill_observation(snapshot(dense=2, grouped=0, calls=4, rows=2048),
        snapshot(dense=2, grouped=0)) > 0
    assert microscaling_prefill_observation(snapshot(dense=0, grouped=3, calls=4, rows=2048),
        snapshot(dense=0, grouped=3)) > 0
    final = snapshot(calls=4, rows=2048)
    final["microscaling_prefill"]["counts"]["engaged_rows"] = True
    assert microscaling_prefill_observation(final, initial) == 0


def test_qualification_loader_requires_feature_and_approximate_tier(tmp_path):
    settings = {"mtp": False, "max_context": 4096, "microscaling_prefill": {
        "enabled": True, "revision": "bound-law", "kernel_revision": "bound-kernel",
        "artifact_fingerprint": "weights", "projection_paths": ["projection"]}}
    checks = (REQUIRED_CHECKS | required_generic_checks(QWEN4_FLASH_NEXT) | {"structured_output"}
        | required_feature_checks(settings) | required_descriptor_checks(QWEN4_FLASH_NEXT, settings))
    record = {"passed": True, "runtime": {"source": "synthetic-contract"}, "artifact": "weights",
        "settings": settings, "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {name: {"passed": True} for name in checks}}
    path = tmp_path / "record.json"
    args = dict(runtime=record['runtime'], artifact='weights', settings=settings,
        descriptor=QWEN4_FLASH_NEXT, name='synthetic-sc-contract')
    path.write_text(json.dumps(record))
    assert load_qualified_route(path, **args).profile.fidelity is Fidelity.APPROXIMATE
    del record['checks']['feature_microscaling_prefill']
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='lack observed qualification'):
        load_qualified_route(path, **args)
    record['checks']['feature_microscaling_prefill'] = {'passed': True}
    path.write_text(json.dumps(record))
    changed = deepcopy(settings); changed['microscaling_prefill']['revision'] = 'other-law'
    with pytest.raises(ValueError, match='serving settings'):
        load_qualified_route(path, **{**args, 'settings': changed})


def test_reviewed_harness_hash_matches_current_producer():
    producer = Path(__file__).resolve().parents[1] / 'scripts/qualify_serving.py'
    assert hashlib.sha256(producer.read_bytes()).hexdigest() == APPROVED_QUALIFICATION_HARNESS['sha256']
