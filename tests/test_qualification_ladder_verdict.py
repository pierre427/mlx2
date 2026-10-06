"""Thermal ladder and qualification verdict false greens (sweep H2, H3).

The scripts live in the latest qualification run directory, which the next
run copies (qualification/runs/qualify-e8861bb5-uncensored).

H2: a warm request that reused only the chat preamble (cached_tokens 54 of
    32,000) counted as an APCv2 hit.
H3: the verdict bound no identity, compared against a reference from another
    model/route/artifact, and qualified runs_per_cell 0.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "qualification" / "runs" / "qualify-e8861bb5-uncensored"


def _load_exact_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load qualification module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_modules():
    missing = [name for name in ("thermal_ladder.py", "qualification_verdict.py")
               if not (RUN / name).is_file()]
    if missing:
        pytest.skip(f"private qualification run copies absent ({RUN.relative_to(ROOT)}: "
                    f"{', '.join(missing)}); not exported to the public mirror")
    previous_path = list(sys.path)
    sys.path.insert(0, str(RUN))
    previous = os.environ.get("MLX2_CAMPAIGN_ROOT")
    os.environ["MLX2_CAMPAIGN_ROOT"] = str(ROOT)
    unique = {
        "thermal": "_mlx2_qualify_e8861bb5_thermal_ladder",
        "verdict": "_mlx2_qualify_e8861bb5_qualification_verdict",
    }
    shadowed = {
        name: sys.modules.pop(name, None)
        for name in ("ladder", "run_qualification_matrix")
    }
    try:
        yield (
            _load_exact_module(unique["thermal"], RUN / "thermal_ladder.py"),
            _load_exact_module(unique["verdict"], RUN / "qualification_verdict.py"),
        )
    finally:
        for name in unique.values():
            sys.modules.pop(name, None)
        for name, module in shadowed.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module
        sys.path[:] = previous_path
        if previous is None:
            os.environ.pop("MLX2_CAMPAIGN_ROOT", None)
        else:
            os.environ["MLX2_CAMPAIGN_ROOT"] = previous


def _row(cached, prompt=32000):
    return {"done": True, "content": "NEEDLE-X-0", "prompt_tokens": prompt,
            "completion_tokens": 128, "cached_tokens": cached,
            "server_ttft_seconds": 20.0, "client_ttft_seconds": 20.0,
            "prefill_tokens_per_second": 1500.0, "decode_tokens_per_second": 60.0}


def test_preamble_only_warm_reuse_is_not_an_apc_hit(run_modules):
    ladder, _ = run_modules
    cold = {"rows": [_row(54)], "aggregate_completion_tokens_per_second": 60.0}
    preamble = ladder.summarize_run(cold, {"rows": [_row(54)]}, ["NEEDLE-X-0"])
    assert preamble["warm_apc_hits"] == 0
    full = ladder.summarize_run(cold, {"rows": [_row(31999)]}, ["NEEDLE-X-0"])
    assert full["warm_apc_hits"] == 1
    assert full["warm_cached_min_fraction"] > 0.99


def _run(i, warm_cached=31999, decode=60.0, prefill=1500.0):
    summary = {"streams_done": True, "needle_correct": 2, "needle_total": 2,
               "warm_apc_hits": 1, "prefill_tokens_per_second": prefill,
               "decode_tokens_per_second": decode}
    return {"run": i, "summary": summary,
            "requests": {"warm": {"rows": [_row(warm_cached)]}}}


RUNTIME = {"source_sha256": "5" * 64, "mlx_native_sha256": "7" * 64, "mlx": "0.32.2"}


def _ladder(runs, model="flash-next", route="mtp2", artifact="A", runs_per_cell=3):
    return {"finished_at": 1, "runs_per_cell": runs_per_cell, "model": model, "route": route,
            "max_context": 65536,
            "initial": {"artifact": artifact, "runtime": RUNTIME},
            "cells": [{"requested_tokens": 32768, "width": 1, "runs": runs}]}


def test_verdict_fails_a_preamble_only_warm_hit(run_modules):
    _, verdict = run_modules
    assert verdict.verdict(_ladder([_run(i) for i in range(3)]))["qualified"] is True
    result = verdict.verdict(_ladder([_run(i, warm_cached=54) for i in range(3)]))
    assert result["qualified"] is False
    assert any("warm APC hits 0/1" in f for f in result["failures"])


def test_verdict_refuses_a_foreign_reference(run_modules):
    _, verdict = run_modules
    slow = _ladder([_run(i, decode=12.0, prefill=120.0) for i in range(3)])
    foreign = _ladder([_run(i, decode=10.0, prefill=100.0) for i in range(3)],
                      model="some-slow-model", route="ordinary", artifact="Z")
    result = verdict.verdict(slow, foreign)
    assert result["qualified"] is False
    assert any("reference is not the same" in f for f in result["failures"])
    own = _ladder([_run(i) for i in range(3)])
    regressed = verdict.verdict(slow, own)
    assert regressed["qualified"] is False
    assert any("regressed" in f for f in regressed["failures"])


def test_verdict_binds_identity_and_refuses_zero_runs(run_modules):
    _, verdict = run_modules
    zero = _ladder([], runs_per_cell=0)
    result = verdict.verdict(zero)
    assert result["qualified"] is False
    assert result["identity"]["model"] == "flash-next"
    assert result["identity"]["artifact"] == "A"
    anonymous = _ladder([_run(i) for i in range(3)])
    del anonymous["model"], anonymous["initial"]
    assert verdict.verdict(anonymous)["qualified"] is False


# --- review item 7: runtime identity ----------------------------------------


@pytest.mark.parametrize("runtime", [
    None, {}, {"source_sha256": "S", "mlx_native_sha256": "7" * 64},
    {"source_sha256": "5" * 64}, {"source_sha256": "5" * 64, "mlx_native_sha256": None},
    "e151ee5e",
])
def test_verdict_requires_a_runtime_identity(run_modules, runtime):
    """Codex review item 7: removing initial.runtime from an otherwise passing
    ladder still qualified, with runtime_source_sha256 null."""
    _, verdict = run_modules
    ladder = _ladder([_run(i) for i in range(3)])
    if runtime is None:
        del ladder["initial"]["runtime"]
    else:
        ladder["initial"]["runtime"] = runtime
    result = verdict.verdict(ladder)
    assert result["qualified"] is False
    assert any("runtime identity" in f for f in result["failures"])


def test_verdict_preserves_the_runtime_identity(run_modules):
    _, verdict = run_modules
    result = verdict.verdict(_ladder([_run(i) for i in range(3)]))
    assert result["qualified"] is True
    assert result["identity"]["runtime_source_sha256"] == "5" * 64
    assert result["identity"]["runtime_mlx_native_sha256"] == "7" * 64
    assert result["identity"]["runtime_mlx"] == "0.32.2"


@pytest.mark.parametrize("name", ["ladder-short.json", "ladder-long.json"])
def test_real_e8861bb5_ladders_still_qualify(run_modules, name):
    import json

    _, verdict = run_modules
    path = RUN / "results" / "ladder-flash-next-uncensored-mtp2" / name
    if not path.is_file():
        pytest.skip(f"private ladder evidence absent: {path.relative_to(ROOT)}")
    result = verdict.verdict(json.loads(path.read_text()))
    assert result["qualified"] is True, result["failures"]
    assert result["identity"]["runtime_source_sha256"] == (
        "e151ee5e976869deef1e76f8463c6bf5b230e0f22f44ead931130a111fe0a159")
