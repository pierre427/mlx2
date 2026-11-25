"""Exercise the emitted zsh unit, including propagation through its cleanup trap."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
ZSH = shutil.which("zsh")
pytestmark = pytest.mark.skipif(not ZSH, reason="the qualification drivers use zsh")
RAON = "raon1_mini_dflash2_affine_unit_20261125"
LING = "ling32_air_defaults_check_20261125"
HISTORICAL = "raon1_mini_window"


def generate(name, tmp_path, **settings):
    original = (ROOT / "scripts" / f"{name}.sh").read_text()
    heredoc = re.search(r"cat > \$RUNS/inner.sh <<(\w+)\n.*?\n\1\n", original, re.S)
    assert heredoc
    worker = tmp_path / "python-stub"
    worker.write_text(
        '#!/bin/zsh\n'
        'print -r -- "$*" >> "$CALLS"\n'
        'if [[ -n "$FAIL_COMMAND" && "$*" == *"$FAIL_COMMAND"* ]]; then exit 37; fi\n'
        'exit 0\n'
    )
    worker.chmod(0o755)
    env = dict(os.environ, SNAP=str(ROOT), RUNS=str(tmp_path), PY=str(worker), PORT="9999",
               DRIVER_DIR=str(ROOT / "scripts"), ARMS_EPOCH="4102444800", END_EPOCH="4102444800",
               MODEL="fixture", AFFINE="affine", NVFP4="nvfp4", DRAFT="draft",
               QUESTIONS="0", CK="fixture@fixture", CALLS=str(tmp_path / "calls"),
               FAIL_COMMAND="", FAIL_READINESS="", REFUSAL_MESSAGE="", CAP_S="3600")
    env.update(settings)
    subprocess.run([ZSH, "-c", heredoc.group()], env=env, check=True)
    inner = (tmp_path / "inner.sh").read_text()
    subprocess.run([ZSH, "-n"], input=inner, text=True, check=True)
    return inner, env


def run_generated(name, tmp_path, **settings):
    inner, env = generate(name, tmp_path, **settings)
    # Replace only external device/process probes, keeping the real server-arm
    # control flow, phase collection and EXIT trap. No server process is started.
    stubs = """
avail() { echo 128; }
swaps() { echo 0; }
swapins() { echo 0; }
lsof() { return 1; }
owned_server_exec() { print -r -- "$REFUSAL_MESSAGE"; return 2; }
owned_server_healthy() { [[ $name != $FAIL_READINESS ]]; }
kill() { return 1; }
curl() { echo '{}'; }
stop_server() { :; }
"""
    inner = inner.replace("trap stop_server EXIT", stubs + "\ntrap stop_server EXIT")
    proc = subprocess.run([ZSH], input=inner, cwd=tmp_path, env=env,
                          text=True, capture_output=True, timeout=10)
    failures = tmp_path / "phase-failures.tsv"
    return proc, (failures.read_text() if failures.exists() else ""), (tmp_path / "calls").read_text()


@pytest.mark.parametrize("unit,command,later", [
    (RAON, "raon1_mini_reference_oracle.py", "raon1_mini_prefill_ladder.py"),
    (LING, "spec_acceptance_gsm8k.py", "ling32_air_long_decode.py compare"),
])
def test_required_measurement_survives_successful_later_phase(tmp_path, unit, command, later):
    proc, failures, calls = run_generated(unit, tmp_path, FAIL_COMMAND=command)
    assert proc.returncode == 37, (proc.stdout, proc.stderr)
    assert failures.startswith("37\t")
    assert later in calls


@pytest.mark.parametrize("unit,arm,later", [
    (RAON, "div-ordinary", "raon1_mini_prefill_ladder.py"),
    (LING, "default-b16", "ling32_air_long_decode.py compare"),
])
def test_readiness_failure_survives_another_required_failure(tmp_path, unit, arm, later):
    proc, failures, calls = run_generated(unit, tmp_path, FAIL_READINESS=arm, FAIL_COMMAND=later)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert failures.startswith(f"1\tserver readiness {arm}\n")
    assert "37\t" in failures and later in calls


@pytest.mark.parametrize("unit", [RAON, LING])
def test_successful_unit_exits_zero(tmp_path, unit):
    proc, failures, _ = run_generated(unit, tmp_path)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert not failures


@pytest.mark.parametrize(("reference", "expected"), [("", 0), ("ref.npz", 37)])
def test_historical_raon_snapshot_without_helper(tmp_path, reference, expected):
    snapshot = tmp_path / "old-snapshot"
    snapshot.mkdir()
    proc, failures, calls = run_generated(
        HISTORICAL, tmp_path, SNAP=str(snapshot), REF=reference, MXFP8E="",
        FAIL_COMMAND="reference_oracle.py compare",
    )
    assert proc.returncode == expected, (proc.stdout, proc.stderr)
    assert bool(failures) == bool(reference)
    assert "spec_acceptance_gsm8k.py" in calls


def test_selected_ordinary_only_skips_long_compare(tmp_path):
    proc, failures, calls = run_generated("ling32_air_long_decode_unit", tmp_path, ARMS="ordinary")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert not failures
    assert "ling32_air_long_decode.py run" in calls
    assert "ling32_air_long_decode.py compare" not in calls


@pytest.mark.parametrize(("message", "expected"), [
    ("Ling-3.2-air vision mode serves the ordinary route only", 0),
    ("weights unavailable", 1),
])
def test_vision_refusal_identifies_the_adapter_error(tmp_path, message, expected):
    proc, failures, calls = run_generated("ling32_air_vision_window", tmp_path, REFUSAL_MESSAGE=message)
    assert proc.returncode == expected, (proc.stdout, proc.stderr)
    assert bool(failures) == bool(expected)
    assert "ling32_air_vision_smoke.py compare" in calls
